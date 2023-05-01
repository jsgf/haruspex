use std::{borrow::Cow, ops::Range, sync::Arc};

use crate::{error, DataSource};

/// A slice of data from a DataSource.
///
/// This represents a reference to a range of data buffered by a [`DataSource`].
/// The data is logically a single contiguous range, but it may be stored as
/// multiple non-contiguous segments.
///
/// A [`DataSlice`] is refcounted, and may outlive the DataSource that created
/// it.
#[derive(Debug, Clone)]
pub struct DataSlice {
    /// The data itself, considered contiguous.
    data: Vec<Arc<[u8]>>,
    /// The offset of the first byte in the `data` in the overall DataSource.
    offset: u64,
    /// Range of offsets into the `data`.
    range: Range<usize>,
}

impl DataSlice {
    /// Construct a new data slice. This shares ownership with the
    /// [`DataSource`]'s internal buffers, but obviously may outlive the buffer.
    ///
    /// Constraints:
    /// - only data segments covered by `range` will be kept
    /// - `range` must not go out of bounds of data segments
    /// - `data` may be empty, but individual data elements must not be empty
    ///
    /// This is only intended to be used by [`DataSource`] implementations.
    ///
    /// TODO: Do we need index data elements by offset?
    pub fn new(
        data: impl IntoIterator<Item = Arc<[u8]>>,
        offset: u64,
        range: Range<usize>,
    ) -> Self {
        let mut range = range;
        let mut offset = offset;

        let data: Vec<_> = data
            .into_iter()
            .filter_map({
                let mut doff = 0;
                let offset = &mut offset;
                move |data| {
                    assert!(!data.is_empty(), "Empty data slice"); // or just filter?

                    let len = doff + data.len();
                    doff += len;
                    if len < range.start {
                        range.start -= len;
                        range.end -= len;
                        *offset += len as u64;
                        None
                    } else {
                        Some(data)
                    }
                }
            })
            .collect();
        assert!(range.start <= range.end, "Invalid range: {range:?}");
        assert!(
            !data.is_empty() || range.is_empty(),
            "data.len {} range {range:?}",
            data.len()
        );

        Self {
            data,
            offset,
            range,
        }
    }

    /// Length of slice in bytes.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// True is slice represents no data.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Return the range of data in the slice.
    pub fn range(&self) -> Range<u64> {
        self.abs_range(&self.range)
    }

    /// Return the range of data as a single contiguous slice. If it's already
    /// contiguous, then return a direct slice, otherwise glue the discontinuous
    /// slices together.
    ///
    /// The returned offset may be higher than the start of the range if
    /// beginning of the range does not overlap with data in the slice. The
    /// returned data may likewise be truncated.
    ///
    /// If the range does not overlap the slice at all, then it returns an empty
    /// data at the given offset.
    pub fn get_contiguous(&self, range: Range<u64>) -> (u64, Cow<[u8]>) {
        let mut it = self.iter_range(range.clone());

        match (it.next(), it.next()) {
            (None, None) => (range.start, Cow::Borrowed(&[])),
            (Some((off, data)), None) => (off, Cow::Borrowed(data)),
            (None, Some(_)) => panic!("Broken iterator"),
            (Some((o1, d1)), Some((o2, d2))) => {
                debug_assert_eq!(o1 + d1.len() as u64, o2);
                let mut v: Vec<u8> = Vec::with_capacity((range.end - range.start) as usize);
                [d1, d2]
                    .into_iter()
                    .chain(it.map(|(_, d)| d))
                    .for_each(|d| {
                        v.extend_from_slice(d);
                    });
                (o1, Cow::Owned(v))
            }
        }
    }

    fn abs_range(&self, range: &Range<usize>) -> Range<u64> {
        range.start as u64 + self.offset..range.end as u64 + self.offset
    }

    fn local_range(&self, range: &Range<u64>) -> Range<usize> {
        debug_assert!(
            range.start >= self.offset,
            "Range {range:?} offset {}",
            self.offset
        );
        (range.start - self.offset) as usize..(range.end - self.offset) as usize
    }

    /// Iterate all the data slices, along with their offsets.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &'_ [u8])> {
        let range = self.abs_range(&self.range);
        self.iter_range(range)
    }

    /// Iterate all the data slices in the given range, with their offsets.
    pub fn iter_range(&self, range: Range<u64>) -> impl Iterator<Item = (u64, &'_ [u8])> {
        // normalize range to be within the slice
        let range = self.local_range(&range);

        let slices = self
            .data
            .iter()
            .map({
                let mut offset = 0;
                move |d| {
                    let o = offset;
                    offset += d.len();
                    (o..offset, &**d)
                }
            })
            .filter_map({
                move |(srange, data)| {
                    let remains = range_intersect(&srange, &range);
                    if remains.is_empty() {
                        None
                    } else {
                        Some((
                            *remains.start,
                            &data[(*remains.start - srange.start)..(*remains.end - srange.start)],
                        ))
                    }
                }
            })
            .map(|(start, data)| (self.offset + start as u64, data));

        slices
    }

    /// Generate a sub-slice of this slice.
    /// The returned slice will share the same data as this slice.
    pub fn sub_slice(&self, range: Range<u64>) -> DataSlice {
        let _ = range;
        todo!()
    }
}

fn range_intersect<'a, T: Ord>(a: &'a Range<T>, b: &'a Range<T>) -> Range<&'a T> {
    let start = (&a.start).max(&b.start);
    let end = (&a.end).min(&b.end);
    if end >= start {
        start..end
    } else {
        start..start
    }
}

#[async_trait::async_trait]
impl DataSource for DataSlice {
    async fn get_data(&self, range: Range<u64>) -> error::Result<Option<DataSlice>> {
        Ok(Some(self.sub_slice(range)))
    }

    async fn size(&self) -> error::Result<Option<u64>> {
        Ok(Some(self.range.len() as u64))
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    use super::*;

    fn range_strategy<T: Arbitrary + Ord + Copy>() -> impl Strategy<Value = Range<T>> {
        (any::<T>(), any::<T>())
            .prop_map(|(start, end)| if start < end { start..end } else { end..start })
    }

    // Custom strategy to generate DataSlice
    fn dataslice_strategy() -> impl Strategy<Value = DataSlice> {
        proptest::collection::vec(1..=8usize, 0..=8usize)
            .prop_flat_map(|lengths| {
                let total_length: usize = lengths.iter().sum();

                let data: Vec<Arc<[u8]>> = lengths
                    .into_iter()
                    .map(|len| {
                        let v: Vec<u8> = (0..len).map(|_| rand::random::<u8>()).collect();
                        Arc::from(v)
                    })
                    .collect();

                (Just(data), 0..=total_length, 0..=total_length)
            })
            .prop_map(|(data, start, end)| {
                let range = if start < end { start..end } else { end..start };
                println!("data: {:?}, range: {:?}", data, range);
                DataSlice::new(data, 0, range)
            })
    }

    prop_compose! {
        fn dataslice_and_range_strategy()
            (dataslice in dataslice_strategy())
            (start in dataslice.range.clone(), end in dataslice.range.clone(), dataslice in Just(dataslice)) -> (DataSlice, Range<u64>) {
            let start = dataslice.offset + start as u64;
            let end = dataslice.offset + end as u64;

            (dataslice, if start < end { start..end } else { end..start })
        }
    }

    proptest! {
        #[test]
        fn test_dataslice_with_proptest(
            dataslice in dataslice_strategy(),
            range in range_strategy::<u64>()) {
            let (offset, contiguous) = dataslice.get_contiguous(range.clone());

            let expected_data: Vec<u8> = dataslice.iter_range(range.clone())
                .flat_map(|(o, d)| {
                    if o <= offset {
                        d[(offset - o) as usize..].iter().cloned().collect::<Vec<u8>>()
                    } else {
                        d.iter().cloned().collect::<Vec<u8>>()
                    }
                })
                .collect();

            assert_eq!(&*contiguous, expected_data.as_slice());
        }
    }

    fn reference_iter_range(data: &[Arc<[u8]>], mut range: Range<usize>) -> Vec<u8> {
        let mut result = Vec::new();

        for slice in data {
            if range.start >= slice.len() {
                range = range.start - slice.len()..range.end - slice.len();
                continue;
            }

            let slicerange = range.start..range.end.min(slice.len());
            result.extend(&slice[slicerange.clone()]);

            range.start = slicerange.end;
            if range.is_empty() {
                break;
            }
        }

        result
    }

    proptest! {
        #[test]
        fn test_iter_range(
            ref dataslice in dataslice_strategy(),
            range in range_strategy::<usize>()
        ) {
            let expected_data: Vec<u8> = reference_iter_range(&dataslice.data, range.clone());
            let result_data: Vec<u8> = dataslice.iter_range(dataslice.abs_range(&range)).map(|(_, chunk)| chunk).flatten().copied().collect();
            assert_eq!(expected_data, result_data);
        }
    }

    proptest! {
        #[test]
        fn test_get_contiguous(ref dataslice in dataslice_strategy(),
                               range in range_strategy::<usize>()     ) {
            let contiguous = dataslice.get_contiguous(dataslice.abs_range(&range));
            let expected_data: Vec<u8> = reference_iter_range(&dataslice.data, range);
            assert_eq!(&*contiguous.1, expected_data.as_slice());
        }
    }

    proptest! {
        #[test]
        fn test_get_contiguous_borrowed(
            ref dataslice in dataslice_strategy(),
            range in range_strategy::<usize>()
        ) {
            let contiguous = dataslice.get_contiguous(dataslice.abs_range(&range));
            let iter: Vec<_> = dataslice.iter_range(dataslice.abs_range(&range)).collect();

            match (contiguous.1, &*iter) {
                (Cow::Borrowed([]), []) | (Cow::Borrowed([_]), [_]) => {},
                (Cow::Owned(_), [_, ..]) => {},
                (contiguous, iter) => panic!("contig {contiguous:?} iter {iter:?}"),
            }
        }
    }
}
