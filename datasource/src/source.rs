use std::{ops::Range, sync::Arc};

use crate::DataSlice;

//mod file;
//
//pub use file::FileDataSource;

/// Trait representing a data source.
#[async_trait::async_trait]
pub trait DataSource: Sync + Send {
    /// Returns a DataSlice for the given range. If the DataSource already has
    /// this data, it returns immediately, otherwise it will block until the
    /// data is available. If the source is finished and the range is out of the
    /// bounds of the source, it returns None.
    async fn get_data(&self, range: Range<u64>) -> anyhow::Result<Option<DataSlice>>;

    /// Waits until the data source has finished, and returns the final size. If
    /// the source is infinite, it returns None. For sources that are inherently
    /// sized, like files, it may return a size before all the data is read.
    async fn size(&self) -> anyhow::Result<Option<u64>>;
}

#[async_trait::async_trait]
impl DataSource for Box<dyn DataSource> {
    async fn get_data(&self, range: Range<u64>) -> anyhow::Result<Option<DataSlice>> {
        self.as_ref().get_data(range).await
    }

    async fn size(&self) -> anyhow::Result<Option<u64>> {
        self.as_ref().size().await
    }
}

#[async_trait::async_trait]
impl DataSource for Arc<dyn DataSource> {
    async fn get_data(&self, range: Range<u64>) -> anyhow::Result<Option<DataSlice>> {
        self.as_ref().get_data(range).await
    }

    async fn size(&self) -> anyhow::Result<Option<u64>> {
        self.as_ref().size().await
    }
}
