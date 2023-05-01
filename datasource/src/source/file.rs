use std::{ops::Range, path::PathBuf, sync::Arc};

use futures::{
    future::{self, BoxFuture},
    FutureExt, TryFutureExt,
};

use crate::{error, DataSlice, DataSource};

/// A data source that is backed by a file.
///
/// This is a simple implementation that just reads the file directly, in one
/// chunk.
pub struct FileDataSource {
    // TODO: chunk file and read chunks on demand so 1) better incremental IO,
    // and 2) a single slice doesn't hold the entire file in memory
    content: future::Shared<BoxFuture<'static, error::Result<DataSlice>>>,
}

impl FileDataSource {
    pub fn new(path: impl Into<PathBuf>) -> error::Result<Self> {
        let content = tokio::fs::read(path.into())
            .map_err(|err| error::Error::from(err))
            .map_ok(|data| DataSlice::new([Arc::from(&*data)], 0, 0..data.len()))
            .boxed()
            .shared();

        Ok(Self { content })
    }
}

#[async_trait::async_trait]
impl DataSource for FileDataSource {
    async fn get_data(&self, _range: Range<u64>) -> error::Result<Option<DataSlice>> {
        let slice = self.content.clone().await?;

        // TODO sub-slice range

        Ok(Some(slice))
    }

    async fn size(&self) -> error::Result<Option<u64>> {
        let content = self.content.clone().await?;

        Ok(Some(content.len() as u64))
    }
}
