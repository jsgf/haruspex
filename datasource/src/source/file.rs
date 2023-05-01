use std::ops::Range;

use crate::{DataSlice, DataSource};

/// A data source that is backed by a file.
///
/// This is a simple implementation that just reads the file directly, in one
/// chunk.
pub struct FileDataSource {
    file: tokio::fs::File,
    size: u64,
}

impl FileDataSource {
    pub async fn new(file: tokio::fs::File) -> anyhow::Result<Self> {
        let size = file.metadata().await?.len();
        Ok(Self {
            file,
            size: Some(size),
        })
    }
}

#[async_trait::async_trait]
impl DataSource for FileDataSource {
    async fn get_data(&self, range: Range<u64>) -> anyhow::Result<Option<DataSlice>> {
        todo!()
    }

    async fn size(&self) -> anyhow::Result<Option<u64>> {
        Ok(Some(self.size))
    }
}
