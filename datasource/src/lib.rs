pub mod error;
mod slice;
mod source;

pub use slice::DataSlice;
pub use source::DataSource;
pub use source::FileDataSource;

pub use error::Error;
