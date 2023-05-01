use std::sync::Arc;

/// A cloneable wrapper around [`anyhow::Error`] to satisfy
/// [`future::Shared`](futures::future::Shared).
#[derive(Clone)]
pub struct Error(Arc<anyhow::Error>);

pub type Result<T> = std::result::Result<T, Error>;

impl<E> From<E> for Error
where
    anyhow::Error: From<E>,
{
    fn from(error: E) -> Self {
        Self(Arc::new(anyhow::Error::from(error)))
    }
}
