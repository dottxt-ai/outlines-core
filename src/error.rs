use thiserror::Error;

pub type Result<T, E = crate::Error> = std::result::Result<T, E>;

#[derive(Error, Debug)]
#[error("{0}")]
pub struct TokenizersError(pub tokenizers::Error);

impl PartialEq for TokenizersError {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_string() == other.0.to_string()
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum Error {
    #[error("The vocabulary does not allow us to build a sequence that matches the input")]
    IndexError,
    #[error(transparent)]
    TokenizersError(#[from] TokenizersError),
    #[error("Unsupported tokenizer for {model}: {reason}, please open an issue with the full error message: https://github.com/dottxt-ai/outlines-core/issues")]
    UnsupportedTokenizer { model: String, reason: String },
    #[error("Unable to locate EOS token for {model}")]
    UnableToLocateEosTokenId { model: String },
    #[error("Tokenizer is not supported by token processor")]
    UnsupportedByTokenProcessor,
    #[error("Decoder unpacking failed for token processor")]
    DecoderUnpackingFailed,
    #[error("Token processing failed for byte level processor")]
    ByteProcessorFailed,
    #[error("Token processing failed for byte fallback level processor")]
    ByteFallbackProcessorFailed,
}

#[cfg(feature = "python-bindings")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        use pyo3::{exceptions::PyValueError, PyErr};
        PyErr::new::<PyValueError, _>(e.to_string())
    }
}
