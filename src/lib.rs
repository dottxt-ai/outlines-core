pub mod index;
pub mod json_schema;
pub mod prelude;
pub mod primitives;
pub mod regex;
pub mod vocabulary;

#[cfg(feature = "python-bindings")]
mod python_bindings;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("The vocabulary does not allow us to build a sequence that matches the input")]
    IndexError,
}

#[cfg(feature = "python-bindings")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        use pyo3::{exceptions::PyValueError, PyErr};
        PyErr::new::<PyValueError, _>(e.to_string())
    }
}
