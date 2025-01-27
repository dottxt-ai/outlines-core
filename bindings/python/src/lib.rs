//! Provides tools and interfaces to integrate the crate's functionality with Python.

mod guide;
mod index;
mod vocabulary;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{exceptions::PyValueError, PyErr};

use ::outlines_core::json_schema;
use ::outlines_core::Error as CoreError;

use crate::guide::PyGuide;
use crate::index::PyIndex;
use crate::vocabulary::PyVocabulary;

pub struct Error(CoreError);

impl From<CoreError> for Error {
    fn from(e: CoreError) -> Self {
        Error(e)
    }
}

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        PyErr::new::<PyValueError, _>(e.to_string())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[pyfunction(name = "build_regex_from_schema")]
#[pyo3(signature = (json_schema, whitespace_pattern=None))]
pub fn build_regex_from_schema_py(
    json_schema: String,
    whitespace_pattern: Option<&str>,
) -> PyResult<String> {
    let value = serde_json::from_str(&json_schema).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a valid JSON string.")
    })?;
    json_schema::regex_from_value(&value, whitespace_pattern)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn outlines_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BOOLEAN", json_schema::BOOLEAN)?;
    m.add("DATE", json_schema::DATE)?;
    m.add("DATE_TIME", json_schema::DATE_TIME)?;
    m.add("INTEGER", json_schema::INTEGER)?;
    m.add("NULL", json_schema::NULL)?;
    m.add("NUMBER", json_schema::NUMBER)?;
    m.add("STRING", json_schema::STRING)?;
    m.add("STRING_INNER", json_schema::STRING_INNER)?;
    m.add("TIME", json_schema::TIME)?;
    m.add("UUID", json_schema::UUID)?;
    m.add("WHITESPACE", json_schema::WHITESPACE)?;
    m.add("EMAIL", json_schema::EMAIL)?;
    m.add("URI", json_schema::URI)?;

    m.add_function(wrap_pyfunction!(build_regex_from_schema_py, m)?)?;

    m.add_class::<PyIndex>()?;
    m.add_class::<PyVocabulary>()?;
    m.add_class::<PyGuide>()?;

    Ok(())
}
