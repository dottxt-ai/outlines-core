//! Provides Vocabulary python interface.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use outlines_core::prelude::*;
use outlines_core::vocabulary::Vocabulary;

use crate::Error;

macro_rules! type_name {
    ($obj:expr) => {
        // Safety: obj is always initialized and tp_name is a C-string
        unsafe { std::ffi::CStr::from_ptr((&*(&*$obj.as_ptr()).ob_type).tp_name) }
    };
}

#[pyclass(name = "Vocabulary", module = "outlines_core")]
#[derive(Clone, Debug, Encode, Decode)]
pub struct PyVocabulary(pub Vocabulary);

#[pymethods]
impl PyVocabulary {
    #[new]
    fn __new__(py: Python<'_>, eos_token_id: TokenId, map: Py<PyAny>) -> PyResult<PyVocabulary> {
        if let Ok(dict) = map.extract::<HashMap<String, Vec<TokenId>>>(py) {
            return Ok(PyVocabulary(
                Vocabulary::try_from((eos_token_id, dict))
                    .map_err(|e| <Error as Into<PyErr>>::into(Error::from(e)))?,
            ));
        }
        if let Ok(dict) = map.extract::<HashMap<Vec<u8>, Vec<TokenId>>>(py) {
            return Ok(PyVocabulary(
                Vocabulary::try_from((eos_token_id, dict))
                    .map_err(|e| <Error as Into<PyErr>>::into(Error::from(e)))?,
            ));
        }

        let message = "Expected a dict with keys of type str or bytes and values of type list[int]";
        let tname = type_name!(map).to_string_lossy();
        if tname == "dict" {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Dict keys or/and values of the wrong types. {message}"
            )))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "{message}, got {tname}"
            )))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (model, revision=None, token=None))]
    fn from_pretrained(
        model: String,
        revision: Option<String>,
        token: Option<String>,
    ) -> PyResult<PyVocabulary> {
        let mut params = FromPretrainedParameters::default();
        if let Some(r) = revision {
            params.revision = r
        }
        if token.is_some() {
            params.token = token
        }
        let v = Vocabulary::from_pretrained(model.as_str(), Some(params))
            .map_err(|e| <Error as Into<PyErr>>::into(Error::from(e)))?;
        Ok(PyVocabulary(v))
    }

    fn insert(&mut self, py: Python<'_>, token: Py<PyAny>, token_id: TokenId) -> PyResult<()> {
        if let Ok(t) = token.extract::<String>(py) {
            return Ok(self
                .0
                .try_insert(t, token_id)
                .map_err(|e| <Error as Into<PyErr>>::into(Error::from(e)))?);
        }
        if let Ok(t) = token.extract::<Token>(py) {
            return Ok(self
                .0
                .try_insert(t, token_id)
                .map_err(|e| <Error as Into<PyErr>>::into(Error::from(e)))?);
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    fn remove(&mut self, py: Python<'_>, token: Py<PyAny>) -> PyResult<()> {
        if let Ok(t) = token.extract::<String>(py) {
            self.0.remove(t);
            return Ok(());
        }
        if let Ok(t) = token.extract::<Token>(py) {
            self.0.remove(t);
            return Ok(());
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    fn get(&self, py: Python<'_>, token: Py<PyAny>) -> PyResult<Option<Vec<TokenId>>> {
        if let Ok(t) = token.extract::<String>(py) {
            return Ok(self.0.token_ids(t.into_bytes()).cloned());
        }
        if let Ok(t) = token.extract::<Token>(py) {
            return Ok(self.0.token_ids(&t).cloned());
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    fn get_eos_token_id(&self) -> TokenId {
        self.0.eos_token_id()
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __eq__(&self, other: &PyVocabulary) -> bool {
        self.0 == other.0
    }

    fn __len__(&self) -> usize {
        self.0.tokens().len()
    }

    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyVocabulary(self.0.clone())
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import_bound(py, "outlines_core")?.getattr("Vocabulary")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(self, bincode::config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "Serialization of Vocabulary failed: {}",
                        e
                    ))
                })?;
            Ok((cls.getattr("from_binary")?.to_object(py), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (guide, _): (PyVocabulary, usize) = bincode::decode_from_slice(
            &binary_data[..],
            bincode::config::standard(),
        )
        .map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Deserialization of Vocabulary failed: {}", e))
        })?;
        Ok(guide)
    }
}
