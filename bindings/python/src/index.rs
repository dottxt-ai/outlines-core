//! Provides Index python interface.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use outlines_core::index::Index;
use outlines_core::prelude::*;

use crate::vocabulary::PyVocabulary;
use crate::Error;

#[pyclass(name = "Index", module = "outlines_core")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyIndex(pub Arc<Index>);

#[pymethods]
impl PyIndex {
    #[new]
    fn __new__(py: Python<'_>, regex: &str, vocabulary: &PyVocabulary) -> PyResult<Self> {
        py.allow_threads(|| {
            Index::new(regex, &vocabulary.0)
                .map(|x| PyIndex(Arc::new(x)))
                .map_err(|e| Error::from(e).into())
        })
    }

    pub fn get_allowed_tokens(&self, state: StateId) -> Option<Vec<TokenId>> {
        self.0.allowed_tokens(&state)
    }

    pub fn get_next_state(&self, state: StateId, token_id: TokenId) -> Option<StateId> {
        self.0.next_state(&state, &token_id)
    }

    pub fn is_final_state(&self, state: StateId) -> bool {
        self.0.is_final_state(&state)
    }

    pub fn get_final_states(&self) -> HashSet<StateId> {
        self.0.final_states().clone()
    }

    pub fn get_transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>> {
        self.0.transitions().clone()
    }

    pub fn get_initial_state(&self) -> StateId {
        self.0.initial_state()
    }
    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __eq__(&self, other: &PyIndex) -> bool {
        *self.0 == *other.0
    }

    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyIndex(Arc::new((*self.0).clone()))
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import_bound(py, "outlines_core")?.getattr("Index")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&self.0, bincode::config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Serialization of Index failed: {}", e))
                })?;
            Ok((cls.getattr("from_binary")?.to_object(py), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (index, _): (Index, usize) =
            bincode::decode_from_slice(&binary_data[..], bincode::config::standard()).map_err(
                |e| {
                    PyErr::new::<PyValueError, _>(format!("Deserialization of Index failed: {}", e))
                },
            )?;
        Ok(PyIndex(Arc::new(index)))
    }
}
