//! Provides Guide python interface.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use outlines_core::prelude::*;

use crate::index::PyIndex;

#[pyclass(name = "Guide", module = "outlines_core")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyGuide {
    state: StateId,
    index: PyIndex,
}

#[pymethods]
impl PyGuide {
    #[new]
    fn __new__(index: PyIndex) -> Self {
        PyGuide {
            state: index.get_initial_state(),
            index,
        }
    }

    fn get_state(&self) -> StateId {
        self.state
    }

    fn get_tokens(&self) -> PyResult<Vec<TokenId>> {
        self.index
            .get_allowed_tokens(self.state)
            // Since Guide advances only through the states offered by the Index, it means
            // None here shouldn't happen and it's an issue at Index creation step
            .ok_or(PyErr::new::<PyValueError, _>(format!(
                "No allowed tokens available for the state {}",
                self.state
            )))
    }

    fn advance(&mut self, token_id: TokenId) -> PyResult<Vec<TokenId>> {
        match self.index.get_next_state(self.state, token_id) {
            Some(new_state) => {
                self.state = new_state;
                self.get_tokens()
            }
            None => Err(PyErr::new::<PyValueError, _>(format!(
                "No next state found for the current state: {} with token ID: {token_id}",
                self.state
            ))),
        }
    }

    fn is_finished(&self) -> bool {
        self.index.is_final_state(self.state)
    }

    fn __repr__(&self) -> String {
        format!(
            "Guide object with the state={:#?} and {:#?}",
            self.state, self.index
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Guide object with the state={} and {}",
            self.state, self.index.0
        )
    }

    fn __eq__(&self, other: &PyGuide) -> bool {
        self == other
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import_bound(py, "outlines_core")?.getattr("Guide")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(self, bincode::config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Serialization of Guide failed: {}", e))
                })?;
            Ok((cls.getattr("from_binary")?.to_object(py), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (guide, _): (PyGuide, usize) =
            bincode::decode_from_slice(&binary_data[..], bincode::config::standard()).map_err(
                |e| {
                    PyErr::new::<PyValueError, _>(format!("Deserialization of Guide failed: {}", e))
                },
            )?;
        Ok(guide)
    }
}
