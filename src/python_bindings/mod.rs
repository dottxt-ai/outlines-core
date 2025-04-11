//! Provides tools and interfaces to integrate the crate's functionality with Python.

use std::sync::Arc;

use bincode::{config, Decode, Encode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::wrap_pyfunction;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tokenizers::FromPretrainedParameters;

use crate::index::Index;
use crate::json_schema;
use crate::prelude::*;

macro_rules! type_name {
    ($obj:expr) => {
        // Safety: obj is always initialized and tp_name is a C-string
        unsafe { std::ffi::CStr::from_ptr((&*(&*$obj.as_ptr()).ob_type).tp_name) }
    };
}

/// Guide object based on Index.
#[pyclass(name = "Guide", module = "outlines_core")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyGuide {
    state: StateId,
    index: PyIndex,
}

#[pymethods]
impl PyGuide {
    /// Creates a Guide object based on Index.
    #[new]
    fn __new__(index: PyIndex) -> Self {
        PyGuide {
            state: index.get_initial_state(),
            index,
        }
    }

    /// Retrieves current state id of the Guide.
    fn get_state(&self) -> StateId {
        self.state
    }

    /// Gets the list of allowed tokens for the current state.
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

    /// Guide moves to the next state provided by the token id and returns a list of allowed tokens, unless return_tokens is False.
    #[pyo3(signature = (token_id, return_tokens=None))]
    fn advance(
        &mut self,
        token_id: TokenId,
        return_tokens: Option<bool>,
    ) -> PyResult<Option<Vec<TokenId>>> {
        match self.index.get_next_state(self.state, token_id) {
            Some(new_state) => {
                self.state = new_state;
                if return_tokens.unwrap_or(true) {
                    self.get_tokens().map(Some)
                } else {
                    Ok(None)
                }
            }
            None => Err(PyErr::new::<PyValueError, _>(format!(
                "No next state found for the current state: {} with token ID: {token_id}",
                self.state
            ))),
        }
    }

    /// Checks if the automaton is in a final state.
    fn is_finished(&self) -> bool {
        self.index.is_final_state(self.state)
    }

    /// Write the mask of allowed tokens into the memory specified by data_ptr.
    /// Size of the memory to be written to is indicated by `numel`, and `element_size`.
    /// `element_size` must be 4.
    ///
    /// `data_ptr` should be the data ptr to a `torch.tensor`, or `np.ndarray`, `mx.array` or other
    /// contiguous memory array.
    fn write_mask_into(&self, data_ptr: usize, numel: usize, element_size: usize) -> PyResult<()> {
        let expected_elements = self.index.0.vocab_size().div_ceil(32);
        if element_size != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid element size: got {} bytes per element, expected 4 bytes (32-bit integer).",
                    element_size
                ),
            ));
        } else if data_ptr == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid data pointer: received a null pointer.",
            ));
        } else if data_ptr % 4 != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid data pointer alignment: pointer address {} is not a multiple of 4.",
                data_ptr
            )));
        } else if numel < expected_elements {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid buffer size: got {} elements ({} bytes), expected {} elements ({} bytes). \
                    Ensure that the mask tensor has shape (1, (vocab_size + 31) // 32) and uses 32-bit integers.",
                    numel,
                    numel * element_size,
                    expected_elements,
                    expected_elements * 4
                )
            ));
        }
        unsafe {
            std::ptr::write_bytes(data_ptr as *mut u8, 0, numel * 4);
        }
        if let Some(tokens) = self.index.0.allowed_tokens_iter(&self.state) {
            let slice = unsafe { std::slice::from_raw_parts_mut(data_ptr as *mut u32, numel) };
            for &token in tokens {
                let bucket = (token as usize) / 32;
                if bucket < slice.len() {
                    slice[bucket] |= 1 << ((token as usize) % 32);
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.state = self.index.get_initial_state();
    }

    /// Gets the debug string representation of the guide.
    fn __repr__(&self) -> String {
        format!(
            "Guide object with the state={:#?} and {:#?}",
            self.state, self.index
        )
    }

    /// Gets the string representation of the guide.
    fn __str__(&self) -> String {
        format!(
            "Guide object with the state={} and {}",
            self.state, self.index.0
        )
    }

    /// Compares whether two guides are the same.
    fn __eq__(&self, other: &PyGuide) -> bool {
        self == other
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import(py, "outlines_core")?.getattr("Guide")?;
            let binary_data: Vec<u8> =
                bincode::encode_to_vec(self, config::standard()).map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Serialization of Guide failed: {}", e))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (guide, _): (PyGuide, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Deserialization of Guide failed: {}", e))
            })?;
        Ok(guide)
    }
}

/// Index object based on regex and vocabulary.
#[pyclass(name = "Index", module = "outlines_core")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyIndex(Arc<Index>);

#[pymethods]
impl PyIndex {
    /// Creates an index from a regex and vocabulary.
    #[new]
    fn __new__(py: Python<'_>, regex: &str, vocabulary: &PyVocabulary) -> PyResult<Self> {
        py.allow_threads(|| {
            Index::new(regex, &vocabulary.0)
                .map(|x| PyIndex(Arc::new(x)))
                .map_err(Into::into)
        })
    }

    /// Returns allowed tokens in this state.
    fn get_allowed_tokens(&self, state: StateId) -> Option<Vec<TokenId>> {
        self.0.allowed_tokens(&state)
    }

    /// Updates the state.
    fn get_next_state(&self, state: StateId, token_id: TokenId) -> Option<StateId> {
        self.0.next_state(&state, &token_id)
    }

    /// Determines whether the current state is a final state.
    fn is_final_state(&self, state: StateId) -> bool {
        self.0.is_final_state(&state)
    }

    /// Get all final states.
    fn get_final_states(&self) -> HashSet<StateId> {
        self.0.final_states().clone()
    }

    /// Returns the Index as a Python Dict object.
    fn get_transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>> {
        self.0.transitions().clone()
    }

    /// Returns the ID of the initial state of the index.
    fn get_initial_state(&self) -> StateId {
        self.0.initial_state()
    }

    /// Gets the debug string representation of the index.
    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    /// Gets the string representation of the index.
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Compares whether two indexes are the same.
    fn __eq__(&self, other: &PyIndex) -> bool {
        *self.0 == *other.0
    }

    /// Makes a deep copy of the Index.
    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyIndex(Arc::new((*self.0).clone()))
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import(py, "outlines_core")?.getattr("Index")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&self.0, config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Serialization of Index failed: {}", e))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (index, _): (Index, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Deserialization of Index failed: {}", e))
            })?;
        Ok(PyIndex(Arc::new(index)))
    }
}

/// LLM vocabulary.
#[pyclass(name = "Vocabulary", module = "outlines_core")]
#[derive(Clone, Debug, Encode, Decode)]
pub struct PyVocabulary(Vocabulary);

#[pymethods]
impl PyVocabulary {
    /// Creates a vocabulary from eos token id and a map of tokens to token ids.
    #[new]
    fn __new__(py: Python<'_>, eos_token_id: TokenId, map: Py<PyAny>) -> PyResult<PyVocabulary> {
        if let Ok(dict) = map.extract::<HashMap<String, Vec<TokenId>>>(py) {
            return Ok(PyVocabulary(Vocabulary::try_from((eos_token_id, dict))?));
        }
        if let Ok(dict) = map.extract::<HashMap<Vec<u8>, Vec<TokenId>>>(py) {
            return Ok(PyVocabulary(Vocabulary::try_from((eos_token_id, dict))?));
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

    /// Creates the vocabulary of a pre-trained model.
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
        let v = Vocabulary::from_pretrained(model.as_str(), Some(params))?;
        Ok(PyVocabulary(v))
    }

    /// Inserts new token with token_id or extends list of token_ids if token already present.
    fn insert(&mut self, py: Python<'_>, token: Py<PyAny>, token_id: TokenId) -> PyResult<()> {
        if let Ok(t) = token.extract::<String>(py) {
            return Ok(self.0.try_insert(t, token_id)?);
        }
        if let Ok(t) = token.extract::<Token>(py) {
            return Ok(self.0.try_insert(t, token_id)?);
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    /// Removes a token from vocabulary.
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

    /// Gets token ids of a given token.
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

    /// Gets the end of sentence token id.
    fn get_eos_token_id(&self) -> TokenId {
        self.0.eos_token_id()
    }

    /// Gets the debug string representation of the vocabulary.
    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    /// Gets the string representation of the vocabulary.
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Compares whether two vocabularies are the same.
    fn __eq__(&self, other: &PyVocabulary) -> bool {
        self.0 == other.0
    }

    /// Returns length of Vocabulary's tokens, excluding EOS token.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Makes a deep copy of the Vocabulary.
    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyVocabulary(self.0.clone())
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import(py, "outlines_core")?.getattr("Vocabulary")?;
            let binary_data: Vec<u8> =
                bincode::encode_to_vec(self, config::standard()).map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "Serialization of Vocabulary failed: {}",
                        e
                    ))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (guide, _): (PyVocabulary, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!(
                    "Deserialization of Vocabulary failed: {}",
                    e
                ))
            })?;
        Ok(guide)
    }
}

/// Creates regex string from JSON schema with optional whitespace pattern.
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

fn register_child_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "json_schema")?;
    parent_module.add_submodule(&m)?;

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
    m.add_function(wrap_pyfunction!(build_regex_from_schema_py, &m)?)?;

    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules_bind = sys.as_ref().getattr("modules")?;
    let sys_modules = sys_modules_bind.downcast::<PyDict>()?;
    sys_modules.set_item("outlines_core.json_schema", &m)?;

    Ok(())
}

/// This package provides core functionality for structured generation, providing a convenient way to:
///
/// - build regular expressions from JSON schemas
///
/// - construct an Index object by combining a Vocabulary and regular expression to efficiently map tokens from a given Vocabulary to state transitions in a finite-state automation
#[pymodule]
fn outlines_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let version = env!("CARGO_PKG_VERSION");
    m.add("__version__", version)?;

    m.add_class::<PyIndex>()?;
    m.add_class::<PyVocabulary>()?;
    m.add_class::<PyGuide>()?;
    register_child_module(m)?;

    Ok(())
}
