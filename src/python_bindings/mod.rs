//! Provides tools and interfaces to integrate the crate's functionality with Python.

use std::sync::Arc;

use bincode::{config, Decode, Encode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::buffer::PyBuffer;
use pyo3::wrap_pyfunction;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tokenizers::FromPretrainedParameters;

use crate::v2_index::V2Index;
use crate::json_schema;
use crate::prelude::*;

macro_rules! type_name {
    ($obj:expr) => {
        // Safety: obj is always initialized and tp_name is a C-string
        unsafe { std::ffi::CStr::from_ptr((&*(&*$obj.as_ptr()).ob_type).tp_name) }
    };
}

#[pyclass(name = "Guide", module = "outlines_core.outlines_core_rs")]
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
    #[pyo3(signature = (mask=None))]
    fn get_tokens(&self, mask: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<TokenId>> {
        
        // If a mask reference is passed from parameters then it's filled with inner_mask value.
        // else, the inner mask is turned to a vec of TokenId and the vec is returned.
        // It allows no breaking change for outlines-core users. 
        // Of course, use the Guide with mask reference to get better speed perfomance.

        let  vec_result: Vec<TokenId> = Vec::new();

        let inner_mask_opt = self.index
            .get_allowed_tokens(self.state);
        
        if let Some(inner_mask) = inner_mask_opt {
            if let  Some(outer_mask_py) = mask{
                let buffer: PyBuffer<u64> = PyBuffer::get(outer_mask_py)?;
                if buffer.item_size() != std::mem::size_of::<u64>() {
                    return Err(PyErr::new::<PyValueError, _>(
                        "Buffer must contain u64 elements",
                    ));
                }
                let buffer_ptr = buffer.buf_ptr() as *mut u64;
                let buffer_len = buffer.len_bytes();
                unsafe {
                    let outer_mask = std::slice::from_raw_parts_mut(buffer_ptr, buffer_len);
                    outer_mask[..inner_mask.len()].copy_from_slice(inner_mask);
                }

            }else {
                
                return Ok(mask_to_list(inner_mask));
            }
        } else {
            // Since Guide advances only through the states offered by the Index, it means
            // None here shouldn't happen and it's an issue at Index creation step
            return Err(PyErr::new::<PyValueError, _>(format!(
                "No allowed tokens available for the state {}",
                self.state
            )));
        }   
        Ok(vec_result)
    }
    #[pyo3(signature = (token_id, mask=None))]
    fn advance(&mut self, token_id: TokenId, mask: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<TokenId>> {
        match self.index.get_next_state(self.state, token_id) {
            Some(new_state) => {
                self.state = new_state;
                self.get_tokens(mask)
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
            let cls = PyModule::import(py, "outlines_core.outlines_core_rs")?.getattr("Guide")?;
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

#[pyclass(name = "Index", module = "outlines_core.outlines_core_rs")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyIndex(Arc<V2Index>);

#[pymethods]
impl PyIndex {
    #[new]
    fn __new__(py: Python<'_>, regex: &str, vocabulary: &PyVocabulary) -> PyResult<Self> {
        py.allow_threads(|| {
            V2Index::new(regex, &vocabulary.0)
                .map(|x| PyIndex(Arc::new(x)))
                .map_err(Into::into)
        })
    }

    fn get_allowed_tokens(&self, state: StateId) -> Option<&Vec<u64>> {
        
        self.0.allowed_tokens(&state)
      
    }

    fn get_next_state(&self, state: StateId, token_id: TokenId) -> Option<StateId> {
        self.0.next_state(&state, &token_id)
    }

    fn is_final_state(&self, state: StateId) -> bool {
        self.0.is_final_state(&state)
    }

    fn get_final_states(&self) -> HashSet<StateId> {
        self.0.final_states().clone()
    }

    /// WARNING : VERY COSTLY FUNCTION
    fn get_transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>> {
        self.0.transitions().clone()
    }

    fn get_initial_state(&self) -> StateId {
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
            let cls = PyModule::import(py, "outlines_core.outlines_core_rs")?.getattr("Index")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&self.0, config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Serialization of Index failed: {}", e))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (index, _): (V2Index, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Deserialization of Index failed: {}", e))
            })?;
        Ok(PyIndex(Arc::new(index)))
    }
}

#[pyclass(name = "Vocabulary", module = "outlines_core.outlines_core_rs")]
#[derive(Clone, Debug, Encode, Decode)]
pub struct PyVocabulary(Vocabulary);

#[pymethods]
impl PyVocabulary {
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
            let cls =
                PyModule::import(py, "outlines_core.outlines_core_rs")?.getattr("Vocabulary")?;
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
fn outlines_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
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


fn mask_to_list(inner_mask: &Vec<u64>)-> Vec<TokenId>{
    let mut result = Vec::with_capacity(inner_mask.iter().map(|b| b.count_ones() as usize).sum());
    for (chunk_index, &chunk) in inner_mask.iter().enumerate() {
        let base_id = (chunk_index * 64) as u32;
        let mut bit_pos = chunk;
        
        while bit_pos != 0 {
            let bit = bit_pos.trailing_zeros(); 
            result.push(base_id + bit);
            bit_pos &= bit_pos - 1; 
        }
    }
    return result;
}