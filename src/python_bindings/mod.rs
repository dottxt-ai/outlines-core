use crate::index::Index;
use crate::json_schema;
use crate::prelude::*;
use bincode::config;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde_json::Value;


#[pyclass(name = "Index", module = "outlines_core.fsm.outlines_core_rs")]
pub struct PyIndex(Index);

#[pymethods]
impl PyIndex {
    #[new]
    fn new(py: Python<'_>, regex: &str, vocabulary: &PyVocabulary) -> PyResult<Self> {
        py.allow_threads(|| {
            Index::new(regex, &vocabulary.0)
                .map(PyIndex)
                .map_err(Into::into)
        })
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import_bound(py, "outlines_core.fsm.outlines_core_rs")?
                .getattr("Index")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&self.0, config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("Serialization of Index failed: {}", e))
                })?;
            Ok((cls.getattr("from_binary")?.to_object(py), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (index, _): (Index, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Deserialization of Index failed: {}", e))
            })?;
        Ok(PyIndex(index))
    }

    fn get_allowed_tokens(&self, state: u32) -> Option<Vec<u32>> {
        self.0.allowed_tokens(state)
    }

    fn get_next_state(&self, state: u32, token_id: u32) -> Option<u32> {
        self.0.next_state(state, token_id)
    }

    fn is_final_state(&self, state: u32) -> bool {
        self.0.is_final(state)
    }

    fn final_states(&self) -> HashSet<StateId> {
        self.0.final_states().clone()
    }

    fn get_transitions(&self) -> HashMap<u32, HashMap<u32, u32>> {
        self.0.transitions().clone()
    }

    fn get_initial_state(&self) -> u32 {
        self.0.initial()
    }
}

#[pyfunction(name = "build_regex_from_schema")]
#[pyo3(signature = (json, whitespace_pattern=None))]
pub fn build_regex_from_schema_py(
    json: String,
    whitespace_pattern: Option<&str>,
) -> PyResult<String> {
    json_schema::build_regex_from_schema(&json, whitespace_pattern)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction(name = "to_regex")]
#[pyo3(signature = (json, whitespace_pattern=None))]
pub fn to_regex_py(json: Bound<PyDict>, whitespace_pattern: Option<&str>) -> PyResult<String> {
    let json_value: Value = serde_pyobject::from_pyobject(json)?;
    json_schema::to_regex(&json_value, whitespace_pattern)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "Vocabulary")]
pub struct PyVocabulary(Vocabulary);

#[pymethods]
impl PyVocabulary {
    #[staticmethod]
    fn from_dict(map: HashMap<String, Vec<TokenId>>) -> PyVocabulary {
        PyVocabulary(Vocabulary::from(map))
    }

    #[staticmethod]
    fn from_dict_with_eos_token_id(
        map: HashMap<String, Vec<TokenId>>,
        eos_token_id: TokenId,
    ) -> PyVocabulary {
        let v = Vocabulary::from(map).with_eos_token_id(Some(eos_token_id));
        PyVocabulary(v)
    }

    #[staticmethod]
    fn from_pretrained(model: String) -> PyResult<PyVocabulary> {
        let v = Vocabulary::from_pretrained(model.as_str(), None)?;
        Ok(PyVocabulary(v))
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
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
    m.add_function(wrap_pyfunction!(to_regex_py, m)?)?;

    m.add_class::<PyIndex>()?;
    m.add_class::<PyVocabulary>()?;

    Ok(())
}
