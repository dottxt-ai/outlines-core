mod json_schema;
mod regex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use regex::_walk_fsm;
use regex::create_fsm_index_end_to_end;
use regex::get_token_transition_keys;
use regex::get_vocabulary_transition_keys;
use regex::state_scan_tokens;
use regex::FSMInfo;
use serde_json::Value;

#[pymodule]
fn outlines_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_walk_fsm, m)?)?;
    m.add_function(wrap_pyfunction!(state_scan_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(get_token_transition_keys, m)?)?;
    m.add_function(wrap_pyfunction!(get_vocabulary_transition_keys, m)?)?;
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end, m)?)?;

    m.add_class::<FSMInfo>()?;

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

    m.add_function(wrap_pyfunction!(build_regex_from_schema, m)?)?;
    m.add_function(wrap_pyfunction!(to_regex, m)?)?;

    Ok(())
}

#[pyfunction(name = "build_regex_from_schema")]
#[pyo3(signature = (json, whitespace_pattern=None))]
pub fn build_regex_from_schema(json: String, whitespace_pattern: Option<&str>) -> PyResult<String> {
    json_schema::build_regex_from_schema(&json, whitespace_pattern)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction(name = "to_regex")]
#[pyo3(signature = (json, whitespace_pattern=None))]
pub fn to_regex(json: Bound<PyDict>, whitespace_pattern: Option<&str>) -> PyResult<String> {
    let json_value: Value = serde_pyobject::from_pyobject(json).unwrap();
    json_schema::to_regex(&json_value, whitespace_pattern, &json_value)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
