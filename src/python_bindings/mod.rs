use crate::regex::get_token_transition_keys_internal;
use crate::regex::get_vocabulary_transition_keys_internal;
use crate::regex::state_scan_tokens_internal;
use crate::regex::walk_fsm_internal;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use std::collections::{HashMap, HashSet};

#[pyclass]
pub struct FSMInfo {
    #[pyo3(get)]
    initial: u32,
    #[pyo3(get)]
    finals: HashSet<u32>,
    #[pyo3(get)]
    transitions: HashMap<(u32, u32), u32>,
    #[pyo3(get)]
    alphabet_anything_value: u32,
    #[pyo3(get)]
    alphabet_symbol_mapping: HashMap<String, u32>,
}

#[pymethods]
impl FSMInfo {
    #[new]
    fn new(
        initial: u32,
        finals: HashSet<u32>,
        transitions: HashMap<(u32, u32), u32>,
        alphabet_anything_value: u32,
        alphabet_symbol_mapping: HashMap<String, u32>,
    ) -> Self {
        Self {
            initial,
            finals,
            transitions,
            alphabet_anything_value,
            alphabet_symbol_mapping,
        }
    }
}

#[pyfunction(name = "_walk_fsm")]
#[pyo3(
    text_signature = "(fsm_transitions, fsm_initial, fsm_finals, token_transition_keys, start_state, full_match)"
)]
pub fn _walk_fsm(
    fsm_transitions: HashMap<(u32, u32), u32>,
    fsm_initial: u32,
    fsm_finals: HashSet<u32>,
    token_transition_keys: Vec<u32>,
    start_state: u32,
    full_match: bool,
) -> PyResult<Vec<u32>> {
    Ok(walk_fsm_internal(
        &fsm_transitions,
        fsm_initial,
        &fsm_finals,
        &token_transition_keys,
        start_state,
        full_match,
    ))
}

#[pyfunction(name = "state_scan_tokens")]
#[pyo3(
    text_signature = "(fsm_transitions, fsm_initial, fsm_finals, vocabulary, vocabulary_transition_keys, start_state)"
)]
pub fn state_scan_tokens(
    fsm_transitions: HashMap<(u32, u32), u32>,
    fsm_initial: u32,
    fsm_finals: HashSet<u32>,
    vocabulary: Vec<(String, Vec<u32>)>,
    vocabulary_transition_keys: Vec<Vec<u32>>,
    start_state: u32,
) -> PyResult<HashSet<(u32, u32)>> {
    Ok(state_scan_tokens_internal(
        &fsm_transitions,
        fsm_initial,
        &fsm_finals,
        &vocabulary,
        &vocabulary_transition_keys,
        start_state,
    ))
}

#[pyfunction(name = "get_token_transition_keys")]
#[pyo3(text_signature = "(alphabet_symbol_mapping, alphabet_anything_value, token_str)")]
pub fn get_token_transition_keys(
    alphabet_symbol_mapping: HashMap<String, u32>,
    alphabet_anything_value: u32,
    token_str: String,
) -> PyResult<Vec<u32>> {
    Ok(get_token_transition_keys_internal(
        &alphabet_symbol_mapping,
        alphabet_anything_value,
        &token_str,
    ))
}

#[pyfunction(name = "get_vocabulary_transition_keys")]
#[pyo3(
    text_signature = "(alphabet_symbol_mapping, alphabet_anything_value, vocabulary, frozen_tokens)"
)]
pub fn get_vocabulary_transition_keys(
    alphabet_symbol_mapping: HashMap<String, u32>,
    alphabet_anything_value: u32,
    vocabulary: Vec<(String, Vec<u32>)>,
    frozen_tokens: HashSet<String>,
) -> PyResult<Vec<Vec<u32>>> {
    Ok(get_vocabulary_transition_keys_internal(
        &alphabet_symbol_mapping,
        alphabet_anything_value,
        &vocabulary,
        &frozen_tokens,
    ))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "create_fsm_index_end_to_end")]
#[pyo3(text_signature = "(fsm_info, vocabulary, frozen_tokens)")]
pub fn create_fsm_index_end_to_end<'py>(
    py: Python<'py>,
    fsm_info: &FSMInfo,
    vocabulary: Vec<(String, Vec<u32>)>,
    frozen_tokens: HashSet<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let states_to_token_subsets = PyDict::new_bound(py);
    let mut seen: HashSet<u32> = HashSet::new();
    let mut next_states: HashSet<u32> = HashSet::from_iter(vec![fsm_info.initial]);

    let vocabulary_transition_keys = get_vocabulary_transition_keys_internal(
        &fsm_info.alphabet_symbol_mapping,
        fsm_info.alphabet_anything_value,
        &vocabulary,
        &frozen_tokens,
    );

    while let Some(start_state) = next_states.iter().cloned().next() {
        next_states.remove(&start_state);

        // TODO: Return Pydict directly at construction
        let token_ids_end_states = state_scan_tokens_internal(
            &fsm_info.transitions,
            fsm_info.initial,
            &fsm_info.finals,
            &vocabulary,
            &vocabulary_transition_keys,
            start_state,
        );

        for (token_id, end_state) in token_ids_end_states {
            if let Ok(Some(existing_dict)) = states_to_token_subsets.get_item(start_state) {
                existing_dict.set_item(token_id, end_state).unwrap();
            } else {
                let new_dict = PyDict::new_bound(py);
                new_dict.set_item(token_id, end_state).unwrap();
                states_to_token_subsets
                    .set_item(start_state, new_dict)
                    .map_err(|e| PyValueError::new_err(format!("Failed to set item: {}", e)))?;
            }

            if !seen.contains(&end_state) {
                next_states.insert(end_state);
            }
        }

        seen.insert(start_state);
    }

    Ok(states_to_token_subsets)
}

#[pymodule]
fn outlines_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_walk_fsm, m)?)?;
    m.add_function(wrap_pyfunction!(state_scan_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(get_token_transition_keys, m)?)?;
    m.add_function(wrap_pyfunction!(get_vocabulary_transition_keys, m)?)?;
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end, m)?)?;

    m.add_class::<FSMInfo>()?;

    Ok(())
}
