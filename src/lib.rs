mod regex;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use regex::_walk_fsm;
use regex::create_fsm_index_end_to_end;
use regex::get_token_transition_keys;
use regex::get_vocabulary_transition_keys;
use regex::state_scan_tokens;
use regex::FSMInfo;

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
