use crate::interegular::fsm::Fsm;
use crate::interegular::patterns::parse_pattern;
use crate::interegular::patterns::RegexElement;
use crate::json_schema;
use crate::primitives::TransitionKey;
use crate::regex::get_token_transition_keys;
use crate::regex::get_vocabulary_transition_keys;
use crate::regex::state_scan_tokens;
use crate::regex::walk_fsm;
use crate::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[pyclass]
pub struct FSMInfo {
    #[pyo3(get)]
    initial: State,
    #[pyo3(get)]
    finals: HashSet<State>,
    #[pyo3(get)]
    transitions: HashMap<(State, TransitionKey), State>,
    #[pyo3(get)]
    alphabet_anything_value: TransitionKey,
    #[pyo3(get)]
    alphabet_symbol_mapping: HashMap<String, TransitionKey>,
}

#[pymethods]
impl FSMInfo {
    #[new]
    fn new(
        initial: State,
        finals: HashSet<State>,
        transitions: HashMap<(State, TransitionKey), State>,
        alphabet_anything_value: TransitionKey,
        alphabet_symbol_mapping: HashMap<String, TransitionKey>,
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
    let json_value: Value = serde_pyobject::from_pyobject(json).unwrap();
    json_schema::to_regex(&json_value, whitespace_pattern, &json_value)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction(name = "_walk_fsm")]
#[pyo3(
    text_signature = "(fsm_transitions, fsm_initial, fsm_finals, token_transition_keys, start_state, full_match)"
)]
pub fn walk_fsm_py(
    fsm_transitions: HashMap<(State, TransitionKey), State>,
    fsm_initial: State,
    fsm_finals: HashSet<State>,
    token_transition_keys: Vec<TransitionKey>,
    start_state: State,
    full_match: bool,
) -> PyResult<Vec<State>> {
    Ok(walk_fsm(
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
pub fn state_scan_tokens_py(
    fsm_transitions: HashMap<(State, TransitionKey), State>,
    fsm_initial: State,
    fsm_finals: HashSet<State>,
    vocabulary: &PyVocabulary,
    vocabulary_transition_keys: HashMap<String, Vec<TransitionKey>>,
    start_state: State,
) -> PyResult<HashSet<(TokenId, State)>> {
    Ok(state_scan_tokens(
        &fsm_transitions,
        fsm_initial,
        &fsm_finals,
        &vocabulary.0,
        &vocabulary_transition_keys,
        start_state,
    ))
}

#[pyfunction(name = "get_token_transition_keys")]
#[pyo3(text_signature = "(alphabet_symbol_mapping, alphabet_anything_value, token_str)")]
pub fn get_token_transition_keys_py(
    alphabet_symbol_mapping: HashMap<String, TransitionKey>,
    alphabet_anything_value: TransitionKey,
    token_str: String,
) -> PyResult<Vec<TransitionKey>> {
    Ok(get_token_transition_keys(
        &alphabet_symbol_mapping,
        alphabet_anything_value,
        &token_str,
    ))
}

#[pyfunction(name = "get_vocabulary_transition_keys")]
#[pyo3(
    text_signature = "(alphabet_symbol_mapping, alphabet_anything_value, vocabulary, frozen_tokens)"
)]
pub fn get_vocabulary_transition_keys_py(
    alphabet_symbol_mapping: HashMap<String, TransitionKey>,
    alphabet_anything_value: TransitionKey,
    vocabulary: &PyVocabulary,
    frozen_tokens: HashSet<String>,
) -> PyResult<HashMap<String, Vec<TransitionKey>>> {
    Ok(get_vocabulary_transition_keys(
        &alphabet_symbol_mapping,
        alphabet_anything_value,
        &vocabulary.0,
        &frozen_tokens,
    ))
}

#[pyfunction(name = "create_fsm_index_end_to_end")]
#[pyo3(text_signature = "(fsm_info, vocabulary, frozen_tokens)")]
pub fn create_fsm_index_end_to_end_py<'py>(
    py: Python<'py>,
    fsm_info: &FSMInfo,
    vocabulary: &PyVocabulary,
    frozen_tokens: HashSet<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let states_to_token_subsets = PyDict::new_bound(py);
    let mut seen: HashSet<State> = HashSet::new();
    let mut next_states: HashSet<State> = HashSet::from_iter(vec![fsm_info.initial]);

    let vocabulary_transition_keys = get_vocabulary_transition_keys(
        &fsm_info.alphabet_symbol_mapping,
        fsm_info.alphabet_anything_value,
        &vocabulary.0,
        &frozen_tokens,
    );

    while let Some(start_state) = next_states.iter().cloned().next() {
        next_states.remove(&start_state);

        // TODO: Return Pydict directly at construction
        let token_ids_end_states = state_scan_tokens(
            &fsm_info.transitions,
            fsm_info.initial,
            &fsm_info.finals,
            &vocabulary.0,
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
                    .unwrap();
            }

            if !seen.contains(&end_state) {
                next_states.insert(end_state);
            }
        }

        seen.insert(start_state);
    }

    Ok(states_to_token_subsets)
}

#[pyclass(name = "Vocabulary")]
pub struct PyVocabulary(Vocabulary);

#[pymethods]
impl PyVocabulary {
    #[staticmethod]
    fn from_dict(map: HashMap<Token, Vec<TokenId>>) -> PyVocabulary {
        PyVocabulary(Vocabulary::from(map))
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[pyclass]
#[derive(Clone)]
struct PyLiteral {
    #[pyo3(get)]
    value: char,
}

#[pymethods]
impl PyLiteral {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Literal('{}')", self.value))
    }
}

#[pyclass]
#[derive(Clone)]
struct PyCharGroup {
    #[pyo3(get)]
    chars: Vec<char>,
    #[pyo3(get)]
    inverted: bool,
}

#[pymethods]
impl PyCharGroup {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CharGroup(chars='{:?}', inverted={})",
            self.chars, self.inverted
        ))
    }
}

#[pyclass]
struct PyRepeated {
    #[pyo3(get)]
    element: PyObject,
    #[pyo3(get)]
    min: usize,
    #[pyo3(get)]
    max: Option<usize>,
}

#[pymethods]
impl PyRepeated {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Repeated(element='{}', min={}, max={:?})",
            self.element, self.min, self.max
        ))
    }
}

#[pyclass]
struct PyConcatenation {
    #[pyo3(get)]
    elements: Vec<PyObject>,
}

#[pymethods]
impl PyConcatenation {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Concatenation(elements='{:?}')", self.elements))
    }
}

#[pyclass]
struct PyAlternation {
    #[pyo3(get)]
    elements: Vec<PyObject>,
}

#[pymethods]
impl PyAlternation {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Alternation(elements='{:?}')", self.elements))
    }
}

#[pyclass]
struct PyCapture {
    #[pyo3(get)]
    element: PyObject,
}

#[pymethods]
impl PyCapture {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Capture(element='{}')", self.element))
    }
}

#[pyclass]
struct PyGroup {
    #[pyo3(get)]
    element: PyObject,
}

#[pymethods]
impl PyGroup {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Group(element='{}')", self.element))
    }
}

#[pyclass]
#[derive(Clone)]
struct PyAnchor {
    #[pyo3(get)]
    anchor_type: String,
}

#[pymethods]
impl PyAnchor {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Anchor(anchor_type='{}')", self.anchor_type))
    }
}

#[pyclass]
struct PyFlag {
    #[pyo3(get)]
    element: PyObject,
    #[pyo3(get)]
    added: Vec<String>,
    #[pyo3(get)]
    removed: Vec<String>,
}

impl Clone for PyRepeated {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyRepeated {
            element: self.element.clone_ref(py),
            min: self.min,
            max: self.max,
        })
    }
}

impl Clone for PyConcatenation {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyConcatenation {
            elements: self.elements.iter().map(|e| e.clone_ref(py)).collect(),
        })
    }
}

impl Clone for PyAlternation {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyAlternation {
            elements: self.elements.iter().map(|e| e.clone_ref(py)).collect(),
        })
    }
}

impl Clone for PyCapture {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyCapture {
            element: self.element.clone_ref(py),
        })
    }
}

impl Clone for PyGroup {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyGroup {
            element: self.element.clone_ref(py),
        })
    }
}

impl Clone for PyFlag {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyFlag {
            element: self.element.clone_ref(py),
            added: self.added.clone(),
            removed: self.removed.clone(),
        })
    }
}

fn convert_to_py_regex_element(py: Python, element: &RegexElement) -> PyResult<PyObject> {
    match element {
        RegexElement::Literal(c) => Ok(PyLiteral { value: *c }.into_py(py)),
        RegexElement::CharGroup { chars, inverted } => Ok(PyCharGroup {
            chars: chars.iter().cloned().collect(),
            inverted: *inverted,
        }
        .into_py(py)),
        RegexElement::Repeated { element, min, max } => {
            let py_element = convert_to_py_regex_element(py, element)?;
            Ok(PyRepeated {
                element: py_element,
                min: *min,
                max: *max,
            }
            .into_py(py))
        }
        RegexElement::Concatenation(elements) => {
            let py_elements: PyResult<Vec<PyObject>> = elements
                .iter()
                .map(|e| convert_to_py_regex_element(py, e))
                .collect();
            Ok(PyConcatenation {
                elements: py_elements?,
            }
            .into_py(py))
        }
        RegexElement::Alternation(elements) => {
            let py_elements: PyResult<Vec<PyObject>> = elements
                .iter()
                .map(|e| convert_to_py_regex_element(py, e))
                .collect();
            Ok(PyAlternation {
                elements: py_elements?,
            }
            .into_py(py))
        }
        RegexElement::Capture(element) => {
            let py_element = convert_to_py_regex_element(py, element)?;
            Ok(PyCapture {
                element: py_element,
            }
            .into_py(py))
        }
        RegexElement::Group(element) => {
            let py_element = convert_to_py_regex_element(py, element)?;
            Ok(PyGroup {
                element: py_element,
            }
            .into_py(py))
        }
        RegexElement::Anchor(anchor_type) => Ok(PyAnchor {
            anchor_type: format!("{:?}", anchor_type),
        }
        .into_py(py)),
        RegexElement::Flag {
            element,
            added,
            removed,
        } => {
            let py_element = convert_to_py_regex_element(py, element)?;
            Ok(PyFlag {
                element: py_element,
                added: added.iter().map(|f| format!("{:?}", f)).collect(),
                removed: removed.iter().map(|f| format!("{:?}", f)).collect(),
            }
            .into_py(py))
        }
    }
}

#[pyfunction(name = "parse_pattern")]
#[pyo3(text_signature = "(pattern: &str)")]
pub fn parse_pattern_internal(py: Python, pattern: &str) -> PyResult<PyObject> {
    match parse_pattern(pattern) {
        Ok(regex_element) => convert_to_py_regex_element(py, &regex_element),
        Err(_) => Err(PyValueError::new_err("Invalid pattern")),
    }
}

#[pyclass]
pub struct InteregularFSMInfo {
    #[pyo3(get)]
    initial: u32,
    #[pyo3(get)]
    finals: HashSet<u32>,
    #[pyo3(get)]
    states: HashSet<u32>,
    #[pyo3(get)]
    map: HashMap<u32, HashMap<u32, u32>>,
    #[pyo3(get)]
    symbol_mapping: HashMap<char, usize>,
    #[pyo3(get)]
    by_transition: HashMap<usize, Vec<char>>,
}

use crate::interegular::fsm::Alphabet;
use crate::interegular::patterns::Flag;

#[pyfunction(name = "parse_pattern_to_fsm")]
#[pyo3(text_signature = "(pattern: &str)")]
pub fn parse_pattern_to_fsm_internal(pattern: &str) -> PyResult<InteregularFSMInfo> {
    let regex_element =
        parse_pattern(pattern).map_err(|_| PyValueError::new_err("Invalid pattern"))?;

    let prefix_postfix = None;
    let flags = None;

    let default_alphabet = Alphabet::<char>::default();
    let empty_flags: HashSet<Flag> = HashSet::new();
    let patterns_alphabet: Alphabet<char> = regex_element.get_alphabet(&empty_flags);

    // TODO: this is a hack to build a alphabet with the same symbols as the patterns
    // and ensure that \0 is the anything symbol at 0. However, this is not a good solution
    // and should be handled by an improved alphabet implementation
    let mut my_new_symbol_mapping = HashMap::new();
    my_new_symbol_mapping.insert('\0', 0 as usize); // add \0 as the anything symbol at 0

    let mut counter = 1;
    for (symbol, _) in patterns_alphabet.symbol_mapping.iter() {
        if *symbol != '\0' {
            my_new_symbol_mapping.insert(*symbol, counter as usize);
            counter += 1;
        }
    }

    let alphabet = Alphabet::new(my_new_symbol_mapping);
    let fsm_info = regex_element.to_fsm(Some(alphabet.clone()), prefix_postfix, flags);

    // convert into u32 for python
    let map: HashMap<u32, HashMap<u32, u32>> = fsm_info
        .map
        .iter()
        .map(|(key, map)| {
            // let u32_key = u32::from(*key);
            let u32_key = *key as u32;
            let map_as_u32s = map
                .iter()
                .map(|(key, value)| {
                    (
                        // u32::from(*key), u32::from(*value)
                        *key as u32,
                        *value as u32,
                    )
                })
                .collect();
            (u32_key, map_as_u32s)
        })
        .collect();

    let python_symbol_mapping: HashMap<char, usize> = alphabet
        .symbol_mapping
        .iter()
        .map(|(k, v)| (*k, (*v).into()))
        .collect();

    let python_by_transition: HashMap<usize, Vec<char>> = alphabet
        .by_transition
        .iter()
        .map(|(k, v)| (usize::from(*k), v.iter().map(|&c| c).collect()))
        .collect();

    Ok(InteregularFSMInfo {
        initial: fsm_info.initial as u32,
        finals: fsm_info.finals.iter().map(|f| (*f as u32)).collect(),
        states: fsm_info.states.iter().map(|s| (*s as u32)).collect(),
        map,
        symbol_mapping: python_symbol_mapping,
        by_transition: python_by_transition,
    })
}

#[pymodule]
fn outlines_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(walk_fsm_py, m)?)?;
    m.add_function(wrap_pyfunction!(state_scan_tokens_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_token_transition_keys_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_vocabulary_transition_keys_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pattern_internal, m)?)?;
    m.add_class::<PyLiteral>()?;
    m.add_class::<PyCharGroup>()?;
    m.add_class::<PyRepeated>()?;
    m.add_class::<PyConcatenation>()?;
    m.add_class::<PyAlternation>()?;
    m.add_class::<PyCapture>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyAnchor>()?;
    m.add_class::<PyFlag>()?;

    m.add_function(wrap_pyfunction!(parse_pattern_to_fsm_internal, m)?)?;
    m.add_class::<InteregularFSMInfo>()?;

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

    m.add_function(wrap_pyfunction!(build_regex_from_schema_py, m)?)?;
    m.add_function(wrap_pyfunction!(to_regex_py, m)?)?;

    m.add_class::<PyVocabulary>()?;

    Ok(())
}
