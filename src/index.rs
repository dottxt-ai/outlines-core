//! Building an `Index` to efficiently map vocabulary tokens to state transitions.

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::prelude::*;
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};

/// `Index` efficiently maps vocabulary tokens to state transitions.
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Index {
    /// The ID of the initial state in the automaton, processing begins from this state.
    initial_state: StateId,
    /// A collection of states considered as terminal states.
    final_states: HashSet<StateId>,
    /// A mapping of state transitions, defined by tokens ids and their corresponding state changes.
    ///
    /// ### Example
    /// ```ignore
    /// transitions = {
    ///    1: {10: 2, 15: 3},
    ///    2: {20: 4, 25: 3},
    ///    3: {30: 4},
    ///    4: {40: 4},
    /// }
    ///  +--------------------------------------+
    ///  |               State 1                |
    ///  |            Initial State             |
    ///  +--------------------------------------+
    ///              |                     |
    ///              +                     |
    ///         Token ID 10                |
    ///  +-----------------------+         |
    ///  |        State 2        |         |
    ///  +-----------------------+         |
    ///       |             |              |
    ///       |             +              +
    ///       |        Token ID 25    Token ID 15
    ///       |        +------------------------+
    ///       |        |        State 3         |
    ///       |        +------------------------+
    ///       |                            |
    ///       +                            +
    ///  Token ID 20                  Token ID 30
    ///  +--------------------------------------+
    ///  |               State 4                |
    ///  |             Final state              |
    ///  +--------------------------------------+
    /// ```
    transitions: HashMap<StateId, HashMap<TokenId, StateId>>,
    /// The token ID reserved for the "end-of-sequence" token.
    eos_token_id: TokenId,
    /// The size of the vocabulary used to build the index.
    vocab_size: usize,
}
/// The `Index` structure is designed to efficiently map tokens from a given vocabulary
/// to state transitions within a finite-state automaton.
///
/// ## Usage:
/// The `Index` is typically constructed by combining a vocabulary and regular expressions.
/// Once built, it can be used to efficiently evaluate token sequences or to validate input data.
///
/// ## Example:
/// ```rust
/// use outlines_core::prelude::*;
///
/// # fn run() -> Result<(), outlines_core::Error> {
/// let regex = "0|[1-9][0-9]*";
/// let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", None)?;
/// let index = Index::new(regex, &vocabulary)?;
///
/// let initial_state = index.initial_state();
/// println!("Initial state is {}", initial_state);
/// println!("Is initial state a final state? {}", index.is_final_state(&initial_state));
///
/// let allowed_tokens = index.allowed_tokens(&initial_state).expect("Some allowed tokens");
/// println!("Allowed tokens at initial state are {:?}", allowed_tokens);
///
/// let token_id = allowed_tokens.first().expect("First token");
/// println!("Next state for the token_id {} is {:?}", token_id, index.next_state(&initial_state, token_id));
///
/// println!("Final states are {:?}", index.final_states());
/// println!("Index has exactly {} transitions", index.transitions().len());
/// # Ok(())
/// # }
///
/// ```
///
/// ## Performance:
/// - **Complexity**:
///   The `Index` can accommodate large vocabularies and complex regular expressions.
///   However, its size may grow significantly with the complexity of the input.
/// - **Construction Cost**:
///   Building the `Index` involves processing the vocabulary and regular expressions,
///   which may require a considerable amount of time and computational resources.
impl Index {
    /// Builds an `Index` from regular expression and vocabulary tokens.
    pub fn new(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
        let vocab_size = vocabulary.len();
        let eos_token_id = vocabulary.eos_token_id();
        let dfa = DFA::new(regex).map_err(Box::new)?;
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::DfaHasNoStartState),
        };

        let mut transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        let mut final_states: HashSet<StateId> = HashSet::default();

        let mut seen: HashSet<AutomataStateId> = HashSet::from_iter([start_state]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];

        while let Some(current_state) = next_states.pop() {
            let mut has_valid_transitions = false;

            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state.as_u32());
                has_valid_transitions = true;
            }

            'token_loop: for (token, ids) in vocabulary.tokens().iter() {
                if ids.contains(&eos_token_id) {
                    continue;
                }

                let mut next_state = current_state;
                for transition_byte in token {
                    next_state = dfa.next_state(next_state, *transition_byte);
                    if dfa.is_dead_state(next_state) || dfa.is_quit_state(next_state) {
                        continue 'token_loop;
                    }
                }

                let is_intermediate_state = !dfa.is_match_state(next_state);
                let is_full_match_state = dfa.is_match_state(dfa.next_eoi_state(next_state));
                if is_intermediate_state || is_full_match_state {
                    has_valid_transitions = true;
                    for token_id in ids {
                        transitions
                            .entry(current_state.as_u32())
                            .or_default()
                            .insert(*token_id, next_state.as_u32());
                    }
                }
                if !seen.contains(&next_state) {
                    seen.insert(next_state);
                    next_states.push(next_state);
                }
            }

            // If the current state has no valid transitions and is not a match state,
            // it means the vocabulary is incompatible with the regex.
            if !has_valid_transitions && !dfa.is_match_state(current_state) {
                let mut valid_characters = Vec::new();
                for byte in 0..=255u8 {
                    let test_state = dfa.next_state(current_state, byte);
                    if !dfa.is_dead_state(test_state) && !dfa.is_quit_state(test_state) {
                        if byte.is_ascii() {
                            valid_characters.push(char::from(byte).to_string());
                        } else {
                            valid_characters.push(format!("\\x{:02x}", byte));
                        }
                    }
                }

                return Err(Error::IncompatibleVocabulary {
                    regex: regex.to_string(),
                    error_state: current_state.as_u32(),
                    missing_tokens: valid_characters,
                });
            }
        }

        // Populate `transitions` with mappings from `final_states` to `eos_token_id`
        for &final_state in &final_states {
            transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
        }

        Ok(Self {
            initial_state: start_state.as_u32(),
            final_states,
            transitions,
            eos_token_id,
            vocab_size,
        })
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        use std::io::Write;
        let mut buffer = Vec::new();
        
        // Write vocab_size (32 bits)
        buffer.extend_from_slice(&(self.vocab_size as u32).to_le_bytes());
        
        // Write eos_token_id (32 bits)
        buffer.extend_from_slice(&self.eos_token_id.to_le_bytes());
        
        // Write initial_state_id (32 bits)
        buffer.extend_from_slice(&self.initial_state.to_le_bytes());
        
        // Write number of final states (32 bits)
        buffer.extend_from_slice(&(self.final_states.len() as u32).to_le_bytes());
        
        // Write final states (32 bits each)
        for &final_state in &self.final_states {
            buffer.extend_from_slice(&final_state.to_le_bytes());
        }
        
        // Write index type (8 bits) - Type 1 for now
        buffer.push(1u8);
        
        // Write number of states with transitions (32 bits)
        buffer.extend_from_slice(&(self.transitions.len() as u32).to_le_bytes());
        
        // Write transitions for each state
        for (&state_id, transitions_map) in &self.transitions {
            // Write state ID (32 bits)
            buffer.extend_from_slice(&state_id.to_le_bytes());
            
            // Write number of transitions (32 bits)
            buffer.extend_from_slice(&(transitions_map.len() as u32).to_le_bytes());
            
            // Write each transition (TokenId -> StateId)
            for (&token_id, &next_state_id) in transitions_map {
                buffer.extend_from_slice(&token_id.to_le_bytes());
                buffer.extend_from_slice(&next_state_id.to_le_bytes());
            }
        }
        
        // Write compressed data to file
        let compressed = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut encoder = compressed;
        encoder.write_all(&buffer).map_err(|e| Error::IOError(e.to_string()))?;
        let compressed_data = encoder.finish().map_err(|e| Error::IOError(e.to_string()))?;
        
        std::fs::write(path, compressed_data).map_err(|e| Error::IOError(e.to_string()))?;
        
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use std::io::Read;
        
        // Read and decompress file
        let compressed_data = std::fs::read(path).map_err(|e| Error::IOError(e.to_string()))?;
        let mut decoder = flate2::read::GzDecoder::new(&compressed_data[..]);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).map_err(|e| Error::IOError(e.to_string()))?;
        
        let mut cursor = 0;
        
        // Helper to read u32
        let read_u32 = |buf: &[u8], pos: &mut usize| -> Result<u32> {
            if *pos + 4 > buf.len() {
                return Err(Error::IOError("Unexpected end of buffer".to_string()));
            }
            let value = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
            *pos += 4;
            Ok(value)
        };
        
        // Read vocab_size (32 bits)
        let vocab_size = read_u32(&buffer, &mut cursor)? as usize;
        
        // Read eos_token_id (32 bits)
        let eos_token_id = read_u32(&buffer, &mut cursor)?;
        
        // Read initial_state_id (32 bits)
        let initial_state = read_u32(&buffer, &mut cursor)?;
        
        // Read number of final states (32 bits)
        let num_final_states = read_u32(&buffer, &mut cursor)? as usize;
        
        // Read final states
        let mut final_states = HashSet::default();
        for _ in 0..num_final_states {
            let final_state = read_u32(&buffer, &mut cursor)?;
            final_states.insert(final_state);
        }
        
        // Read index type (8 bits)
        if cursor >= buffer.len() {
            return Err(Error::IOError("Unexpected end of buffer".to_string()));
        }
        let index_type = buffer[cursor];
        cursor += 1;
        
        if index_type != 1 {
            return Err(Error::IOError(format!("Unsupported index type: {}", index_type)));
        }
        
        // Read number of states with transitions (32 bits)
        let num_states = read_u32(&buffer, &mut cursor)? as usize;
        
        // Read transitions
        let mut transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        for _ in 0..num_states {
            // Read state ID (32 bits)
            let state_id = read_u32(&buffer, &mut cursor)?;
            
            // Read number of transitions (32 bits)
            let num_transitions = read_u32(&buffer, &mut cursor)? as usize;
            
            // Read each transition
            let mut state_transitions = HashMap::default();
            for _ in 0..num_transitions {
                let token_id = read_u32(&buffer, &mut cursor)?;
                let next_state_id = read_u32(&buffer, &mut cursor)?;
                state_transitions.insert(token_id, next_state_id);
            }
            
            transitions.insert(state_id, state_transitions);
        }
        
        Ok(Self {
            initial_state,
            final_states,
            transitions,
            eos_token_id,
            vocab_size,
        })
    }

    /// Returns the ID of the initial state in the automaton.
    pub fn initial_state(&self) -> StateId {
        self.initial_state
    }

    /// Returns set of final states.
    pub fn final_states(&self) -> &HashSet<StateId> {
        &self.final_states
    }

    /// Returns state transitions map of tokens ids and their corresponding transition states.
    pub fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        &self.transitions
    }

    /// Checks if state is in final states set or not.
    pub fn is_final_state(&self, state: &StateId) -> bool {
        self.final_states.contains(state)
    }

    /// Lists allowed tokens for a give state ID or `None` if it is not found in `Index`.
    pub fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        self.transitions
            .get(state)
            .map(|res| res.keys().cloned().collect())
    }

    pub fn allowed_tokens_iter(&self, state: &StateId) -> Option<impl Iterator<Item = &TokenId>> {
        self.transitions.get(state).map(|map| map.keys())
    }

    /// Returns transition state for a given state and token id or `None` otherwise.
    pub fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if token_id == &self.eos_token_id {
            return None;
        }
        Some(*self.transitions.get(state)?.get(token_id)?)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index object with transitions:")?;
        for (state_id, token_ids) in self.transitions.iter() {
            writeln!(f, "{:?} -> {:#?}", state_id, token_ids)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let eos_token_id = 4;
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let initial_state = index.initial_state();
        assert_eq!(initial_state, 40);
        assert_eq!(index.final_states(), &HashSet::from_iter([24, 48, 56]));
        assert!(!index.is_final_state(&initial_state));

        let expected = HashMap::from_iter([
            (24, HashMap::from_iter([(3, 24), (4, 24), (2, 24)])),
            (48, HashMap::from_iter([(4, 48)])),
            (40, HashMap::from_iter([(3, 48), (2, 56)])),
            (56, HashMap::from_iter([(3, 24), (4, 56), (2, 24)])),
        ]);
        assert_eq!(index.transitions(), &expected);

        let allowed_tokens = index
            .allowed_tokens(&initial_state)
            .expect("No allowed tokens");
        let token_id = allowed_tokens.first().expect("No first tokens");

        let state = 48;
        assert_eq!(index.next_state(&initial_state, token_id), Some(state));
        assert!(index.is_final_state(&state));

        assert_eq!(index.next_state(&state, &eos_token_id), None);
        assert_eq!(index.next_state(&state, token_id), None);
    }

    #[test]
    fn index_from_regex_initital_in_allowed() {
        let regex = "`\\n(\\.\\n)?`\\n";
        let mut vocabulary = Vocabulary::new(104);
        for (token, token_id) in [("\n", 103), (".", 102), ("`", 101)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let allowed = index
            .allowed_tokens(&index.initial_state())
            .expect("No allowed tokens");
        assert!(allowed.contains(&101));
    }

    #[test]
    fn index_from_regex_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let mut vocabulary = Vocabulary::new(8);
        for (token, token_id) in [(" 😍", 5), ("blah", 0), ("😇", 2), ("😈a", 1), ("😍", 3)]
        {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        for (token, token_id) in [
            (vec![32, 240, 159, 152, 136], 7),
            (vec![32, 240, 159, 152, 141], 6),
            (vec![240, 159, 152, 141], 4),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.final_states(), &HashSet::from_iter([208, 128]));

        let expected = HashMap::from_iter([
            (
                208,
                HashMap::from_iter([(3, 208), (8, 208), (4, 208), (2, 208)]),
            ),
            (
                80,
                HashMap::from_iter([(2, 128), (7, 208), (5, 208), (6, 208)]),
            ),
            (128, HashMap::from_iter([(8, 128)])),
        ]);
        assert_eq!(index.transitions(), &expected);
    }

    #[test]
    fn index_incompatible_vocabulary_error() {
        let regex = "0 1";
        let mut vocabulary = Vocabulary::new(3);
        for (token, token_id) in [("0", 0), ("0 ", 1), ("1", 2)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let result = Index::new(regex, &vocabulary);
        assert!(result.is_err());

        if let Err(Error::IncompatibleVocabulary {
            regex: _,
            missing_tokens,
            ..
        }) = result
        {
            assert!(missing_tokens.contains(&" ".to_string()));
        } else {
            panic!("Expected IncompatibleVocabulary error");
        }
    }

    #[test]
    fn index_incompatible_vocabulary_error_non_ascii() {
        let regex = "😈😍";
        let mut vocabulary = Vocabulary::new(3);
        for (token, token_id) in [("😈", 0), (" ", 1), ("b", 2)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let result = Index::new(regex, &vocabulary);
        assert!(result.is_err());

        if let Err(Error::IncompatibleVocabulary {
            regex: _,
            missing_tokens,
            ..
        }) = result
        {
            assert!(missing_tokens.contains(&"\\xf0".to_string()));
        } else {
            panic!("Expected IncompatibleVocabulary error");
        }
    }

    #[test]
    fn test_save_and_load() {
        let regex = "0|[1-9][0-9]*";
        let eos_token_id = 4;
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        
        let original_index = Index::new(regex, &vocabulary).expect("Index failed");
        
        // Save to temporary file
        let temp_path = std::env::temp_dir().join("test_index.bin");
        original_index.save(&temp_path).expect("Save failed");
        
        // Load from file
        let loaded_index = Index::load(&temp_path).expect("Load failed");
        
        // Cleanup
        std::fs::remove_file(&temp_path).ok();
        
        // Verify equality
        assert_eq!(original_index, loaded_index);
        assert_eq!(original_index.initial_state(), loaded_index.initial_state());
        assert_eq!(original_index.final_states(), loaded_index.final_states());
        assert_eq!(original_index.transitions(), loaded_index.transitions());
        assert_eq!(original_index.vocab_size(), loaded_index.vocab_size());
    }

    #[test]
    fn test_save_and_load_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let mut vocabulary = Vocabulary::new(8);
        for (token, token_id) in [(" 😍", 5), ("blah", 0), ("😇", 2), ("😈a", 1), ("😍", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        for (token, token_id) in [
            (vec![32, 240, 159, 152, 136], 7),
            (vec![32, 240, 159, 152, 141], 6),
            (vec![240, 159, 152, 141], 4),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let original_index = Index::new(regex, &vocabulary).expect("Index failed");
        
        let temp_path = std::env::temp_dir().join("test_index_multibyte.bin");
        original_index.save(&temp_path).expect("Save failed");
        let loaded_index = Index::load(&temp_path).expect("Load failed");
        std::fs::remove_file(&temp_path).ok();
        
        assert_eq!(original_index, loaded_index);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = Index::load("/nonexistent/path/index.bin");
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::IOError(_))));
    }

    #[test]
    fn test_load_corrupted_file() {
        let temp_path = std::env::temp_dir().join("test_corrupted.bin");
        std::fs::write(&temp_path, b"corrupted data").expect("Write failed");
        
        let result = Index::load(&temp_path);
        std::fs::remove_file(&temp_path).ok();
        
        assert!(result.is_err());
    }

    #[test]
    fn test_save_preserves_file_size() {
        let regex = "0|[1-9][0-9]*";
        let mut vocabulary = Vocabulary::new(4);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        
        let index = Index::new(regex, &vocabulary).expect("Index failed");
        let temp_path = std::env::temp_dir().join("test_size.bin");
        
        index.save(&temp_path).expect("Save failed");
        let metadata = std::fs::metadata(&temp_path).expect("Metadata failed");
        
        // File should exist and be non-empty
        assert!(metadata.len() > 0);
        
        // Gzip compression should make it smaller than raw data
        // Rough estimate: at least 5 * 4 bytes for basic fields + transitions
        assert!(metadata.len() < 10000); // Should be much smaller for this simple case
        
        std::fs::remove_file(&temp_path).ok();
    }
}
