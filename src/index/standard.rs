//! Building an `Index` to efficiently map vocabulary tokens to state transitions.
use std::mem::{size_of, size_of_val};

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::index::{IndexBehavior, TokenIdIterator};
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
    /// The usefull size of the vocabulary
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
/// let Index = Index::new(regex, &vocabulary)?;
///
/// let initial_state = Index.initial_state();
/// println!("Initial state is {}", initial_state);
/// println!("Is initial state a final state? {}", Index.is_final_state(&initial_state));
///
/// let allowed_tokens = Index.allowed_tokens(&initial_state).expect("Some allowed tokens");
/// println!("Allowed tokens at initial state are {:?}", allowed_tokens);
///
/// let token_id = allowed_tokens.first().expect("First token");
/// println!("Next state for the token_id {} is {:?}", token_id, Index.next_state(&initial_state, token_id));
///
/// println!("Final states are {:?}", Index.final_states());
/// println!("Index has exactly {} transitions", Index.transitions().len());
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
        let eos_token_id = vocabulary.eos_token_id();
        let vocab_size = vocabulary.tokens().len();
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
            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state.as_u32());
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

    pub fn memory_usage(&self) -> usize {
        let struct_size = size_of::<Index>();

        let initial_state_size = size_of::<StateId>();
        let eos_token_id_size = size_of::<TokenId>();
        let vocab_size_size = size_of::<usize>();

        let final_states_container = size_of_val(&self.final_states);

        let final_states_content = self.final_states.len() * (size_of::<StateId>() + 1);
        let final_states_size = final_states_container + final_states_content;

        let transitions_container = size_of_val(&self.transitions);

        let mut inner_maps_size = 0;
        let mut transitions_content_size = 0;
        #[allow(clippy::for_kv_map)]
        for (_, inner_map) in &self.transitions {
            inner_maps_size += size_of_val(inner_map);

            transitions_content_size +=
                inner_map.len() * (size_of::<TokenId>() + size_of::<StateId>() + 1);
        }

        let transitions_total_size =
            transitions_container + inner_maps_size + transitions_content_size;

        let total_size = struct_size
            + initial_state_size
            + eos_token_id_size
            + vocab_size_size
            + final_states_size
            + transitions_total_size;

        println!("Index memory usage:");
        println!("  Structure de base: {} bytes", struct_size);
        println!("  initial_state: {} bytes", initial_state_size);
        println!("  final_states: {} bytes", final_states_size);
        println!("  transitions: {} bytes", transitions_total_size);
        println!("    - container: {} bytes", transitions_container);
        println!("    - inner maps: {} bytes", inner_maps_size);
        println!("    - transitions data: {} bytes", transitions_content_size);
        println!("  eos_token_id: {} bytes", eos_token_id_size);
        println!("  vocab_size: {} bytes", vocab_size_size);
        println!(
            "Total memory usage: {} bytes ({:.2} MB)",
            total_size,
            total_size as f64 / (1024.0 * 1024.0)
        );

        println!("\nTransitions statistics:");
        println!("  Number of states: {}", self.transitions.len());
        let total_transitions = self.transitions.values().map(|m| m.len()).sum::<usize>();
        println!("  Total transitions: {}", total_transitions);
        println!(
            "  Average transitions per state: {:.2}",
            if self.transitions.is_empty() {
                0.0
            } else {
                total_transitions as f64 / self.transitions.len() as f64
            }
        );

        total_size
    }
}

impl IndexBehavior for Index {
    /// Returns the ID of the initial state in the automaton.
    fn initial_state(&self) -> StateId {
        self.initial_state
    }

    /// Returns set of final states.
    fn final_states(&self) -> &HashSet<StateId> {
        &self.final_states
    }

    fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }

    /// Returns state transitions map of tokens ids and their corresponding transition states.
    fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        &self.transitions
    }

    /// Checks if state is in final states set or not.
    fn is_final_state(&self, state: &StateId) -> bool {
        self.final_states.contains(state)
    }

    /// Lists allowed tokens for a give state ID or `None` if it is not found in `Index`.
    fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        self.transitions
            .get(state)
            .map(|res| res.keys().cloned().collect())
    }

    fn allowed_tokens_iter(&self, state: &StateId) -> Option<TokenIdIterator> {
        self.transitions
            .get(state)
            .map(|res| TokenIdIterator::new(res.keys()))
    }

    fn allowed_tokens_mask(&self, _state: &StateId) -> Option<&Vec<u64>> {
        None
    }

    /// Returns transition state for a given state and token id or `None` otherwise.
    fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if token_id == &self.eos_token_id {
            return None;
        }
        Some(*self.transitions.get(state)?.get(token_id)?)
    }

    /// Returns the size of the vocabulary
    fn vocab_size(&self) -> usize {
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
            (vec![32, 240, 159, 152], 7),
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
                HashMap::from_iter([(2, 128), (7, 192), (5, 208), (6, 208)]),
            ),
            (128, HashMap::from_iter([(8, 128)])),
        ]);
        assert_eq!(index.transitions(), &expected);
    }

    #[test]
    fn test_allowed_tokens_iter() {
        let mut vocabulary = Vocabulary::new(8);

        for (token, token_id) in [
            (vec![32, 240, 159, 152], 7),
            (vec![32, 240, 159, 152, 141], 6),
            (vec![240, 159, 152, 141], 4),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = Index::new("[ ]?.?", &vocabulary).unwrap();
        let initial_state = index.initial_state();

        let tokens: Vec<_> = index
            .allowed_tokens_iter(&initial_state)
            .unwrap()
            .cloned()
            .collect();
        assert_eq!(tokens, vec![7, 6, 4, 8]); // Vérifie les TokenId retournés
    }

    #[test]
    fn test_index_memory_size() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string", "format": "email"}}, "required": ["name", "age", "email"]}"#;

        // Generate regex from schema
        let regex = json_schema::regex_from_str(schema, None).unwrap();

        //let regex : &str =  r"[a-zA-Z]{100}";
        // println!("Generated regex: {}", regex);

        let vocabulary = Vocabulary::from_pretrained("gpt2", None).unwrap();

        let index = Index::new(&regex, &vocabulary).expect("Index failed");

        let memory_usage = index.memory_usage();
        println!("Index utilise {} bytes", memory_usage);
    }
}
