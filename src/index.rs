/// Construct an Index.
use crate::prelude::*;
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};
use bincode::{Decode, Encode};
use regex_automata::dfa::{dense::DFA, Automaton};
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Index {
    initial: StateId,
    finals: HashSet<StateId>,
    states_to_token_subsets: HashMap<StateId, HashMap<TokenId, StateId>>,
    eos_token_id: TokenId,
}

impl Index {
    pub(crate) fn new(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
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
            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state.as_u32());
            }

            'token_loop: for (token, ids) in vocabulary.tokens_to_ids().iter() {
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

        // Check if there is at least one valid mapping
        let is_valid = transitions.values().any(|mapping| {
            mapping
                .values()
                .any(|end_state| final_states.contains(end_state))
        });

        if is_valid {
            Ok(Self {
                initial: start_state.as_u32(),
                finals: final_states,
                states_to_token_subsets: transitions,
                eos_token_id,
            })
        } else {
            Err(Error::InsufficientVocabulary)
        }
    }

    pub(crate) fn allowed_tokens(&self, state: StateId) -> Option<Vec<TokenId>> {
        self.states_to_token_subsets
            .get(&state)
            .map_or_else(|| None, |res| Some(res.keys().cloned().collect()))
    }

    pub(crate) fn next_state(&self, state: StateId, token_id: TokenId) -> Option<StateId> {
        if token_id == self.eos_token_id {
            return None;
        }
        Some(*self.states_to_token_subsets.get(&state)?.get(&token_id)?)
    }

    pub(crate) fn initial(&self) -> StateId {
        self.initial
    }

    pub(crate) fn is_final(&self, state: StateId) -> bool {
        self.finals.contains(&state)
    }

    pub(crate) fn final_states(&self) -> &HashSet<StateId> {
        &self.finals
    }

    pub(crate) fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        &self.states_to_token_subsets
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index object with transitions:")?;
        for (state_id, token_ids) in self.states_to_token_subsets.iter() {
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
        let mut vocabulary = Vocabulary::new(4);
        for (token, token_id) in [("blah", 0), ("1a", 1), ("2", 2), ("0", 3)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index = Index::new(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.initial(), 40);
        assert_eq!(index.final_states(), &HashSet::from_iter([24, 48, 56]));

        let expected = HashMap::from_iter([
            (24, HashMap::from_iter([(3, 24), (4, 24), (2, 24)])),
            (48, HashMap::from_iter([(4, 48)])),
            (40, HashMap::from_iter([(3, 48), (2, 56)])),
            (56, HashMap::from_iter([(3, 24), (4, 56), (2, 24)])),
        ]);
        assert_eq!(index.transitions(), &expected);
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
            .allowed_tokens(index.initial())
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
}
