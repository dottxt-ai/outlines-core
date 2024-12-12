/// Construct an Index.
use crate::prelude::*;
use crate::regex::{get_vocabulary_transition_keys, state_scan_tokens};
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};
use bincode::{Decode, Encode};
use regex_automata::dfa::{dense::DFA, Automaton};
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug)]
pub struct FSMInfo {
    pub(crate) initial: State,
    pub(crate) finals: FxHashSet<State>,
    pub(crate) transitions: FxHashMap<(State, TransitionKey), State>,
    pub(crate) alphabet_anything_value: TransitionKey,
    pub(crate) alphabet_symbol_mapping: FxHashMap<String, TransitionKey>,
}

impl FSMInfo {
    pub fn new(
        initial: State,
        finals: FxHashSet<State>,
        transitions: FxHashMap<(State, TransitionKey), State>,
        alphabet_anything_value: TransitionKey,
        alphabet_symbol_mapping: FxHashMap<String, TransitionKey>,
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

#[derive(Debug, Encode, Decode)]
pub struct Index {
    initial: u32,
    finals: FxHashSet<u32>,
    states_to_token_subsets: FxHashMap<u32, FxHashMap<u32, u32>>,
    eos_token_id: u32,
}

impl Index {
    pub fn new(
        fsm_info: &FSMInfo,
        vocabulary: &Vocabulary,
        eos_token_id: u32,
        frozen_tokens: FxHashSet<String>,
    ) -> Result<Self> {
        let mut states_to_token_subsets: FxHashMap<u32, FxHashMap<u32, u32>> = FxHashMap::default();
        let mut seen: FxHashSet<State> = FxHashSet::default();
        let mut next_states: FxHashSet<State> = FxHashSet::from_iter([fsm_info.initial]);

        let vocabulary_transition_keys = get_vocabulary_transition_keys(
            &fsm_info.alphabet_symbol_mapping,
            fsm_info.alphabet_anything_value,
            vocabulary,
            &frozen_tokens,
        );

        while let Some(start_state) = next_states.iter().cloned().next() {
            next_states.remove(&start_state);

            let token_ids_end_states = state_scan_tokens(
                &fsm_info.transitions,
                fsm_info.initial,
                &fsm_info.finals,
                vocabulary,
                &vocabulary_transition_keys,
                start_state,
            );

            for (token_id, end_state) in &token_ids_end_states {
                let inner_map = states_to_token_subsets.entry(start_state).or_default();
                inner_map.insert(*token_id, *end_state);

                if !seen.contains(end_state) {
                    next_states.insert(*end_state);
                }
            }

            if fsm_info.finals.contains(&start_state) && !token_ids_end_states.is_empty() {
                let inner_map = states_to_token_subsets.entry(start_state).or_default();
                inner_map.insert(eos_token_id, start_state);
            }

            seen.insert(start_state);
        }

        let is_valid = states_to_token_subsets
            .values()
            .flat_map(|token_id_end_states| token_id_end_states.values())
            .any(|end_state| fsm_info.finals.contains(end_state));

        if is_valid {
            Ok(Self {
                initial: fsm_info.initial,
                finals: fsm_info.finals.clone(),
                states_to_token_subsets,
                eos_token_id,
            })
        } else {
            Err(Error::InsufficientVocabulary)
        }
    }

    pub(crate) fn from_regex(regex: &str, vocabulary: &Vocabulary) -> Result<Self> {
        let eos_token_id = match vocabulary.eos_token_id() {
            Some(s) => s,
            None => return Err(Error::IndexEosTokenIdNotAvailable),
        };

        let dfa = DFA::builder().build(regex).map_err(Box::new)?;
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::IndexNoAnchoredUniversalStartState),
        };

        let mut index: FxHashMap<State, FxHashMap<TokenId, State>> = FxHashMap::default();
        let mut seen: FxHashSet<AutomataStateId> = FxHashSet::default();
        let mut final_states: FxHashSet<State> = FxHashSet::default();
        let mut next_states: FxHashSet<AutomataStateId> = FxHashSet::from_iter([start_state]);

        while let Some(start_state) = next_states.iter().cloned().next() {
            next_states.remove(&start_state);
            seen.insert(start_state);

            if dfa.is_match_state(dfa.next_eoi_state(start_state)) {
                final_states.insert(start_state.as_u32());
            }

            'token_loop: for (token, ids) in vocabulary.tokens_to_ids().iter() {
                if ids.contains(&eos_token_id) {
                    continue;
                }

                let mut next_state = start_state;
                for transition_byte in token.as_bytes() {
                    next_state = dfa.next_state(next_state, *transition_byte);
                    if dfa.is_dead_state(next_state) || dfa.is_quit_state(next_state) {
                        continue 'token_loop;
                    }
                }

                if dfa.is_match_state(next_state) {
                    // Token either matched or matched except the last character.
                    // Check what happens if the input suddenly ends after reaching this state.
                    // If the automata still matches, then token is exactly matched, if not
                    // then token didn't match.
                    let next_eoi_state = dfa.next_eoi_state(next_state);
                    let token_matched = dfa.is_match_state(next_eoi_state);
                    if !token_matched {
                        continue;
                    }
                }

                for token_id in ids {
                    let mapping = index.entry(start_state.as_u32()).or_default();
                    mapping.insert(*token_id, next_state.as_u32());

                    if !seen.contains(&next_state) {
                        next_states.insert(next_state);
                    }
                }
            }
        }

        let start_state = start_state.as_u32();

        // Populate `index` with mappings from `final_states` to `eos_token_id`
        for &final_state in &final_states {
            index
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
        }
        // Check if there is at least one valid mapping
        let is_valid = index.values().any(|mapping| {
            mapping
                .values()
                .any(|end_state| final_states.contains(end_state))
        });

        if is_valid {
            Ok(Self {
                initial: start_state,
                finals: final_states,
                states_to_token_subsets: index,
                eos_token_id,
            })
        } else {
            Err(Error::InsufficientVocabulary)
        }
    }

    pub(crate) fn allowed_tokens(&self, state: u32) -> Option<Vec<u32>> {
        self.states_to_token_subsets
            .get(&state)
            .map_or_else(|| None, |res| Some(res.keys().cloned().collect()))
    }

    pub(crate) fn next_state(&self, state: u32, token_id: u32) -> Option<u32> {
        if token_id == self.eos_token_id {
            return None;
        }
        Some(*self.states_to_token_subsets.get(&state)?.get(&token_id)?)
    }

    pub(crate) fn initial(&self) -> u32 {
        self.initial
    }

    pub(crate) fn is_final(&self, state: u32) -> bool {
        self.finals.contains(&state)
    }

    pub(crate) fn final_states(&self) -> &FxHashSet<State> {
        &self.finals
    }

    pub(crate) fn transitions(&self) -> &FxHashMap<u32, FxHashMap<u32, u32>> {
        &self.states_to_token_subsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let vocabulary = Vocabulary::new(Some(4))
            .insert("blah", 0)
            .insert("1a", 1)
            .insert("2", 2)
            .insert("0", 3)
            .insert("<eos>", 4);

        let index = Index::from_regex(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.initial(), 40);
        assert_eq!(index.final_states(), &FxHashSet::from_iter([24, 48, 56]));
        assert_eq!(
            "{24: {3: 24, 4: 24, 2: 24}, 48: {4: 48}, 40: {3: 48, 2: 56}, 56: {3: 24, 4: 56, 2: 24}}",
            format!("{:?}", index.transitions())
        );
    }
}
