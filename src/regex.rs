use crate::*;
use std::collections::{HashMap, HashSet};

pub fn walk_fsm(
    fsm_transitions: &HashMap<(State, TransitionKey), State>,
    _fsm_initial: State,
    fsm_finals: &HashSet<State>,
    token_transition_keys: &[TransitionKey],
    start_state: State,
    full_match: bool,
) -> Vec<State> {
    let mut state = start_state;
    let mut accepted_states = Vec::new();
    let mut last_final_idx = 0;

    for (i, &trans_key) in token_transition_keys.iter().enumerate() {
        match fsm_transitions.get(&(state, trans_key)) {
            Some(&new_state) => {
                state = new_state;
                if fsm_finals.contains(&state) {
                    last_final_idx = i + 1;
                }
                accepted_states.push(state);
            }
            None => {
                if !full_match && last_final_idx > 0 {
                    return accepted_states[..last_final_idx].to_vec();
                }
                return Vec::new();
            }
        }
    }

    if full_match && last_final_idx != token_transition_keys.len() {
        return Vec::new();
    }

    accepted_states
}

pub fn state_scan_tokens(
    fsm_transitions: &HashMap<(State, TransitionKey), State>,
    fsm_initial: State,
    fsm_finals: &HashSet<State>,
    vocabulary: &[(String, Vec<TokenId>)],
    vocabulary_transition_keys: &[Vec<TransitionKey>],
    start_state: State,
) -> HashSet<(TokenId, State)> {
    let mut res = HashSet::new();

    for (vocab_item, token_transition_keys) in
        vocabulary.iter().zip(vocabulary_transition_keys.iter())
    {
        let token_ids: Vec<TokenId> = vocab_item.1.clone();

        let state_seq = walk_fsm(
            fsm_transitions,
            fsm_initial,
            fsm_finals,
            token_transition_keys,
            start_state,
            false,
        );

        if state_seq.len() < token_transition_keys.len() {
            continue;
        }

        for &token_id in &token_ids {
            res.insert((token_id, *state_seq.last().unwrap()));
        }
    }

    res
}

pub fn get_token_transition_keys(
    alphabet_symbol_mapping: &HashMap<String, TransitionKey>,
    alphabet_anything_value: TransitionKey,
    token_str: &str,
) -> Vec<TransitionKey> {
    let mut token_transition_keys = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = token_str.chars().collect();

    while i < chars.len() {
        let symbol;
        if chars[i] == '\0' && i != chars.len() - 1 {
            if i + 2 < chars.len() {
                symbol = format!("\0{}{}", chars[i + 1], chars[i + 2]);
                i += 3;
            } else {
                symbol = chars[i].to_string();
                i += 1;
            }
        } else {
            symbol = chars[i].to_string();
            i += 1;
        }

        let transition_key = *alphabet_symbol_mapping
            .get(&symbol)
            .unwrap_or(&alphabet_anything_value);
        token_transition_keys.push(transition_key);
    }

    token_transition_keys
}

pub fn get_vocabulary_transition_keys(
    alphabet_symbol_mapping: &HashMap<String, TransitionKey>,
    alphabet_anything_value: TransitionKey,
    vocabulary: &[(String, Vec<TokenId>)],
    frozen_tokens: &HashSet<String>,
) -> Vec<Vec<TransitionKey>> {
    let mut vocab_transition_keys: Vec<Vec<TransitionKey>> = Vec::new();

    for item in vocabulary.iter() {
        let token_str = item.0.clone();

        let mut token_transition_keys;

        // Since these tokens are not expanded into byte-level transitions, we
        // can simply get their transition keys directly.
        if frozen_tokens.contains(&token_str) {
            token_transition_keys = Vec::new();
            token_transition_keys.push(
                *alphabet_symbol_mapping
                    .get(&token_str)
                    .unwrap_or(&alphabet_anything_value),
            )
        } else {
            token_transition_keys = get_token_transition_keys(
                alphabet_symbol_mapping,
                alphabet_anything_value,
                &token_str,
            );
        }

        vocab_transition_keys.push(token_transition_keys);
    }

    vocab_transition_keys
}
