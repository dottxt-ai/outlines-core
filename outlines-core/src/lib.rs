use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::thread;

pub const FLAG: bool = true;

pub fn create_fsm_index_end_to_end_rust(
    fsm_transitions: &BTreeMap<(i32, i32), i32>,
    alphabet_symbol_mapping: &BTreeMap<char, i32>,
    alphabet_anything_value: i32,
    fsm_initial: i32,
    fsm_finals: &BTreeSet<i32>,
    vocabulary: &BTreeMap<String, Vec<i32>>,
) -> BTreeMap<i32, BTreeSet<(i32, i32)>> {
    let mut states_to_token_subsets: BTreeMap<i32, BTreeMap<i32, i32>> = BTreeMap::new();
    let mut seen: BTreeSet<i32> = BTreeSet::new();
    let mut next_states = vec![fsm_initial];

    // TODO adjust dynamically
    let n_threads = 16;

    let tokens: Vec<String> = vocabulary.keys().cloned().collect();
    let _token_ids: Vec<Vec<i32>> = vocabulary.values().cloned().collect();

    let n_tokens = tokens.len();
    let _chunk_size = n_tokens / n_threads;

    let _start_time = std::time::Instant::now();

    while let Some(start_state) = next_states.pop() {
        let _start = std::time::Instant::now();
        let token_ids_end_states = state_scan_tokens(
            fsm_transitions,
            alphabet_symbol_mapping,
            alphabet_anything_value,
            fsm_initial,
            fsm_finals,
            vocabulary,
            start_state,
        );
        for token_id_and_end_state in token_ids_end_states {
            let end_state = token_id_and_end_state.1;
            if !seen.contains(&end_state) {
                next_states.push(end_state);
            }
            states_to_token_subsets
                .entry(start_state)
                .or_default()
                .insert(token_id_and_end_state.0, token_id_and_end_state.1);
        }
        // println!("state_scan_tokens: {:?}", start.elapsed());
        seen.insert(start_state);
    }

    let mut states_to_token_subsets_btree: BTreeMap<i32, BTreeSet<(i32, i32)>> = BTreeMap::new();

    for (k, v) in states_to_token_subsets.iter() {
        let token_subsets: BTreeSet<(i32, i32)> = v.iter().map(|(&k1, &v1)| (k1, v1)).collect();
        states_to_token_subsets_btree.insert(*k, token_subsets);
    }

    states_to_token_subsets_btree
}

fn walk_fsm(
    fsm_transitions: &BTreeMap<(i32, i32), i32>,
    alphabet_symbol_mapping: &BTreeMap<char, i32>,
    alphabet_anything_value: i32,
    _fsm_initial: i32,
    fsm_finals: &BTreeSet<i32>,
    input_string: &str,
    start_state: i32,
    full_match: bool,
) -> Vec<i32> {
    let mut state = start_state;
    let mut accepted_states = Vec::new();
    let mut is_final_state_reached = false;

    for symbol in input_string.chars() {
        let trans_key = alphabet_symbol_mapping
            .get(&symbol)
            .unwrap_or(&alphabet_anything_value);
        if let Some(&new_state) = fsm_transitions.get(&(state, *trans_key)) {
            state = new_state;
            if fsm_finals.contains(&state) {
                is_final_state_reached = true;
            }
            accepted_states.push(state);
        } else {
            // Exit early if not full match and a final state was reached before
            if !full_match && is_final_state_reached {
                break;
            }
            return Vec::new();
        }
    }

    if full_match && !is_final_state_reached {
        return Vec::new();
    }

    accepted_states
}

fn state_scan_tokens(
    fsm_transitions_map: &BTreeMap<(i32, i32), i32>,
    alphabet_symbol_mapping_map: &BTreeMap<char, i32>,
    alphabet_anything_value: i32,
    fsm_initial: i32,
    fsm_finals_set: &BTreeSet<i32>,
    vocabulary_map: &BTreeMap<String, Vec<i32>>,
    start_state: i32,
) -> Vec<(i32, i32)> {
    let _start_time = std::time::Instant::now();

    // TODO choose dynamically
    let mut n_threads = 16;

    // Convert fsm_transitions to BTreeMap and two vectors
    let mut tokens = Vec::new();
    let mut token_ids = Vec::new();

    for (k, v) in vocabulary_map.iter() {
        tokens.push(k.clone());
        token_ids.push(v.clone());
    }

    let n_tokens = tokens.len();
    let _chunk_size = n_tokens / n_threads;

    // Prepare for multithreading
    let fsm_transitions_map_arc = Arc::new(fsm_transitions_map);
    let alphabet_symbol_mapping_map_arc = Arc::new(alphabet_symbol_mapping_map);
    let fsm_finals_set_arc = Arc::new(fsm_finals_set);
    let tokens_arc = Arc::new(tokens);
    let token_ids_arc = Arc::new(token_ids);

    let mut token_chunks = Vec::new();
    let tokens_per_thread = (n_tokens as f32 / n_threads as f32).ceil() as usize;

    if n_tokens > 1000 {
        for i in 0..n_threads {
            let start = i * tokens_per_thread;
            let mut end = start + tokens_per_thread;

            // Make sure we don't go out of bounds on the last chunk
            if end > n_tokens {
                end = n_tokens;
            }

            // Only add chunks that have data to process
            if start < n_tokens {
                token_chunks.push((start, end));
            }
        }
    } else {
        n_threads = 1;
        token_chunks.push((0, n_tokens));
    }

    let all_outputs: Vec<Vec<(i32, i32)>> = thread::scope(|s| {
        (0..n_threads)
            .map(|thread_id| {
                let _start_time = std::time::Instant::now();

                let start = token_chunks[thread_id].0;
                let end = token_chunks[thread_id].1;

                let token_chunk = tokens_arc[start..end].to_vec();
                let token_ids_chunk = token_ids_arc[start..end].to_vec();

                let fsm_transitions_map_clone = Arc::clone(&fsm_transitions_map_arc);
                let alphabet_symbol_mapping_map_clone =
                    Arc::clone(&alphabet_symbol_mapping_map_arc);
                let fsm_finals_set_clone = Arc::clone(&fsm_finals_set_arc);
                let _token_ids_arc_clone = Arc::clone(&token_ids_arc);
                s.spawn(move || {
                    let mut res = Vec::new();
                    // zip the token_chunk with the token_ids_chunk
                    for i in 0..token_chunk.len() {
                        let token = &token_chunk[i];
                        let token_ids = &token_ids_chunk[i];
                        let state_seq = walk_fsm(
                            &fsm_transitions_map_clone,
                            &alphabet_symbol_mapping_map_clone,
                            alphabet_anything_value,
                            fsm_initial,
                            &fsm_finals_set_clone,
                            token,
                            start_state,
                            false,
                        );
                        if state_seq.len() < token.len() {
                            continue;
                        }

                        for token_id in token_ids {
                            res.push((*token_id, state_seq[state_seq.len() - 1]));
                        }
                    }
                    res
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            // wait for each thread to finish and collect their results
            .map(|handle| handle.join().expect("Thread failed"))
            .collect::<Vec<_>>()
    });
    let res = all_outputs.into_iter().flatten().collect();
    res
}
