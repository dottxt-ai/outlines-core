//! Compare two Index Structures with one regex to be sure they have the same states and transitions.

#[cfg(test)]
mod tests {

    use crate::index::{CompressedIndex, Index, IndexBehavior};
    use crate::prelude::*;
    use crate::vocabulary::Vocabulary;

    fn mock_vocabulary() -> Vocabulary {
        Vocabulary::from_pretrained("gpt2", None).unwrap()
    }

    #[test]
    fn test_index_vs_compressed_index() {
        let regex = r".{5}$";
        let vocab = mock_vocabulary();
        let index = Index::new(regex, &vocab).expect("Failed to create Index");
        let compressed_index = CompressedIndex::new(&index, vocab.eos_token_id());

        assert_eq!(
            index.initial_state(),
            compressed_index.initial_state(),
            "Initial states differ"
        );
        assert_eq!(
            index.final_states(),
            compressed_index.final_states(),
            "Final states differ"
        );
        assert_eq!(
            index.eos_token_id(),
            compressed_index.eos_token_id(),
            "EOS token IDs differ"
        );
        assert_eq!(
            index.vocab_size(),
            compressed_index.vocab_size(),
            "Vocab sizes differ"
        );

        for (state_id, transitions) in index.transitions() {
            let idx = match compressed_index.state_to_index.get(state_id) {
                Some(&idx) => idx,
                None => panic!("State {} missing in CompressedIndex", state_id),
            };

            let offset = compressed_index.state_offsets[idx];
            let next_transitions_len = if idx + 1 < compressed_index.state_offsets.len() {
                compressed_index.state_offsets[idx + 1] - offset
            } else {
                compressed_index.next_states.len() - offset
            };

            let compressed_next_states =
                &compressed_index.next_states[offset..offset + next_transitions_len];

            // Count active tokens in the token mask
            let mut compressed_transitions = Vec::new();
            for word_idx in 0..compressed_index.token_masks[idx].len() {
                let mask = compressed_index.token_masks[idx][word_idx];
                for bit_idx in 0..64 {
                    if (mask & (1u64 << bit_idx)) != 0 {
                        let token_id = (word_idx * 64 + bit_idx) as TokenId;
                        compressed_transitions.push(token_id);
                    }
                }
            }

            // Check transitions count
            assert_eq!(
                transitions.len(),
                compressed_transitions.len(),
                "Transition count mismatch for state {}: Index has {}, CompressedIndex has {}",
                state_id,
                transitions.len(),
                compressed_transitions.len()
            );

            // Check every transitions
            for (token_id, expected_next_state) in transitions {
                let word_idx = (*token_id as usize) / 64;
                let bit_idx = (*token_id as usize) % 64;
                let mask = compressed_index.token_masks[idx][word_idx];

                assert!(
                    (mask & (1u64 << bit_idx)) != 0,
                    "Token {} not found in mask for state {}",
                    token_id,
                    state_id
                );

                // find the position of the token compressed_transitions
                let pos = compressed_transitions
                    .iter()
                    .position(|&t| t == *token_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "Token {} not found in compressed transitions for state {}",
                            token_id, state_id
                        )
                    });

                let next_state = compressed_next_states[pos];
                assert_eq!(
                    next_state, *expected_next_state,
                    "Next state mismatch for state {}, token {}: expected {}, got {}",
                    state_id, token_id, expected_next_state, next_state
                );
            }
        }

        assert_eq!(
            index.transitions().len(),
            compressed_index.state_to_index.len(),
            "Number of states differs"
        );
    }
}
