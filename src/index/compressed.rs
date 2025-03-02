use std::mem::{size_of, size_of_val};

use bincode::{Decode, Encode};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::index::{Index, IndexBehavior, TokenIdIterator};
use crate::prelude::*;

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct CompressedIndex {
    initial_state: StateId,
    final_states: HashSet<StateId>,

    pub state_to_index: HashMap<StateId, usize>,
    pub state_offsets: Vec<usize>,
    pub next_states: Vec<StateId>,

    pub token_masks: Vec<Vec<u64>>,

    eos_token_id: TokenId,
    vocab_size: usize,
    transitions: HashMap<StateId, HashMap<TokenId, StateId>>, // Useless but needed to be IndexBehavior Compliant
}

impl CompressedIndex {
    pub fn new(index: &Index, eos_token_id: TokenId) -> Self {
        let vocab_size = index.vocab_size();
        let final_states = index.final_states().clone();

        let mut state_to_index = HashMap::default();

        let mut all_states = HashSet::default();
        for (&state_id, transitions) in index.transitions() {
            all_states.insert(state_id);
            for &next_state in transitions.values() {
                all_states.insert(next_state);
            }
        }
        for (idx, &state_id) in all_states.iter().enumerate() {
            state_to_index.insert(state_id, idx);
        }

        let bits_per_state = ((vocab_size + 1 + 63) / 64) * 64; // +1 for the eoi_token
        let words_per_state = bits_per_state / 64;

        let mut token_masks = vec![vec![0u64; words_per_state]; state_to_index.len()];
        let mut next_states = Vec::new();
        let mut state_offsets = vec![0; state_to_index.len()];

        let mut current_offset = 0;

        let mut sorted_states: Vec<_> = state_to_index.iter().collect();
        sorted_states.sort_by_key(|&(_, &idx)| idx);

        for (state_id, idx) in sorted_states {
            state_offsets[*idx] = current_offset;

            if let Some(transitions) = index.transitions().get(state_id) {
                let mut sorted_transitions: Vec<_> = transitions.iter().collect();
                sorted_transitions.sort_by_key(|(&token, _)| token);

                for (&token, &next_state) in sorted_transitions {
                    let word_idx = (token as usize) / 64;
                    let bit_idx = (token as usize) % 64;
                    token_masks[*idx][word_idx] |= 1u64 << bit_idx;
                    next_states.push(next_state);
                }
                current_offset += transitions.len();
            }
        }

        CompressedIndex {
            initial_state: index.initial_state(),
            final_states,
            state_to_index,
            next_states,
            token_masks,
            state_offsets,
            eos_token_id,
            vocab_size,
            transitions: HashMap::default(),
        }
    }

    pub fn memory_usage(&self) -> usize {
        let struct_size = size_of::<CompressedIndex>();

        let initial_state_size = size_of::<StateId>();
        let eos_token_id_size = size_of::<TokenId>();
        let vocab_size_size = size_of::<usize>();

        let final_states_size =
            size_of_val(&self.final_states) + (self.final_states.len() * size_of::<StateId>());

        let mut token_masks_size = 0;
        for mask in &self.token_masks {
            token_masks_size += size_of::<Vec<u64>>() + (mask.len() * size_of::<u64>());
        }

        let next_states_container_size = size_of::<Vec<StateId>>();
        let next_states_content_size = self.next_states.len() * size_of::<StateId>();
        let total_next_states_size = next_states_container_size + next_states_content_size;

        let state_offsets_size =
            size_of::<Vec<usize>>() + (self.state_offsets.len() * size_of::<usize>());

        let mapping_size = size_of_val(&self.state_to_index);
        let mapping_content_size =
            self.state_to_index.len() * (size_of::<StateId>() + size_of::<usize>());
        let total_mapping_size = mapping_size + mapping_content_size;

        let total_size = struct_size
            + final_states_size
            + token_masks_size
            + state_offsets_size
            + total_next_states_size
            + initial_state_size
            + eos_token_id_size
            + vocab_size_size
            + total_mapping_size;

        println!("CompressedIndex memory usage:");
        println!("  Structure de base: {} bytes", struct_size);
        println!("  initial_state: {} bytes", initial_state_size);
        println!("  final_states: {} bytes", final_states_size);
        println!("  state_to_index mapping: {} bytes", total_mapping_size);
        println!("  token_masks: {} bytes", token_masks_size);
        println!("  next_states: {} bytes", total_next_states_size);
        println!("  state_offsets: {} bytes", state_offsets_size);
        println!("  eos_token_id: {} bytes", eos_token_id_size);
        println!("  vocab_size: {} bytes", vocab_size_size);
        println!(
            "Total memory usage: {} bytes ({:.2} MB)",
            total_size,
            total_size as f64 / (1024.0 * 1024.0)
        );

        total_size
    }
}

impl IndexBehavior for CompressedIndex {
    fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if token_id == &self.eos_token_id {
            return None;
        }

        let state_idx = *self.state_to_index.get(state)?;

        let word_idx = (*token_id as usize) / 64;
        let bit_idx = (*token_id as usize) % 64;

        if word_idx >= self.token_masks[state_idx].len()
            || (self.token_masks[state_idx][word_idx] & (1u64 << bit_idx)) == 0
        {
            return None;
        }

        // Check if state has transitions
        let transition_count = if state_idx + 1 < self.state_offsets.len() {
            self.state_offsets[state_idx + 1] - self.state_offsets[state_idx]
        } else {
            self.next_states.len() - self.state_offsets[state_idx]
        };
        if transition_count == 0 {
            return None;
        }

        let offset = self.state_offsets[state_idx];
        let mut pos = 0;

        // Reconstruire la liste triée des token_id actifs
        for w in 0..word_idx {
            // Compter tous les bits à 1 dans les mots précédents
            pos += self.token_masks[state_idx][w].count_ones() as usize;
        }

        // Compter les bits à 1 dans le mot courant jusqu'à bit_idx
        let mask_before = (1u64 << bit_idx) - 1;
        pos += (self.token_masks[state_idx][word_idx] & mask_before).count_ones() as usize;

        // Vérifier que la position est valide
        if pos < transition_count {
            return Some(self.next_states[offset + pos]);
        }

        None
    }

    fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        let state_idx = *(self.state_to_index.get(state)?);
        if state_idx >= self.token_masks.len() {
            return None;
        }

        let mut tokens = Vec::new();
        for (word_idx, &word) in self.token_masks[state_idx].iter().enumerate() {
            let base = word_idx * 64;
            for bit_idx in 0..64 {
                if word & (1u64 << bit_idx) != 0 {
                    tokens.push((base + bit_idx) as TokenId);
                }
            }
        }
        Some(tokens)
    }

    fn allowed_tokens_mask(&self, state: &StateId) -> Option<&Vec<u64>> {
        let state_idx = *(self.state_to_index.get(state)?);
        if state_idx >= self.token_masks.len() {
            return None;
        }

        Some(&self.token_masks[state_idx])
    }

    fn initial_state(&self) -> StateId {
        self.initial_state
    }

    fn final_states(&self) -> &HashSet<StateId> {
        &self.final_states
    }

    fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        &self.transitions
    }

    fn is_final_state(&self, state: &StateId) -> bool {
        self.final_states.contains(state)
    }

    fn allowed_tokens_iter(&self, _state: &StateId) -> Option<TokenIdIterator> {
        None
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }
}

impl std::fmt::Display for CompressedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CompressedIndex object")?;

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
        let index =
            IndexVariant::create_compressed(regex, &vocabulary).expect("Compressed Index failed");
        let initial_state = index.initial_state();
        assert_eq!(initial_state, 40);
        assert_eq!(index.final_states(), &HashSet::from_iter([24, 48, 56]));
        assert!(!index.is_final_state(&initial_state));

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
        let mut vocabulary = Vocabulary::new(3);
        for (token, token_id) in [("\n", 2), (".", 1), ("`", 0)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }

        let index =
            IndexVariant::create_compressed(regex, &vocabulary).expect("CompressedIndex failed");

        let allowed = index
            .allowed_tokens(&index.initial_state())
            .expect("No allowed tokens");
        assert!(allowed.contains(&0));
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

        let index = IndexVariant::create_compressed(regex, &vocabulary).expect("Index failed");
        assert_eq!(index.final_states(), &HashSet::from_iter([208, 128]));
    }

    #[test]
    fn test_allowed_tokens_mask() {
        let mut vocabulary = Vocabulary::new(3);

        for (token, token_id) in [
            (vec![32, 240, 159, 152], 2),
            (vec![32, 240, 159, 152, 141], 1),
            (vec![240, 159, 152, 141], 0),
        ] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = IndexVariant::create_compressed("[ ]?.?", &vocabulary).unwrap();
        let initial_state = index.initial_state();

        let mask = index.allowed_tokens_mask(&initial_state).unwrap();
        let expect_mask: Vec<u64> = vec![15]; // Bits 0, 1, 2, 3 activés (0b1111 = 15)
        assert_eq!(mask, &expect_mask);
    }

    #[test]
    fn test_index_memory_size() {
        //let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}"#;

        // Generate regex from schema
        //let regex = json_schema::regex_from_str(schema, None).unwrap();

        let regex: &str = r".{1,255}$";
        // println!("Generated regex: {}", regex);

        let vocabulary = Vocabulary::from_pretrained("gpt2", None).unwrap();

        let index = IndexVariant::create_compressed(regex, &vocabulary).expect("Index failed");

        if let IndexVariant::Compressed(compressed_index) = &index {
            let memory_usage = compressed_index.memory_usage();
            println!("CompressedIndex utilise {} bytes", memory_usage);
        } else {
            println!("L'index n'est pas un CompressedIndex");
        }
    }

    #[test]
    fn test_compressed_sequence() {
        let regex = ".{50}";
        let vocab = Vocabulary::from_pretrained("gpt2", None).unwrap();
        let index = Index::new(regex, &vocab).unwrap();
        let compressed = CompressedIndex::new(&index, vocab.eos_token_id());

        let mut state = 96;
        let mut compressed_state = 96;
        let tokens = vec![17081, 17081, 807, 48773, 11303, 41065, 5504, 11693, 31739];

        for &token in &tokens {
            let next = index.next_state(&state, &token);
            let compressed_next = compressed.next_state(&compressed_state, &token);
            println!(
                "Token {}: Standard {} -> {:?}, Compressed {} -> {:?}",
                token, state, next, compressed_state, compressed_next
            );
            assert_eq!(next, compressed_next, "Divergence at token {}", token);
            if let Some(next_state) = next {
                state = next_state;
            }
            if let Some(compressed_next_state) = compressed_next {
                compressed_state = compressed_next_state;
            }
        }
    }
}
