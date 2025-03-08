use bincode::{Decode, Encode};
use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};

use crate::tokens_dfa::TokensDFA;
use crate::vocabulary::Vocabulary;
use crate::Result;
use crate::primitives::{StateId, TokenId};


#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct V2Index {
    tokens_dfa: TokensDFA
}

impl V2Index {
    pub fn new(regex: &str, vocabulary: &Vocabulary) -> Result<Self>{
        
    let tokens_dfa = TokensDFA::new(regex, vocabulary)?;
       
       Ok(V2Index{
            tokens_dfa
        })
    }

    pub fn initial_state(&self) -> StateId {
        self.tokens_dfa.start_state
    }

    pub fn final_states(&self) -> &HashSet<StateId> {
        &self.tokens_dfa.final_states
    }

    pub fn transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>> {
        self.tokens_dfa.transitions_table.get_transitions()
    }

    pub fn is_final_state(&self, state: &StateId) -> bool {
        self.tokens_dfa.final_states.contains(state)
    }

    pub fn allowed_tokens(&self, state: &StateId) -> Option<&Vec<u64>> {
        self.tokens_dfa.transitions_table.allowed_transitions(state)
    }

    pub fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        if *token_id  == self.tokens_dfa.eos_token_id {
            return None;
        }

        self.tokens_dfa.transitions_table.next_state(state, &token_id)
    }

   

}

#[cfg(any(feature = "run_benchmarks", debug_assertions))]
impl V2Index{
    pub fn size(&self) -> usize {
        self.tokens_dfa.transitions_table.size()
    }
}

impl std::fmt::Display for V2Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index object with transitions:")?;
        writeln!(f, "{}", self.tokens_dfa.transitions_table)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use std::io::Write;

    use crate::json_schema;
    use crate::index::Index;

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
        let index = V2Index::new(regex, &vocabulary).expect("Index failed");
      
        assert_eq!(index.initial_state(), 0);
        assert_eq!(index.final_states(), &HashSet::from_iter([1,2,3]));
        assert!(!index.is_final_state(&index.initial_state()));
        
        let mut expected:Vec<u64> = Vec::new();
        expected.push(0);
        expected[0] |= 1 << 2;
        expected[0] |= 1 << 3;
        assert_eq!(index.allowed_tokens(&0).unwrap(), &expected);
        let state = 1;
        assert_eq!(index.next_state(&state, &eos_token_id), None);

        let allowed_token_id = 2;
        assert_eq!(index.next_state(&0, &allowed_token_id), Some(1));


    }

    #[test]
    fn index_from_regex_initial_in_allowed(){
        let regex = "`\\n(\\.\\n)?`\\n";
        let mut vocabulary = Vocabulary::new(3);
        for (token, token_id) in [("\n", 2), (".", 1), ("`", 0)] {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        let index = V2Index::new(regex, &vocabulary).expect("index failed");
        let allowed = index.allowed_tokens(&index.initial_state()).unwrap();
        let expect:Vec<u64> = vec![1];
        assert_eq!(*allowed, expect);
        
    }
    #[test]
    fn index_from_regex_multibyte() {
        let regex = "😇|(😈 😍)";
        let mut vocabulary = Vocabulary::new(4);
        for (token, token_id) in [(" 😍", 5), ("blah", 0), ("😇", 2), ("😈", 1), ("😍", 3)]
        {
            vocabulary
                .try_insert(token, token_id as u32)
                .expect("Insert failed");
        }
        
        let index = V2Index::new(regex, &vocabulary).expect("V2Index failed");
        
        let initial_state = index.initial_state();
        
        let mut expected:Vec<u64> = Vec::new();
        expected.push(0);
        expected[0] |= 1 << 1;
        expected[0] |= 1 << 2;
        let allowed_tokens = index.allowed_tokens(&initial_state).expect("No allowed tokens for intiial state");
        assert_eq!(expected, *allowed_tokens);

        let next_state = index.next_state(&initial_state, &2).expect("No Next state");
        assert!(index.final_states().contains(&next_state));

        let next_state_2 = index.next_state(&initial_state, &1).expect("No next state");
        expected[0] = 0;
        expected[0] |= 1 << 5;
        let allowed_tokens = index.allowed_tokens(&next_state_2).expect("No allowed tokens for next_state_2");
        assert_eq!(expected, *allowed_tokens);


    }

    #[test]
    fn test_sample(){
        let regex = r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?";
        //let sch =r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}"#;
        //let regex = &json_schema::regex_from_str(sch, None).unwrap(); 
         println!("{}", regex);
        let model_name = "unsloth/Llama-3.1-8B-Instruct";
        let vocab = Vocabulary::from_pretrained(model_name, None).unwrap();
        

        
        let start_optimized = Instant::now();
        let index_optimized = V2Index::new(regex, &vocab).expect("Failed to create Index with new_optimized");
        let duration_optimized = start_optimized.elapsed();

        println!("Time V2Index : {:?}", duration_optimized);
        
        let start_optimized = Instant::now();
        let indexd = Index::new(regex, &vocab).expect("Failed to create Index with new_optimized");
        let duration_optimized = start_optimized.elapsed();
        println!("Time Index : {:?}", duration_optimized);
       
       }

    #[test]
    fn test_minimal_index() {
        let mut vocab = Vocabulary::new(7);
        vocab.try_insert(b"a".to_vec(), 0).unwrap();
        vocab.try_insert(b"b".to_vec(), 1).unwrap();
        vocab.try_insert(b"c".to_vec(), 2).unwrap();
        vocab.try_insert(b"d".to_vec(), 3).unwrap();
        vocab.try_insert(b"e".to_vec(), 4).unwrap();
        vocab.try_insert(b"f".to_vec(), 5).unwrap();
        vocab.try_insert(b"abcd".to_vec(), 6).unwrap();

        let regex = "a[a|b]cd(ef){1,2}$";
        // With Muted litteral feature, regex is : (∟1)[a|b](∟2∟3)((∟4∟5)){1,2}$
        let v2_index = V2Index::new(regex, &vocab).unwrap();
        let index = Index::new(regex, &vocab).unwrap();
        let index_allowed_tokens = index.allowed_tokens(&index.initial_state()).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_index.initial_state()).unwrap();
        
        assert_eq!(index_allowed_tokens.len(), 2); // Token ID 0 and Token ID 6
        assert_eq!(index_allowed_tokens[0], 0); // Token A
        assert_eq!(index_allowed_tokens[1], 6); // Token ABCD
        assert_eq!(v2_allowed_tokens[0], 1); // BIT 0 activated

        let next_state = index.next_state(&index.initial_state(), &0).unwrap();
        let v2_next_state = v2_index.next_state(&v2_index.initial_state(), &0).unwrap();
      
        let index_allowed_tokens = index.allowed_tokens(&next_state).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(index_allowed_tokens.len(), 2); // Token ID 4
        assert_eq!(index_allowed_tokens[0], 0); // Token A
        assert_eq!(index_allowed_tokens[1], 1); // Token B
        assert_eq!(v2_allowed_tokens[0], 3); // BIT 0 and 1  activated

        let next_state = index.next_state(&next_state, &1).unwrap();
        let v2_next_state = v2_index.next_state(&v2_next_state, &1).unwrap();
       
        let index_allowed_tokens = index.allowed_tokens(&next_state).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(index_allowed_tokens.len(), 1); // Token ID 2
        assert_eq!(index_allowed_tokens[0], 2); // Token c
        assert_eq!(v2_allowed_tokens[0], 4); // BIT 3  activated

        let next_state = index.next_state(&next_state, &2).unwrap();
        let v2_next_state = v2_index.next_state(&v2_next_state, &2).unwrap();

        let index_allowed_tokens = index.allowed_tokens(&next_state).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(index_allowed_tokens.len(), 1); // Token ID 3
        assert_eq!(index_allowed_tokens[0], 3); // Token D
        assert_eq!(v2_allowed_tokens[0], 8); // BIT 3  activated

        let next_state = index.next_state(&next_state, &3).unwrap();
        let v2_next_state = v2_index.next_state(&v2_next_state, &3).unwrap();

        let index_allowed_tokens = index.allowed_tokens(&next_state).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(index_allowed_tokens.len(), 1); // Token ID 5
        assert_eq!(index_allowed_tokens[0], 4); // Token ID E
        assert_eq!(v2_allowed_tokens[0], 16); // BIT 4  activated


        let next_state = index.next_state(&next_state, &4).unwrap();
        let v2_next_state = v2_index.next_state(&v2_next_state, &4).unwrap();

        let index_allowed_tokens = index.allowed_tokens(&next_state).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(index_allowed_tokens.len(), 1); // Token ID 5
        assert_eq!(index_allowed_tokens[0], 5); // Token ID 5
        assert_eq!(v2_allowed_tokens[0], 32); // BIT 5  activated

        let next_state = index.next_state(&next_state, &5).unwrap();
        let v2_next_state = v2_index.next_state(&v2_next_state, &5).unwrap();

        assert!(index.is_final_state(&next_state));
        assert!(v2_index.is_final_state(&v2_next_state));

        let index_allowed_tokens = index.allowed_tokens(&next_state).unwrap();
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(index_allowed_tokens.len(), 2); // Token ID 4 and EOI
        assert_eq!(index_allowed_tokens[0], 7); // Token ID 4 and EOI
        assert_eq!(index_allowed_tokens[1], 4);
        assert_eq!(v2_allowed_tokens[0], 144); // BIT 4 and 7  activated


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
        let index = V2Index::new("[ ]?.?", &vocabulary).unwrap();
        let initial_state = index.initial_state();

        let mask = index.allowed_tokens(&initial_state).unwrap();
        let expect_mask: Vec<u64> = vec![15]; // Bits 0, 1, 2 activated
        assert_eq!(mask, &expect_mask);
        assert!(index.final_states().contains(&initial_state));

    }
    #[test]
    fn test_minimal_2_index() {
        let mut vocab = Vocabulary::new(3);
        vocab.try_insert(b"file".to_vec(), 0).unwrap();
        vocab.try_insert(b"-".to_vec(), 1).unwrap();
        vocab.try_insert(b"name".to_vec(), 2).unwrap();
        let regex = "file-name$";
       
        let v2_index = V2Index::new(regex, &vocab).unwrap();
        
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_index.initial_state()).unwrap();
        
        assert_eq!(v2_allowed_tokens[0], 1); // BIT 0 activated

        let v2_next_state = v2_index.next_state(&v2_index.initial_state(), &0).unwrap();
        
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(v2_allowed_tokens[0], 2); // BIT 1  activated

        let v2_next_state = v2_index.next_state(&v2_next_state, &1).unwrap();
        
        let v2_allowed_tokens = v2_index.allowed_tokens(&v2_next_state).unwrap();

        assert_eq!(v2_allowed_tokens[0], 4); // BIT 3 activated
    }

    
}


