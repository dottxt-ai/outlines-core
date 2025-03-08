use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use bincode::{Decode, Encode};

use crate::prelude::*;

use super::{token_classes::TokenClassId, TokenClasses, token_classes::TokenIdsByClass};

pub type TokenIdMask = Vec<u64>;

#[derive(Clone, Debug, PartialEq, Encode, Decode, Default)]
pub struct MasksTable {
    alphabet_len: usize,
    eos_token_class_id: TokenClassId,
    token_classes: TokenClasses,
    classes_len : usize,
    

    
    /// The bitset masks of transitions for every state.
    /// masks[state_id] -> The mask of allowed token_ids for the State(state_id)
    /// Every bit in masks[state_id] represents a token_id. if egal 1 then the Token(token_id) is allowed for the State(state_id)
    masks: Vec<Vec<u64>>,

    /// The destination state for every (state_id; class_id) transitions
    /// next_states\[i\][pos] -> i : state_id, pos : index of the class_id in
    next_states: Vec<HashMap<TokenClassId,StateId>>,
    
    /// temp_transitions\[3\] means the transitions table for state_id = 3  
    /// Vec<HashMap<TokenClassId, StateId>> 
    temp_transitions: Vec<HashMap<TokenClassId, StateId>>, // temp_transitions[3] means the transitions table for state_id = 3   
    token_ids_by_class : TokenIdsByClass
}


impl MasksTable {
    
    pub fn new(alphabet_len:usize) -> Self {
       
        MasksTable{
            alphabet_len,
            eos_token_class_id:0 as TokenClassId,
            token_classes: TokenClasses::new(),
            classes_len: 0 as usize,
            token_ids_by_class: TokenIdsByClass::with_capacity(0),
            masks: Vec::new(),
            next_states: Vec::new(),
            temp_transitions : Vec::new()
        }
    }

    pub fn set_eos_token_class_id(&mut self, eos_token_class_id:TokenClassId){
        self.eos_token_class_id = eos_token_class_id;
    }

    /// Return an estimation of the size of the TokensDFA in bytes
    pub fn size(&self) -> usize {

        let mut sum = self.token_classes.size();

        let masks_size = std::mem::size_of::<Vec<Vec<u64>>>() + 
        (self.masks.capacity() * std::mem::size_of::<Vec<u64>>()) + 
        self.masks.iter().map(|v| v.capacity() * std::mem::size_of::<u64>()).sum::<usize>(); 
        
        
        let next_states_size = std::mem::size_of::<Vec<HashMap<TokenClassId, StateId>>>() + // Taille du Vec externe
            self.next_states.capacity() * std::mem::size_of::<HashMap<TokenClassId, StateId>>() + // Capacité allouée pour les HashMap
            self.next_states.iter().map(|map| {
                
                let bucket_count = map.capacity();
                std::mem::size_of::<usize>() * bucket_count + 
                map.len() * (std::mem::size_of::<TokenClassId>() + std::mem::size_of::<StateId>() + std::mem::size_of::<usize>())
            }).sum::<usize>();
       
       sum += masks_size + next_states_size;
       sum
    }
    

    pub fn get_token_classes(&mut self) -> &mut TokenClasses {
        &mut self.token_classes
    }

    pub fn get_token_ids_by_class(&mut self) -> &mut TokenIdsByClass {
        &mut self.token_ids_by_class
    }

    pub fn add_transition(&mut self, departure: &StateId, class_id: TokenClassId, arrival: &StateId){
        
        if *departure as usize >= self.temp_transitions.len() {
            for _ in 0..((departure+1) as usize - self.temp_transitions.len()){
                self.temp_transitions.push(HashMap::default());
            }
        }

        self.temp_transitions.get_mut(*departure as usize).unwrap().insert(class_id, *arrival);
    }

    pub fn allowed_transitions(&self, state:&StateId) -> Option<&Vec<u64>> {
        if *state as usize >= self.masks.len() {return None;}
        return Some(&self.masks[*state as usize]);
    }

    pub fn next_state(&self, state_id: &StateId, token_id: &TokenId) -> Option<StateId> {
        
        let class_id = self.token_classes[*token_id];
        
        return Some(*self.next_states[*state_id as usize].get(&class_id)?);
    }
    /// WARNING : VERY COSTLY FUNCTION
    pub fn get_transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>>{
        let mut transition_map: HashMap<u32, HashMap<u32, u32>> = HashMap::default();

        for (state_id, transitions) in self.next_states.iter().enumerate() {
            let mut token_map = HashMap::default();

            for (&token_id, &class_id) in self.token_classes.iter() {
                if let Some(&next_state) = transitions.get(&class_id) {
                    token_map.insert(token_id, next_state);
                }
            }

            if !token_map.is_empty() {
                transition_map.insert(state_id as u32, token_map);
            }
        }

        transition_map
    }

    /// Reduce the transitions table by building masks from temp_transitions ;
    
    pub fn reduce(&mut self, muted_list:HashSet<TokenId>) {
        let bits_per_state = ((self.alphabet_len + 1 + 63) / 64) * 64; // +1 for the eoi_token
        let words_per_state = bits_per_state / 64;
        
        
        self.masks = vec![vec![0u64; words_per_state]; self.temp_transitions.len()];
        self.next_states = vec![HashMap::default(); self.temp_transitions.len() as usize];

        
        self.classes_len = self.token_classes.len();
     
        for (idx, map_state_transitions) in self.temp_transitions.iter().enumerate(){
            // For every transition (class_id -> next_state_id) of the state_id 
            for(class_id, next_state_id) in map_state_transitions {
                // For every Token(token_id) belonging to the Class(class_id)
                // TokenIds are sorted inside token_ids_by_class.
                //let mut real_class_id = class_id.clone();
                //let tokens = self.token_ids_by_class.get_token_ids(*class_id);
                
                // Tokens which have been muted in determinist segment can share 
                // classes with other tokens in non-determinist segment.
                // So, We have to check that the class contains only one token
                // before the unmute.
                // if tokens.len() == 1 && muted_list.contains(&tokens[0]) {

                //     real_class_id = self.token_classes[tokens[0]];
                // }
                let tokens = self.token_ids_by_class.get_token_ids(*class_id);
                for token_id in tokens {
                    mask_set_token_id_unchecked(&mut self.masks[idx], *token_id);
                }
                
                let mut real_class_id = class_id.clone();

                if tokens.len() == 1 && muted_list.contains(&tokens[0]) { 
                    
                    real_class_id = self.token_classes[tokens[0]]
                }


                self.next_states[idx].insert(real_class_id, *next_state_id);
                
            }
                 
        }
        self.temp_transitions.clear();
        self.token_ids_by_class = TokenIdsByClass::with_capacity(0)
    }
}

impl std::fmt::Display for MasksTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        
        for row  in self.next_states.iter().enumerate(){
            writeln!(f, "Transitions for State {}", row.0)?;
            for class in row.1 {
                writeln!(f, "Token IDs: [{:?}] -> State : {:?}", self.token_ids_by_class.get_token_ids(*class.0), class.1)?;
            }
        }
       
        Ok(())
    }
}


fn mask_set_token_id_unchecked( mask: &mut TokenIdMask, token_id: TokenId){
    // if mask_len(mask) <= token_id as usize {
    //     panic!("mask_set_token_id() :: token_id superior to mask:TokenIdMask length");
    // }

    let word_idx = (token_id as usize) / 64;
    let bit_idx = (token_id as usize) % 64;
     mask[word_idx] |= 1u64 << bit_idx;
}


