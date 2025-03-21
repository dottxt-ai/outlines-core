use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use bincode::{Decode, Encode};

use crate::prelude::*;

use super::token_classes::{TokenClass, TokenClassId};

pub type TokenIdMask = Vec<u64>;



/// The purpose of this structure is to centralize 
/// everything about the links between Tokens and TokensClass.
/// It allows to avoid multiple copies by value, 
/// and quicker access
#[derive(Clone, Debug, PartialEq, Encode, Decode, Default)]
pub struct EquivalentGrid{
    /// The current number of different classes
    size: usize,
    /// The list of all the TokenClasses in the Vocabulary.
    /// The position of the TokenClass in the vec is the TokenClassId
    classes: Vec<TokenClass>, 
    /// The list of class IDS sorted by the length of the TokenClasses
    sorted_classes: Vec<TokenClassId>,
    /// The list of token IDS sharing the same class id
    /// The position in the outer vec is the TokenClassId
    token_ids_by_class_id : Vec<Vec<TokenId>>,
    /// The TokenClassID for each TokenId
    /// The position in the vec is the TokenId
    class_id_by_token_id : Vec<TokenClassId>,

    classes_history: HashMap<TokenClass, TokenClassId>
    
}

impl EquivalentGrid {

    pub fn new(vocab_len: usize, class_nb: usize) -> Self {
   
        EquivalentGrid{
            size:0,
            classes: Vec::with_capacity(class_nb),
            sorted_classes: Vec::with_capacity(class_nb),
            token_ids_by_class_id: Vec::with_capacity(class_nb),
            class_id_by_token_id: vec![0;vocab_len],
            classes_history:HashMap::default()
        }

    }

    pub fn get_class_id_by_token_id(&self) -> &Vec<TokenClassId>{
        &self.class_id_by_token_id
    }

    #[inline(always)]
    pub fn insert_class(&mut self, class:TokenClass) -> TokenClassId {
        if let Some(idx) = self.classes_history.get(&class){
            return idx.clone();
        }

        let new_id = self.size;
        self.classes_history.insert(class.clone(), new_id as u32);
        self.classes.push(class.clone());
        self.token_ids_by_class_id.push(Vec::new());
        self.size += 1;
        
        new_id as u32
    }

    #[inline(always)]
    pub fn bind_token_id_and_class_id(&mut self, token_id: TokenId, class_id: TokenClassId){
        self.token_ids_by_class_id[class_id as usize].push(token_id);
        self.class_id_by_token_id[token_id as usize] = class_id;
    }

    #[inline(always)]
    pub fn mute_bind_token_id_and_class_id(&mut self, token_id: TokenId, class_id: TokenClassId){
        self.token_ids_by_class_id[class_id as usize].push(token_id);
    }

    #[inline(always)]
    pub fn get_class_id_from_token_id(&self, token_id: TokenId) -> &TokenClassId {
        &self.class_id_by_token_id[token_id as usize]
    }

    #[inline(always)]
    pub fn get_token_ids_from_class_id(&self, class_id: TokenClassId) -> &Vec<TokenId> {
        &self.token_ids_by_class_id[class_id as usize]
    }

    #[inline(always)]
    pub fn get_class_from_class_id(&self, class_id: TokenClassId) -> &TokenClass {
        &self.classes[class_id as usize]
    }

    #[inline(always)]
    pub fn get_token_id_position_in_class(&self, token_id: TokenId) -> usize {
        self.token_ids_by_class_id[*self.get_class_id_from_token_id(token_id) as usize].iter().position(|&x| x==token_id).unwrap()
    }

    #[inline]
    pub fn sort_classes(&mut self) {
        
        let indices: Vec<u32> = (0..self.size as u32).collect();
        self.sorted_classes.extend_from_slice(&indices);
        self.sorted_classes.sort_unstable_by_key(|&id| self.classes[id as usize].len());
       
    }

    #[inline(always)]
    pub fn get_sorted_classes(&self) -> &Vec<TokenClassId> {
        &self.sorted_classes
    }

    #[inline(always)]
    pub fn get_classes(&self) -> &Vec<TokenClass> {
        &self.classes
    }

    /// Reduce the memory used by the EquivalentGrid once the compilation is done.
    /// We keep only what we need to serve the guide.
    pub fn reduce(&mut self){
        // Do not need the token_ids_by_class_id
        self.token_ids_by_class_id.clear();
        self.token_ids_by_class_id.shrink_to_fit();

        // Resizing the class_id_by_token_id if the capacity was to large
        self.class_id_by_token_id.shrink_to_fit();

        // Do not need the classes, only need the class ids
        self.classes.clear();
        self.classes.shrink_to_fit();
    }

    pub fn memory_size(&self) -> usize {
        0
    }

}


#[derive(Clone, Debug, PartialEq, Encode, Decode, Default)]
pub struct MasksTable {

    vocab_len: usize,

    eos_token_class_id: TokenClassId,
        
    equivalent_grid: EquivalentGrid,
    
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

}


impl MasksTable {
    
    pub fn new(vocab_len:usize) -> Self {
        let classes_nb = (vocab_len + 255) >> 8;
        MasksTable{
            vocab_len,
            eos_token_class_id:0 as TokenClassId,
            equivalent_grid: EquivalentGrid::new(vocab_len, classes_nb),
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

        let mut sum = self.equivalent_grid.memory_size();

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
    
    #[inline(always)]
    pub fn get_equivalent_grid(&mut self) -> &mut EquivalentGrid {
        &mut self.equivalent_grid
    }


    pub fn add_transition(&mut self, departure: &StateId, class_id: TokenClassId, arrival: &StateId){
        
        if *departure as usize >= self.temp_transitions.len() {
            self.temp_transitions.resize_with(
                (*departure as usize) + 1,
                HashMap::default
            );
        }
    
        self.temp_transitions[*departure as usize].insert(class_id, *arrival);
    }

    pub fn allowed_transitions(&self, state:&StateId) -> Option<&Vec<u64>> {
        if *state as usize >= self.masks.len() {return None;}
        return Some(&self.masks[*state as usize]);
    }

    pub fn next_state(&self, state_id: &StateId, token_id: &TokenId) -> Option<StateId> {
        
        let class_id = *self.equivalent_grid.get_class_id_from_token_id(*token_id);
        
        return Some(*self.next_states[*state_id as usize].get(&class_id)?);
    }
    /// WARNING : VERY COSTLY FUNCTION
    pub fn get_transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>>{
        let mut transition_map: HashMap<u32, HashMap<u32, u32>> = HashMap::default();

        for (state_id, transitions) in self.next_states.iter().enumerate() {
            let mut token_map = HashMap::default();

            for (token_id, &class_id) in self.equivalent_grid.get_class_id_by_token_id().iter().enumerate() {
                if let Some(&next_state) = transitions.get(&class_id) {
                    token_map.insert(token_id as u32, next_state);
                }
            }

            if !token_map.is_empty() {
                transition_map.insert(state_id as u32, token_map);
            }
        }

        transition_map
    }

    /// Reduce the transitions table by building masks from temp_transitions ;
    
    pub fn reduce(&mut self, muted_list:HashSet<TokenId>, final_states: &mut HashSet<u32>) {
        let bits_per_state = ((self.vocab_len + 1 + 63) / 64) * 64; // +1 for the eoi_token
        let words_per_state = bits_per_state / 64;
        
        
        self.masks = vec![vec![0u64; words_per_state]; self.temp_transitions.len()];
        self.next_states = vec![HashMap::default(); self.temp_transitions.len() as usize];

        for (idx, map_state_transitions) in self.temp_transitions.iter_mut().enumerate(){
            // For every transition (class_id -> next_state_id) of the state_id 
            if map_state_transitions.is_empty() { 
                map_state_transitions.insert(self.eos_token_class_id, idx as u32);
                final_states.insert(idx as u32);
            }

            for(class_id, next_state_id) in map_state_transitions {
                // For every Token(token_id) belonging to the Class(class_id)
                // TokenIds are sorted inside token_ids_by_class.
                //let mut real_class_id = class_id.clone();
                //let tokens = self.token_ids_by_class.get_token_ids(*class_id);
                
                // Tokens which have been muted in determinist segment can share 
                // classes with other tokens in non-determinist segment.
                // So, We have to check that the class contains only one token
                // before the unmute.
               
                let tokens = self.equivalent_grid.get_token_ids_from_class_id(*class_id);
                for token_id in tokens {
                    mask_set_token_id_unchecked(&mut self.masks[idx], *token_id);
                }
                
                let mut real_class_id = class_id.clone();

                if tokens.len() == 1 && muted_list.contains(&tokens[0]) { 
                    real_class_id = *self.equivalent_grid.get_class_id_from_token_id(tokens[0]);
                }

                self.next_states[idx].insert(real_class_id, *next_state_id);
                
            }
                 
        }
        self.temp_transitions.clear();
        self.equivalent_grid.reduce();
    }
}

impl std::fmt::Display for MasksTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        
        for row  in self.next_states.iter().enumerate(){
            writeln!(f, "Transitions for State {}", row.0)?;
            for class in row.1 {
                writeln!(f, "Token IDs: [{:?}] -> State : {:?}", self.equivalent_grid.get_token_ids_from_class_id(*class.0), class.1)?;
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


