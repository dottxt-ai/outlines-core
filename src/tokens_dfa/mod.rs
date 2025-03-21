mod token_classes;
mod token_classes_graph;
mod transitions_table;
mod regex;
mod reduce;

use std::collections::hash_map::Entry;

use bincode::{Decode, Encode};

use reduce::{build_prefix_based_graphes, minimizing_alphabet, mute_literals};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use regex_automata::util::alphabet::Unit;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use token_classes_graph::{PrefixGraph, PrefixGraphes};
pub use transitions_table::MasksTable;

use crate::prelude::*;
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};

pub use regex::compile_dead_byte_classes;




#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct TokensDFA
 {
    pub eos_token_id:u32,
    pub eos_class_id:u32,
    pub start_state: StateId,
    pub final_states: HashSet<StateId>,
    pub transitions_table: MasksTable,
    

}

impl TokensDFA
{

    pub fn new(regex: &str, vocabulary: &Vocabulary)-> Result<Self>
    
    {
        let eos_token_id = vocabulary.eos_token_id();

        let mut additionnal_tokens: Vec<(Vec<u8>, TokenId)> =Vec::new();
       
        let (muted_regex, muted_list) = mute_literals(regex, vocabulary, &mut additionnal_tokens);
        

        let real_len: usize = if vocabulary.len_alphabet() < eos_token_id as usize {
            eos_token_id as usize
        } else {
            vocabulary.len_alphabet()
        };

        let alphabet_len = real_len + 1 + additionnal_tokens.len(); // Real number of different token_id
        
        
        let dfa = DFA::new(&muted_regex).map_err(Box::new)?;
        
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::DfaHasNoStartState),
        };
            
        let mut transitions_table = MasksTable::new(alphabet_len); 
        //if byte_classes.alphabet_len() <- Gives the number of different classes
        // We can introduce different behaviors depending on the result of this previous line.
        // Result is in [2..257].  '2' means really permissive regex. '257' means "pain in the ass" regex
        let byte_classes = dfa.byte_classes();
        let dead_byte_classes:HashSet<u8> = compile_dead_byte_classes(&muted_regex, &byte_classes);
        

        let mut final_states:HashSet<StateId> = HashSet::default();
        
        let mut graphes: PrefixGraphes = PrefixGraphes::new();
        
        let eos_class_id = minimizing_alphabet(
            vocabulary.tokens(), 
            &additionnal_tokens, 
            byte_classes, 
            dead_byte_classes, 
            transitions_table.get_equivalent_grid(), 
            eos_token_id);

        transitions_table.set_eos_token_class_id(eos_class_id);

        build_prefix_based_graphes(
            &transitions_table.get_equivalent_grid(), 
            &mut graphes);
        
        let mut state_map: HashMap<AutomataStateId, StateId> = HashMap::default();
        let mut seen_states: HashSet<AutomataStateId> = HashSet::from_iter([start_state]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];
        let mut state_counter: StateId = 0;
        
        let initial_state_id = *state_map.entry(start_state)
                .or_insert_with(|| {
                    state_counter += 1;
                    state_counter - 1
                });
        
        let mut allowed_prefixes: Vec<u8> = vec![];
        let mut allowed_graphes : Vec<&PrefixGraph> = vec![];
        
        while let Some(current_state) = next_states.pop() {
            
            let current_state_id = *state_map.entry(current_state)
                .or_insert_with(|| {
                    state_counter += 1;
                    state_counter - 1
                });
            

            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state_id);
            }
            
            dfa.get_valid_classes_from_state(current_state, &mut allowed_prefixes);
           
            graphes.get_graphes_from_prefix(&allowed_prefixes, &mut allowed_graphes);

            for graph in &allowed_graphes{

              
                let mut graph_iterator = graph.iterator();
                graph_iterator.init();
               
                let mut remember_vec = Vec::new();
                let representative_bytes: Vec<Unit> = byte_classes.representatives(..).collect();

                while let Some(current_node) = graph_iterator.get_current() {
                    
                    let token_class = transitions_table.get_equivalent_grid().get_class_from_class_id(current_node.get_class_id()).clone();

                    
                    let mut valid = true;
                    let mut prefix_len = 0;
                    let mut temp_state = current_state;
                    if let Some((p_l, jump_state)) = remember_vec.pop() {
                        prefix_len = p_l;
                        temp_state = jump_state;
                    }
                
                    let token_bytes = token_class.as_bytes();
                    let bytes_to_process = &token_bytes[prefix_len..];

                    for &class_byte in bytes_to_process {
                        
                        let rep_byte = representative_bytes[class_byte as usize];

                        temp_state = dfa.next_state(temp_state, rep_byte.as_u8().unwrap());
                        if dfa.is_dead_state(temp_state) || dfa.is_quit_state(temp_state) {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                       
                        let is_intermediate = !dfa.is_match_state(temp_state);
                        let is_final = dfa.is_match_state(dfa.next_eoi_state(temp_state));
                        
                        if is_final || is_intermediate {
                           
                            let entry = state_map.entry(temp_state);
                            let next_state_id = match entry {
                                Entry::Occupied(occupied) => *occupied.get(),
                                Entry::Vacant(vacant) => {
                                    state_counter += 1;
                                    *vacant.insert(state_counter - 1)
                                }
                            };
                            transitions_table.add_transition(&current_state_id, current_node.get_class_id(), &next_state_id);
                            
                            if seen_states.insert(temp_state) {
                                next_states.push(temp_state);
                            }
                        }
                        
                       
                        if current_node.get_child().len() > 0 {
                            remember_vec.push((token_bytes.len(), temp_state));
                        }
                        graph_iterator.accept_and_advance();
                    
                    } else {
                        graph_iterator.reject_and_advance();
                    }
                    
                }
             
            }
            
        }

        for &final_state in &final_states {
            transitions_table.add_transition(&final_state, eos_class_id, &final_state);
        }

        transitions_table.reduce(muted_list, &mut final_states);
        
         Ok(
        TokensDFA{ 
            eos_token_id, 
            eos_class_id: eos_class_id,
            start_state: initial_state_id, 
            final_states, 
            transitions_table: std::mem::take(&mut transitions_table)
        }
       )
    }

}



