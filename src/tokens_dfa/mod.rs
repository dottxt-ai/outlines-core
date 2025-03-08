mod token_classes;
mod token_classes_graph;
mod transitions_table;
mod regex;
mod reduce;

use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::*;

use bincode::{Decode, Encode};

use reduce::{init_classes_and_graph_optimized, mute_literals};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use regex_automata::util::alphabet::Unit;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use token_classes_graph::ReadOnlyTokenClassesGraphIterator;
pub use transitions_table::MasksTable;

use crate::{prelude::*, vocabulary};
use crate::vocabulary::Vocabulary;
use crate::{Error, Result};

pub use token_classes::TokenClasses;
pub use token_classes_graph::TokenClassesGraph;
pub use regex::compile_dead_byte_classes;

const GRAPH_THREADS_THRESHOLD:usize = 5;


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
        //let start_time = Instant::now();
        //let mutable_vocabulary: &mut Vocabulary =  &mut vocabulary.clone();
        //let time_vocabulary = start_time.elapsed();
        let mut additionnal_tokens: Vec<(Vec<u8>, TokenId)> =Vec::new();
        //let start_muting = Instant::now();
        let (muted_regex, muted_list) = mute_literals(regex, vocabulary, &mut additionnal_tokens);
        //let time_muting = start_muting.elapsed();
        // println!("> Muted List {:?}", muted_list);
        // println!("> Additionnal : {:?}", additionnal_tokens);
        let alphabet_len = vocabulary.len_alphabet()+additionnal_tokens.len(); // Real number of different token_id
        //let start_dfa = Instant::now();
        let dfa = DFA::new(&muted_regex).map_err(Box::new)?;
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::DfaHasNoStartState),
        };
        //let time_dfa = start_dfa.elapsed();
        
        //let start_minimizing = Instant::now();
        let mut transitions_table = MasksTable::new(alphabet_len); 
        //if byte_classes.alphabet_len() <- Gives the number of different classes
        // We can introduce different behaviors depending on the result of this previous line.
        // Result is in [2..257].  '2' means really permissive regex. '257' means "pain in the ass" regex
        let byte_classes = dfa.byte_classes();
        let mut dead_byte_classes:HashSet<u8> = compile_dead_byte_classes(&muted_regex, &byte_classes);
        
        //let time_compile_dead_byte = start_minimizing.elapsed();
        let eos_token_id = vocabulary.eos_token_id();
        let mut final_states:HashSet<StateId> = HashSet::default();
        let mut token_classes_graph = TokenClassesGraph::new();
        
        let eos_class_id = init_classes_and_graph_optimized(
            vocabulary.tokens(), 
            &additionnal_tokens,
            &mut token_classes_graph, 
            &mut transitions_table,
            byte_classes, 
            &mut dead_byte_classes,
            eos_token_id);
        
        //let time_minimizing = start_minimizing.elapsed();

        //let start_transitions = Instant::now();
        let mut state_map: HashMap<AutomataStateId, StateId> = HashMap::default();
        let mut seen_states: HashSet<AutomataStateId> = HashSet::from_iter([start_state]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];
        let mut state_counter: StateId = 0;
        
        let initial_state_id = *state_map.entry(start_state)
                .or_insert_with(|| {
                    state_counter += 1;
                    state_counter - 1
                });
        
       
        let dfa = Arc::new(dfa.clone());
        let byte_classes = Arc::new(byte_classes);
        let start_graph_read_only = Instant::now();
        let read_only_graph = token_classes_graph.to_read_only();
        let time_read_only = start_graph_read_only.elapsed();
        let active_threads = read_only_graph.get_roots().len() > GRAPH_THREADS_THRESHOLD;

        while let Some(current_state) = next_states.pop() {
            
            let current_state_id = *state_map.entry(current_state)
                .or_insert_with(|| {
                    state_counter += 1;
                    state_counter - 1
                });
            

            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state_id);
                
            }
            
            
            
            
            let roots = read_only_graph.get_roots();
            
            
            if active_threads {
                
                let local_state_map = Arc::new(Mutex::new(state_map));
                let local_transitions_table = Arc::new(Mutex::new(transitions_table));
                let local_seen_states = Arc::new(Mutex::new(seen_states));
                let local_next_states = Arc::new(Mutex::new(next_states));
                let local_states_counter = Arc::new(Mutex::new(state_counter));


                roots.par_iter()
                    .for_each(|root| {
                        let mut graph_iterator = ReadOnlyTokenClassesGraphIterator::new(&vec![root.1.get_first_node()]);
                        let mut current_class = graph_iterator.init();
                        let mut remember_vec = Vec::new();
                        let mut thread_local_seen: HashSet<AutomataStateId> = HashSet::default();

                        while let Some((token_class, class_id)) = current_class {
                            let mut valid = true;
                            let mut prefix_len = 0;
                            let mut temp_state = current_state;
                            if let Some((p_l, jump_state)) = remember_vec.pop() {
                                prefix_len = p_l;
                                temp_state = jump_state;
                            }
                            for &class_byte in &token_class.as_bytes()[prefix_len..] {
                                let rep_byte = byte_classes.elements(Unit::u8(class_byte)).next().unwrap();
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
                                    let next_state_id: u32;
                                    {
                                        let mut lock_state_map = local_state_map.lock().unwrap();
                                        next_state_id = *lock_state_map.entry(temp_state).or_insert_with(|| {
                                                        let mut counter = local_states_counter.lock().unwrap();
                                                        *counter += 1;
                                                        *counter - 1
                                                        });
                                    }
                                    {local_transitions_table.lock().unwrap().add_transition(&current_state_id, class_id, &next_state_id);}
                                    
                                   
                                    
                                    if !thread_local_seen.contains(&temp_state) {
                                        thread_local_seen.insert(temp_state);
                                        
                                        let mut seen = local_seen_states.lock().unwrap();
                                        if !seen.contains(&temp_state) {
                                            seen.insert(temp_state);
                                            local_next_states.lock().unwrap().push(temp_state);
                                        }
                                    }
                                }
                                if graph_iterator.has_child() {
                                    remember_vec.push((token_class.as_bytes().len(), temp_state));
                                }
                                current_class = graph_iterator.accept_and_advance();
                            } else {
                                current_class = graph_iterator.reject_and_advance();
                            }
                        }
                        
                    });
                
                state_map = match Arc::try_unwrap(local_state_map) {
                            Ok(mutex) => mutex.into_inner().unwrap(),
                            Err(arc) => {
                                // Fallback si d'autres références existent encore
                                arc.lock().unwrap().clone()
                            }
                };


                transitions_table = match Arc::try_unwrap(local_transitions_table) {
                    Ok(mutex) => mutex.into_inner().unwrap(),
                    Err(arc) => {
                        // Fallback au cas où d'autres références existent encore
                        arc.lock().unwrap().clone()
                    }
                };

                // Pour seen_states
                seen_states = match Arc::try_unwrap(local_seen_states) {
                    Ok(mutex) => mutex.into_inner().unwrap(),
                    Err(arc) => {
                        // Fallback au cas où d'autres références existent encore
                        arc.lock().unwrap().clone()
                    }
                };

                // Pour next_states
                next_states = match Arc::try_unwrap(local_next_states) {
                    Ok(mutex) => mutex.into_inner().unwrap(),
                    Err(arc) => {
                        // Fallback au cas où d'autres références existent encore
                        arc.lock().unwrap().clone()
                    }
                };

                state_counter = match Arc::try_unwrap(local_states_counter) {
                    Ok(mutex) => mutex.into_inner().unwrap(),
                    Err(arc) => {
                        // Fallback au cas où d'autres références existent encore
                        arc.lock().unwrap().clone()
                    }
                };
                
            } else {
                let transitions: Vec<(u32, u32, AutomataStateId)> = roots.iter()
                    .flat_map(|root| {
                        let mut graph_iterator = ReadOnlyTokenClassesGraphIterator::new(&vec![root.1.get_first_node()]);
                        let mut current_class = graph_iterator.init();
                        let mut transitions = Vec::new();
                        let mut remember_vec = Vec::new();

                        while let Some((token_class, class_id)) = current_class {
                            let mut valid = true;
                            let mut prefix_len = 0;
                            let mut temp_state = current_state;
                            if let Some((p_l, jump_state)) = remember_vec.pop() {
                                prefix_len = p_l;
                                temp_state = jump_state;
                            }
                            for &class_byte in &token_class.as_bytes()[prefix_len..] {
                                let rep_byte = byte_classes.elements(Unit::u8(class_byte)).next().unwrap();
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
                                    
                                    transitions.push((current_state_id, class_id, temp_state));
                                
                                }
                                if graph_iterator.has_child() {
                                    remember_vec.push((token_class.as_bytes().len(), temp_state));
                                }
                                current_class = graph_iterator.accept_and_advance();
                            } else {
                                current_class = graph_iterator.reject_and_advance();
                            }
                        }
                        transitions
                    })
                    .collect();
            

                for (state_depart, class_id, state_target) in transitions {
                    let next_state_id = *state_map.entry(state_target)
                    .or_insert_with(|| {
                                            state_counter += 1;
                                            state_counter - 1
                                        });
                    
                    transitions_table.add_transition(&state_depart, class_id, &next_state_id);
                    if !seen_states.contains(&state_target) {
                        seen_states.insert(state_target);
                        next_states.push(state_target);
                    }
                }
            }
                
        }

        
        for &final_state in &final_states {
            transitions_table.add_transition(&final_state, eos_class_id, &final_state);
        }

        //let time_transitions_table = start_transitions.elapsed();


        //let start_reducing = Instant::now();
        transitions_table.reduce(muted_list);
        //let time_reducing = start_reducing.elapsed();

        // println!("> Copy Vocabulary : {:?}", time_vocabulary);
        // println!("> Muting Literal : {:?}", time_muting);
        // println!("> DFA : {:?}", time_dfa);
        // println!("> Minimizing Alphabet : {:?} dont compile byte dead : {:?}", time_minimizing, time_compile_dead_byte);
        // println!("> Read-Only Graph : {:?}", time_read_only);
        // println!("> Transition table {:?}", time_transitions_table);
        // println!("> Reducing Table {:?}" , time_reducing);


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



