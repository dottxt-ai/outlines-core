
use std::default;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use regex_automata::util::alphabet::ByteClasses;
use rayon::prelude::*;

use super::token_classes::TokenClassId;
pub use super::transitions_table::MasksTable;

use crate::prelude::*;
use crate::vocabulary::Vocabulary;

pub use super::token_classes::{TokenClass, MapTokenClassTokenClassId};
pub use super::token_classes_graph::TokenClassesGraph;
pub use super::regex::{extract_literals, replace_literals};

const MUTE_BYTE: u8 = 0x1C;  // En hexadécimal
const MUTE_INDEX_BYTE: u8 = 0x1E;


pub fn init_classes_and_graph_optimized(
    tokens: &HashMap<Vec<u8>, Vec<TokenId>>, // Token comme Vec<u8>
    additionnal_tokens: &Vec<(Vec<u8>, TokenId)>,
    token_classes_graph: &mut TokenClassesGraph, // Référence mutable inchangée
    transitions_table: &mut MasksTable, // Référence mutable inchangée
    byte_classes: &ByteClasses,
    dead_byte_classes: &mut HashSet<u8>, // Référence mutable inchangée
    eos_token_id: u32,
) -> TokenClassId {
    
    let mut results: Vec<(Vec<TokenId>, TokenClass)> = tokens
    .par_iter() 
    .filter_map(|(token, token_ids)| {
        if token_ids.contains(&eos_token_id) {
            return None; 
        }
        if token_ids[0] == 216 {return None;} // BUG IN THE VOCABULARY. token_id 216 ""\u011c"" is interpreted as \x1C, (Byte 28)
        let t_class = get_token_class(token, byte_classes);
        
        if t_class.as_bytes().iter().any(|byte| dead_byte_classes.contains(byte)) && !token_ids.iter().any(|&id| additionnal_tokens.iter().any(|(_, add_id)| *add_id == id)) {
            return None; 
        }
        
        Some((token_ids.clone(), t_class)) 
    })
    .collect();

    let mut classes_set = MapTokenClassTokenClassId::new();
    
    let mut additionnal_classes:HashSet<TokenClass> = HashSet::default();
    for token in additionnal_tokens {
        let t_class = get_token_class(&token.0, byte_classes);
        additionnal_classes.insert(t_class.clone());
        
        results.push((vec![token.1], t_class))
    }


    results.sort_by_key(|a| a.1.len());
    
    for (token_ids, t_class) in results {
        
        let class_id = classes_set.insert(&t_class);

        for &id in &token_ids {
            transitions_table.get_token_ids_by_class().add_token_id(class_id, id);
            // Avoid to override the Tokens class of muted token.
            if additionnal_classes.contains(&t_class){
                continue;
            
            }
            transitions_table.get_token_classes()[id] = class_id;
            
            
        }
  
        token_classes_graph.add_class(t_class.clone(), class_id);
    
    }
    

    let eos_u8 = eos_token_id as u8;
    let eos_token_class = byte_classes.get(eos_u8);
    let eos_token_class_id = classes_set.insert(&TokenClass::from_bytes(vec![eos_token_class; 1]));
    transitions_table.set_eos_token_class_id(eos_token_class_id);
    transitions_table.get_token_classes()[eos_token_id] = eos_token_class_id;
    transitions_table.get_token_ids_by_class().add_token_id(eos_token_class_id, eos_token_id);
    eos_token_class_id


}



fn get_token_class(token: &Vec<u8>, byte_classes: &ByteClasses) -> TokenClass {
    TokenClass::from_token_with_classes(token, byte_classes)
}

fn update_vocabulary(
    vocab: &Vocabulary, 
    decompositions: &HashMap<String, (Vec<(Vec<u8>, Vec<u32>)>, Vec<usize>)>,
    additionnal_tokens: &mut Vec<(Vec<u8>, TokenId)>) -> 
    (Vec<(String, String, Vec<usize>)>, HashSet<TokenId>) {

          // Calculer le nombre total de tokens pour déterminer la taille du padding
          let token_count = decompositions.values()
          .map(|(tokens, _)| tokens.iter()
              .map(|(_, ids)| ids.len())
              .sum::<usize>())
          .sum::<usize>();
      
      // Déterminer la longueur nécessaire pour le padding
      // log10(n) + 1 donne le nombre de chiffres nécessaires
      let padding_length = (token_count as f64).log10().ceil() as usize;
      
      let mut results: Vec<(String, String, Vec<usize>)> = Vec::default();
      let mut muted_list: HashSet<TokenId> = HashSet::default();
      
      let mut i: usize = 1;
      for literal_row in decompositions {
          let mut new_literal_value = String::new();
          new_literal_value += "(";
          let (tokens, pos) = &literal_row.1;
              
          for token in tokens {
              for id in token.1.as_slice() {
                  // Formater i avec un padding dynamique basé sur le nombre total de tokens
                  let i_padded = format!("{:0width$}", i, width = padding_length);
                  
                  let mut new_token = vec![MUTE_BYTE];
                  new_token.extend_from_slice(i_padded.as_bytes());
                  
                  new_literal_value += &String::from_utf8_lossy(&new_token);
                  
                  additionnal_tokens.push((new_token.clone(), *id));
                  muted_list.insert(*id);
                  
                  i += 1;
              }
          }
          
          new_literal_value += ")";
          
          results.push((literal_row.0.clone(), new_literal_value, pos.clone()));
      }

      return (results, muted_list);



        // let mut results: Vec<(String, String, Vec<usize>)> = Vec::default();
        // let mut muted_list:HashSet<TokenId> = HashSet::default();
        
        // let mut i: usize = 1;
        // for literal_row in decompositions{
            
        //     let mut new_literal_value = String::new();
        //     new_literal_value += "(";
        //     let (tokens, pos) = literal_row.1 ;
                
        //         for token in tokens {
        //             for id in token.1.as_slice(){
        //                 let i_str = i.to_string();

        //                 let mut new_token = vec![MUTE_BYTE];
        //                 new_token.extend_from_slice(i_str.as_bytes());
                       
        //                 new_literal_value += &String::from_utf8_lossy(&new_token);
                        
                
        //                 additionnal_tokens.push((new_token,*id) );
        //                 muted_list.insert(*id);
                      
        //                 i+=1;
                        
        //             }
        //         }
        //     new_literal_value += ")";
                
               
        //     results.push((literal_row.0.clone(), new_literal_value, pos.clone()));
        
        // }

        // return (results, muted_list);
}

pub fn mute_literals(regex: &str, vocabulary: &Vocabulary, additionnal_tokens: &mut Vec<(Vec<u8>, TokenId)>) -> (String, HashSet<TokenId>) {
    
    let literals_raw = extract_literals(regex);
    
    if literals_raw.len() == 0 {return (regex.to_string(), HashSet::default());}

    let tokens: &HashMap<Token, Vec<TokenId>> =  vocabulary.tokens();

    let decompositions = decompose_all_literals_optimized(&literals_raw, &tokens);
    
    let (literals_updated, muted_list )= update_vocabulary(vocabulary, &decompositions, additionnal_tokens);
    
    return (replace_literals(regex, &literals_updated), muted_list);
    
}



struct Trie {
    children: HashMap<u8, Trie>,
    token_ids: Option<Vec<u32>>,
}

impl Trie {
    fn new() -> Self {
        Trie {
            children: HashMap::default(),
            token_ids: None,
        }
    }

    fn insert(&mut self, token: &[u8], ids: &[u32]) {
        let mut node = self;
        for &byte in token {
            node = node.children.entry(byte).or_insert_with(Trie::new);
        }
        node.token_ids = Some(ids.to_vec());
    }

    fn find_tokens_at_position<'a>(&'a self, text: &[u8], pos: usize) -> Vec<(usize, &'a Vec<u32>)> {
        let mut result = Vec::new();
        let mut node = self;
        let mut offset = 0;

        while pos + offset < text.len() {
            let byte = text[pos + offset];
            if let Some(next_node) = node.children.get(&byte) {
                offset += 1;
                node = next_node;
                
                if let Some(ids) = &node.token_ids {
                    result.push((offset, ids));
                }
            } else {
                break;
            }
        }
        
        result
    }
}

/// Construit un Trie à partir d'une HashMap de tokens
fn build_trie(tokens: &HashMap<Vec<u8>, Vec<u32>>) -> Trie {
    let mut trie = Trie::new();
    for (token, ids) in tokens {
        trie.insert(token, ids);
    }
    trie
}

/// Trouve la décomposition optimale de tous les littéraux en une seule passe
fn decompose_all_literals_optimized(
    literals: &HashMap<String, Vec<usize>>,
    tokens: &HashMap<Vec<u8>, Vec<u32>>
) -> HashMap<String, (Vec<(Vec<u8>, Vec<u32>)>, Vec<usize>)> {
    let trie = build_trie(tokens);
    let mut result = HashMap::default();
    
    for (literal, positions) in literals {
        let literal_bytes = literal.as_bytes();
        let n = literal_bytes.len();
        
        let mut dp: Vec<Option<(usize, usize, usize)>> = vec![None; n + 1];
        dp[0] = Some((0, 0, 0));  // Base case
        
        for i in 0..n {
            if dp[i].is_none() {
                continue;
            }
            
            // Trouver tous les tokens qui commencent à la position i
            let tokens_at_pos = trie.find_tokens_at_position(literal_bytes, i);
            
            for (token_len, token_ids) in tokens_at_pos {
                let next_pos = i + token_len;
                
                // Si aucune décomposition n'existe pour next_pos ou
                // si la nouvelle décomposition utilise moins de tokens
                if dp[next_pos].is_none() || 
                   dp[next_pos].unwrap().0 > dp[i].unwrap().0 + 1 {
                    dp[next_pos] = Some((dp[i].unwrap().0 + 1, i, token_len));
                }
            }
        }
        
 
        if let Some(_) = dp[n] {
            // Reconstruire la séquence de tokens
            let mut token_sequence = Vec::new();
            let mut pos = n;
            
            while pos > 0 {
                let (_, prev_pos, token_len) = dp[pos].unwrap();
                let token_bytes = literal_bytes[prev_pos..prev_pos + token_len].to_vec();
                
                // Trouver les IDs correspondant à ce token
                if let Some(token_ids) = tokens.get(&token_bytes) {
                    token_sequence.push((token_bytes, token_ids.clone()));
                }
                
                pos = prev_pos;
            }
            
            token_sequence.reverse();
            result.insert(literal.clone(), (token_sequence, positions.clone()));
        } 
    }
    
    result
}


