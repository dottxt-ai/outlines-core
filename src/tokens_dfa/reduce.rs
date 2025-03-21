
use core::time;
use std::sync::Mutex;
use std::time::Instant;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use regex_automata::util::alphabet::ByteClasses;
use rayon::prelude::*;

use super::token_classes::{from_token_to_token_class, TokenClassId, TokenClass};
use super::token_classes_graph::PrefixGraphes;
use super::transitions_table::EquivalentGrid;


use crate::prelude::*;
use crate::vocabulary::Vocabulary;

pub use super::regex::{extract_literals, replace_literals};

const MUTE_BYTE: u8 = 0x1C;  // En hexadécimal


pub fn minimizing_alphabet(
    tokens: &HashMap<Vec<u8>, Vec<TokenId>>, // Token comme Vec<u8>
    additionnal_tokens: &Vec<(Vec<u8>, TokenId)>,
    byte_classes: &ByteClasses,
    dead_byte_classes: HashSet<u8>,
    equivalent_grid: &mut EquivalentGrid,
    eos_token_id: u32
) -> TokenClassId {

    let start_filter = Instant::now();

    let filtered_tokens: Vec<_> = tokens.par_iter()
            .filter_map(|(token, token_ids)| {
                if token_ids.iter().any(|id| *id == eos_token_id || *id == 216) {
                    return None;
                }

                let token_class = from_token_to_token_class(token, byte_classes);

                if token_class.as_bytes().iter().any(|byte| dead_byte_classes.contains(byte)) 
                {
                    return None; 
                }

                Some((token_class, token_ids))
        })
        .collect();
    
    let time_filter = start_filter.elapsed();
    
    let start_tokens = Instant::now();
    for (token_class, token_ids) in filtered_tokens {
        let class_id = equivalent_grid.insert_class(token_class);
        
        for token_id in token_ids {
            equivalent_grid.bind_token_id_and_class_id(*token_id, class_id);
        }
    }

    let time_tokens = start_tokens.elapsed();
    

    for (token, token_id) in additionnal_tokens.iter(){
        
        let token_class = from_token_to_token_class(&token, &byte_classes);
        
        let class_id = equivalent_grid.insert_class(token_class.clone());
        
        equivalent_grid.mute_bind_token_id_and_class_id(*token_id, class_id);
    }

    let eos_u8 = eos_token_id as u8;
    let eos_token_class: u8 = byte_classes.get_by_unit(byte_classes.eoi()) as u8;
    let eos_token_class_id = equivalent_grid.insert_class(TokenClass::from_bytes(vec![eos_token_class;1]));
    equivalent_grid.bind_token_id_and_class_id(eos_token_id, eos_token_class_id);
    equivalent_grid.sort_classes();
    eos_token_class_id

}

pub fn build_prefix_based_graphes(
    equivalent_grid: &EquivalentGrid,
    graphes: &mut PrefixGraphes
){
   let sorted_classes: &Vec<TokenClassId> = equivalent_grid.get_sorted_classes();
    for class_id in sorted_classes {
        let class = equivalent_grid.get_class_from_class_id(*class_id);
        graphes.add_class(class, *class_id, equivalent_grid.get_classes());
    }
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

}

pub fn mute_literals(regex: &str, vocabulary: &Vocabulary, additionnal_tokens: &mut Vec<(Vec<u8>, TokenId)>) -> (String, HashSet<TokenId>) {
    
    let literals_raw = extract_literals(regex);
    
    if literals_raw.len() == 0 {return (regex.to_string(), HashSet::default());}
    let tokens: &HashMap<Token, Vec<TokenId>> =  vocabulary.tokens();

    let decompositions = decompose_all_literals_optimized(&literals_raw, &tokens);
    let (literals_updated, muted_list )= update_vocabulary(vocabulary, &decompositions, additionnal_tokens);

    return (replace_literals(regex, &literals_updated), muted_list);
    
}



/// Trouve la décomposition optimale de tous les littéraux en une seule passe
fn decompose_all_literals_optimized(
    literals: &HashMap<String, Vec<usize>>,
    tokens: &HashMap<Vec<u8>, Vec<u32>>
) -> HashMap<String, (Vec<(Vec<u8>, Vec<u32>)>, Vec<usize>)> {

    let mut result = HashMap::default();

    for (literal, positions) in literals {
        let literal_bytes = literal.as_bytes();
        let n = literal_bytes.len();
        
        // Tableau DP pour stocker le nombre minimal de tokens, position précédente et longueur
        let mut dp: Vec<Option<(usize, usize, usize)>> = vec![None; n + 1];
        dp[0] = Some((0, 0, 0)); // Cas de base

        // Parcours de chaque position dans le littéral
        for i in 0..n {
            if dp[i].is_none() {
                continue;
            }

            // Chercher tous les tokens possibles commençant à la position i
            let max_len = n - i; // Longueur maximale restante
            for len in 1..=max_len {
                let token_bytes = &literal_bytes[i..i + len];
                if let Some(token_ids) = tokens.get(token_bytes) {
                    let next_pos = i + len;
                    // Si aucune décomposition n'existe ou si on trouve une meilleure
                    if dp[next_pos].is_none() || dp[next_pos].unwrap().0 > dp[i].unwrap().0 + 1 {
                        dp[next_pos] = Some((dp[i].unwrap().0 + 1, i, len));
                    }
                }
            }
        }

        // Si une décomposition complète est trouvée
        if let Some(_) = dp[n] {
            // Reconstruire la séquence de tokens
            let mut token_sequence = Vec::new();
            let mut pos = n;

            while pos > 0 {
                let (_, prev_pos, token_len) = dp[pos].unwrap();
                let token_bytes = literal_bytes[prev_pos..prev_pos + token_len].to_vec();
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


