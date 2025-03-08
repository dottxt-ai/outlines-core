#[cfg(any(feature = "run_benchmarks", debug_assertions))]
#[cfg(test)]
mod benchmark {
    use std::io::Write;
    use std::time::{Duration, Instant};
    use rand::Rng;
    use std::collections::HashSet;
    use crate::index::Index;
    use crate::v2_index::V2Index;
    use crate::vocabulary::Vocabulary;
    use crate::json_schema;


    #[test]
    fn bench_indexes_constructors() {
        let model_name = "unsloth/Llama-3.1-8B-Instruct";
        let regexes = get_bench_regexes();
        let vocab = Vocabulary::from_pretrained(model_name, None).unwrap();
        
        println!("> Benchmark constructors :  Index vs V2Index ({}) :", model_name);
        println!(
            "{:<45} | {:<20} | {:<20} | {:<15}",
            "Regex", "new()", "new_optimized()", "ratio"
        );
        println!(
            "{:<45} | {:<20} | {:<20} | {:<15} ",
            "-".repeat(45),
            "-".repeat(20),
            "-".repeat(20),
            "-".repeat(15),
           
        );
    
        for (name, regex) in &regexes{
            
            let schema: String;
            let regex_str = if name.contains("schema") {
                schema = json_schema::regex_from_str(regex, None).unwrap();
                schema.as_str()
            } else {
                regex
            };
    
            let start_new = Instant::now();
            let _index_new = Index::new(regex_str, &vocab).expect("Failed to create Index with new");
            let duration_new = start_new.elapsed();
    
            let start_optimized = Instant::now();
            let _index_optimized = V2Index::new(regex_str, &vocab).expect("Failed to create Index with new_optimized");
            let duration_optimized = start_optimized.elapsed();
    
            let time_new_ms = duration_new.as_secs_f64() * 1000.0;
            let time_optimized_ms = duration_optimized.as_secs_f64() * 1000.0;
            let ratio = if time_optimized_ms > 0.0 {
                time_new_ms / time_optimized_ms
            } else {
                f64::INFINITY
            };
    
            println!(
                "{:<45} | {:<20?} | {:<20?} | {:<15.2}x",
                name,
                duration_new,
                duration_optimized,
                ratio
            );
            let _ = std::io::stdout().flush();
        }
    
    }

    #[test]
    fn bench_indexes_memory(){
      let model_name = "unsloth/Llama-3.1-8B-Instruct";
        let regexes = get_bench_regexes();
        let vocab = Vocabulary::from_pretrained(model_name, None).unwrap();
        
        println!("> Benchmark constructors :  Index vs V2Index ({}) :", model_name);
        println!(
            "{:<45} | {:<20} | {:<20} | {:<15}",
            "Regex", "Index (MB)", "V2Index (MB)", "ratio"
        );
        println!(
            "{:<45} | {:<20} | {:<20} | {:<15} ",
            "-".repeat(45),
            "-".repeat(20),
            "-".repeat(20),
            "-".repeat(15),
           
        );
    
        for (name, regex) in &regexes{
            
            let schema: String;
            let regex_str = if name.contains("schema") {
                schema = json_schema::regex_from_str(regex, None).unwrap();
                schema.as_str()
            } else {
                regex
            };
    
           
            let _index_new = Index::new(regex_str, &vocab).expect("Failed to create Index with new");        
            let _index_optimized = V2Index::new(regex_str, &vocab).expect("Failed to create Index with new_optimized");
            
            let v2_index_size = _index_optimized.size();
            let index_size = _index_new.size();

            let savings_percent = if v2_index_size > index_size {
              -((v2_index_size as f64 - index_size as f64) / index_size as f64 * 100.0)
            } else {
                (index_size as f64 - v2_index_size as f64) / index_size as f64 * 100.0
            };

            println!(
                "{:<45} | {:<20?} | {:<20?} | {:<15.2}x",
                name,
                index_size as f64 / (1024.0 * 1024.0),
                v2_index_size as f64 / (1024.0 * 1024.0),
                savings_percent
            );
            let _ = std::io::stdout().flush();
        }
    
    }
    




    // Checking if the V2_Index has the same possible path as the Index has. For a given regex.
    #[test]
    fn test_index_vs_v2_index_compliance() {
        // let regex=  (
        //     "email",
        //     r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
        // );
        let sch =r###"{
      "$schema": "http://json-schema.org/draft-04/schema#",
      "title": "Schema for a recording",
      "type": "object",
      "definitions": {
        "artist": {
          "type": "object",
          "properties": {
            "id": {"type": "number"},
            "name": {"type": "string"},
            "functions": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["id", "name", "functions"]
        }
      },
      "properties": {
        "id": {"type": "number"},
        "work": {
          "type": "object",
          "properties": {
            "id": {"type": "number"},
            "name": {"type": "string"},
            "composer": {"$ref": "#/definitions/artist"}
          }
        },
        "recording_artists": {
          "type": "array",
          "items": {"$ref": "#/definitions/artist"}
        }
      },
      "required": ["id", "work", "recording_artists"]
    }"###;
         let regex = &json_schema::regex_from_str(sch, None).unwrap(); 
        
        let vocab = Vocabulary::from_pretrained("gpt2", None).unwrap();
        println!("Vocabulary loaded with size: {}", vocab.tokens().len());
    
        // Création des indices
        let standard_index = Index::new(regex, &vocab).unwrap();
        let v2_index = V2Index::new(regex, &vocab).unwrap();
        println!("Standard final states: {:?}", standard_index.final_states());
        println!(
            "v2 final states: {:?}",
            v2_index.final_states()
        );
    
        // Initialisation des guides
        let standard_guide=  standard_index;
        let v2_guide = v2_index;
    
        let mut iterations = 0;
        let mut standard_total_time = Duration::new(0, 0);
        let mut v2_total_time = Duration::new(0, 0);
        let mut current_token_id = None;
    
        // Fonction pour vérifier si un guide est terminé
        let is_finished = |guide: &Index| guide.is_final_state(&guide.initial_state());
    
        // Première itération
        if !is_finished(&standard_guide) {
            iterations += 1;
    
            // Temps pour Standard
            let start_standard = Instant::now();
            let standard_tokens = match standard_guide.allowed_tokens(&standard_guide.initial_state()) {
                Some(tokens) => tokens,
                None => panic!("No tokens available for standard"),
            };
    
            let standard_time = start_standard.elapsed();
            standard_total_time += standard_time;
    
            // Temps pour v2
            let start_v2 = Instant::now();
            let v2_tokens =
                match v2_guide.allowed_tokens(&v2_guide.initial_state()) {
                    Some(tokens) => tokens,
                    None => panic!("No tokens available for v2"),
                };
    
            let v2_time = start_v2.elapsed();
            v2_total_time += v2_time;
    
    
            // Vérification que le token est dans les deux guides
            let mask_tokens: Vec<u32> = v2_tokens
                .iter()
                .enumerate()
                .flat_map(|(word_idx, &word)| {
                    (0..64).filter_map(move |bit_idx| {
                        if word & (1u64 << bit_idx) != 0 {
                            Some((word_idx * 64 + bit_idx) as u32)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            println!("mask_tokens: {:?}", mask_tokens);
            println!("index_tokens: {:?}", standard_tokens);
            let random_idx = rand::rng().random_range(0..mask_tokens.len());
            current_token_id = Some(mask_tokens[random_idx]);

            assert!(
                standard_tokens.contains(&current_token_id.unwrap()),
                "Token {} from V2Index not found in Index, Iteration {}",
                current_token_id.unwrap(),
                iterations
            );
        }
    
        // Boucle principale
        let mut standard_state = standard_guide.initial_state();
        let mut v2_state = v2_guide.initial_state();
        while !v2_guide.is_final_state(&v2_state) {
            iterations += 1;
            // println!("> Iterations : {}", iterations );
            let token_id = current_token_id.unwrap();
    
            // Avancer Standard
            let start_standard = Instant::now();
            let new_standard_state = match standard_guide.next_state(&standard_state, &token_id) {
                Some(state) => state,
                None => {
                    println!("Iteration : {}\n state : {} - token_id : {}", iterations,  standard_state, token_id);
                    panic!("No next state found for standard guide");},
            };
    
            standard_state = new_standard_state; // Mise à jour de l'état
            let standard_tokens = match standard_guide.allowed_tokens(&standard_state) {
                Some(tokens) => tokens,
                None => panic!("No tokens available for standard"),
            };
            let standard_time = start_standard.elapsed();
            standard_total_time += standard_time;
    
            // Avancer v2
            let start_v2 = Instant::now();
            let new_v2_state = match v2_guide.next_state(&v2_state, &token_id) {
                Some(state) => state,
                None =>{ println!("Token ID: {}", token_id); panic!("No next state found for v2 guide")},
            };
    
            v2_state = new_v2_state; // Mise à jour de l'état
            let v2_tokens = match v2_guide.allowed_tokens(&v2_state) {
                Some(tokens) => tokens,
                None => panic!("No tokens available for v2"),
            };
            let v2_time = start_v2.elapsed();
            v2_total_time += v2_time;
    
            // Maintenant les vérifications sur les états
            assert!(
                standard_guide.is_final_state(&standard_state)
                    == v2_guide.is_final_state(&v2_state),
                "Guides out of sync, Iteration {}",
                iterations
            );
    
            if !v2_guide.is_final_state(&v2_state) {
                 // Vérification que le token est dans les deux guides
                let mask_tokens: Vec<u32> = v2_tokens
                .iter()
                .enumerate()
                .flat_map(|(word_idx, &word)| {
                    (0..64).filter_map(move |bit_idx| {
                        if word & (1u64 << bit_idx) != 0 {
                            Some((word_idx * 64 + bit_idx) as u32)
                        } else {
                            None
                        }
                    })
                })
                .collect();
                // println!("mask_tokens: {:?}", mask_tokens);
                // println!("index_tokens: {:?}", standard_tokens);
                let random_idx = rand::rng().random_range(0..mask_tokens.len());
                current_token_id = Some(mask_tokens[random_idx]);
               
                // println!("Token choose : {}", current_token_id.unwrap());
                
                assert!(
                    standard_tokens.contains(&current_token_id.unwrap()),
                    "Token {} from V2Index not found in Index, Iteration {}\n mask_tokens : {:?} \n index tokens : {:?}",
                    current_token_id.unwrap(),
                    iterations,
                    mask_tokens,
                    standard_tokens
                    
                );
            }
        }
    
        let standard_total_time_us = standard_total_time.as_micros() as f64;
        let v2_total_time_us = v2_total_time.as_micros() as f64;
    
        println!("Total iterations (Number of tokens): {}", iterations);
        println!(
            "Guide with Standard Index: {:.2} µs ({:.2} µs per iteration)",
            standard_total_time_us,
            standard_total_time_us / iterations as f64
        );
        println!(
            "Guide with v2 Index: {:.2} µs ({:.2} µs per iteration)",
            v2_total_time_us,
            v2_total_time_us / iterations as f64
        );
        println!(
            "Speedup ratio: {:.2}x",
            standard_total_time_us / v2_total_time_us
        );
    }
    

    fn get_bench_regexes() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "email",
                r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
            ),
            ("simple_phone", r"\+?[1-9][0-9]{7,14}"),
            (
                "complex_phone",
                r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            ),
            ("permissive_any", r".{255}$"),
            ("permissive_words", r"[a-zA-Z]{100}"),
            (
                "schema_simple",
                r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#,
            ),
            (
                "schema_simple_phone",
                r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}"#,
            ),
            ("https", r#"(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?"#),
            (
                "schema_complexe",
                r###"{
      "$schema": "http://json-schema.org/draft-04/schema#",
      "title": "Schema for a recording",
      "type": "object",
      "definitions": {
        "artist": {
          "type": "object",
          "properties": {
            "id": {"type": "number"},
            "name": {"type": "string"},
            "functions": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["id", "name", "functions"]
        }
      },
      "properties": {
        "id": {"type": "number"},
        "work": {
          "type": "object",
          "properties": {
            "id": {"type": "number"},
            "name": {"type": "string"},
            "composer": {"$ref": "#/definitions/artist"}
          }
        },
        "recording_artists": {
          "type": "array",
          "items": {"$ref": "#/definitions/artist"}
        }
      },
      "required": ["id", "work", "recording_artists"]
    }"###,
            ),
            ("schema_curriculum" ,
            r###"{
                "$schema": "http://json-schema.org/draft-04/schema#",
                "title": "Schema for a Curriculum Vitae",
                "type": "object",
                "definitions": {
                    "experienceEntry": {
                    "type": "object",
                    "properties": {
                        "date": {
                        "type": "string",
                        "format": "date"
                        },
                        "position": {
                        "type": "string"
                        }
                    },
                    "required": ["date", "position"]
                    }
                },
                "properties": {
                    "name": {
                    "type": "string"
                    },
                    "surname": {
                    "type": "string"
                    },
                    "email": {
                    "type": "string",
                    "pattern": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
                    },
                    "phone": {
                    "type": "string",
                    "pattern": "\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}"
                    },
                    "website": {
                    "type": "string",
                    "pattern": "(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?"
                    },
                    "resume": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/experienceEntry"
                    }
                    }
                },
                "required": ["name", "surname", "email", "phone", "resume"]
              }"###
        )
        ]
}





}
