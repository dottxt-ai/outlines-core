use std::collections::HashSet;
use std::time::{Duration, Instant};

use rand::Rng;

use crate::index::{CompressedIndex, Index, IndexBehavior, IndexVariant};
use crate::json_schema;
use crate::prelude::*;
#[allow(dead_code)]
fn run_benchmark(regex: &str) {
    // Initialisation du vocabulaire
    let vocab = Vocabulary::from_pretrained("gpt2", None).unwrap();
    println!("Vocabulary loaded with size: {}", vocab.tokens().len());

    // Création des indices
    let standard_index = Index::new(regex, &vocab).unwrap();
    let compressed_index = CompressedIndex::new(&standard_index, vocab.eos_token_id());
    println!("Standard final states: {:?}", standard_index.final_states());
    println!(
        "Compressed final states: {:?}",
        compressed_index.final_states()
    );

    // Initialisation des guides
    let standard_guide: IndexVariant = IndexVariant::Standard(standard_index);
    let compressed_guide = IndexVariant::Compressed(compressed_index);

    let mut iterations = 0;
    let mut standard_total_time = Duration::new(0, 0);
    let mut compressed_total_time = Duration::new(0, 0);
    let mut current_token_id = None;

    // Fonction pour vérifier si un guide est terminé
    let is_finished = |guide: &IndexVariant| guide.is_final_state(&guide.initial_state());

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

        // Temps pour Compressed
        let start_compressed = Instant::now();
        let compressed_tokens =
            match compressed_guide.allowed_tokens_mask(&compressed_guide.initial_state()) {
                Some(tokens) => tokens,
                None => panic!("No tokens available for compressed"),
            };

        let compressed_time = start_compressed.elapsed();
        compressed_total_time += compressed_time;

        // Choix aléatoire d’un token
        let random_idx = rand::rng().random_range(0..standard_tokens.len());
        current_token_id = Some(standard_tokens[random_idx]);

        // Vérification que le token est dans les deux guides
        let mask_tokens: HashSet<u64> = compressed_tokens
            .iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| {
                (0..64).filter_map(move |bit_idx| {
                    if word & (1u64 << bit_idx) != 0 {
                        Some((word_idx * 64 + bit_idx) as u64)
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            mask_tokens.contains(&(current_token_id.unwrap() as u64)),
            "Token {} from Standard not found in Compressed, Iteration {}",
            current_token_id.unwrap(),
            iterations
        );
    }

    // Boucle principale
    let mut standard_state = standard_guide.initial_state();
    let mut compressed_state = compressed_guide.initial_state();
    while !standard_guide.is_final_state(&standard_state) {
        iterations += 1;

        let token_id = current_token_id.unwrap();

        // Avancer Standard
        let start_standard = Instant::now();
        let new_standard_state = match standard_guide.next_state(&standard_state, &token_id) {
            Some(state) => state,
            None => panic!("No next state found for standard guide"),
        };

        standard_state = new_standard_state; // Mise à jour de l'état
        let standard_tokens = match standard_guide.allowed_tokens(&standard_state) {
            Some(tokens) => tokens,
            None => panic!("No tokens available for standard"),
        };
        let standard_time = start_standard.elapsed();
        standard_total_time += standard_time;

        // Avancer Compressed
        let start_compressed = Instant::now();
        let new_compressed_state = match compressed_guide.next_state(&compressed_state, &token_id) {
            Some(state) => state,
            None => panic!("No next state found for compressed guide"),
        };

        compressed_state = new_compressed_state; // Mise à jour de l'état
        let compressed_tokens = match compressed_guide.allowed_tokens_mask(&compressed_state) {
            Some(tokens) => tokens,
            None => panic!("No tokens available for compressed"),
        };
        let compressed_time = start_compressed.elapsed();
        compressed_total_time += compressed_time;

        // Maintenant les vérifications sur les états
        assert!(
            standard_guide.is_final_state(&standard_state)
                == compressed_guide.is_final_state(&compressed_state),
            "Guides out of sync, Iteration {}",
            iterations
        );

        if !standard_guide.is_final_state(&standard_state) {
            let random_idx = rand::rng().random_range(0..standard_tokens.len());
            current_token_id = Some(standard_tokens[random_idx]);

            let mask_tokens: HashSet<u64> = compressed_tokens
                .iter()
                .enumerate()
                .flat_map(|(word_idx, &word)| {
                    (0..64).filter_map(move |bit_idx| {
                        if word & (1u64 << bit_idx) != 0 {
                            Some((word_idx * 64 + bit_idx) as u64)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            assert!(
                mask_tokens.contains(&(current_token_id.unwrap() as u64)),
                "Token {} from Standard not found in Compressed, Iteration {}",
                current_token_id.unwrap(),
                iterations
            );
        }
    }

    let standard_total_time_us = standard_total_time.as_micros() as f64;
    let compressed_total_time_us = compressed_total_time.as_micros() as f64;

    println!("Total iterations (Number of tokens): {}", iterations);
    println!(
        "Guide with Standard Index: {:.2} µs ({:.2} µs per iteration)",
        standard_total_time_us,
        standard_total_time_us / iterations as f64
    );
    println!(
        "Guide with Compressed Index: {:.2} µs ({:.2} µs per iteration)",
        compressed_total_time_us,
        compressed_total_time_us / iterations as f64
    );
    println!(
        "Speedup ratio: {:.2}x",
        standard_total_time_us / compressed_total_time_us
    );
}

#[allow(dead_code)]
pub fn main() {
    let regexes = vec![
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
    ];
    let schemas = [
        (
            "schema_simple",
            r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#,
        ),
        (
            "schema_simple_phone",
            r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}"#,
        ),
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
    ];

    println!("Starting benchmark in Rust...");
    for (name, regex) in regexes {
        println!("> Benchmark IndexVariant for regex '{}'", name);
        run_benchmark(regex);
    }

    for (name, schema) in schemas {
        println!("> Benchmark IndexVariant for schema '{}'", name);
        let regex = json_schema::regex_from_str(schema, None).unwrap();
        run_benchmark(&regex);
    }

    // println!("> Benchmark IndexVariant for regex simple");
    // run_benchmark(r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}");
}

#[cfg(test)]
mod tests {
    use crate::index::test_bench_index_variant::main;
    #[test]
    fn test_benchmarck() {
        main();
    }
}
