//! Compare the memory used by two index structures with different regexes.

use std::time::Instant;

use rustc_hash::FxHashMap as HashMap;

use crate::index::{CompressedIndex, Index, IndexBehavior, IndexVariant};
use crate::prelude::*;

#[allow(dead_code)]
pub fn compare_memory_usage(show_details: bool) {
    println!("Comparing memory usage between Index and CompressedIndex...\n");

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

    println!(
        "{:<45} | {:<12} | {:<12} | {:<12} | {:<12}",
        "Regex", "Index (MB)", "Compressed (MB)", "Diff (MB)", "Savings %"
    );
    println!(
        "{:<45} | {:<12} | {:<12} | {:<12} | {:<12}",
        "-".repeat(30),
        "-".repeat(12),
        "-".repeat(12),
        "-".repeat(12),
        "-".repeat(12)
    );

    let vocabulary =
        Vocabulary::from_pretrained("gpt2", None).expect("Failed to load GPT-2 vocabulary");

    for (name, regex) in &regexes {
        println!("Testing regex: {}", name);

        let start = Instant::now();

        let schema: String;
        let regex_str = if name.contains("schema") {
            schema = json_schema::regex_from_str(regex, None).unwrap();
            schema.as_str()
        } else {
            regex
        };

        let index = Index::new(regex_str, &vocabulary)
            .unwrap_or_else(|_| panic!("Failed to create Index for {}", name));
        let index_creation_time = start.elapsed();

        let index_size = if show_details {
            index.memory_usage()
        } else {
            calculate_size_without_output(&index)
        };

        let start = Instant::now();
        let compressed = IndexVariant::create_compressed(regex_str, &vocabulary)
            .unwrap_or_else(|_| panic!("Failed to create CompressedIndex for {}", name));
        let compressed_creation_time = start.elapsed();

        let compressed_size = match &compressed {
            IndexVariant::Compressed(comp_idx) => {
                if show_details {
                    comp_idx.memory_usage()
                } else {
                    calculate_compressed_size_without_output(comp_idx)
                }
            }
            _ => panic!("Expected CompressedIndex variant"),
        };

        let size_diff = if compressed_size > index_size {
            compressed_size - index_size
        } else {
            index_size - compressed_size
        };

        let savings_percent = if compressed_size > index_size {
            -((compressed_size as f64 - index_size as f64) / index_size as f64 * 100.0)
        } else {
            (index_size as f64 - compressed_size as f64) / index_size as f64 * 100.0
        };

        println!(
            "{:<45} | {:<12.2} | {:<12.2} | {:<12.2} | {:<12.2}%",
            name,
            index_size as f64 / (1024.0 * 1024.0),
            compressed_size as f64 / (1024.0 * 1024.0),
            size_diff as f64 / (1024.0 * 1024.0),
            savings_percent
        );

        if show_details {
            println!("Creation times:");
            println!("  Index: {:?}", index_creation_time);
            println!("  CompressedIndex: {:?}", compressed_creation_time);
            println!();
        }
    }
}

#[allow(unused_mut)]
fn calculate_size_without_output(index: &Index) -> usize {
    use std::mem::{size_of, size_of_val};

    let struct_size = size_of::<Index>();

    let initial_state_size = size_of::<StateId>();
    let eos_token_id_size = size_of::<TokenId>();
    let vocab_size_size = size_of::<usize>();

    let final_states_container = size_of_val(index.final_states());
    let final_states_content = index.final_states().len() * (size_of::<StateId>() + 1);
    let final_states_size = final_states_container + final_states_content;

    let transitions_container = size_of_val(index.transitions());

    let mut inner_maps_size = 0;
    let mut transitions_content_size = 0;
    #[allow(clippy::for_kv_map)]
    for (_, inner_map) in index.transitions() {
        inner_maps_size += size_of_val::<HashMap<u32, u32>>(inner_map);
        transitions_content_size +=
            inner_map.len() * (size_of::<TokenId>() + size_of::<StateId>() + 1);
    }

    let transitions_total_size = transitions_container + inner_maps_size + transitions_content_size;

    struct_size
        + initial_state_size
        + eos_token_id_size
        + vocab_size_size
        + final_states_size
        + transitions_total_size
}

#[allow(unused_mut)]
fn calculate_compressed_size_without_output(index: &CompressedIndex) -> usize {
    use std::mem::{size_of, size_of_val};

    let struct_size = size_of::<CompressedIndex>();

    let initial_state_size = size_of::<StateId>();
    let eos_token_id_size = size_of::<TokenId>();
    let vocab_size_size = size_of::<usize>();
    #[allow(clippy::size_of_ref)]
    let final_states_size =
        size_of_val(&index.final_states()) + (index.final_states().len() * size_of::<StateId>());

    let mapping_size = size_of_val(&index.state_to_index);
    let mapping_content_size =
        index.state_to_index.len() * (size_of::<StateId>() + size_of::<usize>());
    let total_mapping_size = mapping_size + mapping_content_size;

    let mut token_masks_size = 0;
    for mask in &index.token_masks {
        token_masks_size += size_of::<Vec<u64>>() + (mask.len() * size_of::<u64>());
    }

    let next_states_container_size = size_of::<Vec<StateId>>();
    let next_states_content_size = index.next_states.len() * size_of::<StateId>();
    let total_next_states_size = next_states_container_size + next_states_content_size;

    let state_offsets_size =
        size_of::<Vec<usize>>() + (index.state_offsets.len() * size_of::<usize>());

    struct_size
        + final_states_size
        + initial_state_size
        + eos_token_id_size
        + vocab_size_size
        + total_mapping_size
        + token_masks_size
        + total_next_states_size
        + state_offsets_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_comparison() {
        compare_memory_usage(true);
    }

    #[test]
    fn test_memory_comparison_simple() {
        compare_memory_usage(false);
    }
}
#[allow(clippy::items_after_test_module)]
#[allow(dead_code)]
pub fn run_benchmarks(show_details: bool) {
    compare_memory_usage(show_details);
}
