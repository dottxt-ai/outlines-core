mod helpers;
mod parsing;
mod types;

pub use types::*;

use anyhow::{anyhow, Result};
use serde_json::Value;

#[derive(Debug, Copy, Clone)]
enum SchemaKeyword {
    Properties,
    AllOf,
    AnyOf,
    OneOf,
    PrefixItems,
    Enum,
    Const,
    Ref,
    Type,
    EmptyObject,
}

pub fn build_regex_from_schema(json: &str, whitespace_pattern: Option<&str>) -> Result<String> {
    let json_value: Value = serde_json::from_str(json)?;
    to_regex(&json_value, whitespace_pattern, &json_value)
}

pub fn to_regex(
    json: &Value,
    whitespace_pattern: Option<&str>,
    full_schema: &Value,
) -> Result<String> {
    let whitespace_pattern = whitespace_pattern.unwrap_or(types::WHITESPACE);

    match json {
        Value::Object(obj) => {
            let keyword = if obj.is_empty() {
                SchemaKeyword::EmptyObject
            } else {
                [
                    ("properties", SchemaKeyword::Properties),
                    ("allOf", SchemaKeyword::AllOf),
                    ("anyOf", SchemaKeyword::AnyOf),
                    ("oneOf", SchemaKeyword::OneOf),
                    ("prefixItems", SchemaKeyword::PrefixItems),
                    ("enum", SchemaKeyword::Enum),
                    ("const", SchemaKeyword::Const),
                    ("$ref", SchemaKeyword::Ref),
                    ("type", SchemaKeyword::Type),
                ]
                .iter()
                .find_map(|&(key, schema_keyword)| {
                    if obj.contains_key(key) {
                        Some(schema_keyword)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| anyhow!("Unsupported JSON Schema structure {} \nMake sure it is valid to the JSON Schema specification and check if it's supported by Outlines.\nIf it should be supported, please open an issue.", json))?
            };

            match keyword {
                SchemaKeyword::Properties => {
                    parsing::parse_properties(obj, whitespace_pattern, full_schema)
                }
                SchemaKeyword::AllOf => parsing::parse_all_of(obj, whitespace_pattern, full_schema),
                SchemaKeyword::AnyOf => parsing::parse_any_of(obj, whitespace_pattern, full_schema),
                SchemaKeyword::OneOf => parsing::parse_one_of(obj, whitespace_pattern, full_schema),
                SchemaKeyword::PrefixItems => {
                    parsing::parse_prefix_items(obj, whitespace_pattern, full_schema)
                }
                SchemaKeyword::Enum => parsing::parse_enum(obj, whitespace_pattern),
                SchemaKeyword::Const => parsing::parse_const(obj, whitespace_pattern),
                SchemaKeyword::Ref => parsing::parse_ref(obj, whitespace_pattern, full_schema),
                SchemaKeyword::Type => parsing::parse_type(obj, whitespace_pattern, full_schema),
                SchemaKeyword::EmptyObject => {
                    parsing::parse_empty_object(whitespace_pattern, full_schema)
                }
            }
        }
        _ => Err(anyhow!("Invalid JSON Schema: expected an object")),
    }
}
