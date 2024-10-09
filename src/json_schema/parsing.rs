use anyhow::{anyhow, Result};
use regex::escape;
use serde_json::json;
use serde_json::Value;

use crate::json_schema::helpers;
use crate::json_schema::to_regex;
use crate::json_schema::types;

pub fn parse_properties(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let mut regex = String::from(r"\{");

    let properties = obj
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("'properties' not found or not an object"))?;

    let required_properties = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();

    let is_required: Vec<bool> = properties
        .keys()
        .map(|item| required_properties.contains(&item.as_str()))
        .collect();

    if is_required.iter().any(|&x| x) {
        let last_required_pos = is_required
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value)
            .map(|(i, _)| i)
            .max()
            .unwrap();

        for (i, (name, value)) in properties.iter().enumerate() {
            let mut subregex = format!(
                r#"{whitespace_pattern}"{}"{}:{}"#,
                escape(name),
                whitespace_pattern,
                whitespace_pattern
            );
            subregex += &to_regex(value, Some(whitespace_pattern), full_schema)?;

            match i {
                i if i < last_required_pos => {
                    subregex = format!("{}{},", subregex, whitespace_pattern)
                }
                i if i > last_required_pos => {
                    subregex = format!("{},{}", whitespace_pattern, subregex)
                }
                _ => (),
            }

            regex += &if is_required[i] {
                subregex
            } else {
                format!("({})?", subregex)
            };
        }
    } else {
        let mut property_subregexes = Vec::new();
        for (name, value) in properties.iter() {
            let mut subregex = format!(
                r#"{whitespace_pattern}"{}"{}:{}"#,
                escape(name),
                whitespace_pattern,
                whitespace_pattern
            );

            subregex += &to_regex(value, Some(whitespace_pattern), full_schema)?;
            property_subregexes.push(subregex);
        }

        let mut possible_patterns = Vec::new();
        for i in 0..property_subregexes.len() {
            let mut pattern = String::new();
            for subregex in &property_subregexes[..i] {
                pattern += &format!("({}{},)?", subregex, whitespace_pattern);
            }
            pattern += &property_subregexes[i];
            for subregex in &property_subregexes[i + 1..] {
                pattern += &format!("({},{})?", whitespace_pattern, subregex);
            }
            possible_patterns.push(pattern);
        }

        regex += &format!("({})?", possible_patterns.join("|"));
    }

    regex += &format!("{}\\}}", whitespace_pattern);

    Ok(regex)
}

pub fn parse_all_of(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    match obj.get("allOf") {
        Some(Value::Array(all_of)) => {
            let subregexes: Result<Vec<String>> = all_of
                .iter()
                .map(|t| to_regex(t, Some(whitespace_pattern), full_schema))
                .collect();

            let subregexes = subregexes?;
            let combined_regex = subregexes.join("");

            Ok(format!(r"({})", combined_regex))
        }
        _ => Err(anyhow!("'allOf' must be an array")),
    }
}

pub fn parse_any_of(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    match obj.get("anyOf") {
        Some(Value::Array(any_of)) => {
            let subregexes: Result<Vec<String>> = any_of
                .iter()
                .map(|t| to_regex(t, Some(whitespace_pattern), full_schema))
                .collect();

            let subregexes = subregexes?;

            Ok(format!(r"({})", subregexes.join("|")))
        }
        _ => Err(anyhow!("'anyOf' must be an array")),
    }
}

pub fn parse_one_of(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    match obj.get("oneOf") {
        Some(Value::Array(one_of)) => {
            let subregexes: Result<Vec<String>> = one_of
                .iter()
                .map(|t| to_regex(t, Some(whitespace_pattern), full_schema))
                .collect();

            let subregexes = subregexes?;

            let xor_patterns: Vec<String> = subregexes
                .into_iter()
                .map(|subregex| format!(r"(?:{})", subregex))
                .collect();

            Ok(format!(r"({})", xor_patterns.join("|")))
        }
        _ => Err(anyhow!("'oneOf' must be an array")),
    }
}

pub fn parse_prefix_items(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    match obj.get("prefixItems") {
        Some(Value::Array(prefix_items)) => {
            let element_patterns: Result<Vec<String>> = prefix_items
                .iter()
                .map(|t| to_regex(t, Some(whitespace_pattern), full_schema))
                .collect();

            let element_patterns = element_patterns?;

            let comma_split_pattern = format!("{},{}", whitespace_pattern, whitespace_pattern);
            let tuple_inner = element_patterns.join(&comma_split_pattern);

            Ok(format!(
                r"\[{whitespace_pattern}{tuple_inner}{whitespace_pattern}\]"
            ))
        }
        _ => Err(anyhow!("'prefixItems' must be an array")),
    }
}

pub fn parse_enum(
    obj: &serde_json::Map<String, Value>,
    _whitespace_pattern: &str,
) -> Result<String> {
    match obj.get("enum") {
        Some(Value::Array(enum_values)) => {
            let choices: Result<Vec<String>> = enum_values
                .iter()
                .map(|choice| match choice {
                    Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                        let json_string = serde_json::to_string(choice)?;
                        Ok(regex::escape(&json_string))
                    }
                    _ => Err(anyhow!("Unsupported data type in enum: {:?}", choice)),
                })
                .collect();

            let choices = choices?;
            Ok(format!(r"({})", choices.join("|")))
        }
        _ => Err(anyhow!("'enum' must be an array")),
    }
}

pub fn parse_const(
    obj: &serde_json::Map<String, Value>,
    _whitespace_pattern: &str,
) -> Result<String> {
    match obj.get("const") {
        Some(const_value) => match const_value {
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                let json_string = serde_json::to_string(const_value)?;
                Ok(regex::escape(&json_string))
            }
            _ => Err(anyhow!("Unsupported data type in const: {:?}", const_value)),
        },
        None => Err(anyhow!("'const' key not found in object")),
    }
}

pub fn parse_ref(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let ref_path = obj["$ref"]
        .as_str()
        .ok_or_else(|| anyhow!("'$ref' must be a string"))?;

    let parts: Vec<&str> = ref_path.split('#').collect();

    match parts.as_slice() {
        [fragment] | ["", fragment] => {
            let path_parts: Vec<&str> = fragment.split('/').filter(|&s| !s.is_empty()).collect();
            let referenced_schema = resolve_local_ref(full_schema, &path_parts)?;
            to_regex(referenced_schema, Some(whitespace_pattern), full_schema)
        }
        [base, fragment] => {
            if let Some(id) = full_schema["$id"].as_str() {
                if *base == id || base.is_empty() {
                    let path_parts: Vec<&str> =
                        fragment.split('/').filter(|&s| !s.is_empty()).collect();
                    let referenced_schema = resolve_local_ref(full_schema, &path_parts)?;
                    return to_regex(referenced_schema, Some(whitespace_pattern), full_schema);
                }
            }
            Err(anyhow!(
                "External references are not supported: {}",
                ref_path
            ))
        }
        _ => Err(anyhow!("Invalid reference format: {}", ref_path)),
    }
}

fn resolve_local_ref<'a>(schema: &'a Value, path_parts: &[&str]) -> Result<&'a Value> {
    let mut current = schema;
    for &part in path_parts {
        current = current
            .get(part)
            .ok_or_else(|| anyhow!("Invalid reference path: {}", part))?;
    }
    Ok(current)
}

pub fn parse_type(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let instance_type = obj["type"]
        .as_str()
        .ok_or_else(|| anyhow!("'type' must be a string"))?;
    match instance_type {
        "string" => parse_string_type(obj),
        "number" => parse_number_type(obj),
        "integer" => parse_integer_type(obj),
        "array" => parse_array_type(obj, whitespace_pattern, full_schema),
        "object" => parse_object_type(obj, whitespace_pattern, full_schema),
        "boolean" => parse_boolean_type(),
        "null" => parse_null_type(),
        _ => Err(anyhow!("Unsupported type: {}", instance_type)),
    }
}

pub fn parse_empty_object(whitespace_pattern: &str, full_schema: &Value) -> Result<String> {
    // JSON Schema Spec: Empty object means unconstrained, any json type is legal
    let types = vec![
        json!({"type": "boolean"}),
        json!({"type": "null"}),
        json!({"type": "number"}),
        json!({"type": "integer"}),
        json!({"type": "string"}),
        json!({"type": "array"}),
        json!({"type": "object"}),
    ];

    let regexes: Result<Vec<String>> = types
        .iter()
        .map(|t| to_regex(t, Some(whitespace_pattern), full_schema))
        .collect();

    let regexes = regexes?;

    let wrapped_regexes: Vec<String> = regexes.into_iter().map(|r| format!("({})", r)).collect();

    Ok(wrapped_regexes.join("|"))
}

fn parse_boolean_type() -> Result<String> {
    let format_type = types::JsonType::Boolean;
    Ok(format_type.to_regex().to_string())
}

fn parse_null_type() -> Result<String> {
    let format_type = types::JsonType::Null;
    Ok(format_type.to_regex().to_string())
}

fn parse_string_type(obj: &serde_json::Map<String, Value>) -> Result<String> {
    if obj.contains_key("maxLength") || obj.contains_key("minLength") {
        let max_items = obj.get("maxLength");
        let min_items = obj.get("minLength");

        match (min_items, max_items) {
            (Some(min), Some(max)) if min.as_f64() > max.as_f64() => {
                return Err(anyhow::anyhow!(
                    "maxLength must be greater than or equal to minLength"
                ));
            }
            _ => {}
        }

        let formatted_max = max_items
            .and_then(Value::as_u64)
            .map_or("".to_string(), |n| format!("{}", n));
        let formatted_min = min_items
            .and_then(Value::as_u64)
            .map_or("0".to_string(), |n| format!("{}", n));

        Ok(format!(
            r#""{}{{{},{}}}""#,
            types::STRING_INNER,
            formatted_min,
            formatted_max,
        ))
    } else if let Some(pattern) = obj.get("pattern").and_then(Value::as_str) {
        if pattern.starts_with('^') && pattern.ends_with('$') {
            Ok(format!(r#"("{}")"#, &pattern[1..pattern.len() - 1]))
        } else {
            Ok(format!(r#"("{}")"#, pattern))
        }
    } else if let Some(format) = obj.get("format").and_then(Value::as_str) {
        match types::FormatType::from_str(format) {
            Some(format_type) => Ok(format_type.to_regex().to_string()),
            None => Err(anyhow::anyhow!(
                "Format {} is not supported by Outlines",
                format
            )),
        }
    } else {
        Ok(types::JsonType::String.to_regex().to_string())
    }
}

fn parse_number_type(obj: &serde_json::Map<String, Value>) -> Result<String> {
    let bounds = [
        "minDigitsInteger",
        "maxDigitsInteger",
        "minDigitsFraction",
        "maxDigitsFraction",
        "minDigitsExponent",
        "maxDigitsExponent",
    ];

    let has_bounds = bounds.iter().any(|&key| obj.contains_key(key));

    if has_bounds {
        let (min_digits_integer, max_digits_integer) = helpers::validate_quantifiers(
            obj.get("minDigitsInteger").and_then(Value::as_u64),
            obj.get("maxDigitsInteger").and_then(Value::as_u64),
            1,
        )?;

        let (min_digits_fraction, max_digits_fraction) = helpers::validate_quantifiers(
            obj.get("minDigitsFraction").and_then(Value::as_u64),
            obj.get("maxDigitsFraction").and_then(Value::as_u64),
            0,
        )?;

        let (min_digits_exponent, max_digits_exponent) = helpers::validate_quantifiers(
            obj.get("minDigitsExponent").and_then(Value::as_u64),
            obj.get("maxDigitsExponent").and_then(Value::as_u64),
            0,
        )?;

        let integers_quantifier = match (min_digits_integer, max_digits_integer) {
            (Some(min), Some(max)) => format!("{{{},{}}}", min, max),
            (Some(min), None) => format!("{{{},}}", min),
            (None, Some(max)) => format!("{{1,{}}}", max),
            (None, None) => "*".to_string(),
        };

        let fraction_quantifier = match (min_digits_fraction, max_digits_fraction) {
            (Some(min), Some(max)) => format!("{{{},{}}}", min, max),
            (Some(min), None) => format!("{{{},}}", min),
            (None, Some(max)) => format!("{{0,{}}}", max),
            (None, None) => "+".to_string(),
        };

        let exponent_quantifier = match (min_digits_exponent, max_digits_exponent) {
            (Some(min), Some(max)) => format!("{{{},{}}}", min, max),
            (Some(min), None) => format!("{{{},}}", min),
            (None, Some(max)) => format!("{{0,{}}}", max),
            (None, None) => "+".to_string(),
        };

        Ok(format!(
            r"((-)?(0|[1-9][0-9]{}))(\.[0-9]{})?([eE][+-][0-9]{})?",
            integers_quantifier, fraction_quantifier, exponent_quantifier
        ))
    } else {
        let format_type = types::JsonType::Number;
        Ok(format_type.to_regex().to_string())
    }
}

fn parse_integer_type(obj: &serde_json::Map<String, Value>) -> Result<String> {
    if obj.contains_key("minDigits") || obj.contains_key("maxDigits") {
        let (min_digits, max_digits) = helpers::validate_quantifiers(
            obj.get("minDigits").and_then(Value::as_u64),
            obj.get("maxDigits").and_then(Value::as_u64),
            1,
        )?;

        let quantifier = match (min_digits, max_digits) {
            (Some(min), Some(max)) => format!("{{{},{}}}", min, max),
            (Some(min), None) => format!("{{{},}}", min),
            (None, Some(max)) => format!("{{0,{}}}", max),
            (None, None) => "*".to_string(),
        };

        Ok(format!(r"(-)?(0|[1-9][0-9]{})", quantifier))
    } else {
        let format_type = types::JsonType::Integer;
        Ok(format_type.to_regex().to_string())
    }
}

fn parse_object_type(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let min_properties = obj.get("minProperties").and_then(|v| v.as_u64());
    let max_properties = obj.get("maxProperties").and_then(|v| v.as_u64());

    let num_repeats = helpers::get_num_items_pattern(min_properties, max_properties);

    if num_repeats.is_none() {
        return Ok(format!(r"\{{{}\}}", whitespace_pattern));
    }

    let allow_empty = if min_properties.unwrap_or(0) == 0 {
        "?"
    } else {
        ""
    };

    let additional_properties = obj.get("additionalProperties");

    let value_pattern = match additional_properties {
        None | Some(&Value::Bool(true)) => {
            // parse unconstrained object case
            let mut legal_types = vec![
                json!({"type": "string"}),
                json!({"type": "number"}),
                json!({"type": "boolean"}),
                json!({"type": "null"}),
            ];

            let depth = obj.get("depth").and_then(|v| v.as_u64()).unwrap_or(2);
            if depth > 0 {
                legal_types.push(json!({"type": "object", "depth": depth - 1}));
                legal_types.push(json!({"type": "array", "depth": depth - 1}));
            }

            let any_of = json!({"anyOf": legal_types});
            to_regex(&any_of, Some(whitespace_pattern), full_schema)?
        }
        Some(props) => to_regex(props, Some(whitespace_pattern), full_schema)?,
    };

    let key_value_pattern = format!(
        "{}{whitespace_pattern}:{whitespace_pattern}{value_pattern}",
        types::STRING
    );
    let key_value_successor_pattern =
        format!("{whitespace_pattern},{whitespace_pattern}{key_value_pattern}");
    let multiple_key_value_pattern =
        format!("({key_value_pattern}({key_value_successor_pattern}){{0,}}){allow_empty}");

    let res = format!(
        r"\{{{}{}{}\}}",
        whitespace_pattern, multiple_key_value_pattern, whitespace_pattern
    );

    Ok(res)
}

fn parse_array_type(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let num_repeats = helpers::get_num_items_pattern(
        obj.get("minItems").and_then(Value::as_u64),
        obj.get("maxItems").and_then(Value::as_u64),
    )
    .unwrap_or_else(|| String::from(""));

    if num_repeats.is_empty() {
        return Ok(format!(r"\[{0}\]", whitespace_pattern));
    }

    let allow_empty = if obj.get("minItems").and_then(Value::as_u64).unwrap_or(0) == 0 {
        "?"
    } else {
        ""
    };

    if let Some(items) = obj.get("items") {
        let items_regex = to_regex(items, Some(whitespace_pattern), full_schema)?;
        Ok(format!(
            r"\[{0}(({1})(,{0}({1})){2}){3}{0}\]",
            whitespace_pattern, items_regex, num_repeats, allow_empty
        ))
    } else {
        let mut legal_types = vec![
            json!({"type": "boolean"}),
            json!({"type": "null"}),
            json!({"type": "number"}),
            json!({"type": "integer"}),
            json!({"type": "string"}),
        ];

        let depth = obj.get("depth").and_then(Value::as_u64).unwrap_or(2);
        if depth > 0 {
            legal_types.push(json!({"type": "object", "depth": depth - 1}));
            legal_types.push(json!({"type": "array", "depth": depth - 1}));
        }

        let regexes: Result<Vec<String>> = legal_types
            .iter()
            .map(|t| to_regex(t, Some(whitespace_pattern), full_schema))
            .collect();

        let regexes = regexes?;
        let regexes_joined = regexes.join("|");

        Ok(format!(
            r"\[{0}(({1})(,{0}({1})){2}){3}{0}\]",
            whitespace_pattern, regexes_joined, num_repeats, allow_empty
        ))
    }
}
