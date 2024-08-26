use anyhow::{anyhow, Result};
use jsonschema::JSONSchema;
use regex::escape;
use serde_json::json;
use serde_json::Value;
use std::num::NonZeroU64;

// allow `\"`, `\\`, or any character which isn't a control sequence
pub static STRING_INNER: &str = r#"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])"#;
pub static STRING: &str = r#""([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*""#;

pub static INTEGER: &str = r#"(-)?(0|[1-9][0-9]*)"#;
pub static NUMBER: &str = r#"((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?"#;
pub static BOOLEAN: &str = r#"(true|false)"#;
pub static NULL: &str = r#"null"#;

pub static WHITESPACE: &str = r#"[ ]?"#;

#[derive(Debug, PartialEq)]
pub enum JsonType {
    String,
    Integer,
    Number,
    Boolean,
    Null,
}

impl JsonType {
    pub fn to_regex(&self) -> &'static str {
        match self {
            JsonType::String => STRING,
            JsonType::Integer => INTEGER,
            JsonType::Number => NUMBER,
            JsonType::Boolean => BOOLEAN,
            JsonType::Null => NULL,
        }
    }
}

pub static DATE_TIME: &str = r#""(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?""#;
pub static DATE: &str = r#""(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])""#;
pub static TIME: &str = r#""(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?""#;
pub static UUID: &str = r#""[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}""#;

#[derive(Debug, PartialEq)]
pub enum FormatType {
    DateTime,
    Date,
    Time,
    Uuid,
}

impl FormatType {
    pub fn to_regex(&self) -> &'static str {
        match self {
            FormatType::DateTime => DATE_TIME,
            FormatType::Date => DATE,
            FormatType::Time => TIME,
            FormatType::Uuid => UUID,
        }
    }

    pub fn from_str(s: &str) -> Option<FormatType> {
        match s {
            "date-time" => Some(FormatType::DateTime),
            "date" => Some(FormatType::Date),
            "time" => Some(FormatType::Time),
            "uuid" => Some(FormatType::Uuid),
            _ => None,
        }
    }
}

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
    let _compiled_schema = JSONSchema::compile(&json_value)
        .map_err(|e| anyhow!("Failed to compile JSON schema: {}", e))?;

    to_regex(&json_value, whitespace_pattern, &json_value)
}

pub fn to_regex(
    json: &Value,
    whitespace_pattern: Option<&str>,
    full_schema: &Value,
) -> Result<String> {
    let whitespace_pattern = whitespace_pattern.unwrap_or(WHITESPACE);

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
                    handle_properties(obj, whitespace_pattern, full_schema)
                }
                SchemaKeyword::AllOf => handle_all_of(obj, whitespace_pattern, full_schema),
                SchemaKeyword::AnyOf => handle_any_of(obj, whitespace_pattern, full_schema),
                SchemaKeyword::OneOf => handle_one_of(obj, whitespace_pattern, full_schema),
                SchemaKeyword::PrefixItems => {
                    handle_prefix_items(obj, whitespace_pattern, full_schema)
                }
                SchemaKeyword::Enum => handle_enum(obj, whitespace_pattern),
                SchemaKeyword::Const => handle_const(obj, whitespace_pattern),
                SchemaKeyword::Ref => handle_ref(obj, whitespace_pattern, full_schema),
                SchemaKeyword::Type => handle_type(obj, whitespace_pattern, full_schema),
                SchemaKeyword::EmptyObject => handle_empty_object(whitespace_pattern, full_schema),
            }
        }
        _ => Err(anyhow!("Invalid JSON Schema: expected an object")),
    }
}

fn handle_properties(
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

            if i < last_required_pos {
                subregex = format!("{}{},", subregex, whitespace_pattern);
            } else if i > last_required_pos {
                subregex = format!("{},{}", whitespace_pattern, subregex);
            }

            regex += &if is_required[i] {
                subregex
            } else {
                format!("({})?", subregex)
            };
        }
    } else {
        let mut property_subregexes = Vec::new();
        for (name, value) in properties.iter().rev() {
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

fn handle_all_of(
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

fn handle_any_of(
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

fn handle_one_of(
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

fn handle_prefix_items(
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

fn handle_enum(obj: &serde_json::Map<String, Value>, _whitespace_pattern: &str) -> Result<String> {
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

fn handle_const(obj: &serde_json::Map<String, Value>, _whitespace_pattern: &str) -> Result<String> {
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

fn handle_ref(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let ref_path = obj["$ref"]
        .as_str()
        .ok_or_else(|| anyhow!("'$ref' must be a string"))?;

    // TODO Only handle local references for now, maybe add support for remote references later
    if !ref_path.starts_with("#/") {
        return Err(anyhow!("Only local references are supported"));
    }

    let path_parts: Vec<&str> = ref_path[2..].split('/').collect();
    let referenced_schema = resolve_local_ref(full_schema, &path_parts)?;

    to_regex(referenced_schema, Some(whitespace_pattern), full_schema)
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

fn handle_type(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let instance_type = obj["type"]
        .as_str()
        .ok_or_else(|| anyhow!("'type' must be a string"))?;
    match instance_type {
        "string" => handle_string_type(obj),
        "number" => handle_number_type(obj),
        "integer" => handle_integer_type(obj),
        "array" => handle_array_type(obj, whitespace_pattern, full_schema),
        "object" => handle_object_type(obj, whitespace_pattern, full_schema),
        "boolean" => handle_boolean_type(),
        "null" => handle_null_type(),
        _ => Err(anyhow!("Unsupported type: {}", instance_type)),
    }
}

pub fn handle_empty_object(whitespace_pattern: &str, full_schema: &Value) -> Result<String> {
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

pub fn handle_boolean_type() -> Result<String> {
    let format_type = JsonType::Boolean;
    Ok(format_type.to_regex().to_string())
}

pub fn handle_null_type() -> Result<String> {
    let format_type = JsonType::Null;
    Ok(format_type.to_regex().to_string())
}

pub fn handle_string_type(obj: &serde_json::Map<String, Value>) -> Result<String> {
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
            .map_or("".to_string(), |n| format!("{}", n));

        Ok(format!(
            r#""{}{{{},{}}}""#,
            STRING_INNER, formatted_min, formatted_max,
        ))
    } else if let Some(pattern) = obj.get("pattern").and_then(Value::as_str) {
        if pattern.starts_with('^') && pattern.ends_with('$') {
            Ok(format!(r#"("{}")"#, &pattern[1..pattern.len() - 1]))
        } else {
            Ok(format!(r#"("{}")"#, pattern))
        }
    } else if let Some(format) = obj.get("format").and_then(Value::as_str) {
        match FormatType::from_str(format) {
            Some(format_type) => Ok(format_type.to_regex().to_string()),
            None => Err(anyhow::anyhow!(
                "Format {} is not supported by Outlines",
                format
            )),
        }
    } else {
        Ok(JsonType::String.to_regex().to_string())
    }
}

pub fn handle_number_type(obj: &serde_json::Map<String, Value>) -> Result<String> {
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
        let (min_digits_integer, max_digits_integer) = validate_quantifiers(
            obj.get("minDigitsInteger").and_then(Value::as_u64),
            obj.get("maxDigitsInteger").and_then(Value::as_u64),
            1,
        )?;

        let (min_digits_fraction, max_digits_fraction) = validate_quantifiers(
            obj.get("minDigitsFraction").and_then(Value::as_u64),
            obj.get("maxDigitsFraction").and_then(Value::as_u64),
            0,
        )?;

        let (min_digits_exponent, max_digits_exponent) = validate_quantifiers(
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
            (None, Some(max)) => format!("{{,{}}}", max),
            (None, None) => "+".to_string(),
        };

        let exponent_quantifier = match (min_digits_exponent, max_digits_exponent) {
            (Some(min), Some(max)) => format!("{{{},{}}}", min, max),
            (Some(min), None) => format!("{{{},}}", min),
            (None, Some(max)) => format!("{{,{}}}", max),
            (None, None) => "+".to_string(),
        };

        Ok(format!(
            r"((-)?(0|[1-9][0-9]{}))(\.[0-9]{})?([eE][+-][0-9]{})?",
            integers_quantifier, fraction_quantifier, exponent_quantifier
        ))
    } else {
        let format_type = JsonType::Number;
        Ok(format_type.to_regex().to_string())
    }
}
pub fn handle_integer_type(obj: &serde_json::Map<String, Value>) -> Result<String> {
    if obj.contains_key("minDigits") || obj.contains_key("maxDigits") {
        let (min_digits, max_digits) = validate_quantifiers(
            obj.get("minDigits").and_then(Value::as_u64),
            obj.get("maxDigits").and_then(Value::as_u64),
            1,
        )?;

        let quantifier = match (min_digits, max_digits) {
            (Some(min), Some(max)) => format!("{{{},{}}}", min, max),
            (Some(min), None) => format!("{{{},}}", min),
            (None, Some(max)) => format!("{{,{}}}", max),
            (None, None) => "*".to_string(),
        };

        Ok(format!(r"(-)?(0|[1-9][0-9]{})", quantifier))
    } else {
        let format_type = JsonType::Integer;
        Ok(format_type.to_regex().to_string())
    }
}
pub fn handle_object_type(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let min_properties = obj.get("minProperties").and_then(|v| v.as_u64());
    let max_properties = obj.get("maxProperties").and_then(|v| v.as_u64());

    let num_repeats = get_num_items_pattern(min_properties, max_properties);

    if num_repeats.is_none() {
        return Ok(format!(r"\{{{}}}", whitespace_pattern));
    }

    let allow_empty = if min_properties.unwrap_or(0) == 0 {
        "?"
    } else {
        ""
    };

    let additional_properties = obj.get("additionalProperties");

    let value_pattern =
        if additional_properties.is_none() || additional_properties == Some(&Value::Bool(true)) {
            // Handle unconstrained object case
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
            to_regex(&any_of, Some(whitespace_pattern), full_schema)
        } else {
            to_regex(
                additional_properties.unwrap(),
                Some(whitespace_pattern),
                full_schema,
            )
        };

    // TODO handle the unwrap
    let value_pattern = value_pattern.unwrap();

    let key_value_pattern = format!(
        "{}{whitespace_pattern}:{whitespace_pattern}{value_pattern}",
        STRING
    );
    let key_value_successor_pattern =
        format!("{whitespace_pattern},{whitespace_pattern}{key_value_pattern}");
    let multiple_key_value_pattern = format!(
        "({key_value_pattern}({key_value_successor_pattern}){{0,}}){allow_empty}"
    );
 
    let res = format!(
        r"\{{{}{}{}\}}",
        whitespace_pattern, multiple_key_value_pattern, whitespace_pattern
    );

    Ok(res)
}

pub fn handle_array_type(
    obj: &serde_json::Map<String, Value>,
    whitespace_pattern: &str,
    full_schema: &Value,
) -> Result<String> {
    let num_repeats = get_num_items_pattern(
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

/// HELPER FUNCTIONS

fn validate_quantifiers(
    min_bound: Option<u64>,
    max_bound: Option<u64>,
    start_offset: u64,
) -> Result<(Option<NonZeroU64>, Option<NonZeroU64>)> {
    let min_bound = min_bound.map(|n| NonZeroU64::new(n.saturating_sub(start_offset)));
    let max_bound = max_bound.map(|n| NonZeroU64::new(n.saturating_sub(start_offset)));

    if let (Some(min), Some(max)) = (min_bound, max_bound) {
        if max < min {
            return Err(anyhow!(
                "max bound must be greater than or equal to min bound"
            ));
        }
    }

    Ok((min_bound.flatten(), max_bound.flatten()))
}

fn get_num_items_pattern(min_items: Option<u64>, max_items: Option<u64>) -> Option<String> {
    let min_items = min_items.unwrap_or(0);

    match max_items {
        None => Some(format!("{{{},}}", min_items.saturating_sub(1))),
        Some(max_items) => {
            if max_items < 1 {
                None
            } else {
                Some(format!(
                    "{{{},{}}}",
                    min_items.saturating_sub(1),
                    max_items.saturating_sub(1)
                ))
            }
        }
    }
}
