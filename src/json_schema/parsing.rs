use std::num::NonZeroU64;

use regex::escape;
use serde_json::json;
use serde_json::Value;

use crate::json_schema::types;
use crate::JsonSchemaParserError;

type Result<T> = std::result::Result<T, JsonSchemaParserError>;

pub(crate) struct Parser<'a> {
    root: &'a Value,
    whitespace_pattern: &'a str,
    recursion_depth: usize,
    max_recursion_depth: usize,
}

impl<'a> Parser<'a> {
    // Max recursion depth is defined at level 3.
    // Defining recursion depth higher than that should be done cautiously, since
    // each +1 step on the depth blows up regex's size exponentially.
    //
    // For example, for simple referential json schema level 5 will produce regex size over 700K,
    // which seems counterproductive and likely to introduce performance issues.
    // It also breaks even `regex` sensible defaults with `CompiledTooBig` error.
    pub fn new(root: &'a Value) -> Self {
        Self {
            root,
            whitespace_pattern: types::WHITESPACE,
            recursion_depth: 0,
            max_recursion_depth: 3,
        }
    }

    pub fn with_whitespace_pattern(self, whitespace_pattern: &'a str) -> Self {
        Self {
            whitespace_pattern,
            ..self
        }
    }

    #[allow(dead_code)]
    pub fn with_max_recursion_depth(self, max_recursion_depth: usize) -> Self {
        Self {
            max_recursion_depth,
            ..self
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_regex(&mut self, json: &Value) -> Result<String> {
        match json {
            Value::Object(obj) if obj.is_empty() => self.parse_empty_object(),
            Value::Object(obj) if obj.contains_key("properties") => self.parse_properties(obj),
            Value::Object(obj) if obj.contains_key("allOf") => self.parse_all_of(obj),
            Value::Object(obj) if obj.contains_key("anyOf") => self.parse_any_of(obj),
            Value::Object(obj) if obj.contains_key("oneOf") => self.parse_one_of(obj),
            Value::Object(obj) if obj.contains_key("prefixItems") => self.parse_prefix_items(obj),
            Value::Object(obj) if obj.contains_key("enum") => self.parse_enum(obj),
            Value::Object(obj) if obj.contains_key("const") => self.parse_const(obj),
            Value::Object(obj) if obj.contains_key("$ref") => self.parse_ref(obj),
            Value::Object(obj) if obj.contains_key("type") => self.parse_type(obj),
            json => Err(JsonSchemaParserError::UnsupportedJsonSchema(Box::new(
                json.clone(),
            ))),
        }
    }

    fn parse_empty_object(&mut self) -> Result<String> {
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
        let regex = types
            .iter()
            .try_fold(Vec::with_capacity(types.len()), |mut acc, object| {
                self.to_regex(object).map(|string| {
                    acc.push(format!("({})", string));
                    acc
                })
            })?
            .join("|");
        Ok(regex)
    }

    fn parse_properties(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let mut regex = String::from(r"\{");

        let properties = obj
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| JsonSchemaParserError::PropertiesNotFound)?;

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
                let mut subregex =
                    format!(r#"{0}"{1}"{0}:{0}"#, self.whitespace_pattern, escape(name));
                subregex += &mut match self.to_regex(value) {
                    Ok(regex) => regex,
                    Err(e) if e.is_recursion_limit() => continue,
                    Err(e) => return Err(e),
                };
                match i {
                    i if i < last_required_pos => {
                        subregex = format!("{}{},", subregex, self.whitespace_pattern)
                    }
                    i if i > last_required_pos => {
                        subregex = format!("{},{}", self.whitespace_pattern, subregex)
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
                let mut subregex =
                    format!(r#"{0}"{1}"{0}:{0}"#, self.whitespace_pattern, escape(name));
                subregex += &mut match self.to_regex(value) {
                    Ok(regex) => regex,
                    Err(e) if e.is_recursion_limit() => continue,
                    Err(e) => return Err(e),
                };
                property_subregexes.push(subregex);
            }

            let mut possible_patterns = Vec::new();
            for i in 0..property_subregexes.len() {
                let mut pattern = String::new();
                for subregex in &property_subregexes[..i] {
                    pattern += &format!("({}{},)?", subregex, self.whitespace_pattern);
                }
                pattern += &property_subregexes[i];
                for subregex in &property_subregexes[i + 1..] {
                    pattern += &format!("({},{})?", self.whitespace_pattern, subregex);
                }
                possible_patterns.push(pattern);
            }

            regex += &format!("({})?", possible_patterns.join("|"));
        }

        regex += &format!("{}\\}}", self.whitespace_pattern);
        Ok(regex)
    }

    fn parse_all_of(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        match obj.get("allOf") {
            Some(Value::Array(all_of)) => {
                let subregexes: Result<Vec<String>> =
                    all_of.iter().map(|t| self.to_regex(t)).collect();

                let subregexes = subregexes?;
                let combined_regex = subregexes.join("");

                Ok(format!(r"({})", combined_regex))
            }
            _ => Err(JsonSchemaParserError::AllOfMustBeAnArray),
        }
    }

    fn parse_any_of(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        match obj.get("anyOf") {
            Some(Value::Array(any_of)) => {
                let subregexes: Result<Vec<String>> =
                    any_of.iter().map(|t| self.to_regex(t)).collect();

                let subregexes = subregexes?;

                Ok(format!(r"({})", subregexes.join("|")))
            }
            _ => Err(JsonSchemaParserError::AnyOfMustBeAnArray),
        }
    }

    fn parse_one_of(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        match obj.get("oneOf") {
            Some(Value::Array(one_of)) => {
                let subregexes: Result<Vec<String>> =
                    one_of.iter().map(|t| self.to_regex(t)).collect();

                let subregexes = subregexes?;
                let xor_patterns: Vec<String> = subregexes
                    .into_iter()
                    .map(|subregex| format!(r"(?:{})", subregex))
                    .collect();

                Ok(format!(r"({})", xor_patterns.join("|")))
            }
            _ => Err(JsonSchemaParserError::OneOfMustBeAnArray),
        }
    }

    fn parse_prefix_items(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        match obj.get("prefixItems") {
            Some(Value::Array(prefix_items)) => {
                let element_patterns: Result<Vec<String>> =
                    prefix_items.iter().map(|t| self.to_regex(t)).collect();

                let element_patterns = element_patterns?;

                let comma_split_pattern = format!("{0},{0}", self.whitespace_pattern);
                let tuple_inner = element_patterns.join(&comma_split_pattern);

                Ok(format!(r"\[{0}{tuple_inner}{0}\]", self.whitespace_pattern))
            }
            _ => Err(JsonSchemaParserError::PrefixItemsMustBeAnArray),
        }
    }

    fn parse_enum(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        match obj.get("enum") {
            Some(Value::Array(enum_values)) => {
                let choices: Result<Vec<String>> = enum_values
                    .iter()
                    .map(|choice| match choice {
                        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                            let json_string = serde_json::to_string(choice)?;
                            Ok(regex::escape(&json_string))
                        }
                        _ => Err(JsonSchemaParserError::UnsupportedEnumDataType(Box::new(
                            choice.clone(),
                        ))),
                    })
                    .collect();

                let choices = choices?;
                Ok(format!(r"({})", choices.join("|")))
            }
            _ => Err(JsonSchemaParserError::EnumMustBeAnArray),
        }
    }

    fn parse_const(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        match obj.get("const") {
            Some(const_value) => match const_value {
                Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                    let json_string = serde_json::to_string(const_value)?;
                    Ok(regex::escape(&json_string))
                }
                _ => Err(JsonSchemaParserError::UnsupportedConstDataType(Box::new(
                    const_value.clone(),
                ))),
            },
            None => Err(JsonSchemaParserError::ConstKeyNotFound),
        }
    }

    fn parse_ref(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        if self.recursion_depth > self.max_recursion_depth {
            return Err(JsonSchemaParserError::RefRecursionLimitReached(
                self.max_recursion_depth,
            ));
        }
        self.recursion_depth += 1;
        let ref_path = obj["$ref"]
            .as_str()
            .ok_or_else(|| JsonSchemaParserError::RefMustBeAString)?;

        let parts: Vec<&str> = ref_path.split('#').collect();

        let result = match parts.as_slice() {
            [fragment] | ["", fragment] => {
                let path_parts: Vec<&str> =
                    fragment.split('/').filter(|&s| !s.is_empty()).collect();
                let referenced_schema = Self::resolve_local_ref(self.root, &path_parts)?;
                self.to_regex(referenced_schema)
            }
            [base, fragment] => {
                if let Some(id) = self.root["$id"].as_str() {
                    if *base == id || base.is_empty() {
                        let path_parts: Vec<&str> =
                            fragment.split('/').filter(|&s| !s.is_empty()).collect();
                        let referenced_schema = Self::resolve_local_ref(self.root, &path_parts)?;
                        return self.to_regex(referenced_schema);
                    }
                }
                Err(JsonSchemaParserError::ExternalReferencesNotSupported(
                    Box::from(ref_path),
                ))
            }
            _ => Err(JsonSchemaParserError::InvalidReferenceFormat(Box::from(
                ref_path,
            ))),
        };
        self.recursion_depth -= 1;
        result
    }

    fn parse_type(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let instance_type = obj["type"]
            .as_str()
            .ok_or_else(|| JsonSchemaParserError::TypeMustBeAString)?;
        match instance_type {
            "string" => self.parse_string_type(obj),
            "number" => self.parse_number_type(obj),
            "integer" => self.parse_integer_type(obj),
            "array" => self.parse_array_type(obj),
            "object" => self.parse_object_type(obj),
            "boolean" => self.parse_boolean_type(),
            "null" => self.parse_null_type(),
            _ => Err(JsonSchemaParserError::UnsupportedType(Box::from(
                instance_type,
            ))),
        }
    }

    fn parse_boolean_type(&mut self) -> Result<String> {
        let format_type = types::JsonType::Boolean;
        Ok(format_type.to_regex().to_string())
    }

    fn parse_null_type(&mut self) -> Result<String> {
        let format_type = types::JsonType::Null;
        Ok(format_type.to_regex().to_string())
    }

    fn parse_string_type(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        if obj.contains_key("maxLength") || obj.contains_key("minLength") {
            let max_items = obj.get("maxLength");
            let min_items = obj.get("minLength");

            match (min_items, max_items) {
                (Some(min), Some(max)) if min.as_f64() > max.as_f64() => {
                    return Err(JsonSchemaParserError::MaxBoundError)
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
                None => Err(JsonSchemaParserError::StringTypeUnsupportedFormat(
                    Box::from(format),
                )),
            }
        } else {
            Ok(types::JsonType::String.to_regex().to_string())
        }
    }

    fn parse_number_type(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
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
            let (min_digits_integer, max_digits_integer) = Self::validate_quantifiers(
                obj.get("minDigitsInteger").and_then(Value::as_u64),
                obj.get("maxDigitsInteger").and_then(Value::as_u64),
                1,
            )?;

            let (min_digits_fraction, max_digits_fraction) = Self::validate_quantifiers(
                obj.get("minDigitsFraction").and_then(Value::as_u64),
                obj.get("maxDigitsFraction").and_then(Value::as_u64),
                0,
            )?;

            let (min_digits_exponent, max_digits_exponent) = Self::validate_quantifiers(
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

    fn parse_integer_type(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        if obj.contains_key("minDigits") || obj.contains_key("maxDigits") {
            let (min_digits, max_digits) = Self::validate_quantifiers(
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

    fn parse_object_type(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let min_properties = obj.get("minProperties").and_then(|v| v.as_u64());
        let max_properties = obj.get("maxProperties").and_then(|v| v.as_u64());

        let num_repeats = Self::get_num_items_pattern(min_properties, max_properties);

        if num_repeats.is_none() {
            return Ok(format!(r"\{{{}\}}", self.whitespace_pattern));
        }

        let allow_empty = if min_properties.unwrap_or(0) == 0 {
            "?"
        } else {
            ""
        };

        let additional_properties = obj.get("additionalProperties");

        let value_pattern = match additional_properties {
            None | Some(&Value::Bool(true)) => {
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

                let any_of = json!({"anyOf": &legal_types});
                self.to_regex(&any_of)?
            }
            Some(props) => self.to_regex(props)?,
        };

        let key_value_pattern = format!(
            "{}{1}:{1}{value_pattern}",
            types::STRING,
            self.whitespace_pattern,
        );
        let key_value_successor_pattern =
            format!("{0},{0}{key_value_pattern}", self.whitespace_pattern,);
        let multiple_key_value_pattern =
            format!("({key_value_pattern}({key_value_successor_pattern}){{0,}}){allow_empty}");

        let res = format!(
            r"\{{{0}{1}{0}\}}",
            self.whitespace_pattern, multiple_key_value_pattern
        );

        Ok(res)
    }

    fn parse_array_type(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String> {
        let num_repeats = Self::get_num_items_pattern(
            obj.get("minItems").and_then(Value::as_u64),
            obj.get("maxItems").and_then(Value::as_u64),
        )
        .unwrap_or_else(|| String::from(""));

        if num_repeats.is_empty() {
            return Ok(format!(r"\[{0}\]", self.whitespace_pattern));
        }

        let allow_empty = if obj.get("minItems").and_then(Value::as_u64).unwrap_or(0) == 0 {
            "?"
        } else {
            ""
        };

        if let Some(items) = obj.get("items") {
            let items_regex = self.to_regex(items)?;
            Ok(format!(
                r"\[{0}(({1})(,{0}({1})){2}){3}{0}\]",
                self.whitespace_pattern, items_regex, num_repeats, allow_empty
            ))
        } else {
            // parse unconstrained object case
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

            let regexes: Result<Vec<String>> =
                legal_types.iter().map(|t| self.to_regex(t)).collect();

            let regexes = regexes?;
            let regexes_joined = regexes.join("|");

            Ok(format!(
                r"\[{0}(({1})(,{0}({1})){2}){3}{0}\]",
                self.whitespace_pattern, regexes_joined, num_repeats, allow_empty
            ))
        }
    }

    fn resolve_local_ref<'b>(schema: &'b Value, path_parts: &[&str]) -> Result<&'b Value> {
        let mut current = schema;
        for &part in path_parts {
            current = current
                .get(part)
                .ok_or_else(|| JsonSchemaParserError::InvalidRefecencePath(Box::from(part)))?;
        }
        Ok(current)
    }

    fn validate_quantifiers(
        min_bound: Option<u64>,
        max_bound: Option<u64>,
        start_offset: u64,
    ) -> Result<(Option<NonZeroU64>, Option<NonZeroU64>)> {
        let min_bound = min_bound.map(|n| NonZeroU64::new(n.saturating_sub(start_offset)));
        let max_bound = max_bound.map(|n| NonZeroU64::new(n.saturating_sub(start_offset)));

        if let (Some(min), Some(max)) = (min_bound, max_bound) {
            if max < min {
                return Err(JsonSchemaParserError::MaxBoundError);
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
}
