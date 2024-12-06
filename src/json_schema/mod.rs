mod parsing;
mod types;

pub use types::*;

use serde_json::Value;

use crate::JsonSchemaParserError;

type Result<T> = std::result::Result<T, JsonSchemaParserError>;

pub fn build_regex_from_schema(json: &str, whitespace_pattern: Option<&str>) -> Result<String> {
    let json_value: Value = serde_json::from_str(json)?;
    to_regex(&json_value, whitespace_pattern)
}

pub fn to_regex(json: &Value, whitespace_pattern: Option<&str>) -> Result<String> {
    let mut parser = parsing::Parser::new(json);
    if let Some(pattern) = whitespace_pattern {
        parser = parser.with_whitespace_pattern(pattern)
    }
    parser.to_regex(json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    fn should_match(re: &Regex, value: &str) {
        // Asserts that value is fully matched.
        match re.find(value) {
            Some(matched) => {
                assert_eq!(
                    matched.as_str(),
                    value,
                    "Value should match, but does not for: {value}, re:\n{re}"
                );
                assert_eq!(matched.range(), 0..value.len());
            }
            None => unreachable!(
                "Value should match, but does not, in unreachable for: {value}, re:\n{re}"
            ),
        }
    }

    fn should_not_match(re: &Regex, value: &str) {
        // Asserts that regex does not find a match or not a full match.
        if let Some(matched) = re.find(value) {
            assert_ne!(
                matched.as_str(),
                value,
                "Value should NOT match, but does for: {value}, re:\n{re}"
            );
            assert_ne!(matched.range(), 0..value.len());
        }
    }

    #[test]
    fn test_schema_matches_regex() {
        for (schema, regex, a_match, not_a_match) in [
            // ==========================================================
            //                       Integer Type
            // ==========================================================
            (
                r#"{"title": "Foo", "type": "integer"}"#,
                INTEGER,
                vec!["0", "1", "-1"],
                vec!["01", "1.3", "t"],
            ),
            // Required integer property
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {"count": {"title": "Count", "type": "integer"}},
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?\}"#,
                vec![r#"{ "count": 100 }"#],
                vec![r#"{ "count": "a" }"#, ""],
            ),
            // Integer with minimum digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {"title": "Count", "type": "integer", "minDigits": 3}
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?(-)?(0|[1-9][0-9]{2,})[ ]?\}"#,
                vec![r#"{ "count": 100 }"#, r#"{ "count": 1000 }"#],
                vec![r#"{ "count": 10 }"#],
            ),
            // Integer with maximum digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {"title": "Count", "type": "integer", "maxDigits": 3}
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?(-)?(0|[1-9][0-9]{0,2})[ ]?\}"#,
                vec![r#"{ "count": 100 }"#, r#"{ "count": 10 }"#],
                vec![r#"{ "count": 1000 }"#],
            ),
            // Integer with minimum and maximum digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {
                            "title": "Count",
                            "type": "integer",
                            "minDigits": 3,
                            "maxDigits": 5
                        }
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?(-)?(0|[1-9][0-9]{2,4})[ ]?\}"#,
                vec![r#"{ "count": 100 }"#, r#"{ "count": 10000 }"#],
                vec![r#"{ "count": 10 }"#, r#"{ "count": 100000 }"#],
            ),
            // ==========================================================
            //                       Number Type
            // ==========================================================
            (
                r#"{"title": "Foo", "type": "number"}"#,
                NUMBER,
                vec!["1", "0", "1.3", "-1.3", "1.3e+9"],
                vec!["01", ".3", "1.3e9"],
            ),
            // Required number property
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {"count": {"title": "Count", "type": "number"}},
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?\}"#,
                vec![r#"{ "count": 100 }"#, r#"{ "count": 100.5 }"#],
                vec![""],
            ),
            // Number with min and max integer digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {
                            "title": "Count",
                            "type": "number",
                            "minDigitsInteger": 3,
                            "maxDigitsInteger": 5
                        }
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]{2,4}))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?\}"#,
                vec![r#"{ "count": 100.005 }"#, r#"{ "count": 10000.005 }"#],
                vec![r#"{ "count": 10.005 }"#, r#"{ "count": 100000.005 }"#],
            ),
            // Number with min and max fraction digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {
                            "title": "Count",
                            "type": "number",
                            "minDigitsFraction": 3,
                            "maxDigitsFraction": 5
                        }
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]{3,5})?([eE][+-][0-9]+)?[ ]?\}"#,
                vec![r#"{ "count": 1.005 }"#, r#"{ "count": 1.00005 }"#],
                vec![r#"{ "count": 1.05 }"#, r#"{ "count": 1.000005 }"#],
            ),
            // Number with min and max exponent digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {
                            "title": "Count",
                            "type": "number",
                            "minDigitsExponent": 3,
                            "maxDigitsExponent": 5
                        }
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]{3,5})?[ ]?\}"#,
                vec![r#"{ "count": 1.05e+001 }"#, r#"{ "count": 1.05e-00001 }"#],
                vec![r#"{ "count": 1.05e1 }"#, r#"{ "count": 1.05e0000001 }"#],
            ),
            // Number with min and max integer, fraction and exponent digits
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {
                        "count": {
                            "title": "Count",
                            "type": "number",
                            "minDigitsInteger": 3,
                            "maxDigitsInteger": 5,
                            "minDigitsFraction": 3,
                            "maxDigitsFraction": 5,
                            "minDigitsExponent": 3,
                            "maxDigitsExponent": 5
                        }
                    },
                    "required": ["count"]
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]{2,4}))(\.[0-9]{3,5})?([eE][+-][0-9]{3,5})?[ ]?\}"#,
                vec![r#"{ "count": 100.005e+001 }"#, r#"{ "count": 10000.00005e-00001 }"#],
                vec![r#"{ "count": 1.05e1 }"#, r#"{ "count": 100000.0000005e0000001 }"#],
            ),
            // ==========================================================
            //                       Array Type
            // ==========================================================
            (
                r#"{"title": "Foo", "type": "array", "items": {"type": "number"}}"#,
                format!(r#"\[{WHITESPACE}(({NUMBER})(,{WHITESPACE}({NUMBER})){{0,}})?{WHITESPACE}\]"#).as_str(),
                vec!["[1e+9,1.3]", "[]"], vec!["[1", r#"["a"]"#],
            ),
            // Array with a set min length
            (
                r#"{
                    "title": "Foo",
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 3
                }"#,
                format!(r#"\[{WHITESPACE}(({INTEGER})(,{WHITESPACE}({INTEGER})){{2,}}){WHITESPACE}\]"#).as_str(),
                vec!["[1,2,3]", "[1,2,3,4]"], vec!["[1]", "[1,2]", "[]"],
            ),
            // Array with a set max length
            (
                r#"{
                    "title": "Foo",
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 3
                }"#,
                format!(r#"\[{WHITESPACE}(({INTEGER})(,{WHITESPACE}({INTEGER})){{0,2}})?{WHITESPACE}\]"#).as_str(),
                vec!["[1,2,3]", "[1,2]", "[]"], vec!["[1,2,3,4]"],
            ),
            // Array with a set min/max length
            (
                r#"{
                    "title": "Foo",
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": 1
                }"#,
                format!(r#"\[{WHITESPACE}(({INTEGER})(,{WHITESPACE}({INTEGER})){{0,0}}){WHITESPACE}\]"#).as_str(),
                vec!["[1]"], vec!["[1, 2]", r#"["a"]"#, "[]"],
            ),
            // Array with zero length
            (
                r#"{
                    "title": "Foo",
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 0,
                    "maxItems": 0
                }"#,
                format!(r#"\[{WHITESPACE}\]"#).as_str(),
                vec!["[]"], vec!["[1, 2]", "[1]", "[1,2,3,4]"],
            ),
            // ==========================================================
            //                       String Type
            // ==========================================================
            (
                r#"{"title": "Foo", "type": "string"}"#,
                STRING,
                vec![
                    r#""(parenthesized_string)""#,
                    r#""malformed) parenthesis (((() string""#,
                    r#""quoted_string""#,
                    r#""double_\\escape""#,
                    r#""\\n""#,
                    r#""escaped \" quote""#,
                ],
                vec![
                    "unquotedstring",
                    r#""escape_\character""#,
                    r#""\n""#,
                    r#""unescaped " quote""#,
                ],
            ),
            (
                r#"{"title": "Foo", "type": "boolean"}"#,
                BOOLEAN,
                vec!["true", "false"],
                vec!["null", "0"],
            ),
            (
                r#"{"title": "Foo", "type": "null"}"#,
                NULL,
                vec!["null"],
                vec!["true", "0"],
            ),
            // String with maximum length
            (
                r#"{"title": "Foo", "type": "string", "maxLength": 3}"#,
                format!(r#""{STRING_INNER}{{0,3}}""#).as_str(),
                vec![r#""ab""#], vec![r#""a"""#, r#""abcd""#],
            ),
            // String with minimum length
            (
                r#"{"title": "Foo", "type": "string", "minLength": 3}"#,
                format!(r#""{STRING_INNER}{{3,}}""#).as_str(),
                vec![r#""abcd""#], vec![r#""ab""#, r#""abc"""#],
            ),
            // String with both minimum and maximum length
            (
                r#"{"title": "Foo", "type": "string", "minLength": 3, "maxLength": 5}"#,
                format!(r#""{STRING_INNER}{{3,5}}""#).as_str(),
                vec![r#""abcd""#], vec![r#""ab""#, r#""abcdef"""#],
            ),
            // String defined by a regular expression
            (
                r#"{"title": "Foo", "type": "string", "pattern": "^[a-z]$"}"#,
                r#"("[a-z]")"#,
                vec![r#""a""#], vec![r#""1""#],
            ),
            // Make sure strings are escaped with regex escaping
            (
                r#"{"title": "Foo", "const": ".*", "type": "string"}"#,
                r#""\.\*""#,
                vec![r#"".*""#], vec![r#""\s*""#, r#""\.\*""#],
            ),
            // Make sure strings are escaped with JSON escaping
            (
                r#"{"title": "Foo", "const": "\"", "type": "string"}"#,
                r#""\\"""#,
                vec![r#""\"""#], vec![r#"""""#],
            ),
            // ==========================================================
            //                       Const
            // ==========================================================
            // Const string
            (
                r#"{"title": "Foo", "const": "Marc", "type": "string"}"#,
                r#""Marc""#,
                vec![r#""Marc""#], vec![r#""Jonh""#, r#""Mar""#],
            ),
            // Const integer
            (
                r#"{"title": "Foo", "const": 0, "type": "integer"}"#,
                "0",
                vec!["0"], vec!["1", "a"],
            ),
            // Const float
            (
                r#"{"title": "Foo", "const": 0.2, "type": "float"}"#,
                r#"0\.2"#,
                vec!["0.2"], vec!["032"],
            ),
            // Const boolean
            (
                r#"{"title": "Foo", "const": true, "type": "boolean"}"#,
                "true",
                vec!["true"], vec!["false", "null"],
            ),
            // Const null
            (
                r#"{"title": "Foo", "const": null, "type": "null"}"#,
                "null",
                vec!["null"], vec!["none", ""],
            ),
            // ==========================================================
            //                      Enum
            // ==========================================================
            (
                r#"{"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"}"#,
                r#"("Marc"|"Jean")"#,
                vec![r#""Marc""#, r#""Jean""#], vec![r#""Jonh""#],
            ),
            // Enum with regex and JSON escaping
            (
                r#"{"title": "Foo", "enum": [".*", "\\s*"], "type": "string"}"#,
                r#"("\.\*"|"\\\\s\*")"#,
                vec![r#"".*""#, r#""\\s*""#], vec![r#""\.\*""#],
            ),
            // Enum integer
            (
                r#"{"title": "Foo", "enum": [0, 1], "type": "integer"}"#,
                r#"(0|1)"#,
                vec!["0", "1"], vec!["a"],
            ),
            // Enum mix of types
            (
                r#"{"title": "Foo", "enum": [6, 5.3, "potato", true, null]}"#,
                r#"(6|5\.3|"potato"|true|null)"#,
                vec!["6", "5.3", r#""potato""#, "true", "null"], vec!["none", "53"],
            ),
            // ==========================================================
            //                      UUID
            // ==========================================================
            (
                r#"{"title": "Foo", "type": "string", "format": "uuid"}"#,
                UUID,
                vec![
                    r#""123e4567-e89b-12d3-a456-426614174000""#,
                ],
                vec![
                    r#"123e4567-e89b-12d3-a456-426614174000"#,
                    r#""123e4567-e89b-12d3-a456-42661417400""#,
                    r#""123e4567-e89b-12d3-a456-42661417400g""#,
                    r#""123e4567-e89b-12d3-a456-42661417400-""#,
                    r#""""#,
                ],
            ),
            // Nested UUID
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {"uuid": {"type": "string", "format": "uuid"}}
                }"#,
                format!(r#"\{{([ ]?"uuid"[ ]?:[ ]?{UUID})?[ ]?\}}"#).as_str(),
                vec![
                    r#"{"uuid": "123e4567-e89b-12d3-a456-426614174000"}"#,
                ],
                vec![
                    r#"{"uuid":"123e4567-e89b-12d3-a456-42661417400"}"#,
                    r#"{"uuid":"123e4567-e89b-12d3-a456-42661417400g"}"#,
                    r#"{"uuid":"123e4567-e89b-12d3-a456-42661417400-"}"#,
                    r#"{"uuid":123e4567-e89b-12d3-a456-426614174000}"#, // missing quotes for value
                    r#"{"uuid":""}"#,
                ],
            ),
            // ==========================================================
            //                     DATE & TIME
            // ==========================================================
            // DATE-TIME
            (
                r#"{"title": "Foo", "type": "string", "format": "date-time"}"#,
                DATE_TIME,
                vec![
                    r#""2018-11-13T20:20:39Z""#,
                    r#""2016-09-18T17:34:02.666Z""#,
                    r#""2008-05-11T15:30:00Z""#,
                    r#""2021-01-01T00:00:00""#,
                ],
                vec![
                    "2018-11-13T20:20:39Z",
                    r#""2022-01-10 07:19:30""#, // missing T
                    r#""2022-12-10T10-04-29""#, // incorrect separator
                    r#""2023-01-01""#,
                ],
            ),
            // DATE
            (
                r#"{"title": "Foo", "type": "string", "format": "date"}"#,
                DATE,
                vec![
                    r#""2018-11-13""#,
                    r#""2016-09-18""#,
                    r#""2008-05-11""#,
                ],
                vec![
                    "2018-11-13",
                    r#""2015-13-01""#, // incorrect month
                    r#""2022-01""#, // missing day
                    r#""2022/12/01""#, // incorrect separator
                ],
            ),
            // TIME
            (
                r#"{"title": "Foo", "type": "string", "format": "time"}"#,
                TIME,
                vec![
                    r#""20:20:39Z""#,
                    r#""15:30:00Z""#,
                ],
                vec![
                    "20:20:39Z",
                    r#""25:30:00""#, // incorrect hour
                    r#""15:30""#, // missing seconds
                    r#""15:30:00.000""#, // missing Z
                    r#""15-30-00""#, // incorrect separator
                    r#""15:30:00+01:00""#, // incorrect separator
                ],
            ),
            // Nested DATE-TIME
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {"dateTime": {"type": "string", "format": "date-time"}}
                }"#,
                format!(r#"\{{([ ]?"dateTime"[ ]?:[ ]?{DATE_TIME})?[ ]?\}}"#).as_str(),
                vec![
                    r#"{"dateTime": "2018-11-13T20:20:39Z"}"#,
                    r#"{"dateTime":"2016-09-18T17:34:02.666Z"}"#,
                    r#"{"dateTime":"2008-05-11T15:30:00Z"}"#,
                    r#"{"dateTime":"2021-01-01T00:00:00"}"#,
                ],
                vec![
                    r#"{"dateTime":"2022-01-10 07:19:30"}"#, // missing T
                    r#"{"dateTime":"2022-12-10T10-04-29"}"#, // incorrect separator
                    r#"{"dateTime":2018-11-13T20:20:39Z}"#, // missing quotes for value
                    r#"{"dateTime":"2023-01-01"}"#,
                ],
            ),
            // Nested DATE
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {"date": {"type": "string", "format": "date"}}
                }"#,
                format!(r#"\{{([ ]?"date"[ ]?:[ ]?{DATE})?[ ]?\}}"#).as_str(),
                vec![
                    r#"{"date": "2018-11-13"}"#,
                    r#"{"date":"2016-09-18"}"#,
                    r#"{"date":"2008-05-11"}"#,
                ],
                vec![
                    r#"{"date":"2015-13-01"}"#, // incorrect month
                    r#"{"date":"2022-01"}"#, // missing day
                    r#"{"date":"2022/12/01"}"#, // incorrect separator
                    r#"{"date":2018-11-13}"#, // missing quotes for value
                ],
            ),
            // Nested TIME
            (
                r#"{
                    "title": "Foo",
                    "type": "object",
                    "properties": {"time": {"type": "string", "format": "time"}}
                }"#,
                format!(r#"\{{([ ]?"time"[ ]?:[ ]?{TIME})?[ ]?\}}"#).as_str(),
                vec![
                    r#"{"time": "20:20:39Z"}"#,
                    r#"{"time":"15:30:00Z"}"#,
                ],
                vec![
                    r#"{"time":"25:30:00"}"#, // incorrect hour
                    r#"{"time":"15:30"}"#, // missing seconds
                    r#"{"time":"15:30:00.000"}"#, // missing Z
                    r#"{"time":"15-30-00"}"#, // incorrect separator
                    r#"{"time":"15:30:00+01:00"}"#, // incorrect separator
                    r#"{"time":20:20:39Z}"#, // missing quotes for value
                ],
            ),
            // ==========================================================
            //                     ... Of
            // ==========================================================
            // oneOf
            (
                r#"{
                    "title": "Foo",
                    "oneOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]
                }"#,
                format!(r#"((?:"{STRING_INNER}*")|(?:{NUMBER})|(?:{BOOLEAN}))"#).as_str(),
                vec!["12.3", "true", r#""a""#],
                vec![
                    "null",
                    "",
                    "12true",
                    r#"1.3"a""#,
                    r#"12.3true"a""#,
                ],
            ),
            // anyOf
            (
                r#"{
                    "title": "Foo",
                    "anyOf": [{"type": "string"}, {"type": "integer"}]
                }"#,
                format!(r#"({STRING}|{INTEGER})"#).as_str(),
                vec!["12", r#""a""#],
                vec![r#"1"a""#],
            ),
            // allOf
            (
                r#"{
                    "title": "Foo",
                    "allOf": [{"type": "string"}, {"type": "integer"}]
                }"#,
                format!(r#"({STRING}{INTEGER})"#).as_str(),
                vec![r#""a"1"#],
                vec![r#""a""#, r#""1""#],
            ),
            // ==========================================================
            //                     Object
            // ==========================================================
            (
                r#"{
                    "title": "TestSchema",
                    "type": "object",
                    "properties": {
                        "test_dict": {
                            "title": "Test Dict",
                            "additionalProperties": {"type": "string"},
                            "type": "object"
                        }
                    },
                    "required": ["test_dict"]
                }"#,
                format!(r#"\{{{WHITESPACE}"test_dict"{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}{STRING}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}{STRING}){{0,}})?{WHITESPACE}\}}{WHITESPACE}\}}"#).as_str(),
                vec![
                    r#"{ "test_dict":{"foo":"bar","baz": "bif"}}"#,
                    r#"{ "test_dict":{"foo":"bar" }}"#,
                    r#"{ "test_dict":{}}"#,
                ],
                vec![
                    r#"{ "WRONG_KEY":{}}"#,
                    r#"{ "test_dict":{"wrong_type" 1}}"#,
                ],
            ),
            // Object containing object with undefined keys
            (
                r#"{
                    "title": "TestSchema",
                    "type": "object",
                    "properties": {
                        "test_dict": {
                            "title": "Test Dict",
                            "additionalProperties": {
                                "additionalProperties": {"type": "integer"},
                                "type": "object"
                            },
                            "type": "object"
                        }
                    },
                    "required": ["test_dict"]
                }"#,
                format!(r#"\{{{WHITESPACE}"test_dict"{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}){{0,}})?{WHITESPACE}\}}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}){{0,}})?{WHITESPACE}\}}){{0,}})?{WHITESPACE}\}}{WHITESPACE}\}}"#).as_str(),
                vec![
                    r#"{"test_dict": {"foo": {"bar": 123, "apple": 99}, "baz": {"bif": 456}}}"#,
                    r#"{"test_dict": {"anykey": {"anykey": 123}, "anykey2": {"bif": 456}}}"#,
                    r#"{"test_dict": {}}"#,
                    r#"{"test_dict": {"dict of empty dicts are ok": {} }}"#,
                ],
                vec![
                    r#"{"test_dict": {"anykey": {"ONLY Dict[Dict]": 123}, "No Dict[int]" 1: }}"#,
                    r#"{"test_dict": {"anykey": {"anykey": 123}, "anykey2": {"bif": "bof"}}}"#,
                ],
            ),
            // Object contains object with defined keys
            (
                r#"{
                    "title": "Bar",
                    "type": "object",
                    "properties": {
                        "fuzz": {
                            "title": "Foo",
                            "type": "object",
                            "properties": {"spam": {"title": "Spam", "type": "integer"}},
                            "required": ["spam"]
                        }
                    },
                    "required": ["fuzz"]
                }"#,
                format!(r#"\{{[ ]?"fuzz"[ ]?:[ ]?\{{[ ]?"spam"[ ]?:[ ]?{INTEGER}[ ]?\}}[ ]?\}}"#).as_str(),
                vec![r#"{ "fuzz": { "spam": 100 }}"#],
                vec![r#"{ "fuzz": { "spam": 100, "notspam": 500 }}"#, r#"{ "fuzz": {}}"#, r#"{ "spam": 5}"#],
            ),
            // Object with internal reference: #/
            (
                r##"{
                    "title": "User",
                    "type": "object",
                    "properties": {
                        "user_id": {"title": "User Id", "type": "integer"},
                        "name": {"title": "Name", "type": "string"},
                        "a": {"$ref": "#/properties/name"}
                    },
                    "required": ["user_id", "name", "a"]
                }"##,
                format!(r#"\{{[ ]?"user_id"[ ]?:[ ]?{INTEGER}[ ]?,[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"a"[ ]?:[ ]?{STRING}[ ]?\}}"#).as_str(),
                vec![r#"{"user_id": 100, "name": "John", "a": "Marc"}"#],
                vec![r#"{"user_id": 100, "name": "John", "a": 0}"#],
            ),
            // Object with internal reference: #/$defs
            (
                r##"{
                    "title": "User",
                    "type": "object",
                    "$defs": {"name": {"title": "Name2", "type": "string"}},
                    "properties": {
                        "user_id": {"title": "User Id", "type": "integer"},
                        "name": {"title": "Name", "type": "string"},
                        "name2": {"$ref": "#/$defs/name"}
                    },
                    "required": ["user_id", "name", "name2"]
                }"##,
                format!(r#"\{{[ ]?"user_id"[ ]?:[ ]?{INTEGER}[ ]?,[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"name2"[ ]?:[ ]?{STRING}[ ]?\}}"#).as_str(),
                vec![r#"{"user_id": 100, "name": "John", "name2": "Marc"}"#],
                vec![r#"{"user_id": 100, "name": "John", "name2": 0}"#],
            ),
            // Object with internal reference to $id: $id#/$defs
            // And required list requires more than being defined
            (
                r##"{
                    "$id": "customer",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "Customer",
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "address": {"$ref": "customer#/$defs/address"}
                    },
                    "required": [
                        "name",
                        "first_name",
                        "last_name",
                        "address",
                        "shipping_address",
                        "billing_address"
                    ],
                    "$defs": {
                        "address": {
                            "title": "Address",
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["street_address", "city", "state"],
                            "definitions": {
                                "state": {
                                    "type": "object",
                                    "title": "State",
                                    "properties": {"name": {"type": "string"}},
                                    "required": ["name"]
                                }
                            }
                        }
                    }
                }"##,
                format!(r#"\{{[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"last_name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"address"[ ]?:[ ]?\{{[ ]?"city"[ ]?:[ ]?{STRING}[ ]?\}}[ ]?\}}"#).as_str(),
                vec![
                    r#"{"name": "John", "last_name": "Doe", "address": {"city": "Paris"}}"#,
                ],
                vec![
                    r#"{"name": "John", "last_name": "Doe", "address": {}}"#,
                    r#"{"name": "John", "last_name": "Doe"}"#,
                ],
            ),
            // Object with optional properties:
            // - last required property in first position
            (
                r#"{
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "weapon": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                    },
                    "required": ["name"],
                    "title": "Character",
                    "type": "object"
                }"#,
                format!(r#"\{{[ ]?"name"[ ]?:[ ]?{STRING}([ ]?,[ ]?"age"[ ]?:[ ]?({INTEGER}|null))?([ ]?,[ ]?"weapon"[ ]?:[ ]?({STRING}|null))?[ ]?\}}"#).as_str(),
                vec![
                    r#"{ "name" : "Player" }"#,
                    r#"{ "name" : "Player", "weapon" : "sword" }"#,
                ],
                vec![
                    r#"{ "age" : 10, "weapon" : "sword" }"#,
                ],
            ),
            // Object with optional properties:
            // - last required property in middle position
            (
                r#"{
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "weapon": {"type": "string"},
                        "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                    },
                    "required": ["name", "weapon"],
                    "title": "Character",
                    "type": "object"
                }"#,
                format!(r#"\{{[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,([ ]?"age"[ ]?:[ ]?({INTEGER}|null)[ ]?,)?[ ]?"weapon"[ ]?:[ ]?{STRING}([ ]?,[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?[ ]?\}}"#).as_str(),
                vec![
                    r#"{ "name" : "Player" , "weapon" : "sword" }"#,
                    r#"{ "name" : "Player", "age" : 10, "weapon" : "sword" , "strength" : 10 }"#,
                ],
                vec![
                    r#"{ "weapon" : "sword" }"#,
                ],
            ),
            // Object with optional properties:
            // - last required property in last position
            (
                r#"{
                    "properties": {
                        "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "age": {"type": "integer"},
                        "armor": {"type": "string"},
                        "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "weapon": {"title": "Weapon", "type": "string"}
                    },
                    "required": ["age", "armor", "weapon"],
                    "title": "Character",
                    "type": "object"
                }"#,
                format!(r#"\{{([ ]?"name"[ ]?:[ ]?({STRING}|null)[ ]?,)?[ ]?"age"[ ]?:[ ]?{INTEGER}[ ]?,[ ]?"armor"[ ]?:[ ]?{STRING}[ ]?,([ ]?"strength"[ ]?:[ ]?({INTEGER}|null)[ ]?,)?[ ]?"weapon"[ ]?:[ ]?{STRING}[ ]?\}}"#).as_str(),
                vec![
                    r#"{ "name" : "Player", "age" : 10, "armor" : "plate", "strength" : 11, "weapon" : "sword" }"#,
                    r#"{ "age" : 10, "armor" : "plate", "weapon" : "sword" }"#,
                ],
                vec![
                    r#"{ "name" : "Kahlhanbeh", "armor" : "plate", "weapon" : "sword" }"#,
                ],
            ),
            // Object with all optional properties
            (
                r#"{
                    "properties": {
                        "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                    },
                    "title": "Character",
                    "type": "object"
                }"#,
                format!(r#"\{{([ ]?"name"[ ]?:[ ]?({STRING}|null)([ ]?,[ ]?"age"[ ]?:[ ]?({INTEGER}|null))?([ ]?,[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?|([ ]?"name"[ ]?:[ ]?({STRING}|null)[ ]?,)?[ ]?"age"[ ]?:[ ]?({INTEGER}|null)([ ]?,[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?|([ ]?"name"[ ]?:[ ]?({STRING}|null)[ ]?,)?([ ]?"age"[ ]?:[ ]?({INTEGER}|null)[ ]?,)?[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?[ ]?\}}"#).as_str(),
                vec![
                    r#"{ "name" : "Player" }"#,
                    r#"{ "name" : "Player", "age" : 10, "strength" : 10 }"#,
                    r#"{ "age" : 10, "strength" : 10 }"#,
                    "{ }",
                ],
                vec![r#"{ "foo": 0 } "#],
            ),
            // ==========================================================
            //                    Misc
            // ==========================================================
            // prefixItems
            (
                r#"{
                    "title": "Foo",
                    "prefixItems": [{"type": "string"}, {"type": "integer"}]
                }"#,
                format!(r#"\[{WHITESPACE}{STRING}{WHITESPACE},{WHITESPACE}{INTEGER}{WHITESPACE}\]"#).as_str(),
                vec![r#"["a", 1]"#],
                vec![r#"["a", 1, 1]"#, "[]"],
            ),
            // Unconstrained value (no schema)
            // (huge regex, but important test to verify matching it explicitely)
            (
                "{}",
                "((true|false))|(null)|(((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?)|((-)?(0|[1-9][0-9]*))|(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")|(\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\])(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\])){0,})?[ ]?\\])|(\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\])([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|\\{[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)([ ]?,[ ]?\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"[ ]?:[ ]?(\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\"|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(true|false)|null)){0,})?[ ]?\\}|\\[[ ]?(((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")(,[ ]?((true|false)|null|((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|(-)?(0|[1-9][0-9]*)|\"([^\"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\[\"\\\\])*\")){0,})?[ ]?\\])){0,})?[ ]?\\])){0,})?[ ]?\\})",
                vec![
                    r#""aaabbuecuh""#,
                    "5.554",
                    "true",
                    "null",
                    "5999",
                    r#"["a", "b"]"#,
                    r#"{"key": {"k2": "value"}}"#,
                ],
                vec!["this isnt valid json"],
            ),
        ] {
            let json: Value = serde_json::from_str(schema).expect("Can't parse json");
            let result = to_regex(&json, None).expect("To regex failed");
            assert_eq!(result, regex);

            let re = Regex::new(&result).expect("Regex failed");
            for m in a_match {
                should_match(&re, m);
            }
            for not_m in not_a_match {
                should_not_match(&re, not_m);
            }
        }
    }

    #[test]
    fn test_unconstrained_others() {
        for (schema, a_match, not_a_match) in [
            // Unconstrained Object
            (
                r#"{
                    "title": "Foo",
                    "type": "object"
                }"#,
                vec![
                    "{}",
                    r#"{"a": 1, "b": null}"#,
                    r#"{"a": {"z": {"g": 4}}, "b": null}"#,
                ],
                vec![
                    "1234",          // not an object
                    r#"["a", "a"]"#, // not an array
                ],
            ),
            // Unconstrained Array
            (
                r#"{"type": "array"}"#,
                vec![
                    r#"[1, {}, false]"#,
                    r#"[{}]"#,
                    r#"[{"a": {"z": "q"}, "b": null}]"#,
                    r#"[{"a": [1, 2, true], "b": null}]"#,
                    r#"[{"a": [1, 2, true], "b": {"a": "b"}}, 1, true, [1, [2]]]"#,
                ],
                vec![
                    // too deep, default unconstrained depth limit = 2
                    r#"[{"a": [1, 2, true], "b": {"a": "b"}}, 1, true, [1, [2, [3]]]]"#,
                    r#"[{"a": {"z": {"g": 4}}, "b": null}]"#,
                    r#"[[[[1]]]]"#,
                    // not an array
                    r#"{}"#,
                    r#"{"a": 1, "b": null}"#,
                    r#"{"a": {"z": {"g": 4}}, "b": null}"#,
                    "1234",
                    r#"{"a": "a"}"#,
                ],
            ),
        ] {
            let json: Value = serde_json::from_str(schema).expect("Can't parse json");
            let regex = to_regex(&json, None).expect("To regex failed");
            let re = Regex::new(&regex).expect("Regex failed");
            for m in a_match {
                should_match(&re, m);
            }
            for not_m in not_a_match {
                should_not_match(&re, not_m);
            }
        }
    }

    #[test]
    fn direct_recursion_in_array_and_default_behaviour() {
        let json = r##"
        {
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "children": {
                    "type": "array",
                    "items": { "$ref": "#" }
                }
            }
        }"##;

        let json_value: Value = serde_json::from_str(json).expect("Can't parse json");
        let regex = to_regex(&json_value, None);
        assert!(regex.is_ok(), "{:?}", regex);

        // Confirm the depth of 3 recursion levels by default, recursion level starts
        // when children start to have children
        let re = Regex::new(&regex.unwrap()).expect("Regex failed");
        for lvl in [
            // level 0
            r#"{ "name": "Az"}"#,
            r#"{ "name": "Az", "children": [] }"#,
            r#"{ "name": "Az", "children": [{"name": "Bo"}] }"#,
            // level 1
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [] }] }"#,
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li"}] }] }"#,
            // level 2
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li", "children": [] }] }] }"#,
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li", "children": [{"name": "Ho"}] }] }] }"#,
            // level 3
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li", "children": [{"name": "Ho", "children": [] }] }] }] }"#,
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li", "children": [{"name": "Ho", "children": [{"name": "Ro"}] }] }] }] }"#,
        ] {
            should_match(&re, lvl);
        }

        for lvl in [
            // level 4
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li", "children": [{"name": "Ho", "children": [{"name": "Ro", "children": [] }] }] }] }] }"#,
            r#"{ "name": "Az", "children": [{"name": "Bo", "children": [{"name": "Li", "children": [{"name": "Ho", "children": [{"name": "Ro", "children": [{"name": "Ks"}] }] }] }] }] }"#,
        ] {
            should_not_match(&re, lvl);
        }
    }

    #[test]
    fn indirect_recursion_with_recursion_level_regex_match() {
        let json = r##"{
          "type": "object",
          "properties": {
              "node": { "$ref": "#/definitions/node" }
          },
          "definitions": {
              "node": {
                  "type": "object",
                  "properties": {
                      "value": { "type": "integer" },
                      "next": { "$ref": "#/definitions/node" }
                  }
              }
          }
        }"##;
        let json_value: Value = serde_json::from_str(json).expect("Can't parse json");
        let mut parser = parsing::Parser::new(&json_value).with_max_recursion_depth(0);

        let result = parser.to_regex(&json_value);
        assert!(result.is_ok(), "{:?}", result);
        let regex = result.unwrap();
        assert_eq!(
            r#"\{([ ]?"node"[ ]?:[ ]?\{([ ]?"value"[ ]?:[ ]?(-)?(0|[1-9][0-9]*))?[ ]?\})?[ ]?\}"#,
            regex,
        );

        //  More readable version to confirm that logic is correct.
        //  Recursion depth 1:
        //  {
        //      ("node":
        //          {
        //              ("value":(-)?(0|[1-9][0-9]*)(,"next":{("value":(-)?(0|[1-9][0-9]*))?})?
        //              |
        //              ("value":(-)?(0|[1-9][0-9]*),)?"next":{("value":(-)?(0|[1-9][0-9]*))?})?
        //          }
        //      )?
        //  }
        //  Recursion depth 2:
        //  {
        //      ("node":
        //          {
        //              ("value":(-)?(0|[1-9][0-9]*)(,"next":{
        //                  ("value":(-)?(0|[1-9][0-9]*)(,"next":{("value":(-)?(0|[1-9][0-9]*))?})?
        //                  |
        //                  ("value":(-)?(0|[1-9][0-9]*),)?"next":{("value":(-)?(0|[1-9][0-9]*))?})?
        //              })?
        //              |
        //              ("value":(-)?(0|[1-9][0-9]*),)?"next":{
        //                  ("value":(-)?(0|[1-9][0-9]*)(,"next":{("value":(-)?(0|[1-9][0-9]*))?})?
        //                  |
        //                  ("value":(-)?(0|[1-9][0-9]*),)?"next":{("value":(-)?(0|[1-9][0-9]*))?})?
        //              })?
        //          }
        //      )?
        // }
        let mut parser = parser.with_max_recursion_depth(1);
        let result = parser.to_regex(&json_value);
        assert!(result.is_ok(), "{:?}", result);
        let regex = result.unwrap();
        assert_eq!(
            r#"\{([ ]?"node"[ ]?:[ ]?\{([ ]?"value"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)([ ]?,[ ]?"next"[ ]?:[ ]?\{([ ]?"value"[ ]?:[ ]?(-)?(0|[1-9][0-9]*))?[ ]?\})?|([ ]?"value"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?,)?[ ]?"next"[ ]?:[ ]?\{([ ]?"value"[ ]?:[ ]?(-)?(0|[1-9][0-9]*))?[ ]?\})?[ ]?\})?[ ]?\}"#,
            regex,
        );
    }

    #[test]
    fn triple_recursion_doesnt_fail() {
        let json = r##"
        {
            "definitions": {
                "typeA": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "child": { "$ref": "#/definitions/typeB" }
                    },
                    "required": ["name"]
                },
                "typeB": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "number" },
                        "next": { "$ref": "#/definitions/typeC" }
                    },
                    "required": ["value"]
                },
                "typeC": {
                    "type": "object",
                    "properties": {
                        "flag": { "type": "boolean" },
                        "parent": { "$ref": "#/definitions/typeA" }
                    },
                    "required": ["flag"]
                }
           },
          "$ref": "#/definitions/typeA"
        }"##;

        let json_value: Value = serde_json::from_str(json).expect("Can't parse json");
        let regex = to_regex(&json_value, None);
        assert!(regex.is_ok(), "{:?}", regex);
    }
}
