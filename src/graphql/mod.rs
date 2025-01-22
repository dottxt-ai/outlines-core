mod error;
mod parsing;
mod types;

use error::GraphQLParserError;
pub use types::*;

type Result<T> = std::result::Result<T, GraphQLParserError>;

pub fn build_regex_from_schema(
    graphql_schema: &str,
    whitespace_pattern: Option<&str>,
) -> Result<String> {
    let mut parser = parsing::Parser::new(graphql_schema)?;
    if let Some(pattern) = whitespace_pattern {
        parser = parser.with_whitespace_pattern(pattern)
    }
    parser.to_regex()
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    use super::*;

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
            // Required integer property
            (
                r#"type Query {
                    count: Int!
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?\}"#,
                vec![r#"{ "count": 100 }"#],
                vec![r#"{ "count": "a" }"#, ""],
            ),
            (
                r#"type Query {
                    count: Int
                }"#,
                r#"\{([ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))?)?[ ]?\}"#,
                vec![r#"{ "count": 100 }"#],
                vec![r#"{ "count": "a" }"#, ""],
            ),
            // ==========================================================
            //                       Number Type
            // ==========================================================
            // Required number property
            (
                r#"type Query {
                    count: Float!
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[ ]?\}"#,
                vec![r#"{ "count": 100 }"#, r#"{ "count": 100.5 }"#],
                vec![""],
            ),
            (
                r#"type Query {
                    count: Float
                }"#,
                r#"\{([ ]?"count"[ ]?:[ ]?(((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?)?)?[ ]?\}"#,
                vec![r#"{ "count": 100 }"#, r#"{ "count": 100.5 }"#],
                vec![""],
            ),
            // ==========================================================
            //                       Array Type
            // ==========================================================
            // Required number property
            (
                r#"type Query {
                    count: [Float]!
                }"#,
                r#"\{[ ]?"count"[ ]?:[ ]?\[[ ]?(((((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?)?)(,[ ]?((((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?)?)){0,})[ ]?\][ ]?\}"#,
                vec![r#"{ "count": [100.5] }"#, r#"{ "count": [100] }"#],
                vec![""],
            ),
        ] {
            let result = build_regex_from_schema(schema, None).expect("To regex failed");
            assert_eq!(result, regex, "JSON Schema {} didn't match", schema);

            let re = Regex::new(&result).expect("Regex failed");
            for m in a_match {
                should_match(&re, m);
            }
            for not_m in not_a_match {
                should_not_match(&re, not_m);
            }
        }
    }
}
