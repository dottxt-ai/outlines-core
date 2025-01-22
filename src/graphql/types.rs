// allow `\"`, `\\`, or any character which isn't a control sequence
pub const STRING_INNER: &str = r#"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])"#;
pub const STRING: &str = r#""([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*""#;

pub const INTEGER: &str = r#"(-)?(0|[1-9][0-9]*)"#;
pub const NUMBER: &str = r#"((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?"#;
pub const BOOLEAN: &str = r#"(true|false)"#;
pub const NULL: &str = r#"null"#;

pub const WHITESPACE: &str = r#"[ ]?"#;

#[derive(Debug, PartialEq)]
pub enum GraphQLType {
    String,
    Integer,
    Number,
    Boolean,
    Null,
}

impl GraphQLType {
    pub fn to_regex(&self) -> &'static str {
        match self {
            GraphQLType::String => STRING,
            GraphQLType::Integer => INTEGER,
            GraphQLType::Number => NUMBER,
            GraphQLType::Boolean => BOOLEAN,
            GraphQLType::Null => NULL,
        }
    }
}
