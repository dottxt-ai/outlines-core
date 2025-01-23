// allow `\"`, `\\`, or any character which isn't a control sequence
pub const STRING_INNER: &str = r#"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])"#;
pub const STRING: &str = r#""([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*""#;

pub const INTEGER: &str = r#"(-)?(0|[1-9][0-9]*)"#;
pub const NUMBER: &str = r#"((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?"#;
pub const BOOLEAN: &str = r#"(true|false)"#;
pub const NULL: &str = r#"null"#;

pub const DATE_TIME: &str = r#""(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?""#;
pub const DATE: &str = r#""(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])""#;
pub const TIME: &str = r#""(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?""#;
pub const UUID: &str = r#""[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}""#;
pub const URI: &str = r#"^(https?|ftp):\/\/([^\s:@]+(:[^\s:@]*)?@)?([a-zA-Z\d.-]+\.[a-zA-Z]{2,}|localhost)(:\d+)?(\/[^\s?#]*)?(\?[^\s#]*)?(#[^\s]*)?$|^urn:[a-zA-Z\d][a-zA-Z\d\-]{0,31}:[^\s]+$"#;
pub const EMAIL: &str = r#"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])$"#;

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
