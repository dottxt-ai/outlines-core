//! Static collection of regular expressions for JSON and format types used
//! in generating a regular expression string based on a given JSON schema.

// allow `\"`, `\\`, or any character which isn't a control sequence
pub static STRING_INNER: &str = r#"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])"#;
pub static STRING: &str = r#""([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])*""#;
pub static INTEGER: &str = r#"(-)?(0|[1-9][0-9]*)"#;
pub static NUMBER: &str = r#"((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?"#;
pub static BOOLEAN: &str = r#"(true|false)"#;
pub static NULL: &str = r#"null"#;

/// Default whitespace pattern used for generating a regular expression from JSON schema.
///
/// It's being imposed since letting the model choose the number of white spaces and
/// new lines led to pathological behaviors, especially for small models,
/// see [example](https://github.com/dottxt-ai/outlines/issues/484)
pub static WHITESPACE: &str = r#"[ ]?"#;

/// Supported JSON types.
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

// https://www.iso.org/obp/ui/#iso:std:iso:8601:-1:ed-1:v1:en and https://stackoverflow.com/questions/3143070/regex-to-match-an-iso-8601-datetime-string
pub static DATE_TIME: &str = r#""(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?""#;
pub static DATE: &str = r#""(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])""#;
pub static TIME: &str = r#""(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?""#;
// https://datatracker.ietf.org/doc/html/rfc9562 and https://stackoverflow.com/questions/136505/searching-for-uuids-in-text-with-regex
pub static UUID: &str = r#""[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}""#;
// https://datatracker.ietf.org/doc/html/rfc3986#appendix-B
pub static URI: &str = r#""(?:(https?|ftp):\/\/([^\s:@]+(:[^\s:@]*)?@)?([a-zA-Z\d.-]+\.[a-zA-Z]{2,}|localhost)(:\d+)?(\/[^\s?#]*)?(\?[^\s#]*)?(#[^\s]*)?|urn:[a-zA-Z\d][a-zA-Z\d\-]{0,31}:[^\s]+)""#;
// https://www.rfc-editor.org/rfc/rfc5322 and https://stackoverflow.com/questions/13992403/regex-validation-of-email-addresses-according-to-rfc5321-rfc5322
pub static EMAIL: &str = r#""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""#;

/// Supported format type of the `JsonType::String`.
#[derive(Debug, PartialEq)]
pub enum FormatType {
    DateTime,
    Date,
    Time,
    Uuid,
    Uri,
    Email,
}

impl FormatType {
    pub fn to_regex(&self) -> &'static str {
        match self {
            FormatType::DateTime => DATE_TIME,
            FormatType::Date => DATE,
            FormatType::Time => TIME,
            FormatType::Uuid => UUID,
            FormatType::Uri => URI,
            FormatType::Email => EMAIL,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<FormatType> {
        match s {
            "date-time" => Some(FormatType::DateTime),
            "date" => Some(FormatType::Date),
            "time" => Some(FormatType::Time),
            "uuid" => Some(FormatType::Uuid),
            "uri" => Some(FormatType::Uri),
            "email" => Some(FormatType::Email),
            _ => None,
        }
    }
}
