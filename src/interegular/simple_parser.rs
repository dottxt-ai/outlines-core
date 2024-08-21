use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Formatter;
use std::marker::PhantomData;

#[derive(Debug, Clone, PartialEq)]
pub struct NoMatch {
    data: String,
    index: usize,
    expected: Vec<String>,
}

impl NoMatch {
    pub fn new(data: &str, index: usize, expected: Vec<String>) -> Self {
        NoMatch {
            data: data.to_string(),
            index,
            expected,
        }
    }
}

impl Display for NoMatch {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let context_start = self.index.saturating_sub(10);
        let context_end = (self.index + 10).min(self.data.len());
        let got = if self.index < self.data.len() {
            self.data[self.index..self.data.len().min(self.index + 5)].to_string()
        } else {
            "<end of input>".to_string()
        };

        write!(
            f,
            "Can not match at index {}. Got {:?}, expected any of {:?}.\nContext(data[{}:{}]): {:?}",
            self.index,
            got,
            self.expected,
            context_start,
            context_end,
            &self.data[context_start..context_end]
        )
    }
}

#[derive(Debug)]
pub struct SimpleParser<T> {
    pub data: String,
    pub index: usize,
    pub expected: HashMap<usize, Vec<String>>,
    _phantom: PhantomData<T>,
}

impl<T> SimpleParser<T> {
    pub fn new(data: &str) -> Self {
        SimpleParser {
            data: data.to_string(),
            index: 0,
            expected: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    pub fn peek_static(&mut self, expected: &str) -> bool {
        if self.data[self.index..].starts_with(expected) {
            true
        } else {
            self.expected
                .entry(self.index)
                .or_default()
                .push(expected.to_string());
            false
        }
    }

    pub fn static_match(&mut self, expected: &str) -> Result<(), NoMatch> {
        let len = expected.len();
        if self.index + len <= self.data.len()
            && &self.data[self.index..self.index + len] == expected
        {
            self.index += len;
            Ok(())
        } else {
            self.expected
                .entry(self.index)
                .or_default()
                .push(expected.to_string());
            Err(NoMatch::new(
                &self.data,
                self.index,
                vec![expected.to_string()],
            ))
        }
    }

    pub fn static_b(&mut self, expected: &str) -> bool {
        if self.data[self.index..].starts_with(expected) {
            self.index += expected.len();
            true
        } else {
            self.expected
                .entry(self.index)
                .or_default()
                .push(expected.to_string());
            false
        }
    }

    pub fn anyof(&mut self, strings: &[&str]) -> Result<String, NoMatch> {
        for &s in strings {
            if self.static_b(s) {
                return Ok(s.to_string());
            }
        }
        Err(NoMatch::new(
            &self.data,
            self.index,
            strings.iter().map(|&s| s.to_string()).collect(),
        ))
    }

    pub fn anyof_b(&mut self, strings: &[&str]) -> bool {
        for &s in strings {
            if self.static_b(s) {
                return true;
            }
        }
        false
    }

    pub fn any(&mut self, length: usize) -> Result<String, NoMatch> {
        if self.index + length <= self.data.len() {
            let res = self.data[self.index..self.index + length].to_string();
            self.index += length;
            Ok(res)
        } else {
            self.expected
                .entry(self.index)
                .or_default()
                .push(format!("<Any {}>", length));
            Err(NoMatch::new(
                &self.data,
                self.index,
                vec![format!("<Any {}>", length)],
            ))
        }
    }

    pub fn any_but(&mut self, strings: &[&str], length: usize) -> Result<String, NoMatch> {
        if self.index + length <= self.data.len() {
            let res = self.data[self.index..self.index + length].to_string();
            if !strings.contains(&&res[..]) {
                self.index += length;
                Ok(res)
            } else {
                self.expected
                    .entry(self.index)
                    .or_default()
                    .push(format!("<Any {} except {:?}>", length, strings));
                Err(NoMatch::new(
                    &self.data,
                    self.index,
                    vec![format!("<Any {} except {:?}>", length, strings)],
                ))
            }
        } else {
            self.expected
                .entry(self.index)
                .or_default()
                .push(format!("<Any {} except {:?}>", length, strings));
            Err(NoMatch::new(
                &self.data,
                self.index,
                vec![format!("<Any {} except {:?}>", length, strings)],
            ))
        }
    }

    pub fn multiple(
        &mut self,
        chars: &str,
        min: usize,
        max: Option<usize>,
    ) -> Result<String, NoMatch> {
        let mut result = String::new();

        // match minimum required characters
        for _ in 0..min {
            if let Some(c) = self.data[self.index..].chars().next() {
                if chars.contains(c) {
                    result.push(c);
                    self.index += c.len_utf8();
                } else {
                    self.expected
                        .entry(self.index)
                        .or_default()
                        .extend(chars.chars().map(|c| c.to_string()));
                    return Err(NoMatch::new(
                        &self.data,
                        self.index,
                        chars.chars().map(|c| c.to_string()).collect(),
                    ));
                }
            } else {
                return Err(NoMatch::new(
                    &self.data,
                    self.index,
                    chars.chars().map(|c| c.to_string()).collect(),
                ));
            }
        }

        // match additional characters up to max
        match max {
            Some(max) => {
                for _ in min..max {
                    if let Some(c) = self.data[self.index..].chars().next() {
                        if chars.contains(c) {
                            result.push(c);
                            self.index += c.len_utf8();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            None => {
                while let Some(c) = self.data[self.index..].chars().next() {
                    if chars.contains(c) {
                        result.push(c);
                        self.index += c.len_utf8();
                    } else {
                        break;
                    }
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peek_static() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert!(parser.peek_static("hello"));
        assert!(!parser.peek_static("world"));
        assert_eq!(parser.index, 0);
    }

    #[test]
    fn test_static_match() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert!(parser.static_match("hello").is_ok());
        assert_eq!(parser.index, 5);
        assert!(parser.static_b(" "));
        assert_eq!(parser.index, 6);
        assert!(parser.static_match("world").is_ok());
        assert_eq!(parser.index, 11);
        assert!(parser.static_match("!").is_err());
    }

    #[test]
    fn test_static_b() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert!(parser.static_b("hello"));
        assert_eq!(parser.index, 5);
        assert!(parser.static_b(" "));
        assert_eq!(parser.index, 6);
        assert!(!parser.static_b("hello"));
    }

    #[test]
    fn test_anyof() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert_eq!(parser.anyof(&["hi", "hello"]), Ok("hello".to_string()));
        assert_eq!(parser.index, 5);
        assert!(parser.anyof(&["hi", "hello"]).is_err());
    }

    #[test]
    fn test_anyof_b() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert!(parser.anyof_b(&["hi", "hello"]));
        assert_eq!(parser.index, 5);
        assert!(!parser.anyof_b(&["hi", "hello"]));
    }

    #[test]
    fn test_any() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert_eq!(parser.any(5), Ok("hello".to_string()));
        assert_eq!(parser.index, 5);
        assert_eq!(parser.any(1), Ok(" ".to_string()));
        assert!(parser.any(10).is_err());
    }

    #[test]
    fn test_any_but() {
        let mut parser = SimpleParser::<()>::new("hello world");
        assert_eq!(parser.any_but(&["world"], 5), Ok("hello".to_string()));
        assert_eq!(parser.index, 5);
        assert!(parser.any_but(&[" "], 1).is_err());
    }

    #[test]
    fn test_multiple() {
        let mut parser = SimpleParser::<()>::new("aaabbbccc");
        assert_eq!(parser.multiple("ab", 2, Some(4)), Ok("aaab".to_string()));
        assert_eq!(parser.index, 4);
        assert_eq!(parser.multiple("b", 1, None), Ok("bb".to_string()));
        assert_eq!(parser.index, 6);
        assert!(parser.multiple("d", 1, None).is_err());
    }

    #[test]
    fn test_no_match_display() {
        let no_match = NoMatch::new(
            //
            "hello world",
            6,
            vec!["a".to_string(), "b".to_string()],
        );
        let display = format!("{}", no_match);
        assert!(display.contains("index 6"));
        assert!(display.contains("Got \"world\""));
        assert!(display.contains("expected any of [\"a\", \"b\"]"));
        assert!(display.contains("Context(data[0:11]): \"hello world\""));
    }

    #[test]
    fn test_parser_with_complex_input() {
        let mut parser = SimpleParser::<()>::new("key1=value1;key2=value2");
        assert!(parser.static_b("key1"));
        assert!(parser.static_b("="));
        assert_eq!(
            parser.multiple("abcdefghijklmnopqrstuvwxyz123456789", 1, None),
            Ok("value1".to_string())
        );
        assert!(parser.static_b(";"));
        assert!(parser.static_b("key2"));
        assert!(parser.static_b("="));
        assert_eq!(
            parser.multiple("abcdefghijklmnopqrstuvwxyz123456789", 1, None),
            Ok("value2".to_string())
        );
        assert_eq!(parser.index, parser.data.len());
    }
}
