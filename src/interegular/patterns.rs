#![allow(dead_code, unused_imports, unused_variables)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;
use std::vec;

use crate::interegular::fsm::SymbolTrait;
use crate::interegular::fsm::TransitionKey;
use crate::interegular::fsm::{Alphabet, Fsm};

const SPECIAL_CHARS_INNER: [&str; 2] = ["\\", "]"];
const SPECIAL_CHARS_STANDARD: [&str; 11] = ["+", "?", "*", ".", "$", "^", "\\", "(", ")", "[", "|"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexElement {
    Literal(char),
    CharGroup {
        chars: HashSet<char>,
        inverted: bool,
    },
    Repeated {
        element: Box<RegexElement>,
        min: usize,
        max: Option<usize>,
    },
    Concatenation(Vec<RegexElement>),
    Alternation(Vec<RegexElement>),
    Capture(Box<RegexElement>),
    Group(Box<RegexElement>),
    Anchor(AnchorType),
    Flag {
        element: Box<RegexElement>,
        added: Vec<Flag>,
        removed: Vec<Flag>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnchorType {
    StartOfLine,
    EndOfLine,
    WordBoundary,
    NotWordBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Flag {
    CaseInsensitive,
    Multiline,
    DotMatchesNewline,
    Unicode,
}

impl From<usize> for Flag {
    fn from(value: usize) -> Self {
        match value {
            0 => Flag::CaseInsensitive,
            1 => Flag::Multiline,
            2 => Flag::DotMatchesNewline,
            3 => Flag::Unicode,
            _ => panic!("Invalid flag value"),
        }
    }
}

fn is_alphabetic(c: String) -> bool {
    c.chars().next().unwrap().is_alphabetic()
}

fn _combine_char_groups(groups: &[RegexElement], negate: bool) -> RegexElement {
    let mut pos = HashSet::new();
    let mut neg = HashSet::new();

    for group in groups {
        match group {
            RegexElement::CharGroup { chars, inverted } => {
                if *inverted {
                    neg.extend(chars.iter().copied());
                } else {
                    pos.extend(chars.iter().copied());
                }
            }
            _ => panic!("Invalid group type"),
        }
    }

    if !neg.is_empty() {
        RegexElement::CharGroup {
            chars: neg.difference(&pos).copied().collect(),
            inverted: !negate,
        }
    } else {
        RegexElement::CharGroup {
            chars: pos.difference(&neg).copied().collect(),
            inverted: negate,
        }
    }
}

impl RegexElement {
    #[must_use]
    pub fn repeat(self, min: usize, max: Option<usize>) -> Self {
        RegexElement::Repeated {
            element: Box::new(self),
            min,
            max,
        }
    }

    #[must_use]
    pub fn capture(self) -> Self {
        RegexElement::Capture(Box::new(self))
    }

    #[must_use]
    pub fn group(self) -> Self {
        RegexElement::Group(Box::new(self))
    }

    #[must_use]
    pub fn with_flags(self, added: Vec<Flag>, removed: Vec<Flag>) -> Self {
        RegexElement::Flag {
            element: Box::new(self),
            added,
            removed,
        }
    }
}

impl RegexElement {
    #[must_use]
    pub fn to_fsm(
        &self,
        alphabet: Option<Alphabet<char>>,
        prefix_postfix: Option<(usize, Option<usize>)>,
        flags: Option<HashSet<Flag>>,
    ) -> Fsm<char> {
        match self {
            RegexElement::Literal(c) => {
                let alphabet = alphabet
                    .unwrap_or_else(|| self.get_alphabet(&flags.clone().unwrap_or_default()));
                let prefix_postfix = prefix_postfix.unwrap_or_else(|| self.get_prefix_postfix());

                let case_insensitive = flags
                    .clone()
                    .as_ref()
                    .map_or(false, |f| f.contains(&Flag::CaseInsensitive));

                let mut mapping = HashMap::<_, HashMap<_, _>>::new();
                let symbol = alphabet.get(c);

                let mut m = std::collections::HashMap::new();
                m.insert(symbol, TransitionKey::Symbol(1_usize));
                mapping.insert(TransitionKey::Symbol(0_usize), m);

                // states based on the symbols
                let unique_symbols = alphabet
                    .by_transition
                    .keys()
                    .copied()
                    .collect::<HashSet<_>>();

                let states = unique_symbols.iter().copied().collect();
                let finals = (1..=1).map(std::convert::Into::into).collect();

                Fsm::new(
                    alphabet,
                    states, // {0, 1}
                    0.into(),
                    finals, // {1}
                    mapping,
                )
            }
            RegexElement::CharGroup { chars, inverted } => {
                let alphabet = alphabet
                    .unwrap_or_else(|| self.get_alphabet(&flags.clone().unwrap_or_default()));
                let prefix_postfix = prefix_postfix.unwrap_or_else(|| self.get_prefix_postfix());

                assert!(
                    prefix_postfix == (0, Some(0)),
                    "Cannot have prefix/postfix on CharGroup-level"
                );

                let case_insensitive = flags
                    .clone()
                    .as_ref()
                    .map_or(false, |f| f.contains(&Flag::CaseInsensitive));

                let mut mapping = HashMap::<_, HashMap<_, _>>::new();

                if *inverted {
                    let chars = chars.clone();
                    let alphabet = alphabet.clone();
                    let alphabet_set = alphabet
                        .clone()
                        .by_transition
                        .keys()
                        .copied()
                        .collect::<HashSet<_>>();

                    let char_as_usize = chars
                        .iter()
                        .map(|c| TransitionKey::Symbol(*c as usize))
                        .collect();
                    let diff = alphabet_set
                        .difference(&char_as_usize)
                        .copied()
                        .collect::<Vec<_>>();

                    let mut m = std::collections::HashMap::new();
                    for symbol in diff {
                        m.insert(symbol, TransitionKey::Symbol(1_usize));
                    }
                    mapping.insert(TransitionKey::Symbol(0_usize), m);
                } else {
                    let chars = chars.clone();
                    for symbol in chars {
                        let mut m = std::collections::HashMap::new();
                        let symbol_value = alphabet.get(&symbol);
                        m.insert(symbol_value, TransitionKey::Symbol(1_usize));
                        mapping.insert(TransitionKey::Symbol(0_usize), m);
                    }
                }

                let states = (0..=1).map(std::convert::Into::into).collect();
                let finals = (1..=1).map(std::convert::Into::into).collect();

                Fsm::new(
                    alphabet,
                    states, // {0, 1}
                    0.into(),
                    finals, // {1}
                    mapping,
                )
            }
            RegexElement::Repeated { element, min, max } => {
                let unit = element.to_fsm(alphabet.clone(), None, flags.clone());
                let alphabet = alphabet
                    .unwrap_or_else(|| self.get_alphabet(&flags.clone().unwrap_or_default()));
                let mandatory = std::iter::repeat(unit.clone()).take(*min).fold(
                    Fsm::new(
                        // TODO: fix if alphabet is None
                        alphabet.clone(),
                        HashSet::new(),
                        0.into(),
                        HashSet::new(),
                        std::collections::HashMap::new(),
                    ),
                    |acc, f| Fsm::concatenate(&[acc, f]),
                );

                let optional = if max.is_none() {
                    unit.star()
                } else {
                    let mut optional = unit.clone();
                    optional.finals.insert(optional.initial);
                    optional = std::iter::repeat(optional.clone())
                        .take(max.unwrap() - min)
                        .fold(
                            Fsm::new(
                                alphabet.clone(),
                                HashSet::new(),
                                0.into(),
                                HashSet::new(),
                                std::collections::HashMap::new(),
                            ),
                            |acc, f| Fsm::concatenate(&[acc, f]),
                        );

                    optional
                };

                Fsm::concatenate(&[mandatory, optional])
            }
            RegexElement::Concatenation(parts) => {
                let mut current = vec![];
                for part in parts {
                    current.push(part.to_fsm(alphabet.clone(), None, flags.clone()));
                }

                Fsm::concatenate(&current)
            }
            RegexElement::Alternation(options) => {
                let mut current = vec![];
                for option in options {
                    current.push(option.to_fsm(alphabet.clone(), None, flags.clone()));
                }

                Fsm::union(&current)
            }
            // throw on non implemented variants
            _ => unimplemented!("FSM conversion not implemented for this variant"),
        }
    }

    #[must_use]
    pub fn get_alphabet<T: SymbolTrait>(&self, flags: &HashSet<Flag>) -> Alphabet<T> {
        match self {
            RegexElement::CharGroup { chars, .. } => {
                let relevant = if flags.contains(&Flag::CaseInsensitive) {
                    chars
                        .iter()
                        // .flat_map(|c| vec![c.to_ascii_lowercase(), c.to_ascii_uppercase()])
                        .flat_map(|c| vec![])
                        .collect()
                } else {
                    chars.iter().map(|c| (*c).into()).collect()
                };
                // Alphabet::from_groups(&[relevant, HashSet::from([TransitionKey::AnythingElse])])
                Alphabet::from_groups(&[relevant, HashSet::from(['\0'.into()])])
            }
            RegexElement::Literal(c) => Alphabet::from_groups(&[HashSet::from([(*c).into()])]),
            RegexElement::Repeated { element, .. } => element.get_alphabet(flags),
            RegexElement::Alternation(options) => {
                let mut alphabet = Alphabet::empty();
                for option in options {
                    let alphabets = vec![alphabet, option.get_alphabet(flags)];
                    let (res, new_to_old) = Alphabet::union(alphabets.as_slice());
                    alphabet = res;
                }
                alphabet
            }
            RegexElement::Concatenation(parts) => {
                let mut alphabet = Alphabet::empty();
                for part in parts {
                    let alphabets = vec![alphabet, part.get_alphabet(flags)];
                    let (res, new_to_old) = Alphabet::union(alphabets.as_slice());
                    alphabet = res;
                }
                alphabet
            }
            _ => unimplemented!("Alphabet not implemented for this variant"),
        }
    }

    #[must_use]
    pub fn get_prefix_postfix(&self) -> (usize, Option<usize>) {
        match self {
            RegexElement::CharGroup { .. } => (0, Some(0)),
            RegexElement::Literal(_) => (1, Some(1)),
            RegexElement::Repeated { element, min, max } => {
                let (l, h) = element.get_prefix_postfix();
                (l * min, max.and_then(|max| h.map(|h| h * max)))
            }
            RegexElement::Concatenation(parts) => {
                let mut pre = 0;
                let mut post = Some(0);

                for part in parts {
                    let (o_pre, o_post) = part.get_prefix_postfix();
                    pre = pre.max(o_pre);
                    post = match (post, o_post) {
                        (Some(p), Some(o)) => Some(p.max(o)),
                        (None, o) => o,
                        (p, None) => p,
                    };
                }

                (pre, post)
            }
            RegexElement::Alternation(options) => {
                let mut pre = 0;
                let mut post = Some(0);

                for option in options {
                    let (o_pre, o_post) = option.get_prefix_postfix();
                    pre = pre.max(o_pre);
                    post = match (post, o_post) {
                        (Some(p), Some(o)) => Some(p.max(o)),
                        (None, o) => o,
                        (p, None) => p,
                    };
                }

                (pre, post)
            }
            RegexElement::Capture(inner) => inner.get_prefix_postfix(),
            RegexElement::Group(inner) => inner.get_prefix_postfix(),
            RegexElement::Anchor(_) => (0, Some(0)),
            RegexElement::Flag { element, .. } => element.get_prefix_postfix(),
        }
    }

    #[must_use]
    pub fn get_lengths(&self) -> (usize, Option<usize>) {
        match self {
            RegexElement::CharGroup { .. } => (1, Some(1)),
            RegexElement::Literal(_) => (1, Some(1)),
            RegexElement::Repeated { element, min, max } => {
                let (l, h) = element.get_lengths();
                (l * min, max.and_then(|max| h.map(|h| h * max)))
            }
            RegexElement::Concatenation(parts) => {
                let mut low = 0;
                let mut high = Some(0);

                for part in parts {
                    let (l, h) = part.get_lengths();
                    low += l;
                    high = high.and_then(|high| h.map(|h| high + h));
                }

                (low, high)
            }
            RegexElement::Alternation(options) => {
                let mut low = None;
                let mut high = Some(0);

                for option in options {
                    let (l, h) = option.get_lengths();
                    low = Some(low.map_or(l, |low: usize| low.min(l)));
                    high = match (high, h) {
                        (Some(high), Some(h)) => Some(high.max(h)),
                        _ => None,
                    };
                }

                (low.unwrap_or(0), high)
            }
            RegexElement::Capture(inner) => inner.get_lengths(),
            RegexElement::Group(inner) => inner.get_lengths(),
            RegexElement::Anchor(_) => (0, Some(0)),
            RegexElement::Flag { element, .. } => element.get_lengths(),
        }
    }

    #[must_use]
    pub fn simplify(&self) -> Rc<RegexElement> {
        Rc::new(self.clone())
    }

    #[must_use]
    pub fn to_concrete(&self) -> RegexElement {
        self.clone()
    }
}

pub struct ParsePattern<'a> {
    parser: crate::interegular::simple_parser::SimpleParser<RegexElement>,
    flags: Option<Vec<usize>>,
    data: &'a str,
}

impl<'a> ParsePattern<'a> {
    #[must_use]
    pub fn new(data: &'a str) -> Self {
        ParsePattern {
            parser: crate::interegular::simple_parser::SimpleParser::new(data),
            flags: None,
            data,
        }
    }

    pub fn parse(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        let result = self.start()?;
        if self.parser.index < self.data.len() {
            let max_index = *self.parser.expected.keys().max().unwrap_or(&0);
            Err(crate::interegular::simple_parser::NoMatch::new(
                self.data,
                max_index,
                self.parser
                    .expected
                    .get(&max_index)
                    .unwrap_or(&vec![])
                    .clone(),
            ))
        } else {
            Ok(result)
        }
    }

    fn start(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        self.flags = None;
        let p = self.pattern()?;
        if let Some(flags) = self.flags.take() {
            Ok(p.with_flags(flags.iter().map(|f| Flag::from(*f)).collect(), vec![]))
        } else {
            Ok(p)
        }
    }

    fn pattern(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        let mut options = vec![self.conc()?];
        while self.parser.static_b("|") {
            options.push(self.conc()?);
        }
        Ok(RegexElement::Alternation(options))
    }

    fn conc(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        let mut parts = vec![];
        while let Ok(obj) = self.obj() {
            parts.push(obj);
        }
        Ok(RegexElement::Concatenation(parts))
    }

    fn obj(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        if self.parser.static_b("(") {
            self.group()
        } else {
            match self.atom() {
                Ok(atom) => self.repetition(atom),
                Err(_) => Err(crate::interegular::simple_parser::NoMatch::new(
                    self.data,
                    self.parser.index,
                    vec!["(".to_string()],
                )),
            }
        }
    }

    fn atom(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        if self.parser.static_b("[") {
            match self.chargroup() {
                Ok(cg) => self.repetition(cg),
                Err(_) => Err(crate::interegular::simple_parser::NoMatch::new(
                    self.data,
                    self.parser.index,
                    vec!["[".to_string()],
                )),
            }
        } else if self.parser.static_b("\\") {
            match self.escaped(false) {
                Ok(cg) => self.repetition(cg),
                Err(_) => Err(crate::interegular::simple_parser::NoMatch::new(
                    self.data,
                    self.parser.index,
                    vec!["\\".to_string()],
                )),
            }
        } else if self.parser.static_b(".") {
            let cg = RegexElement::CharGroup {
                chars: vec!['\n'].into_iter().collect(),
                inverted: true,
            };
            self.repetition(cg)
        } else if self.parser.static_b("$") {
            // Unsupported
            Err(crate::interegular::simple_parser::NoMatch::new(
                self.data,
                self.parser.index,
                vec!["'$'".to_string()],
            ))
        } else if self.parser.static_b("^") {
            // Unsupported
            Err(crate::interegular::simple_parser::NoMatch::new(
                self.data,
                self.parser.index,
                vec!["'^'".to_string()],
            ))
        } else {
            let c = self.parser.any_but(&SPECIAL_CHARS_STANDARD, 1)?;
            // Ok(RegexElement::CharGroup {
            //     chars: vec![c.chars().next().unwrap()].into_iter().collect(),
            //     inverted: false,
            // })
            Ok(RegexElement::Literal(c.chars().next().unwrap()))
        }
    }

    fn group(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        if self.parser.static_b("?") {
            self.extension_group()
        } else {
            let p = self.pattern().unwrap();
            self.parser.static_b(")");
            self.repetition(p)
        }
    }

    fn extension_group(
        &mut self,
    ) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        let c = self.parser.any(1)?;
        if "aiLmsux-".contains(&c) {
            self.parser.index -= 1;
            let added_flags = self.parser.multiple("aiLmsux", 0, None)?;
            let removed_flags = if self.parser.static_b("-") {
                self.parser.multiple("aiLmsux", 1, None)?
            } else {
                String::new()
            };

            // TODO: missing cases
        }
        unimplemented!("Missing cases")
    }

    fn repetition(
        &mut self,
        base: RegexElement,
    ) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        if self.parser.static_b("*") {
            self.parser.static_b("?");
            Ok(RegexElement::Repeated {
                element: Box::new(base),
                min: 0,
                max: None,
            })
        } else if self.parser.static_b("+") {
            self.parser.static_b("?");
            Ok(RegexElement::Repeated {
                element: Box::new(base),
                min: 1,
                max: None,
            })
        } else if self.parser.static_b("?") {
            self.parser.static_b("?");
            Ok(RegexElement::Repeated {
                element: Box::new(base),
                min: 0,
                max: Some(1),
            })
        } else if self.parser.static_b("{") {
            let n = self.number().unwrap_or(0);
            let m = if self.parser.static_b(",") {
                match self.number() {
                    Ok(num) => Some(num),
                    Err(_) => None,
                }
            } else {
                Some(n)
            };
            let _ = self.parser.static_match("}");
            self.parser.static_b("?");
            Ok(RegexElement::Repeated {
                element: Box::new(base),
                min: n,
                max: m,
            })
        } else {
            Ok(base)
        }
    }

    fn number(&mut self) -> Result<usize, crate::interegular::simple_parser::NoMatch> {
        let num = self.parser.multiple("0123456789", 1, None)?;
        Ok(num.parse().unwrap())
    }

    fn chargroup(&mut self) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        let negate = self.parser.static_b("^");
        let mut groups = vec![];
        while let Ok(group) = self.chargroup_inner() {
            groups.push(group);
        }
        let _ = self.parser.static_match("]");
        if groups.len() == 1 {
            let f = groups[0].clone();
            match f {
                RegexElement::CharGroup { chars, inverted } => Ok(RegexElement::CharGroup {
                    chars,
                    inverted: inverted ^ negate,
                }),
                _ => panic!("Invalid group type"),
            }
        } else if groups.is_empty() {
            Ok(RegexElement::CharGroup {
                chars: HashSet::new(),
                inverted: negate,
            })
        } else {
            Ok(_combine_char_groups(&groups, negate))
        }
    }

    fn chargroup_inner(
        &mut self,
    ) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        let base = if self.parser.static_b("\\") {
            self.escaped(true)
        } else {
            let c = self.parser.any_but(&SPECIAL_CHARS_INNER, 1)?;
            Ok(RegexElement::CharGroup {
                chars: vec![c.chars().next().unwrap()].into_iter().collect(),
                inverted: false,
            })
        };
        let base_copy = base.clone();
        if self.parser.static_b("-") {
            let end = if self.parser.static_b("\\") {
                self.escaped(true)?
            } else if self.parser.peek_static("]") {
                _combine_char_groups(
                    &[
                        base_copy?,
                        RegexElement::CharGroup {
                            chars: vec!['-'].into_iter().collect(),
                            inverted: false,
                        },
                    ],
                    false,
                )
            } else {
                let c = self.parser.any_but(&SPECIAL_CHARS_INNER, 1)?;
                RegexElement::CharGroup {
                    chars: vec![c.chars().next().unwrap()].into_iter().collect(),
                    inverted: false,
                }
            };

            let low = match base? {
                RegexElement::CharGroup { chars, .. } => *chars.iter().next().unwrap(),
                _ => panic!("Invalid group type"),
            };
            let high = match end {
                RegexElement::CharGroup { chars, .. } => *chars.iter().next().unwrap(),
                _ => panic!("Invalid group type"),
            };

            assert!(low <= high, "Invalid Character-range");

            let chars = (low..=high).collect();
            return Ok(RegexElement::CharGroup {
                chars,
                inverted: false,
            });
        }

        base
    }

    fn escaped(
        &mut self,
        inner: bool,
    ) -> Result<RegexElement, crate::interegular::simple_parser::NoMatch> {
        if self.parser.static_b("x") {
            let n = self.parser.multiple("0123456789abcdefABCDEF", 2, Some(2))?;
            let c = char::from_u32(u32::from_str_radix(&n, 16).unwrap()).unwrap();
            Ok(RegexElement::CharGroup {
                chars: vec![c].into_iter().collect(),
                inverted: false,
            })
        } else if self.parser.static_b("0") {
            let n = self.parser.multiple("01234567", 1, Some(2))?;
            let c = char::from_u32(u32::from_str_radix(&n, 8).unwrap()).unwrap();
            Ok(RegexElement::CharGroup {
                chars: vec![c].into_iter().collect(),
                inverted: false,
            })
        } else if self.parser.anyof_b(&["N", "p", "P", "u", "U"]) {
            unimplemented!("regex module unicode properties are not supported.")
        } else if !inner {
            let n = self
                .parser
                .multiple("01234567", 3, Some(3))
                .unwrap_or_default();
            if !n.is_empty() {
                let c = char::from_u32(u32::from_str_radix(&n, 8).unwrap()).unwrap();
                Ok(RegexElement::CharGroup {
                    chars: vec![c].into_iter().collect(),
                    inverted: false,
                })
            } else {
                let n = self
                    .parser
                    .multiple("0123456789", 1, Some(2))
                    .unwrap_or_default();
                if !n.is_empty() {
                    unimplemented!("Group references are not implemented")
                } else {
                    let n = self.parser.any_but(
                        &[
                            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
                            "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B",
                            "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
                            "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                        ],
                        1,
                    )?;
                    let c = n.chars().next().unwrap();
                    if c.is_alphabetic() {
                        Err(crate::interegular::simple_parser::NoMatch::new(
                            self.data,
                            self.parser.index,
                            vec![n],
                        ))
                    } else {
                        Ok(RegexElement::CharGroup {
                            chars: vec![c].into_iter().collect(),
                            inverted: false,
                        })
                    }
                }
            }
        } else {
            let n = self
                .parser
                .multiple("01234567", 1, Some(3))
                .unwrap_or_default();
            if !n.is_empty() {
                let c = char::from_u32(u32::from_str_radix(&n, 8).unwrap()).unwrap();
                Ok(RegexElement::CharGroup {
                    chars: vec![c].into_iter().collect(),
                    inverted: false,
                })
            } else {
                let c = self.parser.anyof(&[
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                ])?;
                if is_alphabetic(c.clone()) {
                    Err(crate::interegular::simple_parser::NoMatch::new(
                        self.data,
                        self.parser.index,
                        vec![c],
                    ))
                } else {
                    Ok(RegexElement::CharGroup {
                        chars: c.chars().collect(),
                        inverted: false,
                    })
                }
            }
        }
    }
}

pub fn parse_pattern(pattern: &str) -> Result<RegexElement, super::simple_parser::NoMatch> {
    let mut parser = ParsePattern::new(pattern);

    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pattern_simple() {
        let pattern: &str = "a";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![RegexElement::Literal('a')])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_alternation() {
        let pattern = "a|b|c";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![RegexElement::Literal('a')]),
                RegexElement::Concatenation(vec![RegexElement::Literal('b')]),
                RegexElement::Concatenation(vec![RegexElement::Literal('c')])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_concatenation() {
        let pattern = "abc";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![
                    RegexElement::Literal('a'),
                    RegexElement::Literal('b'),
                    RegexElement::Literal('c')
                ])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_repetition() {
        let pattern = "a*b+c?";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![
                    RegexElement::Repeated {
                        element: Box::new(RegexElement::Literal('a')),
                        min: 0,
                        max: None
                    },
                    RegexElement::Repeated {
                        element: Box::new(RegexElement::Literal('b')),
                        min: 1,
                        max: None
                    },
                    RegexElement::Repeated {
                        element: Box::new(RegexElement::Literal('c')),
                        min: 0,
                        max: Some(1)
                    }
                ])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_chargroup() {
        let pattern = "[abc]";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![RegexElement::CharGroup {
                    chars: vec!['a', 'b', 'c'].into_iter().collect(),
                    inverted: false
                }])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_negated_chargroup() {
        let pattern = "[^abc]";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![RegexElement::CharGroup {
                    chars: vec!['a', 'b', 'c'].into_iter().collect(),
                    inverted: true
                }])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_escaped_chars() {
        let pattern = r"\.\*\+\?\|\(\)\[\]\{\}\^\$";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![
                    RegexElement::CharGroup {
                        chars: HashSet::from(['.']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['*']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['+']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['?']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['|']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['(']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from([')']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['[']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from([']']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['{']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['}']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['^']),
                        inverted: false
                    },
                    RegexElement::CharGroup {
                        chars: HashSet::from(['$']),
                        inverted: false
                    }
                ]),
            ]))
        );
    }

    // #[test]
    // fn test_parse_pattern_complex() {
    //     let pattern = r"(a|b)*c+[def]\d{2,4}";
    //     let result = parse_pattern(pattern);
    //     assert_eq!(
    //         result,
    //         Ok(RegexElement::Alternation(vec![
    //             RegexElement::Concatenation(vec![
    //                 RegexElement::Repeated {
    //                     element: Box::new(RegexElement::Group(Box::new(
    //                         RegexElement::Alternation(vec![
    //                             RegexElement::Concatenation(vec![RegexElement::Literal('a')]),
    //                             RegexElement::Concatenation(vec![RegexElement::Literal('b')])
    //                         ])
    //                     ))),
    //                     min: 0,
    //                     max: None
    //                 },
    //                 RegexElement::Repeated {
    //                     element: Box::new(RegexElement::Literal('c')),
    //                     min: 1,
    //                     max: None
    //                 },
    //                 RegexElement::CharGroup {
    //                     chars: vec!['d', 'e', 'f'].into_iter().collect(),
    //                     inverted: false
    //                 },
    //                 RegexElement::Repeated {
    //                     element: Box::new(RegexElement::CharGroup {
    //                         chars: ('0'..='9').collect(),
    //                         inverted: false
    //                     }),
    //                     min: 2,
    //                     max: Some(4)
    //                 }
    //             ])
    //         ]))
    //     );
    // }

    #[test]
    fn test_parse_pattern_dot() {
        let pattern = "a.b";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![
                    RegexElement::Literal('a'),
                    RegexElement::CharGroup {
                        chars: vec!['\n'].into_iter().collect(),
                        inverted: true
                    },
                    RegexElement::Literal('b')
                ])
            ]))
        );
    }

    #[test]
    fn test_parse_range() {
        let pattern = "[a-f]";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![RegexElement::CharGroup {
                    chars: ('a'..='f').collect(),
                    inverted: false
                }])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_repeat() {
        let pattern = "a{3,6}";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![RegexElement::Repeated {
                    element: Box::new(RegexElement::Literal('a')),
                    min: 3,
                    max: Some(6)
                }])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_anchors() {
        let pattern = "abc$";
        let result = parse_pattern(pattern);
        assert_eq!(
            result,
            Ok(RegexElement::Alternation(vec![
                RegexElement::Concatenation(vec![
                    RegexElement::Literal('a'),
                    RegexElement::Literal('b'),
                    RegexElement::Literal('c'),
                ])
            ]))
        );
    }

    #[test]
    fn test_parse_pattern_invalid() {
        let pattern = ")(";
        let result = parse_pattern(pattern);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_pattern_simple_to_fsm() {
        let pattern: &str = "a";
        let result = parse_pattern(pattern).unwrap();

        let alphabet = Alphabet {
            symbol_mapping: HashMap::from([
                ('a', TransitionKey::Symbol(1)),
                ('\0', TransitionKey::Symbol(0)),
            ]),
            by_transition: HashMap::from([
                (TransitionKey::Symbol(0), vec!['\0']),
                (TransitionKey::Symbol(1), vec!['a']),
            ]),
        };

        let result = result.to_fsm(Some(alphabet.clone()), None, None);

        let expected = Fsm {
            alphabet,
            states: HashSet::from([TransitionKey::Symbol(0), TransitionKey::Symbol(1)]),
            initial: TransitionKey::Symbol(0),
            finals: HashSet::from([TransitionKey::Symbol(1)]),
            map: HashMap::from([
                (
                    TransitionKey::Symbol(0),
                    HashMap::from([(TransitionKey::Symbol(1), TransitionKey::Symbol(1))]),
                ),
                (TransitionKey::Symbol(1), HashMap::new()),
            ]),
        };

        assert_eq!(
            result
                .alphabet
                .symbol_mapping
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            expected
                .alphabet
                .symbol_mapping
                .keys()
                .copied()
                .collect::<HashSet<_>>()
        );

        assert_eq!(
            result
                .alphabet
                .by_transition
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            expected
                .alphabet
                .by_transition
                .keys()
                .copied()
                .collect::<HashSet<_>>()
        );

        assert_eq!(
            result.states.iter().copied().collect::<HashSet<_>>(),
            expected.states.iter().copied().collect::<HashSet<_>>()
        );

        assert_eq!(result.initial, expected.initial);

        assert_eq!(
            result.finals.iter().copied().collect::<HashSet<_>>(),
            expected.finals.iter().copied().collect::<HashSet<_>>()
        );

        assert_eq!(
            result.map.keys().copied().collect::<HashSet<_>>(),
            expected.map.keys().copied().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn test_parse_pattern_two_chars_to_fsm() {
        let pattern: &str = "ab";
        let result = parse_pattern(pattern).unwrap();

        let alphabet = Alphabet {
            symbol_mapping: HashMap::from([
                ('\0', TransitionKey::Symbol(0)),
                ('a', TransitionKey::Symbol(1)),
                ('b', TransitionKey::Symbol(2)),
            ]),
            by_transition: HashMap::from([
                (TransitionKey::Symbol(0), vec!['\0']),
                (TransitionKey::Symbol(1), vec!['a']),
                (TransitionKey::Symbol(2), vec!['b']),
            ]),
        };

        let result = result.to_fsm(Some(alphabet.clone()), None, None);

        let expected = Fsm {
            alphabet,
            states: HashSet::from([
                TransitionKey::Symbol(0),
                TransitionKey::Symbol(1),
                TransitionKey::Symbol(2),
            ]),
            initial: TransitionKey::Symbol(0),
            finals: HashSet::from([TransitionKey::Symbol(2)]),
            map: HashMap::from([
                (
                    TransitionKey::Symbol(0),
                    HashMap::from([(TransitionKey::Symbol(1), TransitionKey::Symbol(1))]),
                ),
                (
                    TransitionKey::Symbol(1),
                    HashMap::from([(TransitionKey::Symbol(2), TransitionKey::Symbol(2))]),
                ),
                (TransitionKey::Symbol(2), HashMap::new()),
            ]),
        };

        assert_eq!(
            result
                .alphabet
                .symbol_mapping
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            expected
                .alphabet
                .symbol_mapping
                .keys()
                .copied()
                .collect::<HashSet<_>>()
        );

        assert_eq!(
            result
                .alphabet
                .by_transition
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            expected
                .alphabet
                .by_transition
                .keys()
                .copied()
                .collect::<HashSet<_>>()
        );

        assert_eq!(
            result.states.iter().copied().collect::<HashSet<_>>(),
            expected.states.iter().copied().collect::<HashSet<_>>()
        );

        assert_eq!(result.initial, expected.initial);

        assert_eq!(
            result.finals.iter().copied().collect::<HashSet<_>>(),
            expected.finals.iter().copied().collect::<HashSet<_>>()
        );

        assert_eq!(
            result.map.keys().copied().collect::<HashSet<_>>(),
            expected.map.keys().copied().collect::<HashSet<_>>()
        );
    }
}
