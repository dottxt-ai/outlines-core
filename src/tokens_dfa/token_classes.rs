
use bincode::{Decode, Encode};
use regex_automata::util::alphabet::ByteClasses;
use crate::primitives::TokenId;

pub type TokenClassId = TokenId;


#[inline(always)]
pub fn from_token_to_token_class(token: &[u8], byte_classes: &ByteClasses) ->  TokenClass{
    let mut data = Vec::with_capacity(token.len());
    for &byte in token {
        data.push(byte_classes.get(byte));
    }
   TokenClass(data)
}

/// 'TokenClass' is a classification of a given Token based on the ByteClasses of a given Regex
#[derive(Clone, PartialEq, Eq, Hash, Debug, Encode, Decode)]
pub struct TokenClass(Vec<u8>);

impl TokenClass{
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        TokenClass(bytes)
    }

    
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    #[inline(always)]
    pub fn prefix(&self) -> u8 {
        self.0[0]
    }

    pub fn starts_with(&self, prefix: &TokenClass) -> bool {
        self.as_bytes().starts_with(prefix.as_bytes())
    }

    pub fn starts_with_byte(&self, prefix: u8)-> bool {
        self.as_bytes()[0] == prefix
    }
    
    pub fn add_byte(&mut self, b: u8){
        self.0.push(b);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl AsRef<[u8]> for TokenClass {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<&[u8]> for TokenClass {
    fn from(bytes: &[u8]) -> Self {
        TokenClass(bytes.to_vec())
    }
}

impl From<Vec<u8>> for TokenClass {
    fn from(bytes: Vec<u8>) -> Self {
        TokenClass(bytes)
    }
}

impl std::fmt::Display for TokenClass {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Convertir en String pour l'affichage (avec gestion des erreurs)
        match std::str::from_utf8(&self.0) {
            Ok(s) => write!(f, "{}", s),
            Err(_) => {
                // Fallback: affichage hexadécimal pour les données non-UTF8
                write!(f, "0x")?;
                for byte in &self.0 {
                    write!(f, "{:02X}", byte)?;
                }
                Ok(())
            }
        }
    }
}

