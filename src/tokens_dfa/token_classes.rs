use std::ops::{Index, IndexMut};
use bincode::{Decode, Encode};
use regex_automata::util::alphabet::ByteClasses;
use smallvec::SmallVec;
use rustc_hash::FxHashMap as HashMap;

use crate::primitives::TokenId;


pub type TokenClassId = TokenId;

/// 'TokenClass' is a classification of a given Token based on the ByteClasses of a given Regex
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TokenClass(SmallVec<[u8; 8]>);

/// The ClassId for each TokenId
/// TokenClasses\[TokenId\] = TokenClassId 
#[derive(Clone, Debug, PartialEq, Encode, Decode, Default)]
pub struct TokenClasses(HashMap<TokenId, TokenClassId>);

/// The TokenIds for each ClassId
/// Already sorted from the lowest TokenId to the highest
#[derive(Clone, Debug, PartialEq, Encode, Decode, Default)]
pub struct TokenIdsByClass(Vec<Vec<TokenId>>);

// Map TokenClasses with their respective IDs
pub struct MapTokenClassTokenClassId(HashMap<TokenClass, TokenClassId>);




impl TokenClass{
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        TokenClass(SmallVec::from_vec(bytes))
    }

    
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn starts_with(&self, prefix: &TokenClass) -> bool {
        self.as_bytes().starts_with(prefix.as_bytes())
    }
    
    pub fn from_token_with_classes(token: &[u8], byte_classes: &ByteClasses) -> Self {
        let mut data = SmallVec::with_capacity(token.len());
        for &byte in token {
            data.push(byte_classes.get(byte));
        }
        TokenClass(data)
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
        TokenClass(SmallVec::from_slice(bytes))
    }
}

impl From<Vec<u8>> for TokenClass {
    fn from(bytes: Vec<u8>) -> Self {
        TokenClass(SmallVec::from_vec(bytes))
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

impl TokenClasses {
    pub fn new() -> Self {
        TokenClasses(HashMap::default())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TokenId, &TokenClassId) > {
        self.0.iter()
    }

    pub fn size(&self) -> usize{
        // 1. Taille de la structure de base
        let mut total_size = size_of_val(&self.0);
        
        // 2. Calculer la taille du tableau de buckets
        // Note: ceci est une estimation approximative car nous n'avons pas d'accès
        // direct aux champs internes de HashMap dans la std
        let capacity = self.0.capacity();
        
        // Taille approximative d'un bucket (métadonnées + pointeur)
        let bucket_size = 16;
        total_size += capacity * bucket_size;
        
        // 3. Taille de chaque entrée
        // Chaque entrée contient une clé, une valeur et des métadonnées
        let entry_size = size_of::<u32>() + size_of::<u32>() + 24; // 4 + 4 + 24 = 32 octets environ
        total_size += self.len() * entry_size;
        
        total_size

    }


}

impl Index<u32> for TokenClasses {
    type Output = TokenClassId;
    
    fn index(&self, index: TokenId) -> &Self::Output {
        &self.0.get(&index).unwrap()
    }
}

impl IndexMut<u32> for TokenClasses {
    fn index_mut(&mut self, index: TokenId) -> &mut Self::Output {
        self.0.entry(index)
        .or_insert(0)
    }
}

impl MapTokenClassTokenClassId {

    pub fn new()-> Self {
        MapTokenClassTokenClassId(HashMap::default())
    }

    pub fn insert(&mut self, token_class: &TokenClass) -> TokenClassId {
        let new_id = self.0.len() as TokenClassId;
        *self.0.entry(token_class.clone()).or_insert(new_id)
    }

}

impl TokenIdsByClass {
  
    pub fn with_capacity(class_id_len:usize) -> Self {
        let mut classes = Vec::with_capacity(class_id_len);
        
        
        for _ in 0..class_id_len {
            classes.push(Vec::new());
        }
        
        TokenIdsByClass(classes)
    }

    pub fn get_token_ids(&self, class_id:TokenClassId) -> &Vec<TokenId> {
        &self.0[class_id as usize]
    }

    pub fn add_token_id(&mut self, class_id:TokenClassId, token_id:TokenId){
        if self.0.len() <= class_id as usize {
            for _ in 0..(class_id as usize +1  - self.0.len()){
                self.0.push(Vec::new());
            }
        }
        self.0[class_id as usize].push(token_id);
        //self.0[class_id as usize].sort();
    }
    
    pub fn _get_position(&self, class_id: TokenClassId, token_id: TokenId) -> Option<usize> {
        self.0[class_id as usize].iter().position(|id| *id == token_id)
    }

    pub fn _len(&self) -> usize {
        self.0.len()
    }
}