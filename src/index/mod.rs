mod compressed;
mod standard;
mod test_bench_index_variant;
mod test_bench_memory;
mod test_index_comparison;

use std::collections::hash_map::Keys;

use bincode::{Decode, Encode};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub use crate::index::compressed::CompressedIndex;
pub use crate::index::standard::Index;
use crate::prelude::*;
use crate::{Error, Result};

pub struct TokenIdIterator<'a> {
    inner: Keys<'a, TokenId, StateId>,
}

impl<'a> TokenIdIterator<'a> {
    pub fn new(keys: Keys<'a, TokenId, StateId>) -> Self {
        TokenIdIterator { inner: keys }
    }
}

impl<'a> Iterator for TokenIdIterator<'a> {
    type Item = &'a TokenId;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub trait IndexBehavior {
    fn initial_state(&self) -> StateId;
    fn final_states(&self) -> &HashSet<StateId>;
    fn is_final_state(&self, state: &StateId) -> bool;
    fn eos_token_id(&self) -> TokenId;
    fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>>;
    fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId>;
    fn vocab_size(&self) -> usize;
    fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>>;
    fn allowed_tokens_iter(&self, state: &StateId) -> Option<TokenIdIterator>;
    fn allowed_tokens_mask(&self, _state: &StateId) -> Option<&Vec<u64>>;
}
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub enum IndexVariant {
    Standard(Index),
    Compressed(CompressedIndex),
}

impl IndexVariant {
    pub fn create_index(regex: &str, vocabulary: &Vocabulary) -> Result<Self, Error> {
        let index = Index::new(regex, vocabulary)?;
        Ok(IndexVariant::Standard(index))
    }

    pub fn create_compressed(regex: &str, vocabulary: &Vocabulary) -> Result<Self, Error> {
        let index = Index::new(regex, vocabulary)?;
        let compressed = CompressedIndex::new(&index, vocabulary.eos_token_id());
        Ok(IndexVariant::Compressed(compressed))
    }

    pub fn initial_state(&self) -> StateId {
        match self {
            IndexVariant::Standard(index) => index.initial_state(),
            IndexVariant::Compressed(index) => index.initial_state(),
        }
    }

    pub fn final_states(&self) -> &HashSet<StateId> {
        match self {
            IndexVariant::Standard(index) => index.final_states(),
            IndexVariant::Compressed(index) => index.final_states(),
        }
    }

    pub fn is_final_state(&self, state: &StateId) -> bool {
        match self {
            IndexVariant::Standard(index) => index.is_final_state(state),
            IndexVariant::Compressed(index) => index.is_final_state(state),
        }
    }

    pub fn eos_token_id(&self) -> TokenId {
        match self {
            IndexVariant::Standard(index) => index.eos_token_id(),
            IndexVariant::Compressed(index) => index.eos_token_id(),
        }
    }

    pub fn transitions(&self) -> &HashMap<StateId, HashMap<TokenId, StateId>> {
        match self {
            IndexVariant::Standard(index) => index.transitions(),
            IndexVariant::Compressed(index) => index.transitions(),
        }
    }

    pub fn next_state(&self, state: &StateId, token_id: &TokenId) -> Option<StateId> {
        match self {
            IndexVariant::Standard(index) => index.next_state(state, token_id),
            IndexVariant::Compressed(index) => index.next_state(state, token_id),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            IndexVariant::Standard(index) => index.vocab_size(),
            IndexVariant::Compressed(index) => index.vocab_size(),
        }
    }

    pub fn allowed_tokens(&self, state: &StateId) -> Option<Vec<TokenId>> {
        match self {
            IndexVariant::Standard(index) => index.allowed_tokens(state),
            IndexVariant::Compressed(index) => index.allowed_tokens(state),
        }
    }

    pub fn allowed_tokens_iter(&self, state: &StateId) -> Option<TokenIdIterator> {
        match self {
            IndexVariant::Standard(index) => index.allowed_tokens_iter(state),
            IndexVariant::Compressed(index) => index.allowed_tokens_iter(state),
        }
    }

    pub fn allowed_tokens_mask(&self, state: &StateId) -> Option<&Vec<u64>> {
        match self {
            IndexVariant::Standard(index) => index.allowed_tokens_mask(state),
            IndexVariant::Compressed(index) => index.allowed_tokens_mask(state),
        }
    }
}

impl std::fmt::Display for IndexVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexVariant::Standard(index) => {
                writeln!(f, "Standard Index:")?;
                write!(f, "{}", index)
            }
            IndexVariant::Compressed(index) => {
                writeln!(f, "Compressed Index:")?;
                write!(f, "{}", index)
            }
        }
    }
}
