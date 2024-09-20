use crate::*;

/// Vocabulary of an LLM.
///
/// ## Examples
///
/// ```rust
/// # use outlines_core::*;
/// #
/// let vocabulary = Vocabulary::new()
///     .insert(0, "blah")
///     .insert(1, "1a")
///     .insert(2, "2")
///     .insert(3, "0");
/// ```
#[derive(Clone, Debug, Default)]
pub struct Vocabulary(HashMap<Token, Vec<TokenId>>);

impl Vocabulary {
    /// Creates an empty vocabulary.
    pub fn new() -> Vocabulary {
        Vocabulary::default()
    }
}

impl Vocabulary {
    /// Inserts a token to the vocabulary with the specified identifier.
    pub fn insert(mut self, id: TokenId, token: impl Into<Token>) -> Vocabulary {
        let token = token.into();
        self.0.entry(token).or_default().push(id);
        self
    }

    /// Extends the vocabulary with tokens and their identifiers.
    pub fn extend<T: Into<Token>, I: IntoIterator<Item = TokenId>>(
        mut self,
        tokens_and_ids: impl IntoIterator<Item = (T, I)>,
    ) -> Vocabulary {
        for (token, ids) in tokens_and_ids.into_iter() {
            let token = token.into();
            for id in ids {
                self = self.insert(id, token.clone());
            }
        }
        self
    }
}

impl Deref for Vocabulary {
    type Target = HashMap<Token, Vec<TokenId>>;

    fn deref(&self) -> &HashMap<Token, Vec<TokenId>> {
        &self.0
    }
}

impl<T, I> FromIterator<(T, I)> for Vocabulary
where
    T: Into<Token>,
    I: IntoIterator<Item = TokenId>,
{
    fn from_iter<A: IntoIterator<Item = (T, I)>>(tokens_and_ids: A) -> Self {
        Vocabulary::new().extend(tokens_and_ids)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn insert() {
        let vocabulary = Vocabulary::new()
            .insert(0, "blah")
            .insert(1, "1a")
            .insert(2, "2")
            .insert(3, "0");

        assert_eq!(vocabulary.len(), 4);
        assert_eq!(vocabulary["blah"], &[0]);
        assert_eq!(vocabulary["1a"], &[1]);
        assert_eq!(vocabulary["2"], &[2]);
        assert_eq!(vocabulary["0"], &[3]);
    }

    #[test]
    fn extend() {
        let vocabulary = Vocabulary::new().extend([
            ("blah", vec![0]),
            ("1a", vec![1]),
            ("2", vec![2]),
            ("0", vec![3]),
        ]);

        assert_eq!(vocabulary.len(), 4);
        assert_eq!(vocabulary["blah"], &[0]);
        assert_eq!(vocabulary["1a"], &[1]);
        assert_eq!(vocabulary["2"], &[2]);
        assert_eq!(vocabulary["0"], &[3]);
    }
}
