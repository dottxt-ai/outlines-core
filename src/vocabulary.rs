use crate::prelude::*;

/// Vocabulary of an LLM.
///
/// ## Examples
///
/// ```rust
/// # use outlines_core::prelude::*;
/// #
/// let vocabulary = Vocabulary::new()
///     .insert("blah", 0)
///     .insert("1a", 1)
///     .insert("2", 2)
///     .insert("0", 3);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Vocabulary(pub(crate) HashMap<Token, Vec<TokenId>>);

impl Vocabulary {
    /// Creates an empty vocabulary.
    pub fn new() -> Vocabulary {
        Vocabulary::default()
    }
}

impl Vocabulary {
    /// Inserts a token to the vocabulary with the specified identifier.
    pub fn insert(mut self, token: impl Into<Token>, id: TokenId) -> Vocabulary {
        self.insert_in_place(token, id);
        self
    }

    /// Extends the vocabulary with tokens and their identifiers.
    pub fn extend<T: Into<Token>, I: IntoIterator<Item = TokenId>>(
        mut self,
        tokens_and_ids: impl IntoIterator<Item = (T, I)>,
    ) -> Vocabulary {
        self.extend_in_place(tokens_and_ids);
        self
    }
}

impl Vocabulary {
    /// Inserts a token to the vocabulary with the specified identifier, in place.
    pub fn insert_in_place(&mut self, token: impl Into<Token>, id: TokenId) {
        let token = token.into();
        self.0.entry(token).or_default().push(id);
    }

    /// Extends the vocabulary with tokens and their identifiers, in place.
    pub fn extend_in_place<T: Into<Token>, I: IntoIterator<Item = TokenId>>(
        &mut self,
        tokens_and_ids: impl IntoIterator<Item = (T, I)>,
    ) {
        for (token, ids) in tokens_and_ids.into_iter() {
            let token = token.into();
            self.0.entry(token).or_default().extend(ids);
        }
    }
}

impl Deref for Vocabulary {
    type Target = HashMap<Token, Vec<TokenId>>;

    fn deref(&self) -> &HashMap<Token, Vec<TokenId>> {
        &self.0
    }
}

impl Display for Vocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (token, token_ids)) in self.iter().enumerate() {
            if index != (self.len() - 1) {
                writeln!(f, "{:?} -> {:?}", token, token_ids)?;
            } else {
                write!(f, "{:?} -> {:?}", token, token_ids)?;
            }
        }
        Ok(())
    }
}

impl From<HashMap<Token, Vec<TokenId>>> for Vocabulary {
    fn from(map: HashMap<Token, Vec<TokenId>>) -> Vocabulary {
        Vocabulary(map)
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
    use crate::prelude::*;

    #[test]
    fn insert() {
        let vocabulary = Vocabulary::new()
            .insert("blah", 0)
            .insert("1a", 1)
            .insert("2", 2)
            .insert("0", 3);

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
