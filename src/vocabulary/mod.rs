use rustc_hash::FxHashMap as HashMap;

use tokenizers::normalizers::Sequence;
use tokenizers::{FromPretrainedParameters, NormalizerWrapper, Tokenizer};

use crate::prelude::*;
use crate::{Error, Result};

use locator::{HFLocator, Locator};
use processor::TokenProcessor;

mod locator;
mod processor;

/// Vocabulary of an LLM.
///
/// ## Examples
///
/// ```rust
/// # use outlines_core::prelude::*;
/// #
/// let vocabulary = Vocabulary::new(None)
///     .insert("blah", 0)
///     .insert("1a", 1)
///     .insert("2", 2)
///     .insert("0", 3);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Vocabulary {
    // TODO: Option is temp for back compatibility
    eos_token_id: Option<TokenId>,
    tokens: HashMap<Token, Vec<TokenId>>,
}

impl Vocabulary {
    /// Creates an empty vocabulary.
    pub fn new(eos_token_id: Option<TokenId>) -> Self {
        Self {
            eos_token_id,
            tokens: HashMap::default(),
        }
    }

    pub fn with_eos_token_id(self, eos_token_id: Option<TokenId>) -> Self {
        Self {
            eos_token_id,
            ..self
        }
    }

    /// Creates the vocabulary of pre-trained model from Hugging Face Hub.
    pub fn from_pretrained(
        model: &str,
        parameters: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        Self::from_pretrained_with_locator::<HFLocator>(model, parameters)
    }

    #[doc(hidden)]
    #[inline(always)]
    fn from_pretrained_with_locator<L: Locator>(
        model: &str,
        parameters: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_pretrained(model, parameters.clone())?;
        Self::filter_prepend_normalizers(&mut tokenizer);

        // Locate eos_token_id in defined locations.
        let eos_token_id = L::locate_eos_token_id(model, &tokenizer, &parameters);
        let Some(eos_token_id) = eos_token_id else {
            return Err(Error::UnsupportedTokenizer {
                model: model.to_string(),
                reason: "EOS token id".to_string(),
            });
        };

        // Start building the vocabulary from eos_token_id and added tokens.
        let mut vocabulary = Vocabulary::new(Some(eos_token_id));
        for (id, added_token) in tokenizer.get_added_tokens_decoder().iter() {
            if !added_token.special {
                vocabulary = vocabulary.insert(added_token.content.clone(), *id);
            }
        }

        // Process each vocabulary token according to the tokenizer's level.
        let Ok(processor) = TokenProcessor::new(&tokenizer) else {
            return Err(Error::UnsupportedTokenizer {
                model: model.to_string(),
                reason: "Token processor".to_string(),
            });
        };
        for (token, token_id) in tokenizer.get_vocab(false) {
            let processed_token = processor.process(token)?;
            vocabulary = vocabulary.insert(processed_token, token_id);
        }

        Ok(vocabulary)
    }

    /// Returns all tokens with their token ids in vocabulary
    pub fn tokens_to_ids(&self) -> &HashMap<Token, Vec<TokenId>> {
        &self.tokens
    }

    /// Per provided token returns vector of `TokenId`s if available in the vocabulary.
    pub fn token_to_ids(&self, token: &Token) -> Option<&Vec<TokenId>> {
        self.tokens.get(token)
    }

    /// Gets the identifier of the special end of the sentence token.
    pub fn eos_token_id(&self) -> Option<TokenId> {
        self.eos_token_id
    }

    /// Filters out `Prepend` kind of tokenizer's normalizers.
    fn filter_prepend_normalizers(tokenizer: &mut Tokenizer) {
        // Main concern is prepend normalizers, for example https://github.com/google/sentencepiece
        // In `sentencepiece` tokenizer, `▁` is used to denote spaces in the source text,
        // e.g. `Hello World.` could be tokenized as: [Hello] [▁Wor] [ld] [.]
        //
        // We don't want to deal with the special characters, so we remove `Prepend` normalizers.
        if let Some(normalizer) = tokenizer.get_normalizer() {
            match normalizer {
                NormalizerWrapper::Sequence(normalization_sequence) => {
                    let new_sequence = Sequence::new(
                        normalization_sequence
                            .get_normalizers()
                            .iter()
                            .filter_map(|normalizer| match normalizer {
                                NormalizerWrapper::Prepend(_) => None,
                                _ => Some(normalizer.clone()),
                            })
                            .collect(),
                    );
                    tokenizer.with_normalizer(new_sequence.into());
                }
                NormalizerWrapper::Prepend(_) => {
                    tokenizer.with_normalizer(None::<NormalizerWrapper>);
                }
                _ => {}
            }
        }
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
        // TODO: return error if eos token id is inserted
        let token = token.into();
        self.tokens.entry(token).or_default().push(id);
    }

    /// Extends the vocabulary with tokens and their identifiers, in place.
    pub fn extend_in_place<T: Into<Token>, I: IntoIterator<Item = TokenId>>(
        &mut self,
        tokens_and_ids: impl IntoIterator<Item = (T, I)>,
    ) {
        for (token, ids) in tokens_and_ids.into_iter() {
            let token = token.into();
            self.tokens.entry(token).or_default().extend(ids);
        }
    }
}

impl std::ops::Deref for Vocabulary {
    type Target = HashMap<Token, Vec<TokenId>>;

    fn deref(&self) -> &HashMap<Token, Vec<TokenId>> {
        &self.tokens
    }
}

impl std::fmt::Display for Vocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    fn from(tokens: HashMap<Token, Vec<TokenId>>) -> Vocabulary {
        Vocabulary {
            eos_token_id: None,
            tokens,
        }
    }
}

impl From<HashMap<String, Vec<TokenId>>> for Vocabulary {
    fn from(tokens: HashMap<String, Vec<TokenId>>) -> Vocabulary {
        Vocabulary {
            eos_token_id: None,
            tokens: tokens
                .into_iter()
                .map(|(k, v)| (k.as_bytes().to_vec(), v))
                .collect::<HashMap<Token, Vec<TokenId>>>(),
        }
    }
}

impl<T, I> FromIterator<(T, I)> for Vocabulary
where
    T: Into<Token>,
    I: IntoIterator<Item = TokenId>,
{
    fn from_iter<A: IntoIterator<Item = (T, I)>>(tokens_and_ids: A) -> Self {
        Vocabulary::new(None).extend(tokens_and_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

    #[test]
    fn insert() {
        let vocabulary = Vocabulary::new(None)
            .insert("blah", 0)
            .insert("1a", 1)
            .insert("2", 2)
            .insert("0", 3);

        assert_eq!(vocabulary.len(), 4);
        assert_eq!(vocabulary["blah".as_bytes()], &[0]);
        assert_eq!(vocabulary["1a".as_bytes()], &[1]);
        assert_eq!(vocabulary["2".as_bytes()], &[2]);
        assert_eq!(vocabulary["0".as_bytes()], &[3]);
    }

    #[test]
    fn extend() {
        let vocabulary = Vocabulary::new(None).extend([
            ("blah", vec![0]),
            ("1a", vec![1]),
            ("2", vec![2]),
            ("0", vec![3]),
        ]);

        assert_eq!(vocabulary.len(), 4);
        assert_eq!(vocabulary["blah".as_bytes()], &[0]);
        assert_eq!(vocabulary["1a".as_bytes()], &[1]);
        assert_eq!(vocabulary["2".as_bytes()], &[2]);
        assert_eq!(vocabulary["0".as_bytes()], &[3]);
    }

    #[test]
    fn new_empty_vocabulary() {
        let vocabulary = Vocabulary::new(None);
        assert!(vocabulary.eos_token_id.is_none());
        assert!(vocabulary.tokens.is_empty());
    }

    #[test]
    fn new_empty_vocabulary_from_hashmap() {
        let map: HashMap<Token, Vec<TokenId>> = HashMap::default();
        let vocabulary = Vocabulary::from(map);
        assert!(vocabulary.eos_token_id.is_none());
        assert!(vocabulary.tokens.is_empty());
    }

    #[test]
    fn new_vocabulary_from_iterator() {
        let token: Token = "abc".as_bytes().to_vec();
        let id: Vec<TokenId> = vec![1];
        let it = vec![(token, id)];
        let vocabulary = Vocabulary::from_iter(it);
        assert!(vocabulary.eos_token_id.is_none());
        assert!(!vocabulary.tokens.is_empty());
    }

    #[test]
    fn supported_pretrained_models() {
        // Support is expected for these:
        for model in [
            // GPT 2
            "openai-community/gpt2",
            // Llama 2
            "hf-internal-testing/Llama-2-7B-GPTQ",
            // Llama 3
            // OpenCoder: shares llama tokenizers
            "hf-internal-testing/llama-3-8b-internal",
            // Qwen
            "Qwen/Qwen2-7B-Instruct",
            // Salamandra
            "BSC-LT/salamandra-2b",
        ] {
            let vocabulary = Vocabulary::from_pretrained(model, None);
            match vocabulary {
                Ok(v) => {
                    assert!(v.eos_token_id().is_some());
                    assert_eq!(v.eos_token_id, v.eos_token_id());
                    assert!(!v.tokens.is_empty());
                }
                Err(_) => unreachable!(),
            }
        }
    }

    #[test]
    fn pretrained_from_gpt2() {
        let model = "openai-community/gpt2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let vocabulary = Vocabulary::from_pretrained(model, None).expect("Vocabulary failed");

        let v_eos = vocabulary.eos_token_id;
        assert_eq!(v_eos, vocabulary.eos_token_id());
        assert!(v_eos.is_some());

        let v_eos = v_eos.unwrap();
        assert_eq!(v_eos, 50256);
        assert_eq!(
            tokenizer.id_to_token(v_eos).expect("Token not found"),
            "<|endoftext|>"
        );

        let token = "Ġal";
        let btoken = token.as_bytes().to_vec();
        assert!(vocabulary.token_to_ids(&btoken).is_none());
        assert!(tokenizer.token_to_id(token).is_some());

        for (v_token, t_token_expected) in [("abc", "abc"), (" O", "ĠO")] {
            let v_ids = vocabulary.token_to_ids(&v_token.as_bytes().to_vec());
            assert!(v_ids.is_some());
            for v_id in v_ids.unwrap() {
                let t_token = tokenizer
                    .id_to_token(*v_id)
                    .expect("Token id not found in tokenizer");
                assert_eq!(&t_token, t_token_expected);
            }
        }
    }

    #[test]
    fn pretrained_from_llama() {
        let model = "hf-internal-testing/llama-tokenizer";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let vocabulary = Vocabulary::from_pretrained(model, None).expect("Vocabulary failed");

        let v_eos = vocabulary.eos_token_id;
        assert_eq!(v_eos, vocabulary.eos_token_id());
        assert!(v_eos.is_some());

        let v_eos = v_eos.unwrap();
        assert_eq!(v_eos, 2);
        assert_eq!(
            tokenizer.id_to_token(v_eos).expect("Token not found"),
            "</s>"
        );

        let tests: &[(Vec<u8>, &[&str])] = &[
            ("abc".as_bytes().to_vec(), &["abc"]),
            (" al".as_bytes().to_vec(), &["▁al"]),
            (" O".as_bytes().to_vec(), &["▁O"]),
            ("   ".as_bytes().to_vec(), &["▁▁▁"]),
            (" ".as_bytes().to_vec(), &["▁", "<0x20>"]),
            ("a".as_bytes().to_vec(), &["a", "<0x61>"]),
            (vec![0xFF], &["<0xFF>"]),
            (vec![0x20], &["▁", "<0x20>"]),
        ];
        for (v_token, t_tokens_expected) in tests {
            let v_ids = vocabulary.token_to_ids(v_token);
            assert!(v_ids.is_some());

            let t_tokens = v_ids
                .unwrap()
                .iter()
                .map(|v_id| {
                    tokenizer
                        .id_to_token(*v_id)
                        .expect("Token id not found in tokenizer")
                })
                .collect::<HashSet<String>>();
            let expected = HashSet::from_iter(t_tokens_expected.iter().map(|s| s.to_string()));
            assert_eq!(t_tokens, expected)
        }
    }

    #[test]
    fn token_processor_error() {
        let model = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM";
        let vocabulary = Vocabulary::from_pretrained(model, None);

        match vocabulary {
            Err(Error::UnsupportedTokenizer { model, reason }) => {
                assert_eq!(model, model.to_string());
                assert_eq!(&reason, "Token processor");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn tokenizer_error() {
        let model = "hf-internal-testing/some-non-existent-model";
        let vocabulary = Vocabulary::from_pretrained(model, None);

        match vocabulary {
            Err(Error::TokenizersError(e)) => assert!(!e.to_string().is_empty()),
            _ => unreachable!(),
        }
    }

    struct NoneLocator;
    impl Locator for NoneLocator {
        fn locate_eos_token_id(
            _model: &str,
            _tokenizer: &Tokenizer,
            _parameters: &Option<FromPretrainedParameters>,
        ) -> Option<TokenId> {
            None
        }
    }

    #[test]
    fn unable_to_locate_eos_token_id_error() {
        let model = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM";
        let vocabulary = Vocabulary::from_pretrained_with_locator::<NoneLocator>(model, None);

        match vocabulary {
            Err(Error::UnsupportedTokenizer { model, reason }) => {
                assert_eq!(model, model.to_string());
                assert_eq!(&reason, "EOS token id");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn prepend_normalizers_filtered_out() {
        use tokenizers::normalizers::{Prepend, Sequence};

        let prepend = Prepend::new("_".to_string());
        let prepend_normalizer = NormalizerWrapper::Prepend(prepend);
        let sequence = Sequence::new(vec![prepend_normalizer.clone()]);
        let sequence_normalizer = NormalizerWrapper::Sequence(sequence);

        let model = "hf-internal-testing/llama-tokenizer";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");

        for normalizer in [prepend_normalizer, sequence_normalizer] {
            let mut normalized_t = tokenizer.clone();
            normalized_t.with_normalizer(Some(normalizer));
            Vocabulary::filter_prepend_normalizers(&mut normalized_t);
            if let Some(n) = normalized_t.get_normalizer() {
                match n {
                    NormalizerWrapper::Sequence(seq) => {
                        for n in seq.get_normalizers() {
                            if let NormalizerWrapper::Prepend(_) = n {
                                unreachable!()
                            }
                        }
                    }
                    NormalizerWrapper::Prepend(_) => unreachable!(),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn other_normalizers_being_kept() {
        use tokenizers::normalizers::BertNormalizer;

        let model = "hf-internal-testing/llama-tokenizer";
        let normalizer = NormalizerWrapper::BertNormalizer(BertNormalizer::default());
        let mut tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        tokenizer.with_normalizer(Some(normalizer));

        Vocabulary::filter_prepend_normalizers(&mut tokenizer);

        assert!(tokenizer.get_normalizer().is_some());
    }
}
