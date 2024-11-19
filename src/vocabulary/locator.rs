use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::{Deserialize, Serialize};
use tokenizers::{FromPretrainedParameters, Tokenizer};

use crate::primitives::*;

/// Mapping of characters to bytes for GPT-2 like tokenizers.
/// List of common eos token locations appearing on hugging face hub, ordered by priority.
const COMMON_LOCATIONS: &[EosTokenLocation] = &[
    // Most projects have `generation_config.json` that looks like:
    // {
    //   ...
    //   "eos_token_id": 50256,
    //   ...
    // }
    // So it's the first place we look for the eos token id.
    //
    // For example:
    // - https://huggingface.co/openai-community/gpt2/blob/main/generation_config.json
    EosTokenLocation {
        file: "generation_config.json",
        location: EosTokenField::Id,
    },
    // The ones that don't have `generation_config.json` usually have `tokenizer_config.json`:
    // {
    //   ...
    //   "eos_token": "<|endoftext|>",
    //   ...
    // }
    // Once we have the eos token content, we can get its id from the tokenizer.
    //
    // For example:
    // - https://huggingface.co/microsoft/phi-2/blob/main/tokenizer_config.json
    EosTokenLocation {
        file: "tokenizer_config.json",
        location: EosTokenField::Value,
    },
    // Sometimes `tokenizer_config.json` can have the following format as well:
    // {
    //  "eos_token": {
    //     ...
    //     "content": "</s>",
    //     ...
    //   },
    // }
    // Once we have the eos token content, we can get its id from the tokenizer.
    //
    // For example:
    // - https://huggingface.co/hf-internal-testing/llama-tokenizer/blob/main/tokenizer_config.json
    EosTokenLocation {
        file: "tokenizer_config.json",
        location: EosTokenField::Object,
    },
];

/// `Id` kind of `EosTokenField`, when `eos_token_id` provided as an id.
#[derive(Debug, Serialize, Deserialize)]
struct Id {
    eos_token_id: u64,
}

/// `Value` kind of `EosTokenField`, when `eos_token` provided as a text, so that its id
/// will be fetched from the tokenizer.
#[derive(Debug, Serialize, Deserialize)]
struct Value {
    eos_token: String,
}

/// `Object` kind of `EosTokenField`, when `eos_token` provided as a `Content`.
#[derive(Debug, Serialize, Deserialize)]
struct Object {
    eos_token: Content,
}

/// `eos_token` provided in a `Content`.
#[derive(Debug, Serialize, Deserialize)]
struct Content {
    content: String,
}

/// Specifies in which part in config's json to check for eos token id.
enum EosTokenField {
    Id,
    Value,
    Object,
}

/// Defines location of the end of sentence token id in the config file.
struct EosTokenLocation {
    file: &'static str,
    location: EosTokenField,
}

/// Locates eos token id.
pub(crate) trait Locator {
    /// Locates eos token id in defined locations by `Locator`.
    fn locate_eos_token_id(
        model: &str,
        tokenizer: &Tokenizer,
        parameters: &Option<FromPretrainedParameters>,
    ) -> Option<TokenId>;
}

/// Locates eos token id by searching in defined common locations in hugging face.
pub(crate) struct HFLocator;

impl Locator for HFLocator {
    /// Locates eos token id in defined locations.
    fn locate_eos_token_id(
        model: &str,
        tokenizer: &Tokenizer,
        parameters: &Option<FromPretrainedParameters>,
    ) -> Option<TokenId> {
        COMMON_LOCATIONS
            .iter()
            .find_map(|location| location.lookup(model, tokenizer, parameters))
    }
}

impl EosTokenLocation {
    /// Finds eos token within defined location in a related config file.
    fn lookup(
        &self,
        model: &str,
        tokenizer: &Tokenizer,
        parameters: &Option<FromPretrainedParameters>,
    ) -> Option<TokenId> {
        let file_path = Self::download_config(model, self.file, parameters).ok()?;
        let file = std::fs::File::open(file_path).ok()?;

        match self.location {
            EosTokenField::Id => {
                let config: Id = serde_json::from_reader(file).ok()?;
                u32::try_from(config.eos_token_id).ok()
            }
            EosTokenField::Value => {
                let config: Value = serde_json::from_reader(file).ok()?;
                tokenizer.token_to_id(&config.eos_token)
            }
            EosTokenField::Object => {
                let config: Object = serde_json::from_reader(file).ok()?;
                tokenizer.token_to_id(&config.eos_token.content)
            }
        }
    }

    /// Downloads related config file from Hugging Face Hub.
    fn download_config(
        project: &str,
        file: &str,
        parameters: &Option<FromPretrainedParameters>,
    ) -> tokenizers::Result<std::path::PathBuf> {
        // Adapted from
        // https://github.com/huggingface/tokenizers/blob/9b77c054ef4297c7057fa8db875368c7c02f1bfc/tokenizers/src/utils/from_pretrained.rs#L26

        let params = parameters.clone().unwrap_or_default();

        // Validation checks are coming as a literal adaptation logic from HF.
        // In this case project is a model name, which if invalid expected to fail much earlier.
        // So it seems a bit redundant to validate it this way, but no harm in doing so too.
        Self::validate(project)?;
        Self::validate(&params.revision)?;

        let repo = Repo::with_revision(project.to_string(), RepoType::Model, params.revision);
        let api = ApiBuilder::new()
            .with_token(params.token)
            .build()?
            .repo(repo);

        Ok(api.get(file)?)
    }

    fn validate(input: &str) -> tokenizers::Result<()> {
        let valid_chars = ['-', '_', '.', '/'];

        if !input
            .chars()
            .all(|c: char| c.is_alphanumeric() || valid_chars.contains(&c))
        {
            return Err(format!(
                "Input {input} contains invalid characters, expected only alphanumeric or {}",
                valid_chars
                    .iter()
                    .map(|x| format!("'{}'", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_locations() {
        for (model, expected_token_id, expected_token) in &[
            ("openai-community/gpt2", 50256, "<|endoftext|>"),
            ("microsoft/phi-2", 50256, "<|endoftext|>"),
            ("hf-internal-testing/llama-tokenizer", 2, "</s>"),
        ] {
            let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
            let located = HFLocator::locate_eos_token_id(model, &tokenizer, &None)
                .expect("Token id is not located");

            assert_eq!(located, *expected_token_id);
            assert_eq!(
                tokenizer.id_to_token(located).expect("Token is not found"),
                expected_token.to_string()
            );
        }
    }

    #[test]
    fn bad_location() {
        let bad_location = EosTokenLocation {
            file: "tokenizer_config.json",
            location: EosTokenField::Id,
        };
        let model = "microsoft/phi-2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");

        let token_id = bad_location.lookup(model, &tokenizer, &None);
        assert!(token_id.is_none());

        let bad_file = EosTokenLocation {
            file: "generation_config.json",
            location: EosTokenField::Value,
        };
        let token_id = bad_file.lookup(model, &tokenizer, &None);
        assert!(token_id.is_none());
    }

    #[test]
    fn validate_config_input() {
        let input = "bad_model_name*";
        assert!(EosTokenLocation::validate(input).is_err());
    }
}
