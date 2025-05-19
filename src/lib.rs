//! # Outlines_core
//!
//! `outlines_core` crate provides a convenient way to:
//!
//! - build regular expressions from JSON schemas
//!
//! - construct an [`index::Index`] object by combining a [`vocabulary::Vocabulary`] and regular
//!   expression to efficiently map tokens from a given `Vocabulary` to state transitions in a
//!   finite-state automation
//!
//! ## `json_schema`
//!
//! [`json_schema`] module provides interfaces to generate a regular expression based on a given JSON schema, depending on its type:
//! - [`json_schema::regex_from_str`]
//! - [`json_schema::regex_from_value`]
//!
//! Whitespace pattern could be customized, otherwise the default [`json_schema::WHITESPACE`] pattern is used.
//!
//! Note, that not all the features of JSON schema are supported for regex generation: [Supported Features](json_schema#supported-features)
//!
//! ## `Index`
//!
//! Once [`index::Index`] is built, it can be used to evaluate or validate token sequences.
//!
//! ### Complexity and construction cost
//!
//! `Index` can accommodate large vocabularies and complex regular expressions. However, its size **may** grow
//! significantly with the complexity of the input, as well as time and computational resources.
//!
//! ## Python bindings
//!
//! Additionally, crate provides interfaces to integrate the crate's functionality with Python.
//!
//! ## Support
//!
//! `Outlines_core` is primarily used in structured text generation project [`outlines`](https://github.com/dottxt-ai/outlines),
//! if you need support, consider reaching out to its maintainers, you can also open an issue or start a discussion
//! on [github](https://github.com/dottxt-ai/outlines-core)
//!
//! ## Example
//!
//! Basic example of how it all fits together.
//!
//! ```rust
//! # use outlines_core::Error;
//! use outlines_core::prelude::*;
//!
//! # fn main() -> Result<(), Error> {
//! // Define a JSON schema
//! let schema = r#"{
//!     "type": "object",
//!     "properties": {
//!         "name": { "type": "string" },
//!         "age": { "type": "integer" }
//!     },
//!     "required": ["name", "age"]
//! }"#;
//!
//! // Generate a regular expression from it
//! let regex = json_schema::regex_from_str(&schema, None, None)?;
//! println!("Generated regex: {}", regex);
//!
//! // Create `Vocabulary` from pretrained large language model (but manually is also possible)
//! let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", None)?;
//!
//! // Create new `Index` from regex and a given `Vocabulary`
//! let index = Index::new(&regex, &vocabulary)?;
//!
//! let initial_state = index.initial_state();
//! println!("Is initial state {} a final state? {}", initial_state, index.is_final_state(&initial_state));
//!
//! let allowed_tokens = index.allowed_tokens(&initial_state).expect("Some allowed tokens");
//! println!("Allowed tokens at initial state are {:?}", allowed_tokens);
//!
//! let token_id = allowed_tokens.first().expect("First token");
//! println!("Next state for the token_id {} is {:?}", token_id, index.next_state(&initial_state, token_id));
//! println!("Final states are {:?}", index.final_states());
//! println!("Index has exactly {} transitions", index.transitions().len());
//! # Ok(())
//! }
//! ```

pub mod error;
pub mod index;
pub mod json_schema;
pub mod prelude;
pub mod primitives;
pub mod vocabulary;

pub use error::{Error, Result};

#[cfg(feature = "python-bindings")]
mod python_bindings;
