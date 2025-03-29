<div align="center" style="margin-bottom: 1em;">

<img src="./docs/assets/images/logo.png" alt="Outlines-core Logo" width=500></img>

[![Latest Version]][crates.io] [![License]][github] ![MSRV][version]

[Latest Version]: https://img.shields.io/crates/v/outlines-core.svg
[crates.io]: https://crates.io/crates/outlines-core
[License]: https://img.shields.io/github/license/dottxt-ai/outlines-core.svg?color=blue&cachedrop
[github]: https://github.com/dottxt-ai/outlines-core/blob/main/LICENSE
[MSRV]: MSRV
[version]: https://img.shields.io/crates/msrv/outlines-core.svg?label=msrv&color=lightgrayy

*Structured generation (in Rust).*

</div>

## Outlines-core

This package provides the core functionality for structured generation, formerly implemented in [Outlines][outlines],
with a focus on performance and portability, it offers a convenient way to:

- build regular expressions from JSON schemas

- construct an `Index` object by combining a `Vocabulary` and regular expression to efficiently map tokens from a given vocabulary to state transitions in a finite-state automation

### Example

Basic example of how it all fits together.

```rust
use outlines_core::prelude::*;

// Define a JSON schema
let schema = r#"{
    "type": "object",
    "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" }
    },
    "required": ["name", "age"]
}"#;

// Generate a regular expression from it
let regex = json_schema::regex_from_str(&schema, None)?;

// Create `Vocabulary` from pretrained large language model (but manually is also possible)
let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", None)?;

// Create new `Index` from regex and a given `Vocabulary`
let index = Index::new(&regex, &vocabulary)?;

let initial_state = index.initial_state();
let allowed_tokens = index.allowed_tokens(&initial_state).expect("Some allowed token ids");
let token_id = allowed_tokens.first().expect("First token id");
let next_state = index.next_state(&initial_state, token_id);
let final_states = index.final_states();
```

## Python Bindings

Additionally, project provides interfaces to integrate the crate's functionality with Python.

``` python
import json

from outlines_core.json_schema import build_regex_from_schema
from outlines_core.guide import Guide, Index, Vocabulary

schema =  {
  "title": "Foo",
  "type": "object",
  "properties": {"date": {"type": "string", "format": "date"}}
}
regex = build_regex_from_schema(json.dumps(schema))

vocabulary = Vocabulary.from_pretrained("openai-community/gpt2")
index = Index(regex, vocabulary)
guide = Guide(index)

# Get current state of the Guide:
current_state = guide.get_state()

# Get allowed tokens for the current state of the Guide:
allowed_tokens = guide.get_tokens()

# Advance Guide to the next state via some token_id and return allowed tokens for that new state:
next_allowed_tokens = guide.advance(allowed_tokens[-1])

# To check if Guide is finished:
guide.is_finished()

# If it's finished then this assertion holds:
assert guide.get_tokens() == [vocabulary.get_eos_token_id()]
```

## How to contribute?

### Setup

Fork the repository on GitHub and clone the fork locally:

```bash
git clone git@github.com/YourUserName/outlines-core.git
cd outlines-core
```

Create a new virtual environment and install the dependencies in editable mode:

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pre-commit install
```

### Before pushing your code

If working with Python bindings don't forget to build Rust extension before testing, for example, in debug mode:

```bash
make build-extension-debug
```

Run Python tests:

``` bash
pytest
```

Run Rust tests:

``` bash
cargo test
```

Or alternatively using Makefile for both:

``` bash
make test
```

Finally, run the code style checks:

``` bash
pre-commit run --all-files
```

Or using Makefile:

``` bash
make pcc
```

If necessary you can run benchmarks locally:

``` bash
make pybench
```

## Join us

- 💡 **Have an idea?** Come chat with us on [Discord][discord]
-  **Found a bug?** Open an [issue](https://github.com/dottxt-ai/outlines-core/issues)

[outlines]: https://github.com/dottxt-ai/outlines
[discord]: https://discord.gg/R9DSu34mGd
