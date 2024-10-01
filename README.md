<div align="center" style="margin-bottom: 1em;">

<img src="./docs/assets/images/logo.png" alt="Outlines-core Logo" width=500></img>

[![Contributors][contributors-badge]][contributors]

*Structured generation (in Rust).*
</div>

This package provides the core functionality for structured generation, formerly implemented in [Outlines][outlines], with a focus on performance and portability.

# Install

We provide bindings to the following languages:
- [Rust][rust-implementation] (Original implementation)
- [Python][python-bindings]

The latest release of the Python bindings is available on PyPi using `pip`:

``` python
pip install outlines-core
```

The current development branch of `outlines-core` can be installed from GitHub, also using `pip`:

``` shell
pip install git+https://github.com/outlines-dev/outlines-core
```

Or install in a rust project with cargo:
``` bash
cargo add outlines-core
```

# How to contribute?

## Setup

First, fork the repository on GitHub and clone the fork locally:

```bash
git clone git@github.com/YourUserName/outlines-core.git
cd outlines-core
```

Create a new virtual environment:

``` bash
python -m venv .venv
source .venv/bin/activate
```

Then install the dependencies in editable mode, and install the pre-commit hooks:

``` bash
pip install -e ".[test]"
pre-commit install
```

## Before pushing your code

Run the tests:


``` bash
pytest
```

And run the code style checks:

``` bash
pre-commit run --all-files
```


[outlines]: https://github.com/dottxt-ai/outlines
[contributors]: https://github.com/outlines-dev/outlines-core/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/outlines-dev/outlines-core?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[rust-implementation]: https://github.com/outlines-dev/outlines-core/tree/readme/src
[python-bindings]: https://github.com/outlines-dev/outlines-core/tree/readme/python/outlines_core
