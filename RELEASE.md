# How to release

## Python
The python package uses the setuptools-scm plugin to automatically determine the version from git tags. When a release is created, it checks the tag and the CI automatically builds and publishes the package to PyPI.

No internvention should be required.

## Rust
The rust crate is similarly pushed through a Github Action that triggers on a release. But the version is determined by the Cargo.toml file, which has to be updated manually. Generally, the version should be the same as the python package but this isn't a strict requirement.

Currently we fail the rust release if the version in Cargo.toml doesn't match the tag. If that happens, just manually update the Cargo.toml version to match the tag. Push a new commit and rerun the job.
