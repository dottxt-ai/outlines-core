## Overview

Release to PyPI and crates.io (`publish` workflow) is triggered automatically when a GitHub release is created.

Python and Rust depend on package version in `Cargo.toml` which is dynamically updated to match release tag.

### Python

`publish` workflow relies on `build_wheels` reusable workflow with `auto_bump` flag enabled.
It then activates `bump_version` custom action to ensure that version in `Cargo.toml` was automatically updated to match the release tag before building the wheels with `maturin` backend.
And `maturin` propagates this version to the wheels.

### Rust

The same `bump_version` action is used to update the version for building and publishing Rust.

### Dry-run

Before cutting a release, make sure that the `dry_run_publish` workflow on `main` was finished successfully.

It uses `build_wheels` reusable workflow without `auto_bump` flag to build wheels (thus wheels produced with 0.0.0 version) and doesn't use `bump_version` action for Rust's publishing dry-run.

### Notes

Ensure that the required secrets: PYPI_SECRET, CARGO_REGISTRY_TOKEN are set up.

If something went wrong `publish` workflow could be triggered manually via `workflow_dispatch` with a specified release tag.
