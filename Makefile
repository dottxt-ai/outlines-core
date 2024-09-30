# Optional target to test/benchmark.
TARGET ?=

.ONESHELL:
.PHONY: venv setup install install-release build-extension-debug build-extension-release watch-extension watch-extension-release pcc test test-rust test-python bench pybench doc dist clean check-clean-git
.SILENT:

# Create a fresh virtual environment with the latest pip.
venv:
	rm -rf .venv
	python -m venv .venv
	source .venv/bin/activate
	pip install -U pip

# Setup the active virtual environment for development.
setup:
	pip install -e .[test]
	pre-commit install
	cargo install --force cargo-watch
	cargo install --force cargo-run-script

# Build the latest changes in the rust bindings and install it to the active environment.
install:
	pip install -e .

# Build the latest changes in the rust bindings in release mode and install it to the active environment.
install-release:
	pip install .

# Build only the Rust Python extension (in debug mode)
build-extension-debug:
	python setup.py build_rust --inplace --debug

# Build only the Rust Python extension (in release mode)
build-extension-release:
	python setup.py build_rust --inplace --release

# Watches changes in the rust bindings and updates the python extension in place.
watch-extension:
	cargo watch -x 'run-script build-python-extension' -w src -w Cargo.toml

# Watches changes in the rust bindings in release mode and updates the python extension in place.
watch-extension-release:
	cargo watch -x 'run-script build-python-extension-release' -w src -w Cargo.toml

# Run pre-commit checks.
pcc:
	pre-commit run --all-files

test: test-python test-rust

# Run rust tests.
test-rust:
	cargo test "$(TARGET)"

# Run python tests.
test-python: build-extension-debug
	pytest -svv tests -k "$(TARGET)" \
		--cov=outlines_core \
		--cov-report=term-missing:skip-covered

# Run rust benchmarks.
bench:
ifeq ($(TARGET),)
	cargo bench
else
	cargo bench -- $(TARGET)
endif

# Run python benchmarks.
pybench: check-clean-git
ifeq ($(TARGET),)
	asv run --config benchmarks/asv.conf.json
else
	asv run --config benchmarks/asv.conf.json -b "$(TARGET)"
endif

# Build the documentation of the rust crate and open it.
doc:
	cargo doc --document-private-items --open

# Create wheels for distribution.
dist:
	pip install build
	python -m build

# Clean build and distribution files.
clean:
	cargo clean
	rm -rf dist

# Make sure that git diff is clean.
check-clean-git:
	git diff-index --quiet HEAD \
	|| (echo "Unable to perform the action due to uncommited local changes." && exit 1)
