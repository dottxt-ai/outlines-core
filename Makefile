# Optional target to test/benchmark.
TARGET ?=

.ONESHELL:
.PHONY:
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

# Build the latest changes in the rust bindings and install it to the active environment.
install:
	pip install -e .

# Run pre-commit checks.
pcc:
	pre-commit run --all-files

# Run rust tests.
test:
	cargo test "$(TARGET)"

# Run python tests.
pytest: install
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

# Build the documentation of the python package and open it.
pydoc:
	echo "Unable to perform the action as it's not implemented yet."

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
