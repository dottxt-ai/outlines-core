import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


rust_extensions = [
    RustExtension(
        "outlines_core",
        f"{CURRENT_DIR}/Cargo.toml",
        binding=Binding.PyO3,
        rustc_flags=["--crate-type=cdylib"],
    ),
]

setup(
    rust_extensions=rust_extensions,
    package_dir={"": f"{CURRENT_DIR}/py_src/outlines_core"},
)
