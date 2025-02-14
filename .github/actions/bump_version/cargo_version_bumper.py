# From https://github.com/Intreecom/scyllapy/blob/05fdab32dd7468c26533de5fdfe9627fa3e38445/scripts/version_bumper.py

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        "-t",
        dest="target",
        type=Path,
        default="Cargo.toml",
    )
    parser.add_argument("version", type=str)
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    with args.target.open("r") as f:
        contents = f.read()

    contents = re.sub(
        r"version\s*=\s*\"(.*)\"",
        f'version = "{args.version}"',
        contents,
        count=1,
    )

    with args.target.open("w") as f:
        f.write(contents)


if __name__ == "__main__":
    main()
