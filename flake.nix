{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };
  outputs = { flake-utils, nixpkgs, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.pre-commit
            pkgs.python3
            pkgs.rustup
          ];
          shellHook = ''
            FLAKE_DIR=$(dirname $(dirname $out))
            source $FLAKE_DIR/.venv/bin/activate
          '';
        };
      });
}
