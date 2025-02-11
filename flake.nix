# How to setup this project?
#
# 1. Install Nix:
#
#     curl -sSf -L https://install.lix.systems/lix | sh -s -- install
#
# 2. Enter the reproducible developer shell:
#
#     nix develop
#
# 3. Sync your Python venv with uv.lock file:
#
#     uv sync
#
# Everything's good! You can also install http://direnv.net/ and enjoy automagic
# activation of your environment ;)
{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { flake-utils, nixpkgs, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
      in {
        devShell = import ./shell.nix { inherit pkgs system; };
      });
}
