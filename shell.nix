# If you plan to use `nix-shell` you have first to:
#
# $ nix-channel --add https://github.com/oxalica/rust-overlay/archive/master.tar.gz rust-overlay
# $ nix-channel --update
{ pkgs ? import <nixpkgs> { overlays = [ (import <rust-overlay>) ]; }
, system ? builtins.currentSystem }:
with pkgs;
let
    toolchain = (rust-bin.fromRustupToolchainFile ./rust-toolchain).override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" ];
    };
in mkShell {
    buildInputs = [ toolchain openssl pre-commit pkg-config python3 uv ];
}
