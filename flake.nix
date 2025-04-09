{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python-env = pkgs.python3.withPackages (
          ps: with ps; [
            ipdb
            ipython
            polars
            python-lsp-server
            datasets
            tiktoken
            tinygrad
            tokenizers
            tqdm
          ]
        );
      in
      {
        devShell = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            python-env
            rocmPackages_6.clr
            rocmPackages_6.rocm-runtime
            rocmPackages_6.rocm-comgr
          ];
          nativeBuildInputs = buildInputs;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        };
      }
    );
}
