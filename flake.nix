{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad-src = {
      url = "github:tinygrad/tinygrad";
      flake = false;
    };
  };

  outputs =
    { self, nixpkgs, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        tinygrad-olay = self: super: {
          python3 = super.python3.override {
            packageOverrides = py-self: py-super: {
              tinygrad = py-super.tinygrad.overrideAttrs (oa: {
                version = "master";
                src = inputs.tinygrad-src;
                disabledTests = oa.disabledTests ++ ["test_valid_becomes_const1_z3"];
                nativeCheckInputs = oa.nativeCheckInputs ++ [py-super.z3];
              });
            };
          };
        };
        pkgs = import nixpkgs {
          inherit system;
          overlays = [tinygrad-olay];
          config = {
            allowUnfree = true;
          };
        };
        python-env = pkgs.python3.withPackages (
          ps: with ps; [
            black
            ipdb
            ipython
            matplotlib
            mypy
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
            cudatoolkit
            cudaPackages.cuda_cudart.dev
            python-env
            rocmPackages_6.clr
            rocmPackages_6.rocm-runtime
            rocmPackages_6.rocm-comgr
          ];
          nativeBuildInputs = buildInputs;
          CUDA_PATH=pkgs.cudatoolkit.out;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
        };
      }
    );
}
