{ pkgs ? import (builtins.fetchTarball "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz") {
    config.allowUnfree = true;
  }
}:

let
  # Current nixpkgs has ROCm 6.4.3 (verified: rocmPackages.clr.version = "6.4.3")
  rocmPkgs = pkgs.rocmPackages;
  
  pythonEnv = pkgs.python312.withPackages (ps:
    with ps; [
      pip
      pyyaml
      datasets
      evaluate
      requests
      # Jupyter
      jupyter
      ipykernel
      # Visualization
      matplotlib
      seaborn
      # Observability
      tqdm
      # ML / data
      numpy
      pandas
      scikit-learn
      sentencepiece
      tokenizers
      safetensors
      huggingface-hub
      (transformers.override { torch = ps.torchWithRocm; })
      (accelerate.override { torch = ps.torchWithRocm; })
      ps.torchWithRocm
    ]);
in
pkgs.mkShell {
  packages = [
    pythonEnv
    rocmPkgs.rocm-smi
    rocmPkgs.rocminfo
    pkgs.git
    pkgs.which
  ];

  shellHook = ''
    # export PYTORCH_HIP_ARCH=gfx1100
    # export HSA_OVERRIDE_GFX_VERSION=11.0.0
    # export ROCM_PATH=${rocmPkgs.clr}
    export OPENBLAS_NUM_THREADS=64
    if [ -f .env ]; then
      set -a
      # shellcheck disable=SC1091
      source .env
      set +a
      echo ".env loaded into environment."
    else
      echo "No .env found; skipping environment variable load."
    fi
    
    echo "ROCm shell active (ROCm 6.4.3, gfx1100, torch 2.6). Run: python - <<'PY' ... to verify torch."
  '';
}
