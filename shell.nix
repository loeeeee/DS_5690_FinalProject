{ pkgs ? import (builtins.fetchTarball "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz") {
    config.allowUnfree = true;
  }
}:

let
  rocmPkgs = pkgs.rocmPackages;
  pythonEnv = pkgs.python313.withPackages (ps:
    let
      t = ps.torchWithRocm.override { rocmPackages = rocmPkgs; };
    in
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
      (transformers.override { torch = t; })
      (accelerate.override { torch = t; })
      t
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
    export PYTORCH_HIP_ARCH=gfx1100
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export ROCM_PATH=${rocmPkgs.clr}
    echo "ROCm shell active (ROCm 6.4, gfx1100). Run: python - <<'PY' ... to verify torch."
  '';
}

