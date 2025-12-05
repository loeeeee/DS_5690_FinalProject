Intent (user words)
- “setup the local environment with nix-shell.nix to make sure the scripts in this project can run on my ROCm enabled RX7900XT.”

Rephrase and scope
- Create a reproducible Nix shell targeting Python 3.13 with ROCm 6.4 for an RX 7900 XT so PyTorch (and torchvision/torchaudio) run on ROCm.
- Keep dependencies aligned with existing requirements (transformers, accelerate, pandas, matplotlib, tqdm) and add tooling (jupyter, seaborn, numpy, scikit-learn, spacy + en_core_web_sm).
- Pin to nixos-unstable to access ROCm 6.4 packages.

Environment decisions
- Python: 3.13 (per .clinerules).
- GPU stack: ROCm 6.4 (user request) with torchWithRocm for compatibility with RX 7900 XT.
- Layout: add nix-shell.nix at repo root; prefer nix-shell entrypoint for local runs; keep SLURM CUDA flow unchanged.
- Verification: include a short torch ROCm check (device count, is_available, torch.version.hip).

Planned outputs for this task
- New nix-shell.nix exposing python environment with ROCm PyTorch ecosystem and requested packages.
- README update documenting nix-shell usage and ROCm verification snippet.
- Post-task note here with exact commands to enter the shell and validate torch ROCm on RX 7900 XT.

Post-task status and usage
- Added nix-shell.nix (nixos-unstable, ROCm 6.4, python313 torchWithRocm) with gfx1100 tuning for RX 7900 XT.
- Enter shell: `nix-shell nix-shell.nix`
- Verify torch ROCm:
  - `python - <<'PY'`
  - `import torch, platform`
  - `print("torch", torch.__version__)`
  - `print("hip version", torch.version.hip)`
  - `print("cuda available", torch.cuda.is_available())`
  - `print("device", torch.cuda.get_device_name(0))`
  - `print("arch", torch.cuda.get_device_capability())`
  - `print("platform", platform.platform())`
  - `PY`
- Notes: PYTORCH_HIP_ARCH=gfx1100 and HSA_OVERRIDE_GFX_VERSION=11.0.0 exported in shellHook; ROCm tools rocminfo/rocm-smi included for debugging.

