User intent (verbatim):
- "Now you need to create a download script that prepare data and models."

Concise rephrase:
- Provide a reproducible script to fetch required Hugging Face models (baseline LLaMA 3 8B and LLaDA 8B target) and the Wikitext dataset so local ROCm runs can proceed without runtime download failures.

Scope and notes:
- Must align with ROCm/RX 7900 XT environment via `nix-shell.nix`.
- Should use existing experiment config defaults where possible.
- Needs to respect HF authentication for gated models and allow configurable cache dirs.


