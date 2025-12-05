# Dataclass design for experiment variables and metrics

User intent (verbatim): "create dataclass definition of both experiment variables, and metrics we are measuring in the project."

Intent rephrased: define strongly typed dataclasses that capture all experiment knobs/constants/protocol fields and metric definitions so downstream scripts can validate configurations and reporting is reproducible.

Scope:
- Cover independent variables (architecture, diffusion steps, batch size, sequence length) with allowed levels/ranges from `.clinerules/experiment_variables.md`.
- Cover control variables (hardware, precision, output token count handling, software stack).
- Cover procedural variables (warm-up passes, sampling temperature) as part of the protocol.
- Capture metric definitions from `.clinerules/metrics.md` (latency, inter-token latency, throughput, peak VRAM, steps-to-quality parity/speedup).

Decisions:
- Implement new module `src/config/structures.py` using `dataclasses` with type hints.
- Store allowed ranges/levels in dataclass fields (lists/tuples) to keep them close to configuration validation.
- Provide small helper containers grouping experiment variables and metric specs for easy import by benchmarking/profiling helpers.
- Avoid module-level constants; keep values inside dataclasses via defaults.

Next steps:
- Add `structures.py` with experiment and metric dataclasses.
- Add lightweight references/imports in evaluation code and README to guide usage.


