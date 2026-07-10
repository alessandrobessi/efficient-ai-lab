# Models

Model binaries (safetensors, GGUF, etc.) are **never committed to this repository** —
they're large, license-encumbered, and reproducible from a model ID + revision.

Python experiments download models via the Hugging Face Hub into the standard HF cache
(`~/.cache/huggingface/`) unless `HF_HOME` is overridden. `llama.cpp`-based experiments
(Week 2+) download GGUF files into `models/gguf/` (gitignored).

Every experiment README and config records the exact model ID and revision/quantization
used, so results are reproducible without committing the weights themselves.
