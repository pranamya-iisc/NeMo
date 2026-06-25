# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeMo Speech — toolkit for training/deploying speech models (ASR, TTS, Speech LLM). Active collections: `asr`, `tts`, `audio`, `speechlm2`, `common`. No Megatron / Megatron Core / Transformer Engine — parallelism is PyTorch-native (DDP, FSDP2, TP/SP via DTensor).

## Build & Install

See the canonical installation guide — [`docs/source/starthere/install.rst`](docs/source/starthere/install.rst) (published at https://docs.nvidia.com/nemo/speech/nightly/) — for the uv, pip (bring-your-own Python/PyTorch/CUDA), Docker, and optional `compiled` (SpeechLM2/Automodel) install paths.

Dev quickstart: `uv sync --extra all --extra cu13` (Python 3.12+, PyTorch 2.7+; `test`/`docs` are `--group`s, not extras).

For SpeechLM2 development, also install compiled extras: `uv sync --extra compiled` (adds flash-attn, transformer-engine, mamba-ssm).

## Code Style

- **Line length: 119** (not default 88) — consistent across black, isort, flake8
- Black with `skip_string_normalization = true`
- isort with `profile = black`
- Check: `isort --check <path> && black --check <path>`
- Fix: `isort <path> && black <path>`
- Jupyter Notebooks are excluded from automatic black reformatting (see `extend-exclude`), but can be still reformatted when passed directly. Do not reformat notebooks outside your changes.

## Testing

```bash
pytest tests/collections/asr -m "not pleasefixme" -v     # ASR tests, skip broken
pytest tests/collections/tts -m unit -v                  # TTS unit tests
pytest tests/collections/audio -m unit -v                # Audio tests
pytest tests/collections/speechlm2 -m unit -v            # SpeechLM2 tests
pytest -k "test_name" tests/                             # Single test by name
```

Markers: `unit`, `integration`, `system`, `pleasefixme` (broken — skip), `skipduringci`.

Tests mirror the `nemo/collections/` layout: `tests/collections/{asr,tts,audio,speechlm2,common,speaker_tasks}/`. Core framework tests live in `tests/core/`, `tests/core_ptl/`, `tests/hydra/`, `tests/lightning/`.

## CI & PRs

- NVIDIA developers: feature branches off `main`; community: fork-based workflow
- CI triggered by adding **"Run CICD"** label to the PR
- E2E nightly tests: only when really needed. Add both **"Run e2e nightly"** and **"Run CICD"** labels
- `skip-linting` / `skip-docs` labels bypass those checks
- Formatting CI auto-commits black/isort fixes back to the PR branch
- CI: GitHub Actions in `.github/workflows/`

## Documentation

Sphinx-based docs live in `docs/source/`. Build with:

```bash
uv sync --locked --group docs                        # one-time setup (matches CI)
uv run make -C docs clean html                       # full rebuild
uv run make -C docs html                             # incremental rebuild
```

Output goes to `docs/build/html/`. Open `docs/build/html/index.html` to preview locally.

Other useful targets: `make -C docs linkcheck` (verify external links), `make -C docs doctest` (run embedded doctests).

## Architecture

### Model Hierarchy

All models inherit from `ModelPT` (`nemo/core/classes/modelPT.py`), which combines PyTorch Lightning's `LightningModule` with NeMo's `Model` base class. The standard mixin stack for speech models:

- **`ModelPT`** — training/validation loops, checkpoint save/restore, optimizer setup
- **`Exportable`** (`nemo/core/classes/exportable.py`) — ONNX/TorchScript export
- **`HuggingFaceFileIO`** (`nemo/core/classes/mixins/hf_io_mixin.py`) — push/pull model weights from HuggingFace Hub
- **`AdapterModuleMixin`** (`nemo/core/classes/mixins/adapter_mixins.py`) — LoRA-like adapter fine-tuning support

Individual modules (encoders, decoders, heads) inherit from `NeuralModule` (`nemo/core/classes/module.py`).

### Neural Type System

`nemo/core/neural_types/` defines typed tensor annotations (e.g., `AudioSignal`, `LabelsType`, `LengthsType`, `SpectrogramType`). These appear as `input_types` / `output_types` on `NeuralModule` subclasses and are checked at runtime. When adding a new module, annotate its I/O; when debugging shape/type errors, check these annotations first.

### Collections

Each collection (`nemo/collections/<name>/`) has a consistent internal layout: `data/`, `losses/`, `metrics/`, `models/`, `modules/`, `parts/`.

**`asr/`** — ASR models follow an Encoder-Decoder pattern with task-specific heads:
- CTC: `EncDecCTCModel`, `EncDecCTCModelBPE`
- RNN-T: `EncDecRNNTModel`, `EncDecRNNTBPEModel`
- Hybrid RNN-T/CTC: `EncDecHybridRNNTCTCModel`, `EncDecHybridRNNTCTCBPEModel`
- Classification/Speaker: `EncDecClassificationModel`, `EncDecSpeakerLabelModel`
- Self-supervised: `SpeechEncDecSelfSupervisedModel`
- Base class: `ASRModel` in `nemo/collections/asr/models/asr_model.py`

**`tts/`** — TTS includes G2P (`tts/g2p/`) for grapheme-to-phoneme conversion and `tts/torch/` for torch-native utilities.

**`audio/`** — Audio enhancement/separation models (distinct from ASR; no transcript output).

**`speechlm2/`** — Speech Language Models with streaming support (`speechlm2/streaming/`) and vLLM inference integration (`speechlm2/vllm/`).

**`common/`** — Shared across all collections: tokenizers (BPE, SentencePiece in `common/tokenizers/`), prompts (`common/prompts/`), preprocessing (`common/parts/preprocessing/`), and shared module primitives (RNN, MLP, transformer utils in `common/parts/`).

### Config & Data Loading

- **Hydra + OmegaConf** for all config management; configs live in `examples/<collection>/conf/`
- **Lhotse** (>=1.32.2) for audio data loading: dynamic bucketing, tarred datasets, multi-dataset sampling
- The `scripts/speech_recognition/` helpers (oomptimizer, estimate_duration_bins, estimate_data_weights) are tightly coupled to the Lhotse data pipeline

### Checkpoint Serialization

`SaveRestoreConnector` (`nemo/core/connectors/`) handles `.nemo` file format (a tar archive containing model weights + config). Use `ModelPT.save_to()` / `ModelPT.restore_from()` rather than raw `torch.save`.

## Training & Inference

Entry-point scripts live under `examples/<collection>/`.

All scripts follow the same Hydra pattern — a `@hydra_runner` decorator points to a YAML config in a nearby `conf/` directory:

```python
@hydra_runner(config_path="conf", config_name="fast-conformer_transducer_bpe")
def main(cfg):
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)
```

Override any config value from the CLI with Hydra syntax: `python script.py model.optim.lr=1e-4 trainer.max_epochs=50`. Browse configs with `ls examples/<collection>/conf/`.

## Handy Scripts

Utility scripts live under `scripts/`. Key subdirectories: `speech_recognition/`, `speechlm2/`, `speaker_tasks/`, `tokenizers/`, `dataset_processing/`, `asr_language_modeling/`.

Four frequently used data/training helpers:

- **`scripts/speech_recognition/estimate_duration_bins.py`** — estimate Lhotse dynamic-bucketing duration bins from a manifest or YAML input config. Usage: `python scripts/speech_recognition/estimate_duration_bins.py <input> -b 30 -n 100000`
- **`scripts/speech_recognition/oomptimizer.py`** — find the largest batch size per bucket that fits in GPU memory. Usage: `python scripts/speech_recognition/oomptimizer.py --pretrained-name nvidia/canary-1b` or point to a config with `--config-path`.
- **`scripts/speech_recognition/estimate_data_weights.py`** — compute per-dataset sampling weights from YAML input configs, with optional temperature re-weighting. Usage: `python scripts/speech_recognition/estimate_data_weights.py input.yaml output.yaml -t 0.5`
- **`scripts/speech_recognition/convert_to_tarred_audio_dataset.py`** — shard audio+manifest into tar files. Usage: `python scripts/speech_recognition/convert_to_tarred_audio_dataset.py --manifest_path=m.json --target_dir=./tar --num_shards=512 --max_duration=60.0`

## Subdirectory Instructions

Module-specific instructions can be added as `CLAUDE.md` or `AGENTS.md` files in subdirectories.

## Issue Reproduction

When fixing a bug, always:
1. First reproduce the issue with a minimal test case
2. Add the reproduction as a unit test
3. Then fix the issue
4. Verify the test passes

## Forbidden Operations

- Never push directly to `main`
- Never modify `.github/workflows/` without explicit instruction
- Never delete test files without explicit instruction
