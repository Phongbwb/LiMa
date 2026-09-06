# DATN: Retrieval & Recognition Models

This repository contains code, experiment scripts, and pretrained checkpoints for retrieval
and recognition tasks built with PyTorch. It includes training, evaluation and inference
utilities, dataset json manifests, and tools to reproduce experiments.

## Repository layout

- `configs/` — experiment and model configuration files
- `data/` — data and JSON manifests used by experiments
- `experiments/` — training / evaluation scripts and helpers
- `models/` — model definitions
- `checkpoints/` — model checkpoints (weights)
- `outputs/` — experiment outputs and logs
- `src/` — primary training/evaluation/inference entrypoints
- `visualizations/` — visualization utilities

See the project root and folders for additional helpers and utilities.

## Requirements

- Python 3.8+ recommended
- PyTorch (compatible with your GPU) — e.g. `torch` and `torchvision`
- Other common packages: `numpy`, `tqdm`, `scipy`, `opencv-python`, `matplotlib`

If a `requirements.txt` is not provided, create a virtual environment and install typical deps:

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows use: .venv\\Scripts\\activate
pip install --upgrade pip
pip install torch torchvision numpy tqdm scipy opencv-python matplotlib
```

If you maintain a `requirements.txt`, install with:

```bash
pip install -r requirements.txt
```

## Data preparation

Put your dataset and JSON manifests under the `data/` folder following the existing
structure. Example manifests are provided in `data/json/` (e.g. intent and color jsons).

## Quick Start — Training

Many experiments live under the `experiments/` and top-level `src/` scripts. Common commands:

```bash
# Train using the experiments script
python experiments/train.py --config configs/your_config.yaml

# Or use the main training entry in src/ if configured
python src/train.py --config configs/your_config.yaml
```

Typical flags:
- `--config` path to a config file in `configs/`
- `--ckpt` path to checkpoint for resuming

Check `experiments/train.py` and `src/train.py` for script-specific CLI options.

## Quick Start — Inference / Evaluation

Run inference or evaluation with the provided scripts:

```bash
# Run top-level inference
python src/infer.py --config configs/your_config.yaml --ckpt checkpoints/model.pth --output outputs/predictions.json

# Use specific experiment evaluation scripts
python experiments/eval_train.py --config configs/your_config.yaml --ckpt checkpoints/model.pth
```

There are also specialized infer scripts in `experiments/` (e.g. `infer_trajectory.py`, `infer_color.py`).

## Checkpoints and Outputs

Save and load model weights from `checkpoints/`. Experiment outputs (predictions, logs,
visualizations) are stored under `outputs/` and `visualizations/`.

## Reproducing Submission

If you need to produce `submission.json`, inspect the top-level scripts that generate it
or run the inference pipeline and convert outputs to the expected submission format.

## Debugging & Visualization

- Use `visualizations/visualization.py` to inspect results and generate plots.
- Use `experiments/visual.py` for quick visual debugging of outputs.

## Contributing

If you add features or experiments, please:

1. Add configuration files under `configs/`.
2. Place experiments under `experiments/` and log outputs to `outputs/`.
3. Add README updates describing new usage.

## Contact / Notes

This README provides a high-level guide to get started. If you want, I can:

- generate a `requirements.txt` from imports in the repo
- add example config files and a minimal end-to-end run script

The updated README is written to the project root: [README.md](README.md)
