# PIECED — Part-Aware Self-Distillation for Continual Skeleton Representation Learning

Minimal training code for the part-transformer-distillation variant of PIECED on
NTU60, released alongside the paper.

## Layout

```
PIECED/
├── pieced/                # Python package (methods, distillers, backbones, …)
│   ├── args/              # Argument parsing
│   ├── backbone/          # ST-GCN + part / transformer pooling
│   ├── distillers/        # base, predictive (PRSD)
│   ├── losses/            # BYOL loss
│   ├── methods/           # BaseModel, BYOL, LinearModel
│   └── utils/             # Data loaders, checkpointer, LARS, KNN, task_orders
├── trainer/
│   ├── main_continual.py  # Multi-task loop driver (calls main_pretrain.py)
│   ├── main_pretrain.py   # Per-task pre-training
│   └── main_linear.py     # Linear-probe evaluation
├── bash_files/
│   ├── ntu60_xsub/{FT.sh, PIECED.sh, semi_FT.sh, semi_PIECED.sh}
│   └── ntu60_xview/{FT.sh, PIECED.sh, semi_FT.sh, semi_PIECED.sh}
├── task_orders.json       # Class-to-task assignments (NTU60/120, PKU-MMD)
├── job_launcher.py        # Pre-training launcher
└── job_semi.py            # Semi-supervised linear-eval launcher
```

## Requirements

- Python ≥ 3.10
- PyTorch + CUDA
- PyTorch Lightning
- `pl_bolts`, `torchmetrics`, `wandb` (optional), `scikit-learn`, `scipy`, `umap-learn` (optional)

## Data

NTU60 skeletons live under `./data/ntu60/xsub` and `./data/ntu60/xview`.
Each split directory contains `train_data_joint.npy`, `train_label.pkl`,
`val_data_joint.npy`, `val_label.pkl`.

## Quick start

The two Python launchers (`job_launcher.py`, `job_semi.py`) `chdir` to the repo
root and inject `--data_dir` / env vars into the bash scripts. The defaults at
the top of each launcher target NTU60 XSub; edit `data_name` / `training_model`
to switch, or override on the CLI.

> **Note**: `bash_files/` is `.gitignored` (it contains user-specific values
> like `--entity`, GPU id, and per-run `--pretrained_model` paths). Create the
> scripts locally before running the launchers. The exact arguments used in the
> paper are shown below.

### Pre-training (`job_launcher.py`)

```bash
# Task 0 — Fine-tuning baseline (NTU60 XSub, default)
python3 job_launcher.py

# Tasks 1-4 — PIECED continual pre-training
python3 job_launcher.py --script bash_files/ntu60_xsub/PIECED.sh

# NTU60 XView side
python3 job_launcher.py \
    --script bash_files/ntu60_xview/FT.sh \
    --base_experiment_dir ./exp/ntu60_xview \
    --data_dir ./data/ntu60/xview
```

**Chaining FT → PIECED**: each task's checkpoint is logged at the end of its
run, e.g. `exp/ntu60_xsub/<timestamp>-FT/<run_hash>/FT-task0-ep=499-<run_hash>.ckpt`.
Before running `PIECED.sh`, open the script and set `--pretrained_model` to
the **task-0 FT checkpoint path** produced above (do not leave `<run_id>` /
`<hash>` placeholders — `<` and `>` are shell redirection operators and will
fail with `cannot open run_id`). Example:

```bash
--pretrained_model exp/ntu60_xsub/2026_05_19_07_36_31-FT/b3h5vezf/FT-task0-ep=499-b3h5vezf.ckpt \
```

`main_continual.py` then loops tasks 1 → 4, chaining each task's checkpoint
into the next automatically.

### Semi-supervised linear evaluation (`job_semi.py`)

```bash
# Default — 100% labels, PIECED checkpoint on NTU60 XSub
# (edit CKPT_PATH in bash_files/ntu60_xsub/semi_PIECED.sh first)
python3 job_semi.py

# Sweep label ratios
python3 job_semi.py --ratios 0.01 0.1 1.0

# FT checkpoint instead of PIECED
python3 job_semi.py --script bash_files/ntu60_xsub/semi_FT.sh

# NTU60 XView, custom GPU
python3 job_semi.py \
    --script bash_files/ntu60_xview/semi_PIECED.sh \
    --data_dir ./data/ntu60/xview \
    --gpu 0
```

`job_semi.py` injects `SEMI_RATIO`, `CHECKPOINT_DIR`, and `DATA_DIR` env vars
into the bash script and runs each ratio sequentially.

## Class-task assignments

[`task_orders.json`](./task_orders.json) holds the exact class splits used in
the paper for `NTU60_x{sub,view}`, `NTU120_x{sub,set}`, `PKU_phase{1,2}_xsub`.
The bash scripts pass `--task_split_key NTU60_xsub` (etc.) and
`trainer/main_pretrain.py` / `main_linear.py` load the corresponding splits
through [`pieced/utils/task_orders.py`](./pieced/utils/task_orders.py). Leaving
`--task_split_key` unset falls back to the legacy `randperm(seed=5)` path.

## Key flags

- `--pooling_mode part` — part-aware pooling
- `--attention_mode transformer` — transformer attention head
- `--method byol` — BYOL backbone objective
- `--distiller predictive` — PIECED / PRSD predictive distillation
- `--distill_lamb 1.0` — distillation loss weight
- `--num_tasks 5 --split_strategy class` — class-incremental 5-task protocol
- `--task_split_key NTU60_xsub` — load class splits from `task_orders.json`
- `--wandb --project PIECED_ntu60_xsub` — enable W&B logging (entity uses your
  default account; pass `--entity <name>` to override)
