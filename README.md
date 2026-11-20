# PIECED: Part-Aware Inter-Joint Exemplar-free Continual Self-Supervised Encoding and Distillation for Skeleton-Based Action Recognition

Training and evaluation code for PIECED, released alongside the paper.

<img width="2304" height="1323" alt="image" src="https://github.com/user-attachments/assets/5da6921b-bafa-47d2-9b03-ea4cddb9b0db" />

## Layout

```
PIECED/
├── pieced/                # Python package (methods, distillers, backbones, …)
│   ├── args/              # Argument parsing
│   ├── backbone/          # ST-GCN + part / transformer pooling
│   ├── distillers/        # base (frozen old network), predictive (part-level distillation)
│   ├── losses/            # BYOL loss
│   ├── methods/           # BaseModel, BYOL, LinearModel
│   └── utils/             # Data loaders, checkpointer, LARS, KNN, task_orders
├── trainer/
│   ├── main_continual.py  # Multi-task loop driver (calls main_pretrain.py)
│   ├── main_pretrain.py   # Per-task pre-training
│   └── main_linear.py     # Linear-probe evaluation
├── bash_files/{FT.sh, PIECED.sh, eval.sh}   # Configuration used in the paper (NTU RGB+D 60 xview)
├── scripts/
│   └── compute_cl_metrics.py   # AA, F, and LA from accuracy matrices
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

Skeleton data live under `./data/<dataset>/<split>` (e.g. `./data/ntu60/xview`).
Each split directory contains `train_data_joint.npy`, `train_label.pkl`,
`val_data_joint.npy`, `val_label.pkl`.

The datasets used in this study are publicly available: NTU RGB+D 60 and NTU RGB+D 120 at https://rose1.ntu.edu.sg/dataset/actionRecognition/, and PKU-MMD at https://struct002.github.io/PKUMMD/.

## Configuration

[`bash_files/`](./bash_files) holds the configuration used in the paper,
shown for NTU RGB+D 60 xview. Tasks are 0-indexed in the code, so task 0 here is task 1 in the paper.

| Script | Stage |
|---|---|
| `FT.sh` | Sequential training without distillation. The checkpoint saved after task 0 is the starting point of PIECED (add `--end_task_idx 1` to stop after task 0). |
| `PIECED.sh` | Tasks 1-4 with part-level distillation (`--distiller predictive --distill_lamb 1.0`). |
| `eval.sh` | Linear evaluation of the checkpoint saved after each task. |

The same hyperparameters are used for all six evaluation splits. To run another split, change
only the following arguments in the three scripts:

| Evaluation split | `--dataset` | `--data_dir` | `--task_split_key` | `--num_tasks` |
|---|---|---|---|---|
| NTU RGB+D 60 xview | `ntu60` | `./data/ntu60/xview` | `NTU60_xview` | 5 |
| NTU RGB+D 60 xsub | `ntu60` | `./data/ntu60/xsub` | `NTU60_xsub` | 5 |
| NTU RGB+D 120 xset | `ntu120` | `./data/ntu120/xset` | `NTU120_xset` | 10 |
| NTU RGB+D 120 xsub | `ntu120` | `./data/ntu120/xsub` | `NTU120_xsub` | 10 |
| PKU-MMD phase 1 | `pkuv1` | `./data/pkuv1/xsub` | `PKU_phase1_xsub` | 5 |
| PKU-MMD phase 2 | `pkuv2` | `./data/pkuv2/xsub` | `PKU_phase2_xsub` | 5 |

**Random seeds.** The results in the paper are averaged over five seeds: 5, 15, 25, 35, and 45
(`--seed`). The class-task assignment is fixed and does not depend on the seed.

**Frozen old network.** From task 1 onward, the old network is a frozen copy of the entire
current network at the end of the previous task: the ST-GCN backbone, the part transformer, and
the part-wise projection heads ([`pieced/distillers/base.py`](./pieced/distillers/base.py)).

## Quick start

`job_launcher.py` and `job_semi.py` `chdir` to the repo root and inject `--data_dir` / env vars
into the bash scripts.

### Pre-training (`job_launcher.py`)

```bash
# Task 0 (training without distillation)
python3 job_launcher.py --script bash_files/FT.sh

# Tasks 1-4 with PIECED
python3 job_launcher.py --script bash_files/PIECED.sh
```

Each task's checkpoint is saved at the end of its run under `exp/ntu60_xview/<timestamp>-<name>/`.
Before running `PIECED.sh`, set `--pretrained_model` in the script to the task-0 checkpoint
produced by `FT.sh`. `main_continual.py` then loops over tasks 1 to 4, passing each task's
checkpoint to the next automatically.

To run another split, also pass `--base_experiment_dir` and `--data_dir`, for example
`--base_experiment_dir ./exp/ntu60_xsub --data_dir ./data/ntu60/xsub`.

### Linear evaluation (`job_semi.py`)

Set the five checkpoint paths (`CKPTS`) in `bash_files/eval.sh`, then run:

```bash
# 100% of the labels (protocol used for AA, F, and LA)
python3 job_semi.py --script bash_files/eval.sh --data_dir ./data/ntu60/xview

# Label-efficiency sweep
python3 job_semi.py --script bash_files/eval.sh --data_dir ./data/ntu60/xview \
    --ratios 0.01 0.1 1.0
```

After each task, the backbone and the part transformer are frozen, and a linear classifier is
trained on the labeled training data of all classes. The accuracy is reported for the test
samples of each task as `val_acc1_task{n}` (0-indexed task, value in [0, 1]).

## Metrics

[`scripts/compute_cl_metrics.py`](./scripts/compute_cl_metrics.py) computes the average accuracy
(AA), forgetting (F), and learning accuracy (LA) defined in the paper from accuracy matrices.
Each CSV file holds the N x N matrix of one seed without a header, where row n is the evaluated
task, column m is the training stage (after training up to task m), and the entry is a_{n,m} in
percent, that is, 100 x `val_acc1_task{n-1}` at the last epoch of the linear evaluation of the
task-m checkpoint. Cells with m < n are not used and may be left empty.

```bash
python3 scripts/compute_cl_metrics.py seed5.csv seed15.csv seed25.csv seed35.csv seed45.csv
```

## Class-task assignments

[`task_orders.json`](./task_orders.json) holds the exact class splits used in
the paper for `NTU60_x{sub,view}`, `NTU120_x{sub,set}`, `PKU_phase{1,2}_xsub`.
The bash scripts pass `--task_split_key NTU60_xview` (etc.) and
`trainer/main_pretrain.py` / `main_linear.py` load the corresponding splits
through [`pieced/utils/task_orders.py`](./pieced/utils/task_orders.py). Leaving
`--task_split_key` unset falls back to the legacy `randperm(seed=5)` path.

## Key flags

- `--pooling_mode part` — part-aware pooling
- `--attention_mode transformer` — transformer attention head
- `--method byol` — BYOL backbone objective
- `--distiller predictive` — part-level distillation of PIECED
- `--distill_lamb 1.0` — distillation loss weight
- `--num_tasks 5 --split_strategy class` — class-incremental 5-task protocol
- `--task_split_key NTU60_xview` — load class splits from `task_orders.json`
- `--wandb --project PIECED_ntu60_xview` — enable W&B logging (entity uses your
  default account; pass `--entity <name>` to override)