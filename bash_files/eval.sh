#!/bin/bash
# Linear evaluation of PIECED on NTU RGB+D 60 xview.
# The classifier is trained on all classes after each task, and the accuracy is
# reported per task (val_acc1_task{n}), which gives one column of the accuracy
# matrix a_{n,m} used by scripts/compute_cl_metrics.py.
#
# Run through job_semi.py, or directly: bash bash_files/eval.sh

DATA_DIR=${DATA_DIR:-./data/ntu60/xview}
SEMI_RATIO=${SEMI_RATIO:-1.0}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./exp/linear_eval/ntu60_xview}

# Checkpoints saved at the end of each task (task 0 from FT.sh, tasks 1-4 from PIECED.sh).
CKPTS=(
    "./exp/ntu60_xview/PATH_TO_TASK0_CHECKPOINT.ckpt"
    "./exp/ntu60_xview/PATH_TO_TASK1_CHECKPOINT.ckpt"
    "./exp/ntu60_xview/PATH_TO_TASK2_CHECKPOINT.ckpt"
    "./exp/ntu60_xview/PATH_TO_TASK3_CHECKPOINT.ckpt"
    "./exp/ntu60_xview/PATH_TO_TASK4_CHECKPOINT.ckpt"
)

for TASK in "${!CKPTS[@]}"; do
    python3 trainer/main_linear.py \
        --dataset ntu60 \
        --data_dir "$DATA_DIR" \
        --split_strategy class \
        --num_tasks 5 \
        --task_split_key NTU60_xview \
        --seed 5 \
        --encoder stgcn \
        --pooling_mode part \
        --classifier_type linear \
        --pretrained_feature_extractor "${CKPTS[$TASK]}" \
        --semi_supervised "$SEMI_RATIO" \
        --max_epochs 100 \
        --batch_size 128 \
        --num_workers 5 \
        --gpus 0 \
        --precision 16 \
        --optimizer sgd \
        --scheduler step \
        --lr_decay_steps 60 80 \
        --lr 0.3 \
        --weight_decay 0 \
        --name "PIECED_task${TASK}" \
        --project PIECED_linear_ntu60_xview \
        --wandb \
        --checkpoint_dir "$CHECKPOINT_DIR/task${TASK}"
done