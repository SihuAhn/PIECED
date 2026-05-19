#!/bin/bash

# ============================================================================
# 1. 사용자 정의 설정 (User Hardcoded Settings)
# 이 파일에서 직접 모델 경로와 프로젝트 설정을 관리합니다.
# ============================================================================

# [수정] 평가할 체크포인트 경로
CKPT_PATH="exp/ntu60_xsub/SA/2026_01_02_09_00_26-FT/wwzrxjbk/FT-task4-ep=499-wwzrxjbk.ckpt"

# [수정] 데이터셋 경로
DATA_DIR="${DATA_DIR:-./data/ntu60/xsub}"

# [수정] WandB 프로젝트 및 기본 Job 이름
# (나중에 뒤에 _1p, _10p가 붙습니다)
BASE_PROJECT_NAME="results_ntu60_xsub"
BASE_JOB_NAME="FT"

# ============================================================================
# 2. 동적 변수 처리 (Dynamic Variables from Python)
# job_semi.py에서 SEMI_RATIO와 CHECKPOINT_DIR만 주입받습니다.
# ============================================================================

RATIO="${SEMI_RATIO:-0.01}"           # 없으면 기본값 1%
SAVE_DIR="${CHECKPOINT_DIR:-./exp/semi_eval_manual}"

# [신규 추가] 파이썬에서 넘겨주는 하이퍼파라미터 수신
# ${변수명:-기본값}: 환경변수가 있으면 쓰고, 없으면 기본값 사용
TYPE="${CLASSIFIER_TYPE:-linear}"     # (기본값 linear)
WD="${WEIGHT_DECAY:-0}"               # (기본값 0)

# [로직] 비율에 따라 Suffix 생성 (예: 1p, 10p)
if [ "$RATIO" = "0.01" ]; then
    SUFFIX="1p"
elif [ "$RATIO" = "0.05" ]; then
    SUFFIX="5p"
elif [ "$RATIO" = "0.1" ]; then
    SUFFIX="10p"
elif [ "$RATIO" = "0.2" ]; then
    SUFFIX="20p"
elif [ "$RATIO" = "0.3" ]; then
    SUFFIX="30p"
elif [ "$RATIO" = "0.5" ]; then
    SUFFIX="50p"
elif [ "$RATIO" = "1.0" ]; then
    SUFFIX="100p"
else
    SUFFIX="custom"
fi

# [핵심 변경] Job Name과 Project Name에 Suffix 적용
FINAL_JOB_NAME="${BASE_JOB_NAME}_${TYPE}_wd${WD}"
FINAL_PROJECT_NAME="${BASE_PROJECT_NAME}_${SUFFIX}"

echo "----------------------------------------------------------------"
echo "Running Linear Eval:"
echo "Checkpoint: $CKPT_PATH"
echo "Ratio:      $RATIO ($SUFFIX)"
echo "Output Dir: $SAVE_DIR"
echo "Project:    $FINAL_PROJECT_NAME"
echo "Job Name:   $FINAL_JOB_NAME"
echo "----------------------------------------------------------------"

# ============================================================================
# 실행 명령어
# ============================================================================
python3 trainer/main_linear.py \
    --dataset ntu60 \
    --encoder stgcn \
    --data_dir "$DATA_DIR" \
    --pretrained_feature_extractor "$CKPT_PATH" \
    --semi_supervised "$RATIO" \
    --checkpoint_dir "$SAVE_DIR" \
    --name "$FINAL_JOB_NAME" \
    --split_strategy class \
    --num_tasks 5 \
    --max_epochs 100 \
    --batch_size 128 \
    --num_workers 5 \
    --classifier_type "$TYPE" \
    --weight_decay "$WD" \
    --lr 0.3 \
    --lr_decay_steps 60 80 \
    --optimizer sgd \
    --scheduler step \
    --gpus 0 \
    --precision 16 \
    --wandb \
    --project "$FINAL_PROJECT_NAME" \
    --seed 5 \
    --pooling_mode part \
    --task_split_key NTU60_xsub
