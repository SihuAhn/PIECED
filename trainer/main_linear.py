import os
import sys
import argparse
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath("."))

from pieced.backbone.stgcn import STGCN
from pieced.args.setup import parse_args_linear
# [중요] 수정된 LinearModel 임포트
from pieced.methods.linear import LinearModel
from pieced.utils.classification_dataloader import prepare_data
from pieced.utils.checkpointer import Checkpointer
from pieced.utils.task_orders import load_task_splits


class BackboneWithPartAttention(nn.Module):
    """STGCN 출력(part feats list) 위에 part_attention 적용 후 list로 다시 반환.

    byol pretraining 시 attention_mode='transformer' 또는 'attention' 인 경우
    encoder 다음에 part_attention 이 붙어서 학습됨. linear eval 시에도 같은 흐름이
    되도록 backbone 을 감싼다.
    """

    def __init__(self, backbone: nn.Module, part_attention: nn.Module, attention_mode: str):
        super().__init__()
        self.backbone = backbone
        self.part_attention = part_attention
        self.attention_mode = attention_mode

    def forward(self, X):
        feats = self.backbone(X)
        if not isinstance(feats, list):
            return feats
        # feats: list of (B, C) for part_pool / List 길이 = num_parts
        # byol 학습 forward 와 동일하게 (B, num_parts, C) 로 stack
        feats_stack = torch.stack(feats, dim=1)
        if self.attention_mode == 'transformer':
            refined = self.part_attention(feats_stack)
        elif self.attention_mode == 'attention':
            refined, _ = self.part_attention(feats_stack, feats_stack, feats_stack)
        else:
            refined = feats_stack
        return [refined[:, i, :] for i in range(refined.size(1))]


def _build_part_attention(state_dict, feature_dim=256):
    """ckpt state_dict 안의 part_attention.* / momentum_part_attention.* 을 보고
    TransformerEncoderLayer 또는 MultiheadAttention 을 복원해 weights 로드.
    반환: (part_attention_module, attention_mode) 또는 (None, None)
    """
    pa_keys = [k for k in state_dict if k.startswith("part_attention.")]
    if not pa_keys:
        return None, None
    pa_state = {k.replace("part_attention.", ""): v for k, v in state_dict.items() if k.startswith("part_attention.")}

    # TransformerEncoderLayer 는 linear1.weight 가 존재
    if any("linear1.weight" in k for k in pa_state):
        # dim_feedforward 추정: linear1.weight shape = (dim_ff, d_model)
        dim_ff = pa_state["linear1.weight"].shape[0]
        d_model = pa_state["linear1.weight"].shape[1]
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4,
            dim_feedforward=dim_ff, dropout=0.1, batch_first=True,
        )
        msg = layer.load_state_dict(pa_state, strict=False)
        print(f"> Loaded part_attention (transformer): {msg}")
        return layer, 'transformer'

    # MultiheadAttention 케이스 (in_proj_weight 만 있고 linear1 없음)
    if any("in_proj_weight" in k for k in pa_state):
        d_model = pa_state["in_proj_weight"].shape[1]
        layer = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True,
        )
        msg = layer.load_state_dict(pa_state, strict=False)
        print(f"> Loaded part_attention (attention): {msg}")
        return layer, 'attention'

    return None, None

def main():
    args_for_seed = argparse.ArgumentParser(add_help=False)
    args_for_seed.add_argument("--seed", type=int, default=5)
    known, _ = args_for_seed.parse_known_args()
    seed_everything(known.seed)

    # =============================================================
    # 인자 가로채기 (Interception)
    # =============================================================
    intercept_parser = argparse.ArgumentParser(add_help=False)
    intercept_parser.add_argument("--pooling_mode", type=str, default="whole")
    intercept_parser.add_argument("--checkpoint_dir", type=str, default="./exp/linear_logs")
    # [추가] classifier_type 인자 추가
    intercept_parser.add_argument("--classifier_type", type=str, default="linear") 
    
    custom_args, leftover_args = intercept_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + leftover_args

    args = parse_args_linear()

    args.pooling_mode = custom_args.pooling_mode
    args.checkpoint_dir = custom_args.checkpoint_dir
    args.classifier_type = custom_args.classifier_type # 전달
    # =============================================================

    tasks = None
    if args.split_strategy == "class":
        if args.task_split_key is not None:
            tasks = load_task_splits(args.task_split_json, args.task_split_key)
            assert len(tasks) == args.num_tasks, (
                f"task_split_key '{args.task_split_key}' yields {len(tasks)} tasks "
                f"but --num_tasks is {args.num_tasks}"
            )
        else:
            # Legacy randperm path
            permuted_indices = torch.randperm(args.num_classes)
            tasks = []
            start_idx = 0
            base_size = args.num_classes // args.num_tasks
            remainder = args.num_classes % args.num_tasks
            for i in range(args.num_tasks):
                current_size = base_size + (1 if i < remainder else 0)
                task_indices = permuted_indices[start_idx : start_idx + current_size]
                tasks.append(task_indices)
                start_idx += current_size

        # Apply optional reorder (1-based, e.g. "3,1,5,2,4")
        if args.task_order is not None:
            order_indices = [int(idx.strip()) - 1 for idx in args.task_order.split(',')]
            tasks = [tasks[i] for i in order_indices]

    if args.encoder == "stgcn":
        graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        if hasattr(args, 'graph_layout'): graph_args['layout'] = args.graph_layout
        if hasattr(args, 'graph_strategy'): graph_args['strategy'] = args.graph_strategy

        backbone = STGCN(in_channels=3, num_class=args.num_classes, 
                         graph_args=graph_args,
                         edge_importance_weighting=True,
                         hidden_channels=64,
                         hidden_dim=256,
                         dropout=0.0,
                         pooling_mode=args.pooling_mode ,
                         pool_type=args.pool_type
                         )
    else:
        raise ValueError("Only [stgcn] is currently supported.")

    backbone.fc = nn.Identity()
    
    if os.path.exists(args.pretrained_feature_extractor):
        print(f"> Loading backbone from {args.pretrained_feature_extractor}")
        checkpoint = torch.load(args.pretrained_feature_extractor, map_location='cpu')
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        backbone_state = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                backbone_state[k.replace("encoder.", "")] = v
            elif k.startswith("momentum_encoder.") and len(backbone_state) == 0:
                 backbone_state[k.replace("momentum_encoder.", "")] = v

        msg = backbone.load_state_dict(backbone_state, strict=False)
        print(f"> Loaded backbone weights: {msg}")

        # pooling_mode='part' & ckpt에 part_attention 가 있으면 같이 로드해서 감싼다.
        if args.pooling_mode == 'part':
            part_attn, attn_mode = _build_part_attention(state)
            if part_attn is not None:
                backbone = BackboneWithPartAttention(backbone, part_attn, attn_mode)
                print(f"> Backbone wrapped with part_attention (mode={attn_mode})")
    else:
        print(f"Warning: Checkpoint not found at {args.pretrained_feature_extractor}")

    # [LinearModel 생성]
    # 중복 인자 제거
    model_kwargs = args.__dict__.copy()
    keys_to_remove = ['pooling_mode', 'tasks', 'num_classes', 'backbone', 'feature_dim', 'classifier_type']
    for key in keys_to_remove:
        if key in model_kwargs: model_kwargs.pop(key)

    model = LinearModel(
        backbone=backbone, 
        num_classes=args.num_classes,
        tasks=tasks,
        pooling_mode=args.pooling_mode, 
        feature_dim=256, 
        classifier_type=args.classifier_type,
        **model_kwargs 
    )

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        semi_supervised=args.semi_supervised, 
    )

    callbacks = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100, log_graph=False)
        wandb_logger.log_hyperparams(args)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir, 
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True), 
        enable_checkpointing=False,
        accelerator="gpu", 
        devices=[0] if args.gpus else 0,
        # gradient_clip_val=1.0,          
        # gradient_clip_algorithm="norm", 
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()