import argparse
import os
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pieced.losses.byol import byol_loss_func
from pieced.methods.base import BaseMomentumModel
from pieced.utils.momentum import initialize_momentum_params

class BYOL(BaseMomentumModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements BYOL with Configurable Self-Attention, Distillation, and Replay Support."""
        super().__init__(**kwargs)

        self.pooling_mode = self.hparams.pooling_mode
        self.num_parts = getattr(self.hparams, 'num_parts', 5)
        self.attention_mode = kwargs.get('attention_mode', 'transformer') 

        self.output_dim = output_dim
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        real_feature_dim = kwargs.get('hidden_dim', 256) 
        
        # -------------------------------------------------------------------------
        # [Replay] 메모리 버퍼 초기화
        # -------------------------------------------------------------------------
        self.use_replay = getattr(self.hparams, "replay_method", "None") in ["ER", "LUMP", "CroMo"]
        
        if self.use_replay:
            self.replay_buffer_v1 = []
            self.replay_buffer_v2 = []
            self.replay_past_v1 = None
            self.replay_past_v2 = None
            self._replay_saved = False

        # Classifier 설정
        if self.pooling_mode == 'part':
            clf_input_dim = real_feature_dim * self.num_parts
            if hasattr(self, 'classifier'):
                self.classifier = nn.Linear(clf_input_dim, self.hparams.num_classes)
            if hasattr(self, 'momentum_classifier'):
                self.momentum_classifier = nn.Linear(clf_input_dim, self.hparams.num_classes)
                for param in self.momentum_classifier.parameters():
                    param.requires_grad = False

        # Helper functions
        def _build_projector():
            return nn.Sequential(
                nn.Linear(real_feature_dim, self.proj_hidden_dim),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.output_dim),
            )

        def _build_predictor():
            return nn.Sequential(
                nn.Linear(self.output_dim, self.pred_hidden_dim),
                nn.BatchNorm1d(self.pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.pred_hidden_dim, self.output_dim),
            )

        # -------------------------------------------------------------------------
        # Attention Module 초기화
        # -------------------------------------------------------------------------
        if self.pooling_mode == 'part':
            if self.attention_mode == 'transformer':
                self.part_attention = nn.TransformerEncoderLayer(
                    d_model=real_feature_dim, nhead=4, 
                    dim_feedforward=real_feature_dim * 4, dropout=0.1, batch_first=True
                )
                self.momentum_part_attention = nn.TransformerEncoderLayer(
                    d_model=real_feature_dim, nhead=4, 
                    dim_feedforward=real_feature_dim * 4, dropout=0.1, batch_first=True
                )
            elif self.attention_mode == 'attention':
                self.part_attention = nn.MultiheadAttention(
                    embed_dim=real_feature_dim, num_heads=4, dropout=0.1, batch_first=True
                )
                self.momentum_part_attention = nn.MultiheadAttention(
                    embed_dim=real_feature_dim, num_heads=4, dropout=0.1, batch_first=True
                )
            elif self.attention_mode == 'None':
                self.part_attention = None
                self.momentum_part_attention = None

        # Projector & Predictor 초기화
        if self.pooling_mode == 'whole':
            self.projector = _build_projector()
            self.momentum_projector = _build_projector()
            self.predictor = _build_predictor()
            
        elif self.pooling_mode == 'part':
            self.projector = nn.ModuleList([_build_projector() for _ in range(self.num_parts)])
            self.momentum_projector = nn.ModuleList([_build_projector() for _ in range(self.num_parts)])
            self.predictor = nn.ModuleList([_build_predictor() for _ in range(self.num_parts)])

        # Momentum 초기화
        if self.pooling_mode == 'whole':
            initialize_momentum_params(self.projector, self.momentum_projector)
        elif self.pooling_mode == 'part':
            for proj, mom_proj in zip(self.projector, self.momentum_projector):
                initialize_momentum_params(proj, mom_proj)
            if self.attention_mode in ['transformer', 'attention']:
                initialize_momentum_params(self.part_attention, self.momentum_part_attention)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BYOL, BYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        parser.add_argument("--attention_mode", type=str, default="transformer", 
                            choices=["None", "attention", "transformer"])
        
        parser.add_argument("--replay_method", type=str, default="None", 
                            choices=["None", "ER", "LUMP", "CroMo"],
                            help="None, ER, LUMP, CroMo")
        parser.add_argument("--replay_mem_size", type=int, default=500)
        parser.add_argument("--mixup_alpha", type=float, default=0.2)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        if self.pooling_mode == 'part' and self.attention_mode in ['transformer', 'attention']:
            extra_learnable_params.append({"params": self.part_attention.parameters()})
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        if self.pooling_mode == 'whole':
            extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        else:
            extra_momentum_pairs = list(zip(self.projector, self.momentum_projector))
            if self.attention_mode in ['transformer', 'attention']:
                extra_momentum_pairs.append((self.part_attention, self.momentum_part_attention))
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        out = super().forward(X, *args, **kwargs)
        all_feats = out["all_feats"]

        if self.pooling_mode == 'whole':
            z = self.projector(all_feats)
            p = self.predictor(z)
            return {**out, "z": z, "p": p}

        elif self.pooling_mode == 'part':
            feats_stack = torch.stack(all_feats, dim=1)
            
            if self.attention_mode == 'transformer':
                refined_feats = self.part_attention(feats_stack)
            elif self.attention_mode == 'attention':
                refined_feats, _ = self.part_attention(feats_stack, feats_stack, feats_stack)
            else:
                refined_feats = feats_stack
            
            z_parts = [self.projector[i](refined_feats[:, i, :]) for i in range(self.num_parts)]
            p_parts = [self.predictor[i](z) for i, z in enumerate(z_parts)]
            
            z_avg = torch.stack(z_parts, dim=0).mean(dim=0)
            p_avg = torch.stack(p_parts, dim=0).mean(dim=0)
            
            return {**out, "z": z_avg, "p": p_avg, "z_parts": z_parts, "p_parts": p_parts}

    # -------------------------------------------------------------------------
    # [Replay] 버퍼 Load (Task 1 경로 역추적 로직 포함)
    # -------------------------------------------------------------------------
    def on_train_start(self):
        super().on_train_start()
        
        current_idx = getattr(self, "current_task_idx", 0)
        
        if self.use_replay and current_idx > 0:
            load_idx = current_idx - 1 
            buffer_path = None
            
            # [수정] Task 1일 경우, pretrained_model이 위치한 LUMP 폴더에서 buffer_0.pt를 우선 탐색
            if current_idx == 1 and getattr(self.hparams, 'pretrained_model', None):
                pt_dir = os.path.dirname(self.hparams.pretrained_model) # e.g., .../wofv8nh6/
                pt_parent_dir = os.path.dirname(pt_dir)                 # e.g., .../2026_02_25_06_01_15-LUMP/
                
                # buffer_0.pt가 저장되어 있을 만한 유력한 후보지들
                candidates = [
                    os.path.join(pt_dir, f"buffer_{load_idx}.pt"),
                    os.path.join(pt_parent_dir, f"buffer_{load_idx}.pt"),
                    os.path.join(self.hparams.checkpoint_dir, f"buffer_{load_idx}.pt")
                ]
                
                for p in candidates:
                    if os.path.exists(p):
                        buffer_path = p
                        break
            
            # Task 2 이상이거나 pretrained_model에서 못 찾았을 경우 기본 폴더(CroMo 진행 폴더) 탐색
            if buffer_path is None:
                buffer_path = os.path.join(self.hparams.checkpoint_dir, f"buffer_{load_idx}.pt")
                
            # 최종 Load
            if os.path.exists(buffer_path):
                buffers = torch.load(buffer_path)
                self.replay_past_v1 = buffers['v1']
                self.replay_past_v2 = buffers['v2']
                print(f"\n[*] Replay({self.hparams.replay_method}): Loaded buffer with {len(self.replay_past_v1)} samples from {buffer_path}")
            else:
                print(f"\n[!] Replay({self.hparams.replay_method}): Buffer file not found at {buffer_path} despite task_idx > 0!")

    # -------------------------------------------------------------------------
    # [Replay] 버퍼 Save
    # -------------------------------------------------------------------------
    def _save_replay_buffer(self):
        if getattr(self, "_replay_saved", False): return
        os.makedirs(self.hparams.checkpoint_dir, exist_ok=True)
        
        current_idx = getattr(self, "current_task_idx", 0)
        buffer_path = os.path.join(self.hparams.checkpoint_dir, f"buffer_{current_idx}.pt")
        
        new_v1 = list(self.replay_past_v1) if self.replay_past_v1 is not None else []
        new_v2 = list(self.replay_past_v2) if self.replay_past_v2 is not None else []
        new_v1.extend(self.replay_buffer_v1)
        new_v2.extend(self.replay_buffer_v2)
        
        torch.save({'v1': new_v1, 'v2': new_v2}, buffer_path)
        print(f"\n[*] Replay({self.hparams.replay_method}): Saved buffer with {len(new_v1)} samples to {buffer_path}")
        self._replay_saved = True

    def on_train_epoch_end(self):
        if hasattr(super(), "on_train_epoch_end"): super().on_train_epoch_end()
        if self.use_replay and self.current_epoch == self.trainer.max_epochs - 1:
            self._save_replay_buffer()

    def on_train_end(self):
        super().on_train_end()
        if self.use_replay: self._save_replay_buffer()

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        # -------------------------------------------------------------------------
        # [Replay] 데이터 증강 & Mixup
        # -------------------------------------------------------------------------
        if self.use_replay:
            is_dict_batch = isinstance(batch, dict)
            task_key = f"task{self.current_task_idx}" if hasattr(self, "current_task_idx") else "task0"
            actual_batch = batch[task_key] if is_dict_batch else batch
            
            if len(actual_batch) == 3:
                idx, views, target = actual_batch
            else:
                views, target = actual_batch
                idx = None
                
            v1, v2 = views[0], views[1]
            batch_size = v1.size(0)
            
            # 미래 Task를 위해 현재 데이터 원본 수집
            if len(self.replay_buffer_v1) < self.hparams.replay_mem_size:
                for i in range(batch_size):
                    if len(self.replay_buffer_v1) < self.hparams.replay_mem_size:
                        self.replay_buffer_v1.append(v1[i].detach().cpu())
                        self.replay_buffer_v2.append(v2[i].detach().cpu())

            # 과거 데이터가 있으면 Replay 적용
            if self.replay_past_v1 is not None and len(self.replay_past_v1) > 0:
                rand_idx = torch.randint(0, len(self.replay_past_v1), (batch_size,))
                mem_v1 = torch.stack([self.replay_past_v1[i] for i in rand_idx]).to(v1.device)
                mem_v2 = torch.stack([self.replay_past_v2[i] for i in rand_idx]).to(v2.device)

                if self.hparams.replay_method == "ER":
                    mask = torch.bernoulli(torch.full((batch_size,), 0.5)).to(v1.device)
                    mask_shape = [-1] + [1] * (v1.dim() - 1)
                    mask = mask.view(*mask_shape)
                    v1 = v1 * mask + mem_v1 * (1.0 - mask)
                    v2 = v2 * mask + mem_v2 * (1.0 - mask)

                elif self.hparams.replay_method in ["LUMP", "CroMo"]:
                    lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
                    v1 = lam * v1 + (1.0 - lam) * mem_v1
                    v2 = lam * v2 + (1.0 - lam) * mem_v2

            if idx is not None:
                actual_batch = (idx, [v1, v2], target)
            else:
                actual_batch = ([v1, v2], target)
                
            if is_dict_batch:
                batch[task_key] = actual_batch
            else:
                batch = actual_batch
        
        # -------------------------------------------------------------------------
        # 원본 BYOL 로직 진행
        # -------------------------------------------------------------------------
        out = super().training_step(batch, batch_idx)
        
        total_neg_cos_sim = 0
        all_z = [] 
        cromo_feature_loss = torch.tensor(0.0, device=self.device) 

        if self.pooling_mode == 'whole':
            feats1, feats2 = out["feats"]
            momentum_feats1, momentum_feats2 = out["momentum_feats"]
            z1 = self.projector(feats1); z2 = self.projector(feats2)
            p1 = self.predictor(z1); p2 = self.predictor(z2)
            
            with torch.no_grad():
                z1_momentum = self.momentum_projector(momentum_feats1)
                z2_momentum = self.momentum_projector(momentum_feats2)

            total_neg_cos_sim += byol_loss_func(p1, z2_momentum) + byol_loss_func(p2, z1_momentum)
            all_z.extend([z1, z2])
            out.update({"z": [z1, z2]})
            
            z_avg_current = (z1 + z2) / 2.0

        elif self.pooling_mode == 'part':
            feats1_list, feats2_list = out["feats"]
            momentum_feats1_list, momentum_feats2_list = out["momentum_feats"]

            f1_stack = torch.stack(feats1_list, dim=1)
            f2_stack = torch.stack(feats2_list, dim=1)
            mf1_stack = torch.stack(momentum_feats1_list, dim=1)
            mf2_stack = torch.stack(momentum_feats2_list, dim=1)

            if self.attention_mode == 'transformer':
                ref_f1 = self.part_attention(f1_stack)
                ref_f2 = self.part_attention(f2_stack)
            elif self.attention_mode == 'attention':
                ref_f1, _ = self.part_attention(f1_stack, f1_stack, f1_stack)
                ref_f2, _ = self.part_attention(f2_stack, f2_stack, f2_stack)
            else:
                ref_f1 = f1_stack; ref_f2 = f2_stack
            
            with torch.no_grad():
                if self.attention_mode == 'transformer':
                    ref_mf1 = self.momentum_part_attention(mf1_stack)
                    ref_mf2 = self.momentum_part_attention(mf2_stack)
                elif self.attention_mode == 'attention':
                    ref_mf1, _ = self.momentum_part_attention(mf1_stack, mf1_stack, mf1_stack)
                    ref_mf2, _ = self.momentum_part_attention(mf2_stack, mf2_stack, mf2_stack)
                else:
                    ref_mf1 = mf1_stack; ref_mf2 = mf2_stack

            z1_parts, z2_parts = [], []
            ssl_part_losses = []

            for i in range(self.num_parts):
                z1 = self.projector[i](ref_f1[:, i, :])
                z2 = self.projector[i](ref_f2[:, i, :])
                p1 = self.predictor[i](z1)
                p2 = self.predictor[i](z2)

                with torch.no_grad():
                    z1_momentum = self.momentum_projector[i](ref_mf1[:, i, :])
                    z2_momentum = self.momentum_projector[i](ref_mf2[:, i, :])
                
                loss_part_i = byol_loss_func(p1, z2_momentum) + byol_loss_func(p2, z1_momentum)
                ssl_part_losses.append(loss_part_i.flatten())
                all_z.extend([z1, z2])
                z1_parts.append(z1); z2_parts.append(z2)
            
            all_part_losses_tensor = torch.stack(ssl_part_losses, dim=0).transpose(0, 1) # (B, P)
            total_neg_cos_sim += all_part_losses_tensor.sum()
            out.update({"z": [z1_parts, z2_parts]})
            
            z_avg_current = torch.stack(z1_parts, dim=0).mean(dim=0)

        # -------------------------------------------------------------------------
        # 🟢 CroMo: Feature Mixup (Distillation)
        # -------------------------------------------------------------------------
        if self.hparams.replay_method == "CroMo" and getattr(self, "current_task_idx", 0) > 0:
            old_model = getattr(self, "distiller", getattr(self, "teacher", None))
            
            if old_model is not None:
                old_model.eval()
                with torch.no_grad():
                    old_out = old_model(v1)
                    if isinstance(old_out, dict) and "z" in old_out:
                        old_z = old_out["z"]
                    else:
                        old_z = old_out 
                        
                if isinstance(old_z, list):
                    old_z_avg = torch.stack(old_z, dim=0).mean(dim=0)
                else:
                    old_z_avg = old_z
                    
                cromo_feature_loss = byol_loss_func(z_avg_current, old_z_avg)

        # Calculate STD
        if len(all_z) > 0:
            z_std = torch.stack([F.normalize(z, dim=-1).std(dim=0) for z in all_z]).mean()
        else:
            z_std = 0.0

        metrics = {
            "train_neg_cos_sim": total_neg_cos_sim,
            "train_z_std": z_std,
        }
        if cromo_feature_loss > 0:
            metrics["train_cromo_loss"] = cromo_feature_loss

        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        out.update({"loss": out["loss"] + total_neg_cos_sim + cromo_feature_loss})
        
        return out