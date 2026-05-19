import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from pieced.distillers.base import base_distill_wrapper
from pieced.losses.byol import byol_loss_func


def predictive_distill_wrapper(Method=object):
    class PredictiveDistillWrapper(base_distill_wrapper(Method)):
        def __init__(self, distill_lamb: float, distill_proj_hidden_dim, **kwargs):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            output_dim = kwargs["output_dim"]
            self.pooling_mode = self.hparams.pooling_mode
            self.num_parts = self.hparams.num_parts

            def _build_distill_predictor():
                return nn.Sequential(
                    nn.Linear(output_dim, distill_proj_hidden_dim),
                    nn.BatchNorm1d(distill_proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(distill_proj_hidden_dim, output_dim),
                )

            if self.pooling_mode == 'whole':
                self.distill_predictor = _build_distill_predictor()
            
            elif self.pooling_mode == 'part':
                self.distill_predictor = nn.ModuleList(
                    [_build_distill_predictor() for _ in range(self.num_parts)]
                )

        @staticmethod
        def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")
            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            extra_learnable_params = []
            
            if self.hparams.pooling_mode == 'whole':
                extra_learnable_params.append({
                    "name": "distill_predictor",
                    "params": self.distill_predictor.parameters(),
                    "lr": self.hparams.lr if self.distill_lamb >= 1 else self.hparams.lr / self.distill_lamb,
                    "weight_decay": self.hparams.weight_decay,
                })
            elif self.hparams.pooling_mode == 'part':
                extra_learnable_params.append({
                    "name": "distill_predictor",
                    "params": self.distill_predictor.parameters(),
                    "lr": self.hparams.lr if self.distill_lamb >= 1 else self.hparams.lr / self.distill_lamb,
                    "weight_decay": self.hparams.weight_decay,
                })

            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            
            z_views = out["z"]
            frozen_z_views = out["frozen_z"]
            
            z1_features, z2_features = z_views
            frozen_z1_features, frozen_z2_features = frozen_z_views

            total_distill_loss = 0

            if self.hparams.pooling_mode == 'whole':
                p1 = self.distill_predictor(z1_features)
                p2 = self.distill_predictor(z2_features)
                total_distill_loss = (byol_loss_func(p1, frozen_z1_features) + byol_loss_func(p2, frozen_z2_features)) / 2
            
            elif self.hparams.pooling_mode == 'part':
                distill_part_losses = []
                for i in range(self.num_parts):
                    p1_part = self.distill_predictor[i](z1_features[i])
                    p2_part = self.distill_predictor[i](z2_features[i])
                    
                    loss = byol_loss_func(p1_part, frozen_z1_features[i]) + \
                           byol_loss_func(p2_part, frozen_z2_features[i])
                    
                    distill_part_losses.append(loss.flatten() / 2.0)

                # 평균 Loss 계산
                all_part_losses_tensor = torch.stack(distill_part_losses, dim=0).transpose(0, 1) # (B, P)
                total_distill_loss = all_part_losses_tensor.sum()

            self.log("train_predictive_distill_loss", total_distill_loss, on_epoch=True, sync_dist=True)
            return out["loss"] + self.distill_lamb * total_distill_loss

    return PredictiveDistillWrapper