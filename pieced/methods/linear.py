from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pieced.utils.lars import LARSWrapper
from pieced.utils.metrics import accuracy_at_k, weighted_mean
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.sigma is not None:
            nn.init.constant_(self.sigma, 15.0)

    def forward(self, x):
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        cosine_logits = F.linear(x_norm, w_norm)
        if self.sigma is not None:
            return cosine_logits * self.sigma
        return cosine_logits

class LinearModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        exclude_bias_n_norm: bool,
        extra_optimizer_args: dict,
        scheduler: str,
        split_strategy: str,
        lr_decay_steps: Optional[Sequence[int]] = None,
        tasks: list = None,
        pooling_mode: str = 'whole',
        feature_dim: int = 256,
        classifier_type: str = 'linear', 
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        
        self.backbone = backbone
        self.pooling_mode = pooling_mode
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.feature_dim = feature_dim

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # -----------------------------------------------------------
        # [Strategy 설정]
        # -----------------------------------------------------------
        self.projector = None 
        self.noise_std = 0.1 

        # [수정] bottleneck 계열 통합 처리 (bottleneck, bottleneck_linear)
        if 'bottleneck' in classifier_type:
            concat_dim = feature_dim * 5 if pooling_mode == 'part' else feature_dim
            self.input_dim = feature_dim 
            
            # 공통 Projector
            self.projector = nn.Sequential(
                nn.Linear(concat_dim, self.input_dim, bias=False),
                nn.BatchNorm1d(self.input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5) 
            )
            
            # [분기] Cosine 여부 결정
            if classifier_type == 'bottleneck':
                self.classifier = CosineLinear(self.input_dim, num_classes)
                print(f"> Strategy: BOTTLENECK (Cosine) | {concat_dim} -> {self.input_dim} -> Cosine Classify")
            else:
                # bottleneck_linear
                self.classifier = nn.Linear(self.input_dim, num_classes)
                print(f"> Strategy: BOTTLENECK (Linear) | {concat_dim} -> {self.input_dim} -> Linear Classify")

        elif classifier_type == 'shared_cosine':
            self.input_dim = feature_dim
            if self.pooling_mode == 'part':
                self.part_scales = nn.Parameter(torch.ones(5))
            else:
                self.part_scales = None
            self.classifier = CosineLinear(self.input_dim, num_classes)
            print(f"> Strategy: SHARED COSINE | Input Dim: {self.input_dim}")
        
        else:
            # Normal Linear/Cosine
            if self.pooling_mode == 'part':
                self.input_dim = feature_dim * 5
            else:
                self.input_dim = feature_dim
            
            if classifier_type == 'cosine':
                self.classifier = CosineLinear(self.input_dim, num_classes)
            else:
                self.classifier = nn.Linear(self.input_dim, num_classes)
            print(f"> Strategy: {classifier_type.upper()} | Input Dim: {self.input_dim}")

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.split_strategy = split_strategy
        self.lr_decay_steps = lr_decay_steps
        self.tasks = tasks
        self.extra_args = kwargs

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("linear")
        SUPPORTED_NETWORKS = ['stgcn']
        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS, type=str)
        parser.add_argument("--zero_init_residual", action="store_true")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]
        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--exclude_bias_n_norm", action="store_true")
        SUPPORTED_SCHEDULERS = ["reduce", "cosine", "warmup_cosine", "step", "exponential", "none"]
        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        
        # [수정] bottleneck_linear 추가
        parser.add_argument("--classifier_type", type=str, default="linear", 
                            choices=["linear", "cosine", "shared_cosine", "bottleneck", "bottleneck_linear"])
        return parent_parser

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        if isinstance(X, list): X = X[0]
        if isinstance(X, torch.Tensor) and X.ndim == 4: X = X.unsqueeze(-1)

        with torch.no_grad():
            feats = self.backbone(X)

        # -----------------------------------------------------------
        # [Forward Strategy]
        # -----------------------------------------------------------
        # [수정] bottleneck 혹은 bottleneck_linear 둘 다 여기로 진입
        if 'bottleneck' in self.classifier_type:
            if isinstance(feats, list):
                feats = torch.cat(feats, dim=1) 
            
            if self.training:
                noise = torch.randn_like(feats) * self.noise_std
                feats = feats + noise
            
            projected_feats = self.projector(feats)
            logits = self.classifier(projected_feats)
            feats_ret = feats

        elif self.classifier_type == 'shared_cosine' and isinstance(feats, list):
            logits_list = []
            for i, part_feat in enumerate(feats):
                if self.training:
                     part_feat = part_feat + (torch.randn_like(part_feat) * self.noise_std)
                
                part_logit = self.classifier(part_feat)
                if hasattr(self, 'part_scales') and self.part_scales is not None:
                    part_logit = part_logit * self.part_scales[i]
                logits_list.append(part_logit)
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            feats_ret = torch.cat(feats, dim=1)
            
        else:
            if isinstance(feats, list):
                feats = torch.cat(feats, dim=1)
            logits = self.classifier(feats)
            feats_ret = feats

        return {"logits": logits, "feats": feats_ret}
    
    # ... (나머지 unpack_batch, step 함수들은 기존과 완전히 동일) ...
    def _unpack_batch(self, batch):
        x, y = None, None
        for item in batch:
            if isinstance(item, torch.Tensor) and item.ndim >= 3:
                x = item
            elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                if item[0].ndim >= 3:
                    x = item[0]
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                if item is not x:
                    y = item
        if x is None and len(batch) >= 1: x = batch[0]
        if y is None and len(batch) >= 2: y = batch[1]
        if x is None: raise ValueError(f"Batch Error")
        return x, y

    def shared_step(self, batch: Tuple, batch_idx: int) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        X, target = self._unpack_batch(batch)
        batch_size = X.size(0) if isinstance(X, torch.Tensor) else X[0].size(0)
        out = self(X)
        logits = out["logits"]
        loss = F.cross_entropy(logits, target)
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5, logits, target

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.backbone.eval() 
        _, loss, acc1, acc5, _, _ = self.shared_step(batch, batch_idx)
        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        if hasattr(self, 'part_scales') and self.part_scales is not None:
             for i, scale in enumerate(self.part_scales):
                 log[f"part_scale_{i}"] = scale
        self.log_dict(log, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        batch_size, loss, acc1, acc5, logits, targets = self.shared_step(batch, batch_idx)
        return {"batch_size": batch_size, "val_loss": loss, "val_acc1": acc1, "val_acc5": acc5, "logits": logits, "targets": targets}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        if not self.trainer.sanity_checking:
            preds = torch.cat([o["logits"].max(-1)[1] for o in outs]).cpu().numpy()
            targets = torch.cat([o["targets"] for o in outs]).cpu().numpy()
            mask_correct = preds == targets
            if self.split_strategy == "class" and self.tasks is not None:
                for task_idx, task in enumerate(self.tasks):
                    task_arr = np.array(task) if not isinstance(task, torch.Tensor) else task.cpu().numpy()
                    mask_task = np.isin(targets, task_arr)
                    if mask_task.sum() > 0:
                        correct_task = np.logical_and(mask_task, mask_correct).sum()
                        log[f"val_acc1_task{task_idx}"] = correct_task / mask_task.sum()
        self.log_dict(log, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer == "sgd": opt_cls = torch.optim.SGD
        elif self.optimizer == "adam": opt_cls = torch.optim.Adam
        elif self.optimizer == "adamw": opt_cls = torch.optim.AdamW
        else: raise ValueError(f"Unknown optimizer")

        params = list(self.classifier.parameters())
        if self.projector is not None:
            params.extend(list(self.projector.parameters()))
        if hasattr(self, 'part_scales') and self.part_scales is not None:
            params.append(self.part_scales)

        optimizer = opt_cls(params, lr=self.lr, weight_decay=self.weight_decay, **self.extra_optimizer_args)

        if self.lars: optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.exclude_bias_n_norm)
        if self.scheduler == "none": return optimizer
        
        if self.scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.max_epochs//10, max_epochs=self.max_epochs)
            return [optimizer], [scheduler]
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.max_epochs)
            return [optimizer], [scheduler]
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps or [], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=self.weight_decay)
            return [optimizer], [scheduler]
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}
        raise ValueError(f"Unknown scheduler")