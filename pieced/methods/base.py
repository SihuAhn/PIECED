import operator
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import (
    LinearWarmupCosineAnnealingLR,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from pieced.backbone.skeleton_info import BONES_INFO
from pieced.utils.knn import WeightedKNNClassifier
from pieced.utils.lars import LARSWrapper
from pieced.utils.metrics import accuracy_at_k, weighted_mean
from pieced.utils.momentum import MomentumUpdater, initialize_momentum_params


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    """Applies static learning rates to specific parameter groups."""
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.
        """
        super().__init__()
        # Save all hyperparameters (lr, batch_size, etc.) to self.hparams
        # You can access them via self.hparams.lr, self.hparams.batch_size, etc.
        self.save_hyperparameters()

        # Handle derived or conditional parameters
        self.hparams.num_small_crops = (
            self.hparams.num_small_crops if self.hparams.multicrop else 0
        )
        self.online_eval = self.hparams.online_eval_batch_size is not None

        # Scale LRs if accumulating gradients
        if self.hparams.accumulate_grad_batches:
            self.hparams.lr *= self.hparams.accumulate_grad_batches
            self.hparams.classifier_lr *= self.hparams.accumulate_grad_batches
            self.hparams.min_lr *= self.hparams.accumulate_grad_batches
            self.hparams.warmup_start_lr *= self.hparams.accumulate_grad_batches

        self.domains = [
            "real", "quickdraw", "painting", "sketch", "infograph", "clipart"
        ]

        # Build model components
        self._build_model()

    def _build_model(self):
        """Initializes the encoder, classifier, and KNN."""
        assert self.hparams.encoder in ["stgcn"]
        from pieced.backbone.stgcn import STGCN

        self.base_model = {"stgcn": STGCN}[self.hparams.encoder]

        if self.hparams.encoder == "stgcn":
            # 데이터셋 이름에 따라 Graph Layout 자동 매핑
            dataset_name = self.hparams.dataset
            
            if 'kinetics' in dataset_name or 'openpose' in dataset_name:
                layout = 'openpose'
            elif 'ntu' in dataset_name: # ntu60, ntu120
                layout = 'ntu-rgb+d'
            elif 'pku' in dataset_name: # pkuv1, pkuv2
                layout = 'ntu-rgb+d' # PKU use same layout as NTU (Kinect v2)
            else:
                # 기본값 혹은 사용자가 지정한 값 (self.hparams.graph_layout)
                layout = getattr(self.hparams, 'graph_layout', 'ntu-rgb+d')

            print(f"[BaseModel] Auto-detected layout for {dataset_name}: {layout}")

        if self.hparams.encoder == "stgcn":
           
            # Create graph_args dict from hparams
            graph_args = {
                "layout": layout,
                "strategy": getattr(self.hparams, 'graph_strategy', 'spatial')
            }

            self.encoder = self.base_model(
                self.hparams.in_channels,
                self.hparams.hidden_channels,
                self.hparams.hidden_dim,
                self.hparams.num_class,
                graph_args,  # Pass the constructed dict
                self.hparams.edge_importance_weighting,
                self.hparams.dropout,
                pooling_mode=self.hparams.pooling_mode,
                # ==========================================
                # 🚨 [수정 1] pool_type을 kwargs에서 꺼내서 STGCN에 전달
                # ==========================================
                pool_type=getattr(self.hparams, 'pool_type', 'max'),
            )
            self.features_dim = (
                self.encoder.fc.in_features
                if hasattr(self.encoder.fc, "in_features")
                else 256
            )
            if self.hparams.pooling_mode == "part":
                self.features_dim *= self.hparams.num_parts

            self.encoder.fc = nn.Identity()

        self.classifier = nn.Linear(self.features_dim, self.hparams.num_class)

        if not self.hparams.disable_knn_eval:
            self.knn = WeightedKNNClassifier(
                k=self.hparams.knn_k, distance_fx="euclidean"
            )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods."""
        parser = parent_parser.add_argument_group("base")

        # encoder args
        SUPPORTED_NETWORKS = ["stgcn"]
        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS, type=str)
        parser.add_argument("--zero_init_residual", action="store_true")

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]
        parser.add_argument(
            "--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True
        )
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce", "cosine", "warmup_cosine", "step", "exponential", "none"
        ]
        parser.add_argument(
            "--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # knn eval
        parser.add_argument("--disable_knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=20, type=int)

        return parent_parser

    @property
    def current_task_idx(self) -> int:
        return getattr(self, "_current_task_idx", None)

    @current_task_idx.setter
    def current_task_idx(self, new_task):
        if hasattr(self, "_current_task_idx"):
            assert new_task >= self._current_task_idx
        self._current_task_idx = new_task

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class."""
        return [
            {"name": "encoder", "params": self.encoder.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.hparams.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Builds the optimizer."""
        optimizer_name = self.hparams.optimizer
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{optimizer_name} not in (sgd, adam, adamw)")

        optimizer = optimizer(
            self.learnable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            **getattr(self.hparams, "extra_optimizer_args", {}),
        )
        if getattr(self.hparams, "lars", False):
            optimizer = LARSWrapper(
                optimizer,
                eta=self.hparams.eta_lars,
                clip=self.hparams.grad_clip_lars,
                exclude_bias_n_norm=self.hparams.exclude_bias_n_norm,
            )
        return optimizer

    def _get_scheduler(self, optimizer: torch.optim.Optimizer):
        """Builds the learning rate scheduler."""
        scheduler_name = self.hparams.scheduler
        if scheduler_name == "none":
            return None
        elif scheduler_name == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.max_epochs,
                warmup_start_lr=self.hparams.warmup_start_lr,
                eta_min=self.hparams.min_lr,
            )
        elif scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, self.hparams.max_epochs, eta_min=self.hparams.min_lr
            )
        elif scheduler_name == "step":
            scheduler = MultiStepLR(optimizer, self.hparams.lr_decay_steps)
        else:
            raise ValueError(
                f"{scheduler_name} not in (warmup_cosine, cosine, step, none)"
            )
        return scheduler

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer and learning rate scheduler."""
        # Get parameter groups that should not have scheduler applied
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)

        if scheduler is None:
            return [optimizer]

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.hparams.lr] * len(idxs_no_scheduler),
            )
            scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def forward(self, *args, **kwargs) -> Dict:
        """Dummy forward, calls base forward."""
        return self.base_forward(*args, **kwargs)

    def base_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Basic forward pass.
        Handles all pooling modes ('whole', 'part', 'hybrid') from STGCN.
        Returns a dict with:
            - "feats": A single tensor for linear evaluation.
                       (global_feat or avg(part_feats))
            - "all_feats": The raw output from the encoder.
                           (Tensor, List[Tensor], or Dict)
        """
        raw_features = self.encoder(X)

        if getattr(self.hparams, "pooling_mode", "whole") == "hybrid":
            # raw_features is Dict{'global': T, 'parts': List[T]}
            eval_feats = raw_features["global"]
            all_feats = raw_features
        elif getattr(self.hparams, "pooling_mode", "whole") == "part":
            # raw_features is List[T]
            eval_feats = torch.cat(raw_features, dim=1)
            all_feats = raw_features
        else:  # 'whole'
            # raw_features is T
            eval_feats = raw_features
            all_feats = raw_features

        return {"feats": eval_feats, "all_feats": all_feats}

    def _online_eval_shared_step(self, X: torch.Tensor, targets) -> Dict:
        """Forwards a batch for online evaluation."""
        with torch.no_grad():
            outs = self.base_forward(X)
        
        # 'feats' key now always contains the correct tensor for the classifier
        feats = outs["feats"].detach()
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        return {
            **outs,
            "logits": logits,
            "loss": loss,
            "acc1": acc1.detach(),
            "acc5": acc5.detach(),
        }

    def _get_features(self, X_task: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Helper to run all augmented views through the encoder and format the output.
        Handles 'whole', 'part', and 'hybrid' modes.
        """
        all_features = [self.encoder(x) for x in X_task]

        is_dict_features = isinstance(all_features[0], dict)
        is_list_features = isinstance(all_features[0], list)
        
        main_features = all_features[: self.hparams.num_crops]

        if is_dict_features:
            # Hybrid mode: group features by key
            keys = main_features[0].keys()
            feats = {k: [dic[k] for dic in main_features] for k in keys}
        
        elif is_list_features:
            # Part mode: feats is a List of Lists
            feats = main_features 
        
        else:
            # Whole mode: list of tensors
            feats = main_features

        outs_task = {"feats": feats}

        # Add small crops if multicrop is enabled
        if getattr(self.hparams, "multicrop", False):
            small_crop_feats = all_features[self.hparams.num_crops :]
            if is_dict_features:
                for k in keys:
                    outs_task["feats"][k].extend([dic[k] for dic in small_crop_feats])
            elif is_list_features:
                outs_task["feats"].extend(small_crop_feats)
            else:
                outs_task["feats"].extend(small_crop_feats)

        return outs_task


    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning."""
        _, X_task, _ = batch[f"task{self.current_task_idx}"]
        X_task = [X_task] if isinstance(X_task, torch.Tensor) else X_task

        assert len(X_task) == self.hparams.num_crops + getattr(self.hparams, "num_small_crops", 0)

        # 1. Get features from all augmented views
        outs_task = self._get_features(X_task)

        # 2. Handle online evaluation
        if self.online_eval:
            *_, X_online_eval, targets_online_eval = batch["online_eval"]
            outs_online_eval = self._online_eval_shared_step(
                X_online_eval, targets_online_eval
            )
            outs_online_eval = {
                f"online_eval_{k}": v for k, v in outs_online_eval.items()
            }

            metrics = {
                "train_online_eval_loss": outs_online_eval["online_eval_loss"],
                "train_online_eval_acc1": outs_online_eval["online_eval_acc1"],
                "train_online_eval_acc5": outs_online_eval["online_eval_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            if (
                not getattr(self.hparams, "disable_knn_eval", False)
                and not self.trainer.sanity_checking
                and self.knn.has_train_features
            ):
                self.knn(
                    train_features=outs_online_eval["online_eval_feats"].detach(),
                    train_targets=targets_online_eval,
                )

            loss = outs_online_eval.pop("online_eval_loss")
            return {**outs_task, **outs_online_eval, "loss": loss}
        
        return {**outs_task, "loss": 0}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step."""
        if not self.online_eval:
            return {}

        *_, X, targets = batch
        batch_size = targets.size(0)
        out = self._online_eval_shared_step(X, targets)

        if not getattr(self.hparams, "disable_knn_eval", False) and getattr(self, "knn", None) and self.knn.has_train_features:
            self.knn.update(test_features=out["feats"].detach(), test_targets=targets)

        metrics = {
            "batch_size": batch_size,
            "targets": targets,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        if getattr(self.hparams, "split_strategy", None) == "domain" and len(batch) == 3:
            metrics["domains"] = batch[0]

        return {**metrics, **out}

    def _log_split_metrics(self, outs: List[Dict[str, Any]], log: Dict[str, Any]):
        """Helper to log task/domain-specific metrics."""
        preds = torch.cat([o["logits"].max(-1)[1] for o in outs]).cpu().numpy()
        targets = torch.cat([o["targets"] for o in outs]).cpu().numpy()
        mask_correct = preds == targets

        if getattr(self.hparams, "split_strategy", None) == "class":
            assert getattr(self.hparams, "tasks", None) is not None
            for task_idx, task in enumerate(self.hparams.tasks):
                mask_task = np.isin(targets, np.array(task))
                if mask_task.sum() > 0:
                    correct_task = np.logical_and(mask_task, mask_correct).sum()
                    log[f"val_acc1_task{task_idx}"] = correct_task / mask_task.sum()

        elif getattr(self.hparams, "split_strategy", None) == "domain":
            assert getattr(self.hparams, "tasks", None) is None
            domains = [o["domains"] for o in outs]
            domains_flat = [d for sublist in domains for d in sublist]
            domains = np.array(domains_flat)

            for task_idx, domain in enumerate(self.domains):
                mask_domain = np.isin(domains, np.array([domain]))
                if mask_domain.sum() > 0:
                    correct_domain = np.logical_and(mask_domain, mask_correct).sum()
                    log[f"val_acc1_{domain}_{task_idx}"] = (
                        correct_domain / mask_domain.sum()
                    )

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages validation metrics."""
        if not self.online_eval or not outs:
            return

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if not self.trainer.sanity_checking:
            self._log_split_metrics(outs, log)

            if not getattr(self.hparams, "disable_knn_eval", False) and getattr(self, "knn", None) and self.knn.has_train_features:
                val_knn_acc1, val_knn_acc5 = self.knn.compute()
                log.update(
                    {"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5}
                )

        self.log_dict(log, sync_dist=True)


class BaseMomentumModel(BaseModel):
    def __init__(self, **kwargs):
        """Base model for methods that use a momentum encoder."""
        super().__init__(**kwargs)
        
        self._build_momentum_model()

        # momentum updater
        self.momentum_updater = MomentumUpdater(
            self.hparams.base_tau_momentum, self.hparams.final_tau_momentum
        )

    def _build_momentum_model(self):
        """Initializes the momentum encoder and classifier."""
        if self.hparams.encoder == "stgcn":
            dataset_name = self.hparams.dataset
            
            if 'kinetics' in dataset_name or 'openpose' in dataset_name:
                layout = 'openpose'
            elif 'ntu' in dataset_name:
                layout = 'ntu-rgb+d'
            elif 'pku' in dataset_name:
                layout = 'ntu-rgb+d'
            else:
                layout = getattr(self.hparams, 'graph_layout', 'ntu-rgb+d')

            # Create graph_args dict from hparams
            graph_args = {
                "layout": layout,
                "strategy": getattr(self.hparams, 'graph_strategy', 'spatial')
            }

            self.momentum_encoder = self.base_model(
                self.hparams.in_channels,
                self.hparams.hidden_channels,
                self.hparams.hidden_dim,
                self.hparams.num_class,
                graph_args, # Pass the constructed dict
                self.hparams.edge_importance_weighting,
                self.hparams.dropout,
                pooling_mode=self.hparams.pooling_mode,
                # ==========================================
                # 🚨 [수정 2] pool_type을 kwargs에서 꺼내서 STGCN에 전달
                # ==========================================
                pool_type=getattr(self.hparams, 'pool_type', 'max'),
            )
            self.momentum_encoder.fc = nn.Identity()
        
        initialize_momentum_params(self.encoder, self.momentum_encoder)
        
        # features_dim is already scaled by num_parts in BaseModel._build_model
        # momentum classifier
        self.momentum_classifier = (
            nn.Linear(self.features_dim, self.hparams.num_class)
            if getattr(self.hparams, "momentum_classifier", False)
            else None
        )

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the learnable parameters."""
        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.hparams.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines momentum pairs that will be updated."""
        return [(self.encoder, self.momentum_encoder)]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds momentum-specific arguments."""
        parent_parser = super(
            BaseMomentumModel, BaseMomentumModel
        ).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("base_momentum")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", action="store_true")

        # ST-GCN args (moved here as they are backbone-specific)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--hidden_channels", type=int, default=64)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--edge_importance_weighting", action="store_true")
        parser.add_argument("--graph_layout", type=str, default="ntu-rgb+d")
        parser.add_argument("--graph_strategy", type=str, default="spatial")
        
        # PIECED: Add pooling_mode and num_parts here
        parser.add_argument("--pooling_mode", type=str, default="whole",
                            choices=["whole", "part", "hybrid"],
                            help="Pooling strategy for STGCN output.")
        parser.add_argument("--num_parts", type=int, default=5,
                            help="Number of body parts (used for 'part' and 'hybrid' modes).")
        
        # [수정 3] Argument Parser에도 pool_type을 명시적으로 추가하여 안전하게 받아옵니다.
        parser.add_argument("--pool_type", type=str, default="max", choices=["max", "avg"],
                            help="Pooling type for extracting feature (max or avg)")

        return parent_parser

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        super().on_train_start()
        self.last_step = 0

    @torch.no_grad()
    def base_forward_momentum(self, X: torch.Tensor) -> Dict:
        """Forward pass through the momentum encoder."""
        raw_features = self.momentum_encoder(X)

        if getattr(self.hparams, "pooling_mode", "whole") == "hybrid":
            eval_feats = raw_features["global"]
            all_feats = raw_features
        elif getattr(self.hparams, "pooling_mode", "whole") == "part":
            eval_feats = torch.cat(raw_features, dim=1)
            all_feats = raw_features
        else:  # 'whole'
            eval_feats = raw_features
            all_feats = raw_features

        return {"feats": eval_feats, "all_feats": all_feats}


    def _online_eval_shared_step_momentum(
        self, X: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, Any]:
        """Forward pass for momentum online evaluation."""
        out = self.base_forward_momentum(X)

        if self.momentum_classifier is not None:
            feats = out["feats"]
            logits = self.momentum_classifier(feats)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
            out.update(
                {"logits": logits, "loss": loss, "acc1": acc1.detach(), "acc5": acc5.detach()}
            )
        return out

    def _get_momentum_features(self, X_task: List[torch.Tensor]) -> Dict[str, Any]:
        """Helper to run views through the momentum encoder."""
        all_momentum_features = [self.momentum_encoder(x) for x in X_task]

        is_dict_features = isinstance(all_momentum_features[0], dict)
        is_list_features = isinstance(all_momentum_features[0], list)
        
        if is_dict_features:
            # Hybrid mode
            keys = all_momentum_features[0].keys()
            momentum_feats = {
                k: [dic[k] for dic in all_momentum_features] for k in keys
            }
        elif is_list_features:
            # Part mode: List of Lists
            momentum_feats = all_momentum_features
        else:
            # Whole mode
            momentum_feats = all_momentum_features
        
        return {"momentum_feats": momentum_feats}

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for momentum models."""
        outs_parent = super().training_step(batch, batch_idx)

        _, X_task, _ = batch[f"task{self.current_task_idx}"]
        X_task = [X_task] if isinstance(X_task, torch.Tensor) else X_task

        # Only use main crops for momentum features (as in original)
        X_task_main_crops = X_task[: self.hparams.num_crops]

        # 1. Get momentum features
        outs_task = self._get_momentum_features(X_task_main_crops)

        # 2. Handle online eval for momentum classifier
        if self.online_eval:
            *_, X_online_eval, targets_online_eval = batch["online_eval"]
            outs_online_eval = self._online_eval_shared_step_momentum(
                X_online_eval, targets_online_eval
            )
            outs_online_eval = {
                f"online_eval_momentum_{k}": v for k, v in outs_online_eval.items()
            }

            if self.momentum_classifier is not None:
                metrics = {
                    "train_online_eval_momentum_class_loss": outs_online_eval[
                        "online_eval_momentum_loss"
                    ],
                    "train_online_eval_momentum_acc1": outs_online_eval[
                        "online_eval_momentum_acc1"
                    ],
                    "train_online_eval_momentum_acc5": outs_online_eval[
                        "online_eval_momentum_acc5"
                    ],
                }
                self.log_dict(metrics, on_epoch=True, sync_dist=True)

                # Add momentum classifier loss to main loss
                outs_parent["loss"] += outs_online_eval.pop(
                    "online_eval_momentum_loss"
                )
            
            return {**outs_parent, **outs_task, **outs_online_eval}
        
        return {**outs_parent, **outs_task}

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int
    ):
        """Performs the momentum update."""
        if self.trainer.global_step > self.last_step:
            # Update momentum encoder and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            
            # Log tau
            self.log("tau", self.momentum_updater.cur_tau)
            
            # Update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step
                * getattr(self.trainer, "accumulate_grad_batches", 1),
                max_steps=len(self.trainer.train_dataloader)
                * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for momentum models."""
        if not self.online_eval:
            return {}, {}

        parent_metrics = super().validation_step(batch, batch_idx)

        *_, X, targets = batch
        batch_size = targets.size(0)
        out = self._online_eval_shared_step_momentum(X, targets)

        metrics = {}
        if self.momentum_classifier is not None:
            metrics = {
                "batch_size": batch_size,
                "momentum_val_loss": out["loss"],
                "momentum_val_acc1": out["acc1"],
                "momentum_val_acc5": out["acc5"],
            }
        return parent_metrics, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages validation metrics for momentum models."""
        if not self.online_eval or not outs:
            return

        parent_outs = [out[0] for out in outs if out[0]]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifier is not None:
            momentum_outs = [out[1] for out in outs if out[1]]
            if not momentum_outs:
                return

            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            val_acc1 = weighted_mean(
                momentum_outs, "momentum_val_acc1", "batch_size"
            )
            val_acc5 = weighted_mean(
                momentum_outs, "momentum_val_acc5", "batch_size"
            )

            log = {
                "momentum_val_loss": val_loss,
                "momentum_val_acc1": val_acc1,
                "momentum_val_acc5": val_acc5,
            }
            self.log_dict(log, sync_dist=True)