from copy import deepcopy
from typing import Any, Sequence, Tuple, List, Dict
import torch
from torch import nn


def base_distill_wrapper(Method=object):
    class BaseDistillWrapper(Method):
        def __init__(self, **kwargs) -> None:
            # Call super() first to initialize self.hparams
            super().__init__(**kwargs)

            self.output_dim = kwargs["output_dim"]
            
            # Get pooling_mode and num_parts from hparams (saved by BaseModel)
            self.pooling_mode = self.hparams.pooling_mode
            self.num_parts = self.hparams.num_parts

            # Deepcopy the encoder and projector
            # self.projector is already correctly structured (Sequential or ModuleList)
            # by byol.py's __init__
            self.frozen_encoder = deepcopy(self.encoder)
            self.frozen_projector = deepcopy(self.projector)

        def on_train_start(self):
            """Called at the beginning of training."""
            super().on_train_start()

            # After task 0, create a new frozen model from the updated online model
            if self.current_task_idx > 0:
                self.frozen_encoder = deepcopy(self.encoder)
                self.frozen_projector = deepcopy(self.projector)

                # Freeze all parameters in the frozen models
                for pg in self.frozen_encoder.parameters():
                    pg.requires_grad = False
                for pg in self.frozen_projector.parameters():
                    pg.requires_grad = False

        @torch.no_grad()
        def frozen_forward(self, X: torch.Tensor) -> Tuple[Any, Any]:
            """
            Performs forward pass through the frozen encoder and projector,
            handling 'whole', 'part', and 'hybrid' modes.
            """
            self.frozen_encoder.eval()
            self.frozen_projector.eval()
            
            # Get the raw output (Tensor, List, or Dict) directly from the STGCN model
            feats = self.frozen_encoder(X)

            # 2. Apply frozen projector based on pooling_mode
            if self.pooling_mode == "whole":
                # feats: Tensor
                # self.frozen_projector: nn.Sequential
                frozen_z = self.frozen_projector(feats)
                return feats, frozen_z

            elif self.pooling_mode == "part":
                # feats: List[Tensor] (length 5)
                # self.frozen_projector: nn.ModuleList (length 5)
                frozen_z = [
                    self.frozen_projector[i](feats[i]) for i in range(self.num_parts)
                ]
                return feats, frozen_z # Return List, List

            elif self.pooling_mode == "hybrid":
                # feats: Dict{'global': T, 'parts': List[T]}
                # self.frozen_projector: nn.ModuleList (length 6, [global] + [5 parts])
                global_feat = feats["global"]
                part_feats = feats["parts"]
                
                # self.frozen_projector[0] is global_momentum_projector
                frozen_z_global = self.frozen_projector[0](global_feat) 
                
                # self.frozen_projector[1:] is part_momentum_projectors
                frozen_z_parts = [
                    self.frozen_projector[i+1](part_feats[i]) for i in range(self.num_parts)
                ]
                
                # Match BYOL's 'z' output structure: [global, p1, ..., p5]
                frozen_z = [frozen_z_global] + frozen_z_parts
                return feats, frozen_z # Return Dict, List

            else:
                raise ValueError(f"Unknown pooling_mode in frozen_forward: {self.pooling_mode}")

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            _, X, _ = batch[f"task{self.current_task_idx}"]
            
            # Assuming num_crops is 2
            X1, X2 = X[0], X[1] 

            # 1. Call super().training_step() (e.g., byol.py's training_step)
            # This computes the SSL loss and returns out["z"]
            out = super().training_step(batch, batch_idx)

            # 2. Call frozen_forward, which now correctly handles all pooling modes
            frozen_feats1, frozen_z1 = self.frozen_forward(X1)
            frozen_feats2, frozen_z2 = self.frozen_forward(X2)

            # 3. Update the output dict
            # The structure of frozen_z[i] will now match out["z"][i]
            # (whole -> T), (part -> List[T]), (hybrid -> List[T])
            out.update(
                {"frozen_feats": [frozen_feats1, frozen_feats2], "frozen_z": [frozen_z1, frozen_z2]}
            )
            return out

    return BaseDistillWrapper