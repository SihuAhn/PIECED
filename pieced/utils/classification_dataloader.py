# classification_dataloader.py
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from pieced.utils.datasets import NTU60Dataset, KineticsSkeletonDataset
from pieced.utils.data_utils import (
    build_shear_temporalcrop_pipeline,
    build_eval_pipeline,
)

# ====== transform factories (분류 전용) ======
def prepare_transforms(dataset: str, target_len: int = 64):
    """
    Train:  Resize → TemporalCrop → Shear
    Val:    Resize only
    """
    T_train = build_shear_temporalcrop_pipeline(
        target_len=target_len, shear_amplitude=0.5, temporal_padding_ratio=6
    )
    T_val = build_eval_pipeline(target_len=target_len)
    return T_train, T_val

def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    train_domain: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:

    if data_dir is None:
        sandbox_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_dir / "datasets"
    else:
        data_dir = Path(data_dir)

    if dataset in ["ntu60", "ntu120", "pkuv1", "pkuv2"]:
        train_dataset = NTU60Dataset(data_path=data_dir, split="train", transform=T_train)
        val_dataset = NTU60Dataset(data_path=data_dir, split="val", transform=T_val)
    elif dataset == "kinetics400":
        train_dataset = KineticsSkeletonDataset(data_path=data_dir, split="train", transform=T_train)
        val_dataset = KineticsSkeletonDataset(data_path=data_dir, split="val", transform=T_val)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_dataset, val_dataset

def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

def prepare_data(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    train_domain: str = None,
    semi_supervised: float = None,
) -> Tuple[DataLoader, DataLoader]:

    T_train, T_val = prepare_transforms(dataset)
    train_dataset, val_dataset = prepare_datasets(
        dataset, T_train, T_val, data_dir=data_dir, train_dir=train_dir, val_dir=val_dir, train_domain=train_domain
    )

    if semi_supervised!=None and semi_supervised<1.0:
        from torch.utils.data import Subset
        idxs = train_test_split(
            range(len(train_dataset)),
            train_size=semi_supervised,
            stratify=train_dataset.targets,
            random_state=5,
        )[0]
        train_dataset = Subset(train_dataset, idxs)

    return prepare_dataloaders(train_dataset, val_dataset, batch_size=batch_size, num_workers=num_workers)
