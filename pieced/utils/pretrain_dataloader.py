# pretrain_dataloader.py
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pieced.utils.datasets import NTU60Dataset, KineticsSkeletonDataset

from pieced.utils.data_utils import (
    MulticropSkeletonTransform,
    SkeletonTransform,            # eval/online_eval 용
)

def split_dataset(
    dataset: Dataset, task_idx: List[int], num_tasks: int, split_strategy: str, tasks: list = None
):
    if split_strategy == "class":
        assert len(dataset.classes) == sum([len(t) for t in tasks])
        mask = [(c in tasks[task_idx]) for c in dataset.targets]
        indexes = torch.tensor(mask).nonzero()
        task_dataset = Subset(dataset, indexes)
    elif split_strategy == "data":
        assert tasks is None
        lengths = [len(dataset) // num_tasks] * num_tasks
        lengths[0] += len(dataset) - sum(lengths)
        task_dataset = torch.utils.data.random_split(
            dataset, lengths, generator=torch.Generator().manual_seed(5)
        )[task_idx]
    elif split_strategy == "domain":
        assert tasks is None
        raise NotImplementedError
    return task_dataset, tasks

def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)
    return DatasetWithIndex

# ====== transform factory (pretrain에서만 쓰는 멀티크롭) ======
def prepare_transform(dataset: str, multicrop: bool = False, **kwargs) -> Any:
    """
    multicrop=False → SkeletonTransform(target_len) (eval-like)
    multicrop=True  → MulticropSkeletonTransform(target_len, num_crops, shear_amplitude, temporal_padding_ratio)
    """
    target_len = kwargs.get("target_len", 64)
    if not multicrop:
        return SkeletonTransform(target_len=target_len)

    return MulticropSkeletonTransform(
        target_len=target_len,
        num_crops=kwargs.get("num_crops", 2),
        shear_amplitude=kwargs.get("shear_amplitude", 0.5),
        temporal_padding_ratio=kwargs.get("temporal_padding_ratio", 6),
    )

def prepare_datasets(
    dataset: str,
    task_transform: Callable,
    online_eval_transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    no_labels: Optional[Union[str, Path]] = False,
    encoder: Optional[str] = "stgcn_hisdi",
) -> Dataset:
    online_eval_dataset = None
    if dataset in ["ntu60", "ntu120", "pkuv1", "pkuv2"]:
        dataset = NTU60Dataset(data_path=data_dir, split="train", transform=task_transform)
        online_eval_dataset = NTU60Dataset(data_path=data_dir, split="train", transform=online_eval_transform)
    elif dataset == "kinetics400":
        dataset = KineticsSkeletonDataset(data_path=data_dir, split="train", transform=task_transform)
        online_eval_dataset = KineticsSkeletonDataset(data_path=data_dir, split="train", transform=online_eval_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return dataset, online_eval_dataset

def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
