#./trainer/main_pretrain.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pprint import pprint

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from pieced.args.setup import parse_args_pretrain
from pieced.methods import METHODS
from pieced.distillers import DISTILLERS

try:
    from pieced.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from pieced.utils.checkpointer import Checkpointer
from pieced.utils.classification_dataloader import prepare_data as prepare_data_classification
from pieced.utils.task_orders import load_task_splits
from pieced.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_transform,
    split_dataset,
    SkeletonTransform,
    MulticropSkeletonTransform,
)

def main():
    args = parse_args_pretrain()
    seed_everything(args.seed)
    if args.dataset in 'ntu60':
        args.num_joints = 25
        args.num_class = 60
        args.in_channels = 3
    elif args.dataset in 'ntu120':
        args.num_joints = 25
        args.num_class = 120
        args.in_channels = 3
    elif args.dataset in 'pkuv1':
        args.num_joints = 25
        args.num_class = 51
        args.in_channels = 3
    elif args.dataset in 'pkuv2':
        args.num_joints = 25
        args.num_class = 41
        args.in_channels = 3
    elif args.dataset == 'kinetics400':
        args.num_joints = 18
        args.num_class = 400
        args.in_channels = 3
            
    # online eval dataset reloads when task dataset is over
    args.multiple_trainloader_mode = "min_size"

    # set online eval batch size and num workers
    args.online_eval_batch_size = int(args.batch_size) if args.dataset in ["ntu60", "ntu120", "kinetics400", "pkuv1", "pkuv2"] else None

    # # split classes into tasks
    # tasks = None
    # if args.split_strategy == "class":
    #     # assert args.num_class % args.num_tasks == 0
    #     # tasks = torch.randperm(args.num_class).chunk(args.num_tasks)

    #     permuted_indices = torch.randperm(args.num_class).numpy()
    #     tasks = np.array_split(permuted_indices, args.num_tasks)
    #     tasks = [torch.from_numpy(t) for t in tasks]
    #     if args.task_reverse==True:
    #         tasks.reverse()

    # print(tasks)
    # split classes into tasks
    tasks = None
    if args.split_strategy == "class":
        if args.task_split_key is not None:
            tasks = load_task_splits(args.task_split_json, args.task_split_key)
            # Keep RNG advance identical to the legacy path so downstream
            # init / shuffle behaviour matches across runs with the same seed.
            _ = torch.randperm(args.num_class)
            assert len(tasks) == args.num_tasks, (
                f"task_split_key '{args.task_split_key}' yields {len(tasks)} tasks "
                f"but --num_tasks is {args.num_tasks}"
            )
        else:
            # Legacy randperm path (seed=5 for class assignment, args.seed for downstream).
            cpu_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )

            torch.manual_seed(5)
            permuted_indices = torch.randperm(args.num_class).numpy()

            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            _ = torch.randperm(args.num_class)

            tasks = np.array_split(permuted_indices, args.num_tasks)
            tasks = [torch.from_numpy(t) for t in tasks]

        # Apply optional reorder (1-based, e.g. "3,1,5,2,4")
        if args.task_order is not None:
            order_indices = [int(idx.strip()) - 1 for idx in args.task_order.split(',')]
            tasks = [tasks[i] for i in order_indices]

    print(tasks)

    # asymmetric augmentations
    if args.unique_augs > 1:
        transform = [
            prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
            for kwargs in args.transform_kwargs
        ]
    else:
        transform = prepare_transform(
            args.dataset, multicrop=args.multicrop, **args.transform_kwargs
        )

    if args.debug_augmentations:
        print("Transforms:")
        pprint(transform)


    if args.num_crops != 2:
        assert args.method == "wmse"
    
    if args.dataset in ["ntu60", "ntu120", "kinetics400", "pkuv1", "pkuv2"]:
        task_transform = MulticropSkeletonTransform()
        online_eval_transform = SkeletonTransform()  # 4D로 reshape + normalize만 수행

    train_dataset, online_eval_dataset = prepare_datasets(
        args.dataset,
        task_transform=task_transform,
        online_eval_transform=online_eval_transform,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        no_labels=args.no_labels,
        encoder=args.encoder
    )

    task_dataset, tasks = split_dataset(
        train_dataset,
        tasks=tasks,
        task_idx=args.task_idx,
        num_tasks=args.num_tasks,
        split_strategy=args.split_strategy,
    )

    task_loader = prepare_dataloader(
        task_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_loaders = {f"task{args.task_idx}": task_loader}

    if args.online_eval_batch_size:
        online_eval_loader = prepare_dataloader(
            online_eval_dataset,
            batch_size=args.online_eval_batch_size,
            num_workers=args.num_workers,
        )
        train_loaders.update({"online_eval": online_eval_loader})
    # normal dataloader for when it is available
    print(args.dataset, args.data_dir, args.train_dir, args.val_dir, args.batch_size, args.num_workers)
    _, val_loader = prepare_data_classification(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    
    print(args.in_channels, args.num_class, args.num_joints)
    
    # check method
    assert args.method in METHODS, f"Choose from {METHODS.keys()}" 
    # build method
    MethodClass = METHODS[args.method]
    if args.distiller:
        MethodClass = DISTILLERS[args.distiller](MethodClass)
    model = MethodClass(**args.__dict__, tasks=tasks if args.split_strategy == "class" else None)

    # only one resume mode can be true
    assert [args.resume_from_checkpoint, args.pretrained_model].count(True) <= 1

    if args.resume_from_checkpoint:
        pass  # handled by the trainer
    elif args.pretrained_model:
        print(f"Loading previous task checkpoint {args.pretrained_model}...")
        state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    callbacks = []
    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}-task{args.task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            reinit=True,
        )
        if args.task_idx == 0:
            wandb_logger.watch(model, log="gradients", log_freq=100)
            wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)
    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        # gradient_clip_val=1.0,          
        # gradient_clip_algorithm="norm",  
    )

    model.current_task_idx = args.task_idx
    trainer.fit(model, train_loaders, val_loader)


if __name__ == "__main__":
    main()