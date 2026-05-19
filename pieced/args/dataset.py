from argparse import ArgumentParser
from pathlib import Path


def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    SUPPORTED_DATASETS = [
        "ntu60",
        "ntu120",
        "kinetics400",
        "pkuv1",
        "pkuv2"
    ]

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)

    # dataset path
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--train_dir", type=Path, default=None)
    parser.add_argument("--val_dir", type=Path, default=None)

    # custom dataset only
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument("--semi_supervised", default=None, type=float)


def augmentations_args(parser: ArgumentParser):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # cropping
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--num_crops", type=int, default=2)
    parser.add_argument("--num_small_crops", type=int, default=0)

    # augmentations
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")

    # for imagenet or custom dataset
    parser.add_argument("--size", type=int, default=[224], nargs="+")

    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", action="store_true")
