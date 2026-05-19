from pieced.utils import (
    checkpointer,
    classification_dataloader,
    datasets,
    gather_layer,
    knn,
    lars,
    metrics,
    momentum,
    pretrain_dataloader,
    sinkhorn_knopp,
)

__all__ = [
    "classification_dataloader",
    "pretrain_dataloader",
    "checkpointer",
    "datasets",
    "gather_layer",
    "knn",
    "lars",
    "metrics",
    "momentum",
    "sinkhorn_knopp",
]


try:
    from pieced.utils import auto_umap  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("auto_umap")
