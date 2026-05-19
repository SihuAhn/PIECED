from pieced.methods.base import BaseModel
from pieced.methods.byol import BYOL
from pieced.methods.linear import LinearModel

METHODS = {
    # base classes
    "base": BaseModel,
    "linear": LinearModel,
    # methods
    "byol": BYOL,
}
__all__ = [
    "BYOL",
    "BaseModel",
    "LinearModel",
]
