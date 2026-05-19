from pieced.distillers.base import base_distill_wrapper
from pieced.distillers.predictive import predictive_distill_wrapper


__all__ = [
    "base_distill_wrapper",
    "predictive_distill_wrapper",
]

DISTILLERS = {
    "base": base_distill_wrapper,
    "predictive": predictive_distill_wrapper,
}
