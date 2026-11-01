"""Hugging Face checkpoint support: config, names, weights (S1.2).

Mirrors the deepinfer/hugging_face/ package. DeepInfer re-exports lazily
through ``__getattr__``; plain imports are enough here.
"""

from mini_deepinfer.hugging_face.config import HFModelConfig, load_hf_model_config
from mini_deepinfer.hugging_face.loading import (
    cover_tied_parameters,
    prepare_hf_coverage,
    validate_hf_shapes,
)
from mini_deepinfer.hugging_face.parameters import model_params
from mini_deepinfer.hugging_face.weights import HFCoverage, load_covered_params

__all__ = [
    "HFCoverage",
    "HFModelConfig",
    "cover_tied_parameters",
    "load_covered_params",
    "load_hf_model_config",
    "model_params",
    "prepare_hf_coverage",
    "validate_hf_shapes",
]
