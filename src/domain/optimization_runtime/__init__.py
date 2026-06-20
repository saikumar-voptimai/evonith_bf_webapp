from domain.optimization_runtime.config_adapter import build_runtime_config

try:
    from domain.optimization_runtime.dataset_context import DatasetContextService
except ModuleNotFoundError as exc:
    # Loading persisted model artifacts should not require Streamlit. The app
    # runtime has Streamlit installed; this fallback is only for offline tests and
    # model-bundle validation in lightweight environments.
    if exc.name != "streamlit":
        raise
    DatasetContextService = None  # type: ignore[assignment]
from domain.optimization_runtime.feature_utils import (
    normalize_feature_name,
    parse_lag_feature_name,
)
from domain.optimization_runtime.feature_vector import FeatureVectorBuilder
from domain.optimization_runtime.model_bundle import ModelBundleService
from domain.optimization_runtime.optimizer_runner import OptimizerRunner
from domain.optimization_runtime.types import (
    FeatureBuildResult,
    ModelBundleInfo,
    ObjectiveResult,
    OptimizationContext,
    OptimizationResult,
)

__all__ = [
    "build_runtime_config",
    "DatasetContextService",
    "ModelBundleService",
    "FeatureVectorBuilder",
    "OptimizerRunner",
    "normalize_feature_name",
    "parse_lag_feature_name",
    "OptimizationContext",
    "ModelBundleInfo",
    "FeatureBuildResult",
    "ObjectiveResult",
    "OptimizationResult",
]
