"""Configuration adapter for shared optimization runtime services.

This module normalizes page-level configuration into the dataset, model bundle,
feature policy, and optimizer blocks consumed by BMO runtime services.
"""

from __future__ import annotations

from typing import Any, Mapping


def build_runtime_config(
    raw_cfg: Mapping[str, Any] | None,
    *,
    default_dataset_path: str | None = None,
    default_model_bundle: Mapping[str, Any] | None = None,
    default_optimizer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return a normalized optimization-runtime configuration.

    Page-level settings can define dataset, model, feature-policy, and optimizer
    blocks in different places. This adapter merges those values with explicit
    defaults so downstream services can read one predictable runtime structure.

    Args:
         - raw_cfg: Mapping[str, Any] | None - Page or feature configuration.
         - default_dataset_path: str | None - Dataset path fallback.
         - default_model_bundle: Mapping[str, Any] | None - Model bundle fallback.
         - default_optimizer: Mapping[str, Any] | None - Optimizer fallback.

    Returns:
         - return dict[str, Any] - Normalized dataset, model, feature, and optimizer config.
    """

    cfg = dict(raw_cfg or {})
    runtime = dict(cfg.get("optimization_runtime", {}) or {})

    data_sources = dict(cfg.get("data_sources", {}) or {})
    dataset = dict(runtime.get("dataset", {}) or {})
    dataset.setdefault(
        "static_dataset_path",
        data_sources.get("static_dataset_path")
        or cfg.get("DATA")
        or default_dataset_path,
    )
    dataset.setdefault("refresh_enabled", bool(dataset.get("refresh_enabled", False)))
    dataset.setdefault(
        "refresh_rm_choice", str(dataset.get("refresh_rm_choice", "RM Charge"))
    )

    model_bundle = dict(runtime.get("model_bundle", {}) or {})
    if not model_bundle and default_model_bundle:
        model_bundle.update(default_model_bundle)
    if not model_bundle and cfg.get("model_bundle"):
        model_bundle.update(dict(cfg.get("model_bundle", {}) or {}))

    feature_policy = dict(runtime.get("feature_policy", {}) or {})
    feature_policy.setdefault("missing_feature_policy", "default_warn")

    optimizer = dict(runtime.get("optimizer", {}) or {})
    if not optimizer and default_optimizer:
        optimizer.update(default_optimizer)
    if not optimizer and cfg.get("optimization"):
        optimizer.update(dict(cfg.get("optimization", {}) or {}))
    optimizer.setdefault("strategy", "best1bin")
    optimizer.setdefault("maxiter", 4)
    optimizer.setdefault("popsize", 4)
    optimizer.setdefault("tol", 0.03)
    optimizer.setdefault("polish", False)
    optimizer.setdefault("seed", 42)

    return {
        "dataset": dataset,
        "model_bundle": model_bundle,
        "feature_policy": feature_policy,
        "optimizer": optimizer,
    }
