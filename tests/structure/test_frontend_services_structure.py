"""Structure checks for canonical frontend services, config, UI, and shims."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]

SERVICE_MODULES = [
    "api_client",
    "api_errors",
    "backend_status",
    "auth_api",
    "admin_api",
    "data_api",
    "dataset_api",
    "feedback_api",
    "material_balance_api",
    "recommendations_api",
    "blend_optimizer_api",
    "copilot_api",
    "furnacemind_api",
    "status_api",
    "ops_api",
]

CONFIG_MODULES = [
    "config_loader",
    "frontend_settings",
    "page_registry",
]

UI_MODULES = [
    "backend_status_badge",
    "login_page",
    "streamlit_fragments",
    "styles",
    "bmo.components",
    "bmo.editor_inputs",
    "furnacemind.reports",
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(str(node.module or ""))
    return modules


def _assert_no_forbidden_imports(
    *,
    roots: set[str],
    prefixes: set[str] | None = None,
) -> None:
    service_dirs = [
        REPO_ROOT / "apps" / "frontend_streamlit" / "services",
        REPO_ROOT / "src" / "services",
    ]
    prefixes = prefixes or set()
    failures: list[str] = []
    for service_dir in service_dirs:
        for module_name in SERVICE_MODULES:
            path = service_dir / f"{module_name}.py"
            for module in _imports(path):
                root = module.split(".", 1)[0]
                if root in roots:
                    failures.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
                if any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes):
                    failures.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert failures == []


def _discard_incomplete_utils_session_stub() -> None:
    module = sys.modules.get("utils.session")
    if module is not None and not hasattr(module, "login_user"):
        sys.modules.pop("utils.session", None)


def test_canonical_frontend_services_import():
    for module_name in SERVICE_MODULES:
        module = importlib.import_module(f"apps.frontend_streamlit.services.{module_name}")
        assert module is not None


def test_old_src_services_wrappers_import():
    for module_name in SERVICE_MODULES:
        module = importlib.import_module(f"src.services.{module_name}")
        assert module is not None


def test_canonical_config_imports():
    for module_name in CONFIG_MODULES:
        module = importlib.import_module(f"apps.frontend_streamlit.config.{module_name}")
        assert module is not None


def test_old_src_config_wrappers_import():
    for module_name in CONFIG_MODULES:
        module = importlib.import_module(f"src.config.{module_name}")
        assert module is not None


def test_canonical_ui_imports():
    _discard_incomplete_utils_session_stub()
    for module_name in UI_MODULES:
        module = importlib.import_module(f"apps.frontend_streamlit.ui.{module_name}")
        assert module is not None


def test_old_src_ui_wrappers_import():
    _discard_incomplete_utils_session_stub()
    for module_name in UI_MODULES:
        module = importlib.import_module(f"src.ui.{module_name}")
        assert module is not None


def test_frontend_api_adapters_do_not_import_backend_internals():
    _assert_no_forbidden_imports(
        roots={"app", "data", "domain", "furnace_data"},
        prefixes={"apps.backend_api"},
    )


def test_frontend_api_adapters_do_not_import_db_clients():
    _assert_no_forbidden_imports(
        roots={"influxdb3", "psycopg2", "sqlalchemy", "sqlite3"}
    )


def test_frontend_api_adapters_do_not_import_qdrant_clients():
    _assert_no_forbidden_imports(roots={"qdrant_client"})


def test_frontend_api_adapters_do_not_import_llm_provider_sdks():
    _assert_no_forbidden_imports(
        roots={"anthropic", "langchain", "langchain_openai", "openai"}
    )


def test_frontend_api_adapters_do_not_import_model_loaders():
    _assert_no_forbidden_imports(
        roots={
            "easyocr",
            "joblib",
            "paddleocr",
            "sentence_transformers",
            "sklearn",
            "torch",
            "xgboost",
        }
    )
