"""The Jetson's Caddyfile (#279): valid, and its routes are those the console needs.

Needs the caddy binary (PATH, or ``BF_CADDY_BIN``); skipped without it, failed instead with
``BF_TEST_REQUIRE_SERVICES=1`` (CI). The tests read the configuration Caddy derives from the file (``caddy adapt``),
so they check what Caddy will do rather than the text of the file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CADDYFILE = ROOT / "deploy" / "jetson" / "Caddyfile"
CADDY = os.environ.get("BF_CADDY_BIN") or shutil.which("caddy")
pytestmark = pytest.mark.requires_service(CADDY is not None, "caddy is not installed (or set BF_CADDY_BIN)")


def _run(*args: str, env: dict[str, str] | None = None, tmp: Path) -> subprocess.CompletedProcess[str]:
    assert CADDY is not None
    clean = {k: v for k, v in os.environ.items() if not k.startswith("BF_")}
    environment = {**clean, "XDG_DATA_HOME": str(tmp / "data"), "XDG_CONFIG_HOME": str(tmp / "config"), **(env or {})}
    return subprocess.run(
        [CADDY, *args, "--config", str(CADDYFILE), "--adapter", "caddyfile"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )


def _adapt(tmp: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    done = _run("adapt", env=env, tmp=tmp)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _route(config: dict[str, Any], path: str) -> dict[str, Any]:
    """The route of the site that matches ``path`` exactly as written in a path matcher."""
    for node in _walk(config):
        for match in node.get("match", []) if isinstance(node.get("match"), list) else []:
            if path in match.get("path", []):
                return node
    raise AssertionError(f"no route for {path}")


def _handlers(node: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [n for n in _walk(node) if n.get("handler") == name]


def test_caddyfile_is_valid(tmp_path: Path) -> None:
    done = _run("validate", tmp=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "Valid configuration" in done.stdout + done.stderr


def test_api_is_proxied_unbuffered_and_the_stream_is_not_compressed(tmp_path: Path) -> None:
    config = _adapt(tmp_path)
    (proxy,) = _handlers(_route(config, "/api/*"), "reverse_proxy")
    assert proxy["upstreams"] == [{"dial": "127.0.0.1:8100"}]
    assert proxy["flush_interval"] == -1  # every server-sent event leaves at once
    (encode,) = _handlers(config, "encode")
    route = next(n for n in _walk(config) if encode in n.get("handle", []))
    assert route["match"] == [{"not": [{"path": ["/api/stream"]}]}]


def test_streamlit_keeps_its_prefix_and_its_websocket(tmp_path: Path) -> None:
    config = _adapt(tmp_path)
    route = _route(config, "/legacy/*")
    assert route["match"] == [{"path": ["/legacy", "/legacy/*"]}]
    (proxy,) = _handlers(route, "reverse_proxy")
    assert proxy["upstreams"] == [{"dial": "127.0.0.1:8501"}]
    assert not _handlers(route, "rewrite")  # no prefix stripping: Streamlit runs with --server.baseUrlPath legacy


def test_spa_falls_back_to_200_html_and_caches_only_hashed_assets(tmp_path: Path) -> None:
    config = _adapt(tmp_path, {"BF_SPA_ROOT": "/srv/spa"})
    assets = _route(config, "/_app/immutable/*")
    assert _handlers(assets, "file_server") and not _handlers(assets, "rewrite")  # a missing asset is a 404
    (cache,) = _handlers(assets, "headers")
    assert cache["response"]["set"]["Cache-Control"] == ["public, max-age=31536000, immutable"]
    fallback = [
        n
        for n in _walk(config)
        if {"file": {"try_files": ["{http.request.uri.path}", "/200.html"]}} in (n.get("match") or [])
    ]
    assert fallback and _handlers(fallback[0], "rewrite"), "no 200.html fallback"
    assert not [n for n in _walk(assets) if "file" in str(n.get("match"))]  # assets never fall back
    roots = {n["root"] for n in _handlers(config, "vars") if "root" in n}
    assert roots == {"/srv/spa"}


def test_site_redirects_http_and_takes_its_name_and_certificate_from_the_environment(tmp_path: Path) -> None:
    default = _adapt(tmp_path)
    server = default["apps"]["http"]["servers"]["srv0"]
    assert server["listen"] == [":443"]
    assert server["routes"][0]["match"] == [{"host": ["bf02.local"]}]
    assert "automatic_https" not in server or not server["automatic_https"].get("disable_redirects")
    policies = default["apps"]["tls"]["automation"]["policies"]
    assert policies[0]["issuers"] == [{"module": "internal"}]

    plant = _adapt(tmp_path, {"BF_SITE": "bf02.evonith.local", "BF_TLS": "/etc/bf/tls/cert.pem /etc/bf/tls/key.pem"})
    server = plant["apps"]["http"]["servers"]["srv0"]
    assert server["routes"][0]["match"] == [{"host": ["bf02.evonith.local"]}]
    assert plant["apps"]["tls"]["certificates"]["load_files"] == [
        {"certificate": "/etc/bf/tls/cert.pem", "key": "/etc/bf/tls/key.pem", "tags": ["cert0"]}
    ]
