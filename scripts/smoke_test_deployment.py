#!/usr/bin/env python
"""Run concise smoke tests against a Phase 13 deployment."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from typing import Any

from deployment_common import CheckResult, exit_code, print_results


PUBLIC_ENDPOINTS = [
    "/health",
    "/readiness",
    "/status",
    "/openapi.json",
    "/data/sources",
    "/datasets",
    "/feedback/config",
    "/material-balance/config",
    "/recommendations/config",
    "/blend-optimizer/context",
    "/copilot/config",
    "/furnacemind/config",
]


def _request(url: str, *, timeout: float, token: str | None = None) -> tuple[int, dict[str, str], bytes]:
    headers = {"X-Request-ID": "phase13-smoke-test"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def _json_request(url: str, payload: dict[str, Any], *, timeout: float) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-Request-ID": "phase13-smoke-test"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def _extract_token(body: bytes) -> str | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        return data.get("access_token") or data.get("token")
    return None


def run(args: argparse.Namespace) -> list[CheckResult]:
    base = args.backend_url.rstrip("/")
    results: list[CheckResult] = []
    token: str | None = None
    for endpoint in PUBLIC_ENDPOINTS:
        url = base + endpoint
        try:
            status, headers, _body = _request(url, timeout=args.timeout)
            ok = 200 <= status < 500
            result_status = "pass" if ok else "fail"
            message = f"{endpoint} returned {status}"
            if endpoint == "/health" and "x-request-id" not in {key.lower() for key in headers}:
                result_status = "fail"
                message += " without X-Request-ID"
            results.append(CheckResult(endpoint, result_status, message))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            results.append(CheckResult(endpoint, "fail", f"request failed: {exc.__class__.__name__}"))

    if not args.skip_auth and args.username and args.password:
        status, _headers, body = _json_request(base + "/auth/login", {"username": args.username, "password": args.password}, timeout=args.timeout)
        if 200 <= status < 300:
            token = _extract_token(body)
            results.append(CheckResult("auth_login", "pass" if token else "fail", "auth login returned token" if token else "auth login did not return token"))
        else:
            results.append(CheckResult("auth_login", "fail", f"auth login returned {status}"))
    else:
        results.append(CheckResult("auth_login", "warn", "auth smoke skipped"))

    if token:
        for endpoint in ("/auth/me", "/status/dependencies", "/metrics"):
            status, _headers, _body = _request(base + endpoint, timeout=args.timeout, token=token)
            results.append(CheckResult(endpoint, "pass" if 200 <= status < 500 else "fail", f"{endpoint} returned {status}"))

    status, _headers, body = _request(base + "/does-not-exist-for-smoke", timeout=args.timeout)
    structured = False
    try:
        payload = json.loads(body.decode("utf-8"))
        structured = isinstance(payload, dict) and ("error" in payload or "detail" in payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        structured = False
    results.append(CheckResult("structured_error", "pass" if status >= 400 and structured else "fail", f"bad route returned {status}"))

    if args.frontend_url:
        try:
            status, _headers, _body = _request(args.frontend_url.rstrip("/"), timeout=args.timeout)
            results.append(CheckResult("frontend", "pass" if 200 <= status < 500 else "warn", f"frontend returned {status}"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            results.append(CheckResult("frontend", "warn" if not args.strict else "fail", f"frontend request failed: {exc.__class__.__name__}"))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://localhost:8080/api/v1")
    parser.add_argument("--frontend-url", default="")
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--skip-auth", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.password:
        _ = base64.b64encode(b"redacted")  # keeps password intentionally unused in output
    results = run(args)
    print_results(results, json_output=args.json)
    return exit_code(results, strict_warnings=args.strict)


if __name__ == "__main__":
    sys.exit(main())

