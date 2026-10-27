"""Append-only JSON-lines files shared by processes (SV-1.5, #285): ``utils.jsonl_log``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from utils.jsonl_log import JsonlLog

REPO = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "furnace_data")])}


def test_reads_are_incremental_and_skip_a_partly_written_line(tmp_path) -> None:
    log = JsonlLog(tmp_path / "a.jsonl")
    assert log.records() == []
    log.append({"n": 1})
    log.append({"n": 2, "text": "ηCO"})
    assert [r["n"] for r in log.records()] == [1, 2]
    with open(log.path, "a", encoding="utf-8") as handle:  # another writer, half way through a line
        handle.write('{"n": 3')
    assert [r["n"] for r in log.records()] == [1, 2]
    with open(log.path, "a", encoding="utf-8") as handle:
        handle.write(', "late": true}\n{"n": 4}\n')
    assert [r["n"] for r in log.records()] == [1, 2, 3, 4]
    assert JsonlLog(log.path).records() == log.records()  # one cache per path
    assert log.path.read_text(encoding="utf-8").splitlines()[1] == '{"n": 2, "text": "ηCO"}'  # format as before


def test_a_replaced_or_shortened_file_is_read_again(tmp_path) -> None:
    path = tmp_path / "b.jsonl"
    log = JsonlLog(path)
    for n in range(3):
        log.append({"n": n})
    assert len(log.records()) == 3
    other = tmp_path / "b.new"
    other.write_text('{"n": 9}\n', encoding="utf-8")
    os.replace(other, path)  # another inode
    assert [r["n"] for r in log.records()] == [9]
    path.write_text("", encoding="utf-8")  # same inode, shorter
    assert log.records() == []
    path.unlink()
    assert log.records() == []


def test_exclusive_is_reentrant_in_one_thread(tmp_path) -> None:
    log = JsonlLog(tmp_path / "c.jsonl")
    with log.exclusive():
        with log.exclusive():
            log.append({"n": 1})  # append takes the lock again
        log.append({"n": 2})
    assert len(log.records()) == 2


def _child(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], env=ENV, cwd=REPO)


def test_exclusive_excludes_another_process(tmp_path) -> None:
    path, marker = tmp_path / "d.jsonl", tmp_path / "held"
    child = _child(
        "import time, pathlib; from utils.jsonl_log import JsonlLog\n"
        f"log = JsonlLog({str(path)!r})\n"
        "with log.exclusive():\n"
        f"    pathlib.Path({str(marker)!r}).write_text('1'); time.sleep(1.5)\n"
    )
    try:
        deadline = time.monotonic() + 30
        while not marker.exists():
            assert time.monotonic() < deadline and child.poll() is None, "the child did not take the lock"
            time.sleep(0.02)
        log = JsonlLog(path)
        with pytest.raises(TimeoutError, match="held by another process"):
            with log.exclusive(timeout_s=0.3):
                pass
        t0 = time.monotonic()
        with log.exclusive(timeout_s=30):
            waited = time.monotonic() - t0
        assert waited > 0.3  # it waited for the child to release the lock
    finally:
        child.wait(timeout=30)


def test_appends_of_two_processes_never_interleave(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    code = (
        "from utils.jsonl_log import JsonlLog\n"
        f"log = JsonlLog({str(path)!r})\n"
        "for i in range(150):\n"
        "    with log.exclusive():\n"
        "        known = {r['key'] for r in log.records()}\n"
        "        if str(i) not in known:  # read, check, append: the pattern of the ledgers\n"
        "            log.append({'key': str(i), 'who': 'WHO', 'pad': 'x' * 3000})\n"
    )
    children = [_child(code.replace("WHO", "a")), _child(code.replace("WHO", "b"))]
    for child in children:
        assert child.wait(timeout=120) == 0
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]  # every line is one whole record
    assert sorted(r["key"] for r in rows) == sorted(str(i) for i in range(150))  # each key exactly once
