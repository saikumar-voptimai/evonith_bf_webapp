"""utils.memo: the st.cache_data semantics the service layer relies on (#271)."""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from utils import memo as memo_mod
from utils.memo import clear_all, memo


def test_key_is_argument_values_and_kwargs_order_does_not_matter() -> None:
    calls = []

    @memo()
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert f(a=1, b=2) == 3  # a different call shape is a different key (computed once more)
    assert f(2, b=2) == 4
    assert calls == [(1, 2), (1, 2), (2, 2)]


def test_ttl_expires_entries(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(memo_mod.time, "monotonic", lambda: clock[0])
    calls = []

    @memo(ttl_seconds=5)
    def f(x):
        calls.append(x)
        return x

    f(1)
    clock[0] += 4.9
    f(1)
    clock[0] += 0.2
    f(1)
    assert calls == [1, 1]


def test_max_entries_evicts_least_recently_used() -> None:
    calls = []

    @memo(max_entries=2)
    def f(x):
        calls.append(x)
        return x

    f(1), f(2), f(1), f(3)  # 2 is the least recently used when 3 arrives
    f(1)
    f(2)
    assert calls == [1, 2, 3, 2]
    assert len(f) == 2


def test_copy_protects_the_cached_frame() -> None:
    @memo(copy=True)
    def frame():
        return pd.DataFrame({"a": [1, 2]})

    first = frame()
    first.loc[0, "a"] = 99
    assert frame().loc[0, "a"] == 1


def test_without_copy_the_same_object_is_returned() -> None:
    @memo()
    def obj():
        return {"x": 1}

    assert obj() is obj()


def test_exceptions_are_not_cached() -> None:
    attempts = []

    @memo()
    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("down")
        return "ok"

    with pytest.raises(ConnectionError):
        flaky()
    assert flaky() == "ok"
    assert len(attempts) == 2


def test_clear_and_clear_all() -> None:
    calls = []

    @memo()
    def f():
        calls.append(1)
        return 1

    f()
    f.clear()
    f()
    clear_all()
    f()
    assert len(calls) == 3


def test_unpicklable_arguments_fall_back_to_repr() -> None:
    lock = threading.Lock()  # not picklable

    @memo()
    def f(x):
        return 7

    assert f(lock) == 7
    assert f(lock) == 7


def test_concurrent_callers_get_the_value() -> None:
    @memo()
    def slow(x):
        time.sleep(0.01)
        return x * 2

    out = []
    threads = [threading.Thread(target=lambda: out.append(slow(21))) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert out == [42] * 8
