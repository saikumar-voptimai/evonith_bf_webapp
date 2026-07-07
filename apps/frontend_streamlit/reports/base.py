"""Generic ABCs for the report pipeline pattern.

Every report type follows the same three-stage pipeline:
  Fetcher  → pulls raw source data (InfluxDB, DB, etc.)
  Builder  → computes all metrics from raw data (pure Python, no I/O)
  Analyser → sends compact metrics to an LLM for narrative text
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

RawT = TypeVar("RawT")
ReportT = TypeVar("ReportT")


class ReportFetcher(ABC, Generic[RawT]):
    @abstractmethod
    def fetch(self, **kwargs) -> RawT: ...


class ReportBuilder(ABC, Generic[RawT, ReportT]):
    @abstractmethod
    def build(self, raw: RawT) -> ReportT: ...


class ReportAnalyser(ABC, Generic[ReportT]):
    @abstractmethod
    def analyse(self, current: ReportT, previous: Optional[ReportT] = None) -> str: ...
