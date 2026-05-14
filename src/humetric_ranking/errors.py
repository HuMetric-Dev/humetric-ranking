from __future__ import annotations

from dataclasses import dataclass

from humetric_core import HumetricError


@dataclass(frozen=True, slots=True)
class TrainFailed(HumetricError):
    reason: str


@dataclass(frozen=True, slots=True)
class PredictFailed(HumetricError):
    reason: str


@dataclass(frozen=True, slots=True)
class ModelMissing(HumetricError):
    path: str


@dataclass(frozen=True, slots=True)
class ModelLoadFailed(HumetricError):
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class CrossEncoderUnavailable(HumetricError):
    reason: str


type RankingError = (
    TrainFailed | PredictFailed | ModelMissing | ModelLoadFailed | CrossEncoderUnavailable
)
