"""Pinned nnU-Net oracle: metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ForwardParityMetrics:
    seed: int
    tolerance_rtol: float
    tolerance_atol: float
    maximum_absolute_error: float
    mean_absolute_error: float
    p99_absolute_error: float
    relative_l2_error: float
    argmax_agreement: float
    passed: bool

    def to_payload(self) -> dict[str, object]:
        """Serialize the measured raw-logit parity metrics."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ForwardParityResult:
    expected: FloatArray
    actual: FloatArray
    metrics: ForwardParityMetrics


@dataclass(frozen=True, slots=True)
class ArrayParityMetrics:
    name: str
    maximum_absolute_error: float
    relative_l2_error: float
    tolerance_rtol: float
    tolerance_atol: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ExactParityResult:
    name: str
    expected: object
    actual: object

    @property
    def passed(self) -> bool:
        """Report whether expected and actual values are equal."""
        return self.actual == self.expected

    def to_payload(self) -> dict[str, object]:
        """Serialize the exact comparison and its verdict."""
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True, slots=True)
class ContractParityResult:
    arrays: tuple[ArrayParityMetrics, ...]
    exact: tuple[ExactParityResult, ...] = ()

    @property
    def passed(self) -> bool:
        """Require every numerical and exact contract comparison to pass."""
        return all(metric.passed for metric in self.arrays) and all(
            result.passed for result in self.exact
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize comparison results and their combined verdict."""
        return {
            "arrays": [asdict(metric) for metric in self.arrays],
            "exact": [result.to_payload() for result in self.exact],
            "passed": self.passed,
        }


def nested_tuple(value: object) -> object:
    """Convert nested arrays, lists and NumPy scalars into comparable Python values."""
    if isinstance(value, np.ndarray):
        return nested_tuple(value.tolist())
    if isinstance(value, list | tuple):
        return tuple(nested_tuple(item) for item in cast(Sequence[object], value))
    if isinstance(value, np.generic):
        return cast(object, value.item())
    return value


def exact_array_parity(
    expected: npt.NDArray[np.generic], actual: npt.NDArray[np.generic], *, name: str
) -> ExactParityResult:
    """Compare array dtypes and values exactly for a named parity result."""
    return ExactParityResult(
        name=name,
        expected=True,
        actual=bool(expected.dtype == actual.dtype and np.array_equal(expected, actual)),
    )


def array_metrics(
    expected: FloatArray,
    actual: FloatArray,
    *,
    name: str,
    rtol: float,
    atol: float,
) -> ArrayParityMetrics:
    """Report errors and an elementwise ``allclose`` verdict against ``expected``.

    Array shape compatibility is the caller's responsibility; NumPy broadcasting
    applies. The relative L2 metric is reported but is not a separate pass gate.
    """
    residual = actual - expected
    denominator = max(float(np.linalg.norm(expected.reshape(-1))), np.finfo(np.float32).eps)
    return ArrayParityMetrics(
        name=name,
        maximum_absolute_error=float(np.max(np.abs(residual))),
        relative_l2_error=float(np.linalg.norm(residual.reshape(-1)) / denominator),
        tolerance_rtol=rtol,
        tolerance_atol=atol,
        passed=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
    )




def measure_forward_parity(
    expected: FloatArray,
    actual: FloatArray,
    *,
    seed: int,
    rtol: float,
    atol: float,
) -> ForwardParityMetrics:
    """Report raw-logit differences and channel-axis argmax agreement.

    ``passed`` uses elementwise ``allclose`` with the supplied tolerances;
    argmax agreement and the other reported statistics are not additional gates.
    Callers supply compatible arrays and the expected channel axis at index 1.
    """

    residual = np.abs(actual - expected)
    denominator = max(float(np.linalg.norm(expected.reshape(-1))), np.finfo(np.float32).eps)
    label_matches = np.argmax(actual, axis=1) == np.argmax(expected, axis=1)
    argmax_agreement = np.count_nonzero(label_matches) / label_matches.size
    return ForwardParityMetrics(
        seed=seed,
        tolerance_rtol=rtol,
        tolerance_atol=atol,
        maximum_absolute_error=float(residual.max()),
        mean_absolute_error=float(residual.mean()),
        p99_absolute_error=float(np.quantile(residual, 0.99)),
        relative_l2_error=float(np.linalg.norm((actual - expected).reshape(-1)) / denominator),
        argmax_agreement=float(argmax_agreement),
        passed=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
    )
