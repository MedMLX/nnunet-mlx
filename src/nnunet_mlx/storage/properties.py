"""Restricted loading for nnU-Net property sidecars."""

from __future__ import annotations

import math
import pickle
import pickletools
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.errors import ContractError

_MAX_PROPERTIES_BYTES = 64 * 1024 * 1024
_MAX_PROPERTIES_DEPTH = 64
_NUMPY_ARRAY_SENTINEL = object()
_MISSING = object()
_UNSAFE_PICKLE_OPCODES = frozenset({"BINPERSID", "EXT1", "EXT2", "EXT4", "PERSID"})


def _as_object(value: object) -> object:
    """Keep strict typing from leaking NumPy's intentionally broad generics."""

    return value


def _reject_unsafe_opcodes(serialized: bytes) -> None:
    """Reject pickle extensions and persistent IDs before unpickling metadata."""
    for opcode, _argument, _position in pickletools.genops(serialized):
        if opcode.name in _UNSAFE_PICKLE_OPCODES:
            raise pickle.UnpicklingError(
                f"properties use unsupported pickle opcode {opcode.name}"
            )


def _safe_numpy_dtype(
    descriptor: object,
    align: object = False,
    copy: object = False,
) -> np.dtype[np.generic]:
    """Build a NumPy dtype for unpickling, rejecting object and oversized dtypes."""
    if not isinstance(align, bool) or not isinstance(copy, bool):
        raise pickle.UnpicklingError("NumPy dtype flags must be booleans")
    try:
        dtype = np.dtype(cast(npt.DTypeLike, descriptor), align=align, copy=copy)
    except (TypeError, ValueError, OverflowError, MemoryError) as error:
        raise pickle.UnpicklingError("NumPy dtype is invalid") from error
    if dtype.hasobject or dtype.itemsize > _MAX_PROPERTIES_BYTES:
        raise pickle.UnpicklingError("object or oversized NumPy dtypes are not supported")
    # np.dtype(..., copy=True) can still return the same mutable dtype object.
    # Arrays must not share one that a later pickle BUILD can change in place.
    try:
        snapshot = cast(np.dtype[np.generic], deepcopy(dtype))
    except (TypeError, ValueError, OverflowError, MemoryError, RecursionError) as error:
        raise pickle.UnpicklingError("NumPy dtype is invalid") from error
    if snapshot.hasobject or snapshot.itemsize > _MAX_PROPERTIES_BYTES:
        raise pickle.UnpicklingError("object or oversized NumPy dtypes are not supported")
    return snapshot


def _safe_numpy_shape(value: object) -> tuple[int, ...]:
    """Require a tuple of nonnegative integer array dimensions, excluding booleans."""
    if not isinstance(value, tuple):
        raise pickle.UnpicklingError("NumPy array shape must be a tuple")
    dimensions = cast(tuple[object, ...], value)
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
        for dimension in dimensions
    ):
        raise pickle.UnpicklingError("NumPy array shape must contain non-negative integers")
    return cast(tuple[int, ...], dimensions)


def _safe_numpy_nbytes(shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> int:
    """Calculate array size and reject values above the properties allocation limit."""
    size = math.prod(shape) * dtype.itemsize
    if size > _MAX_PROPERTIES_BYTES:
        raise pickle.UnpicklingError("NumPy array exceeds the properties size limit")
    return size


class _ArrayBudget:
    """Bound aggregate allocation caused by one properties payload."""

    def __init__(self) -> None:
        """Start an aggregate array allocation budget for one properties payload."""
        self._remaining = _MAX_PROPERTIES_BYTES

    def consume(self, size: int) -> None:
        """Debit an allocation or reject it if the aggregate limit would be exceeded."""
        if size > self._remaining:
            raise pickle.UnpicklingError(
                "NumPy values exceed the aggregate properties size limit"
            )
        self._remaining -= size


class _PendingNumpyArray:
    """Validate the state supplied by NumPy's legacy array pickle protocol."""

    def __init__(
        self,
        budget: _ArrayBudget,
        value: npt.NDArray[np.generic] | None = None,
    ) -> None:
        """Retain the shared array budget until the pickle state is validated."""
        self._budget = budget
        self._value = value

    def __setstate__(self, state: object) -> None:
        """Validate legacy NumPy state, charge its allocation and copy the array."""
        if self._value is not None or not isinstance(state, tuple):
            raise pickle.UnpicklingError("NumPy array state is invalid")
        state_values = cast(tuple[object, ...], state)
        if len(state_values) != 5:
            raise pickle.UnpicklingError("NumPy array state is invalid")
        version, shape, dtype, is_fortran, buffer = state_values
        if isinstance(version, bool) or version not in (0, 1):
            raise pickle.UnpicklingError("NumPy array state version is unsupported")
        parsed_shape = _safe_numpy_shape(shape)
        if not isinstance(dtype, np.dtype):
            raise pickle.UnpicklingError("NumPy array dtype is invalid")
        parsed_dtype = _safe_numpy_dtype(cast(np.dtype[np.generic], dtype))
        if not isinstance(is_fortran, bool):
            raise pickle.UnpicklingError("NumPy array order flag must be boolean")
        if not isinstance(buffer, (bytes, bytearray)):
            raise pickle.UnpicklingError("NumPy array buffer must contain bytes")
        expected = _safe_numpy_nbytes(parsed_shape, parsed_dtype)
        if len(buffer) != expected:
            raise pickle.UnpicklingError(
                "NumPy array buffer length does not match its shape"
            )
        self._budget.consume(expected)
        order: Literal["C", "F"] = "F" if is_fortran else "C"
        values = cast(npt.NDArray[np.generic], np.frombuffer(buffer, dtype=parsed_dtype))
        self._value = values.reshape(parsed_shape, order=order).copy()

    def materialize(self) -> npt.NDArray[np.generic]:
        """Return the validated array, rejecting missing pickle state."""
        if self._value is None:
            raise pickle.UnpicklingError("NumPy array state is missing")
        if self._value.dtype.hasobject or self._value.nbytes > _MAX_PROPERTIES_BYTES:
            raise pickle.UnpicklingError("NumPy array is unsafe")
        return self._value


class _PropertiesUnpickler(pickle.Unpickler):
    """Resolve only the NumPy constructors used by plain metadata arrays."""

    def __init__(self, file: BytesIO) -> None:
        """Create a restricted unpickler with a shared NumPy allocation budget."""
        super().__init__(file)
        self._array_budget = _ArrayBudget()
        self._dtypes: list[np.dtype[np.generic]] = []

    def validate_dtypes(self) -> None:
        """Reject dtype objects mutated after an array or scalar was rebuilt."""
        if any(
            dtype.hasobject or dtype.itemsize > _MAX_PROPERTIES_BYTES
            for dtype in self._dtypes
        ):
            raise pickle.UnpicklingError("object or oversized NumPy dtypes are not supported")

    def find_class(self, module: str, name: str) -> object:
        """Resolve only the small allowlist of NumPy reconstruction globals.

        Each resolves to a validating replacement rather than the real NumPy callable;
        any other global raises ``pickle.UnpicklingError``.
        """
        global_name = (module, name)
        if global_name == ("numpy", "dtype"):
            return self._numpy_dtype
        if global_name == ("numpy", "ndarray"):
            return _NUMPY_ARRAY_SENTINEL
        if global_name in {
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
        }:
            return self._numpy_reconstruct
        if global_name in {
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "scalar"),
        }:
            return self._numpy_scalar
        if global_name in {
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        }:
            return self._numpy_frombuffer
        raise pickle.UnpicklingError(
            f"properties reference unsupported global {module}.{name}"
        )

    def _numpy_dtype(
        self,
        descriptor: object,
        align: object = False,
        copy: object = False,
    ) -> np.dtype[np.generic]:
        """Build a dtype through the restricted NumPy dtype validator."""
        dtype = _safe_numpy_dtype(descriptor, align=align, copy=copy)
        self._dtypes.append(dtype)
        return dtype

    def _numpy_frombuffer(
        self,
        buffer: object,
        dtype: object,
        shape: object,
        order: object,
    ) -> _PendingNumpyArray:
        """Rebuild an array from raw bytes, requiring the buffer length to match shape and dtype.

        The result is a writable copy and counts against the unpickler's array budget.
        """
        if not isinstance(buffer, (bytes, bytearray)):
            raise pickle.UnpicklingError("NumPy array buffer must contain bytes")
        parsed_dtype = _safe_numpy_dtype(dtype)
        parsed_shape = _safe_numpy_shape(shape)
        if order not in ("C", "F"):
            raise pickle.UnpicklingError("NumPy array order must be C or F")
        parsed_order: Literal["C", "F"] = order
        expected = _safe_numpy_nbytes(parsed_shape, parsed_dtype)
        if len(buffer) != expected:
            raise pickle.UnpicklingError(
                "NumPy array buffer length does not match its shape"
            )
        self._array_budget.consume(expected)
        values = cast(npt.NDArray[np.generic], np.frombuffer(buffer, dtype=parsed_dtype))
        return _PendingNumpyArray(
            self._array_budget, values.reshape(parsed_shape, order=parsed_order).copy()
        )

    def _numpy_reconstruct(
        self,
        subtype: object,
        shape: object,
        dtype: object,
    ) -> _PendingNumpyArray:
        """Admit only plain NumPy array reconstruction with validated arguments."""
        if subtype is not _NUMPY_ARRAY_SENTINEL:
            raise pickle.UnpicklingError("only plain NumPy arrays are supported")
        _safe_numpy_shape(shape)
        _safe_numpy_dtype(dtype)
        return _PendingNumpyArray(self._array_budget)

    def _numpy_scalar(self, dtype: object, data: object) -> np.generic:
        """Validate and reconstruct a bounded NumPy scalar from bytes."""
        if not isinstance(data, (bytes, bytearray)):
            raise pickle.UnpicklingError("NumPy scalar payload must contain bytes")
        parsed_dtype = _safe_numpy_dtype(dtype)
        if len(data) != parsed_dtype.itemsize:
            raise pickle.UnpicklingError("NumPy scalar length does not match its dtype")
        self._array_budget.consume(parsed_dtype.itemsize)
        return cast(np.generic, np.frombuffer(data, dtype=parsed_dtype, count=1)[0])


def _materialize_value(
    value: object,
    *,
    source: Path,
    depth: int,
    active: set[int],
    memo: dict[int, object],
) -> object:
    """Validate and materialize metadata containers with depth and cycle limits."""
    plain_value = _as_object(value)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, bytearray)):
        return value
    if isinstance(value, _PendingNumpyArray):
        return value.materialize()
    if isinstance(value, np.dtype):
        if value.hasobject:
            raise ContractError(f"{source} contains an object NumPy dtype")
        return plain_value
    if isinstance(value, np.ndarray):
        array = cast(npt.NDArray[np.generic], value)
        if array.dtype.hasobject or array.nbytes > _MAX_PROPERTIES_BYTES:
            raise ContractError(f"{source} contains an unsafe NumPy array")
        return array
    if isinstance(value, np.generic):
        return plain_value
    if depth >= _MAX_PROPERTIES_DEPTH:
        raise ContractError(f"{source} exceeds the properties nesting limit")
    if type(value) not in (dict, list, tuple):
        raise ContractError(f"{source} contains unsupported value {type(value).__name__}")
    identity = id(value)
    if identity in active:
        raise ContractError(f"{source} contains a recursive properties value")
    materialized = memo.get(identity, _MISSING)
    if materialized is not _MISSING:
        return materialized
    active.add(identity)
    try:
        if isinstance(value, dict):
            mapping = cast(dict[object, object], value)
            result: dict[object, object] = {}
            memo[identity] = result
            for key, item in mapping.items():
                parsed_key = _materialize_value(
                    key,
                    source=source,
                    depth=depth + 1,
                    active=active,
                    memo=memo,
                )
                parsed_item = _materialize_value(
                    item,
                    source=source,
                    depth=depth + 1,
                    active=active,
                    memo=memo,
                )
                try:
                    result[parsed_key] = parsed_item
                except TypeError as error:
                    raise ContractError(f"{source} contains an invalid mapping key") from error
            return result
        if isinstance(value, list):
            sequence = cast(Sequence[object], value)
            result_list: list[object] = []
            memo[identity] = result_list
            result_list.extend(
                _materialize_value(
                    item,
                    source=source,
                    depth=depth + 1,
                    active=active,
                    memo=memo,
                )
                for item in sequence
            )
            return result_list
        sequence = cast(Sequence[object], value)
        result_tuple = tuple(
            _materialize_value(
                item,
                source=source,
                depth=depth + 1,
                active=active,
                memo=memo,
            )
            for item in sequence
        )
        memo[identity] = result_tuple
        return result_tuple
    finally:
        active.remove(identity)


def load_properties_pickle(path: str | Path) -> dict[str, object]:
    """Load plain nnU-Net metadata without admitting executable pickle globals.

    Plain Python containers/scalars and non-object NumPy arrays, scalars, and
    dtypes are supported. The 64 MiB file/array limit bounds ordinary sidecars;
    this loader validates structure but does not establish file provenance.

    Raises:
        ContractError: If the file is oversized, malformed, executable, or not
            a string-keyed metadata mapping.
    """

    source = Path(path)
    try:
        with source.open("rb") as handle:
            serialized = handle.read(_MAX_PROPERTIES_BYTES + 1)
            if len(serialized) > _MAX_PROPERTIES_BYTES:
                raise ContractError(f"properties file exceeds 64 MiB: {source}")
        _reject_unsafe_opcodes(serialized)
        stream = BytesIO(serialized)
        unpickler = _PropertiesUnpickler(stream)
        payload = cast(object, unpickler.load())
        if stream.read(1):
            raise pickle.UnpicklingError("trailing data after properties payload")
        unpickler.validate_dtypes()
    except ContractError:
        raise
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        MemoryError,
    ) as error:
        raise ContractError(f"properties file could not be loaded safely: {source}") from error
    try:
        materialized = _materialize_value(
            payload,
            source=source,
            depth=0,
            active=set(),
            memo={},
        )
    except pickle.UnpicklingError as error:
        raise ContractError(f"properties file could not be loaded safely: {source}") from error
    if not isinstance(materialized, dict):
        raise ContractError(f"properties file must contain an object: {source}")
    raw = cast(dict[object, object], materialized)
    result: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ContractError("property keys must be strings")
        result[key] = value
    return result


__all__ = ["load_properties_pickle"]
