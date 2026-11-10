"""Persisted nnU-Net properties reject executable pickle globals."""

from __future__ import annotations

import copyreg
import pickle
import pickletools
from pathlib import Path

import numpy as np
import pytest

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.errors import ContractError
from nnunet_mlx.prediction.ensembling import merge_probability_files
from nnunet_mlx.storage.properties import load_properties_pickle


def _touch(path: str) -> None:
    """Create a marker if an unsafe pickle callable is executed."""
    Path(path).touch()


class _ExecutableProperties:
    def __init__(self, marker: Path) -> None:
        """Retain a marker path for an executable-properties fixture."""
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        """Encode the marker-writing callable in an adversarial pickle."""
        return _touch, (str(self.marker),)


def _write_executable_pickle(path: Path, marker: Path) -> None:
    """Write a fixture pickle that would execute the marker callable if unsafe."""
    with path.open("wb") as handle:
        pickle.dump(_ExecutableProperties(marker), handle)


def test_case_properties_reject_executable_pickle_without_running_it(tmp_path: Path) -> None:
    """Reject an executable properties pickle without creating its marker."""
    marker = tmp_path / "case-properties-executed"
    properties = tmp_path / "case.pkl"
    _write_executable_pickle(properties, marker)

    with pytest.raises(ContractError):
        load_properties_pickle(properties)

    assert not marker.exists()


def test_case_properties_reject_cached_pickle_extension(tmp_path: Path) -> None:
    """Reject cached pickle extension opcodes before execution."""
    marker = tmp_path / "cached-extension-executed"
    properties = tmp_path / "case.pkl"
    extension_code = 42_420
    copyreg.add_extension(__name__, "_touch", extension_code)
    try:
        payload = pickle.dumps(_ExecutableProperties(marker), protocol=4)
        # Seed the extension cache that can bypass a restricted find_class.
        pickle.loads(payload)
        assert marker.exists()
        marker.unlink()
        properties.write_bytes(payload)

        with pytest.raises(ContractError):
            load_properties_pickle(properties)

        assert not marker.exists()
    finally:
        copyreg.remove_extension(__name__, "_touch", extension_code)
        copyreg.clear_extension_cache()
        marker.unlink(missing_ok=True)


def test_probability_merge_rejects_executable_pickle_without_running_it(
    tmp_path: Path,
) -> None:
    """Reject executable probability properties without running their callable."""
    probabilities = tmp_path / "case.npz"
    np.savez_compressed(
        probabilities,
        probabilities=np.full((2, 1, 2, 3), 0.5, dtype=np.float32),
    )
    marker = tmp_path / "probability-properties-executed"
    _write_executable_pickle(probabilities.with_suffix(".pkl"), marker)
    semantics = parse_dataset_label_semantics({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "target": 1},
        "file_ending": ".nii.gz",
    })

    with pytest.raises(ContractError):
        merge_probability_files(
            (probabilities,),
            tmp_path / "merged",
            file_ending=".nii.gz",
            image_reader_writer="SimpleITKIO",
            semantics=semantics,
            overwrite=False,
            save_probabilities=False,
        )

    assert not marker.exists()


@pytest.mark.parametrize("protocol", [4, pickle.HIGHEST_PROTOCOL])
def test_case_properties_accept_plain_numpy_metadata(tmp_path: Path, protocol: int) -> None:
    """Accept supported plain NumPy values in properties metadata."""
    properties = tmp_path / "case.pkl"
    expected: dict[str, object] = {
        "spacing": (np.float32(1.5), 2.0, 3.0),
        "affine": np.eye(4, dtype=np.float64),
    }
    properties.write_bytes(pickle.dumps(expected, protocol=protocol))

    loaded = load_properties_pickle(properties)

    assert loaded["spacing"] == expected["spacing"]
    np.testing.assert_array_equal(np.asarray(loaded["affine"]), np.eye(4))


@pytest.mark.parametrize("protocol", [4, pickle.HIGHEST_PROTOCOL])
def test_case_properties_reject_dtype_mutation_after_array_build(
    tmp_path: Path, protocol: int
) -> None:
    """Reject a memoized dtype changed to object-bearing after an array uses it."""
    dtype = np.dtype("i8", copy=True)
    serialized = pickle.dumps({"array": np.zeros(1, dtype=dtype)}, protocol=protocol)
    memo_index = 0
    dtype_memo: int | None = None
    reduced = False
    for opcode, _argument, _position in pickletools.genops(serialized):
        if opcode.name == "REDUCE":
            reduced = True
        elif opcode.name == "MEMOIZE":
            if reduced:
                dtype_memo = memo_index
            memo_index += 1
            reduced = False
        elif opcode.name == "BUILD":
            break
        else:
            reduced = False
    assert dtype_memo is not None and dtype_memo < 256
    assert serialized[2:3] == b"\x95"  # The small fixture occupies one pickle frame.
    dtype_state = (3, "<", None, None, None, -1, -1, 1)
    mutation = (
        b"h" + bytes((dtype_memo,)) + pickle.dumps(dtype_state, protocol=2)[2:-1] + b"b0"
    )
    frame_size = int.from_bytes(serialized[3:11], "little") + len(mutation)
    properties = tmp_path / "case.pkl"
    properties.write_bytes(
        serialized[:3]
        + frame_size.to_bytes(8, "little")
        + serialized[11:-1]
        + mutation
        + b"."
    )

    with pytest.raises(ContractError):
        load_properties_pickle(properties)


def test_case_properties_reject_rebuilding_frombuffer_array(tmp_path: Path) -> None:
    """A pickle BUILD must not bypass validation on an already rebuilt array."""
    serialized = pickle.dumps({"array": np.zeros(1, dtype="i8")}, protocol=5)
    reduce_memos: list[int] = []
    memo_index = 0
    reduced = False
    for opcode, _argument, _position in pickletools.genops(serialized):
        if opcode.name == "REDUCE":
            reduced = True
        elif opcode.name == "MEMOIZE":
            if reduced:
                reduce_memos.append(memo_index)
            memo_index += 1
            reduced = False
        else:
            reduced = False
    assert len(reduce_memos) == 2
    dtype_memo, array_memo = reduce_memos
    assert dtype_memo < 256 and array_memo < 256
    assert serialized[2:3] == b"\x95"
    # Replay an otherwise valid legacy state on the protocol-5 array result.
    rebuild = (
        b"h"
        + bytes((array_memo,))
        + b"(K\x01K\x01\x85h"
        + bytes((dtype_memo,))
        + b"\x89C\x08"
        + bytes(8)
        + b"tb0"
    )
    frame_size = int.from_bytes(serialized[3:11], "little") + len(rebuild)
    properties = tmp_path / "case.pkl"
    properties.write_bytes(
        serialized[:3] + frame_size.to_bytes(8, "little") + serialized[11:-1] + rebuild + b"."
    )

    with pytest.raises(ContractError):
        load_properties_pickle(properties)
