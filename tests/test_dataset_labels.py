"""Malformed dataset labels must be rejected without label-value-sized allocations."""

import pytest

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.errors import ContractError


@pytest.mark.parametrize("label", [2**63, 2**100])
def test_sparse_label_metadata_is_rejected_without_expanding_its_range(label: int) -> None:
    with pytest.raises(ContractError, match="consecutive"):
        parse_dataset_label_semantics({"labels": {"background": 0, "foreground": label}})
