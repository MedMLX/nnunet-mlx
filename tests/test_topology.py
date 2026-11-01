"""Networks built from plans compared with the pinned upstream architecture."""

from __future__ import annotations

from typing import cast

import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.model.topology import PlainConvUNetSpec


@pytest.mark.oracle
def test_forward_oracle_constructs_from_original_architecture_kwargs(
    tiny_model_plans: dict[str, object],
) -> None:
    """Construct the source forward oracle from original architecture keywords."""
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("dynamic_network_architectures.architectures.unet")
    from importlib import import_module

    import numpy as np
    from nnunet_reference.oracle.weights import convert_state_dict, load_converted_state

    from nnunet_mlx.model.network import PlainConvUNet
    from nnunet_mlx.plans import PlansContract
    from nnunet_mlx.runtime import import_mlx

    plans = PlansContract.from_payload(tiny_model_plans, configuration="3d_fullres")
    architecture = cast(dict[str, object], plans.configuration.raw["architecture"])
    kwargs = dict(cast(dict[str, object], architecture["arch_kwargs"]))
    kwargs["n_stages"] = 2
    kwargs["nonlin_first"] = False
    kwargs["dropout_op"] = None
    kwargs["dropout_op_kwargs"] = None
    for key in ("conv_op", "norm_op", "nonlin"):
        module, name = cast(str, kwargs[key]).rsplit(".", 1)
        kwargs[key] = getattr(import_module(module), name)
    torch.manual_seed(31)
    source = upstream.PlainConvUNet(input_channels=1, num_classes=2, **kwargs).eval()
    spec = PlainConvUNetSpec.from_plans(
        plans.topology_payload(),
        configuration="3d_fullres",
        input_channels=1,
        num_classes=2,
    )
    candidate = PlainConvUNet(spec)
    load_converted_state(candidate, convert_state_dict(source.state_dict()))
    values = np.random.default_rng(31).normal(size=(1, 1, 8, 8, 8)).astype(np.float32)
    with torch.no_grad():
        expected = source(torch.from_numpy(values)).numpy()
    mx, _ = import_mlx()
    actual = candidate(mx.array(values.transpose(0, 2, 3, 4, 1)))
    mx.eval(actual)
    np.testing.assert_allclose(
        host_array(actual).transpose(0, 4, 1, 2, 3),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
