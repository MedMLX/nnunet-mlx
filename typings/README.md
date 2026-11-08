These stubs describe the dependency APIs used by nnunet-mlx.

- MLX 0.32.4: python/mlx/nn/layers/base.py, convolution, transpose-convolution
  and pooling layers, nn/utils.py, and utils.py from the installed pinned
  release. Parameter trees retain their real dictionary/list/tuple containers
  and array leaves. A module's state is the module itself.
- Blosc2 4.11.0 and lock-pinned 4.14.1: src/blosc2/ndarray.py and src/blosc2/schunk.py from the
  installed runtime. Integer and slice array reads return NumPy arrays;
  chunk reads return bytes or strings. The declarations cover this slice
  interface, rather than Blosc2's expression/query indexing APIs.

MLX already ships its native core stubs. The custom Metal kernel callback
contract is owned in nnunet_mlx.kernels.types, because the upstream core
stub declares the kernel factory result as object.

The nn.Module declaration is a partial interface. Runtime networks still inherit
from the genuine MLX Module, including its dictionary container behavior.
MLX's `update(parameters, strict=True)` returns the module, unlike
`dict.update`, which returns `None`. The stub preserves the MLX signature.

SciPy types come from the maintained `scipy-stubs` development dependency.
