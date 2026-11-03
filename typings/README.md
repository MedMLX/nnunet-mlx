These declarations cover the dependency APIs used by nnunet-mlx. They do not
replace dependency implementations or change runtime imports.

- MLX 0.32.4: python/mlx/nn/layers/base.py, convolution, transpose-convolution
  and pooling layers, nn/utils.py, and utils.py from the installed pinned
  release. Parameter trees retain their real dictionary/list/tuple containers
  and array leaves. A module's state is the module itself.
- Blosc2 4.11.0 and lock-pinned 4.14.1: src/blosc2/ndarray.py and src/blosc2/schunk.py from the
  installed runtime. Integer and slice array reads return NumPy arrays;
  chunk reads return bytes or strings. The declarations cover this slice
  interface, rather than Blosc2's expression/query indexing APIs.
- SciPy 1.18.1: the three `scipy.ndimage` functions
  `nnunet_mlx.prediction.postprocessing` uses for connected-component
  filtering (`generate_binary_structure`, `label`, `sum_labels`), declared
  only for the exact call shapes used there. SciPy ships no type stubs at
  all for `scipy.ndimage`.

MLX already ships its native core stubs. The custom Metal kernel callback
contract is owned in nnunet_mlx.kernels.types, because the upstream core
stub declares the kernel factory result as object.

The nn.Module declaration is a partial interface. Runtime networks still inherit
from the genuine MLX Module, including its dictionary container behavior.
The declaration does not promise substitutability with dict.update: MLX's
update(parameters, strict=True) returns the module, while dict.update returns
None. The used MLX update signature and return value are declared directly.
This is not complete typing coverage for upstream MLX or Blosc2.
