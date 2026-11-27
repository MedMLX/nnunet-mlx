"""Repository-used optimizer APIs from MLX 0.32.5."""

from mlx.core import array
from mlx.nn import Module, Parameters

class AdamW:
    state: Parameters
    def __init__(
        self, learning_rate: float, *, weight_decay: float = 0.01, bias_correction: bool = False
    ) -> None: ...
    def init(self, parameters: Parameters) -> None: ...
    def update(self, model: Module, gradients: Parameters) -> None: ...

def clip_grad_norm(gradients: Parameters, max_norm: float) -> tuple[Parameters, array]: ...
