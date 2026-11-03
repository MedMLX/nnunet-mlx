"""scipy.ndimage surface used by nnunet_mlx's connected-component postprocessing."""

import numpy as np
import numpy.typing as npt

def generate_binary_structure(rank: int, connectivity: int) -> npt.NDArray[np.bool_]: ...
def label(
    input: npt.NDArray[np.bool_], structure: npt.NDArray[np.bool_] | None = None
) -> tuple[npt.NDArray[np.int32], int]: ...
def sum_labels(
    input: npt.NDArray[np.bool_],
    labels: npt.NDArray[np.int32],
    index: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]: ...
