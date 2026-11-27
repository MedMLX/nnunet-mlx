"""Released nnU-Net models, one module each, with fixed configurations.

Each module declares one MedMLX model as ``MODEL`` and its ``run``, which
returns the label map as ``outputs["mask"]`` on the input grid. Importing a
module does not initialize MLX.
"""

READINESS = "nnunet_mlx.medmlx:readiness"
