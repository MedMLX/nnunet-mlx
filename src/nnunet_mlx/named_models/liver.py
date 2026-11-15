"""nnU-Net liver and liver-cancer CT segmentation, MedMLX model ``nnunet-liver``."""

from __future__ import annotations

from medmlx_core.files import PathLike
from medmlx_core.weights import model_folder

from nnunet_mlx import medmlx
from nnunet_mlx.medmlx_types import RunResult
from nnunet_mlx.named_models import READINESS, segment_folder

MODEL = {
    "id": "nnunet-liver", "label": "nnU-Net liver CT", "task": "segmentation",
    "readiness": READINESS,
    "run": "nnunet_mlx.named_models.liver:run",
    "cohort": "nnunet_mlx.named_models.liver:load_model",
    # Measured MLX peak for a 768x768x101 abdominal CT; larger inputs need more.
    "peak_memory_gb": 5.6,
    "summary": "Liver and liver cancer segmentation of one CT volume.",
    "labels": {"1": "liver", "2": "cancer"},
}


def run(
    *, image_path: PathLike, output_dir: PathLike, weights_path: PathLike | None = None,
    overwrite: bool = False, use_mirroring: bool = True, step_fraction: float = 0.5,
) -> RunResult:
    """Segment liver (1) and liver cancer (2) in one CT, on the CT's grid.

    The defaults run the released configuration: three-axis mirroring (eight
    network passes per tile) and a 0.5 step. ``use_mirroring=False`` runs one pass
    per tile and a larger ``step_fraction`` runs fewer tiles; both change the
    prediction.

    Args:
        image_path: CT NIfTI volume in Hounsfield units.
        output_dir: Folder for ``mask.nii.gz``; created if missing.
        weights_path: Model folder; omitted weights download MedMLX/nnunet-liver-mlx.
        overwrite: Replace an existing mask; input/output aliases are always refused.
        use_mirroring: Average the checkpoint's test-time flips.
        step_fraction: Sliding-window step as a fraction of the patch, in (0, 1].
    """
    images, _ = medmlx.check_request(
        [image_path], output_dir=output_dir, overwrite=overwrite,
        use_mirroring=use_mirroring, step_fraction=step_fraction,
    )
    folder = model_folder("nnunet-liver", weights_path)
    return segment_folder(
        folder, images, output_dir, "3d_fullres", None, overwrite=overwrite,
        grid_of="image_path", use_mirroring=use_mirroring, step_fraction=step_fraction,
    )


class LiverModel:
    """The liver model loaded once; :meth:`run` takes the per-CT arguments."""

    def __init__(self, model: medmlx.NnUNetModel) -> None:
        """Own ``model``, released by :meth:`close`."""
        self._model = model

    def run(
        self, *, image_path: PathLike, output_dir: PathLike, overwrite: bool = False,
    ) -> RunResult:
        """Segment one CT with the loaded model; see :func:`run`.

        Raises:
            RuntimeError: The model is closed.
        """
        return self._model.segment_case(
            image_paths=[image_path], output_dir=output_dir, overwrite=overwrite,
            grid_of="image_path",
        )

    def close(self) -> None:
        """Release the model; later :meth:`run` calls raise ``RuntimeError``."""
        self._model.close()

    def __enter__(self) -> LiverModel:
        """Return this model for a ``with`` block that closes it on exit."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the model."""
        self.close()


def load_model(
    *, weights_path: PathLike | None = None, use_mirroring: bool = True,
    step_fraction: float = 0.5,
) -> LiverModel:
    """Load the liver model once to segment several CTs, or for a MedMLX cohort.

    The arguments mean what they mean in :func:`run`.
    """
    folder = model_folder("nnunet-liver", weights_path)
    return LiverModel(medmlx.load_model(
        weights_path=folder, configuration="3d_fullres",
        use_mirroring=use_mirroring, step_fraction=step_fraction,
    ))
