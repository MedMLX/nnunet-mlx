# Technical notes

## Native training

The native planner uses the training split to choose median voxel spacing and patch
geometry. The trainer reads memory-mapped cases, samples foreground patches, and
prefetches one batch. AdamW updates the highest-resolution head and its encoder and
decoder using a Dice and cross-entropy loss. Class targets use indexed reductions
with no dense one-hot tensor. Region targets use independent sigmoid outputs.

FP32 is the default. BF16 reduces convolution and activation precision while
keeping the loss, normalization statistics, parameters, and optimizer state in FP32.
Use `--recompute` to rebuild encoder and decoder stage activations during backward.
The resume file stores the last completed epoch, model weights, and optimizer state.
Validation pools Dice over held-out cases on the preprocessed grid with mirroring
disabled. The final model folder can be passed directly to `model predict-folder`.
The best checkpoint uses the mean of the pooled foreground Dice scores. The cosine
schedule retains its original step budget across resume and holds at 5% of the
configured learning rate after that budget.

The M5 Max test used one 64³ batch-one crop from the HCC-TACE-Seg `HCC_063` CT and
liver mask. MLX 0.32.5 ran with `MLX_ENABLE_TF32=0`. Each value is a median across
eight interleaved runs. Update time includes forward, backward, clipping, AdamW,
and synchronization.

| Training mode | Warm update (ms) | Peak MLX allocation (GiB) |
| :--- | ---: | ---: |
| Eager FP32 | 56.7 | 2.73 |
| Compiled FP32 | 44.4 | 3.97 |
| Compiled BF16 | 35.9 | 2.26 |
| Compiled BF16 with recomputation | 43.1 | 1.63 |

### Selective activation recomputation

`--recompute-stages N` selects the highest-resolution stages in each path. The
residual stem is included when recomputation is enabled. The six-round paired
measurements below used a 96³ batch-one Decathlon CT patch per task, compiled
BF16 on M5 Max and MLX 0.32.5, with `MLX_ENABLE_TF32=0`.

| Task | Stages per path | Warm update (ms) | Peak MLX allocation (GiB) |
| :--- | ---: | ---: | ---: |
| Liver | 0 | 115.7 | 4.50 |
| Liver | 2 | 130.1 | 3.57 |
| Pancreas | 0 | 114.5 | 4.45 |
| Pancreas | 2 | 127.2 | 3.10 |

### Decathlon convergence pilot

The liver and pancreas pilots each used 20 training, five validation, and five
held-out patients from the labeled Decathlon cases. All images were resampled
to 2 mm isotropic spacing. Runs used compiled BF16, batch size one, 100 updates
per epoch, and whole-volume validation every three epochs. The same patient
splits and initialization seed were used across recipes.

The table reports pairs of Dice scores at the epoch with the highest mean pooled
foreground Dice within 1,200 updates. Liver selected warmup and decay, lesion-focused
sampling, in-plane rotations, and balanced cross-entropy. Pancreas selected the
same recipe with unweighted cross-entropy.

| Task | Recipe | Patch | Organ Dice | Tumor Dice | Time (s) | Peak MLX (GiB) | Peak process footprint (GiB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Liver | Baseline | 64³ | 0.439 | 0.000 | 72.4 | 2.31 | 4.32 |
| Liver | Selected | 64³ | 0.701 | 0.208 | 72.1 | 2.31 | 4.34 |
| Liver | Selected, two recomputed stages | 96³ | 0.840 | 0.000 | 190.5 | 3.63 | 8.05 |
| Pancreas | Baseline | 64³ | 0.068 | 0.000 | 72.3 | 1.87 | 4.68 |
| Pancreas | Selected | 64³ | 0.178 | 0.000 | 74.1 | 1.87 | 4.68 |
| Pancreas | Selected, two recomputed stages | 96³ | 0.331 | 0.000 | 187.4 | 3.16 | 9.00 |

The extended 64³ comparison stopped at the first validation with organ Dice of
at least 0.75 for liver or 0.55 for pancreas, up to 3,000 updates. Baseline times
sum the active segments before and after resume. Time includes validation and
checkpoint writes. The 96³ trials retained the 1,200-update budget.

| Task | Recipe | Patch | Time to target (s) | Highest validation organ Dice |
| :--- | :--- | ---: | ---: | ---: |
| Liver | Baseline | 64³ | >179.1 | 0.670 |
| Liver | Selected | 64³ | 125.3 | 0.755 |
| Liver | Selected, two recomputed stages | 96³ | 143.6 | 0.840 |
| Pancreas | Baseline | 64³ | >174.8 | 0.132 |
| Pancreas | Selected | 64³ | >175.5 | 0.338 |
| Pancreas | Selected, two recomputed stages | 96³ | >187.4 | 0.331 |

The validation-selected checkpoints were evaluated on the five held-out patients
per task. Dice and surface Dice are patient means on the preprocessed grid.
Surface Dice uses voxel centers and a 2 mm tolerance. Lesion detection requires
prediction of at least 10% of the voxels in a six-connected reference lesion.

| Task | Recipe | Patch | Organ Dice | Organ surface Dice | Tumor Dice | Tumor surface Dice | Lesions detected |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Liver | Extended baseline | 64³ | 0.684 | 0.233 | 0.033 | 0.052 | 4/45 |
| Liver | Extended selected | 64³ | 0.792 | 0.440 | 0.273 | 0.183 | 13/45 |
| Liver | Selected, two recomputed stages | 96³ | 0.870 | 0.559 | 0.000 | 0.000 | 0/45 |
| Pancreas | Extended baseline | 64³ | 0.098 | 0.077 | 0.000 | 0.000 | 0/5 |
| Pancreas | Extended selected | 64³ | 0.259 | 0.191 | 0.000 | 0.000 | 0/5 |
| Pancreas | Selected, two recomputed stages | 96³ | 0.326 | 0.202 | 0.000 | 0.000 | 0/5 |

## Accumulation, compilation and arithmetic

The MedMLX runners and the CLI's `predict` and `predict-folder` compute the same thing:
uncompiled models, with sliding windows and folds accumulated in float16 as nnU-Net v2.8.1
does. The logits stay float16 into restoration, where spline resampling interpolates in
float64 and rounds back to float16 before the softmax or argmax, again as nnU-Net does; at
near ties the first class wins. `ModelFolderPredictor` used directly accumulates in float32.
Its `compile_models=True` compiles only where that reproduces uncompiled results exactly:
`PlainConvUNet` with up to 3 outputs, and the final upsampling of 3D residual `BasicBlockD`
models with up to 27 outputs. Compilation is off by default because it is no faster than
the fused native kernels (128³ forward, M5 Max: 96.8 vs 98.2 ms uncompiled vs compiled for
the liver `PlainConvUNet`).

Compilation requires fixed model state. Checkpoint-backed multi-fold bundles reject
`compile_models=True` with a specific error because their shared weights change between folds.
Prediction results record `accumulation` and `decoding`; MedMLX returns these in
`details`, together with `save_probabilities`. Decoding is `logit-argmax`,
`softmax-argmax`, or `sigmoid-region-threshold`.

`--precision source-fp32` (`precision="source-fp32"`) still runs entirely in
MLX on Metal. It selects local kernels that sum float32 values in the pinned PyTorch CPU
order, which some released models need to reproduce their reference labels exactly. It
disables model compilation. Profiles that list it reject any other precision, and no other
profile accepts it.

## Compatibility profiles

Loading checks each checkpoint's trainer, configuration and plans against the selected
profile. Profiles cover inference only, not training or segmentation quality. Profiles
without a ✓ in the README are accepted and run within the supported configurations,
without a published full-model comparison.

Accepted convolution kernels are positive and odd, and LeakyReLU slopes are in [0, 1].
These bounds keep padding and fused activation semantics aligned with the pinned architecture.

The original TotalSegmentator `v2.0.0-weights` plans use older architecture fields. For
the `totalsegmentator-v2-ct` identities the architecture is reconstructed in memory with
the pinned nnU-Net defaults, and the plans files are unchanged.

## Model folder contract

`dataset.json` must define `channel_names`. Labels must be consecutive non-negative
integers or ASCII decimal strings; boolean, fractional, padded and Unicode values are
rejected even where upstream's `int()` accepts them. `conversion.json` is optional export
metadata: inference does not read it, and `read_conversion_manifest` validates it and its
payload files only for inspection.

The automatic conversion cache also checks the complete source plans and dataset JSON,
a conversion-contract version, checkpoint size/mtime and cached payload availability.
Changing metadata rebuilds the conversion even when the checkpoint is unchanged.
No content digests are required.

`details.weights_revision` records the downloaded Hugging Face commit, or `null` for a
folder without a MedMLX download record.

## Outputs

| Role | File | Content |
| --- | --- | --- |
| `mask` | `mask.nii.gz` (`mask.nii` when `dataset.json` says so) | Label map with the input image's shape and geometry |
| `probabilities` | `probabilities.npz` | Channel-first class or region probabilities (`save_probabilities` only) |
| `probability_properties` | `probabilities.pkl` | nnU-Net properties for the probability grid (`save_probabilities` only) |

Probabilities stay on the image reader's array grid; read them with their properties
sidecar. As in nnU-Net, requesting probabilities makes class models take the argmax after
float32 softmax rather than of the logits, so near ties can segment differently.

Saved probability metadata records the array shape, ZYX axis convention and active
Nibabel array affine. Ensembling compares active reader grids as well as original
geometry and rejects native/reoriented arrays that do not align.
All file-based prediction APIs publish staged, read-back-validated outputs through one
helper. The completion receipt is written under the final names last. Overwrites remove
obsolete probability sidecars; `overwrite=False` protects the complete generation and
uses no-clobber publication. Outputs that alias inputs, including hardlinks, are rejected.
`details` holds the `dataset.json` label table, `output_kind` (`binary_mask` or
`label_map`), the selected options, checkpoint signatures and timings. The Python runner
does not expose cascade stages; use the CLI's `--previous-stage`.

Errors are [medmlx-core](https://github.com/MedMLX/medmlx/tree/main/packages/core) types,
each chaining the package's `ContractError`: `InvalidInputError` for bad images, options
or existing outputs; `AssetNotReadyError` for a missing, malformed or mismatched model
folder; `MissingDependencyError` when the pinned MLX or a Metal device is unavailable; and
`ModelExecutionError` when reading, predicting or writing the case fails, including a
channel-count mismatch.

## Folder prediction, ensembling and postprocessing

`predict-folder --continue-existing` skips cases with matching completion receipts and
replaces stale ones; it cannot be combined with `--overwrite`. Receipts compare file size
and modification time.

`ensemble` averages saved probabilities in `--input-folder` order and decodes the labels,
as nnU-Net's folder ensemble does. Save predictions with `--save-probabilities` first and
pass `--plans-json` and `--dataset-json` from the model folder. Every folder must hold the
same `.npz` cases, each with its `.pkl` sidecar, on the same grid.

`apply-postprocessing` applies a released model's `postprocessing.pkl` to exported
segmentations, as `nnUNetv2_apply_postprocessing` does. Prediction never applies it
automatically.

## Untrusted files

Inference loads safetensors weights, never PyTorch `.pth` checkpoints. nnU-Net's `.pkl`
sidecars (case properties and probability metadata) are read by a restricted loader that
accepts plain Python containers and non-object NumPy metadata up to 64 MiB, and rejects
executable globals, extension and persistent-ID opcodes, object arrays, recursion and
trailing data. A released `postprocessing.pkl` may hold one payload up to 1 MiB naming
the one supported upstream postprocessing function. These checks cover structure, not
provenance.

## CT normalization experiment (removed)

CT normalization uses NumPy clipping, subtraction and division before resampling.
An experimental no-resampling route retained normalization, padding and patch
extraction in MLX. It was removed because the complete-prediction comparison
below found no reliable end-to-end benefit.

The experiment supported CT-only, non-cascade cases whose target shape matched
the cropped input. Validated parameters were passed per channel as a float32
array containing `[lower, upper, mean, max(std, 1e-8)]`. One compiled function
specialized on input shape and dtype, with statistics kept as explicit inputs.
Padding used normalized zero, as in the host path.

Measured on 13 November 2026 with Apple M5 Max (128 GB), macOS 27.2, Python
3.13.18 and MLX 0.32.4 on Metal. HCC-TACE HCC-063 CT supplied 25,952,256
float32 voxels. Liver checkpoint statistics were lower -15 HU, upper 197 HU,
mean 99.48019 HU and standard deviation 37.138412 HU. Six paired fresh-process
rounds used synchronized outputs, 0.1 s discarded warmup and 12 timed calls for
arithmetic or 10 for preprocessing. Metal startup was warmed separately;
first calls include shape compilation, while imports and input loading are excluded.

| Operation | First call (ms) | Warm median (ms) | Peak added process memory (MB) |
| --- | ---: | ---: | ---: |
| NumPy CT arithmetic | 17.56 | 21.51 | 99.1 |
| Eager MLX CT arithmetic, device inputs | 47.14 | 1.74 | 111.7 |
| Compiled MLX CT arithmetic, device inputs | 45.93 | 0.56 | 210.3 |
| Host preprocessing plus MLX upload | 58.85 | 65.77 | 198.2 |
| Preprocessing with compiled MLX normalization | 129.38 | 23.66 | 309.9 |

The preprocessing comparison assigned model-grid spacing to the source array to
isolate the no-resampling route. Both arithmetic and full preprocessing matched
NumPy exactly across all 25,952,256 values. A separate real HU crop of shape
`(1, 32, 64, 64)`, padded to the liver checkpoint's 128³ patch, produced identical
final logits and labels with fold 0, mirroring disabled and nnU-Net float16
accumulation. Additional parity checks covered overlapping 2-D/3-D tiles,
padding, mirroring, two folds and the standard-deviation floor against pinned
nnU-Net 2.8.1. The compiled variant improved warmed preprocessing while
increasing first-call latency and process memory for this workload.

### Complete prediction on three real CT volumes

A follow-up on 14 November 2026 used six paired runs on the same Mac and runtime,
with liver fold 0, 128³ patches, step fraction 0.5, float32 network arithmetic,
float16 accumulation, model compilation disabled and mirroring disabled. Timings
include file reading through segmentation export, with model loading excluded.
Each fresh process predicted the three cases in table order, so the first case
includes network startup and later cases introduce new normalization shapes.

The inputs were full CT volumes resampled once, outside the timed prediction,
using linear interpolation of raw HU onto the released model's grid:
`(1.0, 0.767578125, 0.767578125)` mm in ZYX order. Both routes received identical
files. All five inspected native CT scans required resampling and were
ineligible for the experimental normalization route; the comparison measures
already-model-grid inputs.

| Case | Shape (Z, Y, X) | Tiles | Host median (s) | MLX median (s) |
| --- | --- | ---: | ---: | ---: |
| MSD Pancreas 052 | 222 × 430 × 430 | 108 | 15.56 | 15.03 |
| HCC-TACE HCC-063 | 248 × 573 × 573 | 192 | 26.78 | 29.17 |
| AMOS22 0013 | 410 × 521 × 521 | 384 | 54.12 | 58.08 |

Preprocessing-stage medians fell by 34–81 ms. Total-time paired ratio intervals
were 0.832–1.056, 0.945–1.172 and 0.960–1.201 respectively (96.875% paired-sign
intervals, six pairs, 2% practical noise threshold). Every interval includes no
change, so the runs establish neither a reliable speedup nor a regression.
Peak resident memory was flat: 5630 MiB for the host route and 5637 MiB for MLX.
Saved labels and affines matched exactly for all cases, with nonempty foreground
and the input geometry preserved.

## Liver 3d_fullres inference speed (Apple M5 Max, MLX 0.32.4)

Measured on an Apple M5 Max (40-core GPU, 128 GB) with MLX 0.32.4 on Metal,
float32 network arithmetic (`MLX_ENABLE_TF32=0`, as set by `medmlx-core`), using
the `nnunet-liver` checkpoint (`PlainConvUNet`, features 32-320, six stages,
patch 128³). The full case is AMOS22 `amos_0008`, preprocessed to
`[1, 505, 616, 616]`: 567 tiles at step 0.5, each run 8 times for mirroring
on all three axes (4,536 forward passes), with nnU-Net's float16 Gaussian
accumulation.

### Where the time goes

The tile loop is GPU-bound. During an 81-tile run the GPU was active 99% of
the wall time and the process accounted for 98% of it, so host transfers,
`mx.eval` boundaries and the slab copy thread do not stall the GPU. About 90%
of tile time is the eight network passes per tile; Gaussian accumulation and
mirroring add the rest. Sustained runs are slower per tile than short bursts
(an isolated mirrored tile varied from 0.93 to 1.23 s over 80 calls) because
the GPU does not hold its burst clock under continuous load.

One forward pass is 955 GFLOP, almost all 3×3×3 convolutions. MLX's float32
matmul peaks at 14.1 TFLOP/s on this GPU (8192³), and most of the network's
convolutions reach 12.7-14 TFLOP/s, so they are close to the hardware limit.
The 0.6.4 forward pass (116 ms) had three avoidable costs:

| Layer (128³) | 0.6.4 | Cause |
| --- | --- | --- |
| Decoder conv 64→32 | 31.8 ms | MLX picks 64×64 output tiles once a 3-D convolution has ≥64 input channels, so half of each tile computes absent output channels. A 32→32 conv at the same size takes 9.1 ms. |
| Encoder conv 1→32 | 6.3 ms | MLX pads the single input channel to 16 for its implicit GEMM, 16 times the needed products. |
| Bias and LeakyReLU | ~1.3 ms per conv block | Two extra full-tensor read and write passes around each instance norm. |

### Changes

- The decoder convolution over the concatenated upsampled and skip tensors
  runs as one convolution per part on the matching weight slice, summed, when
  MLX would otherwise use oversized tiles (two 32-channel parts to 32
  outputs). The concatenated tensor is no longer formed.
- For native 3-D layers the convolution bias and LeakyReLU run inside the
  instance-normalization Metal kernels. The output is bitwise equal to the
  separate operations.

The 1→32 encoder convolution kept MLX's padded path in this change; an im2col
and matmul replacement took 14 ms, slower than that path. It now runs on the
kernel described under
[Single-channel first convolution](#single-channel-first-convolution).

### Results

Paired benchmarks alternate the two versions in fresh processes (perfkit,
99% intervals for the forward pass, 97% for the mirrored tile):

| Measurement | 0.6.4 | This change |
| --- | --- | --- |
| One 128³ forward pass | 115.9 ms | 94.7 ms (×0.81-0.83) |
| One mirrored tile (8 passes) | 1,039 ms | 842 ms (×0.75-0.85) |
| Peak footprint, one forward pass | 3.35 GB | 3.10 GB |
| Peak footprint, one mirrored tile | 12.1 GB | 10.8 GB |

The full `amos_0008` case through `nnunet_mlx.named_models.liver`, one run
each, 0.6.4 first:

| Measurement | 0.6.4 | This change |
| --- | --- | --- |
| Tile inference | 634.2 s | 465.6 s |
| Total wall | 645.0 s | 476.6 s |
| Peak MLX memory | 9.82 GB | 8.95 GB |

Single sequential runs include the sustained-clock variation described above,
so the paired per-tile numbers are the better estimate of the speedup.

Each mirrored view is now queued for evaluation as soon as its graph is built, so
the eight forwards of a tile run one after another instead of interleaved. On a
27-tile, 192×256×256 crop of `amos_0008` with the released weights, the logits are
bitwise unchanged, the time is unchanged (15.4 vs 15.6 s), and the peak footprint
falls from 11.6 to 4.9 GB (MLX peak 6.5 to 3.3 GB). For one mirrored 128³ tile, paired
runs give 924 vs 914 ms (×0.98-1.05) and a footprint of 10.7 vs 4.0 GB.

For comparison, the pinned upstream PyTorch network (nnU-Net 2.8.1,
dynamic-network-architectures 0.4.4, torch 2.13.0) with the same weights takes
291 ms per 128³ forward pass on MPS and 4,148 ms on the CPU.

### Parity

On four 128³ tiles of `amos_0008` that contain liver, the pinned upstream
PyTorch CPU float32 network with the same weights is the reference. The
largest absolute logit difference from it is 1.6e-4 for 0.6.4 and 1.8e-4 for
this change, at most 3.3e-6 of the largest logit, and neither changes the
argmax of any voxel. Splitting the decoder convolution changes logits by at
most 5.3e-5 relative to 0.6.4; fusing bias and LeakyReLU changes no bits.
The full-case label maps of 0.6.4 and this change are identical in all
59,572,224 voxels.

## Host resampling

SciPy resamples large float16 and float32 volumes one channel at a time. For linear
and cubic interpolation, up to eight threads write disjoint slabs on the half-pixel
grid. Linear interpolation reads only the source planes that bracket each slab.
Cubic interpolation computes spline coefficients once across the full channel so slab
boundaries do not affect the result. Float16 logits retain their original rounding
before softmax or argmax.

Separate-axis resampling distributes independent in-plane slices across threads.
On the M5 Max, resampling 57 slices from 5 mm to 1 mm took 180 ms with threading
and 688 ms on one thread. An earlier implementation parallelized whole channels
within a 4 GiB budget. For three float16 logit channels resized from 160×384×384
to 200×480×480, that reduced time from 2.64 to 1.05 s and increased host memory
from 0.55 to 1.43 GB.

## Torch-grid resampling on Metal

`resample_torch_fornnunet` calculates sampling coordinates inside each Metal
thread. It preserves the float32 half-pixel multiply/subtract order, replicated
edges, nearest and nearest-exact modes, and scale-1000 float16 label scores with
the selected tie rule. Label candidates come from the interpolation stencil;
there are no full-volume coordinate, indicator or label-score buffers.
Separate-axis passes retain their intermediate arrays on Metal. Preprocessing
with `prefer_mlx=True` also retains the final image for prediction.

The kernel reuses compensated float32 products for SciPy-style accumulation.
Power-of-two scaling protects finite extremes, and integer mantissa handling
preserves subnormal storage. MLX 0.32.5's built-in `Upsample` uses a different
coordinate expression and intermediate rounding, and does not supply the
nearest-exact and label-selection contract used here.

On Apple M5 Max, macOS 27.2, MLX 0.32.4, NumPy 2.5.4 and SciPy 1.18.1
(November 14, 2026), one float32 HCC_063 CT channel cropped to 128×128×48
and divided by 1024 was linearly resampled to 512³. The public NumPy-to-NumPy
resampling call includes transfers and finite-value validation. Six paired fresh
processes each ran one cold and one subsequent warm call, with device startup
outside the measured call. Medians were:

| Metric | Previous SciPy Torch-grid path | Metal Torch-grid path |
| --- | ---: | ---: |
| Peak added process footprint | 3,073.59 MiB | 661.42 MiB |
| Cold call | 3,232.42 ms | 163.45 ms |
| Warm call | 3,106.89 ms | 23.73 ms |

Peak added footprint fell 78.48%. All 134,217,728 output values in this volume
were bitwise equal. Nine smaller image/precision cases met the existing
`rtol=1e-5, atol=5e-6` criterion (largest absolute error 7.45e-9), and 45 label
cases covering all three interpolation and tie modes across five integer dtypes
were exact. The 125 recorded-upstream, pipeline and precision checks also passed.
These measurements cover resampling, including its host boundary.

## Converting trained models

`nnunet-mlx convert` and `nnunet-mlx predict` read upstream `checkpoint_final.pth`
files with `torch.load(weights_only=True)`, allowing only the NumPy scalar types that
nnU-Net's training log stores. Network weights are renamed and transposed to MLX's
channels-last layout, and loaded strictly into the network built from `plans.json`,
so a missing, extra or misshapen tensor fails the conversion. Plans from before
nnU-Net 2.4 get the `architecture` block that nnU-Net 2.8.1's `ConfigurationManager`
rebuilds for them.

Checks against released checkpoints (Apple M5 Max, MLX 0.32.4):

| Model | Result |
| --- | --- |
| Liver (Zenodo 18807203), `PlainConvUNet`, fold 0 | Weights and `state.json` bitwise identical to the published `MedMLX/nnunet-liver-mlx` bundle; a CT prediction through `nnunet-mlx predict -d 3 -c 3d_fullres` matches that bundle's in every voxel |
| AutoPET II (Zenodo 8362371), `ResidualEncoderUNet`, pre-2.4 plans, folds 0 and 3 | On a random 128³ two-channel patch, the largest logit difference from the pinned PyTorch network is 9.0e-5 and 1.1e-4 (largest logit 33.7 and 36.3); every voxel's label agrees |

Converting the five AutoPET folds takes about 5 s.

## SIAM v0.3

`siam()` runs the five-fold release on Zenodo 17733431 (`Dataset1007_Partialization_MixLowDill`,
`nnUNetTrainerNoDA`, ResEnc XL `3d_fullres`, 256×256×192 patch, 0.75 mm). Its folds
convert with `nnunet-mlx convert` to the tensors published as `MedMLX/siam`. The
upstream reference runs `nnUNetPredictor` with the settings `siam-pred -m 0` uses
(0.5 step, Gaussian weighting, no mirroring, all folds) under PyTorch 2.13 and nnU-Net
2.8.1 on the CPU in float32.

| Case | Folds | Voxels agreeing | Minimum Dice | Largest probability difference |
| --- | --- | --- | --- | --- |
| Full-head T1, 0.88 mm, RAS | 0–4 | 99.995% (496 of 9,144,320 differ) | 0.998 (`Anomalies`, 0.4 mL); all others ≥ 0.9996 | 0.004 (mean 4.3e-6) |
| Full-head T1, 0.88 mm, RAS | 0 | 99.994% | 0.997 (`Anomalies`) | 0.0038 |
| Skull-stripped glioma T1, 1 mm, LPS | 0 | 99.990% | 0.9995 | 0.0040 |

Dice in this table measures agreement with upstream predictions.

The glioma case goes through the runner's reorientation: the same 879 voxels differ
whether its LPS output or the canonical prediction is compared, so restoring the input
grid adds no differences.

On an Apple M5 Max (MLX 0.32.4), the five-fold full-head T1 run takes 22 s (20 tiles,
2.2 s preprocessing, 19.7 s sliding window) with a 16.0 GB MLX peak; the PyTorch CPU
reference takes 1,328 s. Peak memory follows the field of view: a 400 mm head and
shoulders CT reached 27.3 GB.

### Five-fold CPU comparisons, 2026-11-17

Both cases used the pinned CPU predictor described above, with FP32 network
execution and nnU-Net's FP16 sliding-window accumulation. Each of the five
folds covered eight tiles for the full-head T1 and one for the glioma T1.
The full-head CPU run used Linux aarch64 and the glioma CPU run used macOS
arm64. The Metal predictions used MLX 0.32.4 on M5 Max with
`MLX_ENABLE_TF32=0`.

| Case | Differing labels / voxels | Lowest foreground Dice | Max absolute probability difference | Mean absolute probability difference |
| --- | ---: | ---: | ---: | ---: |
| OpenNeuro ds000001 full-head T1 | 2 / 5,898,240 | 0.999998 (`WM`) | 0.0038964 | 2.54e-08 |
| Isotropic skull-stripped glioma T1 | 14 / 8,928,000 | 0.999979 (`CSFv`) | 0.0038977 | 2.38e-07 |

Label-only and saved-probability exports gave the same disagreement counts and
per-class Dice on the input grids. All 17 foreground classes were present in
both cases. The `Anomalies` labels matched exactly. Every six-connected
component overlapped a component of the same class in the other prediction,
in both directions. Rerunning the Metal predictor at `7f62c73` reproduced the
saved masks, affines and probability arrays exactly.

## Bounded allocation and current comparisons

Label-only inference checks restoration geometry and resampling policy before
allocating complete device logits. Unsupported policies take the host path with
bounded device slabs. Host restoration uses one level of resize parallelism, with
concurrency limited by estimated working sets. `--max-inflight-bytes` controls
prepared and queued payloads; oversized preparations drain the queue and run alone,
and oversized exports write inline. Active preprocessing/restoration scratch is
additional to that payload budget.

Mirrored outputs are added in the original order. Running sums are materialized when
retaining all views would exceed 256 MiB. Smaller heads keep queued execution: forcing
every sum to synchronize increased a three-class liver tile from 802 to 836 ms in the
diagnostic probe. With conditional materialization, that tile measured 792 ms and kept
the same 2,999 MiB MLX peak. In a float32 128³, 30-channel `tanh(x * [1,...,30])` buffer
probe, the peak fell from 1,976 to 736 MiB with bitwise-equal outputs.

Checks on Apple M5 Max, MLX 0.32.4, `MLX_ENABLE_TF32=0` (November 13, 2026):

| Comparison | Result |
| --- | --- |
| Released liver fold 0, four real CT 128³ tiles, pinned PyTorch 2.13.0 / nnU-Net 2.8.1 / architecture 0.4.4 | Maximum logit error 7.35e-5; allclose at rtol=atol=1e-4; zero label disagreements |
| Full HCC_063 CT, fold 0, 0.5 step, eight mirrored views: previous versus revised MLX mirror accumulation | All 25,952,256 labels identical; per-class Dice 1.0 |
| Full HCC_063 CT, label-only and saved-probability paths | Both exported and read back on the input grid |
| Full HCC_063 CT versus the pinned CPU predictor, fold 0, 0.5 step, eight mirrored views, each output mode | All 25,952,256 labels identical; liver and cancer Dice 1.0 |
| Full HCC_063 saved probabilities versus the pinned CPU predictor | Maximum absolute difference 9.40e-4; mean absolute difference 6.87e-10 |
| SIAM five-fold full-head T1 (5,898,240 voxels) and isotropic glioma T1 (8,928,000 voxels), output handling | Both decoding modes give identical labels per case; source grids preserved; named-runner LPS reorientation is exact |

The previous-versus-revised MLX row measures an optimization regression. The pinned
CPU rows measure upstream agreement. The
[qualification workflow](../tests/README.md#full-volume-comparisons) records comparison
types separately and reports per-class and component metrics.

The separate HCC_063 annotation comparison selects model class 2 (cancer) and the
sourced DICOM SEG `Mass` segment. Dice is 0.8183 and predicted volume exceeds the
annotation by 10.85 mL. Both annotation components overlap the prediction; seven
candidate components have no annotation overlap, including one at most 100 mm³.
Label-only and saved-probability outputs give the same result on this case.

Two additional prototypes remain experiments. Timings are synchronized warm medians
under the shared GPU lock; their local measurements do not estimate whole-case gains.

| Probe | Baseline → prototype | Numerical result | Decision |
| --- | --- | --- | --- |
| Three-class normalization/label reduction, float32 128³; one thread per voxel | 1.02 → 0.22 ms | Equal labels and validity flags; float16/32, half rounding, ties and invalid-value controls also agree | Retain current kernel pending workflow measurement |
| SIAM fold 0, 256×256×192 forward; asynchronous stage boundaries | 1,060.5 → 1,059.9 ms; 10,206 → 14,466 MiB MLX peak | Bitwise-equal output | Retain synchronous boundaries: similar time with higher memory |

## Sliding-window tile lifetime

The accumulator builds each Gaussian-weighted update before it starts the next
tile. The update owns its inputs, so the accumulator can release its Python
references to the tile output and weighted scores sooner. The same additions
run in tile order.

Measured November 18, 2026 on an Apple M1 Max (64 GB, macOS 27.2.1, Python
3.13.17, MLX 0.32.4, `MLX_ENABLE_TF32=0`). Perfkit alternated fresh processes
at baseline `2cdbac7` and with the tile lifetime change under the shared GPU
lock. Both variants used float32 network arithmetic, float16 sliding-window
accumulation, step 0.5 and `save_probabilities=True`. SIAM ran five folds
(40 tiles, 256×256×192) without mirroring. Pancreas ran the released fold
(80 tiles, 192³) with eight mirrored views per tile.

| Model and full-volume input | Counted pairs | Peak MLX memory (MiB), baseline → candidate | Median process wall time (s), baseline → candidate | Paired wall-time ratio, candidate / baseline |
| --- | ---: | ---: | ---: | --- |
| SIAM, OpenNeuro ds000001 sub-01 T1w (160×192×192) | 6 | 13,156 → 11,428 | 111.43 → 111.42 | 0.9969–1.0032 (96.875% interval) |
| Pancreas, AMOS22 `amos_0008` CT (768×768×101) | 2 | 6,911 → 6,967 | 671.31 → 671.95 | 1.0008–1.0012 (observed range) |

The runner reads the MLX allocator peak after export. Each variant reached
the same rounded peak across its counted runs. The uncounted first pancreas
runs reached 7,129 MiB and 6,967 MiB respectively. SIAM's counted peak fell
13.1% and pancreas's rose 0.8%. SIAM wall time stayed within measurement noise.
Two pancreas pairs cannot establish a 95% timing interval.

The exported mask values, mask affines and float32 probability bits match
exactly. The comparisons cover 5,898,240 mask voxels and 106,168,320
probabilities for SIAM, and 59,572,224 mask voxels and 178,716,672 probabilities
for pancreas.

A separate six-pair forward probe used SIAM fold 0 and the first tile of the
same T1. Adding synchronous evaluation after each full-resolution convolution
and normalization lowered the peak through the first encoder stage from
8,350 to 6,796 MiB. The final decoder still set a 10,252 MiB forward peak.
All 226,492,416 float32 logits matched bit for bit. Warm forward medians were 1,825
and 1,820 ms (96.875% paired ratio interval 0.985–1.013). The model keeps the
existing stage boundaries.

## Full liver CT and TotalSegmentator on M5 Max

On November 22, 2026, nnunet-mlx 0.9.0 and the optimized wheel ran the full
HCC-TACE-Seg HCC_063 CT (512×512×99) on an M5 Max with 128 GB, macOS 27.2.1,
Python 3.13.17 and MLX 0.32.4. The liver model used fold 0, 192 tiles of 128³,
step 0.5, eight mirrored views per tile and float32 probability export.
TotalSegmentator MLX (`b4fa955`) used its default mask output for the 3 mm fast
model (297) and all five 1.5 mm standard models (291–295).

Each process loaded the same released weights and wrote its prediction to disk.
The table gives medians over paired runs in alternating order after one discarded
warmup per variant. Wall time includes process startup and export. Process memory
is the macOS peak physical footprint. MLX memory is the allocator peak. The
paired intervals give optimized time divided by 0.9.0 time, with 96.875%
confidence for six pairs and 96.14% for twelve.

| Workload | Pairs | Wall time (s), 0.9.0 → optimized | Peak process (GiB) | Peak MLX (MiB) | Paired wall-time ratio interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Liver with probabilities | 6 | 204.58 → 184.12 | 9.80 → 6.29 | 3,694 → 2,995 | 0.880–0.978 |
| TotalSegmentator fast | 12 | 1.010 → 0.915 | 7.07 → 6.35 | 5,439 → 4,509 | 0.883–0.919 |
| TotalSegmentator standard | 12 | 10.235 → 10.035 | 7.01 → 6.69 | 4,207 → 3,659 | 0.972–0.989 |

Median liver preprocessing fell from 10.36 to 2.39 s and tile inference from
181.80 to 169.60 s. Restoration and export took 12.12 and 11.82 s respectively.
The optimized wheel finished sooner in all six liver pairs. TotalSegmentator
standard's interval spans the prespecified 2% wall-time improvement threshold.

The compiled tile update weights and adds scores in float32 before rounding once
to float16. Unmirrored FP16 label prediction queues the next network pass after
submitting that update to the accumulation window. This releases the previous
tile's scores while the next forward can still overlap label reduction.
In a separate six-round comparison on TotalSegmentator fast, the later submission
reduced peak MLX memory by another 311 MiB.

Both TotalSegmentator masks and the liver mask matched 0.9.0 voxel for voxel on
the original CT grid. All 77,856,768 float32 liver probabilities were bitwise
identical, and their geometry sidecars matched. Against nnU-Net 2.8.1 on CPU,
the liver and tumor masks each had Dice 1.0 across 25,952,256 voxels. Absolute
probability differences had a maximum of 9.39e-4 and a mean of 6.86e-10.

A single liver pair before these optimizations compared 0.9.0 with `32d3331`,
which already included the single-channel convolution and earlier tile release.
Wall time was 206.48 → 208.83 s, peak process memory was 9.79 → 8.76 GiB and
peak MLX memory was 3,694 → 3,003 MiB. The saved outputs matched.
An earlier six-pair build with the optimized probability path, before the
label-only scheduling change, gave 209.62 → 195.33 s and a wider paired
wall-time ratio interval of 0.854–1.180.

### Eight-view convolution on HCC_063

The saved wheel from the preceding comparison supplied the baseline for this
experiment. Both builds used the same M5 Max, CT, fold, step fraction and
probability export. Each view runs four full-resolution 32→32 convolutions,
one in the encoder, two on the separate decoder inputs and one after their sum.
The revised Metal kernel computes all 32 output channels for 16 adjacent voxels
per SIMD group. It preserves the order of MLX’s float32 matrix products over
16-channel blocks and the 27 spatial taps.

After 12 warmup tiles from the CT, a GPU capture recorded one complete 128³ tile
with all eight views. The table ranks the leading MLX float32 3-D implicit-GEMM
kernels by GPU time in serial replay. Decoder weight copies were submitted
separately for timing. Single-kernel command buffers accounted for 99.13% of
the 709.49 ms measured on the GPU.

| Metal output block (rows × columns) | Dispatches | Total GPU time (ms) | Share |
| --- | ---: | ---: | ---: |
| 32 × 32 | 32 | 271.20 | 38.2% |
| 64 × 64 | 56 | 197.37 | 27.8% |
| 32 × 64 | 88 | 74.85 | 10.5% |

Across four paired fresh processes per build, the median complete mirrored tile
took 773.7 ms with the saved wheel and 732.2 ms with the revised convolution.
All 31,457,280 float32 logits matched bit for bit across five real tiles.

The full-CT comparisons interleaved fresh processes under the shared GPU lock,
with one discarded warmup per build. The table gives median wall time and
median per-process peaks, from the saved wheel to the revised kernel. The final
column is the paired-sign interval for revised wall time divided by saved-wheel
time, with 96.875% confidence for six pairs and 96.14% for twelve.

| HCC_063 workload | Pairs | Wall time (s) | Peak process (GiB) | Peak MLX (MiB) | Paired time ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Liver, eight views and probabilities | 6 | 186.325 → 179.430 | 6.277 → 6.271 | 2,995.09 → 2,995.20 | 0.899–0.997 |
| TotalSegmentator fast (297) | 12 | 0.990 → 0.985 | 6.351 → 6.352 | 4,509.03 → 4,509.03 | 0.990–1.040 |
| Standard, all five parts | 6 | 10.730 → 10.720 | 6.677 → 6.674 | 3,658.80 → 3,658.80 | 0.935–1.043 |
| Organs (291) | 6 | 2.840 → 2.845 | 6.544 → 6.541 | 3,497.76 → 3,497.76 | 0.961–1.071 |
| Vertebrae (292) | 6 | 2.745 → 2.700 | 6.659 → 6.659 | 3,539.70 → 3,539.71 | 0.899–1.041 |
| Cardiac (293) | 6 | 2.680 → 2.700 | 6.189 → 6.189 | 3,372.01 → 3,372.01 | 0.981–1.085 |
| Muscles (294) | 6 | 2.670 → 2.675 | 6.483 → 6.481 | 3,476.80 → 3,476.80 | 0.989–1.027 |
| Ribs (295) | 6 | 2.685 → 2.690 | 6.658 → 6.657 | 3,539.71 → 3,539.71 | 0.978–1.041 |

The revised kernel finished sooner in all six liver pairs, saving 0.51–18.80 s
per pair. Median tile inference fell from 171.63 to 164.67 s. The highest
process footprint across the six runs was 6.288 GiB for the saved wheel and
6.292 GiB for the revised kernel. All 25,952,256 labels and all 77,856,768
float32 probability bit patterns matched. The probability properties files
were identical. The masks retained the CT dimensions, affine, qform and sform.
TotalSegmentator uses ordered float32 arithmetic. Its fast, merged standard
and five individual part masks matched voxel for voxel, and each wall-time
interval included 1.0.

### Constant tiles and probability export on HCC_063

On a host tile where every voxel in each channel has the same float32 bits,
each mirrored input gives the evaluation network identical data. The built-in
plain and residual U-Nets run once on these tiles. Padding can give asymmetric
logits, so all eight outputs are still reoriented and added in the original
order. On one constant 128³ CT tile, this reduced median time from 851 to
103 ms over four repetitions. A central tissue tile used all eight forwards.
Across the two tiles, all 12,582,912 float32 logits matched bit for bit.
Six of the CT's 192 tiles were constant. Reusing their predictions reduced
the number of network evaluations from 1,536 to 1,494.

Probability export streams the same float32 NumPy array into the NPZ archive
with DEFLATE level 3. The eight-view HCC_063 archive went from 199,741,329 to
199,571,434 bytes.

Six full-CT rounds under the shared GPU lock used all six orderings of the
three wheels, with one fresh process per wheel per round and one discarded
warmup per wheel. The table gives medians and the 96.875% paired-sign interval
for the new wall time divided by each reference time. The saved wheel is the
qualified baseline for the convolution experiment above.

| Eight-view build | Wall (s) | Tiles (s) | Restoration and export (s) | Peak process (GiB) | Peak MLX (MiB) | New/reference wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Saved wheel | 182.610 | 167.94 | 11.86 | 6.289 | 2,995.09 | 0.837–0.901 |
| Revised convolution (`1ce73e5`) | 169.850 | 155.21 | 11.90 | 6.290 | 2,995.20 | 0.920–0.965 |
| Constant-tile reuse and level-3 export | 157.555 | 148.68 | 6.13 | 6.282 | 2,995.20 | 1.000 |

The new build saved 17.69–30.67 s per pair against the saved wheel and
5.71–13.56 s against the revised convolution. Its highest process footprint
was 6.291 GiB. Every run reproduced all 25,952,256 labels, all 77,856,768
float32 probability bit patterns and the probability properties file.
The output dimensions, affine, qform and sform matched the source CT.

TotalSegmentator fast, merged standard and each standard part reproduced every
label on the source grid. The table compares the saved wheel with the new build.
The paired wall-time intervals include 1.0 for all seven workloads, with
96.14% confidence for fast (12 rounds) and 96.875% for the others (six rounds).
Median peak MLX memory differed by less than 0.01 MiB in every workload.

| TotalSegmentator workload | Wall (s), saved → new | Peak process (GiB), saved → new | Peak MLX (MiB), new | New/saved wall ratio |
| --- | ---: | ---: | ---: | ---: |
| Fast | 0.920 → 0.935 | 6.353 → 6.354 | 4,509.03 | 1.000–1.033 |
| Standard, all five parts | 10.035 → 10.010 | 7.029 → 6.692 | 3,658.80 | 0.995–1.010 |
| Organs | 2.570 → 2.605 | 6.542 → 6.542 | 3,497.76 | 1.000–1.213 |
| Vertebrae | 2.610 → 2.580 | 6.657 → 6.657 | 3,539.71 | 0.977–1.023 |
| Cardiac | 2.565 → 2.575 | 6.192 → 6.194 | 3,372.01 | 0.985–1.032 |
| Muscles | 3.040 → 3.095 | 6.483 → 6.483 | 3,476.80 | 0.968–1.056 |
| Ribs | 3.225 → 3.215 | 6.654 → 6.660 | 3,539.70 | 0.985–1.035 |

### One view on HCC_063

With `use_mirroring=False`, the saved wheel completed one fresh-process run in
33.17 s, 150.95 s faster than its recorded eight-view median of 184.12 s. The
two masks had Dice of 0.99145 for liver and 0.97815 for tumor. Against the
HCC_063 DICOM SEG Mass annotation, tumor Dice was 0.82002 with one view and
0.81830 with eight. Both predictions overlapped both annotated components using
six-neighbor connectivity. The revised convolution reproduced the one-view mask
and probabilities bit for bit in a 35.20 s repeat. With level-3 export, one
fresh process reproduced the same one-view outputs in 28.41 s, saving 129.15 s
against the new eight-view median. Its peak process footprint was 6.439 GiB
and peak MLX memory was 2,819.20 MiB. The default uses eight views.

## MLX buffer-cache sweep

The AMOS22 `amos_0008` pancreas run spent most of its peak process footprint
on cached Metal buffers and large host allocations. The table below is from
a separate attribution run on the Apple M1 Max with the default MLX cache
limit. It uses the full 768×768×101 CT, fold 0, step 0.5, mirroring disabled
and saved probabilities.

| Point in the run | MLX active (GiB) | MLX cache (GiB) | MLX peak so far (GiB) | Physical footprint (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Model loaded | 0.38 | 0.26 | 0.52 | 0.78 |
| Input preprocessed | 0.38 | 0.26 | 0.52 | 3.55 |
| Tile inference finished | 0.38 | 11.79 | 6.49 | 15.12 |
| Probabilities and labels restored | 0.38 | 11.79 | 6.49 | 21.48 |
| Peak sampled during output read-back | 0.38 | 11.79 | 6.49 | 21.62 |
| Files published | <0.01 | 12.17 | 6.49 | 21.62 |

The 50 ms sampler read the MLX allocator counters and the macOS physical
footprint together. The peak sample matched the kernel's footprint high-water
mark. The cache accounted for 11.79 GiB, or 54.5% of that peak. There was
9.45 GiB outside the active and cached MLX totals. A macOS `footprint`
snapshot during restoration attributed 12.18 GiB to graphics
and 8.64 GiB to large malloc regions at a footprint of 20.98 GiB. The initial
probe peaked at 21.65 GiB during restoration, with 11.78 GiB cached.

In this run, the host retained a 431 MiB float32 network input and 646 MiB of
float16 logits through restoration. Export also held 682 MiB of float32 probabilities,
114 MiB of uint16 labels and a 227 MiB int32 mask read back for validation.
Large malloc regions also cover resampling workspace and pages retained by
the host allocator.

With the experimental 512 MiB limit, the same case peaked at 10.51 GiB during
read-back. At the peak, MLX held 0.38 GiB active and 0.68 GiB cached. MLX trims
the cache on allocation, so frees can take it above the configured limit
between allocations. [MLX 0.32.5 Metal allocator](https://github.com/ml-explore/mlx/blob/v0.32.5/mlx/backend/metal/allocator.cpp#L174-L210)

Host restoration and export still set the peak. The anisotropic resampler
used a 2.22 GiB float64 in-plane workspace. Later, the 682 MiB resampled
logits and the 682 MiB probability volume were live together during decoding.
At read-back, 9.45 GiB of the footprint lay outside the active and cached MLX
totals. Single-case prediction now releases the preprocessed CT when inference
finishes and retains its geometry and shape for export.

### Paired results

Measured November 18, 2026 on an Apple M1 Max with 64 GB of unified memory,
macOS 27.2.1, Python 3.13.17 and MLX 0.32.4. The baseline was `2cdbac7`
with the sliding-window tile-lifetime change above. Each fresh process called
`medmlx.run` with fold 0, step 0.5, mirroring disabled, FP32 network arithmetic
(`MLX_ENABLE_TF32=0`), FP16 accumulation and probability export. The default
cache limit on this machine was 60.8 GiB. Each variant changed only the
process-local MLX cache limit.

The inputs were the paired CT and SUV PET for AutoPET `0117d7f11f` (400×400×284),
the AMOS22 `amos_0008` and MSD `pancreas_001` CTs (768×768×101 and 512×512×110),
and the four-sequence BraTS `BRATS_001` MRI (240×240×155). The released AutoPET,
pancreas and BraTS checkpoints were from Hugging Face revisions
`655190e49d0bb8cced452984fb458f2f69cd5348`,
`6bbcfea84c4a317b9deb8854eafb7392cd0cc8a6` and
`b77b65309ac6f3cd747d5b849700ee866eac220a`, respectively.
The runs processed 144, 80, 18 and 8 tiles for AutoPET, AMOS22, MSD and BraTS,
respectively.

Perfkit interleaved full processes under the shared GPU lock. One warm-up per
variant was excluded. Wall time includes loading through output publication.
Tile time is the runner's inference-stage metric. The memory columns show the
largest counted MLX allocator peak and macOS `time -l` physical footprint.
Six-round ratios use perfkit's 96.875% paired-sign interval. Ratio bounds from
two-round sweeps are observed ranges. The AMOS22 sweeps each have their own
paired default-limit baseline.

| Full-volume case | Cache limit | Counted rounds | Median wall (s) | Median tile inference (s) | Peak MLX (MiB) | Peak footprint (GiB) | Paired wall-time ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BraTS `BRATS_001` | Default | 6 | 5.56 | 2.25 | 3,063.41 | 5.21 | — |
| BraTS `BRATS_001` | 0 | 6 | 5.72 | 2.40 | 3,063.41 | 3.41 | 1.0211–1.0331 |
| BraTS `BRATS_001` | 256 MiB | 6 | 5.72 | 2.41 | 3,063.41 | 3.43 | 1.0177–1.0357 |
| BraTS `BRATS_001` | 512 MiB | 6 | 5.66 | 2.32 | 3,063.41 | 3.89 | 1.0058–1.0244 |
| BraTS `BRATS_001` | 1 GiB | 6 | 5.62 | 2.30 | 3,063.41 | 4.07 | 1.0050–1.0188 |
| BraTS `BRATS_001` | 2 GiB | 6 | 5.59 | 2.28 | 3,063.43 | 4.71 | 0.9959–1.0103 |
| AMOS22 `amos_0008` | Default | 6 | 104.58 | 83.50 | 6,858.52 | 21.64 | — |
| AMOS22 `amos_0008` | 512 MiB | 6 | 124.08 | 102.92 | 6,640.82 | 10.51 | 1.1800–1.1913 |
| AMOS22 `amos_0008` | 1 GiB | 6 | 120.15 | 99.03 | 6,640.82 | 10.88 | 1.1301–1.1494 |
| AMOS22 `amos_0008` | Default | 2 | 104.88 | 83.50 | 6,640.82 | 21.63 | — |
| AMOS22 `amos_0008` | 0 | 2 | 125.49 | 104.29 | 6,640.82 | 9.83 | 1.1917–1.2013 (observed) |
| AMOS22 `amos_0008` | 256 MiB | 2 | 125.47 | 104.37 | 6,640.82 | 10.51 | 1.1944–1.1981 (observed) |
| AMOS22 `amos_0008` | 2 GiB | 2 | 119.59 | 98.46 | 6,640.82 | 11.98 | 1.1369–1.1437 (observed) |
| MSD `pancreas_001` | Default | 2 | 30.35 | 19.88 | 6,155.88 | 13.32 | — |
| MSD `pancreas_001` | 0 | 2 | 34.93 | 24.47 | 6,155.88 | 6.55 | 1.1499–1.1524 (observed) |
| MSD `pancreas_001` | 256 MiB | 2 | 34.95 | 24.52 | 6,155.88 | 6.81 | 1.1453–1.1582 (observed) |
| MSD `pancreas_001` | 512 MiB | 2 | 35.00 | 24.58 | 6,155.88 | 6.99 | 1.1534–1.1536 (observed) |
| MSD `pancreas_001` | 1 GiB | 2 | 33.76 | 23.35 | 6,155.88 | 7.55 | 1.1115–1.1137 (observed) |
| MSD `pancreas_001` | 2 GiB | 2 | 33.76 | 23.25 | 6,155.88 | 8.57 | 1.1115–1.1134 (observed) |
| AutoPET `0117d7f11f` | Default | 2 | 58.94 | 49.13 | 2,853.42 | 7.02 | — |
| AutoPET `0117d7f11f` | 0 | 2 | 72.62 | 62.86 | 2,620.61 | 4.01 | 1.2287–1.2354 (observed) |
| AutoPET `0117d7f11f` | 256 MiB | 2 | 69.18 | 59.36 | 2,620.61 | 4.28 | 1.1719–1.1755 (observed) |
| AutoPET `0117d7f11f` | 512 MiB | 2 | 69.38 | 59.62 | 2,620.62 | 4.67 | 1.1762–1.1779 (observed) |
| AutoPET `0117d7f11f` | 1 GiB | 2 | 66.60 | 56.81 | 2,620.62 | 5.08 | 1.1288–1.1311 (observed) |
| AutoPET `0117d7f11f` | 2 GiB | 2 | 59.74 | 49.96 | 2,620.61 | 5.45 | 1.0126–1.0147 (observed) |

Direct file comparison found identical mask, probability archive and probability
properties bytes against the default-limit output for every input and limit.

No cache limit qualified. On BraTS, the 0 and 256 MiB caps had wall-time upper
bounds of +3.31% and +3.57%, and 2 GiB increased the maximum MLX peak by
20,352 bytes. Both observed AMOS22 pairs at 2 GiB also exceeded +13.6% wall
time. On that CT, 512 MiB and 1 GiB cut peak process footprint by
51.4% and 49.8%. Their wall-time intervals were +18.0% to +19.1% and +13.0% to
+14.9%, exceeding the +3% limit.

## Single-channel first convolution

MLX 0.32.5 zero-pads the input and weight of a 3-D convolution up to a multiple of
16 channels when the input channel count is not a multiple of 16. The first
convolution of a one-channel model therefore computes sixteen products per kernel tap
and holds a 16-channel copy of each tile. A convolution from one float32 channel to 32 outputs with
a 3×3×3 kernel, stride 1, padding 1 and dilation 1, the first convolution of the
released liver model, now runs on a Metal kernel: one thread per voxel reads the 27
taps once and accumulates every output channel with `fma` in weight order. The bias
stays in the instance-normalization kernel. Every other convolution keeps `mx.conv3d`.

The kernel returns the bits of `mx.conv3d`. On MLX 0.32.5, nine inputs were compared
with the released liver weights and again with random weights, 943,017,312 output
values each time, without a differing bit pattern: five 128³ HCC_063 CT tiles
(interior, first, last, edge, and mirrored on all axes), a 97×153×131 crop, a batch
of two tiles, a 64³ array with zeros, negative zeros, subnormals and rescaled values,
and a random 256×256×192 array. SIAM and the pancreas model have the same first
convolution; their fold-0 logits on one HCC_063 CT patch each (256×256×192 and 192³)
are also bitwise unchanged, as are the liver fold-0 logits summed over eight mirrored
passes of one 128³ tile.

Measured on Apple M5 Max, MLX 0.32.4, `MLX_ENABLE_TF32=0` (November 17, 2026). Paired
runs alternate the two paths in fresh processes under the shared GPU lock for six
rounds (perfkit, 97% intervals):

| Measurement | `mx.conv3d` | Metal kernel |
| --- | ---: | ---: |
| Bias-free 1→32 convolution of one 128³ CT tile | 5.29 ms | 0.84 ms (×0.157-0.163) |
| Peak MLX memory of that convolution, buffer cache cleared | 511 MiB | 383 MiB |
| One mirrored liver tile (8 passes) | 872.9 ms | 810.1 ms (×0.914-0.940) |
| Peak MLX memory, one mirrored tile | 2,999 MiB | 2,356 MiB |

MLX 0.32.5 marks the sliced output of its padded small-channel convolution as
contiguous, so the normalization kernel no longer copies the `mx.conv3d` output.
Eight mirrored passes of one 128³ liver tile, each evaluated before the next, peak
at 2,340 MiB on both paths with MLX 0.32.5, against 2,419 MiB (`mx.conv3d`) and
2,212 MiB (kernel) with MLX 0.32.4 in the same script. On MLX 0.32.5 the kernel
saves compute, not tile memory.

The full HCC_063 CT with the released liver weights, fold 0, 0.5 step and eight
mirrored views (192 tiles, 1,536 network passes), label-only output, as medians and
ranges of the six rounds:

| Measurement | `mx.conv3d` | Metal kernel |
| --- | ---: | ---: |
| Whole-case wall time | 189.4 s (185.7-197.3) | 168.3 s (163.9-178.1) |
| Tile inference | 179.0 s (175.0-186.9) | 158.1 s (153.7-167.8) |
| Peak MLX memory | 4,160 MiB | 3,517 MiB |

The kernel path was faster in every round, by 12.0 to 30.7 s (×0.842-0.937); the
mirrored-tile difference alone accounts for 12 s over 192 tiles. Each path reached the
same peak in every run. With saved probabilities the peak fell from 3,694 to 3,051 MiB.

The label-only and saved-probability outputs of the two paths are identical: the masks
agree in all 25,952,256 voxels and the 77,856,768 saved float32 probabilities have
equal bit patterns, which are also those of the November 13 Metal outputs. The
comparisons with the pinned CPU predictor above are therefore unchanged: no label
disagreements, liver and cancer Dice 1.0, maximum absolute probability difference
9.39e-4 and mean 6.86e-10.
