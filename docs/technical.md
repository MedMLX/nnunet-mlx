# Technical notes

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

The 1→32 encoder convolution is unchanged; an im2col and matmul replacement
took 14 ms, slower than MLX's padded path.

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

The glioma case goes through the runner's reorientation: the same 879 voxels differ
whether its LPS output or the canonical prediction is compared, so restoring the input
grid adds no differences.

On an Apple M5 Max (MLX 0.32.4), the five-fold full-head T1 run takes 22 s (20 tiles,
2.2 s preprocessing, 19.7 s sliding window) with a 16.0 GB MLX peak; the PyTorch CPU
reference takes 1,328 s. Peak memory follows the field of view: a 400 mm head and
shoulders CT reached 27.3 GB.
