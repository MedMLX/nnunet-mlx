# Local checks

`make test-host` runs self-contained metadata, cache, publication, security and
controlled numerical checks. It needs neither MLX nor sourced images or untracked
recordings. Collection keeps network imports inside the tests that execute them.

`make preflight` reports every missing qualification dependency and input together,
with the pinned versions, asset locations and recording commands. `make test-qualified`
runs that preflight and then the suite on Apple silicon. `make qa` runs Ruff and strict
Pyright in the existing environment. Use `make test TESTS=tests/test_export.py` for an
affected test file; `PYTEST_ARGS` passes through pytest options.

## Image I/O and upstream recordings

The image I/O and Torch oracle tests read HCC-TACE-Seg HCC_063's contrast CT,
its liver mask, and a chest X-ray from `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`,
`hcc_063_liver.nii.gz`, and `chest_xray.jpg`. `MEDMLX_TEST_CT` and
`MEDMLX_TEST_XRAY` override those paths. The CT I/O cases use crops and
set controlled geometry to compare the readers and writers with nnU-Net. The
network, sliding-window, and ensemble parity tests use CT crops for 3D inputs
and X-ray crops for 2D inputs.

Run `python scripts/record_upstream_runner.py` in the provisioned nnU-Net
oracle environment to regenerate the HCC-TACE runner, fold, and restoration
references under `work/real-reference`. The tracked archive retains independent
head-conversion and rounding probes; it does not contain CT voxels.

`scripts/record_torch_resampling.py --source PATH_TO_UPSTREAM` records the
resampling and raw-case preprocessing references from upstream commit
`17cb295b6258c660b7c6ea84f3bcb67f1c7a1e3f`, using Torch 2.13.0 and NumPy 2.5.4.
Its output is `work/real-reference/torch_resampling.npz`. The tracked archive
keeps label, class-logit and crop-mask controls; sourced image operands and their
interpolated outputs stay in the ignored recording.

## Full-volume comparisons

Run the released fold configuration on each complete input and retain separate
label-only and probability-saving reference masks. Include isotropic and anisotropic
images, class and region heads. Keep model revisions, exact plans/dataset JSON,
checkpoint sizes and modification times, runtime versions, arithmetic, accumulation,
folds and mirroring settings in the reference and candidate information files. Record
the candidate Git revision and its patch when the tree has uncommitted changes.

The pinned upstream CPU recorder writes both decoding paths from one inference:

```bash
.venv/bin/python scripts/record_model_reference.py \
  --model models/upstream --input images/case_a.nii.gz \
  --work results/reference --threads 6 --timeout-seconds 10800
```

Every available fold is used unless `--fold` is supplied; mirroring follows the
checkpoint unless `--disable-mirroring` is supplied. Completed mirrored tiles and
folds are cached without changing upstream accumulation or fold averaging. Repeating
the same command resumes an interrupted run. Changed inputs, plans, checkpoints,
runtime versions or settings require a new work directory. Tile caches are removed
after their fold completes. `--cache-folds-only` limits disk use for fast references.
This recorder supports trusted NIfTI model folders without a cascade. Run large CPU
references serially when their convolution working sets would exceed available RAM.

Append each comparison to one versioned matrix:

```bash
.venv/bin/python scripts/qualification_report.py \
  --case case_a --mode label-only --comparison upstream \
  --candidate-mask results/candidate/mask.nii.gz \
  --reference-mask results/reference/label-mask.nii.gz \
  --candidate-info results/candidate/run.json \
  --reference-info results/reference/reference.json \
  --dataset-json models/model/dataset.json \
  --output results/qualification.json
```

Use `--mode probabilities` and the corresponding masks for the other decoding path;
add `--candidate-probabilities` and `--reference-probabilities` to compare their `.npz`
arrays and reader-grid `.pkl` metadata. `--comparison previous-mlx` records an
optimization regression comparison; `--comparison annotation` records comparison
with annotated labels. With `upstream`, Dice measures agreement with upstream output.

The matrix reports per-class disagreement, Dice, volume change, symmetric boundary
voxel-center distances, missed reference components and unmatched candidate components.
A component matches when it overlaps the other mask. Boundaries and components use six-neighbor
connectivity; distances use the full affine under nnU-Net's millimeter input contract.
Small components are at most 100 mm³. Absent-class Dice is null. A rerun replaces the
same case/mode/comparison row. Geometry must agree before metrics are calculated.
Keep matrices and prediction files in ignored `results/`; library documentation retains
only the summary measurements needed to interpret them.
