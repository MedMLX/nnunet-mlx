# Image I/O tests

The image I/O and Torch oracle tests read HCC-TACE-Seg HCC_063's contrast CT
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
