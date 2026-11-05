# Third-party notices

This repository implements compatibility with the following upstream projects.
Their licenses apply to upstream code and dependencies; the repository's own
license remains in `LICENSE`.

## nnU-Net

- Project: MIC-DKFZ nnU-Net
- Source: <https://github.com/MIC-DKFZ/nnUNet>
- Compatibility pin: v2.8.1,
  `468cf803df9b267150ae2b6c0c59b8ac84f16227`
- License: Apache License 2.0

Selected later changes are adapted for the MLX project without changing that
reference pin: foreground sampling store v1 at
[the September 14 merge](https://github.com/MIC-DKFZ/nnUNet/commit/ded2aa3a4c81a9caae37054224d8ebac2d19a061)
and [float32 resampling intermediates](https://github.com/MIC-DKFZ/nnUNet/commit/54c7480e2a8dc1798856b6463ca8fe11197d7e36).
Safetensors checkpoint import follows the format-v1 inference fields and tensor
references defined by [the upstream checkpoint reader](https://github.com/MIC-DKFZ/nnUNet/blob/17cb295b6258c660b7c6ea84f3bcb67f1c7a1e3f/nnunetv2/utilities/checkpoints.py).
Explicit segmentation tie-breaking follows
[the resampler at the same revision](https://github.com/MIC-DKFZ/nnUNet/blob/17cb295b6258c660b7c6ea84f3bcb67f1c7a1e3f/nnunetv2/preprocessing/resampling/default_resampling.py).

## batchgenerators

- Project: MIC-DKFZ batchgenerators
- Source: <https://github.com/MIC-DKFZ/batchgenerators>
- Label-indicator argmax and tie-breaking reference: 0.25.5
- License: Apache License 2.0

## dynamic-network-architectures

- Project: MIC-DKFZ dynamic-network-architectures
- Source: <https://github.com/MIC-DKFZ/dynamic-network-architectures>
- Reference pin: v0.4.4, `d35f85c7b92cbf3e9443b412fdc7c81f8e2e7b12`
- License: Apache License 2.0

## MLX

- Project: Apple MLX
- Source: <https://github.com/ml-explore/mlx>
- Historical MLX reference recorded for this port: v0.32.3,
  `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`
- Declared runtime dependency: `mlx==0.32.4` (see `pyproject.toml`).
  This runtime version is separate from the historical source reference above.
- Derived kernels retain Apple copyright and MIT terms.
- License: MIT

Pretrained nnU-Net weights remain subject to the licenses and terms of their
training datasets and distributors. This repository does not redistribute
weights.

## Project notice

nnU-Net MLX

This product includes software developed by Joseph Sandoval and is licensed
under the Apache License, Version 2.0 (see LICENSE).

Kernels in src/nnunet_mlx/kernels/ are derived in part from Apple MLX,
Copyright 2023 Apple Inc., licensed under the MIT License
(see THIRD_PARTY_NOTICES.md).

See THIRD_PARTY_NOTICES.md for the upstream projects this software is
compatible with.


## Example images

`docs/example.png` is derived from an
[AMOS22](https://zenodo.org/records/7262581) CT and a
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) brain MRI
(CC BY-SA 4.0). The AMOS22 Zenodo record lists CC BY 4.0, while the dataset's
own metadata lists CC BY-SA 4.0. The derived `docs/example.png` is shared
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) pending
clarification of the AMOS22 terms. These images retain their own terms;
this repository's Apache-2.0 license does not relicense them.

## Full license: MLX-LICENSE

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
