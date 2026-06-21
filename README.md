# LDE-MAR Inference

Inference and evaluation code for **Latent-Dictionary Encoding for
Motion-Artifact Reduction (LDE-MAR)** in cardiac cine MRI. This repository
contains the final artifact-reduction model definition and testing pipeline;
training code and training data are intentionally not included.

This release is intended for research use only. It has not been validated for
clinical diagnosis or patient-care decisions.

## Contents

```text
.
|-- infer.py                     # One paired sequence; exports PNG frames
|-- evaluate.py                  # Dataset-level MAE/SSIM/PSNR evaluation
|-- check_dicom_phi.py           # Release-time DICOM metadata check
|-- deidentify_dicom.py          # Build a metadata-stripped public DICOM copy
|-- dcmreader.py                 # DICOM loading and paired-path discovery
|-- inference_utils.py           # Shared inference and metric functions
|-- runtime_utils.py             # Model construction and strict weight loading
|-- networks/                    # Complete LDE-MAR model definition
|-- weights/best.pth            # Final checkpoint (Git LFS)
`-- examples/public_test_data/   # Metadata-stripped paired test set (Git LFS)
```

## Requirements

- Python 3.9 or newer
- PyTorch 2.0 or newer
- CUDA or Apple Silicon MPS acceleration recommended; CPU inference is supported but slow
- Git LFS for downloading the checkpoint and example DICOM files

The released network has 226,816,753 parameters. The default batch size is one
to limit inference memory use.

## Installation

Install Git LFS before cloning, then pull the release assets:

```bash
git lfs install
git clone https://github.com/gaoningn/LDE-MAR.git
cd LDE-MAR
git lfs pull
```

Create an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Release assets

The default commands use paths anchored to the repository directory, so they
work regardless of the current shell directory:

```text
weights/best.pth
examples/public_test_data/Original/SHtest/group_004/
examples/public_test_data/Artificial/SHtest/cardiac_arr_004/
```

The clean and artifact series must have the same frame count. The checkpoint
must match `image_size=256`, `style_dim=512`, `feature_dim=40`,
`motion_dim=40`, and `pure_data=False`. Missing LFS content and incompatible
checkpoints produce explicit errors.

## Single-sequence inference

Run the included example with automatic device selection:

```bash
python infer.py
```

Outputs are written to `outputs/example/{artifact,clean,prediction}/`. The
command also reports frame-averaged MAE, SSIM, and PSNR.

Custom paths and device:

```bash
python infer.py \
  --clean-path /path/to/Original/SHtest/group_004 \
  --artifact-path /path/to/Artificial/SHtest/cardiac_arr_004 \
  --model-path /path/to/best.pth \
  --device cuda:0 \
  --output-dir outputs/custom
```

## Dataset evaluation

The evaluation root must contain matched subject and numeric group names:

```text
<data-root>/
|-- Original/<subject>/group_NNN/
`-- Artificial/<subject>/cardiac_arr_NNN/
```

Evaluate the included sample or a complete test set:

```bash
python evaluate.py

python evaluate.py \
  --data-root /path/to/test \
  --model-path weights/best.pth \
  --device cuda:0 \
  --log-file outputs/test_metrics.tsv
```

The TSV contains per-sequence metrics and a final frame-weighted mean.

## Preprocessing

Each DICOM series is loaded as `[frames, 1, height, width]`. Frames are
zero-padded to square, resized to `256 x 256`, clipped using the sequence-level
0.5th and 99.5th intensity percentiles, and mapped to `[-1, 1]`.

## Privacy check

The tracked public copy contains newly written pixel data and only the minimal
metadata needed to form a DICOM series. Keep source DICOM files outside the Git
repository. Maintainers can rebuild and verify a public copy with:

```bash
python deidentify_dicom.py /path/to/private_test_data examples/public_test_data --overwrite
python check_dicom_phi.py examples/public_test_data
```

The command fails on common patient, institution, and physician fields, private
DICOM tags, unreadable files, or an empty input directory. This check reduces
risk but does not replace institutional review and authorization.

## Reference result

Using the included `weights/best.pth` and all five public test pairs (100 frames),
the frame-weighted result with PyTorch 2.8.0 on Apple MPS was:

| MAE | SSIM | PSNR |
|---:|---:|---:|
| 0.025720 | 0.929048 | 30.276496 |

Per-sequence results are written to `outputs/test_metrics.tsv` by
`python evaluate.py`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python -m pyflakes .
```

## Checksum

The committed checksum for `weights/best.pth` is stored in
`weights/SHA256SUMS`:

```bash
shasum -a 256 -c weights/SHA256SUMS
```

## License

The project code is released under the MIT License. See `LICENSE` and retain
the third-party notices in `THIRD_PARTY_NOTICES.md` when redistributing it.
