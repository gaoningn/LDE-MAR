import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def read_dcm(folder_path, target_size=(256, 256)):
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {folder_path}")

    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(str(folder_path))
    if not dicom_series:
        raise ValueError(f"No DICOM series found in: {folder_path}")

    reader.SetFileNames(dicom_series)
    image3d = reader.Execute()
    volume = torch.from_numpy(sitk.GetArrayFromImage(image3d).astype(np.float32))[:, None]
    _, _, height, width = volume.shape
    if height != width:
        size = max(height, width)
        images = volume.new_zeros((volume.shape[0], 1, size, size))
        if height > width:
            start = (height - width) // 2
            images[:, :, :, start:start + width] = volume
        else:
            start = (width - height) // 2
            images[:, :, start:start + height, :] = volume
    else:
        images = volume

    images = F.interpolate(images, size=target_size, mode="bilinear", align_corners=False)
    flat = images.contiguous().reshape(-1)
    low = torch.quantile(flat, 0.005)
    high = torch.quantile(flat, 0.995)
    images = torch.clamp(images, low, high)
    images = (images - low) / (high - low).clamp_min(1e-8)
    return images * 2.0 - 1.0


def _numeric_suffix(name):
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def _subject_dirs(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
    return sorted(path for path in root_dir.iterdir() if path.is_dir())


def collect_sequence_pairs(root_dir, max_subjects=None):
    root_dir = Path(root_dir)
    original_subjects = {path.name: path for path in _subject_dirs(root_dir / "Original")}
    artificial_subjects = {path.name: path for path in _subject_dirs(root_dir / "Artificial")}
    subject_names = sorted(original_subjects.keys() & artificial_subjects.keys())
    if max_subjects is not None:
        if max_subjects <= 0:
            raise ValueError("max_subjects must be positive")
        subject_names = subject_names[:max_subjects]

    pairs = []
    for subject in subject_names:
        originals = {
            _numeric_suffix(path.name): path
            for path in original_subjects[subject].iterdir()
            if path.is_dir() and path.name.startswith("group_") and _numeric_suffix(path.name) is not None
        }
        artificials = {
            _numeric_suffix(path.name): path
            for path in artificial_subjects[subject].iterdir()
            if path.is_dir()
            and path.name.startswith("cardiac_arr_")
            and _numeric_suffix(path.name) is not None
        }
        for index in sorted(originals.keys() & artificials.keys()):
            pairs.append((subject, originals[index], artificials[index]))

    if not pairs:
        raise RuntimeError(
            f"No matched Original/group_* and Artificial/cardiac_arr_* directories found under: {root_dir}"
        )
    return pairs
