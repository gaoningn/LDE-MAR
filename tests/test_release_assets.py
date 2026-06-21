import hashlib

import pytest
import torch

from check_dicom_phi import scan_dicom_files
from networks.motion_denoise_generator import MDGenerator
from runtime_utils import DEFAULT_DATA_ROOT, DEFAULT_WEIGHT_PATH, is_git_lfs_pointer


ASSETS_PRESENT = DEFAULT_WEIGHT_PATH.is_file() and DEFAULT_DATA_ROOT.is_dir()


@pytest.mark.skipif(not ASSETS_PRESENT, reason="Release weight and example DICOM assets are not present")
def test_release_assets_are_materialized_and_deidentified():
    assert not is_git_lfs_pointer(DEFAULT_WEIGHT_PATH)
    files, findings = scan_dicom_files([DEFAULT_DATA_ROOT])
    assert len(files) == 200
    assert findings == []


@pytest.mark.skipif(not ASSETS_PRESENT, reason="Release weight and example DICOM assets are not present")
def test_release_checkpoint_checksum_and_structure():
    digest = hashlib.sha256()
    with DEFAULT_WEIGHT_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == "f3f5e290f5fa1446c8d82671c7b93a6e8b8608d52b5c1ca14cc6357b4402591b"

    state_dict = torch.load(DEFAULT_WEIGHT_PATH, map_location="meta", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    with torch.device("meta"):
        model = MDGenerator(
            image_size=256,
            style_dim=512,
            feature_dim=40,
            motion_dim=40,
            scale=1,
            pure_data=False,
        )
    expected = model.state_dict()
    assert state_dict.keys() == expected.keys()
    assert all(state_dict[key].shape == expected[key].shape for key in expected)
