import pytest
import torch

from networks.motion_denoise_generator import MDGenerator
from runtime_utils import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_CLEAN_PATH,
    DEFAULT_WEIGHT_PATH,
    PROJECT_ROOT,
    is_git_lfs_pointer,
    load_model_weights,
    load_state_dict,
    resolve_device,
)


def test_release_default_paths_are_repository_anchored():
    assert DEFAULT_WEIGHT_PATH == PROJECT_ROOT / "weights" / "best.pth"
    assert DEFAULT_CLEAN_PATH == PROJECT_ROOT / "examples/public_test_data/Original/SHtest/group_004"
    assert DEFAULT_ARTIFACT_PATH == PROJECT_ROOT / "examples/public_test_data/Artificial/SHtest/cardiac_arr_004"


def test_checkpoint_wrapper_loads_strictly(tmp_path):
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    target = torch.nn.Linear(3, 2)
    load_model_weights(target, checkpoint, torch.device("cpu"))
    for source_value, target_value in zip(source.parameters(), target.parameters()):
        assert torch.equal(source_value, target_value)


def test_lfs_pointer_has_actionable_error(tmp_path):
    pointer = tmp_path / "model.pth"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 1\n",
        encoding="ascii",
    )
    assert is_git_lfs_pointer(pointer)
    with pytest.raises(RuntimeError, match="git lfs pull"):
        load_state_dict(pointer, torch.device("cpu"))


def test_unavailable_cuda_is_rejected():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            resolve_device("cuda:0")


def test_artifact_model_forward_shapes_on_meta_device():
    with torch.device("meta"):
        model = MDGenerator(
            image_size=256,
            style_dim=512,
            feature_dim=40,
            motion_dim=40,
            scale=1,
            pure_data=False,
        )
    model = model.to_empty(device="meta")
    source = torch.empty(1, 1, 256, 256, device="meta")
    target = torch.empty(2, 1, 256, 256, device="meta")
    output, coefficients = model.pure_motion_img(source, target)
    assert output.shape == (2, 1, 256, 256)
    assert [value.shape for value in coefficients] == [(2, 40), (2, 40), (2, 40)]
