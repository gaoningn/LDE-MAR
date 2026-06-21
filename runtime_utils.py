from pathlib import Path

import torch

from networks.motion_denoise_generator import MDGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "weights" / "best.pth"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "examples" / "public_test_data"
DEFAULT_CLEAN_PATH = DEFAULT_DATA_ROOT / "Original" / "SHtest" / "group_004"
DEFAULT_ARTIFACT_PATH = DEFAULT_DATA_ROOT / "Artificial" / "SHtest" / "cardiac_arr_004"


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but MPS is unavailable")
    return device


def validate_image_size(image_size):
    if image_size != 256:
        raise ValueError("The released checkpoint requires --img-size 256")


def _create_model(image_size=256, style_dim=512):
    validate_image_size(image_size)
    if style_dim != 512:
        raise ValueError("The released checkpoint requires --style-dim 512")
    return MDGenerator(
        image_size=image_size,
        style_dim=style_dim,
        feature_dim=40,
        motion_dim=40,
        scale=1,
        pure_data=False,
    )


def build_model(device, image_size=256, style_dim=512):
    return _create_model(image_size, style_dim).to(device)


def is_git_lfs_pointer(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")


def validate_weight_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weight not found: {path}. Ensure Git LFS is installed and run `git lfs pull`."
        )
    if is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Model weight is still a Git LFS pointer: {path}. Run `git lfs pull`."
        )
    return path


def load_state_dict(path, device):
    path = validate_weight_file(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state_dict checkpoint, got {type(checkpoint).__name__}: {path}")
    return checkpoint


def load_model_weights(model, path, device):
    state_dict = load_state_dict(path, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint is incompatible with the released LDE-MAR architecture "
            "(image_size=256, style_dim=512, feature_dim=40, motion_dim=40, pure_data=False)."
        ) from error


def load_model(path, device, image_size=256, style_dim=512):
    state_dict = load_state_dict(path, device)
    with torch.device("meta"):
        model = _create_model(image_size, style_dim)
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint is incompatible with the released LDE-MAR architecture "
            "(image_size=256, style_dim=512, feature_dim=40, motion_dim=40, pure_data=False)."
        ) from error
    return model
