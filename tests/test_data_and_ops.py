from pathlib import Path

import torch

from dcmreader import collect_sequence_pairs, read_dcm
from inference_utils import read_paired_sequences
from networks.op import FusedLeakyReLU, upfirdn2d


def test_dataset_path_pairing(tmp_path):
    directories = [
        "Original/SH001/group_001",
        "Original/SH001/group_010",
        "Artificial/SH001/cardiac_arr_010",
        "Artificial/SH001/cardiac_arr_001",
        "Original/SH002/group_002",
        "Artificial/SH002/cardiac_arr_003",
    ]
    for directory in directories:
        (tmp_path / directory).mkdir(parents=True)

    pairs = collect_sequence_pairs(tmp_path)
    assert [(subject, clean.name, artifact.name) for subject, clean, artifact in pairs] == [
        ("SH001", "group_001", "cardiac_arr_001"),
        ("SH001", "group_010", "cardiac_arr_010"),
    ]


def test_read_dcm_preprocessing(tmp_path, dicom_series_factory):
    series = dicom_series_factory(tmp_path / "series")
    output = read_dcm(series, target_size=(64, 64))
    assert output.shape == (3, 1, 64, 64)
    assert output.dtype == torch.float32
    assert output.min() >= -1 and output.max() <= 1


def test_pair_frame_mismatch_is_rejected(tmp_path, dicom_series_factory, monkeypatch):
    clean = dicom_series_factory(tmp_path / "clean")
    artifact = dicom_series_factory(tmp_path / "artifact")
    original_read = read_dcm

    def mismatched_read(path, target_size=(256, 256)):
        result = original_read(path, target_size)
        return result[:2] if Path(path) == artifact else result

    monkeypatch.setattr("inference_utils.read_dcm", mismatched_read)
    try:
        read_paired_sequences(clean, artifact)
    except ValueError as error:
        assert "different frame counts" in str(error)
    else:
        raise AssertionError("Expected mismatched frame counts to fail")


def test_pure_pytorch_operators_support_gradients():
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    kernel_1d = torch.tensor([1.0, 3.0, 3.0, 1.0])
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]) / 64
    output = upfirdn2d(input_tensor, kernel, up=2, pad=(1, 1))
    assert output.shape == (2, 3, 15, 15)
    output.sum().backward()
    assert torch.isfinite(input_tensor.grad).all()

    activation = FusedLeakyReLU(3, negative_slope=0.1)
    value = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])
    expected = torch.tensor([[[[-0.1]], [[0.0]], [[1.0]]]]) * (2 ** 0.5)
    assert torch.allclose(activation(value), expected)
