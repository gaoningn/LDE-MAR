import torch

from check_dicom_phi import scan_dicom_files
from dcmreader import read_dcm
from deidentify_dicom import deidentify_tree


def test_deidentified_dicom_passes(tmp_path, dicom_series_factory):
    series = dicom_series_factory(tmp_path / "safe", {"0010|0010": "ANONYMIZED"})
    files, findings = scan_dicom_files([series])
    assert len(files) == 3
    assert findings == []


def test_patient_name_is_rejected(tmp_path, dicom_series_factory):
    series = dicom_series_factory(tmp_path / "unsafe", {"0010|0010": "Example^Patient"})
    files, findings = scan_dicom_files([series])
    assert len(files) == 3
    assert any(tag == "0010|0010" for _, tag, _ in findings)


def test_deidentification_preserves_model_input(tmp_path, dicom_series_factory):
    raw_root = tmp_path / "raw"
    raw_series = dicom_series_factory(
        raw_root / "Original" / "SH001" / "group_001",
        {"0010|0010": "Example^Patient", "0010|0020": "PRIVATE-ID"},
    )
    for path in raw_series.glob("*.dcm"):
        path.rename(path.with_suffix(".DCM"))
    public_root = tmp_path / "public"
    assert deidentify_tree(raw_root, public_root) == 1
    public_series = public_root / "Original" / "SH001" / "group_001"
    files, findings = scan_dicom_files([public_root])
    assert len(files) == 3
    assert findings == []
    assert torch.equal(read_dcm(raw_series), read_dcm(public_series))
