import argparse
import re
from pathlib import Path

import SimpleITK as sitk


SENSITIVE_TAGS = {
    "0008|0080": "Institution Name",
    "0008|0081": "Institution Address",
    "0008|0090": "Referring Physician Name",
    "0008|1050": "Performing Physician Name",
    "0008|1070": "Operators' Name",
    "0010|0010": "Patient Name",
    "0010|0020": "Patient ID",
    "0010|0030": "Patient Birth Date",
    "0010|0032": "Patient Birth Time",
    "0010|0040": "Patient Sex",
    "0010|1000": "Other Patient IDs",
    "0010|1001": "Other Patient Names",
    "0010|1010": "Patient Age",
    "0010|1040": "Patient Address",
    "0010|2154": "Patient Telephone Numbers",
    "0032|1032": "Requesting Physician",
}
ALLOWED_REDACTED_VALUES = {
    "",
    "ANON",
    "ANONYMOUS",
    "ANONYMIZED",
    "DEIDENTIFIED",
    "DE-IDENTIFIED",
    "REMOVED",
}
TAG_PATTERN = re.compile(r"^([0-9a-fA-F]{4})\|([0-9a-fA-F]{4})$")


def dicom_files(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".dcm" else []
    if not path.is_dir():
        raise FileNotFoundError(f"DICOM path not found: {path}")
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() == ".dcm")


def scan_dicom_files(paths):
    files = []
    for path in paths:
        files.extend(dicom_files(path))
    files = sorted(set(files))
    if not files:
        raise RuntimeError("No .dcm files found in the requested paths")

    findings = []
    for file in files:
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(str(file))
        try:
            reader.ReadImageInformation()
        except RuntimeError as error:
            findings.append((file, "READ_ERROR", str(error).splitlines()[0]))
            continue

        for key in reader.GetMetaDataKeys():
            normalized_key = key.lower()
            value = reader.GetMetaData(key).strip()
            if normalized_key in SENSITIVE_TAGS:
                if value.upper() not in ALLOWED_REDACTED_VALUES:
                    findings.append((file, normalized_key, f"{SENSITIVE_TAGS[normalized_key]}={value!r}"))
                continue

            match = TAG_PATTERN.match(normalized_key)
            if match and int(match.group(1), 16) % 2 == 1:
                findings.append((file, normalized_key, "Private DICOM tag is present"))
    return files, findings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fail when common PHI fields or private DICOM tags are present."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    files, findings = scan_dicom_files(args.paths)
    if findings:
        print(f"PHI check failed: {len(findings)} finding(s) in {len(files)} DICOM file(s).")
        for file, tag, detail in findings:
            print(f"{file}\t{tag}\t{detail}")
        raise SystemExit(1)
    print(f"PHI check passed: {len(files)} DICOM file(s), no blocked metadata found.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}")
