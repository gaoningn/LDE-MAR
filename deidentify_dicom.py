import argparse
import shutil
import uuid
from pathlib import Path

import SimpleITK as sitk


SUPPORTED_PIXEL_TYPES = {sitk.sitkUInt16, sitk.sitkInt16}


def deterministic_uid(value):
    return f"2.25.{uuid.uuid5(uuid.NAMESPACE_URL, 'ldemar-public:' + value).int}"


def read_series(folder):
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder))
    if not files:
        raise ValueError(f"No DICOM series found in: {folder}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetPixelID() not in SUPPORTED_PIXEL_TYPES:
        raise TypeError(
            f"Unsupported DICOM pixel type in {folder}: {image.GetPixelIDTypeAsString()}"
        )
    return image


def write_sanitized_series(image, output_dir, identity):
    output_dir.mkdir(parents=True, exist_ok=True)
    study_uid = deterministic_uid(identity + ":study")
    series_uid = deterministic_uid(identity + ":series")
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for index in range(image.GetDepth()):
        image_slice = image[:, :, index]
        image_slice.SetOrigin((0.0, 0.0))
        image_slice.SetSpacing((1.0, 1.0))
        image_slice.SetDirection((1.0, 0.0, 0.0, 1.0))
        metadata = {
            "0008|0008": "DERIVED\\SECONDARY\\DEIDENTIFIED",
            "0008|0016": "1.2.840.10008.5.1.4.1.1.4",
            "0008|0018": deterministic_uid(f"{identity}:instance:{index}"),
            "0008|0060": "MR",
            "0012|0062": "YES",
            "0012|0063": "Minimal public research export",
            "0020|000d": study_uid,
            "0020|000e": series_uid,
            "0020|0013": str(index + 1),
            "0020|0032": f"0\\0\\{index}",
            "0020|0037": "1\\0\\0\\0\\1\\0",
        }
        for key, value in metadata.items():
            image_slice.SetMetaData(key, value)
        writer.SetFileName(str(output_dir / f"frame_{index:03d}.dcm"))
        writer.Execute(image_slice)


def deidentify_tree(source_root, output_root, overwrite=False):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output directory must not be inside the source directory")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)

    series_dirs = sorted(
        {
            path.parent
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".dcm"
        }
    )
    if not series_dirs:
        raise RuntimeError(f"No .dcm files found under: {source_root}")
    for index, folder in enumerate(series_dirs, start=1):
        relative = folder.relative_to(source_root)
        print(f"[{index}/{len(series_dirs)}] {relative}")
        image = read_series(folder)
        write_sanitized_series(image, output_root / relative, relative.as_posix())
    return len(series_dirs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a metadata-stripped DICOM tree for public model evaluation."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    count = deidentify_tree(args.source, args.output, args.overwrite)
    print(f"Created {count} sanitized DICOM series in: {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}")
