import numpy as np
import pytest
import SimpleITK as sitk


@pytest.fixture
def dicom_series_factory():
    def write(folder, metadata=None):
        folder.mkdir(parents=True, exist_ok=True)
        array = np.stack(
            [
                np.arange(24 * 32, dtype=np.int16).reshape(24, 32) + index * 100
                for index in range(3)
            ]
        )
        volume = sitk.GetImageFromArray(array)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        series_uid = "1.2.826.0.1.3680043.2.1125.1"
        for index in range(volume.GetDepth()):
            image_slice = volume[:, :, index]
            image_slice.SetMetaData("0008|0060", "MR")
            image_slice.SetMetaData("0008|0018", f"{series_uid}.{index + 1}")
            image_slice.SetMetaData("0020|000e", series_uid)
            image_slice.SetMetaData("0020|0013", str(index + 1))
            image_slice.SetMetaData("0020|0032", f"0\\0\\{index}")
            image_slice.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
            for key, value in (metadata or {}).items():
                image_slice.SetMetaData(key, value)
            writer.SetFileName(str(folder / f"{index:03d}.dcm"))
            writer.Execute(image_slice)
        return folder

    return write
