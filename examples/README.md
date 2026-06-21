# Test data

`public_test_data/` contains the public, metadata-stripped DICOM copy used by
the default inference and evaluation commands:

```text
examples/public_test_data/
|-- Original/SHtest/group_001..005/*.dcm
`-- Artificial/SHtest/cardiac_arr_001..005/*.dcm
```

Keep source DICOM files outside the Git repository. The public copy can be
generated and verified with:

```bash
python deidentify_dicom.py /path/to/private_test_data examples/public_test_data --overwrite
python check_dicom_phi.py examples/public_test_data
```
