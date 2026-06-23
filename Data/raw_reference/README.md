# Raw Reference Data

Raw CALCE reference files are not redistributed in this repository.

Place initial-capacity and low-current OCV reference files here.

Expected reference-file patterns:

```text
*Initial*capacity*.xls*
*lowcurrentOCV*.xls*
*low current OCV*.xlsx
```

The initial-capacity file is used to infer capacity when `--capacity-ah` is not
provided. Low-current OCV files are reference/provenance files only. They are
not dynamic driving-profile records and should not be placed in
`Data/raw_dynamic/`.

After placing the dynamic files in `Data/raw_dynamic/` and the reference files
here, run:

```bash
python Data/prepare_calce_nmc.py
```
