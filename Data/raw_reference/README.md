# Raw Reference Data

Raw CALCE reference files are not redistributed in this repository.

Place low-current OCV reference files here. The initial-capacity file may be
kept here for provenance, but the manuscript SOC label uses temperature-specific
low-current OCV capacity, not a fixed 2 Ah capacity.

Expected reference-file patterns:

```text
*Initial*capacity*.xls*
*lowcurrentOCV*.xls*
*low current OCV*.xlsx
```

Low-current OCV files are used to build the temperature-specific
`SOC0_OCV_inferred` and `Q_ref_lc_ocv_Ah` references. They are not dynamic
driving-profile records and should not be placed in `Data/raw_dynamic/`.

After placing the dynamic files in `Data/raw_dynamic/` and the reference files
here, run:

```bash
python Data/prepare_calce_nmc.py
```
