Place CALCE dynamic driving-profile Excel files here.

Do not put low-current OCV-only files here. OCV files belong in
`Data/raw_reference/` and are not sufficient to build the profile-holdout
training dataset.

Expected profile tokens:
- BJDST
- DST
- US06
- FUDS

Expected temperature tokens:
- 0C
- 25C
- 45C

Expected start-SOC token:
- 80SOC for the manuscript dynamic-profile set

Required manuscript dynamic-profile file patterns:

```text
*0C_BJDST_80SOC.xls*
*0C_DST_80SOC.xls*
*0C_US06_80SOC.xls*
*0C_FUDS_80SOC.xls*
*25C_BJDST_80SOC.xls*
*25C_DST_80SOC.xls*
*25C_US06_80SOC.xls*
*25C_FUDS_80SOC.xls*
*45C_BJDST_80SOC.xls*
*45C_DST_80SOC.xls*
*45C_US06_80SOC.xls*
*45C_FUDS_80SOC.xls*
```

Date and cell prefixes are allowed. Example:

```text
02_24_2016_SP20-2_0C_DST_80SOC.xls
```

`Data/prepare_calce_nmc.py` requires the full 12-file 80SOC set by default.
Use `--allow-incomplete` only for a partial conversion test.
