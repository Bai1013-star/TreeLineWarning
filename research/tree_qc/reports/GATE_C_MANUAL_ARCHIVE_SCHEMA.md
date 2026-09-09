# Gate C manual archive schema checkpoint

Manual archive verification PASS at verifier fix771144a. Supervisor reviewed two publisher MD5 matches, full isolated WinRAR CRC exit0, safe central directories and combined disk budget before authorizing extraction. Standalone verifier remains test-only; extraction was a separate authorized step.

Controlled extraction ran from D:/TreeQCTraining/logs/gate_c_manual_20260909/extract_schema.py into fresh D:/TreeQCData/extracted_manual_verified/{site}. All1699 output paths, exact byte lengths and full per-file CRC passed. Original ZIPs and litchfield.zip.part remain intact. Final free113582182400bytes; reserve35GiB plus10GiB training remains satisfied. Evidence source/hash, bound inventories, commands and markers reside in the external logs directory.

Independent read-only reviewer validated every saved PLY header against archive entry size: header_bytes + vertex_count * actual ordered dtype stride. All1699 are binary little-endian vertex-only. Litchfield record sizes12/32/56bytes, Wytham12/20bytes. Some headers lack semantic/instance; some include RGB. Parse actual fields, and exclude RGB from network input. No explicit stem field exists. Directory tree-file occurrences and basename counts are not verified tree identities.

Coordinate provenance: the official pinned evaluator documents overlap at1cm and rounds unscaled XYZ to2decimals, supporting metres as an explicit inference from authors implementation; GT trees are cropped directly against plotbbox. This supports native shared local metric coordinates, not proven EPSG/global georeferencing. Actual tree/plot agreement remains to be checked. Sources: https://github.com/qforestlab/TreeInstSegEval/blob/6210682210d2783e6a5e58e4a1d63d2632e20748/base_evaluation.py#L533-L548 and #L80-L108; README.md#L45. No official implementation copied.

GATE_C = IN_PROGRESS. Dedup, disjoint spatial split/context and coordinate consistency are not yet accepted. No generator/training/held-out scoring/local challenge. Gate A/B PASS unchanged. PRODUCT_INTEGRATION=NO; FROZEN_V0_1_MODIFIED=NO; SCIENCE_MODIFIED=NO; PUSH=NO.
