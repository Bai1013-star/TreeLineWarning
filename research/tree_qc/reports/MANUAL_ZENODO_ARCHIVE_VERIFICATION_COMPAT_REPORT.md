# MANUAL_ZENODO_ARCHIVE_VERIFICATION_COMPAT_REPORT

Both manually downloaded official ZIPs passed actual exact size, official MD5, central directory and full decompression/CRC testing. No archive redownload or dependency installation was needed. Python3.11.9 stdlib does not implement actual ZIP method9 / Deflate64: PYTHON_STDLIB_LIMITATION, not archive corruption.

LITCHFIELD_BYTES = 2331915728
LITCHFIELD_MD5 = 8e374d940d7e651484a84d7d63f18636
LITCHFIELD_SHA256 = 4d45f0ad215ff49d665b0d881e15a1cfd5f85ae1e832c1da057461205b5156f3
LITCHFIELD_METHODS = 9 / Deflate64
LITCHFIELD_INTEGRITY = PASS

WYTHAM_BYTES = 3166118961
WYTHAM_MD5 = 8d7918ed1270d603efc83aae5a8777c1
WYTHAM_SHA256 = 2ffe674247fba13fe80fa154b72af013592e5b3be8328b7f592409d7bc6534a5
WYTHAM_METHODS = 9 / Deflate64
WYTHAM_INTEGRITY = PASS

PYTHON_STDLIB_SUPPORTED = NO
EXTERNAL_VERIFIER = D:\Program Files\WinRAR\WinRAR.exe
EXTERNAL_VERIFIER_VERSION = 7.13.0
VERIFIER_FIX_COMMIT = resolve this report introducing commit; exact value also in D:\TreeQCTraining\MANUAL_ARCHIVE_FINAL_GIT_STATE.json
GIT_CLEAN = recorded after commit in the external final state
MANUAL_ARCHIVE_VERIFICATION = PASS
GATE_C_ALLOWED = YES
FROZEN_V0_1_MODIFIED = NO
PUSH = NO

WinRAR test-only commands used t -cfg- -y -p- -ibck with inherited RAR options removed. Both returned0: Litchfield23.844seconds, Wytham25.219seconds. GUI stdout/stderr streams were empty and preserved as such; no fabricated per-entry output. Full commands, versions, exit codes and log prefixes are in MANUAL_ZENODO_ARCHIVE_VERIFICATION_COMPAT_EVIDENCE.json. Source hash was recorded before final execution. Fifteen regression tests passed including deflate, unsupported dispatch, external success/failure, no available tool, wrong size, wrong MD5 and inherited configuration exclusion. Independent reviewer reran all15 and closed the configuration issue.

Central directory totals:1699files; compressed member bytes5497779949; ZIP archive bytes5498034689; expanded14414401594bytes. All paths passed Windows-safe path/type/duplicate/encryption checks. Current free space already accounts for ZIPs and retainedparts. Deducting expansion and10GiB trainingreserve leaves about95.79GiB, greater than35GiB guard. Supervisor permits controlled extraction only after this acceptance; the standalone verifier itself still never extracts. Gate C has started and is not yet PASS. No Gate A/B rerun or science/model/product change.

Method9 mapping primary source: https://github.com/ip7z/7zip/blob/main/DOC/Methods.txt . Actual methods were obtained from every ZipInfo.compress_type, not inferred from names.
