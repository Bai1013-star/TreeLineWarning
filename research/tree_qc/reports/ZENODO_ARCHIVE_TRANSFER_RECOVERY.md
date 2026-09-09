# Official Zenodo archive recovery, 2026-09-09

This recovery continues frozen Gate A/B PASS from HEAD `0b67e3d`. It does not recreate environments, modify the product, extract archives, or authorize training.

Current official record and file-list JSON were retrieved afresh through TLS-verified `curl.exe`, following record `links.files`. The two allowed identities still match the pinned manifest:

| Filename | Expected bytes | Official MD5 | Current content URL |
|---|---:|---|---|
| litchfield.zip | 2331915728 | 8e374d940d7e651484a84d7d63f18636 | https://zenodo.org/api/records/16875688/files/litchfield.zip/content |
| wytham.zip | 3166118961 | 8d7918ed1270d603efc83aae5a8777c1 | https://zenodo.org/api/records/16875688/files/wytham.zip/content |

The file metadata links equal the corresponding content URL with `/content` removed. Fresh metadata snapshots are `zenodo_archive_recovery_current_metadata.json` and `zenodo_archive_recovery_current_files.json`. Record GET: HTTP 200, 13895 bytes, 2.733647 seconds. File-list GET: HTTP 200, 8904 bytes, 2.316588 seconds.

Initial Litchfield HEAD returned HTTP 200 and Content-Length 2331915728. Explicit range `bytes=0-1048575` returned HTTP 206, Content-Range `bytes 0-1048575/2331915728`, Accept-Ranges `bytes`, and exactly 1048576 bytes in 4.505648 seconds (232724 bytes/s). This valid prefix was preserved at `D:\TreeQCData\downloads\litchfield.zip.part` with a matching source sidecar.

An open-ended `curl.exe --location --continue-at - --retry 2 --retry-all-errors` request successfully resumed at byte 1048576 and returned the correct HTTP 206 range through byte 2331915727. Single-stream throughput was approximately 0.65 MB/s. The worker terminated only its own Python/curl process IDs (99352, 49128, 100176) before switching methods, preserving a contiguous 62451712-byte prefix.

Four simultaneous 16 MiB explicit ranges then succeeded, with exact Content-Range checks and ordered durable merges. The first batch achieved 2661149 bytes/s. Segments have three bounded attempts with exponential 2/4/8-second backoff, 15-second connect timeout and 120-second transfer timeout. A later segment returned HTTP 504 twice with zero bytes, then succeeded on its third attempt; prior successful pieces were retained. At that stage, BITS was not started because a functioning official resumable transfer had been established.

Raw evidence and progress: `D:\TreeQCTraining\logs\zenodo_recovery_20260909\`. Exact per-segment headers and curl logs are retained under `D:\TreeQCData\downloads\litchfield.zip.ranges\`. Downloaded bytes remain provisional until expected total length, publisher MD5, local SHA-256, ZIP central directory and all ZIP CRC checks pass. The worker processes Litchfield first and starts Wytham only after Litchfield passes these checks.

The recovery tool never extracts. After both archives pass it emits combined expanded bytes, compressed bytes, file counts and current D: disk budget for Supervisor review against the 35 GiB final-free guard. The separate double-click `VERIFY_MANUAL_ZENODO_ARCHIVES.cmd` performs only offline validation using the existing Python environment and the pinned fresh metadata snapshots.

Final status: **ZENODO_ARCHIVE_TRANSFER_BLOCKED**. No archive was promoted to a verified `.zip`; Gate A/B remain PASS, and no extraction or training began.

The four-range transfer progressed to a 1203302400-byte continuous prefix. The next batch's first three 16 MiB pieces succeeded, but range `1253634048-1270411263` returned HTTP 504 with 0 bytes on all three attempts (32.196894, 31.310641 and 32.103284 seconds). A final reduced 1 MiB probe `1253634048-1254682623` also returned HTTP 504, 0 bytes, in 32.237611 seconds. No retry budget was reset afterward.

A separate owned BITS probe against the same current official content link returned `TransientError`, HTTP 504, 0 bytes after 35.1565 seconds. Job `4e97ba62-2baa-4c41-a988-5704a35beee1` was removed. Subsequent read-only checks found zero instances of that job and no recovery Python/curl processes.

The three successful pieces were explicitly audited against the current-run source sidecar, successful transfer log, final HTTP 206 and exact Content-Range headers; local SHA-256 values were recorded in `legacy_piece_provenance_migration.json`. They were appended with flush/fsync to preserve all useful data. The original pieces remain as evidence. Final recoverable continuous prefix:

- Path: `D:\TreeQCData\downloads\litchfield.zip.part`
- Bytes: **1253634048** of 2331915728; next resume offset **1253634048**.
- Partial SHA-256: `ac9eea2c41f38d80645d73a06a1ba0ff92a4c1598f2483e7ee257a53ecfbe766`.
- Range capability: confirmed HTTP 206 / `Accept-Ranges: bytes` on successful Litchfield requests. Current gateway failure does not revoke that observed capability.
- Wytham: 0 bytes, no `.part`; its Range capability was not independently tested because Litchfield has not completed.
- D: free at stop: 133504053248 bytes. Expanded size remains unknown, and `allow_extract = false`.

These partial hashes are recovery evidence, not a publisher-integrity PASS. The complete publisher MD5 and ZIP CRC remain unverified.

The revised resume utility adds explicit current-metadata refresh for online actions, per-piece and nonce-payload provenance, final-response-only Range validation, fsync before consuming intermediate payloads, a mandatory complete Litchfield verification before Wytham, and successful completion handling on the final allowed direct-transfer attempt. Seven offline adversarial boundary tests pass. Existing old payloads without enough provenance are retained and never silently consumed. The executing Litchfield version was reconstructed from this session's patches and saved as `litchfield_executed_version_reconstructed.py`, SHA-256 `4ddb638170de265cde4bcd3bbb1b77cd4f6706e119213e411196294c9230f40f`; this is explicitly a reconstruction, not a pre-execution snapshot. The revised version was not used to restart downloads after the bounded failure.

Root-level `DOWNLOAD_TREE_QC_PILOT_DATA.cmd` now uses the revised resume utility, 4 MiB ranges and `downloads` destination. Do not launch it again automatically in this stopped session. The authorized fallback is a normal browser download of these same two files from https://zenodo.org/records/16875688 to `D:\TreeQCData\downloads\`, preserving the existing `.part`. Double-click root-level `VERIFY_MANUAL_ZENODO_ARCHIVES.cmd` after both complete ZIPs arrive; it makes no network requests and does not extract. Manual download from the same official record preserves scientific provenance.

Final independent read-only review: PASS_WITH_LIMITATIONS. The direct probe now shares final-response Range validation; both root launchers were inspected. Offline verification makes no network request and does not extract. The reviewer independently confirmed the final partial length and accurate blocked/incomplete status; prior seven offline tests passed. Whole-archive integrity remains NOT_VERIFIED.
