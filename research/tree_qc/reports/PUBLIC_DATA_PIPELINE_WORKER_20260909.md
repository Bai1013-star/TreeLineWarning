# Public data pipeline worker evidence

Status: **PUBLIC_DATA_DOWNLOAD_NETWORK_BLOCKED**. Gate A/B supervisor approvals remain valid. No public archive bytes were received, no extraction or training occurred. This is not a model failure.

## Pinned identity

Zenodo record 16875688, exactly two files:

| File | Bytes | Official MD5 |
|---|---:|---|
| litchfield.zip | 2331915728 | 8e374d940d7e651484a84d7d63f18636 |
| wytham.zip | 3166118961 | 8d7918ed1270d603efc83aae5a8777c1 |

Primary URLs:
- https://zenodo.org/api/records/16875688/files/litchfield.zip/content
- https://zenodo.org/api/records/16875688/files/wytham.zip/content

Official alternate URLs:
- https://zenodo.org/records/16875688/files/litchfield.zip?download=1
- https://zenodo.org/records/16875688/files/wytham.zip?download=1

## Observed network attempts

Litchfield download routes and both sites' tiny explicit Range probes were attempted. Wytham full archive download remains **NOT_ATTEMPTED**, because neither of its tiny probes received a response.

1. Python urllib API endpoint: HTTP 504 Gateway Time-out, no archive bytes. A prior initial attempt and restarted attempt both observed this response.
2. Python urllib records endpoint: HTTP 504 Gateway Time-out, no archive bytes. Whole remaining range attempt interrupted after confirming both routes failed.
3. Native curl records endpoint, Range bytes=0-1023: curl exit 28, 30.005 seconds, HTTP 000, 0 bytes, 0 B/s.
4. Native curl API endpoint, Range bytes=0-65535: curl exit 28, 30.003 seconds, HTTP 000, 0 bytes, 0 B/s.
5. Native curl records endpoint, Range bytes=0-65535: curl exit 28, 30.002 seconds, HTTP 000, 0 bytes, 0 B/s.
6. Wytham native curl API endpoint, Range bytes=0-65535: curl exit 28, 25.010759 seconds, HTTP 000, 0 bytes, 0 B/s.
7. Wytham native curl records endpoint, Range bytes=0-65535: curl exit 28, 25.009188 seconds, HTTP 000, 0 bytes, 0 B/s.

HTTP 000 means no HTTP response arrived before curl timed out, not an HTTP server status. Curl header files are empty. Four-way segment downloading was not attempted because neither tiny range probe established a usable response. No further retries are running.

Raw logs: D:\TreeQCTraining\logs\public_data_download.log and public_data_download_retry.log. Tiny curl header/probe paths use zenodo_api_range, zenodo_records_range, and zenodo_records_64k prefixes under D:\TreeQCTraining\logs and tmp. The only archive-directory file is D:\TreeQCData\archives\litchfield.source.json (pinned provenance, not data); no .zip or .zip.part exists. Downloaded archive bytes: **0** for both sites.

## Implementation and limitations

public_data.py implements pinned provenance, streaming resumable .part transfer, alternate official routes, strict Content-Range/length validation before append, retained interrupted files, official MD5 plus local SHA256 on completion, and D-drive projections. Download approval reserves both archives, 40 GiB potential expansion, 10 GiB training, and at least 35 GiB final free space.

ZIP inventory rejects traversal, absolute/drive paths, Windows special names, symlinks/special files, duplicate paths, encryption, and expansion ratios over 1000. It records every central-directory size/CRC. Default inventory and every current CLI call read no member payloads, matching central-directory-only approval. The separate bounded PLY-header helper is dormant unless future approved code explicitly enables include_headers=True; it immediately rejects EOF before end_header. Extraction independently requires allow_extract=true and enforces the cumulative two-site expansion cap before extraction. Existing partial extraction files require manual review; extraction resume is intentionally not automatic. Full response Range support remains unverified due network failure; data-integrity checks and ZIP inventory cannot run without an archive.

No coordinate, schema, duplicate-tree or spatial-overlap findings can be claimed. These remain NOT_RUN, and Gate C remains pending. Six offline safety tests pass for absent approval, ignored/off-by-one Range rejection, traversal/Windows path rejection, expansion cap, truncated PLY header EOF, and central-directory-only inventory (ZipFile.open is mocked to raise on any attempted payload access). See public_data_safety_test.log. Synthetic tiny fixture artifacts remain on D:.

## Recovery

After official Zenodo access recovers, first repeat a tiny explicit range probe; only continue if status 206 and matching Content-Range arrive. Then run from the research worktree:

```cmd
DOWNLOAD_TREE_QC_PILOT_DATA.cmd download --approval research\tree_qc\reports\PUBLIC_DATA_DOWNLOAD_APPROVAL.json
```

The command preserves provenance and resumes a trustworthy existing .part. The current implementation retries alternating official endpoints using the remaining contiguous range; if only bounded ranges work, implement and test bounded segmented transfer before continuing. After completed archives pass MD5/SHA256, submit actual central-directory totals to the supervisor before payload/header access or extraction. No additional license/sites/weights are authorized.
