# Dataset Metadata Recovery — 2026-09-09

Current status: OFFICIAL_METADATA_RECOVERED; earlier bounded network failures below are historical. See the successful recovery section at the end. Gate B approval remains with the supervisor. This narrow follow-up supplements DATASET_AUDIT.md and DATA_LICENSE_AUDIT.md; it does not replace their completed review and does not approve Gate A or Gate B.

No public dataset archives, images, pretrained weights, or archive Range bytes were downloaded. No product or training code changed.

## Bounded recovery attempts

| Official metadata endpoint | Tool/result | Received bytes |
|---|---|---:|
| https://zenodo.org/api/records/16875688 | curl.exe -L --connect-timeout 8 --max-time 20 --fail; exit 28; HTTP 000; timeout 20.009 seconds | 0 |
| https://zenodo.org/records/16875688/export/json | curl.exe -L --connect-timeout 8 --max-time 20 --fail; exit 28; HTTP 000; timeout 20.004 seconds | 0 |
| https://zenodo.org/records/16875688 | web open: timeout fetching | no successful page |

A focused web search for this record's bytes/license did not yield the record JSON. No successful JSON snapshot was obtained. TLS verification was not disabled and no third-party mirror was used.

## Evidence still missing

- Exact metadata license object and consistency with the prior official page's CC BY 4.0 statement.
- Version and latest-version relationship; record 16875688 remains the fixed candidate, not a proven latest version.
- Per-file exact bytes and links from official record JSON. The URLs in DATASET_AUDIT.md remain candidate routes, not newly verified links.
- Uncompressed archive size or official expanded manifest. No official metadata giving expanded bytes was retrieved. The prior 2–4x expansion example is a planning scenario, not an empirical bound.

The compressed display sizes and publisher MD5 values already recorded are unchanged and were not independently reconfirmed in this recovery attempt.

## Supervisor decision support

This recovery is insufficient to close the missing metadata/capacity conditions for Gate B. It is a metadata network obstruction, not evidence of model failure. Gate A must independently PASS before any public data download. The supervisor retains Gate B authority.

Recovery: retry one of the two exact official JSON endpoints above with the same bounded timeout after network recovery; save the successful JSON and inspect id, DOI, license, version/latest links, and file key/size/checksum/link fields. Resolve version links using their actual returned values. Before extraction, obtain an official expanded manifest if available; otherwise, only after the supervisor permits dataset bytes, inspect the ZIP central directory, enforce an explicit extraction disk budget, and account for caches/checkpoints. Do not infer an exact expansion budget from rounded compressed sizes or assume free space alone proves capacity.

## Final process-only network diagnostic

No system proxy, DNS, routing, or certificate settings were changed.

- System DNS query returned zenodo.org A = 198.18.0.70 (fake-IP range).
- curl.exe --noproxy '*' against https://zenodo.org/api/records/16875688: timeout after 15.003 s, HTTP 000, 0 bytes.
- HTTPS Google DNS query https://dns.google/resolve?name=zenodo.org&type=A succeeded and returned 188.184.103.118, 188.184.98.114, 137.138.153.219, 188.185.43.153, 188.185.48.75, 137.138.52.235 (TTL 13 seconds at lookup).
- A single immediate curl.exe --noproxy '*' --resolve zenodo.org:443:188.184.103.118 -L --connect-timeout 8 --max-time 20 --fail against the same official metadata endpoint timed out after 20.012 s, HTTP 000, 0 bytes. HTTPS URL and TLS hostname validation remained zenodo.org; certificate verification was not disabled. No dataset file URL was requested.

This diagnostic did not restore metadata access. Other returned IPs were not iterated. ZENODO_METADATA_NETWORK_BLOCKED remains the bounded observed result; the fake-IP answer alone is not proven to be the root cause because the process-level resolved-IP attempt also failed.


## Successful recovery — supersedes network-blocked status

On 2026-09-09 the supervisor obtained the official record JSON in zenodo_16875688_verified_metadata.json. Independent metadata review confirms id 16875688, conceptrecid 14615492, DOI 10.5281/zenodo.16875688, metadata.license.id = cc-by-4.0, access_right = open, and version relation index 2 / is_last true. The record's description repeats its CC BY 4.0 statement and paper citation requirement.

This reviewer followed the actual links.latest value, https://zenodo.org/api/records/16875688/versions/latest, using HTTPS with TLS validation enabled. curl followed its redirect and returned HTTP 200, 13,895 bytes, effective URL https://zenodo.org/api/records/16875688 in about 3.4 seconds. Saved response: zenodo_16875688_latest_verified_metadata.json. Its parsed id is 16875688. Thus the official latest-version chain confirms the pinned record is latest at this check; this does not imply it remains latest indefinitely.

| Pilot archive | Exact compressed bytes | Publisher checksum | Official API content URL |
|---|---:|---|---|
| litchfield.zip | 2331915728 | md5:8e374d940d7e651484a84d7d63f18636 | https://zenodo.org/api/records/16875688/files/litchfield.zip/content |
| wytham.zip | 3166118961 | md5:8d7918ed1270d603efc83aae5a8777c1 | https://zenodo.org/api/records/16875688/files/wytham.zip/content |

Total compressed pilot archive bytes: 5,498,034,689 (about 5.120 GiB). All values above come from files[].size/checksum/links.self, not constructed URLs. These remain publisher checksums, not locally computed data-file hashes. Metadata SHA256 is recorded in pinned_pilot_manifest.json; only these two archives are selected.

The recovered JSON does not supply expanded bytes, ZIP central-directory entries, per-split unique tree counts, or coordinate/overlap validation. Those remaining evidence conditions cannot be fabricated from compressed size. Supervisor can now decide the Dataset/License download Gate using verified identity, latest chain, exact compressed bytes and license plus an explicit disk-budget/extraction policy. This reviewer does not approve that Gate. No archive, image, weight, or archive Range data was downloaded by this reviewer.
