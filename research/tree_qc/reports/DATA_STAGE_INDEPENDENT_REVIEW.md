# Independent data-stage review and final recheck

Reviewer: gate_a_runtime_reviewer; read-only, did not implement data pipeline.
Final verdict: PASS_WITH_LIMITATIONS. Gate A/B PASS; PUBLIC_DATA_DOWNLOAD_NETWORK_BLOCKED; Gates C–I NOT_RUN.

Verified official metadata/manifest hash, pinned record/latest/license and both file sizes/MD5/URLs; observed source metadata only in D:\TreeQCData, no archive/part/extraction. Confirmed allow_extract=false, cumulative two-archive disk/expanded cap, range/provenance checks and official checksum validation logic. Initial four offline tests passed independently.

Found two issues before any archive access: malformed PLY header EOF could loop indefinitely; inventory read payload headers despite central-directory-only scope. Worker fixed immediate EOF error and inventory default/CLI include_headers=false. Final independent rerun passed all six offline tests, including EOF and a mock that forbids archive member payload access. Both findings closed.

Final network evidence includes Litchfield HTTP504 and tiny Range timeouts, plus Wytham two official64KiB attempts ending25-second timeout/0 bytes. Do not conflate tiny probes with completed archive transfer. All download/probe processes stopped. Real full Range transfer, archive checksum and science/coordinates/dedup remain unverified until actual bytes arrive. No model-failure conclusion is supported.
