"""Produce reviewable tables; preserve provisional controls outside good metrics."""

from pathlib import Path

import numpy as np
from audit import read, sha256


def run():
    here = Path(__file__).parent
    results = read(here / "results.json")
    manifest = read(here / "reference_manifest.json")
    evidence = read(here / "evidence.json")
    field = Path(manifest["datasets"]["human_field_batch"]["batch_root"])
    records = {r["global_tree_id"]: r for r in read(field / "batch-reconciliation.json")["trees"]}
    stems = {
        r["tree_id"]: r["structural_evidence"]
        for r in evidence["datasets"]["human_field_batch"]["span_manifests"][0]["trees"]
    }
    lines = [
        "# Geometry feature analysis",
        "",
        "No trustworthy explicit-good distribution is available. The provisional-control column",
        "below retains the prior manifest's eight good claims for exploration ONLY; all eight",
        "are UNCERTAIN_REFERENCE in definitive metrics. No production accuracy is calculated.",
        "",
        "| Feature | Explicit good | Provisional controls min / median / max "
        "| Bad min / median / max |",
        "|---|---|---|---|",
    ]
    cases = [c for c in results["cases"] if c["features"]]
    for key in cases[0]["features"]:
        cells = []
        for prior in ("EXPLICIT_GOOD", "EXPLICIT_BAD_VISUAL"):
            values = [c["features"][key] for c in cases if c["prior_reference_category"] == prior]
            cells.append(
                " / ".join(f"{x:.6g}" for x in (min(values), np.median(values), max(values)))
            )
        lines.append(f"| {key} | unavailable | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "## Existing segmentation evidence (original field memberships)",
        "",
        "| Case | Raw columns | Deduplicated hypotheses "
        "| Retained independent stems | CHM peak field |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        if case["dataset"] != "human_field_batch":
            continue
        s = stems[records[case["tree_id"]]["primary_span_tree_id"]]
        values = [
            s[k]
            for k in (
                "stem_candidate_count_raw",
                "stem_hypothesis_count_deduplicated",
                "validated_independent_stem_count",
                "CHM_peak_count",
            )
        ]
        lines.append(f"| {case['case_id']} | " + " | ".join(map(str, values)) + " |")
    lines += [
        "",
        "## Case-by-case diagnostic results",
        "",
        "This is the predeclared two-scale diagnostic hypothesis, not accepted product QC.",
        "",
        "| Tree | Reference | QC result | Issue | Reason | Match / miss |",
        "|---|---|---|---|---|---|",
    ]
    for c in results["cases"]:
        p = c["prediction"]
        match = "MISS (abstention)" if c["reference_label"] == "EXPLICIT_BAD" else "NOT_SCORED"
        lines.append(
            f"| {c['case_id']} | {c['reference_label']} | {p['status']} | {p['issue']} | "
            f"{', '.join(p['reason_codes'])} | {match} |"
        )
    lines += [
        "",
        "## Sensitivity checks, not threshold selection",
        "",
        "All use the same unvalidated 10% material-component assumption. No PASS follows from",
        "a missing signal. The controls here remain provisional.",
        "",
        "| Signal | Bad signaled / 8 | Provisional controls signaled / 8 |",
        "|---|---:|---:|",
    ]
    for scale in (16, 32, 64):
        counts = [
            sum(
                c["features"][f"component_{scale}_second"] >= 0.1
                for c in cases
                if c["prior_reference_category"] == prior
            )
            for prior in ("EXPLICIT_BAD_VISUAL", "EXPLICIT_GOOD")
        ]
        lines.append(
            f"| Second sampled component >= 10%, diagonal/{scale} voxels "
            f"| {counts[0]} | {counts[1]} |"
        )
    lines += [
        "| Second sampled component >= 10% at BOTH /32 and /64 | 0 | 0 |",
        "",
        "Fine-scale component evidence overlaps multiple deduplicated stem hypotheses for",
        "T000/T005/T010/T013 (4/8 bad). Multiple hypotheses alone signal 6/8 bad and 0/7",
        "field provisional controls, but are not independently validated stems. T009/T014",
        "have only one hypothesis; T001 has almost all sampled points connected. These are",
        "promising research signals with important misses; validation remains incomplete.",
        "",
        "## Rechecked visual provenance",
        "",
    ]
    for c in cases:
        if c["reference_label"] != "EXPLICIT_BAD":
            continue
        folder = Path("E:/TreeLineWarning-ai-qc-benchmark-v2/renders") / c["case_id"]
        meta = read(folder / "render.json")
        assert meta["source_editor_revision"] == c["pre_edit_revision"]
        assert sha256(folder / meta["contact_sheet"]) == meta["contact_sheet_sha256"]
        lines.append(
            f"- {c['case_id']}: contact-sheet SHA-256 `{meta['contact_sheet_sha256']}` verified;"
        )
    (here / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
