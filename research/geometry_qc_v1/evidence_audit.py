"""Audit neutral reference evidence and persisted geometry without product mutation."""

import base64
import json
import zlib
from pathlib import Path

import numpy as np
from audit import membership_digest, read, sha256


def run():
    here = Path(__file__).parent
    manifest = read(here / "reference_manifest.json")
    result = {"datasets": {}, "cases": []}
    for name, dataset in manifest["datasets"].items():
        root = Path(dataset["batch_root"])
        journal = read(root / "tree_edits.json")
        info = {
            "operations": [
                {k: v for k, v in op.items() if k != "removed_indices_b64"}
                for op in journal["operations"]
            ],
            "source_sha256_in_journal": journal.get("source_sha256"),
            "span_manifests": [],
            "dbh_versions": [],
        }
        result["datasets"][name] = info
        for path in sorted(root.glob("span*/tree/span_tree_manifest.json")):
            value = read(path)
            info["span_manifests"].append(
                {"path": str(path), "sha256": sha256(path), "trees": value["trees"]}
            )
        # All historical versions are inventoried; none are imported into pre-edit features.
        for path in sorted(root.glob("dbh_v1/*/finalized_geometry.json")):
            value = read(path)
            info["dbh_versions"].append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "recovery_version": value.get("recovery_version"),
                    "binding": value["binding"],
                    "trees": [
                        {
                            k: row.get(k)
                            for k in (
                                "global_tree_id",
                                "persistent_tree_id",
                                "dbh_method",
                                "dbh_fit_status",
                                "stem_candidate_count",
                                "valid_dbh_roi_count",
                                "stem_association_status",
                            )
                        }
                        for row in value["trees"]
                    ],
                }
            )
        if not (root / "tree_segmentation_original.npz").is_file():
            info["membership_verification"] = "UNAVAILABLE"
            continue
        loaded = {}
        for source in ("original", "final"):
            path = root / f"tree_segmentation_{source}.npz"
            assert sha256(path) == dataset[f"{source}_artifact_sha256"]
            with np.load(path, allow_pickle=False) as arrays:
                loaded[source] = (arrays["original_index"], arrays["tree_id"])
        oi, ot = loaded["original"]
        fi, ft = loaded["final"]
        removed = np.concatenate(
            [
                np.cumsum(
                    np.frombuffer(
                        zlib.decompress(base64.b64decode(op["removed_indices_b64"])), dtype="<i8"
                    ),
                    dtype=np.int64,
                )
                for op in journal["operations"]
                if op["type"] == "REMOVE_POINTS"
            ]
            or [np.empty(0, dtype="<i8")]
        )
        for case in (c for c in manifest["cases"] if c["dataset"] == name):
            tree = case["tree_id"]
            before, after = oi[ot == tree], fi[ft == tree]
            assert membership_digest(before) == case["pre_edit_revision"]
            if len(after):
                assert membership_digest(after) == case["post_edit_revision"]
            else:
                assert case["post_edit_revision"] == "DELETED_BY_HUMAN_EDIT_SEQUENCE_1"
            expected = before[~np.isin(before, removed)]
            assert np.array_equal(np.sort(expected), np.sort(after)), "EDITOR_REPLAY_MISMATCH"
            result["cases"].append(
                {
                    "case_id": case["case_id"],
                    "before_count": len(before),
                    "after_count": len(after),
                    "removed_count": len(before) - len(after),
                    "pre_membership_sha256": membership_digest(before),
                    "post_membership_sha256": membership_digest(after),
                    "exact_index_replay": True,
                }
            )
        info["membership_verification"] = "PRE_POST_HASH_AND_EXACT_INDEX_REPLAY_PASS"
    (here / "evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
