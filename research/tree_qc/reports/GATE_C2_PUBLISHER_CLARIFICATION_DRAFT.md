# Publisher clarification draft (conditional — send only if the Supervisor judges it needed)

Prepared by the Gate C2 worker. Not sent. Use only if the independent Supervisor
decides that the residual ambiguity in §5 of
`GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2.md` blocks acceptance of the canonical
identity layer. Everything below is answerable from the archive except the three
questions marked **Q1–Q3**, which are the ones the archive cannot settle.

---

Subject: Clarification on tree identity fields in "TLS Forest Instance Segmentation Benchmark" (Zenodo 10.5281/zenodo.16875688)

Dear Dr Cherlet,

We are using the Litchfield and Wytham portions of your benchmark (Zenodo record
16875688, DOI 10.5281/zenodo.16875688) for a tree-quality-control study, and we want
to describe the data exactly as you intended rather than rely on our own inference.
We have verified both ZIP archives (MD5 and full CRC) and read the record description
and the pinned evaluation code. Three points in the release remain ambiguous for our
use case and we would be grateful for a short clarification.

**Q1 — Scope of the `instance` field.** The description states that `instance` is
"sequential, starting at 1, -1 for ground points". In the released files the three
area clouds of one site (`<site>_train.ply`, `<site>_val.ply`, `<site>_test.ply`)
each contain a subset of one site-wide numbering (for Wytham, IDs 1–876 spread over
the three files) rather than each restarting at 1. Is the intended reading that the
`instance` namespace is shared across the three area files of a site, so that a given
tree keeps the same ID in every area file in which it appears? Are the individual
tree PLY files guaranteed to use that same site-wide ID namespace?

**Q2 — Boundary duplicates.** The description says that "all trees overlapping with
each area are in the corresponding folders, so some edge trees are duplicated in
multiple folders". We observe 113 such duplicates in the two sites, and in every case
the duplicated files hold an identical, complete point set (not a per-area crop). Is
it correct that the individual tree files are always complete trees, and that the
folder only records which areas the tree overlaps?

**Q3 — Naming series and high-overlap pairs.** Wytham tree files use two numbering
series (`wytham_winter_<3-digit>` and `wytham_winter_<4-digit>`, e.g. 1610 and 8361).
A small number of pairs from different series share a large fraction of exactly equal
coordinates — up to 81 % of the smaller file — with centroids about 0.9 m apart, while
the rest of the population shares well under 10 %. Does each individual tree file
correspond to exactly one field-measured tree, and are the two numbering series
independent trees rather than re-segmentations of the same stems?

**Q4 — Individual tree ↔ plot mapping.** For our local-context construction we match
tree files to the labelled area clouds by exact coordinates. Almost every tree maps
onto one dominant `instance` ID, and the match is exact for the vast majority of
points; the residual points fall on a neighbouring instance because the raw scan
contains duplicated points that were assigned to two adjacent trees. Could you
confirm that this is the expected behaviour, i.e. that a small number of points may
carry two different `instance` labels at the same coordinates?

If any of this is already documented elsewhere, a pointer would be enough.

Thank you for releasing this dataset.

Kind regards,
[researcher]
[institution]

---

## What the worker already resolved without publisher input

- 1,693 tree file occurrences reduce to 1,579 exact-alias canonical trees
  (703 Litchfield / 876 Wytham), which equals the per-site tree counts reported in
  the associated publication; the 114 extra occurrences are the documented boundary
  duplicates and are exact coordinate aliases.
- The published evaluation-tree counts (128 Litchfield / 181 Wytham) equal the
  `test/in_plot_th0.90` file counts exactly.
- Instance IDs are site-global and the native IDs carried by `train/`/`val/` tree
  files agree with the plot instance their points map onto.

The three questions above are the residual items that the archive alone cannot close.
