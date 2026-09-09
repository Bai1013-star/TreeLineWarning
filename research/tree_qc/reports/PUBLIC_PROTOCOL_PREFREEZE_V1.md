# Public pipeline pre-registration V1 (before public geometry access)

Supervisor supplement; original RESEARCH_PLAN_V1 and user targets remain controlling. This document specifies implementation requirements, not claims about unseen archives. Gate A is not repeated. Site direction frozen now: Litchfield training-site, Wytham held-out. Do not swap after learning results. No public geometry or local challenge is available to choose these rules.

## Identity, space and eligibility

Every sample lists canonical root identity, every context tree identity, source point IDs and spatial background blocks. The train/validation disjointness assertion covers all these source pools, not just candidate IDs. First establish recoverable common coordinates and metric units from real file schema and metadata; if individual trees are separately centered without transforms, stop true-context gate. Never infer scale solely from plausible numbers.

Use original IDs only within proven site/plot namespaces. Exact coordinate hashes plus bbox/centroid/count and spatial overlap identify duplicate candidates; resolve ambiguous matches conservatively as same group or quarantine. Set quantization from source coordinate precision, before validation results. Preserve detailed dedup evidence. Training-site spatial blocks need a neighborhood buffer and whole-tree assignment; do not randomly break a giant context-connected component. If an independent validation pool cannot be formed, report the limitation instead of mixing sites.

Held-out processing is allowed for schema, integrity, coordinate/identity consistency, and application of frozen eligibility/split rules only. No held-out example visualization, generator-yield-driven redesign, model score or label-conditioned inspection may inform training decisions. Local challenge images/point clouds remain unopened; retain prior historical-statistics exposure disclosure.

## Representation and sampling

Use translated local metric XYZ plus membership only, with no extra scalars initially. Origin and local-neighborhood lookup must be obtainable from current candidate membership and local context at deployment, not privileged correct GT centroid. Preserve metric scale; use identical transformations in both classes. No absolute world coordinates enter network.

For4096 and2048 samples, use equal candidate/context quotas, stratified inside each by lower/middle/upper height, XY coverage and membership boundary; final shuffle uses a common label-independent pipeline. Replacement/duplicate handling must be identical for every label. Do not create fake context from candidate points if context is absent; mark ineligible under a predeclared applicability rule. Report eligibility/duplicate ratios and exclusions. Retain natural irregular correct GT trees as GOOD.

A sample's neighborhood derives from actual spatial neighbors, never unrelated tree selection. Removed tree parts remain nonmember context when spatially available. Include original point provenance to verify that corruption changed membership rather than silently dropping all defect evidence during sampling.

## Label validity and generator realism

Every BAD stores exact added/removed point IDs, donor canonical IDs, contiguous geometric region, realized error fractions, variant and RNG seed. Identity/root split comes before corruption. Unchanged BAD is prohibited. Each core corruption needs at least two distinct contiguous variants and held-out seeds/severity combinations. Training-site visual and numerical audit must precede training; do not infer realism merely from classifier separability.

MERGED: actual spatial neighbor substantial connected structure. CONTAMINATION: coherent real neighboring boundary region. OVERSEG: a physically continuous target subtree/region. CROWN_INCOMPLETE: coherent upper/crown region, not independent random deletion. STEM_MISSING requires observable stem evidence in reference; lowest points alone are not proof of a stem. NOISE requires actual nonmember local evidence. Missing evidence is unavailable, not a fabricated class.

Operational visibility check to be implemented: realized membership change must be nonzero and substantial under defect-specific severity definitions frozen at Gate D; after sampling at least32 changed-membership evidence points must remain. Permit a fixed finite resampling budget determined before training, report failures by class, and never filter using model score. The32-point rule is an engineering representation requirement, not a validated forestry threshold. Severity fractions/ranges must be frozen from training-site geometry review before any held-out examples/scores, and may not be derived from local challenge statistics.

Audit fixed label-specific counts/order/replacement/RNG fingerprints. Equal quotas suppress candidate-fraction shortcuts. Physical geometric consequences of real corruption are not automatically leakage; implementation-created signals are the target of this audit.

## Independent support and continuation interpretation

Original overall moderate/high targets remain recall>=.85, bad false-pass<=.15, good false-review<=.25 and balanced accuracy>=.80. Supervisor operationalization of the previously qualitative core-class requirement: each MERGED/CONTAMINATION/OVERSEG must have at least30 unique root identities in validation and in held-out evaluation, recall>=.75 and balanced accuracy>=.75 against the shared GOOD pool at the same frozen threshold. This is a pilot minimum, not statistical power assurance. Report donor counts and spatial-group counts; multiple corruptions do not increase independent-tree counts. If support is insufficient, report INSUFFICIENT_INDEPENDENT_SUPPORT and withhold continuation success rather than hiding it in the aggregate. Uncertainty does not count as REVIEW recall.

PointNet and PointNet++ must use identical identity pools, corruption recipes, sampling and training/label budget. Freeze actual public optimizer/budget/early-stopping and validation-only model-selection rule before Gate E based on measured resource feasibility, not held-out scores. Freeze the entire seed/robustness evaluation package and checkpoint/threshold hashes before Gate F. Training seeds1729/2718/3141 retained from original contract. No automatic model enlargement or epoch extension after failure.
