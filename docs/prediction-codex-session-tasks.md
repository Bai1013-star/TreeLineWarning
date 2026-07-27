\# TreeLineWarning Prediction Codex Session Tasks



\## 1. Purpose



This document defines the Codex session breakdown for the TreeLineWarning vegetation-growth prediction and risk-warning module.



The prediction work is developed on:



\- Branch: `feature/prediction`

\- Requirements: `docs/prediction-risk-requirements.md`

\- Repository baseline: `9e3a304 feat: add local PCA geometry features`



The prediction computer is responsible for:



\- prediction contracts;

\- explainable slow, normal, and fast growth scenarios;

\- future clearances;

\- time to warning threshold;

\- current and future risk grading;

\- warning reasons and messages;

\- JSON, CSV, and Excel export;

\- prediction tests and reviews;

\- integration with real current-clearance JSON from the main computer.



The main computer remains responsible for point-cloud processing and final branch integration.



\## 2. Core collaboration rules



All Codex sessions shall follow these rules:



1\. Work only on `feature/prediction`.

2\. Do not commit or merge directly into `main`.

3\. Read `docs/prediction-risk-requirements.md` before editing.

4\. Inspect the current repository and Git status before changing files.

5\. Do not modify unrelated modules.

6\. Do not install CUDA, PyTorch, TensorFlow, Open3D, or model weights.

7\. Do not read or modify LAS, LAZ, NPZ, `data/`, or `outputs/`.

8\. Do not hard-code `D:\\TreeLineWarning` or `E:\\TreeLineWarning` inside Python code.

9\. Do not disable, delete, skip, or weaken existing tests.

10\. Do not use `git reset --hard`, force-push, history rewriting, or automatic merging.

11\. Stop when the assigned stage is complete.

12\. Do not begin the next stage without review and approval.

13\. Every implementation stage shall update its review document.

14\. Every implementation stage shall leave the repository in a testable state.

15\. If requirements are ambiguous, report the ambiguity instead of inventing behavior.



\## 3. Required checks before every Codex session



Each new Codex session shall first run or inspect:



```powershell

git branch --show-current

```



Expected branch:



```text

feature/prediction

```



Then:



```powershell

git status --short

```



The session shall report any existing modification before editing.



Then inspect:



```powershell

git log -1 --oneline --decorate

```



The session shall read:



```text

docs/prediction-risk-requirements.md

docs/prediction-codex-session-tasks.md

```



The session shall also inspect files owned by the stage before proposing changes.



\## 4. Required verification commands



After every implementation stage, run:



```powershell

python -m pytest -q

```



```powershell

\& "$env:LOCALAPPDATA\\Microsoft\\WinGet\\Links\\ruff.exe" check .

```



```powershell

python -m compileall algorithms core tests scripts

```



```powershell

git diff --check

```



Also run:



```powershell

git status --short

```



and:



```powershell

git diff --stat

```



The stage is not complete if any required test or check fails.



\## 5. Session dependency map



The implementation sequence is:



```text

S0 repository audit

&#x20;       ↓

P0 prediction contracts

&#x20;       ↓

P0 review and approval

&#x20;       ↓

P1 growth calculations

&#x20;       ↓

P1 review and approval

&#x20;       ↓

P2 future-clearance and threshold timing

&#x20;       ↓

P2 review and approval

&#x20;       ↓

P3 risk grading and warnings

&#x20;       ↓

P3 review and approval

&#x20;       ↓

P4 JSON, CSV, and Excel export

&#x20;       ↓

P4 review and approval

&#x20;       ↓

P5 real-interface integration and final review

```



P0 through P5 shall not be implemented simultaneously in separate sessions because later stages depend on the contracts and behavior approved in earlier stages.



Independent review-only sessions may run after an implementation session has stopped editing.



\## 6. Session S0: Repository audit and implementation plan



\### 6.1 Session type



Read-only planning session.



\### 6.2 Objective



Confirm that the repository, branch, requirements, environment, and file boundaries are ready for P0.



\### 6.3 Allowed actions



\- Read repository files.

\- Run Git status and log commands.

\- Run tests and code checks.

\- Compare the requirements document with existing project conventions.

\- Produce a written implementation plan.



\### 6.4 Prohibited actions



\- Do not create prediction implementation files.

\- Do not modify contracts.

\- Do not modify tests.

\- Do not commit or push.

\- Do not install new dependencies.



\### 6.5 Required output



The session shall report:



\- current branch and commit;

\- worktree status;

\- existing contract conventions;

\- existing test conventions;

\- planned P0 files;

\- possible merge-conflict risks;

\- unresolved requirements;

\- recommendation to begin or delay P0.



\### 6.6 Prompt for Codex



```text

You are performing a read-only repository audit for the TreeLineWarning prediction module.



Repository rules:

\- Work on feature/prediction.

\- Read docs/prediction-risk-requirements.md.

\- Read docs/prediction-codex-session-tasks.md.

\- Do not edit any file.

\- Do not commit or push.



Tasks:

1\. Inspect the current branch, HEAD, and worktree.

2\. Inspect existing contract and contract-test conventions.

3\. Confirm the planned P0 file scope.

4\. Identify any ambiguous or contradictory prediction requirements.

5\. Run the existing tests and required static checks if the environment permits.

6\. Produce a concise P0 implementation plan.

7\. Stop without making changes.

```



\## 7. Session P0: Prediction contracts and fixtures



\### 7.1 Objective



Implement the prediction data contracts only.



P0 shall define:



\- `PredictionInputRecord`

\- `GrowthScenarioConfig`

\- `RiskThresholdConfig`

\- `PredictionResult`

\- `PredictionStatistics`



P0 shall not implement prediction algorithms.



\### 7.2 Allowed files



P0 may create or modify only:



```text

core/contracts/prediction.py

tests/contracts/test\_prediction\_contracts.py

tests/fixtures/prediction/prediction\_input\_v1.json

tests/fixtures/prediction/prediction\_config\_v1.json

reviews/P0-prediction-contract-review.md

docs/prediction-risk-requirements.md

docs/prediction-codex-session-tasks.md

```



The two documentation files may be modified only to correct an approved requirement error.



\### 7.3 Prohibited files



P0 shall not modify:



```text

core/contracts/\_\_init\_\_.py

algorithms/io/

algorithms/preprocessing/

algorithms/features/

algorithms/wires/

algorithms/vegetation/

algorithms/prediction/

data/

outputs/

```



\### 7.4 Implementation requirements



Use existing project conventions:



\- `@dataclass(frozen=True)`

\- strict `\_\_post\_init\_\_()` validation;

\- `to\_dict()`;

\- `from\_dict()`;

\- `require\_fields()`;

\- `reject\_unknown\_fields()`;

\- JSON-safe values;

\- finite-number validation;

\- rejection of Boolean values as numbers;

\- immutable tuple storage where appropriate;

\- `Literal` values for fixed string categories.



P0 shall validate ISO 8601 observation times and supported schema version `"1.0"`.



P0 shall reject unknown fields.



P0 shall not use Pydantic unless separately approved.



\### 7.5 Required tests



Tests shall cover:



\- valid construction;

\- default values;

\- round-trip serialization;

\- unknown fields;

\- missing fields;

\- empty identifiers;

\- invalid schema versions;

\- invalid ISO times;

\- invalid measurement quality;

\- negative or non-finite distances;

\- Boolean numeric values;

\- inconsistent current 3D clearance;

\- invalid scenarios;

\- invalid confidence;

\- invalid horizons;

\- invalid threshold ordering;

\- inconsistent statistics;

\- duplicate skipped identifiers;

\- JSON serializability.



\### 7.6 Required review



Update:



```text

reviews/P0-prediction-contract-review.md

```



The review shall include:



\- branch and baseline;

\- modified files;

\- final contract fields;

\- validation rules;

\- fixture summary;

\- test and Ruff results;

\- confirmation that prohibited files were untouched;

\- known limitations;

\- approval recommendation.



\### 7.7 P0 acceptance criteria



\- All five models exist.

\- Contracts match the approved requirements.

\- Fixtures exist and are small synthetic JSON files.

\- Contract tests pass.

\- Existing tests still pass.

\- Ruff and compilation checks pass.

\- `core/contracts/\_\_init\_\_.py` is unchanged.

\- No prediction calculation is implemented.

\- Work stops after P0.



\### 7.8 Prompt for Codex



```text

Implement P0 prediction contracts for TreeLineWarning.



Read first:

\- docs/prediction-risk-requirements.md

\- docs/prediction-codex-session-tasks.md

\- existing core/contracts files

\- existing tests/contracts files



Branch:

\- feature/prediction



Allowed files only:

\- core/contracts/prediction.py

\- tests/contracts/test\_prediction\_contracts.py

\- tests/fixtures/prediction/prediction\_input\_v1.json

\- tests/fixtures/prediction/prediction\_config\_v1.json

\- reviews/P0-prediction-contract-review.md



Do not modify:

\- core/contracts/\_\_init\_\_.py

\- algorithms/

\- data/

\- outputs/

\- existing unrelated tests



Implement only:

\- PredictionInputRecord

\- GrowthScenarioConfig

\- RiskThresholdConfig

\- PredictionResult

\- PredictionStatistics



Follow the requirements exactly.

Use frozen dataclasses, strict validation, to\_dict/from\_dict, JSON-safe values, and existing contract helper functions.

Do not implement growth, clearance, risk, pipeline, or export algorithms.



Run all required tests and checks.

Update the P0 review with actual results.

Do not commit, push, merge, or begin P1.

Stop and summarize modified files, tests, limitations, and review recommendation.

```



\## 8. Session P0-R: Independent contract review



\### 8.1 Session type



Read-only review session after P0 implementation.



\### 8.2 Objective



Review P0 without changing files.



\### 8.3 Review questions



\- Do all public fields match the approved requirements?

\- Are unknown fields rejected?

\- Are missing fields clearly reported?

\- Are Boolean values rejected as numbers?

\- Are all floats finite?

\- Is ISO time parsing correct?

\- Are thresholds ordered correctly?

\- Are count invariants enforced?

\- Are fixtures representative?

\- Are tests meaningful rather than superficial?

\- Was `core/contracts/\_\_init\_\_.py` left unchanged?

\- Did P0 accidentally implement later-stage behavior?



\### 8.4 Prompt for Codex



```text

Perform an independent read-only review of the completed TreeLineWarning P0 prediction contracts.



Read:

\- docs/prediction-risk-requirements.md

\- core/contracts/prediction.py

\- tests/contracts/test\_prediction\_contracts.py

\- tests/fixtures/prediction/

\- reviews/P0-prediction-contract-review.md



Do not edit files.

Do not commit or push.



Check contract completeness, strict validation, serialization, fixture quality, test coverage, prohibited-file protection, and scope control.



Run focused and repository-wide tests if possible.



Report:

\- blocking issues;

\- non-blocking issues;

\- missing tests;

\- requirement deviations;

\- merge-conflict risks;

\- approve or reject recommendation.



Stop after the review.

```



\## 9. Session P1: Three-scenario growth calculations



\### 9.1 Dependency



P0 must be reviewed and approved first.



\### 9.2 Objective



Implement deterministic slow, normal, and fast growth accumulation.



\### 9.3 Allowed files



```text

algorithms/prediction/\_\_init\_\_.py

algorithms/prediction/growth.py

tests/prediction/test\_growth.py

reviews/P1-growth-review.md

```



P1 may import approved contracts from:



```text

core/contracts/prediction.py

```



but shall not modify that contract file unless a blocking contract defect is separately approved.



\### 9.4 Prohibited scope



P1 shall not implement:



\- future clearance;

\- threshold timing;

\- risk grading;

\- warning messages;

\- export;

\- point-cloud integration;

\- UI.



\### 9.5 Required behavior



Implement:



```text

predicted\_vertical\_growth\_m =

vertical\_growth\_rate\_m\_per\_year × horizon\_months / 12

```



```text

predicted\_horizontal\_growth\_m =

horizontal\_growth\_rate\_m\_per\_year × horizon\_months / 12

```



Validate the complete scenario set:



\- exactly one `slow`;

\- exactly one `normal`;

\- exactly one `fast`;

\- vertical rates ordered `slow <= normal <= fast`;

\- horizontal rates ordered `slow <= normal <= fast`.



Results shall be deterministic and finite.



\### 9.6 Required tests



\- 6-, 12-, and 24-month calculations;

\- zero growth;

\- fractional annual rates;

\- invalid scenario duplication;

\- missing scenario;

\- invalid ordering;

\- deterministic ordering;

\- no mutation of input configurations.



\### 9.7 Prompt for Codex



```text

Implement P1 growth calculations only for TreeLineWarning.



Read the approved requirements and P0 contracts first.



Allowed files:

\- algorithms/prediction/\_\_init\_\_.py

\- algorithms/prediction/growth.py

\- tests/prediction/test\_growth.py

\- reviews/P1-growth-review.md



Do not modify P0 contracts unless you find a blocking defect; report such a defect and stop instead of silently changing the contract.



Implement deterministic linear accumulated vertical and horizontal growth for slow, normal, and fast scenarios at configured horizons.



Validate the complete three-scenario set and stable result ordering.



Do not implement future clearance, risk grading, warnings, export, or integration.



Run all required tests and static checks.

Update the P1 review.

Do not commit, push, merge, or begin P2.

```



\## 10. Session P2: Future clearance and threshold timing



\### 10.1 Dependency



P1 must be reviewed and approved first.



\### 10.2 Objective



Implement:



\- predicted horizontal clearance;

\- predicted vertical clearance;

\- predicted 3D clearance;

\- clearance reductions;

\- safety margins;

\- months to warning threshold;

\- threshold dimension selection.



\### 10.3 Allowed files



```text

algorithms/prediction/growth.py

algorithms/prediction/pipeline.py

tests/prediction/test\_clearance.py

tests/prediction/test\_pipeline.py

reviews/P2-clearance-review.md

```



A separate helper module may be proposed, but it shall not be created without clearly documenting why it is necessary.



\### 10.4 Required formulas



Use the formulas approved in the requirements document.



Public future clearances shall be clamped to zero.



Safety margins shall remain signed.



`months\_to\_threshold` shall:



\- be `0` if already at warning level;

\- return the earliest dimension;

\- return `None` if not reached in the search window;

\- use stable tie priority `3d`, `horizontal`, `vertical`.



\### 10.5 Prohibited scope



P2 shall not implement full risk grading, warning text, export, or real-data integration.



\### 10.6 Required tests



\- each future-clearance formula;

\- 3D recomputation;

\- zero clamping;

\- safety-margin signs;

\- threshold at month zero;

\- threshold reached in future;

\- threshold never reached;

\- zero growth;

\- simultaneous threshold tie;

\- deterministic calculations.



\### 10.7 Prompt for Codex



```text

Implement P2 future-clearance and threshold-timing calculations only.



Read:

\- docs/prediction-risk-requirements.md

\- approved P0 contracts

\- approved P1 growth implementation and review



Allowed files:

\- algorithms/prediction/growth.py

\- algorithms/prediction/pipeline.py

\- tests/prediction/test\_clearance.py

\- tests/prediction/test\_pipeline.py

\- reviews/P2-clearance-review.md



Implement predicted horizontal, vertical, and 3D clearance, reductions, warning-threshold safety margins, months to threshold, threshold dimension, and search-window behavior.



Follow approved formulas and deterministic tie rules exactly.



Do not implement full risk grading, warning messages, export, UI, or real-data integration.



Run all required tests and checks.

Update the P2 review.

Do not commit, push, merge, or begin P3.

```



\## 11. Session P3: Risk grading and warning generation



\### 11.1 Dependency



P2 must be reviewed and approved first.



\### 11.2 Objective



Implement:



\- current risk;

\- future risk;

\- most severe dimension;

\- risk worsening;

\- warning-required logic;

\- stable warning-reason codes;

\- human-readable warning messages;

\- result confidence;

\- deterministic worst-case selection.



\### 11.3 Allowed files



```text

algorithms/prediction/risk.py

algorithms/prediction/pipeline.py

algorithms/prediction/\_\_init\_\_.py

tests/prediction/test\_risk.py

tests/prediction/test\_pipeline.py

reviews/P3-risk-review.md

```



\### 11.4 Required risk order



```text

safe < watch < warning < critical

```



The final level is the most severe among:



\- 3D;

\- horizontal;

\- vertical.



Equality counts as reaching a threshold.



\### 11.5 Required warning behavior



Warning reasons shall be:



\- machine-readable;

\- deduplicated;

\- deterministic;

\- stable across identical runs.



Human-readable messages shall explain:



\- scenario;

\- horizon;

\- controlling clearance dimension;

\- future level;

\- threshold timing where available.



Messages shall not claim official regulatory violations when demonstration thresholds are used.



\### 11.6 Required tests



\- every risk level;

\- equality boundaries;

\- three-dimension severity selection;

\- current/future comparison;

\- warning-required cases;

\- warning-reason ordering;

\- suspect-quality confidence reduction;

\- invalid-quality skipping;

\- worst-case priority and ties.



\### 11.7 Prompt for Codex



```text

Implement P3 risk grading and warning generation for TreeLineWarning.



Read all approved P0-P2 requirements, code, tests, and reviews.



Allowed files:

\- algorithms/prediction/risk.py

\- algorithms/prediction/pipeline.py

\- algorithms/prediction/\_\_init\_\_.py

\- tests/prediction/test\_risk.py

\- tests/prediction/test\_pipeline.py

\- reviews/P3-risk-review.md



Implement current and future risk grading, controlling dimensions, risk worsening, warning-required logic, stable reason codes, human-readable messages, result confidence, and deterministic worst-case selection.



Do not implement export, UI, or real main-computer integration.



Do not describe demonstration thresholds as official standards.



Run all required tests and checks.

Update the P3 review.

Do not commit, push, merge, or begin P4.

```



\## 12. Session P4: JSON, CSV, and Excel export



\### 12.1 Dependency



P3 must be reviewed and approved first.



\### 12.2 Objective



Implement result export without changing prediction calculations.



\### 12.3 Allowed files



```text

algorithms/prediction/io.py

algorithms/prediction/\_\_init\_\_.py

tests/prediction/test\_export.py

reviews/P4-export-review.md

```



A fixture or expected-output file may be added under:



```text

tests/fixtures/prediction/

```



only when needed by tests.



\### 12.4 Dependencies



P4 may require minimal export dependencies such as an Excel-writing library.



Before installing a new dependency, Codex shall:



1\. inspect existing dependencies;

2\. explain why the dependency is needed;

3\. avoid heavyweight unrelated packages;

4\. record the dependency requirement in the review.



Codex shall not modify environment configuration silently.



\### 12.5 Required exports



JSON:



\- metadata;

\- statistics;

\- results;

\- configurations;

\- warnings;

\- UTF-8;

\- no NaN or infinity.



CSV:



\- one row per `PredictionResult`;

\- stable columns;

\- flattened configuration fields;

\- deterministic row order;

\- documented list separator.



Excel sheets:



```text

Prediction Results

Risk Summary

Scenario Curves

Configuration

Warnings

```



\### 12.6 File-safety requirements



\- No hard-coded project paths.

\- Parent directories created only when requested.

\- No overwrite unless explicitly enabled.

\- Temporary files cleaned after failure.

\- Generated export artifacts must remain ignored by Git.



\### 12.7 Prompt for Codex



```text

Implement P4 prediction result exporters only.



Read the approved requirements and P0-P3 implementation first.



Allowed files:

\- algorithms/prediction/io.py

\- algorithms/prediction/\_\_init\_\_.py

\- tests/prediction/test\_export.py

\- reviews/P4-export-review.md

\- tests/fixtures/prediction/ only when required



Implement JSON, CSV, and Excel export with stable ordering, UTF-8 handling, configuration preservation, statistics, risk summary, scenario curves, and warning sheets.



Do not change prediction formulas or risk behavior.



Do not overwrite existing files unless explicitly requested.

Do not hard-code local paths.

Do not commit generated export artifacts.



If an additional minimal dependency is required, explain it before installation and record it in the review.



Run all required tests and checks.

Update the P4 review.

Do not commit, push, merge, or begin P5.

```



\## 13. Session P5: Main-computer interface integration



\### 13.1 Dependency



P4 must be reviewed and approved first.



The main computer must provide representative real current-clearance JSON records.



\### 13.2 Objective



Verify that real main-computer output can replace simulated JSON without rewriting prediction algorithms.



\### 13.3 Allowed files



Depending on the approved integration approach:



```text

algorithms/prediction/pipeline.py

algorithms/prediction/io.py

tests/prediction/test\_integration.py

tests/fixtures/prediction/

reviews/P5-integration-review.md

docs/

```



Changes to public contracts require separate review.



\### 13.4 Integration checks



Verify:



\- schema version;

\- required fields;

\- unknown fields;

\- identifiers;

\- units;

\- observation time;

\- measurement quality;

\- quality flags;

\- 3D, horizontal, and vertical clearance consistency;

\- real-record contract validation;

\- prediction with real-shaped records;

\- no LAS or NPZ read;

\- no computer-specific path dependency.



\### 13.5 Compatibility behavior



If real JSON does not match schema `"1.0"`:



\- do not silently rename or discard fields;

\- document every mismatch;

\- propose an explicit adapter or schema revision;

\- stop before implementing an unapproved compatibility layer.



\### 13.6 Required tests



\- simulated fixture integration;

\- representative real-record validation;

\- batch integration;

\- invalid real-record handling;

\- deterministic result order;

\- export of integrated results;

\- no prohibited data access.



\### 13.7 Prompt for Codex



```text

Perform P5 integration of the approved prediction pipeline with representative real current-clearance JSON from the main computer.



Read all approved requirements, code, tests, and reviews from P0-P4.



Do not read LAS, LAZ, or NPZ files.

Do not rerun point-cloud processing.

Do not recalculate current clearance.



Validate the real records against PredictionInputRecord.

Document all schema, type, unit, identifier, and quality-field mismatches.



If the real records match schema 1.0, add focused integration tests and verify that simulated JSON can be replaced without prediction-code redesign.



If the records do not match, stop and propose an explicit reviewed compatibility plan. Do not silently discard or rename fields.



Run all required tests and checks.

Update reviews/P5-integration-review.md.

Do not merge into main.

```



\## 14. Session F-R: Final independent review



\### 14.1 Objective



Conduct a read-only final review before the main computer fetches or merges the branch.



\### 14.2 Review scope



Review:



\- requirements compliance;

\- public contracts;

\- formulas;

\- risk grading;

\- warnings;

\- confidence;

\- worst-case selection;

\- exporters;

\- tests;

\- documentation;

\- Git status;

\- prohibited-file protection;

\- main-computer compatibility.



\### 14.3 Required report



Classify findings as:



\- blocker;

\- major;

\- minor;

\- suggestion.



The final recommendation shall be one of:



\- approve for main-computer review;

\- approve with non-blocking follow-up;

\- reject until blocking issues are fixed.



\### 14.4 Prompt for Codex



```text

Perform a final read-only review of the complete TreeLineWarning prediction and risk-warning module.



Do not edit files.

Do not commit, push, or merge.



Review:

\- docs/prediction-risk-requirements.md

\- all prediction contracts and algorithms

\- all prediction tests and fixtures

\- all P0-P5 reviews

\- Git diff and status

\- prohibited directories

\- exported schema behavior



Run the full test suite and required static checks.



Report findings by severity:

\- blocker

\- major

\- minor

\- suggestion



Verify that:

\- no point-cloud processing was duplicated;

\- no paths were hard-coded;

\- no official threshold claims were invented;

\- no failing test was weakened;

\- no generated data was tracked;

\- simulated and real-shaped JSON use the same interface.



End with an approve or reject recommendation.

```



\## 15. Safe Git workflow after each approved stage



After a stage is implemented, reviewed, and approved by the project owner, check:



```powershell

git status

```



Review:



```powershell

git diff

```



Stage only the approved files.



Example P0 staging:



```powershell

git add core/contracts/prediction.py

git add tests/contracts/test\_prediction\_contracts.py

git add tests/fixtures/prediction

git add reviews/P0-prediction-contract-review.md

git add docs/prediction-risk-requirements.md

git add docs/prediction-codex-session-tasks.md

```



Before committing, run all required tests again.



Recommended P0 commit:



```powershell

git commit -m "feat(prediction): add prediction contracts"

```



Then push:



```powershell

git push

```



Do not commit until the project owner has reviewed the diff and explicitly approved the stage.



\## 16. Multiple Codex session safety



Different Codex sessions shall not edit the same working tree simultaneously.



Recommended practice:



1\. Finish one implementation session.

2\. Stop its editing.

3\. Run a separate read-only review session.

4\. Fix approved issues in one designated fix session.

5\. Run tests.

6\. Commit the approved stage.

7\. Push the branch.

8\. Start the next stage in a new Codex session.



Do not open P1, P2, and P3 implementation sessions simultaneously.



Safe parallel activity is limited to:



\- one implementation session;

\- one read-only reviewer after implementation has stopped;

\- one documentation-analysis session that does not edit repository files.



\## 17. Session handoff template



At the end of every Codex implementation session, require the following handoff:



```text

Stage:

Branch:

Baseline commit:

Final HEAD:

Modified files:

New files:

Tests run:

Test results:

Static checks:

Known limitations:

Requirement deviations:

Unresolved issues:

Review document:

Recommended next action:

```



The next Codex session shall read this handoff before beginning.



\## 18. Stop conditions



A Codex session shall stop immediately when:



\- the current branch is not `feature/prediction`;

\- unrelated uncommitted changes already exist;

\- a requirement contradiction is found;

\- an approved contract must be changed by a later stage;

\- a prohibited file appears necessary to modify;

\- existing tests fail before the assigned changes;

\- a required dependency cannot be justified;

\- real integration data does not match schema `"1.0"`;

\- the assigned scope is complete.



The session shall report the reason instead of continuing blindly.



\## 19. Project-owner approval gates



The project owner shall explicitly approve:



\- P0 contracts before P1;

\- P1 growth formulas before P2;

\- P2 clearance and timing behavior before P3;

\- P3 risk behavior before P4;

\- P4 export structure before P5;

\- P5 integration before the main computer merges the branch.



No Codex session may treat silence as approval.

