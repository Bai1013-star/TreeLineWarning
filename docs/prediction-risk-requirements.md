\# TreeLineWarning Prediction and Risk Warning Requirements



\## 1. Document status



\- Document type: module requirements specification.

\- Module: vegetation growth prediction and risk warning.

\- Project: `TreeLineWarning`.

\- Development branch: `feature/prediction`.

\- Baseline commit: `9e3a304 feat: add local PCA geometry features`.

\- Current status: requirements definition only.

\- Prediction implementation must not begin until this document and the Codex session task breakdown are reviewed.



\## 2. Module ownership



The prediction computer is responsible for predicting future vegetation–conductor clearance and determining current and future risk.



The module owner is responsible for:



\- prediction input and output contracts;

\- explainable vegetation growth scenarios;

\- future three-dimensional clearance;

\- future horizontal clearance;

\- future vertical clearance;

\- safety-margin calculation;

\- estimated months to warning thresholds;

\- current and future risk levels;

\- worst-case scenario selection;

\- confidence values;

\- warning reasons and warning messages;

\- JSON, CSV, and Excel exports;

\- prediction tests and review documents;

\- later integration with real clearance records from the main computer.



The main computer remains responsible for:



\- LAS and point-cloud input;

\- point-cloud preprocessing;

\- geometric feature extraction;

\- conductor candidate detection;

\- conductor reconstruction;

\- vegetation-region identification;

\- current clearance calculation;

\- final review and branch integration.



\## 3. Functional objective



The module shall accept a small structured record describing the current relationship between one vegetation object and one conductor.



It shall use configurable, explainable growth assumptions to estimate how the clearance may change under three scenarios:



\- `slow`;

\- `normal`;

\- `fast`.



The initial supported prediction horizons are:



\- 6 months;

\- 12 months;

\- 24 months.



The module shall produce machine-readable results that can be reviewed independently and later consumed by reporting or user-interface modules.

\## 4. Prediction input contract



The prediction module shall receive structured JSON data only. It shall not receive LAS, LAZ, NPZ, point-cloud arrays, model weights, or hard-coded local paths.



The public input model shall be named:



`PredictionInputRecord`



\### 4.1 Required fields



| Field | Type | Unit | Requirement |

| --- | --- | --- | --- |

| `schema\_version` | string | — | Required; P0 supports `"1.0"` |

| `record\_id` | string | — | Required and non-empty; uniquely identifies the input record |

| `vegetation\_id` | string | — | Required and non-empty |

| `wire\_id` | string | — | Required and non-empty |

| `observation\_time` | string | — | Required; valid ISO 8601 date-time |

| `current\_clearance\_3d\_m` | number | m | Required; finite and non-negative |

| `horizontal\_clearance\_m` | number | m | Required; finite and non-negative |

| `vertical\_clearance\_m` | number | m | Required; finite and non-negative |

| `vegetation\_height\_m` | number | m | Required; finite and greater than zero |

| `crown\_radius\_m` | number | m | Required; finite and non-negative |

| `measurement\_quality` | string | — | Required; one of `valid`, `suspect`, or `invalid` |

| `quality\_flags` | integer | bitmask | Required; non-negative integer |



\### 4.2 Input validation rules



\- Unknown JSON fields shall be rejected.

\- Missing required fields shall be rejected with field-specific error messages.

\- Boolean values shall not be accepted as numeric values.

\- `NaN`, positive infinity, and negative infinity shall be rejected.

\- Identifier fields shall not be empty or whitespace-only.

\- `observation\_time` shall be parseable as an ISO 8601 date-time.

\- A timezone offset is recommended but is not mandatory for schema version `1.0`.

\- `current\_clearance\_3d\_m` shall not be smaller than either the horizontal or vertical clearance, except for a small floating-point tolerance.

\- `measurement\_quality="invalid"` shall not be used for normal prediction calculations.

\- `measurement\_quality="suspect"` may be processed, but the result shall contain a warning and reduced confidence.

\- `quality\_flags` shall be preserved for traceability and shall not be silently discarded.

\- Input validation shall not read any point-cloud file or inspect any local data directory.



\### 4.3 Version 1.0 example



```json

{

&#x20; "schema\_version": "1.0",

&#x20; "record\_id": "record\_001",

&#x20; "vegetation\_id": "vegetation\_001",

&#x20; "wire\_id": "wire\_001",

&#x20; "observation\_time": "2026-07-21T12:00:00",

&#x20; "current\_clearance\_3d\_m": 2.8,

&#x20; "horizontal\_clearance\_m": 2.4,

&#x20; "vertical\_clearance\_m": 1.6,

&#x20; "vegetation\_height\_m": 8.5,

&#x20; "crown\_radius\_m": 2.1,

&#x20; "measurement\_quality": "valid",

&#x20; "quality\_flags": 0

}

\## 5. Prediction configuration contracts



P0 shall define two public configuration models:



\- `GrowthScenarioConfig`

\- `RiskThresholdConfig`



These configuration models shall contain only small JSON-safe values and shall not contain file paths, point-cloud arrays, model weights, or computer-specific directories.

\### 5.1 GrowthScenarioConfig



One `GrowthScenarioConfig` object represents one explainable vegetation-growth scenario.



The first version supports three scenario names:



\- `slow`

\- `normal`

\- `fast`



Each scenario describes assumed vegetation growth rates and the prediction months to be calculated.

Required fields:



| Field | Type | Unit | Requirement |

| --- | --- | --- | --- |

| `scenario` | string | — | One of `slow`, `normal`, or `fast` |

| `vertical\_growth\_rate\_m\_per\_year` | number | m/year | Finite and non-negative |

| `horizontal\_growth\_rate\_m\_per\_year` | number | m/year | Finite and non-negative |

| `confidence` | number | 0–1 | Finite value from 0.0 to 1.0 |

| `prediction\_horizons\_months` | integer list | months | Positive, unique, ascending integers |

Default prediction horizons shall be:



```json

\[6, 12, 24]

Validation rules:



\- Boolean values shall not be accepted as numeric values.

\- Growth rates shall be finite and non-negative.

\- `confidence` shall be between 0.0 and 1.0.

\- Prediction months shall be positive integers.

\- Duplicate prediction months shall be rejected.

\- Prediction months shall be stored in ascending order.

\- The scenario name shall be preserved in every prediction result.

Cross-scenario rules:



\- A complete prediction run shall normally contain exactly one `slow`, one `normal`, and one `fast` scenario.

\- For the same input record, expected growth-rate ordering shall be:



```text

slow <= normal <= fast

\### 5.2 RiskThresholdConfig



`RiskThresholdConfig` defines configurable clearance thresholds for risk grading.



The first version shall evaluate three clearance dimensions independently:



\- three-dimensional clearance;

\- horizontal clearance;

\- vertical clearance.



The final risk level shall use the most severe result among the three dimensions.

Required fields:



| Field | Type | Unit |

| --- | --- | --- |

| `watch\_clearance\_3d\_m` | number | m |

| `warning\_clearance\_3d\_m` | number | m |

| `critical\_clearance\_3d\_m` | number | m |

| `watch\_horizontal\_clearance\_m` | number | m |

| `warning\_horizontal\_clearance\_m` | number | m |

| `critical\_horizontal\_clearance\_m` | number | m |

| `watch\_vertical\_clearance\_m` | number | m |

| `warning\_vertical\_clearance\_m` | number | m |

| `critical\_vertical\_clearance\_m` | number | m |

| `threshold\_search\_max\_months` | integer | months |

Validation rules:



\- All clearance thresholds shall be finite and non-negative.

\- Boolean values shall not be accepted as threshold numbers.

\- `threshold\_search\_max\_months` shall be a positive integer.

\- For each clearance dimension, the threshold order shall be:



```text

critical <= warning <= watch

Risk grading rules:



\- A clearance greater than the `watch` threshold shall be classified as `safe`.

\- A clearance at or below the `watch` threshold shall be classified as at least `watch`.

\- A clearance at or below the `warning` threshold shall be classified as at least `warning`.

\- A clearance at or below the `critical` threshold shall be classified as `critical`.

\- Threshold comparisons shall include equality.

\- Three-dimensional, horizontal, and vertical clearances shall be graded independently.

\- The final risk level shall be the most severe level among the three clearance dimensions.

\- The result shall record every clearance dimension that contributed to the final risk level.

Threshold usage rules:



\- Threshold values used during simulated development shall be treated as demonstration configuration only.

\- Demonstration thresholds shall not be described as official legal, regulatory, utility-company, or voltage-level safety standards.

\- The threshold configuration used for every prediction run shall be preserved in the result for traceability.

\- Future versions may provide different threshold sets for different voltage levels, operating environments, or management standards.

\- Changing threshold values shall not require changes to `PredictionInputRecord`.

\## 6. Prediction output contracts



P0 shall define two public output models:



\- `PredictionResult`

\- `PredictionStatistics`



`PredictionResult` represents the prediction and risk result for one input record.



`PredictionStatistics` represents summary counts, runtime information, and warnings for one prediction run.



All output values shall be JSON-safe, traceable to the original input record, and independent of computer-specific file paths.

\## 6.1 PredictionResult



`PredictionResult` shall contain the prediction output, risk assessment, and traceability information for one input record.



Each result shall correspond to exactly one `record\_id` from `PredictionInputRecord`.



The model shall preserve enough information to explain why a record is classified into a specific risk level.

\### 6.1.1 Traceability and current-state fields



| Field | Type | Requirement |

| --- | --- | --- |

| `schema\_version` | string | Required; P0 supports `"1.0"` |

| `record\_id` | string | Must match the source `PredictionInputRecord.record\_id` |

| `vegetation\_id` | string | Preserved from the input record |

| `wire\_id` | string | Preserved from the input record |

| `observation\_time` | string | Preserved from the input record |

| `measurement\_quality` | string | Preserved from the input record |

| `quality\_flags` | integer | Preserved from the input record |

| `current\_clearance\_3d\_m` | number | Current three-dimensional clearance in meters |

| `current\_horizontal\_clearance\_m` | number | Current horizontal clearance in meters |

| `current\_vertical\_clearance\_m` | number | Current vertical clearance in meters |

| `current\_risk\_level` | string | One of `safe`, `watch`, `warning`, or `critical` |

| `current\_risk\_dimensions` | string list | Clearance dimensions responsible for the current final risk level |

\### 6.1.2 Scenario and prediction-horizon fields



One `PredictionResult` shall represent one input record under one growth scenario at one prediction horizon.



With the default configuration, one valid input record shall normally produce nine results:



\- 3 growth scenarios;

\- 3 prediction horizons;

\- `3 × 3 = 9` result records.



| Field | Type | Requirement |

| --- | --- | --- |

| `scenario` | string | One of `slow`, `normal`, or `fast` |

| `horizon\_months` | integer | Positive prediction horizon in months |

| `vertical\_growth\_rate\_m\_per\_year` | number | Vertical growth rate used for this result |

| `horizontal\_growth\_rate\_m\_per\_year` | number | Horizontal growth rate used for this result |

| `scenario\_confidence` | number | Scenario confidence from 0.0 to 1.0 |

\### 6.1.3 Predicted growth and clearance fields



| Field | Type | Unit | Requirement |

| --- | --- | --- | --- |

| `predicted\_vertical\_growth\_m` | number | m | Non-negative vertical growth accumulated over the prediction horizon |

| `predicted\_horizontal\_growth\_m` | number | m | Non-negative horizontal crown growth accumulated over the prediction horizon |

| `predicted\_clearance\_3d\_m` | number | m | Predicted three-dimensional clearance; finite and not less than zero |

| `predicted\_horizontal\_clearance\_m` | number | m | Predicted horizontal clearance; finite and not less than zero |

| `predicted\_vertical\_clearance\_m` | number | m | Predicted vertical clearance; finite and not less than zero |

| `clearance\_reduction\_3d\_m` | number | m | Reduction from current three-dimensional clearance |

| `horizontal\_clearance\_reduction\_m` | number | m | Reduction from current horizontal clearance |

| `vertical\_clearance\_reduction\_m` | number | m | Reduction from current vertical clearance |

\### 6.1.4 Safety-margin and threshold-timing fields



Safety margins shall be calculated relative to the configured `warning` thresholds.



| Field | Type | Unit | Requirement |

| --- | --- | --- | --- |

| `safety\_margin\_3d\_m` | number | m | Predicted 3D clearance minus the configured 3D warning threshold |

| `horizontal\_safety\_margin\_m` | number | m | Predicted horizontal clearance minus the configured horizontal warning threshold |

| `vertical\_safety\_margin\_m` | number | m | Predicted vertical clearance minus the configured vertical warning threshold |

| `months\_to\_threshold` | number or null | months | Earliest estimated time from observation until any warning threshold is reached |

| `threshold\_dimension` | string or null | — | One of `3d`, `horizontal`, or `vertical` when a threshold is reached |

| `threshold\_reached\_within\_search\_window` | boolean | — | Whether a warning threshold is reached within `threshold\_search\_max\_months` |



A positive safety margin indicates that the predicted clearance remains above the warning threshold.



A zero safety margin indicates that the predicted clearance is exactly at the warning threshold.



A negative safety margin indicates that the predicted clearance has crossed below the warning threshold.



When no warning threshold is reached within the configured search window:



\- `months\_to\_threshold` shall be `null`;

\- `threshold\_dimension` shall be `null`;

\- `threshold\_reached\_within\_search\_window` shall be `false`.

\### 6.1.5 Future-risk and warning fields



| Field | Type | Requirement |

| --- | --- | --- |

| `future\_risk\_level` | string | One of `safe`, `watch`, `warning`, or `critical` |

| `future\_risk\_dimensions` | string list | Clearance dimensions responsible for the future final risk level |

| `is\_risk\_worsened` | boolean | Whether the future risk level is more severe than the current risk level |

| `warning\_required` | boolean | Whether the result requires a warning |

| `warning\_reasons` | string list | Stable machine-readable reason codes |

| `warning\_message` | string | Human-readable explanation of the risk |

| `is\_worst\_case` | boolean | Whether this result is the worst result among all scenarios and horizons for the same input record |



`warning\_reasons` may contain values such as:



\- `CURRENT\_CLEARANCE\_LOW`

\- `PREDICTED\_3D\_CLEARANCE\_LOW`

\- `PREDICTED\_HORIZONTAL\_CLEARANCE\_LOW`

\- `PREDICTED\_VERTICAL\_CLEARANCE\_LOW`

\- `WARNING\_THRESHOLD\_REACHED`

\- `CRITICAL\_THRESHOLD\_REACHED`

\- `FAST\_GROWTH\_SCENARIO`

\- `SUSPECT\_MEASUREMENT\_QUALITY`

\### 6.1.6 Configuration and provenance fields



| Field | Type | Requirement |

| --- | --- | --- |

| `scenario\_config` | object | Complete `GrowthScenarioConfig` used to produce this result |

| `risk\_threshold\_config` | object | Complete `RiskThresholdConfig` used for risk grading |

| `result\_confidence` | number | Final confidence from 0.0 to 1.0 after considering scenario confidence and measurement quality |

| `prediction\_method` | string | Stable method identifier; version 1 uses `linear\_scenario\_v1` |

| `algorithm\_version` | string | Non-empty prediction algorithm version |

| `created\_at` | string | ISO 8601 date-time when the result was generated |

| `assumptions` | string list | Human-readable assumptions used by the prediction |



Rules:



\- The exact scenario and risk-threshold configurations used for a result shall be preserved.

\- `result\_confidence` shall be finite and between 0.0 and 1.0.

\- A `suspect` measurement shall reduce result confidence.

\- An `invalid` measurement shall not produce a normal prediction result.

\- Configuration snapshots shall contain JSON-safe values only.

\- Computer-specific paths shall not be included in provenance fields.

\### 6.2 PredictionStatistics



`PredictionStatistics` shall summarize one complete prediction run.



It shall record how many input records were received, how many were processed or skipped, how many scenario results were produced, and how many results entered each risk level.



It shall also preserve runtime information and non-fatal warnings produced during the prediction run.

Required count fields:



| Field | Type | Requirement |

| --- | --- | --- |

| `input\_record\_count` | integer | Total number of received input records |

| `processed\_record\_count` | integer | Number of records that produced prediction results |

| `skipped\_record\_count` | integer | Number of records skipped because prediction was not allowed |

| `suspect\_record\_count` | integer | Number of processed records with suspect measurement quality |

| `scenario\_result\_count` | integer | Total number of generated scenario-and-horizon results |

| `safe\_result\_count` | integer | Number of results classified as `safe` |

| `watch\_result\_count` | integer | Number of results classified as `watch` |

| `warning\_result\_count` | integer | Number of results classified as `warning` |

| `critical\_result\_count` | integer | Number of results classified as `critical` |

| `warning\_required\_count` | integer | Number of results requiring a warning |

| `worst\_case\_result\_count` | integer | Number of results marked as worst-case results |

Runtime and warning fields:



| Field | Type | Requirement |

| --- | --- | --- |

| `elapsed\_seconds` | number | Total prediction runtime; finite and non-negative |

| `started\_at` | string | ISO 8601 date-time when the prediction run started |

| `completed\_at` | string | ISO 8601 date-time when the prediction run completed |

| `warnings` | string list | Non-fatal warnings produced during the prediction run |

| `skipped\_record\_ids` | string list | Record identifiers skipped during prediction |

| `statistics\_version` | string | Non-empty statistics-schema version |



Validation rules:



\- All count fields shall be non-negative integers.

\- Boolean values shall not be accepted as integer counts.

\- `elapsed\_seconds` shall be finite and non-negative.

\- `processed\_record\_count + skipped\_record\_count` shall equal `input\_record\_count`.

\- The sum of `safe\_result\_count`, `watch\_result\_count`, `warning\_result\_count`, and `critical\_result\_count` shall equal `scenario\_result\_count`.

\- `skipped\_record\_ids` shall contain no duplicate identifiers.

\- Warnings and skipped identifiers shall be stored as immutable sequences in the Python contract and exported as JSON lists.

\## 7. Measurement-quality and invalid-input handling



The prediction module shall distinguish between structurally invalid data, invalid measurements, and suspect measurements.



Structurally invalid records shall be rejected during contract validation.



Records with `measurement\_quality="invalid"` may be accepted as valid JSON records, but they shall not enter normal growth prediction or risk projection.



Records with `measurement\_quality="suspect"` may be predicted, but every resulting prediction shall carry reduced confidence and an explicit warning reason.

\### 7.1 Handling rules



\- Structurally invalid records shall raise a clear validation error and shall not be counted as processed records.

\- Records with `measurement\_quality="invalid"` shall be added to `skipped\_record\_ids`.

\- Invalid-measurement records shall increase `skipped\_record\_count`.

\- Invalid-measurement records shall not produce `PredictionResult` objects.

\- Records with `measurement\_quality="suspect"` shall increase `suspect\_record\_count`.

\- Suspect records shall produce prediction results only when all other required fields are valid.

\- Every result produced from a suspect record shall include `SUSPECT\_MEASUREMENT\_QUALITY` in `warning\_reasons`.

\- The final confidence for a suspect record shall be lower than the original scenario confidence.

\- The prediction module shall not silently replace missing or invalid clearance values with zero.

\## 8. Prediction calculation boundaries



The first implementation shall use deterministic and explainable linear growth scenarios.



It shall not train, download, or load any machine-learning model.



Prediction calculations shall use only:



\- the validated `PredictionInputRecord`;

\- the selected `GrowthScenarioConfig`;

\- the selected `RiskThresholdConfig`;

\- explicit formulas documented in the implementation and review files.

\### 8.1 Growth accumulation rules



For prediction horizon `horizon\_months`, version 1 shall calculate accumulated growth as:



```text

predicted\_vertical\_growth\_m =

vertical\_growth\_rate\_m\_per\_year × horizon\_months / 12

\### 8.1 Growth accumulation rules



For prediction horizon `horizon\_months`, version 1 shall calculate accumulated growth as:



```text

predicted\_vertical\_growth\_m =

vertical\_growth\_rate\_m\_per\_year × horizon\_months / 12

```



```text

predicted\_horizontal\_growth\_m =

horizontal\_growth\_rate\_m\_per\_year × horizon\_months / 12

```



Rules:



\- Growth shall be calculated from the original observation time.

\- Version 1 shall use simple linear accumulation without compounding.

\- Predicted growth shall never be negative.

\- The same input and configuration shall always produce the same result.

\- Version 1 shall not automatically adjust growth for species, season, weather, pruning, drought, or disease.

\- Such factors may be represented later through different scenario configurations without changing the input contract.



\### 8.2 Future-clearance calculation rules



Version 1 shall estimate future horizontal and vertical clearance as:



```text

predicted\_horizontal\_clearance\_m =

max(0, current\_horizontal\_clearance\_m - predicted\_horizontal\_growth\_m)

```



```text

predicted\_vertical\_clearance\_m =

max(0, current\_vertical\_clearance\_m - predicted\_vertical\_growth\_m)

```



The predicted three-dimensional clearance shall be calculated from the predicted horizontal and vertical components:



```text

predicted\_clearance\_3d\_m =

sqrt(

&#x20;   predicted\_horizontal\_clearance\_m²

&#x20;   + predicted\_vertical\_clearance\_m²

)

```



Clearance reductions shall be calculated as:



```text

clearance\_reduction\_3d\_m =

max(0, current\_clearance\_3d\_m - predicted\_clearance\_3d\_m)

```



```text

horizontal\_clearance\_reduction\_m =

max(0, current\_horizontal\_clearance\_m - predicted\_horizontal\_clearance\_m)

```



```text

vertical\_clearance\_reduction\_m =

max(0, current\_vertical\_clearance\_m - predicted\_vertical\_clearance\_m)

```



Rules:



\- Public predicted-clearance values shall never be negative.

\- A calculated clearance of zero represents contact with or intrusion into the conductor-clearance boundary.

\- The implementation shall not silently increase any future clearance above its current value.

\- All calculation outputs shall be finite.

\- Floating-point calculations shall use an explicitly documented tolerance.

\- Version 1 assumes that horizontal crown growth reduces horizontal clearance and vertical growth reduces vertical clearance.

\- More advanced growth-direction or conductor-motion models are outside the version 1 scope.



\### 8.3 Safety-margin rules



Safety margins shall be calculated relative to the configured `warning` thresholds:



```text

safety\_margin\_3d\_m =

predicted\_clearance\_3d\_m - warning\_clearance\_3d\_m

```



```text

horizontal\_safety\_margin\_m =

predicted\_horizontal\_clearance\_m - warning\_horizontal\_clearance\_m

```



```text

vertical\_safety\_margin\_m =

predicted\_vertical\_clearance\_m - warning\_vertical\_clearance\_m

```



Interpretation:



\- A positive margin means the predicted clearance remains above the warning threshold.

\- A zero margin means the predicted clearance is exactly at the warning threshold.

\- A negative margin means the predicted clearance is below the warning threshold.

\- Safety margins shall remain signed values and shall not be clamped to zero.



\### 8.4 Months-to-threshold rules



`months\_to\_threshold` shall represent the earliest estimated time from the observation date until any configured warning threshold is reached.



Rules:



\- The search shall begin at month zero.

\- If the current record is already at or below a warning threshold, `months\_to\_threshold` shall be `0`.

\- Horizontal and vertical threshold times may be calculated directly from their linear growth rates.

\- The three-dimensional threshold time may be calculated analytically or with a deterministic bounded search.

\- The implementation method shall be documented and covered by tests.

\- The earliest valid threshold time among the three clearance dimensions shall be returned.

\- The result shall not exceed `threshold\_search\_max\_months`.

\- When no warning threshold is reached within the configured search window:

&#x20; - `months\_to\_threshold` shall be `null`;

&#x20; - `threshold\_dimension` shall be `null`;

&#x20; - `threshold\_reached\_within\_search\_window` shall be `false`.

\- When two or more dimensions reach a threshold at the same calculated time, the stable reporting priority shall be:

&#x20; 1. `3d`;

&#x20; 2. `horizontal`;

&#x20; 3. `vertical`.

\- All dimensions reaching the same risk level may still be included in the warning reasons.



\### 8.5 Risk-level severity rules



Risk-level severity shall follow this fixed order:



```text

safe < watch < warning < critical

```



Each clearance dimension shall be graded independently.



For each dimension:



\- clearance greater than `watch` threshold → `safe`;

\- clearance at or below `watch` threshold → `watch`;

\- clearance at or below `warning` threshold → `warning`;

\- clearance at or below `critical` threshold → `critical`.



Rules:



\- More severe threshold matches shall override less severe matches.

\- Equality shall count as reaching the corresponding threshold.

\- The final risk level shall be the most severe result among:

&#x20; - three-dimensional clearance;

&#x20; - horizontal clearance;

&#x20; - vertical clearance.

\- `current\_risk\_level` shall be calculated from current clearances.

\- `future\_risk\_level` shall be calculated from predicted clearances.

\- `is\_risk\_worsened` shall be `true` only when the future risk level is more severe than the current risk level.

\- Risk levels shall not be inferred from warning-message text.



\### 8.6 Warning-generation rules



`warning\_required` shall be `true` when at least one of the following is true:



\- the current risk level is `warning` or `critical`;

\- the future risk level is `warning` or `critical`;

\- a warning threshold is predicted to be reached within the configured search window;

\- the future risk level is more severe than the current risk level.



`warning\_reasons` shall use stable machine-readable reason codes.



Rules:



\- Reason codes shall not contain computer paths or uncontrolled free text.

\- Duplicate reason codes shall be removed.

\- Reason codes shall be stored in a stable deterministic order.

\- `warning\_message` shall explain the most important risk in human-readable language.

\- The message shall include the scenario and prediction horizon when the warning is based on a future result.

\- The message shall not claim regulatory non-compliance unless an approved official threshold configuration is explicitly used.

\- A suspect measurement shall always add `SUSPECT\_MEASUREMENT\_QUALITY`.

\- An invalid measurement shall not produce a normal warning result.



\### 8.7 Confidence rules



`scenario\_confidence` shall come directly from `GrowthScenarioConfig`.



`result\_confidence` shall consider:



\- scenario confidence;

\- measurement quality;

\- quality flags;

\- whether the calculation required fallback handling.



Version 1 rules:



\- `valid` measurement quality shall not reduce confidence by default.

\- `suspect` measurement quality shall reduce confidence using a documented deterministic factor.

\- `invalid` measurement quality shall not produce normal prediction results.

\- Confidence shall remain between `0.0` and `1.0`.

\- Confidence rules shall be configurable or centrally defined rather than duplicated across files.

\- Confidence shall represent confidence in the scenario result, not the probability that an accident will occur.

\- Version 1 shall not describe confidence as a trained statistical probability.



\### 8.8 Worst-case selection rules



For one input record, the prediction pipeline shall compare all scenario-and-horizon results.



The worst-case result shall be selected using the following priority:



1\. more severe `future\_risk\_level`;

2\. smaller `predicted\_clearance\_3d\_m`;

3\. smaller `predicted\_horizontal\_clearance\_m`;

4\. smaller `predicted\_vertical\_clearance\_m`;

5\. earlier `months\_to\_threshold`;

6\. longer prediction horizon;

7\. scenario order `fast`, `normal`, `slow`.



Rules:



\- Exactly one result shall normally have `is\_worst\_case=true` for each successfully processed input record.

\- Worst-case selection shall be deterministic.

\- Selection shall not depend on dictionary order or operating-system file order.

\- If no normal prediction result is produced, no result shall be marked as worst case.



\## 9. Pipeline requirements



The prediction pipeline shall accept:



\- one or more validated `PredictionInputRecord` objects;

\- one complete set of growth-scenario configurations;

\- one risk-threshold configuration.



The pipeline shall produce:



\- a collection of `PredictionResult` objects;

\- one `PredictionStatistics` object.



\### 9.1 Pipeline sequence



The required version 1 processing sequence is:



1\. validate the input record;

2\. inspect measurement quality;

3\. skip records that are not eligible for normal prediction;

4\. calculate current risk;

5\. validate the complete scenario configuration set;

6\. calculate growth for each scenario and horizon;

7\. calculate future clearances;

8\. calculate safety margins;

9\. calculate months to warning threshold;

10\. calculate future risk;

11\. generate warning reasons and warning message;

12\. calculate result confidence;

13\. select the worst-case result;

14\. calculate run statistics;

15\. return JSON-safe contract objects.



\### 9.2 Determinism



For identical inputs and identical configurations:



\- result values shall be identical;

\- result ordering shall be identical;

\- warning-reason ordering shall be identical;

\- worst-case selection shall be identical;

\- exported rows shall appear in a stable order.



The pipeline shall not depend on random numbers in version 1.



\### 9.3 Batch behavior



\- One invalid JSON object shall raise a contract-validation error before normal prediction begins for that object.

\- A structurally valid record with `measurement\_quality="invalid"` shall be skipped and recorded in statistics.

\- Other valid records in the same batch may continue to be processed.

\- Input order shall be preserved when producing grouped output.

\- Within each input record, result ordering shall be:

&#x20; 1. `slow`;

&#x20; 2. `normal`;

&#x20; 3. `fast`;

&#x20; and then by ascending prediction horizon.

\- Batch processing shall not require loading point-cloud data.



\## 10. Export requirements



P4 shall support:



\- JSON;

\- CSV;

\- Excel.



Export implementation is outside P0, but the contracts shall be designed to support these formats without field redesign.



\### 10.1 JSON export



JSON export shall support:



\- complete result records;

\- run statistics;

\- configuration snapshots;

\- schema and algorithm versions;

\- warnings and skipped-record identifiers.



Rules:



\- JSON shall use UTF-8 encoding.

\- Output shall reject `NaN` and infinity.

\- JSON keys shall use the public contract field names.

\- Dataclass objects shall be converted through their contract `to\_dict()` methods.

\- Export shall not include absolute local computer paths.

\- The default structure shall contain metadata, statistics, and a result list.



\### 10.2 CSV export



CSV export shall contain one row for each `PredictionResult`.



Rules:



\- Nested configurations shall be flattened using stable column names.

\- Lists such as warning reasons shall use a documented separator.

\- Numeric units shall remain visible in column names through suffixes such as `\_m` and `\_months`.

\- UTF-8 with BOM may be used to improve compatibility with Chinese spreadsheet software.

\- CSV ordering shall be deterministic.

\- Statistics may be written to a separate summary CSV file.



\### 10.3 Excel export



Excel export shall use an `.xlsx` workbook.



The workbook shall contain at least:



\- `Prediction Results`;

\- `Risk Summary`;

\- `Scenario Curves`;

\- `Configuration`;

\- `Warnings`.



Rules:



\- `Prediction Results` shall contain one row per result.

\- `Risk Summary` shall aggregate counts by risk level and scenario.

\- `Scenario Curves` shall provide data suitable for plotting clearance against prediction month.

\- `Configuration` shall preserve the growth and risk-threshold settings used.

\- `Warnings` shall contain warning records and skipped input records.

\- Excel generation shall not require Microsoft Excel to be installed.

\- Exported files shall not be committed to Git.



\## 11. Simulated fixtures



Before real main-computer clearance output is available, development shall use simulated JSON fixtures.



P0 shall provide:



\- `tests/fixtures/prediction/prediction\_input\_v1.json`;

\- `tests/fixtures/prediction/prediction\_config\_v1.json`.



\### 11.1 Input fixture requirements



The input fixture shall contain representative records covering:



\- valid and currently safe data;

\- valid data near the watch threshold;

\- valid data near the warning threshold;

\- valid data already at critical risk;

\- suspect measurement quality;

\- invalid measurement quality;

\- non-zero quality flags.



\### 11.2 Configuration fixture requirements



The configuration fixture shall contain:



\- one `slow` scenario;

\- one `normal` scenario;

\- one `fast` scenario;

\- prediction horizons `\[6, 12, 24]`;

\- demonstration risk thresholds for all three clearance dimensions;

\- a threshold search window.



The fixture shall clearly state that demonstration thresholds are not official standards.



\### 11.3 Fixture stability



\- Fixture field names shall match the public contracts exactly.

\- Fixtures shall be small enough for Git.

\- Fixtures shall not contain LAS, LAZ, NPZ, images, model weights, or private data.

\- Fixture identifiers shall be synthetic.

\- Changing fixtures shall require corresponding test updates and review.



\## 12. Testing requirements



Each development stage shall add focused tests without weakening existing tests.



\### 12.1 P0 contract tests



P0 tests shall verify:



\- default values;

\- valid object construction;

\- `to\_dict()` and `from\_dict()` round trips;

\- rejection of unknown fields;

\- rejection of missing required fields;

\- rejection of empty identifiers;

\- ISO 8601 observation-time validation;

\- finite-number validation;

\- rejection of Boolean values as numbers;

\- non-negative clearance validation;

\- valid measurement-quality values;

\- scenario-name validation;

\- confidence range validation;

\- prediction-horizon validation;

\- risk-threshold ordering;

\- statistics count consistency;

\- immutable warning and identifier sequences;

\- JSON serializability.



\### 12.2 P1 growth tests



P1 tests shall verify:



\- slow, normal, and fast scenario calculations;

\- 6-, 12-, and 24-month growth;

\- zero growth;

\- deterministic results;

\- cross-scenario ordering validation;

\- non-negative accumulated growth.



\### 12.3 P2 clearance tests



P2 tests shall verify:



\- horizontal-clearance calculation;

\- vertical-clearance calculation;

\- three-dimensional-clearance calculation;

\- zero-clamping behavior;

\- safety-margin calculation;

\- threshold time at month zero;

\- threshold reached later;

\- threshold not reached within the search window;

\- stable threshold-dimension selection.



\### 12.4 P3 risk tests



P3 tests shall verify:



\- all four risk levels;

\- equality at each threshold;

\- final selection of the most severe dimension;

\- current and future risk comparison;

\- warning-required behavior;

\- warning-reason generation;

\- suspect-measurement confidence reduction;

\- deterministic worst-case selection.



\### 12.5 P4 export tests



P4 tests shall verify:



\- JSON reload;

\- CSV column stability;

\- Excel workbook and sheet names;

\- exported result counts;

\- preservation of identifiers and configuration;

\- UTF-8 handling;

\- absence of absolute local paths;

\- no `NaN` or infinity in exported artifacts.



\### 12.6 Repository-wide checks



Before every prediction-stage commit, run:



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



Acceptance requires:



\- all existing tests pass;

\- all new prediction tests pass;

\- Ruff passes;

\- compilation passes;

\- `git diff --check` passes;

\- no prohibited files are modified;

\- no data or output artifacts are tracked.



\## 13. File ownership and modification boundaries



\### 13.1 Prediction-owned files



The prediction branch may add or modify:



\- `core/contracts/prediction.py`;

\- `algorithms/prediction/\_\_init\_\_.py`;

\- `algorithms/prediction/growth.py`;

\- `algorithms/prediction/risk.py`;

\- `algorithms/prediction/pipeline.py`;

\- `algorithms/prediction/io.py`;

\- `tests/contracts/test\_prediction\_contracts.py`;

\- `tests/prediction/`;

\- `tests/fixtures/prediction/`;

\- `reviews/P0-prediction-contract-review.md`;

\- later prediction review documents;

\- prediction-specific documentation under `docs/`.



\### 13.2 Prohibited files and directories



Prediction work shall not modify:



\- `algorithms/io/`;

\- `algorithms/preprocessing/`;

\- `algorithms/features/`;

\- `algorithms/wires/`;

\- `algorithms/vegetation/`;

\- `data/`;

\- `outputs/`.



During the first prediction stages, prediction work should also avoid modifying:



\- `core/contracts/\_\_init\_\_.py`.



This restriction reduces merge conflicts with the main computer's `feature/algorithm` branch.



\### 13.3 Path rules



\- Python code shall not hard-code `D:\\TreeLineWarning`.

\- Python code shall not hard-code `E:\\TreeLineWarning`.

\- Paths shall be supplied by arguments, configuration, or project-root helpers.

\- Public prediction contracts shall not require local file paths.

\- Export code shall create parent directories only when explicitly requested.

\- Existing files shall not be overwritten unless an explicit overwrite option is provided.



\## 14. Git and collaboration requirements



\- Prediction work shall remain on `feature/prediction`.

\- Codex sessions shall not commit directly to `main`.

\- Each stage shall have a focused commit.

\- Each stage shall be reviewed before beginning the next stage.

\- Do not repeatedly copy the entire project directory between computers.

\- Push prediction commits to GitHub.

\- The main computer shall use `git fetch` to inspect the prediction branch.

\- The main computer shall run tests before merging.

\- Merge conflicts shall be resolved through review, not by blindly accepting one side.

\- Large point-cloud files, generated outputs, model weights, and environments shall not be committed.



Recommended commit-message pattern:



```text

feat(prediction): add prediction contracts

feat(prediction): add scenario growth calculations

feat(prediction): add future clearance calculations

feat(prediction): add risk grading and warnings

feat(prediction): add result exporters

test(prediction): expand prediction integration coverage

docs(prediction): add stage review

```



\## 15. Main-computer integration requirements



The main computer shall eventually provide records matching `PredictionInputRecord`.



Integration shall not require the prediction computer to:



\- read LAS files;

\- run preprocessing;

\- detect conductors;

\- segment vegetation;

\- calculate current clearance again.



\### 15.1 Interface replacement rule



During development:



```text

simulated JSON → prediction pipeline

```



After the main computer provides real output:



```text

real clearance JSON → the same prediction pipeline

```



The prediction implementation shall not require rewriting when simulated records are replaced by real records with the same schema.



\### 15.2 Schema compatibility



\- Both computers shall use `schema\_version`.

\- P0 supports schema version `"1.0"`.

\- Unsupported schema versions shall raise a clear error.

\- New optional fields may be introduced only through a reviewed schema update.

\- Existing field meanings and units shall not change silently.

\- Meter units shall remain explicit through `\_m` suffixes.

\- Month units shall remain explicit through `\_months` suffixes.



\### 15.3 Integration review



Before integration:



\- the main computer shall provide representative real JSON output;

\- prediction contract validation shall be run against it;

\- field names, types, units, and identifiers shall be checked;

\- missing or additional fields shall be reviewed;

\- no compatibility adapter shall silently discard unknown safety-related fields.



\## 16. Version 1 non-goals



Version 1 shall not include:



\- LSTM;

\- attention mechanisms;

\- neural-network training;

\- PyTorch;

\- TensorFlow;

\- CUDA;

\- GPU inference;

\- model-weight downloads;

\- species-recognition models;

\- weather-data integration;

\- automatic seasonal adjustment;

\- point-cloud reading;

\- point-cloud preprocessing;

\- conductor detection;

\- conductor reconstruction;

\- vegetation segmentation;

\- current-clearance calculation;

\- Streamlit total-system integration;

\- official regulatory-threshold claims.



These items may be considered only after version 1 is validated and suitable historical data becomes available.



\## 17. Performance and resource requirements



Version 1 is a CPU-only module.



Requirements:



\- one small input record shall complete within normal interactive time;

\- batch processing of small JSON records shall not require GPU resources;

\- the module shall not allocate point-cloud-scale arrays;

\- memory usage shall remain proportional to the number of structured input records and generated result rows;

\- the implementation shall avoid unnecessary copies of complete result collections;

\- export operations may stream or write incrementally when practical;

\- performance statistics shall be recorded in `PredictionStatistics`.



No CUDA installation is required.



\## 18. Documentation and review requirements



Each stage shall create or update a review document.



Planned review documents:



\- `reviews/P0-prediction-contract-review.md`;

\- `reviews/P1-growth-review.md`;

\- `reviews/P2-clearance-review.md`;

\- `reviews/P3-risk-review.md`;

\- `reviews/P4-export-review.md`;

\- `reviews/P5-integration-review.md`.



Each review shall include:



\- branch and baseline commit;

\- scope;

\- modified files;

\- public contracts;

\- formulas and assumptions;

\- tests and code-check results;

\- known limitations;

\- Git protection checks;

\- recommendation to approve or reject the stage.



A stage shall not be described as complete only because the code runs once.



\## 19. Codex multi-session development rule



Prediction development shall be divided into isolated Codex sessions.



Each session shall receive:



\- one clearly defined stage;

\- an explicit allowed-file list;

\- an explicit prohibited-file list;

\- required tests;

\- acceptance criteria;

\- a required review output;

\- instructions not to merge into `main`.



Codex sessions shall inspect the current repository before editing.



Codex shall not:



\- invent missing requirements;

\- silently broaden the stage scope;

\- modify unrelated modules;

\- disable failing tests;

\- delete existing tests;

\- change Git history;

\- commit generated data;

\- install CUDA or deep-learning frameworks;

\- begin the next stage without approval.



A separate Codex session task document shall define the exact prompts and responsibilities for P0 through P5.



\## 20. P0 acceptance criteria



P0 shall be accepted only when:



\- `core/contracts/prediction.py` is implemented;

\- the five planned public models are defined:

&#x20; - `PredictionInputRecord`;

&#x20; - `GrowthScenarioConfig`;

&#x20; - `RiskThresholdConfig`;

&#x20; - `PredictionResult`;

&#x20; - `PredictionStatistics`;

\- all fields follow the approved requirements;

\- strict validation is implemented;

\- `to\_dict()` and `from\_dict()` are implemented;

\- simulated fixtures are added;

\- prediction contract tests pass;

\- all existing repository tests still pass;

\- Ruff and compilation checks pass;

\- `core/contracts/\_\_init\_\_.py` remains unchanged during the first P0 implementation;

\- prohibited directories remain unchanged;

\- the P0 review document records the final result;

\- no prediction algorithm beyond the contract scope is implemented.



\## 21. Approval gate



This document defines the approved target behavior for the first prediction and risk-warning implementation.



The next development action after requirements approval shall be:



```text

P0: implement and test prediction contracts only

```



Growth calculations, future-clearance calculations, risk grading, exports, and real-data integration shall remain separate reviewed stages.

