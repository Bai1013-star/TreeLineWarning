\# P0 Review: Prediction Input and Output Contracts



\## 1. Scope and baseline



\- Project: `TreeLineWarning`.

\- Branch: `feature/prediction`.

\- Baseline HEAD: `9e3a304 feat: add local PCA geometry features`.

\- This stage defines the public data contracts for vegetation growth prediction and risk warning.

\- The prediction computer receives only small structured JSON records. It does not read or process LAS point-cloud files.

\- Input records describe the current clearance between one vegetation object and one conductor.

\- The first version supports three explainable growth scenarios: `slow`, `normal`, and `fast`.

\- The initial prediction horizons are 6, 12, and 24 months.

\- This P0 stage is PLAN ONLY. It does not implement growth calculations, future-clearance calculations, risk grading, export functions, or user interfaces.

\- No CUDA, PyTorch, TensorFlow, Open3D, LSTM, attention mechanism, model weights, or neural-network training is included.

\- `core/contracts/\_\_init\_\_.py` will not be modified during the first P0 implementation, to reduce merge conflicts with the main computer's `feature/algorithm` branch.



\## 2. Planned files



P0 will later add the following files:



\- `core/contracts/prediction.py`

\- `tests/contracts/test\_prediction\_contracts.py`

\- `tests/fixtures/prediction/prediction\_input\_v1.json`

\- `tests/fixtures/prediction/prediction\_config\_v1.json`

\- `reviews/P0-prediction-contract-review.md`



P0 must not modify:



\- `algorithms/io/`

\- `algorithms/preprocessing/`

\- `algorithms/features/`

\- `algorithms/wires/`

\- `algorithms/vegetation/`

\- `data/`

\- `outputs/`

\- `core/contracts/\_\_init\_\_.py`

\## 3. Development ownership and Codex workflow



\- The prediction computer is responsible for vegetation growth prediction and risk warning.

\- The main computer remains responsible for point-cloud IO, preprocessing, geometry features, conductor reconstruction, vegetation regions, and current clearance calculation.

\- Prediction development will be completed through multiple isolated Codex sessions.

\- No prediction implementation may begin until the prediction requirements document and the Codex session task breakdown have been reviewed and approved.

\- Planned Codex sessions:

&#x20; - `P0`: prediction contracts and simulated JSON fixtures.

&#x20; - `P1`: slow, normal, and fast vegetation growth scenarios.

&#x20; - `P2`: future clearances, safety margins, and months to threshold.

&#x20; - `P3`: current and future risk grading, worst-case scenario, warning reasons, and warning messages.

&#x20; - `P4`: JSON, CSV, and Excel result export.

&#x20; - `P5`: tests, review, and integration with the main computer's real clearance output.

\- Each Codex session must have an isolated file scope, explicit acceptance criteria, required tests, and a review document.

\- Codex sessions must not merge directly into `main`. All work remains on `feature/prediction` until review and approval.

