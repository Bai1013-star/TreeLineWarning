# Windows launcher diagnostics acceptance

The launcher now supports exclusive JSON diagnostics for a windowed EXE,
where console output may be unavailable. Diagnostic failures return a nonzero
process status, including missing project arguments and failed Survey/backend
probes. Existing diagnostic files are never replaced.

GUI launch checks the registry locations used by the selected pywebview 5.4
backend: WebView2 Runtime `pv` must be a valid version at least 86.0.622.0, and
the HKLM .NET release must satisfy 4.6.2. It also checks the packaged x64 loader
and asks its native API for the available browser version. These checks do not
import WinForms/MSHTML or install prerequisites. GUI failures display a native
message box; command-line probes remain noninteractive. Launch explicitly
requests EdgeChromium and records the actual renderer after the window closes.

The browser-version API and returned-memory ownership follow Microsoft's
[WebView2 Win32 API reference](https://learn.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/webview2-idl?view=webview2-1.0.2478.35).

The shipped offline installer is resolved relative to the package root, including
PyInstaller's `_internal` resource layout. A size check is reported separately
from a content-hash check; an unchecked hash is null, never a verified claim.
The application does not run the installer automatically.

## Supported probes

- `--dependency-health --diagnostic-output <new.json>`: native operations,
  frozen/resource/module origins, and loader/prerequisite status.
- `--survey-smoke --project-root <project> --diagnostic-output <new.json>`:
  read the modern finalized current report and persisted W1 curves, then stop.
- `--backend-smoke --project-root <project> --diagnostic-output <new.json>`:
  serve a local URL until the specified stop signal or finite timeout; publish
  a separate completion artifact. `--stop-signal` and
  `--smoke-timeout-seconds` control this acceptance mode (maximum 600 seconds).

Survey probes do not analyze, edit, finalize, or export. The legacy prediction
smoke remains separate and is not used as current Survey acceptance.

## Independent checks

Sol ran the launcher, desktop-shell, and current-report suites after final
corrections: **56 passed in 2.89 seconds**. Changed-file Ruff passed.
Coverage includes malformed/too-old/version-only registry values, registry
visibility, missing/stale native loader results, output exclusivity, native
error-dialog dispatch, nonzero failure exits, and bounded read-only probes.

A separate actual host probe passed without importing WinForms or MSHTML and
without opening a GUI: loader version **1.0.2957.106**, available browser
**151.0.4129.107**, and installed .NET **4.8.09221**. Evidence:

- `acceptance_evidence/final_delivery_20260830/sol_packaged_launcher_final_tests.log`
- `acceptance_evidence/final_delivery_20260830/sol_launcher_prerequisite_probe.json`

This is source/launcher acceptance. It is not a frozen EXE, visual GUI, or clean-PC
acceptance claim. A completed GUI lifecycle diagnostic explicitly leaves visual
acceptance UNVERIFIED until separately inspected.
