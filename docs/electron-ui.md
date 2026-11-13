# Optional Electron desktop

The Electron UI uses the same trainer, TOMLs, defaults, and validation as the
PySide UI. It adds a dockable workspace styled after the Physics Workshop:
dark neutral surfaces, a lavender accent, compact controls, and shared dialog
styling. Its code and assets live in this repository; the Workshop checkout is
not needed to build or run it.

## Install and launch

Run the existing trainer installer first. Electron setup reuses its `venv/`
and does not create, replace, or reinstall the Python environment.

Install **Node.js 22.12 or newer**, including npm, then use:

| Platform | Set up / rebuild Electron | Launch |
|---|---|---|
| Windows | `setup-electron.bat` | `start-electron.bat` |
| Linux | `bash setup-electron.sh` | `bash start-electron.sh` |

Setup checks the Python backend, installs the locked Node dependencies with
`npm ci`, and builds the local renderer. Repeat setup after pulling UI changes.
The launcher checks the build fingerprint and explains when a rebuild is needed.
No development server is required.

For an existing alternative Python environment, set `BLUFLOW_PYTHON` to its
Python executable before running either script. It must already contain the
trainer's dependencies. Paths containing spaces are supported.

The familiar startup options work with the Electron launcher:

```text
start-electron.bat --config-dir "G:\My training configs"
bash start-electron.sh --config-dir "/home/me/training configs"
```

`--no-update-check` skips the Git update check; `--no-dependency-check` skips
Python package refresh checks. Approved updates use the current branch's upstream
and fast-forward-only Git updates. A branch without an upstream skips that check.
Package refresh and renderer rebuild happen after the running backend has closed,
then the launcher restarts the application. The existing PySide launch scripts
and `python -m trainer.gui` remain available.

## Workspace

- The left Configuration panel selects Dataset, Training, Optimizer, Method,
  and Experimental settings. Search finds labels, help text, and TOML keys across
  all visible categories.
- Settings and Live Metrics share the center. Run controls remain on the right;
  Console and Run Queue start below the center.
- Drag panel tabs to rearrange or group them. The Panels menu restores, floats,
  or hides individual panels, and resets the workspace. Layout and window size
  are remembered for Electron.
- Open, Save, Save As, Reload, and the preset picker work with existing TOMLs.
  Ctrl+O opens a config and Ctrl+S saves. Hover labels for the existing help text.
- Preferences retain the existing configuration folder, Autosave, experimental
  visibility, and Windows allocator settings. Active experimental features reveal
  their controls even when the visibility preference is off.

Operational preferences retain `mage-flow-trainer/gui.json` under the operating
system's generic configuration directory. Electron presentation state is separate:
`BluFlow-Trainer-Electron/workspace.json` under Electron's application-data directory.

## Feature coverage

| Workflow | Electron behavior |
|---|---|
| Configuration | All current GUI fields, labels, tooltips, custom tables, defaults, locks, advisories, and edit transitions share Python definitions. TOML-only values survive round trips. |
| Saving | Managed filenames follow the run name; external configs are copied; renamed runs preserve previous files; collisions ask before replacing another config. Autosave saves at Start, not on every edit. |
| Dataset preparation | Audit, latent cache, dry-run, subsets, resolution tiers, staged-resolution preview, and timestep distribution preview. |
| Training | GPU selection, Windows attention repair, allocator choice, INT8 calibration, toolchain preflight, cache-before-training, process-tree Stop, and close-while-running confirmation. |
| Queue | Independent TOML snapshots, duplicate-output protection, live pending-entry changes, retries, continue-after-error, and cancellation of dependent steps. Session-local queue contents are not restored after exit. |
| Metrics | Loss, EMA, MSE/HF, LR, step time, per-rank peak memory, elapsed time, stage details, Pause and Clear. Wheel to zoom, drag to pan, hover for values, and reset the view. Long runs retain the full timeline with local extrema. Successful queued runs export `metrics-final.png` even when Metrics is hidden. |
| Console | Live output, replaceable progress lines, follow mode, paged access to complete compressed history, copy the full log, and Clear. |
| Utilities | Native Electron dialogs for alpha/strength editing, fused/separated MOSS QKV, fixed-rank/sv_fro resizing, and TOML recovery. Operations run in independent Python workers. |
| Startup | Current-branch update check, dependency refresh, supervised rebuild/restart, and readable backend-failure reporting. |

Alpha editing protects its source and asks before replacing an existing output
copy. The other utilities create new files and reject existing destinations.
DoRA strength baking can use its training base model. Rank resizing retains the
original `sv_fro=0.99` default and optional rank cap.

## Development and verification

```text
npm --prefix electron-ui ci
node electron-ui/node_modules/electron/install.js
npm --prefix electron-ui run build
npm --prefix electron-ui test
npm --prefix electron-ui run test:startup
npm --prefix electron-ui run test:ui
venv\Scripts\python.exe -m trainer.electron_backend --check
venv\Scripts\python.exe -m pytest tests/test_electron_backend.py tests/test_electron_utilities.py
```

On Linux use `venv/bin/python`; run the UI smoke test under `xvfb-run -a` when
there is no display. All UI tests must use hidden, offscreen Electron windows.
The smoke harness exercises the real launcher and desktop workflows with isolated
preferences/configs and small synthetic data. Captures and test profiles go to
`.artifacts/electron-startup/` and `.artifacts/electron-smoke/`, which are ignored
by Git. It never starts GPU training.
`pytest` is a development prerequisite and is not installed by Electron setup.

The Windows/Linux CI matrix runs the headless backend and existing GUI checks,
utility checks, Node pipe tests, production build, and Electron workflow smoke.
Synthetic workers exercise queue failures and Stop without loading a model.
An actual training run still needs locally installed model weights and datasets.

The renderer calls a typed `window.bluflow` preload bridge. An allowlist in
Electron forwards requests to `python -m trainer.electron_backend` over stdio;
responses use `{id, result}` or `{id, error}`, and asynchronous messages use
`{event, ...payload}`. Shared services are in `trainer/ui_core/`. The backend
imports no Qt and does not load models or initialize CUDA. Configuration loading
does import Torch because the existing training dataclasses depend on it.

The custom `bluflow-app` protocol serves built local assets. Renderers use context
isolation and sandboxing with Node integration disabled. Native file dialogs and
process execution remain in Electron/Python.

## Troubleshooting

- **Missing trainer environment:** run the existing installer, or point
  `BLUFLOW_PYTHON` at the configured trainer interpreter.
- **Missing or stale UI build:** run the platform's Electron setup script.
- **Backend stopped:** inspect the terminal output, correct the Python environment,
  and relaunch. Pending UI operations report the failure instead of hanging.
- **Unusable saved layout:** use Panels → Reset workspace layout. If the app cannot
  render, rename Electron's `workspace.json` and relaunch. Training preferences
  and TOMLs are stored separately.

This version runs from the source checkout. It does not bundle Python/model
weights or produce a standalone application installer.
