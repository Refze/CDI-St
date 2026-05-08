# Publishing CDI-ST: Complete Step-by-Step Guide

This guide walks you through everything to go from your current set of `.py`
files to **`pip install cdi-st`** working from PyPI, with `cdi-st` as a
terminal command.

---

## Prerequisites

- A GitHub account (username will be in the repo URL)
- A PyPI account at <https://pypi.org/account/register/>
- Optionally a TestPyPI account at <https://test.pypi.org/account/register/> (recommended for first try)
- Local Python ≥ 3.9 with `pip`
- `git` installed locally

---

## Phase 1 — Restructure the project

Your current code has flat files (`bcdi_gui.py`, `bcdi_core.py`, `nn_*.py`).
PyPI packaging requires a proper package directory.

### 1.1 Create the new directory layout

```
CDI-St/
├── .github/
│   └── workflows/
│       ├── publish.yml
│       └── ci.yml
├── src/
│   └── cdi_st/
│       ├── __init__.py
│       ├── __main__.py
│       ├── bcdi_gui.py             ← move here
│       ├── bcdi_core.py            ← move here
│       ├── nn_gui_tabs.py          ← move here
│       ├── nn_data_generator.py    ← move here
│       ├── nn_dataset.py           ← move here
│       ├── nn_phase_model.py       ← move here
│       ├── nn_autophase_model.py   ← move here
│       ├── nn_train.py             ← move here
│       ├── nn_autophase_train.py   ← move here
│       ├── nn_phase_retrieval.py   ← move here
│       ├── nn_autophase_infer.py   ← move here
│       ├── nn_experimental_loader.py ← move here
│       ├── nn_demo.py              ← move here
│       ├── nn_visualize.py         ← move here
│       └── CDI_ST_logo.png         ← move here
├── tests/
│   └── test_smoke.py               ← create (see below)
├── docs/
│   └── screenshots/
│       └── CDI_ST_logo.png         ← copy logo here too for README
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .gitignore
└── PUBLISHING.md                   ← this file
```

### 1.2 Convert internal imports to package-relative

Inside each `.py` file under `src/cdi_st/`, change all internal imports.
For example, `bcdi_gui.py` currently has lines like:

```python
from bcdi_core import MATERIAL_PRESETS, BCDIConfig, ...
from nn_gui_tabs import T_Gen, T4, T4_Sup, T5, T6
```

These need to become **explicit relative imports**:

```python
from .bcdi_core import MATERIAL_PRESETS, BCDIConfig, ...
from .nn_gui_tabs import T_Gen, T4, T4_Sup, T5, T6
```

The leading `.` means "relative to this package". Without it, Python looks
for a top-level `bcdi_core` module, which won't exist after packaging.

You can do this in one shot with `sed`:

```bash
cd src/cdi_st
for f in *.py; do
    sed -i.bak \
        -e 's/^from bcdi_core /from .bcdi_core /' \
        -e 's/^from nn_gui_tabs /from .nn_gui_tabs /' \
        -e 's/^from nn_data_generator /from .nn_data_generator /' \
        -e 's/^from nn_dataset /from .nn_dataset /' \
        -e 's/^from nn_phase_model /from .nn_phase_model /' \
        -e 's/^from nn_autophase_model /from .nn_autophase_model /' \
        -e 's/^from nn_phase_retrieval /from .nn_phase_retrieval /' \
        -e 's/^from nn_autophase_infer /from .nn_autophase_infer /' \
        -e 's/^from nn_experimental_loader /from .nn_experimental_loader /' \
        -e 's/^from nn_visualize /from .nn_visualize /' \
        -e 's/^import bcdi_core/from . import bcdi_core/' \
        -e 's/^import nn_gui_tabs/from . import nn_gui_tabs/' \
        "$f"
done
rm *.bak  # remove backup files after sanity-checking
```

Also update the `try: import nn_gui_tabs ... ` blocks similarly:

```python
# Before
try:
    from nn_gui_tabs import T_Gen, T4, T4_Sup, T5, T6
    _HAS_NN = True
except ImportError:
    _HAS_NN = False

# After
try:
    from .nn_gui_tabs import T_Gen, T4, T4_Sup, T5, T6
    _HAS_NN = True
except ImportError:
    _HAS_NN = False
```

Inside dynamic imports (e.g. inside worker `run()` methods that import lazily):

```python
# Before
from nn_data_generator import generate_single_sample

# After
from cdi_st.nn_data_generator import generate_single_sample
```

For dynamic imports inside functions, **absolute** imports are clearer than
relative — both work.

### 1.3 Verify the package installs locally

```bash
cd CDI-St
pip install -e .
cdi-st       # ← should launch the GUI
```

If `cdi-st` is not found, ensure your shell's PATH includes the Python `Scripts`
or `bin` directory. Re-open the terminal if needed.

If the GUI launches and the splash screen shows the logo correctly, you're done
with phase 1.

### 1.4 Add a smoke test

Create `tests/test_smoke.py`:

```python
"""Smoke tests — does the package even import?"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_version():
    import cdi_st
    assert cdi_st.__version__


def test_gui_imports():
    # Make sure the main entry point at least imports without crashing.
    from cdi_st import bcdi_gui
    assert hasattr(bcdi_gui, "main")
    assert hasattr(bcdi_gui, "Launcher")


def test_logo_present():
    from cdi_st import bcdi_gui
    p = bcdi_gui._logo_path()
    # Logo should be discoverable inside the installed package
    assert p is not None
    assert os.path.exists(p)
```

---

## Phase 2 — Create the GitHub repository

### 2.1 Create the repo on GitHub

1. Go to <https://github.com/new>
2. **Owner**: your username (or an org you control)
3. **Repository name**: `CDI-St` (with that exact case — but note GitHub URLs are case-insensitive)
4. **Description**: "Coherent Diffraction Imaging — Simulation Tools"
5. **Public** (for open source, MIT license)
6. **Initialize** with: nothing (we'll push from local)
7. Click **Create repository**

### 2.2 Push your code

In the project directory:

```bash
git init
git add .
git commit -m "Initial commit: CDI-ST v0.1.0 alpha"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CDI-St.git
git push -u origin main
```

### 2.3 Configure repository settings

Go to your repo on GitHub, then **Settings**:

- **General → Features**: enable Issues, Discussions
- **General → Pull Requests**: enable "Automatically delete head branches"
- **Pages** (optional): enable to publish docs from `/docs` later
- **Code security → Dependabot**: enable Dependabot security updates

Add a few topics on the main repo page (gear icon next to "About"):
`bcdi`, `xrd`, `coherent-diffraction-imaging`, `phase-retrieval`,
`neural-network`, `synchrotron`, `pyqt6`

Add a description: *"Coherent Diffraction Imaging — Simulation Tools.
End-to-end BCDI: crystal building, simulation, NN phase retrieval, 3D viewer."*

### 2.4 Add screenshots to README

Take a few screenshots of your GUI:
- `docs/screenshots/launcher.png`
- `docs/screenshots/material_tab.png`
- `docs/screenshots/results_simulation.png`
- `docs/screenshots/3d_viewer.png`

Then add them to README. Example:

```markdown
## Screenshots

<table>
  <tr>
    <td><img src="docs/screenshots/launcher.png" width="280"/></td>
    <td><img src="docs/screenshots/material_tab.png" width="280"/></td>
  </tr>
  <tr>
    <td>Launcher</td><td>Material tab</td>
  </tr>
</table>
```

---

## Phase 3 — Set up PyPI Trusted Publishing

This is the modern, secure way to upload to PyPI: no API tokens, GitHub Actions
authenticates via OpenID Connect.

### 3.1 Reserve the project name on PyPI

1. Go to <https://pypi.org/manage/account/publishing/> (sign in if needed)
2. Scroll to **Add a new pending publisher**
3. Fill in:
   - **PyPI Project Name**: `cdi-st`
   - **Owner**: your GitHub username (e.g. `Refze`)
   - **Repository name**: `CDI-St`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
4. Click **Add**

This creates a **pending publisher** — PyPI will trust this exact GitHub workflow
to publish a project named `cdi-st` once it tries.

### 3.2 Create the GitHub Actions environment

1. On GitHub, go to your repo → **Settings → Environments**
2. Click **New environment**
3. Name it `pypi`
4. (Optional but recommended) Add a **Required reviewers** rule with yourself
   as a reviewer — this means every PyPI publish requires your manual approval.

### 3.3 Confirm the workflow file

The workflow at `.github/workflows/publish.yml` should already be in your
repo (it's part of the template). Verify the values:

```yaml
environment:
  name: pypi
  url: https://pypi.org/p/cdi-st
permissions:
  id-token: write
```

These match the PyPI pending publisher exactly.

> **Important**: The values must match. If you renamed the workflow file or the
> PyPI environment, update both sides.

---

## Phase 4 — Test on TestPyPI first (recommended)

TestPyPI is a sandbox at <https://test.pypi.org>. You can publish, install,
and verify before touching production PyPI.

### 4.1 Add a TestPyPI pending publisher

Same as step 3.1, but at <https://test.pypi.org/manage/account/publishing/>,
with environment name `testpypi`.

### 4.2 Add a parallel workflow

Create `.github/workflows/publish-test.yml`:

```yaml
name: Publish to TestPyPI

on:
  workflow_dispatch:   # manual trigger only

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: python -m pip install --upgrade build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with: { name: dist, path: dist/ }

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: testpypi
      url: https://test.pypi.org/p/cdi-st
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
```

Trigger it via **Actions → Publish to TestPyPI → Run workflow**.

### 4.3 Install from TestPyPI to confirm

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            cdi-st
cdi-st
```

The `--extra-index-url` is needed because TestPyPI doesn't host all
dependencies (only your package); regular PyPI fills the rest.

If everything works, you're ready for production PyPI.

---

## Phase 5 — Cut the first release: v0.1.0 Alpha

### 5.1 Update version + changelog

Make sure `pyproject.toml` and `src/cdi_st/__init__.py` both have
`version = "0.1.0"` / `__version__ = "0.1.0"`.

Update `CHANGELOG.md` so the [0.1.0] entry is dated correctly.

### 5.2 Commit and push to main

```bash
git add pyproject.toml src/cdi_st/__init__.py CHANGELOG.md
git commit -m "Prepare v0.1.0 release"
git push origin main
```

### 5.3 Create the tag and GitHub release

Two equivalent ways. Either via web UI:

1. Go to your repo → **Releases → Draft a new release**
2. **Choose a tag** → type `v0.1.0` → "Create new tag on publish"
3. **Release title**: `Alpha v0.1.0`
4. **Description**: copy the [0.1.0] section from CHANGELOG.md
5. Check **Set as a pre-release** (this is alpha)
6. Click **Publish release**

Or via command line:

```bash
git tag -a v0.1.0 -m "Alpha v0.1.0 — initial public release"
git push origin v0.1.0
gh release create v0.1.0 --title "Alpha v0.1.0" --notes-file <(awk '/## \[0.1.0\]/,/## \[/{print}' CHANGELOG.md | head -n -1) --prerelease
```

### 5.4 Watch the workflow run

The tag push triggers the `publish.yml` workflow:

1. **build** job — downloads code, runs `python -m build`, makes wheel + sdist
2. **publish-pypi** job — waits for your approval (if you added the reviewer rule),
   then uploads to PyPI via OIDC trusted publishing

Go to **Actions → Publish to PyPI** to watch progress. Approve the deployment
when prompted.

### 5.5 Verify

Within a minute of successful publish:

```bash
pip install cdi-st
cdi-st
```

Visit <https://pypi.org/project/cdi-st/> — your project page should appear with
the README rendered, version `0.1.0`, "Pre-release" tag (because we used `0.1.0`
+ "Development Status :: 3 - Alpha" classifier).

🎉 You're live.

---

## Phase 6 — Maintenance and future releases

### Bumping versions

Use [Semantic Versioning](https://semver.org/):

- `0.1.1` — bug-fix release
- `0.2.0` — new features, backwards-compatible
- `1.0.0` — first stable release (drops "alpha", change classifier to "5 - Production/Stable")

Workflow for any new release:

```bash
# 1. Edit pyproject.toml + src/cdi_st/__init__.py with new version
# 2. Update CHANGELOG.md with the new section
# 3. Commit and push
git add . && git commit -m "Release v0.2.0" && git push

# 4. Tag and release
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0

# 5. Create GitHub release at https://github.com/YOU/CDI-St/releases/new
#    Choose tag v0.2.0, title "v0.2.0", paste changelog section.
#    Click Publish — Actions takes care of the rest.
```

### Pre-release versions (PEP 440)

If you want strict pre-release labels:
- Alpha: `0.2.0a1`, `0.2.0a2`
- Beta: `0.2.0b1`
- Release candidate: `0.2.0rc1`
- Dev: `0.2.0.dev1`

`pip install cdi-st` ignores pre-releases by default. Users must opt in with
`pip install --pre cdi-st`.

For your **first** release, `0.1.0` with the "Development Status :: 3 - Alpha"
classifier is the most user-friendly choice (`pip install cdi-st` just works,
and the alpha status is communicated via the classifier and the GitHub release
"pre-release" badge).

### Receiving feedback

You already have the in-app **Reports & Suggestions** button. For GitHub-driven
feedback:

- **Issues**: bugs and feature requests
- **Discussions**: open-ended Q&A (enable in Settings → Features)
- **Pull Requests**: contributions

Add an `.github/ISSUE_TEMPLATE/bug_report.yml` to standardize bug reports
(optional but appreciated).

### Citation

Once stable, get a DOI for citations via [Zenodo](https://zenodo.org/).
Connect your GitHub repo to Zenodo, and every release auto-archives with a DOI.
Update the README citation block with the DOI badge.

---

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `pip install cdi-st` says "no matching distribution" | Trusted Publishing failed silently, or version already exists | Check Actions tab for failed workflow; bump version |
| `cdi-st` command not found after install | Python `Scripts`/`bin` not in PATH | Re-open terminal; on Linux check `~/.local/bin` is in `$PATH` |
| Splash screen shows fallback text instead of logo | Logo not packaged | Verify `[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml` and that `CDI_ST_logo.png` is in `src/cdi_st/` |
| `ImportError: No module named 'bcdi_core'` | Imports not converted to relative | Run the `sed` commands in section 1.2 |
| GitHub Actions: `403 Forbidden` from PyPI | PyPI Trusted Publisher mismatch | Check workflow filename + environment name + PyPI publisher config all match |
| Linux: `Could not load the Qt platform plugin "xcb"` | Missing system Qt deps | `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1` |

---

## Summary checklist

- [ ] Move all `.py` files to `src/cdi_st/`
- [ ] Add `__init__.py` and `__main__.py`
- [ ] Convert internal imports to relative (`from . import ...`)
- [ ] Add `pyproject.toml`, `LICENSE`, `README.md`, `CHANGELOG.md`, `.gitignore`
- [ ] Add `.github/workflows/publish.yml` and `ci.yml`
- [ ] Verify `pip install -e .` and `cdi-st` work locally
- [ ] Push to GitHub
- [ ] Create PyPI pending publisher
- [ ] Create GitHub `pypi` environment
- [ ] (Optional) Test the flow on TestPyPI first
- [ ] Tag `v0.1.0` and create GitHub release titled "Alpha v0.1.0"
- [ ] Watch the workflow → approve the deployment → verify on PyPI
- [ ] Run `pip install cdi-st && cdi-st` on a clean environment to confirm

🎉 You're now an open-source maintainer.
