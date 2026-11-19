# Releasing BBarolo

Development happens on `main`. A release is a tag `vX.Y.Z` with a GitHub Release; the workflow
`.github/workflows/release.yml` then builds the sdist and the wheel, tests the wheel (Linux with Python 3.10 and 3.13,
import on macOS and Windows) and publishes them on [PyPI](https://pypi.org/p/bbarolo). Read the Docs builds `stable`
from the latest tag and `latest` from `main`.

Versions follow [semantic versioning](https://semver.org): `2.0.1` for fixes, `2.1.0` for new features and keywords,
`3.0.0` for changes that break parameter files or the Python interface. Release candidates are `2.1.0rc1`, ...

## One-time setup

1. On [pypi.org](https://pypi.org/manage/account/publishing/) and [test.pypi.org](https://test.pypi.org/manage/account/publishing/),
   add a *pending trusted publisher*: project `bbarolo`, owner `editeodoro`, repository `bbarolo2`, workflow
   `release.yml`, environment `pypi` (on test.pypi.org: `testpypi`).
2. On GitHub, Settings → Environments: create `pypi` and `testpypi` (on `pypi`, optionally require your approval).
3. Optionally, enable the repository on [Zenodo](https://zenodo.org/account/settings/github/): every GitHub Release
   then gets a DOI.

## Making a release

1. Run the tests and `performance/compare_examples.py`.
2. Set the version in `bbarolo/_version.py` (e.g. `2.0.0`), and in `CHANGES.md` rename "Unreleased" to
   `2.0.0 (YYYY-MM-DD)`, with a new empty "Unreleased" above it. Commit and push.
3. Optional, for a release candidate: Actions → Release → *Run workflow* publishes the current `main` on TestPyPI.
   Check it in a clean environment:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bbarolo==2.0.0rc1
   ```
   TestPyPI, like PyPI, refuses a version it already has: bump `rc1` → `rc2` to upload again.
4. On GitHub, Releases → *Draft a new release*: tag `v2.0.0` (created on `main`), title `BBarolo 2.0.0`, notes from
   `CHANGES.md`. *Publish release* starts the workflow; it stops if the tag and `_version.py` differ.
5. After the release, set `_version.py` to the next development version (e.g. `2.0.1.dev0`).
