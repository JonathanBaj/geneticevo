# Alpha release preparation: geneticevo 0.1.0a2

This document records local preparation and validation for the next release.
At review on 2026-10-20, PyPI listed 0.1.0a1. Building or editing this checkout
does not publish 0.1.0a2.

## Release configuration

- Distribution/import: `geneticevo`; version: `0.1.0a2` in both `pyproject.toml`
  and the public `__version__` attribute.
- Status: alpha. Public APIs and checkpoint compatibility can change.
- Python: >=3.10 declared; only 3.11.9 on Windows has been exercised.
- Runtime core: NumPy >=1.26,<3 and Gymnasium >=1.2,<1.3. The resolver can select
  Python-compatible NumPy versions; the tested NumPy 2.4.4 requires Python >=3.11.
- Optional extras: `torch`, `sb3`, `shared`, `plot`, `dev`, `bench`, `release`.
  Optional upstream Python/platform constraints still apply. Their exact tested
  versions are listed in the README.
- Build: setuptools >=80,<82 and wheel >=0.45,<0.49. The setuptools ceiling also
  agrees with the installed PyTorch build's `setuptools<82` requirement.
- Explicit src package discovery includes `geneticevo` and its subpackages,
  with namespace discovery disabled.
- License: user-selected MIT, expressed as SPDX `MIT`; `LICENSE` is included
  in source/wheel metadata using `license-files`.
- README: Markdown long description; runnable local quickstart, source/wheel
  installation, feature matrix, limitations, and CPU/GPU requirements.
- Public metadata links point to the verified GitHub repository, README
  documentation, issue tracker, and release history. No author email is invented.

The metadata follows [PyPA's pyproject guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
and [SPDX license metadata guidance](https://packaging.python.org/en/latest/guides/licensing-examples-and-user-scenarios/).
The selected license text follows [GitHub's MIT template](https://choosealicense.com/licenses/mit/).

## Reproducible local checks

Install the development/release tools in a project environment:

```powershell
python -m pip install -e ".[dev,release]"
python -m pytest -q
python -m build --no-isolation
python -m twine check --strict dist/geneticevo-0.1.0a2*
```

Check the built wheel's name/version/Python/dependency/extra metadata, alpha
status, bundled license, and source contents. Install that exact wheel into an
environment with only NumPy/Gymnasium core dependencies, verify none of PyTorch,
SB3, PettingZoo, or Matplotlib is installed, run the core tests, and execute the
README quickstart plus vector/neural/custom-genome examples.

In the full environment, run all examples, including PPO, DQN, native PPO with
discrete and continuous actions, shared evolution/fixed-message learning, and
generalization/adaptation. A compatibility check should verify package metadata
and import versions agree. `pip check` should pass in both environments.

## GPU claim boundary

CUDA availability is checked before tests/examples. If CUDA is unavailable,
native tensor mathematics and PPO are tested on CPU, the conditional CUDA test
is skipped, and an explicit unavailable CUDA request must fail clearly.
This does not verify CUDA correctness, GPU driver compatibility, mixed-device
behavior, throughput, or speedups. The README describes the implemented CUDA
backend and explicitly records that GPU execution/performance has not been
validated here. A CUDA-equipped machine
must validate these before hardware-success or speedup claims.

## Historical alpha-1 validation

The earlier alpha-1 build was validated locally on 2026-10-20:

- Full environment: **43 passed, 1 CUDA test skipped**.
- Installed alpha wheel, core-only environment: **13 core tests passed**;
  PyTorch, SB3, PettingZoo, and Matplotlib are absent. The complete suite in this
  environment passed **24 applicable tests**, with **20 optional cases skipped**.
- All eleven documented example command variants passed: vector/neural/custom
  genome evolution, custom learning, generalization/adaptation, SB3 PPO and DQN,
  native categorical and continuous PPO on CPU, shared evolution, and
  fixed-message shared PPO.
- The exact README quickstart and vector/neural/custom-genome examples passed
  using the core-only installed wheel.
- Editable installation and source/wheel builds passed. `twine check --strict`
  passed for both alpha distributions.
- Archive inspection verified name/version, Python requirement, every runtime
  dependency/extra, alpha classifier, Markdown metadata, and exact MIT license
  contents in both distributions. Wheel modules match source files; results and
  virtual environments are excluded from the source archive.
- Import version and installed distribution version both equal `0.1.0a1`.
  `pip check` passed in the full and core-only environments.
- CUDA reported no available devices. CPU tensor/PPO checks passed; the
  advertised unavailable-device error path also passed. CUDA execution and
  performance remain **unverified**.

Historical timings in `benchmark_snapshot.json` retain the pre-alpha build
version they were measured with; packaging edits do not constitute new
performance evidence.

## GitHub release workflow

`.github/workflows/release.yml` runs on `release: published`, which includes
published prereleases. It checks out the event's tagged commit SHA, checks the
release tag against both package-version declarations, installs the full CPU
integration test dependencies, runs pytest, then builds and strictly checks
exactly one wheel and one source distribution. Accepted tag forms are the exact
version or that version prefixed by `v`, such as `v0.1.0a2`.

The build job uploads `python-distributions`. A separate `publish` job downloads
that same artifact and uses `pypa/gh-action-pypi-publish@release/v1`. Only that job
has `id-token: write`, and it uses the GitHub environment named `pypi`. No PyPI
API token or secret is referenced.

Before publishing a release, the PyPI Trusted Publisher must identify the correct
repository owner/name, workflow filename **release.yml**, and environment
**pypi**. Configure the GitHub `pypi` environment as appropriate for the project.
Commit and push the workflow before tagging the release; the tagged commit must
contain it. Publishing a new GitHub release or prerelease starts the workflow.

The workflow targets GitHub-hosted Ubuntu and Python 3.11. It exercises native
PPO on CPU; its conditional CUDA test is skipped on these runners. Local static
workflow validation does not prove GitHub/PyPI identity configuration or perform
an upload. See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [GitHub release events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release).


## Alpha 2 release repair

The published GitHub `v0.1.0a2` release points to a commit where `pyproject.toml`
says `0.1.0a2` but the package's `__version__` says `0.1.0a1`. Its release workflow
failed at **Confirm release tag matches package version** ([workflow run](https://github.com/JonathanBaj/geneticevo/actions/runs/39533466314)), before installation,
building, or publishing. The local repair aligns both versions at `0.1.0a2`,
updates the PyPI-facing README, and supplies project links in distribution metadata.

The existing `v0.1.0a2` tag still identifies the earlier commit. Re-running that
unchanged release will use the old code and fail again. Publishing these changes
requires a release whose tag identifies the corrected commit. Replacing an
existing published GitHub tag/release is a separate repository operation; a new
version/tag is an alternative. Neither operation is performed by this local fix.

The README and project links shown on PyPI come from uploaded distribution
metadata. Local edits and builds do not change the currently published page.


## Alpha 2 validation

Validated locally on 2026-10-20 after the repair:

- Both package-version declarations and installed metadata equal `0.1.0a2`.
  The workflow's actual version-check script accepts `v0.1.0a2`.
- Full suite: **43 passed, 1 CUDA case skipped**. CUDA remains inaccessible to
  PyTorch, including a check outside the sandbox; no GPU success is claimed.
- Isolated source/wheel builds and strict Twine checks passed. The wheel carries
  all five public project URLs, the PyPI-first README, and the MIT license.
- The basic `pip install geneticevo` command succeeded in a fresh environment,
  installing the currently published `0.1.0a1`. Upgrading that environment with
  the local `0.1.0a2` wheel passed the exact README quickstart from outside the
  repository, without PyTorch, SB3, PettingZoo, or Matplotlib. `pip check` passed.
- `dist/` contains only alpha-2 release files. Previous alpha-1 build artifacts
  are preserved under ignored `results/release/0.1.0a1/`.

These results validate the local repair. No commit, tag rewrite, remote push,
release edit, or PyPI upload is performed as part of it.
