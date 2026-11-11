# Alpha release preparation: geneticevo 0.1.0a6

Prepared locally on 2026-11-11. Both `pyproject.toml` and
`src/geneticevo/__init__.py` declare `0.1.0a6`. Use the matching GitHub tag
`v0.1.0a6` and mark the GitHub release as a prerelease.

Short GitHub release description:

> Adds sparse graph brains with arbitrary connections and cycles, lifetime neuron/connection growth and pruning, and inherited topology across generations. Includes PPO learning, save/load, checkpoint resume, tests, and a runnable example.

## Changes in alpha 6

- New `GraphAgentFactory` and `GraphGrowth` support sparse directed brains,
  including recurrent/self connections and disconnected neurons.
- Neurons and connections can be added or removed during learning and at
  reproduction, within configurable bounds. Surviving genes retain their IDs.
- Both inheritance modes pass on acquired topology; Lamarckian inheritance also
  retains learned weights. Existing MLP behavior remains available.
- Native PPO supports graph brains on CPU/CUDA with `accelerated=False`.
  Portable saves, initial agents, archives, diversity metrics, structural lineage,
  and deterministic generation-boundary checkpoint resume include topology.
- New graph tests, documentation, and `examples/graph_brain.py` cover the feature.
  Cycles use bounded settling per observation; there is no cross-step memory.

## Build and publish alpha 6

Build and check only the new release artifacts in a separate output directory:

```powershell
python -m build --no-isolation --outdir dist/0.1.0a6
python -m twine check --strict dist/0.1.0a6/*
```

Commit and push all feature and release changes, including the new graph module,
tests, example, and documentation. Build outputs, virtual environments, and
`results/` are ignored. Publish a GitHub prerelease tagged `v0.1.0a6` at the
updated commit. The existing `.github/workflows/release.yml` verifies the tag
against both version declarations, tests and builds the tagged source, and
publishes to PyPI through the configured Trusted Publisher and `pypi` environment.
Publishing the GitHub release triggers that workflow; a push or tag alone does
not. The README supplies the PyPI project-page description in the uploaded
distribution metadata. No dependency or workflow version changes are needed.

Validation for the graph feature: **79 passed, 18 skipped** in the local full
suite; skipped cases require unavailable optional integrations or CUDA hardware.
Core-only graph checks also passed. Both documented CPU example variants ran.
These results do not establish CUDA execution for the new graph implementation;
older CUDA reports retain their original MLP validation context. Historical
benchmark metadata is not relabeled for this release.

## Historical alpha-5 release preparation

Prepared locally on 2026-11-06. Both `pyproject.toml` and
`src/geneticevo/__init__.py` declare `0.1.0a5`. Use the matching GitHub tag
`v0.1.0a5` and mark the GitHub release as a prerelease.

This release adds opt-in accelerated CPU/CUDA population PPO, fixed-capacity
masked neural growth, reproducible population chunks, and GPU benchmark/profile
tooling. [GPU correctness and measured results](accelerated_ppo_benchmarks_2026-11-05.md)
cover the implementation. Historical benchmark metadata retains its recorded
versions; it is not relabeled for this release.

Short GitHub release description:

> Opt-in accelerated CUDA PPO with batched neural growth, reproducible population chunking, GPU benchmarks, and validated checkpoint resume.

Build and check only the new release artifacts in a separate output directory:

```powershell
python -m build --no-isolation --outdir dist/0.1.0a5
python -m twine check --strict dist/0.1.0a5/*
```

Commit and push all additions, including new tests, documentation, benchmark
JSON, and the benchmark PNG. `results/`, virtual environments, and build outputs
are ignored and are not release source files. Publish a GitHub prerelease tagged
`v0.1.0a5` at that commit. The existing `release.yml` workflow tests and builds
that tagged source, then publishes its artifacts to PyPI using the configured
Trusted Publisher and GitHub `pypi` environment. A push or a tag alone does not
trigger this workflow; publishing the GitHub release does. No dependency-version
changes are required for these additions. Local preparation does not push a
commit, create a tag, publish a GitHub release, or upload to PyPI.

## Historical alpha-2 release preparation

The following notes preserve their original release and validation context.

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

## GPU validation status

CUDA execution and bounded performance measurements were validated on 2026-10-30
with an NVIDIA GeForce RTX 4060 Laptop GPU, driver 555.97, and PyTorch
2.12.1+cu126. This includes native and hybrid inference/PPO, discrete and
continuous actions, neural growth, checkpoint resume, and SB3 PPO/DQN. See
[CUDA validation](cuda_validation.md) for exact results and reproduction commands.

On machines without usable CUDA, hardware-only cases skip and CPU checks still
run. These measurements establish the tested configurations on this machine;
they do not establish general GPU speedups or coverage of every GPU/driver.
The historical alpha-1/alpha-2 results below describe their original runs.

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
