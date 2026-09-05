# Contributing to MIST

Thanks for your interest in improving MIST! This guide covers how to set up a
development environment and submit changes. For full documentation, see
[mist-medical.readthedocs.io](https://mist-medical.readthedocs.io/).

## Development setup

MIST requires Python >= 3.10.

```bash
git clone https://github.com/mist-medical/MIST.git
cd MIST
python -m pip install --upgrade pip
pip install -e ".[dev]" pytest-cov codespell
```

Installing with `-e .` pulls dependencies from `pyproject.toml`, so your
environment stays consistent with what CI and released builds use.

## Running the checks

Run the test suite under coverage:

```bash
pytest --cov=mist --cov-report=term-missing
```

Run the spell checker (configuration lives under `[tool.codespell]` in
`pyproject.toml`):

```bash
codespell
```

Both run automatically on pull requests via the **Run Tests and Generate
Coverage Badge** and **Lint** workflows. Please add or update tests for any
behavior you change.

## Submitting changes

1. Fork the repository and create a branch for your change.
2. Keep pull requests focused — small, single-purpose changes are easier to
   review and land faster.
3. Open a pull request against `mist-medical/MIST:main` and fill out the pull
   request template.
4. Make sure the test and lint workflows pass.

## Reporting bugs and requesting features

Use the issue templates (**Bug report** / **Feature request**). For bugs, please
include the install method, MIST version, and GPU/PyTorch details the template
asks for — these are essential for reproducing the problem.

For usage questions, please use
[Discussions](https://github.com/mist-medical/MIST/discussions) or the
documentation rather than opening an issue.
