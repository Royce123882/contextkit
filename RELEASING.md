# Releasing contextkit

Step-by-step guide for publishing contextkit to PyPI.

---

## First-Time Setup

Do this once before your first release.

### 1. Create accounts

- Test: https://test.pypi.org/account/register/
- Production: https://pypi.org/account/register/

### 2. Create API tokens

On both sites, go to Account Settings > API tokens. Create a token scoped to your project. For the first upload you may need a token scoped to all projects since the project does not exist yet.

### 3. Install publishing tools

```bash
pip install build twine
```

### 4. Set up Trusted Publishing (recommended)

Trusted Publishing lets GitHub Actions publish to PyPI without storing API tokens as secrets. To enable it:

1. Go to https://pypi.org/manage/project/contextkit/settings/publishing/
2. Add a new publisher:
   - Owner: `Royce123882`
   - Repository: `contextkit`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Save

After this, the GitHub Actions workflow can publish directly. No API token needed.

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "contextkit"
version = "0.1.0"
description = "A context engineering SDK for building reliable AI agents"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = [
    "context-engineering", "llm", "ai-agents",
    "prompt-engineering", "rag", "memory"
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "tiktoken>=0.7.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
sqlite = ["aiosqlite>=0.19.0"]
chroma = ["chromadb>=0.5.0"]
all = ["contextkit[sqlite,chroma]"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4.0",
    "mypy>=1.10",
]

[project.urls]
Homepage = "https://github.com/Royce123882/contextkit"
Documentation = "https://contextkit.readthedocs.io"
Repository = "https://github.com/Royce123882/contextkit"
Issues = "https://github.com/Royce123882/contextkit/issues"
Changelog = "https://github.com/Royce123882/contextkit/blob/main/CHANGELOG.md"

[tool.hatch.build.targets.wheel]
packages = ["src/contextkit"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.mypy]
python_version = "3.10"
strict = true
```

---

## GitHub Actions Workflow

Save this as `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  id-token: write

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

This uses Trusted Publishing. No API token secret needed once the publisher is linked (see First-Time Setup above).

---

## Per-Release Workflow

Run these steps for every release.

### 1. Update the version

Edit `pyproject.toml`:

```
version = "0.1.0"  ->  version = "0.2.0"
```

### 2. Update CHANGELOG.md

Move items from `[Unreleased]` into a new version heading.

### 3. Build distributions

```bash
python -m build
```

This creates two files in `dist/`:
- `contextkit-0.2.0.tar.gz` (source)
- `contextkit-0.2.0-py3-none-any.whl` (wheel)

### 4. Test on TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Verify it installs cleanly:

```bash
pip install --index-url https://test.pypi.org/simple/ contextkit
```

### 5. Publish to PyPI

```bash
twine upload dist/*
```

Or skip this step if using GitHub Actions -- just create a GitHub release and the workflow handles it.

### 6. Tag the release

```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## Pre-Launch Checklist

Run through this before every release.

- [ ] `pip install -e ".[dev]"` works cleanly
- [ ] All tests pass: `pytest`
- [ ] Linting passes: `ruff check src/`
- [ ] Type checking passes: `mypy src/`
- [ ] README renders correctly (preview on GitHub)
- [ ] License file present
- [ ] CHANGELOG.md updated
- [ ] `.gitignore` includes `dist/`, `*.egg-info/`, `__pycache__/`
- [ ] Package name available on PyPI (check https://pypi.org/project/contextkit/)
- [ ] TestPyPI upload succeeds
- [ ] Test install from TestPyPI works
- [ ] GitHub Actions workflow tested

---

## Naming Alternatives

If `contextkit` is taken on PyPI, consider these backups:

- `ctxeng` -- short, technical
- `contextcraft` -- memorable
- `contextpipe` -- emphasizes the pipeline metaphor

Check availability:

```bash
pip index versions contextkit
```

---

## Versioning Reference

| Version | Milestone              |
|---------|------------------------|
| 0.1.0   | Foundation             |
| 0.2.0   | Memory management      |
| 0.3.0   | Prompts & files        |
| 0.4.0   | Tools & RAG            |
| 0.5.0   | Context pipeline       |
| 0.6.0   | Multi-agent & scoping  |
| 1.0.0   | Stable API             |

Pre-1.0, breaking changes are expected between minor versions.
