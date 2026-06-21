# Contributing to FlashEdge

Thank you for your interest in contributing to FlashEdge! This guide will help you get started.

---

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/FlashEdge.git
cd FlashEdge
```

### 2. Set Up Development Environment

```bash
bash setup_env.sh
# or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Maximum line length: 120 characters
- Follow existing code patterns and naming conventions
- Add type hints to all public functions

### Running Tests

```bash
pytest tests/ -v
```

### Running Linter

```bash
ruff check flashedge/
ruff format flashedge/
```

---

## Pull Request Process

1. Ensure all tests pass and linting is clean
2. Update documentation if adding new features
3. Add an entry to `CHANGELOG.md` under `[Unreleased]`
4. Write a clear PR description explaining the change
5. Request review from a maintainer

---

## What to Contribute

- **Bug fixes** — Always welcome!
- **New export backends** — TensorRT, NNAPI, Vulkan support
- **Optimization methods** — Mixed-precision, sparsity, quantization techniques
- **Runtime integrations** — Android NNAPI, iOS CoreML, Raspberry Pi
- **Documentation** — Tutorials, examples, API docs
- **Tests** — Improve coverage
- **Benchmarks** — Performance comparisons across devices

---

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## Questions?

Open an issue on GitHub or start a discussion. We're happy to help!
