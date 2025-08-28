# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EntityFrame is a hybrid Python/Rust package for comparing entity resolutions from different processes. The project uses PyO3 bindings to combine Python's ease of use with Rust's performance for computationally intensive operations.

### Core Architecture

EntityFrame implements a three-layer architecture for high-performance entity evaluation:

1. **String Interning (Layer 1)**: Global string pool mapping record IDs to compact integers
2. **Roaring Bitmaps (Layer 2)**: Compressed bitmap sets optimised for set operations
3. **Entity Hashing (Layer 3)**: Deterministic hashing for fast entity identity and caching

### Key Components

- **StringInterner**: Maps string record IDs to integers for memory efficiency
- **Entity**: Core entity object containing dataset->record_id mappings using roaring bitmaps
- **EntityCollection**: High-level API for managing and comparing multiple entity resolution methods

### Performance Focus

All core operations are implemented in Rust for maximum performance:
- Set operations use SIMD-optimised roaring bitmaps
- String interning reduces memory usage by 10-100x
- Integer-based operations provide excellent cache locality

## Architecture

The project follows UV's proven structure with symmetrical organisation under `src/`:

```
src/
├── python/entityframe/    # Python package with main API
├── rust/entityframe/      # Rust crate with PyO3 bindings
│   ├── Cargo.toml        # Rust-specific configuration
│   └── src/lib.rs        # Rust implementation
└── tests/                # Python test suite
```

**Build system**: 
- Root `Cargo.toml`: Workspace configuration
- Root `pyproject.toml`: Python package config with maturin build backend
- `src/rust/entityframe/Cargo.toml`: Rust crate config with PyO3 dependencies

## Development commands

This project uses `just` as a command runner and `uv` for Python dependency management:

- `just install`: Install all dependencies (Python + dev tools)
- `just build`: Build the Rust extension module
- `just test`: Run quick tests (Python + Rust, excluding slow tests)
- `just test scale`: Run all tests including scale tests (Python + Rust)
- `just test python`: Run Python tests only (excluding slow tests)
- `just test rust`: Run Rust tests only
- `just format`: Format and lint all code (Python + Rust)
- `just clean`: Remove all build artifacts

## Testing strategy

### Test commands
- `just test`: Quick tests (Python + Rust, excluding slow tests) - use for development
- `just test scale`: All tests including scale tests - use before releases  
- `just test python`: Python tests only (excluding slow tests) - use for Python development
- `just test rust`: Rust tests only - use for Rust development

### Test categories

**Quick tests** (default):
- Functional tests that verify correctness
- Performance regression tests at smaller scales (1k-5k entities)
- Complete in under 10 seconds
- Run automatically in CI/development workflow

**Scale tests** (`just test scale`):
- Million-scale performance validation tests (`test_million_scale_performance`)
- Large-scale comparison benchmarks (`test_comparison_scaling_performance`)  
- Take several minutes to complete
- Run manually to validate production readiness

Use `just test` for regular development to get fast feedback. Use `just test scale` when validating performance at target scale.

There is a house style for parameterising Python unit tests.

```python
@pytest.mark.parametrize(
    ["foo", "bar"],
    [
        pytest.param(True, 12, id="test_thing"),
        pytest.param(False, 16, id="test_other_thing"),
    ],
)
def test_something(foo: bool, bar: int):
    """Tests that something does something."""
```

## Development workflow

1. Install dependencies: `just install`
2. Build the project: `just build`
3. Run quick tests: `just test`
4. Check formatting/linting: `just format`
5. Run scale tests before releases: `just test scale`

## Project structure notes

- All source code is contained within `src/` subdirectories
- Root directory contains only configuration files and documentation  
- Symmetrical naming: `src/python/entityframe/` and `src/rust/entityframe/`
- Follows UV's proven structure for Rust/Python hybrid projects
- Rust workspace allows for potential multiple crates (entityframe-core, entityframe-cli, etc.)
- Python package structure supports both pure Python and Rust extension modules

## Writing style guide

When writing documentation (README, comments, etc.):

- Use sentence case for headings: "How to use this" not "How To Use This"
- Add a line break after headings for readability
- Keep writing simple, clear and engaging
- Avoid overly technical jargon when possible
- Write for developers who want to get things done quickly
- Use British English spelling (organised, optimised, colour, etc.)
