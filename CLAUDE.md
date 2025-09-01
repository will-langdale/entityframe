# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

starlings is a hybrid Python/Rust package for systematically exploring and comparing entity resolution results across different thresholds and methods. Instead of forcing threshold decisions at processing time, Starlings preserves the complete resolution space as a hierarchy of merge events, enabling instant threshold exploration, efficient metric computation, and lossless data transport between pipeline stages.

### Core Innovation

Starlings revolutionises entity resolution by storing **merge events** rather than fixed clusters, enabling instant exploration of any threshold without recomputation. This achieves 10-100x performance improvements through O(k) incremental metric updates.

### Key Technical Architecture

**Multi-Collection Model**:
- EntityFrame = (Records, {Hierarchies}, Interning)
- Collections ARE hierarchies that generate partitions at any threshold
- Contextual ownership enables memory sharing between collections

**Performance Characteristics**:
- Hierarchy construction: O(m log m) where m = edges
- Threshold query: O(m) first time, O(1) cached
- Metric updates: O(k) incremental between thresholds
- Memory: ~60-115MB for 1M edges

**Implementation Stack**:
- **Rust core**: Performance-critical operations using RoaringBitmaps, Rayon parallelisation
- **Python interface**: Polars-inspired expression API via PyO3
- **Arrow integration**: Efficient serialisation with dictionary encoding

## Architecture

The project follows UV's proven structure with symmetrical organisation under `src/`:

```
src/
├── python/starlings/    # Python package with main API
├── rust/starlings/      # Rust crate with PyO3 bindings
│   ├── Cargo.toml        # Rust-specific configuration
│   └── src/lib.rs        # Rust implementation
└── tests/                # Python test suite
```

**Build system**: 
- Root `Cargo.toml`: Workspace configuration
- Root `pyproject.toml`: Python package config with maturin build backend
- `src/rust/starlings/Cargo.toml`: Rust crate config with PyO3 dependencies

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
- `just docs`: Run local documentation development server

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

There is a house style for parameterising Python unit tests:

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
- Symmetrical naming: `src/python/starlings/` and `src/rust/starlings/`
- Follows UV's proven structure for Rust/Python hybrid projects
- Rust workspace allows for potential multiple crates
- Python package structure supports both pure Python and Rust extension modules

## Design Documentation

Comprehensive design documents are available in `docs/design/`:

- **overview.md**: High-level system overview and capabilities
- **principles.md**: Mathematical foundations and theoretical guarantees
- **algorithms.md**: Core algorithms and data structures
- **engine.md**: Rust implementation details
- **interface.md**: Python API specification
- **roadmap.md**: Detailed implementation plan with specific tasks

These documents provide the complete technical specification for implementing Starlings from scratch.

## API Design (Target)

The target API follows Polars-inspired expression syntax:

```python
import starlings as sl

# Create frame and add collections
ef = sl.from_records("source", df)
ef.add_collection_from_edges("splink", edges)

# Analyse with expressions
results = ef.analyse(
    sl.col("splink").sweep(0.5, 0.95, 0.01),
    sl.col("truth").at(1.0),
    metrics=[sl.Metrics.eval.f1]
)
# Returns List[Dict]: uniform format for all operations
```

## Writing style guide

When writing documentation (README, comments, etc.):

- Use sentence case for headings: "How to use this" not "How To Use This"
- Add a line break after headings for readability
- Keep writing simple, clear and engaging
- Avoid overly technical jargon when possible
- Write for developers who want to get things done quickly
- Use British English spelling (organised, optimised, colour, etc.)