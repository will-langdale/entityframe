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

## Development Commands

This project uses `just` as a command runner and `uv` for Python dependency management:

- `just install`: Install all dependencies (Python + dev tools)
- `just build`: Build the Rust extension module
- `just test`: Run Python tests
- `just test-rust`: Run Rust checks (PyO3 tests need Python context)
- `just test-all`: Run both Python and Rust tests
- `just check`: Format and lint all code (Python + Rust)
- `just clean`: Remove all build artifacts

For individual operations:
- `just fmt` / `just fmt-rust`: Format Python/Rust code
- `just lint` / `just lint-rust`: Lint Python/Rust code

## Development Workflow

1. Install dependencies: `just install`
2. Build the project: `just build`
3. Run tests: `just test-all`
4. Check formatting/linting: `just check`

## Project Structure Notes

- All source code is contained within `src/` subdirectories
- Root directory contains only configuration files and documentation  
- Symmetrical naming: `src/python/entityframe/` and `src/rust/entityframe/`
- Follows UV's proven structure for Rust/Python hybrid projects
- Rust workspace allows for potential multiple crates (entityframe-core, entityframe-cli, etc.)
- Python package structure supports both pure Python and Rust extension modules