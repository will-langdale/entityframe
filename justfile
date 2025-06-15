# Development commands for entityframe

# Install development dependencies
install:
    uv sync

# Build the Rust extension
build:
    uv run maturin develop

# Run Python tests
test-python:
    uv run pytest

# Run Rust tests (pure Rust unit tests)
test-rust: build
    #!/usr/bin/env bash
    export PYO3_PYTHON=$(uv run which python)
    cargo test --no-default-features

# Run all tests (Python + Rust)
test: test-python test-rust

# Format and lint all code (Python + Rust)
format:
    uvx ruff check src/ --fix
    uvx ruff format src/
    uvx mypy src/python/
    cargo fmt
    cargo clippy

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    find src/ -name "*.pyc" -delete
    find src/ -name "__pycache__" -delete
