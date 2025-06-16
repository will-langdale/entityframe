# Development commands for entityframe

# Install development dependencies
install:
    uv sync

# Build the Rust extension
build:
    uv run maturin develop

# Run Python tests (excluding slow tests)
test-python:
    uv run pytest -m "not slow"

# Run Rust tests (pure Rust unit tests)
test-rust: build
    #!/usr/bin/env bash
    export PYO3_PYTHON=$(uv run which python)
    cargo test --no-default-features

# Run tests - use "just test" for quick tests or "just test scale" for all tests
test *args="":
    @if [ "{{ args }}" = "scale" ]; then \
        echo "Running all tests including scale tests..."; \
        uv run pytest; \
        just test-rust; \
    elif [ -n "{{ args }}" ]; then \
        echo "Unknown test argument: {{ args }}"; \
        echo "Usage: just test [scale]"; \
        exit 1; \
    else \
        just test-python; \
        just test-rust; \
    fi

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
