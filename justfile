# Development commands for entityframe

# Install development dependencies
install:
    uv sync

# Build the Rust extension
build:
    uv run maturin develop

# Run tests - use "just test" for all, "just test scale" for scale tests, "just test python/rust" for specific
test *args="":
    @if [ "{{ args }}" = "scale" ]; then \
        echo "Running all tests including scale tests..."; \
        uv run pytest; \
        just _test-rust-internal; \
    elif [ "{{ args }}" = "python" ]; then \
        echo "Running Python tests only (excluding slow tests)..."; \
        uv run pytest -m "not slow"; \
    elif [ "{{ args }}" = "rust" ]; then \
        echo "Running Rust tests only..."; \
        just _test-rust-internal; \
    elif [ -n "{{ args }}" ]; then \
        echo "Unknown test argument: {{ args }}"; \
        echo "Usage: just test [scale|python|rust]"; \
        exit 1; \
    else \
        echo "Running quick tests (Python + Rust, excluding slow tests)..."; \
        uv run pytest -m "not slow"; \
        just _test-rust-internal; \
    fi

# Internal Rust test runner (use just test rust instead)
_test-rust-internal: build
    #!/usr/bin/env bash
    export PYO3_PYTHON=$(uv run which python)
    cargo test --no-default-features

# Run only slow/scale tests
test-slow:
    uv run pytest -m "slow"

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

# Run a local documentation development server
docs:
    uv run mkdocs serve
