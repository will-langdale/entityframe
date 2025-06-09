# Development commands for entityframe

# Install development dependencies
install:
    uv sync

# Build the Rust extension
build:
    uv run maturin develop

# Run Python tests
test-python:
    uv run pytest src/tests/

# Run Rust tests (pure Rust unit tests + cargo check)
test-rust:
    @echo "Running Rust unit tests..."
    cd src/rust/entityframe && uv run -- cargo test --no-default-features
    @echo "Running cargo check..."
    cd src/rust/entityframe && uv run -- cargo check

# Run all tests (Python + Rust)
test: test-python test-rust

# Format and lint all code (Python + Rust)
format:
    uv run ruff format src/
    uv run ruff check src/
    uv run mypy src/python/
    cd src/rust/entityframe && uv run -- cargo fmt
    cd src/rust/entityframe && uv run -- cargo clippy

# Clean build artifacts
clean:
    cd src/rust/entityframe && cargo clean
    rm -rf target/
    find src/ -name "*.pyc" -delete
    find src/ -name "__pycache__" -delete
