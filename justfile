# Development commands for entityframe

# Install development dependencies
install:
    uv sync
    uv add --dev maturin pytest ruff mypy

# Build the Rust extension
build:
    uv run maturin develop

# Run Python tests
test:
    uv run pytest src/tests/

# Run Rust tests (Note: PyO3 tests require Python context)
test-rust:
    @echo "Note: Rust tests with PyO3 require Python context, use 'just test' instead"
    @echo "Running cargo check instead..."
    cd src/rust/entityframe && cargo check

# Run all tests
test-all: test test-rust

# Format Python code
fmt:
    uv run ruff format src/

# Lint Python code
lint:
    uv run ruff check src/
    uv run mypy src/python/

# Format Rust code
fmt-rust:
    cd src/rust/entityframe && cargo fmt

# Lint Rust code  
lint-rust:
    cd src/rust/entityframe && cargo clippy

# Format and lint all code
check: fmt lint fmt-rust lint-rust

# Clean build artifacts
clean:
    cd src/rust/entityframe && cargo clean
    rm -rf target/
    find src/ -name "*.pyc" -delete
    find src/ -name "__pycache__" -delete
