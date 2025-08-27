# Module imports
mod test 'src/tests/justfile'

# Install development dependencies
install:
    uv sync

# Build the Rust extension
build:
    uv run maturin develop

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
