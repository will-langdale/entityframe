# Contributing to Starlings

Thank you for your interest in contributing to Starlings! This guide will help you understand the project's architecture, development practices, and contribution workflow.

## Getting started

Starlings is a hybrid Python/Rust package that revolutionises entity resolution by preserving complete resolution hierarchies rather than forcing threshold decisions. This enables instant exploration of any threshold and provides 10-100x performance improvements through incremental metric computation.

### Prerequisites

- **Python 3.10+** with [uv](https://github.com/astral-sh/uv) package manager
- **Rust 1.70+** with Cargo
- **just** command runner (`cargo install just`)

### Quick setup

```bash
git clone https://github.com/will-langdale/starlings
cd starlings
just install     # Install development dependencies
just build       # Build the Rust extension  
just test        # Run comprehensive test suite
```

## Architecture overview

Starlings uses a **professional two-crate architecture** following the Polars pattern, implementing the **"Rust Core with Python Bindings"** design:

```
src/
├── rust/
│   ├── starlings-core/     # Pure Rust business logic
│   │   ├── src/core/       # Data structures (DataContext, Key, etc.)
│   │   ├── src/hierarchy/  # Algorithms (PartitionHierarchy, etc.)
│   │   └── benches/        # Performance benchmarks
│   └── starlings-py/       # Minimal PyO3 wrapper
│       └── src/lib.rs      # Python bindings (PyCollection, PyPartition)
├── python/starlings/       # Python package
└── tests/                  # Python integration tests
```

### Why this architecture?

This design provides several critical benefits:

1. **Clean separation**: Business logic is completely independent of Python bindings
2. **Perfect test coverage**: Comprehensive Rust unit tests + Python integration tests
3. **No PyO3 linking issues**: Core crate has zero PyO3 dependencies
4. **Professional standards**: Matches industry patterns (Polars, PyTorch, NumPy)
5. **Future-proof**: Easy to add more language bindings or features
6. **Reusable core**: starlings-core can be used by pure Rust projects

### Architectural patterns

**starlings-core** (Pure Rust):
- Zero dependencies on Python or PyO3
- Contains all business logic, algorithms, and data structures  
- Comprehensive unit test coverage (33+ tests)
- Performance benchmarks and optimisations

**starlings-py** (PyO3 Wrapper):
- Thin wrapper over starlings-core
- Handles Python ↔ Rust type conversions
- Minimal logic, just marshalling data
- Uses standard `#[pymodule]` pattern (no auto-initialize)

## Development environment

The project uses modern development tools optimised for hybrid Rust/Python development:

### Core tools

- **uv**: Fast Python dependency management and virtual environments
- **maturin**: Python extension building via PyO3
- **just**: Command runner replacing complex Makefiles
- **ruff**: Python linting and formatting
- **cargo**: Rust building, testing, and formatting

### Development commands

```bash
just install     # Install all development dependencies
just build       # Build Python extension (maturin develop)
just test        # Run comprehensive test suite
just format      # Format and lint all code (Python + Rust)
just bench       # Run Rust performance benchmarks
just clean       # Clean all build artifacts
just docs        # Run local documentation server
```

### IDE setup

The project works well with:
- **VS Code** with rust-analyzer and Python extensions
- **PyCharm/IntelliJ** with Rust plugin
- **Vim/Neovim** with appropriate LSP configuration

All IDEs benefit from the clean two-crate architecture since core development doesn't require PyO3.

## Testing approach

Starlings uses a **two-layer testing strategy** that achieves comprehensive coverage whilst avoiding complex PyO3 test configuration:

### Layer 1: Pure Rust core tests

**Location**: `src/rust/starlings-core/src/**/*_test.rs` and `#[cfg(test)]` modules

**Coverage**: 33+ tests covering:
- Data structure operations and edge cases
- Algorithm correctness and performance
- Memory management and caching
- Error conditions and boundary cases

**Advantages**:
- Zero PyO3 dependencies = no linking issues
- Fast execution and reliable CI/CD
- Full coverage of business logic
- Easy to debug and maintain

**Run with**: `cargo test -p starlings-core`

### Layer 2: Python integration tests

**Location**: `src/tests/test_*.py`

**Coverage**: 17+ tests covering:
- Python → Rust → Python data flow
- PyO3 wrapper functionality and type conversions
- Error handling and Python exception mapping
- Real-world usage patterns and API contracts
- Performance validation of Python interface

**Advantages**:
- Tests the actual integration boundary
- Validates user-facing API behaviour
- Catches PyO3 wrapper bugs
- Realistic usage scenarios

**Run with**: `uv run pytest`

### Testing philosophy

This approach follows the principle of **"test what you own, integration test what you expose"**:

- **Rust tests**: Comprehensive coverage of all business logic we own
- **Python tests**: Integration testing of the interface we expose

This eliminates the common PyO3 testing problems (linking issues, complex configuration) whilst providing superior coverage compared to attempting to test PyO3 wrapper code in Rust.

### Performance testing

Benchmarks are located in `src/rust/starlings-core/benches/` and test:

- Hierarchy construction performance (1k-10k edges)
- Partition reconstruction scaling
- Caching effectiveness and memory usage
- Quantisation effects on performance

Run with: `just bench`

## Code standards

### Rust code standards

**File organisation**:
- Business logic in `starlings-core/src/`
- Module structure: `core/`, `hierarchy/`, tests inline
- Public API exports in `lib.rs`
- Benchmarks in dedicated `benches/` directory

**Code style**:
- Follow `rustfmt` defaults (runs via `just format`)
- Use `clippy` lints (runs via `just format`)
- Comprehensive documentation comments for public APIs
- Unit tests in `#[cfg(test)]` modules alongside implementation

**Performance considerations**:
- Use `RoaringBitmap` for entity sets (memory efficient)
- Leverage `rayon` for parallelisation where beneficial
- Cache frequently accessed partitions with LRU caching
- Profile with `criterion` benchmarks before optimising

**Error handling**:
- Use `Result<T, E>` for fallible operations
- Create specific error types for different failure modes
- Provide helpful error messages for debugging

### Python code standards

**File organisation**:
- Main API in `src/python/starlings/`
- Integration tests in `src/tests/`
- Follow flat structure for the MVP phase

**Code style**:
- Follow `ruff` configuration (runs via `just format`)
- Use type hints throughout (`mypy` configured)
- Google-style docstrings for public APIs
- pytest for testing with descriptive test names

**Testing patterns**:
```python
@pytest.mark.parametrize(
    ["foo", "bar"],
    [
        pytest.param(True, 12, id="test_specific_case"),
        pytest.param(False, 16, id="test_other_case"),
    ],
)
def test_something(foo: bool, bar: int):
    """Tests that something does something specific."""
```

### PyO3 wrapper patterns

**Conversion functions**:
- Handle Python type → Rust type conversions explicitly
- Provide clear error messages for invalid Python inputs
- Support standard Python types (int, str, bytes)

**Error handling**:
- Map Rust errors to appropriate Python exceptions
- Use `PyValueError` for invalid inputs, `PyTypeError` for wrong types
- Preserve error context when possible

**Memory management**:
- Use `Arc<T>` for shared ownership of large data structures
- Clone data strategically to balance performance and memory usage
- Leverage Rust's ownership system to prevent memory leaks

## Contribution workflow

### Making changes

1. **Fork and clone** the repository
2. **Create a feature branch** from `main`
3. **Set up development environment** with `just install`
4. **Make your changes** following the coding standards
5. **Test thoroughly**:
   ```bash
   just test        # Run full test suite
   just format      # Check code formatting  
   just bench       # Validate performance (if relevant)
   ```
6. **Commit with descriptive messages** following conventional commits
7. **Push to your fork** and create a pull request

### Pull request guidelines

**Before submitting**:
- [ ] All tests pass (`just test`)
- [ ] Code is formatted (`just format`)  
- [ ] New features include appropriate tests
- [ ] Documentation is updated for API changes
- [ ] Performance impact is considered and validated

**PR description should include**:
- **What**: Clear description of changes made
- **Why**: Rationale for the changes (links to issues if relevant)
- **Testing**: Description of testing performed
- **Performance**: Any performance implications noted

### Code review process

1. **Automated checks**: CI/CD runs tests and formatting checks
2. **Technical review**: Focus on correctness, performance, maintainability
3. **Architecture review**: Ensure changes align with project principles
4. **Final approval**: Changes merged after approval

## Performance guidelines

Starlings is designed for high-performance entity resolution. Contributors should consider:

### Performance principles

1. **Rust for hot paths**: Performance-critical code belongs in starlings-core
2. **Efficient data structures**: Use RoaringBitmap, avoid unnecessary allocations
3. **Caching strategies**: Leverage LRU caching for frequently accessed data
4. **Parallelisation**: Use rayon for CPU-bound operations on large datasets
5. **Memory efficiency**: Prefer sharing data with Arc over cloning

### Performance testing

- **Benchmark before optimising**: Use `just bench` to establish baselines
- **Profile with criterion**: All performance claims should be benchmarked
- **Test at scale**: Validate performance with realistic dataset sizes (1k-10k edges)
- **Memory profiling**: Monitor memory usage patterns for large datasets

### Common performance pitfalls

❌ **Avoid**:
- Unnecessary data copying between Rust and Python
- Creating temporary Python objects in tight loops
- Uncontrolled recursive algorithms without caching
- Excessive memory allocations in hot paths

✅ **Prefer**:
- Streaming processing where possible
- Batch operations over item-by-item processing  
- Cached results for expensive computations
- Zero-copy data sharing with Arc references

## Documentation standards

### Writing style

Following the project's established conventions:

- **Sentence case for headings**: "How to contribute" not "How To Contribute"
- **British English spelling**: organised, optimised, colour, etc.
- **Clear and engaging tone**: Write for developers who want to get things done
- **Line breaks after headings**: For improved readability
- **Avoid excessive jargon**: Keep technical language accessible

### Documentation types

**API documentation**:
- Comprehensive docstrings for public functions and classes
- Include usage examples for complex APIs
- Document parameter types, return values, and exceptions
- Cross-reference related functions where helpful

**Architecture documentation**:
- High-level design decisions and rationale
- Detailed algorithm explanations for complex operations
- Performance characteristics and trade-offs
- Integration patterns and best practices

**User guides**:
- Step-by-step tutorials for common workflows
- Real-world examples with realistic datasets
- Troubleshooting guides for common issues
- Migration guides for API changes

## Getting help

### Resources

- **Design documents**: `docs/design/` contains comprehensive technical specifications
- **API reference**: Generated automatically from docstrings
- **GitHub issues**: For bug reports and feature requests
- **GitHub discussions**: For questions and community support

### Common questions

**Q: Why the two-crate architecture?**
A: It provides clean separation, eliminates PyO3 testing complexities, and follows industry best practices from projects like Polars.

**Q: How do I add a new algorithm?**
A: Implement it in `starlings-core`, add comprehensive Rust tests, then add Python wrapper functions in `starlings-py` if needed.

**Q: Should I write Rust tests or Python tests?**
A: Write Rust tests for business logic (algorithms, data structures) and Python tests for integration scenarios (API usage, error handling).

**Q: How do I benchmark performance changes?**
A: Use `just bench` to run criterion benchmarks. Add new benchmarks in `starlings-core/benches/` for new functionality.

### Debugging tips

**Rust development**:
- Use `cargo test -p starlings-core` to test core functionality in isolation
- Add `println!` debugging or use `dbg!` macro for development
- Use `cargo clippy` for additional linting beyond basic compilation

**Python integration**:
- Test Python functionality with `uv run pytest -v` for detailed output
- Use `uv run python -c "import starlings; ..."` for quick interactive testing
- Check Python error messages for PyO3 conversion issues

**Performance issues**:
- Run `just bench` to identify performance regressions
- Use `cargo flamegraph` for detailed performance profiling
- Profile with realistic datasets, not toy examples

---

Thank you for contributing to Starlings! Your contributions help build better entity resolution tools for the entire community.