"""
Performance benchmarks and regression tests for EntityFrame.

This package contains performance and scaling tests that verify:
- Performance regression detection
- Algorithm comparison benchmarks
- Scaling law validation
- Memory efficiency verification

These tests are separated from core functionality tests to allow:
- Fast CI runs (core tests only)
- Optional performance validation
- Clear separation of concerns
"""
