"""Tests for hello world functionality."""

import entityframe


def test_hello_python():
    """Test Python hello function."""
    result = entityframe.hello_python()
    assert result == "Hello from Python!"


def test_hello_rust():
    """Test Rust hello function."""
    result = entityframe.hello_rust()
    assert result == "Hello from Rust!"


def test_hello_world():
    """Test combined hello function."""
    result = entityframe.hello_world()
    assert result == "Hello from Python! Hello from Rust!"