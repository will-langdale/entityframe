"""
EntityFrame - A Python package for comparing entity resolutions from different processes.
"""

from ._rust import hello_rust


def hello_python() -> str:
    """Return a hello message from Python."""
    return "Hello from Python!"


def hello_world() -> str:
    """Return combined hello messages from both Python and Rust."""
    return f"{hello_python()} {hello_rust()}"


__all__ = ["hello_python", "hello_rust", "hello_world"]