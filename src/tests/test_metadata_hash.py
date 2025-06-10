"""Test metadata and hashing functionality."""

import pytest
from entityframe import EntityFrame, EntityWrapper


def test_entity_metadata_basic():
    """Test basic metadata operations on entities."""
    frame = EntityFrame()

    # Add a simple entity
    frame.add_method("test_method", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Set and get metadata
    frame.set_entity_metadata("test_method", 0, "source", b"test_source")
    frame.set_entity_metadata("test_method", 0, "confidence", b"0.95")

    # Retrieve metadata
    assert frame.get_entity_metadata("test_method", 0, "source") == b"test_source"
    assert frame.get_entity_metadata("test_method", 0, "confidence") == b"0.95"
    assert frame.get_entity_metadata("test_method", 0, "nonexistent") is None


def test_entity_wrapper_metadata():
    """Test metadata through EntityWrapper."""
    frame = EntityFrame()

    # Add a simple entity
    frame.add_method("test_method", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Create wrapper
    entity = EntityWrapper(frame, "test_method", 0)

    # Set and get metadata
    entity.set_metadata("hash", b"abc123")
    entity.set_metadata("timestamp", b"2024-01-01")

    assert entity.get_metadata("hash") == b"abc123"
    assert entity.get_metadata("timestamp") == b"2024-01-01"
    assert entity.get_metadata("missing") is None


def test_entity_hash_deterministic():
    """Test that entity hashing is deterministic."""
    frame1 = EntityFrame()
    frame2 = EntityFrame()

    # Add same data to both frames
    data = [{"customers": ["c1", "c2", "c3"], "orders": ["o1", "o2"]}]
    frame1.add_method("method1", data)
    frame2.add_method("method2", data)

    # Hashes should be identical
    hash1 = frame1.hash_entity("method1", 0)
    hash2 = frame2.hash_entity("method2", 0)
    assert hash1 == hash2

    # Different data should produce different hash
    frame1.add_method(
        "method3", [{"customers": ["c1", "c2"], "orders": ["o1", "o2", "o3"]}]
    )
    hash3 = frame1.hash_entity("method3", 0)
    assert hash3 != hash1


def test_entity_hash_algorithms():
    """Test different hash algorithms."""
    frame = EntityFrame()

    # Add test entity
    frame.add_method("test", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Test different algorithms
    sha256 = frame.hash_entity("test", 0, "sha256")
    sha512 = frame.hash_entity("test", 0, "sha512")
    sha3_256 = frame.hash_entity("test", 0, "sha3-256")
    blake3 = frame.hash_entity("test", 0, "blake3")

    # All should be bytes
    assert isinstance(sha256, bytes)
    assert isinstance(sha512, bytes)
    assert isinstance(sha3_256, bytes)
    assert isinstance(blake3, bytes)

    # Different lengths for different algorithms
    assert len(sha256) == 32  # SHA256 is 256 bits = 32 bytes
    assert len(sha512) == 64  # SHA512 is 512 bits = 64 bytes
    assert len(sha3_256) == 32  # SHA3-256 is 256 bits = 32 bytes
    assert len(blake3) == 32  # BLAKE3 default is 256 bits = 32 bytes

    # Different algorithms should produce different hashes
    assert sha256 != sha512
    assert sha256 != blake3


def test_entity_hash_order_independence():
    """Test that hash is independent of insertion order but dependent on content."""
    frame1 = EntityFrame()
    frame2 = EntityFrame()

    # Add same records in different order
    frame1.add_method(
        "test1", [{"customers": ["c3", "c1", "c2"], "orders": ["o2", "o1"]}]
    )

    frame2.add_method(
        "test2", [{"customers": ["c1", "c2", "c3"], "orders": ["o1", "o2"]}]
    )

    # Hashes should be identical (sorted by string value)
    hash1 = frame1.hash_entity("test1", 0)
    hash2 = frame2.hash_entity("test2", 0)
    assert hash1 == hash2


def test_entity_wrapper_hash():
    """Test hashing through EntityWrapper."""
    frame = EntityFrame()

    # Add test entity
    frame.add_method("test", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Create wrapper
    entity = EntityWrapper(frame, "test", 0)

    # Test hash methods
    hash_bytes = entity.hash()
    assert isinstance(hash_bytes, bytes)
    assert len(hash_bytes) == 32  # SHA256 default

    # Test hexdigest
    hex_str = entity.hexdigest()
    assert isinstance(hex_str, str)
    assert len(hex_str) == 64  # 32 bytes * 2 hex chars per byte
    assert hash_bytes.hex() == hex_str

    # Test with different algorithm
    blake3_hex = entity.hexdigest("blake3")
    assert isinstance(blake3_hex, str)
    assert blake3_hex != hex_str  # Different algorithm


def test_metadata_with_hash():
    """Test storing entity hash as metadata."""
    frame = EntityFrame()

    # Add test entity
    frame.add_method("test", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Create wrapper
    entity = EntityWrapper(frame, "test", 0)

    # Compute hash and store as metadata
    entity_hash = entity.hash()
    entity.set_metadata("sha256_hash", entity_hash)

    # Retrieve and verify
    stored_hash = entity.get_metadata("sha256_hash")
    assert stored_hash == entity_hash

    # Store different algorithm hashes
    entity.set_metadata("blake3_hash", entity.hash("blake3"))

    # Verify they're different
    assert entity.get_metadata("sha256_hash") != entity.get_metadata("blake3_hash")


def test_invalid_hash_algorithm():
    """Test error handling for invalid hash algorithm."""
    frame = EntityFrame()
    frame.add_method("test", [{"customers": ["c1"]}])

    with pytest.raises(Exception) as exc_info:
        frame.hash_entity("test", 0, "invalid_algorithm")

    assert "Unsupported hash algorithm" in str(exc_info.value)
