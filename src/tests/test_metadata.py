"""Comprehensive test suite for entity metadata functionality."""

import pytest
from entityframe import EntityFrame, Entity


# Basic Metadata Operations
def test_metadata_basic_python_types():
    """Test storing and retrieving basic Python types as metadata."""
    frame = EntityFrame()

    # Add a simple entity
    frame.add_method("test_method", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Test storing various Python types
    frame.set_entity_metadata("test_method", 0, "int_value", 42)
    frame.set_entity_metadata("test_method", 0, "float_value", 3.14159)
    frame.set_entity_metadata("test_method", 0, "str_value", "hello world")
    frame.set_entity_metadata("test_method", 0, "bool_value", True)
    frame.set_entity_metadata("test_method", 0, "bytes_value", b"binary data")

    # Retrieve and verify types
    assert frame.get_entity_metadata("test_method", 0, "int_value") == 42
    assert isinstance(frame.get_entity_metadata("test_method", 0, "int_value"), int)

    assert frame.get_entity_metadata("test_method", 0, "float_value") == 3.14159
    assert isinstance(frame.get_entity_metadata("test_method", 0, "float_value"), float)

    assert frame.get_entity_metadata("test_method", 0, "str_value") == "hello world"
    assert isinstance(frame.get_entity_metadata("test_method", 0, "str_value"), str)

    assert frame.get_entity_metadata("test_method", 0, "bool_value") is True
    assert isinstance(frame.get_entity_metadata("test_method", 0, "bool_value"), bool)

    assert frame.get_entity_metadata("test_method", 0, "bytes_value") == b"binary data"
    assert isinstance(frame.get_entity_metadata("test_method", 0, "bytes_value"), bytes)

    # Test None value
    assert frame.get_entity_metadata("test_method", 0, "nonexistent") is None


def test_metadata_complex_types():
    """Test storing complex Python types as metadata."""
    frame = EntityFrame()

    # Add entity
    frame.add_method("test", [{"customers": ["c1"]}])

    # Store complex types
    frame.set_entity_metadata("test", 0, "list_value", [1, 2, 3, "four"])
    frame.set_entity_metadata("test", 0, "dict_value", {"name": "Alice", "age": 30})
    frame.set_entity_metadata("test", 0, "tuple_value", (10, 20, 30))
    frame.set_entity_metadata("test", 0, "set_value", {1, 2, 3})

    # Retrieve and verify
    list_val = frame.get_entity_metadata("test", 0, "list_value")
    assert list_val == [1, 2, 3, "four"]
    assert isinstance(list_val, list)

    dict_val = frame.get_entity_metadata("test", 0, "dict_value")
    assert dict_val == {"name": "Alice", "age": 30}
    assert isinstance(dict_val, dict)

    tuple_val = frame.get_entity_metadata("test", 0, "tuple_value")
    assert tuple_val == (10, 20, 30)
    assert isinstance(tuple_val, tuple)

    set_val = frame.get_entity_metadata("test", 0, "set_value")
    assert set_val == {1, 2, 3}
    assert isinstance(set_val, set)


def test_metadata_custom_objects():
    """Test storing custom Python objects as metadata."""

    class CustomData:
        def __init__(self, value):
            self.value = value
            self.processed = False

        def process(self):
            self.processed = True
            return self.value * 2

    frame = EntityFrame()
    frame.add_method("test", [{"customers": ["c1"]}])

    # Store custom object
    custom_obj = CustomData(100)
    frame.set_entity_metadata("test", 0, "custom", custom_obj)

    # Retrieve and verify it's the same object
    retrieved = frame.get_entity_metadata("test", 0, "custom")
    assert isinstance(retrieved, CustomData)
    assert retrieved.value == 100
    assert retrieved.processed is False

    # Process the object
    result = retrieved.process()
    assert result == 200
    assert retrieved.processed is True


def test_metadata_overwrite():
    """Test overwriting metadata values with different types."""
    frame = EntityFrame()
    frame.add_method("test", [{"customers": ["c1"]}])

    # Set initial value as int
    frame.set_entity_metadata("test", 0, "value", 42)
    assert frame.get_entity_metadata("test", 0, "value") == 42

    # Overwrite with string
    frame.set_entity_metadata("test", 0, "value", "forty-two")
    assert frame.get_entity_metadata("test", 0, "value") == "forty-two"

    # Overwrite with list
    frame.set_entity_metadata("test", 0, "value", [4, 2])
    assert frame.get_entity_metadata("test", 0, "value") == [4, 2]

    # Overwrite with dict
    frame.set_entity_metadata("test", 0, "value", {"num": 42})
    assert frame.get_entity_metadata("test", 0, "value") == {"num": 42}


def test_large_metadata():
    """Test storing large metadata objects."""
    frame = EntityFrame()
    frame.add_method("test", [{"customers": ["c1"]}])

    # Create large metadata
    large_list = list(range(10000))
    large_dict = {f"key_{i}": i * 2 for i in range(1000)}
    large_string = "x" * 100000

    # Store large objects
    frame.set_entity_metadata("test", 0, "large_list", large_list)
    frame.set_entity_metadata("test", 0, "large_dict", large_dict)
    frame.set_entity_metadata("test", 0, "large_string", large_string)

    # Retrieve and verify
    assert frame.get_entity_metadata("test", 0, "large_list") == large_list
    assert frame.get_entity_metadata("test", 0, "large_dict") == large_dict
    assert frame.get_entity_metadata("test", 0, "large_string") == large_string


# EntityWrapper Tests
def test_entity_wrapper_metadata():
    """Test metadata through EntityWrapper with various types."""
    frame = EntityFrame()
    frame.add_method("test_method", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Create wrapper
    entity = Entity(frame, "test_method", 0)

    # Set various types of metadata
    entity.set_metadata("score", 0.95)
    entity.set_metadata("tags", ["important", "verified"])
    entity.set_metadata("config", {"threshold": 0.8, "enabled": True})
    entity.set_metadata("timestamp", "2024-01-01")
    entity.set_metadata("binary_data", b"abc123")

    # Retrieve and verify
    assert entity.get_metadata("score") == 0.95
    assert entity.get_metadata("tags") == ["important", "verified"]
    assert entity.get_metadata("config") == {"threshold": 0.8, "enabled": True}
    assert entity.get_metadata("timestamp") == "2024-01-01"
    assert entity.get_metadata("binary_data") == b"abc123"
    assert entity.get_metadata("missing") is None


# Pre-existing Metadata Tests
def test_preexisting_metadata():
    """Test that entities can be created with pre-existing metadata of various types."""
    frame = EntityFrame()

    # Test entities with pre-existing metadata of different types
    entities_with_metadata = [
        {
            "customers": ["c1", "c2"],
            "orders": ["o1"],
            "metadata": {
                "id": 12345,
                "score": 0.92,
                "source": "system_a",
                "tags": ["premium", "verified"],
                "active": True,
                "raw_data": b"binary_content",
            },
        },
        {
            "customers": ["c3"],
            "orders": ["o2", "o3"],
            "metadata": {
                "id": 67890,
                "attributes": {"color": "blue", "size": "large"},
                "values": [1.0, 2.0, 3.0],
            },
        },
    ]

    # Add entities with metadata
    frame.add_method("test_metadata", entities_with_metadata)

    # Test collection access
    collection = frame.test_metadata
    assert len(collection) == 2

    # Test first entity metadata
    entity_0 = collection[0]
    assert "metadata" in entity_0
    assert entity_0["metadata"]["id"] == 12345
    assert entity_0["metadata"]["score"] == 0.92
    assert entity_0["metadata"]["source"] == "system_a"
    assert entity_0["metadata"]["tags"] == ["premium", "verified"]
    assert entity_0["metadata"]["active"] is True
    assert entity_0["metadata"]["raw_data"] == b"binary_content"

    # Test second entity metadata
    entity_1 = collection[1]
    assert "metadata" in entity_1
    assert entity_1["metadata"]["id"] == 67890
    assert entity_1["metadata"]["attributes"] == {"color": "blue", "size": "large"}
    assert entity_1["metadata"]["values"] == [1.0, 2.0, 3.0]

    # Test that second entity doesn't have first entity's keys
    assert "source" not in entity_1["metadata"]
    assert "tags" not in entity_1["metadata"]


# Hashing Tests
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
    entity = Entity(frame, "test", 0)

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


def test_invalid_hash_algorithm():
    """Test error handling for invalid hash algorithm."""
    frame = EntityFrame()
    frame.add_method("test", [{"customers": ["c1"]}])

    with pytest.raises(Exception) as exc_info:
        frame.hash_entity("test", 0, "invalid_algorithm")

    assert "Unsupported hash algorithm" in str(exc_info.value)


# Combined Metadata and Hashing Tests
def test_metadata_with_hash():
    """Test storing entity hash as metadata."""
    frame = EntityFrame()

    # Add test entity
    frame.add_method("test", [{"customers": ["c1", "c2"], "orders": ["o1"]}])

    # Create wrapper
    entity = Entity(frame, "test", 0)

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


def test_mixed_metadata_and_hashing():
    """Test combining pre-existing metadata with new hash generation."""
    frame = EntityFrame()

    # One entity with pre-existing metadata, one without
    entities = [
        {
            "customers": ["c1", "c2"],
            "metadata": {"existing_score": 0.85, "source": "batch_001"},
        },
        {
            "customers": ["c3", "c4"]
            # No metadata
        },
    ]

    frame.add_method("mixed_test", entities)
    collection = frame.mixed_test

    # Add new hashes to the collection
    collection.add_hash("blake3")

    # First entity should have both existing metadata and new hash
    entity_0 = collection[0]
    assert entity_0["metadata"]["existing_score"] == 0.85
    assert entity_0["metadata"]["source"] == "batch_001"
    assert "hash" in entity_0["metadata"]  # New hash added
    assert len(entity_0["metadata"]["hash"]) == 32  # blake3 hash size

    # Second entity should only have the new hash
    entity_1 = collection[1]
    assert "existing_score" not in entity_1["metadata"]
    assert "source" not in entity_1["metadata"]
    assert "hash" in entity_1["metadata"]
    assert len(entity_1["metadata"]["hash"]) == 32

    # Verify hashes
    assert collection.verify_hashes("blake3") is True


def test_metadata_with_hashing():
    """Test that hashing works with mixed metadata types."""
    frame = EntityFrame()

    # Create entities with different types of metadata
    entities = [
        {
            "customers": ["c1", "c2"],
            "metadata": {"score": 0.95, "tags": ["important"], "active": True},
        },
        {
            "customers": ["c3", "c4"],
            # No metadata
        },
    ]

    frame.add_method("test", entities)
    collection = frame.test

    # Add hashes
    collection.add_hash("blake3")

    # Check that hashes were added and existing metadata preserved
    entity_0 = collection[0]
    assert entity_0["metadata"]["score"] == 0.95
    assert entity_0["metadata"]["tags"] == ["important"]
    assert entity_0["metadata"]["active"] is True
    assert "hash" in entity_0["metadata"]
    assert isinstance(entity_0["metadata"]["hash"], bytes)
    assert len(entity_0["metadata"]["hash"]) == 32  # blake3 hash size

    entity_1 = collection[1]
    assert "hash" in entity_1["metadata"]
    assert isinstance(entity_1["metadata"]["hash"], bytes)

    # Verify hashes
    assert collection.verify_hashes("blake3") is True
