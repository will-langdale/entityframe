# EntityFrame

**High-performance entity resolution evaluation for Python**

EntityFrame treats entities as what they truly are: collections of records across multiple datasets. Compare entity resolution methods efficiently using set operations rather than pairwise record comparisons.

## Quickstart

```python
import entityframe as ef

# Your entity resolution results from two different methods
splink_results = [
    {"customers": ["cust_001", "cust_002"], "transactions": ["txn_100"]},
    {"customers": ["cust_003"], "transactions": ["txn_101", "txn_102"]},
]

dedupe_results = [
    {"customers": ["cust_001"], "transactions": ["txn_100"]},
    {"customers": ["cust_002", "cust_003"], "transactions": ["txn_101", "txn_102"]},
]

# Create frame and declare datasets upfront for efficiency
frame = ef.EntityFrame()
frame.declare_dataset("customers")
frame.declare_dataset("transactions")

# Add both collections
frame.add_method("splink", splink_results)
frame.add_method("dedupe", dedupe_results)

# Calculate hashes for integrity and comparison
frame.splink.add_hash("blake3")
frame.dedupe.add_hash("blake3")

# Compare how well the methods agree
comparisons = frame.compare_collections("splink", "dedupe")
avg_similarity = sum(c['jaccard'] for c in comparisons) / len(comparisons)
print(f"Average agreement: {avg_similarity:.2f}")
```

## What is EntityFrame?

Traditional entity resolution evaluation compares pairs of records. EntityFrame compares **entities as complete objects** - collections of records that span multiple datasets.

Instead of asking "do these two records match?", EntityFrame asks "how well do these two methods agree on what constitutes this entity?"

## Core concepts

* `Entity`: A collection of record IDs across multiple datasets

```python
# Entity representing "Michael Smith"
{
    "customers": ["cust_001", "cust_045"],
    "transactions": ["txn_555", "txn_777"], 
    "addresses": ["addr_123"]
}
```

* `EntityCollection`: Entities from one method (like pandas Series)  
* `EntityFrame`: Multiple collections for comparison (like pandas DataFrame)

## Installation

```shell
# From PyPI (when available)
pip install entityframe

# From source
git clone https://github.com/yourusername/entityframe
cd entityframe
just install && just build
```

## Working with hashes and metadata

```python
import entityframe as ef

# Method 1: Conservative clustering
method1 = [
    {"customers": ["john_1"], "emails": ["john@email.com"]},
    {"customers": ["john_2"], "emails": ["j.smith@work.com"]},
]

# Method 2: Aggressive clustering  
method2 = [
    {"customers": ["john_1", "john_2"], "emails": ["john@email.com", "j.smith@work.com"]},
]

# Create frame and declare datasets upfront for efficiency
frame = ef.EntityFrame()
frame.declare_dataset("customers")
frame.declare_dataset("emails")

# Add the collections
frame.add_method("conservative", method1)
frame.add_method("aggressive", method2)

# Calculate hashes for entity integrity (batch processed for performance)
frame.conservative.add_hash("blake3")
frame.aggressive.add_hash("blake3")

# Access entities with metadata
entity = frame.conservative[0]
print(f"Entity: {entity}")
# {'metadata': {'hash': b'\x7a\x2f\x8c\x91...'}, 'customers': ['john_1'], 'emails': ['john@email.com']}

# Verify hash integrity later
if frame.conservative.verify_hashes("blake3"):
    print("All hashes verified successfully")

# See how methods differ
results = frame.compare_collections("conservative", "aggressive")
for result in results:
    print(f"Entity {result['entity_index']}: {result['jaccard']:.2f} similarity")
```

## Working with pre-existing metadata

```python
import entityframe as ef

# Entities with existing metadata (from previous processing)
existing_entities = [
    {
        "customers": ["alice_1", "alice_2"], 
        "emails": ["alice@personal.com", "a.johnson@company.com"],
        "metadata": {
            "id": 1,
            "hash": b"\x3b\x4c\x9d\xa2\x5f\x7e\x8b\x1c",
            "created_at": "2024-01-15T10:30:00Z",
            "source": "import_batch_001"
        }
    },
    {
        "customers": ["bob_smith"],
        "emails": ["bob.smith@email.org"],
        "metadata": {
            "id": 2,
            "hash": b"\x9e\x2a\x7f\x3c\x8d\x1b\x6e\x4a",
            "created_at": "2024-01-15T10:31:00Z", 
            "source": "import_batch_001"
        }
    }
]

# Create frame and add method with existing entities
frame = ef.EntityFrame()
frame.declare_dataset("customers")
frame.declare_dataset("emails")
frame.add_method("imported", existing_entities)

# Access entity with preserved metadata
entity = frame.imported[0]
print(f"ID: {entity['metadata']['id']}")
print(f"Hash: {entity['metadata']['hash'].hex()}")
print(f"Source: {entity['metadata']['source']}")

# Verify existing hashes match the data
if frame.imported.verify_hashes("blake3"):
    print("All imported hashes verified successfully")
```

## Working with individual entities

```python
# Create an entity manually
entity = ef.Entity()
entity.add_records("customers", [1, 2, 3])
entity.add_records("transactions", [100, 101])

# Query the entity
print(f"Total records: {entity.total_records()}")
print(f"Customer records: {entity.get_records('customers')}")
print(f"Has transactions: {entity.has_dataset('transactions')}")

# Compare two entities
other_entity = ef.Entity()
other_entity.add_records("customers", [2, 3, 4])
similarity = entity.jaccard_similarity(other_entity)
print(f"Jaccard similarity: {similarity:.2f}")
```

## Working with multiple methods

EntityFrame makes it simple to compare different entity resolution approaches:

```python
frame = ef.EntityFrame()

# Add results from different methods
frame.add_method("splink", splink_results)
frame.add_method("dedupe", dedupe_results) 
frame.add_method("custom_rules", rules_results)

# Compare any two methods
comparisons = frame.compare_collections("splink", "dedupe")
print(f"Methods agree on {len([c for c in comparisons if c['jaccard'] > 0.8])} entities")

# See what methods you have
print(f"Methods: {frame.get_collection_names()}")
print(f"Total entities per method: {frame.total_entities() // len(frame.get_collection_names())}")
```

## API overview

### `EntityFrame`

```python
frame = ef.EntityFrame()
frame = ef.EntityFrame.with_datasets(["ds1", "ds2"])  # Pre-declare datasets (optional)
frame.add_method(name, entity_data)              # Add method results 
frame.add_collection(name, collection)           # Add pre-built collection  
frame.compare_collections(name1, name2)          # Compare two methods
frame.get_collection(name)                       # Get specific collection
frame.get_collection_names()                     # List all methods
frame.total_entities()                           # Count total entities
frame.entity_has_dataset(method, index, name)    # Check if entity has dataset
```

### `EntityCollection`

```python
collection = ef.EntityCollection("method_name")
collection.get_entity(index)                     # Get entity by index
collection.get_entities()                        # Get all entities
collection.len()                                 # Number of entities
collection.compare_with(other_collection)        # Compare with another collection
collection.total_records()                       # Count all records
collection.is_empty()                            # Check if empty
```

### `Entity`

```python
entity = ef.Entity()
entity.add_record(dataset, record_id)           # Add single record
entity.add_records(dataset, record_ids)         # Add multiple records
entity.get_records(dataset)                     # Get records for dataset
entity.jaccard_similarity(other_entity)         # Compare entities
entity.total_records()                          # Count all records
```

## Data format

EntityFrame expects your entity resolution results as a list of dictionaries:

```python
entity_data = [
    {
        "dataset1": ["record_1", "record_2"],    # Records from dataset1
        "dataset2": ["record_10"],               # Records from dataset2
    },
    {
        "dataset1": ["record_3"],
        "dataset3": ["record_20", "record_21"],  # Different datasets per entity
    },
    # ... more entities
]
```

## Performance

EntityFrame is built for scale:

- **Memory efficient**: Handles millions of entities without breaking a sweat
- **Fast comparisons**: Optimised set operations for entity comparison
- **Rust powered**: High-performance core with Python convenience

Handle millions of entities efficiently.

## Development

```bash
just install    # Install dependencies
just build      # Build Rust extension  
just test       # Run all tests
just format     # Format code
```

## License

MIT License