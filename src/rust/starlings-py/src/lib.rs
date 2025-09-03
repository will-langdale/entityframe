use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyType};
use std::sync::Arc;

use starlings_core::{DataContext, Key, PartitionHierarchy, PartitionLevel};

/// A partition of records into entities at a specific threshold.
///
/// A Partition represents a snapshot of resolved entities at a specific threshold,
/// providing access to the resolved groups and their properties.
#[pyclass(name = "Partition")]
#[derive(Clone)]
pub struct PyPartition {
    partition: PartitionLevel,
}

#[pymethods]
impl PyPartition {
    /// Get the number of entities in this partition.
    ///
    /// Returns:
    ///     int: Number of entities in this partition.
    fn __len__(&self) -> usize {
        self.partition.entities().len()
    }

    /// Get entities as list of lists of record indices.
    ///
    /// Returns resolved entities as a list where each entity is represented
    /// as a list of record indices that belong to that entity.
    ///
    /// Returns:
    ///     List[List[int]]: List of entities, where each entity is a list of record indices.
    ///         
    /// Example:
    ///     ```python
    ///     partition = collection.at(0.8)
    ///     entities = partition.entities
    ///     # [[0, 1, 2], [3, 4], [5]]  # 3 entities
    ///     ```
    #[getter]
    fn entities(&self) -> Vec<Vec<u32>> {
        self.partition
            .entities()
            .iter()
            .map(|bitmap| bitmap.iter().collect())
            .collect()
    }

    /// Get the number of entities in this partition.
    ///
    /// Returns:
    ///     int: Number of entities in this partition.
    ///     
    /// Example:
    ///     ```python
    ///     partition = collection.at(0.8)
    ///     print(f"Found {partition.num_entities} entities")
    ///     ```
    #[getter]
    fn num_entities(&self) -> usize {
        self.partition.entities().len()
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!("Partition(entities={})", self.entities().len())
    }
}

/// Hierarchical partition structure that generates entities at any threshold.
///
/// A Collection stores the complete hierarchy of merge events, enabling instant
/// exploration of partitions at any threshold without recomputation. The first
/// query at a threshold reconstructs the partition (O(m)), while subsequent
/// queries use cached results (O(1)).
///
/// Key Features:
///     - Instant threshold exploration: O(1) cached partition access
///     - Memory efficient: Uses RoaringBitmaps for compact entity storage
///     - Type flexible: Handles int, str, bytes keys seamlessly
///     
/// Performance:
///     - Hierarchy construction: O(m log m) where m = edges
///     - First partition query: O(m) reconstruction  
///     - Cached partition query: O(1) from cache
#[pyclass(name = "Collection")]
#[derive(Clone)]
pub struct PyCollection {
    hierarchy: PartitionHierarchy,
}

#[pymethods]
impl PyCollection {
    /// Build collection from weighted edges.
    ///
    /// Creates a hierarchical partition structure from similarity edges between records.
    /// Records can be any hashable Python type (int, str, bytes) and are automatically
    /// converted to internal indices for efficient processing.
    ///
    /// Args:
    ///     edges (List[Tuple[Any, Any, float]]): List of (record_i, record_j, similarity) tuples.
    ///         Records can be any hashable type (int, str, bytes). Similarities should be
    ///         between 0.0 and 1.0.
    ///     source (str, optional): Source name for record context. Defaults to "default".
    ///
    /// Returns:
    ///     Collection: New Collection with hierarchy of merge events.
    ///
    /// Complexity:
    ///     O(m log m) where m = len(edges)
    ///
    /// Example:
    ///     ```python
    ///     # Basic usage with different key types
    ///     edges = [
    ///         ("cust_123", "cust_456", 0.95),
    ///         (123, 456, 0.85),
    ///         (b"hash1", b"hash2", 0.75)
    ///     ]
    ///     collection = Collection.from_edges(edges)
    ///     
    ///     # Get partition at threshold
    ///     partition = collection.at(0.8)
    ///     print(f"Entities: {len(partition.entities)}")
    ///     ```
    #[classmethod]
    #[pyo3(signature = (edges, *, source=None))]
    fn from_edges(
        _cls: &Bound<'_, PyType>,
        edges: Vec<(Py<PyAny>, Py<PyAny>, f64)>,
        source: Option<String>,
        py: Python,
    ) -> PyResult<Self> {
        #[cfg(debug_assertions)]
        let start_time = std::time::Instant::now();

        let source_name = source.unwrap_or_else(|| "default".to_string());
        let mut context = DataContext::new();
        let mut rust_edges = Vec::with_capacity(edges.len());

        // Pre-allocate space for records (estimate 2 unique records per edge on average)
        context.records.reserve(edges.len());

        // Efficiently convert all Python keys to Rust edges with optimised batching
        #[cfg(debug_assertions)]
        let conversion_start = std::time::Instant::now();

        // Pre-size the vectors for better performance
        rust_edges.reserve(edges.len());

        for (key1_obj, key2_obj, threshold) in edges {
            // Convert Python objects to Rust Keys (optimised for common cases)
            let key1 = python_obj_to_key_fast(key1_obj, py)?;
            let key2 = python_obj_to_key_fast(key2_obj, py)?;

            // Ensure records exist in context
            let id1 = context.ensure_record(&source_name, key1);
            let id2 = context.ensure_record(&source_name, key2);

            rust_edges.push((id1, id2, threshold));
        }
        #[cfg(debug_assertions)]
        let conversion_time = conversion_start.elapsed();

        #[cfg(debug_assertions)]
        let hierarchy_start = std::time::Instant::now();
        #[cfg(debug_assertions)]
        let edge_count = rust_edges.len();
        #[cfg(debug_assertions)]
        let record_count = context.len();
        let hierarchy = PartitionHierarchy::from_edges(rust_edges, Arc::new(context), 6);
        #[cfg(debug_assertions)]
        let hierarchy_time = hierarchy_start.elapsed();

        #[cfg(debug_assertions)]
        let total_time = start_time.elapsed();

        // Production-scale performance metrics (debug builds and large datasets only)
        #[cfg(debug_assertions)]
        if edge_count >= 1_000_000 {
            eprintln!("🏭 Production-scale Collection.from_edges performance:");
            eprintln!(
                "   📊 Scale: {} edges, {} records",
                edge_count, record_count
            );
            eprintln!("   ⚡ Python->Rust conversion: {:?}", conversion_time);
            eprintln!("   🏗️  Hierarchy construction: {:?}", hierarchy_time);
            eprintln!("   📈 Total time: {:?}", total_time);
            eprintln!(
                "   🎯 Edges per second: {:.0}",
                edge_count as f64 / total_time.as_secs_f64()
            );
            eprintln!(
                "   🏆 Target <10s: {}",
                if total_time.as_secs_f64() < 10.0 {
                    "✅ ACHIEVED"
                } else {
                    "❌ MISSED"
                }
            );
        }

        Ok(PyCollection { hierarchy })
    }

    /// Get partition at specific threshold.
    ///
    /// Returns a Partition containing all entities that exist at the specified
    /// similarity threshold. The first call at a threshold reconstructs the partition
    /// from merge events (O(m)), while subsequent calls use cached results (O(1)).
    ///
    /// Args:
    ///     threshold (float): Threshold value between 0.0 and 1.0. Records with
    ///         similarity >= threshold will be merged into the same entity.
    ///
    /// Returns:
    ///     Partition: Partition object with entities at the specified threshold.
    ///
    /// Complexity:
    ///     First call at threshold: O(m) reconstruction
    ///     Subsequent calls: O(1) from cache
    ///
    /// Example:
    ///     ```python
    ///     collection = Collection.from_edges(edges)
    ///     
    ///     # Get partition at different thresholds
    ///     partition_low = collection.at(0.5)   # More, smaller entities
    ///     partition_high = collection.at(0.9)  # Fewer, larger entities
    ///     
    ///     print(f"At 0.5: {len(partition_low.entities)} entities")
    ///     print(f"At 0.9: {len(partition_high.entities)} entities")
    ///     ```
    fn at(&mut self, threshold: f64) -> PyResult<PyPartition> {
        let partition = self.hierarchy.at_threshold(threshold);
        Ok(PyPartition {
            partition: partition.clone(),
        })
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        "Collection".to_string()
    }
}

/// Convert Python object to Rust Key (optimised for performance)
fn python_obj_to_key_fast(obj: Py<PyAny>, py: Python) -> PyResult<Key> {
    // Try integer types first (most common in large datasets)
    if let Ok(i) = obj.extract::<i64>(py) {
        if i < 0 {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Integer key must be non-negative and fit in u64",
            ))
        } else if i <= u32::MAX as i64 {
            Ok(Key::U32(i as u32))
        } else {
            Ok(Key::U64(i as u64))
        }
    } else if let Ok(s) = obj.downcast_bound::<PyString>(py) {
        Ok(Key::String(s.to_string()))
    } else if let Ok(b) = obj.downcast_bound::<PyBytes>(py) {
        Ok(Key::Bytes(b.as_bytes().to_vec()))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Key must be str, bytes, or int",
        ))
    }
}

/// Python module definition
#[pymodule]
fn starlings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCollection>()?;
    m.add_class::<PyPartition>()?;
    Ok(())
}
