use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyType};
use std::collections::HashMap;
use std::sync::Arc;

use starlings_core::test_utils::{GraphConfig, ThresholdConfig};
use starlings_core::{DataContext, Key, PartitionHierarchy, PartitionLevel};

/// Type alias for graph generation result to reduce complexity
type GraphResult = PyResult<(Vec<(i64, i64, f64)>, usize)>;
// use starlings_core::test_utils::{GraphConfig, ThresholdConfig, generate_hierarchical_graph as generate_graph};

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
        let context = DataContext::new();
        let mut rust_edges = Vec::with_capacity(edges.len());

        // Efficiently convert all Python keys to Rust edges with optimised bulk processing
        #[cfg(debug_assertions)]
        let conversion_start = std::time::Instant::now();

        // Phase 1: Bulk Python object extraction with deduplication
        let mut key_to_id: HashMap<Key, u32> = HashMap::new();
        let mut extracted_edges = Vec::with_capacity(edges.len());

        // Pre-allocate estimated capacity based on edge count (assume ~70% unique keys)
        key_to_id.reserve(edges.len().saturating_mul(7) / 10);
        context.reserve(edges.len().saturating_mul(7) / 10);

        for (key1_obj, key2_obj, threshold) in edges {
            // Convert Python objects to Rust Keys (bulk extraction)
            let key1 = python_obj_to_key_fast(key1_obj, py)?;
            let key2 = python_obj_to_key_fast(key2_obj, py)?;

            extracted_edges.push((key1, key2, threshold));
        }

        // Phase 2: Batch key registration with deduplication
        for (key1, key2, _) in &extracted_edges {
            if !key_to_id.contains_key(key1) {
                let id = context.ensure_record(&source_name, key1.clone());
                key_to_id.insert(key1.clone(), id);
            }
            if !key_to_id.contains_key(key2) {
                let id = context.ensure_record(&source_name, key2.clone());
                key_to_id.insert(key2.clone(), id);
            }
        }

        // Phase 3: Build final edge list with ID lookups
        for (key1, key2, threshold) in extracted_edges {
            let id1 = key_to_id[&key1];
            let id2 = key_to_id[&key2];
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

/// Configuration for generating realistic entity resolution graphs.
///
/// This mirrors production entity resolution patterns with hierarchical threshold structures
/// and realistic component distributions for benchmarking and testing.
#[pyclass(name = "GraphConfig")]
#[derive(Clone)]
pub struct PyGraphConfig {
    config: GraphConfig,
}

#[pymethods]
impl PyGraphConfig {
    /// Create a new graph configuration.
    ///
    /// Args:
    ///     n_left (int): Number of left-side records (e.g., customers)
    ///     n_right (int): Number of right-side records (e.g., transactions)
    ///     n_isolates (int): Number of isolated records (no edges)
    ///     thresholds (List[Tuple[float, int]]): List of (threshold, target_entities) pairs
    ///
    /// Returns:
    ///     GraphConfig: New graph configuration
    ///
    /// Example:
    ///     ```python
    ///     config = GraphConfig(1000, 1000, 0, [(0.9, 500), (0.7, 300)])
    ///     ```
    #[new]
    fn new(
        n_left: usize,
        n_right: usize,
        n_isolates: usize,
        thresholds: Vec<(f64, usize)>,
    ) -> Self {
        let threshold_configs: Vec<ThresholdConfig> = thresholds
            .into_iter()
            .map(|(threshold, target_entities)| ThresholdConfig {
                threshold,
                target_entities,
            })
            .collect();

        Self {
            config: GraphConfig {
                n_left,
                n_right,
                n_isolates,
                thresholds: threshold_configs,
            },
        }
    }

    /// Create a production-scale configuration for million-record testing.
    ///
    /// Returns a pre-configured GraphConfig suitable for production-scale testing
    /// with 1.1M records and hierarchical thresholds.
    ///
    /// Returns:
    ///     GraphConfig: Production-scale configuration
    ///
    /// Example:
    ///     ```python
    ///     config = GraphConfig.production_1m()
    ///     # 550k left + 550k right records
    ///     # Thresholds: 0.9 -> 200k entities, 0.7 -> 100k entities, 0.5 -> 50k entities
    ///     ```
    #[classmethod]
    fn production_1m(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            config: GraphConfig::production_1m(),
        }
    }

    /// Create a large-scale configuration for 10M+ record testing.
    ///
    /// Returns a pre-configured GraphConfig suitable for very large-scale testing
    /// with 11M records and hierarchical thresholds.
    ///
    /// Returns:
    ///     GraphConfig: Large-scale configuration  
    ///
    /// Example:
    ///     ```python
    ///     config = GraphConfig.production_10m()
    ///     # 5.5M left + 5.5M right records
    ///     ```
    #[classmethod]
    fn production_10m(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            config: GraphConfig::production_10m(),
        }
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "GraphConfig(n_left={}, n_right={}, n_isolates={}, thresholds={})",
            self.config.n_left,
            self.config.n_right,
            self.config.n_isolates,
            self.config.thresholds.len()
        )
    }
}

/// Generate a hierarchical graph with realistic entity resolution patterns.
///
/// Creates a bipartite graph with exact component counts at specified thresholds,
/// using the hierarchical block construction method to ensure correct component
/// distributions that mirror production entity resolution patterns.
///
/// Args:
///     config (GraphConfig): Configuration specifying graph structure and thresholds
///
/// Returns:
///     Tuple[List[Tuple[Any, Any, float]], int]: (edges, total_nodes)
///         edges: List of (left_id, right_id, similarity) tuples
///         total_nodes: Total number of nodes in the graph
///
/// Complexity:
///     O(n + m) where n = nodes, m = edges
///
/// Example:
///     ```python
///     config = GraphConfig.production_1m()
///     edges, total_nodes = generate_hierarchical_graph(config)
///     
///     collection = Collection.from_edges(edges)
///     partition = collection.at(0.9)
///     print(f"Entities at 0.9: {len(partition.entities)}")
///     ```
#[pyfunction]
fn generate_hierarchical_graph(config: PyGraphConfig, _py: Python<'_>) -> GraphResult {
    let graph_data = starlings_core::test_utils::generate_hierarchical_graph(config.config);

    let python_edges: Vec<(i64, i64, f64)> = graph_data
        .edges
        .into_iter()
        .map(|(id1, id2, weight)| (id1 as i64, id2 as i64, weight))
        .collect();

    Ok((python_edges, graph_data.total_nodes))
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
    m.add_class::<PyGraphConfig>()?;
    m.add_function(wrap_pyfunction!(generate_hierarchical_graph, m)?)?;
    Ok(())
}
