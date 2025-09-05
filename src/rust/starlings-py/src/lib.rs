use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyType};
use std::sync::Arc;

use starlings_core::test_utils::{GraphConfig, ThresholdConfig};
use starlings_core::{DataContext, Key, PartitionHierarchy, PartitionLevel};

type GraphResult = PyResult<(Vec<(i64, i64, f64)>, usize)>;

/// A partition of records into entities at a specific threshold.
#[pyclass(name = "Partition")]
#[derive(Clone)]
pub struct PyPartition {
    partition: PartitionLevel,
}

#[pymethods]
impl PyPartition {
    /// Get the number of entities in this partition.
    fn __len__(&self) -> usize {
        self.partition.entities().len()
    }

    /// Get entities as list of lists of record indices.
    #[getter]
    fn entities(&self) -> Vec<Vec<u32>> {
        self.partition
            .entities()
            .iter()
            .map(|bitmap| bitmap.iter().collect())
            .collect()
    }

    /// Get the number of entities in this partition.
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

        // Pre-calculate capacity based on edge count (assume ~70% unique records)
        let estimated_records = (edges.len() * 14) / 10; // 1.4x edges for safety
        let context = DataContext::with_capacity(estimated_records);

        // Efficiently convert all Python keys to Rust edges with optimised bulk processing
        #[cfg(debug_assertions)]
        let conversion_start = std::time::Instant::now();

        // Phase 1: Bulk Python object extraction with deduplication
        #[cfg(debug_assertions)]
        let phase1_start = std::time::Instant::now();

        use rustc_hash::FxHashMap;
        let mut key_to_id: FxHashMap<Key, u32> = FxHashMap::default();
        let mut extracted_edges = Vec::with_capacity(edges.len());

        // Pre-allocate hash map for key lookups
        key_to_id.reserve(estimated_records);

        for (key1_obj, key2_obj, threshold) in edges {
            // Convert Python objects to Rust Keys (bulk extraction)
            let key1 = python_obj_to_key_fast(key1_obj, py)?;
            let key2 = python_obj_to_key_fast(key2_obj, py)?;

            extracted_edges.push((key1, key2, threshold));
        }

        #[cfg(debug_assertions)]
        let phase1_time = phase1_start.elapsed();

        // Phase 2: Parallel batch key registration with rayon
        #[cfg(debug_assertions)]
        let phase2_start = std::time::Instant::now();

        use rayon::prelude::*;
        use std::sync::Mutex;

        // Collect unique keys in parallel
        let unique_keys: Vec<Key> = {
            let mut keys_set = FxHashMap::default();
            for (key1, key2, _) in &extracted_edges {
                keys_set.entry(key1.clone()).or_insert(());
                keys_set.entry(key2.clone()).or_insert(());
            }
            keys_set.into_keys().collect()
        };

        // Register keys in parallel batches using batch method
        let batch_size = 5000;
        let key_to_id_mutex = Mutex::new(key_to_id);

        unique_keys.par_chunks(batch_size).for_each(|batch| {
            // Call batch registration method
            let ids = context.ensure_records_batch(&source_name, batch);

            // Build local map
            let mut local_map = FxHashMap::default();
            for (key, id) in batch.iter().zip(ids.iter()) {
                local_map.insert(key.clone(), *id);
            }

            // Merge local results back
            let mut global_map = key_to_id_mutex.lock().unwrap();
            global_map.extend(local_map);
        });

        let key_to_id = key_to_id_mutex.into_inner().unwrap();

        #[cfg(debug_assertions)]
        let phase2_time = phase2_start.elapsed();

        // Phase 3: Parallel edge ID mapping
        #[cfg(debug_assertions)]
        let phase3_start = std::time::Instant::now();

        // Use parallel iteration to map keys to IDs
        let rust_edges: Vec<(u32, u32, f64)> = extracted_edges
            .par_iter()
            .map(|(key1, key2, threshold)| {
                let id1 = key_to_id[key1];
                let id2 = key_to_id[key2];
                (id1, id2, *threshold)
            })
            .collect();

        #[cfg(debug_assertions)]
        let phase3_time = phase3_start.elapsed();

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
        if edge_count >= 100_000 {
            eprintln!("🏭 Production-scale Collection.from_edges performance:");
            eprintln!(
                "   📊 Scale: {} edges, {} unique records",
                edge_count, record_count
            );
            eprintln!("   ⚡ Python->Rust conversion breakdown:");
            eprintln!("      Phase 1 (Python extraction): {:?}", phase1_time);
            eprintln!("      Phase 2 (Key interning): {:?}", phase2_time);
            eprintln!("      Phase 3 (ID mapping): {:?}", phase3_time);
            eprintln!("      Total conversion: {:?}", conversion_time);
            eprintln!("   🏗️  Hierarchy construction: {:?}", hierarchy_time);
            eprintln!("   📈 Total time: {:?}", total_time);
            eprintln!(
                "   🎯 Edges per second: {:.0}",
                edge_count as f64 / total_time.as_secs_f64()
            );
            eprintln!(
                "   📍 Time breakdown: {:.1}% conversion, {:.1}% hierarchy",
                (conversion_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0,
                (hierarchy_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
            );
            if edge_count >= 1_000_000 {
                eprintln!(
                    "   🏆 1M edges <10s target: {}",
                    if total_time.as_secs_f64() < 10.0 {
                        "✅ ACHIEVED"
                    } else {
                        "❌ MISSED"
                    }
                );
            }
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
