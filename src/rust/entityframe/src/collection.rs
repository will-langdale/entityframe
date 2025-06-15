use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::entity::Entity;
use crate::interner::StringInterner;

/// Profiling data for performance analysis
#[derive(Debug)]
struct ProfilingData {
    // Component timing (in nanoseconds)
    hasher_creation_time: std::sync::atomic::AtomicU64,
    string_lookup_time: std::sync::atomic::AtomicU64,
    hash_update_time: std::sync::atomic::AtomicU64,
    bitmap_iteration_time: std::sync::atomic::AtomicU64,
    sorting_time: std::sync::atomic::AtomicU64,
    finalization_time: std::sync::atomic::AtomicU64,

    // Operation counters
    fast_path_entities: std::sync::atomic::AtomicUsize,
    fallback_entities: std::sync::atomic::AtomicUsize,
    total_string_lookups: std::sync::atomic::AtomicUsize,
    total_hash_updates: std::sync::atomic::AtomicUsize,
    total_bitmap_iterations: std::sync::atomic::AtomicUsize,
}

/// EntityCollection: A collection of entities from a single process (like pandas Series)
/// Collections should only be created through EntityFrame.create_collection() to ensure shared interner
#[pyclass]
#[derive(Clone)]
pub struct EntityCollection {
    pub entities: Vec<Entity>,
    process_name: String,
    // Simple design: collections don't own interners, they get them from the frame
}

#[pymethods]
impl EntityCollection {
    /// Create EntityCollection (internal constructor)
    /// Users should use EntityFrame.create_collection() instead
    #[new]
    pub fn new(process_name: &str) -> Self {
        Self {
            entities: Vec::new(),
            process_name: process_name.to_string(),
        }
    }

    /// Get all entities in this collection
    pub fn get_entities(&self) -> Vec<Entity> {
        self.entities.clone()
    }

    /// Get the process name for this collection
    #[getter]
    pub fn process_name(&self) -> &str {
        &self.process_name
    }

    /// Get the number of entities in this collection
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the total number of records across all entities
    pub fn total_records(&self) -> u64 {
        self.entities
            .iter()
            .map(|entity| entity.total_records())
            .sum()
    }

    /// Get an entity by index
    pub fn get_entity(&self, index: usize) -> PyResult<Entity> {
        self.entities.get(index).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })
    }

    /// Compare this collection with another collection entity-by-entity
    pub fn compare_with(
        &self,
        other: &EntityCollection,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        if self.entities.len() != other.entities.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Collections must have the same number of entities to compare",
            ));
        }

        Python::with_gil(|py| {
            // Pre-allocate the comparisons vector
            let mut comparisons = Vec::with_capacity(self.entities.len());

            for (i, (entity1, entity2)) in
                self.entities.iter().zip(other.entities.iter()).enumerate()
            {
                let jaccard = entity1.jaccard_similarity(entity2);

                let mut comparison = HashMap::new();
                comparison.insert(
                    "entity_index".to_string(),
                    i.into_pyobject(py).unwrap().into_any().unbind(),
                );
                comparison.insert(
                    "process1".to_string(),
                    self.process_name
                        .clone()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                );
                comparison.insert(
                    "process2".to_string(),
                    other
                        .process_name
                        .clone()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                );
                comparison.insert(
                    "jaccard".to_string(),
                    jaccard.into_pyobject(py).unwrap().into_any().unbind(),
                );

                comparisons.push(comparison);
            }

            Ok(comparisons)
        })
    }
}

impl EntityCollection {
    /// Add entities to this collection using shared interner from frame
    /// This is the main method used by EntityFrame.add_method()
    /// This is a regular Rust method, not exposed to Python
    pub fn add_entities(
        &mut self,
        entity_data: Vec<HashMap<String, Vec<String>>>,
        interner: &mut StringInterner,
        dataset_name_to_id: &mut HashMap<String, u32>,
    ) {
        use roaring::RoaringBitmap;

        // Pre-allocate space for entities
        self.entities.reserve(entity_data.len());

        // Step 1: Process all dataset names first
        for entity_dict in &entity_data {
            for dataset_name in entity_dict.keys() {
                if !dataset_name_to_id.contains_key(dataset_name) {
                    let new_id = interner.intern(dataset_name);
                    dataset_name_to_id.insert(dataset_name.clone(), new_id);
                }
            }
        }

        // Step 2: Process each entity using batch processing per dataset
        for entity_dict in entity_data {
            let mut dataset_bitmaps = HashMap::new();
            let mut dataset_sorted_records = HashMap::new();

            // Group records by dataset for batch processing
            let mut dataset_records_raw = HashMap::new();
            for (dataset_name, record_ids) in entity_dict {
                let dataset_id = dataset_name_to_id[&dataset_name];
                dataset_records_raw.insert(dataset_id, record_ids);
            }

            // Batch intern and sort all records for this entity
            let batch_results = interner.batch_intern_by_dataset(&dataset_records_raw);

            // Create roaring bitmaps and store sorted order
            for (dataset_id, (record_ids, sorted_record_ids)) in batch_results {
                let mut bitmap = RoaringBitmap::new();
                bitmap.extend(&record_ids);
                dataset_bitmaps.insert(dataset_id, bitmap);
                dataset_sorted_records.insert(dataset_id, sorted_record_ids);
            }

            // Create entity with pre-computed sorted order
            let entity = Entity::from_sorted_data(dataset_bitmaps, dataset_sorted_records);
            self.entities.push(entity);
        }
    }

    /// Batch hash all entities in this collection for optimal performance
    pub fn hash_all_entities(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<Vec<u8>>> {
        self.hash_all_entities_with_profiling(interner, algorithm, false)
    }

    /// Batch hash all entities with detailed profiling enabled (Python API)
    pub fn hash_all_entities_profiling(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<Vec<u8>>> {
        self.hash_all_entities_with_profiling(interner, algorithm, true)
    }

    /// Batch hash all entities with detailed profiling enabled
    pub fn hash_all_entities_with_profiling(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
        enable_profiling: bool,
    ) -> PyResult<Vec<Vec<u8>>> {
        use std::sync::atomic::{AtomicU64, AtomicUsize};

        if self.entities.is_empty() {
            return Ok(Vec::new());
        }

        // Profiling data structures (shared across threads)
        let profiling_data = if enable_profiling {
            Some(Arc::new(ProfilingData {
                hasher_creation_time: AtomicU64::new(0),
                string_lookup_time: AtomicU64::new(0),
                hash_update_time: AtomicU64::new(0),
                bitmap_iteration_time: AtomicU64::new(0),
                sorting_time: AtomicU64::new(0),
                finalization_time: AtomicU64::new(0),
                fast_path_entities: AtomicUsize::new(0),
                fallback_entities: AtomicUsize::new(0),
                total_string_lookups: AtomicUsize::new(0),
                total_hash_updates: AtomicUsize::new(0),
                total_bitmap_iterations: AtomicUsize::new(0),
            }))
        } else {
            None
        };

        // Get sorted dataset IDs from interner (one-time cost, requires mutable access)
        let sorted_dataset_ids = interner.get_sorted_ids().to_vec();

        // Clone algorithm string for parallel use
        let algo = algorithm.to_string();

        let total_start = std::time::Instant::now();
        let total_entities = self.entities.len();

        // Adaptive parallelization strategy based on over-parallelization analysis
        let all_hashes = if total_entities < 1000 {
            // Single-threaded for small datasets to avoid thread coordination overhead
            if enable_profiling {
                println!(
                    "\n⚡ Using SINGLE-THREADED approach ({} entities < 1000 threshold)",
                    total_entities
                );
            }
            self.hash_entities_sequential(&sorted_dataset_ids, interner, &algo, &profiling_data)
        } else {
            // Chunked parallel processing for larger datasets
            let optimal_chunk_size =
                std::cmp::max(500, total_entities / rayon::current_num_threads());
            if enable_profiling {
                println!(
                    "\n⚡ Using CHUNKED PARALLEL approach ({} entities, chunk size: {})",
                    total_entities, optimal_chunk_size
                );
            }
            self.hash_entities_chunked_parallel(
                &sorted_dataset_ids,
                interner,
                &algo,
                &profiling_data,
                optimal_chunk_size,
            )
        }?;

        let total_time = total_start.elapsed();
        let entities_per_sec = total_entities as f64 / total_time.as_secs_f64();

        if enable_profiling {
            if let Some(profiling) = &profiling_data {
                self.print_detailed_profiling(profiling, total_time, total_entities);
            }
        } else {
            println!("\nAdaptive parallel hashing performance:");
            println!(
                "  {} entities in {:.1}ms ({:.0} entities/sec)",
                total_entities,
                total_time.as_secs_f64() * 1000.0,
                entities_per_sec
            );
        }

        Ok(all_hashes)
    }

    /// Sequential hashing for small datasets to avoid parallelization overhead
    fn hash_entities_sequential(
        &self,
        sorted_dataset_ids: &[u32],
        interner: &crate::interner::StringInterner,
        algorithm: &str,
        profiling_data: &Option<Arc<ProfilingData>>,
    ) -> PyResult<Vec<Vec<u8>>> {
        let mut all_hashes = Vec::with_capacity(self.entities.len());

        for entity in &self.entities {
            let hash = self.hash_single_entity(
                entity,
                sorted_dataset_ids,
                interner,
                algorithm,
                profiling_data,
            )?;
            all_hashes.push(hash);
        }

        Ok(all_hashes)
    }

    /// Chunked parallel hashing to balance parallelism with coordination overhead
    fn hash_entities_chunked_parallel(
        &self,
        sorted_dataset_ids: &[u32],
        interner: &crate::interner::StringInterner,
        algorithm: &str,
        profiling_data: &Option<Arc<ProfilingData>>,
        chunk_size: usize,
    ) -> PyResult<Vec<Vec<u8>>> {
        use rayon::prelude::*;

        // Chunk entities and process each chunk in parallel
        let all_hashes: Result<Vec<_>, _> = self
            .entities
            .chunks(chunk_size)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|chunk| -> PyResult<Vec<Vec<u8>>> {
                let mut chunk_hashes = Vec::with_capacity(chunk.len());
                for entity in *chunk {
                    let hash = self.hash_single_entity(
                        entity,
                        sorted_dataset_ids,
                        interner,
                        algorithm,
                        profiling_data,
                    )?;
                    chunk_hashes.push(hash);
                }
                Ok(chunk_hashes)
            })
            .collect();

        // Flatten the chunked results
        let all_hashes = all_hashes?;
        let flattened: Vec<Vec<u8>> = all_hashes.into_iter().flatten().collect();

        Ok(flattened)
    }

    /// Hash a single entity with full profiling support
    fn hash_single_entity(
        &self,
        entity: &crate::entity::Entity,
        sorted_dataset_ids: &[u32],
        interner: &crate::interner::StringInterner,
        algorithm: &str,
        profiling_data: &Option<Arc<ProfilingData>>,
    ) -> PyResult<Vec<u8>> {
        use std::sync::atomic::Ordering;

        // 1. Hasher creation timing
        let hasher_start = std::time::Instant::now();
        let mut hasher = crate::hash::create_hasher(algorithm).expect("Failed to create hasher");
        let hasher_time = hasher_start.elapsed();

        if let Some(ref profiling) = profiling_data {
            profiling
                .hasher_creation_time
                .fetch_add(hasher_time.as_nanos() as u64, Ordering::Relaxed);
        }

        // 2. Get entity data
        let datasets_map = entity.get_datasets_map();
        let sorted_records_map = entity.get_sorted_records_map();

        let mut entity_used_fast_path = false;

        // 3. Process datasets in sorted order for deterministic results
        // OPTIMIZATION: Only iterate through datasets that this entity actually has
        // Create a small filtered list instead of checking all global dataset IDs
        let mut entity_datasets: Vec<u32> = datasets_map.keys().copied().collect();
        // Sort using the global order by finding positions (cached lookup would be better)
        entity_datasets.sort_by(|&a, &b| {
            // Find positions in the sorted array - this maintains deterministic order
            let pos_a = sorted_dataset_ids.binary_search(&a).unwrap_or(usize::MAX);
            let pos_b = sorted_dataset_ids.binary_search(&b).unwrap_or(usize::MAX);
            pos_a.cmp(&pos_b)
        });

        for &dataset_id in &entity_datasets {
            if let Some(bitmap) = datasets_map.get(&dataset_id) {
                // Bitmap iteration timing
                let bitmap_start = std::time::Instant::now();
                let bitmap_size = bitmap.len();
                let bitmap_time = bitmap_start.elapsed();

                if let Some(ref profiling) = profiling_data {
                    profiling
                        .bitmap_iteration_time
                        .fetch_add(bitmap_time.as_nanos() as u64, Ordering::Relaxed);
                    profiling
                        .total_bitmap_iterations
                        .fetch_add(bitmap_size as usize, Ordering::Relaxed);
                }

                // String lookup timing for dataset name
                let lookup_start = std::time::Instant::now();
                let dataset_str_opt = interner.get_string_fast(dataset_id);
                let lookup_time = lookup_start.elapsed();

                if let Some(ref profiling) = profiling_data {
                    profiling
                        .string_lookup_time
                        .fetch_add(lookup_time.as_nanos() as u64, Ordering::Relaxed);
                    profiling
                        .total_string_lookups
                        .fetch_add(1, Ordering::Relaxed);
                }

                if let Some(dataset_str) = dataset_str_opt {
                    // Hash update timing for dataset name
                    let hash_start = std::time::Instant::now();
                    hasher.update(dataset_str.as_bytes());
                    let hash_time = hash_start.elapsed();

                    if let Some(ref profiling) = profiling_data {
                        profiling
                            .hash_update_time
                            .fetch_add(hash_time.as_nanos() as u64, Ordering::Relaxed);
                        profiling.total_hash_updates.fetch_add(1, Ordering::Relaxed);
                    }

                    // Process record strings
                    if let Some(sorted_record_ids) = sorted_records_map.get(&dataset_id) {
                        // Fast path: use pre-computed sorted order
                        entity_used_fast_path = true;

                        for &record_id in sorted_record_ids {
                            // String lookup timing for record
                            let lookup_start = std::time::Instant::now();
                            let record_str_opt = interner.get_string_fast(record_id);
                            let lookup_time = lookup_start.elapsed();

                            if let Some(ref profiling) = profiling_data {
                                profiling
                                    .string_lookup_time
                                    .fetch_add(lookup_time.as_nanos() as u64, Ordering::Relaxed);
                                profiling
                                    .total_string_lookups
                                    .fetch_add(1, Ordering::Relaxed);
                            }

                            if let Some(record_str) = record_str_opt {
                                // Hash update timing for record
                                let hash_start = std::time::Instant::now();
                                hasher.update(record_str.as_bytes());
                                let hash_time = hash_start.elapsed();

                                if let Some(ref profiling) = profiling_data {
                                    profiling
                                        .hash_update_time
                                        .fetch_add(hash_time.as_nanos() as u64, Ordering::Relaxed);
                                    profiling.total_hash_updates.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    } else {
                        // Fallback path: sort by string value for deterministic results
                        let sort_start = std::time::Instant::now();
                        let mut record_ids: Vec<u32> = bitmap.iter().collect();
                        record_ids.sort_by(|&a, &b| {
                            let str_a = interner.get_string_fast(a).unwrap_or("");
                            let str_b = interner.get_string_fast(b).unwrap_or("");
                            str_a.cmp(str_b)
                        });
                        let sort_time = sort_start.elapsed();

                        if let Some(ref profiling) = profiling_data {
                            profiling
                                .sorting_time
                                .fetch_add(sort_time.as_nanos() as u64, Ordering::Relaxed);
                        }

                        for record_id in record_ids {
                            // String lookup timing for record (fallback path)
                            let lookup_start = std::time::Instant::now();
                            let record_str_opt = interner.get_string_fast(record_id);
                            let lookup_time = lookup_start.elapsed();

                            if let Some(ref profiling) = profiling_data {
                                profiling
                                    .string_lookup_time
                                    .fetch_add(lookup_time.as_nanos() as u64, Ordering::Relaxed);
                                profiling
                                    .total_string_lookups
                                    .fetch_add(1, Ordering::Relaxed);
                            }

                            if let Some(record_str) = record_str_opt {
                                // Hash update timing for record (fallback path)
                                let hash_start = std::time::Instant::now();
                                hasher.update(record_str.as_bytes());
                                let hash_time = hash_start.elapsed();

                                if let Some(ref profiling) = profiling_data {
                                    profiling
                                        .hash_update_time
                                        .fetch_add(hash_time.as_nanos() as u64, Ordering::Relaxed);
                                    profiling.total_hash_updates.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update entity path counters
        if let Some(ref profiling) = profiling_data {
            if entity_used_fast_path {
                profiling.fast_path_entities.fetch_add(1, Ordering::Relaxed);
            } else {
                profiling.fallback_entities.fetch_add(1, Ordering::Relaxed);
            }
        }

        // 4. Finalization timing
        let finalize_start = std::time::Instant::now();
        let result = hasher.finalize().to_vec();
        let finalize_time = finalize_start.elapsed();

        if let Some(ref profiling) = profiling_data {
            profiling
                .finalization_time
                .fetch_add(finalize_time.as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(result)
    }

    /// Print detailed profiling information
    fn print_detailed_profiling(
        &self,
        profiling: &ProfilingData,
        total_time: std::time::Duration,
        total_entities: usize,
    ) {
        use std::sync::atomic::Ordering;

        // Extract profiling data
        let hasher_creation_ns = profiling.hasher_creation_time.load(Ordering::Relaxed);
        let string_lookup_ns = profiling.string_lookup_time.load(Ordering::Relaxed);
        let hash_update_ns = profiling.hash_update_time.load(Ordering::Relaxed);
        let bitmap_iteration_ns = profiling.bitmap_iteration_time.load(Ordering::Relaxed);
        let sorting_ns = profiling.sorting_time.load(Ordering::Relaxed);
        let finalization_ns = profiling.finalization_time.load(Ordering::Relaxed);

        let fast_path_entities = profiling.fast_path_entities.load(Ordering::Relaxed);
        let fallback_entities = profiling.fallback_entities.load(Ordering::Relaxed);
        let total_string_lookups = profiling.total_string_lookups.load(Ordering::Relaxed);
        let total_hash_updates = profiling.total_hash_updates.load(Ordering::Relaxed);
        let total_bitmap_iterations = profiling.total_bitmap_iterations.load(Ordering::Relaxed);

        let total_ns = total_time.as_nanos() as u64;
        let entities_per_sec = total_entities as f64 / total_time.as_secs_f64();

        println!("\n🔬 DETAILED PROFILING ANALYSIS");
        println!("{}", "=".repeat(50));

        println!("\n📊 Overall Performance:");
        println!(
            "  {} entities in {:.1}ms ({:.0} entities/sec)",
            total_entities,
            total_time.as_secs_f64() * 1000.0,
            entities_per_sec
        );
        println!("  Target: 1,000,000 entities/sec");
        println!(
            "  Shortfall: {:.1}x too slow",
            1_000_000.0 / entities_per_sec
        );

        println!("\n⏱️  Component Timing Breakdown:");
        let components = [
            ("Hasher Creation", hasher_creation_ns),
            ("String Lookups", string_lookup_ns),
            ("Hash Updates", hash_update_ns),
            ("Bitmap Iteration", bitmap_iteration_ns),
            ("Sorting (fallback)", sorting_ns),
            ("Finalization", finalization_ns),
        ];

        for (name, time_ns) in &components {
            let time_ms = *time_ns as f64 / 1_000_000.0;
            let percentage = (*time_ns as f64 / total_ns as f64) * 100.0;
            println!("  {:<18} {:8.1}ms ({:5.1}%)", name, time_ms, percentage);
        }

        let accounted_ns: u64 = components.iter().map(|(_, ns)| ns).sum();
        let unaccounted_ns = total_ns.saturating_sub(accounted_ns);
        let unaccounted_percentage = (unaccounted_ns as f64 / total_ns as f64) * 100.0;
        println!(
            "  {:<18} {:8.1}ms ({:5.1}%)",
            "Other/Overhead",
            unaccounted_ns as f64 / 1_000_000.0,
            unaccounted_percentage
        );

        println!("\n📈 Operation Counters:");
        println!(
            "  Fast path entities:   {:>8} ({:5.1}%)",
            fast_path_entities,
            (fast_path_entities as f64 / total_entities as f64) * 100.0
        );
        println!(
            "  Fallback entities:    {:>8} ({:5.1}%)",
            fallback_entities,
            (fallback_entities as f64 / total_entities as f64) * 100.0
        );
        println!(
            "  String lookups:       {:>8} ({:.1} per entity)",
            total_string_lookups,
            total_string_lookups as f64 / total_entities as f64
        );
        println!(
            "  Hash updates:         {:>8} ({:.1} per entity)",
            total_hash_updates,
            total_hash_updates as f64 / total_entities as f64
        );
        println!(
            "  Bitmap iterations:    {:>8} ({:.1} per entity)",
            total_bitmap_iterations,
            total_bitmap_iterations as f64 / total_entities as f64
        );

        println!("\n⚡ Performance Analysis:");
        if string_lookup_ns > total_ns / 4 {
            println!(
                "  🚨 STRING LOOKUPS are the major bottleneck ({:.1}% of time)",
                (string_lookup_ns as f64 / total_ns as f64) * 100.0
            );
        }
        if hash_update_ns > total_ns / 4 {
            println!(
                "  🚨 HASH UPDATES are the major bottleneck ({:.1}% of time)",
                (hash_update_ns as f64 / total_ns as f64) * 100.0
            );
        }
        if sorting_ns > total_ns / 10 {
            println!(
                "  ⚠️  SORTING fallback consuming significant time ({:.1}% of time)",
                (sorting_ns as f64 / total_ns as f64) * 100.0
            );
        }
        if unaccounted_percentage > 20.0 {
            println!(
                "  ⚠️  HIGH OVERHEAD: {:.1}% time unaccounted (parallelization/memory overhead)",
                unaccounted_percentage
            );
        }

        // Per-operation timing analysis
        if total_string_lookups > 0 {
            let avg_lookup_ns = string_lookup_ns / total_string_lookups as u64;
            println!(
                "  String lookup avg:    {:>8.0}ns per lookup",
                avg_lookup_ns
            );
            if avg_lookup_ns > 100 {
                println!("    🚨 STRING LOOKUPS TOO SLOW (should be <100ns)");
            }
        }

        if total_hash_updates > 0 {
            let avg_hash_ns = hash_update_ns / total_hash_updates as u64;
            println!("  Hash update avg:      {:>8.0}ns per update", avg_hash_ns);
            if avg_hash_ns > 1000 {
                println!("    🚨 HASH UPDATES TOO SLOW (should be <1000ns)");
            }
        }

        println!("{}", "=".repeat(50));
    }

    /// Batch hash all entities returning hex strings for convenience
    pub fn hash_all_entities_hex(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<String>> {
        let hashes = self.hash_all_entities(interner, algorithm)?;
        Ok(hashes.into_iter().map(hex::encode).collect())
    }

    /// Batch hash all entities returning single concatenated byte buffer
    /// This avoids creating individual PyBytes objects and should be much faster
    /// Format: [hash1_length][hash1_bytes][hash2_length][hash2_bytes]...
    pub fn hash_all_entities_concatenated(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<u8>> {
        let hashes = self.hash_all_entities(interner, algorithm)?;

        // Pre-calculate total size needed
        let hash_size = if hashes.is_empty() {
            0
        } else {
            hashes[0].len()
        };
        let total_size = hashes.len() * (4 + hash_size); // 4 bytes for length + hash bytes

        let mut result = Vec::with_capacity(total_size);

        for hash in hashes {
            // Write hash length as 4-byte little-endian
            result.extend_from_slice(&(hash.len() as u32).to_le_bytes());
            // Write hash bytes
            result.extend_from_slice(&hash);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_creation() {
        let collection = EntityCollection::new("test_process");
        assert_eq!(collection.process_name(), "test_process");
        assert_eq!(collection.len(), 0);
        assert!(collection.is_empty());
        assert_eq!(collection.total_records(), 0);
    }

    #[test]
    fn test_add_entities_with_shared_interner() {
        let mut collection = EntityCollection::new("test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c1".to_string(), "c2".to_string()],
                );
                data.insert("orders".to_string(), vec!["o1".to_string()]);
                data
            },
            {
                let mut data = HashMap::new();
                data.insert("customers".to_string(), vec!["c3".to_string()]);
                data
            },
        ];

        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        assert_eq!(collection.len(), 2);
        assert_eq!(collection.total_records(), 4); // c1, c2, o1, c3
        assert_eq!(dataset_name_to_id.len(), 2); // customers, orders
        assert_eq!(interner.len(), 6); // customers, orders, c1, c2, o1, c3

        // Check first entity
        let entity1 = collection.get_entity(0).unwrap();
        assert_eq!(entity1.total_records(), 3); // c1, c2, o1

        // Check second entity
        let entity2 = collection.get_entity(1).unwrap();
        assert_eq!(entity2.total_records(), 1); // c3
    }

    #[test]
    fn test_batch_processing_with_sorted_records() {
        let mut collection = EntityCollection::new("batch_test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        // Create test data with unsorted records to verify sorting works
        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c3".to_string(), "c1".to_string(), "c2".to_string()],
                );
                data.insert(
                    "orders".to_string(),
                    vec!["o2".to_string(), "o1".to_string()],
                );
                data
            },
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c5".to_string(), "c4".to_string()],
                );
                data
            },
        ];

        // Use optimised entity processing
        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        assert_eq!(collection.len(), 2);

        // Check that entities have sorted records cached
        let entity1 = collection.get_entity(0).unwrap();
        let entity2 = collection.get_entity(1).unwrap();

        assert!(entity1.has_sorted_records());
        assert!(entity2.has_sorted_records());

        // Get dataset IDs
        let customers_id = dataset_name_to_id["customers"];
        let orders_id = dataset_name_to_id["orders"];

        // Verify sorted order for entity 1
        if let Some(sorted_customers) = entity1.get_sorted_records(customers_id) {
            let sorted_strings: Vec<&str> = sorted_customers
                .iter()
                .map(|&id| interner.get_string_internal(id).unwrap())
                .collect();
            assert_eq!(sorted_strings, vec!["c1", "c2", "c3"]); // Should be alphabetically sorted
        } else {
            panic!("Entity 1 should have sorted customers records");
        }

        if let Some(sorted_orders) = entity1.get_sorted_records(orders_id) {
            let sorted_strings: Vec<&str> = sorted_orders
                .iter()
                .map(|&id| interner.get_string_internal(id).unwrap())
                .collect();
            assert_eq!(sorted_strings, vec!["o1", "o2"]); // Should be alphabetically sorted
        } else {
            panic!("Entity 1 should have sorted orders records");
        }

        // Verify sorted order for entity 2
        if let Some(sorted_customers) = entity2.get_sorted_records(customers_id) {
            let sorted_strings: Vec<&str> = sorted_customers
                .iter()
                .map(|&id| interner.get_string_internal(id).unwrap())
                .collect();
            assert_eq!(sorted_strings, vec!["c4", "c5"]); // Should be alphabetically sorted
        } else {
            panic!("Entity 2 should have sorted customers records");
        }
    }

    #[test]
    fn test_batch_processing_efficiency() {
        let mut collection = EntityCollection::new("batch_test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        // Same test data for verification
        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c2".to_string(), "c1".to_string()],
                );
                data.insert("orders".to_string(), vec!["o1".to_string()]);
                data
            },
            {
                let mut data = HashMap::new();
                data.insert("customers".to_string(), vec!["c3".to_string()]);
                data
            },
        ];

        // Process with batch method (now the only method)
        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        // Verify results
        assert_eq!(collection.len(), 2);
        assert_eq!(collection.total_records(), 4); // c1, c2, o1, c3

        // Check entities have expected data
        for i in 0..collection.len() {
            let entity = collection.get_entity(i).unwrap();

            // All entities should have sorted records with optimised processing
            assert!(entity.has_sorted_records());
        }
    }

    #[test]
    fn test_batch_hashing() {
        let mut collection = EntityCollection::new("hash_test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        // Create test data
        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c1".to_string(), "c2".to_string()],
                );
                data.insert("orders".to_string(), vec!["o1".to_string()]);
                data
            },
            {
                let mut data = HashMap::new();
                data.insert("customers".to_string(), vec!["c3".to_string()]);
                data.insert(
                    "orders".to_string(),
                    vec!["o2".to_string(), "o3".to_string()],
                );
                data
            },
        ];

        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        // Test batch hashing
        let hashes = collection
            .hash_all_entities(&mut interner, "sha256")
            .unwrap();
        assert_eq!(hashes.len(), 2);

        // Verify hash sizes (SHA-256 = 32 bytes)
        for hash in &hashes {
            assert_eq!(hash.len(), 32);
        }

        // Test hex batch hashing
        let hex_hashes = collection
            .hash_all_entities_hex(&mut interner, "blake3")
            .unwrap();
        assert_eq!(hex_hashes.len(), 2);

        // Verify hex strings (32 bytes = 64 hex chars)
        for hex_hash in &hex_hashes {
            assert_eq!(hex_hash.len(), 64);
        }

        // Test consistency - same algorithm should produce same results
        let hashes2 = collection
            .hash_all_entities(&mut interner, "sha256")
            .unwrap();
        assert_eq!(hashes, hashes2);

        // Test different algorithms produce different results
        let blake_hashes = collection
            .hash_all_entities(&mut interner, "blake3")
            .unwrap();
        assert_ne!(hashes, blake_hashes);
    }
}
