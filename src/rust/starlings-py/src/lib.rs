use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyType};
use std::sync::Arc;

use starlings_core::{DataContext, Key, PartitionHierarchy, PartitionLevel};

/// Python wrapper for PartitionLevel
#[pyclass]
#[derive(Clone)]
pub struct PyPartition {
    partition: PartitionLevel,
}

#[pymethods]
impl PyPartition {
    /// Get the number of entities in this partition
    fn __len__(&self) -> usize {
        self.partition.entities().len()
    }

    /// Get entities as list of lists of record indices
    fn entities(&self) -> Vec<Vec<u32>> {
        self.partition
            .entities()
            .iter()
            .map(|bitmap| bitmap.iter().collect())
            .collect()
    }

    /// String representation for debugging
    fn __repr__(&self) -> String {
        format!("PyPartition(entities={})", self.entities().len())
    }
}

/// Python wrapper for PartitionHierarchy  
#[pyclass]
#[derive(Clone)]
pub struct PyCollection {
    hierarchy: PartitionHierarchy,
}

#[pymethods]
impl PyCollection {
    /// Create a collection from edges
    ///
    /// Args:
    ///     edges: List of (key1, key2, threshold) tuples
    ///     source: Optional source name (defaults to "default")
    #[classmethod]
    #[pyo3(signature = (edges, *, source=None))]
    fn from_edges(
        _cls: &Bound<'_, PyType>,
        edges: Vec<(Py<PyAny>, Py<PyAny>, f64)>,
        source: Option<String>,
        py: Python,
    ) -> PyResult<Self> {
        let source_name = source.unwrap_or_else(|| "default".to_string());
        let mut context = DataContext::new();
        let mut rust_edges = Vec::new();

        for (key1_obj, key2_obj, threshold) in edges {
            // Convert Python objects to Rust Keys
            let key1 = python_obj_to_key(key1_obj, py)?;
            let key2 = python_obj_to_key(key2_obj, py)?;

            // Ensure records exist in context
            let id1 = context.ensure_record(&source_name, key1);
            let id2 = context.ensure_record(&source_name, key2);

            rust_edges.push((id1, id2, threshold));
        }

        let hierarchy = PartitionHierarchy::from_edges(rust_edges, Arc::new(context), 6);

        Ok(PyCollection { hierarchy })
    }

    /// Get partition at specific threshold
    ///
    /// Args:
    ///     threshold: Threshold value between 0.0 and 1.0
    ///         
    /// Returns:
    ///     PyPartition: Partition at the specified threshold
    fn at(&mut self, threshold: f64) -> PyResult<PyPartition> {
        let partition = self.hierarchy.at_threshold(threshold);
        Ok(PyPartition {
            partition: partition.clone(),
        })
    }

    /// String representation for debugging
    fn __repr__(&self) -> String {
        "PyCollection".to_string()
    }
}

/// Convert Python object to Rust Key
fn python_obj_to_key(obj: Py<PyAny>, py: Python) -> PyResult<Key> {
    // Try different Python types
    if let Ok(s) = obj.downcast_bound::<PyString>(py) {
        Ok(Key::String(s.to_string()))
    } else if let Ok(b) = obj.downcast_bound::<PyBytes>(py) {
        Ok(Key::Bytes(b.as_bytes().to_vec()))
    } else if let Ok(i) = obj.extract::<i64>(py) {
        if i < 0 {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Integer key must be non-negative and fit in u64",
            ))
        } else if i <= u32::MAX as i64 {
            Ok(Key::U32(i as u32))
        } else {
            Ok(Key::U64(i as u64))
        }
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
