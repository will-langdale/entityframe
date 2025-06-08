use pyo3::prelude::*;

/// Return a hello message from Rust.
#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_rust() {
        let result = hello_rust().unwrap();
        assert_eq!(result, "Hello from Rust!");
    }
}