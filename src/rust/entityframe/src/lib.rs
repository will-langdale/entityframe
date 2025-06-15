use pyo3::prelude::*;

mod collection;
mod entity;
mod frame;
mod hash;
mod interner;

pub use collection::EntityCollection;
pub use entity::Entity;
pub use frame::EntityFrame;
pub use interner::StringInterner;

/// A Python module implemented in Rust.
#[pymodule]
fn entityframe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StringInterner>()?;
    m.add_class::<Entity>()?;
    m.add_class::<EntityCollection>()?;
    m.add_class::<EntityFrame>()?;
    Ok(())
}
