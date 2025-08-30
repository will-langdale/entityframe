use pyo3::prelude::*;

mod collection;
mod entity;
mod frame;
mod hash;
mod interner;

pub use collection::CollectionCore;
pub use entity::EntityCore;
pub use frame::EntityFrame;
pub use interner::StringInternerCore;

/// A Python module implemented in Rust.
#[pymodule]
fn starlings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StringInternerCore>()?;
    m.add_class::<EntityCore>()?;
    m.add_class::<CollectionCore>()?;
    m.add_class::<EntityFrame>()?;
    Ok(())
}
