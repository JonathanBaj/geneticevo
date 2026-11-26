mod model;
mod population;
mod ppo;

use pyo3::prelude::*;

#[pymodule]
fn _geneticevo_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<population::Population>()?;
    Ok(())
}
