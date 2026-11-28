mod model;
mod population;

use pyo3::prelude::*;

#[pymodule]
fn _geneticevo_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<population::PolicyPopulation>()?;
    Ok(())
}
