use pyo3::prelude::*;
use itertools::Itertools;

#[pyfunction]
fn column_combinations(py_columns: Vec<String>, r: usize) -> Vec<Vec<String>> {
    py_columns.into_iter().combinations(r).collect()
}

#[pymodule]
fn dts_ensemble_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(column_combinations, m)?)?;
    Ok(())
}