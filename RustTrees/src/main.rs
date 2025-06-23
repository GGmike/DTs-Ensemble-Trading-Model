use pyo3::prelude::*;
use itertools::Itertools;

#[pyfunction]
fn column_combinations(py_columns: Vec<String>, r: usize) -> Vec<Vec<String>> {
    py_columns.into_iter().combinations(r).collect()
}