//! Python bindings for Pareto-front extraction over discovered formulas.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::PySymRegConfig;
use super::symreg::PyDiscoveredFormula;

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert `(x, y)` numpy arrays into row-major samples for [`crate::SymRegEngine`].
fn xy_to_rows(
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, usize)> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    let n_samples = y_arr.len();
    let n_features = x_arr.ncols();
    if x_arr.nrows() != n_samples {
        return Err(PyValueError::new_err(format!(
            "X has {} rows but y has {} elements",
            x_arr.nrows(),
            n_samples
        )));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("input arrays must not be empty"));
    }
    let mut inputs: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let val = x_arr
                .get((i, j))
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("index ({i},{j}) out of bounds")))?;
            row.push(val);
        }
        inputs.push(row);
    }
    let targets: Vec<f64> = y_arr.iter().copied().collect();
    Ok((inputs, targets, n_features))
}

/// Discover the Pareto-optimal (MSE vs. complexity) formulas from data.
///
/// Runs the full symbolic-regression search via `SymRegEngine.discover`, then
/// extracts the non-dominated front (sorted by ascending complexity).
#[pyfunction]
pub fn discover_pareto_py<'py>(
    py: Python<'py>,
    config: &PySymRegConfig,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Vec<PyDiscoveredFormula>> {
    let (inputs, targets, n_features) = xy_to_rows(x, y)?;
    let engine = crate::SymRegEngine::new(config.inner.clone());
    let result = py.detach(|| engine.discover_pareto(&inputs, &targets, n_features));
    Ok(result
        .map_err(map_err)?
        .into_iter()
        .map(|f| PyDiscoveredFormula { inner: f })
        .collect())
}

/// Extract the Pareto-optimal subset of an already-discovered formula pool.
///
/// A formula dominates another when it is at least as good on both MSE and
/// complexity, and strictly better on at least one. The result is sorted by
/// ascending complexity.
#[pyfunction]
pub fn pareto_front_py(formulas: Vec<PyDiscoveredFormula>) -> Vec<PyDiscoveredFormula> {
    let inner: Vec<crate::DiscoveredFormula> = formulas.into_iter().map(|f| f.inner).collect();
    crate::pareto_front(&inner)
        .into_iter()
        .map(|f| PyDiscoveredFormula { inner: f })
        .collect()
}
