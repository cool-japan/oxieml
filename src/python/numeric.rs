//! Python bindings for special/numeric functions.

use pyo3::prelude::*;

fn map_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Error function erf(x).
#[pyfunction]
pub fn erf_py(x: f64) -> f64 {
    crate::special::erf(x)
}

/// Complementary error function erfc(x).
#[pyfunction]
pub fn erfc_py(x: f64) -> f64 {
    crate::special::erfc(x)
}

/// Log-gamma function lgamma(x).
#[pyfunction]
pub fn lgamma_py(x: f64) -> PyResult<f64> {
    if x <= 0.0 && x == x.floor() {
        return Err(map_err("lgamma undefined at non-positive integers"));
    }
    Ok(crate::special::lgamma(x))
}

/// Digamma function ψ(x).
#[pyfunction]
pub fn digamma_py(x: f64) -> PyResult<f64> {
    if x <= 0.0 && x == x.floor() {
        return Err(map_err("digamma undefined at non-positive integers"));
    }
    Ok(crate::special::digamma(x))
}

/// Exponential integral Ei(x).
#[pyfunction]
pub fn ei_py(x: f64) -> f64 {
    crate::special::ei(x)
}

/// Sine integral Si(x).
#[pyfunction]
pub fn si_py(x: f64) -> f64 {
    crate::special::si(x)
}

/// Cosine integral Ci(x).
#[pyfunction]
pub fn ci_py(x: f64) -> f64 {
    crate::special::ci(x)
}

/// Principal branch of the Lambert W function W₀(x).
#[pyfunction]
pub fn lambert_w0_py(x: f64) -> PyResult<f64> {
    crate::lambert_w0(x).map_err(map_err)
}

/// Secondary real branch of the Lambert W function W₋₁(x).
#[pyfunction]
pub fn lambert_wm1_py(x: f64) -> PyResult<f64> {
    crate::lambert_wm1(x).map_err(map_err)
}
