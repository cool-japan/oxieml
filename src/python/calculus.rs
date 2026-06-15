//! Python bindings for calculus operations (integration, limits).

use pyo3::prelude::*;

fn map_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Numerically evaluate a definite integral ∫_lo^hi f(x) dx.
///
/// Parameters
/// ----------
/// expr_str : str
///     Expression string (e.g. `"x^2"`, `"sin(x)"`).
/// var : int
///     Variable index (0-based) to integrate over.
/// lo : float
///     Lower bound.
/// hi : float
///     Upper bound.
///
/// Returns
/// -------
/// float
///     Numerical value of the integral.
#[pyfunction]
pub fn integrate_definite_py(expr_str: &str, var: usize, lo: f64, hi: f64) -> PyResult<f64> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let ctx = crate::EvalCtx::new(&[]);
    lowered
        .integrate_definite(var, lo, hi, &ctx)
        .map_err(map_err)
}

/// Evaluate the limit of `expr_str` as variable `var` approaches `at`.
///
/// Parameters
/// ----------
/// expr_str : str
///     Expression string.
/// var : int
///     Variable index to take the limit over.
/// at : float
///     Point to approach. Use `float('inf')` for +∞, `float('-inf')` for -∞.
///
/// Returns
/// -------
/// float
///     The limit value.  Returns `inf` or `-inf` when appropriate.
///
/// Raises
/// ------
/// RuntimeError
///     When the limit does not exist or is indeterminate.
#[pyfunction]
pub fn limit_py(expr_str: &str, var: usize, at: f64) -> PyResult<f64> {
    use pyo3::exceptions::PyRuntimeError;
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let point = if at.is_infinite() && at > 0.0 {
        crate::LimitPoint::PosInf
    } else if at.is_infinite() {
        crate::LimitPoint::NegInf
    } else {
        crate::LimitPoint::Finite(at)
    };
    match lowered.limit(var, point) {
        crate::LimitResult::Finite(v) => Ok(v),
        crate::LimitResult::PosInf => Ok(f64::INFINITY),
        crate::LimitResult::NegInf => Ok(f64::NEG_INFINITY),
        crate::LimitResult::DoesNotExist => Err(PyRuntimeError::new_err("limit does not exist")),
        crate::LimitResult::Indeterminate => Err(PyRuntimeError::new_err("limit is indeterminate")),
    }
}
