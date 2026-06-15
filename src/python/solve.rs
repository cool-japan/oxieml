//! Python bindings for symbolic equation solving.

use pyo3::prelude::*;

fn map_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Find all real symbolic solutions of `expr_str = 0` for the given variable.
///
/// Parameters
/// ----------
/// expr_str : str
///     Expression string for the left-hand side (set equal to zero).
/// var : int
///     Variable index (0-based) to solve for.
///
/// Returns
/// -------
/// list of str
///     Each element is a LaTeX string for a symbolic root.
#[pyfunction]
pub fn solve_for_all_py(expr_str: &str, var: usize) -> PyResult<Vec<String>> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let zero = crate::LoweredOp::Const(0.0);
    let result = crate::solve_for_all(&lowered, &zero, var).map_err(map_err)?;
    Ok(result.roots.iter().map(|r| r.to_latex()).collect())
}

/// Find all complex roots of the polynomial `expr_str` in variable `var`.
///
/// Parameters
/// ----------
/// expr_str : str
///     Expression string for a polynomial expression.
/// var : int
///     Variable index (0-based) to extract the polynomial in.
///
/// Returns
/// -------
/// list of tuple[float, float]
///     Each element is `(re, im)` for a complex root.
#[pyfunction]
pub fn solve_polynomial_complex_py(expr_str: &str, var: usize) -> PyResult<Vec<(f64, f64)>> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let poly = crate::Poly::from_lowered(&lowered, var)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let complex_roots = crate::solve_poly::solve_polynomial_complex(&poly)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(complex_roots
        .roots
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect())
}
