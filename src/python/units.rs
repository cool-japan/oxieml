//! Python bindings for dimensional (units) analysis.
//!
//! Thin wrapper over [`crate::LoweredOp::check_units`] and [`crate::Units`].

use pyo3::prelude::*;

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Check the dimensional consistency of `expr_str` and infer its output units.
///
/// `var_units[i]` gives the physical units of variable `i` as a 7-element
/// integer-exponent tuple `(m, kg, s, A, K, mol, cd)` — length, mass, time,
/// electric current, thermodynamic temperature, amount of substance, and
/// luminous intensity, matching SI base dimension order. Pass an empty list
/// when `expr_str` has no variables.
///
/// Returns the output units in the same 7-exponent form.
///
/// Raises `ValueError` when a dimensional rule is violated (e.g. adding metres
/// to seconds, or a transcendental applied to a non-dimensionless argument),
/// or when the result has a non-integer exponent (e.g. `sqrt` of a unit with
/// an odd exponent).
#[pyfunction]
pub fn check_units_py(expr_str: &str, var_units: Vec<[i8; 7]>) -> PyResult<[i8; 7]> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let units: Vec<crate::Units> = var_units
        .into_iter()
        .map(crate::Units::from_int_exps)
        .collect();
    let result = lowered.check_units(&units).map_err(map_err)?;
    result.try_into_int_exps().ok_or_else(|| {
        map_err("resulting unit has a non-integer exponent; cannot represent as an int8 tuple")
    })
}
