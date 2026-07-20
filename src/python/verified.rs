//! Python bindings for verified (interval-certified) root finding and quadrature.
//!
//! Thin wrappers over [`crate::numeric_verified`]: guaranteed enclosures for
//! definite integrals, and certified root isolation via interval Newton /
//! Krawczyk.

use pyo3::prelude::*;

use crate::numeric_verified::{RootOpts, find_root_verified, integrate_definite_verified};

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Certificate for a verified root-finding result.
#[pyclass(name = "RootCertificate", from_py_object)]
#[derive(Clone)]
pub struct PyRootCertificate {
    inner: crate::RootCertificate,
}

#[pymethods]
impl PyRootCertificate {
    /// Enclosing interval `(lo, hi)` (meaningful only when
    /// `status == "unique_exists"`).
    #[getter]
    pub fn enclosure(&self) -> (f64, f64) {
        (self.inner.enclosure.lo, self.inner.enclosure.hi)
    }

    /// One of ``"unique_exists"``, ``"no_root"``, ``"indeterminate"``.
    #[getter]
    pub fn status(&self) -> &'static str {
        match self.inner.status {
            crate::RootStatus::UniqueExists => "unique_exists",
            crate::RootStatus::NoRoot => "no_root",
            crate::RootStatus::Indeterminate => "indeterminate",
        }
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!(
            "RootCertificate(status={:?}, enclosure={:?})",
            self.status(),
            self.enclosure()
        )
    }
}

/// Find a verified root of `expr_str` in `[lo, hi]` via interval Newton / Krawczyk.
///
/// `var` is the 0-based variable index to solve for; other variables (if any)
/// are held fixed at the values in `bindings`. `max_iter` caps the number of
/// Krawczyk iterations (`0` uses the built-in default of 200).
///
/// Returns a [`PyRootCertificate`] whose `status` distinguishes a *proved*
/// unique root, a *proved* absence of a root, and an indeterminate case
/// (never silently guessed).
#[pyfunction]
#[pyo3(signature = (expr_str, var, lo, hi, bindings=vec![], max_iter=0))]
pub fn find_root_verified_py(
    expr_str: &str,
    var: usize,
    lo: f64,
    hi: f64,
    bindings: Vec<f64>,
    max_iter: usize,
) -> PyResult<PyRootCertificate> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let ctx = crate::EvalCtx::new(&bindings);
    let mut opts = RootOpts::default();
    if max_iter > 0 {
        opts.max_iter = max_iter;
    }
    let cert = find_root_verified(&lowered, var, &ctx, lo, hi, &opts).map_err(map_err)?;
    Ok(PyRootCertificate { inner: cert })
}

/// Compute a verified enclosure `(lo, hi)` of `∫_a^b f(x) dx`.
///
/// `var` is the 0-based integration variable index; other variables (if any)
/// are held fixed at the values in `bindings`. `target_width` (`0` uses the
/// built-in default `1e-8`) bounds the returned enclosure's width;
/// `max_subdivisions` (`0` uses the built-in default `10000`) caps adaptive
/// bisection. The true integral is guaranteed to lie in `[lo, hi]`.
#[pyfunction]
#[pyo3(signature = (expr_str, var, a, b, bindings=vec![], target_width=0.0, max_subdivisions=0))]
pub fn integrate_definite_verified_py(
    expr_str: &str,
    var: usize,
    a: f64,
    b: f64,
    bindings: Vec<f64>,
    target_width: f64,
    max_subdivisions: usize,
) -> PyResult<(f64, f64)> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let ctx = crate::EvalCtx::new(&bindings);
    let mut opts = crate::VerifiedQuadOpts::default();
    if target_width > 0.0 {
        opts.target_width = target_width;
    }
    if max_subdivisions > 0 {
        opts.max_subdivisions = max_subdivisions;
    }
    let enclosure =
        integrate_definite_verified(&lowered, var, &ctx, a, b, &opts).map_err(map_err)?;
    Ok((enclosure.lo, enclosure.hi))
}
