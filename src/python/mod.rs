//! Python bindings for OxiEML via PyO3.
//!
//! Exposes [`PySymRegConfig`], [`PySymRegEngine`], and [`PyDiscoveredFormula`]
//! to Python, matching the Rust API in [`crate::symreg`].
//!
//! Additional utility functions for calculus, solving, and special functions
//! are provided as free Python functions in the `_core` module.
//!
//! # Usage (Python)
//! ```python
//! import numpy as np
//! import oxieml
//!
//! config = oxieml.SymRegConfig.quick()
//! engine = oxieml.SymRegEngine(config)
//!
//! X = np.column_stack([x_data])   # shape (n, n_features)
//! y = y_data                       # shape (n,)
//!
//! formulas = engine.discover(X, y)
//! for f in formulas:
//!     print(f.pretty, f.mse)
//! ```

pub mod algebra;
pub mod calculus;
pub mod discovery;
#[cfg(feature = "jit")]
pub mod jit;
pub mod numeric;
pub mod pareto;
#[cfg(feature = "smt")]
pub mod smt;
pub mod solve;
pub mod symreg;
pub mod units;
pub mod verified;

pub use symreg::PyDiscoveredFormula;
pub use symreg::PySymRegEngine;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PySymRegConfig
// ---------------------------------------------------------------------------

/// Configuration for the symbolic regression engine.
///
/// Use the class-methods `quick`, `balanced`, or `exhaustive` for
/// sensible presets, then override individual attributes as needed.
#[pyclass(name = "SymRegConfig", from_py_object)]
#[derive(Clone)]
pub struct PySymRegConfig {
    pub(crate) inner: crate::symreg::SymRegConfig,
    /// Maximum number of formulas to return from [`PySymRegEngine::discover`].
    ///
    /// The engine may find more candidates; this limits the returned slice.
    /// `0` means unlimited (return all).
    pub max_formulas: usize,
    /// When `true` and `optimizer == LevenbergMarquardt`, compute analytic
    /// parameter confidence intervals via the Laplace approximation.
    pub uq_analytic: bool,
    /// When `true`, enables OxiZ-backed UNSAT pruning for topology search.
    /// Requires the `smt` feature.
    pub smt_prune_solver: bool,
}

#[pymethods]
impl PySymRegConfig {
    /// Create a quick (shallow/fast) configuration preset.
    #[staticmethod]
    pub fn quick() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::quick(),
            max_formulas: 0,
            uq_analytic: false,
            smt_prune_solver: false,
        }
    }

    /// Create a balanced (production-default) configuration preset.
    #[staticmethod]
    pub fn balanced() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::balanced(),
            max_formulas: 0,
            uq_analytic: false,
            smt_prune_solver: false,
        }
    }

    /// Create an exhaustive (slow but thorough) configuration preset.
    #[staticmethod]
    pub fn exhaustive() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::exhaustive(),
            max_formulas: 0,
            uq_analytic: false,
            smt_prune_solver: false,
        }
    }

    /// Maximum tree depth to explore.
    #[getter]
    pub fn depth_limit(&self) -> usize {
        self.inner.max_depth
    }

    /// Set the maximum tree depth.
    #[setter]
    pub fn set_depth_limit(&mut self, v: usize) {
        self.inner.max_depth = v;
    }

    /// Maximum number of formulas to return (0 = unlimited).
    #[getter]
    pub fn get_max_formulas(&self) -> usize {
        self.max_formulas
    }

    /// Set the maximum number of formulas to return.
    #[setter]
    pub fn set_max_formulas(&mut self, v: usize) {
        self.max_formulas = v;
    }

    /// Adam optimizer iteration budget per topology.
    #[getter]
    pub fn adam_steps(&self) -> usize {
        self.inner.max_iter
    }

    /// Set the Adam optimizer iteration budget.
    #[setter]
    pub fn set_adam_steps(&mut self, v: usize) {
        self.inner.max_iter = v;
    }

    /// Optional RNG seed for reproducible runs.
    #[getter]
    pub fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    /// Set the RNG seed (`None` for non-deterministic).
    #[setter]
    pub fn set_seed(&mut self, v: Option<u64>) {
        self.inner.seed = v;
    }

    /// Enable analytic parameter uncertainty quantification (LM optimizer only).
    #[getter]
    pub fn get_uq_analytic(&self) -> bool {
        self.uq_analytic
    }

    /// Set analytic UQ flag (synced to inner config).
    #[setter]
    pub fn set_uq_analytic(&mut self, v: bool) {
        self.uq_analytic = v;
        self.inner.uq_analytic = v;
    }

    /// Enable OxiZ-backed SMT solver pruning (requires `smt` feature).
    #[getter]
    pub fn get_smt_prune_solver(&self) -> bool {
        self.smt_prune_solver
    }

    /// Set SMT pruning flag (synced to inner config).
    #[setter]
    pub fn set_smt_prune_solver(&mut self, v: bool) {
        self.smt_prune_solver = v;
        self.inner.smt_prune_solver = v;
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!(
            "SymRegConfig(depth_limit={}, adam_steps={}, max_formulas={}, uq_analytic={}, smt_prune_solver={})",
            self.inner.max_depth,
            self.inner.max_iter,
            self.max_formulas,
            self.uq_analytic,
            self.smt_prune_solver,
        )
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// Register the `oxieml._core` Python extension module.
#[pymodule]
pub fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PySymRegConfig>()?;
    m.add_class::<PyDiscoveredFormula>()?;
    m.add_class::<PySymRegEngine>()?;

    // Calculus
    m.add_function(wrap_pyfunction!(calculus::integrate_definite_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculus::limit_py, m)?)?;

    // Solving
    m.add_function(wrap_pyfunction!(solve::solve_for_all_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve::solve_polynomial_complex_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve::solve_system_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve::groebner_basis_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve::ideal_member_py, m)?)?;

    // Special / numeric functions
    m.add_function(wrap_pyfunction!(numeric::erf_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::erfc_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::lgamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::digamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::trigamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::ei_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::si_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::lambert_w0_py, m)?)?;
    m.add_function(wrap_pyfunction!(numeric::lambert_wm1_py, m)?)?;

    // ODE solving
    m.add_function(wrap_pyfunction!(symreg::dsolve_py, m)?)?;

    // Algebra: P2 rewrite rules, summation, symbolic matrices
    m.add_function(wrap_pyfunction!(algebra::expand_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::collect_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::apart_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::together_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::powsimp_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::logcombine_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::sum_indefinite_py, m)?)?;
    m.add_function(wrap_pyfunction!(algebra::sum_definite_py, m)?)?;
    m.add_class::<algebra::PyMatrix>()?;

    // Discovery: NSGA-II, SINDy, 1-D PDE
    m.add_class::<discovery::PyNsga2Config>()?;
    m.add_class::<discovery::PyRankedFormula>()?;
    m.add_function(wrap_pyfunction!(discovery::discover_nsga2_py, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::discover_sindy_py, m)?)?;
    m.add_function(wrap_pyfunction!(discovery::discover_pde_py, m)?)?;

    // Pareto-front extraction
    m.add_function(wrap_pyfunction!(pareto::discover_pareto_py, m)?)?;
    m.add_function(wrap_pyfunction!(pareto::pareto_front_py, m)?)?;

    // Dimensional (units) analysis
    m.add_function(wrap_pyfunction!(units::check_units_py, m)?)?;

    // Verified (interval-certified) root finding and quadrature
    m.add_class::<verified::PyRootCertificate>()?;
    m.add_function(wrap_pyfunction!(verified::find_root_verified_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        verified::integrate_definite_verified_py,
        m
    )?)?;

    // JIT batch evaluation (requires the `jit` feature)
    #[cfg(feature = "jit")]
    {
        m.add_class::<jit::PyJitFn>()?;
    }

    // SMT solving (requires the `smt` feature)
    #[cfg(feature = "smt")]
    {
        m.add_class::<smt::PyIncrementalSolver>()?;
        m.add_function(wrap_pyfunction!(smt::max_smt_py, m)?)?;
        m.add_function(wrap_pyfunction!(smt::minimize_unsat_core_py, m)?)?;
    }

    Ok(())
}
