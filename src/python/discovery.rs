//! Python bindings for NSGA-II multi-objective discovery, SINDy ODE discovery,
//! and 1-D PDE discovery.

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

// NSGA-II

/// Hyper-parameters of the NSGA-II generational loop.
#[pyclass(name = "Nsga2Config", from_py_object)]
#[derive(Clone)]
pub struct PyNsga2Config {
    pub(crate) inner: crate::Nsga2Config,
}

#[pymethods]
impl PyNsga2Config {
    /// Create a configuration with the library defaults.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: crate::Nsga2Config::default(),
        }
    }

    /// μ — the parent population size (λ equals μ).
    #[getter]
    pub fn population(&self) -> usize {
        self.inner.population
    }

    /// Set the parent population size.
    #[setter]
    pub fn set_population(&mut self, v: usize) {
        self.inner.population = v;
    }

    /// Number of (μ+λ) generations to run.
    #[getter]
    pub fn generations(&self) -> usize {
        self.inner.generations
    }

    /// Set the number of generations.
    #[setter]
    pub fn set_generations(&mut self, v: usize) {
        self.inner.generations = v;
    }

    /// Tournament size for crowded-comparison mating selection.
    #[getter]
    pub fn tournament_size(&self) -> usize {
        self.inner.tournament_size
    }

    /// Set the tournament size (clamped to ≥ 2 internally).
    #[setter]
    pub fn set_tournament_size(&mut self, v: usize) {
        self.inner.tournament_size = v;
    }

    /// Probability of subtree crossover, in `[0, 1]`.
    #[getter]
    pub fn crossover_rate(&self) -> f64 {
        self.inner.crossover_rate
    }

    /// Set the crossover probability.
    #[setter]
    pub fn set_crossover_rate(&mut self, v: f64) {
        self.inner.crossover_rate = v;
    }

    /// Probability of mutating an offspring, in `[0, 1]`.
    #[getter]
    pub fn mutation_rate(&self) -> f64 {
        self.inner.mutation_rate
    }

    /// Set the mutation probability.
    #[setter]
    pub fn set_mutation_rate(&mut self, v: f64) {
        self.inner.mutation_rate = v;
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!(
            "Nsga2Config(population={}, generations={}, tournament_size={}, crossover_rate={}, mutation_rate={})",
            self.inner.population,
            self.inner.generations,
            self.inner.tournament_size,
            self.inner.crossover_rate,
            self.inner.mutation_rate
        )
    }
}

impl Default for PyNsga2Config {
    fn default() -> Self {
        Self::new()
    }
}

/// A formula annotated with its NSGA-II non-domination rank and crowding distance.
#[pyclass(name = "RankedFormula", from_py_object)]
#[derive(Clone)]
pub struct PyRankedFormula {
    inner: crate::RankedFormula,
}

#[pymethods]
impl PyRankedFormula {
    /// The underlying discovered formula.
    #[getter]
    pub fn formula(&self) -> PyDiscoveredFormula {
        PyDiscoveredFormula {
            inner: self.inner.formula.clone(),
        }
    }

    /// Non-domination rank (`0` = Pareto front).
    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    /// Crowding distance within the formula's own front (may be `+inf`).
    #[getter]
    pub fn crowding(&self) -> f64 {
        self.inner.crowding
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!(
            "RankedFormula(rank={}, crowding={}, pretty={:?})",
            self.inner.rank, self.inner.crowding, self.inner.formula.pretty
        )
    }
}

/// Discover formulas via the full NSGA-II multi-objective (MSE, complexity) search.
///
/// Unlike a scalar-scored search, this returns the **whole final population**
/// annotated with `(rank, crowding)`, sorted best-first; the non-dominated front
/// is every entry with `rank == 0`.
#[pyfunction]
pub fn discover_nsga2_py<'py>(
    py: Python<'py>,
    config: &PySymRegConfig,
    nsga2_config: &PyNsga2Config,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Vec<PyRankedFormula>> {
    let (inputs, targets, n_features) = xy_to_rows(x, y)?;
    let engine = crate::SymRegEngine::new(config.inner.clone());
    let nsga2_cfg = nsga2_config.inner;
    let result = py.detach(|| engine.discover_nsga2(&inputs, &targets, n_features, &nsga2_cfg));
    Ok(result
        .map_err(map_err)?
        .into_iter()
        .map(|f| PyRankedFormula { inner: f })
        .collect())
}

// SINDy

/// One discovered SINDy equation as `(coefficients, active_terms, latex)`.
/// See [`discover_sindy_py`].
type SindyEquationPy = (Vec<f64>, Vec<String>, String);

/// Discover ODEs `dx_k/dt = Σ c_j·library_j(x)` from a multi-variate trajectory
/// via SINDy (STLSQ mode).
///
/// `trajectory[k]` is the time series of state variable `k`; `library` is a
/// list of `(label, expr_str)` candidate terms (variables `x0, x1, …` in each
/// `expr_str` refer to the state vector, in the same order as `trajectory`).
/// Returns one `(coefficients, active_terms, latex)` tuple per state variable.
#[pyfunction]
#[pyo3(signature = (trajectory, dt, library, threshold=0.1, ridge_lambda=1e-6, max_iter=20))]
pub fn discover_sindy_py(
    trajectory: Vec<Vec<f64>>,
    dt: f64,
    library: Vec<(String, String)>,
    threshold: f64,
    ridge_lambda: f64,
    max_iter: usize,
) -> PyResult<Vec<SindyEquationPy>> {
    let mut terms = Vec::with_capacity(library.len());
    for (label, expr_str) in &library {
        let tree = crate::parse(expr_str).map_err(map_err)?;
        let expr = tree.lower().simplify();
        terms.push(crate::LibraryTerm {
            label: label.clone(),
            expr,
        });
    }
    let cfg = crate::SindyConfig {
        library: terms,
        threshold,
        ridge_lambda,
        max_iter,
        mode: crate::SindyMode::Stlsq,
        seed: None,
    };
    let result = crate::discover_ode_sindy(&trajectory, dt, &cfg).map_err(map_err)?;
    Ok(result
        .equations
        .into_iter()
        .map(|eq| (eq.coefficients, eq.active_terms, eq.expr.to_latex()))
        .collect())
}

// PDE (1-D)

/// Result of [`discover_pde_py`] as `(equation_str, latex, coefficients, mse)`.
type PdeResultPy = (String, String, Vec<(String, f64)>, f64);

/// Discover a 1-D PDE `u_t = Σ c_j·library_j(u, u_x, u_xx, …)` from
/// spatiotemporal data, using the built-in 6-term library
/// `{1, u, u², u_x, u·u_x, u_xx}`.
///
/// `field[j][i] = u(x_i, t_j)`, shape `n_time × n_space`. Returns
/// `(equation_str, latex, coefficients, mse)`.
#[pyfunction]
#[pyo3(signature = (field, dx, dt, ridge_lambda=1e-5, threshold=0.01, max_iter=10))]
pub fn discover_pde_py(
    field: Vec<Vec<f64>>,
    dx: f64,
    dt: f64,
    ridge_lambda: f64,
    threshold: f64,
    max_iter: usize,
) -> PyResult<PdeResultPy> {
    let cfg = crate::PdeConfig {
        fd_accuracy: 2,
        trim_boundary: 1,
        ridge_lambda,
        threshold,
        max_iter,
        spatial_dims: None,
        max_deriv_order: None,
        library: None,
        mode: None,
    };
    // `discover_pde`'s engine parameter is reserved for API consistency and is
    // unused internally; a throwaway default engine satisfies the signature.
    let engine = crate::SymRegEngine::new(crate::SymRegConfig::default());
    let result = crate::discover_pde(&engine, &field, dx, dt, &cfg).map_err(map_err)?;
    Ok((
        result.equation,
        result.latex,
        result.coefficients,
        result.mse,
    ))
}
