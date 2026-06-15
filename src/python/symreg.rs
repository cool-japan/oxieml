//! Python bindings for symbolic regression and ODE solving.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn map_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// PyDiscoveredFormula
// ---------------------------------------------------------------------------

/// A symbolic formula discovered by the regression engine.
#[pyclass(name = "DiscoveredFormula", from_py_object)]
#[derive(Clone)]
pub struct PyDiscoveredFormula {
    pub(crate) inner: crate::symreg::DiscoveredFormula,
}

#[pymethods]
impl PyDiscoveredFormula {
    /// Human-readable string representation of the formula.
    #[getter]
    pub fn pretty(&self) -> &str {
        &self.inner.pretty
    }

    /// Mean squared error on the training data.
    #[getter]
    pub fn mse(&self) -> f64 {
        self.inner.mse
    }

    /// Tree node count used as a complexity measure.
    #[getter]
    pub fn complexity(&self) -> usize {
        self.inner.complexity
    }

    /// Combined score: `mse + complexity_penalty * complexity`.
    #[getter]
    pub fn score(&self) -> f64 {
        self.inner.score
    }

    /// Cross-validated MSE, or `None` when CV was not enabled.
    #[getter]
    pub fn cv_mse(&self) -> Option<f64> {
        self.inner.cv_mse
    }

    /// Convert the formula to a LaTeX math expression.
    pub fn to_latex(&self) -> String {
        self.inner.to_latex()
    }

    /// Evaluate the formula at the given variable values.
    ///
    /// `xs` must contain at least as many elements as the number of distinct
    /// variables referenced in the formula.
    pub fn eval(&self, xs: Vec<f64>) -> PyResult<f64> {
        let lowered = self.inner.eml_tree.lower().simplify();
        let n_vars = lowered.count_vars();
        if xs.len() < n_vars {
            return Err(PyValueError::new_err(format!(
                "formula references {} variable(s) but xs has only {} element(s)",
                n_vars,
                xs.len()
            )));
        }
        Ok(lowered.eval(&xs))
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!(
            "DiscoveredFormula(pretty={:?}, mse={:.6}, complexity={})",
            self.inner.pretty, self.inner.mse, self.inner.complexity
        )
    }
}

// ---------------------------------------------------------------------------
// PySymRegEngine
// ---------------------------------------------------------------------------

/// Symbolic regression engine.
///
/// Construct with a `SymRegConfig` and call `discover` with NumPy arrays.
#[pyclass(name = "SymRegEngine")]
pub struct PySymRegEngine {
    pub(crate) config: super::PySymRegConfig,
}

#[pymethods]
impl PySymRegEngine {
    /// Create a new engine from the given configuration.
    #[new]
    pub fn new(config: &super::PySymRegConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Discover symbolic formulas from data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray, shape (n_samples, n_features), dtype float64
    ///     Input feature matrix.
    /// y : numpy.ndarray, shape (n_samples,), dtype float64
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// list of DiscoveredFormula, sorted best-first by score.
    pub fn discover<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Vec<PyDiscoveredFormula>> {
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

        // Convert from row-major matrix to per-row sample vectors.
        let mut inputs: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut row = Vec::with_capacity(n_features);
            for j in 0..n_features {
                let val = x_arr.get((i, j)).copied().ok_or_else(|| {
                    PyValueError::new_err(format!("index ({i},{j}) out of bounds"))
                })?;
                row.push(val);
            }
            inputs.push(row);
        }

        let targets: Vec<f64> = y_arr.iter().copied().collect();
        let engine = crate::symreg::SymRegEngine::new(self.config.inner.clone());
        let max_formulas = self.config.max_formulas;

        // Release the GIL during the compute-intensive discovery pass.
        let result = py.detach(|| engine.discover(&inputs, &targets, n_features));

        let mut formulas = result.map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Optionally truncate to requested number of formulas.
        if max_formulas > 0 && formulas.len() > max_formulas {
            formulas.truncate(max_formulas);
        }

        formulas
            .into_iter()
            .map(|f| Ok(PyDiscoveredFormula { inner: f }))
            .collect()
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!(
            "SymRegEngine(depth_limit={}, adam_steps={})",
            self.config.inner.max_depth, self.config.inner.max_iter
        )
    }
}

// ---------------------------------------------------------------------------
// dsolve_py
// ---------------------------------------------------------------------------

/// Symbolically solve an ODE.
///
/// The ODE is expressed as a residual `expr_str = 0` with variable slots:
/// - `x_var`: independent variable slot
/// - `y_var`: dependent variable slot
/// - `dy_var`: first derivative y′ slot
/// - `d2y_var`: second derivative y″ slot
/// - `c_start`: first free constant slot (must be ≥ max of above + 1)
///
/// Returns a tuple `(solution_latex, kind_str)` where:
/// - `solution_latex` is the LaTeX of the solution expression, or `"Unsolved"`
/// - `kind_str` identifies the family (e.g. `"Separable"`, `"FirstOrderLinear"`)
#[pyfunction]
pub fn dsolve_py(
    expr_str: &str,
    x_var: usize,
    y_var: usize,
    dy_var: usize,
    d2y_var: usize,
    c_start: usize,
) -> PyResult<(String, String)> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let form = crate::OdeForm {
        x: x_var,
        y: y_var,
        dy: dy_var,
        d2y: d2y_var,
        c_start,
    };
    let (sol, kind) = crate::dsolve(&lowered, &form);
    let sol_latex = match sol {
        crate::OdeSolution::Explicit(expr) => expr.to_latex(),
        crate::OdeSolution::Implicit(expr) => expr.to_latex(),
        crate::OdeSolution::Unsolved => "Unsolved".to_owned(),
    };
    let kind_str = format!("{kind:?}");
    Ok((sol_latex, kind_str))
}
