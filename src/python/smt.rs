//! Python bindings for SMT solving (`smt` feature).
//!
//! Thin wrappers over [`crate::IncrementalEmlSolver`], [`crate::smt::maxsmt`],
//! and [`crate::smt::core`]. Constraints are expressed as `(expr_str, op)`
//! pairs, meaning `expr_str <op> 0` with `op` one of ``"eq"``, ``"ne"``,
//! ``"gt"``, ``"ge"``, ``"lt"``, ``"le"``.

use pyo3::prelude::*;

use crate::{
    EmlConstraint, IncrementalEmlSolver, MaxSmtOptimality, MaxSmtOutcome, SmtResult, SoftConstraint,
};

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Parse one `(expr_str, op)` pair into an [`EmlConstraint`].
fn parse_constraint(expr_str: &str, op: &str) -> PyResult<EmlConstraint> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    Ok(match op {
        "eq" => EmlConstraint::EqZero(tree),
        "ne" => EmlConstraint::NeZero(tree),
        "gt" => EmlConstraint::GtZero(tree),
        "ge" => EmlConstraint::GeZero(tree),
        "lt" => EmlConstraint::LtZero(tree),
        "le" => EmlConstraint::LeZero(tree),
        other => {
            return Err(map_err(format!(
                "unknown comparison op {other:?}; expected one of: eq, ne, gt, ge, lt, le"
            )));
        }
    })
}

/// Parse a list of `(expr_str, op)` pairs into [`EmlConstraint`]s.
fn parse_constraints(items: &[(String, String)]) -> PyResult<Vec<EmlConstraint>> {
    items
        .iter()
        .map(|(expr, op)| parse_constraint(expr, op))
        .collect()
}

/// Convert an [`SmtResult`] to `(status, witness)` for Python.
fn smt_result_to_py(result: SmtResult) -> (String, Option<Vec<f64>>) {
    match result {
        SmtResult::Sat(sol) => ("sat".to_string(), Some(sol.assignments)),
        SmtResult::Unsat => ("unsat".to_string(), None),
        SmtResult::Unknown => ("unknown".to_string(), None),
    }
}

/// Incremental EML SMT solver (push/pop scoped) over box-bounded variables.
#[pyclass(name = "IncrementalSolver")]
pub struct PyIncrementalSolver {
    inner: IncrementalEmlSolver,
}

#[pymethods]
impl PyIncrementalSolver {
    /// Create a solver over per-variable bounds `[(lo, hi), …]`.
    ///
    /// `relaxation_samples` is the number of tangent-sample points used for the
    /// `exp`/`ln` linear relaxation (clamped to ≥ 1); `recycle_after` rebuilds
    /// the underlying solver after this many scopes (`None` uses the crate's
    /// default recycling interval, `0` disables recycling entirely).
    #[new]
    #[pyo3(signature = (bounds, relaxation_samples=3, recycle_after=None))]
    pub fn new(
        bounds: Vec<(f64, f64)>,
        relaxation_samples: usize,
        recycle_after: Option<usize>,
    ) -> Self {
        let recycle = recycle_after.unwrap_or(crate::DEFAULT_RECYCLE_AFTER);
        Self {
            inner: IncrementalEmlSolver::with_config(bounds, relaxation_samples, recycle),
        }
    }

    /// Check satisfiability of one constraint `(expr_str, op)`, scoped push/pop.
    ///
    /// Returns `(status, witness)`: `status` is one of ``"sat"``, ``"unsat"``,
    /// ``"unknown"``; `witness` is the satisfying assignment when `status ==
    /// "sat"`, else `None`.
    pub fn check_sat(&mut self, expr_str: &str, op: &str) -> PyResult<(String, Option<Vec<f64>>)> {
        let c = parse_constraint(expr_str, op)?;
        let result = self.inner.check_sat(&c).map_err(map_err)?;
        Ok(smt_result_to_py(result))
    }

    /// Check satisfiability of the conjunction of `constraints` (each an
    /// `(expr_str, op)` pair). Same return convention as [`Self::check_sat`].
    pub fn check_all(
        &mut self,
        constraints: Vec<(String, String)>,
    ) -> PyResult<(String, Option<Vec<f64>>)> {
        let cs = parse_constraints(&constraints)?;
        let result = self.inner.check_all(&cs).map_err(map_err)?;
        Ok(smt_result_to_py(result))
    }
}

/// Result of [`max_smt_py`] as `(outcome, witness, satisfied_indices, weight,
/// optimality)`.
type MaxSmtResultPy = (String, Option<Vec<f64>>, Vec<usize>, u64, String);

/// Solve a MaxSMT instance: maximize satisfied soft-constraint weight subject
/// to hard constraints.
///
/// `hard`/`soft` are lists of `(expr_str, op)` pairs; `weights[i]` is the
/// weight of `soft[i]` (`weights.len()` must equal `soft.len()`).
///
/// Returns `(outcome, witness, satisfied_indices, weight, optimality)`:
/// - `outcome`: ``"solved"``, ``"unsat"``, or ``"unknown"``.
/// - `optimality` (meaningful only when `outcome == "solved"`): ``"optimal"``
///   (proved maximum), ``"maximal"`` (nothing can be added, but a heavier
///   selection may exist), or ``"feasible"`` (neither could be established).
#[pyfunction]
pub fn max_smt_py(
    bounds: Vec<(f64, f64)>,
    relaxation_samples: usize,
    hard: Vec<(String, String)>,
    soft: Vec<(String, String)>,
    weights: Vec<u64>,
) -> PyResult<MaxSmtResultPy> {
    if weights.len() != soft.len() {
        return Err(map_err("weights.len() must equal soft.len()"));
    }
    let hard_c = parse_constraints(&hard)?;
    let soft_c: Vec<SoftConstraint> = parse_constraints(&soft)?
        .into_iter()
        .zip(weights)
        .map(|(c, w)| SoftConstraint::new(c, w))
        .collect();
    let outcome = crate::smt::maxsmt::max_smt(&bounds, relaxation_samples, &hard_c, &soft_c)
        .map_err(map_err)?;
    Ok(match outcome {
        MaxSmtOutcome::Solved(sol) => {
            let optimality = match sol.optimality {
                MaxSmtOptimality::Optimal => "optimal",
                MaxSmtOptimality::Maximal => "maximal",
                MaxSmtOptimality::Feasible => "feasible",
            };
            (
                "solved".to_string(),
                Some(sol.assignments),
                sol.satisfied,
                sol.weight,
                optimality.to_string(),
            )
        }
        MaxSmtOutcome::Unsat => ("unsat".to_string(), None, Vec::new(), 0, "n/a".to_string()),
        MaxSmtOutcome::Unknown => (
            "unknown".to_string(),
            None,
            Vec::new(),
            0,
            "n/a".to_string(),
        ),
    })
}

/// Extract a minimal (irreducible) unsatisfiable subset of `constraints` (each
/// an `(expr_str, op)` pair).
///
/// Returns `None` when the conjunction is not provably unsatisfiable;
/// otherwise `(indices, verified_minimal)`, where `indices` index into
/// `constraints` and `verified_minimal` is `True` only when every deletion
/// trial returned a definite `Sat` (a genuine minimal unsatisfiable subset,
/// rather than minimal-relative-to-an-incomplete-oracle).
#[pyfunction]
pub fn minimize_unsat_core_py(
    bounds: Vec<(f64, f64)>,
    relaxation_samples: usize,
    constraints: Vec<(String, String)>,
) -> PyResult<Option<(Vec<usize>, bool)>> {
    let cs = parse_constraints(&constraints)?;
    let core =
        crate::smt::core::minimize_unsat_core(&bounds, relaxation_samples, &cs).map_err(map_err)?;
    Ok(core.map(|c| (c.indices, c.verified_minimal)))
}
