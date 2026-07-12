//! The safety net: independent numeric verification of the symbolic results.
//!
//! Every symbolic answer in this module is *supposed* to be exact, and the two
//! answers that could be wrong for interesting reasons — anything resting on an
//! undecidable zero test — are flagged as [`Certainty::Conditional`](super::Certainty)
//! rather than trusted. This module is the backstop *behind* that: it evaluates the
//! matrix at seeded sample points and re-derives, by ordinary floating-point linear
//! algebra, the quantities the symbolic engine claims.
//!
//! It checks, at each probe point where the matrix evaluates to finite numbers:
//!
//! | claim | independent check |
//! |---|---|
//! | `det` (Bareiss) | LU with partial pivoting on the evaluated matrix |
//! | `charpoly` (Faddeev–LeVerrier) | Cayley–Hamilton: `Σ cₖ Aᵏ` must vanish |
//! | `inverse` | `A · A⁻¹ − I` must vanish |
//! | `nullspace` | `A · v` must vanish for each basis vector `v` |
//!
//! A disagreement means one of: a bug, or an assumption that did not hold. Either
//! way the caller learns about it here rather than downstream. Note the asymmetry
//! that makes this a *net* and not a *proof*: agreement at finitely many points
//! cannot establish a symbolic identity, but disagreement at even one point
//! definitively refutes it.

use crate::lower::LoweredOp;

use super::zero::ZeroOracle;
use super::{Matrix, MatrixError};

/// Relative tolerance for every numeric comparison in the report.
///
/// Generous by floating-point standards: the symbolic expressions are exact, so any
/// discrepancy larger than accumulated round-off is a real disagreement, and we want
/// no false alarms from a badly conditioned probe point.
pub const VERIFY_TOLERANCE: f64 = 1e-6;

/// The outcome of [`Matrix::verify`].
///
/// Each field is `None` when the corresponding check did not apply (a rectangular
/// matrix has no determinant; a singular matrix has no inverse; a full-rank matrix
/// has an empty nullspace).
#[derive(Clone, Debug, PartialEq)]
pub struct VerifyReport {
    /// How many probe points evaluated to finite numbers and were actually used.
    pub probes_used: usize,
    /// Largest relative disagreement between the symbolic determinant and an
    /// independent numeric LU determinant.
    pub det_max_rel_error: Option<f64>,
    /// Largest entry of `Σ cₖ Aᵏ` — the Cayley–Hamilton residual — relative to the
    /// matrix norm.
    pub cayley_hamilton_max_rel_error: Option<f64>,
    /// Largest entry of `A · A⁻¹ − I`.
    pub inverse_max_abs_error: Option<f64>,
    /// Largest entry of `A · v` over the nullspace basis, relative to `‖A‖·‖v‖`.
    pub nullspace_max_rel_error: Option<f64>,
    /// The tolerance every check was held to.
    pub tolerance: f64,
}

impl VerifyReport {
    /// `true` when every applicable check came in under [`VerifyReport::tolerance`].
    ///
    /// A report with no usable probes is **not** ok: it verified nothing.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        if self.probes_used == 0 {
            return false;
        }
        let within = |v: Option<f64>| v.is_none_or(|e| e.is_finite() && e <= self.tolerance);
        within(self.det_max_rel_error)
            && within(self.cayley_hamilton_max_rel_error)
            && within(self.inverse_max_abs_error)
            && within(self.nullspace_max_rel_error)
    }
}

impl std::fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let show = |v: Option<f64>| match v {
            None => "n/a".to_string(),
            Some(e) => format!("{e:.3e}"),
        };
        write!(
            f,
            "verify[{} probes, tol {:.0e}]: det {} | Cayley–Hamilton {} | inverse {} | nullspace {} → {}",
            self.probes_used,
            self.tolerance,
            show(self.det_max_rel_error),
            show(self.cayley_hamilton_max_rel_error),
            show(self.inverse_max_abs_error),
            show(self.nullspace_max_rel_error),
            if self.is_ok() { "OK" } else { "FAILED" }
        )
    }
}

/// Evaluate an expression at `point`, padding the point when the expression
/// mentions variables the matrix as a whole does not.
fn eval_expr(expr: &LoweredOp, point: &[f64]) -> f64 {
    let needed = expr.count_vars();
    if point.len() >= needed {
        return expr.eval(point);
    }
    let mut padded = point.to_vec();
    padded.resize(needed, 0.0);
    expr.eval(&padded)
}

/// Determinant of a row-major `n × n` `f64` matrix by LU with partial pivoting.
///
/// Independent of everything symbolic in this module — that is the whole point.
fn det_lu(a: &[f64], n: usize) -> f64 {
    let mut m = a.to_vec();
    let mut det = 1.0f64;
    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = m[k * n + k].abs();
        for r in k + 1..n {
            let candidate = m[r * n + k].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = r;
            }
        }
        if pivot_abs == 0.0 {
            return 0.0;
        }
        if pivot_row != k {
            for j in 0..n {
                m.swap(k * n + j, pivot_row * n + j);
            }
            det = -det;
        }
        let pivot = m[k * n + k];
        det *= pivot;
        for i in k + 1..n {
            let factor = m[i * n + k] / pivot;
            for j in k..n {
                let update = factor * m[k * n + j];
                m[i * n + j] -= update;
            }
        }
    }
    det
}

/// Row-major `f64` matrix product.
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Largest absolute entry.
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |acc, x| acc.max(x.abs()))
}

/// Run every applicable check.
pub(crate) fn verify(m: &Matrix, oracle: &ZeroOracle) -> Result<VerifyReport, MatrixError> {
    m.check_cap()?;
    let n_vars = m.n_vars();
    let square = m.is_square();
    let n = m.rows;

    // Symbolic quantities, computed once.
    let det_expr = if square { Some(m.det()?) } else { None };
    let charpoly = if square { Some(m.charpoly()?) } else { None };
    let inverse = if square {
        match m.inverse_assuming_nonsingular(oracle) {
            Ok(certified) => Some(certified.value),
            // A proved-singular matrix simply has no inverse to check.
            Err(MatrixError::SingularMatrix) => None,
            Err(e) => return Err(e),
        }
    } else {
        None
    };
    let nullspace = m.nullspace_certified(oracle)?.value;

    let mut probes_used = 0usize;
    let mut det_error: Option<f64> = None;
    let mut charpoly_error: Option<f64> = None;
    let mut inverse_error: Option<f64> = None;
    let mut nullspace_error: Option<f64> = None;

    let accumulate = |slot: &mut Option<f64>, value: f64| {
        *slot = Some(slot.map_or(value, |current: f64| current.max(value)));
    };

    for probe in 0..oracle.probes() {
        let point = oracle.probe_point(probe, n_vars);
        let a = m.eval_at(&point);
        if a.iter().any(|x| !x.is_finite()) {
            continue;
        }
        probes_used += 1;
        let scale = max_abs(&a).max(1.0);

        if square {
            if let Some(det_expr) = &det_expr {
                let symbolic = eval_expr(det_expr, &point);
                let numeric = det_lu(&a, n);
                if symbolic.is_finite() {
                    let error = (symbolic - numeric).abs() / (1.0 + numeric.abs());
                    accumulate(&mut det_error, error);
                }
            }

            if let Some(charpoly) = &charpoly {
                // Σ cₖ Aᵏ must vanish (Cayley–Hamilton).
                let mut residual = vec![0.0f64; n * n];
                let mut power: Vec<f64> = {
                    let mut identity = vec![0.0f64; n * n];
                    for i in 0..n {
                        identity[i * n + i] = 1.0;
                    }
                    identity
                };
                let mut usable = true;
                for c in &charpoly.coeffs {
                    let value = eval_expr(c, &point);
                    if !value.is_finite() {
                        usable = false;
                        break;
                    }
                    for (slot, p) in residual.iter_mut().zip(power.iter()) {
                        *slot += value * p;
                    }
                    power = mat_mul(&a, &power, n);
                }
                if usable {
                    let denominator = scale.powi(n as i32).max(1.0);
                    accumulate(&mut charpoly_error, max_abs(&residual) / denominator);
                }
            }

            if let Some(inverse) = &inverse {
                let inv = inverse.eval_at(&point);
                if inv.iter().all(|x| x.is_finite()) {
                    let mut product = mat_mul(&a, &inv, n);
                    for i in 0..n {
                        product[i * n + i] -= 1.0;
                    }
                    accumulate(&mut inverse_error, max_abs(&product));
                }
            }
        }

        for basis_vector in &nullspace {
            let v: Vec<f64> = basis_vector.iter().map(|e| eval_expr(e, &point)).collect();
            if v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
            let mut worst = 0.0f64;
            for i in 0..m.rows {
                let mut acc = 0.0;
                for j in 0..m.cols {
                    acc += a[i * m.cols + j] * v[j];
                }
                worst = worst.max(acc.abs());
            }
            accumulate(&mut nullspace_error, worst / (scale * v_norm));
        }
    }

    Ok(VerifyReport {
        probes_used,
        det_max_rel_error: det_error,
        cayley_hamilton_max_rel_error: charpoly_error,
        inverse_max_abs_error: inverse_error,
        nullspace_max_rel_error: nullspace_error,
        tolerance: VERIFY_TOLERANCE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_matrix_verifies() {
        let m =
            Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
        let report = m.verify().expect("verify");
        assert!(report.is_ok(), "{report}");
        assert!(report.det_max_rel_error.is_some());
        assert!(report.inverse_max_abs_error.is_some());
    }

    #[test]
    fn symbolic_matrix_verifies() {
        let m = Matrix::of_vars(3, 3).expect("dims");
        let report = m.verify().expect("verify");
        assert!(report.is_ok(), "{report}");
    }

    #[test]
    fn singular_matrix_has_no_inverse_to_check_but_still_verifies() {
        let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 2.0, 4.0]).expect("dims");
        let report = m.verify().expect("verify");
        assert_eq!(report.inverse_max_abs_error, None);
        assert!(report.nullspace_max_rel_error.is_some(), "{report}");
        assert!(report.is_ok(), "{report}");
    }
}
