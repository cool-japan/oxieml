//! Multivariate Newton solver for systems of equations.
//!
//! Provides damped Newton iteration with Armijo backtracking line search
//! for solving `F(x) = 0` given a system of `LoweredOp` expressions.

use crate::error::EmlError;
use crate::linalg;
use crate::lower::LoweredOp;

/// Options for the multivariate Newton solver.
#[derive(Clone, Copy, Debug)]
pub struct SystemOpts {
    /// Maximum Newton iterations. Default: 100.
    pub max_iter: usize,
    /// Convergence tolerance (‖F(x)‖ < tol). Default: 1e-10.
    pub tol: f64,
    /// Maximum Armijo step halvings. Default: 20.
    pub max_halvings: usize,
}

impl Default for SystemOpts {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-10,
            max_halvings: 20,
        }
    }
}

/// Evaluate the system F at point x.
fn eval_system(fs: &[LoweredOp], x: &[f64]) -> Vec<f64> {
    fs.iter().map(|f| f.eval(x)).collect()
}

/// Squared norm ‖v‖².
fn norm_sq(v: &[f64]) -> f64 {
    v.iter().map(|&vi| vi * vi).sum()
}

/// Solve the nonlinear system F(x) = 0 using damped Newton with Armijo backtracking.
///
/// Builds the symbolic Jacobian once, then re-evaluates it at each Newton step.
/// Solves J·Δ = -F via LU decomposition. Uses Armijo condition for step acceptance.
pub fn solve_system_newton(
    fs: &[LoweredOp],
    x0: &[f64],
    opts: SystemOpts,
) -> Result<Vec<f64>, EmlError> {
    let n = fs.len();
    if n == 0 {
        return Err(EmlError::InvalidParameter(
            "system must have at least one equation",
        ));
    }
    if x0.len() != n {
        return Err(EmlError::DimensionMismatch(n, x0.len()));
    }

    // Build symbolic Jacobian (n rows × n cols)
    // jac[i * n + j] = ∂f_i/∂x_j
    let jac_exprs: Vec<Vec<LoweredOp>> = fs.iter().map(|fi| fi.jacobian(n)).collect();

    let mut x = x0.to_vec();

    for iter in 0..opts.max_iter {
        let f_val = eval_system(fs, &x);
        let f_norm_sq = norm_sq(&f_val);

        if f_norm_sq.sqrt() < opts.tol {
            return Ok(x);
        }

        if f_norm_sq.is_nan() {
            return Err(EmlError::NanEncountered);
        }

        // Build Jacobian matrix (row-major)
        let mut jac_mat: Vec<f64> = jac_exprs
            .iter()
            .flat_map(|row| row.iter().map(|e| e.eval(&x)))
            .collect();

        // RHS: -F(x)
        let mut rhs: Vec<f64> = f_val.iter().map(|&v| -v).collect();

        // Solve J·Δ = -F for the Newton step Δ
        linalg::solve_lu(&mut jac_mat, &mut rhs, n)?;
        let delta = rhs; // delta now contains Δx

        // Armijo backtracking: find step α s.t.
        // ‖F(x + α·Δ)‖² ≤ ‖F(x)‖² · (1 - 0.01·α)
        let mut alpha = 1.0_f64;
        let armijo_c = 0.01;
        let mut accepted = false;

        for _ in 0..opts.max_halvings {
            let x_new: Vec<f64> = x
                .iter()
                .zip(delta.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            let f_new = eval_system(fs, &x_new);
            let f_new_norm_sq = norm_sq(&f_new);

            if f_new_norm_sq <= f_norm_sq * (1.0 - armijo_c * alpha) {
                x = x_new;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            // Take the step anyway with smallest alpha (avoids stagnation)
            let x_new: Vec<f64> = x
                .iter()
                .zip(delta.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            x = x_new;
        }

        // Check convergence after step
        let f_after = eval_system(fs, &x);
        if norm_sq(&f_after).sqrt() < opts.tol {
            return Ok(x);
        }

        let _ = iter; // suppress lint
    }

    // Check final residual
    let f_final = eval_system(fs, &x);
    if norm_sq(&f_final).sqrt() < opts.tol {
        return Ok(x);
    }

    Err(EmlError::NonConvergence {
        method: "solve_system_newton",
        iterations: opts.max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::LoweredOp;
    use std::sync::Arc;

    #[test]
    fn test_newton_circle_line() {
        // Solve: x² + y² = 1, x - y = 0 → (1/√2, 1/√2) or (-1/√2, -1/√2)
        // f0 = x0² + x1² - 1
        let f0 = LoweredOp::Sub(
            Arc::new(LoweredOp::Add(
                Arc::new(LoweredOp::Pow(
                    Arc::new(LoweredOp::Var(0)),
                    Arc::new(LoweredOp::Const(2.0)),
                )),
                Arc::new(LoweredOp::Pow(
                    Arc::new(LoweredOp::Var(1)),
                    Arc::new(LoweredOp::Const(2.0)),
                )),
            )),
            Arc::new(LoweredOp::Const(1.0)),
        );
        // f1 = x0 - x1
        let f1 = LoweredOp::Sub(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
        let x0 = vec![0.5, 0.5]; // initial guess near (1/√2, 1/√2)
        let sol = solve_system_newton(&[f0, f1], &x0, SystemOpts::default()).unwrap();
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!((sol[0] - expected).abs() < 1e-8, "x={}", sol[0]);
        assert!((sol[1] - expected).abs() < 1e-8, "y={}", sol[1]);
    }
}
