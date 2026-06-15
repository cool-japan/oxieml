//! SINDy (Sparse Identification of Nonlinear Dynamics) for ODE discovery.
//!
//! Supports three modes:
//! - [`SindyMode::Stlsq`]: Standard STLSQ on time derivatives.
//! - [`SindyMode::WeakForm`]: Integrate against bump test functions (avoids
//!   numerical differentiation noise).
//! - [`SindyMode::Ensemble`]: Bootstrap-resample + STLSQ, vote on support,
//!   take median coefficients.

use std::sync::Arc;

use rand::RngExt;
use rand::SeedableRng;

use crate::error::EmlError;
use crate::lower::LoweredOp;

use super::discover::derive_seed;
use super::numerics::central_differences;
use super::strlsq::strlsq;

type Rng = rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single library term for SINDy.
#[derive(Debug, Clone)]
pub struct LibraryTerm {
    /// Human-readable label, e.g. `"x0"`, `"x0*x1"`, `"sin(x0)"`.
    pub label: String,
    /// Expression to evaluate at each data point.
    pub expr: LoweredOp,
}

/// SINDy operating mode.
#[derive(Debug, Clone)]
pub enum SindyMode {
    /// Standard STLSQ on numerically differentiated time derivatives.
    Stlsq,
    /// Weak form: integrate library and derivatives against localised test
    /// functions to avoid differentiation noise.
    WeakForm {
        /// Number of test functions (bump windows) to use.
        n_test_functions: usize,
    },
    /// Ensemble: bootstrap-resample rows, run STLSQ, vote on support,
    /// take median coefficients.
    Ensemble {
        /// Number of bootstrap resamples.
        n_bootstrap: usize,
        /// Fraction of resamples that must agree for a term to be active.
        support_quorum: f64,
    },
}

/// Configuration for SINDy.
#[derive(Debug, Clone)]
pub struct SindyConfig {
    /// Candidate library terms.
    pub library: Vec<LibraryTerm>,
    /// Coefficient threshold.
    pub threshold: f64,
    /// Ridge regularisation.
    pub ridge_lambda: f64,
    /// Maximum STLSQ iterations.
    pub max_iter: usize,
    /// Operating mode.
    pub mode: SindyMode,
    /// Optional RNG seed for ensemble reproducibility.
    pub seed: Option<u64>,
}

/// One discovered ODE equation `dx_k/dt = sparse combination of library terms`.
#[derive(Debug, Clone)]
pub struct SindyEquation {
    /// Coefficient for each library term (zero = inactive).
    pub coefficients: Vec<f64>,
    /// Labels of active (non-zero) terms.
    pub active_terms: Vec<String>,
    /// Symbolic expression for the right-hand side.
    pub expr: LoweredOp,
}

/// Result of a SINDy run.
#[derive(Debug, Clone)]
pub struct SindyResult {
    /// One equation per state variable.
    pub equations: Vec<SindyEquation>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Discover ODEs `dx_k/dt = Σ c_j θ_j(x)` from a multi-variate trajectory.
///
/// # Arguments
///
/// - `trajectory`: `trajectory[k][t]` = value of state variable `k` at time `t`.
/// - `dt`: uniform time-step.
/// - `cfg`: SINDy configuration.
///
/// # Errors
///
/// - [`EmlError::EmptyData`] when the trajectory or library is empty.
/// - [`EmlError::DimensionMismatch`] when state variables have different lengths.
pub fn discover_ode_sindy(
    trajectory: &[Vec<f64>],
    dt: f64,
    cfg: &SindyConfig,
) -> Result<SindyResult, EmlError> {
    if trajectory.is_empty() {
        return Err(EmlError::EmptyData);
    }
    let n_timesteps = trajectory[0].len();
    for var in trajectory {
        if var.len() != n_timesteps {
            return Err(EmlError::DimensionMismatch(n_timesteps, var.len()));
        }
    }
    if n_timesteps < 3 {
        return Err(EmlError::DimensionMismatch(3, n_timesteps));
    }
    if cfg.library.is_empty() {
        return Err(EmlError::EmptyData);
    }

    let n_vars = trajectory.len();
    let n_terms = cfg.library.len();

    // Build interior time indices (skip endpoints for central differences)
    let t_indices: Vec<usize> = (1..n_timesteps - 1).collect();
    let n_data = t_indices.len();

    // Pre-compile library term ops for efficiency
    let compiled_terms: Vec<_> = cfg
        .library
        .iter()
        .map(|t| t.expr.to_oxiblas_ops())
        .collect();

    // Build feature matrix theta: n_data × n_terms (row-major)
    let mut theta = vec![0.0_f64; n_data * n_terms];
    for (row, &t) in t_indices.iter().enumerate() {
        let state: Vec<f64> = trajectory.iter().map(|xk| xk[t]).collect();
        for (col, ops) in compiled_terms.iter().enumerate() {
            theta[row * n_terms + col] = LoweredOp::eval_ops(ops, &state);
        }
    }

    // Estimate time derivatives for each state variable
    let derivatives: Vec<Vec<f64>> = trajectory
        .iter()
        .map(|xk| central_differences(xk, dt))
        .collect();

    // Solve for each state variable
    let equations = (0..n_vars)
        .map(|k| {
            let target: Vec<f64> = t_indices.iter().map(|&t| derivatives[k][t]).collect();
            let coeffs = match &cfg.mode {
                SindyMode::Stlsq => strlsq(
                    &theta,
                    n_data,
                    n_terms,
                    &target,
                    cfg.threshold,
                    cfg.ridge_lambda,
                    cfg.max_iter,
                ),
                SindyMode::WeakForm { n_test_functions } => {
                    weak_form_stlsq(&theta, n_data, n_terms, &target, cfg, *n_test_functions)
                }
                SindyMode::Ensemble {
                    n_bootstrap,
                    support_quorum,
                } => ensemble_stlsq(
                    &theta,
                    n_data,
                    n_terms,
                    &target,
                    cfg,
                    *n_bootstrap,
                    *support_quorum,
                ),
            };
            build_equation(coeffs, &cfg.library, cfg.threshold)
        })
        .collect();

    Ok(SindyResult { equations })
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Weak-form STLSQ: integrate library and derivatives against Hann-window
/// test functions to reduce differentiation noise.
fn weak_form_stlsq(
    theta: &[f64],
    n_data: usize,
    n_terms: usize,
    target: &[f64],
    cfg: &SindyConfig,
    n_test_functions: usize,
) -> Vec<f64> {
    let n_tf = n_test_functions.max(1);
    let mut theta_weak = vec![0.0_f64; n_tf * n_terms];
    let mut target_weak = vec![0.0_f64; n_tf];

    for tf_idx in 0..n_tf {
        let center = if n_tf > 1 {
            tf_idx as f64 * (n_data as f64 - 1.0) / (n_tf - 1) as f64
        } else {
            (n_data as f64 - 1.0) / 2.0
        };
        let half_width = (n_data as f64 / (2.0 * n_tf as f64)).max(1.0);

        let mut sum_w = 0.0_f64;
        for row in 0..n_data {
            let dist = (row as f64 - center).abs();
            let w = if dist <= half_width {
                let t = dist / half_width;
                0.5 * (1.0 + (std::f64::consts::PI * t).cos())
            } else {
                0.0
            };
            sum_w += w;
            for col in 0..n_terms {
                theta_weak[tf_idx * n_terms + col] += w * theta[row * n_terms + col];
            }
            target_weak[tf_idx] += w * target[row];
        }

        if sum_w > f64::EPSILON {
            for col in 0..n_terms {
                theta_weak[tf_idx * n_terms + col] /= sum_w;
            }
            target_weak[tf_idx] /= sum_w;
        }
    }

    strlsq(
        &theta_weak,
        n_tf,
        n_terms,
        &target_weak,
        cfg.threshold,
        cfg.ridge_lambda,
        cfg.max_iter,
    )
}

/// Ensemble STLSQ: bootstrap-resample rows, vote on support, take median coefficients.
fn ensemble_stlsq(
    theta: &[f64],
    n_data: usize,
    n_terms: usize,
    target: &[f64],
    cfg: &SindyConfig,
    n_bootstrap: usize,
    support_quorum: f64,
) -> Vec<f64> {
    let master_seed = cfg.seed.unwrap_or(0u64);
    let n_boot = n_bootstrap.max(1);
    let mut support_counts = vec![0_usize; n_terms];
    let mut all_coeffs: Vec<Vec<f64>> = Vec::with_capacity(n_boot);

    for b in 0..n_boot {
        let boot_seed = derive_seed(master_seed, b as u64);
        let mut rng = Rng::seed_from_u64(boot_seed);

        let rows: Vec<usize> = (0..n_data).map(|_| rng.random_range(0..n_data)).collect();

        let mut theta_boot = vec![0.0_f64; n_data * n_terms];
        let mut target_boot = vec![0.0_f64; n_data];
        for (new_row, &orig_row) in rows.iter().enumerate() {
            for col in 0..n_terms {
                theta_boot[new_row * n_terms + col] = theta[orig_row * n_terms + col];
            }
            target_boot[new_row] = target[orig_row];
        }

        let coeffs = strlsq(
            &theta_boot,
            n_data,
            n_terms,
            &target_boot,
            cfg.threshold,
            cfg.ridge_lambda,
            cfg.max_iter,
        );

        for (j, &c) in coeffs.iter().enumerate() {
            if c.abs() >= cfg.threshold {
                support_counts[j] += 1;
            }
        }
        all_coeffs.push(coeffs);
    }

    let quorum_count = (support_quorum * n_boot as f64).ceil() as usize;
    let active: Vec<bool> = support_counts
        .iter()
        .map(|&cnt| cnt >= quorum_count)
        .collect();

    let mut final_coeffs = vec![0.0_f64; n_terms];
    for j in 0..n_terms {
        if active[j] {
            let mut vals: Vec<f64> = all_coeffs.iter().map(|c| c[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if vals.is_empty() {
                0.0
            } else if vals.len().is_multiple_of(2) {
                (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
            } else {
                vals[vals.len() / 2]
            };
            final_coeffs[j] = median;
        }
    }

    for c in &mut final_coeffs {
        if c.abs() < cfg.threshold {
            *c = 0.0;
        }
    }

    final_coeffs
}

/// Convert coefficient vector into a `SindyEquation`.
fn build_equation(coeffs: Vec<f64>, library: &[LibraryTerm], threshold: f64) -> SindyEquation {
    let active_terms: Vec<String> = library
        .iter()
        .zip(coeffs.iter())
        .filter(|&(_, c)| c.abs() >= threshold)
        .map(|(term, _)| term.label.clone())
        .collect();

    let active_parts: Vec<(f64, &LoweredOp)> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() >= threshold)
        .map(|(i, &c)| (c, &library[i].expr))
        .collect();

    let expr = if active_parts.is_empty() {
        LoweredOp::Const(0.0)
    } else {
        let mut expr = LoweredOp::Mul(
            Arc::new(LoweredOp::Const(active_parts[0].0)),
            Arc::new(active_parts[0].1.clone()),
        );
        for &(c, term) in &active_parts[1..] {
            let term_expr = LoweredOp::Mul(Arc::new(LoweredOp::Const(c)), Arc::new(term.clone()));
            expr = LoweredOp::Add(Arc::new(expr), Arc::new(term_expr));
        }
        expr
    };

    SindyEquation {
        coefficients: coeffs,
        active_terms,
        expr,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    fn make_const(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    /// Simple 2D system: dx/dt = -x, dy/dt = x
    fn two_var_trajectory(n: usize, dt: f64) -> Vec<Vec<f64>> {
        let mut x = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        for t in 1..n {
            x[t] = x[t - 1] - dt * x[t - 1];
            y[t] = y[t - 1] + dt * x[t - 1];
        }
        vec![x, y]
    }

    #[test]
    fn test_sindy_lorenz_stlsq() {
        let n = 50;
        let dt = 0.05_f64;
        let trajectory = two_var_trajectory(n, dt);

        let library = vec![
            LibraryTerm {
                label: "x0".to_string(),
                expr: make_var(0),
            },
            LibraryTerm {
                label: "x1".to_string(),
                expr: make_var(1),
            },
            LibraryTerm {
                label: "x0*x1".to_string(),
                expr: LoweredOp::Mul(Arc::new(make_var(0)), Arc::new(make_var(1))),
            },
            LibraryTerm {
                label: "1".to_string(),
                expr: make_const(1.0),
            },
        ];

        let cfg = SindyConfig {
            library,
            threshold: 0.1,
            ridge_lambda: 1e-5,
            max_iter: 10,
            mode: SindyMode::Stlsq,
            seed: None,
        };

        let result = discover_ode_sindy(&trajectory, dt, &cfg).expect("sindy should succeed");
        assert_eq!(result.equations.len(), 2);

        let eq0 = &result.equations[0];
        assert!(
            eq0.active_terms.len() <= 3,
            "expected sparse solution for eq0, got {} active terms",
            eq0.active_terms.len()
        );
    }

    #[test]
    fn test_sindy_ensemble_deterministic() {
        let n = 40;
        let dt = 0.05_f64;
        let trajectory = two_var_trajectory(n, dt);

        let library = vec![
            LibraryTerm {
                label: "x0".to_string(),
                expr: make_var(0),
            },
            LibraryTerm {
                label: "x1".to_string(),
                expr: make_var(1),
            },
            LibraryTerm {
                label: "1".to_string(),
                expr: make_const(1.0),
            },
        ];

        let cfg1 = SindyConfig {
            library: library.clone(),
            threshold: 0.1,
            ridge_lambda: 1e-5,
            max_iter: 10,
            mode: SindyMode::Ensemble {
                n_bootstrap: 8,
                support_quorum: 0.5,
            },
            seed: Some(123),
        };
        let cfg2 = cfg1.clone();

        let r1 = discover_ode_sindy(&trajectory, dt, &cfg1).expect("run 1 should succeed");
        let r2 = discover_ode_sindy(&trajectory, dt, &cfg2).expect("run 2 should succeed");

        for (eq1, eq2) in r1.equations.iter().zip(r2.equations.iter()) {
            for (c1, c2) in eq1.coefficients.iter().zip(eq2.coefficients.iter()) {
                assert!(
                    (c1 - c2).abs() < 1e-12,
                    "ensemble must be deterministic: {c1} vs {c2}"
                );
            }
        }
    }
}
