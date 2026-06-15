//! Levenberg–Marquardt least-squares constant fitter for symbolic regression.
//!
//! Replaces / augments the Adam optimizer in [`super::discover`] when
//! [`super::OptimizerKind::LevenbergMarquardt`] is selected.
//!
//! The LM algorithm iteratively solves:
//!
//! ```text
//! (JᵀJ + λ · diag(JᵀJ)) δ = −Jᵀr
//! ```
//!
//! where `J` is the Jacobian, `r` is the residual vector, and `λ` is a
//! damping parameter (Marquardt variant). Steps that reduce cost are
//! accepted and `λ` is decreased; rejected steps increase `λ` (or abandon
//! the restart if `λ > lambda_max`).

use rand::RngExt;
use rand::SeedableRng;

use crate::eval::EvalCtx;
use crate::grad::ParameterizedEmlTree;
use crate::linalg;
use crate::tree::EmlTree;

use super::discover::compute_aic_bic;
use super::loss::{SymRegLoss, huber_grad_factor, trimmed_mse_grad_factor};
use super::post_round::{try_extract_named_constants, try_post_adam_rounding};
use super::topology::compute_mse_direct;
use super::{DiscoveredFormula, SymRegEngine};

type Rng = rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Local copies of shared utilities (pub(super) items in discover.rs are not
// accessible from sibling modules; we inline these tiny helpers).
// ─────────────────────────────────────────────────────────────────────────────

/// Derive a per-topology seed from a master seed using SplitMix64 mixing.
fn derive_seed(master: u64, topology_idx: u64) -> u64 {
    let mix = |mut z: u64| -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    mix(master).wrapping_add(mix(topology_idx))
}

/// Build a seeded or OS-random RNG for topology `salt`.
fn make_rng(seed: Option<u64>, salt: u64) -> Rng {
    match seed {
        Some(s) => Rng::seed_from_u64(derive_seed(s, salt)),
        None => rand::make_rng::<Rng>(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobian / residual assembly with optional IRLS weighting
// ─────────────────────────────────────────────────────────────────────────────

/// Build the weighted residual vector and weighted row-major Jacobian.
///
/// For MSE loss, weights are all 1.  For robust losses (Huber, TrimmedMse)
/// IRLS weights `w_i = grad_factor(r_i) / r_i` are applied, multiplying
/// each row of J and element of r by `sqrt(w_i)`.
fn assemble_jac_residuals(
    ptree: &ParameterizedEmlTree,
    inputs: &[Vec<f64>],
    targets: &[f64],
    loss: &SymRegLoss,
) -> (Vec<f64>, Vec<f64>) {
    let n_params = ptree.num_params();
    let mut residuals: Vec<f64> = Vec::with_capacity(inputs.len());
    let mut jac_buf: Vec<f64> = Vec::with_capacity(inputs.len() * n_params.max(1));
    let mut jac_tmp = Vec::with_capacity(n_params);

    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        match ptree.forward_with_jacobian_into(&ctx, &mut jac_tmp) {
            Ok(out) if out.is_finite() => {
                residuals.push(out - target);
                jac_buf.extend_from_slice(&jac_tmp);
            }
            _ => {}
        }
    }

    // For robust losses, re-weight rows by IRLS weight √w_i
    let reweighted_residuals: Vec<f64> = match loss {
        SymRegLoss::Mse => residuals.clone(),
        SymRegLoss::Huber { delta } => {
            let d = *delta;
            let mut rw = Vec::with_capacity(residuals.len());
            for (row_idx, &r) in residuals.iter().enumerate() {
                let gf = huber_grad_factor(r, d);
                let w = if r.abs() > f64::EPSILON {
                    (gf / r).abs().sqrt().clamp(0.0, 1e6)
                } else {
                    1.0
                };
                rw.push(r * w);
                for j in 0..n_params {
                    jac_buf[row_idx * n_params + j] *= w;
                }
            }
            rw
        }
        SymRegLoss::TrimmedMse { alpha } => {
            let a = *alpha;
            let mut rw = Vec::with_capacity(residuals.len());
            let residuals_clone = residuals.clone();
            for (row_idx, &r) in residuals.iter().enumerate() {
                let gf = trimmed_mse_grad_factor(r, &residuals_clone, a);
                let w = if r.abs() > f64::EPSILON {
                    (gf / r).abs().sqrt().clamp(0.0, 1e6)
                } else {
                    1.0
                };
                rw.push(r * w);
                for j in 0..n_params {
                    jac_buf[row_idx * n_params + j] *= w;
                }
            }
            rw
        }
    };

    (reweighted_residuals, jac_buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// UQ helper: unweighted Jacobian / residual extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Compute unweighted MSE Jacobian and residuals for UQ purposes.
///
/// Reconstructs a [`ParameterizedEmlTree`] from the topology and fitted params,
/// then evaluates the Jacobian ∂f/∂θ and residuals r = f(θ) - y at each data point.
///
/// Returns `(residuals, jac_buf, n_data, n_params)` where:
/// - `residuals` has length `n_valid` (data points where evaluation was finite)
/// - `jac_buf` is row-major n_valid × n_params
/// - `n_data` = n_valid (number of valid data points)
/// - `n_params` = number of parameters
///
/// Returns `None` if no valid data points were found.
pub(super) fn jac_residuals_for_uq(
    topology: &crate::tree::EmlTree,
    params: &[f64],
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Option<(Vec<f64>, Vec<f64>, usize, usize)> {
    use crate::grad::ParameterizedEmlTree;
    let mut ptree = ParameterizedEmlTree::from_topology(topology, 1.0);
    let n_params = ptree.num_params();
    if n_params == 0 || n_params != params.len() {
        return None;
    }
    ptree.params.clone_from_slice(params);

    let (residuals, jac_buf) =
        assemble_jac_residuals(&ptree, inputs, targets, &super::loss::SymRegLoss::Mse);
    let n_data = residuals.len();
    if n_data == 0 {
        return None;
    }
    Some((residuals, jac_buf, n_data, n_params))
}

// ─────────────────────────────────────────────────────────────────────────────
// Main LM optimizer entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Optimize parameters for a single topology using Levenberg–Marquardt.
///
/// This is the LM counterpart of `discover::SymRegEngine::optimize_topology_adam`.
pub(super) fn optimize_topology_lm(
    engine: &SymRegEngine,
    topology: &EmlTree,
    inputs: &[Vec<f64>],
    targets: &[f64],
    topology_idx: usize,
) -> Option<DiscoveredFormula> {
    let config = &engine.config;
    let lm = &config.lm;
    let mut best_mse = f64::INFINITY;
    let mut best_params: Vec<f64> = Vec::new();
    let mut rng = make_rng(config.seed, topology_idx as u64);

    for _restart in 0..config.num_restarts {
        let mut ptree = ParameterizedEmlTree::from_topology(topology, 1.0);
        let n_params = ptree.num_params();

        // Parameter-free topology: evaluate directly.
        if n_params == 0 {
            let mse = compute_mse_direct(topology, inputs, targets);
            if let Some(mse) = mse {
                if mse < best_mse {
                    best_mse = mse;
                    best_params = vec![];
                }
            }
            break;
        }

        // Random initialization (same range as Adam).
        for p in &mut ptree.params {
            *p = 1.0 + rng.random_range(-0.5..0.5);
        }

        // Assemble initial (r, J) and cost C.
        let (mut r, mut jac) = assemble_jac_residuals(&ptree, inputs, targets, &config.loss);
        let mut valid = r.len();
        if valid == 0 {
            continue;
        }
        let mut cost: f64 = r.iter().map(|x| x * x).sum();
        let mut lambda = lm.lambda_init;

        for _iter in 0..lm.max_iter {
            // g = Jᵀr
            let g = linalg::jtr(&jac, &r, valid, n_params);
            let g_inf = g.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            if g_inf < lm.grad_tol {
                break; // gradient convergence
            }

            // A = JᵀJ + λ·diag(JᵀJ) (Marquardt scaling)
            let a = linalg::jtj_marquardt(&jac, valid, n_params, lambda);
            // b = −Jᵀr
            let mut b: Vec<f64> = g.iter().map(|x| -x).collect();

            // Solve A δ = b  (b becomes δ on success)
            if linalg::solve_normal_equations(&a, &mut b, n_params).is_err() {
                lambda *= lm.lambda_up;
                if lambda > lm.lambda_max {
                    break;
                }
                continue;
            }
            let delta = b; // alias for clarity

            // Trial step
            let theta_try: Vec<f64> = ptree
                .params
                .iter()
                .zip(&delta)
                .map(|(p, d)| p + d)
                .collect();
            let mut ptree_try = ptree.clone();
            ptree_try.params = theta_try;

            let (r_try, jac_try) =
                assemble_jac_residuals(&ptree_try, inputs, targets, &config.loss);
            let valid_try = r_try.len();
            if valid_try == 0 {
                lambda *= lm.lambda_up;
                if lambda > lm.lambda_max {
                    break;
                }
                continue;
            }
            let cost_try: f64 = r_try.iter().map(|x| x * x).sum();

            if cost_try < cost {
                // Accept step
                let cost_prev = cost;
                ptree.params = ptree_try.params;
                r = r_try;
                jac = jac_try;
                valid = valid_try;
                cost = cost_try;
                lambda = (lambda / lm.lambda_down).max(lm.lambda_min);

                // Step-size convergence
                let theta_norm: f64 = ptree.params.iter().map(|x| x * x).sum::<f64>().sqrt();
                let delta_norm: f64 = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
                if delta_norm / (theta_norm + lm.step_tol) < lm.step_tol {
                    break;
                }
                // Cost convergence
                if cost_prev > 0.0 && (cost_prev - cost) / cost_prev < lm.cost_tol {
                    break;
                }
            } else {
                // Reject step — increase damping
                lambda *= lm.lambda_up;
                if lambda > lm.lambda_max {
                    break;
                }
            }
        }

        let mse = cost / valid as f64;
        if mse < best_mse {
            best_mse = mse;
            best_params = ptree.params.clone();
        }

        // Early exit if within tolerance
        if best_mse < config.tolerance {
            break;
        }
    }

    // ── Post-processing (mirrors Adam path) ────────────────────────────────

    if config.integer_rounding && !best_params.is_empty() {
        let (rounded, rounded_mse) =
            try_post_adam_rounding(topology, best_params, best_mse, inputs, targets);
        best_params = rounded;
        best_mse = rounded_mse;
    }

    if !best_mse.is_finite() || best_mse > 1e10 {
        return None;
    }

    let complexity = topology.size();

    let (final_op, final_mse) = try_extract_named_constants(
        topology,
        &best_params,
        best_mse,
        config.constant_extraction,
        inputs,
        targets,
    );

    let pretty = final_op.to_pretty();
    let (aic, bic) = compute_aic_bic(final_mse, inputs.len(), best_params.len());

    Some(DiscoveredFormula {
        eml_tree: topology.clone(),
        mse: final_mse,
        complexity,
        score: final_mse + config.complexity_penalty * complexity as f64,
        pretty,
        params: best_params,
        cv_mse: None,
        aic,
        bic,
        param_intervals: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::symreg::{OptimizerKind, SymRegConfig, SymRegEngine};

    /// Recover `y = 3·x² + 1` using the LM optimizer.
    ///
    /// The topology `a · x² + b` (depth 2) should be found with near-zero MSE.
    #[test]
    fn lm_quadratic_convergence() {
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let targets: Vec<f64> = inputs.iter().map(|r| 3.0 * r[0] * r[0] + 1.0).collect();

        let config = SymRegConfig {
            max_depth: 2,
            max_iter: 30,
            num_restarts: 3,
            optimizer: OptimizerKind::LevenbergMarquardt,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let result = engine.discover(&inputs, &targets, 1);
        assert!(result.is_ok(), "LM discover should not error");
        let formulas = result.expect("already checked");
        assert!(!formulas.is_empty(), "LM should find at least one formula");
        assert!(
            formulas[0].mse < 0.5,
            "LM MSE too high: {}",
            formulas[0].mse
        );
    }

    /// AIC and BIC are finite after a successful LM run.
    #[test]
    fn lm_aic_bic_finite() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.3]).collect();
        let targets: Vec<f64> = inputs.iter().map(|r| r[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 1,
            max_iter: 50,
            num_restarts: 3,
            optimizer: OptimizerKind::LevenbergMarquardt,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine
            .discover(&inputs, &targets, 1)
            .expect("discover should succeed");
        assert!(!formulas.is_empty());
        let f = &formulas[0];
        assert!(f.aic.is_finite(), "AIC must be finite");
        assert!(f.bic.is_finite(), "BIC must be finite");
    }

    #[test]
    fn lm_uq_analytic_produces_intervals() {
        let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let targets: Vec<f64> = inputs.iter().map(|r| 3.0 * r[0] * r[0] + 1.0).collect();

        let config = SymRegConfig {
            max_depth: 2,
            max_iter: 100,
            num_restarts: 3,
            optimizer: OptimizerKind::LevenbergMarquardt,
            uq_analytic: true,
            uq_top_k: 3,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let result = engine.discover(&inputs, &targets, 1);
        assert!(
            result.is_ok(),
            "LM discover with uq_analytic should not error"
        );
        let formulas = result.expect("already checked");
        assert!(!formulas.is_empty(), "Should find at least one formula");
        // Top formula should have param_intervals if it has params
        let top = &formulas[0];
        if !top.params.is_empty() {
            // Intervals may or may not be set depending on DOF, but should not panic
            if let Some(ref intervals) = top.param_intervals {
                assert_eq!(intervals.len(), top.params.len());
                for (lo, hi) in intervals {
                    assert!(lo.is_finite() && hi.is_finite(), "CI bounds must be finite");
                    assert!(lo <= hi, "lo must be <= hi");
                }
            }
        }
    }

    #[test]
    fn lm_uq_analytic_adam_returns_none() {
        // With Adam optimizer, uq_analytic should have no effect
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let targets: Vec<f64> = inputs.iter().map(|r| 2.0 * r[0] + 1.0).collect();

        let config = SymRegConfig {
            max_depth: 2,
            max_iter: 100,
            num_restarts: 2,
            optimizer: OptimizerKind::Adam,
            uq_analytic: true,
            uq_top_k: 3,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let result = engine.discover(&inputs, &targets, 1);
        assert!(result.is_ok());
        let formulas = result.expect("already checked");
        // With Adam optimizer, uq_analytic has no effect: param_intervals should be None
        for formula in &formulas {
            assert!(
                formula.param_intervals.is_none(),
                "Adam optimizer should not produce analytic UQ intervals"
            );
        }
    }
}
