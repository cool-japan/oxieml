//! Uncertainty quantification for discovered formula parameters.
//!
//! Provides [`compute_bootstrap_intervals`]: a non-parametric bootstrap
//! procedure that resamples the data `B` times, refits parameters using a
//! short Adam warm-start, and returns percentile confidence intervals.
//!
//! Also provides [`inv_norm_cdf`]: a pure-Rust implementation of the
//! inverse normal CDF (Acklam's rational approximation).

use rand::RngExt;
use rand::SeedableRng;

use crate::eval::EvalCtx;
use crate::grad::ParameterizedEmlTree;

use super::{DiscoveredFormula, SymRegEngine};

type Rng = rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Inverse normal CDF — Acklam's rational approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse normal CDF (probit function) via Acklam's rational approximation.
///
/// Returns the `p`-quantile of the standard normal distribution for `p ∈ (0, 1)`.
/// Returns `f64::NEG_INFINITY` for `p ≤ 0` and `f64::INFINITY` for `p ≥ 1`.
///
/// Maximum absolute error < 1.15 × 10⁻⁹ (Acklam 2010).
pub fn inv_norm_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for the rational approximation
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail (by symmetry)
        -inv_norm_cdf(1.0 - p)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap interval computation
// ─────────────────────────────────────────────────────────────────────────────

/// Refit parameters of `formula` on resampled data using a condensed Adam loop.
///
/// Returns the parameter vector after `max_iter` Adam steps, warm-started from
/// `init_params`.
fn refit_params_adam(
    ptree_template: &ParameterizedEmlTree,
    inputs: &[Vec<f64>],
    targets: &[f64],
    init_params: &[f64],
    max_iter: usize,
    learning_rate: f64,
) -> Vec<f64> {
    let mut ptree = ptree_template.clone();
    let n_params = ptree.num_params();

    if n_params == 0 {
        return vec![];
    }

    // Warm-start from the provided initial parameters
    if init_params.len() == n_params {
        ptree.params.clone_from_slice(init_params);
    }

    let mut m = vec![0.0_f64; n_params];
    let mut v = vec![0.0_f64; n_params];
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let epsilon = 1e-8_f64;

    for t in 1..=max_iter {
        let mut grads = vec![0.0_f64; n_params];
        let mut valid = 0_usize;

        for (input, &target) in inputs.iter().zip(targets) {
            let ctx = EvalCtx::new(input);
            match ptree.forward_backward(&ctx, target) {
                Ok((loss, g)) if loss.is_finite() => {
                    for (tg, gv) in grads.iter_mut().zip(&g) {
                        if gv.is_finite() {
                            *tg += gv;
                        }
                    }
                    valid += 1;
                }
                _ => {}
            }
        }

        if valid == 0 {
            break;
        }
        let n_f = valid as f64;
        for i in 0..n_params {
            let g = grads[i] / n_f;
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
            let v_hat = v[i] / (1.0 - beta2.powi(t as i32));
            ptree.params[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    ptree.params
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute bootstrap confidence intervals for each parameter of `formula`.
///
/// Resamples the dataset `bootstrap_samples` times (with replacement), refits
/// parameters on each resample using a short warm-started Adam loop, then
/// returns percentile CIs at level `confidence_level`.
///
/// Returns `None` when:
/// - `bootstrap_samples == 0`, or
/// - `formula.params.is_empty()`.
pub fn compute_bootstrap_intervals(
    engine: &SymRegEngine,
    formula: &DiscoveredFormula,
    inputs: &[Vec<f64>],
    targets: &[f64],
    bootstrap_samples: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> Option<Vec<(f64, f64)>> {
    if bootstrap_samples == 0 || formula.params.is_empty() {
        return None;
    }
    let n = inputs.len();
    if n == 0 {
        return None;
    }

    let n_params = formula.params.len();
    let ptree_template = ParameterizedEmlTree::from_topology(&formula.eml_tree, 1.0);

    // Seed the master RNG
    let master_seed: u64 = seed.unwrap_or(0x1234_5678_9abc_def0);
    let mut rng = Rng::seed_from_u64(master_seed);

    // Collect bootstrap parameter estimates
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(bootstrap_samples);

    let boot_iters = 500_usize;
    let lr = engine.config.learning_rate;

    for b in 0..bootstrap_samples {
        // Resample indices with replacement
        let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();

        let boot_inputs: Vec<Vec<f64>> = indices.iter().map(|&i| inputs[i].clone()).collect();
        let boot_targets: Vec<f64> = indices.iter().map(|&i| targets[i]).collect();

        // Per-bootstrap RNG offset for seeded reproducibility
        let boot_params = refit_params_adam(
            &ptree_template,
            &boot_inputs,
            &boot_targets,
            &formula.params,
            boot_iters,
            lr,
        );

        // Sanity: only accept if all params are finite
        if boot_params.iter().all(|p| p.is_finite()) {
            samples.push(boot_params);
        }

        // Suppress unused warning on `b`
        let _ = b;
    }

    if samples.is_empty() {
        return None;
    }

    // Compute percentile CIs
    let alpha = 1.0 - confidence_level;
    let lo_frac = alpha / 2.0;
    let hi_frac = 1.0 - alpha / 2.0;

    let intervals: Vec<(f64, f64)> = (0..n_params)
        .map(|j| {
            let mut vals: Vec<f64> = samples.iter().map(|s| s[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo_idx = ((lo_frac * vals.len() as f64).floor() as usize).min(vals.len() - 1);
            let hi_idx = ((hi_frac * vals.len() as f64).floor() as usize).min(vals.len() - 1);
            (vals[lo_idx], vals[hi_idx])
        })
        .collect();

    Some(intervals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytic (Laplace/Hessian) uncertainty quantification
// ─────────────────────────────────────────────────────────────────────────────

/// Compute analytic (Laplace/Hessian) confidence intervals for LM-fitted parameters.
///
/// Uses the Laplace approximation: Σ = σ̂²(JᵀJ)⁻¹ where σ̂² = RSS/(n−k).
/// CIs: θ̂ ± z_{α/2} · √(Σ_{ii}) where z is from the standard normal.
///
/// Assumptions: MSE loss, Gaussian residuals, n > k (more data than params).
/// This is an asymptotic approximation; bootstrap CIs may be more reliable for small samples.
///
/// Returns `None` if:
/// - `n_data ≤ n_params` (insufficient degrees of freedom)
/// - JᵀJ is singular (Cholesky and pinv both fail, or produce non-finite diagonal)
/// - Any CI bound is non-finite
pub fn compute_analytic_intervals(
    jac: &[f64],
    residuals: &[f64],
    params: &[f64],
    n_data: usize,
    n_params: usize,
    confidence: f64,
) -> Option<Vec<(f64, f64)>> {
    if n_data <= n_params || n_params == 0 {
        return None;
    }
    let dof = (n_data - n_params) as f64;
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma_sq = rss / dof;

    // Plain JᵀJ (no damping)
    let jtj = crate::linalg::jtj(jac, n_data, n_params, 0.0);

    // Covariance Σ = σ̂² (JᵀJ)⁻¹ — try SPD inversion, fall back to pinv
    let cov_unnorm = match crate::linalg::invert_spd(&jtj, n_params) {
        Ok(inv) => inv,
        Err(_) => crate::linalg::pinv(&jtj, n_params, n_params, None).ok()?,
    };

    // z-score for two-sided CI at the given confidence level
    let alpha = 1.0 - confidence;
    let z = inv_norm_cdf(1.0 - alpha / 2.0);

    let mut intervals = Vec::with_capacity(n_params);
    for i in 0..n_params {
        let var_i = cov_unnorm[i * n_params + i] * sigma_sq;
        if var_i < 0.0 || !var_i.is_finite() {
            return None;
        }
        let half_width = z * var_i.sqrt();
        let lo = params[i] - half_width;
        let hi = params[i] + half_width;
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        intervals.push((lo, hi));
    }
    Some(intervals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_norm_cdf_spot_checks() {
        // p = 0.5 → 0.0
        assert!(inv_norm_cdf(0.5).abs() < 1e-9, "ppf(0.5) should be 0");
        // p = 0.975 → ≈ 1.959963984
        let z975 = inv_norm_cdf(0.975);
        assert!((z975 - 1.959_963_984).abs() < 1e-6, "ppf(0.975) = {z975}");
        // p = 0.025 → ≈ −1.959963984
        let z025 = inv_norm_cdf(0.025);
        assert!((z025 + 1.959_963_984).abs() < 1e-6, "ppf(0.025) = {z025}");
        // Boundary
        assert_eq!(inv_norm_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(inv_norm_cdf(1.0), f64::INFINITY);
    }

    #[test]
    fn bootstrap_returns_none_for_zero_samples() {
        use crate::canonical::Canonical;
        use crate::symreg::{DiscoveredFormula, SymRegConfig, SymRegEngine};

        let tree = Canonical::nat(1);
        let formula = DiscoveredFormula {
            eml_tree: tree,
            mse: 0.0,
            complexity: 1,
            score: 0.0,
            pretty: "1".to_string(),
            params: vec![1.0],
            cv_mse: None,
            aic: 0.0,
            bic: 0.0,
            param_intervals: None,
        };
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = compute_bootstrap_intervals(
            &engine,
            &formula,
            &[vec![1.0]],
            &[1.0],
            0, // bootstrap_samples = 0
            0.95,
            None,
        );
        assert!(result.is_none());
    }

    #[test]
    fn bootstrap_returns_none_for_empty_params() {
        use crate::canonical::Canonical;
        use crate::symreg::{DiscoveredFormula, SymRegConfig, SymRegEngine};

        let tree = Canonical::nat(1);
        let formula = DiscoveredFormula {
            eml_tree: tree,
            mse: 0.0,
            complexity: 1,
            score: 0.0,
            pretty: "1".to_string(),
            params: vec![], // no params
            cv_mse: None,
            aic: 0.0,
            bic: 0.0,
            param_intervals: None,
        };
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result =
            compute_bootstrap_intervals(&engine, &formula, &[vec![1.0]], &[1.0], 50, 0.95, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_analytic_intervals_insufficient_dof() {
        // n_data == n_params → None
        let j = vec![1.0_f64, 0.0, 0.0, 1.0]; // 2×2
        let r = vec![0.0_f64, 0.0];
        let params = vec![1.0_f64, 1.0];
        let result = compute_analytic_intervals(&j, &r, &params, 2, 2, 0.95);
        assert!(
            result.is_none(),
            "Should return None when n_data == n_params"
        );
    }

    #[test]
    fn test_analytic_intervals_simple_linear() {
        // y = 2*x, perfect fit: J = x values (column), r = 0
        // With small noise added to residuals to make sigma > 0
        let n = 20usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let true_a = 2.5_f64;
        let params = vec![true_a];
        // J is n×1, jac[i*1+0] = x[i]
        let jac: Vec<f64> = x.clone();
        // Small residuals (not exactly zero to get finite sigma)
        let residuals: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 - 10.0)).collect();
        let result = compute_analytic_intervals(&jac, &residuals, &params, n, 1, 0.95);
        assert!(
            result.is_some(),
            "Should return Some for well-conditioned problem"
        );
        let cis = result.expect("just checked");
        assert_eq!(cis.len(), 1);
        let (lo, hi) = cis[0];
        assert!(lo.is_finite() && hi.is_finite(), "CI bounds must be finite");
        assert!(
            lo < true_a && true_a < hi,
            "95% CI should contain true param: [{lo}, {hi}]"
        );
    }

    #[test]
    fn test_analytic_intervals_zero_residuals() {
        // With exactly zero residuals, sigma_sq = 0, half_width = 0
        // CI degenerates to point interval [theta, theta]
        let n = 10usize;
        let jac: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect(); // n×1
        let residuals = vec![0.0_f64; n];
        let params = vec![3.0_f64];
        let result = compute_analytic_intervals(&jac, &residuals, &params, n, 1, 0.95);
        // With sigma_sq=0, half_width=0, so lo==hi==theta — still valid (Some)
        assert!(result.is_some());
        let cis = result.expect("just checked");
        assert_eq!(cis[0].0, 3.0);
        assert_eq!(cis[0].1, 3.0);
    }
}
