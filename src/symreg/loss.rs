//! Loss functions for the Adam optimiser in symbolic regression.
//!
//! Defines [`SymRegLoss`] and the companion math helpers used by the
//! per-iteration gradient computation in `discover.rs`.

/// Loss function used by the Adam optimiser in symbolic regression.
///
/// Controls how the residual `r = prediction − target` is penalised.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SymRegLoss {
    /// Mean squared error: `L(r) = r²`.
    #[default]
    Mse,
    /// Huber loss: quadratic for `|r| ≤ delta`, linear beyond.
    ///
    /// More robust to outliers than MSE while remaining smooth everywhere.
    Huber {
        /// Transition point between quadratic and linear regime.
        delta: f64,
    },
    /// Trimmed MSE: drops the top `alpha` fraction of largest residuals
    /// before averaging.
    ///
    /// For the Adam gradient step a smooth sigmoid-weight approximation is
    /// used; the exact trim is applied for scoring.
    TrimmedMse {
        /// Fraction of points to trim (0.0 = no trim = MSE).
        alpha: f64,
    },
}

/// Compute Huber loss for a slice of residuals.
///
/// `L(r) = r²/2` when `|r| ≤ delta`, else `delta·(|r| - delta/2)`.
pub(super) fn huber_loss(residuals: &[f64], delta: f64) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let sum: f64 = residuals
        .iter()
        .map(|&r| {
            let ar = r.abs();
            if ar <= delta {
                0.5 * r * r
            } else {
                delta * (ar - 0.5 * delta)
            }
        })
        .sum();
    sum / residuals.len() as f64
}

/// Gradient scaling factor for Huber loss at residual `r`.
///
/// Returns `r` when `|r| ≤ delta`, else `delta · sign(r)`.
/// The Adam gradient is `2 * huber_grad_factor(r, delta)` (matching the
/// MSE gradient structure `2r` so callers just replace the `2r` factor).
pub(super) fn huber_grad_factor(r: f64, delta: f64) -> f64 {
    if r.abs() <= delta {
        r
    } else {
        delta * r.signum()
    }
}

/// Compute trimmed MSE for a slice of residuals, dropping the top `alpha`
/// fraction by absolute residual.
pub(super) fn trimmed_mse(residuals: &[f64], alpha: f64) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let keep = ((1.0 - alpha) * sorted.len() as f64).ceil() as usize;
    let keep = keep.max(1).min(sorted.len());
    sorted[..keep].iter().sum::<f64>() / keep as f64
}

/// Smooth sigmoid-based weight for trimmed-MSE gradient at residual `r`.
///
/// Returns a value in (0, 1] that downweights large residuals via a
/// soft-threshold sigmoid rather than a hard trim. This keeps the Adam
/// gradient step smooth and avoids discontinuities that stall convergence.
///
/// `w(r) = sigmoid(α_k - |r|/q)` where `q` is the `alpha`-quantile of
/// `|residuals|` and `α_k = 3.0` controls the sharpness.
pub(super) fn trimmed_mse_grad_factor(r: f64, residuals: &[f64], alpha: f64) -> f64 {
    if residuals.is_empty() || alpha <= 0.0 {
        return r;
    }
    let mut abs_res: Vec<f64> = residuals.iter().map(|x| x.abs()).collect();
    abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_idx = ((1.0 - alpha) * (abs_res.len() - 1) as f64).round() as usize;
    let q = abs_res[q_idx.min(abs_res.len() - 1)].max(1e-12);
    let sharpness = 3.0_f64;
    let w = 1.0 / (1.0 + (r.abs() / q - (1.0 - alpha)).exp() * sharpness.exp());
    w.clamp(0.0, 1.0) * r
}
