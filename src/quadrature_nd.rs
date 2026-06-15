//! Multidimensional numerical quadrature.
//!
//! Provides tensor-product Gauss-Legendre (up to 4 variables) and
//! quasi-Monte Carlo (Halton sequence) integration for `LoweredOp` expressions.

use crate::error::EmlError;
use crate::lower::LoweredOp;

// Gauss-Legendre 5-point nodes and weights on [-1, 1].
// High-precision values: weights sum to 2.0 in f64 arithmetic.
// w3 = 128/225 exactly; w1=w5, w2=w4 from the Gauss formula.
const GL5_NODES: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683_1,
    0.0,
    0.538_469_310_105_683_1,
    0.906_179_845_938_664,
];
const GL5_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189,
    0.478_628_670_499_366_5,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_5,
    0.236_926_885_056_189,
];

/// Method selector for multidimensional quadrature.
#[derive(Clone, Copy, Debug)]
pub enum QuadNdMethod {
    /// Tensor-product Gauss-Legendre (exact for polynomials; best for n_vars ≤ 4).
    GaussLegendre,
    /// Quasi-Monte Carlo using Halton sequence.
    MonteCarlo {
        /// Number of Halton samples.
        n_samples: usize,
    },
}

/// Options for multidimensional quadrature.
#[derive(Clone, Copy, Debug)]
pub struct QuadNdOpts {
    /// Integration method.
    pub method: QuadNdMethod,
    /// Points per dimension (only used for GaussLegendre). Default: 5.
    pub points_per_dim: usize,
}

impl Default for QuadNdOpts {
    fn default() -> Self {
        Self {
            method: QuadNdMethod::GaussLegendre,
            points_per_dim: 5,
        }
    }
}

/// Halton sequence value for prime base.
fn halton(index: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    let mut i = index;
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}

/// First few primes for Halton sequence bases.
const HALTON_PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// Multidimensional quadrature of `expr` over the box `[lo[i], hi[i]]` for each var index.
///
/// - `vars`: slice of variable indices (length = number of integration dimensions)
/// - `lo`/`hi`: lower and upper bounds per variable (same length as `vars`)
/// - `opts`: method and point-count options
pub fn quadrature_nd(
    expr: &LoweredOp,
    vars: &[usize],
    lo: &[f64],
    hi: &[f64],
    opts: &QuadNdOpts,
) -> Result<f64, EmlError> {
    let ndim = vars.len();
    if ndim == 0 {
        return Err(EmlError::InvalidParameter("vars must be non-empty"));
    }
    if lo.len() != ndim || hi.len() != ndim {
        return Err(EmlError::DimensionMismatch(ndim, lo.len().min(hi.len())));
    }
    if ndim > HALTON_PRIMES.len() {
        return Err(EmlError::InvalidParameter("too many dimensions (max 16)"));
    }

    // Jacobian of the affine map (product of (hi-lo)/2 for each dim when GL,
    // or (hi-lo) for each dim when MC)
    match opts.method {
        QuadNdMethod::GaussLegendre => gauss_legendre_nd(expr, vars, lo, hi),
        QuadNdMethod::MonteCarlo { n_samples } => {
            monte_carlo_halton_nd(expr, vars, lo, hi, n_samples)
        }
    }
}

/// Tensor-product Gauss-Legendre integration.
fn gauss_legendre_nd(
    expr: &LoweredOp,
    vars: &[usize],
    lo: &[f64],
    hi: &[f64],
) -> Result<f64, EmlError> {
    let ndim = vars.len();

    // Jacobian: product of (hi[i] - lo[i]) / 2 for each dimension
    let mut jacobian = 1.0_f64;
    for i in 0..ndim {
        jacobian *= (hi[i] - lo[i]) / 2.0;
    }

    // Affine map: t ∈ [-1,1] → x = lo + (1+t)/2 * (hi-lo) = (lo+hi)/2 + t*(hi-lo)/2
    let centers: Vec<f64> = (0..ndim).map(|i| (lo[i] + hi[i]) / 2.0).collect();
    let half_widths: Vec<f64> = (0..ndim).map(|i| (hi[i] - lo[i]) / 2.0).collect();

    // Total number of GL points
    let n_pts = GL5_NODES.len().pow(ndim as u32);

    let mut sum = 0.0_f64;

    // Iterate over all combinations of 5 nodes per dimension via mixed-radix indexing
    for flat_idx in 0..n_pts {
        let mut weight = 1.0_f64;
        let mut x_vals: Vec<(usize, f64)> = Vec::with_capacity(ndim);

        let mut idx = flat_idx;
        for i in 0..ndim {
            let node_idx = idx % GL5_NODES.len();
            idx /= GL5_NODES.len();
            let node = GL5_NODES[node_idx];
            let w = GL5_WEIGHTS[node_idx];
            let x = centers[i] + half_widths[i] * node;
            weight *= w;
            x_vals.push((vars[i], x));
        }

        // Build variable vector
        let n_vars_total = vars.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut var_vec = vec![0.0_f64; n_vars_total];
        for (idx_var, val) in &x_vals {
            if *idx_var < var_vec.len() {
                var_vec[*idx_var] = *val;
            }
        }

        let f_val = expr.eval(&var_vec);
        sum += weight * f_val;
    }

    Ok(jacobian * sum)
}

/// Quasi-Monte Carlo integration using Halton sequence.
fn monte_carlo_halton_nd(
    expr: &LoweredOp,
    vars: &[usize],
    lo: &[f64],
    hi: &[f64],
    n_samples: usize,
) -> Result<f64, EmlError> {
    if n_samples == 0 {
        return Err(EmlError::InvalidParameter("n_samples must be > 0"));
    }
    let ndim = vars.len();

    let volume: f64 = (0..ndim).map(|i| hi[i] - lo[i]).product();
    let n_vars_total = vars.iter().copied().max().map(|m| m + 1).unwrap_or(1);

    let mut sum = 0.0_f64;
    for s in 0..n_samples {
        let mut var_vec = vec![0.0_f64; n_vars_total];
        for (dim, &var_idx) in vars.iter().enumerate() {
            let h = halton(s + 1, HALTON_PRIMES[dim]);
            let x = lo[dim] + h * (hi[dim] - lo[dim]);
            if var_idx < var_vec.len() {
                var_vec[var_idx] = x;
            }
        }
        sum += expr.eval(&var_vec);
    }

    Ok(volume * sum / n_samples as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::LoweredOp;
    use std::sync::Arc;

    #[test]
    fn test_gauss_legendre_2d_linear() {
        // ∫₀¹∫₀¹ (x + y) dx dy = 1
        let expr = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
        let opts = QuadNdOpts {
            method: QuadNdMethod::GaussLegendre,
            points_per_dim: 5,
        };
        let result = quadrature_nd(&expr, &[0, 1], &[0.0, 0.0], &[1.0, 1.0], &opts).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_gauss_legendre_1d_cubic() {
        // ∫₋₁¹ x³ dx = 0 (odd function)
        let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(3.0)));
        let opts = QuadNdOpts {
            method: QuadNdMethod::GaussLegendre,
            points_per_dim: 5,
        };
        let result = quadrature_nd(&expr, &[0], &[-1.0], &[1.0], &opts).unwrap();
        assert!(result.abs() < 1e-12, "Expected 0, got {}", result);
    }
}
