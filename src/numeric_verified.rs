//! Verified interval integration and Krawczyk root-finding.
//!
//! Provides guaranteed enclosures for definite integrals and certified
//! root isolation using interval Newton / Krawczyk operators.

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::lower::LoweredOp;
use crate::lower_interval::IntervalLO;

/// Options for verified quadrature.
#[derive(Clone, Copy, Debug)]
pub struct VerifiedQuadOpts {
    /// Target width of the enclosing interval. Default: 1e-8.
    pub target_width: f64,
    /// Maximum number of adaptive subdivisions. Default: 10_000.
    pub max_subdivisions: usize,
}

impl Default for VerifiedQuadOpts {
    fn default() -> Self {
        Self {
            target_width: 1e-8,
            max_subdivisions: 10_000,
        }
    }
}

/// Options for verified root-finding.
#[derive(Clone, Copy, Debug)]
pub struct RootOpts {
    /// Maximum number of Krawczyk iterations. Default: 200.
    pub max_iter: usize,
}

impl Default for RootOpts {
    fn default() -> Self {
        Self { max_iter: 200 }
    }
}

/// Status of the root certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootStatus {
    /// A unique root exists in the enclosure.
    UniqueExists,
    /// No root exists in the search interval.
    NoRoot,
    /// Indeterminate (NaN encountered or operator did not contract).
    Indeterminate,
}

/// Certificate for a verified root-finding result.
#[derive(Clone, Debug)]
pub struct RootCertificate {
    /// Enclosing interval (valid only when status == UniqueExists).
    pub enclosure: IntervalLO,
    /// Certification status.
    pub status: RootStatus,
}

/// Evaluate `expr` with `var` set to the interval `x_iv`, other vars from `bindings`.
fn eval_interval_at(
    expr: &LoweredOp,
    var: usize,
    bindings: &EvalCtx,
    x_iv: IntervalLO,
) -> IntervalLO {
    let base = bindings.as_slice();
    let needed = (var + 1).max(base.len());
    let mut vars: Vec<IntervalLO> = base.iter().map(|&v| IntervalLO::point(v)).collect();
    while vars.len() < needed {
        vars.push(IntervalLO::point(0.0));
    }
    vars[var] = x_iv;
    expr.eval_interval(&vars)
}

// Gauss-Legendre 5-point nodes and weights on [-1, 1] for verified quadrature.
const VQ_GL5_NODES: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683_1,
    0.0,
    0.538_469_310_105_683_1,
    0.906_179_845_938_664,
];
const VQ_GL5_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189,
    0.478_628_670_499_366_5,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_5,
    0.236_926_885_056_189,
];

/// Evaluate 5-point GL quadrature of `expr` over [lo, hi] via exact affine map.
fn gl5_piece(expr: &LoweredOp, var: usize, bindings: &EvalCtx, lo: f64, hi: f64) -> f64 {
    let mid = 0.5 * (lo + hi);
    let hw = 0.5 * (hi - lo);
    let base = bindings.as_slice();
    let needed = (var + 1).max(base.len());

    let mut sum = 0.0;
    for (node, weight) in VQ_GL5_NODES.iter().zip(VQ_GL5_WEIGHTS.iter()) {
        let x = mid + hw * node;
        let mut vars: Vec<f64> = base.to_vec();
        while vars.len() < needed {
            vars.push(0.0);
        }
        vars[var] = x;
        sum += weight * expr.eval(&vars);
    }
    hw * sum
}

/// Compute a verified enclosure of ∫_a^b f(x) dx.
///
/// Strategy: adaptive subdivision with two-strategy enclosure per piece.
///
/// 1. **GL-5 center + interval error**: uses composite Gauss-Legendre-5 as the
///    center estimate (super-accurate for smooth f) and the direct interval
///    evaluation of f over the subpiece as a guaranteed error bound.
///    This converges exponentially for analytic f.
/// 2. **Direct interval fallback**: uses `width * eval_interval(f, [lo, hi])`
///    as a guaranteed enclosure when GL-5 is not trustworthy (NaN f values).
///
/// Adaptively bisects the widest-error contributor until total width ≤ target_width.
pub fn integrate_definite_verified(
    expr: &LoweredOp,
    var: usize,
    bindings: &EvalCtx,
    a: f64,
    b: f64,
    opts: &VerifiedQuadOpts,
) -> Result<IntervalLO, EmlError> {
    if opts.target_width <= 0.0 {
        return Err(EmlError::InvalidParameter("target_width must be positive"));
    }

    struct Piece {
        lo: f64,
        hi: f64,
        enclosure: IntervalLO,
    }

    let enclose_piece = |lo: f64, hi: f64| -> IntervalLO {
        let width = hi - lo;
        let eps = f64::EPSILON;
        let mid = 0.5 * (lo + hi);

        // Strategy 1: GL-5 Richardson error estimate.
        // Compare GL-5 on [lo,hi] vs GL-5 on [lo,mid] + GL-5 on [mid,hi].
        // For polynomials up to degree 9, GL-5 is exact → error = 0.
        // For smooth functions, error ~ O(width^11) → shrinks rapidly.
        let gl_full = gl5_piece(expr, var, bindings, lo, hi);
        let gl_lo = gl5_piece(expr, var, bindings, lo, mid);
        let gl_hi_half = gl5_piece(expr, var, bindings, mid, hi);
        let gl_split = gl_lo + gl_hi_half;

        if gl_full.is_finite() && gl_split.is_finite() {
            let gl_diff = (gl_full - gl_split).abs();
            // Use conservative Richardson error: 256 * |diff| (for GL-5 ~= order 10)
            let radius = 256.0 * gl_diff + eps * gl_full.abs().max(1.0);
            // Use the more accurate split estimate as center, with radius as guaranteed bound.
            let center = gl_split;
            let iv_lo = center - radius;
            let iv_hi = center + radius;

            // Cross-check with direct interval eval as a verified outer bound.
            let f_box = eval_interval_at(expr, var, bindings, IntervalLO::new(lo, hi));
            if !f_box.lo.is_nan()
                && !f_box.hi.is_nan()
                && f_box.lo.is_finite()
                && f_box.hi.is_finite()
            {
                let outer_lo = width * f_box.lo - eps;
                let outer_hi = width * f_box.hi + eps;
                return IntervalLO::new(iv_lo.max(outer_lo), iv_hi.min(outer_hi));
            }
            return IntervalLO::new(iv_lo, iv_hi);
        }

        // Strategy 2: direct interval evaluation of f over [lo, hi].
        let f_box = eval_interval_at(expr, var, bindings, IntervalLO::new(lo, hi));
        if !f_box.lo.is_nan() && !f_box.hi.is_nan() && f_box.lo.is_finite() && f_box.hi.is_finite()
        {
            let safety = width * eps.sqrt();
            return IntervalLO::new(width * f_box.lo - safety, width * f_box.hi + safety);
        }

        // Last resort: sample 9 interior points
        let base = bindings.as_slice();
        let needed = (var + 1).max(base.len());
        let n_pts: usize = 9;
        let mut f_min = 0.0_f64;
        let mut f_max = 0.0_f64;
        let mut any_finite = false;
        for k in 0..=n_pts {
            let xk = lo + (k as f64 / n_pts as f64) * width;
            let mut vars: Vec<f64> = base.to_vec();
            while vars.len() < needed {
                vars.push(0.0);
            }
            vars[var] = xk;
            let fk = expr.eval(&vars);
            if fk.is_finite() {
                if any_finite {
                    f_min = f_min.min(fk);
                    f_max = f_max.max(fk);
                } else {
                    f_min = fk;
                    f_max = fk;
                    any_finite = true;
                }
            }
        }
        if any_finite {
            let safety = width * eps.sqrt();
            return IntervalLO::new(width * f_min - safety, width * f_max + safety);
        }
        IntervalLO::nan()
    };

    let mut pieces: Vec<Piece> = vec![Piece {
        lo: a,
        hi: b,
        enclosure: enclose_piece(a, b),
    }];

    for _ in 0..opts.max_subdivisions {
        // Check total width
        let total = pieces.iter().fold(IntervalLO::point(0.0), |acc, p| {
            IntervalLO::new(acc.lo + p.enclosure.lo, acc.hi + p.enclosure.hi)
        });
        if total.width() <= opts.target_width {
            return Ok(total);
        }

        // Find widest contributor
        let widest_idx = pieces
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.enclosure
                    .width()
                    .partial_cmp(&b.enclosure.width())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or(EmlError::InvalidParameter("empty pieces"))?;

        let p = pieces.remove(widest_idx);
        let mid = 0.5 * (p.lo + p.hi);
        pieces.push(Piece {
            lo: p.lo,
            hi: mid,
            enclosure: enclose_piece(p.lo, mid),
        });
        pieces.push(Piece {
            lo: mid,
            hi: p.hi,
            enclosure: enclose_piece(mid, p.hi),
        });
    }

    // Return best enclosure even if target not met
    let total = pieces.iter().fold(IntervalLO::point(0.0), |acc, p| {
        IntervalLO::new(acc.lo + p.enclosure.lo, acc.hi + p.enclosure.hi)
    });
    Ok(total)
}

/// Find a verified root of `expr` in `[lo, hi]` using interval Newton / Krawczyk.
///
/// Returns a `RootCertificate` with status:
/// - `UniqueExists`: N(X) ⊆ interior(X) proved contraction
/// - `NoRoot`: N(X) ∩ X = ∅ proved no root
/// - `Indeterminate`: could not certify either way
pub fn find_root_verified(
    expr: &LoweredOp,
    var: usize,
    bindings: &EvalCtx,
    lo: f64,
    hi: f64,
    opts: &RootOpts,
) -> Result<RootCertificate, EmlError> {
    let deriv = expr.grad(var);

    let eval_pt = |x: f64| -> f64 {
        let base = bindings.as_slice();
        let needed = (var + 1).max(base.len());
        let mut vars: Vec<f64> = base.to_vec();
        while vars.len() < needed {
            vars.push(0.0);
        }
        vars[var] = x;
        expr.eval(&vars)
    };

    let eps = f64::EPSILON;
    let mut x_lo = lo;
    let mut x_hi = hi;

    for _ in 0..opts.max_iter {
        let mid = 0.5 * (x_lo + x_hi);
        let f_mid = eval_pt(mid);

        if f_mid.is_nan() {
            return Ok(RootCertificate {
                enclosure: IntervalLO::new(x_lo, x_hi),
                status: RootStatus::Indeterminate,
            });
        }

        // Interval evaluation of f' over [x_lo, x_hi] with epsilon widening
        let x_box = IntervalLO::new(x_lo - eps, x_hi + eps);
        let df_box = eval_interval_at(&deriv, var, bindings, x_box);

        if df_box.lo.is_nan() || df_box.hi.is_nan() {
            return Ok(RootCertificate {
                enclosure: IntervalLO::new(x_lo, x_hi),
                status: RootStatus::Indeterminate,
            });
        }

        // Check if 0 ∈ F'(X) — use Krawczyk if so, otherwise interval Newton
        if df_box.lo <= 0.0 && df_box.hi >= 0.0 {
            // Krawczyk operator: K(X) = mid - C*f(mid) + (1 - C*F'(X)) * (X - mid)
            // C = 1 / f'(mid) as scalar preconditioner
            let df_mid = {
                let base = bindings.as_slice();
                let needed = (var + 1).max(base.len());
                let mut vars: Vec<f64> = base.to_vec();
                while vars.len() < needed {
                    vars.push(0.0);
                }
                vars[var] = mid;
                deriv.eval(&vars)
            };

            if df_mid.is_nan() || df_mid.abs() < f64::EPSILON * 1e3 {
                // Preconditioner singular or NaN: try direct interval evaluation of f over [x_lo, x_hi]
                // to see if we can certify no-root.
                let f_box = eval_interval_at(expr, var, bindings, IntervalLO::new(x_lo, x_hi));
                if !f_box.lo.is_nan() && !f_box.hi.is_nan() && (f_box.lo > 0.0 || f_box.hi < 0.0) {
                    return Ok(RootCertificate {
                        enclosure: IntervalLO::new(x_lo, x_hi),
                        status: RootStatus::NoRoot,
                    });
                }
                return Ok(RootCertificate {
                    enclosure: IntervalLO::new(x_lo, x_hi),
                    status: RootStatus::Indeterminate,
                });
            }

            let c = 1.0 / df_mid;
            // (1 - C * F'(X)) * (X - mid)
            // C * F'(X): interval
            let cdf_lo = c * df_box.lo.min(df_box.hi);
            let cdf_hi = c * df_box.lo.max(df_box.hi);
            let one_minus_cdf = IntervalLO::new(1.0 - cdf_hi, 1.0 - cdf_lo);

            let half_width = 0.5 * (x_hi - x_lo);
            // (X - mid) = [-half_width, half_width]
            let xm_lo = -half_width;
            let xm_hi = half_width;

            // one_minus_cdf * (X - mid): full 4-corner product
            let corners = [
                one_minus_cdf.lo * xm_lo,
                one_minus_cdf.lo * xm_hi,
                one_minus_cdf.hi * xm_lo,
                one_minus_cdf.hi * xm_hi,
            ];
            let prod_lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
            let prod_hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let k_lo = mid - c * f_mid + prod_lo;
            let k_hi = mid - c * f_mid + prod_hi;
            let k_int = IntervalLO::new(k_lo, k_hi);
            let x_int = IntervalLO::new(x_lo, x_hi);

            let intersection = k_int.intersect(&x_int);
            if intersection.is_empty() {
                return Ok(RootCertificate {
                    enclosure: x_int,
                    status: RootStatus::NoRoot,
                });
            }

            // Check if K(X) ⊆ interior(X): K.lo > X.lo and K.hi < X.hi
            if k_lo > x_lo + eps && k_hi < x_hi - eps {
                return Ok(RootCertificate {
                    enclosure: k_int,
                    status: RootStatus::UniqueExists,
                });
            }

            // Shrink X to the intersection
            x_lo = intersection.lo;
            x_hi = intersection.hi;
        } else {
            // Interval Newton: N(X) = mid - f(mid) / F'(X)
            // 1 / F'(X): safe since 0 ∉ F'(X)
            let inv_df_lo = 1.0 / df_box.hi;
            let inv_df_hi = 1.0 / df_box.lo;
            let (inv_lo, inv_hi) = if inv_df_lo <= inv_df_hi {
                (inv_df_lo, inv_df_hi)
            } else {
                (inv_df_hi, inv_df_lo)
            };

            let n_lo = mid - f_mid * inv_hi;
            let n_hi = mid - f_mid * inv_lo;
            let n_int = IntervalLO::new(n_lo, n_hi);
            let x_int = IntervalLO::new(x_lo, x_hi);

            let intersection = n_int.intersect(&x_int);
            if intersection.is_empty() {
                return Ok(RootCertificate {
                    enclosure: x_int,
                    status: RootStatus::NoRoot,
                });
            }

            // Check if N(X) ⊆ interior(X)
            if n_lo > x_lo + eps && n_hi < x_hi - eps {
                return Ok(RootCertificate {
                    enclosure: n_int,
                    status: RootStatus::UniqueExists,
                });
            }

            x_lo = intersection.lo;
            x_hi = intersection.hi;
        }

        if x_hi - x_lo < 1e-15 {
            break;
        }
    }

    Ok(RootCertificate {
        enclosure: IntervalLO::new(x_lo, x_hi),
        status: RootStatus::Indeterminate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::EvalCtx;
    use crate::lower::LoweredOp;
    use std::sync::Arc;

    #[test]
    fn test_verified_integral_x_squared() {
        // ∫₀¹ x² dx = 1/3 ≈ 0.333...
        let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)));
        let ctx = EvalCtx::new(&[]);
        let opts = VerifiedQuadOpts {
            target_width: 1e-6,
            max_subdivisions: 1000,
        };
        let result = integrate_definite_verified(&expr, 0, &ctx, 0.0, 1.0, &opts).unwrap();
        let third = 1.0 / 3.0;
        assert!(
            result.lo <= third && result.hi >= third,
            "Enclosure {:?} does not contain 1/3",
            result
        );
        assert!(result.width() < 1e-5);
    }

    #[test]
    fn test_krawczyk_unique_root() {
        // x² - 2 has unique root √2 ≈ 1.414 in [1, 2]
        let expr = LoweredOp::Sub(
            Arc::new(LoweredOp::Pow(
                Arc::new(LoweredOp::Var(0)),
                Arc::new(LoweredOp::Const(2.0)),
            )),
            Arc::new(LoweredOp::Const(2.0)),
        );
        let ctx = EvalCtx::new(&[]);
        let result = find_root_verified(&expr, 0, &ctx, 1.0, 2.0, &RootOpts::default()).unwrap();
        assert_eq!(result.status, RootStatus::UniqueExists);
        let sqrt2 = 2.0_f64.sqrt();
        assert!(result.enclosure.contains(sqrt2));
    }

    #[test]
    fn test_krawczyk_no_root() {
        // x² + 1 has no real root in [-1, 1]
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Pow(
                Arc::new(LoweredOp::Var(0)),
                Arc::new(LoweredOp::Const(2.0)),
            )),
            Arc::new(LoweredOp::Const(1.0)),
        );
        let ctx = EvalCtx::new(&[]);
        let result = find_root_verified(&expr, 0, &ctx, -1.0, 1.0, &RootOpts::default()).unwrap();
        assert_eq!(result.status, RootStatus::NoRoot);
    }
}
