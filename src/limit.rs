//! Limit computation for `LoweredOp` expression trees.
//!
//! Uses numeric probing as the primary oracle and L'Hôpital's rule as a
//! symbolic accelerator for `0/0` and `∞/∞` indeterminate forms at the top
//! level of a `Div` node.  Results for oscillatory functions or limits that
//! cannot be resolved within the iteration cap are reported as
//! [`LimitResult::DoesNotExist`] or [`LimitResult::Indeterminate`] rather
//! than silently wrong values.

use crate::lower::LoweredOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Numeric tuning constants
// ---------------------------------------------------------------------------

/// Below this magnitude → treat as 0 for indeterminate-form detection.
const INDET_EPS: f64 = 1e-6;
/// Above this magnitude → treat as ∞ for indeterminate-form detection.
const BIG: f64 = 1e12;
/// Maximum number of L'Hôpital applications before falling back to probing.
const LHOPITAL_MAX: usize = 8;
/// Base tolerance for Cauchy-stability checks.
const CAUCHY_TOL: f64 = 1e-7;
/// Relative multiplier applied to `CAUCHY_TOL` when comparing probe values
/// (allows slow-converging limits to still pass).
const CAUCHY_FACTOR: f64 = 1000.0;
/// h-ladder for one-sided probing: values shrink toward the limit point.
const H_LADDER: [f64; 5] = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The point at which to compute a limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LimitPoint {
    /// A finite real number.
    Finite(f64),
    /// Positive infinity (+∞).
    PosInf,
    /// Negative infinity (−∞).
    NegInf,
}

/// Result of a limit computation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LimitResult {
    /// The limit is a finite real number.
    Finite(f64),
    /// The limit is +∞.
    PosInf,
    /// The limit is −∞.
    NegInf,
    /// The one-sided limits exist but differ, or the function oscillates
    /// without settling to any value.
    DoesNotExist,
    /// The limit is genuinely indeterminate: the L'Hôpital cap was exhausted
    /// and numeric probing did not converge to a stable value.
    Indeterminate,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Evaluate `op` with `Var(wrt) = x` and all other variables set to 0.
///
/// Pads the variable slice to at least `wrt + 1` entries so that `eval`
/// never panics on an out-of-bounds variable index.
fn eval_at_wrt(op: &LoweredOp, wrt: usize, x: f64) -> f64 {
    let needed = (wrt + 1).max(op.count_vars()).max(1);
    let mut vars = vec![0.0_f64; needed];
    vars[wrt] = x;
    op.eval(&vars)
}

/// Structurally substitute every `Var(wrt)` occurrence in `op` with
/// `replacement`, leaving all other nodes unchanged.
fn substitute(op: &LoweredOp, wrt: usize, replacement: &LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Var(i) => {
            if *i == wrt {
                replacement.clone()
            } else {
                op.clone()
            }
        }
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => op.clone(),
        LoweredOp::Neg(x) => LoweredOp::Neg(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Exp(x) => LoweredOp::Exp(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Ln(x) => LoweredOp::Ln(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Sin(x) => LoweredOp::Sin(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Cos(x) => LoweredOp::Cos(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Tan(x) => LoweredOp::Tan(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Sinh(x) => LoweredOp::Sinh(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Cosh(x) => LoweredOp::Cosh(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Tanh(x) => LoweredOp::Tanh(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Arcsin(x) => LoweredOp::Arcsin(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Arccos(x) => LoweredOp::Arccos(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Arctan(x) => LoweredOp::Arctan(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Arcsinh(x) => LoweredOp::Arcsinh(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Arccosh(x) => LoweredOp::Arccosh(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Arctanh(x) => LoweredOp::Arctanh(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Erf(x) => LoweredOp::Erf(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::LGamma(x) => LoweredOp::LGamma(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Digamma(x) => LoweredOp::Digamma(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Trigamma(x) => LoweredOp::Trigamma(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Ei(x) => LoweredOp::Ei(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Si(x) => LoweredOp::Si(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Ci(x) => LoweredOp::Ci(Arc::new(substitute(x, wrt, replacement))),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(substitute(a, wrt, replacement)),
            Arc::new(substitute(b, wrt, replacement)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(substitute(a, wrt, replacement)),
            Arc::new(substitute(b, wrt, replacement)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(substitute(a, wrt, replacement)),
            Arc::new(substitute(b, wrt, replacement)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            Arc::new(substitute(a, wrt, replacement)),
            Arc::new(substitute(b, wrt, replacement)),
        ),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            Arc::new(substitute(a, wrt, replacement)),
            Arc::new(substitute(b, wrt, replacement)),
        ),
    }
}

/// Probe `op` from one side of the point `c`, returning:
///
/// - A finite `f64` if the evaluated sequence is Cauchy-convergent.
/// - `f64::INFINITY` / `f64::NEG_INFINITY` if the values grow monotonically
///   without bound.
/// - `f64::NAN` if the values oscillate, contain NaN, or are otherwise
///   non-convergent.
fn probe_side(op: &LoweredOp, wrt: usize, c: f64, from_right: bool) -> f64 {
    let vals: Vec<f64> = H_LADDER
        .iter()
        .map(|&h| {
            let x = if from_right { c + h } else { c - h };
            eval_at_wrt(op, wrt, x)
        })
        .collect();

    let last = match vals.last() {
        Some(&v) => v,
        None => return f64::NAN,
    };
    let n = vals.len();
    let second_last = if n >= 2 { vals[n - 2] } else { f64::NAN };

    // Case A: last two values are consistently ±∞
    if last.is_infinite() && second_last.is_infinite() {
        return if last.signum() == second_last.signum() {
            last
        } else {
            f64::NAN // sign flip — oscillation between +∞ and -∞
        };
    }

    // Case B: last is ∞, second_last is large-finite (overflow just kicked in)
    if last.is_infinite()
        && second_last.is_finite()
        && second_last.abs() > 1.0
        && last.signum() == second_last.signum()
    {
        return last;
    }

    // Case C: any NaN in the last two → oscillation or domain error
    if last.is_nan() || second_last.is_nan() {
        return f64::NAN;
    }

    // Case D: both finite — check for rapid monotonic growth (divergence to ±∞)
    if last.is_finite() && second_last.is_finite() {
        if last.abs() > second_last.abs() * 10.0
            && last.abs() > 1.0
            && last.signum() == second_last.signum()
        {
            let first = vals.first().copied().unwrap_or(f64::NAN);
            if first.is_finite()
                && first.abs() > 0.01
                && last.abs() > first.abs() * 1000.0
                && last.signum() == first.signum()
            {
                return if last > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
            }
        }

        // Cauchy stability: last two values within tolerance
        let diff = (last - second_last).abs();
        let scale = last.abs().max(second_last.abs()).max(1.0);
        if diff <= CAUCHY_TOL * CAUCHY_FACTOR * scale {
            return last;
        }
    }

    // Otherwise: oscillation or non-convergence
    f64::NAN
}

/// Classify a limit result from left and right one-sided probes.
fn classify_probes(right: f64, left: f64) -> LimitResult {
    // Any NaN means oscillation on that side
    if right.is_nan() || left.is_nan() {
        return LimitResult::DoesNotExist;
    }

    match (right.is_infinite(), left.is_infinite()) {
        (true, true) => {
            if right > 0.0 && left > 0.0 {
                LimitResult::PosInf
            } else if right < 0.0 && left < 0.0 {
                LimitResult::NegInf
            } else {
                LimitResult::DoesNotExist // +∞ from one side, -∞ from the other
            }
        }
        (false, false) => {
            let scale = right.abs().max(left.abs()).max(1.0);
            if (right - left).abs() <= CAUCHY_TOL * CAUCHY_FACTOR * scale {
                LimitResult::Finite((right + left) / 2.0)
            } else {
                LimitResult::DoesNotExist
            }
        }
        _ => LimitResult::DoesNotExist, // one side finite, other infinite
    }
}

// ---------------------------------------------------------------------------
// Core limit algorithms
// ---------------------------------------------------------------------------

/// Compute the two-sided limit at a finite point `c`.
///
/// Algorithm:
///
/// 1. Direct substitution — if finite, validate with side probes.
/// 2. L'Hôpital — applied when a `0/0` or `∞/∞` `Div` form is detected.
/// 3. Numeric probing of both sides as fallback.
fn limit_at_finite(op: &LoweredOp, wrt: usize, c: f64, lhopital_count: usize) -> LimitResult {
    let v = eval_at_wrt(op, wrt, c);

    // -----------------------------------------------------------------------
    // Step 1: direct substitution gave a finite value — confirm with probes.
    // -----------------------------------------------------------------------
    if v.is_finite() {
        let right = probe_side(op, wrt, c, true);
        let left = probe_side(op, wrt, c, false);

        // Oscillation on at least one side
        if right.is_nan() || left.is_nan() {
            return LimitResult::DoesNotExist;
        }

        // At least one side diverges
        if right.is_infinite() || left.is_infinite() {
            return match (right.is_infinite(), left.is_infinite()) {
                (true, true) if right.signum() == left.signum() => {
                    if right > 0.0 {
                        LimitResult::PosInf
                    } else {
                        LimitResult::NegInf
                    }
                }
                _ => LimitResult::DoesNotExist,
            };
        }

        // Both sides finite
        let scale = right.abs().max(left.abs()).max(v.abs()).max(1.0);
        let tol = CAUCHY_TOL * CAUCHY_FACTOR * scale;

        if (right - v).abs() <= tol && (left - v).abs() <= tol {
            // Both sides converge to the direct-eval value
            return LimitResult::Finite(v);
        }
        if (right - left).abs() <= tol {
            // Sides agree with each other but not with the IEEE direct-eval
            // (removable singularity or indeterminate-form IEEE artifact).
            return LimitResult::Finite((right + left) / 2.0);
        }
        return LimitResult::DoesNotExist;
    }

    // -----------------------------------------------------------------------
    // Step 2: apply L'Hôpital for 0/0 or ∞/∞ Div forms.
    // -----------------------------------------------------------------------
    if lhopital_count < LHOPITAL_MAX {
        if let LoweredOp::Div(num, den) = op {
            let num_val = eval_at_wrt(num, wrt, c);
            let den_val = eval_at_wrt(den, wrt, c);

            let zero_over_zero = num_val.is_finite()
                && num_val.abs() < INDET_EPS
                && den_val.is_finite()
                && den_val.abs() < INDET_EPS;
            // Also catch near-infinite values (large finite or IEEE ±∞)
            let inf_over_inf = num_val.abs() > BIG && den_val.abs() > BIG;

            if zero_over_zero || inf_over_inf {
                let new_num = num.grad(wrt);
                let new_den = den.grad(wrt);
                let new_expr = LoweredOp::Div(Arc::new(new_num), Arc::new(new_den)).simplify();
                return limit_inner(&new_expr, wrt, LimitPoint::Finite(c), lhopital_count + 1);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: fall back to numeric probing from both sides.
    // -----------------------------------------------------------------------
    let right = probe_side(op, wrt, c, true);
    let left = probe_side(op, wrt, c, false);
    // classify_probes already handles NaN (oscillation) → DoesNotExist
    classify_probes(right, left)
}

/// Compute a one-sided limit at a finite point `c`, probing only from
/// `from_right`.
///
/// Used after substituting x = ±1/t to reduce an infinite-point limit to a
/// one-sided limit at t = 0⁺.
fn limit_at_finite_one_sided(
    op: &LoweredOp,
    wrt: usize,
    c: f64,
    from_right: bool,
    lhopital_count: usize,
) -> LimitResult {
    let v = eval_at_wrt(op, wrt, c);

    // -----------------------------------------------------------------------
    // Step 1: direct substitution — validate with a one-sided probe.
    // -----------------------------------------------------------------------
    if v.is_finite() {
        let side = probe_side(op, wrt, c, from_right);

        if side.is_nan() {
            return LimitResult::DoesNotExist;
        }
        if side.is_infinite() {
            return if side > 0.0 {
                LimitResult::PosInf
            } else {
                LimitResult::NegInf
            };
        }

        // Both finite: probe is trusted over the direct-eval value because
        // IEEE can produce misleading results for indeterminate forms such as
        // 1^∞ = 1 when the true limit is e.
        let scale = side.abs().max(v.abs()).max(1.0);
        if (side - v).abs() <= CAUCHY_TOL * CAUCHY_FACTOR * scale {
            return LimitResult::Finite(v);
        }
        // Probe converged to a value that differs from the direct eval —
        // trust the probe (e.g. (1+1/x)^x → e, not 1).
        return LimitResult::Finite(side);
    }

    // -----------------------------------------------------------------------
    // Step 2: apply L'Hôpital for 0/0 or ∞/∞ Div forms.
    // -----------------------------------------------------------------------
    if lhopital_count < LHOPITAL_MAX {
        if let LoweredOp::Div(num, den) = op {
            let num_val = eval_at_wrt(num, wrt, c);
            let den_val = eval_at_wrt(den, wrt, c);

            let zero_over_zero = num_val.is_finite()
                && num_val.abs() < INDET_EPS
                && den_val.is_finite()
                && den_val.abs() < INDET_EPS;
            // Also catch near-infinite values (large finite or IEEE ±∞)
            let inf_over_inf = num_val.abs() > BIG && den_val.abs() > BIG;

            if zero_over_zero || inf_over_inf {
                let new_num = num.grad(wrt);
                let new_den = den.grad(wrt);
                let new_expr = LoweredOp::Div(Arc::new(new_num), Arc::new(new_den)).simplify();
                return limit_at_finite_one_sided(
                    &new_expr,
                    wrt,
                    c,
                    from_right,
                    lhopital_count + 1,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: fall back to numeric probing from one side.
    // -----------------------------------------------------------------------
    let side = probe_side(op, wrt, c, from_right);

    if side.is_nan() {
        LimitResult::Indeterminate
    } else if side.is_infinite() {
        if side > 0.0 {
            LimitResult::PosInf
        } else {
            LimitResult::NegInf
        }
    } else {
        LimitResult::Finite(side)
    }
}

/// Central dispatch: routes to the appropriate finite/one-sided algorithm.
fn limit_inner(
    op: &LoweredOp,
    wrt: usize,
    point: LimitPoint,
    lhopital_count: usize,
) -> LimitResult {
    match point {
        LimitPoint::Finite(c) => limit_at_finite(op, wrt, c, lhopital_count),
        LimitPoint::PosInf => {
            // x → +∞  ⟺  t = 1/x → 0⁺
            let inv_t = LoweredOp::Div(
                Arc::new(LoweredOp::Const(1.0)),
                Arc::new(LoweredOp::Var(wrt)),
            );
            let substituted = substitute(op, wrt, &inv_t).simplify();
            limit_at_finite_one_sided(&substituted, wrt, 0.0, true, 0)
        }
        LimitPoint::NegInf => {
            // x → −∞  ⟺  t = −1/x → 0⁺
            let neg_inv_t = LoweredOp::Div(
                Arc::new(LoweredOp::Neg(Arc::new(LoweredOp::Const(1.0)))),
                Arc::new(LoweredOp::Var(wrt)),
            );
            let substituted = substitute(op, wrt, &neg_inv_t).simplify();
            limit_at_finite_one_sided(&substituted, wrt, 0.0, true, 0)
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl LoweredOp {
    /// Compute `lim_{x_{wrt} → point} self`.
    ///
    /// Uses numeric probing as the primary oracle, with L'Hôpital's rule
    /// applied symbolically when a `0/0` or `∞/∞` indeterminate `Div` form is
    /// detected.  Results for oscillatory functions or limits that cannot be
    /// resolved within the iteration cap are accurately reported as
    /// [`LimitResult::DoesNotExist`] or [`LimitResult::Indeterminate`] rather
    /// than silently wrong.
    pub fn limit(&self, wrt: usize, point: LimitPoint) -> LimitResult {
        limit_inner(&self.simplify(), wrt, point, 0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Convenience expression builders
    // ------------------------------------------------------------------

    fn var() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Div(Arc::new(a), Arc::new(b))
    }
    fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Sub(Arc::new(a), Arc::new(b))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Arc::new(a), Arc::new(b))
    }
    fn sin(a: LoweredOp) -> LoweredOp {
        LoweredOp::Sin(Arc::new(a))
    }
    fn cos(a: LoweredOp) -> LoweredOp {
        LoweredOp::Cos(Arc::new(a))
    }
    fn exp(a: LoweredOp) -> LoweredOp {
        LoweredOp::Exp(Arc::new(a))
    }
    fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(a), Arc::new(b))
    }

    fn assert_finite(result: LimitResult, expected: f64, tol: f64) {
        match result {
            LimitResult::Finite(v) => {
                assert!(
                    (v - expected).abs() <= tol,
                    "expected Finite({expected}), got Finite({v}); |diff|={}",
                    (v - expected).abs()
                );
            }
            other => panic!("expected Finite({expected}), got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Standard L'Hôpital / removable-singularity limits
    // ------------------------------------------------------------------

    // Test 1: sin(x)/x at x→0 → 1
    #[test]
    fn test_sinc_at_zero() {
        assert_finite(
            div(sin(var()), var()).limit(0, LimitPoint::Finite(0.0)),
            1.0,
            0.001,
        );
    }

    // Test 2: (1 − cos(x))/x² at x→0 → 0.5
    #[test]
    fn test_one_minus_cos_over_x_sq() {
        let expr = div(sub(c(1.0), cos(var())), pow(var(), c(2.0)));
        assert_finite(expr.limit(0, LimitPoint::Finite(0.0)), 0.5, 0.001);
    }

    // Test 3: (exp(x) − 1)/x at x→0 → 1
    #[test]
    fn test_exp_minus_one_over_x() {
        let expr = div(sub(exp(var()), c(1.0)), var());
        assert_finite(expr.limit(0, LimitPoint::Finite(0.0)), 1.0, 0.001);
    }

    // Test 4: (x² − 1)/(x − 1) at x→1 → 2
    #[test]
    fn test_x_sq_minus_1_over_x_minus_1() {
        let expr = div(sub(pow(var(), c(2.0)), c(1.0)), sub(var(), c(1.0)));
        assert_finite(expr.limit(0, LimitPoint::Finite(1.0)), 2.0, 0.001);
    }

    // ------------------------------------------------------------------
    // Does-not-exist cases
    // ------------------------------------------------------------------

    // Test 5: 1/x at x→0 — left = −∞, right = +∞
    #[test]
    fn test_one_over_x_at_zero_dne() {
        let expr = div(c(1.0), var());
        assert_eq!(
            expr.limit(0, LimitPoint::Finite(0.0)),
            LimitResult::DoesNotExist
        );
    }

    // Test 6: sin(1/x) at x→0 — oscillation
    #[test]
    fn test_sin_one_over_x_oscillates() {
        let expr = sin(div(c(1.0), var()));
        assert_eq!(
            expr.limit(0, LimitPoint::Finite(0.0)),
            LimitResult::DoesNotExist
        );
    }

    // ------------------------------------------------------------------
    // Limits at infinity
    // ------------------------------------------------------------------

    // Test 7: 1/x at x→+∞ → 0
    #[test]
    fn test_one_over_x_at_pos_inf() {
        assert_finite(div(c(1.0), var()).limit(0, LimitPoint::PosInf), 0.0, 0.01);
    }

    // Test 8: x/(x+1) at x→+∞ → 1
    #[test]
    fn test_x_over_x_plus_1_at_inf() {
        let expr = div(var(), add(var(), c(1.0)));
        assert_finite(expr.limit(0, LimitPoint::PosInf), 1.0, 0.01);
    }

    // Test 9: (1 + 1/x)^x at x→+∞ → e ≈ 2.718
    //
    // This is the classic 1^∞ indeterminate form.  After the x = 1/t
    // substitution the probe sequence converges well to e.
    #[test]
    fn test_one_plus_inv_x_pow_x() {
        let expr = pow(add(c(1.0), div(c(1.0), var())), var());
        let result = expr.limit(0, LimitPoint::PosInf);
        match result {
            LimitResult::Finite(v) => {
                assert!(
                    (v - std::f64::consts::E).abs() < 0.05,
                    "(1+1/x)^x at +∞: expected ≈e, got {v}"
                );
            }
            other => panic!("(1+1/x)^x at +∞: expected Finite(≈e), got {other:?}"),
        }
    }

    // Test 10: exp(x) at x→−∞ → 0
    #[test]
    fn test_exp_at_neg_inf() {
        assert_finite(exp(var()).limit(0, LimitPoint::NegInf), 0.0, 0.01);
    }

    // ------------------------------------------------------------------
    // Termination guarantee
    // ------------------------------------------------------------------

    // Test 11: the algorithm must return within a reasonable time even for
    // oscillatory or otherwise non-convergent expressions (no infinite loop).
    #[test]
    fn test_limit_terminates() {
        use std::time::Instant;
        // sin(1/x) oscillates without limit at x=0.
        let expr = sin(div(c(1.0), var()));
        let start = Instant::now();
        let result = expr.limit(0, LimitPoint::Finite(0.0));
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 10,
            "limit computation took too long: {elapsed:?}"
        );
        assert_eq!(result, LimitResult::DoesNotExist);
    }
}
