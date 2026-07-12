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

/// The evaluated ladder for one side of the point `c`.
fn probe_vals(op: &LoweredOp, wrt: usize, c: f64, from_right: bool) -> Vec<f64> {
    H_LADDER
        .iter()
        .map(|&h| {
            let x = if from_right { c + h } else { c - h };
            eval_at_wrt(op, wrt, x)
        })
        .collect()
}

/// Probe `op` from one side of the point `c`, returning:
///
/// - A finite `f64` if the evaluated sequence is Cauchy-convergent.
/// - `f64::INFINITY` / `f64::NEG_INFINITY` if the values grow monotonically
///   without bound.
/// - `f64::NAN` if the values oscillate, contain NaN, or are otherwise
///   non-convergent.
fn probe_side(op: &LoweredOp, wrt: usize, c: f64, from_right: bool) -> f64 {
    let vals = probe_vals(op, wrt, c, from_right);
    probe_side_process(&vals)
}

/// The distinct qualitative outcomes of probing one side of a point.
///
/// Distinguishes an *undefined* side (the function lies outside its real domain
/// there, e.g. `ln(x)` for `x < 0`) from an *oscillating* side (finite values
/// that never settle, e.g. `sin(1/x)`). The former lets a two-sided limit
/// collapse to the defined side (a domain-restricted limit such as
/// `lim_{x→0} x·ln x = 0`); the latter forces `DoesNotExist`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum SideOutcome {
    Finite(f64),
    PosInf,
    NegInf,
    /// Finite but non-convergent (oscillation).
    Oscillate,
    /// Outside the real domain on this side (every probe was non-finite).
    Undefined,
}

/// Classify one side of the point, distinguishing domain-undefined from
/// oscillation.
fn probe_side_classified(op: &LoweredOp, wrt: usize, c: f64, from_right: bool) -> SideOutcome {
    let vals = probe_vals(op, wrt, c, from_right);
    // Every probe non-finite ⇒ the function is outside its real domain here.
    if vals.iter().all(|v| !v.is_finite()) {
        return SideOutcome::Undefined;
    }
    let r = probe_side_process(&vals);
    if r.is_nan() {
        SideOutcome::Oscillate
    } else if r.is_infinite() {
        if r > 0.0 {
            SideOutcome::PosInf
        } else {
            SideOutcome::NegInf
        }
    } else {
        SideOutcome::Finite(r)
    }
}

/// Reduce a single classified side to a limit result (for domain-restricted
/// one-sided limits).
fn classify_single(side: SideOutcome) -> LimitResult {
    match side {
        SideOutcome::Finite(v) => LimitResult::Finite(v),
        SideOutcome::PosInf => LimitResult::PosInf,
        SideOutcome::NegInf => LimitResult::NegInf,
        SideOutcome::Oscillate | SideOutcome::Undefined => LimitResult::DoesNotExist,
    }
}

/// Process an already-evaluated ladder into a probe value (see [`probe_side`]).
fn probe_side_process(vals: &[f64]) -> f64 {
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

/// Classify a two-sided limit from the qualitative outcomes of both sides,
/// honouring domain restrictions: a side that is entirely outside the real
/// domain does not veto a limit that is well-defined on the other side.
fn classify_two_sided(right: SideOutcome, left: SideOutcome) -> LimitResult {
    use SideOutcome::{Finite, NegInf, Oscillate, PosInf, Undefined};
    match (right, left) {
        // Both outside the domain: the point is not approachable on the reals.
        (Undefined, Undefined) => LimitResult::DoesNotExist,
        // One side outside the domain ⇒ domain-restricted one-sided limit.
        (Undefined, other) | (other, Undefined) => classify_single(other),
        // Any oscillating (defined) side ⇒ no limit.
        (Oscillate, _) | (_, Oscillate) => LimitResult::DoesNotExist,
        (PosInf, PosInf) => LimitResult::PosInf,
        (NegInf, NegInf) => LimitResult::NegInf,
        (Finite(r), Finite(l)) => {
            let scale = r.abs().max(l.abs()).max(1.0);
            if (r - l).abs() <= CAUCHY_TOL * CAUCHY_FACTOR * scale {
                LimitResult::Finite((r + l) / 2.0)
            } else {
                LimitResult::DoesNotExist
            }
        }
        // Mixed +∞ / −∞ or finite-vs-infinite: no two-sided limit.
        _ => LimitResult::DoesNotExist,
    }
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
    // Domain-restricted limit: if the function is entirely outside its real
    // domain on one side, reduce to the one-sided limit on the defined side
    // (e.g. lim_{x→0} x·ln x = 0, ln undefined for x < 0). Only fires when a
    // whole side is non-finite; ordinary two-sided limits fall through.
    let right_kind = probe_side_classified(op, wrt, c, true);
    let left_kind = probe_side_classified(op, wrt, c, false);
    match (right_kind, left_kind) {
        (SideOutcome::Undefined, SideOutcome::Undefined) => {}
        (SideOutcome::Undefined, other) | (other, SideOutcome::Undefined) => {
            return classify_single(other);
        }
        _ => {}
    }

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
// Higher-level strategies: series expansion and indeterminate-form normalisation
// ---------------------------------------------------------------------------

/// Depth cap for the ordered-strategy recursion (series / normalisation).
const MAX_STRATEGY_DEPTH: u32 = 4;

/// A rough estimate of `lim op` used only to *classify* indeterminate forms
/// (is a factor going to `0`, to `∞`, …). Never used as a final answer.
fn approx_value(op: &LoweredOp, wrt: usize, point: LimitPoint) -> f64 {
    match point {
        LimitPoint::Finite(c) => {
            let r = probe_side(op, wrt, c, true);
            if r.is_nan() {
                probe_side(op, wrt, c, false)
            } else {
                r
            }
        }
        LimitPoint::PosInf => eval_at_wrt(op, wrt, 1e13),
        LimitPoint::NegInf => eval_at_wrt(op, wrt, -1e13),
    }
}

/// The domain-aware two-sided numeric limit at a finite point (used to
/// cross-check the series verdict).
fn numeric_two_sided(op: &LoweredOp, wrt: usize, c: f64) -> LimitResult {
    let right = probe_side_classified(op, wrt, c, true);
    let left = probe_side_classified(op, wrt, c, false);
    classify_two_sided(right, left)
}

/// Whether two limit verdicts are consistent (same category; finite values
/// close). Used to gate the acceptance of a symbolic/series answer against a
/// numeric one so no fabricated value is ever returned.
fn results_consistent(a: LimitResult, b: LimitResult) -> bool {
    match (a, b) {
        (LimitResult::Finite(x), LimitResult::Finite(y)) => {
            let scale = x.abs().max(y.abs()).max(1.0);
            (x - y).abs() <= 1e-3 * scale
        }
        (LimitResult::PosInf, LimitResult::PosInf)
        | (LimitResult::NegInf, LimitResult::NegInf)
        | (LimitResult::DoesNotExist, LimitResult::DoesNotExist)
        | (LimitResult::Indeterminate, LimitResult::Indeterminate) => true,
        _ => false,
    }
}

/// Strategy A — rigorous Laurent-series limit at a finite point, accepted only
/// when it agrees with an independent numeric probe (honesty cross-check).
fn series_strategy(op: &LoweredOp, wrt: usize, c: f64) -> Option<LimitResult> {
    let verdict = crate::series_laurent::limit_from_series(op, wrt, c)?;
    let numeric = numeric_two_sided(op, wrt, c);
    if results_consistent(verdict, numeric) {
        Some(verdict)
    } else {
        None
    }
}

/// Rewrite one indeterminate form into an equivalent expression better suited to
/// L'Hôpital / series evaluation, given the qualitative behaviour of the parts
/// at `point`:
///
/// * `∞ − ∞`  →  a single fraction via [`LoweredOp::together`] (P2), and
/// * `0 · ∞`  →  a quotient `(→∞) / (1 / (→0))`, an `∞/∞` form.
///
/// Power forms (`0⁰`, `1^∞`, `∞⁰`) are handled by [`limit_power_form`], which
/// applies `a^b = exp(b · ln a)`. Returns `None` when `op` is not a recognised
/// indeterminate form at `point`.
fn normalize_indeterminate(op: &LoweredOp, wrt: usize, point: LimitPoint) -> Option<LoweredOp> {
    match op {
        LoweredOp::Sub(a, b) => {
            let av = approx_value(a, wrt, point);
            let bv = approx_value(b, wrt, point);
            if av.abs() > BIG && bv.abs() > BIG && av.signum() == bv.signum() {
                let combined = op.together();
                if &combined != op {
                    return Some(combined);
                }
            }
            None
        }
        LoweredOp::Mul(a, b) => {
            let av = approx_value(a, wrt, point);
            let bv = approx_value(b, wrt, point);
            let a_zero = av.abs() < INDET_EPS;
            let b_zero = bv.abs() < INDET_EPS;
            let a_inf = av.abs() > BIG;
            let b_inf = bv.abs() > BIG;
            if a_zero && b_inf {
                // b / (1/a)  =  ∞ / ∞
                let recip_a = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), a.clone());
                return Some(LoweredOp::Div(b.clone(), Arc::new(recip_a)));
            }
            if b_zero && a_inf {
                let recip_b = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), b.clone());
                return Some(LoweredOp::Div(a.clone(), Arc::new(recip_b)));
            }
            None
        }
        _ => None,
    }
}

/// Resolve an indeterminate power `base^exp` (forms `0⁰`, `1^∞`, `∞⁰`) via
/// `a^b = exp(b · ln a)`, so `lim a^b = exp(lim b·ln a)`. Requires `base > 0`
/// near the point (real logarithm). Accepted only when it agrees with the
/// numeric limit of `base^exp`.
fn limit_power_form(
    op: &LoweredOp,
    base: &Arc<LoweredOp>,
    exp: &Arc<LoweredOp>,
    wrt: usize,
    point: LimitPoint,
    depth: u32,
) -> Option<LimitResult> {
    let base_v = approx_value(base, wrt, point);
    let exp_v = approx_value(exp, wrt, point);
    // Real logarithm requires a positive base near the point.
    if base_v.is_nan() || base_v <= 0.0 {
        return None;
    }
    // Only engage for genuinely indeterminate forms: base → {0⁺, 1, ∞} and
    // exp → {0, ∞}.
    let base_indet = base_v.abs() < INDET_EPS || (base_v - 1.0).abs() < INDET_EPS || base_v > BIG;
    let exp_indet = exp_v.abs() < INDET_EPS || exp_v.abs() > BIG;
    if !(base_indet && exp_indet) {
        return None;
    }

    // g = exp · ln(base)
    let g = LoweredOp::Mul(exp.clone(), Arc::new(LoweredOp::Ln(base.clone()))).simplify();
    let inner = limit_top(&g, wrt, point, depth + 1);
    let verdict = match inner {
        LimitResult::Finite(l) => {
            let v = l.exp();
            if v.is_finite() {
                LimitResult::Finite(v)
            } else {
                LimitResult::PosInf
            }
        }
        LimitResult::NegInf => LimitResult::Finite(0.0),
        LimitResult::PosInf => LimitResult::PosInf,
        _ => return None,
    };

    let reference = limit_inner(op, wrt, point, 0);
    if results_consistent(verdict, reference) {
        Some(verdict)
    } else {
        None
    }
}

/// Strategy B — normalise a recognised indeterminate form, evaluate the
/// normalised expression, and accept the result only when it agrees with the
/// established numeric engine on the *original* expression.
fn normalization_strategy(
    op: &LoweredOp,
    wrt: usize,
    point: LimitPoint,
    depth: u32,
) -> Option<LimitResult> {
    if let LoweredOp::Pow(base, exp) = op {
        if let Some(r) = limit_power_form(op, base, exp, wrt, point, depth) {
            return Some(r);
        }
    }
    if let Some(norm) = normalize_indeterminate(op, wrt, point) {
        let candidate = limit_top(&norm, wrt, point, depth + 1);
        let reference = limit_inner(op, wrt, point, 0);
        if results_consistent(candidate, reference) {
            return Some(candidate);
        }
    }
    None
}

/// Ordered strategy dispatch: series (finite points) → indeterminate-form
/// normalisation → the L'Hôpital + numeric-probing engine.
fn limit_top(op: &LoweredOp, wrt: usize, point: LimitPoint, depth: u32) -> LimitResult {
    if depth < MAX_STRATEGY_DEPTH {
        if let LimitPoint::Finite(c) = point {
            if let Some(r) = series_strategy(op, wrt, c) {
                return r;
            }
        }
        if let Some(r) = normalization_strategy(op, wrt, point, depth) {
            return r;
        }
    }
    limit_inner(op, wrt, point, 0)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl LoweredOp {
    /// Compute `lim_{x_{wrt} → point} self`.
    ///
    /// Resolves the limit with an ordered cascade of strategies, each guarded by
    /// a numeric cross-check so a value is never fabricated:
    ///
    /// 1. **Laurent series** at finite points — rigorous for rational,
    ///    removable, and pole limits (see [`crate::series_laurent`]).
    /// 2. **Indeterminate-form normalisation** — rewrites `0·∞`, `∞−∞` and the
    ///    power forms `0⁰`, `1^∞`, `∞⁰` into shapes amenable to L'Hôpital.
    /// 3. **L'Hôpital + numeric probing** — the general fallback.
    ///
    /// Oscillatory functions, one-sided disagreements, and limits that no
    /// strategy can settle are reported as [`LimitResult::DoesNotExist`] or
    /// [`LimitResult::Indeterminate`] rather than as silently wrong values.
    /// A side that lies entirely outside the real domain (e.g. `ln(x)` for
    /// `x < 0`) does not veto a limit that is well-defined on the other side,
    /// so domain-restricted limits such as `lim_{x→0} x·ln x = 0` are found.
    pub fn limit(&self, wrt: usize, point: LimitPoint) -> LimitResult {
        limit_top(&self.simplify(), wrt, point, 0)
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

    // ------------------------------------------------------------------
    // P5 indeterminate-form spec bullets
    // ------------------------------------------------------------------

    fn ln(a: LoweredOp) -> LoweredOp {
        LoweredOp::Ln(Arc::new(a))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Arc::new(a), Arc::new(b))
    }

    // Spec: lim_{x→0} x·ln x = 0  (the 0·∞ form; ln is defined only for x > 0,
    // so this is a domain-restricted limit from the right).
    #[test]
    fn test_x_ln_x_zero() {
        let expr = mul(var(), ln(var()));
        assert_finite(expr.limit(0, LimitPoint::Finite(0.0)), 0.0, 1e-3);
    }

    // Spec: lim_{x→+∞} (1 + 1/x)^x = e  (the 1^∞ form).
    #[test]
    fn test_one_plus_inv_x_pow_x_equals_e() {
        let expr = pow(add(c(1.0), div(c(1.0), var())), var());
        assert_finite(expr.limit(0, LimitPoint::PosInf), std::f64::consts::E, 0.05);
    }

    // Spec: lim_{x→0} sin(1/x) = DoesNotExist  (oscillation — must NOT be
    // fabricated as a value).
    #[test]
    fn test_sin_inv_x_does_not_exist() {
        let expr = sin(div(c(1.0), var()));
        assert_eq!(
            expr.limit(0, LimitPoint::Finite(0.0)),
            LimitResult::DoesNotExist
        );
    }

    // ∞ − ∞: lim_{x→0} (1/x − 1/sin x) = 0  (resolved by the Laurent-series
    // strategy: 1/x − 1/sin x = −x/6 + O(x³)).
    #[test]
    fn test_inf_minus_inf_zero() {
        let expr = sub(div(c(1.0), var()), div(c(1.0), sin(var())));
        assert_finite(expr.limit(0, LimitPoint::Finite(0.0)), 0.0, 1e-3);
    }

    // 0⁰: lim_{x→0⁺} x^x = 1.
    #[test]
    fn test_x_pow_x_one() {
        let expr = pow(var(), var());
        // x^x is defined only for x > 0; the domain-restricted limit is 1.
        match expr.limit(0, LimitPoint::Finite(0.0)) {
            LimitResult::Finite(v) => {
                assert!((v - 1.0).abs() < 1e-2, "x^x → 1, got {v}");
            }
            other => panic!("x^x at 0⁺: expected Finite(≈1), got {other:?}"),
        }
    }
}
