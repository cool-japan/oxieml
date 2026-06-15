//! Helper functions for u-substitution and trig substitution in symbolic integration.
//!
//! Extracted from `integrate.rs` to keep file size under 2000 lines.

use crate::lower::LoweredOp;
use crate::lower_simplify::ops_struct_hash;
use std::sync::Arc;

#[inline]
fn arc(op: LoweredOp) -> Arc<LoweredOp> {
    Arc::new(op)
}

/// Structural equality via hash comparison.
pub(crate) fn structurally_eq(a: &LoweredOp, b: &LoweredOp) -> bool {
    ops_struct_hash(a) == ops_struct_hash(b)
}

/// Collect all unique subtrees of `op` that contain `Var(wrt)`.
/// Skips `Var(wrt)` itself (that's the trivial substitution).
pub(crate) fn collect_subtrees(
    op: &LoweredOp,
    wrt: usize,
    seen: &mut Vec<u64>,
    out: &mut Vec<LoweredOp>,
) {
    if !op.contains_var(wrt) {
        return;
    }
    // Skip the trivial var itself
    if matches!(op, LoweredOp::Var(i) if *i == wrt) {
        return;
    }
    let h = ops_struct_hash(op);
    if seen.contains(&h) {
        return;
    }
    seen.push(h);
    out.push(op.clone());

    // Recurse into children
    match op {
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            collect_subtrees(a, wrt, seen, out);
            collect_subtrees(b, wrt, seen, out);
        }
        LoweredOp::Neg(a)
        | LoweredOp::Exp(a)
        | LoweredOp::Ln(a)
        | LoweredOp::Sin(a)
        | LoweredOp::Cos(a)
        | LoweredOp::Tan(a)
        | LoweredOp::Sinh(a)
        | LoweredOp::Cosh(a)
        | LoweredOp::Tanh(a)
        | LoweredOp::Arcsin(a)
        | LoweredOp::Arccos(a)
        | LoweredOp::Arctan(a)
        | LoweredOp::Arcsinh(a)
        | LoweredOp::Arccosh(a)
        | LoweredOp::Arctanh(a)
        | LoweredOp::Erf(a)
        | LoweredOp::LGamma(a)
        | LoweredOp::Digamma(a)
        | LoweredOp::Trigamma(a)
        | LoweredOp::Ei(a)
        | LoweredOp::Si(a)
        | LoweredOp::Ci(a) => {
            collect_subtrees(a, wrt, seen, out);
        }
        _ => {}
    }
}

/// Replace all `Var(wrt)` in `op` with `replacement`.
pub(crate) fn substitute_expr(op: &LoweredOp, wrt: usize, replacement: &LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Var(i) if *i == wrt => replacement.clone(),
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            arc(substitute_expr(a, wrt, replacement)),
            arc(substitute_expr(b, wrt, replacement)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            arc(substitute_expr(a, wrt, replacement)),
            arc(substitute_expr(b, wrt, replacement)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            arc(substitute_expr(a, wrt, replacement)),
            arc(substitute_expr(b, wrt, replacement)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            arc(substitute_expr(a, wrt, replacement)),
            arc(substitute_expr(b, wrt, replacement)),
        ),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            arc(substitute_expr(a, wrt, replacement)),
            arc(substitute_expr(b, wrt, replacement)),
        ),
        LoweredOp::Neg(a) => LoweredOp::Neg(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Exp(a) => LoweredOp::Exp(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Ln(a) => LoweredOp::Ln(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Sin(a) => LoweredOp::Sin(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Cos(a) => LoweredOp::Cos(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Tan(a) => LoweredOp::Tan(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Erf(a) => LoweredOp::Erf(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Ei(a) => LoweredOp::Ei(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Si(a) => LoweredOp::Si(arc(substitute_expr(a, wrt, replacement))),
        LoweredOp::Ci(a) => LoweredOp::Ci(arc(substitute_expr(a, wrt, replacement))),
    }
}

/// Replace subtrees that match `target_hash` with `Var(replacement_var)`.
pub(crate) fn substitute_by_hash(
    op: &LoweredOp,
    target_hash: u64,
    target: &LoweredOp,
    replacement_var: usize,
) -> LoweredOp {
    let h = ops_struct_hash(op);
    if h == target_hash && structurally_eq(op, target) {
        return LoweredOp::Var(replacement_var);
    }
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            arc(substitute_by_hash(a, target_hash, target, replacement_var)),
            arc(substitute_by_hash(b, target_hash, target, replacement_var)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            arc(substitute_by_hash(a, target_hash, target, replacement_var)),
            arc(substitute_by_hash(b, target_hash, target, replacement_var)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            arc(substitute_by_hash(a, target_hash, target, replacement_var)),
            arc(substitute_by_hash(b, target_hash, target, replacement_var)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            arc(substitute_by_hash(a, target_hash, target, replacement_var)),
            arc(substitute_by_hash(b, target_hash, target, replacement_var)),
        ),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            arc(substitute_by_hash(a, target_hash, target, replacement_var)),
            arc(substitute_by_hash(b, target_hash, target, replacement_var)),
        ),
        LoweredOp::Neg(a) => LoweredOp::Neg(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Exp(a) => LoweredOp::Exp(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Ln(a) => LoweredOp::Ln(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Sin(a) => LoweredOp::Sin(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Cos(a) => LoweredOp::Cos(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Tan(a) => LoweredOp::Tan(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Erf(a) => LoweredOp::Erf(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Ei(a) => LoweredOp::Ei(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Si(a) => LoweredOp::Si(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
        LoweredOp::Ci(a) => LoweredOp::Ci(arc(substitute_by_hash(
            a,
            target_hash,
            target,
            replacement_var,
        ))),
    }
}

/// Numeric verification: check that `anti`'s derivative numerically matches `orig` at test points.
/// Uses `wrt` as variable 0. Returns true when a majority of finite probe points agree.
///
/// The probe set spans multiple ranges so that functions with restricted domains (e.g.
/// `1/sqrt(1-x^2)` defined only on `(-1,1)`) still accumulate enough agreeing points,
/// while functions defined everywhere can use all of them.
fn numeric_verify_antiderivative(anti: &LoweredOp, orig: &LoweredOp, wrt: usize) -> bool {
    // Wide spread: small values inside restricted domains, plus larger values for unrestricted ones.
    let probe_points: [f64; 10] = [0.1, 0.3, 0.5, 0.7, -0.2, -0.5, 1.2, 1.8, 2.5, -0.4];
    let h = 1e-7_f64;
    let mut ok_count = 0usize;
    let mut finite_count = 0usize;
    for &xv in &probe_points {
        let f_plus = crate::integrate::eval_only_wrt(anti, wrt, xv + h);
        let f_minus = crate::integrate::eval_only_wrt(anti, wrt, xv);
        let fd = (f_plus - f_minus) / h;
        let fv = crate::integrate::eval_only_wrt(orig, wrt, xv);
        if !fd.is_finite() || !fv.is_finite() {
            continue;
        }
        finite_count += 1;
        let tol = if fv.abs() > 1e-10 {
            (fd / fv - 1.0).abs()
        } else {
            fd.abs()
        };
        if tol < 1e-3 {
            ok_count += 1;
        }
    }
    // Need at least 3 finite probe points and all finite ones must agree.
    finite_count >= 3 && ok_count == finite_count
}

/// Try trig substitution for `Pow(inner, expo_val)` where expo_val is +/-0.5
/// and `inner` is a degree-2 polynomial in `wrt`.
///
/// Handles:
/// - sqrt(a^2 - x^2) -> arcsin form
/// - 1/sqrt(a^2 - x^2) -> arcsin
/// - sqrt(a^2 + x^2) -> arcsinh form
/// - 1/sqrt(a^2 + x^2) -> arcsinh
pub(crate) fn try_trig_substitution(
    inner: &LoweredOp,
    expo_val: f64,
    wrt: usize,
) -> Option<LoweredOp> {
    use crate::poly::Poly;

    // Must be +/-0.5
    let is_sqrt = (expo_val - 0.5).abs() < 1e-12;
    let is_inv_sqrt = (expo_val + 0.5).abs() < 1e-12;
    if !is_sqrt && !is_inv_sqrt {
        return None;
    }

    // Try to parse inner as a degree-2 polynomial in wrt
    let poly = Poly::from_lowered(inner, wrt).ok()?;
    if poly.degree() != Some(2) {
        return None;
    }

    // Extract coefficients: inner = c0 + c1*x + c2*x^2
    let coeff_f64 = |idx: usize| -> f64 {
        poly.coeffs
            .get(idx)
            .map_or(0.0, |r| (*r.numer() as f64) / (*r.denom() as f64))
    };
    let c0 = coeff_f64(0);
    let c1 = coeff_f64(1);
    let c2 = coeff_f64(2);

    // We only handle pure a^2 +/- x^2 forms (c1 == 0 for clean trig sub)
    // And c2 must be +/-1 (or be handled by scaling)
    if c1.abs() > 1e-12 {
        return None; // non-standard form, skip
    }

    // c2 * x^2 + c0: need c0 > 0
    if c0 <= 0.0 {
        return None;
    }

    let a_sq = c0;
    let a = a_sq.sqrt();

    let anti: LoweredOp = if c2 < 0.0 {
        // inner = a^2 - |c2|*x^2
        let abs_c2 = c2.abs();
        let scale = abs_c2.sqrt();

        if is_inv_sqrt {
            // integral 1/sqrt(a^2 - |c2|*x^2) dx = (1/sqrt(|c2|)) * arcsin(sqrt(|c2|)*x/a)
            let scaled_x = LoweredOp::Div(
                arc(LoweredOp::Mul(
                    arc(LoweredOp::Const(scale)),
                    arc(LoweredOp::Var(wrt)),
                )),
                arc(LoweredOp::Const(a)),
            );
            LoweredOp::Div(
                arc(LoweredOp::Arcsin(arc(scaled_x))),
                arc(LoweredOp::Const(scale)),
            )
        } else {
            // integral sqrt(a^2 - |c2|*x^2) dx
            // = (x/2)*sqrt(inner) + (a^2/(2*sqrt(|c2|)))*arcsin(sqrt(|c2|)*x/a)
            let sqrt_inner = LoweredOp::Pow(arc(inner.clone()), arc(LoweredOp::Const(0.5)));
            let scaled_x = LoweredOp::Div(
                arc(LoweredOp::Mul(
                    arc(LoweredOp::Const(scale)),
                    arc(LoweredOp::Var(wrt)),
                )),
                arc(LoweredOp::Const(a)),
            );
            let term1 = LoweredOp::Div(
                arc(LoweredOp::Mul(arc(LoweredOp::Var(wrt)), arc(sqrt_inner))),
                arc(LoweredOp::Const(2.0)),
            );
            let term2 = LoweredOp::Mul(
                arc(LoweredOp::Const(a_sq / (2.0 * scale))),
                arc(LoweredOp::Arcsin(arc(scaled_x))),
            );
            LoweredOp::Add(arc(term1), arc(term2))
        }
    } else if c2 > 0.0 {
        // inner = a^2 + c2*x^2
        let scale = c2.sqrt();

        if is_inv_sqrt {
            // integral 1/sqrt(a^2 + c2*x^2) dx = (1/sqrt(c2))*arcsinh(sqrt(c2)*x/a)
            let scaled_x = LoweredOp::Div(
                arc(LoweredOp::Mul(
                    arc(LoweredOp::Const(scale)),
                    arc(LoweredOp::Var(wrt)),
                )),
                arc(LoweredOp::Const(a)),
            );
            LoweredOp::Div(
                arc(LoweredOp::Arcsinh(arc(scaled_x))),
                arc(LoweredOp::Const(scale)),
            )
        } else {
            // integral sqrt(a^2 + c2*x^2) dx
            // = (x/2)*sqrt(inner) + (a^2/(2*sqrt(c2)))*arcsinh(sqrt(c2)*x/a)
            let sqrt_inner = LoweredOp::Pow(arc(inner.clone()), arc(LoweredOp::Const(0.5)));
            let scaled_x = LoweredOp::Div(
                arc(LoweredOp::Mul(
                    arc(LoweredOp::Const(scale)),
                    arc(LoweredOp::Var(wrt)),
                )),
                arc(LoweredOp::Const(a)),
            );
            let term1 = LoweredOp::Div(
                arc(LoweredOp::Mul(arc(LoweredOp::Var(wrt)), arc(sqrt_inner))),
                arc(LoweredOp::Const(2.0)),
            );
            let term2 = LoweredOp::Mul(
                arc(LoweredOp::Const(a_sq / (2.0 * scale))),
                arc(LoweredOp::Arcsinh(arc(scaled_x))),
            );
            LoweredOp::Add(arc(term1), arc(term2))
        }
    } else {
        return None;
    };

    // Build the original expression to verify against
    let orig = LoweredOp::Pow(arc(inner.clone()), arc(LoweredOp::Const(expo_val)));
    let anti_simplified = anti.simplify();
    if numeric_verify_antiderivative(&anti_simplified, &orig, wrt) {
        Some(anti_simplified)
    } else {
        None
    }
}

/// Try general u-substitution: enumerate subtrees g of op, try u=g, check if
/// op = f(g(x)) * g'(x) for some f, integrate f(u) du, substitute back.
pub(crate) fn try_u_substitution(
    op: &LoweredOp,
    wrt: usize,
    depth: u32,
    by_parts_max: u32,
    raw_integrate: &impl Fn(&LoweredOp, usize, u32) -> Option<LoweredOp>,
) -> Option<LoweredOp> {
    if depth >= by_parts_max {
        return None;
    }

    // Collect candidate substitutions
    let mut seen_hashes: Vec<u64> = Vec::new();
    let mut candidates: Vec<LoweredOp> = Vec::new();
    collect_subtrees(op, wrt, &mut seen_hashes, &mut candidates);

    // Use a fresh variable index (u = wrt + 1, well above any used var)
    let u_var = wrt + 1;

    for g in &candidates {
        // Compute g'(x)
        let g_prime = g.grad(wrt).simplify();

        // Substitute g -> u in op: produces f(u) * g'(x_in_terms_of_u?)
        // We substitute every occurrence of g in op with Var(u_var)
        let g_hash = ops_struct_hash(g);
        let op_with_u = substitute_by_hash(op, g_hash, g, u_var);

        // Check that op_with_u / g_prime (in terms of x) yields something that
        // doesn't depend on x (only on u), meaning the substitution works.
        if op_with_u.contains_var(wrt) {
            continue; // g doesn't cover all x occurrences
        }

        // Now op_with_u = f(u)*g'(u...) but in pure u. We need to divide by g_prime
        // (as a function of x, expressed in u via substituting x back).
        let g_prime_in_u = substitute_by_hash(&g_prime, g_hash, g, u_var);

        // If g_prime_in_u still has x, skip
        if g_prime_in_u.contains_var(wrt) {
            continue;
        }

        // f(u) = op_with_u / g_prime_in_u
        let f_of_u = LoweredOp::Div(arc(op_with_u.clone()), arc(g_prime_in_u.clone()));

        // Try to integrate f(u) w.r.t. u_var
        let anti_u = raw_integrate(&f_of_u, u_var, depth + 1)?;

        // Substitute u back: u -> g(x)
        let anti_x = substitute_expr(&anti_u, u_var, g);
        let anti_simplified = anti_x.simplify();

        // Numeric verification: anti_simplified'(x) should equal op(x)
        if numeric_verify_antiderivative(&anti_simplified, op, wrt) {
            return Some(anti_simplified);
        }
    }

    None
}
