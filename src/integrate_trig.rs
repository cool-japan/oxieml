//! Deep trigonometric symbolic integration: power/product reduction formulas,
//! the product-to-sum table, the Weierstrass `t = tan(x/2)` substitution, and
//! the by-parts "solve-for-I" cyclic technique.
//!
//! This module is dispatched to from [`crate::integrate`]; it never produces a
//! result without first re-checking it by differentiation
//! ([`verify_antiderivative`]), so a sign error in any formula degrades to an
//! honest `None` (⇒ `Unsupported`) rather than a wrong closed form.
//!
//! # 1. Powers of a single trig function `∫ tⁿ(a·x+b) dx`
//!
//! Every rule is derived for the *argument as the integration variable*
//! (`H_n(u) = ∫ tⁿ(u) du`) and the chain rule contributes a single `1/a`
//! afterwards, since for `u = a·x + b`, `∫ tⁿ(u) dx = (1/a)·H_n(u)`.
//!
//! ```text
//! ∫ sinⁿ u du = −sinⁿ⁻¹u·cos u / n + (n−1)/n · ∫ sinⁿ⁻² u du       (H₀=u, H₁=−cos u)
//! ∫ cosⁿ u du =  cosⁿ⁻¹u·sin u / n + (n−1)/n · ∫ cosⁿ⁻² u du       (H₀=u, H₁= sin u)
//! ∫ tanⁿ u du =  tanⁿ⁻¹u/(n−1)      −         ∫ tanⁿ⁻² u du        (H₀=u, H₁=−ln cos u)
//! ∫ secⁿ u du =  secⁿ⁻²u·tan u/(n−1) + (n−2)/(n−1) · ∫ secⁿ⁻² u du (S₀=u, S₁= ln|sec u+tan u|)
//! ∫ cscⁿ u du = −cscⁿ⁻²u·cot u/(n−1) + (n−2)/(n−1) · ∫ cscⁿ⁻² u du (C₀=u, C₁=−ln|csc u+cot u|)
//! ∫ cotⁿ u du = −cotⁿ⁻¹u/(n−1)      −         ∫ cotⁿ⁻² u du        (T₀=u, T₁= ln|sin u|)
//! ```
//!
//! `secⁿ`/`cscⁿ`/`cotⁿ` are matched as `Pow(Cos u, −n)` / `Pow(Sin u, −n)` and so
//! on, since [`LoweredOp`] has no dedicated `sec`/`csc`/`cot` node. The recursion
//! is capped at `|n| ≤ 20` (per the P4 spec); beyond that this module returns
//! `None` honestly.
//!
//! # 2. Products `∫ sinᵐ u cosⁿ u du`
//!
//! * `n` odd → substitute `w = sin u`: `∫ wᵐ (1−w²)^{(n−1)/2} dw` (a polynomial).
//! * `m` odd (`n` even) → substitute `w = cos u`:
//!   `−∫ wⁿ (1−w²)^{(m−1)/2} dw`.
//! * both even → expand `cosⁿ = (1−sin²)^{n/2}` and sum `sinᵏ` reductions.
//!
//! # 3. Product-to-sum `∫ sin(ax) cos(bx) dx`
//!
//! [`product_to_sum`] rewrites a product of two `sin`/`cos` with *distinct*
//! arguments `A`, `B` using
//!
//! ```text
//! sin A cos B = ½[sin(A+B) + sin(A−B)]     cos A sin B = ½[sin(A+B) − sin(A−B)]
//! sin A sin B = ½[cos(A−B) − cos(A+B)]     cos A cos B = ½[cos(A−B) + cos(A+B)]
//! ```
//!
//! `product_to_sum` is deliberately `pub(crate)` and integrand-agnostic: the P3
//! Fourier/series work will import it to expand trigonometric products, so it
//! returns the rewritten *expression* (a sum of `sin`/`cos` of `A±B`) rather
//! than an antiderivative.
//!
//! # 4. Weierstrass `t = tan(x/2)`
//!
//! Any `R(sin(a·x+b), cos(a·x+b))` that is *rational* in the two trig functions
//! rationalises under
//!
//! ```text
//! sin u = 2t/(1+t²)    cos u = (1−t²)/(1+t²)    tan u = 2t/(1−t²)    dx = (2/a)/(1+t²) dt
//! ```
//!
//! The rationalised integrand is handed to the shared partial-fraction engine
//! ([`crate::integrate::integrate_rational`]) and the result is back-substituted
//! `t → tan((a·x+b)/2)`. The predicate requires **one** common **affine** trig
//! argument, which is exactly what keeps non-elementary integrands such as
//! `∫ sin(x²)` out (its argument `x²` is not affine) — they stay `Unsupported`.
//!
//! # 5. Solve-for-I `∫ eᵃˣ sin(bx) dx`
//!
//! Two integrations by parts return a multiple of the original integral; solving
//! the resulting linear equation gives the closed forms
//!
//! ```text
//! ∫ eᶠ sin g dx = eᶠ (a·sin g − b·cos g)/(a²+b²)
//! ∫ eᶠ cos g dx = eᶠ (a·cos g + b·sin g)/(a²+b²)      (f = a·x+c, g = b·x+d)
//! ```
//!
//! which [`try_solve_for_i`] emits directly (the by-parts cycle is what these
//! formulas *are*), superseding the cycle-hash bail in
//! [`crate::integrate`]'s `by_parts` for this family.

use crate::integrate::{affine_arg, eval_only_wrt};
use crate::integrate_subst::{structurally_eq, substitute_expr};
use crate::lower::LoweredOp;
use std::sync::Arc;

/// Reduction-recursion / product-degree cap (P4 spec: `n ≤ 20`).
const POWER_CAP: i64 = 20;

// ── small LoweredOp constructors ──────────────────────────────────────────────

#[inline]
fn arc(op: LoweredOp) -> Arc<LoweredOp> {
    Arc::new(op)
}
#[inline]
fn konst(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
#[inline]
fn var(w: usize) -> LoweredOp {
    LoweredOp::Var(w)
}
#[inline]
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(arc(a), arc(b))
}
#[inline]
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(arc(a), arc(b))
}
#[inline]
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(arc(a), arc(b))
}
#[inline]
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(arc(a), arc(b))
}
#[inline]
fn neg(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(arc(a))
}
#[inline]
fn sin(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(arc(a))
}
#[inline]
fn cos(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(arc(a))
}
#[inline]
fn tan(a: LoweredOp) -> LoweredOp {
    LoweredOp::Tan(arc(a))
}
#[inline]
fn exp(a: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(arc(a))
}
#[inline]
fn ln(a: LoweredOp) -> LoweredOp {
    LoweredOp::Ln(arc(a))
}
#[inline]
fn powf(base: LoweredOp, e: f64) -> LoweredOp {
    LoweredOp::Pow(arc(base), arc(konst(e)))
}

/// Recognised elementary trigonometric heads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigKind {
    Sin,
    Cos,
    Tan,
}

/// Read a numeric constant (`Const`/`NamedConst`) out of `op`.
fn as_const(op: &LoweredOp) -> Option<f64> {
    match op {
        LoweredOp::Const(c) => Some(*c),
        LoweredOp::NamedConst(nc) => Some(nc.value()),
        _ => None,
    }
}

/// If `op` is `Sin`/`Cos`/`Tan`, return the head and (a clone of) its argument.
fn trig_head(op: &LoweredOp) -> Option<(TrigKind, LoweredOp)> {
    match op {
        LoweredOp::Sin(a) => Some((TrigKind::Sin, (**a).clone())),
        LoweredOp::Cos(a) => Some((TrigKind::Cos, (**a).clone())),
        LoweredOp::Tan(a) => Some((TrigKind::Tan, (**a).clone())),
        _ => None,
    }
}

/// Central-difference check that `d/dx anti ≈ orig` at fixed probe points.
///
/// Non-finite probes (domain edges, singularities) are skipped; every finite
/// probe must agree to `1e-4` relative and at least three must be finite. This
/// is the single gate every public entry point in this module passes its result
/// through before returning `Some`.
fn verify_antiderivative(anti: &LoweredOp, orig: &LoweredOp, wrt: usize) -> bool {
    const PROBES: [f64; 10] = [0.1, 0.3, 0.5, 0.7, -0.2, -0.5, 1.2, 1.8, 2.5, -0.4];
    let h = 1e-6_f64;
    let mut finite = 0usize;
    let mut agree = 0usize;
    for &xv in &PROBES {
        let fd = (eval_only_wrt(anti, wrt, xv + h) - eval_only_wrt(anti, wrt, xv - h)) / (2.0 * h);
        let fv = eval_only_wrt(orig, wrt, xv);
        if !fd.is_finite() || !fv.is_finite() {
            continue;
        }
        finite += 1;
        let ok = if fv.abs() > 1e-10 {
            (fd / fv - 1.0).abs() < 1e-4
        } else {
            fd.abs() < 1e-6
        };
        if ok {
            agree += 1;
        }
    }
    finite >= 3 && agree == finite
}

// ── binomial helper ────────────────────────────────────────────────────────────

/// Exact small binomial coefficient `C(n, k)` as an `f64` (all inputs ≤ 20).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut r = 1.0_f64;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

// ── §1  single-power reduction formulas (antiderivative w.r.t. the argument) ───

/// `∫ sinⁿ(u) du`, `n ≥ 0`, expressed in `arg = u`.
fn sin_power_antideriv(arg: &LoweredOp, n: i64) -> LoweredOp {
    if n == 0 {
        return arg.clone();
    }
    if n == 1 {
        return neg(cos(arg.clone()));
    }
    let nf = n as f64;
    let head = neg(div(
        mul(powf(sin(arg.clone()), (n - 1) as f64), cos(arg.clone())),
        konst(nf),
    ));
    let tail = mul(konst((nf - 1.0) / nf), sin_power_antideriv(arg, n - 2));
    add(head, tail)
}

/// `∫ cosⁿ(u) du`, `n ≥ 0`, expressed in `arg = u`.
fn cos_power_antideriv(arg: &LoweredOp, n: i64) -> LoweredOp {
    if n == 0 {
        return arg.clone();
    }
    if n == 1 {
        return sin(arg.clone());
    }
    let nf = n as f64;
    let head = div(
        mul(powf(cos(arg.clone()), (n - 1) as f64), sin(arg.clone())),
        konst(nf),
    );
    let tail = mul(konst((nf - 1.0) / nf), cos_power_antideriv(arg, n - 2));
    add(head, tail)
}

/// `∫ tanⁿ(u) du`, `n ≥ 0`, expressed in `arg = u`.
fn tan_power_antideriv(arg: &LoweredOp, n: i64) -> LoweredOp {
    if n == 0 {
        return arg.clone();
    }
    if n == 1 {
        return neg(ln(cos(arg.clone())));
    }
    let head = div(
        powf(tan(arg.clone()), (n - 1) as f64),
        konst((n - 1) as f64),
    );
    sub(head, tan_power_antideriv(arg, n - 2))
}

/// `secᵏ(u) = 1/cosᵏ(u)` as a `LoweredOp` (`k = 0 → 1`).
fn sec_pow(arg: &LoweredOp, k: i64) -> LoweredOp {
    if k == 0 {
        konst(1.0)
    } else {
        div(konst(1.0), powf(cos(arg.clone()), k as f64))
    }
}

/// `cscᵏ(u) = 1/sinᵏ(u)` as a `LoweredOp` (`k = 0 → 1`).
fn csc_pow(arg: &LoweredOp, k: i64) -> LoweredOp {
    if k == 0 {
        konst(1.0)
    } else {
        div(konst(1.0), powf(sin(arg.clone()), k as f64))
    }
}

/// `∫ secⁿ(u) du`, `n ≥ 0`, expressed in `arg = u`.
fn sec_power_antideriv(arg: &LoweredOp, n: i64) -> LoweredOp {
    if n == 0 {
        return arg.clone();
    }
    if n == 1 {
        // ln|sec u + tan u|
        return ln(add(div(konst(1.0), cos(arg.clone())), tan(arg.clone())));
    }
    let head = div(
        mul(sec_pow(arg, n - 2), tan(arg.clone())),
        konst((n - 1) as f64),
    );
    let tail = mul(
        konst((n - 2) as f64 / (n - 1) as f64),
        sec_power_antideriv(arg, n - 2),
    );
    add(head, tail)
}

/// `∫ cscⁿ(u) du`, `n ≥ 0`, expressed in `arg = u`.
fn csc_power_antideriv(arg: &LoweredOp, n: i64) -> LoweredOp {
    if n == 0 {
        return arg.clone();
    }
    if n == 1 {
        // −ln|csc u + cot u| = −ln((1 + cos u)/sin u)
        return neg(ln(div(add(konst(1.0), cos(arg.clone())), sin(arg.clone()))));
    }
    let cot = div(cos(arg.clone()), sin(arg.clone()));
    let head = neg(div(mul(csc_pow(arg, n - 2), cot), konst((n - 1) as f64)));
    let tail = mul(
        konst((n - 2) as f64 / (n - 1) as f64),
        csc_power_antideriv(arg, n - 2),
    );
    add(head, tail)
}

/// `∫ cotⁿ(u) du`, `n ≥ 0`, expressed in `arg = u`.
fn cot_power_antideriv(arg: &LoweredOp, n: i64) -> LoweredOp {
    if n == 0 {
        return arg.clone();
    }
    if n == 1 {
        return ln(sin(arg.clone()));
    }
    let cot_pow = div(
        powf(cos(arg.clone()), (n - 1) as f64),
        powf(sin(arg.clone()), (n - 1) as f64),
    );
    let head = neg(div(cot_pow, konst((n - 1) as f64)));
    sub(head, cot_power_antideriv(arg, n - 2))
}

/// Attempt `∫ tⁿ(a·x+b) dx` for a single trig power `Pow(t(a·x+b), n)`.
///
/// Handles `sinⁿ`/`cosⁿ`/`tanⁿ` for `2 ≤ n ≤ 20` and their reciprocals
/// `cscⁿ`/`secⁿ`/`cotⁿ` (matched as negative exponents) for `1 ≤ |n| ≤ 20`.
pub(crate) fn try_trig_power(op: &LoweredOp, wrt: usize) -> Option<LoweredOp> {
    let LoweredOp::Pow(base, expo) = op else {
        return None;
    };
    let exp_val = as_const(expo)?;
    if exp_val.fract() != 0.0 {
        return None;
    }
    let n = exp_val as i64;
    if n.abs() > POWER_CAP || n == 0 {
        return None;
    }
    let (kind, arg) = trig_head(base)?;
    let (a_coef, _) = affine_arg(&arg, wrt)?;

    let antideriv_in_u = match (kind, n) {
        (TrigKind::Sin, n) if n >= 2 => sin_power_antideriv(&arg, n),
        (TrigKind::Cos, n) if n >= 2 => cos_power_antideriv(&arg, n),
        (TrigKind::Tan, n) if n >= 2 => tan_power_antideriv(&arg, n),
        (TrigKind::Sin, n) if n <= -1 => csc_power_antideriv(&arg, -n),
        (TrigKind::Cos, n) if n <= -1 => sec_power_antideriv(&arg, -n),
        (TrigKind::Tan, n) if n <= -1 => cot_power_antideriv(&arg, -n),
        _ => return None,
    };

    let result = div(antideriv_in_u, konst(a_coef)).simplify();
    verify_antiderivative(&result, op, wrt).then_some(result)
}

// ── §2  sinᵐ cosⁿ products ─────────────────────────────────────────────────────

/// Build `Σ coeffs[k]·base^{k}` for `k ≥ 1` (index 0 is the integration constant,
/// dropped). Near-zero coefficients are skipped.
fn poly_in_base(coeffs: &[f64], base: &LoweredOp) -> LoweredOp {
    let mut acc: Option<LoweredOp> = None;
    for (k, &c) in coeffs.iter().enumerate() {
        if k == 0 || c.abs() < 1e-14 {
            continue;
        }
        let term = mul(konst(c), powf(base.clone(), k as f64));
        acc = Some(match acc {
            Some(prev) => add(prev, term),
            None => term,
        });
    }
    acc.unwrap_or_else(|| konst(0.0))
}

/// Antiderivative of the `w`-polynomial `∫ (Σ p[k] wᵏ) dw = Σ p[k] w^{k+1}/(k+1)`,
/// returned as its coefficient vector indexed by output power.
fn integrate_w_poly(p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; p.len() + 1];
    for (k, &c) in p.iter().enumerate() {
        out[k + 1] = c / (k + 1) as f64;
    }
    out
}

/// Coefficients (indexed by power of `w`) of `wᵖ · (1 − w²)ᵏ`.
fn shifted_one_minus_w_sq_pow(p: usize, k: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; p + 2 * k + 1];
    for j in 0..=k {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        coeffs[p + 2 * j] += binom(k, j) * sign;
    }
    coeffs
}

/// `∫ sinᵐ(u) cosⁿ(u) du`, `m, n ≥ 0`, expressed in `arg = u`.
fn sin_cos_power_antideriv(arg: &LoweredOp, m: i64, n: i64) -> Option<LoweredOp> {
    if m < 0 || n < 0 || m + n > POWER_CAP {
        return None;
    }
    let (m, n) = (m as usize, n as usize);

    if n % 2 == 1 {
        // w = sin u: ∫ wᵐ (1−w²)^{(n−1)/2} dw, then w = sin u.
        let p = shifted_one_minus_w_sq_pow(m, (n - 1) / 2);
        let anti = integrate_w_poly(&p);
        Some(poly_in_base(&anti, &sin(arg.clone())))
    } else if m % 2 == 1 {
        // w = cos u: −∫ wⁿ (1−w²)^{(m−1)/2} dw, then w = cos u.
        let p = shifted_one_minus_w_sq_pow(n, (m - 1) / 2);
        let anti = integrate_w_poly(&p);
        Some(neg(poly_in_base(&anti, &cos(arg.clone()))))
    } else {
        // both even: cosⁿ = (1−sin²)^{n/2}; sum sinᵏ reductions.
        let half = n / 2;
        let mut acc: Option<LoweredOp> = None;
        for j in 0..=half {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let coeff = binom(half, j) * sign;
            if coeff.abs() < 1e-14 {
                continue;
            }
            let term = mul(konst(coeff), sin_power_antideriv(arg, (m + 2 * j) as i64));
            acc = Some(match acc {
                Some(prev) => add(prev, term),
                None => term,
            });
        }
        Some(acc.unwrap_or_else(|| arg.clone()))
    }
}

// ── §3  product-to-sum ─────────────────────────────────────────────────────────

/// Split `op` into `(is_sin, arg)` when it is a bare `Sin`/`Cos` node.
fn as_sin_or_cos(op: &LoweredOp) -> Option<(bool, LoweredOp)> {
    match op {
        LoweredOp::Sin(a) => Some((true, (**a).clone())),
        LoweredOp::Cos(a) => Some((false, (**a).clone())),
        _ => None,
    }
}

/// Product-to-sum identity rewrite for a product of two `sin`/`cos` factors.
///
/// Given `f`, `g` each a bare `Sin(A)` or `Cos(B)`, returns the equivalent
/// `½[… ± …]` sum of `sin`/`cos` of `A±B`. This is a pure trigonometric-identity
/// rewrite (no integration): **P3's Fourier/series work reuses it** to expand
/// trigonometric products, which is why it is `pub(crate)` and returns the
/// rewritten expression rather than an antiderivative.
///
/// Returns `None` when either factor is not a bare `sin`/`cos`.
pub(crate) fn product_to_sum(f: &LoweredOp, g: &LoweredOp) -> Option<LoweredOp> {
    let (f_is_sin, a_arg) = as_sin_or_cos(f)?;
    let (g_is_sin, b_arg) = as_sin_or_cos(g)?;
    let sum = add(a_arg.clone(), b_arg.clone());
    let diff = sub(a_arg, b_arg);
    let half = konst(0.5);
    let expr = match (f_is_sin, g_is_sin) {
        // sin A cos B = ½[sin(A+B) + sin(A−B)]
        (true, false) => mul(half, add(sin(sum), sin(diff))),
        // cos A sin B = ½[sin(A+B) − sin(A−B)]
        (false, true) => mul(half, sub(sin(sum), sin(diff))),
        // sin A sin B = ½[cos(A−B) − cos(A+B)]
        (true, true) => mul(half, sub(cos(diff), cos(sum))),
        // cos A cos B = ½[cos(A−B) + cos(A+B)]
        (false, false) => mul(half, add(cos(diff), cos(sum))),
    };
    Some(expr)
}

/// Integrate a linear combination (constant coefficients) of `sin`/`cos` of
/// affine arguments — exactly the shape [`product_to_sum`] produces.
fn integrate_linear_trig_sum(op: &LoweredOp, wrt: usize) -> Option<LoweredOp> {
    if !op.contains_var(wrt) {
        // Constant term (e.g. sin(A−B) with A, B the same frequency): ∫ c dx = c·x.
        return Some(mul(op.clone(), var(wrt)));
    }
    match op {
        LoweredOp::Add(a, b) => Some(add(
            integrate_linear_trig_sum(a, wrt)?,
            integrate_linear_trig_sum(b, wrt)?,
        )),
        LoweredOp::Sub(a, b) => Some(sub(
            integrate_linear_trig_sum(a, wrt)?,
            integrate_linear_trig_sum(b, wrt)?,
        )),
        LoweredOp::Neg(a) => Some(neg(integrate_linear_trig_sum(a, wrt)?)),
        LoweredOp::Mul(a, b) => {
            if !a.contains_var(wrt) {
                Some(mul((**a).clone(), integrate_linear_trig_sum(b, wrt)?))
            } else if !b.contains_var(wrt) {
                Some(mul((**b).clone(), integrate_linear_trig_sum(a, wrt)?))
            } else {
                None
            }
        }
        LoweredOp::Sin(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫ sin(u) dx = −cos(u)/a
            Some(div(neg(cos((**arg).clone())), konst(a_coef)))
        }
        LoweredOp::Cos(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫ cos(u) dx = sin(u)/a
            Some(div(sin((**arg).clone()), konst(a_coef)))
        }
        _ => None,
    }
}

// ── product dispatch ───────────────────────────────────────────────────────────

/// A parsed multiplicative trig factor: `coeff · head(arg)^power`.
struct TrigFactor {
    coeff: f64,
    kind: TrigKind,
    arg: LoweredOp,
    power: i64,
}

/// Parse `op` as a trig factor with an optional constant multiplier / negation.
fn as_trig_factor(op: &LoweredOp) -> Option<TrigFactor> {
    match op {
        LoweredOp::Sin(a) => Some(TrigFactor {
            coeff: 1.0,
            kind: TrigKind::Sin,
            arg: (**a).clone(),
            power: 1,
        }),
        LoweredOp::Cos(a) => Some(TrigFactor {
            coeff: 1.0,
            kind: TrigKind::Cos,
            arg: (**a).clone(),
            power: 1,
        }),
        LoweredOp::Tan(a) => Some(TrigFactor {
            coeff: 1.0,
            kind: TrigKind::Tan,
            arg: (**a).clone(),
            power: 1,
        }),
        LoweredOp::Pow(base, expo) => {
            let e = as_const(expo)?;
            if e.fract() != 0.0 {
                return None;
            }
            let (kind, arg) = trig_head(base)?;
            Some(TrigFactor {
                coeff: 1.0,
                kind,
                arg,
                power: e as i64,
            })
        }
        LoweredOp::Neg(a) => {
            let mut inner = as_trig_factor(a)?;
            inner.coeff = -inner.coeff;
            Some(inner)
        }
        LoweredOp::Mul(a, b) => {
            if let Some(c) = as_const(a) {
                let mut inner = as_trig_factor(b)?;
                inner.coeff *= c;
                Some(inner)
            } else if let Some(c) = as_const(b) {
                let mut inner = as_trig_factor(a)?;
                inner.coeff *= c;
                Some(inner)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Attempt `∫ (f·g) dx` when both factors are trigonometric: `sinᵐ cosⁿ` (shared
/// argument) or a `sin`/`cos`·`sin`/`cos` product-to-sum (distinct arguments).
pub(crate) fn try_trig_product(a: &LoweredOp, b: &LoweredOp, wrt: usize) -> Option<LoweredOp> {
    let fa = as_trig_factor(a)?;
    let fb = as_trig_factor(b)?;
    let orig = mul(a.clone(), b.clone());
    let coeff = fa.coeff * fb.coeff;

    // ── shared argument: sinᵐ cosⁿ reduction ────────────────────────────────
    if structurally_eq(&fa.arg, &fb.arg) && fa.power >= 1 && fb.power >= 1 {
        let (a_coef, _) = affine_arg(&fa.arg, wrt)?;
        let (m, n) = match (fa.kind, fb.kind) {
            (TrigKind::Sin, TrigKind::Sin) => (fa.power + fb.power, 0),
            (TrigKind::Cos, TrigKind::Cos) => (0, fa.power + fb.power),
            (TrigKind::Sin, TrigKind::Cos) => (fa.power, fb.power),
            (TrigKind::Cos, TrigKind::Sin) => (fb.power, fa.power),
            _ => return None, // products involving `tan` are not in this table
        };
        let antideriv_in_u = sin_cos_power_antideriv(&fa.arg, m, n)?;
        let result = mul(konst(coeff), div(antideriv_in_u, konst(a_coef))).simplify();
        return verify_antiderivative(&result, &orig, wrt).then_some(result);
    }

    // ── distinct arguments: product-to-sum, then integrate the sum ───────────
    if fa.power == 1 && fb.power == 1 && fa.kind != TrigKind::Tan && fb.kind != TrigKind::Tan {
        let f_core = if fa.kind == TrigKind::Sin {
            sin(fa.arg.clone())
        } else {
            cos(fa.arg.clone())
        };
        let g_core = if fb.kind == TrigKind::Sin {
            sin(fb.arg.clone())
        } else {
            cos(fb.arg.clone())
        };
        let sum = product_to_sum(&f_core, &g_core)?;
        let anti = integrate_linear_trig_sum(&sum, wrt)?;
        let result = mul(konst(coeff), anti).simplify();
        return verify_antiderivative(&result, &orig, wrt).then_some(result);
    }

    None
}

// ── §5  solve-for-I: ∫ eᵃˣ sin(bx) / eᵃˣ cos(bx) ──────────────────────────────

/// A parsed factor for the solve-for-I recogniser.
enum ExpTrig {
    /// `coeff · exp(arg)`.
    Exp { coeff: f64, arg: LoweredOp },
    /// `coeff · sin(arg)` (`is_sin`) or `coeff · cos(arg)`.
    Trig {
        coeff: f64,
        is_sin: bool,
        arg: LoweredOp,
    },
}

/// Parse `op` as `exp`/`sin`/`cos` of something, with an optional constant
/// multiplier / negation.
fn as_exp_trig(op: &LoweredOp) -> Option<ExpTrig> {
    match op {
        LoweredOp::Exp(a) => Some(ExpTrig::Exp {
            coeff: 1.0,
            arg: (**a).clone(),
        }),
        LoweredOp::Sin(a) => Some(ExpTrig::Trig {
            coeff: 1.0,
            is_sin: true,
            arg: (**a).clone(),
        }),
        LoweredOp::Cos(a) => Some(ExpTrig::Trig {
            coeff: 1.0,
            is_sin: false,
            arg: (**a).clone(),
        }),
        LoweredOp::Neg(a) => Some(scale_exp_trig(as_exp_trig(a)?, -1.0)),
        LoweredOp::Mul(a, b) => {
            if let Some(c) = as_const(a) {
                Some(scale_exp_trig(as_exp_trig(b)?, c))
            } else if let Some(c) = as_const(b) {
                Some(scale_exp_trig(as_exp_trig(a)?, c))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn scale_exp_trig(f: ExpTrig, s: f64) -> ExpTrig {
    match f {
        ExpTrig::Exp { coeff, arg } => ExpTrig::Exp {
            coeff: coeff * s,
            arg,
        },
        ExpTrig::Trig { coeff, is_sin, arg } => ExpTrig::Trig {
            coeff: coeff * s,
            is_sin,
            arg,
        },
    }
}

/// Attempt `∫ eᶠ · {sin,cos}(g) dx` via the solve-for-I closed forms, for affine
/// `f = a·x+c` and `g = b·x+d`.
pub(crate) fn try_solve_for_i(a: &LoweredOp, b: &LoweredOp, wrt: usize) -> Option<LoweredOp> {
    let fa = as_exp_trig(a)?;
    let fb = as_exp_trig(b)?;

    // Sort into (exp, trig); require exactly one of each.
    let (exp_coeff, exp_arg, trig_coeff, is_sin, trig_arg) = match (fa, fb) {
        (
            ExpTrig::Exp { coeff: ec, arg: ea },
            ExpTrig::Trig {
                coeff: tc,
                is_sin,
                arg: ta,
            },
        )
        | (
            ExpTrig::Trig {
                coeff: tc,
                is_sin,
                arg: ta,
            },
            ExpTrig::Exp { coeff: ec, arg: ea },
        ) => (ec, ea, tc, is_sin, ta),
        _ => return None,
    };

    let (a_coef, _) = affine_arg(&exp_arg, wrt)?;
    let (b_coef, _) = affine_arg(&trig_arg, wrt)?;
    let denom = a_coef * a_coef + b_coef * b_coef;
    if denom.abs() < 1e-300 {
        return None;
    }

    // ∫ eᶠ sin g dx = eᶠ (a·sin g − b·cos g)/(a²+b²)
    // ∫ eᶠ cos g dx = eᶠ (a·cos g + b·sin g)/(a²+b²)
    let inner = if is_sin {
        sub(
            mul(konst(a_coef), sin(trig_arg.clone())),
            mul(konst(b_coef), cos(trig_arg.clone())),
        )
    } else {
        add(
            mul(konst(a_coef), cos(trig_arg.clone())),
            mul(konst(b_coef), sin(trig_arg.clone())),
        )
    };

    let coeff = exp_coeff * trig_coeff / denom;
    let result = mul(konst(coeff), mul(exp(exp_arg), inner)).simplify();

    let orig = mul(a.clone(), b.clone());
    verify_antiderivative(&result, &orig, wrt).then_some(result)
}

// ── §4  Weierstrass t = tan(x/2) ───────────────────────────────────────────────

/// `sin u = 2t/(1+t²)` in the fresh variable `t = Var(t_var)`.
fn weier_sin(t_var: usize) -> LoweredOp {
    div(mul(konst(2.0), var(t_var)), add(konst(1.0), sq_t(t_var)))
}
/// `cos u = (1−t²)/(1+t²)`.
fn weier_cos(t_var: usize) -> LoweredOp {
    div(sub(konst(1.0), sq_t(t_var)), add(konst(1.0), sq_t(t_var)))
}
/// `tan u = 2t/(1−t²)`.
fn weier_tan(t_var: usize) -> LoweredOp {
    div(mul(konst(2.0), var(t_var)), sub(konst(1.0), sq_t(t_var)))
}
/// `t²` written as `t·t` (keeps the whole integrand `Pow`-free for `as_rational`).
fn sq_t(t_var: usize) -> LoweredOp {
    mul(var(t_var), var(t_var))
}

/// Collect the arguments of every `Sin`/`Cos`/`Tan` node in `op` that depends on
/// `wrt`.
fn collect_trig_args(op: &LoweredOp, wrt: usize, out: &mut Vec<LoweredOp>) {
    match op {
        LoweredOp::Sin(a) | LoweredOp::Cos(a) | LoweredOp::Tan(a) if a.contains_var(wrt) => {
            out.push((**a).clone());
        }
        _ => {}
    }
    match op {
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            collect_trig_args(a, wrt, out);
            collect_trig_args(b, wrt, out);
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
        | LoweredOp::Ci(a) => collect_trig_args(a, wrt, out),
        _ => {}
    }
}

/// Replace every `Sin`/`Cos`/`Tan` node whose argument equals `u` with its
/// Weierstrass rational form in `t = Var(t_var)`.
fn weier_substitute(op: &LoweredOp, u: &LoweredOp, t_var: usize) -> LoweredOp {
    match op {
        LoweredOp::Sin(a) if structurally_eq(a, u) => weier_sin(t_var),
        LoweredOp::Cos(a) if structurally_eq(a, u) => weier_cos(t_var),
        LoweredOp::Tan(a) if structurally_eq(a, u) => weier_tan(t_var),
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Add(a, b) => add(weier_substitute(a, u, t_var), weier_substitute(b, u, t_var)),
        LoweredOp::Sub(a, b) => sub(weier_substitute(a, u, t_var), weier_substitute(b, u, t_var)),
        LoweredOp::Mul(a, b) => mul(weier_substitute(a, u, t_var), weier_substitute(b, u, t_var)),
        LoweredOp::Div(a, b) => div(weier_substitute(a, u, t_var), weier_substitute(b, u, t_var)),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            arc(weier_substitute(a, u, t_var)),
            arc(weier_substitute(b, u, t_var)),
        ),
        LoweredOp::Neg(a) => neg(weier_substitute(a, u, t_var)),
        LoweredOp::Exp(a) => exp(weier_substitute(a, u, t_var)),
        LoweredOp::Ln(a) => ln(weier_substitute(a, u, t_var)),
        LoweredOp::Sin(a) => sin(weier_substitute(a, u, t_var)),
        LoweredOp::Cos(a) => cos(weier_substitute(a, u, t_var)),
        LoweredOp::Tan(a) => tan(weier_substitute(a, u, t_var)),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Erf(a) => LoweredOp::Erf(arc(weier_substitute(a, u, t_var))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Ei(a) => LoweredOp::Ei(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Si(a) => LoweredOp::Si(arc(weier_substitute(a, u, t_var))),
        LoweredOp::Ci(a) => LoweredOp::Ci(arc(weier_substitute(a, u, t_var))),
    }
}

/// Attempt `∫ R(sin u, cos u) dx` for a rational `R` via `t = tan(u/2)`, where
/// `u = a·x + b` is the single common affine argument of every trig node.
///
/// This is a **top-level fallback** (called from [`crate::integrate`] only when
/// the structural rules fail); it never fires on integrands whose trig arguments
/// are not one common affine expression, so `∫ sin(x²)` and the like stay
/// `Unsupported`.
pub(crate) fn try_weierstrass(op: &LoweredOp, wrt: usize) -> Option<LoweredOp> {
    // 1. There must be exactly one distinct trig argument, and it must be affine.
    let mut args = Vec::new();
    collect_trig_args(op, wrt, &mut args);
    if args.is_empty() {
        return None; // no trig depending on x ⇒ Weierstrass is inapplicable
    }
    let u = args[0].clone();
    if args.iter().any(|a| !structurally_eq(a, &u)) {
        return None; // mixed / nested trig arguments
    }
    let (a_coef, _) = affine_arg(&u, wrt)?; // non-affine argument ⇒ bail (guards sin(x²))

    // 2. Rationalise: replace trig nodes, attach dx = (2/a)/(1+t²) dt.
    let t_var = wrt + 1;
    let substituted = weier_substitute(op, &u, t_var);
    let dx_factor = div(konst(2.0 / a_coef), add(konst(1.0), sq_t(t_var)));
    let integrand_t = mul(substituted, dx_factor);
    if integrand_t.contains_var(wrt) {
        return None; // x survived outside a trig node ⇒ not rational in t
    }

    // 3. Extract the exact rational function and integrate it via the shared PFD.
    let (num_poly, den_poly) = crate::rewrite::apart::as_rational(&integrand_t, t_var)?;
    if den_poly.is_zero() {
        return None;
    }
    let num_op = num_poly.to_lowered(t_var);
    let den_op = den_poly.to_lowered(t_var);
    let antideriv_t =
        crate::integrate::integrate_rational::integrate_rational(&num_op, &den_op, t_var)?;

    // 4. Back-substitute t → tan(u/2).
    let tan_half = tan(div(u, konst(2.0)));
    let antideriv_x = substitute_expr(&antideriv_t, t_var, &tan_half).simplify();

    verify_antiderivative(&antideriv_x, op, wrt).then_some(antideriv_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> LoweredOp {
        var(0)
    }

    /// d/dx anti at `xv` via central difference.
    fn deriv(anti: &LoweredOp, xv: f64) -> f64 {
        let h = 1e-6;
        (eval_only_wrt(anti, 0, xv + h) - eval_only_wrt(anti, 0, xv - h)) / (2.0 * h)
    }

    fn assert_antideriv(anti: &LoweredOp, f: impl Fn(f64) -> f64, points: &[f64]) {
        for &xv in points {
            let d = deriv(anti, xv);
            let fv = f(xv);
            assert!(
                (d - fv).abs() < 1e-4,
                "d/dx antideriv = {d} but f({xv}) = {fv}"
            );
        }
    }

    #[test]
    fn sin_squared_power() {
        // ∫ sin²x dx, verified by differentiation.
        let f = powf(sin(x()), 2.0);
        let anti = try_trig_power(&f, 0).expect("sin^2 integrates");
        assert_antideriv(&anti, |v| v.sin().powi(2), &[0.3, 0.7, 1.2, -0.5]);
    }

    #[test]
    fn cos_cubed_power() {
        let f = powf(cos(x()), 3.0);
        let anti = try_trig_power(&f, 0).expect("cos^3 integrates");
        assert_antideriv(&anti, |v| v.cos().powi(3), &[0.3, 0.7, 1.2, -0.5, 2.0]);
    }

    #[test]
    fn tan_squared_power() {
        let f = powf(tan(x()), 2.0);
        let anti = try_trig_power(&f, 0).expect("tan^2 integrates");
        assert_antideriv(&anti, |v| v.tan().powi(2), &[0.3, 0.7, 1.2, -0.5]);
    }

    #[test]
    fn sec_squared_power() {
        // sec²x = Pow(cos x, −2); ∫ sec²x dx = tan x.
        let f = powf(cos(x()), -2.0);
        let anti = try_trig_power(&f, 0).expect("sec^2 integrates");
        assert_antideriv(&anti, |v| 1.0 / v.cos().powi(2), &[0.3, 0.7, 1.2, -0.5]);
    }

    #[test]
    fn sin_cos_product_shared_arg() {
        // ∫ sin x cos x dx = sin²x/2.
        let anti = try_trig_product(&sin(x()), &cos(x()), 0).expect("sin·cos integrates");
        assert_antideriv(&anti, |v| v.sin() * v.cos(), &[0.3, 0.7, 1.2, -0.5]);
    }

    #[test]
    fn sin_sq_cos_cube_product() {
        // ∫ sin²x cos³x dx.
        let a = powf(sin(x()), 2.0);
        let b = powf(cos(x()), 3.0);
        let anti = try_trig_product(&a, &b, 0).expect("sin^2 cos^3 integrates");
        assert_antideriv(
            &anti,
            |v| v.sin().powi(2) * v.cos().powi(3),
            &[0.3, 0.7, 1.2, -0.5],
        );
    }

    #[test]
    fn product_to_sum_distinct_freq() {
        // ∫ sin(2x) cos(3x) dx via product-to-sum.
        let a = sin(mul(konst(2.0), x()));
        let b = cos(mul(konst(3.0), x()));
        let anti = try_trig_product(&a, &b, 0).expect("sin2x·cos3x integrates");
        assert_antideriv(
            &anti,
            |v| (2.0 * v).sin() * (3.0 * v).cos(),
            &[0.3, 0.7, 1.2, -0.5, 2.0],
        );
    }

    #[test]
    fn solve_for_i_exp_sin() {
        // ∫ eˣ sin x dx = eˣ(sin x − cos x)/2.
        let anti = try_solve_for_i(&exp(x()), &sin(x()), 0).expect("eˣ sin x integrates");
        assert_antideriv(&anti, |v| v.exp() * v.sin(), &[0.3, 0.7, 1.2, -0.5]);
        // Structural closeness to eˣ(sin x − cos x)/2 at a point.
        let val = eval_only_wrt(&anti, 0, 0.5);
        let want = 0.5_f64.exp() * (0.5_f64.sin() - 0.5_f64.cos()) / 2.0;
        assert!((val - want).abs() < 1e-9, "got {val}, want {want}");
    }

    #[test]
    fn solve_for_i_exp_cos_affine() {
        // ∫ e^{2x} cos(3x) dx.
        let anti = try_solve_for_i(&exp(mul(konst(2.0), x())), &cos(mul(konst(3.0), x())), 0)
            .expect("e^{2x} cos 3x integrates");
        assert_antideriv(
            &anti,
            |v| (2.0 * v).exp() * (3.0 * v).cos(),
            &[0.1, 0.3, 0.5, -0.2],
        );
    }

    #[test]
    fn weierstrass_one_over_two_plus_cos() {
        // ∫ 1/(2 + cos x) dx via t = tan(x/2).
        let f = div(konst(1.0), add(konst(2.0), cos(x())));
        let anti = try_weierstrass(&f, 0).expect("1/(2+cos x) integrates");
        assert_antideriv(&anti, |v| 1.0 / (2.0 + v.cos()), &[0.3, 0.7, 1.2, -0.5]);
    }

    #[test]
    fn sin_x_squared_not_weierstrass() {
        // ∫ sin(x²): argument is not affine ⇒ Weierstrass must decline.
        let f = sin(powf(x(), 2.0));
        assert!(
            try_weierstrass(&f, 0).is_none(),
            "sin(x²) must stay Unsupported"
        );
    }

    #[test]
    fn exp_x_squared_not_weierstrass() {
        // ∫ exp(x²): no trig ⇒ Weierstrass must decline.
        let f = exp(powf(x(), 2.0));
        assert!(
            try_weierstrass(&f, 0).is_none(),
            "exp(x²) must stay Unsupported"
        );
    }

    #[test]
    fn power_cap_declines() {
        // Above the reduction cap, decline honestly rather than loop.
        let f = powf(sin(x()), 21.0);
        assert!(try_trig_power(&f, 0).is_none(), "n=21 exceeds the cap");
    }
}
