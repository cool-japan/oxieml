//! Symbolic summation: Gosper's algorithm, telescoping, and Faulhaber's formula.
//!
//! This module computes closed forms for sums `Σ_k t(k)` of [`LoweredOp`] terms.
//! It provides indefinite summation ([`LoweredOp::sum_indefinite`], the discrete
//! analogue of an antiderivative) and definite summation
//! ([`LoweredOp::sum_definite`]), returning a [`SumResult`] that is honest about
//! partiality: a term that is not hypergeometric, or is hypergeometric but has
//! no hypergeometric closed form, is reported as such rather than approximated.
//!
//! # The three engines
//!
//! Dispatch (in [`LoweredOp::sum_indefinite`]) is driven by the shape of the
//! term ratio `r(k) = t(k+1)/t(k)`:
//!
//! 1. **Polynomial** terms are summed by **Faulhaber's formula** via Bernoulli
//!    numbers (module [`faulhaber`]). Exact.
//! 2. **Constant ratio** `r(k) = B` (geometric terms `c·B^k`) telescope to
//!    `S(n) = B/(B-1) · t(n)`. Exact for any `B ≠ 1`.
//! 3. **Rational ratio** `r(k) = num(k)/den(k)` is handed to **Gosper's
//!    algorithm** (module [`gosper`]), which either produces the hypergeometric
//!    antidifference or proves none exists.
//!
//! Anything whose ratio is not (ℚ-)rational is [`SumResult::NotHypergeometric`].
//! A rational-ratio term that Gosper proves non-summable — e.g. `Σ 1/k`, the
//! harmonic number — is [`SumResult::NotClosedForm`].
//!
//! # Antidifference convention
//!
//! An indefinite sum `S(n)` returned here satisfies
//!
//! ```text
//! S(n) - S(n-1) = t(n),
//! ```
//! i.e. `S(n) = Σ_{k ≤ n} t(k)` up to an additive constant. A definite sum is
//! then `Σ_{k=lo}^{hi} t(k) = S(hi) - S(lo-1)`.
//!
//! # Certificate verification
//!
//! Every returned closed form is **verified** at a fixed set of integer probe
//! points (no randomness, per crate policy): the identity `S(n) - S(n-1) = t(n)`
//! is checked at `n = 2, …, 8`. If the identity fails to hold at these probes the
//! candidate is rejected and the honest partial variant is returned instead. This
//! is what guarantees a wrong closed form is never fabricated for a Gosper
//! certificate.

mod faulhaber;
mod gosper;
mod ratio;

use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::poly::Poly;

use gosper::Gosper;
use ratio::TermRatio;

/// The result of a symbolic summation.
///
/// Mirrors [`crate::IntegrateResult`] and [`crate::SolveResult`]: partiality is
/// modelled as data, never a panic.
#[derive(Clone, Debug)]
pub enum SumResult {
    /// A closed form `S` in the summation variable. For an indefinite sum, `S`
    /// satisfies `S(n) - S(n-1) = t(n)`. For a definite sum, `S` is the value of
    /// `Σ_{k=lo}^{hi} t(k)`.
    Closed(LoweredOp),
    /// The term is not hypergeometric: its ratio `t(k+1)/t(k)` is not a rational
    /// function of `k`.
    NotHypergeometric,
    /// The term is hypergeometric but has no hypergeometric closed form (Gosper's
    /// algorithm proved the key equation unsolvable). `Σ 1/k` is the canonical
    /// example — it is the harmonic number, not a hypergeometric term.
    NotClosedForm,
}

impl SumResult {
    /// Return the closed form, if the summation succeeded.
    #[must_use]
    pub fn closed_form(&self) -> Option<&LoweredOp> {
        match self {
            Self::Closed(op) => Some(op),
            _ => None,
        }
    }

    /// Return `true` if a closed form was found.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        matches!(self, Self::Closed(_))
    }
}

impl LoweredOp {
    /// Compute the indefinite sum `Σ t(k)` of `self` over variable `k`.
    ///
    /// Returns a [`SumResult::Closed`] `S` satisfying `S(n) - S(n-1) = self(n)`
    /// when a closed form exists, or an honest partial variant otherwise. See the
    /// [module documentation](crate::summation) for the algorithm.
    #[must_use]
    pub fn sum_indefinite(&self, k: usize) -> SumResult {
        indefinite_impl(self, k)
    }

    /// Compute the definite sum `Σ_{k=lo}^{hi} self` over variable `k`.
    ///
    /// Evaluates `S(hi) - S(lo-1)` from the indefinite antidifference `S`. Returns
    /// the same partial variants as [`LoweredOp::sum_indefinite`] when no closed
    /// form exists.
    #[must_use]
    pub fn sum_definite(&self, k: usize, lo: &LoweredOp, hi: &LoweredOp) -> SumResult {
        definite_impl(self, k, lo, hi)
    }
}

/// Free-function form of [`LoweredOp::sum_indefinite`].
#[must_use]
pub fn sum_indefinite(term: &LoweredOp, k: usize) -> SumResult {
    indefinite_impl(term, k)
}

/// Free-function form of [`LoweredOp::sum_definite`].
#[must_use]
pub fn sum_definite(term: &LoweredOp, k: usize, lo: &LoweredOp, hi: &LoweredOp) -> SumResult {
    definite_impl(term, k, lo, hi)
}

// ── LoweredOp builder helpers ──────────────────────────────────────────────────

fn arc(op: LoweredOp) -> Arc<LoweredOp> {
    Arc::new(op)
}

fn konst(c: f64) -> LoweredOp {
    LoweredOp::Const(c)
}

fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(arc(a), arc(b))
}

fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(arc(a), arc(b))
}

fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(arc(a), arc(b))
}

fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(arc(a), arc(b))
}

// ── Dispatch ───────────────────────────────────────────────────────────────────

fn indefinite_impl(term: &LoweredOp, var: usize) -> SumResult {
    // 1. Polynomial term -> Faulhaber.
    if let Ok(poly) = Poly::from_lowered(term, var) {
        let sum_poly = faulhaber::sum_polynomial(&poly);
        let closed = sum_poly.to_lowered(var);
        return finalize(term, var, closed, SumResult::NotClosedForm);
    }

    // 2. Classify the term ratio.
    match ratio::term_ratio(term, var) {
        TermRatio::NotHyper => SumResult::NotHypergeometric,
        TermRatio::IrrationalConstant(base) => geometric_or_constant(term, var, base),
        TermRatio::Rational { num, den } => {
            if let Some(base) = ratio::constant_ratio(&num, &den) {
                geometric_or_constant(term, var, base)
            } else {
                gosper_closed(term, var, &num, &den)
            }
        }
    }
}

fn definite_impl(term: &LoweredOp, var: usize, lo: &LoweredOp, hi: &LoweredOp) -> SumResult {
    match indefinite_impl(term, var) {
        SumResult::Closed(s) => {
            let upper = subst(&s, var, hi);
            let lo_minus_one = sub(lo.clone(), konst(1.0));
            let lower = subst(&s, var, &lo_minus_one);
            let result = sub(upper, lower).simplify();
            if definite_numeric_ok(term, var, lo, hi, &result) {
                SumResult::Closed(result)
            } else {
                SumResult::NotClosedForm
            }
        }
        other => other,
    }
}

/// Handle a constant term ratio `base`: geometric when `base ≠ 1`, otherwise a
/// term that is constant in `k`.
fn geometric_or_constant(term: &LoweredOp, var: usize, base: f64) -> SumResult {
    if (base - 1.0).abs() < 1e-12 {
        // t(k) is constant in k: Σ_{k=1}^{n} c = c·n.
        let closed = mul(term.clone(), LoweredOp::Var(var));
        finalize(term, var, closed, SumResult::NotClosedForm)
    } else {
        // Geometric: S(n) = base/(base-1) · t(n).
        let coeff = base / (base - 1.0);
        let closed = mul(konst(coeff), term.clone());
        finalize(term, var, closed, SumResult::NotClosedForm)
    }
}

/// Assemble and verify the Gosper closed form from a rational term ratio.
fn gosper_closed(term: &LoweredOp, var: usize, num: &Poly, den: &Poly) -> SumResult {
    match gosper::gosper_certificate(num, den) {
        Gosper::NoClosedForm => SumResult::NotClosedForm,
        Gosper::Certificate {
            num: cert_num,
            den: cert_den,
        } => {
            // Antidifference T(k) = R(k)·t(k); return S(n) = T(n+1) so that
            // S(n) - S(n-1) = t(n).
            let cert_num_shift = gosper::shift_poly(&cert_num, 1);
            let cert_den_shift = gosper::shift_poly(&cert_den, 1);
            let certificate = div(
                cert_num_shift.to_lowered(var),
                cert_den_shift.to_lowered(var),
            );
            let shifted_term = subst(term, var, &add(LoweredOp::Var(var), konst(1.0)));
            let closed = mul(certificate, shifted_term);
            finalize(term, var, closed, SumResult::NotClosedForm)
        }
    }
}

/// Verify a candidate antidifference at the mandatory integer probe points and
/// return either the closed form or the supplied fallback.
fn finalize(term: &LoweredOp, var: usize, candidate: LoweredOp, on_fail: SumResult) -> SumResult {
    if verify_antidiff(term, &candidate, var) {
        SumResult::Closed(candidate.simplify())
    } else {
        on_fail
    }
}

// ── Substitution ────────────────────────────────────────────────────────────────

/// Substitute every `Var(var)` in `op` with `replacement`.
fn subst(op: &LoweredOp, var: usize, replacement: &LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Var(i) => {
            if *i == var {
                replacement.clone()
            } else {
                LoweredOp::Var(*i)
            }
        }
        LoweredOp::Const(c) => LoweredOp::Const(*c),
        LoweredOp::NamedConst(nc) => LoweredOp::NamedConst(nc.clone()),
        LoweredOp::Add(a, b) => add(subst(a, var, replacement), subst(b, var, replacement)),
        LoweredOp::Sub(a, b) => sub(subst(a, var, replacement), subst(b, var, replacement)),
        LoweredOp::Mul(a, b) => mul(subst(a, var, replacement), subst(b, var, replacement)),
        LoweredOp::Div(a, b) => div(subst(a, var, replacement), subst(b, var, replacement)),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            arc(subst(a, var, replacement)),
            arc(subst(b, var, replacement)),
        ),
        LoweredOp::Neg(a) => LoweredOp::Neg(arc(subst(a, var, replacement))),
        LoweredOp::Exp(a) => LoweredOp::Exp(arc(subst(a, var, replacement))),
        LoweredOp::Ln(a) => LoweredOp::Ln(arc(subst(a, var, replacement))),
        LoweredOp::Sin(a) => LoweredOp::Sin(arc(subst(a, var, replacement))),
        LoweredOp::Cos(a) => LoweredOp::Cos(arc(subst(a, var, replacement))),
        LoweredOp::Tan(a) => LoweredOp::Tan(arc(subst(a, var, replacement))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(arc(subst(a, var, replacement))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(arc(subst(a, var, replacement))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(arc(subst(a, var, replacement))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(arc(subst(a, var, replacement))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(arc(subst(a, var, replacement))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(arc(subst(a, var, replacement))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(arc(subst(a, var, replacement))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(arc(subst(a, var, replacement))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(arc(subst(a, var, replacement))),
        LoweredOp::Erf(a) => LoweredOp::Erf(arc(subst(a, var, replacement))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(arc(subst(a, var, replacement))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(arc(subst(a, var, replacement))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(arc(subst(a, var, replacement))),
        LoweredOp::Ei(a) => LoweredOp::Ei(arc(subst(a, var, replacement))),
        LoweredOp::Si(a) => LoweredOp::Si(arc(subst(a, var, replacement))),
        LoweredOp::Ci(a) => LoweredOp::Ci(arc(subst(a, var, replacement))),
    }
}

// ── Verification ─────────────────────────────────────────────────────────────────

/// Largest variable index used in `op`, if any.
fn max_var(op: &LoweredOp) -> Option<usize> {
    match op {
        LoweredOp::Var(i) => Some(*i),
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => None,
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
        | LoweredOp::Ci(a) => max_var(a),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => match (max_var(a), max_var(b)) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        },
    }
}

/// Slot value used for variables other than the summation variable during
/// verification. Any fixed strictly-positive value works; it must be consistent
/// across the comparisons.
const OTHER_VAR_VALUE: f64 = 1.5;

/// Evaluate `op` with the summation variable set to `value` and every other
/// variable set to [`OTHER_VAR_VALUE`].
fn eval_at(op: &LoweredOp, var: usize, value: f64, slots: usize) -> f64 {
    let mut vars = vec![OTHER_VAR_VALUE; slots];
    if var < vars.len() {
        vars[var] = value;
    }
    op.eval(&vars)
}

/// Number of variable slots needed to evaluate `term` and `candidate`.
fn slot_count(term: &LoweredOp, candidate: &LoweredOp, var: usize) -> usize {
    let m = max_var(term)
        .into_iter()
        .chain(max_var(candidate))
        .chain(std::iter::once(var))
        .max()
        .unwrap_or(var);
    m + 1
}

/// Verify the discrete antidifference identity `S(n) - S(n-1) = t(n)` at the
/// mandatory integer probe points `n = 2, …, 8`.
///
/// Returns `true` only when at least six probes agree to a relative tolerance of
/// `1e-6`. Probes where any evaluation is non-finite (a pole) are skipped. A
/// disagreement immediately returns `false`.
fn verify_antidiff(term: &LoweredOp, candidate: &LoweredOp, var: usize) -> bool {
    let slots = slot_count(term, candidate, var);
    let mut agreed = 0usize;
    for n in 2..=8i64 {
        let n_val = n as f64;
        let s_n = eval_at(candidate, var, n_val, slots);
        let s_prev = eval_at(candidate, var, n_val - 1.0, slots);
        let t_n = eval_at(term, var, n_val, slots);
        if !s_n.is_finite() || !s_prev.is_finite() || !t_n.is_finite() {
            continue;
        }
        let difference = s_n - s_prev;
        let scale = t_n.abs().max(difference.abs()).max(1.0);
        if (difference - t_n).abs() > 1e-6 * scale {
            return false;
        }
        agreed += 1;
    }
    agreed >= 6
}

/// Cross-check a definite closed form by brute force when the bounds are concrete
/// integers. Returns `true` when the check passes or is not applicable, `false`
/// only on a genuine numeric disagreement.
fn definite_numeric_ok(
    term: &LoweredOp,
    var: usize,
    lo: &LoweredOp,
    hi: &LoweredOp,
    result: &LoweredOp,
) -> bool {
    let (Some(lo_v), Some(hi_v)) = (as_constant(lo), as_constant(hi)) else {
        return true; // symbolic bounds: rely on the indefinite verification
    };
    if lo_v.fract() != 0.0 || hi_v.fract() != 0.0 {
        return true;
    }
    let lo_i = lo_v as i64;
    let hi_i = hi_v as i64;
    if hi_i < lo_i - 1 || hi_i - lo_i > 500 {
        return true;
    }

    let slots = slot_count(term, result, var);
    let mut brute = 0.0f64;
    for k in lo_i..=hi_i {
        let value = eval_at(term, var, k as f64, slots);
        if !value.is_finite() {
            return true; // pole inside the range: cannot brute-force
        }
        brute += value;
    }

    let closed = eval_at(result, var, 0.0, slots);
    if !closed.is_finite() {
        return true;
    }
    let scale = brute.abs().max(closed.abs()).max(1.0);
    (brute - closed).abs() <= 1e-6 * scale
}

/// Interpret `op` as a constant value if it is a literal constant.
fn as_constant(op: &LoweredOp) -> Option<f64> {
    match op {
        LoweredOp::Const(c) => Some(*c),
        LoweredOp::NamedConst(nc) => Some(nc.value()),
        LoweredOp::Neg(a) => as_constant(a).map(|v| -v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    #[test]
    fn sum_k_is_triangular() {
        // Σ k = n(n+1)/2.
        let result = var(0).sum_indefinite(0);
        let closed = result
            .closed_form()
            .expect("polynomial sum has a closed form");
        for n in 0..=20i64 {
            let expected = (n * (n + 1) / 2) as f64;
            let got = closed.eval(&[n as f64]);
            assert!((got - expected).abs() < 1e-9, "Σk at n={n}: got {got}");
        }
    }

    #[test]
    fn subst_replaces_variable() {
        let expr = add(var(0), konst(2.0));
        let replaced = subst(&expr, 0, &konst(5.0));
        assert!((replaced.eval(&[0.0]) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn max_var_finds_largest() {
        let expr = add(var(0), mul(var(3), var(1)));
        assert_eq!(max_var(&expr), Some(3));
        assert_eq!(max_var(&konst(1.0)), None);
    }
}
