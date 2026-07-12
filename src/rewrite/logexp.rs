//! `powsimp` / `logcombine` — power and logarithm combination.
//!
//! # `powsimp`
//!
//! Merges factors that share a common base by adding their exponents:
//!
//! ```text
//! xᵃ · xᵇ → xᵃ⁺ᵇ ,   xᵃ / xᵇ → xᵃ⁻ᵇ ,   (xᵃ)ᵇ → xᵃᵇ .
//! ```
//!
//! A product/quotient/power tree is flattened into a constant multiplier and a
//! list of `(base, exponent)` pairs; pairs with a structurally identical base
//! are merged; the product is rebuilt in canonical `const · Π baseⁱ` form.
//!
//! # `logcombine`
//!
//! Merges a sum of logarithms into a single logarithm using
//!
//! ```text
//! c·ln(u) → ln(uᶜ) ,   ln(a) + ln(b) → ln(a·b) ,   ln(a) − ln(b) → ln(a/b) .
//! ```
//!
//! An additive expression is flattened into signed terms; each `c·ln(u)` term
//! contributes the factor `uᶜ` to a single product argument, and every
//! remaining (non-logarithmic) term is preserved. The combination
//! `Σ cᵢ ln(uᵢ) = ln(Π uᵢ^{cᵢ})` is valid on the positive branch, which is
//! exactly the domain covered by the strictly-positive probe points of the
//! verification gate.
//!
//! # Reusable surface (shared with later work)
//!
//! [`logcombine_tree`] and [`powsimp_tree`] are the raw, verification-free
//! rewrite passes; the public [`LoweredOp::logcombine`] / [`LoweredOp::powsimp`]
//! methods simply wrap them with the numeric-verification gate. A future item
//! (P3: trig/exp-log simplify identities) is intended to reuse these passes and
//! to add their inverses here — `logexpand` (`ln(a·b) → ln a + ln b`) and
//! `expexpand`/`expcombine` (`exp(a+b) ↔ exp a · exp b`) — so all four
//! log/exp combine-and-split directions live behind one module boundary.

use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::lower_simplify::ops_struct_hash;

use super::{ladd, lconst, lln, lmul, lpow, verify_agrees};

impl LoweredOp {
    /// Combine powers of a common base (`xᵃ·xᵇ → xᵃ⁺ᵇ`).
    ///
    /// Returns the input unchanged when the numeric-verification gate fails.
    /// Never panics. The merge assumes the positive branch for non-integer
    /// exponents, consistent with the strictly-positive verification points.
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // x^2 · x^3 → x^5
    /// let x2 = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)));
    /// let x3 = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(3.0)));
    /// let expr = LoweredOp::Mul(Arc::new(x2), Arc::new(x3));
    /// let simplified = expr.powsimp();
    /// for xv in [0.5_f64, 1.5, 2.0] {
    ///     assert!((simplified.eval(&[xv]) - xv.powi(5)).abs() < 1e-9);
    /// }
    /// ```
    #[must_use]
    pub fn powsimp(&self) -> LoweredOp {
        let result = powsimp_tree(self);
        if verify_agrees(self, &result, self.count_vars()) {
            result
        } else {
            self.clone()
        }
    }

    /// Combine a sum of logarithms into a single logarithm
    /// (`ln a + ln b → ln(ab)`).
    ///
    /// Returns the input unchanged when the numeric-verification gate fails.
    /// Never panics.
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // ln 2 + ln 3 → ln 6
    /// let expr = LoweredOp::Add(
    ///     Arc::new(LoweredOp::Ln(Arc::new(LoweredOp::Const(2.0)))),
    ///     Arc::new(LoweredOp::Ln(Arc::new(LoweredOp::Const(3.0)))),
    /// );
    /// let combined = expr.logcombine();
    /// assert!((combined.eval(&[]) - 6.0_f64.ln()).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn logcombine(&self) -> LoweredOp {
        let result = logcombine_tree(self);
        if verify_agrees(self, &result, self.count_vars()) {
            result
        } else {
            self.clone()
        }
    }
}

// ── powsimp core ──────────────────────────────────────────────────────────────

/// Raw power-combination pass (no verification gate). See [`LoweredOp::powsimp`].
pub(crate) fn powsimp_tree(op: &LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Mul(_, _) | LoweredOp::Div(_, _) | LoweredOp::Pow(_, _) => {
            rebuild_from_factors(op)
        }
        _ => map_children(op, &powsimp_tree),
    }
}

/// The constant multiplier and `(base, exponent)` list of a product/power tree.
fn collect_factors(op: &LoweredOp) -> (f64, Vec<(LoweredOp, f64)>) {
    match op {
        LoweredOp::Const(c) => (*c, Vec::new()),
        LoweredOp::Mul(a, b) => {
            let (ca, mut fa) = collect_factors(a);
            let (cb, fb) = collect_factors(b);
            fa.extend(fb);
            (ca * cb, fa)
        }
        LoweredOp::Div(a, b) => {
            let (ca, mut fa) = collect_factors(a);
            let (cb, fb) = collect_factors(b);
            if cb.abs() < 1e-300 {
                // Division by a zero constant: keep as an opaque atom.
                return (1.0, vec![(map_children(op, &powsimp_tree), 1.0)]);
            }
            for (base, exp) in fb {
                fa.push((base, -exp));
            }
            (ca / cb, fa)
        }
        LoweredOp::Pow(base, exp) => {
            if let LoweredOp::Const(e) = exp.as_ref() {
                if e.is_finite() {
                    let (cb, fb) = collect_factors(base);
                    if (cb - 1.0).abs() < 1e-15 {
                        let scaled = fb.into_iter().map(|(b, x)| (b, x * e)).collect();
                        return (1.0, scaled);
                    }
                }
            }
            // Symbolic or constant-carrying base: opaque atom.
            (1.0, vec![(map_children(op, &powsimp_tree), 1.0)])
        }
        _ => (1.0, vec![(map_children(op, &powsimp_tree), 1.0)]),
    }
}

/// Rebuild a canonical `const · Π baseⁱ` product from collected factors, merging
/// structurally-identical bases.
fn rebuild_from_factors(op: &LoweredOp) -> LoweredOp {
    let (const_mult, factors) = collect_factors(op);

    // Merge exponents of structurally-equal bases, preserving first-seen order.
    let mut merged: Vec<(u64, LoweredOp, f64)> = Vec::new();
    for (base, exp) in factors {
        let hash = ops_struct_hash(&base);
        if let Some(entry) = merged.iter_mut().find(|(h, b, _)| *h == hash && *b == base) {
            entry.2 += exp;
        } else {
            merged.push((hash, base, exp));
        }
    }

    let mut result: Option<LoweredOp> = None;
    if (const_mult - 1.0).abs() > 1e-15 {
        result = Some(lconst(const_mult));
    }
    for (_, base, exp) in merged {
        if exp.abs() < 1e-15 {
            continue; // base^0 = 1
        }
        let factor = if (exp - 1.0).abs() < 1e-15 {
            base
        } else {
            lpow(base, exp)
        };
        result = Some(match result {
            None => factor,
            Some(acc) => lmul(acc, factor),
        });
    }

    result.unwrap_or_else(|| lconst(const_mult))
}

// ── logcombine core ───────────────────────────────────────────────────────────

/// Raw logarithm-combination pass (no verification gate). See
/// [`LoweredOp::logcombine`].
pub(crate) fn logcombine_tree(op: &LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Add(_, _) | LoweredOp::Sub(_, _) | LoweredOp::Neg(_) => combine_log_sum(op),
        _ => map_children(op, &logcombine_tree),
    }
}

/// Combine the logarithmic terms of an additive expression into a single `ln`.
fn combine_log_sum(op: &LoweredOp) -> LoweredOp {
    let mut logs: Vec<(f64, LoweredOp)> = Vec::new();
    let mut others: Vec<LoweredOp> = Vec::new();
    flatten_sum(op, 1.0, &mut logs, &mut others);

    let mut terms: Vec<LoweredOp> = Vec::new();
    if !logs.is_empty() {
        let mut product: Option<LoweredOp> = None;
        for (coeff, arg) in &logs {
            let factor = if (*coeff - 1.0).abs() < 1e-12 {
                arg.clone()
            } else {
                lpow(arg.clone(), *coeff)
            };
            product = Some(match product {
                None => factor,
                Some(acc) => lmul(acc, factor),
            });
        }
        let argument = product.unwrap_or_else(|| lconst(1.0)).simplify();
        terms.push(lln(argument));
    }
    terms.extend(others);
    terms
        .into_iter()
        .reduce(ladd)
        .unwrap_or_else(|| lconst(0.0))
}

/// Flatten a `+`/`−`/unary-`−` tree into signed logarithmic and non-logarithmic
/// terms.
fn flatten_sum(
    op: &LoweredOp,
    sign: f64,
    logs: &mut Vec<(f64, LoweredOp)>,
    others: &mut Vec<LoweredOp>,
) {
    match op {
        LoweredOp::Add(a, b) => {
            flatten_sum(a, sign, logs, others);
            flatten_sum(b, sign, logs, others);
        }
        LoweredOp::Sub(a, b) => {
            flatten_sum(a, sign, logs, others);
            flatten_sum(b, -sign, logs, others);
        }
        LoweredOp::Neg(a) => flatten_sum(a, -sign, logs, others),
        _ => {
            let term = logcombine_tree(op);
            if let Some((coeff, arg)) = as_log_term(&term) {
                logs.push((sign * coeff, arg));
            } else if sign < 0.0 {
                others.push(LoweredOp::Neg(Arc::new(term)));
            } else {
                others.push(term);
            }
        }
    }
}

/// Recognise a term of the form `c·ln(u)` (or `ln(u)`, or `−ln(u)`), returning
/// `(c, u)`.
fn as_log_term(op: &LoweredOp) -> Option<(f64, LoweredOp)> {
    match op {
        LoweredOp::Ln(u) => Some((1.0, (**u).clone())),
        LoweredOp::Neg(inner) => as_log_term(inner).map(|(c, u)| (-c, u)),
        LoweredOp::Mul(a, b) => {
            if let (LoweredOp::Const(k), LoweredOp::Ln(u)) = (a.as_ref(), b.as_ref()) {
                return Some((*k, (**u).clone()));
            }
            if let (LoweredOp::Ln(u), LoweredOp::Const(k)) = (a.as_ref(), b.as_ref()) {
                return Some((*k, (**u).clone()));
            }
            None
        }
        _ => None,
    }
}

// ── Shared structural recursion ────────────────────────────────────────────────

/// Rebuild `op` with `f` applied to every direct child (leaves are cloned).
///
/// Shared by [`powsimp_tree`] and [`logcombine_tree`] for the "recurse into
/// children without special handling" case.
fn map_children(op: &LoweredOp, f: &dyn Fn(&LoweredOp) -> LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(f(a))),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(f(a))),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(f(a))),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(f(a))),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(f(a))),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(f(a))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(f(a))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(f(a))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(f(a))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(f(a))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(f(a))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(f(a))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(f(a))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(f(a))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(f(a))),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(f(a))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(f(a))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(f(a))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(f(a))),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(f(a))),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(f(a))),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(f(a))),
        LoweredOp::Add(a, b) => LoweredOp::Add(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Div(a, b) => LoweredOp::Div(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(Arc::new(f(a)), Arc::new(f(b))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn y() -> LoweredOp {
        LoweredOp::Var(1)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Arc::new(a), Arc::new(b))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Arc::new(a), Arc::new(b))
    }
    fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Sub(Arc::new(a), Arc::new(b))
    }
    fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(a), Arc::new(b))
    }
    fn ln(a: LoweredOp) -> LoweredOp {
        LoweredOp::Ln(Arc::new(a))
    }

    #[test]
    fn powsimp_same_base_product() {
        // x^2 · x^3 → x^5
        let expr = mul(pow(x(), c(2.0)), pow(x(), c(3.0)));
        let result = expr.powsimp();
        assert!(
            matches!(&result, LoweredOp::Pow(b, e)
                if **b == x() && matches!(e.as_ref(), LoweredOp::Const(v) if (*v - 5.0).abs() < 1e-12)),
            "expected x^5, got {result:?}"
        );
        for xv in [0.5_f64, 1.5, 2.0] {
            assert!((result.eval(&[xv]) - xv.powi(5)).abs() < 1e-9);
        }
    }

    #[test]
    fn powsimp_quotient_same_base() {
        // x^5 / x^2 → x^3
        let expr = LoweredOp::Div(Arc::new(pow(x(), c(5.0))), Arc::new(pow(x(), c(2.0))));
        let result = expr.powsimp();
        for xv in [0.5_f64, 1.5, 2.0] {
            assert!((result.eval(&[xv]) - xv.powi(3)).abs() < 1e-9);
        }
    }

    #[test]
    fn powsimp_nested_power() {
        // (x^2)^3 → x^6
        let expr = pow(pow(x(), c(2.0)), c(3.0));
        let result = expr.powsimp();
        for xv in [0.5_f64, 1.5, 2.0] {
            assert!((result.eval(&[xv]) - xv.powi(6)).abs() < 1e-9);
        }
    }

    #[test]
    fn powsimp_distinct_bases_unchanged_value() {
        // x^2 · y^3 stays value-preserving.
        let expr = mul(pow(x(), c(2.0)), pow(y(), c(3.0)));
        let result = expr.powsimp();
        for (xv, yv) in [(1.5_f64, 2.0_f64), (0.5, 1.25)] {
            let want = xv.powi(2) * yv.powi(3);
            assert!((result.eval(&[xv, yv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn logcombine_ln2_plus_ln3() {
        // ln 2 + ln 3 → ln 6
        let expr = add(ln(c(2.0)), ln(c(3.0)));
        let result = expr.logcombine();
        assert!(
            matches!(&result, LoweredOp::Ln(a)
                if (a.eval(&[]) - 6.0).abs() < 1e-9),
            "expected ln(6), got {result:?}"
        );
        assert!((result.eval(&[]) - 6.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn logcombine_difference_is_quotient() {
        // ln(x) - ln(y) → ln(x/y)
        let expr = sub(ln(x()), ln(y()));
        let result = expr.logcombine();
        for (xv, yv) in [(2.0_f64, 3.0_f64), (1.5, 0.5)] {
            let want = xv.ln() - yv.ln();
            assert!((result.eval(&[xv, yv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn logcombine_coefficient_becomes_power() {
        // 2·ln(x) + ln(y) → ln(x^2 · y)
        let expr = add(mul(c(2.0), ln(x())), ln(y()));
        let result = expr.logcombine();
        for (xv, yv) in [(2.0_f64, 3.0_f64), (1.25, 0.75)] {
            let want = 2.0 * xv.ln() + yv.ln();
            assert!((result.eval(&[xv, yv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn logcombine_keeps_non_log_terms() {
        // ln(x) + ln(y) + x  → ln(x·y) + x
        let expr = add(add(ln(x()), ln(y())), x());
        let result = expr.logcombine();
        for (xv, yv) in [(2.0_f64, 3.0_f64), (1.1, 0.9)] {
            let want = xv.ln() + yv.ln() + xv;
            assert!((result.eval(&[xv, yv]) - want).abs() < 1e-9);
        }
    }
}
