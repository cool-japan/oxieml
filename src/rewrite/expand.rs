//! `expand` — distribute products and integer powers into a sum of monomials.
//!
//! # Algorithm
//!
//! A general elementary expression is not a polynomial, but it is always a
//! polynomial in a finite set of **generators** — the maximal subexpressions
//! that are not themselves built from the polynomial operations
//! `{+, −, ·, unary −, ^n}`. For example
//!
//! ```text
//! (sin x + 1)·(x + 2)
//! ```
//!
//! is a polynomial in the two generators `g₀ = sin x` and `g₁ = x` (the plain
//! variable is already a polynomial generator). `expand` therefore proceeds by:
//!
//! 1. Replacing every generator subexpression by a fresh symbol (a `Var` index
//!    beyond the real variables), recording the mapping. The generator's own
//!    arguments are recursively expanded first, so `sin((x+1)²)` is stored as
//!    `sin(x² + 2x + 1)`.
//! 2. Converting the resulting genuinely-polynomial tree to a
//!    [`MultiPoly`](crate::poly::MultiPoly), whose arithmetic automatically
//!    performs the distribution *and* collects like terms across every
//!    generator.
//! 3. Emitting the canonical sum of monomials with
//!    [`MultiPoly::to_lowered`](crate::poly::MultiPoly::to_lowered) and mapping
//!    the generator symbols back to their subexpressions.
//!
//! Because the multiplication of two sparse polynomials with `p` and `q` terms
//! can produce up to `p·q` terms, the expansion is capped at
//! [`EXPAND_MAX_TERMS`] intermediate terms; hitting the cap makes `expand`
//! return the input unchanged rather than blow up.
//!
//! The whole pipeline is exact over ℚ, so the result equals the input as a
//! function; this is re-confirmed at the probe points before it is returned.

use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::poly::MultiPoly;

use super::verify_agrees;

/// Upper bound on the number of monomials `expand` will materialise.
///
/// Distributing a product of two sums with `p` and `q` terms yields up to `p·q`
/// terms; this cap keeps a pathological input (e.g. `(x+1)^{50}` in several
/// variables) from exhausting memory. The [`MultiPoly`] sparse representation is
/// checked against this bound after conversion.
pub const EXPAND_MAX_TERMS: usize = 4096;

impl LoweredOp {
    /// Expand products and integer powers into a canonical sum of monomials.
    ///
    /// Non-polynomial subexpressions (transcendental calls, divisions, powers
    /// with non-integer or symbolic exponents) are treated as opaque
    /// generators: the distribution happens *around* them and their own
    /// arguments are expanded recursively.
    ///
    /// Never panics: if the expression cannot be represented as a polynomial in
    /// its generators, if the term count would exceed [`EXPAND_MAX_TERMS`], or
    /// if the numeric-verification gate fails, the input is returned unchanged.
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // (x + 1)^2 → x^2 + 2x + 1
    /// let x = Arc::new(LoweredOp::Var(0));
    /// let expr = LoweredOp::Pow(
    ///     Arc::new(LoweredOp::Add(x, Arc::new(LoweredOp::Const(1.0)))),
    ///     Arc::new(LoweredOp::Const(2.0)),
    /// );
    /// let expanded = expr.expand();
    /// // Numerically identical to the original.
    /// for xv in [0.5_f64, 1.5, 2.5] {
    ///     assert!((expanded.eval(&[xv]) - (xv + 1.0).powi(2)).abs() < 1e-9);
    /// }
    /// ```
    #[must_use]
    pub fn expand(&self) -> LoweredOp {
        let real_vars = self.count_vars();
        match expand_via_generators(self, real_vars) {
            Some(result) => {
                let simplified = result.simplify();
                if verify_agrees(self, &simplified, real_vars) {
                    simplified
                } else {
                    self.clone()
                }
            }
            None => self.clone(),
        }
    }
}

/// A recorded generator: its structural hash (for de-duplication) and the
/// (already-expanded) subexpression it stands for.
struct Generator {
    hash: u64,
    expr: LoweredOp,
}

/// Perform the generator substitution, MultiPoly round-trip and back
/// substitution. Returns `None` on any failure (non-representable, over cap).
fn expand_via_generators(op: &LoweredOp, real_vars: usize) -> Option<LoweredOp> {
    let mut generators: Vec<Generator> = Vec::new();
    let replaced = replace_generators(op, real_vars, &mut generators);
    let total_vars = real_vars + generators.len();

    let mp = MultiPoly::from_lowered(&replaced, total_vars).ok()?;
    if mp.num_terms() > EXPAND_MAX_TERMS {
        return None;
    }
    let expanded = mp.to_lowered();
    Some(substitute_generators(&expanded, real_vars, &generators))
}

/// Rewrite `op` so that every non-polynomial subexpression is replaced by a
/// fresh `Var(real_vars + k)` generator symbol. Nodes that are part of the
/// polynomial skeleton (`+ − · unary‑neg  ^nonneg‑int`, constants, real vars)
/// are kept and recursed into.
fn replace_generators(
    op: &LoweredOp,
    real_vars: usize,
    generators: &mut Vec<Generator>,
) -> LoweredOp {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(replace_generators(a, real_vars, generators))),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(replace_generators(a, real_vars, generators)),
            Arc::new(replace_generators(b, real_vars, generators)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(replace_generators(a, real_vars, generators)),
            Arc::new(replace_generators(b, real_vars, generators)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(replace_generators(a, real_vars, generators)),
            Arc::new(replace_generators(b, real_vars, generators)),
        ),
        LoweredOp::Pow(base, exp) => {
            if let LoweredOp::Const(e) = exp.as_ref() {
                if is_nonneg_int_exponent(*e) {
                    return LoweredOp::Pow(
                        Arc::new(replace_generators(base, real_vars, generators)),
                        Arc::clone(exp),
                    );
                }
            }
            // Non-integer / symbolic exponent: opaque generator.
            intern_generator(op, real_vars, generators)
        }
        // Everything else (Div and all transcendental unary nodes) is opaque.
        _ => intern_generator(op, real_vars, generators),
    }
}

/// Register `op` (with its own arguments expanded) as a generator and return the
/// symbol standing in for it. De-duplicates structurally-equal generators.
fn intern_generator(
    op: &LoweredOp,
    real_vars: usize,
    generators: &mut Vec<Generator>,
) -> LoweredOp {
    // Expand the generator's arguments so nested polynomial structure is
    // canonicalised too (e.g. sin((x+1)^2) → sin(x^2 + 2x + 1)).
    let expanded_generator = expand_arguments(op);
    let hash = crate::lower_simplify::ops_struct_hash(&expanded_generator);
    for (idx, g) in generators.iter().enumerate() {
        if g.hash == hash && g.expr == expanded_generator {
            return LoweredOp::Var(real_vars + idx);
        }
    }
    let idx = generators.len();
    generators.push(Generator {
        hash,
        expr: expanded_generator,
    });
    LoweredOp::Var(real_vars + idx)
}

/// Return a copy of `op` whose immediate child arguments have each been fully
/// expanded (via [`LoweredOp::expand`]). The head node itself is preserved.
fn expand_arguments(op: &LoweredOp) -> LoweredOp {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(a.expand())),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(a.expand())),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(a.expand())),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(a.expand())),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(a.expand())),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(a.expand())),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(a.expand())),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(a.expand())),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(a.expand())),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(a.expand())),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(a.expand())),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(a.expand())),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(a.expand())),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(a.expand())),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(a.expand())),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(a.expand())),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(a.expand())),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(a.expand())),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(a.expand())),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(a.expand())),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(a.expand())),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(a.expand())),
        LoweredOp::Add(a, b) => LoweredOp::Add(Arc::new(a.expand()), Arc::new(b.expand())),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(Arc::new(a.expand()), Arc::new(b.expand())),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(Arc::new(a.expand()), Arc::new(b.expand())),
        LoweredOp::Div(a, b) => LoweredOp::Div(Arc::new(a.expand()), Arc::new(b.expand())),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(Arc::new(a.expand()), Arc::new(b.expand())),
    }
}

/// Replace every generator symbol `Var(real_vars + k)` in `op` by its recorded
/// subexpression. Real variables (`Var(i)` with `i < real_vars`) are left alone.
fn substitute_generators(op: &LoweredOp, real_vars: usize, generators: &[Generator]) -> LoweredOp {
    match op {
        LoweredOp::Var(i) if *i >= real_vars => generators
            .get(*i - real_vars)
            .map_or_else(|| op.clone(), |g| g.expr.clone()),
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Neg(a) => {
            LoweredOp::Neg(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Exp(a) => {
            LoweredOp::Exp(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Ln(a) => {
            LoweredOp::Ln(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Sin(a) => {
            LoweredOp::Sin(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Cos(a) => {
            LoweredOp::Cos(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Tan(a) => {
            LoweredOp::Tan(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Sinh(a) => {
            LoweredOp::Sinh(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Cosh(a) => {
            LoweredOp::Cosh(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Tanh(a) => {
            LoweredOp::Tanh(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Arcsin(a) => {
            LoweredOp::Arcsin(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Arccos(a) => {
            LoweredOp::Arccos(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Arctan(a) => {
            LoweredOp::Arctan(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Arcsinh(a) => {
            LoweredOp::Arcsinh(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Arccosh(a) => {
            LoweredOp::Arccosh(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Arctanh(a) => {
            LoweredOp::Arctanh(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Erf(a) => {
            LoweredOp::Erf(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::LGamma(a) => {
            LoweredOp::LGamma(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Digamma(a) => {
            LoweredOp::Digamma(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Trigamma(a) => {
            LoweredOp::Trigamma(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Ei(a) => {
            LoweredOp::Ei(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Si(a) => {
            LoweredOp::Si(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Ci(a) => {
            LoweredOp::Ci(Arc::new(substitute_generators(a, real_vars, generators)))
        }
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(substitute_generators(a, real_vars, generators)),
            Arc::new(substitute_generators(b, real_vars, generators)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(substitute_generators(a, real_vars, generators)),
            Arc::new(substitute_generators(b, real_vars, generators)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(substitute_generators(a, real_vars, generators)),
            Arc::new(substitute_generators(b, real_vars, generators)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            Arc::new(substitute_generators(a, real_vars, generators)),
            Arc::new(substitute_generators(b, real_vars, generators)),
        ),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            Arc::new(substitute_generators(a, real_vars, generators)),
            Arc::new(substitute_generators(b, real_vars, generators)),
        ),
    }
}

/// A finite `f64` exponent that is a non-negative integer within the
/// [`MultiPoly`] degree budget.
fn is_nonneg_int_exponent(e: f64) -> bool {
    e.is_finite() && e >= 0.0 && e.fract() == 0.0 && e <= 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Arc::new(a), Arc::new(b))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Arc::new(a), Arc::new(b))
    }
    fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(a), Arc::new(b))
    }

    fn as_multipoly(op: &LoweredOp, n: usize) -> MultiPoly {
        MultiPoly::from_lowered(op, n).expect("polynomial")
    }

    #[test]
    fn expand_square_of_binomial() {
        // (x + 1)^2 → x^2 + 2x + 1
        let expr = pow(add(x(), c(1.0)), c(2.0));
        let expanded = expr.expand();
        let expected =
            MultiPoly::from_lowered(&add(add(pow(x(), c(2.0)), mul(c(2.0), x())), c(1.0)), 1)
                .expect("poly");
        assert_eq!(as_multipoly(&expanded, 1), expected);
        for xv in [-2.0_f64, 0.5, 3.0] {
            assert!((expanded.eval(&[xv]) - (xv + 1.0).powi(2)).abs() < 1e-9);
        }
    }

    #[test]
    fn expand_product_of_sums() {
        // (x + 1)(x + 2) → x^2 + 3x + 2
        let expr = mul(add(x(), c(1.0)), add(x(), c(2.0)));
        let expanded = expr.expand();
        for xv in [-1.5_f64, 0.0, 2.5] {
            let want = (xv + 1.0) * (xv + 2.0);
            assert!((expanded.eval(&[xv]) - want).abs() < 1e-9);
        }
        // Must be genuinely expanded: 3 monomials.
        assert_eq!(as_multipoly(&expanded, 1).num_terms(), 3);
    }

    #[test]
    fn expand_with_transcendental_generator() {
        // (sin x + 1)^2 = sin(x)^2 + 2 sin(x) + 1, numerically identical.
        let s = LoweredOp::Sin(Arc::new(x()));
        let expr = pow(add(s, c(1.0)), c(2.0));
        let expanded = expr.expand();
        for xv in [0.3_f64, 1.1, 2.2] {
            let want = (xv.sin() + 1.0).powi(2);
            assert!((expanded.eval(&[xv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn expand_with_pi_constant_is_value_preserving() {
        // (x + π)^2 — the π constant is a finite float / exact rational.
        let pi = c(std::f64::consts::PI);
        let expr = pow(add(x(), pi), c(2.0));
        let expanded = expr.expand();
        for xv in [-1.0_f64, 0.5, 4.0] {
            let want = (xv + std::f64::consts::PI).powi(2);
            assert!((expanded.eval(&[xv]) - want).abs() < 1e-7);
        }
    }

    #[test]
    fn expand_leaves_non_polynomial_alone() {
        // 1/x is opaque; expand returns it value-preserved.
        let expr = LoweredOp::Div(Arc::new(c(1.0)), Arc::new(x()));
        let expanded = expr.expand();
        for xv in [0.5_f64, 1.0, 2.0] {
            assert!((expanded.eval(&[xv]) - 1.0 / xv).abs() < 1e-9);
        }
    }
}
