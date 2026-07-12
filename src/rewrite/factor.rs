//! `factor` — factor a univariate polynomial into irreducibles over ℚ.
//!
//! # Algorithm
//!
//! The expression is converted to an exact univariate
//! [`Poly`](crate::poly::Poly) in its single variable and handed to
//! [`Poly::factor`](crate::poly::Poly::factor), which returns
//!
//! ```text
//! f = content · Π factorᵢ^multᵢ
//! ```
//!
//! with `content ∈ ℚ` (carrying the sign of the leading coefficient) and each
//! `factorᵢ` a primitive integer polynomial, irreducible over ℚ, with positive
//! leading coefficient (Yun square-free decomposition followed by the
//! Zassenhaus pipeline — see [`crate::poly::factor`]).
//!
//! The factored tree is rebuilt as the literal product
//! `content · Π factorᵢ^multᵢ`. Crucially it is **not** run through the global
//! [`simplify`](LoweredOp::simplify), whose polynomial canonicaliser would
//! immediately re-expand the product; only the individual factors are
//! simplified in isolation (which cannot re-multiply them). The product is
//! numerically re-verified against the input before being returned.

use crate::lower::LoweredOp;
use crate::poly::{Poly, ratio_to_f64};

use super::{lconst, lmul, lpow, sole_var, verify_agrees};

impl LoweredOp {
    /// Factor a univariate polynomial into irreducible factors over ℚ.
    ///
    /// Returns the input unchanged when the expression is not a polynomial in a
    /// single variable, when factorization exhausts its internal budget, or when
    /// the numeric-verification gate fails. Never panics.
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // x^2 - 1 → (x - 1)(x + 1)
    /// let x = Arc::new(LoweredOp::Var(0));
    /// let expr = LoweredOp::Sub(
    ///     Arc::new(LoweredOp::Pow(x, Arc::new(LoweredOp::Const(2.0)))),
    ///     Arc::new(LoweredOp::Const(1.0)),
    /// );
    /// let factored = expr.factor();
    /// for xv in [0.5_f64, 1.5, 3.0] {
    ///     assert!((factored.eval(&[xv]) - (xv * xv - 1.0)).abs() < 1e-9);
    /// }
    /// ```
    #[must_use]
    pub fn factor(&self) -> LoweredOp {
        let Some(var) = sole_var(self) else {
            return self.clone();
        };
        let Ok(poly) = Poly::from_lowered(self, var) else {
            return self.clone();
        };
        let Ok(factorization) = poly.factor() else {
            return self.clone();
        };

        let content = ratio_to_f64(&factorization.content);

        // Build the product of factor^mult terms.
        let mut product: Option<LoweredOp> = None;
        for (factor, mult) in &factorization.factors {
            let factor_op = factor.to_lowered(var).simplify();
            let term = if *mult == 1 {
                factor_op
            } else {
                lpow(factor_op, *mult as f64)
            };
            product = Some(match product {
                None => term,
                Some(p) => lmul(p, term),
            });
        }

        let result = match product {
            // No non-constant factors: the whole thing is a constant.
            None => lconst(content),
            Some(p) => {
                if (content - 1.0).abs() < 1e-15 {
                    p
                } else {
                    lmul(lconst(content), p)
                }
            }
        };

        if verify_agrees(self, &result, self.count_vars()) {
            result
        } else {
            self.clone()
        }
    }
}

/// Count the number of top-level multiplicative factors in a factored tree.
///
/// A left-associated product `((a · b) · c)` reports `3`. A non-`Mul` node
/// reports `1`. Used by tests to assert that factoring actually split the input.
#[cfg(test)]
fn count_factors(op: &LoweredOp) -> usize {
    match op {
        LoweredOp::Mul(a, b) => count_factors(a) + count_factors(b),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn x() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Sub(Arc::new(a), Arc::new(b))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Arc::new(a), Arc::new(b))
    }
    fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(a), Arc::new(b))
    }

    #[test]
    fn factor_difference_of_squares() {
        // x^2 - 1 → (x - 1)(x + 1)
        let expr = sub(pow(x(), c(2.0)), c(1.0));
        let factored = expr.factor();
        assert_eq!(count_factors(&factored), 2, "should split into two factors");
        for xv in [-2.0_f64, 0.5, 1.5, 3.0] {
            assert!((factored.eval(&[xv]) - (xv * xv - 1.0)).abs() < 1e-9);
        }
        // Re-expanding recovers x^2 - 1.
        let re = factored.expand();
        let want = Poly::from_int_coeffs(&[-1, 0, 1]);
        assert_eq!(Poly::from_lowered(&re, 0).expect("poly"), want);
    }

    #[test]
    fn factor_irreducible_quadratic_is_value_preserving() {
        // x^2 + 1 is irreducible over ℚ; factor returns something equal to it.
        let expr = add(pow(x(), c(2.0)), c(1.0));
        let factored = expr.factor();
        for xv in [-1.0_f64, 0.5, 2.0] {
            assert!((factored.eval(&[xv]) - (xv * xv + 1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn factor_repeated_root() {
        // x^2 - 2x + 1 → (x - 1)^2
        let expr = add(
            sub(
                pow(x(), c(2.0)),
                LoweredOp::Mul(Arc::new(c(2.0)), Arc::new(x())),
            ),
            c(1.0),
        );
        let factored = expr.factor();
        for xv in [-1.0_f64, 0.5, 2.0, 4.0] {
            assert!((factored.eval(&[xv]) - (xv - 1.0).powi(2)).abs() < 1e-9);
        }
    }

    #[test]
    fn factor_with_integer_content() {
        // 2x^2 - 2 → 2 (x - 1)(x + 1)
        let expr = sub(
            LoweredOp::Mul(Arc::new(c(2.0)), Arc::new(pow(x(), c(2.0)))),
            c(2.0),
        );
        let factored = expr.factor();
        for xv in [-2.0_f64, 0.5, 3.0] {
            assert!((factored.eval(&[xv]) - (2.0 * xv * xv - 2.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn factor_non_polynomial_returns_input() {
        // sin(x) is not a polynomial; factor is a no-op.
        let expr = LoweredOp::Sin(Arc::new(x()));
        let factored = expr.factor();
        assert_eq!(factored, expr);
    }
}
