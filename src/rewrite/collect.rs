//! `collect` — group a polynomial by ascending powers of one variable.
//!
//! # Algorithm
//!
//! Given a polynomial `f` that is polynomial in every variable, and a target
//! variable `x = Var(var)`, `collect` rewrites `f` as
//!
//! ```text
//! f = Σ_d  c_d(other vars) · x^d
//! ```
//!
//! where each coefficient `c_d` is the sub-polynomial in the *remaining*
//! variables obtained by fixing the `x`-exponent at `d`. Concretely the
//! expression is converted to a [`MultiPoly`](crate::poly::MultiPoly) and, for
//! each degree `d` from `0` to `deg_x(f)`,
//! [`MultiPoly::coeff_in_var`](crate::poly::MultiPoly::coeff_in_var) extracts
//! `c_d` (a polynomial with the `x` exponent zeroed). The result is assembled as
//! `Σ_d c_d · x^d` and, like [`factor`](LoweredOp::factor), is deliberately
//! **not** run through the global simplifier, whose polynomial canonicaliser
//! would redistribute the grouping. Each coefficient sub-polynomial is
//! simplified on its own (which preserves the grouping), and the whole is
//! numeric-verified against the input.

use crate::lower::LoweredOp;
use crate::poly::MultiPoly;

use super::{ladd, lconst, lmul, lpow, lvar, verify_agrees};

impl LoweredOp {
    /// Collect the expression into a sum of coefficient · `Var(var)`^d terms.
    ///
    /// Returns the input unchanged when the expression is not polynomial (so a
    /// [`MultiPoly`] cannot be formed) or when the numeric-verification gate
    /// fails. Never panics; `var` may be any index (an absent variable simply
    /// yields the whole expression as the degree-0 coefficient).
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // x*y + x + 3  collected in x  →  (y + 1)·x + 3
    /// let x = || Arc::new(LoweredOp::Var(0));
    /// let y = || Arc::new(LoweredOp::Var(1));
    /// let expr = LoweredOp::Add(
    ///     Arc::new(LoweredOp::Add(
    ///         Arc::new(LoweredOp::Mul(x(), y())),
    ///         Arc::new(LoweredOp::Var(0)),
    ///     )),
    ///     Arc::new(LoweredOp::Const(3.0)),
    /// );
    /// let collected = expr.collect(0);
    /// for (xv, yv) in [(1.0_f64, 2.0_f64), (0.5, 3.0)] {
    ///     let want = xv * yv + xv + 3.0;
    ///     assert!((collected.eval(&[xv, yv]) - want).abs() < 1e-9);
    /// }
    /// ```
    #[must_use]
    pub fn collect(&self, var: usize) -> LoweredOp {
        let n_vars = self.count_vars().max(var + 1);
        let Ok(mp) = MultiPoly::from_lowered(self, n_vars) else {
            return self.clone();
        };

        let max_deg = mp.degree_in(var);
        let mut terms: Vec<LoweredOp> = Vec::new();

        for d in 0..=max_deg {
            let coeff_poly = mp.coeff_in_var(var, d as u32);
            if coeff_poly.is_zero() {
                continue;
            }
            let coeff_op = coeff_poly.to_lowered().simplify();
            let term = if d == 0 {
                coeff_op
            } else {
                let var_power = if d == 1 {
                    lvar(var)
                } else {
                    lpow(lvar(var), d as f64)
                };
                if is_const_one(&coeff_op) {
                    var_power
                } else {
                    lmul(coeff_op, var_power)
                }
            };
            terms.push(term);
        }

        let result = terms
            .into_iter()
            .reduce(ladd)
            .unwrap_or_else(|| lconst(0.0));

        if verify_agrees(self, &result, n_vars) {
            result
        } else {
            self.clone()
        }
    }
}

/// Whether `op` is the constant `1`.
fn is_const_one(op: &LoweredOp) -> bool {
    matches!(op, LoweredOp::Const(c) if (*c - 1.0).abs() < 1e-15)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn x() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn y() -> LoweredOp {
        LoweredOp::Var(1)
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

    #[test]
    fn collect_groups_by_x() {
        // x*y + x + 3  →  value preserved and collected in x.
        let expr = add(add(mul(x(), y()), x()), c(3.0));
        let collected = expr.collect(0);
        for (xv, yv) in [(1.0_f64, 2.0_f64), (0.5, 3.0), (2.0, -1.0)] {
            let want = xv * yv + xv + 3.0;
            assert!((collected.eval(&[xv, yv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn collect_quadratic_in_x() {
        // x^2 + x*y + y^2 collected in x is value-preserving.
        let expr = add(add(pow(x(), c(2.0)), mul(x(), y())), pow(y(), c(2.0)));
        let collected = expr.collect(0);
        for (xv, yv) in [(1.0_f64, 2.0_f64), (0.5, 1.5), (3.0, -2.0)] {
            let want = xv * xv + xv * yv + yv * yv;
            assert!((collected.eval(&[xv, yv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn collect_non_polynomial_returns_input() {
        let expr = LoweredOp::Exp(Arc::new(x()));
        assert_eq!(expr.collect(0), expr);
    }
}
