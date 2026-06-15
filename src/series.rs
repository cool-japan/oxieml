//! Taylor and Maclaurin series expansion for `LoweredOp` expressions.
//!
//! Provides order-n Taylor and Maclaurin polynomial approximations for any
//! `LoweredOp` expression tree via iterated symbolic differentiation.

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::lower::LoweredOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// f64 factorial.  Exact (integer-valued) up to 20!; accumulated f64 product
/// for 21 ≤ n ≤ 170.  The result is guaranteed non-zero for n ≤ 170.
fn factorial_f64(n: usize) -> f64 {
    const TABLE: [f64; 21] = [
        1.0,                         // 0!
        1.0,                         // 1!
        2.0,                         // 2!
        6.0,                         // 3!
        24.0,                        // 4!
        120.0,                       // 5!
        720.0,                       // 6!
        5040.0,                      // 7!
        40320.0,                     // 8!
        362_880.0,                   // 9!
        3_628_800.0,                 // 10!
        39_916_800.0,                // 11!
        479_001_600.0,               // 12!
        6_227_020_800.0,             // 13!
        87_178_291_200.0,            // 14!
        1_307_674_368_000.0,         // 15!
        20_922_789_888_000.0,        // 16!
        355_687_428_096_000.0,       // 17!
        6_402_373_705_728_000.0,     // 18!
        121_645_100_408_832_000.0,   // 19!
        2_432_902_008_176_640_000.0, // 20!
    ];
    if n <= 20 {
        return TABLE[n];
    }
    let mut f = TABLE[20];
    for k in 21..=n {
        f *= k as f64;
    }
    f
}

/// Build the k-th Taylor term: `coef * (Var(wrt) − center)^k`.
fn build_taylor_term(wrt: usize, center: f64, k: usize, coef: f64) -> LoweredOp {
    let x = LoweredOp::Var(wrt);
    let coef_op = LoweredOp::Const(coef);

    if k == 0 {
        return coef_op;
    }

    // (x − center), simplified to just x when center is negligibly small
    let x_minus_c = if center.abs() < 1e-15 {
        x
    } else {
        LoweredOp::Sub(Arc::new(x), Arc::new(LoweredOp::Const(center)))
    };

    let power = if k == 1 {
        x_minus_c
    } else {
        LoweredOp::Pow(Arc::new(x_minus_c), Arc::new(LoweredOp::Const(k as f64)))
    };

    // Drop the explicit coefficient when it is exactly 1 (saves a Mul node)
    if (coef - 1.0).abs() < 1e-15 {
        power
    } else {
        LoweredOp::Mul(Arc::new(coef_op), Arc::new(power))
    }
}

// ---------------------------------------------------------------------------
// Public API (methods on LoweredOp)
// ---------------------------------------------------------------------------

impl LoweredOp {
    /// Compute the order-`order` Taylor polynomial of `self` about `center`
    /// with respect to variable `wrt`.
    ///
    /// Returns
    ///
    /// ```text
    /// Σ_{k=0}^{order}  f⁽ᵏ⁾(center) / k!  ·  (x_wrt − center)^k
    /// ```
    ///
    /// as a `LoweredOp` tree (simplified).
    ///
    /// # Errors
    ///
    /// - [`EmlError::InvalidParameter`] if `order > 170`
    ///   (f64 factorial overflows beyond 170!).
    /// - [`EmlError::UndefinedAtPoint`] if any derivative is non-finite at
    ///   `center` (e.g. `ln(x).taylor(0, 0.0, n)` since ln(0) = −∞).
    pub fn taylor(&self, wrt: usize, center: f64, order: usize) -> Result<LoweredOp, EmlError> {
        if order > 170 {
            return Err(EmlError::InvalidParameter(
                "order must be ≤ 170 (factorial overflows f64 beyond that)",
            ));
        }

        let ctx = EvalCtx::new(&[]);
        let mut terms: Vec<LoweredOp> = Vec::new();
        // Start with f⁽⁰⁾ = self; advance to the next derivative each iteration.
        let mut deriv = self.clone();

        for k in 0..=order {
            let fn_val = crate::numeric::eval_at_pub(&deriv, wrt, &ctx, center);

            if !fn_val.is_finite() {
                return Err(EmlError::UndefinedAtPoint(center));
            }

            let fact = factorial_f64(k);
            let coef = fn_val / fact;

            // Skip terms whose coefficient is so small it underflows
            if coef.abs() > f64::MIN_POSITIVE {
                terms.push(build_taylor_term(wrt, center, k, coef));
            }

            if k < order {
                deriv = deriv.grad(wrt);
            }
        }

        if terms.is_empty() {
            return Ok(LoweredOp::Const(0.0));
        }

        // Fold all terms into a left-associated Add chain
        let result = match terms
            .into_iter()
            .reduce(|acc, t| LoweredOp::Add(Arc::new(acc), Arc::new(t)))
        {
            Some(op) => op,
            None => LoweredOp::Const(0.0),
        };

        Ok(result.simplify())
    }

    /// Compute the Maclaurin series (Taylor polynomial about center = 0).
    ///
    /// Equivalent to `self.taylor(wrt, 0.0, order)`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`taylor`](Self::taylor).
    pub fn maclaurin(&self, wrt: usize, order: usize) -> Result<LoweredOp, EmlError> {
        self.taylor(wrt, 0.0, order)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate a polynomial (which uses only `Var(0)`) at `x`.
    fn eval_at(poly: &LoweredOp, x: f64) -> f64 {
        poly.eval(&[x])
    }

    // ------------------------------------------------------------------
    // Maclaurin series for standard functions
    // ------------------------------------------------------------------

    #[test]
    fn test_exp_maclaurin_order5() {
        // Maclaurin: 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
        // Error at x=1 is ~1/720 ≈ 0.0014 ≪ tolerance 0.01
        let exp_x = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
        let poly = exp_x.maclaurin(0, 5).expect("maclaurin(exp, 5)");
        let v = eval_at(&poly, 1.0);
        assert!(
            (v - std::f64::consts::E).abs() < 0.01,
            "exp Maclaurin order 5 at x=1: expected ≈{}, got {v}",
            std::f64::consts::E
        );
    }

    #[test]
    fn test_sin_maclaurin_order7() {
        let sin_x = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
        let poly = sin_x.maclaurin(0, 7).expect("maclaurin(sin, 7)");
        let v = eval_at(&poly, 0.5);
        let expected = 0.5_f64.sin();
        assert!(
            (v - expected).abs() < 1e-5,
            "sin Maclaurin order 7 at x=0.5: expected {expected}, got {v}"
        );
    }

    #[test]
    fn test_cos_maclaurin_order6() {
        let cos_x = LoweredOp::Cos(Arc::new(LoweredOp::Var(0)));
        let poly = cos_x.maclaurin(0, 6).expect("maclaurin(cos, 6)");
        let v = eval_at(&poly, 0.5);
        let expected = 0.5_f64.cos();
        assert!(
            (v - expected).abs() < 1e-6,
            "cos Maclaurin order 6 at x=0.5: expected {expected}, got {v}"
        );
    }

    #[test]
    fn test_ln1px_maclaurin_order5() {
        // ln(1+x) ≈ x − x²/2 + x³/3 − x⁴/4 + x⁵/5
        let ln1px = LoweredOp::Ln(Arc::new(LoweredOp::Add(
            Arc::new(LoweredOp::Const(1.0)),
            Arc::new(LoweredOp::Var(0)),
        )));
        let poly = ln1px.maclaurin(0, 5).expect("maclaurin(ln(1+x), 5)");
        let v = eval_at(&poly, 0.5);
        let expected = 1.5_f64.ln();
        assert!(
            (v - expected).abs() < 0.01,
            "ln(1+x) Maclaurin order 5 at x=0.5: expected {expected}, got {v}"
        );
    }

    #[test]
    fn test_geom_series_maclaurin_order4() {
        // 1/(1−x) ≈ 1 + x + x² + x³ + x⁴
        // At x=0.3: 1 + 0.3 + 0.09 + 0.027 + 0.0081 = 1.4251
        let expr = LoweredOp::Div(
            Arc::new(LoweredOp::Const(1.0)),
            Arc::new(LoweredOp::Sub(
                Arc::new(LoweredOp::Const(1.0)),
                Arc::new(LoweredOp::Var(0)),
            )),
        );
        let poly = expr.maclaurin(0, 4).expect("maclaurin(1/(1-x), 4)");
        let v = eval_at(&poly, 0.3);
        assert!(
            (v - 1.4251_f64).abs() < 0.005,
            "1/(1−x) Maclaurin order 4 at x=0.3: expected ≈1.4251, got {v}"
        );
    }

    // ------------------------------------------------------------------
    // Taylor about a non-zero center
    // ------------------------------------------------------------------

    #[test]
    fn test_taylor_nonzero_center() {
        // exp(x) about center=1, order=3:
        //   p(x) = e + e(x−1) + e/2·(x−1)² + e/6·(x−1)³
        //   p(1) = e exactly.
        let exp_x = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
        let poly = exp_x.taylor(0, 1.0, 3).expect("taylor(exp, center=1, 3)");
        let v = eval_at(&poly, 1.0);
        assert!(
            (v - std::f64::consts::E).abs() < 1e-10,
            "exp Taylor center=1 order 3 at x=1: expected e, got {v}"
        );
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_taylor_order0() {
        // Order-0 Taylor of exp(x) at center=0 is just exp(0) = 1.
        let exp_x = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
        let poly = exp_x.taylor(0, 0.0, 0).expect("taylor order 0");
        let v = eval_at(&poly, 0.0);
        assert!(
            (v - 1.0).abs() < 1e-12,
            "order-0 Taylor should give 1.0, got {v}"
        );
    }

    #[test]
    fn test_taylor_undefined_at_point() {
        // ln(x) at center=0: ln(0) = −∞ → UndefinedAtPoint(0.0)
        let ln_x = LoweredOp::Ln(Arc::new(LoweredOp::Var(0)));
        let result = ln_x.maclaurin(0, 3);
        assert!(
            matches!(result, Err(EmlError::UndefinedAtPoint(x)) if x == 0.0),
            "expected UndefinedAtPoint(0.0), got {result:?}"
        );
    }

    #[test]
    fn test_taylor_invalid_order() {
        let exp_x = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
        let result = exp_x.maclaurin(0, 171);
        assert!(
            matches!(result, Err(EmlError::InvalidParameter(_))),
            "expected InvalidParameter for order=171, got {result:?}"
        );
    }
}
