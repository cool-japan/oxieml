//! Hypergeometric term-ratio classification.
//!
//! # Mathematics
//!
//! A summand `t(k)` is *hypergeometric* in `k` when the **term ratio**
//!
//! ```text
//! r(k) = t(k+1) / t(k)
//! ```
//!
//! is a rational function of `k`. This module computes that ratio structurally
//! from a [`LoweredOp`] and classifies it as
//!
//! - a **rational function** `num(k)/den(k)` over ℚ (exact) — the case Gosper's
//!   algorithm needs, and, when `num`/`den` are both constant, the geometric
//!   case with a rational common ratio;
//! - an **irrational constant** `c` — e.g. `e^{αk}` has `r(k) = e^α`, a constant
//!   that is not represented over ℚ but still yields a geometric closed form; or
//! - **not hypergeometric**.
//!
//! ## Ratio of the standard atoms
//!
//! The computation walks the multiplicative structure of the term, accumulating
//! a rational function `num/den` times a real constant `κ`. The atoms are:
//!
//! | atom `t(k)`             | ratio `r(k)`                                  |
//! |-------------------------|-----------------------------------------------|
//! | constant in `k`         | `1`                                           |
//! | `k`                     | `(k+1)/k`                                      |
//! | polynomial `p(k)`       | `p(k+1)/p(k)`                                  |
//! | `b^{k}` (`b` constant)  | `b`  (routed to ℚ when `b` is rational)       |
//! | `b^{αk+β}`              | `b^α`                                         |
//! | `Γ(g(k))`, `g` linear   | `Γ(g(k)+α)/Γ(g(k)) = ∏_{i=0}^{α-1}(g(k)+i)`   |
//! | `e^{αk+β}`              | `e^α`  (irrational constant `κ`)              |
//!
//! Factorials are handled through `Γ`: `k! = Γ(k+1) = exp(lgamma(k+1))`, so
//! `Γ(k+2)/Γ(k+1) = k+1`. Products and quotients multiply/divide the accumulated
//! ratios; integer powers raise them.
//!
//! When an irrational constant `κ ≠ 1` is mixed with a *non-constant* rational
//! part the overall ratio is not a rational function over ℚ, and the term is
//! honestly reported as not (ℚ-)hypergeometric.

use num_bigint::{BigInt, Sign};

use crate::lower::LoweredOp;
use crate::poly::{Coeff, Poly, coeff_one, coeff_recip, f64_to_ratio, ratio_to_f64};

use super::gosper::shift_poly;

/// Classification of a term's ratio `r(k) = t(k+1)/t(k)`.
pub(crate) enum TermRatio {
    /// `r(k) = num(k) / den(k)`, an exact rational function over ℚ (reduced).
    Rational {
        /// Numerator polynomial.
        num: Poly,
        /// Denominator polynomial.
        den: Poly,
    },
    /// `r(k)` is a (possibly irrational) non-unit real constant.
    IrrationalConstant(f64),
    /// The term is not hypergeometric.
    NotHyper,
}

/// Accumulator representing a ratio `κ · num(k) / den(k)`.
struct Acc {
    num: Poly,
    den: Poly,
    konst: f64,
}

impl Acc {
    fn identity() -> Self {
        Self {
            num: Poly::constant(coeff_one()),
            den: Poly::constant(coeff_one()),
            konst: 1.0,
        }
    }
}

/// Does `op` reference variable `var` anywhere?
fn contains_var(op: &LoweredOp, var: usize) -> bool {
    match op {
        LoweredOp::Var(i) => *i == var,
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => false,
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
        | LoweredOp::Ci(a) => contains_var(a, var),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => contains_var(a, var) || contains_var(b, var),
    }
}

/// Exact rational power `base^exp` for a (possibly negative) integer exponent.
fn coeff_powi(base: &Coeff, exp: i64) -> Option<Coeff> {
    if exp == 0 {
        return Some(coeff_one());
    }
    let (mut acc, mut n) = if exp < 0 {
        (coeff_recip(base)?, (-exp) as u64)
    } else {
        (base.clone(), exp as u64)
    };
    let mut result = coeff_one();
    while n > 0 {
        if n & 1 == 1 {
            result = &result * &acc;
        }
        n >>= 1;
        if n > 0 {
            acc = &acc * &acc;
        }
    }
    Some(result)
}

/// Convert a rational coefficient to a (possibly negative) integer, or `None`.
fn coeff_to_i64(c: &Coeff) -> Option<i64> {
    if !c.is_integer() {
        return None;
    }
    let value = c.to_integer();
    let negative = value.sign() == Sign::Minus;
    let mut digits = value.magnitude().iter_u64_digits();
    let low = digits.next().unwrap_or(0);
    if digits.next().is_some() {
        return None;
    }
    let signed = i64::try_from(low).ok()?;
    Some(if negative { -signed } else { signed })
}

/// Is `p` a non-zero constant polynomial?
fn is_const(p: &Poly) -> bool {
    p.degree() == Some(0)
}

/// Multiply two accumulators (ratio of a product is the product of the ratios).
fn combine_mul(a: Acc, b: Acc) -> Option<Acc> {
    Some(Acc {
        num: a.num.mul(&b.num).ok()?,
        den: a.den.mul(&b.den).ok()?,
        konst: a.konst * b.konst,
    })
}

/// Divide two accumulators (ratio of a quotient is the quotient of the ratios).
fn combine_div(a: Acc, b: Acc) -> Option<Acc> {
    Some(Acc {
        num: a.num.mul(&b.den).ok()?,
        den: a.den.mul(&b.num).ok()?,
        konst: a.konst / b.konst,
    })
}

/// Raise an accumulator to a (possibly negative) integer power.
fn pow_acc(a: Acc, exp: i64) -> Option<Acc> {
    if exp == 0 {
        return Some(Acc::identity());
    }
    let magnitude = exp.unsigned_abs() as usize;
    let (num, den) = if exp > 0 {
        (a.num.pow(magnitude).ok()?, a.den.pow(magnitude).ok()?)
    } else {
        (a.den.pow(magnitude).ok()?, a.num.pow(magnitude).ok()?)
    };
    Some(Acc {
        num,
        den,
        konst: a.konst.powi(exp as i32),
    })
}

/// Build the accumulator for a polynomial atom `p(k)`: ratio `p(k+1)/p(k)`.
fn poly_atom(p: Poly) -> Option<Acc> {
    if p.is_zero() {
        return None;
    }
    Some(Acc {
        num: shift_poly(&p, 1),
        den: p,
        konst: 1.0,
    })
}

/// Accumulator for a `Γ(g(k))` factor with `g` a polynomial.
///
/// `Γ(g(k+1))/Γ(g(k))` is a rational function only when `α = g(k+1) - g(k)` is a
/// constant integer; then it equals `∏_{i=0}^{α-1}(g+i)` for `α > 0` (numerator)
/// or `1/∏_{i=1}^{|α|}(g-i)` for `α < 0` (denominator).
fn gamma_atom(g: &LoweredOp, var: usize) -> Option<Acc> {
    let g_poly = Poly::from_lowered(g, var).ok()?;
    let delta = shift_poly(&g_poly, 1).sub(&g_poly).ok()?;
    if delta.is_zero() {
        return Some(Acc::identity());
    }
    if delta.degree() != Some(0) {
        return None; // g non-linear -> Γ ratio not rational
    }
    let alpha = coeff_to_i64(&delta.leading_coeff())?;
    if alpha == 0 {
        return Some(Acc::identity());
    }
    let mut product = Poly::constant(coeff_one());
    if alpha > 0 {
        for i in 0..alpha {
            let shifted = g_poly
                .add(&Poly::constant(Coeff::from_integer(BigInt::from(i))))
                .ok()?;
            product = product.mul(&shifted).ok()?;
        }
        Some(Acc {
            num: product,
            den: Poly::constant(coeff_one()),
            konst: 1.0,
        })
    } else {
        for i in 1..=(-alpha) {
            let shifted = g_poly
                .sub(&Poly::constant(Coeff::from_integer(BigInt::from(i))))
                .ok()?;
            product = product.mul(&shifted).ok()?;
        }
        Some(Acc {
            num: Poly::constant(coeff_one()),
            den: product,
            konst: 1.0,
        })
    }
}

/// Accumulator for an exponential base `b^{e(k)}` (`b` a positive-or-negative
/// constant, `e` a polynomial in `k`).
fn power_base_atom(b: f64, exp: &LoweredOp, var: usize) -> Option<Acc> {
    if b == 0.0 {
        return None;
    }
    let e_poly = Poly::from_lowered(exp, var).ok()?;
    let delta = shift_poly(&e_poly, 1).sub(&e_poly).ok()?;
    if delta.is_zero() {
        return Some(Acc::identity());
    }
    if delta.degree() != Some(0) {
        return None; // b^{non-linear} is not hypergeometric
    }
    let alpha = delta.leading_coeff();
    match coeff_to_i64(&alpha) {
        Some(alpha_int) => {
            // b^α is a rational constant -> keep it over ℚ (numerator factor).
            let base = f64_to_ratio(b).ok()?;
            let factor = coeff_powi(&base, alpha_int)?;
            Some(Acc {
                num: Poly::constant(factor),
                den: Poly::constant(coeff_one()),
                konst: 1.0,
            })
        }
        None => {
            // Non-integer exponent: b^α is generally irrational.
            let alpha_f = ratio_to_f64(&alpha);
            if b < 0.0 {
                return None; // negative base to a non-integer power is not real
            }
            Some(Acc {
                num: Poly::constant(coeff_one()),
                den: Poly::constant(coeff_one()),
                konst: b.powf(alpha_f),
            })
        }
    }
}

/// Accumulator for `e^{a(k)}` (natural exponential) with `a` a polynomial.
fn exp_atom(a: &LoweredOp, var: usize) -> Option<Acc> {
    let a_poly = Poly::from_lowered(a, var).ok()?;
    let delta = shift_poly(&a_poly, 1).sub(&a_poly).ok()?;
    if delta.is_zero() {
        return Some(Acc::identity());
    }
    if delta.degree() != Some(0) {
        return None;
    }
    let alpha = ratio_to_f64(&delta.leading_coeff());
    Some(Acc {
        num: Poly::constant(coeff_one()),
        den: Poly::constant(coeff_one()),
        konst: alpha.exp(),
    })
}

/// Recursively accumulate the term ratio, or `None` when not hypergeometric.
fn acc_ratio(term: &LoweredOp, var: usize) -> Option<Acc> {
    if !contains_var(term, var) {
        // Constant factor in `k`: ratio 1.
        return Some(Acc::identity());
    }

    match term {
        LoweredOp::Var(i) => {
            debug_assert_eq!(*i, var, "contains_var guaranteed this is the summation var");
            poly_atom(Poly::from_ratios(vec![
                crate::poly::coeff_zero(),
                coeff_one(),
            ]))
        }
        LoweredOp::Neg(a) => acc_ratio(a, var),
        LoweredOp::Add(_, _) | LoweredOp::Sub(_, _) => {
            let p = Poly::from_lowered(term, var).ok()?;
            poly_atom(p)
        }
        LoweredOp::Mul(a, b) => combine_mul(acc_ratio(a, var)?, acc_ratio(b, var)?),
        LoweredOp::Div(a, b) => combine_div(acc_ratio(a, var)?, acc_ratio(b, var)?),
        LoweredOp::Pow(base, exp) => {
            if let LoweredOp::Const(e) = exp.as_ref() {
                // Integer power of a hypergeometric base.
                if e.fract() != 0.0 {
                    return None;
                }
                let exp_int = coeff_to_i64(&f64_to_ratio(*e).ok()?)?;
                pow_acc(acc_ratio(base, var)?, exp_int)
            } else if let LoweredOp::Const(b) = base.as_ref() {
                // b^{e(k)} with variable exponent.
                power_base_atom(*b, exp, var)
            } else if let LoweredOp::NamedConst(nc) = base.as_ref() {
                power_base_atom(nc.value(), exp, var)
            } else {
                None
            }
        }
        LoweredOp::Exp(a) => {
            if let LoweredOp::LGamma(g) = a.as_ref() {
                gamma_atom(g, var)
            } else {
                exp_atom(a, var)
            }
        }
        // Any other var-dependent construct is not a hypergeometric atom.
        _ => None,
    }
}

/// Classify the ratio `r(k) = term(k+1)/term(k)` of a term in variable `var`.
pub(crate) fn term_ratio(term: &LoweredOp, var: usize) -> TermRatio {
    let acc = match acc_ratio(term, var) {
        Some(a) => a,
        None => return TermRatio::NotHyper,
    };
    let num = acc.num.normalized();
    let den = acc.den.normalized();
    if num.is_zero() || den.is_zero() {
        return TermRatio::NotHyper;
    }

    let gcd = Poly::gcd(&num, &den).unwrap_or_else(|_| Poly::constant(coeff_one()));
    let num_reduced = num.div_rem(&gcd).map_or(num.clone(), |(q, _)| q);
    let den_reduced = den.div_rem(&gcd).map_or(den.clone(), |(q, _)| q);

    if acc.konst == 1.0 {
        return TermRatio::Rational {
            num: num_reduced,
            den: den_reduced,
        };
    }

    // κ ≠ 1: only meaningful when the rational part is itself constant.
    if is_const(&num_reduced) && is_const(&den_reduced) {
        let numerator = ratio_to_f64(&num_reduced.leading_coeff());
        let denominator = ratio_to_f64(&den_reduced.leading_coeff());
        if denominator == 0.0 {
            return TermRatio::NotHyper;
        }
        return TermRatio::IrrationalConstant(acc.konst * numerator / denominator);
    }

    TermRatio::NotHyper
}

/// A convenience accessor used by the dispatcher: is this ratio a constant, and
/// if so, what is its `f64` value?
pub(crate) fn constant_ratio(num: &Poly, den: &Poly) -> Option<f64> {
    if is_const(num) && is_const(den) {
        let numerator = ratio_to_f64(&num.leading_coeff());
        let denominator = ratio_to_f64(&den.leading_coeff());
        if denominator == 0.0 {
            return None;
        }
        Some(numerator / denominator)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    #[test]
    fn geometric_two_pow_k() {
        // 2^k -> ratio 2.
        let term = LoweredOp::Pow(Arc::new(c(2.0)), Arc::new(var(0)));
        match term_ratio(&term, 0) {
            TermRatio::Rational { num, den } => {
                assert_eq!(constant_ratio(&num, &den), Some(2.0));
            }
            _ => panic!("2^k should have a rational constant ratio"),
        }
    }

    #[test]
    fn factorial_times_k() {
        // k · k! = k · Γ(k+1) -> ratio (k+1)^2 / k.
        let factorial = LoweredOp::Exp(Arc::new(LoweredOp::LGamma(Arc::new(LoweredOp::Add(
            Arc::new(var(0)),
            Arc::new(c(1.0)),
        )))));
        let term = LoweredOp::Mul(Arc::new(var(0)), Arc::new(factorial));
        match term_ratio(&term, 0) {
            TermRatio::Rational { num, den } => {
                assert_eq!(num, Poly::from_int_coeffs(&[1, 2, 1]));
                assert_eq!(den, Poly::from_int_coeffs(&[0, 1]));
            }
            _ => panic!("k·k! should be rational-ratio hypergeometric"),
        }
    }

    #[test]
    fn reciprocal_harmonic() {
        // 1/k -> ratio k/(k+1).
        let term = LoweredOp::Div(Arc::new(c(1.0)), Arc::new(var(0)));
        match term_ratio(&term, 0) {
            TermRatio::Rational { num, den } => {
                assert_eq!(num, Poly::from_int_coeffs(&[0, 1]));
                assert_eq!(den, Poly::from_int_coeffs(&[1, 1]));
            }
            _ => panic!("1/k should be rational-ratio hypergeometric"),
        }
    }

    #[test]
    fn non_hypergeometric_ln() {
        // ln(k) is not hypergeometric.
        let term = LoweredOp::Ln(Arc::new(var(0)));
        assert!(matches!(term_ratio(&term, 0), TermRatio::NotHyper));
    }

    #[test]
    fn exp_linear_is_irrational_constant() {
        // e^k -> ratio e (irrational constant).
        let term = LoweredOp::Exp(Arc::new(var(0)));
        match term_ratio(&term, 0) {
            TermRatio::IrrationalConstant(v) => {
                assert!((v - std::f64::consts::E).abs() < 1e-12);
            }
            _ => panic!("e^k should be a geometric term with irrational ratio"),
        }
    }
}
