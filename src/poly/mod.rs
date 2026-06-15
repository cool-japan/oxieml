//! Polynomial algebra over exact rational coefficients.
//!
//! This module provides:
//!
//! - [`Poly`] — dense univariate polynomial over `Ratio<i64>`.
//! - [`MultiPoly`] — sparse multivariate polynomial via `BTreeMap<Vec<u32>, Ratio<i64>>`.
//! - [`PolyError`] — error type for polynomial operations.
//!
//! Exact `Ratio<i64>` coefficients are used throughout to support decidable
//! algorithms like GCD, Yun's square-free factorization, and the rational root
//! theorem. All arithmetic uses `i64::checked_*` to detect overflow early;
//! overflow returns [`PolyError::CoeffOverflow`] rather than panicking.
//!
//! The `to_lowered` method emits standard `LoweredOp` trees with `Const(f64)`
//! coefficients, so the symbolic IR layer stays purely `f64`-based.

pub mod factor;
pub mod multivariate;
pub mod sturm;
#[cfg(test)]
mod tests;
pub mod univariate;

pub use factor::Factorization;
pub use multivariate::MultiPoly;
pub use univariate::Poly;

use num_rational::Ratio;

/// Errors that can arise during polynomial operations.
#[derive(Clone, Debug, PartialEq)]
pub enum PolyError {
    /// The `LoweredOp` expression is not a polynomial.
    NotPolynomial,
    /// An intermediate rational coefficient would overflow `i64`.
    CoeffOverflow,
    /// Division by the zero polynomial.
    DivByZero,
}

impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPolynomial => write!(f, "expression is not a polynomial"),
            Self::CoeffOverflow => write!(f, "rational coefficient overflowed i64"),
            Self::DivByZero => write!(f, "division by zero polynomial"),
        }
    }
}

impl std::error::Error for PolyError {}

/// Add two rationals with overflow checking.
pub(super) fn checked_add(a: Ratio<i64>, b: Ratio<i64>) -> Result<Ratio<i64>, PolyError> {
    let numer = (*a.numer())
        .checked_mul(*b.denom())
        .and_then(|x| {
            (*b.numer())
                .checked_mul(*a.denom())
                .and_then(|y| x.checked_add(y))
        })
        .ok_or(PolyError::CoeffOverflow)?;
    let denom = (*a.denom())
        .checked_mul(*b.denom())
        .ok_or(PolyError::CoeffOverflow)?;
    Ok(Ratio::new(numer, denom))
}

/// Subtract two rationals with overflow checking.
pub(super) fn checked_sub(a: Ratio<i64>, b: Ratio<i64>) -> Result<Ratio<i64>, PolyError> {
    let numer = (*a.numer())
        .checked_mul(*b.denom())
        .and_then(|x| {
            (*b.numer())
                .checked_mul(*a.denom())
                .and_then(|y| x.checked_sub(y))
        })
        .ok_or(PolyError::CoeffOverflow)?;
    let denom = (*a.denom())
        .checked_mul(*b.denom())
        .ok_or(PolyError::CoeffOverflow)?;
    Ok(Ratio::new(numer, denom))
}

/// Multiply two rationals with overflow checking.
pub(super) fn checked_mul(a: Ratio<i64>, b: Ratio<i64>) -> Result<Ratio<i64>, PolyError> {
    let numer = (*a.numer())
        .checked_mul(*b.numer())
        .ok_or(PolyError::CoeffOverflow)?;
    let denom = (*a.denom())
        .checked_mul(*b.denom())
        .ok_or(PolyError::CoeffOverflow)?;
    Ok(Ratio::new(numer, denom))
}

/// Negate a rational with overflow checking.
pub(super) fn checked_neg(a: Ratio<i64>) -> Result<Ratio<i64>, PolyError> {
    let numer = (*a.numer()).checked_neg().ok_or(PolyError::CoeffOverflow)?;
    Ok(Ratio::new(numer, *a.denom()))
}

/// Convert a rational to `f64`.
pub(super) fn ratio_to_f64(r: &Ratio<i64>) -> f64 {
    *r.numer() as f64 / *r.denom() as f64
}

/// Try to convert an `f64` to a `Ratio<i64>` with a small denominator (≤ 1000).
pub(super) fn f64_to_ratio(v: f64) -> Result<Ratio<i64>, PolyError> {
    if !v.is_finite() {
        return Err(PolyError::NotPolynomial);
    }
    for denom in 1i64..=1000 {
        let numer_f = v * denom as f64;
        let numer_rounded = numer_f.round() as i64;
        if (numer_f - numer_rounded as f64).abs() < 1e-9 {
            return Ok(Ratio::new(numer_rounded, denom));
        }
    }
    Err(PolyError::NotPolynomial)
}

/// Return all integer divisors (positive and negative) of `n`.
pub(super) fn integer_factors(n: i64) -> Vec<i64> {
    if n == 0 {
        return vec![0];
    }
    let abs_n = n.unsigned_abs();
    let mut factors: Vec<i64> = Vec::new();
    let mut i = 1u64;
    while i * i <= abs_n {
        if abs_n.is_multiple_of(i) {
            factors.push(i as i64);
            factors.push(-(i as i64));
            let other = (abs_n / i) as i64;
            if other != i as i64 {
                factors.push(other);
                factors.push(-other);
            }
        }
        i += 1;
    }
    factors
}

/// Euclidean GCD for i64.
pub(super) fn gcd_i64(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Helper: evaluate integer-coefficient poly at rational point.
pub(super) fn eval_poly_i64_at_ratio(
    coeffs: &[i64],
    x: Ratio<i64>,
) -> Result<Ratio<i64>, PolyError> {
    let mut result = Ratio::new(0i64, 1);
    let mut x_pow = Ratio::new(1i64, 1);
    for &c in coeffs {
        let term = checked_mul(Ratio::new(c, 1), x_pow)?;
        result = checked_add(result, term)?;
        x_pow = checked_mul(x_pow, x)?;
    }
    Ok(result)
}
