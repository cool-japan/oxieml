//! Faulhaber's formula for power sums via Bernoulli numbers.
//!
//! # Mathematics
//!
//! For a fixed non-negative integer `p`, the power sum
//!
//! ```text
//! S_p(n) = Σ_{k=1}^{n} k^p
//! ```
//!
//! is a polynomial in `n` of degree `p + 1`. **Faulhaber's formula** expresses
//! it in closed form through the Bernoulli numbers `B_j`:
//!
//! ```text
//! S_p(n) = 1/(p+1) · Σ_{j=0}^{p} C(p+1, j) · B_j^+ · n^{p+1-j}
//! ```
//!
//! where `C(p+1, j)` is a binomial coefficient and `B_j^+` are the Bernoulli
//! numbers in the convention `B_1 = +1/2` (the "second" Bernoulli numbers). All
//! other `B_j` agree with the usual (`B_1 = -1/2`) convention, because every odd
//! Bernoulli number beyond `B_1` vanishes.
//!
//! ## Bernoulli numbers
//!
//! The Bernoulli numbers (in the `B_1 = -1/2` convention) satisfy the defining
//! recurrence obtained from `Σ_{j=0}^{m} C(m+1, j) B_j = 0` (`m ≥ 1`):
//!
//! ```text
//! B_0 = 1,      B_m = -1/(m+1) · Σ_{j=0}^{m-1} C(m+1, j) B_j   (m ≥ 1)
//! ```
//!
//! yielding `B_1 = -1/2`, `B_2 = 1/6`, `B_3 = 0`, `B_4 = -1/30`, … Every value is
//! computed with exact [`Coeff`] (`BigRational`) arithmetic, so there is no
//! overflow and no rounding — a machine-integer implementation would overflow
//! `i64` already at `B_{34}`.
//!
//! ## Correctness spot-checks
//!
//! - `p = 1`: `S_1(n) = 1/2 (n² + n) = n(n+1)/2`.
//! - `p = 2`: `S_2(n) = n³/3 + n²/2 + n/6 = n(n+1)(2n+1)/6`.
//! - `p = 3`: `S_3(n) = 1/4 (n⁴ + 2n³ + n²) = (n(n+1)/2)²`.

use num_bigint::BigInt;

use crate::poly::{Coeff, Poly, coeff_is_zero, coeff_one, coeff_recip, coeff_zero};

/// Exact binomial coefficient `C(n, k)` as a [`BigInt`].
///
/// Computed as `∏_{i=0}^{k-1} (n - i) / ∏_{i=1}^{k} i`. The final quotient is an
/// exact integer (the numerator product is divisible by the denominator
/// product), so no rounding occurs.
fn binomial(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::from(0i32);
    }
    let k = k.min(n - k);
    let mut numer = BigInt::from(1i32);
    let mut denom = BigInt::from(1i32);
    for i in 0..k {
        numer *= BigInt::from(n - i);
        denom *= BigInt::from(i + 1);
    }
    numer / denom
}

/// Compute the Bernoulli numbers `B_0, B_1, …, B_m` in the `B_1 = -1/2`
/// convention, exactly.
///
/// Uses the recurrence `B_m = -1/(m+1) Σ_{j<m} C(m+1, j) B_j`.
pub(crate) fn bernoulli_numbers(m: usize) -> Vec<Coeff> {
    let mut b: Vec<Coeff> = Vec::with_capacity(m + 1);
    b.push(coeff_one()); // B_0 = 1
    for i in 1..=m {
        let mut sum = coeff_zero();
        for (j, bj) in b.iter().enumerate().take(i) {
            let c = Coeff::from_integer(binomial((i + 1) as u64, j as u64));
            sum = &sum + &(bj * &c);
        }
        // factor = -1 / (i + 1)
        let factor = Coeff::new(BigInt::from(-1i32), BigInt::from((i + 1) as i64));
        b.push(&sum * &factor);
    }
    b
}

/// Return `S_p(n) = Σ_{k=1}^{n} k^p` as an exact polynomial in `n`.
///
/// Implements Faulhaber's formula with the `B_1 = +1/2` convention (obtained by
/// flipping the sign of `B_1` from [`bernoulli_numbers`]).
pub(crate) fn power_sum_poly(p: usize) -> Poly {
    let bern = bernoulli_numbers(p);
    let inv_p1 = coeff_recip(&Coeff::from_integer(BigInt::from((p + 1) as i64)))
        .expect("p + 1 is strictly positive, so its reciprocal exists");

    let mut result = Poly::zero();
    for j in 0..=p {
        // B_j^+ : identical to B_j except B_1 flips sign to +1/2.
        let bj = if j == 1 { -&bern[1] } else { bern[j].clone() };
        if coeff_is_zero(&bj) {
            continue;
        }
        let binom = Coeff::from_integer(binomial((p + 1) as u64, j as u64));
        let coeff = &(&binom * &bj) * &inv_p1;
        let degree = p + 1 - j;
        let mono = Poly::monomial(degree)
            .scale(&coeff)
            .expect("polynomial scaling is infallible");
        result = result
            .add(&mono)
            .expect("polynomial addition is infallible");
    }
    result
}

/// Sum a polynomial term: return `Σ_{k=1}^{n} term(k)` as an exact polynomial in
/// `n`.
///
/// Decomposes `term(k) = Σ_p a_p k^p` and sums each monomial with
/// [`power_sum_poly`], scaled by `a_p`. This is the antidifference `S(n)` that
/// satisfies `S(n) - S(n-1) = term(n)` and `S(0) = 0` (empty sum).
pub(crate) fn sum_polynomial(term: &Poly) -> Poly {
    let normalized = term.normalized();
    let mut result = Poly::zero();
    for (power, coeff) in normalized.coeffs.iter().enumerate() {
        if coeff_is_zero(coeff) {
            continue;
        }
        let contribution = power_sum_poly(power)
            .scale(coeff)
            .expect("polynomial scaling is infallible");
        result = result
            .add(&contribution)
            .expect("polynomial addition is infallible");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_coeff(v: i64) -> Coeff {
        Coeff::from_integer(BigInt::from(v))
    }

    #[test]
    fn bernoulli_known_values() {
        let b = bernoulli_numbers(8);
        assert_eq!(b[0], int_coeff(1));
        assert_eq!(b[1], Coeff::new(BigInt::from(-1i32), BigInt::from(2i32)));
        assert_eq!(b[2], Coeff::new(BigInt::from(1i32), BigInt::from(6i32)));
        assert_eq!(b[3], coeff_zero());
        assert_eq!(b[4], Coeff::new(BigInt::from(-1i32), BigInt::from(30i32)));
        assert_eq!(b[5], coeff_zero());
        assert_eq!(b[6], Coeff::new(BigInt::from(1i32), BigInt::from(42i32)));
        assert_eq!(b[7], coeff_zero());
        assert_eq!(b[8], Coeff::new(BigInt::from(-1i32), BigInt::from(30i32)));
    }

    #[test]
    fn power_sum_linear() {
        // S_1(n) = 1/2 n^2 + 1/2 n
        let s = power_sum_poly(1);
        assert_eq!(s.coeffs[0], coeff_zero());
        assert_eq!(
            s.coeffs[1],
            Coeff::new(BigInt::from(1i32), BigInt::from(2i32))
        );
        assert_eq!(
            s.coeffs[2],
            Coeff::new(BigInt::from(1i32), BigInt::from(2i32))
        );
    }

    #[test]
    fn power_sum_matches_brute_force() {
        for p in 0..=6usize {
            let s = power_sum_poly(p);
            for n in 0..=12i64 {
                let expected: i64 = (1..=n).map(|k| k.pow(p as u32)).sum();
                let got = s
                    .eval(&Coeff::from_integer(BigInt::from(n)))
                    .expect("polynomial evaluation is infallible");
                assert_eq!(
                    got,
                    Coeff::from_integer(BigInt::from(expected)),
                    "power sum mismatch p={p} n={n}"
                );
            }
        }
    }

    #[test]
    fn sum_polynomial_combines_monomials() {
        // term = 3 k^2 + 2 k + 1 ;  Σ_{k=1}^{n} term = n^3 + 5/2 n^2 + ... check brute force.
        let term = Poly::from_int_coeffs(&[1, 2, 3]);
        let s = sum_polynomial(&term);
        for n in 0..=10i64 {
            let expected: i64 = (1..=n).map(|k| 3 * k * k + 2 * k + 1).sum();
            let got = s
                .eval(&Coeff::from_integer(BigInt::from(n)))
                .expect("polynomial evaluation is infallible");
            assert_eq!(got, Coeff::from_integer(BigInt::from(expected)));
        }
    }
}
