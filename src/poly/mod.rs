//! Polynomial algebra over exact, arbitrary-precision rational coefficients.
//!
//! This module provides:
//!
//! - [`Poly`] — dense univariate polynomial over [`Coeff`] (`BigRational`).
//! - [`MultiPoly`] — sparse multivariate polynomial via `BTreeMap<Vec<u32>, Coeff>`.
//! - [`PolyError`] — error type for polynomial operations.
//!
//! # Exactness
//!
//! Coefficients are [`num_rational::BigRational`] — a ratio of two arbitrary
//! precision [`num_bigint::BigInt`]s, always kept in lowest terms with a
//! positive denominator. Consequently **no polynomial operation can overflow**:
//! `add`, `sub`, `mul`, `pow`, `div_rem`, `gcd`, `resultant`, `discriminant`,
//! `content`, `square_free` and `rational_roots` are all exact over ℚ for any
//! input in ℚ. The [`PolyError::CoeffOverflow`] variant is retained for
//! backwards compatibility but is never constructed.
//!
//! # f64 ↔ ℚ conversion
//!
//! Every finite `f64` *is* a rational number (a dyadic fraction `m / 2^k`), so
//! [`f64_to_ratio`] never has to guess. It returns the **simplest** rational
//! that round-trips back to the exact same `f64`, found by walking the
//! continued-fraction (Stern–Brocot) expansion of the exact dyadic value. This
//! recovers `1/3` from `1.0/3.0`, `355/113` from `355.0/113.0` and `1/10` from
//! the literal `0.1`, while still being able to represent a value such as `π`
//! exactly (as the dyadic fraction the `f64` really denotes).
//!
//! [`ratio_to_f64`] is the inverse: a correctly-rounded (round-half-to-even)
//! conversion that works for coefficients far outside the `f64` range, saturating
//! to ±∞ / ±0 rather than producing `NaN`.
//!
//! The `to_lowered` methods emit standard `LoweredOp` trees with `Const(f64)`
//! coefficients, so the symbolic IR layer stays purely `f64`-based.

pub mod factor;
pub mod factor_zassenhaus;
pub mod groebner;
pub mod modular;
pub mod monomial;
pub mod multivariate;
pub mod solve_system;
pub mod sturm;
#[cfg(test)]
mod tests;
pub mod univariate;

pub use factor::Factorization;
pub use groebner::{GroebnerError, GroebnerOpts, GroebnerStats, groebner_basis, ideal_member};
pub use monomial::{MonOrder, Monomial};
pub use multivariate::MultiPoly;
pub use solve_system::{ZeroDimOutcome, solve_zero_dim};
pub use univariate::Poly;

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_rational::BigRational;

/// The exact coefficient type used by every polynomial in this module.
///
/// This is an arbitrary-precision rational number, so coefficient arithmetic is
/// exact and total: it can never overflow.
pub type Coeff = BigRational;

/// Errors that can arise during polynomial operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolyError {
    /// The `LoweredOp` expression is not a polynomial (or an `f64` constant was
    /// not finite and therefore has no rational value).
    NotPolynomial,
    /// An intermediate rational coefficient would overflow.
    ///
    /// Retained for backwards compatibility. Coefficients are arbitrary
    /// precision, so this variant is **never constructed** any more.
    CoeffOverflow,
    /// Division by the zero polynomial.
    DivByZero,
    /// An integer factorization needed by the rational root theorem (or by
    /// Kronecker's method) exceeded the search budget.
    ///
    /// This happens only for coefficients whose prime factorization is genuinely
    /// hard (large semiprimes) or which have an unreasonable number of divisors.
    /// It is an honest "cannot decide" rather than a silently wrong answer.
    FactorizationLimit,
}

impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPolynomial => write!(f, "expression is not a polynomial"),
            Self::CoeffOverflow => write!(f, "rational coefficient overflowed"),
            Self::DivByZero => write!(f, "division by zero polynomial"),
            Self::FactorizationLimit => {
                write!(f, "integer factorization exceeded the search budget")
            }
        }
    }
}

impl std::error::Error for PolyError {}

// ── Coefficient constructors ──────────────────────────────────────────────────

/// The rational constant `0`.
#[must_use]
pub fn coeff_zero() -> Coeff {
    Coeff::new_raw(BigInt::from(0i32), BigInt::from(1i32))
}

/// The rational constant `1`.
#[must_use]
pub fn coeff_one() -> Coeff {
    Coeff::new_raw(BigInt::from(1i32), BigInt::from(1i32))
}

/// Build the rational `n / 1` from a machine integer.
#[must_use]
pub fn coeff_from_i64(n: i64) -> Coeff {
    Coeff::new_raw(BigInt::from(n), BigInt::from(1i32))
}

/// Build the rational `numer / denom` from machine integers.
///
/// # Errors
///
/// Returns [`PolyError::DivByZero`] when `denom == 0`.
pub fn coeff_from_i64_ratio(numer: i64, denom: i64) -> Result<Coeff, PolyError> {
    if denom == 0 {
        return Err(PolyError::DivByZero);
    }
    Ok(Coeff::new(BigInt::from(numer), BigInt::from(denom)))
}

/// Return `true` when `c` is exactly zero.
#[must_use]
pub fn coeff_is_zero(c: &Coeff) -> bool {
    c.numer().sign() == Sign::NoSign
}

/// Multiplicative inverse of `c`, or `None` when `c` is zero.
#[must_use]
pub fn coeff_recip(c: &Coeff) -> Option<Coeff> {
    if coeff_is_zero(c) {
        return None;
    }
    // `Ratio::new` re-normalizes the sign, so a negative numerator is handled.
    Some(Coeff::new(c.denom().clone(), c.numer().clone()))
}

// ── f64 → ℚ (continued fraction / Stern–Brocot) ───────────────────────────────

/// Upper bound on the number of continued-fraction terms explored.
///
/// The exact value of an `f64` is a dyadic rational `m / 2^k` with `m < 2^53`
/// and `k <= 1074`; a Euclidean expansion of such a pair terminates in far fewer
/// than this many steps. The bound is purely defensive.
const MAX_CF_TERMS: usize = 4096;

/// Rationalize an `f64` exactly.
///
/// Returns the **simplest** rational number (smallest denominator, in lowest
/// terms) that converts back to bit-for-bit the same `f64`. There is no cap on
/// the denominator: if the only rational that round-trips is the exact dyadic
/// value of the float, that value is returned.
///
/// # Algorithm
///
/// The exact value `v = m / 2^k` of the float is expanded as a continued
/// fraction `[a0; a1, a2, …]` using exact rational arithmetic. The convergents
/// `h_i / k_i` are generated by the standard recurrence
///
/// ```text
/// h_i = a_i · h_{i-1} + h_{i-2},    h_{-1} = 1, h_{-2} = 0
/// k_i = a_i · k_{i-1} + k_{i-2},    k_{-1} = 0, k_{-2} = 1
/// ```
///
/// and the first convergent whose `f64` rounding equals `v` is returned. By the
/// classical best-approximation property of continued fractions, that convergent
/// is the unique rational with the smallest denominator that rounds to `v`
/// (equivalently: the first node of the Stern–Brocot tree that lands inside the
/// rounding interval of `v`). The expansion is finite because `v` is rational,
/// so the loop always terminates — in the worst case at `v` itself.
///
/// # Errors
///
/// Returns [`PolyError::NotPolynomial`] when `v` is `NaN` or infinite: those have
/// no rational value at all.
pub fn f64_to_ratio(v: f64) -> Result<Coeff, PolyError> {
    if !v.is_finite() {
        return Err(PolyError::NotPolynomial);
    }
    // Exact dyadic value of the float. `from_float` only fails on non-finite
    // input, which we already rejected.
    let exact = Coeff::from_float(v).ok_or(PolyError::NotPolynomial)?;

    // Convergent recurrence state: (h_{i-2}, h_{i-1}) and (k_{i-2}, k_{i-1}).
    let mut h_prev = BigInt::from(0i32);
    let mut h_cur = BigInt::from(1i32);
    let mut k_prev = BigInt::from(1i32);
    let mut k_cur = BigInt::from(0i32);

    let mut x = exact.clone();

    for _ in 0..MAX_CF_TERMS {
        let a: BigInt = x.floor().to_integer();

        let h_next = &a * &h_cur + &h_prev;
        let k_next = &a * &k_cur + &k_prev;
        h_prev = std::mem::replace(&mut h_cur, h_next);
        k_prev = std::mem::replace(&mut k_cur, k_next);

        // k_0 = a0·0 + 1 = 1, and every later a_i >= 1 with k_{i-1} >= 1, so
        // k_i is always strictly positive: `Coeff::new` cannot divide by zero.
        let candidate = Coeff::new(h_cur.clone(), k_cur.clone());
        if ratio_to_f64(&candidate) == v {
            return Ok(candidate);
        }

        let frac = &x - Coeff::from_integer(a);
        match coeff_recip(&frac) {
            // The continued fraction terminated: `candidate` equals `exact`.
            None => break,
            Some(next) => x = next,
        }
    }

    Ok(exact)
}

// ── ℚ → f64 (correctly rounded) ───────────────────────────────────────────────

/// `f64::MANTISSA_DIGITS` as a signed integer.
const F64_MANTISSA_DIGITS: i64 = 53;
/// `f64::MIN_EXP` as a signed integer.
const F64_MIN_EXP: i64 = -1021;
/// `f64::MAX_EXP` as a signed integer.
const F64_MAX_EXP: i64 = 1024;

/// Convert an exact rational to the nearest `f64` (round-half-to-even).
///
/// Unlike `numer as f64 / denom as f64`, this is correct for coefficients of any
/// magnitude: a ratio of two 5000-bit integers whose *quotient* is `1.5` yields
/// exactly `1.5`, rather than `inf / inf = NaN`. Values too large for `f64`
/// saturate to ±∞ and values too small underflow through the subnormal range to
/// ±0, exactly as IEEE-754 arithmetic would.
///
/// # Algorithm
///
/// Let `v = n / d` with `n, d > 0`. Using `bits(n) - bits(d)` we know
/// `2^(diff-1) < v < 2^(diff+1)`, so shifting by
/// `shift = max(diff, MIN_EXP) - 53 - 2` makes the integer quotient
/// `q = ⌊(n · 2^-shift) / d⌋` land in `[2^54, 2^56)` — i.e. 53 mantissa bits plus
/// 2 or 3 rounding bits (fewer when the result is subnormal). The division's
/// remainder supplies the sticky bit, so the round-half-to-even decision is made
/// with full information and the final scaling by `2^shift` is exact.
#[must_use]
pub fn ratio_to_f64(r: &Coeff) -> f64 {
    let numer = r.numer();
    let sign = match numer.sign() {
        Sign::NoSign => return 0.0,
        Sign::Minus => -1.0f64,
        Sign::Plus => 1.0f64,
    };

    // A reduced `Ratio` always keeps its denominator strictly positive.
    let mut n: BigUint = numer.magnitude().clone();
    let mut d: BigUint = r.denom().magnitude().clone();
    if d.bits() == 0 {
        // Only reachable through `Ratio::new_raw` misuse; a zero denominator has
        // no finite value.
        return f64::INFINITY * sign;
    }

    let diff = n.bits() as i64 - d.bits() as i64;
    if diff > F64_MAX_EXP {
        return f64::INFINITY * sign;
    }
    if diff < F64_MIN_EXP - F64_MANTISSA_DIGITS - 1 {
        return 0.0 * sign;
    }

    let shift = diff.max(F64_MIN_EXP) - F64_MANTISSA_DIGITS - 2;
    if shift >= 0 {
        d <<= shift as usize;
    } else {
        n <<= (-shift) as usize;
    }

    let (quotient_big, remainder) = n.div_rem(&d);
    // By construction `quotient_big` has at most 56 bits, so it fits in a u64.
    debug_assert!(quotient_big.bits() <= 56);
    let mut quotient: u64 = quotient_big.iter_u64_digits().next().unwrap_or(0);

    // Number of low bits of `quotient` that are *not* part of the 53-bit
    // mantissa. In the subnormal range the mantissa is shorter, which the
    // `subnormal_bits` term accounts for.
    let quotient_bits = i64::from(64 - quotient.leading_zeros());
    let subnormal_bits = F64_MIN_EXP - shift;
    let n_rounding_bits = (quotient_bits.max(subnormal_bits) - F64_MANTISSA_DIGITS).max(0);

    if n_rounding_bits > 0 {
        let n_rounding_bits = n_rounding_bits as u32;
        let mask = (1u64 << n_rounding_bits) - 1;
        let ls_bit = quotient & (1u64 << n_rounding_bits) != 0;
        let ms_rounding_bit = quotient & (1u64 << (n_rounding_bits - 1)) != 0;
        let ls_rounding_bits = quotient & (mask >> 1) != 0;
        // Round half to even, with the division remainder acting as sticky bit.
        if ms_rounding_bit && (ls_bit || ls_rounding_bits || remainder.bits() != 0) {
            quotient += 1u64 << n_rounding_bits;
        }
        quotient &= !mask;
    }

    // `quotient` is now 53 significant bits followed by trailing zeros, so the
    // cast is exact and `ldexp` introduces at most the single unavoidable
    // rounding of an underflow into the subnormal range.
    ldexp(quotient as f64 * sign, shift)
}

/// Multiply `x` by `2^exp` without ever forming `2^exp` as a single `f64`.
///
/// The exponent is applied in chunks of 512 (a power of two that is always
/// representable), so intermediate products stay exact until the final step,
/// which may legitimately overflow to ±∞ or underflow into the subnormal range.
fn ldexp(x: f64, exp: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    const CHUNK: i64 = 512;
    let up = 2.0f64.powi(CHUNK as i32);
    let down = 2.0f64.powi(-(CHUNK as i32));

    let mut y = x;
    let mut e = exp;
    while e > CHUNK {
        y *= up;
        e -= CHUNK;
        if !y.is_finite() {
            return y;
        }
    }
    while e < -CHUNK {
        y *= down;
        e += CHUNK;
        if y == 0.0 {
            return y;
        }
    }
    y * 2.0f64.powi(e as i32)
}

// ── Exact integer number theory (for the rational root theorem) ────────────────

/// Maximum number of positive divisors enumerated for a single integer.
const MAX_DIVISORS: usize = 1024;

/// Maximum number of `(p, q)` candidates the rational root theorem may inspect.
const MAX_ROOT_CANDIDATES: usize = 1_000_000;

/// Total number of Pollard–Brent iterations allowed per factorization.
const POLLARD_BUDGET: u64 = 4_000_000;

/// Odd trial-division bound. Anything above this is left to Pollard–Brent.
const TRIAL_DIVISION_BOUND: u32 = 1000;

/// Miller–Rabin bases.
///
/// The first 13 primes make the test *deterministic* for every `n < 3.317·10^24`
/// (well past `u64`). Beyond that bound this is a strong probable-prime test with
/// the standard (astronomically small) error profile; a false "prime" verdict
/// could only cause a rational root to be missed, never a wrong root reported.
const MILLER_RABIN_BASES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Strong probable-prime (Miller–Rabin) test with fixed bases.
fn is_probable_prime(n: &BigUint) -> bool {
    let one = BigUint::from(1u32);
    let two = BigUint::from(2u32);
    if *n < two {
        return false;
    }
    for &base in &MILLER_RABIN_BASES {
        let p = BigUint::from(base);
        if *n == p {
            return true;
        }
        if (n % &p).bits() == 0 {
            return false;
        }
    }

    // n - 1 = d · 2^s with d odd.
    let n_minus_one = n - &one;
    let s = n_minus_one.trailing_zeros().unwrap_or(0);
    let d = &n_minus_one >> s;

    'bases: for &base in &MILLER_RABIN_BASES {
        let a = BigUint::from(base);
        let mut x = a.modpow(&d, n);
        if x == one || x == n_minus_one {
            continue;
        }
        for _ in 1..s {
            x = x.modpow(&two, n);
            if x == n_minus_one {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// Pollard's rho with Brent's cycle detection and batched GCDs.
///
/// Returns a non-trivial factor of the odd composite `n`, or `None` when the
/// iteration budget is exhausted. Deterministic: the polynomial constant `c` is
/// walked upward from 1 instead of being drawn at random.
fn pollard_brent(n: &BigUint, budget: &mut u64) -> Option<BigUint> {
    let one = BigUint::from(1u32);
    let two = BigUint::from(2u32);
    if (n % &two).bits() == 0 {
        return Some(two);
    }

    // Batch size for the accumulated-product GCD trick.
    const BATCH: u64 = 128;

    for c_seed in 1u32..=64 {
        let c = BigUint::from(c_seed);
        let mut y = BigUint::from(2u32);
        let mut g = one.clone();
        let mut r: u64 = 1;
        let mut q = one.clone();
        let mut x = BigUint::from(0u32);
        let mut ys = BigUint::from(0u32);

        while g == one {
            x = y.clone();
            for _ in 0..r {
                y = (&y * &y + &c) % n;
            }
            let mut k: u64 = 0;
            while k < r && g == one {
                ys = y.clone();
                let steps = BATCH.min(r - k);
                for _ in 0..steps {
                    y = (&y * &y + &c) % n;
                    let diff = if x > y { &x - &y } else { &y - &x };
                    q = (&q * &diff) % n;
                }
                if *budget < steps {
                    return None;
                }
                *budget -= steps;
                g = q.gcd(n);
                k += BATCH;
            }
            r = r.saturating_mul(2);
            if *budget == 0 {
                return None;
            }
        }

        if g == *n {
            // The batch swallowed the factor: replay the last block one step at
            // a time. Bounded by `r` so this cannot spin forever.
            g = one.clone();
            for _ in 0..r {
                ys = (&ys * &ys + &c) % n;
                let diff = if x > ys { &x - &ys } else { &ys - &x };
                if diff.bits() == 0 {
                    break;
                }
                g = diff.gcd(n);
                if g != one {
                    break;
                }
            }
        }

        if g != one && g != *n {
            return Some(g);
        }
        // Degenerate cycle for this `c`; try the next polynomial.
    }
    None
}

/// Complete prime factorization of `n >= 1`.
///
/// Small primes are peeled off by trial division; the remaining cofactor is split
/// recursively with [`pollard_brent`] and classified with [`is_probable_prime`].
///
/// Returns `None` when the Pollard budget is exhausted (a genuinely hard
/// semiprime), never a partial or wrong answer.
fn factorize(n: &BigUint) -> Option<Vec<(BigUint, u32)>> {
    let one = BigUint::from(1u32);
    let mut factors: Vec<(BigUint, u32)> = Vec::new();
    let mut remaining = n.clone();
    if remaining <= one {
        return Some(factors);
    }

    let mut trial = 2u32;
    while trial <= TRIAL_DIVISION_BOUND {
        let p = BigUint::from(trial);
        if &p * &p > remaining {
            break;
        }
        let mut exp = 0u32;
        loop {
            let (quot, rem) = remaining.div_rem(&p);
            if rem.bits() != 0 {
                break;
            }
            remaining = quot;
            exp += 1;
        }
        if exp > 0 {
            factors.push((p, exp));
        }
        trial = if trial == 2 { 3 } else { trial + 2 };
    }

    if remaining > one {
        let mut budget = POLLARD_BUDGET;
        let mut stack = vec![remaining];
        while let Some(m) = stack.pop() {
            if m <= one {
                continue;
            }
            if is_probable_prime(&m) {
                match factors.iter_mut().find(|(p, _)| *p == m) {
                    Some((_, e)) => *e += 1,
                    None => factors.push((m, 1)),
                }
                continue;
            }
            let divisor = pollard_brent(&m, &mut budget)?;
            let cofactor = &m / &divisor;
            stack.push(divisor);
            stack.push(cofactor);
        }
    }

    factors.sort_by(|a, b| a.0.cmp(&b.0));
    Some(factors)
}

/// All positive divisors of `n >= 1`, or `None` when the factorization fails or
/// there are more than [`MAX_DIVISORS`] of them.
fn positive_divisors(n: &BigUint) -> Option<Vec<BigUint>> {
    let factors = factorize(n)?;
    let mut divisors = vec![BigUint::from(1u32)];
    for (prime, exp) in factors {
        let base_len = divisors.len();
        let mut next = Vec::with_capacity(base_len * (exp as usize + 1));
        let mut power = BigUint::from(1u32);
        for _ in 0..=exp {
            for d in &divisors[..base_len] {
                next.push(d * &power);
            }
            power *= &prime;
        }
        if next.len() > MAX_DIVISORS {
            return None;
        }
        divisors = next;
    }
    Some(divisors)
}

/// Every integer divisor of `n`, positive and negative.
///
/// Mirrors the classical `integer_factors` helper but is exact for arbitrary
/// precision integers. `n == 0` yields `[0]` (every integer divides zero; the
/// callers treat that as "no useful constraint").
///
/// # Errors
///
/// Returns [`PolyError::FactorizationLimit`] if `|n|` cannot be factored within
/// the search budget or has more than [`MAX_DIVISORS`] positive divisors.
pub(crate) fn integer_divisors(n: &BigInt) -> Result<Vec<BigInt>, PolyError> {
    if n.sign() == Sign::NoSign {
        return Ok(vec![BigInt::from(0i32)]);
    }
    let magnitude = positive_divisors(n.magnitude()).ok_or(PolyError::FactorizationLimit)?;
    let mut out = Vec::with_capacity(magnitude.len() * 2);
    for d in magnitude {
        let positive = BigInt::from_biguint(Sign::Plus, d);
        let negative = -positive.clone();
        out.push(positive);
        out.push(negative);
    }
    Ok(out)
}

/// Positive divisors of a non-zero `n`, as `BigInt`s.
///
/// Used for the denominator side `q` of a rational root `p/q`: the sign is
/// carried by `p` alone, so only positive `q` need be enumerated.
///
/// # Errors
///
/// Returns [`PolyError::DivByZero`] when `n == 0` (a normalized polynomial never
/// has a zero leading coefficient), and [`PolyError::FactorizationLimit`] under
/// the same conditions as [`integer_divisors`].
pub(crate) fn positive_integer_divisors(n: &BigInt) -> Result<Vec<BigInt>, PolyError> {
    if n.sign() == Sign::NoSign {
        return Err(PolyError::DivByZero);
    }
    let magnitude = positive_divisors(n.magnitude()).ok_or(PolyError::FactorizationLimit)?;
    Ok(magnitude
        .into_iter()
        .map(|d| BigInt::from_biguint(Sign::Plus, d))
        .collect())
}

/// Guard used by [`Poly::rational_roots`] before enumerating `p/q` candidates.
pub(crate) fn check_candidate_budget(a: usize, b: usize) -> Result<(), PolyError> {
    if a.saturating_mul(b) > MAX_ROOT_CANDIDATES {
        return Err(PolyError::FactorizationLimit);
    }
    Ok(())
}
