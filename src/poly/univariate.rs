//! Dense univariate polynomial over exact, arbitrary-precision rational coefficients.

use std::collections::BTreeSet;
use std::sync::Arc;

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;

use crate::lower::LoweredOp;

use super::{
    Coeff, PolyError, check_candidate_budget, coeff_is_zero, coeff_one, coeff_recip, coeff_zero,
    f64_to_ratio, integer_divisors, positive_integer_divisors, ratio_to_f64,
};

/// Dense univariate polynomial over exact rational [`Coeff`] coefficients.
///
/// Coefficients are stored in ascending degree order: `coeffs[i]` is the
/// coefficient of `x^i`. The zero polynomial is represented by an empty vector
/// or a vector of all-zero rationals.
///
/// Because [`Coeff`] is arbitrary precision, none of the operations on this type
/// can overflow; the only failure modes are division by the zero polynomial and
/// the integer-factorization budget used by [`Poly::rational_roots`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    /// Coefficients in ascending degree order (`coeffs[i]` = coeff of `x^i`).
    pub coeffs: Vec<Coeff>,
}

impl Poly {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create the zero polynomial.
    #[must_use]
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// Create the constant polynomial with value `c`.
    #[must_use]
    pub fn constant(c: Coeff) -> Self {
        if coeff_is_zero(&c) {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// Create the monic monomial `x^degree`.
    #[must_use]
    pub fn monomial(degree: usize) -> Self {
        let mut coeffs = vec![coeff_zero(); degree + 1];
        coeffs[degree] = coeff_one();
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Build a polynomial from machine-integer coefficients in ascending degree
    /// order.
    ///
    /// ```
    /// use oxieml::Poly;
    /// // 1 + 2x + 3x²
    /// let p = Poly::from_int_coeffs(&[1, 2, 3]);
    /// assert_eq!(p.degree(), Some(2));
    /// ```
    #[must_use]
    pub fn from_int_coeffs(coeffs: &[i64]) -> Self {
        let coeffs = coeffs
            .iter()
            .map(|&n| Coeff::new_raw(BigInt::from(n), BigInt::from(1i32)))
            .collect();
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Build a polynomial from exact rational coefficients in ascending degree
    /// order.
    #[must_use]
    pub fn from_ratios(coeffs: Vec<Coeff>) -> Self {
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Return the degree of this polynomial (`None` for the zero polynomial).
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        let mut last = self.coeffs.len();
        while last > 0 && coeff_is_zero(&self.coeffs[last - 1]) {
            last -= 1;
        }
        if last == 0 { None } else { Some(last - 1) }
    }

    /// Return `true` if this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(coeff_is_zero)
    }

    /// Return the leading coefficient, or 0 for the zero polynomial.
    #[must_use]
    pub fn leading_coeff(&self) -> Coeff {
        match self.degree() {
            Some(d) => self.coeffs[d].clone(),
            None => coeff_zero(),
        }
    }

    /// Remove trailing zero coefficients (high-degree zeros).
    pub fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(coeff_is_zero) {
            self.coeffs.pop();
        }
    }

    /// Return a normalized clone.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut p = self.clone();
        p.normalize();
        p
    }

    // ── Conversion from/to LoweredOp ──────────────────────────────────────────

    /// Try to convert a `LoweredOp` expression tree to a univariate polynomial
    /// in variable `wrt`.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::NotPolynomial`] when the expression contains a
    /// transcendental node, a different variable, a non-finite constant, or a
    /// power that is not a small non-negative integer.
    pub fn from_lowered(expr: &LoweredOp, wrt: usize) -> Result<Self, PolyError> {
        match expr {
            LoweredOp::Const(c) => Ok(Self::constant(f64_to_ratio(*c)?)),
            LoweredOp::NamedConst(nc) => Ok(Self::constant(f64_to_ratio(nc.value())?)),
            LoweredOp::Var(i) => {
                if *i == wrt {
                    Ok(Self {
                        coeffs: vec![coeff_zero(), coeff_one()],
                    })
                } else {
                    Err(PolyError::NotPolynomial)
                }
            }
            LoweredOp::Add(a, b) => Self::from_lowered(a, wrt)?.add(&Self::from_lowered(b, wrt)?),
            LoweredOp::Sub(a, b) => Self::from_lowered(a, wrt)?.sub(&Self::from_lowered(b, wrt)?),
            LoweredOp::Mul(a, b) => Self::from_lowered(a, wrt)?.mul(&Self::from_lowered(b, wrt)?),
            LoweredOp::Neg(a) => Self::from_lowered(a, wrt)?.neg(),
            LoweredOp::Pow(base, exp) => {
                if let LoweredOp::Const(e) = exp.as_ref() {
                    let n = *e;
                    if n < 0.0 || n.fract() != 0.0 || n > 200.0 {
                        return Err(PolyError::NotPolynomial);
                    }
                    Self::from_lowered(base, wrt)?.pow(n as usize)
                } else {
                    Err(PolyError::NotPolynomial)
                }
            }
            _ => Err(PolyError::NotPolynomial),
        }
    }

    /// Convert this polynomial back to a `LoweredOp` expression tree using
    /// variable index `wrt` (Horner form).
    #[must_use]
    pub fn to_lowered(&self, wrt: usize) -> LoweredOp {
        let norm = self.normalized();
        if norm.coeffs.is_empty() {
            return LoweredOp::Const(0.0);
        }
        let x = Arc::new(LoweredOp::Var(wrt));

        let n = norm.coeffs.len() - 1;
        let mut acc = LoweredOp::Const(ratio_to_f64(&norm.coeffs[n]));

        for i in (0..n).rev() {
            let c = ratio_to_f64(&norm.coeffs[i]);
            acc = LoweredOp::Add(
                Arc::new(LoweredOp::Const(c)),
                Arc::new(LoweredOp::Mul(x.clone(), Arc::new(acc))),
            );
        }
        acc
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Add two polynomials.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let zero = coeff_zero();
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).unwrap_or(&zero);
            let b = other.coeffs.get(i).unwrap_or(&zero);
            coeffs.push(a + b);
        }
        Ok(Self::from_ratios(coeffs))
    }

    /// Subtract `other` from `self`.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let zero = coeff_zero();
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).unwrap_or(&zero);
            let b = other.coeffs.get(i).unwrap_or(&zero);
            coeffs.push(a - b);
        }
        Ok(Self::from_ratios(coeffs))
    }

    /// Multiply two polynomials (schoolbook convolution, exact).
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        let a = self.normalized();
        let b = other.normalized();
        if a.coeffs.is_empty() || b.coeffs.is_empty() {
            return Ok(Self::zero());
        }
        let n = a.coeffs.len();
        let m = b.coeffs.len();
        let mut coeffs = vec![coeff_zero(); n + m - 1];
        for (i, ca) in a.coeffs.iter().enumerate() {
            if coeff_is_zero(ca) {
                continue;
            }
            for (j, cb) in b.coeffs.iter().enumerate() {
                if coeff_is_zero(cb) {
                    continue;
                }
                coeffs[i + j] = &coeffs[i + j] + &(ca * cb);
            }
        }
        Ok(Self::from_ratios(coeffs))
    }

    /// Scale this polynomial by a rational constant.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn scale(&self, c: &Coeff) -> Result<Self, PolyError> {
        if coeff_is_zero(c) {
            return Ok(Self::zero());
        }
        let coeffs = self.coeffs.iter().map(|a| a * c).collect();
        Ok(Self::from_ratios(coeffs))
    }

    /// Negate this polynomial.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn neg(&self) -> Result<Self, PolyError> {
        Ok(Self {
            coeffs: self.coeffs.iter().map(|a| -a).collect(),
        })
    }

    /// Raise this polynomial to a non-negative integer power by binary
    /// exponentiation.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn pow(&self, n: usize) -> Result<Self, PolyError> {
        if n == 0 {
            return Ok(Self::constant(coeff_one()));
        }
        let mut result = Self::constant(coeff_one());
        let mut base = self.normalized();
        let mut exp = n;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base)?;
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    // ── Polynomial division ───────────────────────────────────────────────────

    /// Euclidean polynomial division over ℚ: returns `(quotient, remainder)`
    /// with `self == quotient · divisor + remainder` and `deg(remainder) <
    /// deg(divisor)`.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] when `divisor` is the zero polynomial.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolyError> {
        let divisor = divisor.normalized();
        let div_deg = divisor.degree().ok_or(PolyError::DivByZero)?;

        let self_norm = self.normalized();
        let self_deg = match self_norm.degree() {
            Some(d) => d,
            None => return Ok((Self::zero(), Self::zero())),
        };

        if self_deg < div_deg {
            return Ok((Self::zero(), self_norm));
        }

        let divisor_lc_inv = coeff_recip(&divisor.leading_coeff()).ok_or(PolyError::DivByZero)?;

        let mut remainder = self_norm.coeffs;
        let mut quotient = vec![coeff_zero(); self_deg - div_deg + 1];

        for i in (div_deg..=self_deg).rev() {
            let factor = &remainder[i] * &divisor_lc_inv;
            if coeff_is_zero(&factor) {
                continue;
            }
            let pos = i - div_deg;
            for j in 0..=div_deg {
                remainder[pos + j] = &remainder[pos + j] - &(&factor * &divisor.coeffs[j]);
            }
            quotient[pos] = factor;
        }

        Ok((Self::from_ratios(quotient), Self::from_ratios(remainder)))
    }

    // ── GCD ───────────────────────────────────────────────────────────────────

    /// Compute the monic GCD of two polynomials with the Euclidean algorithm
    /// over ℚ.
    ///
    /// # Errors
    ///
    /// Infallible in practice: [`PolyError::DivByZero`] can only surface from an
    /// internal division, and the loop never divides by zero.
    pub fn gcd(a: &Self, b: &Self) -> Result<Self, PolyError> {
        let mut u = a.normalized();
        let mut v = b.normalized();

        while !v.is_zero() {
            let (_, r) = u.div_rem(&v)?;
            u = v;
            v = r;
        }

        if u.is_zero() {
            return Ok(Self::zero());
        }
        let inv = coeff_recip(&u.leading_coeff()).ok_or(PolyError::DivByZero)?;
        u.scale(&inv)
    }

    // ── Differentiation ───────────────────────────────────────────────────────

    /// Compute the formal derivative of this polynomial.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn diff(&self) -> Result<Self, PolyError> {
        if self.coeffs.len() <= 1 {
            return Ok(Self::zero());
        }
        let coeffs = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| c * Coeff::from_integer(BigInt::from(i)))
            .collect();
        Ok(Self::from_ratios(coeffs))
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Evaluate this polynomial at an exact rational point (Horner).
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn eval(&self, x: &Coeff) -> Result<Coeff, PolyError> {
        let mut acc = coeff_zero();
        for c in self.coeffs.iter().rev() {
            acc = &(&acc * x) + c;
        }
        Ok(acc)
    }

    /// Evaluate this polynomial at an `f64` point (Horner).
    #[must_use]
    pub fn eval_f64(&self, x: f64) -> f64 {
        let norm = self.normalized();
        if norm.coeffs.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for c in norm.coeffs.iter().rev() {
            acc = acc * x + ratio_to_f64(c);
        }
        acc
    }

    // ── Square-free part ──────────────────────────────────────────────────────

    /// Return the (monic) square-free part `f / gcd(f, f')` of this polynomial.
    ///
    /// # Errors
    ///
    /// Propagates division errors, which cannot occur for a non-zero input.
    pub fn square_free(&self) -> Result<Self, PolyError> {
        let f = self.normalized();
        if f.is_zero() {
            return Ok(Self::zero());
        }
        let df = f.diff()?;
        let g = Self::gcd(&f, &df)?;
        if g.is_zero() || g.degree() == Some(0) {
            return Ok(f);
        }
        let (q, r) = f.div_rem(&g)?;
        if !r.is_zero() {
            return Ok(f);
        }
        if q.is_zero() {
            return Ok(q);
        }
        let inv = coeff_recip(&q.leading_coeff()).ok_or(PolyError::DivByZero)?;
        q.scale(&inv)
    }

    // ── Content and primitive part ────────────────────────────────────────────

    /// Compute the content of this polynomial: the unique positive rational `c`
    /// such that `self = c · pp` where `pp` is a primitive integer polynomial.
    ///
    /// Concretely `c = gcd_i(n_i · L / d_i) / L` where `L = lcm_i(d_i)` and
    /// `c_i = n_i / d_i`. The zero polynomial has content `0`.
    #[must_use]
    pub fn content(&self) -> Coeff {
        let norm = self.normalized();
        if norm.coeffs.is_empty() {
            return coeff_zero();
        }

        let mut lcm_denom = BigInt::from(1i32);
        for c in &norm.coeffs {
            lcm_denom = lcm_denom.lcm(c.denom());
        }
        let mut gcd_numer = BigInt::from(0i32);
        for c in &norm.coeffs {
            let scaled = c.numer() * (&lcm_denom / c.denom());
            gcd_numer = gcd_numer.gcd(&scaled);
        }
        if gcd_numer.sign() == Sign::NoSign {
            return coeff_zero();
        }
        Coeff::new(gcd_numer, lcm_denom)
    }

    /// Return the primitive part of this polynomial: an integer polynomial with
    /// coefficient GCD 1 and a positive leading coefficient.
    ///
    /// # Errors
    ///
    /// Infallible for well-formed input; the `Result` is kept for API stability.
    pub fn primitive_part(&self) -> Result<Self, PolyError> {
        let norm = self.normalized();
        if norm.is_zero() {
            return Ok(Self::zero());
        }
        let content = norm.content();
        let inv = match coeff_recip(&content) {
            Some(inv) => inv,
            None => return Ok(norm),
        };
        let mut result = norm.scale(&inv)?;
        if result.leading_coeff() < coeff_zero() {
            result = result.neg()?;
        }
        Ok(result)
    }

    /// Clear denominators: return the integer coefficient vector of the
    /// primitive part (ascending degree), together with nothing else.
    ///
    /// The result is the unique primitive integer polynomial proportional to
    /// `self` with a positive leading coefficient. Returns an empty vector for
    /// the zero polynomial.
    fn integer_primitive_coeffs(&self) -> Vec<BigInt> {
        let norm = self.normalized();
        if norm.coeffs.is_empty() {
            return Vec::new();
        }

        let mut lcm_denom = BigInt::from(1i32);
        for c in &norm.coeffs {
            lcm_denom = lcm_denom.lcm(c.denom());
        }
        let mut ints: Vec<BigInt> = norm
            .coeffs
            .iter()
            .map(|c| c.numer() * (&lcm_denom / c.denom()))
            .collect();

        let mut g = BigInt::from(0i32);
        for c in &ints {
            g = g.gcd(c);
        }
        if g.sign() != Sign::NoSign {
            for c in &mut ints {
                *c /= &g;
            }
        }
        if ints.last().is_some_and(|c| c.sign() == Sign::Minus) {
            for c in &mut ints {
                *c = -(&*c);
            }
        }
        ints
    }

    // ── Rational root theorem ─────────────────────────────────────────────────

    /// Find every rational root of this polynomial, exactly.
    ///
    /// The polynomial is first reduced to its primitive integer form
    /// `a_n x^n + … + a_0`. Every rational root in lowest terms is then `p/q`
    /// with `p | a_0` and `q | a_n`, so the search enumerates the exact integer
    /// divisors of `a_0` and `a_n` (via trial division + Pollard–Brent rho +
    /// Miller–Rabin). Candidates are pre-screened with the classical
    /// divisibility filters `(q − p) | f(1)` and `(p + q) | f(−1)` — both follow
    /// from `f(x) = (q x − p) · g(x)` with `g` integral (Gauss' lemma) — and the
    /// survivors are evaluated exactly with a homogeneous Horner scheme
    /// `Σ a_i p^i q^{n−i}`, so no rounding is involved anywhere.
    ///
    /// Roots are returned sorted ascending, without duplicates.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::FactorizationLimit`] when `|a_0|` or `|a_n|` cannot be
    /// factored within the search budget, or when the candidate grid would be
    /// unreasonably large.
    pub fn rational_roots(&self) -> Result<Vec<Coeff>, PolyError> {
        let mut int_coeffs = self.integer_primitive_coeffs();
        if int_coeffs.is_empty() {
            return Ok(Vec::new());
        }

        let mut roots: BTreeSet<Coeff> = BTreeSet::new();

        // Factor out x^k: a zero constant term means 0 is a root.
        let mut shift = 0usize;
        while shift < int_coeffs.len() && int_coeffs[shift].sign() == Sign::NoSign {
            shift += 1;
        }
        if shift > 0 {
            roots.insert(coeff_zero());
            int_coeffs.drain(..shift);
        }
        if int_coeffs.len() < 2 {
            // Constant (possibly after removing the x^k factor): no more roots.
            return Ok(roots.into_iter().collect());
        }

        let degree = int_coeffs.len() - 1;
        let constant_term = int_coeffs[0].clone();
        let leading = int_coeffs[degree].clone();

        let p_divisors = integer_divisors(&constant_term)?;
        let q_divisors = positive_integer_divisors(&leading)?;
        check_candidate_budget(p_divisors.len(), q_divisors.len())?;

        // f(1) = Σ a_i and f(−1) = Σ (−1)^i a_i, used as cheap exact filters.
        let mut f_at_one = BigInt::from(0i32);
        let mut f_at_minus_one = BigInt::from(0i32);
        for (i, a) in int_coeffs.iter().enumerate() {
            f_at_one += a;
            if i.is_multiple_of(2) {
                f_at_minus_one += a;
            } else {
                f_at_minus_one -= a;
            }
        }

        let one_unsigned = BigUint::from(1u32);

        // Powers of q are needed by the homogeneous Horner evaluation.
        for q in &q_divisors {
            let mut q_powers: Vec<BigInt> = Vec::with_capacity(degree + 1);
            let mut acc = BigInt::from(1i32);
            for _ in 0..=degree {
                q_powers.push(acc.clone());
                acc *= q;
            }

            for p in &p_divisors {
                if p.magnitude().gcd(q.magnitude()) != one_unsigned {
                    // Not in lowest terms; the reduced form of this candidate is
                    // visited elsewhere in the grid.
                    continue;
                }

                // (q − p) | f(1)
                let q_minus_p = q - p;
                if f_at_one.sign() != Sign::NoSign {
                    if q_minus_p.sign() == Sign::NoSign {
                        continue;
                    }
                    if (&f_at_one % &q_minus_p).sign() != Sign::NoSign {
                        continue;
                    }
                }
                // (q + p) | f(−1)
                let q_plus_p = q + p;
                if f_at_minus_one.sign() != Sign::NoSign {
                    if q_plus_p.sign() == Sign::NoSign {
                        continue;
                    }
                    if (&f_at_minus_one % &q_plus_p).sign() != Sign::NoSign {
                        continue;
                    }
                }

                // Exact homogeneous Horner: Σ a_i p^i q^(n−i).
                let mut acc = int_coeffs[degree].clone();
                for i in (0..degree).rev() {
                    acc = &acc * p + &int_coeffs[i] * &q_powers[degree - i];
                }
                if acc.sign() == Sign::NoSign {
                    roots.insert(Coeff::new(p.clone(), q.clone()));
                }
            }
        }

        Ok(roots.into_iter().collect())
    }

    // ── Resultant and discriminant ────────────────────────────────────────────

    /// Compute the resultant of two polynomials.
    ///
    /// Uses the Euclidean recursion
    ///
    /// ```text
    /// res(a, b) = (−1)^(deg a · deg b) · lc(b)^(deg a − deg r) · res(b, r),   r = a mod b
    /// res(c, b) = c^(deg b)          (deg c = 0)
    /// res(a, c) = c^(deg a)          (deg c = 0)
    /// ```
    ///
    /// which is exact over ℚ. Because [`Coeff`] is arbitrary precision, the
    /// coefficient growth of the rational remainder sequence is a performance
    /// concern only — the value returned is always exact. `res(a, b) = 0` exactly
    /// when `a` and `b` share a non-constant factor.
    ///
    /// # Errors
    ///
    /// Propagates division errors, which cannot occur for non-zero inputs.
    pub fn resultant(a: &Self, b: &Self) -> Result<Coeff, PolyError> {
        let a = a.normalized();
        let b = b.normalized();

        if a.is_zero() || b.is_zero() {
            return Ok(coeff_zero());
        }

        let da = a.degree().unwrap_or(0);
        let db = b.degree().unwrap_or(0);

        if da == 0 {
            return Ok(pow_coeff(&a.leading_coeff(), db));
        }
        if db == 0 {
            return Ok(pow_coeff(&b.leading_coeff(), da));
        }

        let sign = if (da * db).is_multiple_of(2) {
            coeff_one()
        } else {
            -coeff_one()
        };

        if da < db {
            let sub = Self::resultant(&b, &a)?;
            return Ok(&sign * &sub);
        }

        let (_, rem) = a.div_rem(&b)?;
        if rem.is_zero() {
            return Ok(coeff_zero());
        }

        let dr = rem.degree().unwrap_or(0);
        let lc_pow = pow_coeff(&b.leading_coeff(), da - dr);
        let sub = Self::resultant(&b, &rem)?;

        Ok(&(&sign * &lc_pow) * &sub)
    }

    /// Compute the discriminant `(−1)^(n(n−1)/2) · res(f, f′) / lc(f)`.
    ///
    /// Exact and arbitrary precision. For a monic quadratic `x² + b x + c` this
    /// is exactly `b² − 4c`, for any `b`, `c` in ℚ.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] for the zero polynomial's leading
    /// coefficient (unreachable: the zero polynomial short-circuits to `0`).
    pub fn discriminant(&self) -> Result<Coeff, PolyError> {
        let f = self.normalized();
        if f.is_zero() {
            return Ok(coeff_zero());
        }
        let n = f.degree().unwrap_or(0);
        if n == 0 {
            return Ok(coeff_one());
        }

        let df = f.diff()?;
        let res = Self::resultant(&f, &df)?;

        let exponent = n * (n - 1) / 2;
        let sign = if exponent.is_multiple_of(2) {
            coeff_one()
        } else {
            -coeff_one()
        };

        let lc_inv = coeff_recip(&f.leading_coeff()).ok_or(PolyError::DivByZero)?;
        Ok(&(&sign * &res) * &lc_inv)
    }
}

/// Exact integer power of a rational coefficient.
fn pow_coeff(base: &Coeff, exp: usize) -> Coeff {
    let mut result = coeff_one();
    let mut acc = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = &result * &acc;
        }
        e >>= 1;
        if e > 0 {
            acc = &acc * &acc;
        }
    }
    result
}
