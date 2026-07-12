//! Modular polynomial arithmetic for the Zassenhaus factorization pipeline.
//!
//! Two arithmetic layers live here, deliberately kept apart because they have
//! very different overflow profiles.
//!
//! # Layer 1 — [`ModPoly`]: `GF(p)[x]` with `i128` coefficients
//!
//! Used by prime selection, distinct-degree factorization and the
//! Cantor–Zassenhaus equal-degree split. The prime is always a *small* odd
//! prime bounded by [`MAX_MODULUS`] (`2^20`), so every intermediate product of
//! two residues is `< 2^40` and every accumulated convolution sum stays far
//! below `i128::MAX`. Machine-word arithmetic here is provably overflow-free,
//! and it is where the vast majority of the running time is spent.
//!
//! # Layer 2 — [`MpPoly`]: `(ℤ/mℤ)[x]` with [`BigInt`] coefficients
//!
//! Used by Hensel lifting, where `m = p^k` must exceed twice the Landau–Mignotte
//! bound `2^n · ‖f‖₂` of the input. For a degree-40 polynomial with 64-bit
//! coefficients that bound already needs ~110 bits, and the Hensel step forms
//! products of two residues, i.e. ~220 bits — well past `i128`. The task
//! specification suggested `i128` for the whole of `ModPoly`; **using it for the
//! lifted modulus would silently overflow and return a wrong factorization**, so
//! the lifted layer is arbitrary precision instead. Correctness wins over the
//! letter of the spec; the `i128` layer is retained exactly where it is safe.
//!
//! # Determinism
//!
//! Cantor–Zassenhaus is a randomized algorithm. Rather than pull in an external
//! RNG crate, [`Xorshift64`] provides a small seeded xorshift generator. The
//! factorization code seeds it from the polynomial being split, so the whole
//! pipeline is a deterministic function of its input and every test is
//! reproducible bit for bit.

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;

use super::PolyError;

/// Exclusive upper bound on the prime modulus accepted by [`ModPoly`].
///
/// Keeping `p < 2^20` guarantees `p · p < 2^40`, so a convolution of two
/// degree-`d` polynomials accumulates at most `(d + 1) · 2^40` in an `i128`;
/// even for `d = 2^80` that cannot overflow.
pub const MAX_MODULUS: i128 = 1 << 20;

// ── Small-integer modular arithmetic ──────────────────────────────────────────

/// Reduce `a` into `[0, p)`.
#[inline]
fn norm_i128(a: i128, p: i128) -> i128 {
    let r = a % p;
    if r < 0 { r + p } else { r }
}

/// Modular inverse of `a` in `GF(p)`, or `None` when `p | a`.
///
/// Extended Euclid on machine integers; `p` is prime, so every non-zero residue
/// is invertible.
#[must_use]
pub fn mod_inverse(a: i128, p: i128) -> Option<i128> {
    let a = norm_i128(a, p);
    if a == 0 {
        return None;
    }
    let (mut old_r, mut r) = (a, p);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let new_r = old_r - q * r;
        old_r = r;
        r = new_r;
        let new_s = old_s - q * s;
        old_s = s;
        s = new_s;
    }
    if old_r != 1 {
        return None;
    }
    Some(norm_i128(old_s, p))
}

// ── Deterministic PRNG ────────────────────────────────────────────────────────

/// A seeded xorshift64 pseudo-random generator.
///
/// Marsaglia's `xorshift64` with the classical `(13, 7, 17)` triple; it has full
/// period `2^64 − 1` over non-zero states. It is *not* cryptographic — it only
/// has to supply the random field elements that Cantor–Zassenhaus needs, where
/// any generator whose output is not correlated with the polynomial's factor
/// structure gives the standard `≥ 1/2` split probability per attempt.
#[derive(Clone, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a generator from `seed` (a zero seed is remapped, since the
    /// zero state is the generator's fixed point).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Draw the next 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Draw a residue in `[0, p)`.
    ///
    /// Uses a plain modulo, which is very slightly biased towards small
    /// residues (by a factor `1 + p / 2^64 < 1 + 2^-44` for `p < MAX_MODULUS`).
    /// That bias is irrelevant here: Cantor–Zassenhaus only needs the drawn
    /// polynomial to be *unpredictable relative to the factorization*, not
    /// exactly uniform.
    pub fn next_residue(&mut self, p: i128) -> i128 {
        let p_u64 = p.unsigned_abs() as u64;
        i128::from(self.next_u64() % p_u64)
    }
}

// ── GF(p)[x] ──────────────────────────────────────────────────────────────────

/// A dense univariate polynomial over `GF(p)` for a small odd prime `p`.
///
/// Coefficients are stored in ascending degree order, each reduced into
/// `[0, p)`, with all high-order zeros stripped: the zero polynomial has an
/// empty coefficient vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModPoly {
    coeffs: Vec<i128>,
    modulus: i128,
}

impl ModPoly {
    /// Build a polynomial from raw (possibly unreduced, possibly negative)
    /// coefficients in ascending degree order.
    #[must_use]
    pub fn new(coeffs: Vec<i128>, modulus: i128) -> Self {
        let coeffs = coeffs.into_iter().map(|c| norm_i128(c, modulus)).collect();
        let mut p = Self { coeffs, modulus };
        p.normalize();
        p
    }

    /// Reduce an arbitrary-precision integer polynomial modulo `p`.
    #[must_use]
    pub fn from_bigint_coeffs(coeffs: &[BigInt], modulus: i128) -> Self {
        let m = BigInt::from(modulus);
        let reduced = coeffs
            .iter()
            .map(|c| nonneg_bigint_to_i128(&c.mod_floor(&m)))
            .collect();
        Self::new(reduced, modulus)
    }

    /// The zero polynomial over `GF(p)`.
    #[must_use]
    pub fn zero(modulus: i128) -> Self {
        Self {
            coeffs: Vec::new(),
            modulus,
        }
    }

    /// The constant polynomial `1` over `GF(p)`.
    #[must_use]
    pub fn one(modulus: i128) -> Self {
        Self::new(vec![1], modulus)
    }

    /// The polynomial `x` over `GF(p)`.
    #[must_use]
    pub fn identity(modulus: i128) -> Self {
        Self::new(vec![0, 1], modulus)
    }

    /// The prime modulus.
    #[must_use]
    pub fn modulus(&self) -> i128 {
        self.modulus
    }

    /// The coefficients in ascending degree order, each in `[0, p)`.
    #[must_use]
    pub fn coeffs(&self) -> &[i128] {
        &self.coeffs
    }

    /// The degree, or `None` for the zero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// `true` when this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The leading coefficient, or `0` for the zero polynomial.
    #[must_use]
    pub fn leading_coeff(&self) -> i128 {
        self.coeffs.last().copied().unwrap_or(0)
    }

    fn normalize(&mut self) {
        while self.coeffs.last() == Some(&0) {
            self.coeffs.pop();
        }
    }

    /// Add two polynomials over `GF(p)`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            coeffs.push((a + b) % self.modulus);
        }
        let mut p = Self {
            coeffs,
            modulus: self.modulus,
        };
        p.normalize();
        p
    }

    /// Subtract `other` from `self` over `GF(p)`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            coeffs.push(norm_i128(a - b, self.modulus));
        }
        let mut p = Self {
            coeffs,
            modulus: self.modulus,
        };
        p.normalize();
        p
    }

    /// Multiply two polynomials over `GF(p)` (schoolbook convolution).
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.modulus);
        }
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut coeffs = vec![0i128; n + m - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0 {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] = (coeffs[i + j] + a * b) % self.modulus;
            }
        }
        let mut p = Self {
            coeffs,
            modulus: self.modulus,
        };
        p.normalize();
        p
    }

    /// Multiply every coefficient by the scalar `c`.
    #[must_use]
    pub fn scale(&self, c: i128) -> Self {
        let c = norm_i128(c, self.modulus);
        if c == 0 {
            return Self::zero(self.modulus);
        }
        let coeffs = self
            .coeffs
            .iter()
            .map(|&a| (a * c) % self.modulus)
            .collect();
        let mut p = Self {
            coeffs,
            modulus: self.modulus,
        };
        p.normalize();
        p
    }

    /// The formal derivative over `GF(p)`.
    #[must_use]
    pub fn diff(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero(self.modulus);
        }
        let coeffs = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| (c * (i as i128 % self.modulus)) % self.modulus)
            .collect();
        Self::new(coeffs, self.modulus)
    }

    /// Euclidean division: returns `(quotient, remainder)` with
    /// `self == quotient · divisor + remainder` and `deg remainder < deg divisor`.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] when `divisor` is zero (its leading
    /// coefficient is then not invertible).
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolyError> {
        let div_deg = divisor.degree().ok_or(PolyError::DivByZero)?;
        let self_deg = match self.degree() {
            Some(d) => d,
            None => return Ok((Self::zero(self.modulus), Self::zero(self.modulus))),
        };
        if self_deg < div_deg {
            return Ok((Self::zero(self.modulus), self.clone()));
        }

        let p = self.modulus;
        let inv_lc = mod_inverse(divisor.leading_coeff(), p).ok_or(PolyError::DivByZero)?;

        let mut rem = self.coeffs.clone();
        let mut quo = vec![0i128; self_deg - div_deg + 1];

        for i in (div_deg..=self_deg).rev() {
            let factor = (rem[i] * inv_lc) % p;
            if factor == 0 {
                continue;
            }
            let pos = i - div_deg;
            for (j, &d) in divisor.coeffs.iter().enumerate() {
                rem[pos + j] = norm_i128(rem[pos + j] - factor * d % p, p);
            }
            quo[pos] = factor;
        }

        Ok((Self::new(quo, p), Self::new(rem, p)))
    }

    /// The remainder of `self` modulo `divisor`.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] when `divisor` is zero.
    pub fn rem(&self, divisor: &Self) -> Result<Self, PolyError> {
        Ok(self.div_rem(divisor)?.1)
    }

    /// Scale by the inverse of the leading coefficient, making the polynomial
    /// monic. The zero polynomial is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] if the leading coefficient is somehow
    /// not invertible, which cannot happen for a prime modulus.
    pub fn monic(&self) -> Result<Self, PolyError> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        let inv = mod_inverse(self.leading_coeff(), self.modulus).ok_or(PolyError::DivByZero)?;
        Ok(self.scale(inv))
    }

    /// The monic GCD of `a` and `b` over `GF(p)`.
    ///
    /// # Errors
    ///
    /// Propagates division errors, which cannot occur for a prime modulus.
    pub fn gcd(a: &Self, b: &Self) -> Result<Self, PolyError> {
        let mut u = a.clone();
        let mut v = b.clone();
        while !v.is_zero() {
            let r = u.rem(&v)?;
            u = v;
            v = r;
        }
        u.monic()
    }

    /// Extended Euclid: returns `(g, s, t)` with `s·a + t·b = g` and `g` the
    /// monic GCD of `a` and `b`.
    ///
    /// The Euclidean remainder sequence keeps `deg s < deg b − deg g` and
    /// `deg t < deg a − deg g`, which is exactly the degree bound the Hensel
    /// step needs from its Bézout cofactors.
    ///
    /// # Errors
    ///
    /// Propagates division errors, which cannot occur for a prime modulus.
    pub fn xgcd(a: &Self, b: &Self) -> Result<(Self, Self, Self), PolyError> {
        let p = a.modulus;
        let (mut old_r, mut r) = (a.clone(), b.clone());
        let (mut old_s, mut s) = (Self::one(p), Self::zero(p));
        let (mut old_t, mut t) = (Self::zero(p), Self::one(p));

        while !r.is_zero() {
            let (q, rem) = old_r.div_rem(&r)?;
            let new_s = old_s.sub(&q.mul(&s));
            let new_t = old_t.sub(&q.mul(&t));
            old_r = std::mem::replace(&mut r, rem);
            old_s = std::mem::replace(&mut s, new_s);
            old_t = std::mem::replace(&mut t, new_t);
        }

        if old_r.is_zero() {
            return Ok((old_r, old_s, old_t));
        }
        let inv = mod_inverse(old_r.leading_coeff(), p).ok_or(PolyError::DivByZero)?;
        Ok((old_r.scale(inv), old_s.scale(inv), old_t.scale(inv)))
    }

    /// `self^exp mod modulus_poly`, by square-and-multiply over the bits of
    /// `exp` (which routinely has thousands of bits: the Cantor–Zassenhaus
    /// exponent is `(p^d − 1) / 2`).
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] when `modulus_poly` is zero.
    pub fn pow_mod(&self, exp: &BigUint, modulus_poly: &Self) -> Result<Self, PolyError> {
        let p = self.modulus;
        let bits = exp.bits();
        if bits == 0 {
            return Self::one(p).rem(modulus_poly);
        }
        let base = self.rem(modulus_poly)?;
        let mut result = Self::one(p);
        for i in (0..bits).rev() {
            result = result.mul(&result).rem(modulus_poly)?;
            if exp.bit(i) {
                result = result.mul(&base).rem(modulus_poly)?;
            }
        }
        Ok(result)
    }

    /// `true` when `gcd(self, self') = 1`, i.e. `self` is square-free over `GF(p)`.
    ///
    /// # Errors
    ///
    /// Propagates division errors, which cannot occur for a prime modulus.
    pub fn is_square_free(&self) -> Result<bool, PolyError> {
        let d = self.diff();
        if d.is_zero() {
            // f' ≡ 0 with deg f ≥ 1 means f is a p-th power, hence not square-free.
            return Ok(self.degree().unwrap_or(0) == 0);
        }
        Ok(Self::gcd(self, &d)?.degree() == Some(0))
    }
}

// ── (ℤ/mℤ)[x] with arbitrary-precision coefficients ───────────────────────────

/// A dense univariate polynomial over `ℤ/mℤ` with [`BigInt`] coefficients.
///
/// The modulus is always a power `p^k` of the small odd prime chosen by the
/// factorization driver, and can be arbitrarily large — see the module-level
/// note on why this layer is *not* `i128`.
///
/// Coefficients are kept reduced into `[0, m)` and high-order zeros are stripped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MpPoly {
    coeffs: Vec<BigInt>,
    modulus: BigInt,
}

/// Reduce `c` into `[0, m)`.
fn reduce(c: &BigInt, m: &BigInt) -> BigInt {
    c.mod_floor(m)
}

impl MpPoly {
    /// Build a polynomial from raw coefficients, reducing each into `[0, m)`.
    #[must_use]
    pub fn new(coeffs: Vec<BigInt>, modulus: &BigInt) -> Self {
        let coeffs = coeffs.iter().map(|c| reduce(c, modulus)).collect();
        let mut p = Self {
            coeffs,
            modulus: modulus.clone(),
        };
        p.normalize();
        p
    }

    /// The zero polynomial modulo `m`.
    #[must_use]
    pub fn zero(modulus: &BigInt) -> Self {
        Self {
            coeffs: Vec::new(),
            modulus: modulus.clone(),
        }
    }

    /// The constant polynomial `1` modulo `m`.
    #[must_use]
    pub fn one(modulus: &BigInt) -> Self {
        Self::new(vec![BigInt::from(1i32)], modulus)
    }

    /// Lift a `GF(p)` polynomial into `ℤ/mℤ` by taking its `[0, p)`
    /// representatives (which requires `p | m`, always the case here).
    #[must_use]
    pub fn from_mod_poly(f: &ModPoly, modulus: &BigInt) -> Self {
        let coeffs = f.coeffs().iter().map(|&c| BigInt::from(c)).collect();
        Self::new(coeffs, modulus)
    }

    /// Re-interpret this polynomial modulo a larger modulus `m'` with `m | m'`.
    ///
    /// The stored coefficients already lie in `[0, m) ⊆ [0, m')`, so this only
    /// swaps the modulus; it is the "inclusion" map used every time the Hensel
    /// lift doubles its precision.
    #[must_use]
    pub fn with_modulus(&self, modulus: &BigInt) -> Self {
        Self {
            coeffs: self.coeffs.clone(),
            modulus: modulus.clone(),
        }
    }

    /// The modulus `m`.
    #[must_use]
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// The coefficients in ascending degree order, each in `[0, m)`.
    #[must_use]
    pub fn coeffs(&self) -> &[BigInt] {
        &self.coeffs
    }

    /// Consume the polynomial, yielding its `[0, m)` coefficients.
    #[must_use]
    pub fn into_coeffs(self) -> Vec<BigInt> {
        self.coeffs
    }

    /// The degree, or `None` for the zero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// `true` when this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    fn normalize(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.sign() == Sign::NoSign) {
            self.coeffs.pop();
        }
    }

    /// `true` when the leading coefficient is exactly `1`.
    #[must_use]
    pub fn is_monic(&self) -> bool {
        self.coeffs.last() == Some(&BigInt::from(1i32))
    }

    /// Add two polynomials modulo `m`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let zero = BigInt::from(0i32);
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).unwrap_or(&zero);
            let b = other.coeffs.get(i).unwrap_or(&zero);
            coeffs.push(reduce(&(a + b), &self.modulus));
        }
        let mut p = Self {
            coeffs,
            modulus: self.modulus.clone(),
        };
        p.normalize();
        p
    }

    /// Subtract `other` from `self` modulo `m`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let zero = BigInt::from(0i32);
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).unwrap_or(&zero);
            let b = other.coeffs.get(i).unwrap_or(&zero);
            coeffs.push(reduce(&(a - b), &self.modulus));
        }
        let mut p = Self {
            coeffs,
            modulus: self.modulus.clone(),
        };
        p.normalize();
        p
    }

    /// Multiply two polynomials modulo `m` (schoolbook convolution).
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero(&self.modulus);
        }
        let n = self.coeffs.len();
        let k = other.coeffs.len();
        let mut coeffs = vec![BigInt::from(0i32); n + k - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.sign() == Sign::NoSign {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                if b.sign() == Sign::NoSign {
                    continue;
                }
                coeffs[i + j] = reduce(&(&coeffs[i + j] + a * b), &self.modulus);
            }
        }
        let mut p = Self {
            coeffs,
            modulus: self.modulus.clone(),
        };
        p.normalize();
        p
    }

    /// Divide by a **monic** divisor: returns `(quotient, remainder)` with
    /// `self == quotient · divisor + remainder` and `deg remainder < deg divisor`.
    ///
    /// `ℤ/mℤ` is not a field, so division is only well defined when the divisor's
    /// leading coefficient is a unit. Every division performed by the Hensel step
    /// is by a monic polynomial, which is why this is the only division offered.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] when `divisor` is zero or not monic.
    pub fn div_rem_monic(&self, divisor: &Self) -> Result<(Self, Self), PolyError> {
        let div_deg = divisor.degree().ok_or(PolyError::DivByZero)?;
        if !divisor.is_monic() {
            return Err(PolyError::DivByZero);
        }
        let self_deg = match self.degree() {
            Some(d) => d,
            None => return Ok((Self::zero(&self.modulus), Self::zero(&self.modulus))),
        };
        if self_deg < div_deg {
            return Ok((Self::zero(&self.modulus), self.clone()));
        }

        let mut rem = self.coeffs.clone();
        let mut quo = vec![BigInt::from(0i32); self_deg - div_deg + 1];

        for i in (div_deg..=self_deg).rev() {
            let factor = rem[i].clone();
            if factor.sign() == Sign::NoSign {
                continue;
            }
            let pos = i - div_deg;
            for (j, d) in divisor.coeffs.iter().enumerate() {
                rem[pos + j] = reduce(&(&rem[pos + j] - &factor * d), &self.modulus);
            }
            quo[pos] = factor;
        }

        Ok((Self::new(quo, &self.modulus), Self::new(rem, &self.modulus)))
    }

    /// The symmetric representatives of the coefficients, in
    /// `[−(m−1)/2, (m−1)/2]` (the modulus is always an odd prime power here).
    ///
    /// This is the map that turns a lifted factor back into a candidate integer
    /// polynomial: an integer `v` with `|v| ≤ (m−1)/2` is recovered *exactly*
    /// from its residue.
    #[must_use]
    pub fn symmetric_coeffs(&self) -> Vec<BigInt> {
        self.coeffs
            .iter()
            .map(|c| {
                if c * 2i32 > self.modulus {
                    c - &self.modulus
                } else {
                    c.clone()
                }
            })
            .collect()
    }
}

// ── BigInt helpers ────────────────────────────────────────────────────────────

/// Convert a non-negative [`BigInt`] known to be `< 2^63` into an `i128`.
///
/// Only ever called on residues modulo a prime `< MAX_MODULUS`, so the single
/// 64-bit digit read here is the whole value.
#[must_use]
pub fn nonneg_bigint_to_i128(v: &BigInt) -> i128 {
    debug_assert!(v.sign() != Sign::Minus, "expected a non-negative value");
    debug_assert!(v.bits() <= 63, "value does not fit in one 64-bit digit");
    i128::from(v.iter_u64_digits().next().unwrap_or(0))
}

/// Modular inverse of `a` modulo `m`, or `None` when `gcd(a, m) != 1`.
///
/// Extended Euclid over [`BigInt`]. Used to build the monic image
/// `lc(f)^{-1} · f (mod p^k)` that the Hensel lift targets.
#[must_use]
pub fn bigint_mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    let (mut old_r, mut r) = (a.mod_floor(m), m.clone());
    let (mut old_s, mut s) = (BigInt::from(1i32), BigInt::from(0i32));

    while r.sign() != Sign::NoSign {
        let q = &old_r / &r;
        let new_r = &old_r - &q * &r;
        old_r = std::mem::replace(&mut r, new_r);
        let new_s = &old_s - &q * &s;
        old_s = std::mem::replace(&mut s, new_s);
    }

    if old_r != BigInt::from(1i32) {
        return None;
    }
    Some(old_s.mod_floor(m))
}

/// Deterministic primality test for a small `i128` (trial division).
///
/// Only used on candidate moduli below [`MAX_MODULUS`], where trial division up
/// to `sqrt(n) < 2^10` is faster than anything cleverer.
#[must_use]
pub fn is_small_prime(n: i128) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3i128;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}
