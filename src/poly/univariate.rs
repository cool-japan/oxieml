//! Dense univariate polynomial over exact rational coefficients.

use std::sync::Arc;

use num_rational::Ratio;

use crate::lower::LoweredOp;

use super::{
    PolyError, checked_add, checked_mul, checked_neg, checked_sub, eval_poly_i64_at_ratio,
    f64_to_ratio, gcd_i64, integer_factors, ratio_to_f64,
};

/// Dense univariate polynomial over exact rational `Ratio<i64>` coefficients.
///
/// Coefficient vector is stored in ascending degree order: `coeffs[i]` is the
/// coefficient of `x^i`. The zero polynomial is represented by an empty vector
/// or a vector of all-zero rationals.
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    /// Coefficients in ascending degree order (`coeffs[i]` = coeff of `x^i`).
    pub coeffs: Vec<Ratio<i64>>,
}

impl Poly {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create the zero polynomial.
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// Create the constant polynomial with value `c`.
    pub fn constant(c: Ratio<i64>) -> Self {
        if c == Ratio::new(0, 1) {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// Create the monic monomial `x^degree`.
    pub fn monomial(degree: usize) -> Self {
        let mut coeffs = vec![Ratio::new(0, 1); degree + 1];
        coeffs[degree] = Ratio::new(1, 1);
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Return the degree of this polynomial (`None` for the zero polynomial).
    pub fn degree(&self) -> Option<usize> {
        let norm = self.normalized();
        if norm.coeffs.is_empty() {
            None
        } else {
            Some(norm.coeffs.len() - 1)
        }
    }

    /// Return `true` if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c == &Ratio::new(0, 1))
    }

    /// Return the leading coefficient, or 0 for the zero polynomial.
    pub fn leading_coeff(&self) -> Ratio<i64> {
        let norm = self.normalized();
        norm.coeffs
            .last()
            .cloned()
            .unwrap_or_else(|| Ratio::new(0, 1))
    }

    /// Remove trailing zero coefficients (high-degree zeros).
    pub fn normalize(&mut self) {
        let zero = Ratio::new(0, 1);
        while self.coeffs.last() == Some(&zero) {
            self.coeffs.pop();
        }
    }

    /// Return a normalized clone.
    pub fn normalized(&self) -> Self {
        let mut p = self.clone();
        p.normalize();
        p
    }

    // ── Conversion from/to LoweredOp ──────────────────────────────────────────

    /// Try to convert a `LoweredOp` expression tree to a univariate polynomial
    /// in variable `wrt`.
    pub fn from_lowered(expr: &LoweredOp, wrt: usize) -> Result<Self, PolyError> {
        match expr {
            LoweredOp::Const(c) => {
                let r = f64_to_ratio(*c)?;
                Ok(Self::constant(r))
            }
            LoweredOp::NamedConst(nc) => {
                let r = f64_to_ratio(nc.value())?;
                Ok(Self::constant(r))
            }
            LoweredOp::Var(i) => {
                if *i == wrt {
                    Ok(Self {
                        coeffs: vec![Ratio::new(0, 1), Ratio::new(1, 1)],
                    })
                } else {
                    Err(PolyError::NotPolynomial)
                }
            }
            LoweredOp::Add(a, b) => {
                let pa = Self::from_lowered(a, wrt)?;
                let pb = Self::from_lowered(b, wrt)?;
                pa.add(&pb)
            }
            LoweredOp::Sub(a, b) => {
                let pa = Self::from_lowered(a, wrt)?;
                let pb = Self::from_lowered(b, wrt)?;
                pa.sub(&pb)
            }
            LoweredOp::Mul(a, b) => {
                let pa = Self::from_lowered(a, wrt)?;
                let pb = Self::from_lowered(b, wrt)?;
                pa.mul(&pb)
            }
            LoweredOp::Neg(a) => {
                let pa = Self::from_lowered(a, wrt)?;
                pa.neg()
            }
            LoweredOp::Pow(base, exp) => {
                if let LoweredOp::Const(e) = exp.as_ref() {
                    let n = *e;
                    if n < 0.0 || n.fract() != 0.0 || n > 200.0 {
                        return Err(PolyError::NotPolynomial);
                    }
                    let n_u = n as usize;
                    let pb = Self::from_lowered(base, wrt)?;
                    pb.pow(n_u)
                } else {
                    Err(PolyError::NotPolynomial)
                }
            }
            _ => Err(PolyError::NotPolynomial),
        }
    }

    /// Convert this polynomial back to a `LoweredOp` expression tree using
    /// variable index `wrt`.
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
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let zero = Ratio::new(0i64, 1);
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).cloned().unwrap_or(zero);
            let b = other.coeffs.get(i).cloned().unwrap_or(zero);
            coeffs.push(checked_add(a, b)?);
        }
        let mut p = Self { coeffs };
        p.normalize();
        Ok(p)
    }

    /// Subtract `other` from `self`.
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let zero = Ratio::new(0i64, 1);
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).cloned().unwrap_or(zero);
            let b = other.coeffs.get(i).cloned().unwrap_or(zero);
            coeffs.push(checked_sub(a, b)?);
        }
        let mut p = Self { coeffs };
        p.normalize();
        Ok(p)
    }

    /// Multiply two polynomials.
    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut coeffs = vec![Ratio::new(0i64, 1); n + m - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                let term = checked_mul(*a, *b)?;
                coeffs[i + j] = checked_add(coeffs[i + j], term)?;
            }
        }
        let mut p = Self { coeffs };
        p.normalize();
        Ok(p)
    }

    /// Scale this polynomial by a rational constant.
    pub fn scale(&self, c: Ratio<i64>) -> Result<Self, PolyError> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for a in &self.coeffs {
            coeffs.push(checked_mul(*a, c)?);
        }
        let mut p = Self { coeffs };
        p.normalize();
        Ok(p)
    }

    /// Negate this polynomial.
    pub fn neg(&self) -> Result<Self, PolyError> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for a in &self.coeffs {
            coeffs.push(checked_neg(*a)?);
        }
        Ok(Self { coeffs })
    }

    /// Raise this polynomial to a non-negative integer power.
    pub fn pow(&self, n: usize) -> Result<Self, PolyError> {
        if n == 0 {
            return Ok(Self::constant(Ratio::new(1, 1)));
        }
        let mut result = Self::constant(Ratio::new(1, 1));
        for _ in 0..n {
            result = result.mul(self)?;
        }
        Ok(result)
    }

    // ── Polynomial division ───────────────────────────────────────────────────

    /// Euclidean polynomial division: returns `(quotient, remainder)`.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolyError> {
        let divisor = divisor.normalized();
        if divisor.is_zero() {
            return Err(PolyError::DivByZero);
        }
        let self_norm = self.normalized();
        let self_deg = match self_norm.degree() {
            Some(d) => d,
            None => return Ok((Self::zero(), Self::zero())),
        };
        let div_deg = match divisor.degree() {
            Some(d) => d,
            None => return Err(PolyError::DivByZero),
        };

        if self_deg < div_deg {
            return Ok((Self::zero(), self_norm));
        }

        let mut remainder = self_norm.coeffs.clone();
        let divisor_lc = divisor.leading_coeff();
        let q_len = self_deg - div_deg + 1;
        let mut quotient_coeffs = vec![Ratio::new(0i64, 1); q_len];

        for i in (div_deg..=self_deg).rev() {
            let rem_i = *remainder.get(i).unwrap_or(&Ratio::new(0, 1));
            let q_simplified =
                checked_mul(rem_i, Ratio::new(*divisor_lc.denom(), *divisor_lc.numer()))?;
            let pos = i - div_deg;
            quotient_coeffs[pos] = q_simplified;

            for j in 0..=div_deg {
                let sub_val = checked_mul(
                    q_simplified,
                    *divisor.coeffs.get(j).unwrap_or(&Ratio::new(0, 1)),
                )?;
                remainder[pos + j] = checked_sub(remainder[pos + j], sub_val)?;
            }
        }

        let mut q_poly = Self {
            coeffs: quotient_coeffs,
        };
        q_poly.normalize();
        let mut r_poly = Self { coeffs: remainder };
        r_poly.normalize();

        Ok((q_poly, r_poly))
    }

    // ── GCD ───────────────────────────────────────────────────────────────────

    /// Compute the monic GCD of two polynomials using the Euclidean algorithm.
    pub fn gcd(a: &Self, b: &Self) -> Result<Self, PolyError> {
        let mut u = a.normalized();
        let mut v = b.normalized();

        while !v.is_zero() {
            let (_, r) = u.div_rem(&v)?;
            u = v;
            v = r;
            u.normalize();
            v.normalize();
        }

        if u.is_zero() {
            return Ok(Self::zero());
        }
        let lc = u.leading_coeff();
        u.scale(Ratio::new(*lc.denom(), *lc.numer()))
    }

    // ── Differentiation ───────────────────────────────────────────────────────

    /// Compute the formal derivative of this polynomial.
    pub fn diff(&self) -> Result<Self, PolyError> {
        if self.coeffs.len() <= 1 {
            return Ok(Self::zero());
        }
        let mut coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            let k = i as i64;
            let new_c = checked_mul(*c, Ratio::new(k, 1))?;
            coeffs.push(new_c);
        }
        let mut p = Self { coeffs };
        p.normalize();
        Ok(p)
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Evaluate this polynomial at a rational point.
    pub fn eval(&self, x: Ratio<i64>) -> Result<Ratio<i64>, PolyError> {
        let mut result = Ratio::new(0i64, 1);
        let mut x_pow = Ratio::new(1i64, 1);
        for c in &self.coeffs {
            let term = checked_mul(*c, x_pow)?;
            result = checked_add(result, term)?;
            x_pow = checked_mul(x_pow, x)?;
        }
        Ok(result)
    }

    /// Evaluate this polynomial at an `f64` point.
    pub fn eval_f64(&self, x: f64) -> f64 {
        let norm = self.normalized();
        if norm.coeffs.is_empty() {
            return 0.0;
        }
        let n = norm.coeffs.len();
        let mut acc = ratio_to_f64(&norm.coeffs[n - 1]);
        for i in (0..n - 1).rev() {
            acc = acc * x + ratio_to_f64(&norm.coeffs[i]);
        }
        acc
    }

    // ── Square-free part ──────────────────────────────────────────────────────

    /// Return the square-free part of this polynomial using Yun's algorithm.
    pub fn square_free(&self) -> Result<Self, PolyError> {
        let f = self.normalized();
        if f.is_zero() {
            return Ok(Self::zero());
        }
        let df = f.diff()?;
        let g = Self::gcd(&f, &df)?;
        if g.is_zero() || (g.degree() == Some(0)) {
            return Ok(f);
        }
        let (q, r) = f.div_rem(&g)?;
        if !r.is_zero() {
            return Ok(f);
        }
        let mut result = q;
        if !result.is_zero() {
            let lc = result.leading_coeff();
            result = result.scale(Ratio::new(*lc.denom(), *lc.numer()))?;
        }
        Ok(result)
    }

    // ── Rational root theorem ─────────────────────────────────────────────────

    /// Find all rational roots of this polynomial.
    pub fn rational_roots(&self) -> Result<Vec<Ratio<i64>>, PolyError> {
        let norm = self.normalized();
        if norm.is_zero() {
            return Ok(Vec::new());
        }

        let mut denom_lcm: i64 = 1;
        for c in &norm.coeffs {
            let d = *c.denom();
            let g = gcd_i64(denom_lcm, d);
            denom_lcm = denom_lcm
                .checked_mul(d / g)
                .ok_or(PolyError::CoeffOverflow)?;
        }
        let mut int_coeffs: Vec<i64> = Vec::with_capacity(norm.coeffs.len());
        for c in &norm.coeffs {
            let scaled = (*c.numer())
                .checked_mul(denom_lcm / *c.denom())
                .ok_or(PolyError::CoeffOverflow)?;
            int_coeffs.push(scaled);
        }

        let constant_term = *int_coeffs.first().unwrap_or(&0);
        let leading = *int_coeffs.last().unwrap_or(&0);

        if constant_term == 0 {
            let mut roots = vec![Ratio::new(0, 1)];
            let mut shifted = int_coeffs.clone();
            while shifted.first() == Some(&0) {
                shifted.remove(0);
            }
            if shifted.len() < int_coeffs.len() {
                let p2 = Self {
                    coeffs: shifted.iter().map(|&n| Ratio::new(n, 1)).collect(),
                };
                let more = p2.rational_roots()?;
                roots.extend(more);
            }
            roots.sort_by(|a, b| {
                ratio_to_f64(a)
                    .partial_cmp(&ratio_to_f64(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            roots.dedup();
            return Ok(roots);
        }

        let p_factors = integer_factors(constant_term);
        let q_factors = integer_factors(leading);

        let zero = Ratio::new(0i64, 1);
        let mut roots: Vec<Ratio<i64>> = Vec::new();

        for p in &p_factors {
            for q in &q_factors {
                if *q == 0 {
                    continue;
                }
                let candidate = Ratio::new(*p, *q);
                let val = eval_poly_i64_at_ratio(&int_coeffs, candidate)?;
                if val == zero {
                    roots.push(candidate);
                }
            }
        }

        roots.sort_by(|a, b| {
            ratio_to_f64(a)
                .partial_cmp(&ratio_to_f64(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        roots.dedup();
        Ok(roots)
    }

    // ── Content and primitive part ────────────────────────────────────────────

    /// Compute the content (GCD of all coefficients) of this polynomial.
    pub fn content(&self) -> Ratio<i64> {
        let norm = self.normalized();
        if norm.is_zero() {
            return Ratio::new(0, 1);
        }
        let mut g_numer: i64 = 0;
        let mut g_denom: i64 = 1;
        for c in &norm.coeffs {
            let n = c.numer().abs();
            let d = *c.denom();
            let new_numer = gcd_i64(g_numer.abs(), n);
            let gcd_d = gcd_i64(g_denom, d);
            let new_denom = if gcd_d == 0 {
                d
            } else {
                match (g_denom / gcd_d).checked_mul(d) {
                    Some(v) => v,
                    None => g_denom,
                }
            };
            g_numer = new_numer;
            g_denom = new_denom;
        }
        if g_numer == 0 {
            return Ratio::new(0, 1);
        }
        Ratio::new(g_numer.abs(), g_denom)
    }

    /// Return the primitive part of this polynomial (divide by content).
    pub fn primitive_part(&self) -> Result<Self, PolyError> {
        let norm = self.normalized();
        if norm.is_zero() {
            return Ok(Self::zero());
        }
        let c = norm.content();
        if *c.numer() == 0 {
            return Ok(norm);
        }
        let inv_c = Ratio::new(*c.denom(), *c.numer());
        let mut result = norm.scale(inv_c)?;
        if result.leading_coeff() < Ratio::new(0, 1) {
            result = result.neg()?;
        }
        Ok(result)
    }

    /// Internal: compute resultant returning a Ratio<i64> to avoid premature CoeffOverflow.
    fn resultant_ratio(a: &Self, b: &Self) -> Result<Ratio<i64>, PolyError> {
        let a = a.normalized();
        let b = b.normalized();

        if a.is_zero() || b.is_zero() {
            return Ok(Ratio::new(0, 1));
        }

        let da = a.degree().unwrap_or(0);
        let db = b.degree().unwrap_or(0);

        if da == 0 {
            let c = a.leading_coeff();
            let mut r = Ratio::new(1i64, 1);
            for _ in 0..db {
                r = checked_mul(r, c)?;
            }
            return Ok(r);
        }

        if db == 0 {
            let c = b.leading_coeff();
            let mut r = Ratio::new(1i64, 1);
            for _ in 0..da {
                r = checked_mul(r, c)?;
            }
            return Ok(r);
        }

        if da < db {
            let sign = if (da as i64)
                .checked_mul(db as i64)
                .is_some_and(|x| x % 2 != 0)
            {
                Ratio::new(-1i64, 1)
            } else {
                Ratio::new(1i64, 1)
            };
            let sub = Self::resultant_ratio(&b, &a)?;
            return checked_mul(sub, sign);
        }

        let (_, rem) = a.div_rem(&b)?;

        if rem.is_zero() {
            return Ok(Ratio::new(0, 1));
        }

        let sign = if (da as i64)
            .checked_mul(db as i64)
            .is_none_or(|x| x % 2 == 0)
        {
            Ratio::new(1i64, 1)
        } else {
            Ratio::new(-1i64, 1)
        };

        let lc_b = b.leading_coeff();
        let dr = rem.degree().unwrap_or(0);
        let exp = da - dr;
        let mut lc_pow = Ratio::new(1i64, 1);
        for _ in 0..exp {
            lc_pow = checked_mul(lc_pow, lc_b)?;
        }

        let sub_res = Self::resultant_ratio(&b, &rem)?;

        let tmp = checked_mul(sign, lc_pow)?;
        checked_mul(tmp, sub_res)
    }

    /// Compute the resultant of two polynomials via Euclidean recursion.
    pub fn resultant(a: &Self, b: &Self) -> Result<i64, PolyError> {
        let r = Self::resultant_ratio(a, b)?;
        let (n, d) = (*r.numer(), *r.denom());
        if d != 1 {
            return Err(PolyError::CoeffOverflow);
        }
        Ok(n)
    }

    /// Compute the discriminant of this polynomial.
    pub fn discriminant(&self) -> Result<i64, PolyError> {
        let f = self.normalized();
        if f.is_zero() {
            return Ok(0);
        }
        let n = f.degree().unwrap_or(0);
        if n == 0 {
            return Ok(1);
        }

        let df = f.diff()?;
        let res = Self::resultant_ratio(&f, &df)?;

        let exp = (n as i64) * (n as i64 - 1) / 2;
        let sign = if exp % 2 == 0 {
            Ratio::new(1i64, 1)
        } else {
            Ratio::new(-1i64, 1)
        };

        let lc = f.leading_coeff();
        if lc == Ratio::new(0, 1) {
            return Err(PolyError::DivByZero);
        }

        let lc_inv = Ratio::new(*lc.denom(), *lc.numer());
        let num = checked_mul(sign, res)?;
        let result = checked_mul(num, lc_inv)?;
        let (rn, rd) = (*result.numer(), *result.denom());
        if rd != 1 {
            return Err(PolyError::CoeffOverflow);
        }
        Ok(rn)
    }
}
