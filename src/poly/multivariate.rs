//! Sparse multivariate polynomial over exact, arbitrary-precision rational coefficients.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::lower::LoweredOp;

use super::monomial::{self, MonOrder, Monomial};
use super::univariate::Poly;
use super::{
    Coeff, PolyError, coeff_is_zero, coeff_one, coeff_recip, coeff_zero, f64_to_ratio, ratio_to_f64,
};

/// Sparse multivariate polynomial over exact rational [`Coeff`] coefficients.
///
/// Terms are keyed by their exponent vector, so `terms[[2, 1]] = 3` means the
/// monomial `3·x₀²·x₁`. Like [`Poly`], every operation is exact and cannot
/// overflow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPoly {
    /// The number of variables.
    pub num_vars: usize,
    /// Sparse term map: exponent vector → coefficient.
    pub terms: BTreeMap<Vec<u32>, Coeff>,
}

impl MultiPoly {
    /// Create the zero polynomial in `num_vars` variables.
    #[must_use]
    pub fn zero(num_vars: usize) -> Self {
        Self {
            num_vars,
            terms: BTreeMap::new(),
        }
    }

    /// Create the constant polynomial with value `c` in `num_vars` variables.
    #[must_use]
    pub fn constant(c: Coeff, num_vars: usize) -> Self {
        let mut terms = BTreeMap::new();
        if !coeff_is_zero(&c) {
            terms.insert(vec![0u32; num_vars], c);
        }
        Self { num_vars, terms }
    }

    /// Return `true` if this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.terms.values().all(coeff_is_zero)
    }

    fn normalize(&mut self) {
        self.terms.retain(|_, c| !coeff_is_zero(c));
    }

    fn add_term(&mut self, exp: Vec<u32>, coeff: &Coeff) {
        let entry = self.terms.entry(exp).or_insert_with(coeff_zero);
        let sum = &*entry + coeff;
        *entry = sum;
    }

    /// Try to convert a `LoweredOp` expression tree to a multivariate polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::NotPolynomial`] when the expression is not polynomial
    /// in the first `num_vars` variables.
    pub fn from_lowered(expr: &LoweredOp, num_vars: usize) -> Result<Self, PolyError> {
        match expr {
            LoweredOp::Const(c) => Ok(Self::constant(f64_to_ratio(*c)?, num_vars)),
            LoweredOp::NamedConst(nc) => Ok(Self::constant(f64_to_ratio(nc.value())?, num_vars)),
            LoweredOp::Var(i) => {
                if *i >= num_vars {
                    return Err(PolyError::NotPolynomial);
                }
                let mut terms = BTreeMap::new();
                let mut exp = vec![0u32; num_vars];
                exp[*i] = 1;
                terms.insert(exp, coeff_one());
                Ok(Self { num_vars, terms })
            }
            LoweredOp::Add(a, b) => {
                Self::from_lowered(a, num_vars)?.add(&Self::from_lowered(b, num_vars)?)
            }
            LoweredOp::Sub(a, b) => {
                Self::from_lowered(a, num_vars)?.sub(&Self::from_lowered(b, num_vars)?)
            }
            LoweredOp::Mul(a, b) => {
                Self::from_lowered(a, num_vars)?.mul(&Self::from_lowered(b, num_vars)?)
            }
            LoweredOp::Neg(a) => Self::from_lowered(a, num_vars)?.neg(),
            LoweredOp::Pow(base, exp) => {
                if let LoweredOp::Const(e) = exp.as_ref() {
                    let n = *e;
                    if n < 0.0 || n.fract() != 0.0 || n > 100.0 {
                        return Err(PolyError::NotPolynomial);
                    }
                    Self::from_lowered(base, num_vars)?.pow(n as usize)
                } else {
                    Err(PolyError::NotPolynomial)
                }
            }
            _ => Err(PolyError::NotPolynomial),
        }
    }

    /// Convert this multivariate polynomial back to a `LoweredOp` expression.
    #[must_use]
    pub fn to_lowered(&self) -> LoweredOp {
        let mut norm = self.clone();
        norm.normalize();
        if norm.terms.is_empty() {
            return LoweredOp::Const(0.0);
        }

        let mut term_ops: Vec<LoweredOp> = Vec::new();
        for (exps, coeff) in &norm.terms {
            let c_f64 = ratio_to_f64(coeff);
            let mut factors: Vec<LoweredOp> = Vec::new();
            if c_f64 != 1.0 {
                factors.push(LoweredOp::Const(c_f64));
            }
            for (var_idx, &exp) in exps.iter().enumerate() {
                if exp == 0 {
                    continue;
                }
                let x = LoweredOp::Var(var_idx);
                if exp == 1 {
                    factors.push(x);
                } else {
                    factors.push(LoweredOp::Pow(
                        Arc::new(x),
                        Arc::new(LoweredOp::Const(f64::from(exp))),
                    ));
                }
            }
            if factors.is_empty() {
                term_ops.push(LoweredOp::Const(c_f64));
            } else {
                let mut acc = factors.remove(0);
                for f in factors {
                    acc = LoweredOp::Mul(Arc::new(acc), Arc::new(f));
                }
                term_ops.push(acc);
            }
        }

        let mut acc = term_ops.remove(0);
        for op in term_ops {
            acc = LoweredOp::Add(Arc::new(acc), Arc::new(op));
        }
        acc
    }

    /// Add two multivariate polynomials.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        let mut result = self.clone();
        for (exp, coeff) in &other.terms {
            result.add_term(exp.clone(), coeff);
        }
        result.normalize();
        Ok(result)
    }

    /// Subtract `other` from `self`.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        let neg = other.neg()?;
        self.add(&neg)
    }

    /// Multiply two multivariate polynomials.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        let mut result = Self::zero(self.num_vars);
        for (exp_a, coeff_a) in &self.terms {
            for (exp_b, coeff_b) in &other.terms {
                let new_exp: Vec<u32> =
                    exp_a.iter().zip(exp_b.iter()).map(|(a, b)| a + b).collect();
                result.add_term(new_exp, &(coeff_a * coeff_b));
            }
        }
        result.normalize();
        Ok(result)
    }

    /// Scale this polynomial by a rational constant.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn scale(&self, c: &Coeff) -> Result<Self, PolyError> {
        let mut result = Self::zero(self.num_vars);
        if coeff_is_zero(c) {
            return Ok(result);
        }
        for (exp, coeff) in &self.terms {
            let new_coeff = coeff * c;
            if !coeff_is_zero(&new_coeff) {
                result.terms.insert(exp.clone(), new_coeff);
            }
        }
        Ok(result)
    }

    /// Negate this polynomial.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn neg(&self) -> Result<Self, PolyError> {
        let mut result = Self::zero(self.num_vars);
        for (exp, coeff) in &self.terms {
            result.terms.insert(exp.clone(), -coeff);
        }
        Ok(result)
    }

    /// Raise this polynomial to a non-negative integer power by binary
    /// exponentiation.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` is kept for API stability.
    pub fn pow(&self, n: usize) -> Result<Self, PolyError> {
        if n == 0 {
            return Ok(Self::constant(coeff_one(), self.num_vars));
        }
        let mut result = Self::constant(coeff_one(), self.num_vars);
        let mut base = self.clone();
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

    /// Compute the GCD of two multivariate polynomials by projecting both onto
    /// the first variable.
    ///
    /// # Errors
    ///
    /// Propagates errors from the univariate GCD.
    pub fn gcd(a: &Self, b: &Self) -> Result<Self, PolyError> {
        let ua = a.project_to_var(0);
        let ub = b.project_to_var(0);
        let g = Poly::gcd(&ua, &ub)?;
        Ok(Self::from_univariate(&g, 0, a.num_vars))
    }

    fn project_to_var(&self, var: usize) -> Poly {
        let mut coeffs: Vec<Coeff> = Vec::new();
        for (exp, coeff) in &self.terms {
            let is_univariate = exp.iter().enumerate().all(|(i, &e)| i == var || e == 0);
            if is_univariate {
                let degree = exp[var] as usize;
                while coeffs.len() <= degree {
                    coeffs.push(coeff_zero());
                }
                coeffs[degree] = &coeffs[degree] + coeff;
            }
        }
        Poly::from_ratios(coeffs)
    }

    fn from_univariate(p: &Poly, var: usize, num_vars: usize) -> Self {
        let mut result = Self::zero(num_vars);
        for (i, coeff) in p.coeffs.iter().enumerate() {
            if !coeff_is_zero(coeff) {
                let mut exp = vec![0u32; num_vars];
                exp[var] = i as u32;
                result.terms.insert(exp, coeff.clone());
            }
        }
        result
    }

    /// Evaluate this polynomial at an `f64` point vector.
    #[must_use]
    pub fn eval_f64(&self, point: &[f64]) -> f64 {
        let mut result = 0.0f64;
        for (exp, coeff) in &self.terms {
            let mut term = ratio_to_f64(coeff);
            for (i, &e) in exp.iter().enumerate() {
                if e > 0 {
                    let x = if i < point.len() { point[i] } else { 0.0 };
                    term *= x.powi(e as i32);
                }
            }
            result += term;
        }
        result
    }

    /// Return the degree of this polynomial in variable `var_idx`.
    #[must_use]
    pub fn degree_in(&self, var_idx: usize) -> usize {
        self.terms
            .keys()
            .map(|exp| {
                if var_idx < exp.len() {
                    exp[var_idx] as usize
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// Return the leading coefficient polynomial when viewed as univariate in `var_idx`.
    #[must_use]
    pub fn leading_coeff_poly_in(&self, var_idx: usize) -> Self {
        let max_deg = self.degree_in(var_idx);
        let mut result = Self::zero(self.num_vars);
        for (exp, coeff) in &self.terms {
            let d = if var_idx < exp.len() {
                exp[var_idx] as usize
            } else {
                0
            };
            if d == max_deg {
                let mut new_exp = exp.clone();
                if var_idx < new_exp.len() {
                    new_exp[var_idx] = 0;
                }
                result.terms.insert(new_exp, coeff.clone());
            }
        }
        result.normalize();
        result
    }
}

// ── Order-aware operations (Gröbner basis support) ────────────────────────────
//
// Everything below interprets the polynomial's terms through a [`MonOrder`],
// which is what turns the unordered term map into an object with a well-defined
// *leading term* — the foundation of the division algorithm, S-polynomials and
// Buchberger's algorithm.

impl MultiPoly {
    /// The number of (nonzero) terms.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.terms.values().filter(|c| !coeff_is_zero(c)).count()
    }

    /// Total degree `max{ |α| : c_α ≠ 0 }`, or `0` for the zero polynomial.
    #[must_use]
    pub fn total_degree(&self) -> u64 {
        self.terms
            .iter()
            .filter(|(_, c)| !coeff_is_zero(c))
            .map(|(e, _)| monomial::total_degree(e))
            .max()
            .unwrap_or(0)
    }

    /// Return `true` when variable `var` occurs with a positive exponent.
    #[must_use]
    pub fn involves_var(&self, var: usize) -> bool {
        self.terms
            .iter()
            .filter(|(_, c)| !coeff_is_zero(c))
            .any(|(e, _)| var < e.len() && e[var] > 0)
    }

    /// The exponent vector of the leading monomial `LM(f)` under `order`, or
    /// `None` for the zero polynomial.
    #[must_use]
    pub fn leading_exponents(&self, order: MonOrder) -> Option<&[u32]> {
        self.terms
            .iter()
            .filter(|(_, c)| !coeff_is_zero(c))
            .map(|(e, _)| e.as_slice())
            .reduce(|a, b| {
                if order.cmp_exponents(b, a) == Ordering::Greater {
                    b
                } else {
                    a
                }
            })
    }

    /// The leading monomial `LM(f)` under `order`, or `None` for the zero polynomial.
    #[must_use]
    pub fn leading_monomial(&self, order: MonOrder) -> Option<Monomial> {
        self.leading_exponents(order)
            .map(|e| Monomial::new(e.to_vec()))
    }

    /// The leading coefficient `LC(f)` under `order`, or `None` for the zero polynomial.
    #[must_use]
    pub fn leading_coeff(&self, order: MonOrder) -> Option<Coeff> {
        let lm = self.leading_exponents(order)?;
        self.terms.get(lm).cloned()
    }

    /// The leading term `LT(f) = LC(f) · LM(f)` under `order`, as a
    /// `(monomial, coefficient)` pair. `None` for the zero polynomial.
    #[must_use]
    pub fn leading_term(&self, order: MonOrder) -> Option<(Monomial, Coeff)> {
        let lm = self.leading_exponents(order)?.to_vec();
        let lc = self.terms.get(&lm)?.clone();
        Some((Monomial::new(lm), lc))
    }

    /// Divide through by the leading coefficient so that `LC(f) = 1`.
    ///
    /// # Errors
    ///
    /// Returns [`PolyError::DivByZero`] for the zero polynomial.
    pub fn monic(&self, order: MonOrder) -> Result<Self, PolyError> {
        let lc = self.leading_coeff(order).ok_or(PolyError::DivByZero)?;
        let inv = coeff_recip(&lc).ok_or(PolyError::DivByZero)?;
        self.scale(&inv)
    }

    /// Multiply by the single term `c · x^exps`.
    ///
    /// Adding a fixed exponent vector to every key is injective, so no two terms
    /// can collide and the result needs no re-accumulation.
    #[must_use]
    pub fn term_mul(&self, exps: &[u32], c: &Coeff) -> Self {
        let mut out = Self::zero(self.num_vars);
        if coeff_is_zero(c) {
            return out;
        }
        for (e, coeff) in &self.terms {
            let new_coeff = coeff * c;
            if !coeff_is_zero(&new_coeff) {
                out.terms
                    .insert(monomial::mul_exponents(e, exps), new_coeff);
            }
        }
        out
    }

    /// The coefficient of `x_var^degree`, as a polynomial in the *remaining*
    /// variables (the `var` exponent is set to zero in the result).
    #[must_use]
    pub fn coeff_in_var(&self, var: usize, degree: u32) -> Self {
        let mut out = Self::zero(self.num_vars);
        for (e, coeff) in &self.terms {
            if coeff_is_zero(coeff) {
                continue;
            }
            let d = if var < e.len() { e[var] } else { 0 };
            if d != degree {
                continue;
            }
            let mut key = e.clone();
            if var < key.len() {
                key[var] = 0;
            }
            out.terms.insert(key, coeff.clone());
        }
        out
    }

    /// Formal partial derivative `∂f/∂x_var`, exact over ℚ.
    #[must_use]
    pub fn partial_derivative(&self, var: usize) -> Self {
        let mut out = Self::zero(self.num_vars);
        for (e, coeff) in &self.terms {
            if coeff_is_zero(coeff) || var >= e.len() || e[var] == 0 {
                continue;
            }
            let power = Coeff::from_integer(num_bigint::BigInt::from(e[var]));
            let mut key = e.clone();
            key[var] -= 1;
            let new_coeff = coeff * power;
            if !coeff_is_zero(&new_coeff) {
                out.add_term(key, &new_coeff);
            }
        }
        out.normalize();
        out
    }

    /// The **multivariate division algorithm** with an explicit step budget.
    ///
    /// Divides `self` by the ordered tuple `divisors = (g₁, …, g_s)` under
    /// `order`, producing quotients `(q₁, …, q_s)` and a remainder `r` with
    ///
    /// ```text
    /// f = q₁·g₁ + … + q_s·g_s + r
    /// ```
    ///
    /// such that no monomial of `r` is divisible by any `LM(gᵢ)`, and
    /// `LM(qᵢ·gᵢ) ≤ LM(f)` for every `i`.
    ///
    /// # Algorithm
    ///
    /// (Cox–Little–O'Shea, *Ideals, Varieties, and Algorithms*, Ch. 2 §3,
    /// Theorem 3.) Repeatedly inspect `LT(p)` of the running dividend `p`:
    ///
    /// * if some `LM(gᵢ)` divides `LM(p)`, subtract `(LT(p)/LT(gᵢ))·gᵢ` from `p`
    ///   — this cancels `LT(p)` exactly — and accumulate the multiplier into `qᵢ`;
    /// * otherwise move `LT(p)` into the remainder and drop it from `p`.
    ///
    /// # Termination
    ///
    /// Both branches strictly decrease `LM(p)` in the monomial order (the first
    /// by exact cancellation, the second by removal). Because a monomial order is
    /// a **well-ordering**, no infinite strictly-decreasing chain exists, so the
    /// loop always halts. The `budget` is therefore *not* needed for correctness;
    /// it exists only to bound the wall-clock cost of a pathological input, and
    /// exhausting it yields `Ok(None)` rather than a wrong answer.
    ///
    /// Note that the quotients — but not the remainder's *vanishing* — depend on
    /// the order of `divisors`. When `divisors` is a Gröbner basis the remainder
    /// is the unique canonical normal form and is independent of that order.
    ///
    /// # Errors
    ///
    /// * [`PolyError::DivByZero`] if any divisor is the zero polynomial.
    /// * [`PolyError::NotPolynomial`] if a divisor has a different `num_vars`.
    pub fn divide_bounded(
        &self,
        divisors: &[Self],
        order: MonOrder,
        budget: &mut u64,
    ) -> Result<Option<(Vec<Self>, Self)>, PolyError> {
        let n = self.num_vars;
        let mut leading: Vec<(Vec<u32>, Coeff)> = Vec::with_capacity(divisors.len());
        for d in divisors {
            if d.num_vars != n {
                return Err(PolyError::NotPolynomial);
            }
            let (lm, lc) = d.leading_term(order).ok_or(PolyError::DivByZero)?;
            leading.push((lm.0, lc));
        }

        let mut quotients = vec![Self::zero(n); divisors.len()];
        let mut remainder = Self::zero(n);
        let mut p = self.clone();
        p.normalize();

        while let Some((lm_p, lc_p)) = p.leading_term(order) {
            if *budget == 0 {
                return Ok(None);
            }
            *budget -= 1;

            let mut divided = false;
            for (i, (lm_d, lc_d)) in leading.iter().enumerate() {
                let Some(q_exp) = monomial::div_exponents(lm_d, &lm_p.0) else {
                    continue;
                };
                // LT(p) / LT(g_i) — the coefficient quotient is exact over ℚ.
                let inv = coeff_recip(lc_d).ok_or(PolyError::DivByZero)?;
                let q_coeff = &lc_p * &inv;
                quotients[i].add_term(q_exp.clone(), &q_coeff);
                quotients[i].normalize();
                // Cancels LT(p) exactly, so LM(p) strictly decreases.
                p = p.sub(&divisors[i].term_mul(&q_exp, &q_coeff))?;
                divided = true;
                break;
            }

            if !divided {
                remainder.add_term(lm_p.0.clone(), &lc_p);
                p.terms.remove(&lm_p.0);
            }
        }

        remainder.normalize();
        Ok(Some((quotients, remainder)))
    }

    /// The multivariate division algorithm, unbounded.
    ///
    /// See [`divide_bounded`](Self::divide_bounded) for the algorithm, the
    /// termination argument and the meaning of the outputs.
    ///
    /// # Errors
    ///
    /// Same as [`divide_bounded`](Self::divide_bounded).
    pub fn divide(
        &self,
        divisors: &[Self],
        order: MonOrder,
    ) -> Result<(Vec<Self>, Self), PolyError> {
        let mut budget = u64::MAX;
        let divided = self.divide_bounded(divisors, order, &mut budget)?;
        Ok(divided.expect(
            "division by a well-ordering terminates in far fewer than u64::MAX rewriting steps",
        ))
    }

    /// The remainder of `self` on division by `divisors` under `order`.
    ///
    /// When `divisors` is a Gröbner basis of an ideal `I`, this remainder is the
    /// **canonical normal form** of `self` modulo `I`: it is independent of the
    /// order of the divisors, and it is zero **iff** `self ∈ I`. That equivalence
    /// is exactly what makes [`crate::poly::groebner::ideal_member`] decidable.
    ///
    /// # Errors
    ///
    /// Same as [`divide_bounded`](Self::divide_bounded).
    pub fn reduce(&self, divisors: &[Self], order: MonOrder) -> Result<Self, PolyError> {
        Ok(self.divide(divisors, order)?.1)
    }

    /// The **S-polynomial** of `f` and `g` under `order`.
    ///
    /// With `L = lcm(LM(f), LM(g))`,
    ///
    /// ```text
    /// S(f, g) = (L / LT(f)) · f  −  (L / LT(g)) · g
    /// ```
    ///
    /// Both products have leading term exactly `L`, so the subtraction cancels
    /// it: `S(f, g)` is engineered to expose the *leading-term cancellation*
    /// between `f` and `g`. Buchberger's criterion states that a basis `G` is a
    /// Gröbner basis **iff** `S(f, g)` reduces to `0` modulo `G` for every pair
    /// `f, g ∈ G` — because every such cancellation (every syzygy) is generated
    /// by the S-pairs.
    ///
    /// # Errors
    ///
    /// * [`PolyError::DivByZero`] if either argument is the zero polynomial.
    /// * [`PolyError::NotPolynomial`] if the two have different `num_vars`.
    pub fn s_polynomial(f: &Self, g: &Self, order: MonOrder) -> Result<Self, PolyError> {
        if f.num_vars != g.num_vars {
            return Err(PolyError::NotPolynomial);
        }
        let (lm_f, lc_f) = f.leading_term(order).ok_or(PolyError::DivByZero)?;
        let (lm_g, lc_g) = g.leading_term(order).ok_or(PolyError::DivByZero)?;

        let l = lm_f.lcm(&lm_g);
        // `lcm` is divisible by both leading monomials by construction.
        let a = lm_f.div(&l).ok_or(PolyError::NotPolynomial).map(|m| m.0)?;
        let b = lm_g.div(&l).ok_or(PolyError::NotPolynomial).map(|m| m.0)?;

        let inv_f = coeff_recip(&lc_f).ok_or(PolyError::DivByZero)?;
        let inv_g = coeff_recip(&lc_g).ok_or(PolyError::DivByZero)?;

        let left = f.term_mul(&a, &inv_f);
        let right = g.term_mul(&b, &inv_g);
        left.sub(&right)
    }
}
