//! Sparse multivariate polynomial over exact rational coefficients.

use std::collections::BTreeMap;
use std::sync::Arc;

use num_rational::Ratio;

use crate::lower::LoweredOp;

use super::univariate::Poly;
use super::{PolyError, checked_add, checked_mul, checked_neg, f64_to_ratio, ratio_to_f64};

/// Sparse multivariate polynomial over exact rational `Ratio<i64>` coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiPoly {
    /// The number of variables.
    pub num_vars: usize,
    /// Sparse term map: exponent vector → coefficient.
    pub terms: BTreeMap<Vec<u32>, Ratio<i64>>,
}

impl MultiPoly {
    /// Create the zero polynomial in `num_vars` variables.
    pub fn zero(num_vars: usize) -> Self {
        Self {
            num_vars,
            terms: BTreeMap::new(),
        }
    }

    /// Create the constant polynomial with value `c` in `num_vars` variables.
    pub fn constant(c: Ratio<i64>, num_vars: usize) -> Self {
        let mut terms = BTreeMap::new();
        if c != Ratio::new(0, 1) {
            terms.insert(vec![0u32; num_vars], c);
        }
        Self { num_vars, terms }
    }

    /// Return `true` if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.values().all(|c| c == &Ratio::new(0, 1))
    }

    fn normalize(&mut self) {
        self.terms.retain(|_, c| c != &Ratio::new(0, 1));
    }

    fn add_term(&mut self, exp: Vec<u32>, coeff: Ratio<i64>) -> Result<(), PolyError> {
        let entry = self.terms.entry(exp).or_insert(Ratio::new(0, 1));
        *entry = checked_add(*entry, coeff)?;
        Ok(())
    }

    /// Try to convert a `LoweredOp` expression tree to a multivariate polynomial.
    pub fn from_lowered(expr: &LoweredOp, num_vars: usize) -> Result<Self, PolyError> {
        match expr {
            LoweredOp::Const(c) => {
                let r = f64_to_ratio(*c)?;
                Ok(Self::constant(r, num_vars))
            }
            LoweredOp::NamedConst(nc) => {
                let r = f64_to_ratio(nc.value())?;
                Ok(Self::constant(r, num_vars))
            }
            LoweredOp::Var(i) => {
                if *i >= num_vars {
                    return Err(PolyError::NotPolynomial);
                }
                let mut terms = BTreeMap::new();
                let mut exp = vec![0u32; num_vars];
                exp[*i] = 1;
                terms.insert(exp, Ratio::new(1i64, 1));
                Ok(Self { num_vars, terms })
            }
            LoweredOp::Add(a, b) => {
                let pa = Self::from_lowered(a, num_vars)?;
                let pb = Self::from_lowered(b, num_vars)?;
                pa.add(&pb)
            }
            LoweredOp::Sub(a, b) => {
                let pa = Self::from_lowered(a, num_vars)?;
                let pb = Self::from_lowered(b, num_vars)?;
                pa.sub(&pb)
            }
            LoweredOp::Mul(a, b) => {
                let pa = Self::from_lowered(a, num_vars)?;
                let pb = Self::from_lowered(b, num_vars)?;
                pa.mul(&pb)
            }
            LoweredOp::Neg(a) => {
                let pa = Self::from_lowered(a, num_vars)?;
                pa.neg()
            }
            LoweredOp::Pow(base, exp) => {
                if let LoweredOp::Const(e) = exp.as_ref() {
                    let n = *e;
                    if n < 0.0 || n.fract() != 0.0 || n > 100.0 {
                        return Err(PolyError::NotPolynomial);
                    }
                    let n_u = n as usize;
                    let pb = Self::from_lowered(base, num_vars)?;
                    pb.pow(n_u)
                } else {
                    Err(PolyError::NotPolynomial)
                }
            }
            _ => Err(PolyError::NotPolynomial),
        }
    }

    /// Convert this multivariate polynomial back to a `LoweredOp` expression.
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
                        Arc::new(LoweredOp::Const(exp as f64)),
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
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        let mut result = self.clone();
        for (exp, coeff) in &other.terms {
            result.add_term(exp.clone(), *coeff)?;
        }
        result.normalize();
        Ok(result)
    }

    /// Subtract `other` from `self`.
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        let neg = other.neg()?;
        self.add(&neg)
    }

    /// Multiply two multivariate polynomials.
    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        let mut result = Self::zero(self.num_vars);
        for (exp_a, coeff_a) in &self.terms {
            for (exp_b, coeff_b) in &other.terms {
                let new_exp: Vec<u32> =
                    exp_a.iter().zip(exp_b.iter()).map(|(a, b)| a + b).collect();
                let new_coeff = checked_mul(*coeff_a, *coeff_b)?;
                result.add_term(new_exp, new_coeff)?;
            }
        }
        result.normalize();
        Ok(result)
    }

    /// Scale this polynomial by a rational constant.
    pub fn scale(&self, c: Ratio<i64>) -> Result<Self, PolyError> {
        let mut result = Self::zero(self.num_vars);
        for (exp, coeff) in &self.terms {
            let new_coeff = checked_mul(*coeff, c)?;
            if new_coeff != Ratio::new(0, 1) {
                result.terms.insert(exp.clone(), new_coeff);
            }
        }
        Ok(result)
    }

    /// Negate this polynomial.
    pub fn neg(&self) -> Result<Self, PolyError> {
        let mut result = Self::zero(self.num_vars);
        for (exp, coeff) in &self.terms {
            result.terms.insert(exp.clone(), checked_neg(*coeff)?);
        }
        Ok(result)
    }

    /// Raise this polynomial to a non-negative integer power.
    pub fn pow(&self, n: usize) -> Result<Self, PolyError> {
        if n == 0 {
            return Ok(Self::constant(Ratio::new(1, 1), self.num_vars));
        }
        let mut result = Self::constant(Ratio::new(1, 1), self.num_vars);
        for _ in 0..n {
            result = result.mul(self)?;
        }
        Ok(result)
    }

    /// Compute the GCD of two multivariate polynomials.
    pub fn gcd(a: &Self, b: &Self) -> Result<Self, PolyError> {
        let ua = a.project_to_var(0)?;
        let ub = b.project_to_var(0)?;
        let g = Poly::gcd(&ua, &ub)?;
        let g_multi = Self::from_univariate(&g, 0, a.num_vars)?;
        Ok(g_multi)
    }

    fn project_to_var(&self, var: usize) -> Result<Poly, PolyError> {
        let mut coeffs: Vec<Ratio<i64>> = Vec::new();
        for (exp, coeff) in &self.terms {
            let is_univariate = exp.iter().enumerate().all(|(i, &e)| i == var || e == 0);
            if is_univariate {
                let degree = exp[var] as usize;
                while coeffs.len() <= degree {
                    coeffs.push(Ratio::new(0, 1));
                }
                coeffs[degree] = checked_add(coeffs[degree], *coeff)?;
            }
        }
        let mut p = Poly { coeffs };
        p.normalize();
        Ok(p)
    }

    fn from_univariate(p: &Poly, var: usize, num_vars: usize) -> Result<Self, PolyError> {
        let mut result = Self::zero(num_vars);
        for (i, coeff) in p.coeffs.iter().enumerate() {
            if *coeff != Ratio::new(0, 1) {
                let mut exp = vec![0u32; num_vars];
                exp[var] = i as u32;
                result.terms.insert(exp, *coeff);
            }
        }
        Ok(result)
    }

    /// Evaluate this polynomial at an `f64` point vector.
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
                result.terms.insert(new_exp, *coeff);
            }
        }
        result.normalize();
        result
    }
}
