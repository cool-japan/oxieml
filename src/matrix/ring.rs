//! Matrices over the polynomial ring `R = ℚ[atoms]`, and the **exact division**
//! that the fraction-free algorithms are built on.
//!
//! `R` is an integral domain: it has no zero divisors, an exact zero test, and —
//! when the quotient exists in `R` — an exact division. That is precisely the
//! hypothesis under which the Bareiss identity and the Faddeev–LeVerrier recurrence
//! are theorems, so all the heavy lifting happens here rather than on raw
//! `LoweredOp` trees.

use crate::poly::{Coeff, MultiPoly, coeff_is_zero, coeff_one, coeff_recip};
use num_bigint::{BigInt, Sign};
use num_integer::Integer;

use super::MatrixError;

/// `|x|` for a `BigInt` (`num-traits`' `Signed::abs` is not in scope here).
fn magnitude(x: &BigInt) -> BigInt {
    BigInt::from_biguint(Sign::Plus, x.magnitude().clone())
}

/// Hard cap on the number of reduction steps in [`exact_div`].
///
/// The loop is already guaranteed to terminate (the leading monomial strictly
/// decreases in a well-order), so this is purely a defence against a pathological
/// input producing an effectively unbounded run.
const MAX_DIVISION_STEPS: usize = 1_000_000;

/// The lexicographically greatest monomial of `p` and its coefficient.
///
/// `BTreeMap<Vec<u32>, _>` already orders its keys lexicographically and all
/// exponent vectors have the same length, so the last key *is* the lex-leading
/// monomial. Lex is a monomial order (a well-order compatible with multiplication),
/// which is what makes the division loop below terminate.
fn leading_term(p: &MultiPoly) -> Option<(Vec<u32>, Coeff)> {
    p.terms
        .iter()
        .rev()
        .find(|(_, c)| !coeff_is_zero(c))
        .map(|(e, c)| (e.clone(), c.clone()))
}

/// Exact division in `ℚ[atoms]`: return `q` with `q · divisor = dividend`.
///
/// # Algorithm
///
/// Classical multivariate division with respect to the lex monomial order. At each
/// step the leading monomial of the running remainder must be divisible by the
/// leading monomial of the divisor; the corresponding quotient monomial is appended
/// to `q` and its multiple of the divisor is subtracted. Because lex is a
/// well-order and the leading monomial strictly decreases every step, the loop
/// terminates. If the division is exact the remainder reaches `0`.
///
/// # Errors
///
/// * [`MatrixError::Poly`] — `divisor` is the zero polynomial.
/// * [`MatrixError::InexactDivision`] — the quotient does not lie in `R`. Inside
///   Bareiss and Faddeev–LeVerrier this is *unreachable* (both algorithms divide
///   only by quantities the underlying theorems prove to be exact factors), so it
///   doubles as a self-check on the implementation rather than a user-facing error.
pub(crate) fn exact_div(
    dividend: &MultiPoly,
    divisor: &MultiPoly,
) -> Result<MultiPoly, MatrixError> {
    let num_vars = dividend.num_vars;
    if divisor.is_zero() {
        return Err(MatrixError::Poly(crate::poly::PolyError::DivByZero));
    }
    if dividend.is_zero() {
        return Ok(MultiPoly::zero(num_vars));
    }

    let (divisor_exp, divisor_coeff) =
        leading_term(divisor).ok_or(MatrixError::Poly(crate::poly::PolyError::DivByZero))?;

    // Fast path: division by a nonzero constant is a scaling.
    if divisor.terms.len() == 1 && divisor_exp.iter().all(|&e| e == 0) {
        let inv = coeff_recip(&divisor_coeff)
            .ok_or(MatrixError::Poly(crate::poly::PolyError::DivByZero))?;
        return dividend.scale(&inv).map_err(MatrixError::Poly);
    }

    let inv_divisor_coeff =
        coeff_recip(&divisor_coeff).ok_or(MatrixError::Poly(crate::poly::PolyError::DivByZero))?;

    let mut remainder = dividend.clone();
    let mut quotient = MultiPoly::zero(num_vars);

    for _ in 0..MAX_DIVISION_STEPS {
        let Some((rem_exp, rem_coeff)) = leading_term(&remainder) else {
            // Remainder vanished: the division was exact.
            return Ok(quotient);
        };
        if rem_exp.iter().zip(divisor_exp.iter()).any(|(r, d)| r < d) {
            return Err(MatrixError::InexactDivision);
        }
        let step_exp: Vec<u32> = rem_exp
            .iter()
            .zip(divisor_exp.iter())
            .map(|(r, d)| r - d)
            .collect();
        let step_coeff = &rem_coeff * &inv_divisor_coeff;

        let mut step = MultiPoly::zero(num_vars);
        step.terms.insert(step_exp, step_coeff);

        quotient = quotient.add(&step).map_err(MatrixError::Poly)?;
        let subtrahend = step.mul(divisor).map_err(MatrixError::Poly)?;
        remainder = remainder.sub(&subtrahend).map_err(MatrixError::Poly)?;
    }

    Err(MatrixError::InexactDivision)
}

/// The gcd of two non-negative rationals: `gcd(numerators) / lcm(denominators)`.
///
/// This is the largest rational `g` such that both `a/g` and `b/g` are integers,
/// which is what "common content" means over ℚ.
pub(crate) fn rational_gcd(a: &Coeff, b: &Coeff) -> Coeff {
    if coeff_is_zero(a) {
        return b.clone();
    }
    if coeff_is_zero(b) {
        return a.clone();
    }
    Coeff::new(
        magnitude(a.numer()).gcd(&magnitude(b.numer())),
        a.denom().lcm(b.denom()),
    )
}

/// The rational content of `p`: `gcd(numerators) / lcm(denominators)`.
///
/// Dividing by the content clears denominators and common integer factors without
/// changing the polynomial up to a nonzero rational scalar. Returns `None` for the
/// zero polynomial.
pub(crate) fn content(p: &MultiPoly) -> Option<Coeff> {
    let mut numerator_gcd: Option<BigInt> = None;
    let mut denominator_lcm: Option<BigInt> = None;
    for c in p.terms.values() {
        if coeff_is_zero(c) {
            continue;
        }
        numerator_gcd = Some(match numerator_gcd {
            None => magnitude(c.numer()),
            Some(g) => g.gcd(&magnitude(c.numer())),
        });
        denominator_lcm = Some(match denominator_lcm {
            None => c.denom().clone(),
            Some(l) => l.lcm(c.denom()),
        });
    }
    match (numerator_gcd, denominator_lcm) {
        (Some(g), Some(l)) => Some(Coeff::new(g, l)),
        _ => None,
    }
}

/// If `p` is a constant polynomial, its exact rational value.
pub(crate) fn as_constant(p: &MultiPoly) -> Option<Coeff> {
    let mut value: Option<Coeff> = None;
    for (exps, c) in &p.terms {
        if coeff_is_zero(c) {
            continue;
        }
        if exps.iter().any(|&e| e > 0) {
            return None;
        }
        value = Some(c.clone());
    }
    Some(value.unwrap_or_else(crate::poly::coeff_zero))
}

/// A dense row-major matrix over `ℚ[atoms]`.
#[derive(Clone, Debug)]
pub(crate) struct PolyMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of atoms (the polynomial ring's variable count).
    pub num_vars: usize,
    /// Row-major entries, `rows · cols` of them.
    pub data: Vec<MultiPoly>,
}

impl PolyMatrix {
    /// The all-zero `rows × cols` matrix.
    pub(crate) fn zeros(rows: usize, cols: usize, num_vars: usize) -> Self {
        Self {
            rows,
            cols,
            num_vars,
            data: vec![MultiPoly::zero(num_vars); rows * cols],
        }
    }

    /// The `n × n` identity.
    pub(crate) fn identity(n: usize, num_vars: usize) -> Self {
        let mut m = Self::zeros(n, n, num_vars);
        for i in 0..n {
            m.data[i * n + i] = MultiPoly::constant(coeff_one(), num_vars);
        }
        m
    }

    /// Entry `(i, j)`.
    pub(crate) fn at(&self, i: usize, j: usize) -> &MultiPoly {
        &self.data[i * self.cols + j]
    }

    /// Overwrite entry `(i, j)`.
    pub(crate) fn set(&mut self, i: usize, j: usize, v: MultiPoly) {
        self.data[i * self.cols + j] = v;
    }

    /// Exchange two rows.
    pub(crate) fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for j in 0..self.cols {
            self.data.swap(a * self.cols + j, b * self.cols + j);
        }
    }

    /// Matrix product.
    pub(crate) fn mul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.cols, other.cols),
                got: (other.rows, other.cols),
            });
        }
        let mut out = Self::zeros(self.rows, other.cols, self.num_vars);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = MultiPoly::zero(self.num_vars);
                for k in 0..self.cols {
                    let term = self
                        .at(i, k)
                        .mul(other.at(k, j))
                        .map_err(MatrixError::Poly)?;
                    acc = acc.add(&term).map_err(MatrixError::Poly)?;
                }
                out.set(i, j, acc);
            }
        }
        Ok(out)
    }

    /// Entry-wise sum.
    pub(crate) fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        let mut out = Self::zeros(self.rows, self.cols, self.num_vars);
        for idx in 0..self.data.len() {
            out.data[idx] = self.data[idx]
                .add(&other.data[idx])
                .map_err(MatrixError::Poly)?;
        }
        Ok(out)
    }

    /// Add `c · I` in place (only meaningful for a square matrix).
    pub(crate) fn add_scalar_diagonal(&mut self, c: &MultiPoly) -> Result<(), MatrixError> {
        let n = self.rows.min(self.cols);
        for i in 0..n {
            let updated = self.at(i, i).add(c).map_err(MatrixError::Poly)?;
            self.set(i, i, updated);
        }
        Ok(())
    }

    /// Trace (sum of the diagonal).
    pub(crate) fn trace(&self) -> Result<MultiPoly, MatrixError> {
        let n = self.rows.min(self.cols);
        let mut acc = MultiPoly::zero(self.num_vars);
        for i in 0..n {
            acc = acc.add(self.at(i, i)).map_err(MatrixError::Poly)?;
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::LoweredOp;
    use crate::matrix::atoms::AtomSpace;
    use std::sync::Arc;

    fn poly_of(expr: &LoweredOp, space: &AtomSpace) -> MultiPoly {
        space.polynomialize(expr).expect("polynomializable")
    }

    #[test]
    fn exact_division_recovers_the_factor() {
        // (x0 + x1) * (x0 - 2*x1)  ÷  (x0 + x1)  =  x0 - 2*x1
        let a = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
        let b = LoweredOp::Sub(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(2.0)),
                Arc::new(LoweredOp::Var(1)),
            )),
        );
        let space = AtomSpace::from_exprs(&[a.clone(), b.clone()]);
        let pa = poly_of(&a, &space);
        let pb = poly_of(&b, &space);
        let product = pa.mul(&pb).expect("mul");
        let q = exact_div(&product, &pa).expect("exact");
        assert_eq!(q, pb);
    }

    #[test]
    fn inexact_division_is_reported() {
        let space = AtomSpace::from_exprs(&[LoweredOp::Var(0), LoweredOp::Var(1)]);
        let x = poly_of(&LoweredOp::Var(0), &space);
        let y = poly_of(&LoweredOp::Var(1), &space);
        // x ÷ y does not lie in ℚ[x, y]
        assert_eq!(exact_div(&x, &y), Err(MatrixError::InexactDivision));
    }

    #[test]
    fn division_by_constant_is_a_scaling() {
        let space = AtomSpace::from_exprs(&[LoweredOp::Var(0)]);
        let x = poly_of(&LoweredOp::Var(0), &space);
        let two = MultiPoly::constant(crate::poly::coeff_from_i64(2), space.len());
        let half_x = exact_div(&x, &two).expect("exact");
        let doubled = half_x.add(&half_x).expect("add");
        assert_eq!(doubled, x);
    }

    #[test]
    fn identity_multiplies_neutrally() {
        let space = AtomSpace::from_exprs(&[LoweredOp::Var(0)]);
        let nv = space.len();
        let mut a = PolyMatrix::zeros(2, 2, nv);
        a.set(0, 0, poly_of(&LoweredOp::Var(0), &space));
        a.set(1, 1, MultiPoly::constant(coeff_one(), nv));
        let i = PolyMatrix::identity(2, nv);
        let p = a.mul(&i).expect("mul");
        assert_eq!(p.data, a.data);
    }
}
