//! The Faddeev–LeVerrier algorithm: characteristic polynomial, with the
//! determinant and the inverse falling out as by-products.
//!
//! # The recurrence
//!
//! For an `n × n` matrix `A` over a commutative ring in which the integers
//! `1 … n` are invertible (ℚ[atoms] qualifies), write the characteristic polynomial
//! as
//!
//! ```text
//!   p(λ) = det(λI − A) = λⁿ + c_{n−1} λⁿ⁻¹ + … + c₁ λ + c₀ .
//! ```
//!
//! Faddeev–LeVerrier builds it with a matrix recurrence:
//!
//! ```text
//!   M₀ = 0,                       c_n = 1
//!   M_k = A · M_{k−1} + c_{n−k+1} · I           (k = 1 … n)
//!   c_{n−k} = −(1/k) · tr(A · M_k)
//! ```
//!
//! # Why this is the right algorithm here
//!
//! * **No division by matrix entries.** The only divisions are by the *integers*
//!   `1 … n`. Over `ℚ[atoms]` those are exact rational scalings, so the whole
//!   computation stays inside the polynomial ring: no rational functions, no gcds,
//!   no zero tests on symbolic quantities. The charpoly is therefore computed
//!   **unconditionally exactly**, whatever transcendental relations the atoms hide.
//!   (An LU- or Krylov-based charpoly would have to pivot, and pivoting needs a
//!   zero test — which is exactly the undecidable thing.)
//!
//! * **Determinant for free.** `p(0) = det(−A) = (−1)ⁿ det(A)`, so
//!   `det(A) = (−1)ⁿ · c₀`. This is an independent route to the determinant and
//!   [`super::verify`] cross-checks it against the Bareiss answer.
//!
//! * **Inverse for free.** Cayley–Hamilton gives `p(A) = 0`, and the recurrence's
//!   final matrix satisfies `A · M_n = −c₀ · I`. Hence, whenever `c₀` is invertible,
//!
//!   ```text
//!     A⁻¹ = −M_n / c₀ .
//!   ```
//!
//!   Every entry of `M_n` is a polynomial, so the symbolic inverse comes out as a
//!   single common denominator `c₀` (up to sign, the determinant) over polynomial
//!   numerators — the adjugate, without ever forming `n²` cofactor determinants.
//!
//! # The one place uncertainty enters
//!
//! Everything above is exact ring arithmetic. But "`c₀` is invertible" means "`c₀`
//! is not the zero *function*", and that is undecidable for transcendental entries.
//! `c₀ ≠ 0` in `ℚ[atoms]` does **not** settle it. So [`super::Matrix::inverse`]
//! asks the [`ZeroOracle`](super::ZeroOracle) about `c₀` and, if the oracle returns
//! `ProbablyZero`, refuses to invert rather than fabricate an inverse of a matrix it
//! cannot show to be invertible.

use crate::poly::{Coeff, MultiPoly, coeff_from_i64, coeff_one, coeff_recip};

use super::MatrixError;
use super::ring::PolyMatrix;

/// The output of the Faddeev–LeVerrier recurrence.
pub(crate) struct Faddeev {
    /// `coeffs[k]` is `c_k`, ascending; `coeffs[n] = 1`. Length `n + 1`.
    pub coeffs: Vec<MultiPoly>,
    /// The final matrix `M_n`, which satisfies `A · M_n = −c₀ · I`.
    pub m_final: PolyMatrix,
}

impl Faddeev {
    /// `det(A) = (−1)ⁿ · c₀`.
    pub(crate) fn determinant(&self, n: usize) -> Result<MultiPoly, MatrixError> {
        let c0 = self.coeffs.first().ok_or(MatrixError::Empty)?.clone();
        if n.is_multiple_of(2) {
            Ok(c0)
        } else {
            c0.neg().map_err(MatrixError::Poly)
        }
    }
}

/// Run the Faddeev–LeVerrier recurrence on `a`.
///
/// # Errors
///
/// [`MatrixError::NotSquare`] for a non-square matrix; [`MatrixError::Poly`] if the
/// ring arithmetic fails.
pub(crate) fn faddeev_leverrier(a: &PolyMatrix) -> Result<Faddeev, MatrixError> {
    if a.rows != a.cols {
        return Err(MatrixError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }
    let n = a.rows;
    let num_vars = a.num_vars;

    if n == 0 {
        return Ok(Faddeev {
            coeffs: vec![MultiPoly::constant(coeff_one(), num_vars)],
            m_final: PolyMatrix::zeros(0, 0, num_vars),
        });
    }

    // c_n = 1; the rest are filled in from the top down.
    let mut coeffs: Vec<MultiPoly> = vec![MultiPoly::zero(num_vars); n + 1];
    coeffs[n] = MultiPoly::constant(coeff_one(), num_vars);

    let mut m = PolyMatrix::zeros(n, n, num_vars);

    for k in 1..=n {
        // M_k = A · M_{k−1} + c_{n−k+1} · I
        let mut next = a.mul(&m)?;
        next.add_scalar_diagonal(&coeffs[n - k + 1])?;

        // c_{n−k} = −(1/k) · tr(A · M_k)
        let trace = a.mul(&next)?.trace()?;
        let inverse_k: Coeff = coeff_recip(&coeff_from_i64(k as i64))
            .ok_or(MatrixError::Poly(crate::poly::PolyError::DivByZero))?;
        let negated = trace.neg().map_err(MatrixError::Poly)?;
        coeffs[n - k] = negated.scale(&inverse_k).map_err(MatrixError::Poly)?;

        m = next;
    }

    Ok(Faddeev { coeffs, m_final: m })
}

/// Evaluate `p(A) = Σ_k c_k · Aᵏ` — the Cayley–Hamilton residual.
///
/// Cayley–Hamilton is a theorem over any commutative ring, so this **must** come out
/// as the exactly-zero matrix. It is therefore a genuine end-to-end check of the
/// charpoly implementation, not merely a plausibility test: any bug in the
/// recurrence, in the polynomial arithmetic or in the atomization shows up here as a
/// nonzero polynomial.
pub(crate) fn cayley_hamilton_residual(
    a: &PolyMatrix,
    coeffs: &[MultiPoly],
) -> Result<PolyMatrix, MatrixError> {
    let n = a.rows;
    let num_vars = a.num_vars;
    let mut power = PolyMatrix::identity(n, num_vars); // A⁰
    let mut acc = PolyMatrix::zeros(n, n, num_vars);

    for c in coeffs {
        let term = scale_matrix(&power, c)?;
        acc = acc.add(&term)?;
        power = a.mul(&power)?;
    }
    Ok(acc)
}

/// Multiply every entry of `m` by the ring element `c`.
fn scale_matrix(m: &PolyMatrix, c: &MultiPoly) -> Result<PolyMatrix, MatrixError> {
    let mut out = PolyMatrix::zeros(m.rows, m.cols, m.num_vars);
    for i in 0..m.rows {
        for j in 0..m.cols {
            out.set(i, j, m.at(i, j).mul(c).map_err(MatrixError::Poly)?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::matrix::atoms::AtomSpace;

    fn poly_matrix(m: &Matrix) -> (AtomSpace, PolyMatrix) {
        m.to_poly_matrix().expect("polynomializable")
    }

    #[test]
    fn charpoly_of_2x2_numeric() {
        // [[1,2],[3,4]] → λ² − 5λ − 2
        let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("dims");
        let (_space, pm) = poly_matrix(&m);
        let fl = faddeev_leverrier(&pm).expect("fl");
        let c: Vec<f64> = fl
            .coeffs
            .iter()
            .map(|p| {
                crate::matrix::ring::as_constant(p)
                    .map_or(f64::NAN, |c| crate::poly::ratio_to_f64(&c))
            })
            .collect();
        assert!((c[0] - (-2.0)).abs() < 1e-12, "c0 = {}", c[0]);
        assert!((c[1] - (-5.0)).abs() < 1e-12, "c1 = {}", c[1]);
        assert!((c[2] - 1.0).abs() < 1e-12, "c2 = {}", c[2]);
    }

    #[test]
    fn cayley_hamilton_is_exactly_zero() {
        let m =
            Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
        let (_space, pm) = poly_matrix(&m);
        let fl = faddeev_leverrier(&pm).expect("fl");
        let residual = cayley_hamilton_residual(&pm, &fl.coeffs).expect("residual");
        assert!(
            residual.data.iter().all(MultiPoly::is_zero),
            "p(A) must vanish identically"
        );
    }

    #[test]
    fn inverse_identity_a_times_m_final_is_minus_c0_i() {
        let m =
            Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
        let (_space, pm) = poly_matrix(&m);
        let fl = faddeev_leverrier(&pm).expect("fl");
        let product = pm.mul(&fl.m_final).expect("mul");
        let minus_c0 = fl.coeffs[0].neg().expect("neg");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    minus_c0.clone()
                } else {
                    MultiPoly::zero(pm.num_vars)
                };
                assert_eq!(*product.at(i, j), expected, "entry ({i},{j})");
            }
        }
    }
}
