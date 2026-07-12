//! Bareiss fraction-free elimination: exact determinant, echelon form, rref,
//! nullspace and rank.
//!
//! # The Bareiss identity
//!
//! Ordinary Gaussian elimination over a field divides by the pivot at every step,
//! which over `ℚ[atoms]` immediately drags us into the field of fractions and makes
//! the entries swell into unreduced rational functions. Bareiss (1968) avoids this
//! entirely. Write `M⁽ᵏ⁾` for the working matrix after `k` elimination steps and
//! `p_k = M⁽ᵏ⁾[k][k]` for the `k`-th pivot, with `p_{-1} = 1`. The update
//!
//! ```text
//!                    p_k · M⁽ᵏ⁾[i][j]  −  M⁽ᵏ⁾[i][k] · M⁽ᵏ⁾[k][j]
//!   M⁽ᵏ⁺¹⁾[i][j]  =  ───────────────────────────────────────────────
//!                                      p_{k−1}
//! ```
//!
//! is **exact in the ring**: the division by the previous pivot always leaves no
//! remainder. The reason is Sylvester's identity — every intermediate entry is
//! itself a *minor determinant* of the original matrix,
//!
//! ```text
//!   M⁽ᵏ⁺¹⁾[i][j] = det A[ rows 0..k, i | cols 0..k, j ]
//! ```
//!
//! and the numerator above is that minor multiplied by `p_{k−1}` (which is the
//! `k×k` leading minor). So the quotient is a determinant, hence an element of the
//! ring, and no denominators are ever created. The intermediate entries stay as
//! small as determinants can be — no expression swell beyond what the answer itself
//! requires. After `n−1` steps `det A = ± M⁽ⁿ⁻¹⁾[n−1][n−1]`, the sign being the
//! parity of the row swaps.
//!
//! The identity holds over **any integral domain**, which is exactly why we first
//! atomize the transcendental entries into `ℚ[atoms]` (see [`super::atoms`]).
//!
//! # Zero tests: where the oracle is and is not needed
//!
//! * [`bareiss_det`] needs to know only whether a pivot candidate is the zero
//!   *polynomial*, which is an **exact** test. Determinant is a polynomial function
//!   of the entries, so it commutes with the evaluation homomorphism and the answer
//!   is unconditionally correct — even when the atoms satisfy hidden relations. No
//!   oracle, no assumptions, no uncertainty.
//!
//! * [`echelon`] (and therefore rref / nullspace / rank) **divides by pivots** and
//!   scales rows by them, so it needs pivots that are nonzero *as functions*, not
//!   merely as polynomials. That is undecidable, so it consults the
//!   [`ZeroOracle`](super::ZeroOracle) and, when the oracle cannot decide, records a
//!   [`Assumption`] instead of guessing.

use crate::lower::LoweredOp;
use crate::poly::{Coeff, MultiPoly, coeff_is_zero, coeff_one, coeff_recip};

use super::atoms::AtomSpace;
use super::ring::{PolyMatrix, content, exact_div, rational_gcd};
use super::zero::{ZeroOracle, ZeroVerdict};
use super::{AssumedRelation, Assumption, MatrixError};

/// Exact determinant by Bareiss fraction-free elimination.
///
/// The result is the determinant **in the ring** `ℚ[atoms]`, which — because the
/// determinant is a polynomial in the entries and atomization is a ring
/// homomorphism — is also the determinant of the original matrix of functions. This
/// is unconditional: it does not depend on any zero test that might have been wrong.
///
/// # Errors
///
/// [`MatrixError::NotSquare`] for a non-square matrix; [`MatrixError::Poly`] or
/// [`MatrixError::InexactDivision`] only if the ring arithmetic itself fails (the
/// latter being unreachable by Sylvester's identity, and thus a self-check).
pub(crate) fn bareiss_det(mat: &PolyMatrix) -> Result<MultiPoly, MatrixError> {
    if mat.rows != mat.cols {
        return Err(MatrixError::NotSquare {
            rows: mat.rows,
            cols: mat.cols,
        });
    }
    let n = mat.rows;
    let num_vars = mat.num_vars;
    if n == 0 {
        // The determinant of the empty matrix is the empty product, 1.
        return Ok(MultiPoly::constant(coeff_one(), num_vars));
    }

    let mut m = mat.clone();
    let mut sign_is_negative = false;
    let mut previous_pivot = MultiPoly::constant(coeff_one(), num_vars);

    for k in 0..n.saturating_sub(1) {
        if m.at(k, k).is_zero() {
            // Exact test: no oracle needed, and none would help — a pivot that is
            // nonzero in the ring is a legal divisor whether or not it happens to
            // vanish as a function.
            let Some(swap_row) = (k + 1..n).find(|&r| !m.at(r, k).is_zero()) else {
                // The whole trailing column is zero in the ring ⇒ the working matrix
                // is singular ⇒ det = 0.
                return Ok(MultiPoly::zero(num_vars));
            };
            m.swap_rows(k, swap_row);
            sign_is_negative = !sign_is_negative;
        }

        let pivot = m.at(k, k).clone();
        for i in k + 1..n {
            let below = m.at(i, k).clone();
            for j in k + 1..n {
                let cross = pivot.mul(m.at(i, j)).map_err(MatrixError::Poly)?;
                let anti = below.mul(m.at(k, j)).map_err(MatrixError::Poly)?;
                let numerator = cross.sub(&anti).map_err(MatrixError::Poly)?;
                // Exact by Sylvester's identity — the quotient is a minor.
                let updated = exact_div(&numerator, &previous_pivot)?;
                m.set(i, j, updated);
            }
            m.set(i, k, MultiPoly::zero(num_vars));
        }
        previous_pivot = pivot;
    }

    let det = m.at(n - 1, n - 1).clone();
    if sign_is_negative {
        det.neg().map_err(MatrixError::Poly)
    } else {
        Ok(det)
    }
}

/// A fraction-free row-echelon form together with everything the callers need.
pub(crate) struct Echelon {
    /// The echelon matrix. Each row is the original row scaled by a nonzero ring
    /// element (a product of pivots), which is harmless: it does not change the row
    /// space, the pivot positions, or the nullspace.
    pub matrix: PolyMatrix,
    /// `pivot_cols[r]` is the pivot column of echelon row `r`.
    pub pivot_cols: Vec<usize>,
    /// Assumptions the oracle forced us to make. Empty ⇒ the result is *proved*.
    pub assumptions: Vec<Assumption>,
}

/// Choose a pivot in `col` among rows `from_row..rows`.
///
/// Returns the index of a row whose entry is **proved** nonzero as a function, or
/// `None` when no such row exists. In the `None` case any entries the oracle could
/// not decide are recorded as [`Assumption`]s: to declare the column pivot-free we
/// are *assuming* those entries vanish identically, and the caller must be told.
fn select_pivot(
    m: &PolyMatrix,
    from_row: usize,
    col: usize,
    space: &AtomSpace,
    oracle: &ZeroOracle,
    assumptions: &mut Vec<Assumption>,
) -> Option<usize> {
    let mut undecided: Vec<(usize, super::zero::ProbeEvidence)> = Vec::new();

    for r in from_row..m.rows {
        let entry = m.at(r, col);
        if entry.is_zero() {
            continue; // proved zero in the ring ⇒ proved zero as a function
        }
        match oracle.test_poly(entry, space) {
            ZeroVerdict::NonZero(_) => return Some(r),
            ZeroVerdict::Zero(_) => {}
            ZeroVerdict::ProbablyZero(evidence) => undecided.push((r, evidence)),
        }
    }

    // No provably-nonzero entry. Every remaining candidate is one the oracle could
    // not decide; treating the column as pivot-free means assuming they all vanish.
    for (r, evidence) in undecided {
        assumptions.push(Assumption {
            expr: space.poly_to_lowered(m.at(r, col)).simplify(),
            assumed: AssumedRelation::IdenticallyZero,
            context: format!(
                "fraction-free elimination: column {col} row {r} has no provably-nonzero pivot; \
                 the column is treated as pivot-free, which assumes this entry is identically zero"
            ),
            evidence,
        });
    }
    None
}

/// Fraction-free (Bareiss) forward elimination to row-echelon form.
///
/// Works for rectangular matrices and skips columns that have no pivot. The
/// division by the previous pivot stays exact: with column skipping the entries are
/// still the minors `det A[pivot rows, i | pivot cols, j]`, so Sylvester's identity
/// applies verbatim.
///
/// Every column of every row below the current pivot row is updated — including the
/// free (skipped) columns — because the nullspace basis reads its coefficients out
/// of exactly those columns.
pub(crate) fn echelon(
    mat: &PolyMatrix,
    space: &AtomSpace,
    oracle: &ZeroOracle,
) -> Result<Echelon, MatrixError> {
    let mut m = mat.clone();
    let rows = m.rows;
    let cols = m.cols;
    let num_vars = m.num_vars;

    let mut assumptions: Vec<Assumption> = Vec::new();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut previous_pivot = MultiPoly::constant(coeff_one(), num_vars);
    let mut row = 0usize;

    for col in 0..cols {
        if row >= rows {
            break;
        }
        let Some(pivot_row) = select_pivot(&m, row, col, space, oracle, &mut assumptions) else {
            continue; // free column
        };
        m.swap_rows(row, pivot_row);

        let pivot = m.at(row, col).clone();
        for i in row + 1..rows {
            let below = m.at(i, col).clone();
            for j in 0..cols {
                if j == col {
                    continue;
                }
                let cross = pivot.mul(m.at(i, j)).map_err(MatrixError::Poly)?;
                let anti = below.mul(m.at(row, j)).map_err(MatrixError::Poly)?;
                let numerator = cross.sub(&anti).map_err(MatrixError::Poly)?;
                let updated = exact_div(&numerator, &previous_pivot)?;
                m.set(i, j, updated);
            }
            m.set(i, col, MultiPoly::zero(num_vars));
        }

        previous_pivot = pivot;
        pivot_cols.push(col);
        row += 1;
    }

    Ok(Echelon {
        matrix: m,
        pivot_cols,
        assumptions,
    })
}

/// A reduced row-echelon form, in polynomial (numerator) form.
///
/// Row `r` of `matrix` is the RREF row *scaled* by its own pivot entry
/// `matrix[r][pivot_cols[r]]`, which is nonzero by construction. Dividing row `r`
/// through by that entry gives the true RREF; [`super::Matrix::rref_certified`] does
/// exactly that when it renders the answer back into `LoweredOp`.
pub(crate) struct ReducedEchelon {
    /// The back-substituted echelon matrix (unnormalised rows).
    pub matrix: PolyMatrix,
    /// `pivot_cols[r]` is the pivot column of row `r`.
    pub pivot_cols: Vec<usize>,
    /// Assumptions forced by an undecidable pivot; empty ⇒ proved.
    pub assumptions: Vec<Assumption>,
}

/// Back-substitute an echelon form into a (still fraction-free) reduced echelon form.
///
/// For each pivot row `r` from the bottom up, and each row `i` above it, we replace
///
/// ```text
///   row_i  ←  p_r · row_i  −  row_i[c_r] · row_r
/// ```
///
/// which is a pure ring operation — no division, hence no fractions — and zeroes
/// `row_i[c_r]`. Scaling `row_i` by the nonzero pivot `p_r` rescales the whole row,
/// which is exactly what an RREF row is defined up to. Rows are then reduced by
/// their rational content to keep the integers from growing without need; this is a
/// nonzero rational rescaling and likewise harmless here (it would *not* be
/// harmless inside [`bareiss_det`], where it would corrupt both the determinant and
/// the exact-division invariant, and it is not used there).
pub(crate) fn reduced_echelon(
    mat: &PolyMatrix,
    space: &AtomSpace,
    oracle: &ZeroOracle,
) -> Result<ReducedEchelon, MatrixError> {
    let Echelon {
        mut matrix,
        pivot_cols,
        assumptions,
    } = echelon(mat, space, oracle)?;

    let cols = matrix.cols;
    for r in (1..pivot_cols.len()).rev() {
        let pivot_col = pivot_cols[r];
        let pivot = matrix.at(r, pivot_col).clone();
        for i in 0..r {
            let factor = matrix.at(i, pivot_col).clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..cols {
                let scaled = pivot.mul(matrix.at(i, j)).map_err(MatrixError::Poly)?;
                let subtracted = factor.mul(matrix.at(r, j)).map_err(MatrixError::Poly)?;
                let updated = scaled.sub(&subtracted).map_err(MatrixError::Poly)?;
                matrix.set(i, j, updated);
            }
            // Row `i` is only defined up to a nonzero scalar here; strip the content
            // so the coefficients do not grow across successive back-substitutions.
            let row: Vec<MultiPoly> = (0..cols).map(|j| matrix.at(i, j).clone()).collect();
            let row_content_free = strip_row_content(&row)?;
            for (j, entry) in row_content_free.into_iter().enumerate() {
                matrix.set(i, j, entry);
            }
        }
    }

    Ok(ReducedEchelon {
        matrix,
        pivot_cols,
        assumptions,
    })
}

/// Divide a whole row by the rational content common to *all* of its entries.
///
/// The single common factor is `gcd` over the entries of their individual contents.
/// Reducing each entry by its own content would rescale the entries relative to one
/// another and destroy the row; one shared factor keeps the row proportional and so
/// leaves the row space (and the RREF it defines) untouched.
fn strip_row_content(row: &[MultiPoly]) -> Result<Vec<MultiPoly>, MatrixError> {
    let mut factor: Option<Coeff> = None;
    for entry in row {
        let Some(entry_content) = content(entry) else {
            continue; // zero entry contributes nothing
        };
        factor = Some(match factor {
            None => entry_content,
            Some(current) => rational_gcd(&current, &entry_content),
        });
    }
    let Some(factor) = factor else {
        return Ok(row.to_vec()); // the row is entirely zero
    };
    if coeff_is_zero(&factor) {
        return Ok(row.to_vec());
    }
    let inverse =
        coeff_recip(&factor).ok_or(MatrixError::Poly(crate::poly::PolyError::DivByZero))?;
    row.iter()
        .map(|p| p.scale(&inverse).map_err(MatrixError::Poly))
        .collect()
}

/// Render `numerator / denominator` as a `LoweredOp`, cancelling exactly when the
/// division lies in the ring.
pub(crate) fn ratio_to_lowered(
    numerator: &MultiPoly,
    denominator: &MultiPoly,
    space: &AtomSpace,
) -> Result<LoweredOp, MatrixError> {
    if numerator.is_zero() {
        return Ok(LoweredOp::Const(0.0));
    }
    if let Ok(quotient) = exact_div(numerator, denominator) {
        return Ok(space.poly_to_lowered(&quotient).simplify());
    }
    // The quotient is a genuine rational function: keep it as an explicit division.
    // It is exact — nothing is rounded away — merely unsimplified.
    let num = space.poly_to_lowered(numerator);
    let den = space.poly_to_lowered(denominator);
    Ok(LoweredOp::Div(std::sync::Arc::new(num), std::sync::Arc::new(den)).simplify())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::poly::coeff_from_i64;

    fn numeric(rows: usize, cols: usize, values: &[f64]) -> Matrix {
        Matrix::from_f64(rows, cols, values).expect("dimensions")
    }

    #[test]
    fn det_of_2x2_numeric() {
        let m = numeric(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let det = m.det().expect("det");
        assert_eq!(det, LoweredOp::Const(-2.0));
    }

    #[test]
    fn det_of_identity_is_one() {
        let m = Matrix::identity(5);
        let det = m.det().expect("det");
        assert_eq!(det, LoweredOp::Const(1.0));
    }

    #[test]
    fn exact_rational_determinant() {
        // [[1/3, 1/5], [1/7, 1/11]] → 1/33 - 1/35 = (35 - 33)/1155 = 2/1155
        let m = numeric(2, 2, &[1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0, 1.0 / 11.0]);
        let det = m.det_exact().expect("exact det");
        let expected = crate::poly::Coeff::new(
            coeff_from_i64(2).numer().clone(),
            coeff_from_i64(1155).numer().clone(),
        );
        assert_eq!(det, expected);
    }
}
