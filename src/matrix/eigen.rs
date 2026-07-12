//! Eigenvalues and eigenvectors.
//!
//! # What is exact and what is not
//!
//! The characteristic polynomial is computed **exactly** (Faddeev–LeVerrier over
//! `ℚ[atoms]`, see [`super::faddeev`]). Its *roots*, however, are algebraic numbers:
//! for `n ≥ 5` there is provably no radical expression for them (Abel–Ruffini), so
//! there is no closed symbolic form to return in general. This module is therefore
//! honest about its scope:
//!
//! * The charpoly must have **constant** coefficients — i.e. the matrix entries are
//!   numeric. Otherwise the roots are not numbers at all but algebraic functions of
//!   the symbols, and we return [`MatrixError::NotNumeric`] rather than inventing
//!   something. (The *exact rational* eigenvalues of a symbolic matrix can still be
//!   obtained by factoring [`super::Matrix::charpoly`] with the `poly` module.)
//!
//! * Given exact rational charpoly coefficients, the roots are found numerically in
//!   ℂ by the crate's existing Durand–Kerner solver, so complex conjugate pairs come
//!   out properly (a rotation matrix gives `{+i, −i}`, not "no real roots").
//!
//! # Eigenvectors: inverse iteration
//!
//! For an eigenvalue `λ`, `A − λI` is singular, so its nullspace cannot be found by
//! naive elimination in floating point — the "zero" pivot is only zero to within
//! rounding, and which pivot is deemed zero is exactly the fragile decision.
//!
//! Instead we use **inverse iteration**, the standard tool: pick a shift `μ = λ + ε`
//! slightly off the eigenvalue, and iterate `v ← normalise((A − μI)⁻¹ v)`. Writing
//! `v` in an eigenbasis, the component along the eigenvector for `λ` is amplified by
//! `1/|λ − μ| = 1/ε` while every other component is amplified only by
//! `1/|λⱼ − μ|`, so the ratio grows like `|λⱼ − λ| / ε` per step and the iterate
//! collapses onto the eigenvector after a handful of steps. The linear solve is a
//! complex LU with partial pivoting; the deliberate `ε` keeps it away from exact
//! singularity, which is precisely what makes the (otherwise alarming) solve of a
//! near-singular system the *right* thing to do here — the growing solution norm is
//! the signal, not noise, and normalising each step keeps it bounded.
//!
//! Several deterministic starting vectors are tried in turn, so that a start
//! accidentally orthogonal to the wanted eigenvector cannot cause a failure. The
//! result is accepted only if the residual `‖A v − λ v‖ / ‖v‖` is small relative to
//! `‖A‖`; otherwise [`MatrixError::NumericFailure`] is returned rather than a
//! plausible-looking vector that is not an eigenvector.

use num_complex::Complex;

use crate::poly::{Coeff, Poly, ratio_to_f64};
use crate::solve_poly::solve_polynomial_complex;

use super::MatrixError;
use super::ring::{PolyMatrix, as_constant};

/// Relative residual accepted for an eigenvector.
const EIGENVECTOR_RESIDUAL_TOLERANCE: f64 = 1e-6;

/// Number of inverse-iteration steps per starting vector.
const INVERSE_ITERATION_STEPS: usize = 24;

/// Number of distinct starting vectors tried before giving up.
const INVERSE_ITERATION_RESTARTS: usize = 4;

/// An eigenvalue together with a corresponding unit eigenvector.
#[derive(Clone, Debug, PartialEq)]
pub struct Eigenpair {
    /// The eigenvalue `λ` (possibly complex).
    pub value: Complex<f64>,
    /// A unit eigenvector `v` with `A v ≈ λ v`.
    pub vector: Vec<Complex<f64>>,
    /// The achieved relative residual `‖A v − λ v‖ / ‖A‖`, reported so the caller can
    /// judge the quality of the vector rather than having to trust it.
    pub residual: f64,
}

/// Extract the exact rational charpoly from Faddeev–LeVerrier coefficients.
///
/// # Errors
///
/// [`MatrixError::NotNumeric`] when any coefficient is a genuine symbolic
/// expression, in which case the eigenvalues are algebraic *functions* and no
/// numeric root-finder applies.
pub(crate) fn exact_charpoly(coeffs: &[crate::poly::MultiPoly]) -> Result<Poly, MatrixError> {
    let mut exact: Vec<Coeff> = Vec::with_capacity(coeffs.len());
    for c in coeffs {
        exact.push(as_constant(c).ok_or(MatrixError::NotNumeric)?);
    }
    Ok(Poly::from_ratios(exact))
}

/// All eigenvalues of a numeric matrix, as the roots of its exact charpoly.
pub(crate) fn eigenvalues_of(charpoly: &Poly) -> Result<Vec<Complex<f64>>, MatrixError> {
    let roots = solve_polynomial_complex(charpoly).map_err(|_| {
        MatrixError::NumericFailure("characteristic polynomial root-finding did not converge")
    })?;
    Ok(roots.roots)
}

/// The numeric (complex) matrix of a `PolyMatrix` whose entries are all constants.
pub(crate) fn numeric_entries(m: &PolyMatrix) -> Result<Vec<Complex<f64>>, MatrixError> {
    m.data
        .iter()
        .map(|p| {
            as_constant(p)
                .map(|c| Complex::new(ratio_to_f64(&c), 0.0))
                .ok_or(MatrixError::NotNumeric)
        })
        .collect()
}

/// Eigenvectors by inverse iteration, one per eigenvalue.
pub(crate) fn eigenvectors_of(
    a: &[Complex<f64>],
    n: usize,
    eigenvalues: &[Complex<f64>],
) -> Result<Vec<Eigenpair>, MatrixError> {
    let norm_a = matrix_norm(a);
    let scale = if norm_a > 0.0 { norm_a } else { 1.0 };
    let mut pairs = Vec::with_capacity(eigenvalues.len());

    for &lambda in eigenvalues {
        let mut best: Option<(Vec<Complex<f64>>, f64)> = None;

        for restart in 0..INVERSE_ITERATION_RESTARTS {
            // Deliberately offset shift: keeps A − μI invertible while still being
            // far closer to λ than to any other eigenvalue.
            let epsilon = scale * 1e-9 * f64::from(1u8 + restart as u8);
            let mu = lambda + Complex::new(epsilon, epsilon / 2.0);

            let mut shifted = a.to_vec();
            for i in 0..n {
                shifted[i * n + i] -= mu;
            }
            let Some(lu) = lu_factorize(&shifted, n) else {
                continue;
            };

            let mut v = starting_vector(n, restart);
            for _ in 0..INVERSE_ITERATION_STEPS {
                let Some(w) = lu.solve(&v) else { break };
                let Some(normalised) = normalise(&w) else {
                    break;
                };
                v = normalised;
            }

            let residual = eigen_residual(a, n, &v, lambda) / scale;
            if best.as_ref().is_none_or(|(_, r)| residual < *r) {
                best = Some((v, residual));
            }
            if residual <= EIGENVECTOR_RESIDUAL_TOLERANCE {
                break;
            }
        }

        match best {
            Some((vector, residual)) if residual <= EIGENVECTOR_RESIDUAL_TOLERANCE => {
                pairs.push(Eigenpair {
                    value: lambda,
                    vector: canonical_phase(&vector),
                    residual,
                });
            }
            _ => {
                return Err(MatrixError::NumericFailure(
                    "inverse iteration did not converge to an eigenvector (the matrix may be \
                     severely defective or ill-conditioned)",
                ));
            }
        }
    }
    Ok(pairs)
}

/// Frobenius norm.
fn matrix_norm(a: &[Complex<f64>]) -> f64 {
    a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// `‖A v − λ v‖₂`.
fn eigen_residual(a: &[Complex<f64>], n: usize, v: &[Complex<f64>], lambda: Complex<f64>) -> f64 {
    let mut total = 0.0;
    for i in 0..n {
        let mut acc = Complex::new(0.0, 0.0);
        for j in 0..n {
            acc += a[i * n + j] * v[j];
        }
        acc -= lambda * v[i];
        total += acc.norm_sqr();
    }
    total.sqrt()
}

/// Scale a vector to unit 2-norm; `None` if it is (numerically) the zero vector.
fn normalise(v: &[Complex<f64>]) -> Option<Vec<Complex<f64>>> {
    let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    Some(v.iter().map(|z| z / norm).collect())
}

/// Fix the phase so the eigenvector is reproducible: rotate the largest-magnitude
/// component onto the positive real axis.
fn canonical_phase(v: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let Some(pivot) = v
        .iter()
        .copied()
        .max_by(|a, b| {
            a.norm_sqr()
                .partial_cmp(&b.norm_sqr())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|z| z.norm() > 0.0)
    else {
        return v.to_vec();
    };
    let phase = pivot / pivot.norm();
    v.iter().map(|z| z / phase).collect()
}

/// Deterministic starting vectors for inverse iteration.
///
/// The `k`-th start puts a `1` in position `k mod n` and a small deterministic
/// ramp elsewhere, so successive restarts explore genuinely different directions and
/// no eigenvector can be missed by an unlucky orthogonal start.
fn starting_vector(n: usize, k: usize) -> Vec<Complex<f64>> {
    let mut v: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let re = 1.0 / (1.0 + i as f64 + k as f64);
            let im = 1.0 / (2.0 + 2.0 * i as f64 + 3.0 * k as f64);
            Complex::new(re, im)
        })
        .collect();
    if n > 0 {
        v[k % n] += Complex::new(1.0, 0.0);
    }
    normalise(&v).unwrap_or(v)
}

/// A complex LU factorisation with partial pivoting.
struct ComplexLu {
    /// Combined L (unit lower) and U (upper), row-major, `n × n`.
    lu: Vec<Complex<f64>>,
    /// Row permutation.
    perm: Vec<usize>,
    /// Dimension.
    n: usize,
}

impl ComplexLu {
    /// Solve `A x = b` from the factorisation. `None` if a pivot underflowed to zero.
    fn solve(&self, b: &[Complex<f64>]) -> Option<Vec<Complex<f64>>> {
        let n = self.n;
        let mut y = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            let mut acc = b[self.perm[i]];
            for (j, yj) in y.iter().enumerate().take(i) {
                acc -= self.lu[i * n + j] * yj;
            }
            y[i] = acc;
        }
        let mut x = vec![Complex::new(0.0, 0.0); n];
        for i in (0..n).rev() {
            let mut acc = y[i];
            for (j, xj) in x.iter().enumerate().take(n).skip(i + 1) {
                acc -= self.lu[i * n + j] * xj;
            }
            let pivot = self.lu[i * n + i];
            if pivot.norm() == 0.0 {
                return None;
            }
            x[i] = acc / pivot;
        }
        if x.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
            Some(x)
        } else {
            None
        }
    }
}

/// LU factorise a complex matrix with partial pivoting. `None` if it is exactly
/// singular (which inverse iteration avoids by construction, via the shift).
fn lu_factorize(a: &[Complex<f64>], n: usize) -> Option<ComplexLu> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let (pivot_row, pivot_norm) = (k..n).fold((k, 0.0f64), |(best, best_norm), r| {
            let norm = lu[r * n + k].norm();
            if norm > best_norm {
                (r, norm)
            } else {
                (best, best_norm)
            }
        });
        if pivot_norm == 0.0 {
            return None;
        }
        if pivot_row != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot_row * n + j);
            }
            perm.swap(k, pivot_row);
        }
        let pivot = lu[k * n + k];
        for i in k + 1..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in k + 1..n {
                let update = factor * lu[k * n + j];
                lu[i * n + j] -= update;
            }
        }
    }
    Some(ComplexLu { lu, perm, n })
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn rotation_matrix_has_eigenvalues_plus_minus_i() {
        // [[0, -1], [1, 0]] — charpoly λ² + 1
        let m = Matrix::from_f64(2, 2, &[0.0, -1.0, 1.0, 0.0]).expect("dims");
        let values = m.eigenvalues().expect("eigenvalues");
        assert_eq!(values.len(), 2);
        let has = |re: f64, im: f64| {
            values
                .iter()
                .any(|z| (z.re - re).abs() < 1e-9 && (z.im - im).abs() < 1e-9)
        };
        assert!(has(0.0, 1.0), "expected +i in {values:?}");
        assert!(has(0.0, -1.0), "expected -i in {values:?}");
    }

    #[test]
    fn diagonal_eigenvectors_are_axis_aligned() {
        let m = Matrix::from_f64(2, 2, &[3.0, 0.0, 0.0, 5.0]).expect("dims");
        let pairs = m.eigenvectors().expect("eigenvectors");
        assert_eq!(pairs.len(), 2);
        for pair in &pairs {
            assert!(pair.residual < 1e-8, "residual {}", pair.residual);
        }
    }
}
