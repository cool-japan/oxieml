//! Polynomial factorization over ℚ: Yun square-free decomposition + Zassenhaus.
//!
//! Every step here is exact. [`square_free_decomposition`] separates the
//! multiplicities with exact GCDs over ℚ, and each square-free part is then
//! handed to [`super::factor_zassenhaus`], which factors it completely for
//! **arbitrary degree** via Cantor–Zassenhaus modular factorization, quadratic
//! Hensel lifting and Zassenhaus recombination.
//!
//! # Normal form
//!
//! `factor` returns `f = content · Π factorᵢ^multᵢ` where
//!
//! * `content` is the exact rational content of `f`, carrying the sign of its
//!   leading coefficient, and
//! * every `factorᵢ` is a **primitive integer** polynomial with a positive
//!   leading coefficient, irreducible over ℚ.
//!
//! Gauss' lemma guarantees these two facts are compatible: the primitive part of
//! `f` is exactly the product of the primitive irreducible factors, with no
//! leftover constant.

use super::factor_zassenhaus::factor_square_free;
use super::univariate::Poly;
use super::{Coeff, PolyError, coeff_recip, coeff_zero};

/// Result of polynomial factorization: `f = content · Π factorᵢ^multᵢ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factorization {
    /// The exact rational content of `f`, signed like its leading coefficient.
    pub content: Coeff,
    /// Irreducible factors with multiplicities: `(factor, multiplicity)`.
    ///
    /// Each factor is a primitive integer polynomial with a positive leading
    /// coefficient. Factors are sorted by degree, then lexicographically by
    /// coefficients, so the output is canonical.
    pub factors: Vec<(Poly, usize)>,
}

/// Yun's square-free decomposition.
///
/// Returns monic square-free factors paired with their multiplicity, such that
/// the monic part of `f` is `Π factorᵢ^multᵢ` with the factors pairwise coprime.
///
/// # Errors
///
/// Propagates division errors from the internal exact GCD chain.
pub fn square_free_decomposition(f: &Poly) -> Result<Vec<(Poly, usize)>, PolyError> {
    let f = f.normalized();
    if f.is_zero() {
        return Ok(Vec::new());
    }

    let lc_inv = match coeff_recip(&f.leading_coeff()) {
        Some(inv) => inv,
        None => return Ok(Vec::new()),
    };
    let f_monic = f.scale(&lc_inv)?;

    let df = f_monic.diff()?;
    if df.is_zero() {
        return Ok(vec![(f_monic, 1)]);
    }

    let a0 = Poly::gcd(&f_monic, &df)?;
    let (mut b, rem) = f_monic.div_rem(&a0)?;
    if !rem.is_zero() {
        return Ok(vec![(f_monic, 1)]);
    }

    let (mut c, rem2) = df.div_rem(&a0)?;
    if !rem2.is_zero() {
        return Ok(vec![(f_monic, 1)]);
    }

    let mut factors: Vec<(Poly, usize)> = Vec::new();
    let mut i = 1usize;

    loop {
        b.normalize();
        if b.is_zero() || b.degree() == Some(0) {
            break;
        }

        let db = b.diff()?;
        let d = c.sub(&db)?;

        if d.is_zero() {
            if let Some(inv) = coeff_recip(&b.leading_coeff()) {
                factors.push((b.scale(&inv)?, i));
            }
            break;
        }

        let a = Poly::gcd(&b, &d)?;

        if a.degree().unwrap_or(0) >= 1 {
            if let Some(inv) = coeff_recip(&a.leading_coeff()) {
                factors.push((a.scale(&inv)?, i));
            }
        }

        let (new_b, rem_b) = b.div_rem(&a)?;
        let (new_c, rem_c) = d.div_rem(&a)?;

        if !rem_b.is_zero() || !rem_c.is_zero() {
            return Ok(vec![(f_monic, 1)]);
        }

        b = new_b;
        c = new_c;
        i += 1;
    }

    if factors.is_empty() {
        return Ok(vec![(f_monic, 1)]);
    }

    Ok(factors)
}

impl Poly {
    /// Factor this polynomial into irreducibles over ℚ.
    ///
    /// Works for **arbitrary degree**. The multiplicities come from Yun's
    /// square-free decomposition; every square-free part is then split into
    /// irreducibles by [`super::factor_zassenhaus::factor_square_free`]
    /// (distinct-degree factorization → Cantor–Zassenhaus → Hensel lifting →
    /// Zassenhaus recombination).
    ///
    /// The result satisfies `self == content · Π factorᵢ^multᵢ` exactly, with
    /// primitive integer factors of positive leading coefficient.
    ///
    /// ```
    /// use oxieml::Poly;
    /// // (x² + 1)(x² + x + 1) — no rational roots, both factors irreducible.
    /// let f = Poly::from_int_coeffs(&[1, 1, 2, 1, 1]);
    /// let factored = f.factor().unwrap();
    /// assert_eq!(factored.factors.len(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates [`PolyError::FactorizationLimit`] when the Zassenhaus pipeline
    /// exhausts one of its documented budgets (lucky-prime search,
    /// Cantor–Zassenhaus attempts, or the recombination subset cap). It never
    /// returns a wrong factorization and never claims an unproven irreducibility.
    pub fn factor(&self) -> Result<Factorization, PolyError> {
        let f = self.normalized();
        if f.is_zero() {
            return Ok(Factorization {
                content: coeff_zero(),
                factors: Vec::new(),
            });
        }

        // `f = content · primitive_part(f)` requires the content to carry the
        // sign of the leading coefficient: `content()` is always positive, and
        // `primitive_part()` forces a positive leading coefficient.
        let mut content = f.content();
        if f.leading_coeff() < coeff_zero() {
            content = -content;
        }
        let primitive = f.primitive_part()?;

        // Yun returns *monic* square-free parts. Their primitive parts multiply
        // back to `primitive` exactly (Gauss: a product of primitives is
        // primitive, and both sides have a positive leading coefficient), so no
        // stray constant has to be tracked here.
        let sfd = square_free_decomposition(&primitive)?;

        let mut all_factors: Vec<(Poly, usize)> = Vec::new();
        for (sf_factor, mult) in sfd {
            let sf_primitive = sf_factor.primitive_part()?;
            for irred in factor_square_free(&sf_primitive)? {
                all_factors.push((irred, mult));
            }
        }

        // Canonical order: by degree, then lexicographically by coefficients.
        all_factors.sort_by(|a, b| {
            a.0.coeffs
                .len()
                .cmp(&b.0.coeffs.len())
                .then_with(|| a.0.coeffs.cmp(&b.0.coeffs))
        });

        Ok(Factorization {
            content,
            factors: all_factors,
        })
    }
}
