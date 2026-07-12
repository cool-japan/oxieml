//! `apart` / `together` — partial-fraction decomposition and its inverse.
//!
//! # Partial fractions over ℚ
//!
//! Let `f = num/den` be a univariate rational function in `x`. After extracting
//! the polynomial part by long division (`num = q·den + r`, `deg r < deg den`)
//! and cancelling any common factor (`r/den → r̃/d̃` with `gcd(r̃, d̃) = 1`), the
//! denominator is factored over ℚ into monic irreducibles:
//!
//! ```text
//! d̃ = K · Π_i p_i(x)^{m_i}        (K = lc(d̃),  p_i monic, irreducible/ℚ)
//! ```
//!
//! The unique partial-fraction expansion is then
//!
//! ```text
//! r̃/d̃ = Σ_i Σ_{j=1}^{m_i}  N_{i,j}(x) / p_i(x)^j ,     deg N_{i,j} < deg p_i .
//! ```
//!
//! The number of unknown numerator coefficients is `Σ_i m_i·deg p_i = deg d̃`,
//! exactly matching the `deg d̃` linear conditions obtained from the polynomial
//! identity `r̃ = Σ_{i,j} N_{i,j}·(d̃/p_i^j)`. Following the discipline already
//! used by the rational-integration engine, the conditions are sampled at
//! `deg d̃` rational points and solved as a dense `f64` system with the shared
//! [`gaussian_eliminate`](super::gaussian_eliminate); the recovered coefficients
//! are then rationalised (the exact-`f64→ℚ` map recovers clean fractions such as
//! `1/3`) and the reconstruction is checked against `r̃/d̃` before being
//! accepted. A failed check yields `None`, never a wrong decomposition.
//!
//! # `together`
//!
//! The inverse folds a sum of fractions back to a single reduced fraction by
//! walking the tree as a rational function ([`as_rational`]) and cancelling the
//! polynomial `gcd` of numerator and denominator, then normalising the
//! denominator to be monic.
//!
//! # Shared with integration
//!
//! [`partial_fractions`] is the single implementation used both by
//! [`LoweredOp::apart`] and by [`crate::integrate`]'s `integrate_rational`,
//! which integrates the decomposition term by term.

use std::collections::BTreeMap;

use crate::lower::LoweredOp;
use crate::poly::{Poly, coeff_one, coeff_recip, coeff_zero, f64_to_ratio};

use super::{gaussian_eliminate, ladd, lconst, ldiv, lpow, sole_var, verify_agrees};

/// One summand of a partial-fraction expansion: `numerator / factor^power`.
pub(crate) struct PfTerm {
    /// The monic irreducible-over-ℚ denominator factor `p_i`.
    pub(crate) factor: Poly,
    /// The power `j ≥ 1` to which `factor` is raised.
    pub(crate) power: usize,
    /// The numerator polynomial `N_{i,j}`, of degree strictly below
    /// `deg(factor)`.
    pub(crate) numerator: Poly,
}

/// A full partial-fraction decomposition of `num/den`.
pub(crate) struct PartialFractions {
    /// The polynomial quotient from long division (`0` for a proper fraction).
    pub(crate) polynomial_part: Poly,
    /// The proper-fraction summands `N_{i,j}/p_i^j`.
    pub(crate) terms: Vec<PfTerm>,
}

/// Decompose the rational function `num/den` (variables all `_var`) into partial
/// fractions over ℚ.
///
/// Returns `None` when `den` is zero, when the denominator cannot be factored
/// within budget, when the sampled linear system is singular, or when the
/// numeric reconstruction of the recovered decomposition disagrees with
/// `num/den`. See the module docs for the algorithm.
pub(crate) fn partial_fractions(num: &Poly, den: &Poly, _var: usize) -> Option<PartialFractions> {
    let den = den.normalized();
    if den.is_zero() {
        return None;
    }
    let num = num.normalized();
    let den_deg = den.degree()?; // Some, since `den` is nonzero.

    // ── Long division: split off the polynomial part ─────────────────────────
    let (polynomial_part, remainder) = match num.degree() {
        Some(nd) if nd >= den_deg => num.div_rem(&den).ok()?,
        _ => (Poly::zero(), num.clone()),
    };
    if remainder.is_zero() {
        return Some(PartialFractions {
            polynomial_part,
            terms: Vec::new(),
        });
    }

    // ── Cancel the common factor so the fraction is in lowest terms ──────────
    let gcd = Poly::gcd(&remainder, &den).ok()?;
    let (rem_reduced, rem_rest) = remainder.div_rem(&gcd).ok()?;
    let (den_reduced, den_rest) = den.div_rem(&gcd).ok()?;
    if !rem_rest.is_zero() || !den_rest.is_zero() {
        return None; // gcd must divide both exactly; otherwise bail honestly.
    }
    let rem_reduced = rem_reduced.normalized();
    let den_reduced = den_reduced.normalized();
    let den_reduced_deg = den_reduced.degree()?;

    // A constant denominator after reduction means the whole thing was a
    // polynomial; fold it into the polynomial part.
    if den_reduced_deg == 0 {
        let inv = coeff_recip(&den_reduced.leading_coeff())?;
        let extra = rem_reduced.scale(&inv).ok()?;
        let polynomial_part = polynomial_part.add(&extra).ok()?;
        return Some(PartialFractions {
            polynomial_part,
            terms: Vec::new(),
        });
    }

    // ── Factor the reduced denominator into monic irreducibles ───────────────
    let factorization = den_reduced.factor().ok()?;
    let mut monic_factors: Vec<(Poly, usize)> = Vec::new();
    for (factor, mult) in &factorization.factors {
        let inv = coeff_recip(&factor.leading_coeff())?;
        monic_factors.push((factor.scale(&inv).ok()?, *mult));
    }
    if monic_factors.is_empty() {
        return None;
    }

    // ── Enumerate the unknown numerator coefficients ─────────────────────────
    // Column (i, j, k): coefficient of x^k in N_{i,j} (factor i, power j).
    let mut columns: Vec<(usize, usize, usize)> = Vec::new();
    for (i, (factor, mult)) in monic_factors.iter().enumerate() {
        let deg = factor.degree()?; // ≥ 1
        for j in 1..=*mult {
            for k in 0..deg {
                columns.push((i, j, k));
            }
        }
    }
    let n_unknowns = columns.len();
    if n_unknowns == 0 || n_unknowns != den_reduced_deg {
        return None;
    }

    // ── Sample points avoiding the real roots of the denominator/factors ─────
    let points = choose_points(n_unknowns, &den_reduced, &monic_factors);
    if points.len() < n_unknowns {
        return None;
    }

    // ── Build and solve the linear system  r̃ = Σ N·(d̃/p_i^j) ───────────────
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n_unknowns);
    let mut rhs: Vec<f64> = Vec::with_capacity(n_unknowns);
    for &pt in points.iter().take(n_unknowns) {
        let den_val = den_reduced.eval_f64(pt);
        let mut row: Vec<f64> = Vec::with_capacity(n_unknowns);
        for &(i, j, k) in &columns {
            let factor_val = monic_factors[i].0.eval_f64(pt);
            if factor_val.abs() < 1e-10 {
                return None; // sample landed on a factor root; abort honestly.
            }
            let basis = pt.powi(k as i32) * den_val / factor_val.powi(j as i32);
            row.push(basis);
        }
        matrix.push(row);
        rhs.push(rem_reduced.eval_f64(pt));
    }
    let solution = gaussian_eliminate(matrix, rhs)?;

    // ── Reassemble the numerators, rationalising the f64 coefficients ────────
    let mut numerator_coeffs: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    for (idx, &(i, j, k)) in columns.iter().enumerate() {
        let deg = monic_factors[i].0.degree().unwrap_or(0);
        let entry = numerator_coeffs
            .entry((i, j))
            .or_insert_with(|| vec![0.0; deg]);
        if let Some(slot) = entry.get_mut(k) {
            *slot = solution[idx];
        }
    }

    let mut terms: Vec<PfTerm> = Vec::new();
    for ((i, j), coeffs_f64) in numerator_coeffs {
        let mut ratios = Vec::with_capacity(coeffs_f64.len());
        for value in &coeffs_f64 {
            ratios.push(f64_to_ratio(*value).ok()?);
        }
        let numerator = Poly::from_ratios(ratios);
        if numerator.is_zero() {
            continue;
        }
        terms.push(PfTerm {
            factor: monic_factors[i].0.clone(),
            power: j,
            numerator,
        });
    }

    // ── Verify the reconstruction before accepting it ────────────────────────
    if !verify_reconstruction(&rem_reduced, &den_reduced, &terms) {
        return None;
    }

    Some(PartialFractions {
        polynomial_part,
        terms,
    })
}

/// Pick sample points at which `den_reduced` and every factor are safely nonzero.
fn choose_points(n: usize, den_reduced: &Poly, monic_factors: &[(Poly, usize)]) -> Vec<f64> {
    const CANDIDATES: [f64; 28] = [
        0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5, 1.5, -1.5, 4.0, -4.0, 5.0, -5.0, 0.25,
        -0.25, 2.5, -2.5, 7.0, -7.0, 0.3, -0.3, 6.0, -6.0, 1.25, -1.25, 3.5,
    ];
    let mut points = Vec::with_capacity(n);
    for &candidate in &CANDIDATES {
        if points.len() >= n {
            break;
        }
        if den_reduced.eval_f64(candidate).abs() < 1e-9 {
            continue;
        }
        if monic_factors
            .iter()
            .any(|(factor, _)| factor.eval_f64(candidate).abs() < 1e-9)
        {
            continue;
        }
        points.push(candidate);
    }
    points
}

/// Confirm that `Σ N_{i,j}/p_i^j` reproduces `rem_reduced/den_reduced` at signed
/// probe points (requiring at least four finite agreements).
fn verify_reconstruction(rem_reduced: &Poly, den_reduced: &Poly, terms: &[PfTerm]) -> bool {
    const CHECK: [f64; 10] = [0.35, -0.45, 0.85, -1.15, 1.65, -1.9, 2.35, -2.8, 3.3, -3.7];
    let mut agreed = 0usize;
    for &pt in &CHECK {
        let den_val = den_reduced.eval_f64(pt);
        if den_val.abs() < 1e-9 {
            continue;
        }
        let lhs = rem_reduced.eval_f64(pt) / den_val;
        let mut rhs = 0.0f64;
        let mut usable = true;
        for term in terms {
            let factor_val = term.factor.eval_f64(pt);
            if factor_val.abs() < 1e-9 {
                usable = false;
                break;
            }
            rhs += term.numerator.eval_f64(pt) / factor_val.powi(term.power as i32);
        }
        if !usable {
            continue;
        }
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        if (lhs - rhs).abs() > 1e-6 * scale {
            return false;
        }
        agreed += 1;
    }
    agreed >= 4
}

/// Interpret `op` as a univariate rational function in `var`, returning
/// `(numerator, denominator)` as exact polynomials.
///
/// Returns `None` if `op` involves a different variable, a transcendental node,
/// or a power with a non-integer / symbolic exponent.
pub(crate) fn as_rational(op: &LoweredOp, var: usize) -> Option<(Poly, Poly)> {
    match op {
        LoweredOp::Const(c) => Some((Poly::constant(f64_to_ratio(*c).ok()?), one_poly())),
        LoweredOp::NamedConst(nc) => {
            Some((Poly::constant(f64_to_ratio(nc.value()).ok()?), one_poly()))
        }
        LoweredOp::Var(i) => {
            if *i == var {
                Some((
                    Poly::from_ratios(vec![coeff_zero(), coeff_one()]),
                    one_poly(),
                ))
            } else {
                None
            }
        }
        LoweredOp::Neg(a) => {
            let (n, d) = as_rational(a, var)?;
            Some((n.neg().ok()?, d))
        }
        LoweredOp::Add(a, b) => {
            let (na, da) = as_rational(a, var)?;
            let (nb, db) = as_rational(b, var)?;
            let numerator = na.mul(&db).ok()?.add(&nb.mul(&da).ok()?).ok()?;
            let denominator = da.mul(&db).ok()?;
            Some((numerator, denominator))
        }
        LoweredOp::Sub(a, b) => {
            let (na, da) = as_rational(a, var)?;
            let (nb, db) = as_rational(b, var)?;
            let numerator = na.mul(&db).ok()?.sub(&nb.mul(&da).ok()?).ok()?;
            let denominator = da.mul(&db).ok()?;
            Some((numerator, denominator))
        }
        LoweredOp::Mul(a, b) => {
            let (na, da) = as_rational(a, var)?;
            let (nb, db) = as_rational(b, var)?;
            Some((na.mul(&nb).ok()?, da.mul(&db).ok()?))
        }
        LoweredOp::Div(a, b) => {
            let (na, da) = as_rational(a, var)?;
            let (nb, db) = as_rational(b, var)?;
            Some((na.mul(&db).ok()?, da.mul(&nb).ok()?))
        }
        LoweredOp::Pow(base, exp) => {
            let LoweredOp::Const(e) = exp.as_ref() else {
                return None;
            };
            if !e.is_finite() || e.fract() != 0.0 || e.abs() > 64.0 {
                return None;
            }
            let power = e.abs() as usize;
            let (nb, db) = as_rational(base, var)?;
            let numerator = nb.pow(power).ok()?;
            let denominator = db.pow(power).ok()?;
            if *e >= 0.0 {
                Some((numerator, denominator))
            } else {
                Some((denominator, numerator))
            }
        }
        _ => None,
    }
}

/// The constant polynomial `1`.
fn one_poly() -> Poly {
    Poly::constant(coeff_one())
}

impl LoweredOp {
    /// Partial-fraction-decompose a univariate rational function.
    ///
    /// Returns the input unchanged when it is not a single-variable rational
    /// function, when the decomposition cannot be computed, or when the
    /// numeric-verification gate fails. Never panics.
    ///
    /// Unlike the previous rational-integration path — which handled only
    /// linear factors and a single irreducible quadratic — this uses the full
    /// [`Poly::factor`](crate::poly::Poly::factor), so e.g.
    /// `1/((x²+1)(x²+4))` now decomposes successfully.
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // 1 / ((x^2 + 1)(x^2 + 4))
    /// let x2 = || Arc::new(LoweredOp::Pow(
    ///     Arc::new(LoweredOp::Var(0)),
    ///     Arc::new(LoweredOp::Const(2.0)),
    /// ));
    /// let den = LoweredOp::Mul(
    ///     Arc::new(LoweredOp::Add(x2(), Arc::new(LoweredOp::Const(1.0)))),
    ///     Arc::new(LoweredOp::Add(x2(), Arc::new(LoweredOp::Const(4.0)))),
    /// );
    /// let expr = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(den));
    /// let decomposed = expr.apart();
    /// for xv in [0.5_f64, 1.5, 3.0] {
    ///     let want = 1.0 / ((xv * xv + 1.0) * (xv * xv + 4.0));
    ///     assert!((decomposed.eval(&[xv]) - want).abs() < 1e-9);
    /// }
    /// ```
    #[must_use]
    pub fn apart(&self) -> LoweredOp {
        let Some(var) = sole_var(self) else {
            return self.clone();
        };
        let Some((num, den)) = as_rational(self, var) else {
            return self.clone();
        };
        let Some(pf) = partial_fractions(&num, &den, var) else {
            return self.clone();
        };

        let mut parts: Vec<LoweredOp> = Vec::new();
        if !pf.polynomial_part.is_zero() {
            parts.push(pf.polynomial_part.to_lowered(var).simplify());
        }
        for term in &pf.terms {
            let numerator = term.numerator.to_lowered(var).simplify();
            let factor = term.factor.to_lowered(var).simplify();
            let denominator = if term.power == 1 {
                factor
            } else {
                lpow(factor, term.power as f64)
            };
            parts.push(ldiv(numerator, denominator));
        }

        let result = parts
            .into_iter()
            .reduce(ladd)
            .unwrap_or_else(|| lconst(0.0));
        if verify_agrees(self, &result, self.count_vars()) {
            result
        } else {
            self.clone()
        }
    }

    /// Combine a sum of fractions into a single reduced fraction `num/den`.
    ///
    /// Walks the expression as a univariate rational function, cancels the
    /// polynomial `gcd` of numerator and denominator, and normalises the
    /// denominator to be monic. Returns the input unchanged when it is not a
    /// single-variable rational function or when verification fails. Never
    /// panics. This is the inverse of [`apart`](LoweredOp::apart).
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    /// // 1/x + 1/(x+1)  →  (2x + 1) / (x^2 + x)
    /// let a = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(LoweredOp::Var(0)));
    /// let b = LoweredOp::Div(
    ///     Arc::new(LoweredOp::Const(1.0)),
    ///     Arc::new(LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(1.0)))),
    /// );
    /// let expr = LoweredOp::Add(Arc::new(a), Arc::new(b));
    /// let combined = expr.together();
    /// for xv in [0.5_f64, 2.0, 3.0] {
    ///     let want = 1.0 / xv + 1.0 / (xv + 1.0);
    ///     assert!((combined.eval(&[xv]) - want).abs() < 1e-9);
    /// }
    /// ```
    #[must_use]
    pub fn together(&self) -> LoweredOp {
        let Some(var) = sole_var(self) else {
            return self.clone();
        };
        let Some((num, den)) = as_rational(self, var) else {
            return self.clone();
        };
        if den.is_zero() {
            return self.clone();
        }

        let Ok(gcd) = Poly::gcd(&num, &den) else {
            return self.clone();
        };
        let (Ok((num_reduced, _)), Ok((den_reduced, _))) = (num.div_rem(&gcd), den.div_rem(&gcd))
        else {
            return self.clone();
        };

        // Normalise the denominator to be monic.
        let Some(inv) = coeff_recip(&den_reduced.leading_coeff()) else {
            return self.clone();
        };
        let (Ok(num_monic), Ok(den_monic)) = (num_reduced.scale(&inv), den_reduced.scale(&inv))
        else {
            return self.clone();
        };

        let numerator = num_monic.to_lowered(var).simplify();
        let result = if den_monic.degree().unwrap_or(0) == 0 {
            // Denominator reduced to a constant: the value is a polynomial.
            numerator
        } else {
            ldiv(numerator, den_monic.to_lowered(var).simplify())
        };

        if verify_agrees(self, &result, self.count_vars()) {
            result
        } else {
            self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn x() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Arc::new(a), Arc::new(b))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Arc::new(a), Arc::new(b))
    }
    fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Div(Arc::new(a), Arc::new(b))
    }
    fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(a), Arc::new(b))
    }

    fn count_div(op: &LoweredOp) -> usize {
        match op {
            LoweredOp::Div(_, _) => 1,
            LoweredOp::Add(a, b) | LoweredOp::Sub(a, b) | LoweredOp::Mul(a, b) => {
                count_div(a) + count_div(b)
            }
            LoweredOp::Neg(a) => count_div(a),
            _ => 0,
        }
    }

    #[test]
    fn apart_two_irreducible_quadratics_succeeds() {
        // 1 / ((x^2 + 1)(x^2 + 4)) — previously returned None.
        let den = mul(add(pow(x(), c(2.0)), c(1.0)), add(pow(x(), c(2.0)), c(4.0)));
        let expr = div(c(1.0), den);
        let decomposed = expr.apart();

        // It actually decomposed: two separate fractions.
        assert_eq!(count_div(&decomposed), 2, "expected two partial fractions");
        assert_ne!(decomposed, expr, "apart should transform the expression");

        for xv in [0.3_f64, 1.4, 2.7, -1.1, -3.3] {
            let want = 1.0 / ((xv * xv + 1.0) * (xv * xv + 4.0));
            assert!(
                (decomposed.eval(&[xv]) - want).abs() < 1e-9,
                "value mismatch at x={xv}"
            );
        }
    }

    #[test]
    fn apart_then_together_roundtrip() {
        let den = mul(add(pow(x(), c(2.0)), c(1.0)), add(pow(x(), c(2.0)), c(4.0)));
        let expr = div(c(1.0), den);
        let recombined = expr.apart().together();
        for xv in [0.3_f64, 1.4, 2.7, -1.1, -3.3] {
            let want = 1.0 / ((xv * xv + 1.0) * (xv * xv + 4.0));
            assert!(
                (recombined.eval(&[xv]) - want).abs() < 1e-9,
                "roundtrip mismatch at x={xv}"
            );
        }
    }

    #[test]
    fn apart_distinct_linear_roots() {
        // 1 / (x^2 - 1) = 1/2/(x-1) - 1/2/(x+1)
        let expr = div(
            c(1.0),
            LoweredOp::Sub(Arc::new(pow(x(), c(2.0))), Arc::new(c(1.0))),
        );
        let decomposed = expr.apart();
        for xv in [2.0_f64, 3.0, -2.0, 0.5, -0.5] {
            let want = 1.0 / (xv * xv - 1.0);
            assert!((decomposed.eval(&[xv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn together_sum_of_two_fractions() {
        // 1/x + 1/(x+1) → (2x+1)/(x^2+x)
        let a = div(c(1.0), x());
        let b = div(c(1.0), add(x(), c(1.0)));
        let combined = add(a, b).together();
        for xv in [0.5_f64, 2.0, 3.0, -2.5] {
            let want = 1.0 / xv + 1.0 / (xv + 1.0);
            assert!((combined.eval(&[xv]) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn apart_transcendental_returns_input() {
        let expr = LoweredOp::Sin(Arc::new(x()));
        assert_eq!(expr.apart(), expr);
        assert_eq!(expr.together(), expr);
    }

    #[test]
    fn partial_fractions_improper_has_polynomial_part() {
        // (x^3) / (x^2 - 1): polynomial part x, plus proper fraction.
        let num = Poly::from_int_coeffs(&[0, 0, 0, 1]);
        let den = Poly::from_int_coeffs(&[-1, 0, 1]);
        let pf = partial_fractions(&num, &den, 0).expect("decomposes");
        assert!(!pf.polynomial_part.is_zero(), "expected polynomial part");
        assert!(!pf.terms.is_empty(), "expected proper-fraction terms");
    }
}
