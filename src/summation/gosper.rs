//! Gosper's algorithm for indefinite hypergeometric summation.
//!
//! # Mathematics
//!
//! A term `t(k)` is *hypergeometric* when the ratio `r(k) = t(k+1)/t(k)` is a
//! rational function of `k`. Gosper's algorithm decides whether such a `t` is
//! **Gosper-summable**, i.e. whether there is a hypergeometric `T(k)` with
//!
//! ```text
//! T(k+1) - T(k) = t(k)
//! ```
//!
//! and, when one exists, produces it. `T` is written `T(k) = R(k) · t(k)` for a
//! rational *certificate* `R(k)`.
//!
//! ## Step 1 — Gosper–Petkovšek normal form
//!
//! Given `r(k) = f(k)/g(k)` in lowest terms, factor
//!
//! ```text
//! r(k) = a(k)/b(k) · c(k+1)/c(k)
//! ```
//!
//! with polynomials `a, b, c` satisfying `gcd(a(k), b(k+h)) = 1` for every
//! integer `h ≥ 0`. The construction repeatedly peels common factors: the set of
//! candidate shifts is the set of non-negative integer roots `h` of the
//! *dispersion resultant*
//!
//! ```text
//! R(h) = Res_k( f(k), g(k+h) ),
//! ```
//!
//! and for each such `h` (in increasing order) one removes `s(k) = gcd(a(k),
//! b(k+h))` from `a`, removes `s(k-h)` from `b`, and multiplies `c` by
//! `∏_{j=1}^{h} s(k-j)`. This update provably preserves the identity
//! `a/b · c(k+1)/c(k) = r(k)`.
//!
//! ## Step 2 — the Gosper key equation
//!
//! Writing `T(k) = (b(k-1)/c(k)) · x(k) · t(k)` and demanding `T(k+1) - T(k) =
//! t(k)` reduces to the **key polynomial equation**
//!
//! ```text
//! a(k) · x(k+1) - b(k-1) · x(k) = c(k)
//! ```
//!
//! in the unknown polynomial `x(k)`. The term is Gosper-summable **iff** this
//! equation has a polynomial solution.
//!
//! ## Step 3 — degree bound and linear solve
//!
//! A classical degree analysis of the key equation bounds `deg x`. With the
//! bound in hand, the unknown coefficients of `x` are found by exact rational
//! Gaussian elimination on the (generally rectangular, over-determined) linear
//! system obtained by matching coefficients of each power of `k`.
//!
//! When a solution `x` exists, the certificate is `R(k) = b(k-1) x(k) / c(k)` and
//! the antidifference is `T(k) = R(k) t(k)`. If no polynomial `x` exists the term
//! is honestly reported as having no hypergeometric closed form.
//!
//! # Exactness
//!
//! Every arithmetic step uses exact [`Coeff`] (`BigRational`) coefficients: the
//! dispersion resultant is obtained by exact evaluation/interpolation, the
//! decomposition uses exact polynomial GCD and division, and the key equation is
//! solved with an exact rational Gauss–Jordan elimination that handles
//! rectangular and rank-deficient systems (choosing the free-variable-zero
//! particular solution). The f64 solver `crate::rewrite::gaussian_eliminate` is
//! *not* used here because it is square-only and inexact, which would defeat the
//! certificate's exact verification.

use num_bigint::{BigInt, Sign};

use crate::poly::{Coeff, Poly, coeff_is_zero, coeff_one, coeff_recip, coeff_zero};

/// Outcome of running Gosper's algorithm on a rational term ratio.
pub(crate) enum Gosper {
    /// The term is Gosper-summable; the certificate rational function is
    /// `num(k) / den(k)`, so `T(k) = num(k)/den(k) · t(k)`.
    Certificate {
        /// Numerator of the certificate rational function.
        num: Poly,
        /// Denominator of the certificate rational function.
        den: Poly,
    },
    /// The term is hypergeometric but has no hypergeometric antidifference.
    NoClosedForm,
}

/// Substitute `k -> k + m` into a polynomial, exactly.
///
/// Computes `p(k + m)` by Horner evaluation of `p` at the polynomial point
/// `k + m`. Works for negative `m`.
pub(crate) fn shift_poly(p: &Poly, m: i64) -> Poly {
    if m == 0 {
        return p.clone();
    }
    let base = Poly::from_ratios(vec![Coeff::from_integer(BigInt::from(m)), coeff_one()]);
    let mut acc = Poly::zero();
    for coeff in p.coeffs.iter().rev() {
        acc = acc
            .mul(&base)
            .expect("polynomial multiplication is infallible")
            .add(&Poly::constant(coeff.clone()))
            .expect("polynomial addition is infallible");
    }
    acc
}

/// Convert a rational coefficient to a non-negative `i64`, or `None` when it is
/// not a non-negative integer that fits.
fn coeff_to_nonneg_i64(c: &Coeff) -> Option<i64> {
    if !c.is_integer() {
        return None;
    }
    let value = c.to_integer();
    if value.sign() == Sign::Minus {
        return None;
    }
    let mut digits = value.magnitude().iter_u64_digits();
    let low = digits.next().unwrap_or(0);
    if digits.next().is_some() {
        return None; // too large to be a meaningful shift/degree
    }
    i64::try_from(low).ok()
}

/// Convert a rational coefficient to a non-negative `usize`, or `None`.
fn coeff_to_nonneg_usize(c: &Coeff) -> Option<usize> {
    coeff_to_nonneg_i64(c).and_then(|v| usize::try_from(v).ok())
}

/// Exact Lagrange interpolation through the points `(i, ys[i])` for `i = 0, 1,
/// …, ys.len()-1`.
fn lagrange_interpolate(ys: &[Coeff]) -> Poly {
    let n = ys.len();
    let mut result = Poly::zero();
    for (i, y_i) in ys.iter().enumerate() {
        let mut basis = Poly::constant(coeff_one());
        let mut denom = coeff_one();
        for j in 0..n {
            if i == j {
                continue;
            }
            let factor = Poly::from_ratios(vec![
                -Coeff::from_integer(BigInt::from(j as i64)),
                coeff_one(),
            ]);
            basis = basis
                .mul(&factor)
                .expect("polynomial multiplication is infallible");
            let diff = Coeff::from_integer(BigInt::from(i as i64 - j as i64));
            denom = &denom * &diff;
        }
        let denom_inv = coeff_recip(&denom).expect("interpolation nodes are distinct");
        let scale = y_i * &denom_inv;
        let scaled = basis
            .scale(&scale)
            .expect("polynomial scaling is infallible");
        result = result
            .add(&scaled)
            .expect("polynomial addition is infallible");
    }
    result
}

/// Return the sorted set of non-negative integer roots `h` of the dispersion
/// resultant `R(h) = Res_k(f(k), g(k+h))`.
///
/// `R(h)` is a polynomial in `h` of degree `deg(f)·deg(g)`; it is recovered by
/// evaluating the (integer-shift) resultant at `h = 0, 1, …, deg(f)·deg(g)` and
/// interpolating exactly, then extracting its rational roots.
///
/// Returns `None` only when the rational-root search exceeds its factorization
/// budget (an honest "cannot decide").
fn resultant_dispersion(f: &Poly, g: &Poly) -> Option<Vec<usize>> {
    let df = f.degree().unwrap_or(0);
    let dg = g.degree().unwrap_or(0);
    let sample_count = df * dg;

    let mut ys: Vec<Coeff> = Vec::with_capacity(sample_count + 1);
    for h in 0..=sample_count {
        let g_shift = shift_poly(g, h as i64);
        let res = Poly::resultant(f, &g_shift).ok()?;
        ys.push(res);
    }

    let r_poly = lagrange_interpolate(&ys);
    let roots = r_poly.rational_roots().ok()?;
    let mut out: Vec<usize> = roots.iter().filter_map(coeff_to_nonneg_usize).collect();
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// Perform the Gosper–Petkovšek decomposition, returning `(a, b, c)`.
///
/// Returns `None` only if an internal exact division has a non-zero remainder,
/// which cannot happen mathematically and signals a bug guard.
fn gosper_petkovsek(f: &Poly, g: &Poly, roots: &[usize]) -> Option<(Poly, Poly, Poly)> {
    let mut a = f.normalized();
    let mut b = g.normalized();
    let mut c = Poly::constant(coeff_one());

    for &h in roots {
        let b_shift = shift_poly(&b, h as i64); // b(k + h)
        let s = Poly::gcd(&a, &b_shift).ok()?;
        if s.degree().unwrap_or(0) < 1 {
            continue;
        }
        // a := a / s
        let (quotient_a, remainder_a) = a.div_rem(&s).ok()?;
        if !remainder_a.is_zero() {
            return None;
        }
        a = quotient_a;
        // b := b / s(k - h)
        let s_minus_h = shift_poly(&s, -(h as i64));
        let (quotient_b, remainder_b) = b.div_rem(&s_minus_h).ok()?;
        if !remainder_b.is_zero() {
            return None;
        }
        b = quotient_b;
        // c := c · ∏_{j=1}^{h} s(k - j)
        for j in 1..=h {
            let s_minus_j = shift_poly(&s, -(j as i64));
            c = c.mul(&s_minus_j).ok()?;
        }
    }

    Some((a, b, c))
}

/// The coefficient of `x^degree` in `p`, or zero when absent.
fn coeff_at(p: &Poly, degree: usize) -> Coeff {
    p.coeffs.get(degree).cloned().unwrap_or_else(coeff_zero)
}

/// Compute the Gosper degree bound for `x` in `a(k) x(k+1) - b_shift(k) x(k) =
/// c(k)`, where `b_shift(k) = b(k-1)`.
///
/// Returns the maximal possible degree of a polynomial solution (an `i64`,
/// possibly negative, in which case no solution of non-negative degree exists).
fn degree_bound(a: &Poly, b_shift: &Poly, c: &Poly) -> i64 {
    let da = a.degree().unwrap_or(0);
    let db = b_shift.degree().unwrap_or(0);
    let dc = c.degree().map_or(-1i64, |d| d as i64);
    let ell = da.max(db) as i64;

    if da != db {
        return dc - ell;
    }

    let lead_a = a.leading_coeff();
    let lead_b = b_shift.leading_coeff();
    if lead_a != lead_b {
        return dc - ell;
    }

    // Leading terms cancel: analyse the next coefficient.
    if ell == 0 {
        // a and b_shift are equal non-zero constants λ; the equation becomes
        // λ (x(k+1) - x(k)) = c(k), so deg x = deg c + 1.
        return dc + 1;
    }

    let a_sub = coeff_at(a, (ell - 1) as usize);
    let b_sub = coeff_at(b_shift, (ell - 1) as usize);
    let lead_inv = coeff_recip(&lead_a).expect("leading coefficient is non-zero");
    let d0 = &(&b_sub - &a_sub) * &lead_inv;

    let generic = dc - ell + 1;
    match coeff_to_nonneg_i64(&d0) {
        Some(special) => generic.max(special),
        None => generic,
    }
}

/// Solve the exact rational linear system `mat · x = rhs` for `n_unknowns`
/// unknowns.
///
/// `mat` is `m × n_unknowns` (generally over-determined). Uses Gauss–Jordan
/// elimination over ℚ. Returns the particular solution with all free variables
/// set to zero, or `None` if the system is inconsistent.
fn solve_exact(mat: Vec<Vec<Coeff>>, rhs: Vec<Coeff>, n_unknowns: usize) -> Option<Vec<Coeff>> {
    let m = mat.len();
    // Augmented rows: each row has `n_unknowns` coefficients followed by its rhs.
    let mut rows: Vec<Vec<Coeff>> = mat
        .into_iter()
        .zip(rhs)
        .map(|(mut row, r)| {
            row.push(r);
            row
        })
        .collect();

    let mut where_pivot: Vec<Option<usize>> = vec![None; n_unknowns];
    let mut pivot_row = 0usize;

    for col in 0..n_unknowns {
        // Find a pivot at or below the current pivot row.
        let mut selected = None;
        for (offset, row) in rows.iter().enumerate().skip(pivot_row) {
            if !coeff_is_zero(&row[col]) {
                selected = Some(offset);
                break;
            }
        }
        let Some(selected) = selected else {
            continue; // free column
        };
        rows.swap(pivot_row, selected);

        // Normalise the pivot row so that its pivot entry is 1.
        let inv = coeff_recip(&rows[pivot_row][col]).expect("pivot is non-zero");
        for entry in &mut rows[pivot_row] {
            let scaled = &*entry * &inv;
            *entry = scaled;
        }

        // Eliminate this column from every other row.
        let pivot = rows[pivot_row].clone();
        for (idx, row) in rows.iter_mut().enumerate() {
            if idx == pivot_row {
                continue;
            }
            let factor = row[col].clone();
            if coeff_is_zero(&factor) {
                continue;
            }
            for (dst, piv) in row.iter_mut().zip(pivot.iter()) {
                let updated = &*dst - &(&factor * piv);
                *dst = updated;
            }
        }

        where_pivot[col] = Some(pivot_row);
        pivot_row += 1;
        if pivot_row == m {
            break;
        }
    }

    // Consistency check: a row with all-zero coefficients but non-zero rhs is a
    // contradiction (0 = nonzero).
    for row in &rows {
        let all_zero = row[..n_unknowns].iter().all(coeff_is_zero);
        if all_zero && !coeff_is_zero(&row[n_unknowns]) {
            return None;
        }
    }

    let mut solution = vec![coeff_zero(); n_unknowns];
    for (col, pivot) in where_pivot.iter().enumerate() {
        if let Some(row) = pivot {
            solution[col] = rows[*row][n_unknowns].clone();
        }
    }
    Some(solution)
}

/// Solve the Gosper key equation `a(k) x(k+1) - b_shift(k) x(k) = c(k)` for the
/// polynomial `x`, or return `None` when no polynomial solution exists.
fn solve_key_polynomial(a: &Poly, b_shift: &Poly, c: &Poly) -> Option<Poly> {
    let bound = degree_bound(a, b_shift, c);
    if bound < 0 {
        return None;
    }
    let bound = bound as usize;

    let n_plus_one = Poly::from_ratios(vec![coeff_one(), coeff_one()]); // k + 1
    let k_poly = Poly::from_ratios(vec![coeff_zero(), coeff_one()]); // k

    // Column j is the polynomial a(k)·(k+1)^j - b_shift(k)·k^j.
    let mut columns: Vec<Poly> = Vec::with_capacity(bound + 1);
    let mut np1_pow = Poly::constant(coeff_one());
    let mut k_pow = Poly::constant(coeff_one());
    for j in 0..=bound {
        let term_a = a
            .mul(&np1_pow)
            .expect("polynomial multiplication is infallible");
        let term_b = b_shift
            .mul(&k_pow)
            .expect("polynomial multiplication is infallible");
        let column = term_a
            .sub(&term_b)
            .expect("polynomial subtraction is infallible");
        columns.push(column);
        if j < bound {
            np1_pow = np1_pow
                .mul(&n_plus_one)
                .expect("polynomial multiplication is infallible");
            k_pow = k_pow
                .mul(&k_poly)
                .expect("polynomial multiplication is infallible");
        }
    }

    let max_degree = columns
        .iter()
        .filter_map(Poly::degree)
        .max()
        .unwrap_or(0)
        .max(c.degree().unwrap_or(0));

    let mut mat = vec![vec![coeff_zero(); bound + 1]; max_degree + 1];
    for (j, column) in columns.iter().enumerate() {
        for (deg, coeff) in column.coeffs.iter().enumerate() {
            mat[deg][j] = coeff.clone();
        }
    }
    let mut rhs = vec![coeff_zero(); max_degree + 1];
    for (deg, coeff) in c.coeffs.iter().enumerate() {
        rhs[deg] = coeff.clone();
    }

    let solution = solve_exact(mat, rhs, bound + 1)?;
    let x = Poly::from_ratios(solution);

    // Exact verification of the key equation before trusting the solution.
    let lhs = a
        .mul(&shift_poly(&x, 1))
        .expect("polynomial multiplication is infallible")
        .sub(
            &b_shift
                .mul(&x)
                .expect("polynomial multiplication is infallible"),
        )
        .expect("polynomial subtraction is infallible");
    if lhs.normalized() == c.normalized() {
        Some(x)
    } else {
        None
    }
}

/// Run Gosper's algorithm on the reduced term ratio `r(k) = num(k) / den(k)`.
///
/// Returns [`Gosper::Certificate`] with the rational certificate when the term
/// is Gosper-summable, else [`Gosper::NoClosedForm`].
pub(crate) fn gosper_certificate(num: &Poly, den: &Poly) -> Gosper {
    // Reduce to lowest terms defensively.
    let gcd = match Poly::gcd(num, den) {
        Ok(g) => g,
        Err(_) => return Gosper::NoClosedForm,
    };
    let f = match num.div_rem(&gcd) {
        Ok((q, _)) => q,
        Err(_) => return Gosper::NoClosedForm,
    };
    let g = match den.div_rem(&gcd) {
        Ok((q, _)) => q,
        Err(_) => return Gosper::NoClosedForm,
    };
    if f.is_zero() || g.is_zero() {
        return Gosper::NoClosedForm;
    }

    let roots = match resultant_dispersion(&f, &g) {
        Some(r) => r,
        None => return Gosper::NoClosedForm,
    };
    let (a, b, c) = match gosper_petkovsek(&f, &g, &roots) {
        Some(t) => t,
        None => return Gosper::NoClosedForm,
    };

    let b_shift = shift_poly(&b, -1); // b(k - 1)
    let x = match solve_key_polynomial(&a, &b_shift, &c) {
        Some(x) => x,
        None => return Gosper::NoClosedForm,
    };

    // Certificate rational function R(k) = b(k-1) x(k) / c(k).
    let cert_num = b_shift
        .mul(&x)
        .expect("polynomial multiplication is infallible");
    let cert_den = c;

    // Reduce the certificate by its GCD for a cleaner closed form.
    let reduce_gcd =
        Poly::gcd(&cert_num, &cert_den).unwrap_or_else(|_| Poly::constant(coeff_one()));
    let num_reduced = cert_num
        .div_rem(&reduce_gcd)
        .map_or(cert_num.clone(), |(q, _)| q);
    let den_reduced = cert_den
        .div_rem(&reduce_gcd)
        .map_or(cert_den.clone(), |(q, _)| q);

    Gosper::Certificate {
        num: num_reduced,
        den: den_reduced,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly(coeffs: &[i64]) -> Poly {
        Poly::from_int_coeffs(coeffs)
    }

    #[test]
    fn shift_poly_shifts() {
        // p(k) = k^2 ;  p(k+1) = k^2 + 2k + 1
        let p = poly(&[0, 0, 1]);
        let shifted = shift_poly(&p, 1);
        assert_eq!(shifted, poly(&[1, 2, 1]));
        // shift back
        assert_eq!(shift_poly(&shifted, -1), p);
    }

    #[test]
    fn lagrange_recovers_polynomial() {
        // (h - 1)^2 = h^2 - 2h + 1 sampled at 0,1,2 -> 1, 0, 1
        let ys = vec![
            Coeff::from_integer(BigInt::from(1i32)),
            Coeff::from_integer(BigInt::from(0i32)),
            Coeff::from_integer(BigInt::from(1i32)),
        ];
        let r = lagrange_interpolate(&ys);
        assert_eq!(r, poly(&[1, -2, 1]));
    }

    #[test]
    fn dispersion_for_k_times_k_factorial() {
        // r(k) = (k+1)^2 / k -> f = (k+1)^2, g = k. Dispersion set = {1}.
        let f = poly(&[1, 2, 1]);
        let g = poly(&[0, 1]);
        let roots = resultant_dispersion(&f, &g).expect("dispersion computable");
        assert_eq!(roots, vec![1]);
    }

    #[test]
    fn gosper_summable_k_times_k_factorial() {
        // r(k) = (k+1)^2 / k.
        let num = poly(&[1, 2, 1]);
        let den = poly(&[0, 1]);
        match gosper_certificate(&num, &den) {
            Gosper::Certificate { num, den } => {
                // Certificate is 1 / k.
                assert_eq!(num, poly(&[1]));
                assert_eq!(den, poly(&[0, 1]));
            }
            Gosper::NoClosedForm => panic!("k·k! must be Gosper-summable"),
        }
    }

    #[test]
    fn gosper_rejects_harmonic() {
        // r(k) = k / (k+1) for t(k) = 1/k -> not Gosper-summable.
        let num = poly(&[0, 1]);
        let den = poly(&[1, 1]);
        assert!(matches!(
            gosper_certificate(&num, &den),
            Gosper::NoClosedForm
        ));
    }

    #[test]
    fn gosper_telescoping_reciprocal() {
        // t(k) = 1/(k(k+1)) -> r(k) = k/(k+2). Certificate is -(k+1)/1.
        let num = poly(&[0, 1]);
        let den = poly(&[2, 1]);
        match gosper_certificate(&num, &den) {
            Gosper::Certificate { num, den } => {
                // R(k) = b(k-1) x(k) / c(k) = (k+1)·(-1)/1 = -(k+1).
                assert_eq!(den, poly(&[1]));
                assert_eq!(num, poly(&[-1, -1]));
            }
            Gosper::NoClosedForm => panic!("1/(k(k+1)) must telescope"),
        }
    }
}
