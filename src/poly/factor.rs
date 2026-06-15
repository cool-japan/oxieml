//! Polynomial factorization: Yun SFD + rational roots + Kronecker's method.

use num_rational::Ratio;

use super::univariate::Poly;
use super::{PolyError, checked_add, checked_mul, checked_neg, integer_factors};

/// Result of polynomial factorization.
#[derive(Clone, Debug, PartialEq)]
pub struct Factorization {
    /// Content (GCD of coefficients).
    pub content: Ratio<i64>,
    /// Irreducible factors with multiplicities: `(factor, multiplicity)`.
    pub factors: Vec<(Poly, usize)>,
}

/// Yun's square-free decomposition.
pub fn square_free_decomposition(f: &Poly) -> Result<Vec<(Poly, usize)>, PolyError> {
    let f = f.normalized();
    if f.is_zero() {
        return Ok(Vec::new());
    }

    let lc = f.leading_coeff();
    let f_monic = if *lc.numer() == 0 {
        return Ok(Vec::new());
    } else {
        f.scale(Ratio::new(*lc.denom(), *lc.numer()))?
    };

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
            let lc_b = b.leading_coeff();
            if *lc_b.numer() != 0 {
                let b_monic = b.scale(Ratio::new(*lc_b.denom(), *lc_b.numer()))?;
                factors.push((b_monic, i));
            }
            break;
        }

        let a = Poly::gcd(&b, &d)?;

        if a.degree().unwrap_or(0) >= 1 {
            let lc_a = a.leading_coeff();
            if *lc_a.numer() != 0 {
                let a_monic = a.scale(Ratio::new(*lc_a.denom(), *lc_a.numer()))?;
                factors.push((a_monic, i));
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
    /// Factor this polynomial into irreducibles.
    pub fn factor(&self) -> Result<Factorization, PolyError> {
        let f = self.normalized();
        if f.is_zero() {
            return Ok(Factorization {
                content: Ratio::new(0, 1),
                factors: Vec::new(),
            });
        }

        let content = f.content();
        let primitive = f.primitive_part()?;

        let sfd = square_free_decomposition(&primitive)?;

        let mut all_factors: Vec<(Poly, usize)> = Vec::new();
        for (sf_factor, mult) in sfd {
            let irreducibles = split_into_irreducibles(&sf_factor)?;
            for irred in irreducibles {
                all_factors.push((irred, mult));
            }
        }

        Ok(Factorization {
            content,
            factors: all_factors,
        })
    }
}

fn split_into_irreducibles(f: &Poly) -> Result<Vec<Poly>, PolyError> {
    let f = f.normalized();
    if f.is_zero() {
        return Ok(Vec::new());
    }

    let deg = match f.degree() {
        None => return Ok(Vec::new()),
        Some(0) => return Ok(Vec::new()),
        Some(1) => return Ok(vec![f]),
        Some(d) => d,
    };

    let rational_roots = f.rational_roots()?;
    if !rational_roots.is_empty() {
        let mut remaining = f.clone();
        let mut linear_factors: Vec<Poly> = Vec::new();

        for root in rational_roots {
            let lin = Poly {
                coeffs: vec![checked_neg(root)?, Ratio::new(1, 1)],
            };
            let (quot, rem) = remaining.div_rem(&lin)?;
            if rem.is_zero() {
                linear_factors.push(lin);
                remaining = quot.normalized();
            }
        }

        let mut result = linear_factors;
        if !remaining.is_zero() && remaining.degree().unwrap_or(0) >= 1 {
            let more = split_into_irreducibles(&remaining)?;
            result.extend(more);
        }
        return Ok(result);
    }

    if deg == 2 {
        return Ok(vec![f]);
    }

    if deg <= 6 {
        if let Ok(Some(factor)) = kronecker_try_split(&f) {
            let (quot, rem) = f.div_rem(&factor)?;
            if rem.is_zero() && !quot.is_zero() && quot.degree().unwrap_or(0) >= 1 {
                let mut result = split_into_irreducibles(&factor)?;
                result.extend(split_into_irreducibles(&quot.normalized())?);
                return Ok(result);
            }
        }
    }

    Ok(vec![f])
}

fn kronecker_try_split(f: &Poly) -> Result<Option<Poly>, PolyError> {
    let deg = match f.degree() {
        Some(d) if d >= 2 => d,
        _ => return Ok(None),
    };

    let max_factor_deg = deg / 2;
    let eval_points: [i64; 7] = [0, 1, -1, 2, -2, 3, -3];

    for k in 1..=max_factor_deg {
        if k + 1 > eval_points.len() {
            break;
        }

        let pts = &eval_points[..=k];
        let mut vals: Vec<i64> = Vec::new();
        for &pt in pts {
            let v = f.eval(Ratio::new(pt, 1))?;
            let (vn, vd) = (*v.numer(), *v.denom());
            if vd != 1 {
                continue;
            }
            vals.push(vn);
        }

        if vals.len() != k + 1 {
            continue;
        }

        let divisors: Vec<Vec<i64>> = vals.iter().map(|&v| integer_factors(v)).collect();

        let combinations = enumerate_combinations(&divisors, 100);

        for combo in &combinations {
            if let Some(candidate) = lagrange_interpolate(pts, combo) {
                if candidate.is_zero() || candidate.degree() == Some(0) {
                    continue;
                }
                let (_, rem) = f.div_rem(&candidate)?;
                if rem.is_zero() {
                    return Ok(Some(candidate));
                }
            }
        }
    }

    Ok(None)
}

fn enumerate_combinations(divisors: &[Vec<i64>], max_count: usize) -> Vec<Vec<i64>> {
    if divisors.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<Vec<i64>> = vec![Vec::new()];
    for divs in divisors {
        let mut new_result = Vec::new();
        for combo in &result {
            for &d in divs {
                let mut new_combo = combo.clone();
                new_combo.push(d);
                new_result.push(new_combo);
                if new_result.len() >= max_count {
                    return new_result;
                }
            }
        }
        result = new_result;
        if result.len() >= max_count {
            return result;
        }
    }
    result
}

fn lagrange_interpolate(xs: &[i64], ys: &[i64]) -> Option<Poly> {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return None;
    }

    let mut result_coeffs = vec![Ratio::new(0i64, 1); n];

    for i in 0..n {
        let mut denom = 1i64;
        for j in 0..n {
            if j == i {
                continue;
            }
            let diff = xs[i].checked_sub(xs[j])?;
            denom = denom.checked_mul(diff)?;
        }
        if denom == 0 {
            return None;
        }

        let mut num_poly: Vec<Ratio<i64>> = vec![Ratio::new(1, 1)];

        for (j, &xj) in xs.iter().enumerate().take(n) {
            if j == i {
                continue;
            }
            let neg_xj = xj.checked_neg()?;
            let mut new_poly = vec![Ratio::new(0, 1); num_poly.len() + 1];
            for (k, &c) in num_poly.iter().enumerate() {
                new_poly[k + 1] = checked_add(new_poly[k + 1], c).ok()?;
                let scaled = checked_mul(c, Ratio::new(neg_xj, 1)).ok()?;
                new_poly[k] = checked_add(new_poly[k], scaled).ok()?;
            }
            num_poly = new_poly;
        }

        let scale = Ratio::new(ys[i], denom);
        for (k, &c) in num_poly.iter().enumerate() {
            if k < n {
                let term = checked_mul(c, scale).ok()?;
                result_coeffs[k] = checked_add(result_coeffs[k], term).ok()?;
            }
        }
    }

    for c in &result_coeffs {
        if *c.denom() != 1 {
            return None;
        }
    }

    let mut p = Poly {
        coeffs: result_coeffs,
    };
    p.normalize();
    if p.is_zero() {
        return None;
    }
    Some(p)
}
