//! Polynomial and Lambert-W equation solving helpers for [`crate::solve::solve_for_all`].

use crate::error::EmlError;
use crate::lower::LoweredOp;
use crate::numeric::lambert_w0;
use crate::poly::{Poly, PolyError};

use crate::numeric::lambert_wm1;
use num_complex::Complex;
use num_rational::Ratio;

/// All real roots of an equation.
#[derive(Debug, Clone)]
pub struct RootsResult {
    /// Roots sorted ascending by numerical value.
    pub roots: Vec<LoweredOp>,
}

/// Complete root set (real + complex) of a polynomial.
#[derive(Debug, Clone)]
pub struct ComplexRoots {
    /// All roots including complex ones, sorted by real part ascending, then imag ascending.
    pub roots: Vec<Complex<f64>>,
}

impl ComplexRoots {
    /// Filter to real roots only (|imag| < tol).
    pub fn real_roots(&self, tol: f64) -> Vec<f64> {
        self.roots
            .iter()
            .filter(|c| c.im.abs() < tol)
            .map(|c| c.re)
            .collect()
    }
}

fn ratio_to_f64(r: &Ratio<i64>) -> f64 {
    *r.numer() as f64 / *r.denom() as f64
}

/// Solve a univariate polynomial for all real roots.
pub(crate) fn solve_polynomial(poly: &Poly, _var: usize) -> Result<RootsResult, EmlError> {
    let deg = match poly.degree() {
        Some(d) => d,
        None => return Ok(RootsResult { roots: vec![] }),
    };

    match deg {
        0 => Ok(RootsResult { roots: vec![] }),
        1 => {
            let a0 = ratio_to_f64(&poly.coeffs[0]);
            let a1 = ratio_to_f64(&poly.coeffs[1]);
            if a1.abs() < 1e-14 {
                return Ok(RootsResult { roots: vec![] });
            }
            let root = -a0 / a1;
            Ok(RootsResult {
                roots: vec![LoweredOp::Const(root)],
            })
        }
        2 => {
            let a0 = ratio_to_f64(&poly.coeffs[0]);
            let a1 = ratio_to_f64(&poly.coeffs[1]);
            let a2 = ratio_to_f64(&poly.coeffs[2]);
            if a2.abs() < 1e-14 {
                if a1.abs() < 1e-14 {
                    return Ok(RootsResult { roots: vec![] });
                }
                return Ok(RootsResult {
                    roots: vec![LoweredOp::Const(-a0 / a1)],
                });
            }
            let disc = a1 * a1 - 4.0 * a2 * a0;
            if disc < -1e-12 {
                Ok(RootsResult { roots: vec![] })
            } else if disc.abs() <= 1e-12 {
                let root = -a1 / (2.0 * a2);
                Ok(RootsResult {
                    roots: vec![LoweredOp::Const(root)],
                })
            } else {
                let sq = disc.sqrt();
                let r1 = (-a1 - sq) / (2.0 * a2);
                let r2 = (-a1 + sq) / (2.0 * a2);
                let (r1, r2) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
                Ok(RootsResult {
                    roots: vec![LoweredOp::Const(r1), LoweredOp::Const(r2)],
                })
            }
        }
        3 => {
            let a0 = ratio_to_f64(&poly.coeffs[0]);
            let a1 = ratio_to_f64(&poly.coeffs[1]);
            let a2 = ratio_to_f64(&poly.coeffs[2]);
            let a3 = ratio_to_f64(&poly.coeffs[3]);

            if a3.abs() < 1e-14 {
                let deg2 = Poly {
                    coeffs: poly.coeffs[..3].to_vec(),
                };
                return solve_polynomial(&deg2, _var);
            }

            let shift = a2 / (3.0 * a3);
            let p_coef = (3.0 * a3 * a1 - a2 * a2) / (3.0 * a3 * a3);
            let q_coef = (2.0 * a2 * a2 * a2 - 9.0 * a3 * a2 * a1 + 27.0 * a3 * a3 * a0)
                / (27.0 * a3 * a3 * a3);

            let disc = -4.0 * p_coef * p_coef * p_coef - 27.0 * q_coef * q_coef;

            let mut roots: Vec<f64> = if disc > 1e-12 {
                let m = 2.0 * (-p_coef / 3.0).sqrt();
                let theta_arg =
                    (3.0 * q_coef / (2.0 * p_coef) * (-3.0 / p_coef).sqrt()).clamp(-1.0, 1.0);
                let theta = theta_arg.acos();
                let pi = std::f64::consts::PI;
                vec![
                    m * (theta / 3.0).cos() - shift,
                    m * ((theta - 2.0 * pi) / 3.0).cos() - shift,
                    m * ((theta - 4.0 * pi) / 3.0).cos() - shift,
                ]
            } else {
                poly.isolate_real_roots(-1e6, 1e6, 1e-10)
                    .unwrap_or_default()
            };

            roots.retain(|r| r.is_finite() && poly.eval_f64(*r).abs() < 1e-6);
            roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            roots.dedup_by(|a, b| (*a - *b).abs() < 1e-8);

            if roots.is_empty() {
                roots = poly
                    .isolate_real_roots(-1e6, 1e6, 1e-10)
                    .unwrap_or_default();
            }

            Ok(RootsResult {
                roots: roots.into_iter().map(LoweredOp::Const).collect(),
            })
        }
        _ => {
            let numeric_roots = poly
                .isolate_real_roots(-1e6, 1e6, 1e-10)
                .map_err(|_: PolyError| EmlError::NotSolvable)?;
            Ok(RootsResult {
                roots: numeric_roots.into_iter().map(LoweredOp::Const).collect(),
            })
        }
    }
}

/// Attempt to detect and solve x·eˣ = k (Lambert-W pattern).
///
/// Handles both `Sub(x·eˣ, k)` and `Add(k_neg, x·eˣ)` forms,
/// since `simplify()` may rewrite subtraction into addition with a negated constant.
pub(crate) fn try_lambert_w_solve(f: &LoweredOp, var: usize) -> Option<RootsResult> {
    const NEG_INV_E: f64 = -0.367_879_441_171_442_32;

    // Build root list for x·eˣ = k: returns W₀(k) and optionally W₋₁(k)
    let make_lambert_roots = |k: f64| -> Option<RootsResult> {
        let root0 = lambert_w0(k).ok().filter(|r| r.is_finite())?;
        let mut roots = vec![LoweredOp::Const(root0)];
        // W₋₁ exists when -1/e ≤ k < 0
        if (NEG_INV_E - 1e-12..0.0).contains(&k) {
            if let Ok(wm1) = lambert_wm1(k) {
                if wm1.is_finite() && (wm1 - root0).abs() > 1e-10 {
                    roots.push(LoweredOp::Const(wm1));
                    roots.sort_by(|a, b| {
                        if let (LoweredOp::Const(x), LoweredOp::Const(y)) = (a, b) {
                            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    });
                }
            }
        }
        Some(RootsResult { roots })
    };

    match f {
        LoweredOp::Sub(lhs, rhs) => {
            // Form: x·eˣ - k = 0  ⟹  x·eˣ = k  ⟹  x = W₀(k)
            if let LoweredOp::Const(k) = rhs.as_ref() {
                if is_x_exp_x(lhs, var) {
                    if let Some(result) = make_lambert_roots(*k) {
                        return Some(result);
                    }
                }
            }
            // Form: k - x·eˣ = 0  ⟹  x·eˣ = k  ⟹  x = W₀(k)
            if let LoweredOp::Const(k) = lhs.as_ref() {
                if is_x_exp_x(rhs, var) {
                    if let Some(result) = make_lambert_roots(*k) {
                        return Some(result);
                    }
                }
            }
            None
        }
        LoweredOp::Add(lhs, rhs) => {
            // Form: neg_k + x·eˣ = 0  ⟹  x·eˣ = -neg_k
            if let LoweredOp::Const(neg_k) = lhs.as_ref() {
                if is_x_exp_x(rhs, var) {
                    if let Some(result) = make_lambert_roots(-neg_k) {
                        return Some(result);
                    }
                }
            }
            // Form: x·eˣ + neg_k = 0  ⟹  x·eˣ = -neg_k
            if let LoweredOp::Const(neg_k) = rhs.as_ref() {
                if is_x_exp_x(lhs, var) {
                    if let Some(result) = make_lambert_roots(-neg_k) {
                        return Some(result);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn is_x_exp_x(op: &LoweredOp, var: usize) -> bool {
    if let LoweredOp::Mul(a, b) = op {
        let a_is_var = matches!(a.as_ref(), LoweredOp::Var(i) if *i == var);
        let b_is_exp_var = is_exp_var(b, var);
        if a_is_var && b_is_exp_var {
            return true;
        }
        let a_is_exp_var = is_exp_var(a, var);
        let b_is_var = matches!(b.as_ref(), LoweredOp::Var(i) if *i == var);
        if a_is_exp_var && b_is_var {
            return true;
        }
    }
    false
}

fn is_exp_var(op: &LoweredOp, var: usize) -> bool {
    if let LoweredOp::Exp(inner) = op {
        return matches!(inner.as_ref(), LoweredOp::Var(i) if *i == var);
    }
    false
}

/// Durand-Kerner (Weierstrass) method for simultaneous root finding.
/// Makes polynomial monic internally. Square-free pre-division recommended before calling.
fn durand_kerner(poly: &Poly, max_iter: usize, tol: f64) -> Result<Vec<Complex<f64>>, EmlError> {
    let n = match poly.degree() {
        Some(d) => d,
        None => return Ok(vec![]),
    };
    if n == 0 {
        return Ok(vec![]);
    }

    // Convert to f64 monic coefficients (ascending degree)
    let lead = ratio_to_f64(&poly.leading_coeff());
    if lead.abs() < 1e-300 {
        return Err(EmlError::NotSolvable);
    }
    let norm: Vec<f64> = poly.coeffs.iter().map(|c| ratio_to_f64(c) / lead).collect();
    // norm has n+1 entries, norm[n] == 1.0

    // Cauchy bound for root magnitudes: 1 + max(|a_k/a_n|)
    let r = {
        let max_coeff = norm[..n].iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        (1.0 + max_coeff).max(1.0)
    };

    // Initial approximations: slightly perturbed circle
    let pi = std::f64::consts::PI;
    let mut roots: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let theta = 2.0 * pi * k as f64 / n as f64 + 0.1;
            Complex::from_polar(r, theta)
        })
        .collect();

    // Evaluate monic polynomial at complex z via Horner
    // p(z) = z^n + norm[n-1]*z^{n-1} + ... + norm[0]
    // Horner: start with 1 (monic leading coeff), loop from n-1 down to 0
    let eval_at = |z: Complex<f64>| -> Complex<f64> {
        let mut acc = Complex::new(1.0, 0.0);
        for i in (0..n).rev() {
            acc = acc * z + Complex::new(norm[i], 0.0);
        }
        acc
    };

    for _iter in 0..max_iter {
        let mut max_change = 0.0_f64;
        let prev = roots.clone();
        for i in 0..n {
            let pz = eval_at(prev[i]);
            // Weierstrass correction: p(z_i) / Π_{j≠i} (z_i - z_j)
            let denom: Complex<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| prev[i] - prev[j])
                .fold(Complex::new(1.0, 0.0), |acc, x| acc * x);
            if denom.norm() < 1e-300 {
                continue;
            }
            let delta = pz / denom;
            roots[i] = prev[i] - delta;
            max_change = max_change.max(delta.norm());
        }
        if !max_change.is_finite() {
            return Err(EmlError::NonConvergence {
                method: "durand_kerner",
                iterations: max_iter,
            });
        }
        if max_change < tol {
            break;
        }
    }

    // Newton polish: tighten each root with a few Newton steps
    // p'(z) = n*z^{n-1} + (n-1)*norm[n-1]*z^{n-2} + ... + 1*norm[1]
    // Horner for p'(z): start with n, loop i from n-2 down to 0: dacc = dacc*z + (i+1)*norm[i+1]
    for root in &mut roots {
        for _ in 0..10 {
            let pz = eval_at(*root);
            if pz.norm() < tol * tol {
                break;
            }
            if n >= 2 {
                // Compute p'(z) via Horner
                let mut dacc = Complex::new(n as f64, 0.0);
                for i in (0..n - 1).rev() {
                    dacc = dacc * *root + Complex::new((i + 1) as f64 * norm[i + 1], 0.0);
                }
                if dacc.norm() < 1e-300 {
                    break;
                }
                *root -= pz / dacc;
            } else {
                // degree 1: p'(z) = norm[1] (already monic means 1.0)
                break;
            }
        }
    }

    // Sort by real part, then imaginary
    roots.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });

    Ok(roots)
}

/// Apply primitive-part reduction then Durand-Kerner.
fn square_free_dk(poly: &Poly) -> Result<ComplexRoots, EmlError> {
    let prim = poly.primitive_part().map_err(|_| EmlError::NotSolvable)?;
    let mut roots = durand_kerner(&prim, 300, 1e-12)?;
    roots.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(ComplexRoots { roots })
}

/// Solve a polynomial for all roots, including complex ones.
pub fn solve_polynomial_complex(poly: &Poly) -> Result<ComplexRoots, EmlError> {
    let n = match poly.degree() {
        Some(d) => d,
        None => return Ok(ComplexRoots { roots: vec![] }),
    };

    match n {
        0 => Ok(ComplexRoots { roots: vec![] }),
        1 => {
            let a0 = ratio_to_f64(&poly.coeffs[0]);
            let a1 = ratio_to_f64(&poly.coeffs[1]);
            if a1.abs() < 1e-14 {
                return Ok(ComplexRoots { roots: vec![] });
            }
            Ok(ComplexRoots {
                roots: vec![Complex::new(-a0 / a1, 0.0)],
            })
        }
        2 => {
            let a0 = ratio_to_f64(&poly.coeffs[0]);
            let a1 = ratio_to_f64(&poly.coeffs[1]);
            let a2 = ratio_to_f64(&poly.coeffs[2]);
            if a2.abs() < 1e-14 {
                let p1 = Poly {
                    coeffs: poly.coeffs[..2].to_vec(),
                };
                return solve_polynomial_complex(&p1);
            }
            let disc = a1 * a1 - 4.0 * a2 * a0;
            let mut roots = if disc >= 0.0 {
                let sq = disc.sqrt();
                vec![
                    Complex::new((-a1 - sq) / (2.0 * a2), 0.0),
                    Complex::new((-a1 + sq) / (2.0 * a2), 0.0),
                ]
            } else {
                let sq = (-disc).sqrt();
                vec![
                    Complex::new(-a1 / (2.0 * a2), -sq / (2.0 * a2)),
                    Complex::new(-a1 / (2.0 * a2), sq / (2.0 * a2)),
                ]
            };
            roots.sort_by(|a, b| {
                a.re.partial_cmp(&b.re)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
            });
            Ok(ComplexRoots { roots })
        }
        _ => square_free_dk(poly),
    }
}

#[cfg(test)]
mod complex_root_tests {
    use super::*;
    use num_rational::Ratio;

    fn make_poly(int_coeffs: &[i64]) -> Poly {
        Poly {
            coeffs: int_coeffs.iter().map(|&n| Ratio::new(n, 1)).collect(),
        }
    }

    #[test]
    fn test_complex_roots_quadratic_positive_disc() {
        // x² - 5x + 6 = (x-2)(x-3)
        let p = make_poly(&[6, -5, 1]);
        let roots = solve_polynomial_complex(&p).unwrap();
        assert_eq!(roots.roots.len(), 2);
        let reals: Vec<f64> = roots.roots.iter().map(|c| c.re).collect();
        assert!((reals[0] - 2.0).abs() < 1e-9 || (reals[1] - 2.0).abs() < 1e-9);
        assert!((reals[0] - 3.0).abs() < 1e-9 || (reals[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_complex_roots_quadratic_negative_disc() {
        // x² + 1 = 0 → ±i
        let p = make_poly(&[1, 0, 1]);
        let roots = solve_polynomial_complex(&p).unwrap();
        assert_eq!(roots.roots.len(), 2);
        for r in &roots.roots {
            assert!(r.re.abs() < 1e-9, "real part should be 0: {}", r.re);
            assert!(
                (r.im.abs() - 1.0).abs() < 1e-9,
                "imag should be ±1: {}",
                r.im
            );
        }
        // real_roots filter must return empty
        assert!(roots.real_roots(1e-9).is_empty());
    }

    #[test]
    fn test_real_api_back_compat_x2_plus_1() {
        // solve_polynomial (real API) for x²+1 must still return no roots
        let p = make_poly(&[1, 0, 1]);
        let result = solve_polynomial(&p, 0).unwrap();
        assert!(
            result.roots.is_empty(),
            "x²+1 must have no real roots in the real API"
        );
    }

    #[test]
    fn test_complex_roots_x4_minus_1() {
        // x⁴ - 1 = (x-1)(x+1)(x-i)(x+i) → ±1, ±i
        let p = make_poly(&[-1, 0, 0, 0, 1]);
        let roots = solve_polynomial_complex(&p).unwrap();
        assert_eq!(roots.roots.len(), 4);
        // All |r| ≈ 1
        for r in &roots.roots {
            assert!(
                (r.norm() - 1.0).abs() < 1e-6,
                "root not on unit circle: {:?}",
                r
            );
        }
        // Verify p(r) ≈ 0 for each root
        for r in &roots.roots {
            let val = r.powi(4) - Complex::new(1.0, 0.0);
            assert!(val.norm() < 1e-9, "root doesn't satisfy p: {:?}", r);
        }
    }

    #[test]
    fn test_complex_roots_x5_minus_1() {
        // x⁵ - 1 → 5 fifth roots of unity
        let p = make_poly(&[-1, 0, 0, 0, 0, 1]);
        let roots = solve_polynomial_complex(&p).unwrap();
        assert_eq!(roots.roots.len(), 5);
        for r in &roots.roots {
            let val = r.powi(5) - Complex::new(1.0, 0.0);
            assert!(
                val.norm() < 1e-8,
                "5th root residual too large: {:?} → {:?}",
                r,
                val
            );
        }
    }

    #[test]
    fn test_complex_roots_x5_minus_x_minus_1() {
        // x⁵ - x - 1: one real root ≈ 1.1673
        let p = make_poly(&[-1, -1, 0, 0, 0, 1]);
        let roots = solve_polynomial_complex(&p).unwrap();
        assert_eq!(roots.roots.len(), 5);
        let real_roots = roots.real_roots(1e-6);
        assert_eq!(real_roots.len(), 1);
        assert!((real_roots[0] - 1.1673039).abs() < 1e-4);
        // Verify residual
        let x = real_roots[0];
        let residual = x.powi(5) - x - 1.0;
        assert!(residual.abs() < 1e-8);
    }

    #[test]
    fn test_lambert_wm1_two_branches() {
        use crate::numeric::lambert_wm1;
        let x = lambert_wm1(-0.2_f64).unwrap();
        assert!(x < -1.0, "W₋₁ should be < -1: {}", x);
        assert!(
            (x * x.exp() - (-0.2)).abs() < 1e-10,
            "W₋₁(-0.2) must satisfy w·eʷ=-0.2"
        );
    }
}
