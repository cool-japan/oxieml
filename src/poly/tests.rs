//! Tests for the poly module.

use num_rational::Ratio;
use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::named_const::NamedConst;

use super::factor::square_free_decomposition;
use super::ratio_to_f64;
use super::{MultiPoly, Poly, PolyError};

fn ratio(n: i64, d: i64) -> Ratio<i64> {
    Ratio::new(n, d)
}

fn poly_from_coeffs(coeffs: &[(i64, i64)]) -> Poly {
    Poly {
        coeffs: coeffs.iter().map(|&(n, d)| ratio(n, d)).collect(),
    }
}

// ── Basic construction ────────────────────────────────────────────────────────

#[test]
fn test_zero_poly() {
    let p = Poly::zero();
    assert!(p.is_zero());
    assert_eq!(p.degree(), None);
}

#[test]
fn test_constant_poly() {
    let p = Poly::constant(ratio(3, 2));
    assert!(!p.is_zero());
    assert_eq!(p.degree(), Some(0));
}

#[test]
fn test_monomial() {
    let p = Poly::monomial(3);
    assert_eq!(p.degree(), Some(3));
    assert_eq!(p.coeffs.len(), 4);
    assert_eq!(p.coeffs[3], ratio(1, 1));
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

#[test]
fn test_add_polys() {
    // (x + 1) + (x - 1) = 2x
    let p1 = poly_from_coeffs(&[(1, 1), (1, 1)]);
    let p2 = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let sum = p1.add(&p2).unwrap();
    assert_eq!(sum.degree(), Some(1));
    assert_eq!(sum.coeffs[1], ratio(2, 1));
    assert_eq!(sum.coeffs[0], ratio(0, 1));
}

#[test]
fn test_mul_polys() {
    // (x + 1) * (x - 1) = x^2 - 1
    let p1 = poly_from_coeffs(&[(1, 1), (1, 1)]);
    let p2 = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let prod = p1.mul(&p2).unwrap();
    let expected = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    assert_eq!(prod, expected.normalized());
}

// ── from_lowered / to_lowered roundtrip ──────────────────────────────────────

#[test]
fn test_from_lowered_const() {
    let op = LoweredOp::Const(3.0);
    let p = Poly::from_lowered(&op, 0).unwrap();
    assert_eq!(p.coeffs.len(), 1);
    assert_eq!(p.coeffs[0], ratio(3, 1));
}

#[test]
fn test_from_lowered_var() {
    let op = LoweredOp::Var(0);
    let p = Poly::from_lowered(&op, 0).unwrap();
    assert_eq!(p.coeffs.len(), 2);
    assert_eq!(p.coeffs[0], ratio(0, 1));
    assert_eq!(p.coeffs[1], ratio(1, 1));
}

#[test]
fn test_from_lowered_wrong_var() {
    let op = LoweredOp::Var(1);
    let result = Poly::from_lowered(&op, 0);
    assert_eq!(result, Err(PolyError::NotPolynomial));
}

#[test]
fn test_from_lowered_add() {
    // x + 2
    let op = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)));
    let p = Poly::from_lowered(&op, 0).unwrap();
    assert_eq!(p.coeffs[0], ratio(2, 1));
    assert_eq!(p.coeffs[1], ratio(1, 1));
}

#[test]
fn test_from_lowered_transcendental_fails() {
    let op = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let result = Poly::from_lowered(&op, 0);
    assert_eq!(result, Err(PolyError::NotPolynomial));
}

#[test]
fn test_to_lowered_zero() {
    let p = Poly::zero();
    let op = p.to_lowered(0);
    assert_eq!(op, LoweredOp::Const(0.0));
}

#[test]
fn test_roundtrip_from_to_lowered() {
    // x^2 + 3x + 2
    let op = LoweredOp::Add(
        Arc::new(LoweredOp::Add(
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Var(0)),
                Arc::new(LoweredOp::Var(0)),
            )),
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(3.0)),
                Arc::new(LoweredOp::Var(0)),
            )),
        )),
        Arc::new(LoweredOp::Const(2.0)),
    );
    let p = Poly::from_lowered(&op, 0).unwrap();
    let lowered_back = p.to_lowered(0);
    for x in [-2.0, 0.0, 1.0, 3.0] {
        let expected = x * x + 3.0 * x + 2.0;
        let got_poly = p.eval_f64(x);
        let got_lowered = lowered_back.eval(&[x]);
        assert!(
            (got_poly - expected).abs() < 1e-10,
            "poly eval mismatch at x={x}"
        );
        assert!(
            (got_lowered - expected).abs() < 1e-10,
            "lowered eval mismatch at x={x}"
        );
    }
}

// ── div_rem ───────────────────────────────────────────────────────────────────

#[test]
fn test_div_rem_identity() {
    // (x^2 - 1) = (x - 1) * (x + 1) + 0
    let dividend = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let divisor = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let (q, r) = dividend.div_rem(&divisor).unwrap();
    assert_eq!(q, poly_from_coeffs(&[(1, 1), (1, 1)]));
    assert!(r.is_zero());
}

#[test]
fn test_div_rem_with_remainder() {
    // (x^2 + 1) = (x) * x + 1
    let dividend = poly_from_coeffs(&[(1, 1), (0, 1), (1, 1)]);
    let divisor = poly_from_coeffs(&[(0, 1), (1, 1)]);
    let (_, r) = dividend.div_rem(&divisor).unwrap();
    assert_eq!(r, Poly::constant(ratio(1, 1)));
}

#[test]
fn test_div_by_zero() {
    let p = poly_from_coeffs(&[(1, 1), (1, 1)]);
    let result = p.div_rem(&Poly::zero());
    assert_eq!(result, Err(PolyError::DivByZero));
}

// ── GCD ───────────────────────────────────────────────────────────────────────

#[test]
fn test_gcd_basic() {
    // gcd(x^2 - 1, x - 1) = x - 1
    let a = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let b = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let g = Poly::gcd(&a, &b).unwrap();
    assert_eq!(g.degree(), Some(1));
    assert_eq!(g.coeffs[1], ratio(1, 1));
    assert_eq!(g.eval_f64(1.0), 0.0);
}

#[test]
fn test_gcd_coprime() {
    // gcd(x + 1, x + 2) = 1
    let a = poly_from_coeffs(&[(1, 1), (1, 1)]);
    let b = poly_from_coeffs(&[(2, 1), (1, 1)]);
    let g = Poly::gcd(&a, &b).unwrap();
    assert_eq!(g.degree(), Some(0));
}

// ── Differentiation ───────────────────────────────────────────────────────────

#[test]
fn test_diff_quadratic() {
    // d/dx (x^2 + 3x + 2) = 2x + 3
    let p = poly_from_coeffs(&[(2, 1), (3, 1), (1, 1)]);
    let dp = p.diff().unwrap();
    assert_eq!(dp.coeffs[0], ratio(3, 1));
    assert_eq!(dp.coeffs[1], ratio(2, 1));
}

#[test]
fn test_diff_constant() {
    let p = Poly::constant(ratio(5, 1));
    let dp = p.diff().unwrap();
    assert!(dp.is_zero());
}

// ── Square-free (Yun) ─────────────────────────────────────────────────────────

#[test]
fn test_square_free_simple() {
    // x^2 - 1 is already square-free.
    let p = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let sf = p.square_free().unwrap();
    assert_eq!(sf.degree(), Some(2));
}

#[test]
fn test_square_free_removes_double_root() {
    // (x-1)^2*(x+1) → square-free should have lower degree
    let x_minus_1_sq = poly_from_coeffs(&[(1, 1), (-2, 1), (1, 1)]);
    let x_plus_1 = poly_from_coeffs(&[(1, 1), (1, 1)]);
    let p = x_minus_1_sq.mul(&x_plus_1).unwrap();
    let sf = p.square_free().unwrap();
    assert!(sf.degree().unwrap() < p.degree().unwrap());
}

// ── Rational roots ────────────────────────────────────────────────────────────

#[test]
fn test_rational_roots_x_sq_minus_1() {
    // x^2 - 1 has roots ±1.
    let p = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let roots = p.rational_roots().unwrap();
    assert_eq!(roots.len(), 2);
    let vals: Vec<f64> = roots.iter().map(ratio_to_f64).collect();
    assert!(vals.iter().any(|&v| (v - 1.0).abs() < 1e-10));
    assert!(vals.iter().any(|&v| (v + 1.0).abs() < 1e-10));
}

#[test]
fn test_rational_roots_x_sq_minus_2() {
    // x^2 - 2 has no rational roots.
    let p = poly_from_coeffs(&[(-2, 1), (0, 1), (1, 1)]);
    let roots = p.rational_roots().unwrap();
    assert!(roots.is_empty());
}

// ── Real root isolation ───────────────────────────────────────────────────────

#[test]
fn test_isolate_real_roots_x_sq_minus_2() {
    // x^2 - 2 has roots ±√2 ≈ ±1.414.
    let p = poly_from_coeffs(&[(-2, 1), (0, 1), (1, 1)]);
    let roots = p.isolate_real_roots(-3.0, 3.0, 1e-8).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots, got {:?}", roots);
    let sqrt2 = 2.0_f64.sqrt();
    assert!(
        roots.iter().any(|&r| (r - sqrt2).abs() < 1e-6),
        "missing +√2 in {:?}",
        roots
    );
    assert!(
        roots.iter().any(|&r| (r + sqrt2).abs() < 1e-6),
        "missing -√2 in {:?}",
        roots
    );
}

#[test]
fn test_isolate_real_roots_cubic() {
    // x^3 - x = x*(x-1)*(x+1), roots at -1, 0, 1.
    let p = poly_from_coeffs(&[(0, 1), (-1, 1), (0, 1), (1, 1)]);
    let roots = p.isolate_real_roots(-2.0, 2.0, 1e-8).unwrap();
    assert!(roots.len() >= 3, "expected >=3 roots, got {:?}", roots);
    assert!(roots.iter().any(|&r| r.abs() < 1e-6), "missing root at 0");
    assert!(
        roots.iter().any(|&r| (r - 1.0).abs() < 1e-6),
        "missing root at 1"
    );
    assert!(
        roots.iter().any(|&r| (r + 1.0).abs() < 1e-6),
        "missing root at -1"
    );
}

// ── Overflow detection ────────────────────────────────────────────────────────

#[test]
fn test_coeff_overflow_no_panic() {
    let big = Poly::constant(ratio(i64::MAX / 2, 1));
    let result = big.mul(&big);
    assert_eq!(result, Err(PolyError::CoeffOverflow));
}

// ── MultiPoly ─────────────────────────────────────────────────────────────────

#[test]
fn test_multipoly_zero() {
    let p = MultiPoly::zero(3);
    assert!(p.is_zero());
}

#[test]
fn test_multipoly_from_lowered_var() {
    let op = LoweredOp::Var(0);
    let p = MultiPoly::from_lowered(&op, 2).unwrap();
    assert!(!p.is_zero());
    assert_eq!(p.eval_f64(&[3.0, 5.0]), 3.0);
}

#[test]
fn test_multipoly_from_lowered_product() {
    let op = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let p = MultiPoly::from_lowered(&op, 2).unwrap();
    assert!((p.eval_f64(&[3.0, 4.0]) - 12.0).abs() < 1e-10);
}

#[test]
fn test_multipoly_add() {
    let op_a = LoweredOp::Var(0);
    let op_b = LoweredOp::Var(1);
    let a = MultiPoly::from_lowered(&op_a, 2).unwrap();
    let b = MultiPoly::from_lowered(&op_b, 2).unwrap();
    let sum = a.add(&b).unwrap();
    // sum = x0 + x1; at (2, 3) should be 5.
    assert!((sum.eval_f64(&[2.0, 3.0]) - 5.0).abs() < 1e-10);
}

#[test]
fn test_multipoly_to_lowered_roundtrip() {
    // x0^2 + x1 in 2 variables.
    let op = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Var(0)),
        )),
        Arc::new(LoweredOp::Var(1)),
    );
    let p = MultiPoly::from_lowered(&op, 2).unwrap();
    let back = p.to_lowered();
    for (x0, x1) in [(0.0, 1.0), (1.0, 2.0), (3.0, -1.0)] {
        let expected = x0 * x0 + x1;
        let poly_val = p.eval_f64(&[x0, x1]);
        let lowered_val = back.eval(&[x0, x1]);
        assert!(
            (poly_val - expected).abs() < 1e-10,
            "multipoly eval mismatch at ({x0},{x1})"
        );
        assert!(
            (lowered_val - expected).abs() < 1e-10,
            "lowered eval mismatch at ({x0},{x1})"
        );
    }
}

#[test]
fn test_multipoly_transcendental_fails() {
    let op = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    let result = MultiPoly::from_lowered(&op, 1);
    assert_eq!(result, Err(PolyError::NotPolynomial));
}

// ── NamedConst handling ───────────────────────────────────────────────────────

#[test]
fn test_from_lowered_named_const_half() {
    let op = LoweredOp::NamedConst(NamedConst::Half);
    let p = Poly::from_lowered(&op, 0).unwrap();
    assert_eq!(p.degree(), Some(0));
    assert_eq!(p.coeffs[0], ratio(1, 2));
}

// ── 12 new tests ──────────────────────────────────────────────────────────────

#[test]
fn test_square_free_decomposition_x_sq_minus_1() {
    // x^2 - 1 = (x-1)(x+1), square-free → one factor of multiplicity 1
    let p = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let sfd = square_free_decomposition(&p).unwrap();
    assert!(!sfd.is_empty());
    // All multiplicities should be 1 since it's square-free
    for (_, mult) in &sfd {
        assert_eq!(*mult, 1, "expected multiplicity 1, got {mult}");
    }
}

#[test]
fn test_square_free_decomposition_repeated_root() {
    // (x - 1)^2 = x^2 - 2x + 1
    let p = poly_from_coeffs(&[(1, 1), (-2, 1), (1, 1)]);
    let sfd = square_free_decomposition(&p).unwrap();
    // Should have a factor with multiplicity 2
    let max_mult = sfd.iter().map(|(_, m)| *m).max().unwrap_or(0);
    assert!(
        max_mult >= 2,
        "expected max multiplicity >=2, got {max_mult}"
    );
}

#[test]
fn test_factor_difference_of_squares() {
    // x^2 - 1 = (x-1)(x+1)
    let p = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let factored = p.factor().unwrap();
    // Should find 2 linear factors
    assert_eq!(
        factored.factors.len(),
        2,
        "expected 2 factors, got {:?}",
        factored.factors.len()
    );
    // Product of factors should equal original (check at a few points)
    for x in [-2.0, 0.5, 1.5, 3.0] {
        let orig = p.eval_f64(x);
        let prod: f64 = factored
            .factors
            .iter()
            .fold(ratio_to_f64(&factored.content), |acc, (f, mult)| {
                acc * f.eval_f64(x).powi(*mult as i32)
            });
        assert!(
            (orig - prod).abs() < 1e-10,
            "product mismatch at x={x}: orig={orig}, prod={prod}"
        );
    }
}

#[test]
fn test_factor_irreducible_quadratic() {
    // x^2 + 1 is irreducible over rationals
    let p = poly_from_coeffs(&[(1, 1), (0, 1), (1, 1)]);
    let factored = p.factor().unwrap();
    // Should have 1 factor (the polynomial itself)
    assert_eq!(factored.factors.len(), 1);
    assert_eq!(factored.factors[0].1, 1);
}

#[test]
fn test_factor_cubic_three_roots() {
    // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    // coeffs: -6 + 11x - 6x^2 + x^3  → [(-6,1), (11,1), (-6,1), (1,1)]
    let p = poly_from_coeffs(&[(-6, 1), (11, 1), (-6, 1), (1, 1)]);
    let factored = p.factor().unwrap();
    assert_eq!(factored.factors.len(), 3, "expected 3 linear factors");
    // Check product reconstructs original at a test point
    let x = 4.0;
    let orig = p.eval_f64(x);
    let prod: f64 = factored
        .factors
        .iter()
        .fold(ratio_to_f64(&factored.content), |acc, (f, mult)| {
            acc * f.eval_f64(x).powi(*mult as i32)
        });
    assert!(
        (orig - prod).abs() < 1e-8,
        "product mismatch: orig={orig}, prod={prod}"
    );
}

#[test]
fn test_factor_with_repeated_root() {
    // (x - 2)^2 = x^2 - 4x + 4
    let p = poly_from_coeffs(&[(4, 1), (-4, 1), (1, 1)]);
    let factored = p.factor().unwrap();
    // Should have 1 factor with multiplicity 2
    assert!(!factored.factors.is_empty());
    let total_mult: usize = factored.factors.iter().map(|(_, m)| *m).sum();
    assert_eq!(
        total_mult, 2,
        "total degree should be 2 but got {total_mult}"
    );
}

#[test]
fn test_resultant_vanishes_common_root() {
    // res(x-1, x-1) should be 0 (common root)
    let a = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let b = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let res = Poly::resultant(&a, &b).unwrap();
    assert_eq!(res, 0, "resultant of identical polynomials should be 0");
}

#[test]
fn test_resultant_coprime() {
    // res(x - 1, x - 2): for linear polys ax+b, cx+d: res = ad - bc
    // a=1,b=-1,c=1,d=-2: res = 1*(-2) - (-1)*1 = -2+1 = -1
    let a = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let b = poly_from_coeffs(&[(-2, 1), (1, 1)]);
    let res = Poly::resultant(&a, &b).unwrap();
    assert_ne!(res, 0, "coprime polynomials should have non-zero resultant");
}

#[test]
fn test_discriminant_quadratic() {
    // x^2 - 5x + 6 = (x-2)(x-3), discriminant = 25 - 24 = 1
    let p = poly_from_coeffs(&[(6, 1), (-5, 1), (1, 1)]);
    let disc = p.discriminant().unwrap();
    assert_eq!(disc, 1, "expected discriminant 1, got {disc}");
}

#[test]
fn test_discriminant_negative_no_real_roots() {
    // x^2 + 1, discriminant = 0^2 - 4*1*1 = -4
    let p = poly_from_coeffs(&[(1, 1), (0, 1), (1, 1)]);
    let disc = p.discriminant().unwrap();
    assert!(disc < 0, "expected negative discriminant, got {disc}");
}

#[test]
fn test_content_and_primitive_part() {
    // 2x^2 + 4x + 6 has content 2
    let p = poly_from_coeffs(&[(6, 1), (4, 1), (2, 1)]);
    let content = p.content();
    assert_eq!(content, ratio(2, 1), "expected content 2, got {content}");
    let prim = p.primitive_part().unwrap();
    // Primitive part should be x^2 + 2x + 3
    assert_eq!(prim.eval_f64(1.0), 6.0, "primitive part at x=1 should be 6");
    assert_eq!(prim.eval_f64(0.0), 3.0, "primitive part at x=0 should be 3");
}

#[test]
fn test_factorization_product_property() {
    // For any polynomial f, f = content * product(factors^mult)
    // Test with x^3 - x = x(x-1)(x+1)
    let p = poly_from_coeffs(&[(0, 1), (-1, 1), (0, 1), (1, 1)]);
    let factored = p.factor().unwrap();
    // Verify at multiple points
    for x in [-2.0, -0.5, 0.5, 1.5, 2.5] {
        let orig = p.eval_f64(x);
        let prod: f64 = factored
            .factors
            .iter()
            .fold(ratio_to_f64(&factored.content), |acc, (f, mult)| {
                acc * f.eval_f64(x).powi(*mult as i32)
            });
        assert!(
            (orig - prod).abs() < 1e-8,
            "product property failed at x={x}: orig={orig}, prod={prod}"
        );
    }
}
