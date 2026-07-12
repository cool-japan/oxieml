//! Tests for the poly module.

use num_bigint::{BigInt, BigUint};
use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::named_const::NamedConst;

use super::factor::square_free_decomposition;
use super::modular::{ModPoly, MpPoly};
use super::{
    Coeff, Factorization, MultiPoly, Poly, PolyError, coeff_from_i64, coeff_one, coeff_zero,
    f64_to_ratio, ratio_to_f64,
};

fn ratio(n: i64, d: i64) -> Coeff {
    Coeff::new(BigInt::from(n), BigInt::from(d))
}

fn poly_from_coeffs(coeffs: &[(i64, i64)]) -> Poly {
    Poly::from_ratios(coeffs.iter().map(|&(n, d)| ratio(n, d)).collect())
}

/// 10^18, the coefficient used by the overflow-elimination test.
fn ten_pow_18() -> BigInt {
    BigInt::from(10i64).pow(18)
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

#[test]
fn test_from_int_coeffs() {
    // 1 + 2x + 3x^2
    let p = Poly::from_int_coeffs(&[1, 2, 3]);
    assert_eq!(p.degree(), Some(2));
    assert_eq!(p.coeffs[0], coeff_one());
    assert_eq!(p.coeffs[2], coeff_from_i64(3));
    // Trailing zeros are normalized away.
    let q = Poly::from_int_coeffs(&[1, 0, 0]);
    assert_eq!(q.degree(), Some(0));
}

#[test]
fn test_from_ratios() {
    let p = Poly::from_ratios(vec![ratio(1, 2), ratio(-3, 4)]);
    assert_eq!(p.degree(), Some(1));
    assert_eq!(p.coeffs[0], ratio(1, 2));
    assert_eq!(p.coeffs[1], ratio(-3, 4));
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

#[test]
fn test_pow_binary_exponentiation() {
    // (x + 1)^5 = x^5 + 5x^4 + 10x^3 + 10x^2 + 5x + 1
    let p = Poly::from_int_coeffs(&[1, 1]);
    let fifth = p.pow(5).unwrap();
    assert_eq!(fifth, Poly::from_int_coeffs(&[1, 5, 10, 10, 5, 1]));
    assert_eq!(p.pow(0).unwrap(), Poly::constant(coeff_one()));
}

// ── Overflow elimination (N1) ─────────────────────────────────────────────────

#[test]
fn test_no_overflow_big_coefficients_cubed() {
    // [(10^18) x + 1]^3 must succeed exactly — this used to be CoeffOverflow.
    let big = ten_pow_18();
    let p = Poly::from_ratios(vec![coeff_one(), Coeff::from_integer(big.clone())]);
    let cubed = p.pow(3).expect("exact BigRational pow must not overflow");

    assert_eq!(cubed.degree(), Some(3));
    // Binomial expansion: 1 + 3·b·x + 3·b²·x² + b³·x³
    let b = Coeff::from_integer(big);
    assert_eq!(cubed.coeffs[0], coeff_one());
    assert_eq!(cubed.coeffs[1], &coeff_from_i64(3) * &b);
    assert_eq!(cubed.coeffs[2], &coeff_from_i64(3) * &(&b * &b));
    assert_eq!(cubed.coeffs[3], &(&b * &b) * &b);
}

#[test]
fn test_no_overflow_repeated_squaring_product() {
    // (i64::MAX/2)^2 previously produced PolyError::CoeffOverflow.
    let half_max = coeff_from_i64(i64::MAX / 2);
    let big = Poly::constant(half_max.clone());
    let squared = big.mul(&big).expect("exact multiplication cannot overflow");
    assert_eq!(squared.coeffs[0], &half_max * &half_max);
}

#[test]
fn test_no_overflow_gcd_of_huge_polys() {
    // gcd((10^18 x + 1)·(x - 3), (10^18 x + 1)·(x + 7)) == x + 10^-18 (monic).
    let base = Poly::from_ratios(vec![coeff_one(), Coeff::from_integer(ten_pow_18())]);
    let a = base.mul(&Poly::from_int_coeffs(&[-3, 1])).unwrap();
    let b = base.mul(&Poly::from_int_coeffs(&[7, 1])).unwrap();
    let g = Poly::gcd(&a, &b).unwrap();
    assert_eq!(g.degree(), Some(1));
    // Monic form of 10^18·x + 1.
    let expected = Poly::from_ratios(vec![
        Coeff::new(BigInt::from(1i32), ten_pow_18()),
        coeff_one(),
    ]);
    assert_eq!(g, expected);
}

// ── f64_to_ratio (continued fraction / Stern–Brocot) ──────────────────────────

#[test]
fn test_f64_to_ratio_one_third() {
    let r = f64_to_ratio(1.0 / 3.0).unwrap();
    assert_eq!(r, ratio(1, 3), "1.0/3.0 should rationalize to 1/3");
}

#[test]
fn test_f64_to_ratio_355_over_113() {
    let r = f64_to_ratio(355.0 / 113.0).unwrap();
    assert_eq!(
        r,
        ratio(355, 113),
        "355.0/113.0 should rationalize to 355/113"
    );
}

#[test]
fn test_f64_to_ratio_point_one() {
    let r = f64_to_ratio(0.1).unwrap();
    assert_eq!(r, ratio(1, 10), "0.1 should rationalize to 1/10");
}

#[test]
fn test_f64_to_ratio_nan_and_infinities_err() {
    assert_eq!(f64_to_ratio(f64::NAN), Err(PolyError::NotPolynomial));
    assert_eq!(f64_to_ratio(f64::INFINITY), Err(PolyError::NotPolynomial));
    assert_eq!(
        f64_to_ratio(f64::NEG_INFINITY),
        Err(PolyError::NotPolynomial)
    );
}

#[test]
fn test_f64_to_ratio_no_denominator_cap() {
    // 1/1009 has a denominator well past the old ≤1000 cap and must now work.
    let r = f64_to_ratio(1.0 / 1009.0).unwrap();
    assert_eq!(r, ratio(1, 1009));

    // 1/123457 likewise.
    let r = f64_to_ratio(1.0 / 123_457.0).unwrap();
    assert_eq!(r, ratio(1, 123_457));
}

#[test]
fn test_f64_to_ratio_round_trips_exactly() {
    // Whatever rational we produce must convert back to the identical f64.
    for v in [
        0.0,
        -0.0,
        1.0,
        -2.5,
        0.1,
        0.2,
        0.3,
        1.0 / 3.0,
        2.0 / 7.0,
        std::f64::consts::PI,
        std::f64::consts::E,
        1e300,
        1e-300,
        f64::MIN_POSITIVE,
        123_456_789.987_654_32,
    ] {
        let r = f64_to_ratio(v).expect("finite value must rationalize");
        assert_eq!(ratio_to_f64(&r), v, "round-trip failed for {v}");
    }
}

#[test]
fn test_f64_to_ratio_is_simplest() {
    // 0.5 is dyadic and exact; the simplest round-tripping rational is 1/2.
    assert_eq!(f64_to_ratio(0.5).unwrap(), ratio(1, 2));
    // Negative values keep the sign on the numerator.
    let r = f64_to_ratio(-1.0 / 3.0).unwrap();
    assert_eq!(r, ratio(-1, 3));
}

// ── ratio_to_f64 (correct rounding, no overflow to NaN) ───────────────────────

#[test]
fn test_ratio_to_f64_huge_numer_and_denom() {
    // (3 · 10^400) / (2 · 10^400) = 1.5 — a naive f64 cast would give inf/inf = NaN.
    let scale = BigInt::from(10i32).pow(400);
    let r = Coeff::new(BigInt::from(3i32) * &scale, BigInt::from(2i32) * &scale);
    assert_eq!(ratio_to_f64(&r), 1.5);
}

#[test]
fn test_ratio_to_f64_saturates() {
    let huge = Coeff::from_integer(BigInt::from(10i32).pow(400));
    assert_eq!(ratio_to_f64(&huge), f64::INFINITY);
    assert_eq!(ratio_to_f64(&-&huge), f64::NEG_INFINITY);

    let tiny = Coeff::new(BigInt::from(1i32), BigInt::from(10i32).pow(400));
    assert_eq!(ratio_to_f64(&tiny), 0.0);
    assert_eq!(ratio_to_f64(&coeff_zero()), 0.0);
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
fn test_from_lowered_non_finite_const_fails() {
    let op = LoweredOp::Const(f64::NAN);
    assert_eq!(Poly::from_lowered(&op, 0), Err(PolyError::NotPolynomial));
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

#[test]
fn test_div_rem_reconstructs_dividend() {
    // Random-ish rational dividend/divisor: a = q·b + r must hold exactly.
    let a = poly_from_coeffs(&[(1, 3), (-5, 7), (2, 1), (9, 4)]);
    let b = poly_from_coeffs(&[(2, 5), (-1, 1)]);
    let (q, r) = a.div_rem(&b).unwrap();
    let reconstructed = q.mul(&b).unwrap().add(&r).unwrap();
    assert_eq!(reconstructed, a.normalized());
    assert!(r.degree().unwrap_or(0) < b.degree().unwrap());
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
    assert!(roots.contains(&ratio(1, 1)));
    assert!(roots.contains(&ratio(-1, 1)));
}

#[test]
fn test_rational_roots_x_sq_minus_2() {
    // x^2 - 2 has no rational roots.
    let p = poly_from_coeffs(&[(-2, 1), (0, 1), (1, 1)]);
    let roots = p.rational_roots().unwrap();
    assert!(roots.is_empty());
}

#[test]
fn test_rational_roots_fractional() {
    // 6x^2 - 5x + 1 = (2x - 1)(3x - 1): roots 1/2 and 1/3.
    let p = Poly::from_int_coeffs(&[1, -5, 6]);
    let roots = p.rational_roots().unwrap();
    assert_eq!(roots, vec![ratio(1, 3), ratio(1, 2)]);
}

#[test]
fn test_rational_roots_with_zero_root_and_big_coeffs() {
    // x·(10^18·x - 1)·(x + 2): roots 0, 1/10^18, -2.
    let big = ten_pow_18();
    let factor = Poly::from_ratios(vec![-coeff_one(), Coeff::from_integer(big.clone())]);
    let p = Poly::from_int_coeffs(&[0, 1])
        .mul(&factor)
        .unwrap()
        .mul(&Poly::from_int_coeffs(&[2, 1]))
        .unwrap();
    let roots = p.rational_roots().unwrap();
    assert_eq!(roots.len(), 3, "expected 3 rational roots, got {roots:?}");
    assert!(roots.contains(&coeff_zero()));
    assert!(roots.contains(&ratio(-2, 1)));
    assert!(roots.contains(&Coeff::new(BigInt::from(1i32), big)));
}

// ── Real root isolation ───────────────────────────────────────────────────────

#[test]
fn test_isolate_real_roots_x_sq_minus_2() {
    // x^2 - 2 has roots ±√2 ≈ ±1.414.
    let p = poly_from_coeffs(&[(-2, 1), (0, 1), (1, 1)]);
    let roots = p.isolate_real_roots(-3.0, 3.0, 1e-8).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots, got {roots:?}");
    let sqrt2 = 2.0_f64.sqrt();
    assert!(
        roots.iter().any(|&r| (r - sqrt2).abs() < 1e-6),
        "missing +√2 in {roots:?}"
    );
    assert!(
        roots.iter().any(|&r| (r + sqrt2).abs() < 1e-6),
        "missing -√2 in {roots:?}"
    );
}

#[test]
fn test_isolate_real_roots_cubic() {
    // x^3 - x = x*(x-1)*(x+1), roots at -1, 0, 1.
    let p = poly_from_coeffs(&[(0, 1), (-1, 1), (0, 1), (1, 1)]);
    let roots = p.isolate_real_roots(-2.0, 2.0, 1e-8).unwrap();
    assert!(roots.len() >= 3, "expected >=3 roots, got {roots:?}");
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
fn test_multipoly_no_overflow() {
    // (10^18·x0·x1)^3 is exact now.
    let big = Coeff::from_integer(ten_pow_18());
    let x0x1 = MultiPoly::from_lowered(
        &LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1))),
        2,
    )
    .unwrap();
    let scaled = x0x1.scale(&big).unwrap();
    let cubed = scaled.pow(3).expect("exact multivariate pow");
    let expected_coeff = &(&big * &big) * &big;
    assert_eq!(cubed.terms.get(&vec![3u32, 3u32]), Some(&expected_coeff));
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

// ── Square-free decomposition / factorization ────────────────────────────────

#[test]
fn test_square_free_decomposition_x_sq_minus_1() {
    // x^2 - 1 = (x-1)(x+1), square-free → one factor of multiplicity 1
    let p = poly_from_coeffs(&[(-1, 1), (0, 1), (1, 1)]);
    let sfd = square_free_decomposition(&p).unwrap();
    assert!(!sfd.is_empty());
    for (_, mult) in &sfd {
        assert_eq!(*mult, 1, "expected multiplicity 1, got {mult}");
    }
}

#[test]
fn test_square_free_decomposition_repeated_root() {
    // (x - 1)^2 = x^2 - 2x + 1
    let p = poly_from_coeffs(&[(1, 1), (-2, 1), (1, 1)]);
    let sfd = square_free_decomposition(&p).unwrap();
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
    assert_eq!(
        factored.factors.len(),
        2,
        "expected 2 factors, got {:?}",
        factored.factors.len()
    );
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
    assert_eq!(factored.factors.len(), 1);
    assert_eq!(factored.factors[0].1, 1);
}

#[test]
fn test_factor_cubic_three_roots() {
    // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    let p = poly_from_coeffs(&[(-6, 1), (11, 1), (-6, 1), (1, 1)]);
    let factored = p.factor().unwrap();
    assert_eq!(factored.factors.len(), 3, "expected 3 linear factors");
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
    assert!(!factored.factors.is_empty());
    let total_mult: usize = factored.factors.iter().map(|(_, m)| *m).sum();
    assert_eq!(
        total_mult, 2,
        "total degree should be 2 but got {total_mult}"
    );
}

#[test]
fn test_factor_quartic_into_two_quadratics() {
    // (x^2 + 1)(x^2 + 2) = x^4 + 3x^2 + 2 has no rational roots; Zassenhaus splits it.
    let p = Poly::from_int_coeffs(&[2, 0, 3, 0, 1]);
    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        2,
        "expected 2 irreducible quadratics, got {:?}",
        factored.factors
    );
    let mut product = Poly::constant(factored.content.clone());
    for (f, mult) in &factored.factors {
        product = product.mul(&f.pow(*mult).unwrap()).unwrap();
    }
    assert_eq!(product, p.normalized());
}

// ── Resultant / discriminant (exact, arbitrary precision) ─────────────────────

#[test]
fn test_resultant_vanishes_common_root() {
    // res(x-1, x-1) == 0 (common root)
    let a = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let b = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let res = Poly::resultant(&a, &b).unwrap();
    assert_eq!(res, coeff_zero());
}

#[test]
fn test_resultant_coprime() {
    // res(x - 1, x - 2) = -1 for linear a·x+b, c·x+d: res = ad - bc.
    let a = poly_from_coeffs(&[(-1, 1), (1, 1)]);
    let b = poly_from_coeffs(&[(-2, 1), (1, 1)]);
    let res = Poly::resultant(&a, &b).unwrap();
    assert_ne!(res, coeff_zero());
    assert_eq!(res, coeff_from_i64(-1));
}

#[test]
fn test_resultant_x2_minus_2_and_x2_minus_3_is_one() {
    // res(x² − 2, x² − 3) == 1 exactly.
    let a = Poly::from_int_coeffs(&[-2, 0, 1]);
    let b = Poly::from_int_coeffs(&[-3, 0, 1]);
    let res = Poly::resultant(&a, &b).unwrap();
    assert_eq!(res, coeff_one(), "res(x²−2, x²−3) must be exactly 1");
}

#[test]
fn test_discriminant_quadratic() {
    // x^2 - 5x + 6 = (x-2)(x-3), discriminant = 25 - 24 = 1
    let p = poly_from_coeffs(&[(6, 1), (-5, 1), (1, 1)]);
    assert_eq!(p.discriminant().unwrap(), coeff_one());
}

#[test]
fn test_discriminant_negative_no_real_roots() {
    // x^2 + 1, discriminant = 0^2 - 4*1*1 = -4
    let p = poly_from_coeffs(&[(1, 1), (0, 1), (1, 1)]);
    let disc = p.discriminant().unwrap();
    assert!(
        disc < coeff_zero(),
        "expected negative discriminant: {disc}"
    );
    assert_eq!(disc, coeff_from_i64(-4));
}

#[test]
fn test_discriminant_exact_arbitrary_precision() {
    // disc(x² + b·x + c) == b² − 4c for b = 10^10 (b² = 10^20 overflows nothing now).
    let b: i64 = 10_000_000_000;
    let c: i64 = 7;
    let p = Poly::from_int_coeffs(&[c, b, 1]);
    let disc = p.discriminant().unwrap();

    let b_big = Coeff::from_integer(BigInt::from(b));
    let expected = &(&b_big * &b_big) - &(&coeff_from_i64(4) * &coeff_from_i64(c));
    assert_eq!(disc, expected, "disc must be exactly b² − 4c");

    // Sanity: b² − 4c = 10^20 − 28, which does not fit in the old i64 return type.
    let expected_int = BigInt::from(10i32).pow(20) - BigInt::from(28i32);
    assert_eq!(disc, Coeff::from_integer(expected_int));
}

#[test]
fn test_discriminant_cubic_depressed() {
    // disc(x³ + p·x + q) = −4p³ − 27q²; for p = -3, q = 1 → 108 - 27 = 81.
    let p = Poly::from_int_coeffs(&[1, -3, 0, 1]);
    assert_eq!(p.discriminant().unwrap(), coeff_from_i64(81));
}

// ── Content and primitive part ────────────────────────────────────────────────

#[test]
fn test_content_and_primitive_part() {
    // 2x^2 + 4x + 6 has content 2
    let p = poly_from_coeffs(&[(6, 1), (4, 1), (2, 1)]);
    let content = p.content();
    assert_eq!(content, ratio(2, 1), "expected content 2, got {content}");
    let prim = p.primitive_part().unwrap();
    assert_eq!(prim.eval_f64(1.0), 6.0, "primitive part at x=1 should be 6");
    assert_eq!(prim.eval_f64(0.0), 3.0, "primitive part at x=0 should be 3");
}

#[test]
fn test_content_rational_coefficients() {
    // (1/2)x + 1/3 has content 1/6 and primitive part 3x + 2.
    let p = poly_from_coeffs(&[(1, 3), (1, 2)]);
    assert_eq!(p.content(), ratio(1, 6));
    assert_eq!(p.primitive_part().unwrap(), Poly::from_int_coeffs(&[2, 3]));
}

#[test]
fn test_primitive_part_negative_leading() {
    // -2x - 4 → primitive part x + 2 (leading coefficient made positive).
    let p = Poly::from_int_coeffs(&[-4, -2]);
    assert_eq!(p.primitive_part().unwrap(), Poly::from_int_coeffs(&[2, 1]));
}

#[test]
fn test_factorization_product_property() {
    // For any polynomial f, f = content * product(factors^mult)
    let p = poly_from_coeffs(&[(0, 1), (-1, 1), (0, 1), (1, 1)]);
    let factored = p.factor().unwrap();
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

// ── Zassenhaus factorization (arbitrary degree) ───────────────────────────────

/// Multiply a list of polynomials together.
fn product_of(polys: &[Poly]) -> Poly {
    polys
        .iter()
        .fold(Poly::from_int_coeffs(&[1]), |acc, p| acc.mul(p).unwrap())
}

/// Rebuild `content · Π factorᵢ^multᵢ` from a factorization.
fn remultiply(factored: &Factorization) -> Poly {
    let mut product = Poly::constant(factored.content.clone());
    for (factor, mult) in &factored.factors {
        product = product.mul(&factor.pow(*mult).unwrap()).unwrap();
    }
    product.normalized()
}

#[test]
fn test_factor_x_squared_minus_two_is_irreducible() {
    // x² − 2 is irreducible over ℚ (√2 is irrational).
    let p = Poly::from_int_coeffs(&[-2, 0, 1]);
    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        1,
        "x² − 2 must be irreducible, got {:?}",
        factored.factors
    );
    assert_eq!(factored.factors[0].0, p);
    assert_eq!(factored.factors[0].1, 1);
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_swinnerton_dyer_degree_4_is_irreducible() {
    // SD(2,3) = Π (x ± √2 ± √3) = x⁴ − 10x² + 1.
    // Irreducible over ℚ, yet it factors into linear/quadratic pieces modulo
    // *every* prime: the classical worst case for Zassenhaus recombination,
    // which must reject all C(4,1) + C(4,2) = 10 subsets.
    let p = Poly::from_int_coeffs(&[1, 0, -10, 0, 1]);
    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        1,
        "SD(2,3) must be irreducible, got {:?}",
        factored.factors
    );
    assert_eq!(factored.factors[0].0, p);
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_swinnerton_dyer_degree_8_is_irreducible() {
    // SD(2,3,5) = Π (x ± √2 ± √3 ± √5) = x⁸ − 40x⁶ + 352x⁴ − 960x² + 576.
    // Splits into eight linear factors modulo every prime, so recombination has
    // to reject all 8 + 28 + 56 + 70 = 162 subsets of size ≤ 4 before it can
    // conclude irreducibility.
    let p = Poly::from_int_coeffs(&[576, 0, -960, 0, 352, 0, -40, 0, 1]);
    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        1,
        "SD(2,3,5) must be irreducible, got {:?}",
        factored.factors
    );
    assert_eq!(factored.factors[0].0, p);
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_three_irreducible_factors() {
    // (x² + 1)(x² + x + 1)(x³ − 2): degrees 2, 2, 3, none with a rational root.
    let a = Poly::from_int_coeffs(&[1, 0, 1]);
    let b = Poly::from_int_coeffs(&[1, 1, 1]);
    let c = Poly::from_int_coeffs(&[-2, 0, 0, 1]);
    let p = product_of(&[a.clone(), b.clone(), c.clone()]);
    assert_eq!(p.degree(), Some(7));

    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        3,
        "expected 3 factors, got {:?}",
        factored.factors
    );
    for expected in [&a, &b, &c] {
        assert!(
            factored
                .factors
                .iter()
                .any(|(f, m)| f == expected && *m == 1),
            "missing factor {expected:?} in {:?}",
            factored.factors
        );
    }
    assert_eq!(remultiply(&factored), p.normalized());
}

#[test]
fn test_factor_cyclotomic_phi_8_is_irreducible() {
    // Φ₈ = x⁴ + 1 is irreducible over ℚ but reducible modulo *every* prime.
    let p = Poly::from_int_coeffs(&[1, 0, 0, 0, 1]);
    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        1,
        "Φ₈ must be irreducible, got {:?}",
        factored.factors
    );
    assert_eq!(factored.factors[0].0, p);
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_x6_minus_1_into_cyclotomics() {
    // x⁶ − 1 = Φ₁ Φ₂ Φ₃ Φ₆ = (x − 1)(x + 1)(x² + x + 1)(x² − x + 1).
    let p = Poly::from_int_coeffs(&[-1, 0, 0, 0, 0, 0, 1]);
    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        4,
        "expected 4 cyclotomic factors, got {:?}",
        factored.factors
    );
    let expected = [
        Poly::from_int_coeffs(&[-1, 1]),    // Φ₁ = x − 1
        Poly::from_int_coeffs(&[1, 1]),     // Φ₂ = x + 1
        Poly::from_int_coeffs(&[1, 1, 1]),  // Φ₃ = x² + x + 1
        Poly::from_int_coeffs(&[1, -1, 1]), // Φ₆ = x² − x + 1
    ];
    for e in &expected {
        assert!(
            factored.factors.iter().any(|(f, m)| f == e && *m == 1),
            "missing cyclotomic {e:?} in {:?}",
            factored.factors
        );
    }
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_x12_minus_1_many_factors() {
    // x¹² − 1 = Φ₁ Φ₂ Φ₃ Φ₄ Φ₆ Φ₁₂: six irreducible factors of degrees
    // 1, 1, 2, 2, 2, 4.
    let mut coeffs = vec![0i64; 13];
    coeffs[0] = -1;
    coeffs[12] = 1;
    let p = Poly::from_int_coeffs(&coeffs);

    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        6,
        "expected 6 factors, got {:?}",
        factored.factors
    );
    let mut degrees: Vec<usize> = factored
        .factors
        .iter()
        .map(|(f, _)| f.degree().unwrap())
        .collect();
    degrees.sort_unstable();
    assert_eq!(degrees, vec![1, 1, 2, 2, 2, 4]);
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_x11_minus_1_gives_degree_10_irreducible() {
    // x¹¹ − 1 = (x − 1) · Φ₁₁ with Φ₁₁ = 1 + x + … + x¹⁰ irreducible.
    let mut coeffs = vec![0i64; 12];
    coeffs[0] = -1;
    coeffs[11] = 1;
    let p = Poly::from_int_coeffs(&coeffs);

    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        2,
        "expected (x − 1)·Φ₁₁, got {:?}",
        factored.factors
    );
    let mut degrees: Vec<usize> = factored
        .factors
        .iter()
        .map(|(f, _)| f.degree().unwrap())
        .collect();
    degrees.sort_unstable();
    assert_eq!(degrees, vec![1, 10]);
    assert!(
        factored
            .factors
            .iter()
            .any(|(f, _)| *f == Poly::from_int_coeffs(&[1; 11])),
        "Φ₁₁ must appear as a factor"
    );
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_degree_10_product_of_linears() {
    // ∏_{k=1}^{10} (x − k): ten distinct linear factors. The constant term is
    // 10! = 3628800 and the degree is far past the old Kronecker limit of 6.
    let linears: Vec<Poly> = (1..=10i64)
        .map(|k| Poly::from_int_coeffs(&[-k, 1]))
        .collect();
    let p = product_of(&linears);
    assert_eq!(p.degree(), Some(10));

    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        10,
        "expected 10 linear factors, got {:?}",
        factored.factors
    );
    for (f, mult) in &factored.factors {
        assert_eq!(f.degree(), Some(1));
        assert_eq!(*mult, 1);
    }
    for k in 1..=10i64 {
        assert!(
            factored
                .factors
                .iter()
                .any(|(f, _)| *f == Poly::from_int_coeffs(&[-k, 1])),
            "missing linear factor (x − {k})"
        );
    }
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_repeated_and_mixed_multiplicities() {
    // (x − 1)³ (x² + 1)² (x + 2): Yun supplies the multiplicities, Zassenhaus
    // the irreducible factors of each square-free part.
    let p = product_of(&[
        Poly::from_int_coeffs(&[-1, 1]).pow(3).unwrap(),
        Poly::from_int_coeffs(&[1, 0, 1]).pow(2).unwrap(),
        Poly::from_int_coeffs(&[2, 1]),
    ]);

    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        3,
        "expected 3 distinct factors, got {:?}",
        factored.factors
    );
    let multiplicity = |target: &Poly| {
        factored
            .factors
            .iter()
            .find(|(f, _)| f == target)
            .map(|(_, m)| *m)
    };
    assert_eq!(multiplicity(&Poly::from_int_coeffs(&[-1, 1])), Some(3));
    assert_eq!(multiplicity(&Poly::from_int_coeffs(&[1, 0, 1])), Some(2));
    assert_eq!(multiplicity(&Poly::from_int_coeffs(&[2, 1])), Some(1));
    assert_eq!(remultiply(&factored), p.normalized());
}

#[test]
fn test_factor_non_monic_with_content() {
    // 6 · (2x + 1)(3x − 2)(x² + 1) = 36x⁴ − 6x³ + 24x² − 6x − 12.
    let p = product_of(&[
        Poly::from_int_coeffs(&[6]),
        Poly::from_int_coeffs(&[1, 2]),
        Poly::from_int_coeffs(&[-2, 3]),
        Poly::from_int_coeffs(&[1, 0, 1]),
    ]);

    let factored = p.factor().unwrap();
    assert_eq!(factored.content, coeff_from_i64(6));
    assert_eq!(
        factored.factors.len(),
        3,
        "expected 3 factors, got {:?}",
        factored.factors
    );
    for expected in [
        Poly::from_int_coeffs(&[1, 2]),
        Poly::from_int_coeffs(&[-2, 3]),
        Poly::from_int_coeffs(&[1, 0, 1]),
    ] {
        assert!(
            factored.factors.iter().any(|(f, _)| *f == expected),
            "missing factor {expected:?} in {:?}",
            factored.factors
        );
    }
    assert_eq!(remultiply(&factored), p.normalized());
}

#[test]
fn test_factor_negative_leading_coefficient() {
    // −x² + 1 = (−1) · (x − 1)(x + 1): the content carries the sign.
    let p = Poly::from_int_coeffs(&[1, 0, -1]);
    let factored = p.factor().unwrap();
    assert_eq!(factored.content, coeff_from_i64(-1));
    assert_eq!(factored.factors.len(), 2);
    assert_eq!(remultiply(&factored), p.normalized());
}

#[test]
fn test_factor_rational_coefficients() {
    // ½x² − ½ = ½ · (x − 1)(x + 1).
    let p = poly_from_coeffs(&[(-1, 2), (0, 1), (1, 2)]);
    let factored = p.factor().unwrap();
    assert_eq!(factored.content, ratio(1, 2));
    assert_eq!(factored.factors.len(), 2);
    assert_eq!(remultiply(&factored), p.normalized());
}

#[test]
fn test_factor_eisenstein_high_degree_is_irreducible() {
    // x¹⁵ − 2 is Eisenstein at 2, hence irreducible over ℚ for every degree.
    let mut coeffs = vec![0i64; 16];
    coeffs[0] = -2;
    coeffs[15] = 1;
    let p = Poly::from_int_coeffs(&coeffs);

    let factored = p.factor().unwrap();
    assert_eq!(
        factored.factors.len(),
        1,
        "x¹⁵ − 2 must be irreducible, got {:?}",
        factored.factors
    );
    assert_eq!(remultiply(&factored), p);
}

#[test]
fn test_factor_large_coefficients_need_a_big_lift_modulus() {
    // (x² − 10⁶ x + 1)(x³ + 10⁶) — the Mignotte bound here is well past 2^64, so
    // the Hensel lift genuinely needs the arbitrary-precision modulus layer.
    let big = 1_000_000i64;
    let a = Poly::from_int_coeffs(&[1, -big, 1]);
    let b = Poly::from_int_coeffs(&[big, 0, 0, 1]);
    let p = product_of(&[a.clone(), b.clone()]);

    let factored = p.factor().unwrap();
    assert_eq!(remultiply(&factored), p.normalized());
    let total_degree: usize = factored
        .factors
        .iter()
        .map(|(f, m)| f.degree().unwrap_or(0) * m)
        .sum();
    assert_eq!(total_degree, 5);
    for (factor, _) in &factored.factors {
        assert_eq!(
            factor.factor().unwrap().factors.len(),
            1,
            "factor {factor:?} must itself be irreducible"
        );
    }
}

#[test]
fn test_factorization_remultiplies_to_original_property() {
    // Property: content · Π factorᵢ^multᵢ == the original, exactly, and every
    // returned factor is itself irreducible (re-factoring it yields one factor).
    let mut state: u64 = 0x5DEE_CE66_D1CE_B00D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let atoms: [Poly; 10] = [
        Poly::from_int_coeffs(&[-1, 1]),           // x − 1
        Poly::from_int_coeffs(&[2, 1]),            // x + 2
        Poly::from_int_coeffs(&[-3, 2]),           // 2x − 3
        Poly::from_int_coeffs(&[1, 0, 1]),         // x² + 1
        Poly::from_int_coeffs(&[1, 1, 1]),         // x² + x + 1
        Poly::from_int_coeffs(&[-2, 0, 0, 1]),     // x³ − 2
        Poly::from_int_coeffs(&[1, 0, 0, 0, 1]),   // x⁴ + 1
        Poly::from_int_coeffs(&[1, 0, -10, 0, 1]), // SD(2,3)
        Poly::from_int_coeffs(&[-5, 3]),           // 3x − 5
        Poly::from_int_coeffs(&[3, 0, 1]),         // x² + 3
    ];

    for _ in 0..40 {
        let count = 2 + (next() % 3) as usize;
        let chosen: Vec<Poly> = (0..count)
            .map(|_| atoms[(next() as usize) % atoms.len()].clone())
            .collect();
        let p = product_of(&chosen);

        let factored = p.factor().unwrap();
        assert_eq!(
            remultiply(&factored),
            p.normalized(),
            "re-multiplication must reproduce {p:?}"
        );

        let total_degree: usize = factored
            .factors
            .iter()
            .map(|(f, m)| f.degree().unwrap_or(0) * m)
            .sum();
        assert_eq!(
            total_degree,
            p.degree().unwrap_or(0),
            "factor degrees must add up for {p:?}"
        );

        for (factor, _) in &factored.factors {
            let refactored = factor.factor().unwrap();
            assert_eq!(
                refactored.factors.len(),
                1,
                "returned factor {factor:?} is not irreducible"
            );
            assert_eq!(refactored.factors[0].1, 1);
        }
    }
}

#[test]
fn test_factor_is_deterministic() {
    // Cantor–Zassenhaus is randomized, but the PRNG is seeded from the input, so
    // repeated calls must agree exactly.
    let p = product_of(&[
        Poly::from_int_coeffs(&[1, 0, 1]),
        Poly::from_int_coeffs(&[1, 1, 1]),
        Poly::from_int_coeffs(&[1, -1, 1]),
        Poly::from_int_coeffs(&[-2, 0, 0, 1]),
    ]);
    let first = p.factor().unwrap();
    for _ in 0..5 {
        assert_eq!(p.factor().unwrap(), first);
    }
    assert_eq!(first.factors.len(), 4);
}

// ── Modular arithmetic layer (GF(p) and ℤ/p^k) ────────────────────────────────

#[test]
fn test_mod_poly_div_rem_and_gcd() {
    // Over GF(7): x² − 1 = (x − 1)(x + 1).
    let a = ModPoly::new(vec![-1, 0, 1], 7);
    let b = ModPoly::new(vec![-1, 1], 7);

    let (q, r) = a.div_rem(&b).unwrap();
    assert!(r.is_zero());
    assert_eq!(q.coeffs(), &[1, 1], "quotient must be x + 1");

    let g = ModPoly::gcd(&a, &b).unwrap();
    assert_eq!(g.coeffs(), &[6, 1], "gcd must be the monic x − 1 ≡ x + 6");
}

#[test]
fn test_mod_poly_xgcd_bezout_identity() {
    // x² + 1 and x² + x + 1 are coprime over GF(13); s·a + t·b must equal 1.
    let a = ModPoly::new(vec![1, 0, 1], 13);
    let b = ModPoly::new(vec![1, 1, 1], 13);

    let (g, s, t) = ModPoly::xgcd(&a, &b).unwrap();
    assert_eq!(g.degree(), Some(0), "the two quadratics must be coprime");
    assert_eq!(g, ModPoly::one(13));
    assert_eq!(s.mul(&a).add(&t.mul(&b)), g, "Bézout identity must hold");
}

#[test]
fn test_mod_poly_pow_mod_frobenius_fixes_x() {
    // x² + 2 is irreducible over GF(5) (3 is not a square mod 5), so in
    // GF(5)[x]/(x² + 2) the Frobenius x ↦ x^(5²) is the identity.
    let prime = 5i128;
    let f = ModPoly::new(vec![2, 0, 1], prime);
    assert!(f.is_square_free().unwrap());

    let x = ModPoly::identity(prime);
    let frobenius = x.pow_mod(&BigUint::from(25u32), &f).unwrap();
    assert_eq!(frobenius, x, "x^(p²) ≡ x modulo an irreducible quadratic");
}

#[test]
fn test_mod_poly_detects_repeated_factors() {
    // (x − 1)² = x² − 2x + 1 is not square-free over GF(11).
    let square = ModPoly::new(vec![1, -2, 1], 11);
    assert!(!square.is_square_free().unwrap());
    // (x − 1)(x + 1) is.
    let distinct = ModPoly::new(vec![-1, 0, 1], 11);
    assert!(distinct.is_square_free().unwrap());
}

#[test]
fn test_mp_poly_symmetric_representatives_and_monic_division() {
    // Modulus 3³ = 27; the residue 26 stands for the integer −1.
    let modulus = BigInt::from(27i32);
    let f = MpPoly::new(vec![BigInt::from(26i32), BigInt::from(1i32)], &modulus);
    assert!(f.is_monic());
    assert_eq!(
        f.symmetric_coeffs(),
        vec![BigInt::from(-1i32), BigInt::from(1i32)],
        "26 ≡ −1 (mod 27)"
    );

    // (x − 1)(x + 1) divided by (x − 1) leaves (x + 1) with no remainder.
    let g = MpPoly::new(vec![BigInt::from(1i32), BigInt::from(1i32)], &modulus);
    let (q, r) = f.mul(&g).div_rem_monic(&f).unwrap();
    assert!(r.is_zero());
    assert_eq!(q, g);
}
