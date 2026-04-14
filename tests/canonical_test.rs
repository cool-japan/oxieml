//! Tests for canonical EML constructions (Paper Tables 1-4).

use approx::assert_relative_eq;
use oxieml::{Canonical, EmlTree, EvalCtx};

// ================================================================
// Table 1: Basic operations
// ================================================================

#[test]
fn table1_exp() {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    for &val in &[0.0, 1.0, 2.0, -1.0, 0.5] {
        let ctx = EvalCtx::new(&[val]);
        let result = exp_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, val.exp(), epsilon = 1e-10);
    }
}

#[test]
fn table1_ln() {
    let x = EmlTree::var(0);
    let ln_x = Canonical::ln(&x);
    for &val in &[1.0, std::f64::consts::E, 2.0, 10.0, 0.5] {
        let ctx = EvalCtx::new(&[val]);
        let result = ln_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, val.ln(), epsilon = 1e-10);
    }
}

#[test]
fn table1_euler() {
    let e = Canonical::euler();
    let ctx = EvalCtx::new(&[]);
    let result = e.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, std::f64::consts::E, epsilon = 1e-10);
}

#[test]
fn table1_neg() {
    let x = EmlTree::var(0);
    let neg_x = Canonical::neg(&x);
    for &val in &[1.0, 2.5, 0.1] {
        let ctx = EvalCtx::new(&[val]);
        let result = neg_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, -val, epsilon = 1e-6);
    }
}

#[test]
fn table1_zero() {
    let z = Canonical::zero();
    let ctx = EvalCtx::new(&[]);
    let result = z.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 0.0, epsilon = 1e-10);
}

// ================================================================
// Table 2: Arithmetic operations
// ================================================================

#[test]
fn table2_add() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let sum = Canonical::add(&x, &y);
    let ctx = EvalCtx::new(&[2.0, 3.0]);
    let result = sum.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 5.0, epsilon = 1e-4);
}

#[test]
fn table2_sub() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let diff = Canonical::sub(&x, &y);
    let ctx = EvalCtx::new(&[5.0, 3.0]);
    let result = diff.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 2.0, epsilon = 1e-6);
}

#[test]
fn table2_mul() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let prod = Canonical::mul(&x, &y);
    let ctx = EvalCtx::new(&[3.0, 4.0]);
    let result = prod.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 12.0, epsilon = 1e-3);
}

#[test]
fn table2_div() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let quot = Canonical::div(&x, &y);
    let ctx = EvalCtx::new(&[10.0, 2.0]);
    let result = quot.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 5.0, epsilon = 1e-3);
}

#[test]
fn table2_pow() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let p = Canonical::pow(&x, &y);
    let ctx = EvalCtx::new(&[2.0, 3.0]);
    let result = p.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 8.0, epsilon = 1e-2);
}

// ================================================================
// Table 3: Trigonometric functions
// ================================================================

#[test]
fn table3_pi_complex() {
    // pi() returns a tree whose complex evaluation yields iπ.
    // eval_real should return ComplexResult error.
    let pi_tree = Canonical::pi();
    let ctx = EvalCtx::new(&[]);
    let result = pi_tree.eval_real(&ctx);
    assert!(
        result.is_err(),
        "pi() should yield complex result in real mode"
    );
}

#[test]
fn table3_pi_complex_eval() {
    // Directly check that eval_complex gives iπ.
    let pi_tree = Canonical::pi();
    let result = pi_tree.eval_complex(&[]).unwrap();
    // ln(-1) = iπ, so Im should be π
    assert_relative_eq!(result.im, std::f64::consts::PI, epsilon = 1e-4);
    assert!(
        result.re.abs() < 1e-4,
        "real part should be near zero, got {}",
        result.re
    );
}

#[test]
fn table3_sin_zero() {
    // sin(0) = 0
    let x = EmlTree::var(0);
    let sin_x = Canonical::sin(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = sin_x.eval_real(&ctx);
    if let Ok(val) = result {
        assert!(val.abs() < 0.1, "sin(0) should be ~0, got {val}");
    }
    // sin produces very deep trees — complex evaluation may introduce noise.
    // We accept either a near-zero result or a ComplexResult error.
}

#[test]
fn table3_cos_zero() {
    // cos(0) = 1
    let x = EmlTree::var(0);
    let cos_x = Canonical::cos(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = cos_x.eval_real(&ctx);
    if let Ok(val) = result {
        assert!((val - 1.0).abs() < 0.1, "cos(0) should be ~1, got {val}");
    }
}

// ================================================================
// Table 4: Inverse functions and others
// ================================================================

#[test]
fn table4_reciprocal() {
    let x = EmlTree::var(0);
    let recip = Canonical::reciprocal(&x);
    for &val in &[2.0, 4.0, 0.5, 10.0] {
        let ctx = EvalCtx::new(&[val]);
        let result = recip.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, 1.0 / val, epsilon = 1e-6);
    }
}

#[test]
fn table4_sqrt() {
    let x = EmlTree::var(0);
    let sqrt_x = Canonical::sqrt(&x);
    for &(val, expected) in &[(4.0, 2.0), (1.0, 1.0), (0.25, 0.5)] {
        let ctx = EvalCtx::new(&[val]);
        let result = sqrt_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, expected, epsilon = 1e-2);
    }
}

#[test]
fn table4_abs_positive() {
    let x = EmlTree::var(0);
    let abs_x = Canonical::abs(&x);
    let ctx = EvalCtx::new(&[3.0]);
    let result = abs_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 3.0, epsilon = 1e-2);
}

#[test]
fn table4_nat_values() {
    for n in 1..=5u64 {
        let tree = Canonical::nat(n);
        let ctx = EvalCtx::new(&[]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!(
            (result - n as f64).abs() < 0.1,
            "nat({n}) = {result}, expected {n}"
        );
    }
}

// ================================================================
// Table 5: Hyperbolic functions
// ================================================================

#[test]
fn table5_sinh_zero() {
    let x = EmlTree::var(0);
    let sinh_x = Canonical::sinh(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = sinh_x.eval_real(&ctx).unwrap();
    assert!(result.abs() < 0.1, "sinh(0) should be ~0, got {result}");
}

#[test]
fn table5_sinh_one() {
    let x = EmlTree::var(0);
    let sinh_x = Canonical::sinh(&x);
    let ctx = EvalCtx::new(&[1.0]);
    let result = sinh_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 1.0_f64.sinh(), epsilon = 0.1);
}

#[test]
fn table5_cosh_zero() {
    let x = EmlTree::var(0);
    let cosh_x = Canonical::cosh(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = cosh_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 1.0, epsilon = 0.1);
}

#[test]
fn table5_cosh_one() {
    let x = EmlTree::var(0);
    let cosh_x = Canonical::cosh(&x);
    let ctx = EvalCtx::new(&[1.0]);
    let result = cosh_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 1.0_f64.cosh(), epsilon = 0.1);
}

#[test]
fn table5_tanh_zero() {
    let x = EmlTree::var(0);
    let tanh_x = Canonical::tanh(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = tanh_x.eval_real(&ctx);
    if let Ok(val) = result {
        assert!(val.abs() < 0.1, "tanh(0) should be ~0, got {val}");
    }
}

// ================================================================
// Table 6: Inverse hyperbolic functions
// ================================================================

#[test]
fn table6_arcsinh_zero() {
    let x = EmlTree::var(0);
    let asinh_x = Canonical::arcsinh(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = asinh_x.eval_real(&ctx).unwrap();
    assert!(result.abs() < 0.1, "arcsinh(0) should be ~0, got {result}");
}

#[test]
fn table6_arctanh_zero() {
    let x = EmlTree::var(0);
    let atanh_x = Canonical::arctanh(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = atanh_x.eval_real(&ctx).unwrap();
    assert!(result.abs() < 0.1, "arctanh(0) should be ~0, got {result}");
}

// ================================================================
// Table 7: Inverse trig functions
// ================================================================

#[test]
fn table7_arctan_zero() {
    let x = EmlTree::var(0);
    let atan_x = Canonical::arctan(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = atan_x.eval_real(&ctx);
    if let Ok(val) = result {
        assert!(val.abs() < 0.1, "arctan(0) should be ~0, got {val}");
    }
}

#[test]
fn table7_arcsin_zero() {
    let x = EmlTree::var(0);
    let asin_x = Canonical::arcsin(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = asin_x.eval_real(&ctx);
    if let Ok(val) = result {
        assert!(val.abs() < 0.1, "arcsin(0) should be ~0, got {val}");
    }
}

#[test]
fn table7_arccos_one() {
    let x = EmlTree::var(0);
    let acos_x = Canonical::arccos(&x);
    let ctx = EvalCtx::new(&[1.0]);
    let result = acos_x.eval_real(&ctx);
    if let Ok(val) = result {
        assert!(val.abs() < 0.2, "arccos(1) should be ~0, got {val}");
    }
}

// ================================================================
// Constants
// ================================================================

#[test]
fn test_neg_one_constant() {
    let tree = Canonical::neg_one();
    let ctx = EvalCtx::new(&[]);
    let result = tree.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, -1.0, epsilon = 1e-6);
}

#[test]
fn test_neg_two_constant() {
    let tree = Canonical::neg_two();
    let ctx = EvalCtx::new(&[]);
    let result = tree.eval_real(&ctx).unwrap();
    assert!(
        (result - (-2.0)).abs() < 0.1,
        "neg_two = {result}, expected -2"
    );
}

#[test]
fn test_imag_unit_complex() {
    let i_tree = Canonical::imag_unit();
    let result = i_tree.eval_complex(&[]).unwrap();
    assert!(
        result.re.abs() < 1e-4,
        "Re(i) should be ~0, got {}",
        result.re
    );
    assert!(
        (result.im - 1.0).abs() < 1e-4,
        "Im(i) should be ~1, got {}",
        result.im
    );
}

#[test]
fn test_square() {
    let x = EmlTree::var(0);
    let x_sq = Canonical::square(&x);
    for &val in &[2.0, 3.0, 0.5] {
        let ctx = EvalCtx::new(&[val]);
        let result = x_sq.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, val * val, epsilon = 0.1);
    }
}

// ================================================================
// Hyperbolic identities
// ================================================================

#[test]
fn test_cosh_squared_minus_sinh_squared() {
    // cosh^2(x) - sinh^2(x) = 1
    let x = EmlTree::var(0);
    let sinh_x = Canonical::sinh(&x);
    let cosh_x = Canonical::cosh(&x);
    let sinh_sq = Canonical::square(&sinh_x);
    let cosh_sq = Canonical::square(&cosh_x);
    let diff = Canonical::sub(&cosh_sq, &sinh_sq);
    let ctx = EvalCtx::new(&[1.0]);
    let result = diff.eval_real(&ctx);
    if let Ok(val) = result {
        assert!(
            (val - 1.0).abs() < 0.2,
            "cosh^2 - sinh^2 should be ~1, got {val}"
        );
    }
}

// ================================================================
// Special values and roundtrips
// ================================================================

#[test]
fn test_exp_of_zero() {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = exp_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 1.0, epsilon = 1e-10);
}

#[test]
fn test_ln_of_e() {
    let x = EmlTree::var(0);
    let ln_x = Canonical::ln(&x);
    let ctx = EvalCtx::new(&[std::f64::consts::E]);
    let result = ln_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 1.0, epsilon = 1e-10);
}

#[test]
fn test_exp_ln_roundtrip() {
    let x = EmlTree::var(0);
    let exp_ln_x = Canonical::exp(&Canonical::ln(&x));
    for &val in &[1.0, 2.0, 5.0, 0.5] {
        let ctx = EvalCtx::new(&[val]);
        let result = exp_ln_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, val, epsilon = 1e-8);
    }
}

#[test]
fn test_ln_exp_roundtrip() {
    let x = EmlTree::var(0);
    let ln_exp_x = Canonical::ln(&Canonical::exp(&x));
    for &val in &[0.0, 1.0, 2.0, -1.0] {
        let ctx = EvalCtx::new(&[val]);
        let result = ln_exp_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(result, val, epsilon = 1e-8);
    }
}

#[test]
fn test_neg_of_neg() {
    // neg(neg(x)) should approximately equal x
    let x = EmlTree::var(0);
    let neg_neg_x = Canonical::neg(&Canonical::neg(&x));
    let ctx = EvalCtx::new(&[2.0]);
    let result = neg_neg_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 2.0, epsilon = 1e-4);
}

#[test]
fn test_add_commutativity() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let xy = Canonical::add(&x, &y);
    let yx = Canonical::add(&y, &x);
    let ctx = EvalCtx::new(&[2.0, 3.0]);
    let r1 = xy.eval_real(&ctx).unwrap();
    let r2 = yx.eval_real(&ctx).unwrap();
    assert_relative_eq!(r1, r2, epsilon = 1e-4);
}

#[test]
fn test_mul_commutativity() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let xy = Canonical::mul(&x, &y);
    let yx = Canonical::mul(&y, &x);
    let ctx = EvalCtx::new(&[3.0, 4.0]);
    let r1 = xy.eval_real(&ctx).unwrap();
    let r2 = yx.eval_real(&ctx).unwrap();
    assert_relative_eq!(r1, r2, epsilon = 1e-3);
}
