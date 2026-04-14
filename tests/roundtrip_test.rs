//! Roundtrip tests: EML tree -> lower -> evaluate, verify numerical consistency.

use approx::assert_relative_eq;
use oxieml::{Canonical, EmlTree, EvalCtx};

/// Helper: check that EML evaluation and lowered evaluation agree.
fn check_roundtrip(tree: &EmlTree, vars: &[f64], expected: f64, epsilon: f64) {
    let ctx = EvalCtx::new(vars);

    // EML evaluation
    let eml_result = tree.eval_real(&ctx).unwrap();
    assert_relative_eq!(eml_result, expected, epsilon = epsilon);

    // Lowered evaluation
    let lowered = tree.lower();
    let lowered_result = lowered.eval(vars);
    assert_relative_eq!(lowered_result, expected, epsilon = epsilon);

    // EML and lowered should agree
    assert_relative_eq!(eml_result, lowered_result, epsilon = epsilon);
}

#[test]
fn roundtrip_exp() {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    for &val in &[0.0, 1.0, 2.0, -1.0] {
        check_roundtrip(&exp_x, &[val], val.exp(), 1e-10);
    }
}

#[test]
fn roundtrip_euler() {
    let e = Canonical::euler();
    check_roundtrip(&e, &[], std::f64::consts::E, 1e-10);
}

#[test]
fn roundtrip_ln() {
    let x = EmlTree::var(0);
    let ln_x = Canonical::ln(&x);
    for &val in &[1.0, std::f64::consts::E, 2.0, 10.0] {
        check_roundtrip(&ln_x, &[val], val.ln(), 1e-10);
    }
}

#[test]
fn roundtrip_eml_basic() {
    let one = EmlTree::one();
    let e = EmlTree::eml(&one, &one);
    check_roundtrip(&e, &[], std::f64::consts::E, 1e-10);
}

#[test]
fn roundtrip_exp_of_two() {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    check_roundtrip(&exp_x, &[2.0], 2.0_f64.exp(), 1e-10);
}

#[test]
fn roundtrip_e_minus_x() {
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let exp_x = EmlTree::eml(&x, &one);
    let e_minus_x = EmlTree::eml(&one, &exp_x);

    for &val in &[0.0, 1.0, 2.0] {
        let expected = std::f64::consts::E - val;
        check_roundtrip(&e_minus_x, &[val], expected, 1e-10);
    }
}

#[test]
fn roundtrip_reciprocal() {
    let x = EmlTree::var(0);
    let recip = Canonical::reciprocal(&x);
    for &val in &[2.0, 4.0, 0.5] {
        let ctx = EvalCtx::new(&[val]);
        let eml_result = recip.eval_real(&ctx).unwrap();
        assert_relative_eq!(eml_result, 1.0 / val, epsilon = 1e-6);
    }
}

#[test]
fn roundtrip_neg() {
    let x = EmlTree::var(0);
    let neg_x = Canonical::neg(&x);
    for &val in &[1.0, 2.5] {
        let ctx = EvalCtx::new(&[val]);
        let eml_result = neg_x.eval_real(&ctx).unwrap();
        assert_relative_eq!(eml_result, -val, epsilon = 1e-6);
    }
}

#[test]
fn roundtrip_sub() {
    let x = EmlTree::var(0);
    let y = EmlTree::var(1);
    let diff = Canonical::sub(&x, &y);
    let ctx = EvalCtx::new(&[5.0, 3.0]);

    let eml_result = diff.eval_real(&ctx).unwrap();
    let lowered = diff.lower();
    let lowered_result = lowered.eval(&[5.0, 3.0]);

    assert_relative_eq!(eml_result, 2.0, epsilon = 1e-6);
    assert_relative_eq!(lowered_result, 2.0, epsilon = 1e-6);
}

#[test]
fn roundtrip_sqrt() {
    let x = EmlTree::var(0);
    let sqrt_x = Canonical::sqrt(&x);
    let ctx = EvalCtx::new(&[4.0]);
    let result = sqrt_x.eval_real(&ctx).unwrap();
    assert_relative_eq!(result, 2.0, epsilon = 1e-2);
}

#[test]
fn roundtrip_lowered_simplify() {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let lowered = exp_x.lower();
    let simplified = lowered.simplify();

    for &val in &[0.0, 1.0, 2.0] {
        let original = lowered.eval(&[val]);
        let after = simplified.eval(&[val]);
        assert_relative_eq!(original, after, epsilon = 1e-15);
    }
}

#[test]
fn roundtrip_display_consistency() {
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let exp_x = EmlTree::eml(&x, &one);
    let display = format!("{exp_x}");
    assert_eq!(display, "eml(x0, 1)");
}

#[test]
fn roundtrip_compile_eval() {
    // Verify compiled code description matches lowered evaluation
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let code = oxieml::compile::compile_to_rust(&exp_x, "test_fn");
    assert!(code.contains("fn test_fn"));
    assert!(code.contains(".exp()"));
    // The lowered form should give exp(x0)
    let lowered = exp_x.lower();
    assert_relative_eq!(lowered.eval(&[1.0]), std::f64::consts::E, epsilon = 1e-10);
}

#[test]
fn roundtrip_simplify_preserves_eval() {
    use oxieml::simplify::simplify;

    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let ln_exp_x = Canonical::ln(&exp_x);

    for &val in &[0.5, 1.0, 2.0] {
        let ctx = EvalCtx::new(&[val]);
        let before = ln_exp_x.eval_real(&ctx).unwrap();
        let simplified = simplify(&ln_exp_x);
        let after = simplified.eval_real(&ctx).unwrap();
        assert_relative_eq!(before, after, epsilon = 1e-10);
    }
}

#[test]
fn roundtrip_zero() {
    let z = Canonical::zero();
    let ctx = EvalCtx::new(&[]);
    let result = z.eval_real(&ctx).unwrap();
    assert!(result.abs() < 1e-10);
}

#[test]
fn roundtrip_nat_values() {
    for n in 1..=3u64 {
        let tree = Canonical::nat(n);
        let ctx = EvalCtx::new(&[]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!(
            (result - n as f64).abs() < 0.1,
            "nat({n}) = {result}, expected {n}"
        );
    }
}
