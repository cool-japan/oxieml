//! Integration tests for symbolic differentiation on [`LoweredOp`].
//!
//! Verifies [`LoweredOp::grad`] against numerical central-difference
//! derivatives on representative elementary functions, exercises each
//! calculus rule (sum, product, quotient, chain), and locks in a small
//! pretty-printed identity so simplification stays deterministic.

use oxieml::{Canonical, EmlTree, EvalCtx, LoweredOp};
use std::sync::Arc;

const H: f64 = 1e-5;

/// Numerical derivative via central difference, routed through the same
/// lowered IR that `grad` operates on. Used as ground truth.
fn numerical_grad(tree: &EmlTree, vars: &[f64], wrt: usize) -> f64 {
    let mut plus = vars.to_vec();
    let mut minus = vars.to_vec();
    plus[wrt] += H;
    minus[wrt] -= H;
    let ctx_p = EvalCtx::new(&plus);
    let ctx_m = EvalCtx::new(&minus);
    let yp = tree
        .eval_real_lowered(&ctx_p)
        .expect("tree.eval_real_lowered(+) must succeed for finite inputs");
    let ym = tree
        .eval_real_lowered(&ctx_m)
        .expect("tree.eval_real_lowered(-) must succeed for finite inputs");
    (yp - ym) / (2.0 * H)
}

/// Evaluate a `LoweredOp` directly using its recursive f64 evaluator.
fn eval_lowered(op: &LoweredOp, vars: &[f64]) -> f64 {
    op.eval(vars)
}

#[test]
fn grad_exp_matches_numerical() {
    // f(x) = exp(x), f'(x) = exp(x)
    let x = EmlTree::var(0);
    let tree = Canonical::exp(&x);
    let lowered = tree.lower().simplify();
    let d = lowered.grad(0);
    for &v in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
        let sym = eval_lowered(&d, &[v]);
        let num = numerical_grad(&tree, &[v], 0);
        assert!((sym - num).abs() < 1e-7, "exp: v={v}, sym={sym}, num={num}");
    }
}

#[test]
fn grad_sin_is_cos() {
    // f(x) = sin(x), f'(x) = cos(x)
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let lowered = tree.lower().simplify();
    let d = lowered.grad(0);
    for &v in &[-1.5, 0.0, 0.7, 1.2] {
        let sym = eval_lowered(&d, &[v]);
        assert!((sym - v.cos()).abs() < 1e-10);
    }
}

#[test]
fn grad_product_rule() {
    // f(x,y) = x*y, df/dx = y, df/dy = x
    let op = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let dx = op.grad(0);
    let dy = op.grad(1);
    assert!((eval_lowered(&dx, &[3.0, 5.0]) - 5.0).abs() < 1e-12);
    assert!((eval_lowered(&dy, &[3.0, 5.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn grad_polynomial() {
    // f(x) = x^3 + 2x, f'(x) = 3x^2 + 2 ; f'(2) = 14
    let x = Arc::new(LoweredOp::Var(0));
    let x_cubed = LoweredOp::Mul(Arc::new(LoweredOp::Mul(x.clone(), x.clone())), x.clone());
    let two_x = LoweredOp::Mul(Arc::new(LoweredOp::Const(2.0)), x.clone());
    let f = LoweredOp::Add(Arc::new(x_cubed), Arc::new(two_x));
    let df = f.grad(0);
    assert!((eval_lowered(&df, &[2.0]) - 14.0).abs() < 1e-10);
}

#[test]
fn grad_quotient_rule() {
    // f(x) = 1/x, f'(x) = -1/x^2
    let op = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(LoweredOp::Var(0)));
    let d = op.grad(0);
    for &v in &[0.5, 1.0, 2.0, 3.0] {
        let sym = eval_lowered(&d, &[v]);
        let expected = -1.0 / (v * v);
        assert!(
            (sym - expected).abs() < 1e-10,
            "v={v}: sym={sym}, expected={expected}"
        );
    }
}

#[test]
fn grad_wrt_other_var_is_zero() {
    // The expression involves only x0; differentiating wrt x1 must give 0.
    let op = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    let d = op.grad(1);
    assert_eq!(eval_lowered(&d, &[1.0, 99.0]), 0.0);
}

#[test]
fn grad_const_is_zero() {
    let op = LoweredOp::Const(42.0);
    let d = op.grad(0);
    assert_eq!(eval_lowered(&d, &[100.0]), 0.0);
}

#[test]
fn grad_composition_chain_rule() {
    // f(x) = exp(sin(x)), f'(x) = exp(sin(x)) * cos(x)
    let x = EmlTree::var(0);
    let sin_x = Canonical::sin(&x);
    let exp_sin = Canonical::exp(&sin_x);
    let lowered = exp_sin.lower().simplify();
    let d = lowered.grad(0);
    for &v in &[-1.0, 0.0, 0.5, 1.2] {
        let sym = eval_lowered(&d, &[v]);
        let expected = v.sin().exp() * v.cos();
        assert!(
            (sym - expected).abs() < 1e-8,
            "v={v}: sym={sym}, expected={expected}"
        );
    }
}

#[test]
fn grad_of_grad_sin_is_minus_sin() {
    // f(x) = sin(x); f''(x) = -sin(x). Compute grad twice and compare.
    let op = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let d1 = op.grad(0);
    let d2 = d1.grad(0);
    for &v in &[-1.3, -0.4, 0.0, 0.6, 1.8] {
        let sym = eval_lowered(&d2, &[v]);
        let expected = -v.sin();
        assert!(
            (sym - expected).abs() < 1e-12,
            "v={v}: sym={sym}, expected={expected}"
        );
    }
}

#[test]
fn grad_is_idempotent_via_pretty() {
    // f(x) = x * x; symbolic derivative through the product rule yields
    // x'·x + x·x' = 1·x + x·1. After polynomial canonicalization this
    // collapses to the canonical form (2 * x0).
    // Lock this exact surface form so future simplification changes are
    // caught explicitly.
    let x = Arc::new(LoweredOp::Var(0));
    let op = LoweredOp::Mul(x.clone(), x.clone());
    let d = op.grad(0);
    assert_eq!(d.to_pretty(), "(2 * x0)");
    // Numerical sanity check: f'(3) = 2·3 = 6.
    assert!((eval_lowered(&d, &[3.0]) - 6.0).abs() < 1e-12);
}

#[test]
fn grad_neg_flips_sign() {
    // f(x) = -x^2, f'(x) = -2x ; at x=4 should be -8.
    let x = Arc::new(LoweredOp::Var(0));
    let x_sq = LoweredOp::Mul(x.clone(), x);
    let op = LoweredOp::Neg(Arc::new(x_sq));
    let d = op.grad(0);
    assert!((eval_lowered(&d, &[4.0]) - (-8.0)).abs() < 1e-12);
    assert!((eval_lowered(&d, &[-1.5]) - 3.0).abs() < 1e-12);
}

#[test]
fn grad_ln_reciprocal() {
    // f(x) = ln(x), f'(x) = 1/x
    let op = LoweredOp::Ln(Arc::new(LoweredOp::Var(0)));
    let d = op.grad(0);
    for &v in &[0.5, 1.0, 2.0, 7.5] {
        let sym = eval_lowered(&d, &[v]);
        let expected = 1.0 / v;
        assert!(
            (sym - expected).abs() < 1e-12,
            "v={v}: sym={sym}, expected={expected}"
        );
    }
}

#[test]
fn grad_pow_general_rule() {
    // f(x, y) = x^y, df/dx = y·x^(y-1), df/dy = x^y · ln(x).
    let op = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let dx = op.grad(0);
    let dy = op.grad(1);
    let cases = [(2.0_f64, 3.0_f64), (1.5, 2.0), (4.0, 0.5)];
    for (xv, yv) in cases {
        let sym_x = eval_lowered(&dx, &[xv, yv]);
        let sym_y = eval_lowered(&dy, &[xv, yv]);
        let expected_x = yv * xv.powf(yv - 1.0);
        let expected_y = xv.powf(yv) * xv.ln();
        assert!(
            (sym_x - expected_x).abs() < 1e-9,
            "df/dx at ({xv},{yv}): sym={sym_x}, expected={expected_x}"
        );
        assert!(
            (sym_y - expected_y).abs() < 1e-9,
            "df/dy at ({xv},{yv}): sym={sym_y}, expected={expected_y}"
        );
    }
}
