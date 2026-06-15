//! Tests for grad_all, jacobian, hessian, and count_vars utilities.

use oxieml::LoweredOp;
use std::sync::Arc;

/// Build `2·x0 + 3·x1`.
fn linear_expr() -> LoweredOp {
    LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Const(2.0)),
            Arc::new(LoweredOp::Var(0)),
        )),
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Const(3.0)),
            Arc::new(LoweredOp::Var(1)),
        )),
    )
}

/// Evaluate a `LoweredOp` at a fixed point via the standard `eval` method.
fn eval_at(op: &LoweredOp, vars: &[f64]) -> f64 {
    op.eval(vars)
}

#[test]
fn grad_all_linear() {
    let f = linear_expr();
    let grads = f.grad_all();
    assert_eq!(grads.len(), 2, "linear in two variables");

    // Both at any point: ∂f/∂x0 = 2, ∂f/∂x1 = 3
    let point = [5.0, 7.0];
    let d0 = eval_at(&grads[0], &point);
    let d1 = eval_at(&grads[1], &point);
    assert!((d0 - 2.0).abs() < 1e-12, "∂f/∂x0 = 2 but got {d0}");
    assert!((d1 - 3.0).abs() < 1e-12, "∂f/∂x1 = 3 but got {d1}");
}

#[test]
fn jacobian_pads_with_zeros() {
    // f(x0) = x0; jacobian(3) should be [1, 0, 0]
    let f = LoweredOp::Var(0);
    let jac = f.jacobian(3);
    assert_eq!(jac.len(), 3);

    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let hash_of = |op: &LoweredOp| {
        let mut h = DefaultHasher::new();
        op.structural_hash(&mut h);
        h.finish()
    };
    let zero_hash = hash_of(&LoweredOp::Const(0.0));

    // jac[1] and jac[2] must both be Const(0.0)
    assert_eq!(hash_of(&jac[1]), zero_hash, "jac[1] should be Const(0.0)");
    assert_eq!(hash_of(&jac[2]), zero_hash, "jac[2] should be Const(0.0)");

    // jac[0] is the derivative wrt x0, which is 1
    let d0 = eval_at(&jac[0], &[42.0, 0.0, 0.0]);
    assert!((d0 - 1.0).abs() < 1e-12, "∂x0/∂x0 = 1 but got {d0}");
}

#[test]
fn jacobian_truncates() {
    // f(x0, x1, x2) = x0 + x1 + x2; jacobian(2) has length 2
    let f = LoweredOp::Add(
        Arc::new(LoweredOp::Add(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Var(1)),
        )),
        Arc::new(LoweredOp::Var(2)),
    );
    assert_eq!(f.count_vars(), 3);
    let jac = f.jacobian(2);
    assert_eq!(jac.len(), 2, "truncated to 2 columns");
}

#[test]
fn hessian_symmetric() {
    // f(x0, x1) = x0^2 + x0·x1 + x1^2
    let f = LoweredOp::Add(
        Arc::new(LoweredOp::Add(
            Arc::new(LoweredOp::Pow(
                Arc::new(LoweredOp::Var(0)),
                Arc::new(LoweredOp::Const(2.0)),
            )),
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Var(0)),
                Arc::new(LoweredOp::Var(1)),
            )),
        )),
        Arc::new(LoweredOp::Pow(
            Arc::new(LoweredOp::Var(1)),
            Arc::new(LoweredOp::Const(2.0)),
        )),
    );
    let h = f.hessian(2);
    assert_eq!(h.len(), 2);
    assert_eq!(h[0].len(), 2);

    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let hash_of = |op: &LoweredOp| {
        let mut hsh = DefaultHasher::new();
        op.structural_hash(&mut hsh);
        hsh.finish()
    };
    // Schwarz: H[i][j] == H[j][i] structurally
    assert_eq!(
        hash_of(&h[0][1]),
        hash_of(&h[1][0]),
        "H[0][1] and H[1][0] must be structurally equal"
    );
}

#[test]
fn hessian_quadratic_correct() {
    // f(x0, x1) = x0^2 + x1^2; Hessian = [[2, 0], [0, 2]]
    let f = LoweredOp::Add(
        Arc::new(LoweredOp::Pow(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Const(2.0)),
        )),
        Arc::new(LoweredOp::Pow(
            Arc::new(LoweredOp::Var(1)),
            Arc::new(LoweredOp::Const(2.0)),
        )),
    );
    let h = f.hessian(2);
    let point = [1.0, 2.0];
    let h00 = eval_at(&h[0][0], &point);
    let h11 = eval_at(&h[1][1], &point);
    let h01 = eval_at(&h[0][1], &point);
    let h10 = eval_at(&h[1][0], &point);
    assert!((h00 - 2.0).abs() < 1e-10, "H[0][0] = 2 but got {h00}");
    assert!((h11 - 2.0).abs() < 1e-10, "H[1][1] = 2 but got {h11}");
    assert!(h01.abs() < 1e-10, "H[0][1] = 0 but got {h01}");
    assert!(h10.abs() < 1e-10, "H[1][0] = 0 but got {h10}");
}

#[test]
fn count_vars_correct() {
    // Var(3) in a tree → count_vars = 4
    let f = LoweredOp::Add(
        Arc::new(LoweredOp::Var(0)),
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Var(3)),
            Arc::new(LoweredOp::Const(2.0)),
        )),
    );
    assert_eq!(f.count_vars(), 4, "max var index is 3 → count = 4");

    // Const tree: no vars
    let c = LoweredOp::Const(42.0);
    assert_eq!(c.count_vars(), 0, "constant has no vars");

    // Single Var(2)
    let v = LoweredOp::Var(2);
    assert_eq!(v.count_vars(), 3);
}

#[test]
fn grad_all_vs_individual() {
    // f(x0, x1) = sin(x0) * x1 + exp(x1)
    let f = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Sin(Arc::new(LoweredOp::Var(0)))),
            Arc::new(LoweredOp::Var(1)),
        )),
        Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Var(1)))),
    );

    let grads_all = f.grad_all();
    let n = f.count_vars();
    assert_eq!(grads_all.len(), n);

    // For a few fixed points, verify grad_all()[i].eval == grad(i).eval
    let test_points = [[0.5_f64, 1.0_f64], [1.2_f64, 0.3_f64], [-0.8_f64, 2.1_f64]];
    for pt in &test_points {
        for (i, grad_i) in grads_all.iter().enumerate().take(n) {
            let via_all = eval_at(grad_i, pt.as_slice());
            let individual = f.grad(i).simplify();
            let via_individual = eval_at(&individual, pt.as_slice());
            assert!(
                (via_all - via_individual).abs() < 1e-10,
                "grad_all()[{i}] differs from grad({i}) at {pt:?}: {via_all} vs {via_individual}"
            );
        }
    }
}
