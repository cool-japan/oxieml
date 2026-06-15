//! Integration tests for constant folding in `LoweredOp::simplify`.
//! Verifies that all-constant subtrees are folded at compile time,
//! including propagation of NaN and Inf for domain violations.

use oxieml::lower::LoweredOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// 1. fold_add_consts
// ---------------------------------------------------------------------------
#[test]
fn fold_add_consts() {
    let op = LoweredOp::Add(
        Arc::new(LoweredOp::Const(2.0)),
        Arc::new(LoweredOp::Const(3.0)),
    );
    assert_eq!(op.simplify(), LoweredOp::Const(5.0));
}

// ---------------------------------------------------------------------------
// 2. fold_mul_consts
// ---------------------------------------------------------------------------
#[test]
fn fold_mul_consts() {
    let op = LoweredOp::Mul(
        Arc::new(LoweredOp::Const(3.0)),
        Arc::new(LoweredOp::Const(4.0)),
    );
    assert_eq!(op.simplify(), LoweredOp::Const(12.0));
}

// ---------------------------------------------------------------------------
// 3. fold_ln_neg_gives_nan
// ---------------------------------------------------------------------------
#[test]
fn fold_ln_neg_gives_nan() {
    let op = LoweredOp::Ln(Arc::new(LoweredOp::Const(-1.0)));
    let result = op.simplify();
    match result {
        LoweredOp::Const(v) => assert!(v.is_nan(), "ln(-1) should fold to NaN, got Const({v})"),
        other => panic!("expected Const(NaN), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. fold_div_by_zero_gives_inf
// ---------------------------------------------------------------------------
#[test]
fn fold_div_by_zero_gives_inf() {
    let op = LoweredOp::Div(
        Arc::new(LoweredOp::Const(1.0)),
        Arc::new(LoweredOp::Const(0.0)),
    );
    let result = op.simplify();
    match result {
        LoweredOp::Const(v) => assert!(v.is_infinite(), "1/0 should fold to Inf, got Const({v})"),
        other => panic!("expected Const(Inf), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. fold_nested_constant
// ---------------------------------------------------------------------------
#[test]
fn fold_nested_constant() {
    // (2 * 3) + 4 = 10
    let op = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Const(2.0)),
            Arc::new(LoweredOp::Const(3.0)),
        )),
        Arc::new(LoweredOp::Const(4.0)),
    );
    assert_eq!(op.simplify(), LoweredOp::Const(10.0));
}

// ---------------------------------------------------------------------------
// 6. fold_idempotent
// ---------------------------------------------------------------------------
#[test]
fn fold_idempotent() {
    let op = LoweredOp::Add(
        Arc::new(LoweredOp::Const(2.0)),
        Arc::new(LoweredOp::Const(3.0)),
    );
    let s1 = op.simplify();
    let s2 = s1.clone().simplify();
    assert_eq!(s1, s2, "double-simplify should be idempotent");
}

// ---------------------------------------------------------------------------
// 7. fold_sin_const
// ---------------------------------------------------------------------------
#[test]
fn fold_sin_const() {
    let op = LoweredOp::Sin(Arc::new(LoweredOp::Const(0.0)));
    let result = op.simplify();
    match result {
        LoweredOp::Const(v) => assert!(v.abs() < 1e-15, "sin(0) should fold to ~0.0, got {v}"),
        other => panic!("expected Const(0.0), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 8. fold_exp_const
// ---------------------------------------------------------------------------
#[test]
fn fold_exp_const() {
    let op = LoweredOp::Exp(Arc::new(LoweredOp::Const(0.0)));
    assert_eq!(op.simplify(), LoweredOp::Const(1.0));
}

// ---------------------------------------------------------------------------
// 9. partial_fold_leaves_vars
// ---------------------------------------------------------------------------
#[test]
fn partial_fold_leaves_vars() {
    let op = LoweredOp::Add(Arc::new(LoweredOp::Const(2.0)), Arc::new(LoweredOp::Var(0)));
    let simplified = op.simplify();
    // Var(0) prevents folding; the Add node should remain
    assert!(
        matches!(simplified, LoweredOp::Add(_, _)),
        "Add(Const, Var) should not fold, got {simplified:?}"
    );
}
