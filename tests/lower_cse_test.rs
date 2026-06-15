//! Tests for `LoweredOp::cse()` — hash-consing pass.

use oxieml::{LoweredOp, NamedConst};
use std::sync::Arc;

/// Helper: build `Exp(Var(0))` as a fresh Arc (no sharing).
fn exp_var0() -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Var(0))))
}

// ---------------------------------------------------------------------------
// 1. Sharing is real
// ---------------------------------------------------------------------------

#[test]
fn cse_sharing_is_real() {
    // Build Add(Exp(Var(0)), Exp(Var(0))) with two DISTINCT Arc allocations.
    let tree = LoweredOp::Add(exp_var0(), exp_var0());
    let cse_root = tree.cse();
    if let LoweredOp::Add(a, b) = cse_root.as_ref() {
        assert!(
            Arc::ptr_eq(a, b),
            "After CSE, both children of Add should be the same Arc (ptr_eq)"
        );
    } else {
        panic!("CSE root should be Add");
    }
}

// ---------------------------------------------------------------------------
// 2. Behavioural parity
// ---------------------------------------------------------------------------

#[test]
fn cse_behavioural_parity_add() {
    let tree = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Var(1)),
        )),
        Arc::new(LoweredOp::Const(3.0)),
    );
    let vars = [2.0, 5.0];
    let expected = tree.eval(&vars);
    let cse_root = tree.cse();
    let got = cse_root.eval(&vars);
    assert!(
        (got - expected).abs() < 1e-15,
        "eval mismatch: expected {expected}, got {got}"
    );
}

#[test]
fn cse_behavioural_parity_complex() {
    // sin(x)^2 + cos(x)^2 ≈ 1
    let x = Arc::new(LoweredOp::Var(0));
    let sin_x = Arc::new(LoweredOp::Sin(Arc::clone(&x)));
    let cos_x = Arc::new(LoweredOp::Cos(Arc::clone(&x)));
    let tree = LoweredOp::Add(
        Arc::new(LoweredOp::Pow(
            Arc::clone(&sin_x),
            Arc::new(LoweredOp::Const(2.0)),
        )),
        Arc::new(LoweredOp::Pow(
            Arc::clone(&cos_x),
            Arc::new(LoweredOp::Const(2.0)),
        )),
    );
    for i in 0..10 {
        let x_val = i as f64 * 0.3;
        let vars = [x_val];
        let expected = tree.eval(&vars);
        let cse_root = tree.cse();
        let got = cse_root.eval(&vars);
        assert!(
            (got - expected).abs() < 1e-14,
            "parity at x={x_val}: expected {expected}, got {got}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Idempotence
// ---------------------------------------------------------------------------

#[test]
fn cse_idempotent() {
    let tree = LoweredOp::Add(exp_var0(), exp_var0());
    let once = tree.cse();
    let twice = once.cse();
    // Structural equality
    assert_eq!(
        once.as_ref(),
        twice.as_ref(),
        "cse().cse() should be structurally equal to cse()"
    );
    // And same eval
    let vars = [1.5];
    let val_once = once.eval(&vars);
    let val_twice = twice.eval(&vars);
    assert!(
        (val_once - val_twice).abs() < 1e-15,
        "idempotent eval mismatch: {val_once} vs {val_twice}"
    );
}

// ---------------------------------------------------------------------------
// 4. NamedConst(Pi) vs Const(PI) — different structures, should NOT be merged
// ---------------------------------------------------------------------------

#[test]
fn cse_named_const_not_merged_with_const() {
    // structural_hash for NamedConst hashes identically to Const(value()) (see display.rs).
    // However, the PartialEq collision guard in CseInterner compares enum variants:
    // LoweredOp::NamedConst(Pi) != LoweredOp::Const(PI) by derived PartialEq.
    // Therefore they must NOT be merged into the same Arc.
    let pi_named = Arc::new(LoweredOp::NamedConst(NamedConst::Pi));
    let pi_const = Arc::new(LoweredOp::Const(std::f64::consts::PI));
    let tree = LoweredOp::Add(Arc::clone(&pi_named), Arc::clone(&pi_const));
    let cse_root = tree.cse();

    // Children must NOT be ptr_eq (different enum variants, PartialEq guard fires).
    if let LoweredOp::Add(a, b) = cse_root.as_ref() {
        assert!(
            !Arc::ptr_eq(a, b),
            "NamedConst(Pi) and Const(PI) collide on structural_hash but \
             PartialEq guard must prevent merging (different enum variants)"
        );
    } else {
        panic!("CSE root should be Add");
    }

    // Behavioural parity is also required.
    let expected = tree.eval(&[]);
    let got = cse_root.eval(&[]);
    assert!(
        (got - expected).abs() < 1e-15,
        "NamedConst+Const parity failed: expected {expected}, got {got}"
    );
}

// ---------------------------------------------------------------------------
// 5. ±0.0 edge case
// ---------------------------------------------------------------------------

#[test]
fn cse_pos_neg_zero_edge_case() {
    // +0.0 and -0.0 have different bit patterns; structural_hash uses to_bits(),
    // so they will NOT be merged (different hashes).
    let zero_pos = Arc::new(LoweredOp::Const(0.0_f64));
    let zero_neg = Arc::new(LoweredOp::Const(-0.0_f64));
    let tree = LoweredOp::Add(Arc::clone(&zero_pos), Arc::clone(&zero_neg));
    let cse_root = tree.cse();
    // Behavioural parity: 0.0 + (-0.0) = 0.0
    let expected = tree.eval(&[]);
    let got = cse_root.eval(&[]);
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "±0.0 edge case: eval mismatch"
    );
    // Children should NOT be ptr_eq since their bit patterns differ.
    if let LoweredOp::Add(a, b) = cse_root.as_ref() {
        assert!(
            !Arc::ptr_eq(a, b),
            "0.0 and -0.0 should be distinct nodes after CSE (different to_bits)"
        );
    } else {
        panic!("CSE root should be Add");
    }
}

// ---------------------------------------------------------------------------
// 6. grad_all parity
// ---------------------------------------------------------------------------

#[test]
fn cse_grad_all_parity() {
    // f(x0, x1) = x0 * x1 + sin(x0)
    let x0 = Arc::new(LoweredOp::Var(0));
    let x1 = Arc::new(LoweredOp::Var(1));
    let expr = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(Arc::clone(&x0), Arc::clone(&x1))),
        Arc::new(LoweredOp::Sin(Arc::clone(&x0))),
    );

    let n = expr.count_vars();
    let grad_all = expr.grad_all();
    let grad_each: Vec<LoweredOp> = (0..n).map(|i| expr.grad(i)).collect();

    assert_eq!(grad_all.len(), grad_each.len(), "length mismatch");

    let vars = [0.7, 1.3];
    for (i, (ga, ge)) in grad_all.iter().zip(grad_each.iter()).enumerate() {
        let va = ga.eval(&vars);
        let ve = ge.eval(&vars);
        assert_eq!(
            va.to_bits(),
            ve.to_bits(),
            "grad_all[{i}] and grad({i}) differ: {va} vs {ve}"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. CSE reduces shared subtrees to a single Arc
// ---------------------------------------------------------------------------

#[test]
fn cse_three_way_sharing() {
    // Build Add(Add(exp_var0, exp_var0), exp_var0) — three references to Exp(Var(0)).
    let a = exp_var0();
    let b = exp_var0();
    let c = exp_var0();
    let tree = LoweredOp::Add(Arc::new(LoweredOp::Add(a, b)), c);
    let cse_root = tree.cse();
    // After CSE, all three should share the same canonical Arc.
    if let LoweredOp::Add(outer_a, outer_c) = cse_root.as_ref() {
        if let LoweredOp::Add(inner_a, inner_b) = outer_a.as_ref() {
            assert!(
                Arc::ptr_eq(inner_a, inner_b),
                "inner children should be ptr_eq"
            );
            assert!(
                Arc::ptr_eq(inner_a, outer_c),
                "outer child should match inner canonical"
            );
        } else {
            panic!("inner should be Add");
        }
    } else {
        panic!("outer should be Add");
    }
}
