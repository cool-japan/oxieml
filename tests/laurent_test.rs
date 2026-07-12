//! Public-API integration tests for P5: Laurent series + indeterminate-form
//! limits. Exercises the re-exported `oxieml::Series` type, the
//! `LoweredOp::laurent` method, and the extended `LoweredOp::limit` behaviour.

use oxieml::error::EmlError;
use oxieml::{LimitPoint, LimitResult, LoweredOp, Series};
use std::sync::Arc;

fn var() -> LoweredOp {
    LoweredOp::Var(0)
}
fn cst(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Arc::new(a), Arc::new(b))
}
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Arc::new(a), Arc::new(b))
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}
fn sin(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(Arc::new(a))
}
fn ln(a: LoweredOp) -> LoweredOp {
    LoweredOp::Ln(Arc::new(a))
}
fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Pow(Arc::new(a), Arc::new(b))
}

// --- Laurent bullets ------------------------------------------------------

#[test]
fn laurent_one_over_x_leading_x_inv() {
    let s: Series = div(cst(1.0), var())
        .laurent(0, 0.0, 3)
        .expect("laurent(1/x)");
    assert_eq!(s.lo_pow, -1);
    assert!((s.coeffs[0] - 1.0).abs() < 1e-9);
    assert_eq!(s.pole_order(), 1);
}

#[test]
fn laurent_one_over_sin_x_inv_plus_x_over_6() {
    let s = div(cst(1.0), sin(var()))
        .laurent(0, 0.0, 3)
        .expect("laurent(1/sin x)");
    assert_eq!(s.lo_pow, -1);
    // x⁻¹ + 0·x⁰ + (1/6)·x¹
    assert!((s.coeffs[0] - 1.0).abs() < 1e-9);
    assert!(s.coeffs[1].abs() < 1e-9);
    assert!((s.coeffs[2] - 1.0 / 6.0).abs() < 1e-9);
}

#[test]
fn laurent_ln_x_is_branch_point() {
    let result = ln(var()).laurent(0, 0.0, 3);
    assert!(matches!(result, Err(EmlError::BranchPoint)));
}

// --- Indeterminate-form limit bullets ------------------------------------

#[test]
fn limit_x_ln_x_is_zero() {
    // 0·∞ (domain-restricted from the right).
    match mul(var(), ln(var())).limit(0, LimitPoint::Finite(0.0)) {
        LimitResult::Finite(v) => assert!(v.abs() < 1e-3, "expected 0, got {v}"),
        other => panic!("expected Finite(≈0), got {other:?}"),
    }
}

#[test]
fn limit_one_plus_inv_x_pow_x_is_e() {
    // 1^∞.
    match pow(add(cst(1.0), div(cst(1.0), var())), var()).limit(0, LimitPoint::PosInf) {
        LimitResult::Finite(v) => {
            assert!(
                (v - std::f64::consts::E).abs() < 0.05,
                "expected e, got {v}"
            );
        }
        other => panic!("expected Finite(≈e), got {other:?}"),
    }
}

#[test]
fn limit_sin_inv_x_does_not_exist() {
    // Oscillatory: honest DoesNotExist, never a fabricated value.
    assert_eq!(
        sin(div(cst(1.0), var())).limit(0, LimitPoint::Finite(0.0)),
        LimitResult::DoesNotExist
    );
}

#[test]
fn limit_inf_minus_inf_is_zero() {
    // ∞ − ∞ resolved by the series strategy: 1/x − 1/sin x → 0.
    match sub(div(cst(1.0), var()), div(cst(1.0), sin(var()))).limit(0, LimitPoint::Finite(0.0)) {
        LimitResult::Finite(v) => assert!(v.abs() < 1e-3, "expected 0, got {v}"),
        other => panic!("expected Finite(≈0), got {other:?}"),
    }
}

// --- Series reconstruction round-trip ------------------------------------

#[test]
fn series_to_lowered_round_trips() {
    let s = div(cst(1.0), sin(var()))
        .laurent(0, 0.0, 5)
        .expect("laurent");
    let reconstructed = s.to_lowered(0);
    for &x in &[0.15_f64, -0.15, 0.25] {
        let via_series = s.eval(x);
        let via_tree = reconstructed.eval(&[x]);
        assert!(
            (via_series - via_tree).abs() < 1e-9,
            "reconstruction mismatch at {x}: {via_series} vs {via_tree}"
        );
    }
}
