//! Tests for IntervalLO and LoweredOp::eval_interval.

use oxieml::{IntervalLO, LoweredOp};
use std::sync::Arc;

/// Create an interval for a variable binding.
fn iv(lo: f64, hi: f64) -> IntervalLO {
    IntervalLO::new(lo, hi)
}

#[test]
fn containment_add() {
    // Add(Var(0), Var(1)) over [1,2] × [3,4]
    let expr = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let ivar = [iv(1.0, 2.0), iv(3.0, 4.0)];
    let result = expr.eval_interval(&ivar);

    // Must contain every x0+x1 for x0 ∈ [1,2], x1 ∈ [3,4]
    let test_pairs = [(1.0, 3.0), (2.0, 4.0), (1.5, 3.7), (1.9, 3.1)];
    for (x0, x1) in test_pairs {
        let val = x0 + x1;
        assert!(
            result.contains(val),
            "interval {result:?} must contain {x0}+{x1}={val}"
        );
    }
}

#[test]
fn containment_mul() {
    // Mul(Var(0), Var(1)) over [-2, 3] × [-1, 4] (mixed signs)
    let expr = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let ivar = [iv(-2.0, 3.0), iv(-1.0, 4.0)];
    let result = expr.eval_interval(&ivar);

    let test_pairs = [
        (-2.0, -1.0),
        (3.0, 4.0),
        (-2.0, 4.0),
        (3.0, -1.0),
        (1.0, 2.0),
    ];
    for (x0, x1) in test_pairs {
        let val = x0 * x1;
        assert!(
            result.contains(val),
            "interval {result:?} must contain {x0}*{x1}={val}"
        );
    }
}

#[test]
fn containment_sin() {
    // Sin(Var(0)) over [-π, π]
    let expr = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let pi = std::f64::consts::PI;
    let ivar = [iv(-pi, pi)];
    let result = expr.eval_interval(&ivar);

    // sin is in [-1, 1], check containment for several points
    let xs = [-pi, -pi / 2.0, 0.0, pi / 6.0, pi / 4.0, pi / 2.0, pi];
    for x in xs {
        let val = x.sin();
        assert!(
            result.contains(val),
            "interval {result:?} must contain sin({x})={val}"
        );
    }
    // Interval must be contained in [-1, 1]
    assert!(result.lo >= -1.0 - 1e-12 && result.hi <= 1.0 + 1e-12);
}

#[test]
fn sin_full_period_returns_unit_interval() {
    // Sin(Var(0)) over [-4π, 4π]: width ≥ 2π → [-1, 1]
    let expr = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let pi = std::f64::consts::PI;
    let ivar = [iv(-4.0 * pi, 4.0 * pi)];
    let result = expr.eval_interval(&ivar);
    assert!(
        (result.lo - (-1.0)).abs() < 1e-12,
        "lo should be -1, got {}",
        result.lo
    );
    assert!(
        (result.hi - 1.0).abs() < 1e-12,
        "hi should be 1, got {}",
        result.hi
    );
}

#[test]
fn containment_cos() {
    // Cos(Var(0)) over [0, π]
    let expr = LoweredOp::Cos(Arc::new(LoweredOp::Var(0)));
    let pi = std::f64::consts::PI;
    let ivar = [iv(0.0, pi)];
    let result = expr.eval_interval(&ivar);

    let xs = [
        0.0,
        pi / 6.0,
        pi / 4.0,
        pi / 3.0,
        pi / 2.0,
        2.0 * pi / 3.0,
        pi,
    ];
    for x in xs {
        let val = x.cos();
        assert!(
            result.contains(val),
            "interval {result:?} must contain cos({x})={val}"
        );
    }
}

#[test]
fn tan_asymptote_returns_full() {
    // Tan(Var(0)) over [0, π]: contains π/2 (asymptote) → [-∞, +∞]
    let expr = LoweredOp::Tan(Arc::new(LoweredOp::Var(0)));
    let pi = std::f64::consts::PI;
    let ivar = [iv(0.0, pi)];
    let result = expr.eval_interval(&ivar);
    assert!(
        result.lo == f64::NEG_INFINITY && result.hi == f64::INFINITY,
        "result should be (-inf, inf) but got {result:?}"
    );
}

#[test]
fn tan_monotone_branch() {
    // Tan(Var(0)) over [-π/4, π/4]: no asymptote → bounded
    let expr = LoweredOp::Tan(Arc::new(LoweredOp::Var(0)));
    let pi = std::f64::consts::PI;
    let ivar = [iv(-pi / 4.0, pi / 4.0)];
    let result = expr.eval_interval(&ivar);
    assert!(
        result.lo.is_finite() && result.hi.is_finite(),
        "result should be bounded but got {result:?}"
    );
    // Check containment
    for x in [-pi / 4.0, 0.0, pi / 4.0] {
        let val = x.tan();
        assert!(
            result.contains(val),
            "interval {result:?} must contain tan({x})={val}"
        );
    }
}

#[test]
fn containment_exp() {
    // Exp(Var(0)) over [-2, 2]: monotone, check several points
    let expr = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    let ivar = [iv(-2.0, 2.0)];
    let result = expr.eval_interval(&ivar);

    let xs = [-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0];
    for x in xs {
        let val = x.exp();
        assert!(
            result.contains(val),
            "interval {result:?} must contain exp({x})={val}"
        );
    }
}

#[test]
fn containment_sinh_cosh_tanh() {
    // Sinh(Var(0)) point at x=1.0 → [sinh(1), sinh(1)]
    let sinh_expr = LoweredOp::Sinh(Arc::new(LoweredOp::Var(0)));
    let sinh_iv = [IntervalLO::point(1.0)];
    let sinh_result = sinh_expr.eval_interval(&sinh_iv);
    let expected_sinh = 1.0_f64.sinh();
    assert!(
        sinh_result.contains(expected_sinh),
        "sinh result {sinh_result:?} must contain {expected_sinh}"
    );

    // Cosh(Var(0)) at point 0.5
    let cosh_expr = LoweredOp::Cosh(Arc::new(LoweredOp::Var(0)));
    let cosh_iv = [IntervalLO::point(0.5)];
    let cosh_result = cosh_expr.eval_interval(&cosh_iv);
    let expected_cosh = 0.5_f64.cosh();
    assert!(
        cosh_result.contains(expected_cosh),
        "cosh result {cosh_result:?} must contain {expected_cosh}"
    );

    // Tanh(Var(0)) at point -1.5
    let tanh_expr = LoweredOp::Tanh(Arc::new(LoweredOp::Var(0)));
    let tanh_iv = [IntervalLO::point(-1.5)];
    let tanh_result = tanh_expr.eval_interval(&tanh_iv);
    let expected_tanh = (-1.5_f64).tanh();
    assert!(
        tanh_result.contains(expected_tanh),
        "tanh result {tanh_result:?} must contain {expected_tanh}"
    );
}

#[test]
fn arccosh_below_one_is_nan() {
    // Arccosh(Var(0)) over [0.0, 0.5]: domain requires x ≥ 1, so result is NaN
    let expr = LoweredOp::Arccosh(Arc::new(LoweredOp::Var(0)));
    let ivar = [iv(0.0, 0.5)];
    let result = expr.eval_interval(&ivar);
    assert!(
        result.lo.is_nan(),
        "arccosh of [0, 0.5] should return NaN but got {result:?}"
    );
}

#[test]
fn interval_lo_basic_ops() {
    let a = IntervalLO::new(1.0, 3.0);
    let b = IntervalLO::new(2.0, 5.0);
    let u = a.union(&b);
    assert_eq!(u.lo, 1.0);
    assert_eq!(u.hi, 5.0);

    let i = a.intersect(&b);
    assert_eq!(i.lo, 2.0);
    assert_eq!(i.hi, 3.0);

    assert!(!a.is_empty());
    assert!(IntervalLO::new(5.0, 2.0).is_empty());

    assert!(a.contains(2.0));
    assert!(!a.contains(0.5));

    assert!((a.width() - 2.0).abs() < 1e-15);

    assert_eq!(IntervalLO::point(7.0).lo, 7.0);
    assert_eq!(IntervalLO::point(7.0).hi, 7.0);
}
