//! Integration tests for deep trigonometric symbolic integration (P4):
//! trig power/product reduction, the Weierstrass `t = tan(x/2)` substitution,
//! and the by-parts solve-for-I family.
//!
//! Every closed form is checked by differentiation — both symbolically
//! (`F.grad(0)`) and numerically (central difference) — against the original
//! integrand. The final two tests lock in that the non-elementary integrals
//! `∫ sin(x²)` and `∫ exp(x²)` stay `Unsupported`.

use oxieml::IntegrateResult;
use oxieml::LoweredOp;
use std::sync::Arc;

// ── builders ──────────────────────────────────────────────────────────────────

fn x() -> LoweredOp {
    LoweredOp::Var(0)
}
fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}
fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Arc::new(a), Arc::new(b))
}
fn powf(a: LoweredOp, e: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(a), Arc::new(c(e)))
}
fn sin(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(Arc::new(a))
}
fn cos(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Arc::new(a))
}
fn tan(a: LoweredOp) -> LoweredOp {
    LoweredOp::Tan(Arc::new(a))
}
fn exp(a: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Arc::new(a))
}

// ── verification helper ─────────────────────────────────────────────────────────

/// Integrate `f`, assert `Closed`, and verify `d/dx F ≈ f` at `points` via BOTH
/// the symbolic gradient and a numeric central difference.
fn assert_closed_and_verified(f: &LoweredOp, expected: impl Fn(f64) -> f64, points: &[f64]) {
    let IntegrateResult::Closed(anti) = f.integrate(0) else {
        panic!("expected Closed antiderivative for {f}, got Unsupported");
    };
    let grad = anti.grad(0).simplify();
    let h = 1e-6_f64;
    for &xv in points {
        let want = expected(xv);
        // Symbolic derivative of the antiderivative.
        let sym = grad.eval(&[xv]);
        assert!(
            (sym - want).abs() < 1e-6,
            "symbolic F'({xv}) = {sym} but f = {want} for {f}"
        );
        // Numeric central difference of the antiderivative.
        let num = (anti.eval(&[xv + h]) - anti.eval(&[xv - h])) / (2.0 * h);
        assert!(
            (num - want).abs() < 1e-4,
            "numeric F'({xv}) = {num} but f = {want} for {f}"
        );
    }
}

// ── spec bullet 1: ∫ sin²x = x/2 − sin(2x)/4 ─────────────────────────────────

#[test]
fn integrate_sin_squared_closed() {
    let f = powf(sin(x()), 2.0);
    assert_closed_and_verified(&f, |v| v.sin().powi(2), &[0.3, 0.7, 1.2, -0.5, 2.0]);
}

// ── spec bullet 2: ∫ tan²x = tan x − x ───────────────────────────────────────

#[test]
fn integrate_tan_squared_closed() {
    let f = powf(tan(x()), 2.0);
    assert_closed_and_verified(&f, |v| v.tan().powi(2), &[0.3, 0.7, 1.2, -0.5]);
}

// ── spec bullet 3: ∫ 1/(2 + cos x) via Weierstrass ───────────────────────────

#[test]
fn integrate_one_over_two_plus_cos_weierstrass() {
    let f = div(c(1.0), add(c(2.0), cos(x())));
    assert_closed_and_verified(&f, |v| 1.0 / (2.0 + v.cos()), &[0.3, 0.7, 1.2, -0.5, 2.0]);
}

// ── spec bullet 4: ∫ eˣ sin x = eˣ(sin x − cos x)/2 (solve-for-I) ─────────────

#[test]
fn integrate_exp_sin_solve_for_i() {
    let f = mul(exp(x()), sin(x()));
    assert_closed_and_verified(&f, |v| v.exp() * v.sin(), &[0.3, 0.7, 1.2, -0.5]);

    // Exact value check against the known closed form eˣ(sin x − cos x)/2.
    let IntegrateResult::Closed(anti) = f.integrate(0) else {
        panic!("expected Closed");
    };
    for xv in [0.4_f64, 1.1, -0.3] {
        let got = anti.eval(&[xv]);
        let want = xv.exp() * (xv.sin() - xv.cos()) / 2.0;
        // Antiderivatives differ by a constant; compare differences instead.
        let got0 = anti.eval(&[0.0]);
        let want0 = 0.0_f64.exp() * (0.0_f64.sin() - 0.0_f64.cos()) / 2.0;
        assert!(
            ((got - got0) - (want - want0)).abs() < 1e-9,
            "∫eˣsin x mismatch at {xv}: got {got}, want {want}"
        );
    }
}

// ── additional coverage ──────────────────────────────────────────────────────

#[test]
fn integrate_cos_squared_closed() {
    let f = powf(cos(x()), 2.0);
    assert_closed_and_verified(&f, |v| v.cos().powi(2), &[0.3, 0.7, 1.2, -0.5, 2.0]);
}

#[test]
fn integrate_sin_cubed_closed() {
    let f = powf(sin(x()), 3.0);
    assert_closed_and_verified(&f, |v| v.sin().powi(3), &[0.3, 0.7, 1.2, -0.5, 2.0]);
}

#[test]
fn integrate_sec_squared_closed() {
    // sec²x = 1/cos²x = Pow(cos x, −2); ∫ = tan x.
    let f = powf(cos(x()), -2.0);
    assert_closed_and_verified(&f, |v| 1.0 / v.cos().powi(2), &[0.3, 0.7, 1.2, -0.5]);
}

#[test]
fn integrate_sin_sq_cos_sq_product() {
    let f = mul(powf(sin(x()), 2.0), powf(cos(x()), 2.0));
    assert_closed_and_verified(
        &f,
        |v| v.sin().powi(2) * v.cos().powi(2),
        &[0.3, 0.7, 1.2, -0.5],
    );
}

#[test]
fn integrate_sin2x_cos3x_product_to_sum() {
    let f = mul(sin(mul(c(2.0), x())), cos(mul(c(3.0), x())));
    assert_closed_and_verified(
        &f,
        |v| (2.0 * v).sin() * (3.0 * v).cos(),
        &[0.3, 0.7, 1.2, -0.5, 2.0],
    );
}

#[test]
fn integrate_exp_cos_solve_for_i() {
    let f = mul(exp(x()), cos(x()));
    assert_closed_and_verified(&f, |v| v.exp() * v.cos(), &[0.3, 0.7, 1.2, -0.5]);
}

#[test]
fn integrate_one_over_one_plus_sin_cos_weierstrass() {
    // ∫ 1/(1 + sin x + cos x) dx — a genuine R(sin,cos) requiring Weierstrass.
    let f = div(c(1.0), add(add(c(1.0), sin(x())), cos(x())));
    // Avoid the pole where 1 + sin x + cos x = 0 (near x = π); sample small x.
    assert_closed_and_verified(
        &f,
        |v| 1.0 / (1.0 + v.sin() + v.cos()),
        &[0.2, 0.5, 0.9, 1.3, -0.4],
    );
}

// ── CRITICAL GUARDRAIL: non-elementary integrals stay Unsupported ─────────────

#[test]
fn integrate_sin_x_squared_stays_unsupported() {
    // ∫ sin(x²) is a Fresnel integral — must NOT be claimed closed.
    let f = sin(powf(x(), 2.0));
    assert!(
        matches!(f.integrate(0), IntegrateResult::Unsupported),
        "∫ sin(x²) must stay Unsupported"
    );
}

#[test]
fn integrate_exp_x_squared_stays_unsupported() {
    // ∫ exp(x²) is an error-function integral — must NOT be claimed closed.
    let f = exp(powf(x(), 2.0));
    assert!(
        matches!(f.integrate(0), IntegrateResult::Unsupported),
        "∫ exp(x²) must stay Unsupported"
    );
}
