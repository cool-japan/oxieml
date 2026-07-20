//! Regression tests for the digamma-gradient fix (item T3).
//!
//! Before this fix, `d/dx digamma(f)` silently returned `0` (a wrong gradient)
//! in symbolic `grad`, forward-mode (JVP), and reverse-mode (VJP) autodiff. The
//! correct rule is `d/dx digamma(f) = trigamma(f)·f'`. These tests pin that rule
//! across all three differentiation paths and cross-check it against the
//! already-correct `lgamma → digamma` chain and against a finite difference.
//!
//! They also document the *intentional* boundary: `d/dx trigamma(f)` would need
//! the tetragamma function (ψ²), which is a known, reported capability limit and
//! evaluates to `0` by design.

use oxieml::LoweredOp;
use oxieml::special::{digamma, trigamma};
use std::sync::Arc;

/// The evaluation points shared by every test in this file.
const X_POINTS: [f64; 3] = [1.5, 2.5, 3.0];

/// Build the expression `digamma(x0)`.
fn digamma_var0() -> LoweredOp {
    LoweredOp::Digamma(Arc::new(LoweredOp::Var(0)))
}

/// Build the expression `trigamma(x0)`.
fn trigamma_var0() -> LoweredOp {
    LoweredOp::Trigamma(Arc::new(LoweredOp::Var(0)))
}

/// Build the expression `lgamma(x0)`.
fn lgamma_var0() -> LoweredOp {
    LoweredOp::LGamma(Arc::new(LoweredOp::Var(0)))
}

/// Central finite difference of `f` at `x` with step `h`.
fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Symbolic grad: `d/dx digamma(x)` must equal `trigamma(x)`, matched against a
/// central finite difference of `digamma`. It must NOT be the old `0` stub.
#[test]
fn digamma_symbolic_grad_matches_finite_difference() {
    let grad = digamma_var0().grad(0);

    // Structural guard against a regression to the old `Const(0.0)` stub.
    assert_ne!(
        grad,
        LoweredOp::Const(0.0),
        "digamma gradient regressed to the silent zero stub"
    );

    let h = 1e-5;
    let tol = 1e-4;
    for &x in &X_POINTS {
        let symbolic = grad.eval(&[x]);
        let fd = central_diff(digamma, x, h);
        let exact = trigamma(x);

        // The symbolic derivative is exactly trigamma(x)...
        assert!(
            (symbolic - exact).abs() < 1e-12,
            "symbolic grad != trigamma at x={x}: {symbolic} vs {exact}"
        );
        // ...and it agrees with the finite difference of digamma.
        assert!(
            (symbolic - fd).abs() < tol,
            "symbolic grad vs finite diff mismatch at x={x}: {symbolic} vs {fd}"
        );
        // And it is emphatically not zero (trigamma is ~0.39..0.94 here).
        assert!(
            symbolic.abs() > 1e-3,
            "symbolic grad unexpectedly ~0 at x={x}: {symbolic}"
        );
    }
}

/// Forward-mode JVP: `du` of `digamma(x0)` equals `trigamma(x)·tangent`.
#[test]
fn digamma_jvp_equals_trigamma_times_tangent() {
    let expr = digamma_var0();
    for &x in &X_POINTS {
        for &tangent in &[1.0_f64, 2.5, -0.75] {
            let (val, du) = expr.jvp(&[x], &[tangent]);
            assert!(
                (val - digamma(x)).abs() < 1e-12,
                "jvp value != digamma at x={x}: {val}"
            );
            let expected = trigamma(x) * tangent;
            assert!(
                (du - expected).abs() < 1e-12,
                "jvp du != trigamma*tangent at x={x}, t={tangent}: {du} vs {expected}"
            );
            // A unit tangent must not collapse to the old zero.
            if tangent != 0.0 {
                assert!(du.abs() > 1e-3, "jvp du unexpectedly ~0 at x={x}");
            }
        }
    }
}

/// Reverse-mode VJP: the adjoint flowing into `x0` through `digamma` is
/// `trigamma(x)` (with the default incoming adjoint of 1.0).
#[test]
fn digamma_vjp_yields_trigamma_adjoint() {
    let expr = digamma_var0();
    for &x in &X_POINTS {
        let (val, grad) = expr.vjp(&[x]);
        assert!(
            (val - digamma(x)).abs() < 1e-12,
            "vjp value != digamma at x={x}: {val}"
        );
        assert_eq!(grad.len(), 1, "expected a single-variable gradient");
        let expected = trigamma(x);
        assert!(
            (grad[0] - expected).abs() < 1e-12,
            "vjp grad != trigamma at x={x}: {} vs {expected}",
            grad[0]
        );
        assert!(grad[0].abs() > 1e-3, "vjp grad unexpectedly ~0 at x={x}");
    }
}

/// Reverse-mode VJP correctly scales the propagated adjoint: for
/// `k · digamma(x0)` the gradient is `k · trigamma(x)`, exercising the
/// `adj * trigamma(vx)` factor in the sweep.
#[test]
fn digamma_vjp_scales_with_incoming_adjoint() {
    let k = 3.0;
    let expr = LoweredOp::Mul(Arc::new(LoweredOp::Const(k)), Arc::new(digamma_var0()));
    for &x in &X_POINTS {
        let (val, grad) = expr.vjp(&[x]);
        assert!(
            (val - k * digamma(x)).abs() < 1e-12,
            "vjp value != k*digamma at x={x}: {val}"
        );
        let expected = k * trigamma(x);
        assert!(
            (grad[0] - expected).abs() < 1e-12,
            "vjp scaled grad != k*trigamma at x={x}: {} vs {expected}",
            grad[0]
        );
    }
}

/// End-to-end identity through the `lgamma → digamma → trigamma` chain:
/// `d/dx lgamma(x) = digamma(x)` and `d²/dx² lgamma(x) = trigamma(x)`.
/// The second derivative is `grad∘grad`, which now routes through the newly
/// wired digamma-derivative arm.
#[test]
fn lgamma_first_and_second_derivative_identity() {
    let first = lgamma_var0().grad(0);
    let second = first.grad(0);

    // The second derivative must not be the old zero stub.
    assert_ne!(
        second,
        LoweredOp::Const(0.0),
        "d^2/dx^2 lgamma regressed to the silent zero stub"
    );

    for &x in &X_POINTS {
        let first_val = first.eval(&[x]);
        assert!(
            (first_val - digamma(x)).abs() < 1e-12,
            "d/dx lgamma != digamma at x={x}: {first_val}"
        );
        let second_val = second.eval(&[x]);
        assert!(
            (second_val - trigamma(x)).abs() < 1e-12,
            "d^2/dx^2 lgamma != trigamma at x={x}: {second_val}"
        );
    }
}

/// Documented capability boundary: `d/dx trigamma(x)` needs the tetragamma
/// function (ψ²), which is intentionally unsupported. It evaluates to `0` by
/// design across symbolic grad, JVP, and VJP. This is a known, reported
/// limitation — asserted here as the current documented behavior, NOT ignored.
#[test]
fn trigamma_grad_is_documented_tetragamma_boundary_zero() {
    // Symbolic: grad collapses to exactly Const(0.0).
    let grad = trigamma_var0().grad(0);
    assert_eq!(
        grad,
        LoweredOp::Const(0.0),
        "trigamma grad is the documented tetragamma boundary (Const(0.0))"
    );

    let expr = trigamma_var0();
    for &x in &X_POINTS {
        // Forward-mode: value is trigamma(x), but du is 0 at the boundary.
        let (val, du) = expr.jvp(&[x], &[1.0]);
        assert!(
            (val - trigamma(x)).abs() < 1e-12,
            "trigamma jvp value != trigamma at x={x}: {val}"
        );
        assert!(
            du.abs() < 1e-12,
            "trigamma jvp du must be 0 at the tetragamma boundary, got {du}"
        );

        // Reverse-mode: no gradient is propagated (adjoint stays 0).
        let (rval, rgrad) = expr.vjp(&[x]);
        assert!(
            (rval - trigamma(x)).abs() < 1e-12,
            "trigamma vjp value != trigamma at x={x}: {rval}"
        );
        assert!(
            rgrad[0].abs() < 1e-12,
            "trigamma vjp grad must be 0 at the tetragamma boundary, got {}",
            rgrad[0]
        );
    }
}
