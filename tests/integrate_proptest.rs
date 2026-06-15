//! Property-based tests for `LoweredOp::integrate`.
//!
//! For random LoweredOp trees, when `integrate(f)` returns `Closed(F)`,
//! verifies that `F.grad(wrt)` matches `f` numerically at sampled points.

use oxieml::integrate::IntegrateResult;
use oxieml::lower::LoweredOp;
use proptest::prelude::*;
use std::sync::Arc;

/// Recursive strategy producing random `LoweredOp` trees suitable for
/// integration testing.
///
/// Uses only ops with well-behaved antiderivatives: `Var(0)`, constants,
/// `Add`, `Sub`, `Mul` (with one constant child), `Neg`, `Sin`, `Cos`, `Exp`.
/// Deliberately omits `Div`, `Ln`, `Pow`, and inverse-trig ops that would
/// create domain issues or are likely to return `Unsupported`.
fn integrable_op_strategy(depth: u32) -> impl Strategy<Value = LoweredOp> {
    let leaf = prop_oneof![
        Just(LoweredOp::Var(0)),
        (-2.0_f64..2.0_f64).prop_map(LoweredOp::Const),
    ];

    leaf.prop_recursive(depth, 32, 2, |inner| {
        prop_oneof![
            // Unary ops — safe domain everywhere
            inner.clone().prop_map(|x| LoweredOp::Neg(Arc::new(x))),
            inner.clone().prop_map(|x| LoweredOp::Sin(Arc::new(x))),
            inner.clone().prop_map(|x| LoweredOp::Cos(Arc::new(x))),
            // Binary ops: Add and Sub (both children may depend on x)
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| LoweredOp::Add(Arc::new(a), Arc::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| LoweredOp::Sub(Arc::new(a), Arc::new(b))),
            // Mul with a constant on the left (so the constant-factor rule fires)
            ((-2.0_f64..2.0_f64), inner.clone())
                .prop_map(|(c, b)| LoweredOp::Mul(Arc::new(LoweredOp::Const(c)), Arc::new(b))),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// For 512 random trees, when integration returns `Closed(F)`, verify
    /// that `F'(x) ≈ f(x)` at three sampled evaluation points.
    #[test]
    fn antiderivative_derivative_matches_original(
        tree in integrable_op_strategy(4),
        x0 in 0.1_f64..1.9_f64,
        x1 in 0.1_f64..1.9_f64,
        x2 in 0.1_f64..1.9_f64,
    ) {
        let IntegrateResult::Closed(anti) = tree.integrate(0) else { return Ok(()); };
        let anti_grad = anti.grad(0);

        let h = 1e-6_f64;
        let tol_rel = 1e-3_f64;
        let tol_abs = 1e-5_f64;

        for xv in [x0, x1, x2] {
            // Evaluate original f at xv.
            let fv = tree.eval(&[xv]);
            // Evaluate symbolic F'(xv).
            let symbolic = anti_grad.eval(&[xv]);
            // Forward-difference approximation of F'(xv).
            let numerical = (anti.eval(&[xv + h]) - anti.eval(&[xv])) / h;

            // Skip non-finite values (singularities, overflow).
            if !fv.is_finite() || !symbolic.is_finite() || !numerical.is_finite() {
                continue;
            }

            // F'(x) should ≈ f(x).
            let err = (symbolic - fv).abs();
            let tol = tol_rel * fv.abs().max(1.0) + tol_abs;
            prop_assert!(
                err <= tol,
                "F'(x) mismatch at x={xv:.4}: F'={symbolic:.6}, f={fv:.6}, err={err:.2e}"
            );

            // Also verify numerical ≈ symbolic (sanity check on F).
            let err2 = (numerical - symbolic).abs();
            let tol2 = tol_rel * symbolic.abs().max(1.0) + tol_abs;
            prop_assert!(
                err2 <= tol2,
                "numerical F' mismatch at x={xv:.4}: num={numerical:.6}, sym={symbolic:.6}"
            );
        }
    }
}

/// Non-proptest sanity: ∫sin(x) returns Closed and the antiderivative is -cos(x).
#[test]
fn integrate_proptest_sanity() {
    use std::f64::consts::PI;
    let f = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let IntegrateResult::Closed(anti) = f.integrate(0) else {
        panic!("sin(x) must integrate to Closed");
    };
    // F(π) - F(0) should equal ∫₀^π sin(x) dx = 2
    let fa = anti.eval(&[0.0_f64]);
    let fb = anti.eval(&[PI]);
    assert!(
        (fb - fa - 2.0).abs() < 1e-10,
        "∫₀^π sin = 2, got {}",
        fb - fa
    );
}
