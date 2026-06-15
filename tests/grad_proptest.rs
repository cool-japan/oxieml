//! Property-based tests for `LoweredOp::grad`.
//!
//! Verifies that the symbolic gradient matches the numerical central-difference
//! approximation for randomly generated expression trees.

use oxieml::lower::LoweredOp;
use proptest::prelude::*;
use std::sync::Arc;

/// Recursive strategy that produces random `LoweredOp` trees.
///
/// Leaves: `Var(0)`, `Var(1)`, and constants in `[-3.0, 3.0]`.
/// Unary ops: `Neg`, `Sin`, `Cos`, `Tanh`, `Arctan`.
/// Binary ops: `Add`, `Sub`, `Mul`.
///
/// Deliberately omits `Div`, `Ln`, `Exp`, `Pow` and the inverse-trig ops
/// that produce singularities or explode quickly, to keep numerical
/// differentiation reliable across the full `-1..1` evaluation range.
fn lowered_op_strategy(depth: u32) -> impl Strategy<Value = LoweredOp> {
    let leaf = prop_oneof![
        Just(LoweredOp::Var(0)),
        Just(LoweredOp::Var(1)),
        (-3.0_f64..3.0_f64).prop_map(LoweredOp::Const),
    ];

    leaf.prop_recursive(depth, 64, 2, |inner| {
        prop_oneof![
            // Unary ops — all bounded or smooth on the evaluation range
            inner.clone().prop_map(|x| LoweredOp::Neg(Arc::new(x))),
            inner.clone().prop_map(|x| LoweredOp::Sin(Arc::new(x))),
            inner.clone().prop_map(|x| LoweredOp::Cos(Arc::new(x))),
            inner.clone().prop_map(|x| LoweredOp::Tanh(Arc::new(x))),
            inner.clone().prop_map(|x| LoweredOp::Arctan(Arc::new(x))),
            // Binary ops
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| LoweredOp::Add(Arc::new(a), Arc::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| LoweredOp::Sub(Arc::new(a), Arc::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| LoweredOp::Mul(Arc::new(a), Arc::new(b))),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// For 1024 random trees the symbolic gradient must match the central
    /// difference at three randomly sampled points in `[-1, 1]^2`.
    #[ignore = "heavy: slow integration test, run manually"]
    #[test]
    fn grad_matches_central_difference(
        tree in lowered_op_strategy(6),
        x0_a in -1.0_f64..1.0_f64,
        x1_a in -1.0_f64..1.0_f64,
        x0_b in -1.0_f64..1.0_f64,
        x1_b in -1.0_f64..1.0_f64,
        x0_c in -1.0_f64..1.0_f64,
        x1_c in -1.0_f64..1.0_f64,
    ) {
        let h           = 1e-5_f64;
        let tol_rel     = 1e-4_f64;
        let tol_abs     = 1e-5_f64;

        for (x0, x1) in [(x0_a, x1_a), (x0_b, x1_b), (x0_c, x1_c)] {
            let point: [f64; 2] = [x0, x1];

            for wrt in 0..2_usize {
                // Symbolic derivative — grad() already calls simplify() internally.
                let grad_tree = tree.grad(wrt);
                let symbolic  = grad_tree.eval(&point);

                // Central difference.
                let mut hi = point.to_vec();
                hi[wrt] += h;
                let mut lo = point.to_vec();
                lo[wrt] -= h;
                let numerical = (tree.eval(&hi) - tree.eval(&lo)) / (2.0 * h);

                // Skip non-finite evaluations (singularities, overflows).
                if !symbolic.is_finite() || !numerical.is_finite() {
                    continue;
                }

                let err = (symbolic - numerical).abs();
                let tol = tol_rel * numerical.abs().max(1.0) + tol_abs;

                prop_assert!(
                    err <= tol,
                    "grad({wrt}) mismatch at ({x0}, {x1}): \
                     symbolic={symbolic}, numerical={numerical}, err={err}, tol={tol}",
                );
            }
        }
    }
}

/// Non-proptest sanity check: d/dx₀ (x₀ · x₁) at (2, 3) = 3.
#[test]
fn grad_proptest_sanity() {
    let tree = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let g0 = tree.grad(0);
    let val = g0.eval(&[2.0, 3.0]);
    assert!((val - 3.0).abs() < 1e-12, "expected 3.0, got {val}",);
}
