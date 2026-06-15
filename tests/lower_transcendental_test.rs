//! Integration tests for the 10 new transcendental `LoweredOp` variants:
//! Tan, Sinh, Cosh, Tanh, Arcsin, Arccos, Arctan, Arcsinh, Arccosh, Arctanh.

use oxieml::lower::{LoweredOp, OxiOp};
use std::f64::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helper: compute structural hash of a LoweredOp as u64.
// ---------------------------------------------------------------------------
fn struct_hash(op: &LoweredOp) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    op.structural_hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// 1. eval_tan_matches_std
// ---------------------------------------------------------------------------
#[test]
fn eval_tan_matches_std() {
    let op = LoweredOp::Tan(Arc::new(LoweredOp::Var(0)));
    let n = 10usize;
    for i in 0..n {
        let x = -PI / 4.0 + (PI / 2.0) * (i as f64) / (n as f64 - 1.0);
        let got = op.eval(&[x]);
        let expected = x.tan();
        assert!(
            (got - expected).abs() < 1e-12,
            "tan mismatch at x={x}: got={got} expected={expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. eval_sinh_cosh_tanh_matches_std
// ---------------------------------------------------------------------------
#[test]
fn eval_sinh_cosh_tanh_matches_std() {
    let sinh_op = LoweredOp::Sinh(Arc::new(LoweredOp::Var(0)));
    let cosh_op = LoweredOp::Cosh(Arc::new(LoweredOp::Var(0)));
    let tanh_op = LoweredOp::Tanh(Arc::new(LoweredOp::Var(0)));
    let n = 10usize;
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        let vars = &[x];

        let got_sinh = sinh_op.eval(vars);
        assert!(
            (got_sinh - x.sinh()).abs() < 1e-12,
            "sinh mismatch at x={x}: {got_sinh} vs {}",
            x.sinh()
        );

        let got_cosh = cosh_op.eval(vars);
        assert!(
            (got_cosh - x.cosh()).abs() < 1e-12,
            "cosh mismatch at x={x}: {got_cosh} vs {}",
            x.cosh()
        );

        let got_tanh = tanh_op.eval(vars);
        assert!(
            (got_tanh - x.tanh()).abs() < 1e-12,
            "tanh mismatch at x={x}: {got_tanh} vs {}",
            x.tanh()
        );
    }
}

// ---------------------------------------------------------------------------
// 3. eval_arcsin_arccos_arctan_matches_std
// ---------------------------------------------------------------------------
#[test]
fn eval_arcsin_arccos_arctan_matches_std() {
    let arcsin_op = LoweredOp::Arcsin(Arc::new(LoweredOp::Var(0)));
    let arccos_op = LoweredOp::Arccos(Arc::new(LoweredOp::Var(0)));
    let arctan_op = LoweredOp::Arctan(Arc::new(LoweredOp::Var(0)));
    let n = 10usize;
    for i in 0..n {
        let x = -0.9 + 1.8 * (i as f64) / (n as f64 - 1.0);
        let vars = &[x];

        let got = arcsin_op.eval(vars);
        assert!(
            (got - x.asin()).abs() < 1e-12,
            "arcsin mismatch at x={x}: {got} vs {}",
            x.asin()
        );

        let got = arccos_op.eval(vars);
        assert!(
            (got - x.acos()).abs() < 1e-12,
            "arccos mismatch at x={x}: {got} vs {}",
            x.acos()
        );

        let got = arctan_op.eval(vars);
        assert!(
            (got - x.atan()).abs() < 1e-12,
            "arctan mismatch at x={x}: {got} vs {}",
            x.atan()
        );
    }
}

// ---------------------------------------------------------------------------
// 4. eval_arcsinh_arccosh_arctanh_matches_std
// ---------------------------------------------------------------------------
#[test]
fn eval_arcsinh_arccosh_arctanh_matches_std() {
    let arcsinh_op = LoweredOp::Arcsinh(Arc::new(LoweredOp::Var(0)));
    let arccosh_op = LoweredOp::Arccosh(Arc::new(LoweredOp::Var(0)));
    let arctanh_op = LoweredOp::Arctanh(Arc::new(LoweredOp::Var(0)));

    // arcsinh over [-2, 2]
    let n = 10usize;
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        let got = arcsinh_op.eval(&[x]);
        assert!(
            (got - x.asinh()).abs() < 1e-12,
            "arcsinh mismatch at x={x}: {got} vs {}",
            x.asinh()
        );
    }

    // arccosh over [1.1, 3.0]
    for i in 0..n {
        let x = 1.1 + 1.9 * (i as f64) / (n as f64 - 1.0);
        let got = arccosh_op.eval(&[x]);
        assert!(
            (got - x.acosh()).abs() < 1e-12,
            "arccosh mismatch at x={x}: {got} vs {}",
            x.acosh()
        );
    }

    // arctanh over [-0.9, 0.9]
    for i in 0..n {
        let x = -0.9 + 1.8 * (i as f64) / (n as f64 - 1.0);
        let got = arctanh_op.eval(&[x]);
        assert!(
            (got - x.atanh()).abs() < 1e-12,
            "arctanh mismatch at x={x}: {got} vs {}",
            x.atanh()
        );
    }
}

// ---------------------------------------------------------------------------
// 5. grad_tan_central_diff
// ---------------------------------------------------------------------------
#[test]
fn grad_tan_central_diff() {
    let op = LoweredOp::Tan(Arc::new(LoweredOp::Var(0)));
    let x = 0.3_f64;
    let h = 1e-5;
    let numerical = (op.eval(&[x + h]) - op.eval(&[x - h])) / (2.0 * h);
    let symbolic = op.grad(0).eval(&[x]);
    assert!(
        (symbolic - numerical).abs() < 1e-6,
        "grad tan: symbolic={symbolic} numerical={numerical}"
    );
}

// ---------------------------------------------------------------------------
// 6. grad_sinh_central_diff
// ---------------------------------------------------------------------------
#[test]
fn grad_sinh_central_diff() {
    let op = LoweredOp::Sinh(Arc::new(LoweredOp::Var(0)));
    let x = 0.5_f64;
    let h = 1e-5;
    let numerical = (op.eval(&[x + h]) - op.eval(&[x - h])) / (2.0 * h);
    let symbolic = op.grad(0).eval(&[x]);
    assert!(
        (symbolic - numerical).abs() < 1e-6,
        "grad sinh: symbolic={symbolic} numerical={numerical}"
    );
}

// ---------------------------------------------------------------------------
// 7. grad_tanh_central_diff
// ---------------------------------------------------------------------------
#[test]
fn grad_tanh_central_diff() {
    let op = LoweredOp::Tanh(Arc::new(LoweredOp::Var(0)));
    let x = 0.5_f64;
    let h = 1e-5;
    let numerical = (op.eval(&[x + h]) - op.eval(&[x - h])) / (2.0 * h);
    let symbolic = op.grad(0).eval(&[x]);
    assert!(
        (symbolic - numerical).abs() < 1e-6,
        "grad tanh: symbolic={symbolic} numerical={numerical}"
    );
}

// ---------------------------------------------------------------------------
// 8. grad_arctan_central_diff
// ---------------------------------------------------------------------------
#[test]
fn grad_arctan_central_diff() {
    let op = LoweredOp::Arctan(Arc::new(LoweredOp::Var(0)));
    let x = 0.5_f64;
    let h = 1e-5;
    let numerical = (op.eval(&[x + h]) - op.eval(&[x - h])) / (2.0 * h);
    let symbolic = op.grad(0).eval(&[x]);
    assert!(
        (symbolic - numerical).abs() < 1e-6,
        "grad arctan: symbolic={symbolic} numerical={numerical}"
    );
}

// ---------------------------------------------------------------------------
// 9. grad_arctanh_central_diff
// ---------------------------------------------------------------------------
#[test]
fn grad_arctanh_central_diff() {
    let op = LoweredOp::Arctanh(Arc::new(LoweredOp::Var(0)));
    let x = 0.3_f64;
    let h = 1e-5;
    let numerical = (op.eval(&[x + h]) - op.eval(&[x - h])) / (2.0 * h);
    let symbolic = op.grad(0).eval(&[x]);
    assert!(
        (symbolic - numerical).abs() < 1e-6,
        "grad arctanh: symbolic={symbolic} numerical={numerical}"
    );
}

// ---------------------------------------------------------------------------
// 10. simplify_tan_arctan_collapses
// ---------------------------------------------------------------------------
#[test]
fn simplify_tan_arctan_collapses() {
    let inner = LoweredOp::Arctan(Arc::new(LoweredOp::Var(0)));
    let op = LoweredOp::Tan(Arc::new(inner));
    let simplified = op.simplify();
    assert_eq!(
        simplified,
        LoweredOp::Var(0),
        "tan(arctan(x)) should simplify to Var(0), got {simplified:?}"
    );
}

// ---------------------------------------------------------------------------
// 11. simplify_tanh_arctanh_collapses
// ---------------------------------------------------------------------------
#[test]
fn simplify_tanh_arctanh_collapses() {
    let inner = LoweredOp::Arctanh(Arc::new(LoweredOp::Var(0)));
    let op = LoweredOp::Tanh(Arc::new(inner));
    let simplified = op.simplify();
    assert_eq!(
        simplified,
        LoweredOp::Var(0),
        "tanh(arctanh(x)) should simplify to Var(0), got {simplified:?}"
    );
}

// ---------------------------------------------------------------------------
// 12. to_pretty_golden
// ---------------------------------------------------------------------------
#[test]
fn to_pretty_golden() {
    let op = LoweredOp::Tan(Arc::new(LoweredOp::Var(0)));
    let pretty = op.to_pretty();
    // Display uses "x0" for Var(0) — not unicode subscript
    assert_eq!(pretty, "tan(x0)", "expected 'tan(x0)', got '{pretty}'");
}

// ---------------------------------------------------------------------------
// 13. to_latex_golden
// ---------------------------------------------------------------------------
#[test]
fn to_latex_golden() {
    let op = LoweredOp::Sinh(Arc::new(LoweredOp::Var(0)));
    let latex = op.to_latex();
    assert_eq!(
        latex, r"\sinh{x_{0}}",
        "expected r#\"\\sinh{{x_{{0}}}}\"#, got '{latex}'"
    );
}

// ---------------------------------------------------------------------------
// 14. oxiop_emission_parity
// ---------------------------------------------------------------------------
#[test]
fn oxiop_emission_parity() {
    let op = LoweredOp::Tan(Arc::new(LoweredOp::Var(0)));
    let ops = op.to_oxiblas_ops();
    assert!(
        ops.contains(&OxiOp::Tan),
        "expected OxiOp::Tan in ops: {ops:?}"
    );
}

// ---------------------------------------------------------------------------
// 15. canonical_recognizer_tan
// ---------------------------------------------------------------------------
#[test]
fn canonical_recognizer_tan() {
    // Build Div(Sin(Var(0)), Cos(Var(0))) and simplify — should become Tan(Var(0))
    let div_op = LoweredOp::Div(
        Arc::new(LoweredOp::Sin(Arc::new(LoweredOp::Var(0)))),
        Arc::new(LoweredOp::Cos(Arc::new(LoweredOp::Var(0)))),
    );
    let simplified = div_op.simplify();
    assert_eq!(
        simplified,
        LoweredOp::Tan(Arc::new(LoweredOp::Var(0))),
        "Div(Sin(x), Cos(x)).simplify() should give Tan(x), got {simplified:?}"
    );
}

// ---------------------------------------------------------------------------
// 16. canonical_recognizer_sinh
// ---------------------------------------------------------------------------
#[test]
fn canonical_recognizer_sinh() {
    // Build (exp(x) - exp(-x)) / 2 and check structural hash equals Sinh(x)
    let x = Arc::new(LoweredOp::Var(0));
    let exp_x = LoweredOp::Exp(x.clone());
    let neg_x = LoweredOp::Neg(x);
    let exp_neg_x = LoweredOp::Exp(Arc::new(neg_x));
    let sub = LoweredOp::Sub(Arc::new(exp_x), Arc::new(exp_neg_x));
    let div = LoweredOp::Div(Arc::new(sub), Arc::new(LoweredOp::Const(2.0)));

    let simplified = div.simplify();
    let expected = LoweredOp::Sinh(Arc::new(LoweredOp::Var(0)));

    assert_eq!(
        struct_hash(&simplified),
        struct_hash(&expected),
        "canonical sinh recognizer failed: simplified={simplified:?}"
    );
}
