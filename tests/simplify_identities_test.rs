//! Integration tests for the P3 trig/exp-log simplify identities and the
//! lightweight assumptions layer, driven entirely through the public API
//! (`oxieml::{Assumptions, VarAssumption, RewriteFlags, LoweredOp}`).

use oxieml::{Assumptions, LoweredOp, RewriteFlags, VarAssumption};
use std::sync::Arc;

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}
fn cst(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Arc::new(a), Arc::new(b))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}
fn powf(base: LoweredOp, e: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(base), Arc::new(LoweredOp::Const(e)))
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
fn sinh(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sinh(Arc::new(a))
}
fn cosh(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cosh(Arc::new(a))
}
fn ln(a: LoweredOp) -> LoweredOp {
    LoweredOp::Ln(Arc::new(a))
}

// ── Contracting (default-on) Pythagorean identities ─────────────────────────────

#[test]
fn sin_squared_plus_cos_squared_is_one() {
    let expr = add(powf(sin(var(0)), 2.0), powf(cos(var(0)), 2.0));
    let simplified = expr.simplify_with(&Assumptions::default());
    assert_eq!(
        simplified,
        LoweredOp::Const(1.0),
        "sin²x + cos²x should simplify to 1, got {simplified}"
    );
}

#[test]
fn cosh_squared_minus_sinh_squared_is_one() {
    let expr = sub(powf(cosh(var(0)), 2.0), powf(sinh(var(0)), 2.0));
    let simplified = expr.simplify_with(&Assumptions::default());
    assert_eq!(
        simplified,
        LoweredOp::Const(1.0),
        "cosh²x − sinh²x should simplify to 1, got {simplified}"
    );
}

#[test]
fn one_plus_tan_squared_is_reciprocal_cos_squared() {
    let expr = add(cst(1.0), powf(tan(var(0)), 2.0));
    let simplified = expr.simplify_with(&Assumptions::default());

    // It must no longer be an additive `1 + tan²` shape …
    assert!(
        !matches!(&simplified, LoweredOp::Add(_, _)),
        "1 + tan²x should have been rewritten to sec², got {simplified}"
    );
    // … and must equal 1/cos²x = sec²x pointwise.
    for probe in [0.3_f64, 0.7, 1.3, -0.5, 2.1] {
        let want = 1.0 + probe.tan().powi(2);
        let got = simplified.eval(&[probe]);
        assert!(
            (got - want).abs() < 1e-9,
            "sec² mismatch at {probe}: got {got}, want {want}"
        );
    }
}

// ── Assumption-gated radical of a square ────────────────────────────────────────

#[test]
fn sqrt_of_square_with_nonnegative_assumption_is_x() {
    let expr = powf(powf(var(0), 2.0), 0.5); // √(x²)
    let asm = Assumptions::new().assume_nonnegative(0);
    let simplified = expr.simplify_with(&asm);
    assert_eq!(
        simplified,
        LoweredOp::Var(0),
        "√(x²) with x≥0 should be x, got {simplified}"
    );
}

#[test]
fn sqrt_of_square_without_assumption_stays_abs() {
    let expr = powf(powf(var(0), 2.0), 0.5); // √(x²) = |x|
    let simplified = expr.simplify_with(&Assumptions::default());

    // Must NOT have collapsed to the bare variable x …
    assert_ne!(
        simplified,
        LoweredOp::Var(0),
        "√(x²) must stay |x| without a sign assumption, got {simplified}"
    );
    // … and must equal |x| (differs from x at negative points).
    for probe in [-3.0_f64, -0.5, 0.0, 2.0] {
        let got = simplified.eval(&[probe]);
        assert!(
            (got - probe.abs()).abs() < 1e-9,
            "√(x²) should equal |x| at {probe}: got {got}"
        );
    }
    // It is byte-identical to what the plain base `simplify` produces.
    assert_eq!(simplified, expr.simplify());
}

#[test]
fn sqrt_of_square_positive_assumption_also_collapses() {
    let expr = powf(powf(var(0), 2.0), 0.5);
    let asm = Assumptions::new().with(0, VarAssumption::new().positive());
    assert_eq!(expr.simplify_with(&asm), LoweredOp::Var(0));
}

// ── Opt-in expanding rules ──────────────────────────────────────────────────────

#[test]
fn product_to_sum_is_off_by_default_and_on_with_flag() {
    // Distinct arguments → clean product-to-sum expansion.
    let expr = mul(sin(var(0)), cos(var(1)));

    let default = expr.simplify_with(&Assumptions::default());
    assert!(
        matches!(&default, LoweredOp::Mul(_, _)),
        "product-to-sum must not fire by default, got {default}"
    );

    let flags = RewriteFlags::default().with_product_to_sum();
    let enabled = expr.simplify_with_flags(&Assumptions::default(), flags);
    assert!(
        !matches!(&enabled, LoweredOp::Mul(a, b)
            if matches!(a.as_ref(), LoweredOp::Sin(_)) && matches!(b.as_ref(), LoweredOp::Cos(_))),
        "product-to-sum should have expanded sin·cos, got {enabled}"
    );

    // Both forms agree in value.
    for (a, b) in [(0.3_f64, 1.1_f64), (-0.5, 0.7), (2.1, -0.4)] {
        let want = a.sin() * b.cos();
        assert!((default.eval(&[a, b]) - want).abs() < 1e-9);
        assert!((enabled.eval(&[a, b]) - want).abs() < 1e-9);
    }
}

#[test]
fn double_angle_off_by_default() {
    // Without the flag, sin(2x) is left as-is (a single Sin node).
    let expr = sin(mul(cst(2.0), var(0)));
    let default = expr.simplify_with(&Assumptions::default());
    assert!(
        matches!(&default, LoweredOp::Sin(_)),
        "double-angle must not fire by default, got {default}"
    );

    let flags = RewriteFlags::default().with_double_angle();
    let enabled = expr.simplify_with_flags(&Assumptions::default(), flags);
    assert!(
        !matches!(&enabled, LoweredOp::Sin(_)),
        "double-angle should have expanded sin(2x), got {enabled}"
    );
    for probe in [0.3_f64, -0.5, 1.3, 2.1] {
        assert!((enabled.eval(&[probe]) - (2.0 * probe).sin()).abs() < 1e-9);
    }
}

#[test]
fn half_angle_power_reduction_when_enabled() {
    // cos²x → (1 + cos 2x)/2 under the half_angle flag.
    let expr = powf(cos(var(0)), 2.0);
    let flags = RewriteFlags::default().with_half_angle();
    let reduced = expr.simplify_with_flags(&Assumptions::default(), flags);
    for probe in [0.3_f64, -0.5, 1.3, 2.1] {
        let want = probe.cos().powi(2);
        assert!((reduced.eval(&[probe]) - want).abs() < 1e-9);
    }
}

// ── log / exp identities via the logexp path ────────────────────────────────────

#[test]
fn logcombine_ln2_plus_ln3_is_ln6() {
    // logcombine(ln 2 + ln 3) → ln 6, exercised through the public logexp method.
    let expr = add(ln(cst(2.0)), ln(cst(3.0)));
    let combined = expr.logcombine();
    assert!(
        matches!(&combined, LoweredOp::Ln(a) if (a.eval(&[]) - 6.0).abs() < 1e-9),
        "expected ln(6), got {combined}"
    );
    assert!((combined.eval(&[]) - 6.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn log_expand_flag_splits_product_log() {
    // ln(x·y) → ln x + ln y under the log_expand flag (positive vars).
    let expr = ln(mul(var(0), var(1)));
    let flags = RewriteFlags::default().with_log_expand();
    let asm = Assumptions::new().assume_positive(0).assume_positive(1);
    let expanded = expr.simplify_with_flags(&asm, flags);
    for (a, b) in [(1.5_f64, 2.0_f64), (0.5, 3.0)] {
        let want = (a * b).ln();
        assert!((expanded.eval(&[a, b]) - want).abs() < 1e-9);
    }
}

#[test]
fn exp_expand_flag_splits_sum_exp() {
    // exp(x + y) → exp x · exp y under the exp_expand flag.
    let expr = LoweredOp::Exp(Arc::new(add(var(0), var(1))));
    let flags = RewriteFlags::default().with_exp_expand();
    let expanded = expr.simplify_with_flags(&Assumptions::default(), flags);
    for (a, b) in [(0.5_f64, 1.0_f64), (-0.5, 0.75)] {
        let want = (a + b).exp();
        assert!((expanded.eval(&[a, b]) - want).abs() < 1e-9);
    }
}

// ── Regression: default path preserves ordinary canonical forms ─────────────────

#[test]
fn default_matches_base_simplify_for_non_identity_exprs() {
    // (x + 1)·(x − 1) has no trig/sqrt identity, so simplify_with(default) must
    // reproduce the exact base canonical form.
    let x = Arc::new(LoweredOp::Var(0));
    let xp1 = add((*x).clone(), cst(1.0));
    let xm1 = sub((*x).clone(), cst(1.0));
    let expr = mul(xp1, xm1);
    assert_eq!(
        expr.simplify_with(&Assumptions::default()),
        expr.simplify(),
        "default identity layer must not perturb ordinary polynomial canonical forms"
    );
}

#[test]
fn sin_over_cos_still_becomes_tan_under_simplify_with() {
    // The base sin/cos → tan recogniser still runs beneath simplify_with.
    let expr = LoweredOp::Div(Arc::new(sin(var(0))), Arc::new(cos(var(0))));
    let simplified = expr.simplify_with(&Assumptions::default());
    assert!(
        matches!(&simplified, LoweredOp::Tan(_)),
        "sin(x)/cos(x) should still become tan(x), got {simplified}"
    );
}
