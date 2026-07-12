//! Integration tests for symbolic summation (Gosper + telescoping + Faulhaber).
//!
//! Covers every bullet of the P1 specification:
//! - `Σ k = n(n+1)/2`, `Σ k²`, `Σ k³` (Faulhaber);
//! - `Σ 2ᵏ = 2^{n+1} − 1` (geometric);
//! - `Σ k·k! = (n+1)! − 1` (Gosper);
//! - `Σ 1/(k(k+1)) = n/(n+1)` (telescoping);
//! - property verification at ≥ 6 probes;
//! - `Σ 1/k → NotClosedForm` (harmonic number, not hypergeometric-summable).

use oxieml::{LoweredOp, SumResult};
use std::sync::Arc;

// ── LoweredOp constructors ───────────────────────────────────────────────────

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
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
fn powi(base: LoweredOp, exp: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(base), Arc::new(c(exp)))
}
/// `b^k` with a constant base.
fn base_pow(base: f64, exponent: LoweredOp) -> LoweredOp {
    LoweredOp::Pow(Arc::new(c(base)), Arc::new(exponent))
}
/// `k! = Γ(k+1) = exp(lgamma(k+1))`.
fn factorial(arg: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Arc::new(LoweredOp::LGamma(Arc::new(add(arg, c(1.0))))))
}

fn factorial_f64(n: i64) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

fn closed_of(result: &SumResult) -> &LoweredOp {
    result
        .closed_form()
        .expect("expected a closed form for this summand")
}

/// Assert the discrete antidifference identity `S(n) - S(n-1) = t(n)` holds at at
/// least six integer probe points — the specification's mandatory verification.
fn assert_property_holds(term: &LoweredOp, closed: &LoweredOp) {
    let mut agreed = 0;
    for n in 2..=9i64 {
        let n_val = n as f64;
        let s_n = closed.eval(&[n_val]);
        let s_prev = closed.eval(&[n_val - 1.0]);
        let t_n = term.eval(&[n_val]);
        assert!(
            (s_n - s_prev - t_n).abs() <= 1e-6 * t_n.abs().max(1.0),
            "property S(n)-S(n-1)=t(n) failed at n={n}: S={s_n}, S(prev)={s_prev}, t={t_n}"
        );
        agreed += 1;
    }
    assert!(agreed >= 6, "need at least six probes, verified {agreed}");
}

// ── Faulhaber ────────────────────────────────────────────────────────────────

#[test]
fn sum_k_equals_triangular() {
    // Σ k = n(n+1)/2.
    let term = var(0);
    let result = term.sum_indefinite(0);
    let closed = closed_of(&result);
    for n in 0..=30i64 {
        let expected = (n * (n + 1) / 2) as f64;
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-9, "n={n}");
    }
    assert_property_holds(&term, closed);
}

#[test]
fn sum_k_squared() {
    // Σ k² = n(n+1)(2n+1)/6.
    let term = powi(var(0), 2.0);
    let result = term.sum_indefinite(0);
    let closed = closed_of(&result);
    for n in 0..=30i64 {
        let expected = (n * (n + 1) * (2 * n + 1) / 6) as f64;
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-6, "n={n}");
    }
    assert_property_holds(&term, closed);
}

#[test]
fn sum_k_cubed() {
    // Σ k³ = (n(n+1)/2)².
    let term = powi(var(0), 3.0);
    let result = term.sum_indefinite(0);
    let closed = closed_of(&result);
    for n in 0..=30i64 {
        let tri = (n * (n + 1) / 2) as f64;
        let expected = tri * tri;
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-5, "n={n}");
    }
    assert_property_holds(&term, closed);
}

#[test]
fn sum_k_definite_bounds() {
    // Σ_{k=1}^{n} k = n(n+1)/2, evaluated as S(hi) - S(lo-1).
    let term = var(0);
    let result = term.sum_definite(0, &c(1.0), &var(0));
    let closed = closed_of(&result);
    for n in 1..=25i64 {
        let expected = (n * (n + 1) / 2) as f64;
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-9, "n={n}");
    }
}

// ── Geometric ────────────────────────────────────────────────────────────────

#[test]
fn sum_two_pow_k_geometric() {
    // Σ_{k=0}^{n} 2^k = 2^{n+1} − 1.
    let term = base_pow(2.0, var(0));

    // Indefinite antidifference S(n) with S(n)-S(n-1) = 2^n.
    let indef = term.sum_indefinite(0);
    assert_property_holds(&term, closed_of(&indef));

    // Definite closed form.
    let result = term.sum_definite(0, &c(0.0), &var(0));
    let closed = closed_of(&result);
    for n in 0..=20i64 {
        let expected = 2f64.powi((n + 1) as i32) - 1.0;
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-6, "n={n}");
    }
}

#[test]
fn sum_three_pow_k_geometric() {
    // Σ_{k=0}^{n} 3^k = (3^{n+1} − 1)/2.
    let term = base_pow(3.0, var(0));
    let result = term.sum_definite(0, &c(0.0), &var(0));
    let closed = closed_of(&result);
    for n in 0..=15i64 {
        let expected = (3f64.powi((n + 1) as i32) - 1.0) / 2.0;
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-4, "n={n}");
    }
}

// ── Gosper ───────────────────────────────────────────────────────────────────

#[test]
fn sum_k_times_k_factorial_gosper() {
    // Σ_{k=1}^{n} k·k! = (n+1)! − 1.
    let term = mul(var(0), factorial(var(0)));

    // Indefinite antidifference is (n+1)!; verify the property.
    let indef = term.sum_indefinite(0);
    assert!(indef.is_closed(), "k·k! must be Gosper-summable");
    assert_property_holds(&term, closed_of(&indef));

    // Definite closed form (n+1)! − 1.
    let result = term.sum_definite(0, &c(1.0), &var(0));
    let closed = closed_of(&result);
    for n in 1..=8i64 {
        let expected = factorial_f64(n + 1) - 1.0;
        let got = closed.eval(&[n as f64]);
        assert!(
            (got - expected).abs() <= 1e-6 * expected.abs().max(1.0),
            "n={n}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn sum_k_times_k_factorial_concrete_bounds() {
    // Concrete numeric bounds exercise the brute-force cross-check.
    // Σ_{k=1}^{5} k·k! = 1 + 4 + 18 + 96 + 600 = 719 = 6! − 1.
    let term = mul(var(0), factorial(var(0)));
    let result = term.sum_definite(0, &c(1.0), &c(5.0));
    let closed = closed_of(&result);
    let value = closed.eval(&[0.0]);
    assert!((value - 719.0).abs() < 1e-6, "got {value}");
}

// ── Telescoping ──────────────────────────────────────────────────────────────

#[test]
fn sum_reciprocal_telescopes() {
    // Σ_{k=1}^{n} 1/(k(k+1)) = n/(n+1).
    let denom = mul(var(0), add(var(0), c(1.0)));
    let term = div(c(1.0), denom);

    // Indefinite antidifference S(n) = -1/(n+1); verify the property.
    let indef = term.sum_indefinite(0);
    assert!(indef.is_closed(), "1/(k(k+1)) must telescope");
    assert_property_holds(&term, closed_of(&indef));

    // Definite closed form n/(n+1).
    let result = term.sum_definite(0, &c(1.0), &var(0));
    let closed = closed_of(&result);
    for n in 1..=40i64 {
        let expected = n as f64 / (n as f64 + 1.0);
        assert!((closed.eval(&[n as f64]) - expected).abs() < 1e-9, "n={n}");
    }
}

// ── Honest classification ────────────────────────────────────────────────────

#[test]
fn sum_reciprocal_k_is_not_closed_form() {
    // Σ 1/k is the harmonic number: hypergeometric ratio k/(k+1) but no
    // hypergeometric closed form.
    let term = div(c(1.0), var(0));

    let indef = term.sum_indefinite(0);
    assert!(
        matches!(indef, SumResult::NotClosedForm),
        "Σ 1/k must be NotClosedForm, got {indef:?}"
    );

    let def = term.sum_definite(0, &c(1.0), &var(0));
    assert!(
        matches!(def, SumResult::NotClosedForm),
        "definite Σ 1/k must be NotClosedForm, got {def:?}"
    );
}

#[test]
fn sum_reciprocal_k_via_negative_power_is_not_closed_form() {
    // Same term written as k^{-1}.
    let term = powi(var(0), -1.0);
    let indef = term.sum_indefinite(0);
    assert!(matches!(indef, SumResult::NotClosedForm), "{indef:?}");
}

#[test]
fn sum_log_is_not_hypergeometric() {
    // ln(k) is not a hypergeometric term.
    let term = LoweredOp::Ln(Arc::new(var(0)));
    let indef = term.sum_indefinite(0);
    assert!(
        matches!(indef, SumResult::NotHypergeometric),
        "Σ ln(k) must be NotHypergeometric, got {indef:?}"
    );
}

#[test]
fn sum_k_factorial_alone_has_no_closed_form() {
    // Σ k! is not Gosper-summable (ratio is k+1; the key equation has no
    // polynomial solution).
    let term = factorial(var(0));
    let indef = term.sum_indefinite(0);
    assert!(
        matches!(indef, SumResult::NotClosedForm),
        "Σ k! must be NotClosedForm, got {indef:?}"
    );
}

// ── Mixed / general hypergeometric ───────────────────────────────────────────

#[test]
fn sum_k_times_two_pow_k_gosper() {
    // Σ_{k=1}^{n} k·2^k = (n-1)·2^{n+1} + 2. Ratio 2(k+1)/k is rational over ℚ.
    let term = mul(var(0), base_pow(2.0, var(0)));
    let indef = term.sum_indefinite(0);
    assert!(indef.is_closed(), "k·2^k must be Gosper-summable");
    assert_property_holds(&term, closed_of(&indef));

    let result = term.sum_definite(0, &c(1.0), &var(0));
    let closed = closed_of(&result);
    for n in 1..=15i64 {
        let expected: f64 = (1..=n).map(|k| k as f64 * 2f64.powi(k as i32)).sum();
        let got = closed.eval(&[n as f64]);
        assert!(
            (got - expected).abs() <= 1e-6 * expected.abs().max(1.0),
            "n={n}: got {got}, expected {expected}"
        );
    }
}
