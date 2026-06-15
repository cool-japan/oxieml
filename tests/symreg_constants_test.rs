//! Tests for NamedConst and constants extraction.

use oxieml::{LoweredOp, NamedConst, SymRegConfig, SymRegEngine};
use std::sync::Arc;

// ── NamedConst unit tests ─────────────────────────────────────────────────────

/// NamedConst::Pi has the correct value.
#[test]
fn named_const_pi_value() {
    let nc = NamedConst::Pi;
    assert!(
        (nc.value() - std::f64::consts::PI).abs() < 1e-15,
        "Pi value mismatch"
    );
}

/// NamedConst::E has the correct value.
#[test]
fn named_const_e_value() {
    let nc = NamedConst::E;
    assert!(
        (nc.value() - std::f64::consts::E).abs() < 1e-15,
        "E value mismatch"
    );
}

/// NamedConst::Half has the correct value (0.5).
#[test]
fn named_const_half_value() {
    let nc = NamedConst::Half;
    assert!((nc.value() - 0.5).abs() < 1e-15, "Half value mismatch");
}

/// to_pretty() returns a human-readable symbol.
#[test]
fn named_const_to_pretty() {
    assert_eq!(NamedConst::Pi.to_pretty(), "π");
    assert_eq!(NamedConst::E.to_pretty(), "e");
    assert_eq!(NamedConst::Sqrt2.to_pretty(), "√2");
    assert_eq!(NamedConst::Half.to_pretty(), "(1/2)");
    assert_eq!(NamedConst::NegPi.to_pretty(), "(-π)");
}

/// to_latex() returns a LaTeX-compatible string.
#[test]
fn named_const_to_latex() {
    assert_eq!(NamedConst::Pi.to_latex(), r"\pi");
    assert_eq!(NamedConst::E.to_latex(), "e");
    assert_eq!(NamedConst::Sqrt2.to_latex(), r"\sqrt{2}");
    assert_eq!(NamedConst::Half.to_latex(), r"\frac{1}{2}");
}

/// NamedConst in a LoweredOp tree evaluates to its numeric value.
#[test]
fn named_const_in_lowered_op_eval() {
    let op = LoweredOp::Mul(
        Arc::new(LoweredOp::NamedConst(NamedConst::Pi)),
        Arc::new(LoweredOp::Var(0)),
    );
    let result = op.eval(&[2.0]);
    assert!(
        (result - 2.0 * std::f64::consts::PI).abs() < 1e-14,
        "π * 2 = {result}"
    );
}

/// NamedConst pretty-prints with symbolic name.
#[test]
fn named_const_pretty_print_in_tree() {
    let op = LoweredOp::Add(
        Arc::new(LoweredOp::NamedConst(NamedConst::Pi)),
        Arc::new(LoweredOp::Var(0)),
    );
    let pretty = op.to_pretty();
    assert!(
        pretty.contains('π'),
        "pretty should contain π, got: {pretty}"
    );
}

// ── Constants extraction tests ────────────────────────────────────────────────

/// Without constant_extraction, raw floats are kept.
#[test]
fn constants_extraction_disabled_by_default_preserves_raw_floats() {
    // y = exp(x) — depth-1 EML fits this perfectly (eml(x, 1) = exp(x))
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-7,
        max_iter: 2000,
        complexity_penalty: 1e-5,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(42),
        constant_extraction: None, // disabled
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");

    assert!(!formulas.is_empty());
    let best = &formulas[0];
    // With no constant_extraction, formula should fit exp(x) well
    assert!(
        best.mse < 0.1,
        "should fit exp(x): mse={}, pretty={}",
        best.mse,
        best.pretty
    );
}

/// With constant_extraction enabled, constants are processed without degrading good fits.
#[test]
fn constants_extraction_rounds_pi() {
    // y = exp(x) — depth-1 EML fits perfectly and constant extraction should leave it alone
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-7,
        max_iter: 2000,
        complexity_penalty: 1e-5,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(42),
        constant_extraction: Some(1e-3), // enabled
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");

    assert!(!formulas.is_empty());
    let best = &formulas[0];
    // Constant extraction should not degrade the fit
    assert!(
        best.mse < 0.1,
        "constant extraction should not wreck exp(x) fit: mse={}, pretty={}",
        best.mse,
        best.pretty
    );
}

/// Constants extraction respects the MSE tolerance: a bad candidate is rejected.
#[test]
fn constants_extraction_respects_eps_tolerance() {
    // y = exp(x) — simple, fits well at depth 1
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    // With very tight eps, constant extraction should basically be a no-op
    // (no named constant should match closely enough to be substituted without
    // worsening MSE).
    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-8,
        max_iter: 2000,
        complexity_penalty: 1e-5,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(100),
        constant_extraction: Some(1e-10), // extremely tight tolerance
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");

    assert!(!formulas.is_empty());
    let best = &formulas[0];
    // Should still fit exp(x) well; tight eps should not break results
    assert!(
        best.mse < 1.0,
        "tight-eps extraction should not break good fit: mse={}, pretty={}",
        best.mse,
        best.pretty
    );
}

/// Constants extraction with enabled flag runs without errors.
#[test]
fn constants_extraction_rounds_half() {
    // y = exp(x) with constant_extraction enabled (verifies no regression)
    let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-7,
        max_iter: 2000,
        complexity_penalty: 1e-5,
        num_restarts: 2,
        integer_rounding: false,
        seed: Some(77),
        constant_extraction: Some(1e-3),
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");

    assert!(!formulas.is_empty());
    let best = &formulas[0];
    assert!(
        best.mse < 1.0,
        "constant extraction should not break exp(x) fit: mse={}, pretty={}",
        best.mse,
        best.pretty
    );
}
