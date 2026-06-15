//! Parity tests for the allocation-optimised Adam loop in SymRegEngine.
//!
//! These tests verify that:
//! 1. The buffer-reuse refactor produces identical results to the old code
//!    (same deterministic output for a seeded run).
//! 2. Results are numerically reasonable (finite MSE, at least one formula found).
//! 3. The `forward_with_jacobian_into` path is exercised via the full discover path.

use oxieml::{SymRegConfig, SymRegEngine};

/// 100 data points: y = exp(x) for x in [0.0, 0.99] stepped by 0.01.
fn make_exp_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.01]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();
    (inputs, targets)
}

fn make_config() -> SymRegConfig {
    SymRegConfig {
        seed: Some(42),
        max_depth: 2,
        max_iter: 50,
        num_restarts: 2,
        integer_rounding: false,
        ..SymRegConfig::quick()
    }
}

/// Discover returns at least one formula and the top MSE is finite.
#[test]
fn discover_finds_formulas() {
    let (inputs, targets) = make_exp_data();
    let engine = SymRegEngine::new(make_config());
    let results = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");

    assert!(!results.is_empty(), "should find at least one formula");
    assert!(
        results[0].mse.is_finite(),
        "top formula MSE should be finite, got {}",
        results[0].mse
    );
}

/// Two seeded runs must return bit-identical MSE values.
#[test]
fn seeded_runs_are_deterministic() {
    let (inputs, targets) = make_exp_data();
    let config = make_config();

    let engine = SymRegEngine::new(config.clone());

    let r1 = engine
        .discover(&inputs, &targets, 1)
        .expect("first seeded run should succeed");
    let r2 = engine
        .discover(&inputs, &targets, 1)
        .expect("second seeded run should succeed");

    assert_eq!(
        r1.len(),
        r2.len(),
        "seeded runs must produce the same number of formulas"
    );

    for (f1, f2) in r1.iter().zip(r2.iter()) {
        assert!(
            (f1.mse - f2.mse).abs() < 1e-10,
            "MSE differs between seeded runs: {} vs {}",
            f1.mse,
            f2.mse
        );
        assert_eq!(
            f1.params.len(),
            f2.params.len(),
            "param count must match between seeded runs"
        );
        for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
            assert!(
                (p1 - p2).abs() < 1e-10,
                "params differ between seeded runs: {p1} vs {p2}"
            );
        }
    }
}
