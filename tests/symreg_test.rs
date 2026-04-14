//! Tests for symbolic regression.

use oxieml::symreg::{SymRegConfig, SymRegEngine};

#[test]
fn test_discover_exp() {
    // y = exp(x)
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 2000,
        complexity_penalty: 1e-4,
        num_restarts: 3,
        integer_rounding: true,
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();
    assert!(!formulas.is_empty());

    // eml(x, 1) = exp(x) should be discoverable at depth 1
    let best = &formulas[0];
    assert!(best.mse < 0.1, "MSE too high: {}", best.mse);
}

#[test]
fn test_discover_constant() {
    // y = e (constant function)
    let inputs: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
    let targets: Vec<f64> = vec![std::f64::consts::E; 10];

    let config = SymRegConfig {
        max_depth: 1,
        learning_rate: 1e-2,
        tolerance: 1e-8,
        max_iter: 5000,
        complexity_penalty: 1e-4,
        num_restarts: 3,
        integer_rounding: true,
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();
    assert!(!formulas.is_empty());

    // eml(1, 1) = e should be found
    let best = &formulas[0];
    assert!(best.mse < 0.01, "MSE too high: {}", best.mse);
}

#[test]
fn test_discover_linear() {
    // y = x (identity function)
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();

    let config = SymRegConfig {
        max_depth: 2,
        learning_rate: 1e-2,
        tolerance: 1e-6,
        max_iter: 5000,
        complexity_penalty: 1e-4,
        num_restarts: 3,
        integer_rounding: true,
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();
    assert!(!formulas.is_empty());
}

#[test]
fn test_formulas_sorted_by_score() {
    let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

    let config = SymRegConfig {
        max_depth: 1,
        max_iter: 500,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1).unwrap();

    for window in formulas.windows(2) {
        assert!(window[0].score <= window[1].score);
    }
}
