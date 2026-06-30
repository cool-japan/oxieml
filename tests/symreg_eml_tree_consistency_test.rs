//! Regression test for `DiscoveredFormula.eml_tree` being consistent with
//! `mse`/`pretty`/`params`: evaluating `eml_tree` directly on the training
//! data must reproduce (approximately) the reported `mse`.
//!
//! Before the fix, `eml_tree` was the *unparameterized* topology (the
//! optimizer's fitted `params` were never substituted into it), so
//! `eml_tree.eval_batch(...)` silently returned values unrelated to the
//! formula described by `pretty`/`mse`.

use oxieml::symreg::{SymRegConfig, SymRegEngine};
use oxieml::{EvalCtx, OptimizerKind};

fn assert_eml_tree_matches_reported_mse(inputs: &[Vec<f64>], targets: &[f64], config: SymRegConfig) {
    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(inputs, targets, 1).unwrap();
    assert!(!formulas.is_empty());

    let best = &formulas[0];

    // Evaluate the returned `eml_tree` directly, the same way every README
    // example does, and recompute MSE from scratch.
    let mut manual_mse = 0.0;
    let mut n = 0usize;
    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        if let Ok(pred) = best.eml_tree.eval_real(&ctx) {
            if pred.is_finite() {
                manual_mse += (pred - target).powi(2);
                n += 1;
            }
        }
    }
    assert!(n > 0, "eml_tree produced no finite predictions at all");
    manual_mse /= n as f64;

    assert!(
        (manual_mse - best.mse).abs() < 1e-6 * manual_mse.max(1.0),
        "eml_tree's own evaluation (mse={manual_mse:.6e} over {n}/{} points) \
         does not match the formula's reported mse ({:.6e}); pretty = {}, params = {:?}",
        inputs.len(),
        best.mse,
        best.pretty,
        best.params,
    );
}

#[test]
fn eml_tree_matches_reported_mse_adam() {
    // y = exp(x) - 5. Fitting requires eml(x, c) = exp(x) - ln(c) with
    // c = exp(5) =~ 148.4, far from the topology's default "One" leaf value
    // of 1.0 — so an unparameterized `eml_tree` (the pre-fix bug) would be
    // off by a constant 5 everywhere, not accidentally close to correct.
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp() - 5.0).collect();

    let config = SymRegConfig {
        max_depth: 2,
        learning_rate: 1e-2,
        tolerance: 1e-9,
        max_iter: 2000,
        num_restarts: 3,
        ..SymRegConfig::default()
    };
    assert_eml_tree_matches_reported_mse(&inputs, &targets, config);
}

#[test]
fn eml_tree_matches_reported_mse_levenberg_marquardt() {
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp() - 5.0).collect();

    let config = SymRegConfig {
        max_depth: 2,
        tolerance: 1e-9,
        max_iter: 200,
        num_restarts: 3,
        optimizer: OptimizerKind::LevenbergMarquardt,
        ..SymRegConfig::default()
    };
    assert_eml_tree_matches_reported_mse(&inputs, &targets, config);
}
