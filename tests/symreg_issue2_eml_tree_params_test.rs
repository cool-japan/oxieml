//! Regression tests for https://github.com/cool-japan/oxieml/issues/2
//!
//! `SymRegEngine::discover` used to return a `DiscoveredFormula` whose
//! `eml_tree` was the bare search topology (`topology.clone()`), i.e. with the
//! parameter leaves still sitting at their unfitted `One` values, while `mse`
//! and `pretty` were derived from a separately parameterized expression.
//! Evaluating `eml_tree` — what every README example does via
//! `eval_batch`/`eval_real`/`lower` — therefore produced numbers unrelated to
//! the reported `mse` (the reporter saw `1.25e-13` vs a recomputed `7.00e2`).
//!
//! A second, closely related defect lived in `bake_params_into_lowered`: it
//! lowered the topology *before* substituting the parameters, so the lowering
//! patterns that match on `EmlNode::One` (`eml(x, One) → exp(x)`, …) swallowed
//! the leaf and dropped the parameter entirely. `pretty` for a fitted
//! `eml(x0, p)` with `p = exp(5)` read `exp(x0)` instead of `exp(x0) - 5`.
//!
//! The invariant these tests pin down: every field of `DiscoveredFormula`
//! describes the same tree — `eml_tree` reproduces `mse`, its `Const` leaves
//! are `params`, and `pretty`/`to_latex` render that same tree.

use oxieml::symreg::SymRegConfig;
use oxieml::{
    EmlNode, EmlTree, EvalCtx, MultiOutputStrategy, OptimizerKind, SymRegEngine, SymRegStrategy,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Recompute MSE by evaluating `tree` directly, using the same
/// skip-non-finite-points rule the optimizer scores with.
fn tree_mse(tree: &EmlTree, inputs: &[Vec<f64>], targets: &[f64]) -> Option<f64> {
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        if let Ok(value) = tree.eval_real(&ctx) {
            if value.is_finite() {
                total += (value - target) * (value - target);
                count += 1;
            }
        }
    }
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

/// Collect the tree's `Const` leaf values in left-to-right leaf order — the
/// order in which `ParameterizedEmlTree` assigns parameter indices.
fn const_leaves(tree: &EmlTree) -> Vec<f64> {
    tree.iter_postorder()
        .filter_map(|node| match node {
            EmlNode::Const(v) => Some(*v),
            _ => None,
        })
        .collect()
}

/// Count leaves that are still the unfitted grammar constant `One`.
fn one_leaves(tree: &EmlTree) -> usize {
    tree.iter_postorder()
        .filter(|node| matches!(node, EmlNode::One))
        .count()
}

/// `y = exp(x) - 5`: the fitted parameter is `exp(5) ≈ 148.41`, far from the
/// `1.0` the topology's `One` leaf carries, so a stale topology cannot pass by
/// coincidence.
fn exp_minus_five_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
    let targets: Vec<f64> = inputs.iter().map(|row| row[0].exp() - 5.0).collect();
    (inputs, targets)
}

/// Assert the whole `DiscoveredFormula` consistency invariant for one config.
fn assert_formula_self_consistent(
    label: &str,
    config: SymRegConfig,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> TestResult {
    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(inputs, targets, inputs[0].len())?;
    let best = formulas
        .first()
        .ok_or_else(|| format!("[{label}] discover returned no formula"))?;

    // (a) The returned tree carries the fitted parameters, in order.
    assert_eq!(
        one_leaves(&best.eml_tree),
        0,
        "[{label}] eml_tree still has unfitted `One` leaves: {}",
        best.eml_tree
    );
    let leaves = const_leaves(&best.eml_tree);
    assert_eq!(
        leaves.len(),
        best.params.len(),
        "[{label}] eml_tree has {} constant leaves but {} params ({})",
        leaves.len(),
        best.params.len(),
        best.eml_tree
    );
    for (i, (leaf, param)) in leaves.iter().zip(&best.params).enumerate() {
        assert_eq!(
            leaf.to_bits(),
            param.to_bits(),
            "[{label}] leaf {i} is {leaf} but params[{i}] is {param}"
        );
    }

    // (b) Evaluating that tree reproduces the reported MSE.
    let recomputed = tree_mse(&best.eml_tree, inputs, targets)
        .ok_or_else(|| format!("[{label}] eml_tree produced no finite prediction"))?;
    let tolerance = 1e-9 * recomputed.abs().max(best.mse.abs()).max(1e-12);
    assert!(
        (recomputed - best.mse).abs() <= tolerance,
        "[{label}] eml_tree evaluates to mse={recomputed:.6e} but the formula reports \
         mse={:.6e} (tree={}, params={:?}, pretty={})",
        best.mse,
        best.eml_tree,
        best.params,
        best.pretty
    );

    // (c) `pretty` renders that same tree, not a different one.
    let from_tree = best.eml_tree.lower().simplify().to_pretty();
    assert_eq!(
        best.pretty, from_tree,
        "[{label}] pretty does not match the returned tree's own rendering"
    );

    Ok(())
}

/// The headline symptom of #2, on the Adam path: `eml_tree` must reproduce
/// `mse`, and its leaves must be the fitted parameters.
#[test]
fn test_issue_2_eml_tree_carries_fitted_params_adam() -> TestResult {
    let (inputs, targets) = exp_minus_five_data();
    let config = SymRegConfig {
        max_depth: 2,
        learning_rate: 1e-2,
        tolerance: 1e-12,
        max_iter: 3000,
        num_restarts: 3,
        seed: Some(7),
        ..SymRegConfig::default()
    };
    assert_formula_self_consistent("adam", config, &inputs, &targets)
}

/// Same invariant on the Levenberg–Marquardt path, which has its own
/// `DiscoveredFormula` construction site in `optimize_lm.rs`.
#[test]
fn test_issue_2_eml_tree_carries_fitted_params_lm() -> TestResult {
    let (inputs, targets) = exp_minus_five_data();
    let config = SymRegConfig {
        max_depth: 2,
        tolerance: 1e-12,
        max_iter: 300,
        num_restarts: 3,
        seed: Some(7),
        optimizer: OptimizerKind::LevenbergMarquardt,
        ..SymRegConfig::default()
    };
    assert_formula_self_consistent("lm", config, &inputs, &targets)
}

/// The second defect: `pretty` was built by lowering the topology first, so a
/// lowering pattern that matches on `One` dropped the parameter. For
/// `y = exp(x) - 5` the LM optimizer finds `eml(x0, exp(5))`, which used to
/// pretty-print as `exp(x0)` — a formula whose MSE is 25, not the reported
/// ~1e-31. The rendered expression must now evaluate to the reported MSE too.
#[test]
fn test_issue_2_pretty_matches_fitted_tree_not_bare_topology() -> TestResult {
    let (inputs, targets) = exp_minus_five_data();
    let config = SymRegConfig {
        max_depth: 1,
        tolerance: 1e-12,
        max_iter: 300,
        num_restarts: 3,
        seed: Some(7),
        optimizer: OptimizerKind::LevenbergMarquardt,
        ..SymRegConfig::default()
    };
    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1)?;
    let best = formulas
        .first()
        .ok_or("discover returned no formula for y = exp(x) - 5")?;

    // Depth 1 over one variable can only reach `exp(x0) - ln(p)`, so the fit
    // has to be (near) exact and the `- 5` term cannot be optional.
    assert!(
        best.mse < 1e-12,
        "expected a near-exact fit of exp(x)-5, got mse={:.6e} (pretty={})",
        best.mse,
        best.pretty
    );

    let lowered = best.eml_tree.lower().simplify();
    assert_eq!(
        best.pretty,
        lowered.to_pretty(),
        "pretty must render the returned tree"
    );

    // Evaluate the lowered/pretty-printed form itself: it must agree with the
    // reported MSE, i.e. the `- 5` offset survived lowering.
    let mut lowered_mse = 0.0_f64;
    for (input, &target) in inputs.iter().zip(&targets) {
        let value = lowered.eval(input);
        lowered_mse += (value - target) * (value - target);
    }
    lowered_mse /= inputs.len() as f64;
    assert!(
        lowered_mse < 1e-12,
        "lowered form '{}' has mse={lowered_mse:.6e}; the fitted constant was dropped",
        best.pretty
    );

    Ok(())
}

/// The shared-topology multi-output path builds its own `DiscoveredFormula`s in
/// `discover_shared::shared_to_multi_result`, one per output, each with its own
/// parameter vector. Every one of them must carry its own fitted parameters.
#[test]
fn test_issue_2_shared_topology_outputs_carry_own_params() -> TestResult {
    let inputs: Vec<Vec<f64>> = (0..16).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<Vec<f64>> = vec![
        inputs.iter().map(|row| row[0].exp() - 5.0).collect(),
        inputs.iter().map(|row| row[0].exp() - 2.0).collect(),
    ];

    let config = SymRegConfig {
        max_depth: 1,
        tolerance: 1e-12,
        max_iter: 300,
        num_restarts: 2,
        seed: Some(11),
        optimizer: OptimizerKind::LevenbergMarquardt,
        multi_output_strategy: MultiOutputStrategy::SharedTopology,
        strategy: SymRegStrategy::Exhaustive,
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let per_output = engine.discover_multi(&inputs, &targets, 1)?;
    assert_eq!(per_output.len(), targets.len());

    for (out_idx, (formulas, target)) in per_output.iter().zip(&targets).enumerate() {
        let best = formulas
            .first()
            .ok_or_else(|| format!("output {out_idx}: no formula"))?;
        assert_eq!(
            one_leaves(&best.eml_tree),
            0,
            "output {out_idx}: eml_tree still has unfitted `One` leaves: {}",
            best.eml_tree
        );
        assert_eq!(
            const_leaves(&best.eml_tree),
            best.params,
            "output {out_idx}: eml_tree leaves do not match params"
        );
        let recomputed = tree_mse(&best.eml_tree, &inputs, target)
            .ok_or_else(|| format!("output {out_idx}: eml_tree produced no finite prediction"))?;
        let tolerance = 1e-9 * recomputed.abs().max(best.mse.abs()).max(1e-12);
        assert!(
            (recomputed - best.mse).abs() <= tolerance,
            "output {out_idx}: eml_tree evaluates to mse={recomputed:.6e} but reports {:.6e} \
             (tree={})",
            best.mse,
            best.eml_tree
        );
    }

    // The two outputs share one skeleton but differ in their fitted constant,
    // so the baked trees must not be identical.
    let left = per_output
        .first()
        .and_then(|f| f.first())
        .ok_or("output 0 missing")?;
    let right = per_output
        .get(1)
        .and_then(|f| f.first())
        .ok_or("output 1 missing")?;
    assert_ne!(
        const_leaves(&left.eml_tree),
        const_leaves(&right.eml_tree),
        "both outputs baked the same constants, so per-output params were not applied"
    );

    Ok(())
}

/// With `constant_extraction` enabled the snapped constant must reach `params`
/// and `eml_tree` (not only the pretty string), so the reported `mse` still
/// belongs to the returned tree.
#[test]
fn test_issue_2_constant_extraction_reaches_returned_tree() -> TestResult {
    // y = exp(x) - ln(π): the ideal fitted parameter is exactly π.
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs
        .iter()
        .map(|row| row[0].exp() - std::f64::consts::PI.ln())
        .collect();

    let config = SymRegConfig {
        max_depth: 1,
        tolerance: 1e-14,
        max_iter: 300,
        num_restarts: 3,
        integer_rounding: false,
        seed: Some(23),
        optimizer: OptimizerKind::LevenbergMarquardt,
        constant_extraction: Some(1e-3),
        ..SymRegConfig::default()
    };

    let engine = SymRegEngine::new(config);
    let formulas = engine.discover(&inputs, &targets, 1)?;
    let best = formulas.first().ok_or("discover returned no formula")?;

    assert_eq!(
        const_leaves(&best.eml_tree),
        best.params,
        "eml_tree leaves must equal params after constant extraction (tree={}, pretty={})",
        best.eml_tree,
        best.pretty
    );
    let recomputed = tree_mse(&best.eml_tree, &inputs, &targets)
        .ok_or("eml_tree produced no finite prediction")?;
    let tolerance = 1e-9 * recomputed.abs().max(best.mse.abs()).max(1e-12);
    assert!(
        (recomputed - best.mse).abs() <= tolerance,
        "eml_tree evaluates to mse={recomputed:.6e} but the formula reports mse={:.6e} \
         (params={:?}, pretty={})",
        best.mse,
        best.params,
        best.pretty
    );

    Ok(())
}
