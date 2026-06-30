//! Shared-topology multi-output symbolic regression.
//!
//! All output dimensions share a single EML tree skeleton; each output has its
//! own fitted parameter vector. The parsimony benefit: topology complexity is
//! charged once regardless of the number of outputs.

use crate::tree::EmlTree;

use super::constants::{bake_params_into_lowered, bake_params_into_tree};
use super::topology::{dedupe_by_semantics, enumerate_topologies_gated};
use super::{DiscoveredFormula, SharedFormula, SymRegEngine};

/// Run shared-topology symbolic regression across multiple outputs.
///
/// Enumerates candidate topologies using the same search as the `Exhaustive`
/// strategy.  For each topology, all `n_outputs` outputs are fitted
/// independently (each gets its own parameter vector and optimizer call).
/// The total score charges the topology complexity exactly once:
///
/// ```text
/// total_score = Σ_k mse_k + complexity_penalty × node_count(topology)
/// ```
///
/// Returns the top-K results sorted ascending by `total_score`.
///
/// # Arguments
///
/// * `engine` — configured `SymRegEngine` (settings are read from its config).
/// * `inputs` — `n_samples × n_vars` row-major feature matrix.
/// * `targets_multi` — one `Vec<f64>` per output, each of length `n_samples`.
/// * `num_vars` — number of input variables.
///
/// # Returns
///
/// `Vec<SharedFormula>` sorted ascending by `total_score`, truncated to
/// `top_k = num_restarts * 5` (or at most the number of topologies tried).
pub(super) fn run_shared_topology(
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets_multi: &[Vec<f64>],
    num_vars: usize,
) -> Vec<SharedFormula> {
    let n_outputs = targets_multi.len();
    if n_outputs == 0 || inputs.is_empty() {
        return Vec::new();
    }

    let config = &engine.config;

    // Enumerate and deduplicate topologies — mirrors discover_exhaustive.
    let const_leaf = if config.enable_const_leaf {
        Some(config.const_leaf_init)
    } else {
        None
    };
    let raw_topologies = enumerate_topologies_gated(config.max_depth, num_vars, const_leaf);
    let topologies: Vec<EmlTree> = dedupe_by_semantics(raw_topologies);

    let complexity_penalty = config.complexity_penalty;

    let mut shared_results: Vec<SharedFormula> = topologies
        .iter()
        .enumerate()
        .filter_map(|(topo_idx, tree)| {
            fit_shared_topology(
                engine,
                tree,
                topo_idx,
                inputs,
                targets_multi,
                complexity_penalty,
                n_outputs,
            )
        })
        .collect();

    // Sort ascending by total_score.
    shared_results.sort_by(|a, b| {
        a.total_score
            .partial_cmp(&b.total_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep top-K results.  Use the same heuristic as Independent: at most
    // num_restarts * 5 or the number available, whichever is smaller.
    let top_k = (config.num_restarts * 5).max(1);
    shared_results.truncate(top_k);
    shared_results
}

/// Fit a single topology against all outputs and build a [`SharedFormula`].
///
/// Returns `None` when the topology is infeasible for *every* output (all
/// optimizer calls fail).
fn fit_shared_topology(
    engine: &SymRegEngine,
    tree: &EmlTree,
    topo_idx: usize,
    inputs: &[Vec<f64>],
    targets_multi: &[Vec<f64>],
    complexity_penalty: f64,
    n_outputs: usize,
) -> Option<SharedFormula> {
    let mut per_output_params: Vec<Vec<f64>> = Vec::with_capacity(n_outputs);
    let mut per_output_mse: Vec<f64> = Vec::with_capacity(n_outputs);
    let mut total_mse = 0.0_f64;
    let mut any_feasible = false;

    for (out_idx, target) in targets_multi.iter().enumerate() {
        // Each output gets its own seed derived from the topology index and output index.
        let per_output_topo_idx = topo_idx * n_outputs + out_idx;

        match engine.optimize_topology(tree, inputs, target, per_output_topo_idx) {
            Some(df) => {
                per_output_mse.push(df.mse);
                total_mse += df.mse;
                per_output_params.push(df.params);
                any_feasible = true;
            }
            None => {
                // Topology infeasible for this output — mark with infinity.
                per_output_mse.push(f64::INFINITY);
                total_mse = f64::INFINITY;
                per_output_params.push(Vec::new());
            }
        }
    }

    // Discard entirely-infeasible topologies.
    if !any_feasible || !total_mse.is_finite() {
        return None;
    }

    let node_count = tree.size() as f64;
    let total_score = total_mse + complexity_penalty * node_count;

    // Build pretty-printed formula per output (params substituted in).
    let pretty_per_output: Vec<String> = per_output_params
        .iter()
        .map(|params| {
            if params.is_empty() {
                "Infeasible".to_string()
            } else {
                bake_params_into_lowered(tree, params)
                    .simplify()
                    .to_pretty()
            }
        })
        .collect();

    Some(SharedFormula {
        eml_tree: tree.clone(),
        per_output_params,
        per_output_mse,
        total_score,
        pretty_per_output,
    })
}

/// Convert a `Vec<SharedFormula>` into the `Vec<Vec<DiscoveredFormula>>` shape
/// that callers of `discover_multi` expect.
///
/// Each `SharedFormula` produces one `DiscoveredFormula` per output dimension.
/// The `n_outputs` slices are organized as: result[output_idx] = list of
/// `DiscoveredFormula` for that output, sorted by per-output MSE.
pub(super) fn shared_to_multi_result(
    shared: Vec<SharedFormula>,
    complexity_penalty: f64,
    n_outputs: usize,
) -> Vec<Vec<DiscoveredFormula>> {
    if n_outputs == 0 || shared.is_empty() {
        return vec![Vec::new(); n_outputs];
    }

    // For each output dimension, collect one DiscoveredFormula per SharedFormula.
    let mut per_output: Vec<Vec<DiscoveredFormula>> = (0..n_outputs)
        .map(|_| Vec::with_capacity(shared.len()))
        .collect();

    for sf in &shared {
        for (out_idx, (params, &mse)) in sf
            .per_output_params
            .iter()
            .zip(sf.per_output_mse.iter())
            .enumerate()
        {
            if !mse.is_finite() || out_idx >= n_outputs {
                continue;
            }
            let complexity = sf.eml_tree.size();
            let score = mse + complexity_penalty * complexity as f64;
            let pretty = sf
                .pretty_per_output
                .get(out_idx)
                .cloned()
                .unwrap_or_default();
            // Inline AIC/BIC: n_data unknown here, use 1 as a unit proxy.
            let n = 1_f64;
            let k = params.len() as f64;
            let rss_per_n = mse.max(f64::MIN_POSITIVE);
            let aic = n * rss_per_n.ln() + 2.0 * k;
            let bic = n * rss_per_n.ln() + k * n.max(1.0_f64).ln();
            let df = DiscoveredFormula {
                eml_tree: bake_params_into_tree(&sf.eml_tree, params),
                mse,
                complexity,
                score,
                pretty,
                params: params.clone(),
                cv_mse: None,
                aic,
                bic,
                param_intervals: None,
            };
            per_output[out_idx].push(df);
        }
    }

    // Sort each output's list by score ascending.
    for output_list in &mut per_output {
        output_list.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    per_output
}
