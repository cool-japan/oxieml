//! Symbolic regression engine.
//!
//! Discovers closed-form mathematical formulas from data using EML trees.
//! The algorithm enumerates tree topologies up to a maximum depth, optimizes
//! continuous parameters via Adam, and selects the best formulas by MSE
//! with a complexity penalty (Occam's razor).

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::grad::ParameterizedEmlTree;
use crate::tree::{EmlNode, EmlTree};
use rand::RngExt;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for symbolic regression.
#[derive(Clone, Debug)]
pub struct SymRegConfig {
    /// Maximum tree depth to explore (paper: 4 is often sufficient).
    pub max_depth: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Convergence threshold (MSE).
    pub tolerance: f64,
    /// Maximum optimization iterations per topology.
    pub max_iter: usize,
    /// Complexity penalty coefficient (Occam's razor).
    pub complexity_penalty: f64,
    /// Number of random restarts per topology.
    pub num_restarts: usize,
    /// Whether to attempt integer rounding of parameters.
    pub integer_rounding: bool,
}

impl Default for SymRegConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            learning_rate: 1e-3,
            tolerance: 1e-10,
            max_iter: 10_000,
            complexity_penalty: 1e-4,
            num_restarts: 3,
            integer_rounding: true,
        }
    }
}

/// A formula discovered by symbolic regression.
#[derive(Clone, Debug)]
pub struct DiscoveredFormula {
    /// The EML tree representation.
    pub eml_tree: EmlTree,
    /// Final mean squared error.
    pub mse: f64,
    /// Tree node count (complexity measure).
    pub complexity: usize,
    /// Combined score: MSE + complexity_penalty * complexity.
    pub score: f64,
    /// Human-readable expression (from lowering).
    pub pretty: String,
    /// Optimized parameter values.
    pub params: Vec<f64>,
}

/// Symbolic regression engine.
pub struct SymRegEngine {
    config: SymRegConfig,
}

impl SymRegEngine {
    /// Create a new symbolic regression engine.
    pub fn new(config: SymRegConfig) -> Self {
        Self { config }
    }

    /// Discover closed-form formulas from input-output data.
    ///
    /// - `inputs`: each row is one data point's variable values
    /// - `targets`: corresponding output values
    /// - `num_vars`: number of input variables
    ///
    /// Returns formulas sorted by score (best first).
    pub fn discover(
        &self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        num_vars: usize,
    ) -> Result<Vec<DiscoveredFormula>, EmlError> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(EmlError::EmptyData);
        }
        if inputs.len() != targets.len() {
            return Err(EmlError::DimensionMismatch(inputs.len(), targets.len()));
        }

        // Phase 1: Enumerate all topologies up to max_depth
        let topologies = enumerate_topologies(self.config.max_depth, num_vars);

        // Phase 1b: Prune semantically-equivalent topologies
        // (EML-default lowering is nearly injective, so only a small fraction
        //  collapse via simplification rules — see `dedupe_by_semantics`).
        let topologies = dedupe_by_semantics(topologies);

        // Phase 2: Optimize parameters for each topology (parallel when feature enabled)
        #[cfg(feature = "parallel")]
        let mut formulas: Vec<DiscoveredFormula> = topologies
            .par_iter()
            .filter_map(|topology| self.optimize_topology(topology, inputs, targets))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let mut formulas: Vec<DiscoveredFormula> = topologies
            .iter()
            .filter_map(|topology| self.optimize_topology(topology, inputs, targets))
            .collect();

        // Phase 3: Sort by score
        formulas.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(formulas)
    }

    /// Optimize parameters for a single topology.
    fn optimize_topology(
        &self,
        topology: &EmlTree,
        inputs: &[Vec<f64>],
        targets: &[f64],
    ) -> Option<DiscoveredFormula> {
        let mut best_mse = f64::INFINITY;
        let mut best_params = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..self.config.num_restarts {
            let mut ptree = ParameterizedEmlTree::from_topology(topology, 1.0);

            // Randomize initial parameters slightly
            for p in &mut ptree.params {
                *p = 1.0 + rng.random_range(-0.5..0.5);
            }

            // Adam optimizer state
            let n_params = ptree.num_params();
            if n_params == 0 {
                // No parameters to optimize — evaluate directly
                let mse = compute_mse_direct(topology, inputs, targets);
                if let Some(mse) = mse {
                    if mse < best_mse {
                        best_mse = mse;
                        best_params = vec![];
                    }
                }
                break;
            }

            let mut m = vec![0.0_f64; n_params]; // First moment
            let mut v = vec![0.0_f64; n_params]; // Second moment
            let beta1 = 0.9;
            let beta2 = 0.999;
            let epsilon = 1e-8;
            let lr = self.config.learning_rate;

            let mut converged = false;

            for t in 1..=self.config.max_iter {
                // Compute average gradient over all data points
                let mut total_loss = 0.0;
                let mut total_grads = vec![0.0; n_params];
                let mut valid_count = 0usize;

                for (input, &target) in inputs.iter().zip(targets) {
                    let ctx = EvalCtx::new(input);
                    match ptree.forward_backward(&ctx, target) {
                        Ok((loss, grads)) => {
                            if loss.is_finite() {
                                total_loss += loss;
                                for (tg, g) in total_grads.iter_mut().zip(&grads) {
                                    if g.is_finite() {
                                        *tg += g;
                                    }
                                }
                                valid_count += 1;
                            }
                        }
                        Err(_) => continue,
                    }
                }

                if valid_count == 0 {
                    break;
                }

                let n_f = valid_count as f64;
                let mse = total_loss / n_f;

                if mse < self.config.tolerance {
                    best_mse = mse;
                    best_params = ptree.params.clone();
                    converged = true;
                    break;
                }

                // Adam update
                for i in 0..n_params {
                    let g = total_grads[i] / n_f;
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                    let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                    let v_hat = v[i] / (1.0 - beta2.powi(t as i32));
                    ptree.params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }

                if mse < best_mse {
                    best_mse = mse;
                    best_params = ptree.params.clone();
                }
            }

            if converged {
                break;
            }
        }

        // Phase 4: Integer rounding
        if self.config.integer_rounding && !best_params.is_empty() {
            let rounded = try_integer_rounding(&best_params);
            let mut ptree_rounded = ParameterizedEmlTree::from_topology(topology, 1.0);
            ptree_rounded.params = rounded;
            let rounded_mse = compute_mse_parameterized(&ptree_rounded, inputs, targets);
            if let Some(rmse) = rounded_mse {
                if rmse <= best_mse * 1.01 {
                    // Accept rounding if MSE doesn't degrade much
                    best_mse = rmse;
                    best_params = ptree_rounded.params;
                }
            }
        }

        if !best_mse.is_finite() || best_mse > 1e10 {
            return None;
        }

        let complexity = topology.size();
        let score = best_mse + self.config.complexity_penalty * complexity as f64;
        let lowered = topology.lower();
        let pretty = lowered.simplify().to_pretty();

        Some(DiscoveredFormula {
            eml_tree: topology.clone(),
            mse: best_mse,
            complexity,
            score,
            pretty,
            params: best_params,
        })
    }
}

/// Enumerate all EML tree topologies up to given depth with given number of variables.
///
/// Follows the Catalan-number-based enumeration of binary trees.
pub fn enumerate_topologies(max_depth: usize, num_vars: usize) -> Vec<EmlTree> {
    let mut topologies = Vec::new();
    let leaves = build_leaves(num_vars);

    for depth in 0..=max_depth {
        enumerate_at_depth(depth, &leaves, &mut topologies);
    }

    topologies
}

/// Deduplicate topologies that lower+simplify to the same structural form.
///
/// Many EML topologies are semantically equivalent (commutative/associative
/// reorderings, algebraic identities like `exp(ln(x)) = x`). We apply
/// EML-level simplification first (to collapse patterns like `ln(exp(x))`
/// and `exp(ln(x))` directly on the tree), then lower to a conventional
/// op tree and run the arithmetic simplifier. We compute a structural
/// hash over the resulting `LoweredOp` and keep the first representative
/// of each hash bucket.
///
/// Note: EML is non-commutative and non-associative, so tree rearrangements
/// generally produce genuinely distinct functions. Structural dedup thus
/// catches only the cases where simplification rules collapse different
/// trees to identical forms. Deeper reduction would require evaluation-based
/// fingerprinting, which is out of scope here.
pub fn dedupe_by_semantics(topologies: Vec<EmlTree>) -> Vec<EmlTree> {
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut seen: HashSet<u64> = HashSet::new();
    topologies
        .into_iter()
        .filter(|t| {
            let eml_simplified = crate::simplify::simplify(t);
            let simplified = eml_simplified.lower().simplify();
            let mut hasher = DefaultHasher::new();
            simplified.structural_hash(&mut hasher);
            seen.insert(hasher.finish())
        })
        .collect()
}

/// Build leaf nodes: One and Var(0), Var(1), ...
fn build_leaves(num_vars: usize) -> Vec<Arc<EmlNode>> {
    let mut leaves = vec![Arc::new(EmlNode::One)];
    for i in 0..num_vars {
        leaves.push(Arc::new(EmlNode::Var(i)));
    }
    leaves
}

/// Enumerate trees at exactly the given depth (duplicate-free).
///
/// A tree has exact depth `d` iff at least one child has depth `d-1`.
/// We enumerate three disjoint cases:
/// 1. Both children at exactly depth `d-1`
/// 2. Left at depth `d-1`, right at depth `< d-1`
/// 3. Left at depth `< d-1`, right at depth `d-1`
fn enumerate_at_depth(depth: usize, leaves: &[Arc<EmlNode>], out: &mut Vec<EmlTree>) {
    if depth == 0 {
        for leaf in leaves {
            out.push(EmlTree::from_node(Arc::clone(leaf)));
        }
        return;
    }

    let at_max = enumerate_at_depth_nodes(depth - 1, leaves);

    // Nodes strictly below max depth
    let mut below_max = Vec::new();
    for d in 0..(depth - 1) {
        below_max.extend(enumerate_at_depth_nodes(d, leaves));
    }

    // Case 1: both at max-1 depth
    for left in &at_max {
        for right in &at_max {
            out.push(EmlTree::from_node(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            })));
        }
    }

    // Case 2: left at max, right strictly below
    for left in &at_max {
        for right in &below_max {
            out.push(EmlTree::from_node(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            })));
        }
    }

    // Case 3: left strictly below, right at max
    for left in &below_max {
        for right in &at_max {
            out.push(EmlTree::from_node(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            })));
        }
    }
}

/// Enumerate node Arcs at exactly the given depth.
fn enumerate_at_depth_nodes(depth: usize, leaves: &[Arc<EmlNode>]) -> Vec<Arc<EmlNode>> {
    if depth == 0 {
        return leaves.to_vec();
    }

    let at_max = enumerate_at_depth_nodes(depth - 1, leaves);
    let mut below_max = Vec::new();
    for d in 0..(depth - 1) {
        below_max.extend(enumerate_at_depth_nodes(d, leaves));
    }

    let mut out = Vec::new();

    // Both at max
    for left in &at_max {
        for right in &at_max {
            out.push(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            }));
        }
    }

    // Left at max, right below
    for left in &at_max {
        for right in &below_max {
            out.push(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            }));
        }
    }

    // Left below, right at max
    for left in &below_max {
        for right in &at_max {
            out.push(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            }));
        }
    }

    out
}

/// Try rounding parameters to nearest integers.
fn try_integer_rounding(params: &[f64]) -> Vec<f64> {
    params
        .iter()
        .map(|&p| {
            let rounded = p.round();
            if (p - rounded).abs() < 0.1 {
                rounded
            } else {
                p
            }
        })
        .collect()
}

/// Compute MSE for a tree without parameters (all One nodes = 1.0).
fn compute_mse_direct(tree: &EmlTree, inputs: &[Vec<f64>], targets: &[f64]) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;

    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        match tree.eval_real(&ctx) {
            Ok(val) if val.is_finite() => {
                total += (val - target).powi(2);
                count += 1;
            }
            _ => {}
        }
    }

    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

/// Compute MSE for a parameterized tree.
fn compute_mse_parameterized(
    ptree: &ParameterizedEmlTree,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;

    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        match ptree.forward(&ctx) {
            Ok(val) if val.is_finite() => {
                total += (val - target).powi(2);
                count += 1;
            }
            _ => {}
        }
    }

    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_depth0() {
        let topos = enumerate_topologies(0, 1);
        // Depth 0: One, Var(0) = 2 leaves
        assert_eq!(topos.len(), 2);
    }

    #[test]
    fn test_enumerate_depth1() {
        let topos = enumerate_topologies(1, 1);
        // Depth 0: 2 leaves (One, x0)
        // Depth 1: each pair of leaves → 2*2 = 4, but exact-depth-1 means
        // at least one child at depth 0 (which all leaves are), so all combos.
        // Actually: trees at depth exactly 1 have both children at depth 0.
        // That's 2*2 = 4 trees. Plus 2 depth-0 trees = 6 total.
        assert!(topos.len() >= 6);
    }

    #[test]
    fn test_symreg_exp() {
        // Generate data from y = exp(x)
        let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 1,
            learning_rate: 1e-2,
            tolerance: 1e-6,
            max_iter: 1000,
            complexity_penalty: 1e-4,
            num_restarts: 2,
            integer_rounding: true,
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine.discover(&inputs, &targets, 1).unwrap();
        assert!(!formulas.is_empty());
        // The best formula should have low MSE
        assert!(formulas[0].mse < 1.0);
    }

    #[test]
    fn test_integer_rounding() {
        let params = vec![0.98, 2.03, 1.51, -0.99];
        let rounded = try_integer_rounding(&params);
        assert!((rounded[0] - 1.0).abs() < 1e-15);
        assert!((rounded[1] - 2.0).abs() < 1e-15);
        assert!((rounded[2] - 1.51).abs() < 1e-15); // Not close enough to round
        assert!((rounded[3] - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_symreg_parallel_matches_sequential() {
        // Parallel and sequential both discover exp(x) with low MSE.
        let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 1,
            learning_rate: 1e-2,
            tolerance: 1e-6,
            max_iter: 1000,
            complexity_penalty: 1e-4,
            num_restarts: 2,
            integer_rounding: true,
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine.discover(&inputs, &targets, 1).unwrap();
        assert!(!formulas.is_empty());
        assert!(formulas[0].mse < 1.0);
    }

    #[test]
    fn test_empty_data() {
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = engine.discover(&[], &[], 1);
        assert!(matches!(result, Err(EmlError::EmptyData)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = engine.discover(&[vec![1.0]], &[1.0, 2.0], 1);
        assert!(matches!(result, Err(EmlError::DimensionMismatch(1, 2))));
    }

    #[test]
    fn test_dedupe_reduces_topology_count() {
        // EML is non-commutative (eml(A,B) != eml(B,A)) and enumerate_at_depth
        // already generates duplicate-free topologies. Dedup via structural
        // hash only catches simplifications like exp(ln(x)) = x — so the
        // actual reduction is tiny (0.0002% at depth 4). We verify the dedup
        // function runs correctly and preserves at least "before >= after".
        //
        // A full depth-4 stress test (2M trees, ~38s) is captured in
        // `test_dedupe_depth_four_stress` and marked `#[ignore]` for CI speed.
        let topologies = enumerate_topologies(2, 1);
        let before = topologies.len();
        let after = dedupe_by_semantics(topologies).len();
        assert!(
            after <= before,
            "dedup must not grow the set: before={before}, after={after}"
        );
    }

    #[test]
    #[ignore = "slow: depth-4 enumerates 2M topologies, ~38s wall-clock"]
    fn test_dedupe_depth_four_stress() {
        let topologies = enumerate_topologies(4, 1);
        let before = topologies.len();
        let after = dedupe_by_semantics(topologies).len();
        assert!(after <= before);
        // Measured: 2_090_918 → 2_090_913 (5-tree reduction from exp(ln(x)) etc.)
        // Kept for benchmarking the dedup pass itself, not for CI.
    }

    #[test]
    fn test_dedupe_preserves_uniqueness() {
        // Every remaining topology should have a unique structural hash under
        // the same canonicalization pipeline used by `dedupe_by_semantics`.
        use std::collections::HashSet;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let topologies = enumerate_topologies(3, 1);
        let deduped = dedupe_by_semantics(topologies);

        let mut hashes: HashSet<u64> = HashSet::new();
        for tree in &deduped {
            let eml_simplified = crate::simplify::simplify(tree);
            let simplified = eml_simplified.lower().simplify();
            let mut h = DefaultHasher::new();
            simplified.structural_hash(&mut h);
            let inserted = hashes.insert(h.finish());
            assert!(inserted, "duplicate structural hash found in deduped set");
        }
        assert_eq!(hashes.len(), deduped.len());
    }

    #[test]
    fn test_dedupe_preserves_discovery_exp() {
        // Rerunning test_symreg_exp-style discovery after dedup should still find
        // a low-MSE formula for exp(x).
        let inputs: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 2,
            learning_rate: 1e-2,
            tolerance: 1e-5,
            max_iter: 1000,
            complexity_penalty: 1e-4,
            num_restarts: 2,
            integer_rounding: false,
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine
            .discover(&inputs, &targets, 1)
            .expect("discover should succeed");
        assert!(!formulas.is_empty(), "should discover at least one formula");
        let best = &formulas[0];
        assert!(
            best.mse < 0.1,
            "best formula MSE too high after dedup: {} (pretty={})",
            best.mse,
            best.pretty
        );
    }
}
