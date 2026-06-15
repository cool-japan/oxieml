//! Monte-Carlo Tree Search (MCTS) over partial EML tree topologies.
//!
//! Implements UCB1-guided exploration of the EML grammar space:
//!
//! ```text
//! S → One | Var(i) | Eml(S, S)
//! ```
//!
//! Each MCTS node represents a *partial* EML tree (a tree with some leaves
//! still unexpanded, called HOLEs). The algorithm selects the leftmost HOLE
//! for expansion at each step, guaranteeing that every complete tree is
//! reachable by exactly one action sequence (no double-counting).
//!
//! **UCB1 score** (for child `c` with parent `p`):
//!
//! ```text
//! score(c) = c.total_value / c.visits
//!            + exploration * sqrt(ln(p.visits) / c.visits)
//! ```
//!
//! **Reward**: `1.0 / (1.0 + mse)` — bounded in `(0, 1]`, suitable for UCB1.

use std::sync::Arc;

use crate::error::EmlError;
use crate::symreg::topology::topology_interval_feasible;
use crate::symreg::{DiscoveredFormula, SymRegConfig, SymRegEngine};
use crate::tree::{EmlNode, EmlTree};

type Rng = rand::rngs::StdRng;

/// A partial EML tree: a recursive enum that mirrors `EmlNode` but adds a `Hole` variant
/// for unexpanded leaves.
///
/// We avoid the flat `Vec<Option<EmlNode>>` representation because `EmlNode` is
/// `Arc`-recursive; conversion would require double marshalling. A recursive
/// enum converts to `Arc<EmlNode>` in O(n) with a simple `match`.
#[derive(Clone, Debug)]
enum PartialNode {
    /// Unexpanded leaf — will be replaced by One, Var(i), or Eml during expansion.
    Hole,
    /// The constant `1` (corresponds to `EmlNode::One`).
    One,
    /// Input variable `x_i` (corresponds to `EmlNode::Var(i)`).
    Var(usize),
    /// Free constant leaf (activated by `SymRegConfig.enable_const_leaf`).
    Const(f64),
    /// `eml(left, right) = exp(left) − ln(right)`.
    Eml(Box<PartialNode>, Box<PartialNode>),
}

impl PartialNode {
    /// Count HOLEs in the subtree.
    fn hole_count(&self) -> usize {
        match self {
            PartialNode::Hole => 1,
            PartialNode::One | PartialNode::Var(_) | PartialNode::Const(_) => 0,
            PartialNode::Eml(l, r) => l.hole_count() + r.hole_count(),
        }
    }

    /// Find the leftmost HOLE and apply `action` to it.
    ///
    /// Returns `true` if the action was applied (i.e., a HOLE was found).
    fn expand_leftmost(&mut self, action: &MctsAction) -> bool {
        match self {
            PartialNode::Hole => {
                *self = match action {
                    MctsAction::One => PartialNode::One,
                    MctsAction::Var(i) => PartialNode::Var(*i),
                    MctsAction::FreeConst(v) => PartialNode::Const(*v),
                    MctsAction::Expand => {
                        PartialNode::Eml(Box::new(PartialNode::Hole), Box::new(PartialNode::Hole))
                    }
                };
                true
            }
            PartialNode::One | PartialNode::Var(_) | PartialNode::Const(_) => false,
            PartialNode::Eml(l, r) => {
                if l.expand_leftmost(action) {
                    true
                } else {
                    r.expand_leftmost(action)
                }
            }
        }
    }

    /// Complete all remaining HOLEs by sampling from `{One, Var(0), ..., Var(n-1)}`
    /// uniformly at random (no more `Expand` — forces a finite tree).
    fn complete_random(&mut self, num_vars: usize, rng: &mut Rng) {
        use rand::RngExt;
        match self {
            PartialNode::Hole => {
                let choices = 1 + num_vars; // One + Var(0..n-1)
                let idx = rng.random_range(0..choices);
                *self = if idx == 0 {
                    PartialNode::One
                } else {
                    PartialNode::Var(idx - 1)
                };
            }
            PartialNode::One | PartialNode::Var(_) | PartialNode::Const(_) => {}
            PartialNode::Eml(l, r) => {
                l.complete_random(num_vars, rng);
                r.complete_random(num_vars, rng);
            }
        }
    }

    /// Convert a complete (Hole-free) `PartialNode` into `Arc<EmlNode>`.
    ///
    /// Panics in debug builds if any `Hole` remains (invariant violation).
    fn to_eml_node(&self) -> Arc<EmlNode> {
        match self {
            PartialNode::Hole => {
                // This should never happen if called on a complete tree.
                // Return a sentinel (One) instead of panicking in release.
                debug_assert!(false, "to_eml_node called on a Hole — invariant violated");
                Arc::new(EmlNode::One)
            }
            PartialNode::One => Arc::new(EmlNode::One),
            PartialNode::Var(i) => Arc::new(EmlNode::Var(*i)),
            PartialNode::Const(v) => Arc::new(EmlNode::Const(*v)),
            PartialNode::Eml(l, r) => Arc::new(EmlNode::Eml {
                left: l.to_eml_node(),
                right: r.to_eml_node(),
            }),
        }
    }
}

/// The action taken to expand the leftmost HOLE.
#[derive(Clone, Debug)]
enum MctsAction {
    /// Replace the HOLE with the constant `1`.
    One,
    /// Replace the HOLE with input variable `x_i`.
    Var(usize),
    /// Replace the HOLE with a free constant leaf `Const(v)`.
    FreeConst(f64),
    /// Replace the HOLE with `eml(HOLE, HOLE)` — adds two new HOLEs.
    Expand,
}

/// Legal actions for expanding the leftmost HOLE at depth `hole_depth`.
///
/// If `hole_depth >= max_depth`, only terminal actions (One, Var) are legal —
/// adding an Eml node would push children to depth `hole_depth + 1 > max_depth`.
fn legal_actions(
    hole_depth: usize,
    max_depth: usize,
    num_vars: usize,
    enable_const: bool,
    const_init: f64,
) -> Vec<MctsAction> {
    let mut actions = Vec::with_capacity(2 + num_vars + usize::from(enable_const));
    actions.push(MctsAction::One);
    for i in 0..num_vars {
        actions.push(MctsAction::Var(i));
    }
    if enable_const {
        actions.push(MctsAction::FreeConst(const_init));
    }
    if hole_depth < max_depth {
        actions.push(MctsAction::Expand);
    }
    actions
}

/// Compute the depth of the leftmost HOLE in a `PartialNode` tree.
fn leftmost_hole_depth(node: &PartialNode, current: usize) -> Option<usize> {
    match node {
        PartialNode::Hole => Some(current),
        PartialNode::One | PartialNode::Var(_) | PartialNode::Const(_) => None,
        PartialNode::Eml(l, r) => {
            leftmost_hole_depth(l, current + 1).or_else(|| leftmost_hole_depth(r, current + 1))
        }
    }
}

/// A node in the MCTS search tree.
///
/// Uses a flat `Vec<MctsNode>` with index-based parent/child links to avoid
/// `Rc<RefCell<...>>` lifetime complexity.
struct MctsNode {
    /// The partial tree stored at this MCTS node.
    partial: PartialNode,
    /// Number of times this node has been visited.
    visits: u64,
    /// Cumulative reward (`1/(1+mse)`, bounded in `(0,1]`).
    total_value: f64,
    /// Indices of child nodes in the flat arena.
    children: Vec<usize>,
    /// Index of parent node (`usize::MAX` for the root).
    parent: usize,
    /// Whether all legal actions from this node have been tried.
    fully_expanded: bool,
    /// Number of children already expanded (index into `legal_actions`).
    next_action_idx: usize,
    /// Depth of the leftmost HOLE at this node (cached for action generation).
    leftmost_hole_depth: Option<usize>,
}

impl MctsNode {
    fn new(partial: PartialNode, parent: usize) -> Self {
        let hole_depth = leftmost_hole_depth(&partial, 0);
        Self {
            partial,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            parent,
            fully_expanded: false,
            next_action_idx: 0,
            leftmost_hole_depth: hole_depth,
        }
    }

    /// Returns `true` if this partial tree has no remaining HOLEs.
    fn is_complete(&self) -> bool {
        self.partial.hole_count() == 0
    }

    /// UCB1 score for this node given parent's visit count.
    fn ucb1(&self, parent_visits: u64, exploration: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.total_value / self.visits as f64;
        let ln_parent = (parent_visits as f64).ln();
        let exploration_term = exploration * (ln_parent / self.visits as f64).sqrt();
        exploitation + exploration_term
    }
}

/// Convert a complete `PartialNode` to an `EmlTree`.
///
/// `EmlTree::from_node` counts variables internally via `count_vars`.
fn partial_to_tree(node: &PartialNode) -> EmlTree {
    let root = node.to_eml_node();
    EmlTree::from_node(root)
}

/// Run the MCTS algorithm over EML topology space.
///
/// This is the main entry point called from `SymRegEngine::discover_mcts`.
pub(crate) fn run_mcts(
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
    num_vars: usize,
    iterations: usize,
    exploration: f64,
) -> Result<Vec<DiscoveredFormula>, EmlError> {
    if inputs.is_empty() || targets.is_empty() {
        return Err(EmlError::EmptyData);
    }
    if inputs.len() != targets.len() {
        return Err(EmlError::DimensionMismatch(inputs.len(), targets.len()));
    }
    if iterations == 0 {
        return Ok(vec![]);
    }

    let config = &engine.config;
    let max_depth = config.max_depth;

    // Surrogate engine: cheap Adam for rollout simulation.
    let surrogate_iters = config.max_iter.clamp(10, 50);
    let surrogate_config = SymRegConfig {
        max_iter: surrogate_iters,
        num_restarts: 1,
        cv_folds: None,
        ..config.clone()
    };
    let surrogate_engine = SymRegEngine::new(surrogate_config);

    // Interval pruning setup (if enabled).
    let interval_data = if config.interval_pruning {
        use crate::lower_interval::IntervalLO;
        let input_intervals: Vec<IntervalLO> = (0..num_vars)
            .map(|j| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for row in inputs.iter() {
                    if let Some(&v) = row.get(j) {
                        if v < lo {
                            lo = v;
                        }
                        if v > hi {
                            hi = v;
                        }
                    }
                }
                if lo.is_finite() && hi.is_finite() {
                    IntervalLO::new(lo, hi)
                } else {
                    IntervalLO::full()
                }
            })
            .collect();
        let target_lo = targets.iter().copied().fold(f64::INFINITY, f64::min);
        let target_hi = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((input_intervals, target_lo, target_hi))
    } else {
        None
    };

    // RNG: use a different salt from the topology-level seeds.
    let mut rng = make_mcts_rng(config.seed);

    // Flat arena of MCTS nodes (avoids Rc cycles).
    let root_partial = PartialNode::Hole;
    let mut arena: Vec<MctsNode> = vec![MctsNode::new(root_partial, usize::MAX)];

    // Track which complete trees we've encountered and their best rewards.
    // Key = index in `arena`, Value = reward.
    let mut complete_nodes: Vec<(usize, f64)> = Vec::new();

    // Track rollout-completed trees and their surrogate MSE.
    // These are the primary source of interesting candidates when max_depth > 1.
    let mut rollout_candidates: Vec<(EmlTree, f64)> = Vec::new();

    for _iter in 0..iterations {
        // === SELECTION ===
        // Walk from root using UCB1 until we reach a node that:
        //   (a) is a complete tree (terminal), or
        //   (b) has at least one unexpanded action (will be expanded next).
        let mut node_idx = 0usize;
        loop {
            let (is_complete, fully_expanded) = {
                let node = &arena[node_idx];
                (node.is_complete(), node.fully_expanded)
            };

            if is_complete || !fully_expanded {
                // Stop here: either terminal, or has room to expand.
                break;
            }

            // All children visited: select the best UCB1 child.
            let parent_visits = arena[node_idx].visits;
            let children: Vec<usize> = arena[node_idx].children.clone();
            let best_child = children
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    arena[a]
                        .ucb1(parent_visits, exploration)
                        .partial_cmp(&arena[b].ucb1(parent_visits, exploration))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(node_idx);
            node_idx = best_child;
        }

        // === EXPANSION ===
        let expanded_idx = {
            let (is_complete, fully_expanded) = {
                let node = &arena[node_idx];
                (node.is_complete(), node.fully_expanded)
            };

            if is_complete || fully_expanded {
                // Can't expand further: simulate from here.
                node_idx
            } else {
                // Determine legal actions for this node's leftmost HOLE.
                let (hole_depth, actions) = {
                    let node = &arena[node_idx];
                    let hd = node.leftmost_hole_depth.unwrap_or(0);
                    let acts = legal_actions(
                        hd,
                        max_depth,
                        num_vars,
                        config.enable_const_leaf,
                        config.const_leaf_init,
                    );
                    (hd, acts)
                };
                let _ = hole_depth; // captured inside legal_actions call

                let action_idx = arena[node_idx].next_action_idx;
                if action_idx >= actions.len() {
                    // All actions exhausted — mark fully expanded.
                    arena[node_idx].fully_expanded = true;
                    node_idx
                } else {
                    // Apply action to create a new child.
                    let action = &actions[action_idx];
                    let mut new_partial = arena[node_idx].partial.clone();
                    new_partial.expand_leftmost(action);

                    // Update parent's action counter.
                    arena[node_idx].next_action_idx += 1;
                    if arena[node_idx].next_action_idx >= actions.len() {
                        arena[node_idx].fully_expanded = true;
                    }

                    let child_idx = arena.len();
                    let mut child_node = MctsNode::new(new_partial, node_idx);
                    // Recompute leftmost_hole_depth after expansion.
                    child_node.leftmost_hole_depth = leftmost_hole_depth(&child_node.partial, 0);
                    arena.push(child_node);
                    arena[node_idx].children.push(child_idx);
                    child_idx
                }
            }
        };

        // === SIMULATION (rollout) ===
        // Clone the partial tree, complete remaining HOLEs randomly, evaluate.
        // We also keep the rollout tree so we can include it in the candidate pool.
        let (reward, rollout_tree_opt) = {
            let mut rollout_partial = arena[expanded_idx].partial.clone();
            rollout_partial.complete_random(num_vars, &mut rng);

            let tree = partial_to_tree(&rollout_partial);

            // Units pre-filter (gated on Some(unit_filter)).
            let units_feasible = if let Some((ref var_units, target_units)) = config.unit_filter {
                let lowered = tree.lower().simplify();
                matches!(lowered.check_units(var_units), Ok(u) if u == target_units)
            } else {
                true
            };

            // Interval pruning: skip infeasible topologies.
            let feasible = units_feasible
                && if let Some((ref ivs, tlo, thi)) = interval_data {
                    let threshold = config.interval_pruning_depth_threshold;
                    if tree.depth() < threshold {
                        true
                    } else if !topology_interval_feasible(&tree, ivs, tlo, thi) {
                        false
                    } else {
                        #[cfg(feature = "smt")]
                        let smt_feasible = {
                            use crate::smt::Interval;
                            let smt_vars: Vec<Interval> =
                                ivs.iter().map(|iv| Interval::new(iv.lo, iv.hi)).collect();
                            let constraint = crate::smt::EmlConstraint::GeZero(tree.clone());
                            if config.smt_prune_solver {
                                let depth = tree.depth();
                                let min_d = config.interval_pruning_depth_threshold;
                                !super::smt_prune::solver_prune(
                                    &constraint,
                                    &smt_vars,
                                    min_d,
                                    depth,
                                )
                            } else if config.smt_prune {
                                !super::smt_prune::interval_prune(&constraint, &smt_vars)
                            } else {
                                true
                            }
                        };
                        #[cfg(not(feature = "smt"))]
                        let smt_feasible = true;
                        smt_feasible
                    }
                } else {
                    true
                };

            if !feasible {
                // Assign a very low reward for pruned topologies; discard tree.
                (0.0, None)
            } else {
                // Quick Adam fit — keep the tree AND the reward.
                let formula =
                    surrogate_engine.optimize_topology(&tree, inputs, targets, expanded_idx);
                match formula {
                    Some(f) => {
                        let r = 1.0 / (1.0 + f.mse);
                        (r, Some((tree, f.mse)))
                    }
                    None => (0.0, None),
                }
            }
        };

        // Track complete nodes (arena-complete path) AND rollout trees (simulation path).
        if arena[expanded_idx].is_complete() {
            complete_nodes.push((expanded_idx, reward));
        }
        // Always record the rollout tree regardless — this is the primary source of
        // interesting complete trees when max_depth > 1.
        if let Some(rt) = rollout_tree_opt {
            rollout_candidates.push(rt);
        }

        // === BACKPROPAGATION ===
        let mut idx = expanded_idx;
        loop {
            arena[idx].visits += 1;
            arena[idx].total_value += reward;
            let parent = arena[idx].parent;
            if parent == usize::MAX {
                break;
            }
            idx = parent;
        }
    }

    // === FINALIZATION ===
    // Merge three sources of complete trees into a single candidate pool:
    //   1. Rollout trees (simulation-completed partials) — the primary source
    //      for max_depth > 1; each has a surrogate MSE from the quick Adam fit.
    //   2. Arena-complete nodes — trees that were fully expanded during selection
    //      without needing a random completion rollout.
    //
    // We store (tree, mse) for rollout candidates and convert arena-complete nodes
    // to the same format using their accumulated average reward.
    let mut candidate_trees: Vec<(EmlTree, f64)> = rollout_candidates;

    // Arena-complete nodes: convert average reward back to a pseudo-MSE.
    // reward = 1/(1+mse) → mse = 1/reward − 1
    for (node_idx, _) in &complete_nodes {
        let node = &arena[*node_idx];
        if node.is_complete() && node.visits > 0 {
            let avg_reward = node.total_value / node.visits as f64;
            let pseudo_mse = if avg_reward > 0.0 {
                1.0 / avg_reward - 1.0
            } else {
                f64::INFINITY
            };
            let tree = partial_to_tree(&node.partial);
            candidate_trees.push((tree, pseudo_mse));
        }
    }

    // Sort by MSE ascending (lowest = best fit).
    candidate_trees.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // De-duplicate by structural hash, keeping the top-K unique trees.
    let top_k = 20_usize;
    let mut seen_hashes = std::collections::HashSet::new();
    let unique_candidates: Vec<EmlTree> = candidate_trees
        .into_iter()
        .filter_map(|(tree, _)| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hasher;
            let simplified = tree.lower().simplify();
            let mut h = DefaultHasher::new();
            simplified.structural_hash(&mut h);
            let hash = h.finish();
            if seen_hashes.insert(hash) {
                Some(tree)
            } else {
                None
            }
        })
        .take(top_k)
        .collect();

    if unique_candidates.is_empty() {
        return Ok(vec![]);
    }

    // Full Adam optimization on the top candidates.
    engine.optimize_and_finalize(unique_candidates, inputs, targets)
}

/// Create an RNG for MCTS rollouts with a distinct salt from topology seeds.
fn make_mcts_rng(seed: Option<u64>) -> Rng {
    use rand::SeedableRng;
    const MCTS_SALT: u64 = 0xDEAD_BEEF_CAFE_1234;
    match seed {
        Some(s) => {
            // SplitMix64 mixing with the salt.
            let mixed = {
                let mut z = s.wrapping_add(MCTS_SALT);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^ (z >> 31)
            };
            Rng::seed_from_u64(mixed)
        }
        None => rand::make_rng::<Rng>(),
    }
}
