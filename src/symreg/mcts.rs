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
//!
//! # Root-parallel batching and bit-for-bit determinism
//!
//! Under `feature = "parallel"` the *simulation* phase is executed on the rayon
//! pool. The search is therefore organised in fixed-size **batches** of
//! [`MCTS_BATCH_SIZE`] iterations, each batch running three strictly separated
//! phases:
//!
//! 1. **Prepare (sequential).** Selection, expansion and random rollout
//!    completion for every iteration of the batch, in ascending iteration
//!    order. This phase is the only one that mutates the arena's *shape*
//!    (`children`, `next_action_idx`, `fully_expanded`) and the only one that
//!    draws random numbers. A *virtual visit* (`visits += 1` along the path to
//!    the root, with no value added yet) is applied immediately so that the
//!    next iteration of the same batch does not re-select the same in-flight
//!    node — the classic virtual-loss trick, applied deterministically because
//!    the phase is sequential.
//! 2. **Simulate (parallel or sequential).** [`rollout_once`] is a *pure*
//!    function: it reads only its arguments, touches no global state, and seeds
//!    its optimizer RNG from `(config.seed, expanded_idx)`. It is mapped over
//!    the batch with `par_iter` under `feature = "parallel"` and with `iter`
//!    otherwise. Rayon's `collect()` on an indexed parallel iterator preserves
//!    source order, so `outcomes[k]` always corresponds to `tasks[k]`
//!    regardless of how work was stolen between threads.
//! 3. **Merge (sequential).** Rewards are back-propagated in ascending batch
//!    order. This is the *only* place a `f64` is ever added to another `f64`
//!    that came from a different iteration.
//!
//! **Determinism argument.** Floating-point addition is not associative, so the
//! order in which rewards are folded into `MctsNode::total_value` is part of the
//! result. Because (a) the batch boundaries are a compile-time constant and are
//! *not* derived from the thread count, (b) `tasks` is built sequentially, and
//! (c) `total_value += reward` only ever happens in phase 3 walking `tasks` in
//! ascending index order, every node accumulates *exactly* the same sequence of
//! `+=` operations for a given `config.seed`, whatever `RAYON_NUM_THREADS` says.
//! Rayon is never allowed to reduce floats. Consequently the parallel and the
//! sequential paths are bit-for-bit identical, and so are two parallel runs with
//! different thread counts.
//!
//! Setting [`MCTS_BATCH_SIZE`] to `1` degenerates exactly to the textbook
//! sequential MCTS loop (select → expand → simulate → backprop), which is a
//! useful sanity anchor when reasoning about the batched form.

use std::sync::Arc;

use rand::SeedableRng;

use crate::error::EmlError;
use crate::lower_interval::IntervalLO;
use crate::symreg::discover::derive_seed;
use crate::symreg::topology::topology_interval_feasible;
use crate::symreg::{DiscoveredFormula, SymRegConfig, SymRegEngine};
use crate::tree::{EmlNode, EmlTree};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type Rng = rand::rngs::StdRng;

/// Number of MCTS iterations whose simulation phase is evaluated as one batch.
///
/// This is a **compile-time constant on purpose**: it must never be derived from
/// `rayon::current_num_threads()`. The batch boundary is part of the algorithm's
/// definition (back-propagation is deferred to the end of a batch), so a batch
/// size that varied with `RAYON_NUM_THREADS` would change the search trajectory
/// and destroy thread-count invariance.
///
/// The value trades search fidelity against parallel occupancy: small enough
/// that the tree statistics stay close to the sequential trajectory, large
/// enough to keep a typical rayon pool busy with the (comparatively expensive)
/// surrogate Adam fits.
const MCTS_BATCH_SIZE: usize = 8;

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

// ─────────────────────────────────────────────────────────────────────────────
// Simulation phase — pure, side-effect-free, safe to run on the rayon pool
// ─────────────────────────────────────────────────────────────────────────────

/// Per-variable input intervals plus the observed target range.
///
/// Computed once, up-front, from the training data; shared immutably by every
/// rollout (hence `Sync`-by-construction: it contains only `f64`s).
struct IntervalData {
    /// Observed `[min, max]` range of each input variable.
    var_intervals: Vec<IntervalLO>,
    /// Smallest observed target value.
    target_lo: f64,
    /// Largest observed target value.
    target_hi: f64,
}

/// One unit of simulation work, produced by the sequential prepare phase.
struct RolloutTask {
    /// Arena index of the node this rollout was launched from.
    ///
    /// Doubles as the `topology_idx` handed to `optimize_topology`, which makes
    /// the surrogate optimizer's RNG a deterministic function of
    /// `(config.seed, expanded_idx)` — never of the thread that ran it.
    expanded_idx: usize,
    /// The randomly-completed (Hole-free) tree to fit.
    tree: EmlTree,
}

/// The result of one simulation, consumed by the sequential merge phase.
struct RolloutOutcome {
    /// UCB1 reward `1/(1 + mse)`, or `0.0` when the topology was pruned or the
    /// surrogate fit failed.
    reward: f64,
    /// `(tree, surrogate_mse)` when the fit succeeded — fed into the final
    /// candidate pool.
    candidate: Option<(EmlTree, f64)>,
}

/// Run one MCTS simulation (rollout evaluation).
///
/// # Purity / determinism
///
/// This function is the body of the parallel region and is deliberately pure:
///
/// * It reads only its arguments — no statics, no thread-locals, no interior
///   mutability.
/// * It draws **no** random numbers from a shared stream. The surrogate
///   optimizer seeds itself from `(config.seed, task.expanded_idx)` inside
///   `SymRegEngine::optimize_topology`, which is an index-derived seed, so two
///   runs that schedule this task onto different threads still see the identical
///   RNG stream.
/// * Every floating-point reduction it performs (MSE, gradients) happens *within*
///   this single task, in a fixed order. No float produced here is ever combined
///   with a float from another task except in the sequential merge phase.
///
/// Therefore `rollout_once` returns bit-identical `f64`s for identical inputs,
/// which is what makes the whole search reproducible.
fn rollout_once(
    task: &RolloutTask,
    surrogate_engine: &SymRegEngine,
    config: &SymRegConfig,
    inputs: &[Vec<f64>],
    targets: &[f64],
    interval_data: Option<&IntervalData>,
) -> RolloutOutcome {
    let tree = &task.tree;

    // Units pre-filter (gated on Some(unit_filter)).
    let units_feasible = if let Some((ref var_units, target_units)) = config.unit_filter {
        let lowered = tree.lower().simplify();
        matches!(lowered.check_units(var_units), Ok(u) if u == target_units)
    } else {
        true
    };

    if !units_feasible || !interval_feasible(tree, config, interval_data) {
        // Pruned topology: lowest possible reward, no candidate.
        return RolloutOutcome {
            reward: 0.0,
            candidate: None,
        };
    }

    // Quick Adam fit — keep the tree AND the reward.
    match surrogate_engine.optimize_topology(tree, inputs, targets, task.expanded_idx) {
        Some(f) => RolloutOutcome {
            reward: 1.0 / (1.0 + f.mse),
            candidate: Some((tree.clone(), f.mse)),
        },
        None => RolloutOutcome {
            reward: 0.0,
            candidate: None,
        },
    }
}

/// Interval-pruning pre-filter: `false` means the topology cannot match the data.
fn interval_feasible(
    tree: &EmlTree,
    config: &SymRegConfig,
    interval_data: Option<&IntervalData>,
) -> bool {
    let Some(data) = interval_data else {
        return true;
    };
    if tree.depth() < config.interval_pruning_depth_threshold {
        return true;
    }
    if !topology_interval_feasible(tree, &data.var_intervals, data.target_lo, data.target_hi) {
        return false;
    }
    smt_feasible(tree, config, data)
}

/// SMT-backed pruning (`smt` feature): `false` means proved infeasible.
#[cfg(feature = "smt")]
fn smt_feasible(tree: &EmlTree, config: &SymRegConfig, data: &IntervalData) -> bool {
    use crate::smt::Interval;
    let smt_vars: Vec<Interval> = data
        .var_intervals
        .iter()
        .map(|iv| Interval::new(iv.lo, iv.hi))
        .collect();
    let constraint = crate::smt::EmlConstraint::GeZero(tree.clone());
    if config.smt_prune_solver {
        let depth = tree.depth();
        let min_d = config.interval_pruning_depth_threshold;
        !super::smt_prune::solver_prune(&constraint, &smt_vars, min_d, depth)
    } else if config.smt_prune {
        !super::smt_prune::interval_prune(&constraint, &smt_vars)
    } else {
        true
    }
}

/// SMT pruning is unavailable without the `smt` feature — nothing is pruned.
#[cfg(not(feature = "smt"))]
fn smt_feasible(_tree: &EmlTree, _config: &SymRegConfig, _data: &IntervalData) -> bool {
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree-policy phases — all sequential, all arena-mutating
// ─────────────────────────────────────────────────────────────────────────────

/// SELECTION: walk from the root along the best-UCB1 child until we reach a node
/// that is either terminal (no HOLEs) or still has an untried action.
///
/// Ties in `max_by` resolve to the *last* maximal child (Rust's documented
/// `Iterator::max_by` behaviour) — a fixed, data-independent rule, so selection
/// is reproducible.
fn select_leaf(arena: &[MctsNode], exploration: f64) -> usize {
    let mut node_idx = 0usize;
    loop {
        let node = &arena[node_idx];
        if node.is_complete() || !node.fully_expanded {
            return node_idx;
        }
        let parent_visits = node.visits;
        let best_child = node
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                arena[a]
                    .ucb1(parent_visits, exploration)
                    .partial_cmp(&arena[b].ucb1(parent_visits, exploration))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(node_idx);
        if best_child == node_idx {
            // Defensive: a fully-expanded, non-terminal node always has children,
            // so this cannot normally fire. Guarantees loop termination anyway.
            return node_idx;
        }
        node_idx = best_child;
    }
}

/// EXPANSION: apply the next untried action at `node_idx` and return the arena
/// index to simulate from (the new child, or `node_idx` itself when the node is
/// terminal / already fully expanded).
fn expand_node(
    arena: &mut Vec<MctsNode>,
    node_idx: usize,
    max_depth: usize,
    num_vars: usize,
    config: &SymRegConfig,
) -> usize {
    if arena[node_idx].is_complete() || arena[node_idx].fully_expanded {
        return node_idx;
    }

    let hole_depth = arena[node_idx].leftmost_hole_depth.unwrap_or(0);
    let actions = legal_actions(
        hole_depth,
        max_depth,
        num_vars,
        config.enable_const_leaf,
        config.const_leaf_init,
    );

    let action_idx = arena[node_idx].next_action_idx;
    let Some(action) = actions.get(action_idx) else {
        // All actions exhausted — mark fully expanded and simulate from here.
        arena[node_idx].fully_expanded = true;
        return node_idx;
    };

    let mut new_partial = arena[node_idx].partial.clone();
    new_partial.expand_leftmost(action);

    arena[node_idx].next_action_idx += 1;
    if arena[node_idx].next_action_idx >= actions.len() {
        arena[node_idx].fully_expanded = true;
    }

    let child_idx = arena.len();
    arena.push(MctsNode::new(new_partial, node_idx));
    arena[node_idx].children.push(child_idx);
    child_idx
}

/// VIRTUAL LOSS: count an in-flight simulation along the path `idx → root`.
///
/// Only `visits` is bumped; `total_value` is left alone until the real reward is
/// known. Within a batch this makes an in-flight node look pessimistic
/// (`total_value / visits` drops), which stops the next iteration of the same
/// batch from selecting it again — the diversification that root-parallel MCTS
/// needs. Because this runs in the sequential prepare phase, it is applied in a
/// fixed order and involves no floating-point arithmetic at all.
fn apply_virtual_visit(arena: &mut [MctsNode], mut idx: usize) {
    loop {
        arena[idx].visits += 1;
        let parent = arena[idx].parent;
        if parent == usize::MAX {
            break;
        }
        idx = parent;
    }
}

/// BACKPROPAGATION: fold the realised reward into the path `idx → root`.
///
/// `visits` was already incremented by [`apply_virtual_visit`], so only the
/// value is added here. This is the **single** place where rewards from
/// different iterations are summed, and it is called strictly in ascending batch
/// order from the sequential merge phase — which is precisely what pins the
/// (non-associative) `f64` addition order.
fn backpropagate_value(arena: &mut [MctsNode], mut idx: usize, reward: f64) {
    loop {
        arena[idx].total_value += reward;
        let parent = arena[idx].parent;
        if parent == usize::MAX {
            break;
        }
        idx = parent;
    }
}

/// Map the batch of rollout tasks to their outcomes on the rayon pool.
///
/// `par_iter().map(...).collect()` over a slice is an *indexed* parallel
/// iterator: rayon guarantees the collected `Vec` is in source order, so
/// `outcomes[k]` corresponds to `tasks[k]` no matter how the work was split.
/// No float is reduced across tasks here — each task returns its own scalars.
#[cfg(feature = "parallel")]
fn simulate_batch(
    tasks: &[RolloutTask],
    surrogate_engine: &SymRegEngine,
    config: &SymRegConfig,
    inputs: &[Vec<f64>],
    targets: &[f64],
    interval_data: Option<&IntervalData>,
) -> Vec<RolloutOutcome> {
    tasks
        .par_iter()
        .map(|task| {
            rollout_once(
                task,
                surrogate_engine,
                config,
                inputs,
                targets,
                interval_data,
            )
        })
        .collect()
}

/// Sequential twin of [`simulate_batch`] — same order, same arithmetic, same bits.
#[cfg(not(feature = "parallel"))]
fn simulate_batch(
    tasks: &[RolloutTask],
    surrogate_engine: &SymRegEngine,
    config: &SymRegConfig,
    inputs: &[Vec<f64>],
    targets: &[f64],
    interval_data: Option<&IntervalData>,
) -> Vec<RolloutOutcome> {
    tasks
        .iter()
        .map(|task| {
            rollout_once(
                task,
                surrogate_engine,
                config,
                inputs,
                targets,
                interval_data,
            )
        })
        .collect()
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
        let var_intervals: Vec<IntervalLO> = (0..num_vars)
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
        Some(IntervalData {
            var_intervals,
            target_lo,
            target_hi,
        })
    } else {
        None
    };
    let interval_data_ref = interval_data.as_ref();

    // Master seed for the rollout-completion RNGs. Each iteration derives its own
    // RNG from `(rollout_master, global_iteration)` rather than pulling from one
    // long shared stream, so the random completion of iteration `i` depends only
    // on `i` — never on how many iterations happened to be in flight.
    let rollout_master = rollout_master_seed(config.seed);

    // Flat arena of MCTS nodes (avoids Rc cycles).
    let root_partial = PartialNode::Hole;
    let mut arena: Vec<MctsNode> = vec![MctsNode::new(root_partial, usize::MAX)];

    // Track which complete trees we've encountered and their best rewards.
    // Key = index in `arena`, Value = reward.
    let mut complete_nodes: Vec<(usize, f64)> = Vec::new();

    // Track rollout-completed trees and their surrogate MSE.
    // These are the primary source of interesting candidates when max_depth > 1.
    let mut rollout_candidates: Vec<(EmlTree, f64)> = Vec::new();

    let mut tasks: Vec<RolloutTask> = Vec::with_capacity(MCTS_BATCH_SIZE);

    for batch_start in (0..iterations).step_by(MCTS_BATCH_SIZE) {
        let batch_len = MCTS_BATCH_SIZE.min(iterations - batch_start);

        // ── Phase 1: PREPARE (sequential) ──────────────────────────────────
        // Selection + expansion + virtual visit + random rollout completion.
        // Everything that mutates the arena's shape or consumes randomness lives
        // here, in ascending iteration order.
        tasks.clear();
        for k in 0..batch_len {
            let global_iter = batch_start + k;

            let node_idx = select_leaf(&arena, exploration);
            let expanded_idx = expand_node(&mut arena, node_idx, max_depth, num_vars, config);

            // Count the in-flight simulation *now* so the remaining iterations of
            // this batch see a pessimistic (virtual-loss) estimate for this path.
            apply_virtual_visit(&mut arena, expanded_idx);

            let mut rollout_partial = arena[expanded_idx].partial.clone();
            let mut rollout_rng =
                Rng::seed_from_u64(derive_seed(rollout_master, global_iter as u64));
            rollout_partial.complete_random(num_vars, &mut rollout_rng);

            tasks.push(RolloutTask {
                expanded_idx,
                tree: partial_to_tree(&rollout_partial),
            });
        }

        // ── Phase 2: SIMULATE (rayon pool under `feature = "parallel"`) ────
        let outcomes = simulate_batch(
            &tasks,
            &surrogate_engine,
            config,
            inputs,
            targets,
            interval_data_ref,
        );

        // ── Phase 3: MERGE (sequential, ascending batch order) ─────────────
        // The one and only place rewards from different iterations are summed.
        for (task, outcome) in tasks.iter().zip(outcomes) {
            if arena[task.expanded_idx].is_complete() {
                complete_nodes.push((task.expanded_idx, outcome.reward));
            }
            // Always record the rollout tree regardless — this is the primary source
            // of interesting complete trees when max_depth > 1.
            if let Some(rt) = outcome.candidate {
                rollout_candidates.push(rt);
            }
            backpropagate_value(&mut arena, task.expanded_idx, outcome.reward);
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

/// Master seed for the per-iteration rollout-completion RNGs.
///
/// Drawn once, up-front. Every rollout then builds its own RNG from
/// `derive_seed(master, global_iteration)`, so the random completion of a given
/// iteration is a pure function of its **index** — not of its position in some
/// shared stream whose consumption order could, in a batched/parallel setting,
/// depend on scheduling. With `config.seed = None` the master comes from OS
/// entropy, which is the documented "unseeded" behaviour.
fn rollout_master_seed(seed: Option<u64>) -> u64 {
    use rand::RngExt;
    let mut rng = make_mcts_rng(seed);
    rng.random::<u64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symreg::SymRegStrategy;

    fn linear_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let inputs: Vec<Vec<f64>> = (1..=16).map(|i| vec![i as f64 * 0.3]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * x[0]).collect();
        (inputs, targets)
    }

    fn mcts_config(seed: u64, iterations: usize) -> SymRegConfig {
        SymRegConfig {
            max_depth: 2,
            max_iter: 120,
            num_restarts: 1,
            seed: Some(seed),
            strategy: SymRegStrategy::Mcts {
                iterations,
                exploration: 1.0,
            },
            ..SymRegConfig::default()
        }
    }

    /// Two `run_mcts` calls with the same seed must agree on every bit.
    ///
    /// This holds in *both* feature configurations: with `parallel` on it
    /// exercises the rayon path, with it off the sequential twin.
    #[test]
    fn mcts_repeat_run_is_bit_identical() {
        let (inputs, targets) = linear_data();
        let engine = SymRegEngine::new(mcts_config(42, 40));

        let a = run_mcts(&engine, &inputs, &targets, 1, 40, 1.0).expect("first mcts run");
        let b = run_mcts(&engine, &inputs, &targets, 1, 40, 1.0).expect("second mcts run");

        assert_eq!(a.len(), b.len(), "formula count must match");
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(
                fa.mse.to_bits(),
                fb.mse.to_bits(),
                "MSE bits must match: {} vs {}",
                fa.mse,
                fb.mse
            );
            assert_eq!(fa.pretty, fb.pretty, "pretty form must match");
            let pa: Vec<u64> = fa.params.iter().map(|p| p.to_bits()).collect();
            let pb: Vec<u64> = fb.params.iter().map(|p| p.to_bits()).collect();
            assert_eq!(pa, pb, "parameter bits must match");
        }
    }

    /// A batch that is not a multiple of `MCTS_BATCH_SIZE` must still work: the
    /// final partial batch is `iterations % MCTS_BATCH_SIZE` long.
    #[test]
    fn mcts_handles_partial_final_batch() {
        let (inputs, targets) = linear_data();
        // 13 = 1 full batch of 8 + a partial batch of 5.
        assert_ne!(
            13 % MCTS_BATCH_SIZE,
            0,
            "test premise: 13 is not a multiple"
        );
        let engine = SymRegEngine::new(mcts_config(7, 13));
        let out = run_mcts(&engine, &inputs, &targets, 1, 13, 1.0).expect("mcts run");
        assert!(!out.is_empty(), "13 iterations should yield candidates");
    }

    /// Zero iterations is a legal (degenerate) request and must not panic.
    #[test]
    fn mcts_zero_iterations_is_empty() {
        let (inputs, targets) = linear_data();
        let engine = SymRegEngine::new(mcts_config(1, 0));
        let out = run_mcts(&engine, &inputs, &targets, 1, 0, 1.0).expect("zero-iteration mcts");
        assert!(out.is_empty(), "zero iterations must return no formulas");
    }

    /// Build a representative batch of rollout tasks, exactly as the prepare
    /// phase of [`run_mcts`] would.
    #[cfg(feature = "parallel")]
    fn make_tasks(num_vars: usize, count: usize) -> Vec<RolloutTask> {
        let mut arena: Vec<MctsNode> = vec![MctsNode::new(PartialNode::Hole, usize::MAX)];
        let config = mcts_config(42, count);
        let master = rollout_master_seed(config.seed);
        let mut tasks = Vec::with_capacity(count);
        for global_iter in 0..count {
            let node_idx = select_leaf(&arena, 1.0);
            let expanded_idx =
                expand_node(&mut arena, node_idx, config.max_depth, num_vars, &config);
            apply_virtual_visit(&mut arena, expanded_idx);
            let mut partial = arena[expanded_idx].partial.clone();
            let mut rng = Rng::seed_from_u64(derive_seed(master, global_iter as u64));
            partial.complete_random(num_vars, &mut rng);
            tasks.push(RolloutTask {
                expanded_idx,
                tree: partial_to_tree(&partial),
            });
        }
        tasks
    }

    /// **The parallel == sequential proof at the map level.**
    ///
    /// `simulate_batch` is the *only* thing that differs between the `parallel`
    /// and the non-`parallel` build of `run_mcts` — every other line of the
    /// algorithm is shared, un-`cfg`-ed source. This test runs the rayon version
    /// and then, in the same binary, the literal body of the sequential twin, and
    /// asserts the two agree on every bit of every reward and MSE.
    #[cfg(feature = "parallel")]
    #[test]
    fn simulate_batch_parallel_equals_sequential_bitwise() {
        let (inputs, targets) = linear_data();
        let config = mcts_config(42, 32);
        let engine = SymRegEngine::new(config.clone());
        let tasks = make_tasks(1, 32);
        assert!(!tasks.is_empty(), "prepare phase must produce work");

        let parallel = simulate_batch(&tasks, &engine, &config, &inputs, &targets, None);

        // This is byte-for-byte the body of the `#[cfg(not(feature = "parallel"))]`
        // twin of `simulate_batch`.
        let sequential: Vec<RolloutOutcome> = tasks
            .iter()
            .map(|task| rollout_once(task, &engine, &config, &inputs, &targets, None))
            .collect();

        assert_eq!(parallel.len(), sequential.len(), "batch length must match");
        for (k, (p, s)) in parallel.iter().zip(sequential.iter()).enumerate() {
            assert_eq!(
                p.reward.to_bits(),
                s.reward.to_bits(),
                "task {k}: reward bits differ ({} vs {})",
                p.reward,
                s.reward
            );
            match (&p.candidate, &s.candidate) {
                (Some((_, pm)), Some((_, sm))) => assert_eq!(
                    pm.to_bits(),
                    sm.to_bits(),
                    "task {k}: candidate MSE bits differ ({pm} vs {sm})"
                ),
                (None, None) => {}
                _ => panic!("task {k}: candidate presence differs"),
            }
        }
    }

    /// Thread-count invariance of the simulation phase: 1, 2, 3 and 8 worker
    /// threads must all produce the identical bit pattern.
    #[cfg(feature = "parallel")]
    #[test]
    fn simulate_batch_is_thread_count_invariant() {
        let (inputs, targets) = linear_data();
        let config = mcts_config(42, 32);
        let engine = SymRegEngine::new(config.clone());
        let tasks = make_tasks(1, 32);

        let run_with = |threads: usize| -> Vec<u64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool should build");
            pool.install(|| {
                simulate_batch(&tasks, &engine, &config, &inputs, &targets, None)
                    .iter()
                    .map(|o| o.reward.to_bits())
                    .collect()
            })
        };

        let reference = run_with(1);
        for threads in [2usize, 3, 8] {
            assert_eq!(
                run_with(threads),
                reference,
                "reward bits changed with {threads} rayon threads"
            );
        }
    }
}
