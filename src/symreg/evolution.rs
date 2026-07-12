//! Evolutionary symbolic regression: genetic algorithm with island populations.
//!
//! Implements tournament selection, subtree crossover, and multiple mutation
//! types. Island populations evolve in parallel (requires the `parallel` feature)
//! or sequentially, with ring migration between islands.
//!
//! # Parallel fitness and bit-for-bit determinism
//!
//! Under `feature = "parallel"` two independent levels of parallelism are used:
//!
//! * **Island level** — whole islands are independent, so they map cleanly onto
//!   `par_iter`. `collect()` restores source order, so the reduction that picks
//!   the global best formula still sees the islands in ascending index order.
//! * **Intra-island fitness level** — [`evaluate_population`] evaluates a
//!   generation's individuals with a strict **two-phase map-then-merge**:
//!
//!   1. *Prepare (sequential).* Walk the slice in ascending slot order, hash each
//!      individual, and consult the fitness cache **read-only**. A hit is settled
//!      immediately; a miss becomes a work item carrying its own index-derived
//!      `topology_idx` seed.
//!   2. *Map (parallel or sequential).* `optimize_topology` is applied to the
//!      work items. It is a pure function of `(config, tree, data, topology_idx)`
//!      — it seeds its RNG from `(config.seed, topology_idx)` and touches no
//!      global state — so its `f64` output is bit-identical regardless of which
//!      thread runs it.
//!   3. *Merge (sequential).* Results are written back in **ascending slot
//!      order**, replaying the exact cache semantics of the sequential loop: a
//!      slot whose hash was inserted by an *earlier* slot of the same generation
//!      takes the cached formula and discards its own (redundantly computed)
//!      result. This makes the parallel path bit-identical not only to its own
//!      sequential twin but to the original single-threaded implementation.
//!
//! No `f64` is ever reduced across rayon tasks: fitness values are produced
//! independently per individual and only ever *compared* (via `sort_by` /
//! `min_by`, both stable and order-deterministic), never summed.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::Arc;

use rand::RngExt;
use rand::SeedableRng;

use crate::error::EmlError;
use crate::tree::{EmlNode, EmlTree};

use super::discover::derive_seed;
use super::topology::build_leaves;
use super::{DiscoveredFormula, SymRegConfig, SymRegEngine};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type Rng = rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Individual: genome + cached fitness
// ─────────────────────────────────────────────────────────────────────────────

struct Individual {
    tree: EmlTree,
    formula: Option<DiscoveredFormula>,
}

impl Individual {
    fn new(tree: EmlTree) -> Self {
        Self {
            tree,
            formula: None,
        }
    }

    fn score(&self) -> f64 {
        self.formula.as_ref().map_or(f64::INFINITY, |f| f.score)
    }

    fn structural_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.tree.lower().simplify().structural_hash(&mut h);
        h.finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random tree generation
// ─────────────────────────────────────────────────────────────────────────────

fn random_tree(
    rng: &mut Rng,
    num_vars: usize,
    max_depth: usize,
    const_leaf: Option<f64>,
) -> EmlTree {
    let target_depth = rng.random_range(0..=max_depth);
    if target_depth == 0 || rng.random_range(0..2u32) == 0 {
        random_leaf_tree(rng, num_vars, const_leaf)
    } else {
        let node = random_node_at_depth(rng, target_depth, num_vars, const_leaf);
        EmlTree::from_node(node)
    }
}

fn random_leaf_tree(rng: &mut Rng, num_vars: usize, const_leaf: Option<f64>) -> EmlTree {
    let leaves = build_leaves(num_vars, const_leaf);
    let idx = rng.random_range(0..leaves.len());
    EmlTree::from_node(Arc::clone(&leaves[idx]))
}

fn random_node_at_depth(
    rng: &mut Rng,
    depth: usize,
    num_vars: usize,
    const_leaf: Option<f64>,
) -> Arc<EmlNode> {
    if depth == 0 {
        let leaves = build_leaves(num_vars, const_leaf);
        let idx = rng.random_range(0..leaves.len());
        return Arc::clone(&leaves[idx]);
    }
    let left = random_node_at_depth(rng, depth - 1, num_vars, const_leaf);
    let right = random_node_at_depth(rng, depth - 1, num_vars, const_leaf);
    Arc::new(EmlNode::Eml { left, right })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree manipulation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn count_nodes(node: &EmlNode) -> usize {
    match node {
        EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
        EmlNode::Eml { left, right } => 1 + count_nodes(left) + count_nodes(right),
    }
}

fn get_subtree(node: &Arc<EmlNode>, idx: usize) -> Option<Arc<EmlNode>> {
    let mut counter = idx;
    get_subtree_inner(node, &mut counter)
}

fn get_subtree_inner(node: &Arc<EmlNode>, counter: &mut usize) -> Option<Arc<EmlNode>> {
    if *counter == 0 {
        return Some(Arc::clone(node));
    }
    *counter -= 1;
    match node.as_ref() {
        EmlNode::Eml { left, right } => {
            get_subtree_inner(left, counter).or_else(|| get_subtree_inner(right, counter))
        }
        _ => None,
    }
}

fn replace_subtree(node: &Arc<EmlNode>, idx: usize, replacement: &Arc<EmlNode>) -> Arc<EmlNode> {
    let mut counter = idx;
    replace_subtree_inner(node, &mut counter, replacement)
}

fn replace_subtree_inner(
    node: &Arc<EmlNode>,
    counter: &mut usize,
    replacement: &Arc<EmlNode>,
) -> Arc<EmlNode> {
    if *counter == 0 {
        *counter = usize::MAX; // sentinel: done
        return Arc::clone(replacement);
    }
    if *counter == usize::MAX {
        return Arc::clone(node);
    }
    *counter -= 1;
    match node.as_ref() {
        EmlNode::Eml { left, right } => {
            let new_left = replace_subtree_inner(left, counter, replacement);
            let new_right = replace_subtree_inner(right, counter, replacement);
            Arc::new(EmlNode::Eml {
                left: new_left,
                right: new_right,
            })
        }
        _ => Arc::clone(node),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Genetic operators
// ─────────────────────────────────────────────────────────────────────────────

fn tournament_select(pop: &[Individual], k: usize, rng: &mut Rng) -> usize {
    let n = pop.len();
    if n == 0 {
        return 0;
    }
    let mut best_idx = rng.random_range(0..n);
    for _ in 1..k.max(1) {
        let idx = rng.random_range(0..n);
        if pop[idx].score() < pop[best_idx].score() {
            best_idx = idx;
        }
    }
    best_idx
}

fn crossover(a: &EmlTree, b: &EmlTree, max_depth: usize, rng: &mut Rng) -> Option<EmlTree> {
    let n_a = count_nodes(&a.root);
    let n_b = count_nodes(&b.root);
    if n_a == 0 || n_b == 0 {
        return None;
    }
    let idx_a = rng.random_range(0..n_a);
    let idx_b = rng.random_range(0..n_b);
    let subtree_b = get_subtree(&b.root, idx_b)?;
    let new_root = replace_subtree(&a.root, idx_a, &subtree_b);
    let new_tree = EmlTree::from_node(new_root);
    if new_tree.depth() > max_depth {
        None
    } else {
        Some(new_tree)
    }
}

fn mutate_point(
    tree: &EmlTree,
    num_vars: usize,
    const_leaf: Option<f64>,
    rng: &mut Rng,
) -> EmlTree {
    let leaves = build_leaves(num_vars, const_leaf);
    let new_leaf_idx = rng.random_range(0..leaves.len());
    let new_leaf = Arc::clone(&leaves[new_leaf_idx]);

    let n = count_nodes(&tree.root);
    let leaf_positions: Vec<usize> = (0..n)
        .filter(|&i| {
            matches!(
                get_subtree(&tree.root, i).as_deref(),
                Some(EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_))
            )
        })
        .collect();

    if leaf_positions.is_empty() {
        return tree.clone();
    }
    let pos = leaf_positions[rng.random_range(0..leaf_positions.len())];
    let new_root = replace_subtree(&tree.root, pos, &new_leaf);
    EmlTree::from_node(new_root)
}

fn mutate_subtree(
    tree: &EmlTree,
    num_vars: usize,
    max_depth: usize,
    const_leaf: Option<f64>,
    rng: &mut Rng,
) -> EmlTree {
    let n = count_nodes(&tree.root);
    if n == 0 {
        return tree.clone();
    }
    let pos = rng.random_range(0..n);
    let subtree_max = max_depth.saturating_sub(1);
    let depth_choice = rng.random_range(0..=subtree_max);
    let new_sub = random_node_at_depth(rng, depth_choice, num_vars, const_leaf);
    let new_root = replace_subtree(&tree.root, pos, &new_sub);
    let candidate = EmlTree::from_node(new_root);
    if candidate.depth() <= max_depth {
        candidate
    } else {
        tree.clone()
    }
}

fn mutate_const_jitter(tree: &EmlTree, rng: &mut Rng) -> EmlTree {
    let n = count_nodes(&tree.root);
    let const_positions: Vec<usize> = (0..n)
        .filter(|&i| {
            matches!(
                get_subtree(&tree.root, i).as_deref(),
                Some(EmlNode::Const(_))
            )
        })
        .collect();

    if const_positions.is_empty() {
        return tree.clone();
    }

    let pos = const_positions[rng.random_range(0..const_positions.len())];
    let old_val = match get_subtree(&tree.root, pos).as_deref() {
        Some(EmlNode::Const(v)) => *v,
        _ => return tree.clone(),
    };

    // Box-Muller Gaussian noise (sigma=0.1)
    let u1: f64 = rng.random_range(f64::EPSILON..1.0_f64);
    let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
    let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.1;
    let new_val = old_val + noise;
    let new_leaf = Arc::new(EmlNode::Const(new_val));
    let new_root = replace_subtree(&tree.root, pos, &new_leaf);
    EmlTree::from_node(new_root)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitness evaluation with caching — two-phase map-then-merge
// ─────────────────────────────────────────────────────────────────────────────

/// One fitness evaluation that could not be served from the pre-existing cache.
struct FitnessTask {
    /// Index into the population slice. Tasks are built in ascending `slot` order.
    slot: usize,
    /// Structural hash of the simplified tree — the fitness-cache key.
    hash: u64,
    /// Seed handed to `optimize_topology` as its `topology_idx`.
    ///
    /// Derived from `(island_seed, seed_base + slot)`, i.e. purely from indices.
    /// The optimizer's RNG is a function of this value, never of the worker
    /// thread, which is what makes the parallel map reproducible.
    topology_idx: usize,
    /// Index into `tasks` of the first task with the same `hash`, when this task
    /// is *not* that first one.
    ///
    /// The sequential loop gets duplicates for free: the first individual with a
    /// given hash populates the cache and every later one reads it back. We
    /// preserve both that *result* and that *work profile* by only fitting the
    /// leader, and falling back to fitting the followers themselves in the rare
    /// case where the leader's fit failed (a failed fit is never cached, so the
    /// sequential loop would have refitted the follower with its own seed).
    leader: Option<usize>,
}

/// Evaluate the fitness of every individual in `individuals` that does not
/// already carry a formula.
///
/// # Determinism
///
/// Three strictly separated phases:
///
/// 1. **Prepare (sequential).** Ascending slot order; the cache is only *read*.
///    Because the cache never loses entries, a hash present here would also have
///    hit in the fully sequential loop, so settling it now is exact.
/// 2. **Map (parallel under `feature = "parallel"`).** Pure `optimize_topology`
///    calls; `par_iter().map(...).collect()` over a slice is order-preserving.
/// 3. **Merge (sequential).** Ascending slot order again. The cache is written
///    here and only here, reproducing the intra-generation cache hits of the
///    sequential loop bit for bit.
///
/// No floating-point value is ever combined across tasks, so thread count cannot
/// perturb a single bit of the result.
fn evaluate_population(
    individuals: &mut [Individual],
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
    cache: &mut HashMap<u64, DiscoveredFormula>,
    island_seed: u64,
    seed_base: u64,
) {
    // ── Phase 1: PREPARE (sequential, read-only cache probe) ────────────────
    let mut tasks: Vec<FitnessTask> = Vec::new();
    let mut first_by_hash: HashMap<u64, usize> = HashMap::new();

    for (slot, ind) in individuals.iter_mut().enumerate() {
        if ind.formula.is_some() {
            continue;
        }
        let hash = ind.structural_hash();
        if let Some(cached) = cache.get(&hash) {
            ind.formula = Some(cached.clone());
            continue;
        }
        let topology_idx = derive_seed(island_seed, seed_base + slot as u64) as usize;
        let leader = first_by_hash.get(&hash).copied();
        if leader.is_none() {
            first_by_hash.insert(hash, tasks.len());
        }
        tasks.push(FitnessTask {
            slot,
            hash,
            topology_idx,
            leader,
        });
    }

    if tasks.is_empty() {
        return;
    }

    // ── Phase 2: MAP (rayon pool under `feature = "parallel"`) ──────────────
    // Immutable reborrow: the parallel region only *reads* the population.
    let population: &[Individual] = individuals;

    let mut results: Vec<Option<DiscoveredFormula>> = vec![None; tasks.len()];

    // Round A — one fit per distinct structural hash.
    let leaders: Vec<usize> = (0..tasks.len())
        .filter(|&i| tasks[i].leader.is_none())
        .collect();
    let leader_results = map_fitness(&leaders, &tasks, population, engine, inputs, targets);
    for (&task_idx, formula) in leaders.iter().zip(leader_results) {
        results[task_idx] = formula;
    }

    // Round B — followers whose leader failed to fit. A failed fit is never
    // cached, so the sequential loop would have refitted these with their own
    // seed; do exactly that. Usually empty (a `None` fit means a non-finite MSE).
    let stragglers: Vec<usize> = (0..tasks.len())
        .filter(|&i| match tasks[i].leader {
            Some(leader) => results[leader].is_none(),
            None => false,
        })
        .collect();
    if !stragglers.is_empty() {
        let straggler_results =
            map_fitness(&stragglers, &tasks, population, engine, inputs, targets);
        for (&task_idx, formula) in stragglers.iter().zip(straggler_results) {
            results[task_idx] = formula;
        }
    }

    // ── Phase 3: MERGE (sequential, ascending slot order) ───────────────────
    for (task, computed) in tasks.iter().zip(results) {
        // A same-hash task at a *lower* slot may just have populated the cache —
        // the sequential loop would have read it back instead of fitting. Honour
        // that, and drop this task's own (speculatively computed) result.
        if let Some(cached) = cache.get(&task.hash) {
            individuals[task.slot].formula = Some(cached.clone());
            continue;
        }
        if let Some(formula) = computed {
            cache.insert(task.hash, formula.clone());
            individuals[task.slot].formula = Some(formula);
        }
    }
}

/// Fit the tasks named by `indices` (indices into `tasks`) on the rayon pool.
///
/// `par_iter().map(...).collect()` over a slice is an indexed parallel iterator,
/// so the returned `Vec` is in `indices` order regardless of work stealing. Each
/// element is produced independently — rayon never reduces the `f64`s.
#[cfg(feature = "parallel")]
fn map_fitness(
    indices: &[usize],
    tasks: &[FitnessTask],
    population: &[Individual],
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Vec<Option<DiscoveredFormula>> {
    indices
        .par_iter()
        .map(|&i| {
            let task = &tasks[i];
            engine.optimize_topology(
                &population[task.slot].tree,
                inputs,
                targets,
                task.topology_idx,
            )
        })
        .collect()
}

/// Sequential twin of [`map_fitness`] — same order, same arithmetic, same bits.
#[cfg(not(feature = "parallel"))]
fn map_fitness(
    indices: &[usize],
    tasks: &[FitnessTask],
    population: &[Individual],
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Vec<Option<DiscoveredFormula>> {
    indices
        .iter()
        .map(|&i| {
            let task = &tasks[i];
            engine.optimize_topology(
                &population[task.slot].tree,
                inputs,
                targets,
                task.topology_idx,
            )
        })
        .collect()
}

/// Map a closure over all islands, on the rayon pool under `feature = "parallel"`.
///
/// Islands are fully independent (own RNG stream, own population, own cache), so
/// this is embarrassingly parallel. `collect()` restores source order, which
/// keeps the downstream "pick the global best" reduction — a `min_by` over
/// islands — order-deterministic.
#[cfg(feature = "parallel")]
fn map_islands<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(usize, T) -> R + Sync + Send,
{
    items
        .into_par_iter()
        .enumerate()
        .map(|(i, item)| f(i, item))
        .collect()
}

/// Sequential twin of [`map_islands`].
#[cfg(not(feature = "parallel"))]
fn map_islands<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(usize, T) -> R + Sync + Send,
{
    items
        .into_iter()
        .enumerate()
        .map(|(i, item)| f(i, item))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-island state
// ─────────────────────────────────────────────────────────────────────────────

struct IslandState {
    population: Vec<Individual>,
    cache: HashMap<u64, DiscoveredFormula>,
}

impl IslandState {
    fn new(
        pop_size: usize,
        num_vars: usize,
        max_depth: usize,
        const_leaf: Option<f64>,
        rng: &mut Rng,
    ) -> Self {
        let population = (0..pop_size)
            .map(|_| Individual::new(random_tree(rng, num_vars, max_depth, const_leaf)))
            .collect();
        Self {
            population,
            cache: HashMap::new(),
        }
    }

    fn best_formula(&self) -> Option<&DiscoveredFormula> {
        self.population
            .iter()
            .filter_map(|ind| ind.formula.as_ref())
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit-filter helper
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if `tree` passes the unit filter in `config`, or if no filter is set.
fn passes_unit_filter(tree: &EmlTree, config: &SymRegConfig) -> bool {
    if let Some((ref var_units, target_units)) = config.unit_filter {
        let lowered = tree.lower().simplify();
        matches!(lowered.check_units(var_units), Ok(u) if u == target_units)
    } else {
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-island GA
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_island(
    island_seed: u64,
    num_vars: usize,
    config: &SymRegConfig,
    inputs: &[Vec<f64>],
    targets: &[f64],
    engine: &SymRegEngine,
    population: usize,
    generations: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism: usize,
) -> IslandState {
    let mut rng = Rng::seed_from_u64(island_seed);
    let const_leaf = if config.enable_const_leaf {
        Some(config.const_leaf_init)
    } else {
        None
    };
    let max_depth = config.max_depth;

    let mut state = IslandState::new(population, num_vars, max_depth, const_leaf, &mut rng);

    // Initial evaluation (generation 0): seed base 0.
    evaluate_population(
        &mut state.population,
        engine,
        inputs,
        targets,
        &mut state.cache,
        island_seed,
        0,
    );

    for generation in 0..generations {
        let mut next_pop: Vec<Individual> = Vec::with_capacity(population);

        state.population.sort_by(|a, b| {
            a.score()
                .partial_cmp(&b.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 0..elitism.min(state.population.len()) {
            next_pop.push(Individual {
                tree: state.population[i].tree.clone(),
                formula: state.population[i].formula.clone(),
            });
        }

        while next_pop.len() < population {
            let slot = next_pop.len();
            let per_slot_seed = derive_seed(island_seed, (generation * population + slot) as u64);
            let mut slot_rng = Rng::seed_from_u64(per_slot_seed);

            let parent_a_idx = tournament_select(&state.population, tournament_size, &mut slot_rng);
            let parent_a = &state.population[parent_a_idx];

            let child_tree =
                if slot_rng.random::<f64>() < crossover_rate && state.population.len() > 1 {
                    let parent_b_idx =
                        tournament_select(&state.population, tournament_size, &mut slot_rng);
                    let parent_b = &state.population[parent_b_idx];
                    crossover(&parent_a.tree, &parent_b.tree, max_depth, &mut slot_rng)
                        .unwrap_or_else(|| parent_a.tree.clone())
                } else {
                    parent_a.tree.clone()
                };

            let pre_mutation_tree = child_tree.clone();

            let mutated = if slot_rng.random::<f64>() < mutation_rate {
                let mutation_type = slot_rng.random_range(0..3u32);
                match mutation_type {
                    0 => mutate_point(&child_tree, num_vars, const_leaf, &mut slot_rng),
                    1 => {
                        mutate_subtree(&child_tree, num_vars, max_depth, const_leaf, &mut slot_rng)
                    }
                    _ => mutate_const_jitter(&child_tree, &mut slot_rng),
                }
            } else {
                child_tree
            };

            // If unit filter is active and the mutated child fails it, fall back to pre-mutation tree.
            let final_child =
                if config.unit_filter.is_some() && !passes_unit_filter(&mutated, config) {
                    pre_mutation_tree
                } else {
                    mutated
                };

            next_pop.push(Individual::new(final_child));
        }

        let n_elite = elitism.min(population);
        // Two-phase parallel fitness: the non-elite slots of this generation.
        // Seed base keeps the per-slot `topology_idx` identical to the old
        // one-at-a-time loop, so seeded results are unchanged.
        let seed_base = ((generation + 1) * population + n_elite) as u64;
        evaluate_population(
            &mut next_pop[n_elite..],
            engine,
            inputs,
            targets,
            &mut state.cache,
            island_seed,
            seed_base,
        );

        state.population = next_pop;
    }

    state
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Run evolutionary symbolic regression with optional island populations.
///
/// When `n_islands == 1`, runs a single-population GA. When `n_islands > 1`,
/// runs multiple independent islands with ring migration every
/// `migration_interval` generations.
///
/// Islands are parallelized via rayon when the `parallel` feature is enabled;
/// otherwise they run sequentially.
#[allow(clippy::too_many_arguments)]
pub fn run_evolutionary(
    data_xs: &[Vec<f64>],
    data_ys: &[f64],
    config: &SymRegConfig,
    population: usize,
    generations: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism: usize,
    n_islands: usize,
    migration_interval: usize,
    migrants: usize,
) -> Result<DiscoveredFormula, EmlError> {
    if data_xs.is_empty() || data_ys.is_empty() {
        return Err(EmlError::EmptyData);
    }
    if data_xs.len() != data_ys.len() {
        return Err(EmlError::DimensionMismatch(data_xs.len(), data_ys.len()));
    }

    let num_vars = data_xs.first().map_or(0, |v| v.len());
    let n_islands = n_islands.max(1);
    let master_seed = config.seed.unwrap_or(42);

    let island_seeds: Vec<u64> = (0..n_islands)
        .map(|i| derive_seed(master_seed, i as u64))
        .collect();

    if n_islands == 1 || migration_interval == 0 {
        // Simple case: run all islands independently (no migration needed for single island)
        let states = run_islands_parallel(
            island_seeds,
            num_vars,
            config,
            data_xs,
            data_ys,
            population,
            generations,
            tournament_size,
            crossover_rate,
            mutation_rate,
            elitism,
        );

        states
            .iter()
            .filter_map(|s| s.best_formula().cloned())
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(EmlError::EmptyData)
    } else {
        // Multi-island with migration: run in epochs
        let epochs = generations.div_ceil(migration_interval);
        let gens_per_epoch = migration_interval;

        let const_leaf = if config.enable_const_leaf {
            Some(config.const_leaf_init)
        } else {
            None
        };
        let mut island_states: Vec<IslandState> = map_islands(island_seeds, |_, seed| {
            let mut rng = Rng::seed_from_u64(seed);
            let mut s =
                IslandState::new(population, num_vars, config.max_depth, const_leaf, &mut rng);
            let engine = SymRegEngine::new(config.clone());
            evaluate_population(
                &mut s.population,
                &engine,
                data_xs,
                data_ys,
                &mut s.cache,
                seed,
                0,
            );
            s
        });

        for epoch in 0..epochs {
            let actual_gens = if epoch == epochs - 1 {
                generations.saturating_sub(epoch * gens_per_epoch)
            } else {
                gens_per_epoch
            };

            let new_states: Vec<IslandState> = map_islands(island_states, |i, old_state| {
                let seed = derive_seed(master_seed, (epoch * n_islands + i) as u64);
                let engine = SymRegEngine::new(config.clone());
                run_island_from_state(
                    old_state,
                    seed,
                    num_vars,
                    config,
                    data_xs,
                    data_ys,
                    &engine,
                    actual_gens,
                    tournament_size,
                    crossover_rate,
                    mutation_rate,
                    elitism,
                )
            });

            island_states = new_states;

            if epoch + 1 < epochs {
                perform_ring_migration(&mut island_states, migrants);
            }
        }

        island_states
            .iter()
            .filter_map(|s| s.best_formula().cloned())
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(EmlError::EmptyData)
    }
}

/// Run islands on the rayon pool (or sequentially without `feature = "parallel"`).
#[allow(clippy::too_many_arguments)]
fn run_islands_parallel(
    island_seeds: Vec<u64>,
    num_vars: usize,
    config: &SymRegConfig,
    data_xs: &[Vec<f64>],
    data_ys: &[f64],
    population: usize,
    generations: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism: usize,
) -> Vec<IslandState> {
    map_islands(island_seeds, |_, seed| {
        let engine = SymRegEngine::new(config.clone());
        run_island(
            seed,
            num_vars,
            config,
            data_xs,
            data_ys,
            &engine,
            population,
            generations,
            tournament_size,
            crossover_rate,
            mutation_rate,
            elitism,
        )
    })
}

/// Continue evolving an existing island state for more generations.
#[allow(clippy::too_many_arguments)]
fn run_island_from_state(
    mut state: IslandState,
    island_seed: u64,
    num_vars: usize,
    config: &SymRegConfig,
    inputs: &[Vec<f64>],
    targets: &[f64],
    engine: &SymRegEngine,
    generations: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism: usize,
) -> IslandState {
    let const_leaf = if config.enable_const_leaf {
        Some(config.const_leaf_init)
    } else {
        None
    };
    let max_depth = config.max_depth;
    let population = state.population.len();

    for generation in 0..generations {
        let mut next_pop: Vec<Individual> = Vec::with_capacity(population);

        state.population.sort_by(|a, b| {
            a.score()
                .partial_cmp(&b.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 0..elitism.min(state.population.len()) {
            next_pop.push(Individual {
                tree: state.population[i].tree.clone(),
                formula: state.population[i].formula.clone(),
            });
        }

        while next_pop.len() < population {
            let slot = next_pop.len();
            let per_slot_seed = derive_seed(island_seed, (generation * population + slot) as u64);
            let mut slot_rng = Rng::seed_from_u64(per_slot_seed);

            let parent_a_idx = tournament_select(&state.population, tournament_size, &mut slot_rng);
            let parent_a = &state.population[parent_a_idx];

            let child_tree =
                if slot_rng.random::<f64>() < crossover_rate && state.population.len() > 1 {
                    let parent_b_idx =
                        tournament_select(&state.population, tournament_size, &mut slot_rng);
                    let parent_b = &state.population[parent_b_idx];
                    crossover(&parent_a.tree, &parent_b.tree, max_depth, &mut slot_rng)
                        .unwrap_or_else(|| parent_a.tree.clone())
                } else {
                    parent_a.tree.clone()
                };

            let pre_mutation_tree = child_tree.clone();

            let mutated = if slot_rng.random::<f64>() < mutation_rate {
                let mutation_type = slot_rng.random_range(0..3u32);
                match mutation_type {
                    0 => mutate_point(&child_tree, num_vars, const_leaf, &mut slot_rng),
                    1 => {
                        mutate_subtree(&child_tree, num_vars, max_depth, const_leaf, &mut slot_rng)
                    }
                    _ => mutate_const_jitter(&child_tree, &mut slot_rng),
                }
            } else {
                child_tree
            };

            // If unit filter is active and the mutated child fails it, fall back to pre-mutation tree.
            let final_child =
                if config.unit_filter.is_some() && !passes_unit_filter(&mutated, config) {
                    pre_mutation_tree
                } else {
                    mutated
                };

            next_pop.push(Individual::new(final_child));
        }

        let n_elite = elitism.min(population);
        // Two-phase parallel fitness: the non-elite slots of this generation.
        // Seed base keeps the per-slot `topology_idx` identical to the old
        // one-at-a-time loop, so seeded results are unchanged.
        let seed_base = ((generation + 1) * population + n_elite) as u64;
        evaluate_population(
            &mut next_pop[n_elite..],
            engine,
            inputs,
            targets,
            &mut state.cache,
            island_seed,
            seed_base,
        );

        state.population = next_pop;
    }

    state
}

/// Perform ring migration: top `migrants` from island i go to (i+1)%n.
fn perform_ring_migration(islands: &mut [IslandState], migrants: usize) {
    if islands.len() <= 1 || migrants == 0 {
        return;
    }
    let n = islands.len();
    let all_migrants: Vec<Vec<Individual>> = islands
        .iter_mut()
        .map(|island| {
            island.population.sort_by(|a, b| {
                a.score()
                    .partial_cmp(&b.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let m = migrants.min(island.population.len());
            island.population[..m]
                .iter()
                .map(|ind| Individual {
                    tree: ind.tree.clone(),
                    formula: ind.formula.clone(),
                })
                .collect()
        })
        .collect();

    for (i, island_migrants) in all_migrants.iter().enumerate() {
        let dest = (i + 1) % n;
        let dest_pop = &mut islands[dest].population;

        dest_pop.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let replace_count = island_migrants.len().min(dest_pop.len());
        for (j, migrant) in island_migrants[..replace_count].iter().enumerate() {
            if migrant.score() < dest_pop[j].score() {
                dest_pop[j] = Individual {
                    tree: migrant.tree.clone(),
                    formula: migrant.formula.clone(),
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symreg::SymRegConfig;

    #[test]
    fn test_evolutionary_determinism() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * 2.0 + 1.0).collect();
        let config = SymRegConfig {
            max_depth: 2,
            seed: Some(42),
            ..SymRegConfig::quick()
        };
        let r1 = run_evolutionary(&inputs, &targets, &config, 10, 5, 3, 0.7, 0.2, 1, 1, 0, 0);
        let r2 = run_evolutionary(&inputs, &targets, &config, 10, 5, 3, 0.7, 0.2, 1, 1, 0, 0);
        let mse1 = r1.expect("run 1 should succeed").mse;
        let mse2 = r2.expect("run 2 should succeed").mse;
        assert!(
            (mse1 - mse2).abs() < 1e-12,
            "MSEs must match: {mse1} vs {mse2}"
        );
    }

    #[test]
    fn test_crossover_respects_max_depth() {
        let config = SymRegConfig {
            max_depth: 2,
            seed: Some(99),
            ..SymRegConfig::quick()
        };
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect();
        let r = run_evolutionary(&inputs, &targets, &config, 8, 3, 2, 0.9, 0.3, 1, 1, 0, 0);
        let formula = r.expect("should succeed");
        assert!(
            formula.eml_tree.depth() <= 2,
            "depth must not exceed max_depth=2"
        );
    }

    /// Seeded evolutionary runs must agree on every *bit*, not merely to 1e-12.
    #[test]
    fn evolutionary_repeat_run_is_bit_identical() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * 2.0 + 1.0).collect();
        let config = SymRegConfig {
            max_depth: 2,
            seed: Some(42),
            ..SymRegConfig::quick()
        };
        let a = run_evolutionary(&inputs, &targets, &config, 12, 4, 3, 0.7, 0.3, 1, 1, 0, 0)
            .expect("run 1 should succeed");
        let b = run_evolutionary(&inputs, &targets, &config, 12, 4, 3, 0.7, 0.3, 1, 1, 0, 0)
            .expect("run 2 should succeed");
        assert_eq!(a.mse.to_bits(), b.mse.to_bits(), "MSE bits must match");
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "score bits must match"
        );
        assert_eq!(a.pretty, b.pretty, "pretty form must match");
        let pa: Vec<u64> = a.params.iter().map(|p| p.to_bits()).collect();
        let pb: Vec<u64> = b.params.iter().map(|p| p.to_bits()).collect();
        assert_eq!(pa, pb, "parameter bits must match");
    }

    /// Build a population and the corresponding fitness task list, mirroring
    /// what the prepare phase of [`evaluate_population`] produces.
    #[cfg(feature = "parallel")]
    fn make_fitness_fixture() -> (Vec<Individual>, Vec<FitnessTask>, SymRegConfig) {
        let config = SymRegConfig {
            max_depth: 2,
            seed: Some(42),
            ..SymRegConfig::quick()
        };
        let mut rng = Rng::seed_from_u64(1234);
        let population: Vec<Individual> = (0..24)
            .map(|_| Individual::new(random_tree(&mut rng, 1, config.max_depth, None)))
            .collect();
        let tasks: Vec<FitnessTask> = population
            .iter()
            .enumerate()
            .map(|(slot, ind)| FitnessTask {
                slot,
                hash: ind.structural_hash(),
                topology_idx: derive_seed(7, slot as u64) as usize,
                leader: None,
            })
            .collect();
        (population, tasks, config)
    }

    /// **The parallel == sequential proof at the map level.**
    ///
    /// `map_fitness` is the only function whose body differs between the
    /// `parallel` and non-`parallel` builds of the GA. Here the rayon version and
    /// the literal body of its sequential twin are run against the same
    /// population and asserted bit-equal.
    #[cfg(feature = "parallel")]
    #[test]
    fn map_fitness_parallel_equals_sequential_bitwise() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * 2.0 + 1.0).collect();
        let (population, tasks, config) = make_fitness_fixture();
        let engine = SymRegEngine::new(config);
        let indices: Vec<usize> = (0..tasks.len()).collect();

        let parallel = map_fitness(&indices, &tasks, &population, &engine, &inputs, &targets);

        // Byte-for-byte the body of the `#[cfg(not(feature = "parallel"))]` twin.
        let sequential: Vec<Option<DiscoveredFormula>> = indices
            .iter()
            .map(|&i| {
                let task = &tasks[i];
                engine.optimize_topology(
                    &population[task.slot].tree,
                    &inputs,
                    &targets,
                    task.topology_idx,
                )
            })
            .collect();

        assert_eq!(parallel.len(), sequential.len());
        for (k, (p, s)) in parallel.iter().zip(sequential.iter()).enumerate() {
            match (p, s) {
                (Some(pf), Some(sf)) => {
                    assert_eq!(
                        pf.mse.to_bits(),
                        sf.mse.to_bits(),
                        "slot {k}: MSE bits differ ({} vs {})",
                        pf.mse,
                        sf.mse
                    );
                    let pp: Vec<u64> = pf.params.iter().map(|v| v.to_bits()).collect();
                    let sp: Vec<u64> = sf.params.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(pp, sp, "slot {k}: parameter bits differ");
                }
                (None, None) => {}
                _ => panic!("slot {k}: fit success differs between parallel and sequential"),
            }
        }
    }

    /// Thread-count invariance of the GA fitness map: 1, 2, 3 and 8 workers must
    /// all yield the identical bit pattern.
    #[cfg(feature = "parallel")]
    #[test]
    fn map_fitness_is_thread_count_invariant() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * 2.0 + 1.0).collect();
        let (population, tasks, config) = make_fitness_fixture();
        let engine = SymRegEngine::new(config);
        let indices: Vec<usize> = (0..tasks.len()).collect();

        let run_with = |threads: usize| -> Vec<Option<u64>> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool should build");
            pool.install(|| {
                map_fitness(&indices, &tasks, &population, &engine, &inputs, &targets)
                    .iter()
                    .map(|f| f.as_ref().map(|f| f.mse.to_bits()))
                    .collect()
            })
        };

        let reference = run_with(1);
        for threads in [2usize, 3, 8] {
            assert_eq!(
                run_with(threads),
                reference,
                "MSE bits changed with {threads} rayon threads"
            );
        }
    }
}
