//! Evolutionary symbolic regression: genetic algorithm with island populations.
//!
//! Implements tournament selection, subtree crossover, and multiple mutation
//! types. Island populations evolve in parallel (requires the `parallel` feature)
//! or sequentially, with ring migration between islands.

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
// Fitness evaluation with caching
// ─────────────────────────────────────────────────────────────────────────────

fn evaluate_individual(
    ind: &mut Individual,
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
    topology_idx: usize,
    cache: &mut HashMap<u64, DiscoveredFormula>,
) {
    if ind.formula.is_some() {
        return;
    }
    let hash = ind.structural_hash();
    if let Some(f) = cache.get(&hash) {
        ind.formula = Some(f.clone());
        return;
    }
    if let Some(f) = engine.optimize_topology(&ind.tree, inputs, targets, topology_idx) {
        cache.insert(hash, f.clone());
        ind.formula = Some(f);
    }
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

    // Initial evaluation
    for (i, ind) in state.population.iter_mut().enumerate() {
        let slot_seed = derive_seed(island_seed, i as u64);
        evaluate_individual(
            ind,
            engine,
            inputs,
            targets,
            slot_seed as usize,
            &mut state.cache,
        );
    }

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
        for (i, ind) in next_pop[n_elite..].iter_mut().enumerate() {
            let slot_seed = derive_seed(
                island_seed,
                ((generation + 1) * population + n_elite + i) as u64,
            );
            evaluate_individual(
                ind,
                engine,
                inputs,
                targets,
                slot_seed as usize,
                &mut state.cache,
            );
        }

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
            &island_seeds,
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
        let mut island_states: Vec<IslandState> = island_seeds
            .iter()
            .enumerate()
            .map(|(i, &seed)| {
                let mut rng = Rng::seed_from_u64(seed);
                let mut s =
                    IslandState::new(population, num_vars, config.max_depth, const_leaf, &mut rng);
                let engine = SymRegEngine::new(config.clone());
                for (j, ind) in s.population.iter_mut().enumerate() {
                    let slot_seed = derive_seed(seed, j as u64);
                    evaluate_individual(
                        ind,
                        &engine,
                        data_xs,
                        data_ys,
                        slot_seed as usize,
                        &mut s.cache,
                    );
                }
                let _ = i; // suppress unused warning
                s
            })
            .collect();

        for epoch in 0..epochs {
            let actual_gens = if epoch == epochs - 1 {
                generations.saturating_sub(epoch * gens_per_epoch)
            } else {
                gens_per_epoch
            };

            let new_states: Vec<IslandState> = island_states
                .into_iter()
                .enumerate()
                .map(|(i, old_state)| {
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
                })
                .collect();

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

/// Run islands in parallel (or sequentially).
#[allow(clippy::too_many_arguments)]
fn run_islands_parallel(
    island_seeds: &[u64],
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
    #[cfg(feature = "parallel")]
    {
        island_seeds
            .par_iter()
            .map(|&seed| {
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
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        island_seeds
            .iter()
            .map(|&seed| {
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
            .collect()
    }
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
        for (i, ind) in next_pop[n_elite..].iter_mut().enumerate() {
            let slot_seed = derive_seed(
                island_seed,
                ((generation + 1) * population + n_elite + i) as u64,
            );
            evaluate_individual(
                ind,
                engine,
                inputs,
                targets,
                slot_seed as usize,
                &mut state.cache,
            );
        }

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
}
