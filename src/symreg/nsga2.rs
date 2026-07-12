//! NSGA-II multi-objective symbolic regression (Deb, Pratap, Agarwal & Meyarivan, 2002).
//!
//! Where [`SymRegEngine::discover`](super::SymRegEngine::discover) collapses the
//! search onto one scalar score and
//! [`pareto_front`](super::pareto_front) merely *filters* an
//! already-computed pool, this module runs the genuine NSGA-II loop and returns
//! the whole population annotated with its **non-domination rank** and its
//! **crowding distance** — [`RankedFormula`].
//!
//! # The algorithm
//!
//! Two objectives, both **minimised**:
//!
//! | index | objective    | source                                |
//! |-------|--------------|---------------------------------------|
//! | 0     | MSE          | [`DiscoveredFormula::mse`]            |
//! | 1     | complexity   | [`DiscoveredFormula::complexity`]     |
//!
//! Nothing below is specialised to `M = 2`, though: every routine takes an
//! arbitrary-length objective vector, so the module is a complete, reusable
//! NSGA-II core.
//!
//! ## 1. Fast non-dominated sort — `O(M·N²)` time, `O(N²)` space
//!
//! [`fast_nondominated_sort`] is Deb et al.'s Algorithm 1 verbatim, *not* a naive
//! "scan the remaining pool once per front" loop (which would be `O(M·N³)`):
//!
//! ```text
//! for each p in P:                       ── O(M·N²) total
//!     S_p = { q : p ≺ q }                   the set p dominates
//!     n_p = |{ q : q ≺ p }|                 how many dominate p
//!     if n_p == 0: rank(p) = 0, F₀ ∪= {p}
//!
//! i = 0                                  ── peeling, O(N²) total, because every
//! while F_i ≠ ∅:                            edge (p,q) ∈ S_p is relaxed exactly once
//!     H = ∅
//!     for p in F_i, for q in S_p:
//!         n_q -= 1
//!         if n_q == 0: rank(q) = i+1, H ∪= {q}
//!     i += 1; F_i = H
//! ```
//!
//! The domination-count / dominated-set bookkeeping is what buys the `O(M·N²)`
//! bound: the peeling phase never re-tests a domination relation.
//!
//! ## 2. Crowding distance
//!
//! [`crowding_distance`] follows Deb's Algorithm 2, with two explicitly
//! documented edge-case decisions:
//!
//! * **Boundary solutions get `+∞`.** For each objective the front is sorted and
//!   the extreme members are given an infinite distance so that the extremes of
//!   the trade-off curve always survive truncation. A front of size ≤ 2 consists
//!   of nothing *but* boundary members, so every member gets `+∞`.
//! * **A degenerate objective is skipped, never divided by.** If an objective's
//!   `(max − min)` is `0` — or is not finite, which is what a `+∞` objective
//!   value produces — that objective contributes nothing: no division, and no
//!   boundary marking either. Marking boundaries on a fully-tied objective would
//!   award `+∞` to two *arbitrary* tie-broken members and corrupt the diversity
//!   signal with index noise. Deb's pseudocode is silent on the degenerate case;
//!   this is the deliberate refinement.
//!
//! Consequence: a crowding distance is **never `NaN`**. Whenever an objective is
//! used, its range is finite and strictly positive, hence every value in the
//! front is finite for that objective and every increment `(next − prev) / range`
//! is finite. Accumulating finite increments onto a `+∞` boundary keeps `+∞`.
//!
//! ## 3. Crowded-comparison operator `≺ₙ`
//!
//! [`crowded_compare`]: lower rank wins; on a rank tie, *larger* crowding wins
//! (prefer the lonelier solution). Implemented with [`f64::total_cmp`], so it is
//! a total order on **every** input, `NaN` included — an inconsistent comparator
//! can make Rust's sort panic or silently produce garbage, and this one cannot be
//! inconsistent.
//!
//! ## 4. (μ+λ) selection
//!
//! [`environmental_selection`] implements the elitist survivor step: parents `P`
//! (μ) and offspring `Q` (λ = μ) are merged into `R = P ∪ Q`, `R` is
//! fast-non-dominated-sorted, and fronts are copied into the next generation
//! whole until one no longer fits. That last front is truncated by *descending*
//! crowding distance. Because `R` contains the parents, no non-dominated solution
//! can ever be lost.
//!
//! # `NaN` objectives are the worst possible value
//!
//! A formula whose MSE is `NaN` is treated as **maximally bad**: every objective
//! is passed through [`sanitize_objective`], which maps `NaN ↦ +∞`. This is done
//! at *both* the entry point ([`objective_vector`]) and inside the public
//! primitives, so an externally-supplied objective vector cannot poison the
//! ordering either.
//!
//! "Worst" means *worst on that axis*, not *universally dominated* — dominance is
//! a vector relation. A `NaN`-MSE formula loses to every finite fit that is no
//! more complex, which is the whole point; but if it is strictly the simplest
//! member of the pool then nothing dominates it and it stays on the front,
//! behaving exactly as an honest `+∞`-MSE model would. What is ruled out is the
//! pathology: `NaN` can no longer make a comparison neither-true-nor-false and
//! thereby produce an intransitive ordering.
//!
//! **Documented divergence from [`pareto_front`](super::pareto_front).** The
//! existing free function tests `self.mse <= other.mse`, and *any* comparison
//! with `NaN` is `false` — so a `NaN`-MSE formula is vacuously non-dominated and
//! `pareto_front` keeps it on the front. NSGA-II here does the opposite: `NaN` is
//! the worst value, so such a formula is dominated by anything with a finite MSE
//! and no worse complexity, and it sinks to the last rank. On a `NaN`-free pool —
//! i.e. every pool the optimiser actually produces, since
//! `optimize_topology` rejects non-finite fits — the two agree **exactly**:
//! rank 0 is, member for member, `pareto_front`'s output.
//!
//! # Determinism
//!
//! Same discipline as `evolution.rs` / `mcts.rs`:
//!
//! * The only rayon call site is [`map_fits`], a pure *map*: `par_iter().map(…)
//!   .collect()` over an indexed slice, whose output order is the input order
//!   regardless of work stealing. Its `#[cfg(not(feature = "parallel"))]` twin has
//!   a byte-identical body modulo `par_iter` → `iter`.
//! * **No `f64` is ever reduced across rayon tasks.** Objective values are only
//!   ever produced independently and *compared*; they are never summed.
//! * Every RNG is seeded from indices alone — [`stream_seed`] mixes
//!   `(master_seed, generation, tag, slot)` through SplitMix64 — never from a
//!   shared stream whose consumption order would depend on thread scheduling.
//! * Merges are sequential and in ascending index order.
//!
//! Result: `parallel == sequential`, bit for bit, for any thread count.

use std::cmp::Ordering;
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

/// Number of objectives minimised by [`objective_vector`]: `[mse, complexity]`.
pub const NSGA2_NUM_OBJECTIVES: usize = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A [`DiscoveredFormula`] annotated with its NSGA-II non-domination rank and
/// crowding distance.
///
/// * `rank == 0` is the non-dominated (Pareto) front. On a pool with no `NaN`
///   objectives it contains exactly the formulas that
///   [`pareto_front`](super::pareto_front) returns.
/// * `crowding` is `+∞` for the boundary members of a front (see the module
///   docs) and a finite non-negative number otherwise. It is never `NaN`.
///
/// `crowding` may be `+∞`, which has no JSON representation, so this type
/// deliberately does **not** derive `serde` — serialise the inner
/// [`RankedFormula::formula`] instead, together with the two annotations.
#[derive(Clone, Debug)]
pub struct RankedFormula {
    /// The formula itself.
    pub formula: DiscoveredFormula,
    /// Non-domination rank: `0` = Pareto front, `1` = front after removing rank 0, …
    pub rank: usize,
    /// Crowding distance within the formula's own front. `+∞` for boundaries.
    pub crowding: f64,
}

impl RankedFormula {
    /// The crowded-comparison operator `≺ₙ` applied to two ranked formulas.
    ///
    /// [`Ordering::Less`] means `self` is *better*, so a `sort_by` using this
    /// yields best-first order.
    pub fn crowded_cmp(&self, other: &Self) -> Ordering {
        crowded_compare(self.rank, self.crowding, other.rank, other.crowding)
    }
}

/// Hyper-parameters of the NSGA-II generational loop.
///
/// The search-space shape (`max_depth`, the `Const` leaf, the unit filter) and
/// the per-topology optimiser still come from
/// [`SymRegConfig`](super::SymRegConfig); this struct only configures the
/// evolutionary layer on top of it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Nsga2Config {
    /// μ — the parent population size. λ (the offspring count) equals μ.
    pub population: usize,
    /// Number of (μ+λ) generations to run.
    pub generations: usize,
    /// Tournament size for crowded-comparison mating selection. Clamped to ≥ 2.
    pub tournament_size: usize,
    /// Probability of subtree crossover, in `[0, 1]`.
    pub crossover_rate: f64,
    /// Probability of mutating an offspring, in `[0, 1]`.
    pub mutation_rate: f64,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population: 48,
            generations: 20,
            tournament_size: 2,
            crossover_rate: 0.9,
            mutation_rate: 0.2,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Objectives
// ─────────────────────────────────────────────────────────────────────────────

/// Map a raw objective value onto the value NSGA-II actually compares.
///
/// `NaN ↦ +∞`: a formula whose MSE could not be computed is the **worst**
/// possible solution, not an incomparable one. Everything else is returned
/// unchanged (`+∞` itself included — an infinitely bad fit is exactly as bad as
/// a `NaN` one).
///
/// This is what makes every comparator in this module a consistent (total,
/// antisymmetric, transitive) order: after sanitisation there is no `NaN` left
/// to violate trichotomy.
#[inline]
pub fn sanitize_objective(value: f64) -> f64 {
    if value.is_nan() { f64::INFINITY } else { value }
}

/// The minimised objective vector of a formula: `[mse, complexity]`.
///
/// A `NaN` MSE is sanitised to `+∞` (worst) — see [`sanitize_objective`].
pub fn objective_vector(formula: &DiscoveredFormula) -> Vec<f64> {
    vec![sanitize_objective(formula.mse), formula.complexity as f64]
}

/// [`objective_vector`] lifted over a whole pool, preserving order.
pub fn objective_vectors(pool: &[DiscoveredFormula]) -> Vec<Vec<f64>> {
    pool.iter().map(objective_vector).collect()
}

/// Pareto dominance on minimised objective vectors: `a ≺ b`.
///
/// `a` dominates `b` iff `a` is no worse on **every** objective and strictly
/// better on **at least one**. Values are compared after [`sanitize_objective`],
/// so `NaN` behaves as `+∞` (worst) and the relation stays a strict partial
/// order — irreflexive, asymmetric and transitive — which is precisely what
/// [`fast_nondominated_sort`] needs to terminate with correct ranks.
///
/// Objectives are zipped, so only `min(a.len(), b.len())` of them are compared;
/// callers are expected to pass equal-length vectors.
///
/// On finite `[mse, complexity]` vectors this agrees, term for term, with
/// [`DiscoveredFormula::dominates`].
pub fn dominates_objectives(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better_somewhere = false;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = sanitize_objective(x);
        let y = sanitize_objective(y);
        if x > y {
            return false;
        }
        if x < y {
            strictly_better_somewhere = true;
        }
    }
    strictly_better_somewhere
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast non-dominated sort (Deb et al. 2002, Algorithm 1)
// ─────────────────────────────────────────────────────────────────────────────

/// Fast non-dominated sort — the real `O(M·N²)` algorithm.
///
/// Returns the fronts: `fronts[0]` is the non-dominated (rank-0) set, `fronts[1]`
/// is what becomes non-dominated once rank 0 is removed, and so on. Every index
/// in `0..objectives.len()` appears in exactly one front, and the indices inside
/// each front are sorted ascending, so the output is canonical and does not
/// depend on iteration accidents.
///
/// # Algorithm
///
/// Phase 1 (`O(M·N²)`) computes, for every `p`, the set `S_p` of solutions `p`
/// dominates and the count `n_p` of solutions that dominate `p`; the `n_p == 0`
/// solutions form front 0. Phase 2 (`O(N²)`, since each `(p, q) ∈ S_p` edge is
/// relaxed exactly once over the whole run) peels the fronts off by decrementing
/// `n_q` for every `q` dominated by a member of the current front, and promoting
/// `q` the moment its count reaches zero.
///
/// A naive implementation — "re-scan the survivors for non-dominated points, N
/// times" — is `O(M·N³)`. This is not that.
pub fn fast_nondominated_sort(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    if n == 0 {
        return Vec::new();
    }

    // S_p: the solutions dominated by p.
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    // n_p: how many solutions dominate p.
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut current: Vec<usize> = Vec::new();

    // ── Phase 1: domination counts and dominated sets — O(M·N²) ──────────────
    for (p, obj_p) in objectives.iter().enumerate() {
        for (q, obj_q) in objectives.iter().enumerate() {
            if p == q {
                continue;
            }
            if dominates_objectives(obj_p, obj_q) {
                dominated_set[p].push(q);
            } else if dominates_objectives(obj_q, obj_p) {
                domination_count[p] += 1;
            }
        }
        if domination_count[p] == 0 {
            current.push(p);
        }
    }

    // ── Phase 2: peel the fronts — O(N²) over the whole loop ─────────────────
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    while !current.is_empty() {
        let mut next: Vec<usize> = Vec::new();
        for &p in &current {
            for &q in &dominated_set[p] {
                // Dominance is a strict partial order, so `domination_count[q]`
                // is > 0 here; the guard makes the decrement total anyway, which
                // rules out both underflow and a double promotion.
                if domination_count[q] > 0 {
                    domination_count[q] -= 1;
                    if domination_count[q] == 0 {
                        next.push(q);
                    }
                }
            }
        }
        next.sort_unstable();
        fronts.push(current);
        current = next;
    }

    fronts
}

// ─────────────────────────────────────────────────────────────────────────────
// Crowding distance (Deb et al. 2002, Algorithm 2)
// ─────────────────────────────────────────────────────────────────────────────

/// Crowding distance of every member of one front.
///
/// The returned vector is aligned with `front`: `result[k]` is the distance of
/// the solution `front[k]`.
///
/// # Edge cases (all deliberate, all tested)
///
/// * **`|front| ≤ 2`** — every member is a boundary member in every objective, so
///   every member gets `+∞`.
/// * **Boundary members get `+∞`.** Per objective, the front is sorted by that
///   objective and the two extremes are pinned to `+∞`.
/// * **Degenerate objective ⇒ skip, never divide.** If `(max − min)` is `0`, or
///   is not finite (which is what a `+∞` objective value — e.g. a sanitised `NaN`
///   MSE — produces), the objective is skipped entirely: no division by zero, and
///   no boundary marking, because a fully-tied objective has no boundary and the
///   "extremes" would be pure tie-break noise.
/// * **Never `NaN`.** Whenever an objective *is* used its range is finite and
///   positive, hence every value in the front is finite for that objective and
///   every increment is finite.
///
/// Ties inside an objective's sort are broken by ascending solution index, which
/// makes the result independent of the input's incidental ordering.
pub fn crowding_distance(objectives: &[Vec<f64>], front: &[usize]) -> Vec<f64> {
    let l = front.len();
    if l == 0 {
        return Vec::new();
    }
    if l <= 2 {
        // Both (or the sole) members are simultaneously the min and the max of
        // every objective: they are boundary solutions by definition.
        return vec![f64::INFINITY; l];
    }

    let n_obj = front
        .iter()
        .map(|&i| objectives[i].len())
        .min()
        .unwrap_or(0);

    let mut distance = vec![0.0_f64; l];
    let value_at =
        |position: usize, m: usize| -> f64 { sanitize_objective(objectives[front[position]][m]) };

    for m in 0..n_obj {
        // Positions into `front`, sorted by objective `m`. `total_cmp` is a total
        // order for every f64, so this comparator can never be inconsistent.
        let mut order: Vec<usize> = (0..l).collect();
        order.sort_by(|&a, &b| {
            value_at(a, m)
                .total_cmp(&value_at(b, m))
                .then_with(|| front[a].cmp(&front[b]))
        });

        let f_min = value_at(order[0], m);
        let f_max = value_at(order[l - 1], m);
        let range = f_max - f_min;

        // Explicit skip — this is the div-by-zero guard. A non-finite range is
        // caught by the same test (`INFINITY - x` is not finite, `INF - INF` is
        // `NaN`, and `NaN > 0.0` is false).
        if !range.is_finite() || range <= 0.0 {
            continue;
        }

        // Boundary solutions of this objective survive truncation unconditionally.
        distance[order[0]] = f64::INFINITY;
        distance[order[l - 1]] = f64::INFINITY;

        // Interior: normalised distance between the two neighbours. `range` is
        // finite and > 0, so `f_min` and `f_max` are finite, so every value in
        // between is finite and this increment can never be `NaN`.
        for k in 1..l - 1 {
            let next_value = value_at(order[k + 1], m);
            let prev_value = value_at(order[k - 1], m);
            distance[order[k]] += (next_value - prev_value) / range;
        }
    }

    distance
}

// ─────────────────────────────────────────────────────────────────────────────
// Crowded-comparison operator
// ─────────────────────────────────────────────────────────────────────────────

/// The crowded-comparison operator `≺ₙ`.
///
/// `a ≺ₙ b` iff `rank(a) < rank(b)`, or the ranks tie and `crowding(a) >
/// crowding(b)` — i.e. among equally non-dominated solutions the one in the
/// sparser region of objective space wins.
///
/// Returns [`Ordering::Less`] when `a` is **better**, so `sort_by` with this
/// comparator produces best-first order.
///
/// Crowding is compared with [`f64::total_cmp`]: the comparator is a total order
/// on all inputs, `NaN` and `±∞` included. It therefore satisfies the contract of
/// `slice::sort_by` unconditionally — no panic, no garbage ordering — even if a
/// caller hands it a `NaN` crowding value that this module itself cannot produce.
pub fn crowded_compare(rank_a: usize, crowding_a: f64, rank_b: usize, crowding_b: f64) -> Ordering {
    rank_a
        .cmp(&rank_b)
        .then_with(|| crowding_b.total_cmp(&crowding_a))
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking + (μ+λ) environmental selection
// ─────────────────────────────────────────────────────────────────────────────

/// Rank and crowding of every solution, aligned with `objectives`.
///
/// `result[i] = (rank, crowding)` for solution `i`. Runs one
/// [`fast_nondominated_sort`] plus one [`crowding_distance`] per front.
pub fn rank_and_crowding(objectives: &[Vec<f64>]) -> Vec<(usize, f64)> {
    let mut meta = vec![(0_usize, 0.0_f64); objectives.len()];
    for (rank, front) in fast_nondominated_sort(objectives).iter().enumerate() {
        let distance = crowding_distance(objectives, front);
        for (k, &idx) in front.iter().enumerate() {
            meta[idx] = (rank, distance[k]);
        }
    }
    meta
}

/// (μ+λ) environmental selection: the indices of the `mu` survivors.
///
/// Fronts are taken whole, best first, for as long as they fit. The first front
/// that does *not* fit is truncated: its members are sorted by **descending**
/// crowding distance (ties broken by ascending index, so the result is
/// deterministic) and the best `mu − |selected|` are taken.
///
/// Call this on the objectives of the merged parent+offspring pool `R = P ∪ Q`;
/// because `R` contains the parents, the step is elitist — a rank-0 solution can
/// never be lost.
///
/// The returned vector has length `min(mu, objectives.len())` and holds distinct
/// indices into `objectives`.
pub fn environmental_selection(objectives: &[Vec<f64>], mu: usize) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::with_capacity(mu.min(objectives.len()));
    for front in fast_nondominated_sort(objectives) {
        if selected.len() + front.len() <= mu {
            selected.extend_from_slice(&front);
            continue;
        }
        let remaining = mu.saturating_sub(selected.len());
        if remaining == 0 {
            break;
        }
        let distance = crowding_distance(objectives, &front);
        let mut order: Vec<usize> = (0..front.len()).collect();
        order.sort_by(|&a, &b| {
            distance[b]
                .total_cmp(&distance[a])
                .then_with(|| front[a].cmp(&front[b]))
        });
        selected.extend(order.iter().take(remaining).map(|&k| front[k]));
        break;
    }
    selected
}

/// Annotate a pool of formulas with `(rank, crowding)` and sort it best-first.
///
/// This is the pure "rank an existing pool" entry point — no search, no fitting.
/// The result is sorted by the crowded-comparison operator [`crowded_compare`],
/// with `(complexity, mse, pretty)` as a deterministic final tie-break, so the
/// output order never depends on the incidental order of `pool`.
///
/// On a pool with no `NaN` MSE, `result.iter().filter(|r| r.rank == 0)` is
/// exactly [`pareto_front`](super::pareto_front)'s output, member for member.
pub fn rank_formulas(pool: &[DiscoveredFormula]) -> Vec<RankedFormula> {
    let objectives = objective_vectors(pool);
    let mut ranked: Vec<RankedFormula> = Vec::with_capacity(pool.len());
    for (rank, front) in fast_nondominated_sort(&objectives).iter().enumerate() {
        let distance = crowding_distance(&objectives, front);
        for (k, &idx) in front.iter().enumerate() {
            ranked.push(RankedFormula {
                formula: pool[idx].clone(),
                rank,
                crowding: distance[k],
            });
        }
    }
    ranked.sort_by(|a, b| {
        a.crowded_cmp(b)
            .then_with(|| a.formula.complexity.cmp(&b.formula.complexity))
            .then_with(|| {
                sanitize_objective(a.formula.mse).total_cmp(&sanitize_objective(b.formula.mse))
            })
            .then_with(|| a.formula.pretty.cmp(&b.formula.pretty))
    });
    ranked
}

// ─────────────────────────────────────────────────────────────────────────────
// Index-derived seeding
// ─────────────────────────────────────────────────────────────────────────────

/// Number of distinct RNG purposes per generation — the stride of the tag space.
const TAG_STRIDE: u64 = 4;
/// RNG stream that builds the initial population.
const TAG_INIT: u64 = 0;
/// RNG stream that drives mating selection, crossover and mutation.
const TAG_MATE: u64 = 1;
/// RNG stream handed to the per-topology optimiser as its `topology_idx`.
const TAG_FIT: u64 = 2;

/// Seed for the RNG of `(generation, tag, slot)`, derived from indices only.
///
/// Two-level SplitMix64 (via [`derive_seed`]): the outer mix fixes an independent
/// stream per `(generation, tag)`, the inner one indexes into that stream by
/// `slot`. Distinct tags cannot collide because the generation is scaled by
/// [`TAG_STRIDE`] before the tag is added.
///
/// Nothing here depends on thread scheduling, completion order, or how many RNG
/// draws a *sibling* slot happened to make — which is exactly why the parallel
/// and the sequential runs agree bit for bit.
fn stream_seed(master: u64, generation: u64, tag: u64, slot: u64) -> u64 {
    let stream = derive_seed(
        master,
        generation.wrapping_mul(TAG_STRIDE).wrapping_add(tag),
    );
    derive_seed(stream, slot)
}

// ─────────────────────────────────────────────────────────────────────────────
// Individuals
// ─────────────────────────────────────────────────────────────────────────────

/// One member of the NSGA-II population: a topology plus its fit, if it has one.
struct Individual {
    tree: EmlTree,
    /// `None` when the optimiser could not produce a finite fit.
    formula: Option<DiscoveredFormula>,
}

impl Individual {
    fn new(tree: EmlTree) -> Self {
        Self {
            tree,
            formula: None,
        }
    }

    /// The minimised objective vector.
    ///
    /// An individual the optimiser failed to fit has no MSE at all; it is scored
    /// as `+∞` — the same "worst possible" value a `NaN` MSE is mapped to — with
    /// its real node count as the complexity. It is therefore dominated by every
    /// finite fit that is not more complex, and it will lose every tournament,
    /// but it still occupies its slot honestly instead of being silently dropped.
    fn objectives(&self) -> Vec<f64> {
        match &self.formula {
            Some(formula) => objective_vector(formula),
            None => vec![f64::INFINITY, self.tree.size() as f64],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree helpers (node addressing over the uniform EML binary tree)
// ─────────────────────────────────────────────────────────────────────────────

fn count_nodes(node: &EmlNode) -> usize {
    match node {
        EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
        EmlNode::Eml { left, right } => 1 + count_nodes(left) + count_nodes(right),
    }
}

/// The `idx`-th node in pre-order, or `None` when `idx` is past the end.
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

/// Rebuild the tree with the `idx`-th pre-order node replaced by `replacement`.
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
        *counter = usize::MAX; // sentinel: the replacement is already placed
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
// Variation operators
// ─────────────────────────────────────────────────────────────────────────────

fn random_leaf(rng: &mut Rng, num_vars: usize, const_leaf: Option<f64>) -> Arc<EmlNode> {
    let leaves = build_leaves(num_vars, const_leaf);
    let idx = rng.random_range(0..leaves.len());
    Arc::clone(&leaves[idx])
}

/// A complete EML tree of exactly `depth` levels.
fn random_node_at_depth(
    rng: &mut Rng,
    depth: usize,
    num_vars: usize,
    const_leaf: Option<f64>,
) -> Arc<EmlNode> {
    if depth == 0 {
        return random_leaf(rng, num_vars, const_leaf);
    }
    let left = random_node_at_depth(rng, depth - 1, num_vars, const_leaf);
    let right = random_node_at_depth(rng, depth - 1, num_vars, const_leaf);
    Arc::new(EmlNode::Eml { left, right })
}

/// A random tree of depth uniform in `0..=max_depth`.
///
/// The initial population deliberately spans the whole depth range: NSGA-II needs
/// spread on the *complexity* objective from generation 0, otherwise the rank-0
/// front degenerates to a single accuracy-optimal point.
fn random_tree(
    rng: &mut Rng,
    num_vars: usize,
    max_depth: usize,
    const_leaf: Option<f64>,
) -> EmlTree {
    let target_depth = rng.random_range(0..=max_depth);
    EmlTree::from_node(random_node_at_depth(
        rng,
        target_depth,
        num_vars,
        const_leaf,
    ))
}

/// Subtree crossover: a random subtree of `b` is grafted into a random node of `a`.
///
/// Returns `None` when the child would exceed `max_depth`.
fn crossover(a: &EmlTree, b: &EmlTree, max_depth: usize, rng: &mut Rng) -> Option<EmlTree> {
    let n_a = count_nodes(&a.root);
    let n_b = count_nodes(&b.root);
    if n_a == 0 || n_b == 0 {
        return None;
    }
    let idx_a = rng.random_range(0..n_a);
    let idx_b = rng.random_range(0..n_b);
    let subtree_b = get_subtree(&b.root, idx_b)?;
    let child = EmlTree::from_node(replace_subtree(&a.root, idx_a, &subtree_b));
    if child.depth() > max_depth {
        None
    } else {
        Some(child)
    }
}

/// Point mutation: swap one leaf for another leaf of the grammar.
fn mutate_point(
    tree: &EmlTree,
    num_vars: usize,
    const_leaf: Option<f64>,
    rng: &mut Rng,
) -> EmlTree {
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
    let position = leaf_positions[rng.random_range(0..leaf_positions.len())];
    let new_leaf = random_leaf(rng, num_vars, const_leaf);
    EmlTree::from_node(replace_subtree(&tree.root, position, &new_leaf))
}

/// Subtree mutation: replace a random node with a fresh random subtree.
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
    let position = rng.random_range(0..n);
    let subtree_depth = rng.random_range(0..=max_depth.saturating_sub(1));
    let new_subtree = random_node_at_depth(rng, subtree_depth, num_vars, const_leaf);
    let candidate = EmlTree::from_node(replace_subtree(&tree.root, position, &new_subtree));
    if candidate.depth() <= max_depth {
        candidate
    } else {
        tree.clone()
    }
}

/// Shrink mutation: replace a random internal node with one of its own leaves.
///
/// The complexity-reducing counterpart of [`mutate_subtree`]. NSGA-II rewards
/// simplicity explicitly (objective 1), so the operator set needs a move that
/// walks *down* the complexity axis; without it the population drifts towards
/// the deepest trees the depth limit allows.
fn mutate_shrink(
    tree: &EmlTree,
    num_vars: usize,
    const_leaf: Option<f64>,
    rng: &mut Rng,
) -> EmlTree {
    let n = count_nodes(&tree.root);
    let internal_positions: Vec<usize> = (0..n)
        .filter(|&i| {
            matches!(
                get_subtree(&tree.root, i).as_deref(),
                Some(EmlNode::Eml { .. })
            )
        })
        .collect();
    if internal_positions.is_empty() {
        return tree.clone();
    }
    let position = internal_positions[rng.random_range(0..internal_positions.len())];
    let new_leaf = random_leaf(rng, num_vars, const_leaf);
    EmlTree::from_node(replace_subtree(&tree.root, position, &new_leaf))
}

/// Gaussian jitter (σ = 0.1, Box–Muller) on one `Const` leaf.
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
    let position = const_positions[rng.random_range(0..const_positions.len())];
    let old_value = match get_subtree(&tree.root, position).as_deref() {
        Some(EmlNode::Const(v)) => *v,
        _ => return tree.clone(),
    };
    let u1: f64 = rng.random_range(f64::EPSILON..1.0_f64);
    let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
    let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.1;
    let new_leaf = Arc::new(EmlNode::Const(old_value + noise));
    EmlTree::from_node(replace_subtree(&tree.root, position, &new_leaf))
}

/// `true` when `tree` satisfies the config's dimensional-analysis filter, or when
/// no filter is configured.
fn passes_unit_filter(tree: &EmlTree, config: &SymRegConfig) -> bool {
    match config.unit_filter {
        Some((ref var_units, target_units)) => {
            let lowered = tree.lower().simplify();
            matches!(lowered.check_units(var_units), Ok(u) if u == target_units)
        }
        None => true,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitness evaluation — map (parallel) then merge (sequential)
// ─────────────────────────────────────────────────────────────────────────────

/// Fit the individuals named by `slots`, on the rayon pool.
///
/// `seeds[k]` is the `topology_idx` for `slots[k]`; both are pure functions of
/// indices ([`stream_seed`]), so `optimize_topology` — itself a pure function of
/// `(config, tree, data, topology_idx)` — cannot observe the thread it runs on.
/// `par_iter().map(…).collect()` over a slice is an *indexed* parallel iterator,
/// so the returned `Vec` is in `slots` order no matter how work is stolen, and no
/// `f64` is ever combined across tasks.
#[cfg(feature = "parallel")]
fn map_fits(
    slots: &[usize],
    seeds: &[usize],
    population: &[Individual],
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Vec<Option<DiscoveredFormula>> {
    slots
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(&slot, &seed)| {
            engine.optimize_topology(&population[slot].tree, inputs, targets, seed)
        })
        .collect()
}

/// Sequential twin of [`map_fits`] — same order, same arithmetic, same bits.
#[cfg(not(feature = "parallel"))]
fn map_fits(
    slots: &[usize],
    seeds: &[usize],
    population: &[Individual],
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Vec<Option<DiscoveredFormula>> {
    slots
        .iter()
        .zip(seeds.iter())
        .map(|(&slot, &seed)| {
            engine.optimize_topology(&population[slot].tree, inputs, targets, seed)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The NSGA-II driver
// ─────────────────────────────────────────────────────────────────────────────

/// Everything the generational loop needs, bundled so no helper needs a long
/// argument list.
struct Nsga2Run<'a> {
    engine: &'a SymRegEngine,
    inputs: &'a [Vec<f64>],
    targets: &'a [f64],
    num_vars: usize,
    nsga2: Nsga2Config,
    master_seed: u64,
    max_depth: usize,
    const_leaf: Option<f64>,
}

impl Nsga2Run<'_> {
    /// Evaluate every unfitted individual: prepare → map → merge.
    ///
    /// Phase 1 (sequential, ascending slot order) collects the slots that need a
    /// fit and their index-derived seeds. Phase 2 maps them (rayon under
    /// `feature = "parallel"`). Phase 3 merges the results back in ascending slot
    /// order. Nothing about the outcome can depend on thread count.
    fn evaluate(&self, individuals: &mut [Individual], generation: u64, tag: u64) {
        let slots: Vec<usize> = individuals
            .iter()
            .enumerate()
            .filter(|(_, individual)| individual.formula.is_none())
            .map(|(slot, _)| slot)
            .collect();
        if slots.is_empty() {
            return;
        }
        let seeds: Vec<usize> = slots
            .iter()
            .map(|&slot| stream_seed(self.master_seed, generation, tag, slot as u64) as usize)
            .collect();

        let fits = map_fits(
            &slots,
            &seeds,
            individuals,
            self.engine,
            self.inputs,
            self.targets,
        );

        for (&slot, fit) in slots.iter().zip(fits) {
            individuals[slot].formula = fit;
        }
    }

    /// A random tree that satisfies the unit filter, if one can be found in a
    /// bounded number of draws. Falls back to the last draw (which the fitness
    /// step is free to reject) rather than looping forever.
    fn sample_tree(&self, rng: &mut Rng) -> EmlTree {
        const MAX_DRAWS: usize = 16;
        let mut candidate = random_tree(rng, self.num_vars, self.max_depth, self.const_leaf);
        for _ in 1..MAX_DRAWS {
            if passes_unit_filter(&candidate, &self.engine.config) {
                return candidate;
            }
            candidate = random_tree(rng, self.num_vars, self.max_depth, self.const_leaf);
        }
        candidate
    }

    /// Binary (or `tournament_size`-ary) tournament under the crowded-comparison
    /// operator. `meta[i] = (rank, crowding)` of parent `i`.
    fn crowded_tournament(&self, meta: &[(usize, f64)], rng: &mut Rng) -> usize {
        let n = meta.len();
        if n == 0 {
            return 0;
        }
        let mut best = rng.random_range(0..n);
        for _ in 1..self.nsga2.tournament_size.max(2) {
            let challenger = rng.random_range(0..n);
            let (rank_c, crowd_c) = meta[challenger];
            let (rank_b, crowd_b) = meta[best];
            if crowded_compare(rank_c, crowd_c, rank_b, crowd_b) == Ordering::Less {
                best = challenger;
            }
        }
        best
    }

    /// Produce one offspring tree from the parent population.
    fn breed(&self, parents: &[Individual], meta: &[(usize, f64)], rng: &mut Rng) -> EmlTree {
        let parent_a = &parents[self.crowded_tournament(meta, rng)];

        let crossed = if rng.random::<f64>() < self.nsga2.crossover_rate && parents.len() > 1 {
            let parent_b = &parents[self.crowded_tournament(meta, rng)];
            crossover(&parent_a.tree, &parent_b.tree, self.max_depth, rng)
                .unwrap_or_else(|| parent_a.tree.clone())
        } else {
            parent_a.tree.clone()
        };

        let before_mutation = crossed.clone();
        let mutated = if rng.random::<f64>() < self.nsga2.mutation_rate {
            match rng.random_range(0..4u32) {
                0 => mutate_point(&crossed, self.num_vars, self.const_leaf, rng),
                1 => mutate_subtree(
                    &crossed,
                    self.num_vars,
                    self.max_depth,
                    self.const_leaf,
                    rng,
                ),
                2 => mutate_shrink(&crossed, self.num_vars, self.const_leaf, rng),
                _ => mutate_const_jitter(&crossed, rng),
            }
        } else {
            crossed
        };

        if self.engine.config.unit_filter.is_some()
            && !passes_unit_filter(&mutated, &self.engine.config)
        {
            before_mutation
        } else {
            mutated
        }
    }
}

/// Run NSGA-II and return the final population, ranked and sorted best-first.
///
/// See the module documentation for the algorithm. The returned vector holds one
/// entry per surviving individual that the optimiser managed to fit, with exact
/// duplicates collapsed ([`dedupe_identical`]); entries with `rank == 0` form the
/// non-dominated front. It is empty only when *no* topology in the final
/// population could be fitted at all.
///
/// Ranks and crowding distances are computed on the deduplicated pool, so they
/// describe the curve the caller actually receives.
///
/// The master seed is [`SymRegConfig::seed`](super::SymRegConfig::seed), falling
/// back to `42` — the same convention `evolution.rs` uses — so an NSGA-II run is
/// reproducible by default.
pub(super) fn run_nsga2(
    engine: &SymRegEngine,
    inputs: &[Vec<f64>],
    targets: &[f64],
    num_vars: usize,
    nsga2: &Nsga2Config,
) -> Result<Vec<RankedFormula>, EmlError> {
    if inputs.is_empty() || targets.is_empty() {
        return Err(EmlError::EmptyData);
    }
    if inputs.len() != targets.len() {
        return Err(EmlError::DimensionMismatch(inputs.len(), targets.len()));
    }

    let run = Nsga2Run {
        engine,
        inputs,
        targets,
        num_vars,
        nsga2: *nsga2,
        master_seed: engine.config.seed.unwrap_or(42),
        max_depth: engine.config.max_depth,
        const_leaf: if engine.config.enable_const_leaf {
            Some(engine.config.const_leaf_init)
        } else {
            None
        },
    };

    let mu = nsga2.population.max(1);

    // ── Generation 0: initial population ─────────────────────────────────────
    let mut population: Vec<Individual> = (0..mu)
        .map(|slot| {
            let mut rng =
                Rng::seed_from_u64(stream_seed(run.master_seed, 0, TAG_INIT, slot as u64));
            Individual::new(run.sample_tree(&mut rng))
        })
        .collect();
    run.evaluate(&mut population, 0, TAG_FIT);

    // ── The (μ+λ) loop ───────────────────────────────────────────────────────
    for generation in 0..nsga2.generations {
        // Mating selection needs the parents' (rank, crowding).
        let parent_objectives: Vec<Vec<f64>> =
            population.iter().map(Individual::objectives).collect();
        let meta = rank_and_crowding(&parent_objectives);

        let stream = (generation as u64) + 1;
        let mut offspring: Vec<Individual> = (0..mu)
            .map(|slot| {
                let mut rng =
                    Rng::seed_from_u64(stream_seed(run.master_seed, stream, TAG_MATE, slot as u64));
                Individual::new(run.breed(&population, &meta, &mut rng))
            })
            .collect();
        run.evaluate(&mut offspring, stream, TAG_FIT);

        // R = P ∪ Q, then keep the best μ. Elitist by construction.
        let mut combined = population;
        combined.extend(offspring);
        let combined_objectives: Vec<Vec<f64>> =
            combined.iter().map(Individual::objectives).collect();
        let survivors = environmental_selection(&combined_objectives, mu);

        // `survivors` holds distinct indices, so every `take()` yields `Some`.
        let mut slots: Vec<Option<Individual>> = combined.into_iter().map(Some).collect();
        population = survivors
            .iter()
            .filter_map(|&idx| slots[idx].take())
            .collect();
    }

    let pool: Vec<DiscoveredFormula> = dedupe_identical(
        population
            .into_iter()
            .filter_map(|individual| individual.formula)
            .collect(),
    );
    Ok(rank_formulas(&pool))
}

/// Collapse *exactly identical* formulas — same structure, same fit — to one entry.
///
/// A GA population accumulates copies of its winners, and a Pareto front that
/// lists the same formula three times is noise, not a trade-off. Worse, coincident
/// points corrupt the diversity signal: two copies of one solution sit at distance
/// zero from each other, so crowding distance would report the front's most
/// popular point as its most crowded one.
///
/// The key is the structural hash of the lowered, simplified tree **plus** the bit
/// patterns of the fitted MSE and parameters, so this only ever removes genuine
/// duplicates. Two individuals that share a topology but converged to *different*
/// local optima are different formulas — different points on the curve — and both
/// survive. First occurrence wins, and the population order at that point is
/// deterministic, so the result is too.
fn dedupe_identical(pool: Vec<DiscoveredFormula>) -> Vec<DiscoveredFormula> {
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut seen: HashSet<(u64, u64, usize, Vec<u64>)> = HashSet::new();
    pool.into_iter()
        .filter(|formula| {
            let mut hasher = DefaultHasher::new();
            formula
                .eml_tree
                .lower()
                .simplify()
                .structural_hash(&mut hasher);
            seen.insert((
                hasher.finish(),
                formula.mse.to_bits(),
                formula.complexity,
                formula.params.iter().map(|p| p.to_bits()).collect(),
            ))
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symreg::pareto_front;

    fn formula(mse: f64, complexity: usize) -> DiscoveredFormula {
        DiscoveredFormula {
            eml_tree: EmlTree::one(),
            mse,
            complexity,
            score: mse + 1e-4 * complexity as f64,
            pretty: format!("f(mse={mse}, c={complexity})"),
            params: Vec::new(),
            cv_mse: None,
            aic: 0.0,
            bic: 0.0,
            param_intervals: None,
        }
    }

    // ── Dominance ────────────────────────────────────────────────────────────

    #[test]
    fn dominance_matches_discovered_formula_dominates() {
        let pool = [
            formula(1.0, 3),
            formula(1.0, 5),
            formula(0.5, 5),
            formula(2.0, 1),
            formula(1.0, 3),
        ];
        for a in &pool {
            for b in &pool {
                let reference = a.dominates(b);
                let ours = dominates_objectives(&objective_vector(a), &objective_vector(b));
                assert_eq!(
                    reference, ours,
                    "dominance disagrees for ({}, {}) vs ({}, {})",
                    a.mse, a.complexity, b.mse, b.complexity
                );
            }
        }
    }

    #[test]
    fn dominance_is_irreflexive_and_asymmetric() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 3.0];
        assert!(!dominates_objectives(&a, &a));
        assert!(dominates_objectives(&a, &b));
        assert!(!dominates_objectives(&b, &a));
    }

    // ── Fast non-dominated sort ──────────────────────────────────────────────

    /// Textbook fixture with ranks known by hand.
    ///
    /// Objectives (both minimised):
    ///
    /// ```text
    ///   idx : (f1, f2)   expected rank
    ///   0   : (1, 5)     0   ← non-dominated (best f1)
    ///   1   : (2, 3)     0   ← non-dominated
    ///   2   : (4, 1)     0   ← non-dominated (best f2)
    ///   3   : (3, 6)     1   ← dominated by 1 only
    ///   4   : (5, 4)     1   ← dominated by 1 (2,3) and 2? (4,1)≺(5,4) yes → still rank 1
    ///   5   : (6, 7)     2   ← dominated by 3 (3,6) and 4 (5,4), both rank 1
    /// ```
    #[test]
    fn fast_nondominated_sort_textbook_ranks() {
        let objectives = vec![
            vec![1.0, 5.0], // 0
            vec![2.0, 3.0], // 1
            vec![4.0, 1.0], // 2
            vec![3.0, 6.0], // 3
            vec![5.0, 4.0], // 4
            vec![6.0, 7.0], // 5
        ];
        let fronts = fast_nondominated_sort(&objectives);
        assert_eq!(fronts.len(), 3, "expected exactly three fronts: {fronts:?}");
        assert_eq!(fronts[0], vec![0, 1, 2]);
        assert_eq!(fronts[1], vec![3, 4]);
        assert_eq!(fronts[2], vec![5]);

        // Cross-check against the (rank, crowding) view.
        let meta = rank_and_crowding(&objectives);
        let ranks: Vec<usize> = meta.iter().map(|&(rank, _)| rank).collect();
        assert_eq!(ranks, vec![0, 0, 0, 1, 1, 2]);
    }

    #[test]
    fn fast_nondominated_sort_partitions_every_index_exactly_once() {
        let objectives: Vec<Vec<f64>> = (0..23)
            .map(|i| {
                let x = (i % 7) as f64;
                let y = (i % 5) as f64;
                vec![x, y]
            })
            .collect();
        let fronts = fast_nondominated_sort(&objectives);
        let mut seen: Vec<usize> = fronts.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..objectives.len()).collect::<Vec<_>>());
    }

    #[test]
    fn fast_nondominated_sort_all_mutually_nondominated_is_one_front() {
        let objectives = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let fronts = fast_nondominated_sort(&objectives);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn fast_nondominated_sort_totally_ordered_chain_is_n_fronts() {
        let objectives: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, i as f64]).collect();
        let fronts = fast_nondominated_sort(&objectives);
        assert_eq!(fronts.len(), 5);
        for (rank, front) in fronts.iter().enumerate() {
            assert_eq!(front, &vec![rank]);
        }
    }

    #[test]
    fn fast_nondominated_sort_empty_pool() {
        assert!(fast_nondominated_sort(&[]).is_empty());
    }

    #[test]
    fn fast_nondominated_sort_duplicates_share_rank_zero() {
        let objectives = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let fronts = fast_nondominated_sort(&objectives);
        assert_eq!(fronts[0], vec![0, 1]);
        assert_eq!(fronts[1], vec![2]);
    }

    // ── Rank 0 == pareto_front ───────────────────────────────────────────────

    #[test]
    fn rank_zero_equals_pareto_front_on_the_same_pool() {
        let pool: Vec<DiscoveredFormula> = vec![
            formula(0.5, 9),
            formula(1.0, 3),
            formula(2.0, 1),
            formula(1.5, 4),
            formula(0.4, 12),
            formula(1.0, 3),
            formula(3.0, 2),
        ];
        let reference = pareto_front(&pool);
        let ranked = rank_formulas(&pool);

        let key = |f: &DiscoveredFormula| (f.complexity, f.mse.to_bits());
        let mut expected: Vec<(usize, u64)> = reference.iter().map(key).collect();
        let mut actual: Vec<(usize, u64)> = ranked
            .iter()
            .filter(|r| r.rank == 0)
            .map(|r| key(&r.formula))
            .collect();
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected, "rank 0 must equal pareto_front exactly");
    }

    // ── Crowding distance ────────────────────────────────────────────────────

    #[test]
    fn crowding_boundaries_are_positive_infinity() {
        let objectives = vec![
            vec![1.0, 10.0], // boundary on f1 (min) and f2 (max)
            vec![2.0, 6.0],
            vec![3.0, 4.0],
            vec![8.0, 1.0], // boundary on f1 (max) and f2 (min)
        ];
        let front = vec![0, 1, 2, 3];
        let distance = crowding_distance(&objectives, &front);

        assert!(distance[0].is_infinite() && distance[0] > 0.0);
        assert!(distance[3].is_infinite() && distance[3] > 0.0);
        assert!(distance[1].is_finite(), "interior must be finite");
        assert!(distance[2].is_finite(), "interior must be finite");
        assert!(distance.iter().all(|d| !d.is_nan()));

        // Interior values follow Deb's normalisation exactly:
        //   d(1) = (3-1)/(8-1) + (10-4)/(10-1)
        //   d(2) = (8-2)/(8-1) + (6-1)/(10-1)
        let expected_1 = (3.0 - 1.0) / 7.0 + (10.0 - 4.0) / 9.0;
        let expected_2 = (8.0 - 2.0) / 7.0 + (6.0 - 1.0) / 9.0;
        assert!((distance[1] - expected_1).abs() < 1e-12);
        assert!((distance[2] - expected_2).abs() < 1e-12);
    }

    #[test]
    fn crowding_small_fronts_are_all_boundaries() {
        let objectives = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(crowding_distance(&objectives, &[0]), vec![f64::INFINITY]);
        assert_eq!(
            crowding_distance(&objectives, &[0, 1]),
            vec![f64::INFINITY, f64::INFINITY]
        );
        assert!(crowding_distance(&objectives, &[]).is_empty());
    }

    #[test]
    fn crowding_zero_range_objective_is_skipped_not_divided_by() {
        // Objective 1 is constant → range 0 → must be skipped, never divided by.
        let objectives = vec![
            vec![1.0, 5.0],
            vec![2.0, 5.0],
            vec![3.0, 5.0],
            vec![4.0, 5.0],
        ];
        let front = vec![0, 1, 2, 3];
        let distance = crowding_distance(&objectives, &front);

        assert!(
            distance.iter().all(|d| !d.is_nan()),
            "a zero-range objective must not produce NaN: {distance:?}"
        );
        // Only the f0 boundaries are infinite; the degenerate objective adds nothing.
        assert!(distance[0].is_infinite());
        assert!(distance[3].is_infinite());
        assert_eq!(
            distance.iter().filter(|d| d.is_infinite()).count(),
            2,
            "a degenerate objective must not manufacture extra boundaries"
        );
        // Contribution from f0 only: (3-1)/3 and (4-2)/3.
        assert!((distance[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((distance[2] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn crowding_all_objectives_degenerate_is_finite_zero() {
        let objectives = vec![vec![2.0, 2.0]; 5];
        let front: Vec<usize> = (0..5).collect();
        let distance = crowding_distance(&objectives, &front);
        assert!(distance.iter().all(|d| d.is_finite() && *d == 0.0));
    }

    // ── NaN handling ─────────────────────────────────────────────────────────

    #[test]
    fn nan_mse_is_the_worst_objective_value() {
        assert!(sanitize_objective(f64::NAN).is_infinite());
        assert!(sanitize_objective(f64::NAN) > 0.0);

        let good = objective_vector(&formula(1.0, 3));
        let nan_fit = objective_vector(&formula(f64::NAN, 3));
        assert!(dominates_objectives(&good, &nan_fit));
        assert!(!dominates_objectives(&nan_fit, &good));
    }

    #[test]
    fn nan_mse_sinks_to_the_last_rank() {
        let pool = vec![
            formula(f64::NAN, 3),
            formula(1.0, 3),
            formula(0.5, 7),
            formula(2.0, 1),
        ];
        let ranked = rank_formulas(&pool);
        assert_eq!(ranked.len(), 4);

        let nan_entry = ranked
            .iter()
            .find(|r| r.formula.mse.is_nan())
            .expect("the NaN formula must still be present");
        let worst_rank = ranked.iter().map(|r| r.rank).max().unwrap_or(0);
        assert_eq!(
            nan_entry.rank, worst_rank,
            "a NaN MSE must be treated as the worst objective value"
        );
        assert!(nan_entry.rank > 0, "a NaN MSE must not sit on the front");
        // Best-first order: the NaN entry sorts last.
        assert!(
            ranked
                .last()
                .map(|r| r.formula.mse.is_nan())
                .unwrap_or(false)
        );
        assert!(ranked.iter().all(|r| !r.crowding.is_nan()));
    }

    #[test]
    fn nan_crowding_cannot_break_sort_consistency() {
        // `crowded_compare` must be a total order even for inputs this module
        // cannot itself produce — an inconsistent comparator makes `sort_by`
        // panic or emit garbage.
        let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, 1.5];
        for &a in &values {
            for &b in &values {
                let ab = crowded_compare(0, a, 0, b);
                let ba = crowded_compare(0, b, 0, a);
                assert_eq!(ab, ba.reverse(), "comparator is not antisymmetric");
            }
        }
        // Transitivity spot check across the whole set.
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    if crowded_compare(0, a, 0, b) != Ordering::Greater
                        && crowded_compare(0, b, 0, c) != Ordering::Greater
                    {
                        assert_ne!(crowded_compare(0, a, 0, c), Ordering::Greater);
                    }
                }
            }
        }

        let mut entries: Vec<RankedFormula> = vec![
            RankedFormula {
                formula: formula(1.0, 1),
                rank: 0,
                crowding: f64::NAN,
            },
            RankedFormula {
                formula: formula(2.0, 2),
                rank: 0,
                crowding: f64::INFINITY,
            },
            RankedFormula {
                formula: formula(3.0, 3),
                rank: 0,
                crowding: 1.0,
            },
        ];
        entries.sort_by(RankedFormula::crowded_cmp); // must not panic
        assert_eq!(entries.len(), 3);
    }

    // ── Crowded comparison ───────────────────────────────────────────────────

    #[test]
    fn crowded_compare_prefers_lower_rank_then_higher_crowding() {
        assert_eq!(crowded_compare(0, 0.0, 1, f64::INFINITY), Ordering::Less);
        assert_eq!(crowded_compare(1, f64::INFINITY, 0, 0.0), Ordering::Greater);
        assert_eq!(crowded_compare(2, 5.0, 2, 1.0), Ordering::Less);
        assert_eq!(crowded_compare(2, 1.0, 2, 5.0), Ordering::Greater);
        assert_eq!(crowded_compare(2, 1.0, 2, 1.0), Ordering::Equal);
        assert_eq!(
            crowded_compare(0, f64::INFINITY, 0, 1e300),
            Ordering::Less,
            "an infinite crowding distance always wins its rank"
        );
    }

    // ── (μ+λ) selection ──────────────────────────────────────────────────────

    #[test]
    fn environmental_selection_takes_whole_fronts_then_truncates_by_crowding() {
        // Front 0 = {0,1,2,3} (a 4-point trade-off curve), front 1 = {4}.
        let objectives = vec![
            vec![1.0, 10.0],
            vec![2.0, 6.0],
            vec![3.0, 4.0],
            vec![8.0, 1.0],
            vec![9.0, 11.0],
        ];
        // μ = 5 → everything survives.
        assert_eq!(environmental_selection(&objectives, 5), vec![0, 1, 2, 3, 4]);
        // μ = 4 → exactly front 0.
        assert_eq!(environmental_selection(&objectives, 4), vec![0, 1, 2, 3]);
        // μ = 2 → front 0 truncated to its two boundary members (crowding = +∞).
        let two = environmental_selection(&objectives, 2);
        assert_eq!(two.len(), 2);
        let mut sorted = two.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 3], "boundaries must survive truncation");
        // μ = 0 → nothing.
        assert!(environmental_selection(&objectives, 0).is_empty());
    }

    #[test]
    fn environmental_selection_is_elitist_and_distinct() {
        let objectives: Vec<Vec<f64>> = (0..17)
            .map(|i| vec![(i % 5) as f64, (i % 3) as f64])
            .collect();
        let survivors = environmental_selection(&objectives, 8);
        assert_eq!(survivors.len(), 8);
        let mut unique = survivors.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 8, "survivor indices must be distinct");

        // Every rank-0 member that fits must be in there.
        let fronts = fast_nondominated_sort(&objectives);
        if fronts[0].len() <= 8 {
            for idx in &fronts[0] {
                assert!(survivors.contains(idx), "lost a rank-0 solution");
            }
        }
    }

    // ── Seeding ──────────────────────────────────────────────────────────────

    #[test]
    fn stream_seed_separates_tags_generations_and_slots() {
        let mut seeds = Vec::new();
        for generation in 0..4u64 {
            for tag in [TAG_INIT, TAG_MATE, TAG_FIT] {
                for slot in 0..4u64 {
                    seeds.push(stream_seed(7, generation, tag, slot));
                }
            }
        }
        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), seeds.len(), "seed streams collided");
        // Purely a function of its indices.
        assert_eq!(
            stream_seed(7, 2, TAG_MATE, 3),
            stream_seed(7, 2, TAG_MATE, 3)
        );
    }

    // ── Determinism of the only rayon call site ──────────────────────────────

    #[cfg(feature = "parallel")]
    fn determinism_fixture() -> (Vec<Individual>, Vec<usize>, Vec<usize>, SymRegConfig) {
        let config = SymRegConfig {
            max_depth: 2,
            seed: Some(42),
            ..SymRegConfig::quick()
        };
        let mut rng = Rng::seed_from_u64(1234);
        let population: Vec<Individual> = (0..24)
            .map(|_| Individual::new(random_tree(&mut rng, 1, config.max_depth, None)))
            .collect();
        let slots: Vec<usize> = (0..population.len()).collect();
        let seeds: Vec<usize> = slots
            .iter()
            .map(|&slot| stream_seed(42, 1, TAG_FIT, slot as u64) as usize)
            .collect();
        (population, slots, seeds, config)
    }

    /// **The parallel == sequential proof at the map level.**
    ///
    /// [`map_fits`] is the only function whose body differs between the `parallel`
    /// and the non-`parallel` build of NSGA-II; everything else in the driver is
    /// sequential. Here the rayon version and the literal body of its sequential
    /// twin are run on the same population and asserted bit-equal.
    #[cfg(feature = "parallel")]
    #[test]
    fn map_fits_parallel_equals_sequential_bitwise() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * 2.0 + 1.0).collect();
        let (population, slots, seeds, config) = determinism_fixture();
        let engine = SymRegEngine::new(config);

        let parallel = map_fits(&slots, &seeds, &population, &engine, &inputs, &targets);

        // Byte-for-byte the body of the `#[cfg(not(feature = "parallel"))]` twin.
        let sequential: Vec<Option<DiscoveredFormula>> = slots
            .iter()
            .zip(seeds.iter())
            .map(|(&slot, &seed)| {
                engine.optimize_topology(&population[slot].tree, &inputs, &targets, seed)
            })
            .collect();

        assert_eq!(parallel.len(), sequential.len());
        for (k, (p, s)) in parallel.iter().zip(sequential.iter()).enumerate() {
            match (p, s) {
                (Some(pf), Some(sf)) => {
                    assert_eq!(pf.mse.to_bits(), sf.mse.to_bits(), "slot {k}: MSE bits");
                    let pp: Vec<u64> = pf.params.iter().map(|v| v.to_bits()).collect();
                    let sp: Vec<u64> = sf.params.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(pp, sp, "slot {k}: parameter bits");
                    assert_eq!(pf.complexity, sf.complexity, "slot {k}: complexity");
                }
                (None, None) => {}
                _ => panic!("slot {k}: fit success differs between parallel and sequential"),
            }
        }
    }

    /// The rayon map must be invariant under the worker count.
    #[cfg(feature = "parallel")]
    #[test]
    fn map_fits_is_thread_count_invariant() {
        let inputs: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0] * 2.0 + 1.0).collect();
        let (population, slots, seeds, config) = determinism_fixture();
        let engine = SymRegEngine::new(config);

        let run_with = |threads: usize| -> Vec<Option<u64>> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool should build");
            pool.install(|| {
                map_fits(&slots, &seeds, &population, &engine, &inputs, &targets)
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
