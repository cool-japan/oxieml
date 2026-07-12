//! R2 — NSGA-II multi-objective symbolic regression.
//!
//! Spec coverage, one section per bullet of the TODO's `Tests:` line:
//!
//! 1. **Textbook fast-nondominated-sort ranks** — a hand-computed fixture whose
//!    fronts are known by inspection, plus the degenerate shapes (empty, one
//!    front, a totally-ordered chain).
//! 2. **rank-0 == `pareto_front` on the same pool** — the free function is the
//!    reference oracle; NSGA-II's rank 0 must reproduce it member for member, on
//!    synthetic pools *and* on a pool that a real `discover()` run produced.
//! 3. **Crowding boundary `+∞`** — extremes are infinite, interiors are finite
//!    and match Deb's normalisation.
//! 4. **Zero-range objective** — skipped, never divided by; no `NaN` escapes.
//! 5. **`NaN` MSE treated as worst** — it sinks to the last rank instead of
//!    poisoning the comparator, and the comparator stays a total order.
//! 6. **Determinism** — repeat runs are bit-identical via `f64::to_bits`, and
//!    under `feature = "parallel"` the result is invariant under the rayon worker
//!    count (1, 2, 3, 8, 16 threads), which is the in-binary form of
//!    `parallel == sequential`. The map-level proof (rayon `par_iter` output ==
//!    `iter` output, bit for bit) lives in `src/symreg/nsga2.rs`, where the
//!    private `map_fits` is reachable.

use oxieml::EmlTree;
use oxieml::symreg::{
    DiscoveredFormula, Nsga2Config, RankedFormula, SymRegConfig, SymRegEngine, crowded_compare,
    crowding_distance, dominates_objectives, environmental_selection, fast_nondominated_sort,
    objective_vector, objective_vectors, pareto_front, rank_and_crowding, rank_formulas,
};
use std::cmp::Ordering;

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// A formula carrying nothing but the two objectives that NSGA-II reads.
fn formula(mse: f64, complexity: usize) -> DiscoveredFormula {
    DiscoveredFormula {
        eml_tree: EmlTree::one(),
        mse,
        complexity,
        score: mse + 1e-4 * complexity as f64,
        pretty: format!("mse={mse}/c={complexity}"),
        params: Vec::new(),
        cv_mse: None,
        aic: 0.0,
        bic: 0.0,
        param_intervals: None,
    }
}

/// `y = exp(x)` — a target the EML grammar represents exactly.
fn exp_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.15]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();
    (inputs, targets)
}

fn small_engine() -> SymRegEngine {
    SymRegEngine::new(SymRegConfig {
        max_depth: 2,
        max_iter: 60,
        num_restarts: 1,
        seed: Some(2026),
        ..SymRegConfig::quick()
    })
}

fn small_nsga2() -> Nsga2Config {
    Nsga2Config {
        population: 12,
        generations: 3,
        tournament_size: 2,
        crossover_rate: 0.9,
        mutation_rate: 0.3,
    }
}

/// Bit-exact fingerprint of a ranked population. Every `f64` goes through
/// `to_bits` — `Display` would round away exactly the differences we hunt for.
fn fingerprint(ranked: &[RankedFormula]) -> Vec<String> {
    ranked
        .iter()
        .map(|r| {
            let params: Vec<String> = r
                .formula
                .params
                .iter()
                .map(|p| format!("{:016x}", p.to_bits()))
                .collect();
            format!(
                "rank={} crowding={:016x} mse={:016x} score={:016x} complexity={} params=[{}] pretty={}",
                r.rank,
                r.crowding.to_bits(),
                r.formula.mse.to_bits(),
                r.formula.score.to_bits(),
                r.formula.complexity,
                params.join(","),
                r.formula.pretty,
            )
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Textbook fast-nondominated-sort ranks
// ─────────────────────────────────────────────────────────────────────────────

/// Deb-style fixture, ranks computed by hand:
///
/// ```text
///   idx  (f1, f2)   dominated by            rank
///   0    (1, 5)     —                        0
///   1    (2, 3)     —                        0
///   2    (4, 1)     —                        0
///   3    (3, 6)     1 (2,3)                  1
///   4    (5, 4)     1 (2,3), 2 (4,1)         1
///   5    (6, 7)     0,1,2,3,4                2
/// ```
#[test]
fn textbook_fast_nondominated_sort_ranks() {
    let objectives = vec![
        vec![1.0, 5.0],
        vec![2.0, 3.0],
        vec![4.0, 1.0],
        vec![3.0, 6.0],
        vec![5.0, 4.0],
        vec![6.0, 7.0],
    ];
    let fronts = fast_nondominated_sort(&objectives);
    assert_eq!(fronts.len(), 3, "expected three fronts, got {fronts:?}");
    assert_eq!(fronts[0], vec![0, 1, 2]);
    assert_eq!(fronts[1], vec![3, 4]);
    assert_eq!(fronts[2], vec![5]);

    let ranks: Vec<usize> = rank_and_crowding(&objectives)
        .iter()
        .map(|&(rank, _)| rank)
        .collect();
    assert_eq!(ranks, vec![0, 0, 0, 1, 1, 2]);
}

#[test]
fn fast_nondominated_sort_degenerate_shapes() {
    // Empty pool.
    assert!(fast_nondominated_sort(&[]).is_empty());

    // A pure trade-off curve: everything is mutually non-dominated → one front.
    let curve: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 5.0 - i as f64]).collect();
    let fronts = fast_nondominated_sort(&curve);
    assert_eq!(fronts.len(), 1);
    assert_eq!(fronts[0], (0..6).collect::<Vec<usize>>());

    // A total order: N points, N fronts of one.
    let chain: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, i as f64]).collect();
    let fronts = fast_nondominated_sort(&chain);
    assert_eq!(fronts.len(), 6);
    for (rank, front) in fronts.iter().enumerate() {
        assert_eq!(front, &vec![rank]);
    }

    // Every index lands in exactly one front, always.
    let messy: Vec<Vec<f64>> = (0..29)
        .map(|i| vec![(i % 7) as f64, (i % 4) as f64])
        .collect();
    let mut seen: Vec<usize> = fast_nondominated_sort(&messy)
        .iter()
        .flatten()
        .copied()
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..29).collect::<Vec<usize>>());
}

#[test]
fn nsga2_dominance_agrees_with_discovered_formula_dominates() {
    let pool = [
        formula(1.0, 3),
        formula(1.0, 5),
        formula(0.5, 5),
        formula(2.0, 1),
        formula(1.0, 3),
        formula(0.0, 9),
    ];
    for a in &pool {
        for b in &pool {
            assert_eq!(
                a.dominates(b),
                dominates_objectives(&objective_vector(a), &objective_vector(b)),
                "dominance disagrees on ({}, {}) vs ({}, {})",
                a.mse,
                a.complexity,
                b.mse,
                b.complexity
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. rank-0 == pareto_front on the same pool
// ─────────────────────────────────────────────────────────────────────────────

/// Canonical identity of a formula for set comparison — the two objectives, bit
/// exact, plus the label.
fn pool_key(f: &DiscoveredFormula) -> (usize, u64, String) {
    (f.complexity, f.mse.to_bits(), f.pretty.clone())
}

fn assert_rank_zero_equals_pareto_front(pool: &[DiscoveredFormula]) {
    let reference = pareto_front(pool);
    let ranked = rank_formulas(pool);

    let mut expected: Vec<(usize, u64, String)> = reference.iter().map(pool_key).collect();
    let mut actual: Vec<(usize, u64, String)> = ranked
        .iter()
        .filter(|r| r.rank == 0)
        .map(|r| pool_key(&r.formula))
        .collect();
    expected.sort();
    actual.sort();

    assert_eq!(
        actual.len(),
        expected.len(),
        "rank-0 size differs from pareto_front (multiplicity included)"
    );
    assert_eq!(actual, expected, "rank 0 must equal pareto_front exactly");
}

#[test]
fn rank_zero_equals_pareto_front_synthetic_pool() {
    assert_rank_zero_equals_pareto_front(&[
        formula(0.5, 9),
        formula(1.0, 3),
        formula(2.0, 1),
        formula(1.5, 4),
        formula(0.4, 12),
        formula(1.0, 3), // exact duplicate: both stay non-dominated
        formula(3.0, 2),
        formula(0.4, 12), // duplicate of the accuracy champion
    ]);
}

#[test]
fn rank_zero_equals_pareto_front_dense_grid() {
    // A dense grid exercises ties on both objectives simultaneously.
    let pool: Vec<DiscoveredFormula> = (0..6)
        .flat_map(|c| (0..6).map(move |m| formula(m as f64 * 0.25, c + 1)))
        .collect();
    assert_rank_zero_equals_pareto_front(&pool);
}

#[test]
fn rank_zero_equals_pareto_front_on_a_real_discover_pool() {
    let (inputs, targets) = exp_data();
    let engine = small_engine();
    let pool = engine
        .discover(&inputs, &targets, 1)
        .expect("discover should succeed");
    assert!(!pool.is_empty());
    assert!(
        pool.iter().all(|f| !f.mse.is_nan()),
        "the optimiser must never emit a NaN MSE"
    );
    assert_rank_zero_equals_pareto_front(&pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Crowding boundary +∞
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn crowding_boundary_solutions_are_positive_infinity() {
    let objectives = vec![
        vec![1.0, 10.0],
        vec![2.0, 6.0],
        vec![3.0, 4.0],
        vec![8.0, 1.0],
    ];
    let distance = crowding_distance(&objectives, &[0, 1, 2, 3]);

    assert!(distance[0] == f64::INFINITY, "min-f1 boundary must be +∞");
    assert!(distance[3] == f64::INFINITY, "max-f1 boundary must be +∞");
    assert!(distance[1].is_finite());
    assert!(distance[2].is_finite());

    // Deb's normalisation, exactly.
    let expected_1 = (3.0 - 1.0) / 7.0 + (10.0 - 4.0) / 9.0;
    let expected_2 = (8.0 - 2.0) / 7.0 + (6.0 - 1.0) / 9.0;
    assert!((distance[1] - expected_1).abs() < 1e-12);
    assert!((distance[2] - expected_2).abs() < 1e-12);
}

#[test]
fn crowding_fronts_of_size_one_and_two_are_all_boundaries() {
    let objectives = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    assert_eq!(crowding_distance(&objectives, &[0]), vec![f64::INFINITY]);
    assert_eq!(
        crowding_distance(&objectives, &[0, 1]),
        vec![f64::INFINITY, f64::INFINITY]
    );
    assert!(crowding_distance(&objectives, &[]).is_empty());
}

#[test]
fn rank_zero_front_of_a_real_run_has_infinite_boundaries() {
    let pool = vec![
        formula(4.0, 1),
        formula(2.0, 2),
        formula(1.0, 3),
        formula(0.5, 5),
    ];
    let ranked = rank_formulas(&pool);
    let front: Vec<&RankedFormula> = ranked.iter().filter(|r| r.rank == 0).collect();
    assert_eq!(front.len(), 4, "the whole pool is a trade-off curve");
    assert_eq!(
        front.iter().filter(|r| r.crowding.is_infinite()).count(),
        2,
        "exactly the two extremes of the curve are boundaries"
    );
    // Best-first order puts the infinite-crowding boundaries first.
    assert!(front[0].crowding.is_infinite());
    assert!(front[1].crowding.is_infinite());
    assert!(ranked.iter().all(|r| !r.crowding.is_nan()));
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Zero-range objective: explicit skip, never a division by zero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn crowding_zero_range_objective_is_skipped() {
    // Objective 1 is constant → max − min == 0.
    let objectives = vec![
        vec![1.0, 5.0],
        vec![2.0, 5.0],
        vec![3.0, 5.0],
        vec![4.0, 5.0],
    ];
    let distance = crowding_distance(&objectives, &[0, 1, 2, 3]);

    assert!(
        distance.iter().all(|d| !d.is_nan()),
        "0/0 must never happen: {distance:?}"
    );
    assert_eq!(
        distance.iter().filter(|d| d.is_infinite()).count(),
        2,
        "a degenerate objective must not manufacture extra boundaries"
    );
    // Only objective 0 contributes: (3−1)/3 and (4−2)/3.
    assert!((distance[1] - 2.0 / 3.0).abs() < 1e-12);
    assert!((distance[2] - 2.0 / 3.0).abs() < 1e-12);
}

#[test]
fn crowding_all_objectives_degenerate_stays_finite() {
    let objectives = vec![vec![7.0, 7.0]; 6];
    let distance = crowding_distance(&objectives, &(0..6).collect::<Vec<usize>>());
    assert!(
        distance.iter().all(|d| d.is_finite() && *d == 0.0),
        "identical points must give finite, zero crowding: {distance:?}"
    );
}

#[test]
fn equal_complexity_pool_never_divides_by_zero() {
    // Every formula has the same complexity → objective 1 has zero range.
    let pool: Vec<DiscoveredFormula> = (0..7).map(|i| formula(i as f64 * 0.5, 4)).collect();
    let ranked = rank_formulas(&pool);
    assert!(ranked.iter().all(|r| !r.crowding.is_nan()));
    // Only the MSE champion is non-dominated (all complexities tie).
    let front: Vec<&RankedFormula> = ranked.iter().filter(|r| r.rank == 0).collect();
    assert_eq!(front.len(), 1);
    assert_eq!(front[0].formula.mse.to_bits(), 0.0_f64.to_bits());
    assert_rank_zero_equals_pareto_front(&pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. NaN MSE treated as worst, without breaking sort consistency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nan_mse_is_treated_as_the_worst_objective() {
    let good = objective_vector(&formula(1.0, 3));
    let broken = objective_vector(&formula(f64::NAN, 3));

    assert!(broken[0].is_infinite(), "NaN MSE must sanitise to +∞");
    assert!(dominates_objectives(&good, &broken));
    assert!(!dominates_objectives(&broken, &good));

    let pool = vec![
        formula(f64::NAN, 3),
        formula(1.0, 3),
        formula(0.5, 7),
        formula(2.0, 1),
        formula(f64::NAN, 8),
    ];
    let ranked = rank_formulas(&pool);
    assert_eq!(ranked.len(), pool.len(), "no formula may be dropped");

    let worst_rank = ranked.iter().map(|r| r.rank).max().unwrap_or(0);
    for entry in ranked.iter().filter(|r| r.formula.mse.is_nan()) {
        assert!(
            entry.rank > 0,
            "a NaN-MSE formula must never sit on the non-dominated front"
        );
    }
    assert!(
        ranked
            .last()
            .map(|r| r.formula.mse.is_nan())
            .unwrap_or(false),
        "best-first order must put a NaN-MSE formula last"
    );
    assert_eq!(
        ranked
            .iter()
            .find(|r| r.formula.mse.is_nan() && r.formula.complexity == 8)
            .map(|r| r.rank),
        Some(worst_rank)
    );
    assert!(
        ranked.iter().all(|r| !r.crowding.is_nan()),
        "a NaN objective must not leak into the crowding distance"
    );
}

#[test]
fn nan_objectives_keep_the_comparator_consistent() {
    // An inconsistent comparator makes Rust's sort panic (or emit garbage). This
    // one is a total order on every f64, NaN and ±∞ included.
    let values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        1e308,
    ];
    for &a in &values {
        for &b in &values {
            assert_eq!(
                crowded_compare(0, a, 0, b),
                crowded_compare(0, b, 0, a).reverse(),
                "not antisymmetric for ({a}, {b})"
            );
            for &c in &values {
                if crowded_compare(0, a, 0, b) != Ordering::Greater
                    && crowded_compare(0, b, 0, c) != Ordering::Greater
                {
                    assert_ne!(
                        crowded_compare(0, a, 0, c),
                        Ordering::Greater,
                        "not transitive for ({a}, {b}, {c})"
                    );
                }
            }
        }
    }

    // A whole pool of NaN MSEs must still sort without panicking. With a finite
    // fit at the minimal complexity, every NaN individual is strictly dominated.
    let pool: Vec<DiscoveredFormula> = (1..=8)
        .map(|c| formula(f64::NAN, c))
        .chain(std::iter::once(formula(0.25, 1)))
        .collect();
    let ranked = rank_formulas(&pool); // must not panic
    assert_eq!(ranked.len(), 9);
    let best = &ranked[0];
    assert_eq!(best.rank, 0);
    assert!(
        !best.formula.mse.is_nan(),
        "the only finite fit must win rank 0"
    );
    assert_eq!(
        ranked.iter().filter(|r| r.rank == 0).count(),
        1,
        "with a finite fit at minimal complexity, no NaN formula is non-dominated"
    );
    assert!(ranked.iter().skip(1).all(|r| r.formula.mse.is_nan()));
}

/// Honest semantics, pinned down: `NaN → +∞` makes a formula the **worst on the
/// MSE axis**, which is not the same as universally dominated. Dominance is a
/// *vector* relation, so a `NaN`-MSE formula that is strictly the simplest in the
/// pool stays non-dominated — exactly as an honest `+∞`-MSE model would, and
/// exactly as `pareto_front` treats a `+∞`-MSE model. What "worst" buys is that
/// the formula loses to any finite fit that is *no more complex*, which is what
/// the previous test asserts.
#[test]
fn nan_mse_at_strictly_minimal_complexity_stays_non_dominated() {
    let pool = vec![
        formula(f64::NAN, 1), // strictly the simplest — nothing dominates it
        formula(0.25, 4),
        formula(0.10, 6),
    ];
    let ranked = rank_formulas(&pool);
    let front: Vec<&RankedFormula> = ranked.iter().filter(|r| r.rank == 0).collect();
    assert_eq!(front.len(), 3, "the whole pool is a trade-off curve");

    // A `+∞` MSE at the same complexity behaves identically — NaN is not special.
    let with_inf = vec![
        formula(f64::INFINITY, 1),
        formula(0.25, 4),
        formula(0.10, 6),
    ];
    let inf_ranks: Vec<usize> = rank_formulas(&with_inf).iter().map(|r| r.rank).collect();
    let nan_ranks: Vec<usize> = ranked.iter().map(|r| r.rank).collect();
    assert_eq!(nan_ranks, inf_ranks, "NaN must behave exactly like +∞");

    // But add a finite fit at complexity 1 and the NaN formula is dominated.
    let mut pool = pool;
    pool.push(formula(0.9, 1));
    let ranked = rank_formulas(&pool);
    let nan_rank = ranked
        .iter()
        .find(|r| r.formula.mse.is_nan())
        .map(|r| r.rank)
        .expect("NaN formula must still be present");
    assert!(
        nan_rank > 0,
        "a finite fit of equal complexity must dominate the NaN one"
    );
}

#[test]
fn nan_mse_environmental_selection_evicts_it_first() {
    let pool = vec![
        formula(1.0, 2),
        formula(0.5, 4),
        formula(f64::NAN, 1),
        formula(2.0, 1),
    ];
    let survivors = environmental_selection(&objective_vectors(&pool), 3);
    assert_eq!(survivors.len(), 3);
    assert!(
        !survivors.contains(&2),
        "the NaN-MSE individual must be the first one evicted, got {survivors:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. discover_nsga2 end-to-end + determinism
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn discover_nsga2_returns_a_ranked_annotated_population() {
    let (inputs, targets) = exp_data();
    let engine = small_engine();
    let ranked = engine
        .discover_nsga2(&inputs, &targets, 1, &small_nsga2())
        .expect("discover_nsga2 should succeed");

    assert!(!ranked.is_empty(), "NSGA-II must return a population");

    // Ranks are dense from 0 and the output is sorted best-first.
    assert_eq!(ranked[0].rank, 0, "the first entry must be non-dominated");
    for pair in ranked.windows(2) {
        assert_ne!(
            pair[0].crowded_cmp(&pair[1]),
            Ordering::Greater,
            "output must be sorted by the crowded-comparison operator"
        );
    }

    // Rank 0 of the returned population is its own Pareto front.
    let pool: Vec<DiscoveredFormula> = ranked.iter().map(|r| r.formula.clone()).collect();
    assert_rank_zero_equals_pareto_front(&pool);

    // The annotations are well-formed.
    assert!(ranked.iter().all(|r| !r.crowding.is_nan()));
    assert!(ranked.iter().all(|r| r.crowding >= 0.0));
    assert!(ranked.iter().all(|r| !r.formula.mse.is_nan()));

    // NSGA-II must actually fit something on `y = exp(x)`.
    let best_mse = ranked
        .iter()
        .map(|r| r.formula.mse)
        .fold(f64::INFINITY, f64::min);
    assert!(
        best_mse < 1.0,
        "NSGA-II should fit exp(x) far better than MSE 1.0, got {best_mse}"
    );

    // The front spans a genuine trade-off: no member of rank 0 dominates another.
    let front: Vec<&RankedFormula> = ranked.iter().filter(|r| r.rank == 0).collect();
    for a in &front {
        for b in &front {
            if !std::ptr::eq(*a, *b) {
                assert!(!a.formula.dominates(&b.formula) || a.formula.mse == b.formula.mse);
            }
        }
    }
}

/// A GA population accumulates copies of its winners. The returned front must not
/// list the same formula (same structure, same fit) more than once — coincident
/// points are noise on a trade-off curve and they distort crowding distance.
#[test]
fn discover_nsga2_front_has_no_exact_duplicates() {
    let (inputs, targets) = exp_data();
    let engine = small_engine();
    let ranked = engine
        .discover_nsga2(&inputs, &targets, 1, &small_nsga2())
        .expect("discover_nsga2 should succeed");

    let mut keys: Vec<(String, u64, Vec<u64>)> = ranked
        .iter()
        .map(|r| {
            (
                r.formula.pretty.clone(),
                r.formula.mse.to_bits(),
                r.formula.params.iter().map(|p| p.to_bits()).collect(),
            )
        })
        .collect();
    let total = keys.len();
    keys.sort();
    keys.dedup();
    assert_eq!(
        keys.len(),
        total,
        "the returned front repeats an identical formula"
    );
}

#[test]
fn discover_nsga2_rejects_bad_input() {
    let engine = small_engine();
    let config = small_nsga2();
    assert!(engine.discover_nsga2(&[], &[], 1, &config).is_err());
    assert!(
        engine
            .discover_nsga2(&[vec![1.0]], &[1.0, 2.0], 1, &config)
            .is_err()
    );
}

#[test]
fn discover_nsga2_is_bit_identical_across_repeat_runs() {
    let (inputs, targets) = exp_data();
    let engine = small_engine();
    let config = small_nsga2();

    let first = fingerprint(
        &engine
            .discover_nsga2(&inputs, &targets, 1, &config)
            .expect("run 1"),
    );
    for attempt in 2..=3 {
        let again = fingerprint(
            &engine
                .discover_nsga2(&inputs, &targets, 1, &config)
                .expect("run n"),
        );
        assert_eq!(again, first, "run {attempt} diverged from run 1 bit-wise");
    }
    assert!(!first.is_empty());
}

#[test]
fn discover_nsga2_seeds_are_honoured() {
    let (inputs, targets) = exp_data();
    let config = small_nsga2();

    let run_with_seed = |seed: u64| -> Vec<String> {
        let engine = SymRegEngine::new(SymRegConfig {
            max_depth: 2,
            max_iter: 60,
            num_restarts: 1,
            seed: Some(seed),
            ..SymRegConfig::quick()
        });
        fingerprint(
            &engine
                .discover_nsga2(&inputs, &targets, 1, &config)
                .expect("seeded run"),
        )
    };

    assert_eq!(run_with_seed(7), run_with_seed(7), "same seed, same bits");
}

/// **`parallel == sequential`, in-binary form.**
///
/// The rayon worker count is the only thing that distinguishes a parallel run
/// from a sequential one at run time (the `par_iter`/`iter` split itself is
/// proved bit-equal by the unit test in `src/symreg/nsga2.rs`). Pinning the
/// result across 1, 2, 3, 8 and 16 workers — *including* against the ambient
/// global pool — is therefore the strictly stronger statement.
#[cfg(feature = "parallel")]
#[test]
fn discover_nsga2_is_thread_count_invariant() {
    let (inputs, targets) = exp_data();
    let engine = small_engine();
    let config = small_nsga2();

    let run_with = |threads: usize| -> Vec<String> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool should build");
        pool.install(|| {
            fingerprint(
                &engine
                    .discover_nsga2(&inputs, &targets, 1, &config)
                    .expect("threaded run"),
            )
        })
    };

    let ambient = fingerprint(
        &engine
            .discover_nsga2(&inputs, &targets, 1, &config)
            .expect("ambient run"),
    );
    let reference = run_with(1);
    assert_eq!(
        reference, ambient,
        "the single-worker pool must match the ambient global pool"
    );
    for threads in [2usize, 3, 8, 16] {
        assert_eq!(
            run_with(threads),
            reference,
            "NSGA-II result changed with {threads} rayon workers"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (μ+λ) selection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mu_plus_lambda_selection_is_elitist_and_truncates_by_crowding() {
    let objectives = vec![
        vec![1.0, 10.0], // front 0, boundary
        vec![2.0, 6.0],  // front 0, interior
        vec![3.0, 4.0],  // front 0, interior
        vec![8.0, 1.0],  // front 0, boundary
        vec![9.0, 11.0], // front 1
    ];
    assert_eq!(environmental_selection(&objectives, 5), vec![0, 1, 2, 3, 4]);
    assert_eq!(environmental_selection(&objectives, 4), vec![0, 1, 2, 3]);

    let mut two = environmental_selection(&objectives, 2);
    two.sort_unstable();
    assert_eq!(
        two,
        vec![0, 3],
        "truncation must keep the +∞-crowding boundaries"
    );

    assert!(environmental_selection(&objectives, 0).is_empty());
    assert_eq!(
        environmental_selection(&objectives, 99).len(),
        5,
        "μ larger than the pool keeps everything"
    );
}

#[test]
fn mu_plus_lambda_never_loses_a_rank_zero_solution() {
    let objectives: Vec<Vec<f64>> = (0..24)
        .map(|i| vec![(i % 6) as f64, (i % 5) as f64])
        .collect();
    let fronts = fast_nondominated_sort(&objectives);
    let mu = fronts[0].len() + 2;
    let survivors = environmental_selection(&objectives, mu);
    assert_eq!(survivors.len(), mu);
    for idx in &fronts[0] {
        assert!(survivors.contains(idx), "lost rank-0 solution {idx}");
    }
}
