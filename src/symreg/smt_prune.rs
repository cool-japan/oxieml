//! Shared SMT-based pruning for symbolic regression.
//!
//! `interval_prune` (cheap, always available under the `smt` feature): uses
//! `IntervalDomain::propagate` to check feasibility.
//!
//! `solver_prune` (OxiZ-backed, expensive, opt-in): calls the SMT solver.
//! Depth-gated to avoid spending time on tiny partial topologies.
//!
//! # Solver reuse
//!
//! `solver_prune` used to construct a brand-new `EmlSmtSolver` — and therefore a
//! brand-new OxiZ `TermManager` and `Solver` — for *every candidate topology*.
//! Over a search that is thousands of constructions, and for the small formulas a
//! pruner sees the construction cost dominates the solving cost.
//!
//! It now screens each candidate against this thread's pooled
//! [`IncrementalEmlSolver`] (see [`crate::smt::with_thread_local_solver`]) inside
//! its own `push`/`pop` scope. OxiZ's `Solver` is not `Sync`, so a thread-local
//! pool is exactly the right home for it: no locking, no sharing, and the pruner
//! stays usable from the parallel search phases (each worker thread simply gets
//! its own solver). The pool is keyed by the variable box, so a sweep at fixed
//! bounds — the normal case — constructs exactly one solver however many
//! topologies it screens. That is asserted, not assumed, by
//! `solver_construction_count_is_one_across_many_topologies` below.

use crate::smt::{
    EmlConstraint, Interval, IntervalDomain, PropResult, SmtResult, with_thread_local_solver,
};

/// Returns `true` if the topology should be PRUNED (infeasible under interval
/// propagation).
///
/// Uses interval propagation only — cheap and always available when the `smt`
/// feature is enabled. The constraint is wrapped in a `GeZero` envelope before
/// propagation (checks that the tree's output can be ≥ 0).
pub fn interval_prune(constraint: &EmlConstraint, vars: &[Interval]) -> bool {
    let bounds: Vec<(f64, f64)> = vars.iter().map(|iv| (iv.lo, iv.hi)).collect();
    let n = bounds.len();
    let mut domain = IntervalDomain::new(&bounds, n);
    domain.propagate(constraint) == PropResult::Conflict
}

/// Returns `true` if the topology should be PRUNED (proved UNSAT by OxiZ).
///
/// Reuses this thread's live OxiZ solver via `push`/`pop` — still expensive
/// relative to `interval_prune`, but no longer paying for solver construction per
/// candidate. Only call when `smt_prune_solver` is enabled in the config.
/// Depth-gated: returns `false` immediately when `current_depth < min_depth`.
///
/// # Soundness
/// An `Unsat` result is sound — if the LRA relaxation of the constraint is UNSAT
/// then the original (nonlinear) problem is also UNSAT, so pruning is safe. `Sat`
/// and `Unknown` are treated conservatively (no pruning), so an undecided query
/// can only cost time, never a viable topology.
///
/// Reusing the solver does not weaken this. Each candidate is checked inside its
/// own `push`/`pop` scope, and every term asserted in that scope is created
/// inside it, so the assertion set `check()` sees is exactly the one a fresh
/// solver would see. See [`crate::smt::incremental`] for the full argument, and
/// `cached_pruner_agrees_with_one_shot_solver` below for the differential test.
pub fn solver_prune(
    constraint: &EmlConstraint,
    vars: &[Interval],
    min_depth: usize,
    current_depth: usize,
) -> bool {
    if current_depth < min_depth {
        return false;
    }
    let bounds: Vec<(f64, f64)> = vars.iter().map(|iv| (iv.lo, iv.hi)).collect();

    with_thread_local_solver(&bounds, |solver| {
        matches!(solver.check_sat(constraint), Ok(SmtResult::Unsat))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::{EmlConstraint, Interval, solver_construction_count};
    use crate::tree::{EmlNode, EmlTree};
    use std::sync::Arc;

    fn var0_tree() -> EmlTree {
        EmlTree::from_node(Arc::new(EmlNode::Var(0)))
    }

    fn const_neg_one_tree() -> EmlTree {
        EmlTree::from_node(Arc::new(EmlNode::Const(-1.0)))
    }

    #[test]
    fn test_interval_prune_conflict() {
        // A Const(-1.0) tree always outputs -1.0 → GeZero is always false → Conflict
        let c = EmlConstraint::GeZero(const_neg_one_tree());
        let vars = vec![Interval::new(0.0, 3.0)];
        assert!(interval_prune(&c, &vars), "Const -1 >= 0 should conflict");
    }

    #[test]
    fn test_interval_prune_feasible() {
        // var0 ∈ [0.0, 5.0], constraint: GeZero(Var(0)) → var0 >= 0 → feasible
        let c = EmlConstraint::GeZero(var0_tree());
        let vars = vec![Interval::new(0.0, 5.0)];
        assert!(
            !interval_prune(&c, &vars),
            "var0 in [0,5] >= 0 should be feasible"
        );
    }

    #[test]
    fn test_solver_prune_depth_gate() {
        // Even if constraint is UNSAT, depth gate should prevent pruning
        let c = EmlConstraint::GeZero(const_neg_one_tree());
        let vars = vec![Interval::new(0.0, 1.0)];
        // min_depth=3, current_depth=1 → depth gate fires, no prune
        assert!(
            !solver_prune(&c, &vars, 3, 1),
            "Below min_depth should not prune"
        );
    }

    #[test]
    fn test_solver_prune_unsat() {
        // Clear UNSAT: Const(-1.0) >= 0 is never true
        let c = EmlConstraint::GeZero(const_neg_one_tree());
        let vars = vec![Interval::new(0.0, 1.0)];
        assert!(solver_prune(&c, &vars, 0, 0), "Clear UNSAT should prune");
    }

    #[test]
    fn test_solver_prune_sat_not_pruned() {
        // SAT: var0 >= 0 with var0 ∈ [0, 10] → feasible
        let c = EmlConstraint::GeZero(var0_tree());
        let vars = vec![Interval::new(0.0, 10.0)];
        assert!(
            !solver_prune(&c, &vars, 0, 0),
            "SAT constraint should not prune"
        );
    }

    /// The `n`-th of a family of structurally distinct EML topologies: different
    /// shapes and different constants, mixing feasible and infeasible cases, so
    /// the pruner cannot short-cut the sweep by memoising one formula.
    fn topology(n: usize) -> EmlTree {
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let k = EmlTree::from_node(Arc::new(EmlNode::Const(1.0 + (n % 7) as f64)));
        match n % 4 {
            0 => EmlTree::eml(&x, &one),
            1 => EmlTree::eml(&EmlTree::eml(&x, &one), &k),
            2 => EmlTree::eml(&k, &EmlTree::eml(&x, &one)),
            _ => EmlTree::eml(&EmlTree::eml(&k, &one), &EmlTree::eml(&x, &k)),
        }
    }

    /// The reuse guarantee, made measurable: screening 60 distinct topologies at a
    /// fixed variable box must construct **exactly one** OxiZ `Solver`. Before
    /// this change the same sweep constructed 60.
    ///
    /// The counter is thread-local, as is the cached solver, so this is
    /// deterministic even under a parallel test runner.
    #[test]
    fn solver_construction_count_is_one_across_many_topologies() {
        crate::smt::reset_thread_local_solver();
        let vars = vec![Interval::new(0.5, 4.0)];

        let before = solver_construction_count();
        for n in 0..60 {
            let c = EmlConstraint::GeZero(topology(n));
            // The verdict is irrelevant here; we are measuring solver construction.
            let _ = solver_prune(&c, &vars, 0, 0);
        }
        let after = solver_construction_count();

        assert_eq!(
            after - before,
            1,
            "60 topologies at a fixed box must reuse a single OxiZ Solver, but {} were constructed",
            after - before
        );
    }

    /// Reuse must not change verdicts: the cached-solver pruner must agree with a
    /// fresh one-shot `EmlSmtSolver` on every topology of the sweep.
    #[test]
    fn cached_pruner_agrees_with_one_shot_solver() {
        crate::smt::reset_thread_local_solver();
        let vars = vec![Interval::new(0.5, 4.0)];
        let bounds: Vec<(f64, f64)> = vars.iter().map(|iv| (iv.lo, iv.hi)).collect();

        for n in 0..60 {
            let c = EmlConstraint::GeZero(topology(n));

            let cached = solver_prune(&c, &vars, 0, 0);

            let one_shot = crate::smt::EmlSmtSolver::new(bounds.clone());
            let expected = matches!(one_shot.check_sat(&c), Ok(crate::smt::SmtResult::Unsat));

            assert_eq!(
                cached, expected,
                "topology {n}: the reused solver disagreed with a fresh one-shot solver"
            );
        }
    }

    /// A change of variable box must rebuild the solver — screening against a
    /// stale box would check candidates over the wrong domain.
    #[test]
    fn changing_bounds_rebuilds_the_solver() {
        crate::smt::reset_thread_local_solver();
        let c = EmlConstraint::GeZero(var0_tree());

        let before = solver_construction_count();
        let _ = solver_prune(&c, &[Interval::new(0.0, 1.0)], 0, 0);
        let _ = solver_prune(&c, &[Interval::new(0.0, 1.0)], 0, 0);
        let _ = solver_prune(&c, &[Interval::new(2.0, 3.0)], 0, 0);
        let after = solver_construction_count();

        assert_eq!(
            after - before,
            2,
            "the same box must reuse the solver and a changed box must rebuild it"
        );
    }
}
