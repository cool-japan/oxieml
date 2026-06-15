//! Shared SMT-based pruning for symbolic regression.
//!
//! `interval_prune` (cheap, always available under `smt` feature): uses
//! `IntervalDomain::propagate` to check feasibility.
//!
//! `solver_prune` (OxiZ-backed, expensive, opt-in): calls
//! `EmlSmtSolver::check_sat`. Depth-gated to avoid spending time on tiny
//! partial topologies.

use crate::smt::{EmlConstraint, Interval, IntervalDomain, PropResult};

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
/// Uses `EmlSmtSolver::check_sat` — expensive. Only call when
/// `smt_prune_solver` is enabled in the config. Depth-gated: returns `false`
/// immediately when `current_depth < min_depth`.
///
/// # Soundness
/// An `Unsat` result from `check_sat` is sound — if the LRA relaxation is
/// UNSAT, the original (nonlinear) problem is also UNSAT, so pruning is safe.
/// `Sat` and `Unknown` are treated conservatively (no pruning).
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
    let solver = crate::smt::EmlSmtSolver::new(bounds);
    match solver.check_sat(constraint) {
        Ok(crate::smt::SmtResult::Unsat) => true,
        Ok(crate::smt::SmtResult::Sat(_)) | Ok(crate::smt::SmtResult::Unknown) | Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::{EmlConstraint, Interval};
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
}
