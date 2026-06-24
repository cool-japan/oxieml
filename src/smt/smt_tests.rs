use super::helpers::check_constraint;
use super::*;
use crate::EmlTree;
use crate::canonical::Canonical;

#[test]
fn test_smt_sat_exp_positive() {
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let exp_x = EmlTree::eml(&x, &one);
    let c = EmlConstraint::GtZero(exp_x);
    let solver = EmlSmtSolver::new(vec![(-3.0, 3.0)]);
    match solver.check_sat(&c).expect("check_sat error") {
        SmtResult::Sat(_) => {}
        other => panic!("expected Sat, got {other:?}"),
    }
}

#[test]
fn test_smt_ln_bracket() {
    // ln(x) > 0 on [1.1, 5.0] should be Sat.
    let x = EmlTree::var(0);
    let ln_x = Canonical::ln(&x);
    let gt = EmlConstraint::GtZero(ln_x);
    let solver = EmlSmtSolver::new(vec![(1.1, 5.0)]);
    assert!(matches!(
        solver.check_sat(&gt).expect("check_sat error"),
        SmtResult::Sat(_)
    ));
}

#[test]
fn test_smt_ln_of_negative_is_unknown_not_unsat() {
    // ln(x) > 0 with x ∈ [-2, -1]: `Canonical::ln` builds an EML tree whose real
    // value is non-real over this domain (ln of a negative operand), so real
    // interval arithmetic CANNOT soundly certify infeasibility. The old solver
    // claimed `Unsat` via the (unsound) "empty ⇒ conflict" mechanism; after the
    // fix the `ln`-of-non-positive operand is treated as indeterminate and the
    // solver honestly returns `Unknown` instead of a spurious `Unsat`.
    let x = EmlTree::var(0);
    let ln_x = Canonical::ln(&x);
    let c = EmlConstraint::GtZero(ln_x);
    let solver = EmlSmtSolver::new(vec![(-2.0, -1.0)]);
    assert!(matches!(
        solver.check_sat(&c).expect("check_sat error"),
        SmtResult::Unknown
    ));
}

#[test]
fn test_smt_witness_verifies() {
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let exp_x = EmlTree::eml(&x, &one);
    let c = EmlConstraint::GtZero(exp_x);
    let solver = EmlSmtSolver::new(vec![(-1.0, 1.0)]);
    match solver.check_sat(&c).expect("check_sat error") {
        SmtResult::Sat(sol) => {
            let ctx = crate::eval::EvalCtx::new(&sol.assignments);
            assert!(check_constraint(&c, &ctx));
        }
        other => panic!("expected Sat, got {other:?}"),
    }
}

#[test]
fn test_smt_constant_true() {
    // ln(1) = 0 satisfies EqZero trivially; no free variables.
    let one = EmlTree::one();
    let ln_one = Canonical::ln(&one);
    let c = EmlConstraint::EqZero(ln_one);
    let solver = EmlSmtSolver::default();
    assert!(matches!(
        solver.check_sat(&c).expect("check_sat error"),
        SmtResult::Sat(_)
    ));
}

#[cfg(test)]
mod f3_tests {
    use super::super::constraint::EmlConstraint;
    use super::super::helpers::check_constraint;
    use crate::canonical::Canonical;
    use crate::eval::EvalCtx;

    #[test]
    fn lt_zero_basic() {
        let x = crate::EmlTree::var(0);
        let one = crate::EmlTree::one();
        let constraint = EmlConstraint::LtZero(Canonical::sub(&x, &one));
        let ctx_satisfy = EvalCtx::new(&[0.5]);
        let ctx_violate = EvalCtx::new(&[2.0]);
        assert!(check_constraint(&constraint, &ctx_satisfy));
        assert!(!check_constraint(&constraint, &ctx_violate));
    }

    #[test]
    fn not_eq_zero_becomes_ne_zero() {
        let x = crate::EmlTree::var(0);
        let c = EmlConstraint::Not(Box::new(EmlConstraint::EqZero(x)));
        let nnf = c.to_nnf();
        assert!(matches!(nnf, EmlConstraint::NeZero(_)));
    }

    #[test]
    fn ne_zero_excludes_only_zero() {
        let x = crate::EmlTree::var(0);
        let c = EmlConstraint::NeZero(x);
        let ctx_zero = EvalCtx::new(&[0.0]);
        let ctx_nonzero = EvalCtx::new(&[1.0]);
        assert!(!check_constraint(&c, &ctx_zero));
        assert!(check_constraint(&c, &ctx_nonzero));
    }

    #[test]
    fn binary_lt_helper() {
        let a = crate::EmlTree::var(0);
        let b = crate::EmlTree::var(1);
        let c = EmlConstraint::lt(a, b);
        let ctx_pass = EvalCtx::new(&[1.0, 2.0]);
        let ctx_fail = EvalCtx::new(&[3.0, 2.0]);
        assert!(check_constraint(&c, &ctx_pass));
        assert!(!check_constraint(&c, &ctx_fail));
    }
}

#[cfg(test)]
mod f1_tests {
    use super::super::helpers::check_constraint;
    use super::*;
    use crate::eval::EvalCtx;

    #[test]
    fn every_sat_re_verifies_exp_gt_zero() {
        let solver = EmlSmtSolver::default();
        let x = crate::EmlTree::var(0);
        let one = crate::EmlTree::one();
        let exp_x = crate::EmlTree::eml(&x, &one); // exp(x)
        let c = EmlConstraint::GtZero(exp_x);
        if let Ok(SmtResult::Sat(sol)) = solver.check_sat(&c) {
            let ctx = EvalCtx::new(&sol.assignments);
            assert!(
                check_constraint(&c, &ctx),
                "Sat witness should satisfy the constraint"
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// J1 — Bounded quantifiers + J3 — disjunction hull / NeZero splitting tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod j1_j3_tests {
    use super::super::constraint::EmlConstraint;
    use super::super::helpers::{QuantResult, decide_exists, decide_forall};
    use super::super::interval::{Interval, IntervalDomain, PropResult};
    use crate::EmlTree;
    use crate::canonical::Canonical;

    // ── J1: NeZero NNF / negate ───────────────────────────────────────────

    #[test]
    fn nnf_forall_negate_yields_exists() {
        // ¬(∀x∈[0,1].φ) should become ∃x∈[0,1].¬φ
        let x = EmlTree::var(0);
        let body = EmlConstraint::GtZero(x.clone());
        let forall = EmlConstraint::ForAll {
            var: 0,
            lo: 0.0,
            hi: 1.0,
            body: Box::new(body),
        };
        let negated = EmlConstraint::Not(Box::new(forall)).to_nnf();
        assert!(
            matches!(negated, EmlConstraint::Exists { .. }),
            "¬∀ should become ∃"
        );
    }

    #[test]
    fn nnf_exists_negate_yields_forall() {
        // ¬(∃x∈[0,1].φ) should become ∀x∈[0,1].¬φ
        let x = EmlTree::var(0);
        let body = EmlConstraint::GtZero(x.clone());
        let exists = EmlConstraint::Exists {
            var: 0,
            lo: 0.0,
            hi: 1.0,
            body: Box::new(body),
        };
        let negated = EmlConstraint::Not(Box::new(exists)).to_nnf();
        assert!(
            matches!(negated, EmlConstraint::ForAll { .. }),
            "¬∃ should become ∀"
        );
    }

    // ── J1: decide_forall ─────────────────────────────────────────────────

    #[test]
    fn forall_refutation_trivially_true() {
        // ∀x∈[1,10]. x > 0  is trivially true — negation x ≤ 0 conflicts with [1,10].
        let x = EmlTree::var(0);
        // body: x > 0  →  GtZero(x - 0) = GtZero(x)
        let body = EmlConstraint::GtZero(x);
        let result = decide_forall(0, 1.0, 10.0, &body, 1);
        assert!(
            matches!(result, QuantResult::True),
            "∀x∈[1,10].x>0 should be detected as True by interval refutation"
        );
    }

    #[test]
    fn forall_counterexample_falsified() {
        // ∀x∈[-5,5]. x > 0  is false; sampling will find negative counterexample.
        let x = EmlTree::var(0);
        let body = EmlConstraint::GtZero(x);
        let result = decide_forall(0, -5.0, 5.0, &body, 1);
        assert!(
            matches!(result, QuantResult::FalseWithCounterexample { .. }),
            "∀x∈[-5,5].x>0 should be falsified by a counterexample"
        );
    }

    // ── J1: decide_exists ─────────────────────────────────────────────────

    #[test]
    fn exists_witness_found() {
        // ∃x∈[0,10]. x > 5  — midpoint 5.0 barely doesn't satisfy, but 7.5 does.
        let x = EmlTree::var(0);
        let five = EmlTree::const_val(5.0);
        let body = EmlConstraint::GtZero(Canonical::sub(&x, &five));
        let result = decide_exists(0, 0.0, 10.0, &body, 1);
        assert!(
            matches!(result, QuantResult::TrueWithWitness(_)),
            "∃x∈[0,10].x>5 should find a witness"
        );
    }

    #[test]
    fn exists_unknown_when_no_witness_in_narrow_band() {
        // ∃x∈[10,20]. x == 0 — no sample will satisfy this, returns Unknown.
        let x = EmlTree::var(0);
        let body = EmlConstraint::EqZero(x);
        let result = decide_exists(0, 10.0, 20.0, &body, 1);
        // We expect Unknown (never False — sound over-approximation).
        assert!(
            matches!(result, QuantResult::Unknown),
            "∃x∈[10,20].x==0 should return Unknown (not False)"
        );
    }

    // ── J1: EmlSmtSolver integration ──────────────────────────────────────

    #[test]
    fn smt_solver_forall_trivially_true() {
        // ∀x∈[1,5]. exp(x) > 0 — always true; should return Sat.
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let body = EmlConstraint::GtZero(exp_x);
        let c = EmlConstraint::ForAll {
            var: 0,
            lo: 1.0,
            hi: 5.0,
            body: Box::new(body),
        };
        let solver = super::EmlSmtSolver::new(vec![(-10.0, 10.0)]);
        let result = solver.check_sat(&c).expect("check_sat should not error");
        assert!(
            matches!(result, super::SmtResult::Sat(_) | super::SmtResult::Unknown),
            "∀x∈[1,5].exp(x)>0 should be Sat or Unknown, got {result:?}"
        );
    }

    #[test]
    fn smt_solver_exists_finds_witness() {
        // ∃x∈[0,5]. x > 3 — solver should return Sat.
        let x = EmlTree::var(0);
        let three = EmlTree::const_val(3.0);
        let body = EmlConstraint::GtZero(Canonical::sub(&x, &three));
        let c = EmlConstraint::Exists {
            var: 0,
            lo: 0.0,
            hi: 5.0,
            body: Box::new(body),
        };
        let solver = super::EmlSmtSolver::new(vec![(-10.0, 10.0)]);
        let result = solver.check_sat(&c).expect("check_sat should not error");
        assert!(
            matches!(result, super::SmtResult::Sat(_)),
            "∃x∈[0,5].x>3 should be Sat, got {result:?}"
        );
    }

    // ── J3: Or hull tightening ────────────────────────────────────────────

    #[test]
    fn or_with_indeterminate_branch_not_conflict() {
        // Or([exp(x) < 0, ln(x) > 0]) with x ∈ [-2,-1].
        // Branch exp(x) < 0 is genuinely infeasible (exp > 0 everywhere).
        // Branch ln(x) > 0 is INDETERMINATE: `Canonical::ln` applies `ln` to a
        // negative operand, which real interval arithmetic cannot soundly bound,
        // so it is no longer treated as a (false) infeasibility. With one branch
        // indeterminate, the Or is not provably a conflict — `propagate` must NOT
        // return Conflict (the sound fix that avoids the false-Unsat mechanism).
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let ln_x = Canonical::ln(&x);

        let c = EmlConstraint::Or(vec![
            // exp(x) < 0 — always false
            EmlConstraint::LtZero(exp_x),
            // ln(x) > 0 with x ∈ [-2,-1] — indeterminate (ln of negative operand)
            EmlConstraint::GtZero(ln_x),
        ]);
        let mut domain = IntervalDomain::new(&[(-2.0, -1.0)], 1);
        let result = domain.propagate(&c);
        assert_ne!(
            result,
            PropResult::Conflict,
            "ln(x)>0 branch is indeterminate (not provably infeasible) → no Conflict"
        );
    }

    #[test]
    fn or_single_feasible_branch_adopted() {
        // Or([exp(x) < 0, exp(x) > 0]) with x ∈ [-5, 5].
        // Branch exp(x) < 0 is always infeasible (exp > 0 everywhere).
        // Branch exp(x) > 0 is always feasible.
        // Result must not be Conflict.
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one); // exp(x)

        let c = EmlConstraint::Or(vec![
            // exp(x) < 0 — always infeasible
            EmlConstraint::LtZero(exp_x.clone()),
            // exp(x) > 0 — always feasible
            EmlConstraint::GtZero(exp_x),
        ]);
        let mut domain = IntervalDomain::new(&[(-5.0, 5.0)], 1);
        let result = domain.propagate(&c);
        // The feasible branch should prevent Conflict.
        assert_ne!(
            result,
            PropResult::Conflict,
            "At least one Or-branch is feasible"
        );
    }

    // ── J3: NeZero propagation ────────────────────────────────────────────

    #[test]
    fn nezero_conflict_on_point_zero() {
        // x ≠ 0 with x ∈ [0, 0] → Conflict.
        let x = EmlTree::var(0);
        let c = EmlConstraint::NeZero(x);
        let mut domain = IntervalDomain::new(&[(0.0, 0.0)], 1);
        let result = domain.propagate(&c);
        assert_eq!(
            result,
            PropResult::Conflict,
            "x≠0 with x=[0,0] must Conflict"
        );
    }

    #[test]
    fn nezero_nudges_lower_bound_off_zero() {
        // x ≠ 0 with x ∈ [0.0, 5.0] → lower bound should be nudged above 0.
        let x = EmlTree::var(0);
        let c = EmlConstraint::NeZero(x);
        let mut domain = IntervalDomain::new(&[(0.0, 5.0)], 1);
        let result = domain.propagate(&c);
        assert_ne!(
            result,
            PropResult::Conflict,
            "x≠0 with x∈[0,5] should not Conflict"
        );
        let lo = domain.vars[0].lo;
        assert!(
            lo > 0.0,
            "lower bound should be nudged above 0, got lo={lo}"
        );
    }

    #[test]
    fn nezero_nudges_upper_bound_off_zero() {
        // x ≠ 0 with x ∈ [-5.0, 0.0] → upper bound should be nudged below 0.
        let x = EmlTree::var(0);
        let c = EmlConstraint::NeZero(x);
        let mut domain = IntervalDomain::new(&[(-5.0, 0.0)], 1);
        let result = domain.propagate(&c);
        assert_ne!(
            result,
            PropResult::Conflict,
            "x≠0 with x∈[-5,0] should not Conflict"
        );
        let hi = domain.vars[0].hi;
        assert!(
            hi < 0.0,
            "upper bound should be nudged below 0, got hi={hi}"
        );
    }

    #[test]
    fn nezero_stable_when_zero_interior() {
        // x ≠ 0 with x ∈ [-1.0, 1.0] → 0 is interior, can't split — Stable.
        let x = EmlTree::var(0);
        let c = EmlConstraint::NeZero(x);
        let mut domain = IntervalDomain::new(&[(-1.0, 1.0)], 1);
        let result = domain.propagate(&c);
        // Should be Stable (can't represent a hole in a single interval).
        assert_ne!(
            result,
            PropResult::Conflict,
            "x≠0 with 0 interior should not Conflict"
        );
    }

    // ── Display (smoke test) ─────────────────────────────────────────────

    #[test]
    fn forall_exists_display() {
        let x = EmlTree::var(0);
        let body = EmlConstraint::GtZero(x);
        let fa = EmlConstraint::ForAll {
            var: 0,
            lo: 0.0,
            hi: 1.0,
            body: Box::new(body.clone()),
        };
        let ex = EmlConstraint::Exists {
            var: 0,
            lo: -1.0,
            hi: 1.0,
            body: Box::new(body),
        };
        let fa_s = format!("{fa}");
        let ex_s = format!("{ex}");
        assert!(
            fa_s.contains("∀x0"),
            "ForAll display should contain ∀x0, got: {fa_s}"
        );
        assert!(
            ex_s.contains("∃x0"),
            "Exists display should contain ∃x0, got: {ex_s}"
        );
    }

    // ── Interval domain public API smoke tests ────────────────────────────

    #[test]
    fn interval_hull_and_intersect_sanity() {
        let a = Interval::new(0.0, 3.0);
        let b = Interval::new(2.0, 5.0);
        let hull = a.hull(&b);
        assert!((hull.lo - 0.0).abs() < 1e-12);
        assert!((hull.hi - 5.0).abs() < 1e-12);
        let inter = a.intersect(&b);
        assert!((inter.lo - 2.0).abs() < 1e-12);
        assert!((inter.hi - 3.0).abs() < 1e-12);
    }
}
