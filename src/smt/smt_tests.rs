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

// ────────────────────────────────────────────────────────────────────────────
// S1: incremental solver (push/pop), unsat cores, MaxSMT
// ────────────────────────────────────────────────────────────────────────────

/// Differential and hygiene tests for [`IncrementalEmlSolver`].
///
/// The contract under test is the one that makes solver reuse *safe*: a query
/// answered inside a `push`/`pop` scope on a solver that has already answered
/// many other queries must get the **same verdict** as a query answered by a
/// brand-new solver. An assertion that leaked past a `pop()` would show up here
/// as a spurious `Unsat` (or a spurious `Sat` from a leaked disjunct), which is
/// exactly the class of bug that silently destroys a symbolic-regression search.
mod s1_incremental {
    use super::*;
    use crate::smt::{
        EmlSmtSolver, IncrementalEmlSolver, SmtResult, solver_construction_count,
        with_thread_local_solver,
    };
    use crate::tree::EmlNode;
    use std::sync::Arc;

    /// Deterministic 64-bit LCG — no `rand` dependency, fully reproducible.
    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    fn konst(v: f64) -> EmlTree {
        EmlTree::from_node(Arc::new(EmlNode::Const(v)))
    }

    /// A pseudo-random EML tree of the given depth over variables `0..n_vars`.
    fn random_tree(depth: usize, n_vars: usize, rng: &mut u64) -> EmlTree {
        if depth == 0 {
            return match next(rng) % 4 {
                0 => EmlTree::one(),
                1 => konst(1.0 + (next(rng) % 5) as f64),
                2 => konst(-(1.0 + (next(rng) % 3) as f64)),
                _ => EmlTree::var((next(rng) as usize) % n_vars),
            };
        }
        let left = random_tree(depth - 1, n_vars, rng);
        let right = random_tree(depth - 1, n_vars, rng);
        EmlTree::eml(&left, &right)
    }

    /// A pseudo-random constraint. Deliberately spans every atom kind plus `And`
    /// and `Or`, so the encoder's whole surface is exercised — including the
    /// `ln`-of-a-non-positive-interval cases that must come back `Unknown`
    /// (Issue #1) rather than `Unsat`.
    fn random_constraint(n_vars: usize, rng: &mut u64) -> EmlConstraint {
        let depth = 1 + (next(rng) % 3) as usize;
        let tree = random_tree(depth, n_vars, rng);
        match next(rng) % 8 {
            0 => EmlConstraint::GeZero(tree),
            1 => EmlConstraint::GtZero(tree),
            2 => EmlConstraint::LeZero(tree),
            3 => EmlConstraint::LtZero(tree),
            4 => EmlConstraint::EqZero(tree),
            5 => EmlConstraint::NeZero(tree),
            6 => EmlConstraint::And(vec![
                EmlConstraint::GeZero(tree),
                EmlConstraint::LeZero(random_tree(1, n_vars, rng)),
            ]),
            _ => EmlConstraint::Or(vec![
                EmlConstraint::GtZero(tree),
                EmlConstraint::LtZero(random_tree(1, n_vars, rng)),
            ]),
        }
    }

    /// Name of the verdict, for exact comparison and readable failures.
    fn verdict(r: &SmtResult) -> &'static str {
        match r {
            SmtResult::Sat(_) => "Sat",
            SmtResult::Unsat => "Unsat",
            SmtResult::Unknown => "Unknown",
        }
    }

    fn corpus(n_vars: usize, count: usize, seed: u64) -> Vec<EmlConstraint> {
        let mut rng = seed;
        let mut out: Vec<EmlConstraint> = Vec::with_capacity(count + 4);

        // Anchor the corpus with cases whose verdict we know by hand, so the
        // battery can never degenerate into "everything is Unknown, so the two
        // solvers trivially agree".
        let x = EmlTree::var(0);
        // exp(x) > 0 — always true.
        out.push(EmlConstraint::GtZero(EmlTree::eml(&x, &EmlTree::one())));
        // -1 >= 0 — always false.
        out.push(EmlConstraint::GeZero(konst(-1.0)));
        // x >= 0 ∧ x < 0 — contradictory.
        out.push(EmlConstraint::And(vec![
            EmlConstraint::GeZero(EmlTree::var(0)),
            EmlConstraint::LtZero(EmlTree::var(0)),
        ]));
        // exp(x) = 0 — unsatisfiable by positivity of exp.
        out.push(EmlConstraint::EqZero(EmlTree::eml(&x, &EmlTree::one())));

        while out.len() < count {
            out.push(random_constraint(n_vars, &mut rng));
        }
        out
    }

    /// **The spec's headline test.** Push/pop verdicts must equal N independent
    /// `check_sat` calls.
    ///
    /// One `IncrementalEmlSolver` answers the whole corpus in sequence; each
    /// constraint is *also* answered by a freshly constructed `EmlSmtSolver`. The
    /// verdicts must match exactly, constraint by constraint. Because the two
    /// share the encoding and the post-processing pipeline and differ *only* in
    /// solver lifecycle, any disagreement is a push/pop hygiene bug.
    #[test]
    fn push_pop_verdicts_equal_n_independent_check_sat() {
        for (bounds, seed) in [
            (vec![(0.5, 4.0)], 0xC0FFEE_u64),
            (vec![(-2.0, 2.0)], 0xBADF00D_u64),
            (vec![(0.25, 3.0), (0.5, 5.0)], 0x5EED_u64),
        ] {
            let n_vars = bounds.len();
            let constraints = corpus(n_vars, 64, seed);

            let mut incremental = IncrementalEmlSolver::new(bounds.clone());

            let mut agreed_sat = 0usize;
            let mut agreed_unsat = 0usize;

            for (i, c) in constraints.iter().enumerate() {
                let independent = EmlSmtSolver::new(bounds.clone())
                    .check_sat(c)
                    .expect("one-shot check_sat must not error");
                let reused = incremental
                    .check_sat(c)
                    .expect("incremental check_sat must not error");

                assert_eq!(
                    verdict(&reused),
                    verdict(&independent),
                    "bounds {bounds:?}, constraint #{i} ({c}): the reused solver said {} \
                     but an independent solver said {} — an assertion leaked across push/pop",
                    verdict(&reused),
                    verdict(&independent),
                );

                match reused {
                    SmtResult::Sat(sol) => {
                        agreed_sat += 1;
                        // A `Sat` must be a *real* `Sat`: the witness has to
                        // satisfy the original nonlinear constraint.
                        let ctx = crate::EvalCtx::new(&sol.assignments);
                        assert!(
                            check_constraint(c, &ctx),
                            "constraint #{i} ({c}): reported Sat with a witness that does not \
                             satisfy it — {:?}",
                            sol.assignments
                        );
                    }
                    SmtResult::Unsat => agreed_unsat += 1,
                    SmtResult::Unknown => {}
                }
            }

            // Guard against a vacuous pass.
            assert!(
                agreed_sat >= 4 && agreed_unsat >= 2,
                "bounds {bounds:?}: corpus is not discriminating enough \
                 (sat={agreed_sat}, unsat={agreed_unsat})"
            );
        }
    }

    /// Order-independence: the same constraint must get the same verdict wherever
    /// it appears in the query sequence. A leak would make a verdict depend on
    /// what was checked *before* it.
    #[test]
    fn push_pop_verdicts_are_order_independent() {
        let bounds = vec![(0.5, 4.0)];
        let constraints = corpus(1, 56, 0x1234_5678);

        let mut forward = IncrementalEmlSolver::new(bounds.clone());
        let forward_verdicts: Vec<&'static str> = constraints
            .iter()
            .map(|c| verdict(&forward.check_sat(c).expect("check_sat")))
            .collect();

        // Same solver instance, reversed order.
        let mut reverse = IncrementalEmlSolver::new(bounds.clone());
        let mut reverse_verdicts: Vec<&'static str> = constraints
            .iter()
            .rev()
            .map(|c| verdict(&reverse.check_sat(c).expect("check_sat")))
            .collect();
        reverse_verdicts.reverse();

        assert_eq!(
            forward_verdicts, reverse_verdicts,
            "verdicts depend on query order — state is leaking between scopes"
        );
    }

    /// Push/pop stack hygiene.
    ///
    /// The assertion stack must be empty before and after every query, and an
    /// UNSAT query must not poison the queries that follow it. Interleaving a
    /// contradiction with a tautology 40 times is the sharpest form of this: if
    /// `x >= 0 ∧ x < 0` leaked, the very next `exp(x) > 0` would come back
    /// `Unsat`.
    #[test]
    fn push_pop_stack_hygiene() {
        let mut solver = IncrementalEmlSolver::new(vec![(-2.0, 2.0)]);

        let contradiction = EmlConstraint::And(vec![
            EmlConstraint::GeZero(EmlTree::var(0)),
            EmlConstraint::LtZero(EmlTree::var(0)),
        ]);
        let tautology = EmlConstraint::GtZero(EmlTree::eml(&EmlTree::var(0), &EmlTree::one()));

        assert_eq!(
            solver.assertion_depth(),
            0,
            "fresh solver must be at depth 0"
        );

        for round in 0..40 {
            let unsat = solver.check_sat(&contradiction).expect("check_sat");
            assert!(
                unsat.is_unsat(),
                "round {round}: the contradiction must stay Unsat"
            );
            assert_eq!(
                solver.assertion_depth(),
                0,
                "round {round}: depth must return to 0 after an Unsat query"
            );

            let sat = solver.check_sat(&tautology).expect("check_sat");
            assert!(
                sat.is_sat(),
                "round {round}: exp(x) > 0 must stay Sat — the preceding contradiction \
                 leaked past its pop()"
            );
            assert_eq!(
                solver.assertion_depth(),
                0,
                "round {round}: depth must return to 0 after a Sat query"
            );
        }

        // Note this is 40, not 80: the contradiction `x >= 0 ∧ x < 0` is refuted by
        // interval propagation before OxiZ is ever consulted, so it opens no scope.
        // Only the 40 tautology queries reach the LRA layer.
        assert_eq!(
            solver.lra_scopes(),
            40,
            "only the queries that reach the LRA layer open a push/pop scope"
        );
    }

    /// Recycling the underlying solver must not change any verdict.
    #[test]
    fn recycling_preserves_verdicts() {
        let bounds = vec![(0.5, 4.0)];
        let constraints = corpus(1, 52, 0xABCDEF);

        let mut never = IncrementalEmlSolver::new(bounds.clone()).with_recycle_after(0);
        let mut often = IncrementalEmlSolver::new(bounds.clone()).with_recycle_after(3);

        for (i, c) in constraints.iter().enumerate() {
            let a = never.check_sat(c).expect("check_sat");
            let b = often.check_sat(c).expect("check_sat");
            assert_eq!(
                verdict(&a),
                verdict(&b),
                "constraint #{i} ({c}): recycling changed the verdict"
            );
        }
    }

    /// One solver, many topologies: the reuse is real and measurable.
    #[test]
    fn one_solver_construction_across_sixty_topologies() {
        let bounds = vec![(0.5, 4.0)];
        let constraints = corpus(1, 60, 0xFEED);

        let before = solver_construction_count();
        // `recycle_after = 0` ⇒ never rebuilt, so exactly one construction.
        let mut solver = IncrementalEmlSolver::new(bounds).with_recycle_after(0);
        for c in &constraints {
            let _ = solver.check_sat(c).expect("check_sat");
        }
        let after = solver_construction_count();

        assert_eq!(
            after - before,
            1,
            "60 topologies must be answered by a single OxiZ Solver, but {} were constructed",
            after - before
        );
        assert!(
            solver.lra_scopes() > 0,
            "the corpus must actually exercise the LRA layer"
        );
    }

    /// The pooled per-thread solver used by the pruner reuses one instance too.
    #[test]
    fn thread_local_pool_reuses_one_solver() {
        crate::smt::reset_thread_local_solver();
        let bounds = vec![(0.5, 4.0)];
        let constraints = corpus(1, 55, 0x9999);

        let before = solver_construction_count();
        for c in &constraints {
            with_thread_local_solver(&bounds, |solver| {
                let _ = solver.check_sat(c).expect("check_sat");
            });
        }
        let after = solver_construction_count();

        assert_eq!(after - before, 1, "the pool must build exactly one solver");
    }
}

/// Unsat-core extraction: the returned subset must be genuinely infeasible, and
/// **removing any one of its members must make it satisfiable**. That is the
/// definition of minimality and it is checked literally, element by element.
mod s1_unsat_core {
    use super::*;
    use crate::smt::EmlSmtSolver;

    /// Assert the two halves of the minimality contract:
    /// 1. the core itself is UNSAT, and
    /// 2. dropping *any single* member yields a **Sat** verdict.
    fn assert_core_is_minimal(bounds: &[(f64, f64)], all: &[EmlConstraint], core: &[usize]) {
        let solver = EmlSmtSolver::new(bounds.to_vec());

        let core_constraints: Vec<EmlConstraint> = core.iter().map(|&i| all[i].clone()).collect();
        assert!(
            solver
                .check_all(&core_constraints)
                .expect("check_all")
                .is_unsat(),
            "the reported core {core:?} is not actually unsatisfiable"
        );

        for (slot, &dropped) in core.iter().enumerate() {
            let reduced: Vec<EmlConstraint> = core
                .iter()
                .enumerate()
                .filter(|(s, _)| *s != slot)
                .map(|(_, &i)| all[i].clone())
                .collect();
            assert!(
                solver.check_all(&reduced).expect("check_all").is_sat(),
                "core {core:?} is not minimal: dropping constraint #{dropped} \
                 still leaves an infeasible set"
            );
        }
    }

    /// `x >= 0` and `x < 0` are the infeasible pair; the always-true third
    /// constraint is a distractor that must be dropped from the core.
    #[test]
    fn unsat_core_is_minimal_and_drops_the_distractor() {
        let bounds = vec![(-2.0, 2.0)];
        let constraints = vec![
            EmlConstraint::GeZero(EmlTree::var(0)), // 0: x >= 0
            EmlConstraint::LtZero(EmlTree::var(0)), // 1: x < 0
            EmlConstraint::GtZero(EmlTree::eml(&EmlTree::var(0), &EmlTree::one())), // 2: exp(x) > 0
        ];

        let solver = EmlSmtSolver::new(bounds.clone());
        let core = solver
            .unsat_core(&constraints)
            .expect("unsat_core")
            .expect("the set is unsatisfiable, so a core must exist");

        assert_eq!(
            core.indices,
            vec![0, 1],
            "the always-true constraint must not be in the core"
        );
        assert!(
            core.is_verified(),
            "every deletion trial was decidable here, so minimality must be verified"
        );
        assert_core_is_minimal(&bounds, &constraints, &core.indices);
    }

    /// Two *independent* contradictions. Deletion-based extraction returns **one**
    /// irreducible core, not the union of both — and whichever it picks must
    /// satisfy the minimality contract.
    #[test]
    fn unsat_core_with_two_independent_contradictions_is_still_irreducible() {
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let constraints = vec![
            EmlConstraint::GeZero(EmlTree::var(0)), // 0
            EmlConstraint::LtZero(EmlTree::var(0)), // 1  (0 ∧ 1 contradict)
            EmlConstraint::GeZero(EmlTree::var(1)), // 2
            EmlConstraint::LtZero(EmlTree::var(1)), // 3  (2 ∧ 3 contradict)
        ];

        let solver = EmlSmtSolver::new(bounds.clone());
        let core = solver
            .unsat_core(&constraints)
            .expect("unsat_core")
            .expect("the set is unsatisfiable");

        assert_eq!(
            core.len(),
            2,
            "an irreducible core here has exactly two members, got {:?}",
            core.indices
        );
        assert_core_is_minimal(&bounds, &constraints, &core.indices);
    }

    /// A satisfiable set has no core, and we must say so rather than invent one.
    #[test]
    fn no_core_for_a_satisfiable_set() {
        let solver = EmlSmtSolver::new(vec![(-2.0, 2.0)]);
        let constraints = vec![
            EmlConstraint::GeZero(EmlTree::var(0)),
            EmlConstraint::GtZero(EmlTree::eml(&EmlTree::var(0), &EmlTree::one())),
        ];
        assert!(
            solver
                .unsat_core(&constraints)
                .expect("unsat_core")
                .is_none(),
            "a satisfiable set must not produce a core"
        );
    }

    /// A single self-contradictory constraint is its own (unit) core.
    #[test]
    fn singleton_core() {
        let solver = EmlSmtSolver::new(vec![(-2.0, 2.0)]);
        // exp(x) = 0 is unsatisfiable: exp is strictly positive.
        let constraints = vec![EmlConstraint::EqZero(EmlTree::eml(
            &EmlTree::var(0),
            &EmlTree::one(),
        ))];
        let core = solver
            .unsat_core(&constraints)
            .expect("unsat_core")
            .expect("exp(x) = 0 is unsatisfiable");
        assert_eq!(core.indices, vec![0]);
    }
}

/// MaxSMT.
///
/// The instance below is the textbook case where **greedy is provably wrong**:
///
/// | soft | constraint | weight |
/// |------|-----------|--------|
/// | 0    | `x == 0`  | 3      |
/// | 1    | `x > 0`   | 2      |
/// | 2    | `x != 0`  | 2      |
///
/// `{1, 2}` is feasible and worth **4**. Greedy takes the heaviest first — soft 0
/// — and then cannot add either of the others, ending at **3**.
///
/// So the two feature configurations must give *different* answers, and each must
/// label its answer honestly. That is the point: the greedy fallback is a real
/// greedy algorithm that really is suboptimal, and it says so.
mod s1_maxsmt {
    use super::*;
    use crate::smt::{EmlSmtSolver, MaxSmtOptimality, MaxSmtOutcome, SoftConstraint};

    fn greedy_trap() -> (Vec<(f64, f64)>, Vec<EmlConstraint>, Vec<SoftConstraint>) {
        let bounds = vec![(-2.0, 2.0)];
        let hard: Vec<EmlConstraint> = vec![];
        let soft = vec![
            SoftConstraint::new(EmlConstraint::EqZero(EmlTree::var(0)), 3),
            SoftConstraint::new(EmlConstraint::GtZero(EmlTree::var(0)), 2),
            SoftConstraint::new(EmlConstraint::NeZero(EmlTree::var(0)), 2),
        ];
        (bounds, hard, soft)
    }

    fn solved(outcome: MaxSmtOutcome) -> crate::smt::MaxSmtSolution {
        match outcome {
            MaxSmtOutcome::Solved(sol) => sol,
            other => panic!("expected a solved MaxSMT instance, got {other:?}"),
        }
    }

    /// Every reported solution must be *real*: the witness has to satisfy all the
    /// hard constraints and exactly the soft constraints it claims.
    fn assert_solution_is_honest(
        sol: &crate::smt::MaxSmtSolution,
        hard: &[EmlConstraint],
        soft: &[SoftConstraint],
    ) {
        let ctx = crate::EvalCtx::new(&sol.assignments);
        for (i, h) in hard.iter().enumerate() {
            assert!(
                check_constraint(h, &ctx),
                "witness violates hard constraint #{i}"
            );
        }
        let mut weight = 0u64;
        for (i, s) in soft.iter().enumerate() {
            let holds = check_constraint(&s.constraint, &ctx);
            assert_eq!(
                holds,
                sol.satisfied.contains(&i),
                "soft #{i}: reported satisfied = {}, but it actually {} at the witness {:?}",
                sol.satisfied.contains(&i),
                if holds { "holds" } else { "does not hold" },
                sol.assignments
            );
            if holds {
                weight += s.weight;
            }
        }
        assert_eq!(
            weight, sol.weight,
            "reported weight disagrees with the witness"
        );
    }

    /// Exact MaxSMT (`smt-opt`): must find the optimum of 4 that greedy misses.
    #[cfg(feature = "smt-opt")]
    #[test]
    fn maxsmt_finds_the_optimum_that_greedy_misses() {
        let (bounds, hard, soft) = greedy_trap();
        let solver = EmlSmtSolver::new(bounds);

        let sol = solved(solver.max_smt(&hard, &soft).expect("max_smt"));
        assert_solution_is_honest(&sol, &hard, &soft);

        assert_eq!(
            sol.optimality,
            MaxSmtOptimality::Optimal,
            "the core-guided search must prove optimality on this instance"
        );
        assert_eq!(
            sol.weight, 4,
            "the optimum is {{soft 1, soft 2}} = 2 + 2 = 4, got {} (satisfied = {:?})",
            sol.weight, sol.satisfied
        );
        assert_eq!(sol.satisfied, vec![1, 2]);
    }

    /// Greedy fallback (base `smt`): returns the *maximal* selection worth 3, and
    /// labels it `Maximal` rather than pretending it is the optimum.
    #[cfg(not(feature = "smt-opt"))]
    #[test]
    fn maxsmt_greedy_fallback_is_maximal_and_admits_it() {
        let (bounds, hard, soft) = greedy_trap();
        let solver = EmlSmtSolver::new(bounds);

        let sol = solved(solver.max_smt(&hard, &soft).expect("max_smt"));
        assert_solution_is_honest(&sol, &hard, &soft);

        assert_eq!(
            sol.optimality,
            MaxSmtOptimality::Maximal,
            "greedy must not claim optimality"
        );
        assert_ne!(
            sol.optimality,
            MaxSmtOptimality::Optimal,
            "greedy is NOT optimal on this instance and must never say it is"
        );
        assert_eq!(
            sol.weight, 3,
            "weight-ordered greedy commits to the heaviest soft (3) and then gets stuck"
        );
        assert_eq!(sol.satisfied, vec![0]);
    }

    /// An instance where greedy *is* optimal: both engines must agree, and both
    /// must satisfy every soft constraint (they are mutually compatible).
    #[test]
    fn maxsmt_takes_everything_when_the_softs_are_compatible() {
        let bounds = vec![(0.5, 4.0)];
        let hard = vec![EmlConstraint::GtZero(EmlTree::var(0))];
        let soft = vec![
            SoftConstraint::new(
                EmlConstraint::GtZero(EmlTree::eml(&EmlTree::var(0), &EmlTree::one())),
                5,
            ),
            SoftConstraint::new(EmlConstraint::GeZero(EmlTree::var(0)), 1),
        ];

        let solver = EmlSmtSolver::new(bounds);
        let sol = solved(solver.max_smt(&hard, &soft).expect("max_smt"));
        assert_solution_is_honest(&sol, &hard, &soft);

        assert_eq!(
            sol.weight, 6,
            "both soft constraints are satisfiable at once"
        );
        assert_eq!(sol.satisfied, vec![0, 1]);
    }

    /// Unsatisfiable hard constraints ⇒ no solution at all, whatever the softs say.
    #[test]
    fn maxsmt_reports_unsat_hard_constraints() {
        let solver = EmlSmtSolver::new(vec![(-2.0, 2.0)]);
        let hard = vec![
            EmlConstraint::GeZero(EmlTree::var(0)),
            EmlConstraint::LtZero(EmlTree::var(0)),
        ];
        let soft = vec![SoftConstraint::unit(EmlConstraint::GeZero(EmlTree::var(0)))];

        assert!(
            matches!(
                solver.max_smt(&hard, &soft).expect("max_smt"),
                MaxSmtOutcome::Unsat
            ),
            "contradictory hard constraints must yield Unsat"
        );
    }

    /// No soft constraints ⇒ trivially optimal at weight 0.
    #[test]
    fn maxsmt_with_no_soft_constraints() {
        let solver = EmlSmtSolver::new(vec![(0.5, 4.0)]);
        let hard = vec![EmlConstraint::GtZero(EmlTree::var(0))];

        let sol = solved(solver.max_smt(&hard, &[]).expect("max_smt"));
        assert_eq!(sol.weight, 0);
        assert!(sol.satisfied.is_empty());
        assert_eq!(sol.optimality, MaxSmtOptimality::Optimal);
    }
}
