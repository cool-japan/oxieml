//! Regression tests for https://github.com/cool-japan/oxieml/issues/1
//!
//! `EmlConstraint::le/ge/lt/gt/eq` wrap the LHS via `Canonical::sub`, which
//! applies `ln(a)`. When `a` can be <= 0 over the variable domain (e.g.
//! `f = exp(x0) - 1` over `x0 ∈ [-1, 1]` ranges `[-0.632, 1.718]`), the
//! real-domain `Interval::ln` of that operand is empty. The interval walkers
//! used to turn that emptiness into a `Conflict`, producing a spurious
//! `Unsat`. EML's `sub`/`ln` constructions are evaluated in the COMPLEX domain,
//! where such an intermediate `ln` is legitimate (imaginary parts cancel) and
//! the final real value is well-defined, so real interval arithmetic must treat
//! `ln` of a non-positive operand as INDETERMINATE, never as an infeasibility.
//!
//! NOTE: `EmlConstraint`/`IntervalDomain`/`PropResult` live in the `oxieml::smt`
//! module, which is feature-gated (`#[cfg(feature = "smt")]` in lib.rs). The
//! interval-layer anchor is therefore compiled under the same feature gate,
//! mirroring `tests/smt_interval_test.rs`; without the feature this file is an
//! empty test binary.

// (A) Interval-layer anchor: the propagator must never spuriously `Conflict`.
#[cfg(feature = "smt")]
mod interval_anchor {
    use oxieml::{
        EmlTree,
        smt::{EmlConstraint, IntervalDomain, PropResult},
    };

    #[test]
    fn interval_layer_ln_operand_never_conflicts() {
        // f = eml(var(0), const e) = exp(x0) - ln(e) = exp(x0) - 1; reaches <= 0 on [-1,1].
        let f = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(std::f64::consts::E));
        let z = EmlTree::const_val(0.0);

        let cases = [
            ("le", EmlConstraint::le(f.clone(), z.clone())),
            ("ge", EmlConstraint::ge(f.clone(), z.clone())),
            ("lt", EmlConstraint::lt(f.clone(), z.clone())),
            ("gt", EmlConstraint::gt(f.clone(), z.clone())),
            ("eq", EmlConstraint::eq(f.clone(), z.clone())),
        ];

        for (name, c) in cases {
            let mut d = IntervalDomain::new(&[(-1.0, 1.0)], 1);
            assert_ne!(
                d.propagate(&c),
                PropResult::Conflict,
                "interval propagation must not spuriously conflict on `{name}(f, 0)`"
            );
        }
    }
}

// (B) Full solver: f-cases must NEVER be Unsat (and are Sat with a real witness).
#[cfg(feature = "smt")]
mod smt_cases {
    use oxieml::{
        EmlTree, EvalCtx,
        smt::{EmlConstraint, EmlSmtSolver, SmtResult},
    };

    /// Re-evaluate `f = exp(x0) - 1` (real-valued everywhere) at a witness.
    fn f_at(f: &EmlTree, witness: &[f64]) -> f64 {
        f.eval_real(&EvalCtx::new(witness))
            .expect("f = exp(x0) - 1 is real-valued for all x0")
    }

    #[test]
    fn issue1_seven_cases_are_sound() {
        let z = EmlTree::const_val(0.0);
        // g = eml(var0, 1) = exp(x0) - ln(1) = exp(x0); strictly positive on [-1,1].
        let g = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        // f = eml(var0, e) = exp(x0) - ln(e) = exp(x0) - 1; reaches <= 0 on [-1,1].
        let f = EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(std::f64::consts::E));

        // ── g cases: strictly-positive operand ⇒ soundly decidable ──────────
        // le(g, 0): exp(x0) <= 0 is never true ⇒ Unsat.
        let r = EmlSmtSolver::new(vec![(-1.0, 1.0)])
            .check_sat(&EmlConstraint::le(g.clone(), z.clone()))
            .expect("check_sat error");
        assert!(
            matches!(r, SmtResult::Unsat),
            "le(g,0): exp(x0) <= 0 must be Unsat, got {r:?}"
        );
        // ge(g, 0): exp(x0) >= 0 is always true ⇒ Sat.
        let r = EmlSmtSolver::new(vec![(-1.0, 1.0)])
            .check_sat(&EmlConstraint::ge(g.clone(), z.clone()))
            .expect("check_sat error");
        assert!(
            matches!(r, SmtResult::Sat(_)),
            "ge(g,0): exp(x0) >= 0 must be Sat, got {r:?}"
        );

        // ── f cases: the soundness bug. NEVER Unsat; Sat with a verified witness.
        let f_cases = [
            ("le", EmlConstraint::le(f.clone(), z.clone())),
            ("ge", EmlConstraint::ge(f.clone(), z.clone())),
            ("lt", EmlConstraint::lt(f.clone(), z.clone())),
            ("gt", EmlConstraint::gt(f.clone(), z.clone())),
            ("eq", EmlConstraint::eq(f.clone(), z.clone())),
        ];

        for (name, c) in f_cases {
            let r = EmlSmtSolver::new(vec![(-1.0, 1.0)])
                .check_sat(&c)
                .expect("check_sat error");

            // HARD soundness requirement: an f-case must NEVER be reported Unsat.
            assert!(
                !matches!(r, SmtResult::Unsat),
                "{name}(f,0) must NEVER be Unsat (the spurious-Unsat soundness bug), got {r:?}"
            );

            // Per analysis the bisection fallback finds a witness for all five.
            match r {
                SmtResult::Sat(sol) => {
                    let v = f_at(&f, &sol.assignments);
                    let ok = match name {
                        "le" => v <= 1e-9,
                        "ge" => v >= -1e-9,
                        "lt" => v <= 1e-9,
                        "gt" => v >= -1e-9,
                        "eq" => v.abs() <= 1e-6,
                        _ => unreachable!("unknown case {name}"),
                    };
                    assert!(
                        ok,
                        "{name}(f,0): witness {:?} gives f = {v}, which violates the relation",
                        sol.assignments
                    );
                }
                other => panic!("{name}(f,0): expected Sat with a witness, got {other:?}"),
            }
        }
    }
}
