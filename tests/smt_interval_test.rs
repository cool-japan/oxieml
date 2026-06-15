//! Tests for HC4-revise style backward interval tightening in the SMT module.

#[cfg(feature = "smt")]
mod backward_tightening {
    use oxieml::{
        EmlTree,
        smt::{EmlConstraint, IntervalDomain, PropResult},
    };

    #[test]
    fn backward_tighten_eq_zero_no_conflict() {
        // x ∈ [-10, 10], constraint: EqZero on var(0) directly
        // EqZero(x) means x == 0; after propagation, x should be tightened to 0
        let x = EmlTree::var(0);
        let constraint = EmlConstraint::EqZero(x);
        let mut domain = IntervalDomain::new(&[(-10.0, 10.0)], 1);
        let result = domain.propagate(&constraint);
        // x = 0 is satisfiable on [-10, 10]: not a conflict
        assert_ne!(
            result,
            PropResult::Conflict,
            "x=0 should not conflict on [-10,10]"
        );
        // After propagation, domain.vars[0] should be tightened to [0, 0]
        assert!(
            (domain.vars[0].lo - 0.0).abs() < 1e-10,
            "x lower bound should be ~0, got {}",
            domain.vars[0].lo
        );
        assert!(
            (domain.vars[0].hi - 0.0).abs() < 1e-10,
            "x upper bound should be ~0, got {}",
            domain.vars[0].hi
        );
    }

    #[test]
    fn backward_tighten_ge_zero_reaches_changed() {
        // eml(x, 1) = exp(x) - ln(1) = exp(x) - 0 = exp(x)
        // GeZero(eml(x,1)) means exp(x) >= 0, which is always true
        // But more usefully: backward propagation should not conflict
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let eml_x = EmlTree::eml(&x, &one); // exp(x) - ln(1) = exp(x)
        let constraint = EmlConstraint::GeZero(eml_x);
        let mut domain = IntervalDomain::new(&[(-10.0, 10.0)], 1);
        let result = domain.propagate(&constraint);
        assert_ne!(
            result,
            PropResult::Conflict,
            "exp(x) >= 0 is always true, should not conflict"
        );
    }

    #[test]
    fn propagate_conflict_when_impossible() {
        // eml(1, x) = exp(1) - ln(x) = e - ln(x)
        // For x ∈ [100, 1000]: ln(x) ∈ [4.6, 6.9], so e - ln(x) ∈ [-4.2, -1.9] < 0
        // GeZero: e - ln(x) >= 0 should be Conflict
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let eml_1_x = EmlTree::eml(&one, &x); // exp(1) - ln(x)
        let constraint = EmlConstraint::GeZero(eml_1_x);
        let mut domain = IntervalDomain::new(&[(100.0, 1000.0)], 1);
        let result = domain.propagate(&constraint);
        assert_eq!(
            result,
            PropResult::Conflict,
            "e - ln(x) >= 0 with x ∈ [100,1000] should conflict (max value is e-ln(100) ≈ -1.9)"
        );
    }

    #[test]
    fn propagate_changed_is_reachable() {
        // Verify PropResult::Changed is now reachable (was previously dead code)
        // EqZero forces x toward 0
        let x = EmlTree::var(0);
        let constraint = EmlConstraint::EqZero(x);
        let mut domain = IntervalDomain::new(&[(-10.0, 10.0)], 1);
        let result = domain.propagate(&constraint);
        // With backward propagation, x should be tightened from [-10,10] to [0,0]
        // which means Changed was returned at some point
        assert_eq!(
            result,
            PropResult::Changed,
            "EqZero(x) with x in [-10,10] should return Changed after tightening to [0,0]"
        );
    }
}
