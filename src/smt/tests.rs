use super::*;
use crate::EmlTree;
use crate::canonical::Canonical;

#[test]
fn test_eq_zero_trivial() {
    let one = EmlTree::one();
    let ln_one = Canonical::ln(&one);
    let constraint = EmlConstraint::EqZero(ln_one);
    let solver = EmlNraSolver::default();
    assert!(solver.solve(&constraint).is_ok());
}

#[test]
fn test_gt_zero() {
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let exp_x = EmlTree::eml(&x, &one);
    let constraint = EmlConstraint::GtZero(exp_x);
    let solver = EmlNraSolver::new(vec![(-5.0, 5.0)]);
    assert!(solver.solve(&constraint).is_ok());
}

#[test]
fn test_and_constraint() {
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let x_tree = EmlTree::var(0);
    let exp_x = EmlTree::eml(&x, &one);
    let e_minus_x = EmlTree::eml(&one, &exp_x); // e - x
    let constraint = EmlConstraint::And(vec![
        EmlConstraint::GtZero(x_tree),
        EmlConstraint::GtZero(e_minus_x),
    ]);
    let solver = EmlNraSolver::new(vec![(0.1, 2.5)]);
    let result = solver.solve(&constraint).expect("expected Sat solution");
    let x_val = result.assignments[0];
    assert!(x_val > 0.0 && x_val < std::f64::consts::E);
}

#[test]
fn test_interval_exp_forward() {
    let iv = Interval::new(0.0, 2.0);
    let exp_iv = iv.exp();
    assert!((exp_iv.lo - 1.0).abs() < 1e-12);
    assert!((exp_iv.hi - 2.0_f64.exp()).abs() < 1e-12);
}

#[test]
fn test_interval_ln_forward() {
    let iv = Interval::new(1.0, std::f64::consts::E);
    let ln_iv = iv.ln();
    assert!(ln_iv.lo.abs() < 1e-12);
    assert!((ln_iv.hi - 1.0).abs() < 1e-12);
}

#[test]
fn test_interval_ln_negative_empty() {
    let iv = Interval::new(-1.0, 1.0);
    let ln_iv = iv.ln();
    assert!(ln_iv.is_empty());
}

#[test]
fn test_interval_intersect_and_hull() {
    let a = Interval::new(0.0, 2.0);
    let b = Interval::new(1.0, 3.0);
    let inter = a.intersect(&b);
    assert!((inter.lo - 1.0).abs() < 1e-12);
    assert!((inter.hi - 2.0).abs() < 1e-12);
    let hull = a.hull(&b);
    assert!((hull.lo - 0.0).abs() < 1e-12);
    assert!((hull.hi - 3.0).abs() < 1e-12);
}

#[test]
fn test_interval_disjoint_intersect_empty() {
    let a = Interval::new(0.0, 1.0);
    let b = Interval::new(2.0, 3.0);
    assert!(a.intersect(&b).is_empty());
}

#[test]
fn test_propagate_exp_positivity_conflict() {
    // exp(x) = 0 can never be satisfied.
    let x = EmlTree::var(0);
    let one = EmlTree::one();
    let exp_x = EmlTree::eml(&x, &one);
    let c = EmlConstraint::EqZero(exp_x);
    let mut domain = IntervalDomain::new(&[(-5.0, 5.0)], 1);
    assert_eq!(domain.propagate(&c), PropResult::Conflict);
}
