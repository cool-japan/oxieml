//! End-to-end tests for N3: Gröbner bases and nonlinear system solving.
//!
//! These exercise the *public* API surface only — the crate-root re-exports and
//! `oxieml::solve` — as a user would. The per-module unit tests in
//! `src/poly/{monomial,groebner,solve_system}.rs` cover the internals.

use std::sync::Arc;

use oxieml::poly::groebner::groebner_basis_with_stats;
use oxieml::{
    GroebnerOpts, LoweredOp, MonOrder, MultiPoly, SystemSolveResult, groebner_basis, ideal_member,
    solve_linear_system,
};

// ── Builders ─────────────────────────────────────────────────────────────────

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}

fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Arc::new(a), Arc::new(b))
}

fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}

fn powi(a: LoweredOp, e: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(a), Arc::new(c(e)))
}

/// A [`MultiPoly`] from `(coefficient, exponents)` pairs.
fn poly(terms: &[(i64, &[u32])], num_vars: usize) -> MultiPoly {
    let pairs: Vec<_> = terms
        .iter()
        .map(|(k, e)| (oxieml::poly::coeff_from_i64(*k), e.to_vec()))
        .collect();
    oxieml::poly::groebner::multipoly_from_terms(&pairs, num_vars)
        .expect("well-formed test polynomial")
}

/// x² + y² − 1
fn circle() -> MultiPoly {
    poly(&[(1, &[2, 0]), (1, &[0, 2]), (-1, &[0, 0])], 2)
}

/// x − y
fn diagonal() -> MultiPoly {
    poly(&[(1, &[1, 0]), (-1, &[0, 1])], 2)
}

// ── Spec bullet 1: ⟨x²+y²−1, x−y⟩ lex GB → solutions (±1/√2, ±1/√2) ─────────

#[test]
fn lex_groebner_basis_triangularizes_circle_and_diagonal() {
    let basis = groebner_basis(
        &[circle(), diagonal()],
        &GroebnerOpts::with_order(MonOrder::Lex),
    )
    .expect("terminates");

    // The lex basis is {x − y, y² − 1/2}: one polynomial per variable, in a
    // triangular cascade. The eliminant y² − 1/2 lives in ℚ[y] alone.
    assert_eq!(basis.len(), 2, "basis = {basis:?}");

    let eliminant = basis
        .iter()
        .find(|g| !g.involves_var(0))
        .expect("lex must expose an eliminant in y alone");
    assert_eq!(eliminant.degree_in(1), 2);
    // y² − 1/2 vanishes exactly at ±1/√2.
    let r = 1.0 / 2.0_f64.sqrt();
    assert!(eliminant.eval_f64(&[0.0, r]).abs() < 1e-12);
    assert!(eliminant.eval_f64(&[0.0, -r]).abs() < 1e-12);
}

#[test]
fn circle_and_diagonal_solve_to_plus_minus_one_over_root_two() {
    // {x² + y² = 1, x = y} — the headline case from the spec.
    let eq0 = sub(add(powi(var(0), 2.0), powi(var(1), 2.0)), c(1.0));
    let eq1 = sub(var(0), var(1));

    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("solvable");

    let SystemSolveResult::NonlinearSolutions(sols) = result else {
        panic!("expected NonlinearSolutions, got {result:?}");
    };
    assert_eq!(sols.len(), 2, "two real intersection points");

    let r = 1.0 / 2.0_f64.sqrt();
    let points: Vec<(f64, f64)> = sols
        .iter()
        .map(|s| {
            assert_eq!(s.len(), 2);
            (s[0].eval(&[]), s[1].eval(&[]))
        })
        .collect();

    // Both coordinates share a sign, because x = y.
    assert!(
        points
            .iter()
            .any(|(x, y)| (x - r).abs() < 1e-12 && (y - r).abs() < 1e-12),
        "missing (+1/√2, +1/√2): {points:?}"
    );
    assert!(
        points
            .iter()
            .any(|(x, y)| (x + r).abs() < 1e-12 && (y + r).abs() < 1e-12),
        "missing (−1/√2, −1/√2): {points:?}"
    );

    // And every point genuinely solves the original system.
    for (x, y) in &points {
        assert!((x * x + y * y - 1.0).abs() < 1e-12, "off the circle");
        assert!((x - y).abs() < 1e-12, "off the diagonal");
    }
}

// ── Spec bullet 2: ideal membership ──────────────────────────────────────────

#[test]
fn ideal_membership_xy_is_in_x_y_and_one_is_not() {
    let gens = vec![poly(&[(1, &[1, 0])], 2), poly(&[(1, &[0, 1])], 2)]; // ⟨x, y⟩
    let opts = GroebnerOpts::default();

    let xy = poly(&[(1, &[1, 1])], 2);
    assert!(
        ideal_member(&xy, &gens, &opts).expect("terminates"),
        "xy ∈ ⟨x, y⟩"
    );

    let one = poly(&[(1, &[0, 0])], 2);
    assert!(
        !ideal_member(&one, &gens, &opts).expect("terminates"),
        "1 ∉ ⟨x, y⟩"
    );
}

#[test]
fn ideal_membership_agrees_across_all_three_orders() {
    // Membership is a property of the ideal, not of the order used to decide it.
    let gens = vec![circle(), diagonal()];
    let xy = poly(&[(1, &[1, 1])], 2);
    let inside = circle(); // trivially a member

    for order in [MonOrder::Lex, MonOrder::GrLex, MonOrder::GrevLex] {
        let opts = GroebnerOpts::with_order(order);
        assert!(ideal_member(&inside, &gens, &opts).expect("terminates"));
        // xy is NOT in ⟨x²+y²−1, x−y⟩: the variety is two points where xy = 1/2,
        // so xy does not vanish on it.
        assert!(
            !ideal_member(&xy, &gens, &opts).expect("terminates"),
            "xy ∉ I under {}",
            order.name()
        );
    }
}

// ── Spec bullet 3: the product criterion actually prunes ⟨x², y³⟩ ────────────

#[test]
fn product_criterion_prunes_the_single_pair_of_x2_y3() {
    // LM(x²) = x² and LM(y³) = y³ are coprime, so Buchberger's first criterion
    // discards the only critical pair without ever forming an S-polynomial.
    // The counters below are what make this an assertion about *pruning* rather
    // than merely about the (trivially correct) result.
    let gens = vec![poly(&[(1, &[2, 0])], 2), poly(&[(1, &[0, 3])], 2)];

    let (basis, stats) = groebner_basis_with_stats(&gens, &GroebnerOpts::default())
        .expect("⟨x², y³⟩ is already a Gröbner basis");

    assert_eq!(stats.pairs_generated, 1, "exactly one pair exists");
    assert_eq!(
        stats.product_criterion_pruned, 1,
        "the product criterion must have pruned it: {stats:?}"
    );
    assert_eq!(
        stats.s_polynomials_reduced, 0,
        "no S-polynomial may be computed at all: {stats:?}"
    );
    assert_eq!(basis.len(), 2, "the input was already the reduced basis");
}

// ── Spec bullet 4: end-to-end nonlinear solving through the public API ───────

#[test]
fn end_to_end_nonlinear_solve_hyperbola_meets_diagonal() {
    // {x·y = 1, x − y = 0} ⟹ (1, 1) and (−1, −1).
    let eq0 = sub(mul(var(0), var(1)), c(1.0));
    let eq1 = sub(var(0), var(1));

    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("solvable");
    let SystemSolveResult::NonlinearSolutions(sols) = result else {
        panic!("expected NonlinearSolutions, got {result:?}");
    };

    assert_eq!(sols.len(), 2, "{sols:?}");
    let points: Vec<(f64, f64)> = sols
        .iter()
        .map(|s| (s[0].eval(&[]), s[1].eval(&[])))
        .collect();
    assert!(
        points
            .iter()
            .any(|(x, y)| (x - 1.0).abs() < 1e-9 && (y - 1.0).abs() < 1e-9),
        "{points:?}"
    );
    assert!(
        points
            .iter()
            .any(|(x, y)| (x + 1.0).abs() < 1e-9 && (y + 1.0).abs() < 1e-9),
        "{points:?}"
    );
}

#[test]
fn end_to_end_contradictory_nonlinear_system_is_inconsistent() {
    // {x² = 1, x² = 2, y = 0}: the unit ideal — no solutions even over ℂ.
    let eq0 = sub(powi(var(0), 2.0), c(1.0));
    let eq1 = sub(powi(var(0), 2.0), c(2.0));
    let eq2 = var(1);

    let result = solve_linear_system(&[eq0, eq1, eq2], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Inconsistent),
        "got {result:?}"
    );
}

#[test]
fn end_to_end_nonlinear_system_with_no_real_solutions_returns_an_empty_list() {
    // {x² + y² = −1, x = y}: finitely many solutions, all of them complex.
    // This must NOT be reported as `Inconsistent` — solutions do exist over ℂ.
    let eq0 = add(add(powi(var(0), 2.0), powi(var(1), 2.0)), c(1.0));
    let eq1 = sub(var(0), var(1));

    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    match result {
        SystemSolveResult::NonlinearSolutions(sols) => {
            assert!(sols.is_empty(), "no real solutions exist: {sols:?}");
        }
        other => panic!("expected an empty NonlinearSolutions, got {other:?}"),
    }
}

#[test]
fn end_to_end_underdetermined_nonlinear_system_is_underdetermined() {
    // A lone circle is a curve: infinitely many solutions.
    let eq0 = sub(add(powi(var(0), 2.0), powi(var(1), 2.0)), c(1.0));
    let result = solve_linear_system(&[eq0], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Underdetermined),
        "got {result:?}"
    );
}

#[test]
fn end_to_end_three_variable_nonlinear_system() {
    // {x² + y² + z² = 1, x = y, z = 0} ⟹ (±1/√2, ±1/√2, 0).
    let eq0 = sub(
        add(add(powi(var(0), 2.0), powi(var(1), 2.0)), powi(var(2), 2.0)),
        c(1.0),
    );
    let eq1 = sub(var(0), var(1));
    let eq2 = var(2);

    let result = solve_linear_system(&[eq0, eq1, eq2], &[0, 1, 2]).expect("solvable");
    let SystemSolveResult::NonlinearSolutions(sols) = result else {
        panic!("expected NonlinearSolutions, got {result:?}");
    };

    assert_eq!(sols.len(), 2, "{sols:?}");
    let r = 1.0 / 2.0_f64.sqrt();
    for s in &sols {
        let (x, y, z) = (s[0].eval(&[]), s[1].eval(&[]), s[2].eval(&[]));
        assert!((x.abs() - r).abs() < 1e-9, "x = {x}");
        assert!((x - y).abs() < 1e-9, "x ≠ y");
        assert!(z.abs() < 1e-9, "z = {z}");
    }
}

// ── The transcendental guard ─────────────────────────────────────────────────

#[test]
fn transcendental_functions_never_reach_the_groebner_path() {
    // sin(x) is not polynomial. The system is nonlinear-looking but must come back
    // as an honest `Nonlinear`, never as fabricated `NonlinearSolutions`.
    let eq0 = add(LoweredOp::Sin(Arc::new(var(0))), var(1));
    let eq1 = sub(var(0), var(1));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Nonlinear),
        "sin() must not be solved as a polynomial: got {result:?}"
    );

    // Same for exp() combined with a genuinely nonlinear polynomial term, so the
    // nonlinear routing is definitely being attempted.
    let eq0 = sub(LoweredOp::Exp(Arc::new(var(0))), c(1.0));
    let eq1 = sub(powi(var(1), 2.0), var(0));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Nonlinear),
        "exp() must not be solved as a polynomial: got {result:?}"
    );
}

#[test]
fn symbolic_irrational_constants_never_reach_the_groebner_path() {
    // Since N1 removed the denominator cap, `MultiPoly::from_lowered` will happily
    // rationalize π into the exact dyadic fraction its f64 denotes — so the
    // polynomial parse SUCCEEDS here and the guard is the only thing standing
    // between a symbolic irrational and an engine that advertises exactness.
    let pi = LoweredOp::NamedConst(oxieml::NamedConst::Pi);
    assert!(
        MultiPoly::from_lowered(&add(powi(var(0), 2.0), pi.clone()), 2).is_ok(),
        "precondition: from_lowered rationalizes π rather than rejecting it"
    );
    assert!(
        !oxieml::solve::is_groebner_safe(&add(powi(var(0), 2.0), pi.clone())),
        "the guard must reject a symbolic irrational constant"
    );

    // x² + π = 0, x − y = 0 must therefore come back as Nonlinear.
    let eq0 = add(powi(var(0), 2.0), pi);
    let eq1 = sub(var(0), var(1));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Nonlinear),
        "π must not be silently rationalized into the Gröbner path: got {result:?}"
    );
}

#[test]
fn plain_f64_constants_are_still_solved_exactly() {
    // The guard rejects *symbolic* irrationals, not ordinary decimal coefficients.
    // A bare Const carries no symbolic identity: 0.1 IS the rational 1/10, and
    // solving with it is exact, so this must still go down the Gröbner path.
    // {x² − 0.25 = 0, x − y = 0} ⟹ (±0.5, ±0.5).
    let eq0 = sub(powi(var(0), 2.0), c(0.25));
    let eq1 = sub(var(0), var(1));

    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("solvable");
    let SystemSolveResult::NonlinearSolutions(sols) = result else {
        panic!("decimal coefficients must still solve, got {result:?}");
    };
    assert_eq!(sols.len(), 2, "{sols:?}");
    let xs: Vec<f64> = sols.iter().map(|s| s[0].eval(&[])).collect();
    assert!(xs.iter().any(|&x| (x - 0.5).abs() < 1e-12), "{xs:?}");
    assert!(xs.iter().any(|&x| (x + 0.5).abs() < 1e-12), "{xs:?}");
}

#[test]
fn a_variable_outside_vars_is_not_solved() {
    // x² + z = 0 mentions z, which is not among the unknowns: a parametric problem
    // the engine is not being asked to solve.
    let eq0 = add(powi(var(0), 2.0), var(2));
    let eq1 = sub(var(0), var(1));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Nonlinear),
        "got {result:?}"
    );
}

// ── Regression: the linear path is untouched ─────────────────────────────────

#[test]
fn linear_regression_intact_unique_solution() {
    // x + y = 3, x − y = 1 ⟹ x = 2, y = 1. Must still be `Unique`, NOT routed
    // through the new nonlinear machinery.
    let eq0 = sub(add(var(0), var(1)), c(3.0));
    let eq1 = sub(sub(var(0), var(1)), c(1.0));

    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("solvable");
    match result {
        SystemSolveResult::Unique(sols) => {
            assert_eq!(sols.len(), 2);
            assert!((sols[0].eval(&[]) - 2.0).abs() < 1e-10);
            assert!((sols[1].eval(&[]) - 1.0).abs() < 1e-10);
        }
        other => panic!("expected Unique, got {other:?}"),
    }
}

#[test]
fn linear_regression_intact_inconsistent_and_underdetermined() {
    // x + y = 1, x + y = 2 ⟹ inconsistent.
    let eq0 = sub(add(var(0), var(1)), c(1.0));
    let eq1 = sub(add(var(0), var(1)), c(2.0));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Inconsistent),
        "got {result:?}"
    );

    // x + y = 1 alone, two unknowns ⟹ underdetermined.
    let eq0 = sub(add(var(0), var(1)), c(1.0));
    let result = solve_linear_system(&[eq0], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Underdetermined),
        "got {result:?}"
    );

    // Dependent rows with a *consistent* rhs (x + y = 1, 2x + 2y = 2). This is
    // genuinely underdetermined mathematically, but the crate has always reported
    // it as `Inconsistent`: the singular-matrix branch only looks at whether the
    // rhs is nonzero, without a rank/consistency check. That predates N3 and is
    // deliberately left alone here — this assertion pins the existing behavior so
    // the Gröbner routing cannot be blamed for it, and so a future fix to the
    // linear path is a conscious, visible change.
    let eq0 = sub(add(var(0), var(1)), c(1.0));
    let eq1 = sub(add(mul(c(2.0), var(0)), mul(c(2.0), var(1))), c(2.0));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Inconsistent),
        "pre-existing linear-path behavior changed: got {result:?}"
    );
}

#[test]
fn linear_regression_intact_3x3() {
    // x + y + z = 6, y + z = 5, z = 3 ⟹ (1, 2, 3).
    let eq0 = sub(add(add(var(0), var(1)), var(2)), c(6.0));
    let eq1 = sub(add(var(1), var(2)), c(5.0));
    let eq2 = sub(var(2), c(3.0));

    let result = solve_linear_system(&[eq0, eq1, eq2], &[0, 1, 2]).expect("solvable");
    match result {
        SystemSolveResult::Unique(sols) => {
            let v: Vec<f64> = sols.iter().map(|s| s.eval(&[])).collect();
            assert!((v[0] - 1.0).abs() < 1e-10, "{v:?}");
            assert!((v[1] - 2.0).abs() < 1e-10, "{v:?}");
            assert!((v[2] - 3.0).abs() < 1e-10, "{v:?}");
        }
        other => panic!("expected Unique, got {other:?}"),
    }
}

#[test]
fn linear_system_with_transcendental_entry_still_reports_nonlinear() {
    // Pre-existing behavior: a linear-looking system containing ln() is Nonlinear.
    let eq0 = add(LoweredOp::Ln(Arc::new(var(0))), var(1));
    let eq1 = sub(var(0), var(1));
    let result = solve_linear_system(&[eq0, eq1], &[0, 1]).expect("decidable");
    assert!(
        matches!(result, SystemSolveResult::Nonlinear),
        "got {result:?}"
    );
}
