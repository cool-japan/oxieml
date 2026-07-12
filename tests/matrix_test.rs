//! Integration tests for symbolic linear algebra (`oxieml::matrix`).
//!
//! Two groups:
//!
//! 1. **Correctness** — the determinant, Bareiss, rref, nullspace, charpoly,
//!    Cayley–Hamilton, eigenvalues, singularity and inverse behaviours required of
//!    the module.
//! 2. **Honesty** — that an *undecidable* symbolic zero test is never quietly
//!    resolved. These are the tests that would fail if the implementation started
//!    guessing.

use oxieml::LoweredOp;
use oxieml::matrix::{
    AssumedRelation, Certainty, Matrix, MatrixError, NonZeroProof, ZeroOracle, ZeroVerdict,
};
use oxieml::poly::MultiPoly;
use std::sync::Arc;

// ── helpers ──────────────────────────────────────────────────────────────────

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

fn constant(c: f64) -> LoweredOp {
    LoweredOp::Const(c)
}

fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}

fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Arc::new(a), Arc::new(b))
}

fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}

fn pow(a: LoweredOp, e: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(a), Arc::new(LoweredOp::Const(e)))
}

fn sin(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(Arc::new(a))
}

fn cos(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Arc::new(a))
}

fn exp(a: LoweredOp) -> LoweredOp {
    LoweredOp::Exp(Arc::new(a))
}

fn neg(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Arc::new(a))
}

/// Compare two expressions as *polynomials* in `n_vars` variables — an exact
/// symbolic identity check, not a numeric one.
fn assert_same_polynomial(actual: &LoweredOp, expected: &LoweredOp, n_vars: usize) {
    let pa = MultiPoly::from_lowered(actual, n_vars)
        .unwrap_or_else(|e| panic!("actual is not polynomial: {e} — {}", actual.to_pretty()));
    let pb = MultiPoly::from_lowered(expected, n_vars)
        .unwrap_or_else(|e| panic!("expected is not polynomial: {e}"));
    assert_eq!(
        pa,
        pb,
        "\n  actual:   {}\n  expected: {}",
        actual.to_pretty(),
        expected.to_pretty()
    );
}

// ── 1. numeric determinant ───────────────────────────────────────────────────

#[test]
fn numeric_determinant_of_1_2_3_4_is_minus_two() {
    let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("dims");
    assert_eq!(m.det().expect("det"), LoweredOp::Const(-2.0));
}

#[test]
fn numeric_determinant_is_exact_over_the_rationals() {
    // Entries with denominators: the answer must be an exact rational, not a float.
    // [[1/3, 1/5], [1/7, 1/11]] → 1/33 − 1/35 = 2/1155
    let m = Matrix::from_f64(2, 2, &[1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0, 1.0 / 11.0]).expect("dims");
    let exact = m.det_exact().expect("exact det");
    assert_eq!(exact.numer().to_string(), "2");
    assert_eq!(exact.denom().to_string(), "1155");
}

// ── 2. symbolic determinant: [[a, b], [c, d]] → a·d − b·c ────────────────────

#[test]
fn symbolic_determinant_is_ad_minus_bc() {
    let m = Matrix::of_vars(2, 2).expect("dims"); // [[x0, x1], [x2, x3]]
    let det = m.det().expect("det");
    let expected = sub(mul(var(0), var(3)), mul(var(1), var(2)));
    assert_same_polynomial(&det, &expected, 4);
}

// ── 3. Bareiss on a 3×3 Vandermonde ─────────────────────────────────────────

#[test]
fn bareiss_vandermonde_3x3() {
    // | 1  x0  x0² |
    // | 1  x1  x1² |  =  (x1 − x0)(x2 − x0)(x2 − x1)
    // | 1  x2  x2² |
    let rows = vec![
        vec![constant(1.0), var(0), pow(var(0), 2.0)],
        vec![constant(1.0), var(1), pow(var(1), 2.0)],
        vec![constant(1.0), var(2), pow(var(2), 2.0)],
    ];
    let m = Matrix::from_rows(&rows).expect("dims");
    let det = m.det().expect("det");

    let expected = mul(
        mul(
            sub(var(1), var(0)), // (x1 − x0)
            sub(var(2), var(0)), // (x2 − x0)
        ),
        sub(var(2), var(1)), // (x2 − x1)
    );
    assert_same_polynomial(&det, &expected, 3);

    // And the fraction-free path really is exact: the determinant of a Vandermonde
    // with repeated nodes vanishes identically.
    let degenerate = Matrix::from_rows(&[
        vec![constant(1.0), var(0), pow(var(0), 2.0)],
        vec![constant(1.0), var(0), pow(var(0), 2.0)],
        vec![constant(1.0), var(2), pow(var(2), 2.0)],
    ])
    .expect("dims");
    let zero_det = degenerate.det().expect("det");
    assert_eq!(zero_det, LoweredOp::Const(0.0));
}

#[test]
fn bareiss_vandermonde_4x4_matches_the_product_formula() {
    let mut rows = Vec::new();
    for i in 0..4 {
        rows.push(vec![
            constant(1.0),
            var(i),
            pow(var(i), 2.0),
            pow(var(i), 3.0),
        ]);
    }
    let m = Matrix::from_rows(&rows).expect("dims");
    let det = m.det().expect("det");

    // ∏_{i < j} (x_j − x_i)
    let mut expected = constant(1.0);
    for i in 0..4 {
        for j in i + 1..4 {
            expected = mul(expected, sub(var(j), var(i)));
        }
    }
    assert_same_polynomial(&det, &expected, 4);
}

// ── 4. rref and nullspace ────────────────────────────────────────────────────

#[test]
fn rref_of_a_rank_deficient_numeric_matrix() {
    // [[1,2,3],[4,5,6],[7,8,9]] has rank 2; rref = [[1,0,−1],[0,1,2],[0,0,0]]
    let m = Matrix::from_f64(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).expect("dims");
    let rref = m.rref().expect("rref");

    assert_eq!(rref.rank(), 2);
    assert_eq!(rref.pivot_cols, vec![0, 1]);

    let expected = [[1.0, 0.0, -1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, want) in row.iter().enumerate() {
            let got = rref.matrix.get(i, j).expect("in bounds").eval(&[]);
            assert!(
                (got - want).abs() < 1e-12,
                "rref[{i}][{j}] = {got}, expected {want}"
            );
        }
    }
}

#[test]
fn nullspace_of_a_rank_deficient_numeric_matrix() {
    let m = Matrix::from_f64(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).expect("dims");
    let basis = m.nullspace().expect("nullspace");
    assert_eq!(basis.len(), 1, "nullity must be 3 − rank 2 = 1");

    let v: Vec<f64> = basis[0].iter().map(|e| e.eval(&[])).collect();
    // The basis vector is (1, −2, 1) up to scale.
    assert!(
        v.iter().any(|x| x.abs() > 1e-9),
        "must not be the zero vector"
    );
    let ratio = v[2];
    assert!((v[0] / ratio - 1.0).abs() < 1e-9, "{v:?}");
    assert!((v[1] / ratio + 2.0).abs() < 1e-9, "{v:?}");

    // And it really is in the nullspace: A·v = 0.
    let a = m.eval_at(&[]);
    for i in 0..3 {
        let dot: f64 = (0..3).map(|j| a[i * 3 + j] * v[j]).sum();
        assert!(dot.abs() < 1e-9, "row {i} of A·v is {dot}, not 0");
    }
}

#[test]
fn rank_of_a_full_rank_matrix() {
    let m = Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
    assert_eq!(m.rank().expect("rank"), 3);
    assert!(m.nullspace().expect("nullspace").is_empty());
}

#[test]
fn symbolic_nullspace_is_verified_numerically() {
    // [[x0, x1], [2·x0, 2·x1]] — rank 1 for generic (x0, x1); nullspace spanned by
    // (−x1, x0).
    let rows = vec![
        vec![var(0), var(1)],
        vec![mul(constant(2.0), var(0)), mul(constant(2.0), var(1))],
    ];
    let m = Matrix::from_rows(&rows).expect("dims");
    let basis = m.nullspace().expect("nullspace");
    assert_eq!(basis.len(), 1);

    // A·v must vanish at every sample point.
    for point in [[1.0, 2.0], [3.0, -0.5], [0.25, 7.0]] {
        let a = m.eval_at(&point);
        let v: Vec<f64> = basis[0].iter().map(|e| e.eval(&point)).collect();
        assert!(
            v.iter().any(|x| x.abs() > 1e-9),
            "degenerate basis at {point:?}"
        );
        for i in 0..2 {
            let dot: f64 = (0..2).map(|j| a[i * 2 + j] * v[j]).sum();
            assert!(dot.abs() < 1e-9, "A·v row {i} = {dot} at {point:?}");
        }
    }
}

#[test]
fn rref_of_a_rectangular_matrix() {
    // 2×4, rank 2
    let m = Matrix::from_f64(2, 4, &[1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0]).expect("dims");
    let rref = m.rref().expect("rref");
    assert_eq!(rref.pivot_cols, vec![0, 2]);
    assert_eq!(rref.rank(), 2);
    // Nullity = 4 − 2 = 2
    assert_eq!(m.nullspace().expect("nullspace").len(), 2);
}

// ── 5. charpoly + Cayley–Hamilton probe ──────────────────────────────────────

#[test]
fn charpoly_of_a_numeric_matrix() {
    // [[1,2],[3,4]] → λ² − 5λ − 2
    let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("dims");
    let p = m.charpoly().expect("charpoly");
    assert_eq!(p.degree(), 2);
    let c: Vec<f64> = p.coeffs.iter().map(|e| e.eval(&[])).collect();
    assert!((c[0] + 2.0).abs() < 1e-12, "c0 = {}", c[0]);
    assert!((c[1] + 5.0).abs() < 1e-12, "c1 = {}", c[1]);
    assert!((c[2] - 1.0).abs() < 1e-12, "c2 = {}", c[2]);
}

#[test]
fn charpoly_of_a_symbolic_2x2() {
    // [[a,b],[c,d]] → λ² − (a+d)·λ + (a·d − b·c)
    let m = Matrix::of_vars(2, 2).expect("dims");
    let p = m.charpoly().expect("charpoly");

    let trace = add(var(0), var(3));
    assert_same_polynomial(&p.coeffs[1], &neg(trace), 4);

    let det = sub(mul(var(0), var(3)), mul(var(1), var(2)));
    assert_same_polynomial(&p.coeffs[0], &det, 4);
    assert_eq!(p.coeffs[2], LoweredOp::Const(1.0));
}

#[test]
fn cayley_hamilton_probe_numeric() {
    // p(A) must be *exactly* the zero matrix — not merely small.
    let m = Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
    let residual = m.cayley_hamilton_residual().expect("residual");
    for entry in &residual.data {
        assert_eq!(
            *entry,
            LoweredOp::Const(0.0),
            "p(A) must vanish identically"
        );
    }
}

#[test]
fn cayley_hamilton_probe_symbolic() {
    // The same, for a fully generic 3×3 symbolic matrix: every one of the nine
    // entries of p(A) must cancel to the exact zero polynomial. This is a genuine
    // end-to-end check of Faddeev–LeVerrier, the ring arithmetic and the atomization.
    let m = Matrix::of_vars(3, 3).expect("dims");
    let residual = m.cayley_hamilton_residual().expect("residual");
    for (idx, entry) in residual.data.iter().enumerate() {
        assert_eq!(
            *entry,
            LoweredOp::Const(0.0),
            "p(A) entry {idx} did not cancel: {}",
            entry.to_pretty()
        );
    }
}

#[test]
fn cayley_hamilton_probe_transcendental() {
    // And with transcendental entries — the ring identity holds over ℚ[atoms] no
    // matter what the atoms are.
    let rows = vec![
        vec![exp(var(0)), sin(var(1))],
        vec![cos(var(0)), LoweredOp::Ln(Arc::new(var(1)))],
    ];
    let m = Matrix::from_rows(&rows).expect("dims");
    let residual = m.cayley_hamilton_residual().expect("residual");
    for entry in &residual.data {
        assert_eq!(*entry, LoweredOp::Const(0.0), "{}", entry.to_pretty());
    }
}

// ── 6. eigenvalues {±i} ──────────────────────────────────────────────────────

#[test]
fn rotation_matrix_eigenvalues_are_plus_and_minus_i() {
    // [[0, −1], [1, 0]] — the 90° rotation. charpoly λ² + 1 → λ = ±i.
    let m = Matrix::from_f64(2, 2, &[0.0, -1.0, 1.0, 0.0]).expect("dims");
    let values = m.eigenvalues().expect("eigenvalues");
    assert_eq!(values.len(), 2);

    let mut has_plus_i = false;
    let mut has_minus_i = false;
    for z in &values {
        assert!(
            z.re.abs() < 1e-9,
            "eigenvalue {z} should be purely imaginary"
        );
        if (z.im - 1.0).abs() < 1e-9 {
            has_plus_i = true;
        }
        if (z.im + 1.0).abs() < 1e-9 {
            has_minus_i = true;
        }
    }
    assert!(
        has_plus_i && has_minus_i,
        "expected {{+i, −i}}, got {values:?}"
    );
}

#[test]
fn rotation_matrix_eigenvectors_satisfy_a_v_equals_lambda_v() {
    let m = Matrix::from_f64(2, 2, &[0.0, -1.0, 1.0, 0.0]).expect("dims");
    let pairs = m.eigenvectors().expect("eigenvectors");
    assert_eq!(pairs.len(), 2);

    let a = m.eval_at(&[]);
    for pair in &pairs {
        assert!(pair.residual < 1e-8, "residual {}", pair.residual);
        for i in 0..2 {
            let mut acc = num_complex::Complex::new(0.0, 0.0);
            for j in 0..2 {
                acc += num_complex::Complex::new(a[i * 2 + j], 0.0) * pair.vector[j];
            }
            let want = pair.value * pair.vector[i];
            assert!(
                (acc - want).norm() < 1e-8,
                "A·v ≠ λ·v at row {i}: {acc} vs {want}"
            );
        }
    }
}

#[test]
fn symbolic_eigenvalues_are_refused_not_faked() {
    // The eigenvalues of a symbolic matrix are algebraic *functions*, not numbers.
    // Returning plausible numbers here would be a lie.
    let m = Matrix::of_vars(2, 2).expect("dims");
    assert_eq!(m.eigenvalues(), Err(MatrixError::NotNumeric));
    assert_eq!(m.eigenvectors(), Err(MatrixError::NotNumeric));
    // …but the charpoly, which *is* well defined symbolically, still works.
    assert!(m.charpoly().is_ok());
}

// ── 7. singular → Err(SingularMatrix) ────────────────────────────────────────

#[test]
fn singular_matrix_inverse_is_an_error() {
    let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 2.0, 4.0]).expect("dims");
    assert_eq!(m.det().expect("det"), LoweredOp::Const(0.0));
    assert_eq!(m.inverse(), Err(MatrixError::SingularMatrix));
    assert!(m.is_singular().expect("decidable"));
}

#[test]
fn singular_symbolic_matrix_inverse_is_an_error() {
    // [[a, b], [2a, 2b]] — determinant 2ab − 2ab = 0 identically. The Bareiss path
    // proves it (the determinant is the *zero polynomial*), so this is decided, not
    // guessed.
    let rows = vec![
        vec![var(0), var(1)],
        vec![mul(constant(2.0), var(0)), mul(constant(2.0), var(1))],
    ];
    let m = Matrix::from_rows(&rows).expect("dims");
    assert_eq!(m.det().expect("det"), LoweredOp::Const(0.0));
    assert_eq!(m.inverse(), Err(MatrixError::SingularMatrix));
    assert!(m.is_singular().expect("decidable"));
}

// ── 8. A · inv(A) ≈ I ────────────────────────────────────────────────────────

#[test]
fn numeric_a_times_inv_a_is_identity() {
    let m = Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
    let inv = m.inverse().expect("inverse");
    let product = m.mul(&inv).expect("mul");

    for i in 0..3 {
        for j in 0..3 {
            let value = product.get(i, j).expect("in bounds").eval(&[]);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (value - want).abs() < 1e-9,
                "(A·A⁻¹)[{i}][{j}] = {value}, expected {want}"
            );
        }
    }
}

#[test]
fn symbolic_a_times_inv_a_is_identity() {
    // [[a, b], [c, d]] — inverse is the adjugate over the determinant. The product
    // must be the identity at every point where the determinant does not vanish.
    let m = Matrix::of_vars(2, 2).expect("dims");
    let inv = m.inverse().expect("inverse");
    let product = m.mul(&inv).expect("mul");

    for point in [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, 0.0, 5.0],
        [-1.5, 0.25, 3.0, 7.0],
    ] {
        for i in 0..2 {
            for j in 0..2 {
                let value = product.get(i, j).expect("in bounds").eval(&point);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (value - want).abs() < 1e-9,
                    "(A·A⁻¹)[{i}][{j}] = {value} at {point:?}, expected {want}"
                );
            }
        }
    }
}

#[test]
fn transcendental_a_times_inv_a_is_identity() {
    // A provably invertible transcendental matrix: det = exp(x)·ln(y) − 1, which the
    // oracle witnesses as nonzero by probing.
    let rows = vec![
        vec![exp(var(0)), constant(1.0)],
        vec![constant(1.0), LoweredOp::Ln(Arc::new(var(1)))],
    ];
    let m = Matrix::from_rows(&rows).expect("dims");
    assert!(!m.is_singular().expect("oracle should witness nonzero det"));

    let inv = m.inverse().expect("inverse");
    let product = m.mul(&inv).expect("mul");
    for point in [[0.5, 3.0], [1.25, 7.0], [-0.75, 2.0]] {
        for i in 0..2 {
            for j in 0..2 {
                let value = product.get(i, j).expect("in bounds").eval(&point);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (value - want).abs() < 1e-9,
                    "(A·A⁻¹)[{i}][{j}] = {value} at {point:?}"
                );
            }
        }
    }
}

// ── verify(): the safety net ─────────────────────────────────────────────────

#[test]
fn verify_passes_on_numeric_symbolic_and_transcendental_matrices() {
    let numeric =
        Matrix::from_f64(3, 3, &[2.0, -1.0, 0.0, 4.0, 3.0, 7.0, -5.0, 1.0, 6.0]).expect("dims");
    let report = numeric.verify().expect("verify");
    assert!(report.is_ok(), "{report}");
    assert!(report.probes_used > 0);

    let symbolic = Matrix::of_vars(3, 3).expect("dims");
    assert!(symbolic.verify().expect("verify").is_ok());

    let transcendental =
        Matrix::from_rows(&[vec![exp(var(0)), sin(var(1))], vec![cos(var(0)), var(1)]])
            .expect("dims");
    let report = transcendental.verify().expect("verify");
    assert!(report.is_ok(), "{report}");
}

// ═════════════════════════════════════════════════════════════════════════════
//  HONESTY: the undecidable cases must never be silently decided.
// ═════════════════════════════════════════════════════════════════════════════

/// `sin(x)² + cos(x)² − 1` — identically zero, but not provably so: as a polynomial
/// in the atoms `s = sin x` and `c = cos x` it is `s² + c² − 1 ≠ 0`, and no amount of
/// probing can prove that a function which vanishes everywhere we look vanishes
/// *everywhere*.
fn pythagorean_zero() -> LoweredOp {
    sub(
        add(mul(sin(var(0)), sin(var(0))), mul(cos(var(0)), cos(var(0)))),
        constant(1.0),
    )
}

#[test]
fn oracle_reports_the_three_verdicts_correctly() {
    let oracle = ZeroOracle::new();

    // Tier 1 — proved zero (polynomial identity).
    assert!(
        oracle
            .test(&sub(mul(var(0), var(0)), pow(var(0), 2.0)))
            .is_proved_zero()
    );

    // Tier 1b — proved nonzero, exactly, no probing needed.
    assert_eq!(
        oracle.test(&add(pow(var(0), 2.0), constant(1.0))),
        ZeroVerdict::NonZero(NonZeroProof::PolynomialNonZero)
    );

    // Tier 2 — proved nonzero by a probe witness.
    let verdict = oracle.test(&sub(exp(var(0)), constant(1.0)));
    assert!(verdict.is_proved_nonzero(), "{verdict}");
    assert!(matches!(
        verdict,
        ZeroVerdict::NonZero(NonZeroProof::ProbeWitness { .. })
    ));

    // Tier 3 — UNDECIDED. Not zero, not nonzero.
    let verdict = oracle.test(&pythagorean_zero());
    assert!(verdict.is_undecided(), "{verdict}");
    assert!(!verdict.is_proved_zero(), "must not claim it is zero");
    assert!(!verdict.is_proved_nonzero(), "must not claim it is nonzero");
}

#[test]
fn undecidable_singularity_is_refused_not_guessed() {
    // det = sin²+cos²−1, which really *is* 0 — the matrix really *is* singular — but
    // we cannot prove it. The honest answer is "I don't know", not "not singular"
    // (which the nonzero atom-polynomial would suggest) and not "singular" (which we
    // cannot back up).
    let m = Matrix::from_rows(&[
        vec![pythagorean_zero(), constant(0.0)],
        vec![constant(0.0), constant(1.0)],
    ])
    .expect("dims");

    match m.is_singular() {
        Err(MatrixError::Undecidable(assumption)) => {
            assert_eq!(assumption.assumed, AssumedRelation::IdenticallyZero);
            assert!(
                assumption.evidence.usable_probes > 0,
                "the refusal must carry real evidence"
            );
            assert!(assumption.evidence.max_abs_value < 1e-9);
        }
        other => panic!("expected an honest refusal, got {other:?}"),
    }

    // The determinant *expression* is still returned — that part is unconditionally
    // exact, and the oracle's verdict is handed over alongside it for inspection.
    let (det, verdict) = m.determinant_verdict(&ZeroOracle::new()).expect("det");
    assert!(verdict.is_undecided(), "{verdict}");
    for x in [0.3f64, 1.7, -2.5] {
        assert!(det.eval(&[x]).abs() < 1e-12, "the determinant does vanish");
    }
}

#[test]
fn undecidable_inverse_is_refused_but_available_under_an_explicit_assumption() {
    let m = Matrix::from_rows(&[
        vec![pythagorean_zero(), constant(0.0)],
        vec![constant(0.0), constant(1.0)],
    ])
    .expect("dims");

    // The plain API refuses: it cannot show the matrix is invertible.
    match m.inverse() {
        Err(MatrixError::Undecidable(assumption)) => {
            assert_eq!(assumption.assumed, AssumedRelation::NotIdenticallyZero);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }

    // The caller may take the assumption on explicitly — and is then *told* about it.
    let certified = m
        .inverse_assuming_nonsingular(&ZeroOracle::new())
        .expect("conditional inverse");
    assert!(
        !certified.is_proved(),
        "a conditional inverse must never be reported as proved"
    );
    match &certified.certainty {
        Certainty::Conditional(assumptions) => {
            assert_eq!(assumptions.len(), 1);
            assert_eq!(assumptions[0].assumed, AssumedRelation::NotIdenticallyZero);
            assert!(
                assumptions[0].context.contains("determinant"),
                "the assumption must say what it is about: {}",
                assumptions[0].context
            );
        }
        Certainty::Proved => panic!("must not be proved"),
    }
}

#[test]
fn undecidable_pivot_makes_rref_refuse_and_rref_certified_disclose() {
    // Column 0 has one undecidable entry and one proved zero. There is no *provably*
    // nonzero pivot, so declaring the column pivot-free means assuming the
    // undecidable entry vanishes — an assumption that must be surfaced.
    let m = Matrix::from_rows(&[
        vec![pythagorean_zero(), constant(1.0)],
        vec![constant(0.0), constant(0.0)],
    ])
    .expect("dims");

    match m.rref() {
        Err(MatrixError::Undecidable(assumption)) => {
            assert_eq!(assumption.assumed, AssumedRelation::IdenticallyZero);
            assert!(assumption.context.contains("column 0"));
        }
        other => panic!("expected an honest refusal, got {other:?}"),
    }

    let certified = m
        .rref_certified(&ZeroOracle::new())
        .expect("conditional rref");
    assert!(!certified.is_proved());
    assert_eq!(certified.assumptions().len(), 1);
    assert_eq!(
        certified.assumptions()[0].assumed,
        AssumedRelation::IdenticallyZero
    );

    // The rank is likewise not knowable, and says so.
    assert!(matches!(m.rank(), Err(MatrixError::Undecidable(_))));
    let rank = m
        .rank_certified(&ZeroOracle::new())
        .expect("conditional rank");
    assert!(!rank.is_proved(), "rank rests on the pivot assumption");
    assert_eq!(rank.value, 1, "under the assumption, the rank is 1");
}

#[test]
fn the_determinant_never_becomes_uncertain() {
    // The whole point of routing the determinant through Bareiss over ℚ[atoms]: it is
    // a polynomial in the entries, so it commutes with evaluation and needs *no*
    // functional zero test. Even a matrix full of undecidable entries has an
    // unconditionally correct determinant expression.
    let m = Matrix::from_rows(&[
        vec![pythagorean_zero(), exp(var(0))],
        vec![sin(var(0)), pythagorean_zero()],
    ])
    .expect("dims");

    let det = m.det().expect("the determinant is always computable");

    // Cross-check it against a numeric determinant at several points.
    for x in [0.3f64, 1.25, 2.5, -0.7] {
        let a = m.eval_at(&[x]);
        let numeric = a[0] * a[3] - a[1] * a[2];
        let symbolic = det.eval(&[x]);
        assert!(
            (symbolic - numeric).abs() < 1e-9 * (1.0 + numeric.abs()),
            "det mismatch at x = {x}: {symbolic} vs {numeric}"
        );
    }
}

#[test]
fn a_matrix_that_is_secretly_singular_is_not_falsely_inverted() {
    // det = exp(x)·exp(−x) − 1 ≡ 0. The matrix IS singular, but the atom polynomial
    // `a·b − 1` is not zero, so it cannot be proved. Under no circumstances may we
    // hand back an "inverse".
    let m = Matrix::from_rows(&[
        vec![exp(var(0)), constant(1.0)],
        vec![constant(1.0), exp(neg(var(0)))],
    ])
    .expect("dims");

    assert!(
        matches!(m.inverse(), Err(MatrixError::Undecidable(_))),
        "must refuse: the matrix is singular, we just cannot prove it"
    );
    assert!(matches!(m.is_singular(), Err(MatrixError::Undecidable(_))));

    // The determinant expression, however, is exact — and it does vanish.
    let det = m.det().expect("det");
    for x in [0.5f64, 1.5, -2.0] {
        assert!(det.eval(&[x]).abs() < 1e-12, "det should vanish at {x}");
    }
}

#[test]
fn oracle_verdicts_are_deterministic_and_reproducible() {
    let expr = sin(var(0));
    let a = ZeroOracle::with_seed(12345).test(&expr);
    let b = ZeroOracle::with_seed(12345).test(&expr);
    assert_eq!(a, b, "a seeded oracle must be reproducible");

    // More probes can only ever turn "undecided" into "proved nonzero", never into
    // "proved zero".
    let stubborn = ZeroOracle::new().with_probes(200).test(&pythagorean_zero());
    assert!(
        stubborn.is_undecided(),
        "200 probes still prove nothing about a true identity: {stubborn}"
    );
}

// ── shape and cap handling ───────────────────────────────────────────────────

#[test]
fn non_square_matrices_are_rejected_where_squareness_is_required() {
    let m = Matrix::of_vars(2, 3).expect("dims");
    assert_eq!(m.det(), Err(MatrixError::NotSquare { rows: 2, cols: 3 }));
    assert_eq!(
        m.charpoly(),
        Err(MatrixError::NotSquare { rows: 2, cols: 3 })
    );
    // …but rref, nullspace and rank are perfectly well defined.
    assert!(m.rref().is_ok());
    assert!(m.nullspace().is_ok());
}

#[test]
fn symbolic_dimension_cap_is_enforced() {
    let n = oxieml::matrix::MAX_SYMBOLIC_DIM + 1;
    let m = Matrix::of_vars(n, n).expect("dims");
    assert_eq!(
        m.det(),
        Err(MatrixError::TooLarge {
            n,
            cap: oxieml::matrix::MAX_SYMBOLIC_DIM
        })
    );
}

#[test]
fn eight_by_eight_numeric_determinant_reaches_the_cap() {
    // The cap is 8, so an 8×8 must actually be computable — a cap you cannot reach is
    // not a cap. Numeric entries keep this exact and fast (Bareiss over ℚ with the
    // atom count zero). Determinant of a lower-triangular matrix is the product of
    // the diagonal: here 1·2·3·4·5·6·7·8 = 40320.
    let mut m = Matrix::identity(8);
    for i in 0..8 {
        m.set(i, i, constant((i + 1) as f64)).expect("in bounds");
        for j in 0..i {
            m.set(i, j, constant(2.0)).expect("in bounds"); // below the diagonal
        }
    }
    let det = m.det().expect("8×8 numeric determinant");
    assert_eq!(det, LoweredOp::Const(40_320.0));
}

#[test]
fn five_by_five_generic_determinant_is_computable_without_overflow() {
    // A fully generic 5×5 has a 120-term determinant. The point here is robustness:
    // rendering and simplifying a many-term expression must not overflow the stack
    // (the sum is assembled as a balanced tree for exactly this reason).
    let m = Matrix::of_vars(5, 5).expect("dims");
    let det = m.det().expect("5×5 symbolic determinant");
    let p = MultiPoly::from_lowered(&det, 25).expect("polynomial");
    assert_eq!(p.terms.len(), 120, "a 5×5 determinant has 5! = 120 terms");

    // Cross-check against a numeric determinant at a non-degenerate sample point.
    // A non-linear pattern keeps the rows from being affine (which would make the
    // matrix singular and the comparison vacuous).
    let point: Vec<f64> = (0..25)
        .map(|k| 1.0 + ((k * k + 7 * k + 1) % 13) as f64 * 0.31)
        .collect();
    let a = m.eval_at(&point);
    let mut numeric = a.clone();
    // naive f64 Bareiss-free LU determinant for the cross-check
    let n = 5;
    let mut det_lu = 1.0f64;
    for k in 0..n {
        let mut piv = k;
        for r in k + 1..n {
            if numeric[r * n + k].abs() > numeric[piv * n + k].abs() {
                piv = r;
            }
        }
        if piv != k {
            for c in 0..n {
                numeric.swap(k * n + c, piv * n + c);
            }
            det_lu = -det_lu;
        }
        det_lu *= numeric[k * n + k];
        for r in k + 1..n {
            let f = numeric[r * n + k] / numeric[k * n + k];
            for c in k..n {
                numeric[r * n + c] -= f * numeric[k * n + c];
            }
        }
    }
    let symbolic = det.eval(&point);
    assert!(
        (symbolic - det_lu).abs() < 1e-6 * (1.0 + det_lu.abs()),
        "5×5 det mismatch: {symbolic} vs {det_lu}"
    );
}
