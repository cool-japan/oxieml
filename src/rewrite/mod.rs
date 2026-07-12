//! Algebraic term rewriting for the lowered IR ([`crate::lower::LoweredOp`]).
//!
//! This module implements seven classical computer-algebra rewrites as
//! *never-panic* inherent methods on [`LoweredOp`]:
//!
//! | method            | what it does                                                        |
//! |-------------------|---------------------------------------------------------------------|
//! | [`LoweredOp::expand`]     | distribute products/powers into a canonical sum of monomials |
//! | [`LoweredOp::factor`]     | factor a univariate polynomial into irreducibles over ℚ      |
//! | [`LoweredOp::collect`]    | group a polynomial by ascending powers of one variable        |
//! | [`LoweredOp::apart`]      | partial-fraction-decompose a univariate rational function     |
//! | [`LoweredOp::together`]   | combine a sum of fractions into a single reduced fraction      |
//! | [`LoweredOp::powsimp`]    | merge powers of a common base (`xᵃ·xᵇ → xᵃ⁺ᵇ`)                |
//! | [`LoweredOp::logcombine`] | combine a sum of logarithms (`ln a + ln b → ln(ab)`)          |
//!
//! # Never-panic contract
//!
//! Every method returns a [`LoweredOp`]. When the input is outside the method's
//! domain (a transcendental where a polynomial was needed, more than one
//! variable where a univariate routine is required, an exhausted internal
//! budget, …) the method returns the **input unchanged** rather than panicking.
//! There is no indexing that can go out of bounds and no `unwrap` on user data.
//!
//! # Numeric-verification gate
//!
//! An algebraic rewrite must never change the *value* of the expression. Because
//! every rewrite here is built on exact rational arithmetic (`Coeff =
//! BigRational`) the transformed tree is mathematically identical to the input;
//! nonetheless each of [`expand`](LoweredOp::expand),
//! [`factor`](LoweredOp::factor), [`collect`](LoweredOp::collect),
//! [`apart`](LoweredOp::apart), [`together`](LoweredOp::together),
//! [`powsimp`](LoweredOp::powsimp) and [`logcombine`](LoweredOp::logcombine)
//! re-checks the rewritten tree against the original at a fixed set of
//! deterministic rational probe points via [`verify_agrees`]. If they disagree
//! (which can only happen through a bug, or an out-of-domain float such as a
//! logarithm of a negative sample), the method conservatively returns the input
//! unchanged. This is the same discipline the rational-integration engine uses.
//!
//! The transcendental-guard follows from the same gate: the N1 coefficient work
//! made `Poly::from_lowered` succeed on constants such as `π` or `0.1` (they are
//! finite floats and therefore exact dyadic rationals). Treating such a constant
//! as a rational coefficient is *correct* — it round-trips to the identical
//! `f64` — and the probe-point check confirms the value is preserved, so no
//! nonsense can leak out for transcendental-looking inputs.
//!
//! # Shared infrastructure
//!
//! [`gaussian_eliminate`] is the single linear solver used by the shared
//! [`apart::partial_fractions`] helper (which the rational-integration engine in
//! [`crate::integrate`] also calls). There is exactly one copy of it in the
//! whole crate, living here.

pub mod apart;
pub mod collect;
pub mod expand;
pub mod factor;
pub mod logexp;

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::lower::LoweredOp;

// ── LoweredOp builder helpers (shared across the rewrite submodules) ───────────

/// A constant leaf.
pub(crate) fn lconst(c: f64) -> LoweredOp {
    LoweredOp::Const(c)
}

/// A variable leaf.
pub(crate) fn lvar(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

/// `a + b`.
pub(crate) fn ladd(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}

/// `a · b`.
pub(crate) fn lmul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}

/// `a / b`.
pub(crate) fn ldiv(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Arc::new(a), Arc::new(b))
}

/// `base ^ exp` with a floating exponent.
pub(crate) fn lpow(base: LoweredOp, exp: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(base), Arc::new(LoweredOp::Const(exp)))
}

/// `ln(arg)`.
pub(crate) fn lln(arg: LoweredOp) -> LoweredOp {
    LoweredOp::Ln(Arc::new(arg))
}

// ── Variable analysis ─────────────────────────────────────────────────────────

/// Insert every distinct `Var(i)` index occurring in `op` into `out`.
pub(crate) fn collect_vars(op: &LoweredOp, out: &mut BTreeSet<usize>) {
    match op {
        LoweredOp::Var(i) => {
            out.insert(*i);
        }
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => {}
        LoweredOp::Neg(a)
        | LoweredOp::Exp(a)
        | LoweredOp::Ln(a)
        | LoweredOp::Sin(a)
        | LoweredOp::Cos(a)
        | LoweredOp::Tan(a)
        | LoweredOp::Sinh(a)
        | LoweredOp::Cosh(a)
        | LoweredOp::Tanh(a)
        | LoweredOp::Arcsin(a)
        | LoweredOp::Arccos(a)
        | LoweredOp::Arctan(a)
        | LoweredOp::Arcsinh(a)
        | LoweredOp::Arccosh(a)
        | LoweredOp::Arctanh(a)
        | LoweredOp::Erf(a)
        | LoweredOp::LGamma(a)
        | LoweredOp::Digamma(a)
        | LoweredOp::Trigamma(a)
        | LoweredOp::Ei(a)
        | LoweredOp::Si(a)
        | LoweredOp::Ci(a) => collect_vars(a, out),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
    }
}

/// Return the unique variable index used by `op`, or `None` when it uses zero or
/// more than one distinct variable.
///
/// Used by the univariate routines ([`factor`](LoweredOp::factor),
/// [`apart`](LoweredOp::apart), [`together`](LoweredOp::together)) to pick the
/// working variable and to bail out on the multivariate case.
pub(crate) fn sole_var(op: &LoweredOp) -> Option<usize> {
    let mut set = BTreeSet::new();
    collect_vars(op, &mut set);
    if set.len() == 1 {
        set.into_iter().next()
    } else {
        None
    }
}

// ── Numeric verification ──────────────────────────────────────────────────────

/// Deterministic strictly-positive probe points for `n_vars` variables.
///
/// The values are strictly positive so that logarithms and other
/// partial-domain functions stay finite, and the 16 patterns are pairwise
/// distinct, so any nonzero low-degree polynomial/rational difference between
/// two candidate trees is detected. No randomness is used, per the crate's
/// no-`rand` policy.
pub(crate) fn probe_points(n_vars: usize) -> Vec<Vec<f64>> {
    const SCALARS: [f64; 16] = [
        1.5, 2.0, 0.5, 3.0, 2.5, 1.25, 4.0, 0.75, 3.5, 1.75, 5.0, 2.25, 0.625, 2.75, 1.125, 3.25,
    ];
    let n = n_vars.max(1);
    let mut pts = Vec::with_capacity(SCALARS.len());
    for i in 0..SCALARS.len() {
        let mut v = Vec::with_capacity(n);
        for j in 0..n {
            v.push(SCALARS[(i + j) % SCALARS.len()]);
        }
        pts.push(v);
    }
    pts
}

/// Check that `rewritten` agrees with `original` at the probe points.
///
/// Returns `true` only when every finite comparison agrees to a relative
/// tolerance of `1e-6` **and** at least three finite comparisons were made.
/// Samples where either side is non-finite (e.g. a pole or a log of a
/// non-positive argument) are skipped. A `false` verdict causes the caller to
/// return the input unchanged, so a spurious rejection is always safe.
pub(crate) fn verify_agrees(original: &LoweredOp, rewritten: &LoweredOp, n_vars: usize) -> bool {
    let pts = probe_points(n_vars);
    let mut agreed = 0usize;
    for pt in &pts {
        let va = original.eval(pt);
        let vb = rewritten.eval(pt);
        if !va.is_finite() || !vb.is_finite() {
            continue;
        }
        let scale = va.abs().max(vb.abs()).max(1.0);
        if (va - vb).abs() > 1e-6 * scale {
            return false;
        }
        agreed += 1;
    }
    agreed >= 3
}

// ── Shared dense linear solver ─────────────────────────────────────────────────

/// Solve the dense linear system `a · x = b` by Gaussian elimination with
/// partial pivoting.
///
/// Returns `None` when the matrix is singular (a pivot smaller than `1e-12` in
/// magnitude) or when back-substitution produces a non-finite component. This
/// is the single copy of the solver in the crate; it is used by
/// [`apart::partial_fractions`] to recover partial-fraction numerators, which in
/// turn is called both by [`LoweredOp::apart`] and by the rational-integration
/// engine in [`crate::integrate`].
///
/// The `a`/`b` inputs are consumed (they are overwritten in place during
/// elimination).
pub(crate) fn gaussian_eliminate(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&i, &j| {
            a[i][col]
                .abs()
                .partial_cmp(&a[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let pivot = a[col][col];
        if pivot.abs() < 1e-12 {
            return None;
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            let col_vals: Vec<f64> = a[col][col..n].to_vec();
            for (a_rk, a_ck) in a[row][col..n].iter_mut().zip(col_vals) {
                *a_rk -= factor * a_ck;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_points_are_positive_and_sized() {
        let pts = probe_points(3);
        assert_eq!(pts.len(), 16);
        for p in &pts {
            assert_eq!(p.len(), 3);
            assert!(p.iter().all(|&v| v > 0.0));
        }
    }

    #[test]
    fn sole_var_detects_single_variable() {
        let x = LoweredOp::Var(0);
        let expr = LoweredOp::Add(Arc::new(x.clone()), Arc::new(LoweredOp::Const(1.0)));
        assert_eq!(sole_var(&expr), Some(0));

        let two = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
        assert_eq!(sole_var(&two), None);

        assert_eq!(sole_var(&LoweredOp::Const(3.0)), None);
    }

    #[test]
    fn gaussian_eliminate_solves_2x2() {
        // [[2, 1], [1, 3]] x = [3, 5]  →  x = [0.8, 1.4]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![3.0, 5.0];
        let x = gaussian_eliminate(a, b).expect("nonsingular");
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn gaussian_eliminate_singular_is_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert!(gaussian_eliminate(a, b).is_none());
    }

    #[test]
    fn verify_agrees_on_equal_expressions() {
        // (x+1)^2  vs  x^2 + 2x + 1
        let x = || LoweredOp::Var(0);
        let lhs = LoweredOp::Pow(
            Arc::new(LoweredOp::Add(
                Arc::new(x()),
                Arc::new(LoweredOp::Const(1.0)),
            )),
            Arc::new(LoweredOp::Const(2.0)),
        );
        let rhs = LoweredOp::Add(
            Arc::new(LoweredOp::Add(
                Arc::new(LoweredOp::Pow(
                    Arc::new(x()),
                    Arc::new(LoweredOp::Const(2.0)),
                )),
                Arc::new(LoweredOp::Mul(
                    Arc::new(LoweredOp::Const(2.0)),
                    Arc::new(x()),
                )),
            )),
            Arc::new(LoweredOp::Const(1.0)),
        );
        assert!(verify_agrees(&lhs, &rhs, 1));
    }

    #[test]
    fn verify_rejects_unequal_expressions() {
        let x = LoweredOp::Var(0);
        let lhs = LoweredOp::Mul(Arc::new(x.clone()), Arc::new(LoweredOp::Const(2.0)));
        let rhs = LoweredOp::Mul(Arc::new(x), Arc::new(LoweredOp::Const(3.0)));
        assert!(!verify_agrees(&lhs, &rhs, 1));
    }
}
