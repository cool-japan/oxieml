//! Solving zero-dimensional polynomial systems by lex elimination.
//!
//! [`solve_zero_dim`] finds the **real** solutions of a square-or-overdetermined
//! polynomial system `f₁ = … = f_m = 0` in `n` variables, provided the system has
//! only finitely many solutions.
//!
//! # The method
//!
//! ## 1. Triangularize with a lex Gröbner basis
//!
//! Compute the reduced Gröbner basis `G` of `I = ⟨f₁, …, f_m⟩` under **lex** with
//! `x₀ > x₁ > … > x_{n−1}`. The **Elimination Theorem** (Cox–Little–O'Shea, Ch. 3
//! §1, Theorem 2) then says
//!
//! ```text
//! G ∩ ℚ[x_k, …, x_{n−1}]   is a Gröbner basis of the elimination ideal
//! I_k = I ∩ ℚ[x_k, …, x_{n−1}]
//! ```
//!
//! for every `k`. In other words a single lex basis contains, already inside it,
//! a triangular cascade: polynomials in `x_{n−1}` alone, then polynomials adding
//! `x_{n−2}`, and so on — a nonlinear analogue of the row echelon form.
//!
//! The reason this works is the defining property of lex: any monomial containing
//! an *earlier* variable outranks every monomial built only from later ones. So if
//! `LM(g)` happens to involve only `x_k, …, x_{n−1}`, then so does all of `g`.
//!
//! ## 2. Check the system is actually zero-dimensional
//!
//! By the **Finiteness Theorem**, `V(I)` is finite iff for every `i` some `g ∈ G`
//! has `LM(g) = x_i^{m_i}`. See [`groebner::is_zero_dimensional`]. If the check
//! fails the system has infinitely many solutions and we honestly report
//! [`ZeroDimOutcome::PositiveDimensional`] rather than inventing a finite list.
//!
//! A useful corollary of that check plus lex: the element with `LM(g) = x_i^{m_i}`
//! involves *only* `x_i, …, x_{n−1}` (by the paragraph above), and its `x_i^{m_i}`
//! coefficient is a nonzero **constant** — because any monomial `x_i^{m_i}·x_j`
//! with `j > i` would still be lex-greater than `x_i^{m_i}`, and so would have
//! been the leading monomial instead. Hence, after substituting *any* values for
//! `x_{i+1}, …, x_{n−1}`, that polynomial stays of degree exactly `m_i` in `x_i`
//! and never collapses to zero. Back-substitution can therefore never get stuck.
//!
//! ## 3. Back-substitute
//!
//! Solve the univariate eliminant in `x_{n−1}`, then walk `i = n−2 … 0`, at each
//! level extending every partial solution by the roots of the polynomials that
//! involve `x_i` and nothing earlier.
//!
//! Two routes are taken at each level:
//!
//! * **Exact / symbolic** — when some basis element is *linear* in `x_i` (the
//!   generic "shape lemma" situation), `x_i = −c₀/c₁` is a closed form in the
//!   already-known coordinates and is built symbolically, with no root-finding at
//!   all. For `⟨x² + y² − 1, x − y⟩` this is what turns `x − y` into `x = y`.
//! * **Numeric** — otherwise the known coordinates are substituted numerically and
//!   the resulting univariate polynomial is solved. Degree 1 and 2 are still handled
//!   exactly (the discriminant's *sign* is decided over ℚ, so a root is never
//!   invented or missed through rounding); higher degrees go through the crate's
//!   Sturm-based real-root isolator.
//!
//! ## 4. Polish and verify
//!
//! Every candidate is finally checked against the **original** system. Candidates
//! carrying numeric error are refined by a Gauss–Newton step using exact symbolic
//! partial derivatives, and any candidate whose residual is still not at zero is
//! **discarded**. Nothing is returned that has not been verified to solve the
//! system that was actually asked about.
//!
//! # Scope and honesty
//!
//! * Only **real** solutions are returned; complex ones are dropped. This is a
//!   deliberate choice — [`LoweredOp`] is a real-valued IR — and is why an empty
//!   solution list is *not* the same as [`ZeroDimOutcome::NoSolutions`], which
//!   means the ideal is the unit ideal and there are no solutions even over ℂ.
//! * Solutions of even multiplicity (tangencies) are reported once.
//! * If Buchberger hits a cap, the error propagates: no partial answer is faked.

use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::poly::univariate::Poly;

use super::groebner::{self, GroebnerError, GroebnerOpts};
use super::monomial::MonOrder;
use super::multivariate::MultiPoly;
use super::{Coeff, coeff_is_zero, coeff_zero, f64_to_ratio, ratio_to_f64};

/// Residual below which a candidate counts as an exact solution already.
const EXACT_RESIDUAL_TOL: f64 = 1e-11;
/// Residual a candidate must reach (after polishing) to be reported at all.
const ACCEPT_RESIDUAL_TOL: f64 = 1e-6;
/// Loose tolerance used while filtering back-substitution candidates.
const CANDIDATE_FILTER_TOL: f64 = 1e-5;
/// Two solutions closer than this in every coordinate are the same solution.
const DEDUP_TOL: f64 = 1e-7;
/// A univariate coefficient smaller than this counts as absent.
const COEFF_ZERO_TOL: f64 = 1e-12;
/// Gauss–Newton iteration cap.
const NEWTON_ITERS: usize = 64;

/// What [`solve_zero_dim`] concluded about a polynomial system.
#[derive(Clone, Debug)]
pub enum ZeroDimOutcome {
    /// The system has finitely many solutions; these are the **real** ones.
    ///
    /// Each inner vector has length `num_vars`: entry `i` is the value of `x_i`.
    /// The list may legitimately be empty when every solution is complex.
    Solutions(Vec<Vec<LoweredOp>>),
    /// The ideal is the unit ideal: the system is contradictory and has no
    /// solutions at all, not even over ℂ.
    NoSolutions,
    /// The solution set is infinite (positive-dimensional), so it cannot be
    /// enumerated. Reported honestly rather than truncated.
    PositiveDimensional,
}

/// Solve a zero-dimensional polynomial system over ℝ.
///
/// `polys` must all live in `num_vars` variables. See the
/// [module documentation](self) for the method and its guarantees.
///
/// The monomial order in `opts` is **ignored**: elimination requires lex, so lex
/// is used regardless. The caps in `opts` are honoured.
///
/// # Errors
///
/// [`GroebnerError`] when the Gröbner basis cannot be computed within the caps in
/// `opts`, or when the generators are inconsistent.
pub fn solve_zero_dim(
    polys: &[MultiPoly],
    num_vars: usize,
    opts: &GroebnerOpts,
) -> Result<ZeroDimOutcome, GroebnerError> {
    // Elimination is a property of lex and of no other order.
    let lex_opts = GroebnerOpts {
        order: MonOrder::Lex,
        ..*opts
    };
    let order = MonOrder::Lex;

    for p in polys {
        if p.num_vars != num_vars {
            return Err(GroebnerError::DimensionMismatch {
                expected: num_vars,
                got: p.num_vars,
            });
        }
    }

    let basis = groebner::groebner_basis(polys, &lex_opts)?;

    if groebner::is_unit_ideal(&basis) {
        return Ok(ZeroDimOutcome::NoSolutions);
    }
    if basis.is_empty() {
        // The zero ideal: every point is a solution.
        return Ok(if num_vars == 0 {
            ZeroDimOutcome::Solutions(vec![Vec::new()])
        } else {
            ZeroDimOutcome::PositiveDimensional
        });
    }
    if num_vars == 0 {
        // No variables and a non-unit, nonzero ideal cannot happen, but be total.
        return Ok(ZeroDimOutcome::Solutions(vec![Vec::new()]));
    }
    if !groebner::is_zero_dimensional(&basis, order, num_vars) {
        return Ok(ZeroDimOutcome::PositiveDimensional);
    }

    // ── Back-substitution ─────────────────────────────────────────────────────
    let mut partials: Vec<Partial> = vec![Partial::empty(num_vars)];

    for i in (0..num_vars).rev() {
        // Basis elements that constrain x_i and involve no earlier variable.
        // By the Elimination Theorem these are exactly a Gröbner basis of the
        // elimination ideal I_i, restricted to those that actually see x_i.
        let relevant: Vec<&MultiPoly> = basis
            .iter()
            .filter(|g| g.involves_var(i) && (0..i).all(|j| !g.involves_var(j)))
            .collect();

        if relevant.is_empty() {
            // The Finiteness Theorem forbids this, but never assume: an
            // unconstrained variable means an infinite solution set.
            return Ok(ZeroDimOutcome::PositiveDimensional);
        }

        let mut next: Vec<Partial> = Vec::new();
        for partial in &partials {
            extend_partial(partial, i, &relevant, num_vars, &mut next)?;
        }
        partials = next;
    }

    // ── Polish, verify, deduplicate ───────────────────────────────────────────
    let jacobian = build_jacobian(polys, num_vars);
    let mut solutions: Vec<(Vec<f64>, Vec<LoweredOp>)> = Vec::new();

    for partial in partials {
        let Some(point) = partial.numeric_point() else {
            continue;
        };
        let Some(exprs) = partial.symbolic_point() else {
            continue;
        };

        if max_residual(polys, &point) <= EXACT_RESIDUAL_TOL {
            // The symbolic construction already lands on the variety: keep the
            // closed forms untouched.
            solutions.push((point, exprs));
            continue;
        }

        let mut polished = point;
        gauss_newton(polys, &jacobian, num_vars, &mut polished);
        if max_residual(polys, &polished) > ACCEPT_RESIDUAL_TOL {
            // Could not verify it against the original system — drop it rather
            // than report a point that does not solve the problem.
            continue;
        }
        let consts = polished.iter().map(|&v| LoweredOp::Const(v)).collect();
        solutions.push((polished, consts));
    }

    Ok(ZeroDimOutcome::Solutions(dedup_solutions(solutions)))
}

// ── Partial solutions ─────────────────────────────────────────────────────────

/// A partially-built solution: coordinates `i+1 … n−1` are known, `0 … i` are not.
///
/// Each known coordinate carries both a symbolic closed form and its `f64` value.
/// The two are kept in lock-step: the value is always the evaluation of the form.
#[derive(Clone, Debug)]
struct Partial {
    exprs: Vec<Option<LoweredOp>>,
    vals: Vec<Option<f64>>,
}

impl Partial {
    fn empty(num_vars: usize) -> Self {
        Self {
            exprs: vec![None; num_vars],
            vals: vec![None; num_vars],
        }
    }

    fn with(&self, var: usize, expr: LoweredOp, value: f64) -> Self {
        let mut next = self.clone();
        next.exprs[var] = Some(expr);
        next.vals[var] = Some(value);
        next
    }

    /// The `f64` point, or `None` if any coordinate is still unknown or non-finite.
    fn numeric_point(&self) -> Option<Vec<f64>> {
        let mut out = Vec::with_capacity(self.vals.len());
        for v in &self.vals {
            let v = (*v)?;
            if !v.is_finite() {
                return None;
            }
            out.push(v);
        }
        Some(out)
    }

    fn symbolic_point(&self) -> Option<Vec<LoweredOp>> {
        self.exprs.iter().cloned().collect()
    }

    /// Values of the *already-known* coordinates, padded with zeros elsewhere, so
    /// a [`MultiPoly`] whose support lies in the known region can be evaluated.
    fn padded_values(&self, num_vars: usize) -> Vec<f64> {
        (0..num_vars).map(|j| self.vals[j].unwrap_or(0.0)).collect()
    }
}

/// Extend `partial` with every real value of `x_var` consistent with `relevant`.
fn extend_partial(
    partial: &Partial,
    var: usize,
    relevant: &[&MultiPoly],
    num_vars: usize,
    out: &mut Vec<Partial>,
) -> Result<(), GroebnerError> {
    let point = partial.padded_values(num_vars);

    // ── Route A: a basis element linear in x_var gives a closed form ──────────
    //
    // Pick the linear one with the fewest terms, for the tidiest expression.
    let mut best_linear: Option<(&MultiPoly, usize)> = None;
    for g in relevant {
        if g.degree_in(var) != 1 {
            continue;
        }
        let c1 = g.coeff_in_var(var, 1);
        if c1.eval_f64(&point).abs() <= COEFF_ZERO_TOL {
            // The leading coefficient degenerates at this point: this element
            // says nothing about x_var here.
            continue;
        }
        let terms = g.num_terms();
        if best_linear.is_none_or(|(_, best)| terms < best) {
            best_linear = Some((g, terms));
        }
    }

    if let Some((g, _)) = best_linear {
        let c1 = g.coeff_in_var(var, 1);
        let c0 = g.coeff_in_var(var, 0);
        let denom = c1.eval_f64(&point);
        let numer = c0.eval_f64(&point);
        let value = -numer / denom;

        if value.is_finite() {
            // x_var = −c₀ / c₁, built symbolically from the known coordinates.
            let (Some(c0_expr), Some(c1_expr)) = (
                substitute_symbolic(&c0, &partial.exprs),
                substitute_symbolic(&c1, &partial.exprs),
            ) else {
                return Ok(());
            };
            let expr = LoweredOp::Neg(Arc::new(LoweredOp::Div(
                Arc::new(c0_expr),
                Arc::new(c1_expr),
            )))
            .simplify();
            out.push(partial.with(var, expr, value));
        }
        return Ok(());
    }

    // ── Route B: solve a univariate polynomial in x_var numerically ───────────
    //
    // Substitute the known coordinates into every relevant element, then pick the
    // lowest-degree non-degenerate one to supply the candidate roots. As argued in
    // the module docs, the pure-power element guarantees at least one exists.
    let mut candidates: Vec<Vec<f64>> = Vec::new();
    for g in relevant {
        let coeffs = univariate_at(g, var, &point);
        if effective_degree(&coeffs).is_some() {
            candidates.push(coeffs);
        }
    }
    let Some(primary_idx) = candidates
        .iter()
        .enumerate()
        .filter_map(|(k, c)| effective_degree(c).map(|d| (d, k)))
        .min()
        .map(|(_, k)| k)
    else {
        // Every relevant polynomial vanished identically on this branch. The
        // fibre is infinite, contradicting zero-dimensionality, so the branch is
        // spurious: drop it.
        return Ok(());
    };

    for (expr, value) in real_roots_of(&candidates[primary_idx])? {
        // Keep only roots that (loosely) satisfy the *other* relevant elements.
        // Final acceptance is decided later against the original system.
        let consistent = candidates.iter().enumerate().all(|(k, coeffs)| {
            k == primary_idx || eval_univariate(coeffs, value).abs() <= CANDIDATE_FILTER_TOL
        });
        if consistent {
            out.push(partial.with(var, expr, value));
        }
    }

    Ok(())
}

// ── Univariate machinery ──────────────────────────────────────────────────────

/// Coefficients (index = degree) of `g` viewed as a polynomial in `x_var`, with
/// the other variables replaced by their numeric values from `point`.
fn univariate_at(g: &MultiPoly, var: usize, point: &[f64]) -> Vec<f64> {
    let degree = g.degree_in(var);
    let mut coeffs = vec![0.0f64; degree + 1];
    for (exps, coeff) in &g.terms {
        if coeff_is_zero(coeff) {
            continue;
        }
        let d = if var < exps.len() {
            exps[var] as usize
        } else {
            0
        };
        let mut term = ratio_to_f64(coeff);
        for (j, &e) in exps.iter().enumerate() {
            if j == var || e == 0 {
                continue;
            }
            let x = point.get(j).copied().unwrap_or(0.0);
            term *= x.powi(i32::try_from(e).unwrap_or(i32::MAX));
        }
        if d < coeffs.len() {
            coeffs[d] += term;
        }
    }
    coeffs
}

/// Highest index whose coefficient is not numerically zero, if that index is ≥ 1.
///
/// Returns `None` for the zero polynomial and for nonzero constants — neither
/// constrains the variable in a way that yields candidate roots.
fn effective_degree(coeffs: &[f64]) -> Option<usize> {
    let top = coeffs
        .iter()
        .rposition(|c| c.abs() > COEFF_ZERO_TOL && c.is_finite())?;
    if top == 0 { None } else { Some(top) }
}

fn eval_univariate(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0f64, |acc, &c| acc * x + c)
}

/// Real roots of a univariate polynomial given by `f64` coefficients.
///
/// Degrees 1 and 2 are handled **exactly**: the coefficients are rationalized
/// losslessly (every `f64` *is* a rational), so the discriminant's sign is decided
/// over ℚ and a root can be neither invented nor missed by rounding. Higher
/// degrees defer to the crate's Sturm-isolating solver.
///
/// Returns `(closed form, value)` pairs.
fn real_roots_of(coeffs: &[f64]) -> Result<Vec<(LoweredOp, f64)>, GroebnerError> {
    let Some(degree) = effective_degree(coeffs) else {
        return Ok(Vec::new());
    };

    let mut exact: Vec<Coeff> = Vec::with_capacity(degree + 1);
    for &c in &coeffs[..=degree] {
        exact.push(f64_to_ratio(c)?);
    }
    let poly = Poly::from_ratios(exact);

    // Squarefree part: collapses repeated roots, which keeps the numeric solvers
    // well-conditioned and reports a tangency exactly once.
    let squarefree = poly.square_free()?;
    let poly = if squarefree.degree().is_some_and(|d| d >= 1) {
        squarefree
    } else {
        poly
    };

    let Some(degree) = poly.degree() else {
        return Ok(Vec::new());
    };

    match degree {
        0 => Ok(Vec::new()),
        1 => {
            // a₁x + a₀ = 0  ⟹  x = −a₀/a₁, an exact rational.
            let a0 = &poly.coeffs[0];
            let a1 = &poly.coeffs[1];
            if coeff_is_zero(a1) {
                return Ok(Vec::new());
            }
            let root = -(a0 / a1);
            let value = ratio_to_f64(&root);
            Ok(vec![(LoweredOp::Const(value), value)])
        }
        2 => {
            // a₂x² + a₁x + a₀ = 0. The discriminant is computed over ℚ, so its
            // sign — the very thing that decides how many real roots exist — is
            // exact rather than epsilon-guessed.
            let a0 = poly.coeffs[0].clone();
            let a1 = poly.coeffs[1].clone();
            let a2 = poly.coeffs[2].clone();
            if coeff_is_zero(&a2) {
                return Ok(Vec::new());
            }
            let four = Coeff::from_integer(num_bigint::BigInt::from(4i32));
            let disc = &a1 * &a1 - four * &a2 * &a0;
            let zero = coeff_zero();

            if disc < zero {
                return Ok(Vec::new());
            }

            let two_a = ratio_to_f64(&(&a2 + &a2));
            let neg_b = ratio_to_f64(&(-&a1));

            if disc == zero {
                let value = neg_b / two_a;
                return Ok(vec![(LoweredOp::Const(value), value)]);
            }

            let disc_f64 = ratio_to_f64(&disc);
            let sqrt_disc = LoweredOp::Pow(
                Arc::new(LoweredOp::Const(disc_f64)),
                Arc::new(LoweredOp::Const(0.5)),
            );
            let root_of = |sign: f64| -> (LoweredOp, f64) {
                let shifted = if sign < 0.0 {
                    LoweredOp::Sub(
                        Arc::new(LoweredOp::Const(neg_b)),
                        Arc::new(sqrt_disc.clone()),
                    )
                } else {
                    LoweredOp::Add(
                        Arc::new(LoweredOp::Const(neg_b)),
                        Arc::new(sqrt_disc.clone()),
                    )
                };
                let expr =
                    LoweredOp::Div(Arc::new(shifted), Arc::new(LoweredOp::Const(two_a))).simplify();
                let value = (neg_b + sign * disc_f64.sqrt()) / two_a;
                (expr, value)
            };
            Ok(vec![root_of(-1.0), root_of(1.0)])
        }
        _ => {
            // Cubic: closed form; quartic and up: Sturm-sequence isolation with
            // bisection refinement. Both live in `solve_poly` already.
            let roots = crate::solve_poly::solve_polynomial(&poly, 0)
                .map_err(|_| GroebnerError::Poly(super::PolyError::NotPolynomial))?;
            Ok(roots
                .roots
                .into_iter()
                .map(|r| {
                    let v = r.eval(&[]);
                    (r, v)
                })
                .filter(|(_, v)| v.is_finite())
                .collect())
        }
    }
}

// ── Symbolic substitution ─────────────────────────────────────────────────────

/// Rebuild `p` as a [`LoweredOp`], replacing every variable by its closed form.
///
/// Returns `None` when `p` uses a variable that has no closed form yet.
fn substitute_symbolic(p: &MultiPoly, subs: &[Option<LoweredOp>]) -> Option<LoweredOp> {
    let mut terms: Vec<LoweredOp> = Vec::new();

    for (exps, coeff) in &p.terms {
        if coeff_is_zero(coeff) {
            continue;
        }
        let c = ratio_to_f64(coeff);
        let mut factors: Vec<LoweredOp> = Vec::new();
        for (j, &e) in exps.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let base = subs.get(j)?.clone()?;
            factors.push(if e == 1 {
                base
            } else {
                LoweredOp::Pow(Arc::new(base), Arc::new(LoweredOp::Const(f64::from(e))))
            });
        }

        let term = match factors.split_first() {
            None => LoweredOp::Const(c),
            Some((head, rest)) => {
                let mut acc = head.clone();
                for f in rest {
                    acc = LoweredOp::Mul(Arc::new(acc), Arc::new(f.clone()));
                }
                if (c - 1.0).abs() < f64::EPSILON {
                    acc
                } else {
                    LoweredOp::Mul(Arc::new(LoweredOp::Const(c)), Arc::new(acc))
                }
            }
        };
        terms.push(term);
    }

    Some(match terms.split_first() {
        None => LoweredOp::Const(0.0),
        Some((head, rest)) => {
            let mut acc = head.clone();
            for t in rest {
                acc = LoweredOp::Add(Arc::new(acc), Arc::new(t.clone()));
            }
            acc
        }
    })
}

// ── Verification: Gauss–Newton polish against the original system ─────────────

/// `jacobian[k][j] = ∂f_k/∂x_j`, exact symbolic derivatives.
fn build_jacobian(polys: &[MultiPoly], num_vars: usize) -> Vec<Vec<MultiPoly>> {
    polys
        .iter()
        .map(|f| (0..num_vars).map(|j| f.partial_derivative(j)).collect())
        .collect()
}

fn max_residual(polys: &[MultiPoly], point: &[f64]) -> f64 {
    polys
        .iter()
        .map(|f| f.eval_f64(point).abs())
        .fold(0.0f64, f64::max)
}

/// Refine `point` with Gauss–Newton on the (possibly overdetermined) system.
///
/// Solves the normal equations `(JᵀJ)·δ = −Jᵀ·F` each step, which reduces to an
/// ordinary Newton step when the system is square and non-singular. Purely a
/// *refinement*: it cannot rescue a candidate that is not near a genuine root, and
/// the caller verifies the residual afterwards regardless.
fn gauss_newton(
    polys: &[MultiPoly],
    jacobian: &[Vec<MultiPoly>],
    num_vars: usize,
    point: &mut [f64],
) {
    if num_vars == 0 || polys.is_empty() {
        return;
    }

    for _ in 0..NEWTON_ITERS {
        let residual: Vec<f64> = polys.iter().map(|f| f.eval_f64(point)).collect();
        let norm = residual.iter().fold(0.0f64, |a, &r| a.max(r.abs()));
        if norm <= EXACT_RESIDUAL_TOL || !norm.is_finite() {
            return;
        }

        // J (m × n), evaluated at the current point.
        let jac: Vec<Vec<f64>> = jacobian
            .iter()
            .map(|row| row.iter().map(|d| d.eval_f64(point)).collect())
            .collect();

        // Normal equations: JᵀJ (n × n) and −Jᵀr (n).
        let mut ata = vec![0.0f64; num_vars * num_vars];
        let mut atb = vec![0.0f64; num_vars];
        for (row, &r) in jac.iter().zip(residual.iter()) {
            for a in 0..num_vars {
                atb[a] -= row[a] * r;
                for b in 0..num_vars {
                    ata[a * num_vars + b] += row[a] * row[b];
                }
            }
        }

        if crate::linalg::solve_lu(&mut ata, &mut atb, num_vars).is_err() {
            // Singular Jacobian (a multiple root, typically). The candidate is
            // whatever back-substitution produced; the caller still verifies it.
            return;
        }

        let mut improved = false;
        for (x, dx) in point.iter_mut().zip(atb.iter()) {
            if !dx.is_finite() {
                return;
            }
            if dx.abs() > 0.0 {
                improved = true;
            }
            *x += dx;
        }
        if !improved {
            return;
        }
    }
}

/// Drop duplicate solutions (same point to within [`DEDUP_TOL`]).
fn dedup_solutions(solutions: Vec<(Vec<f64>, Vec<LoweredOp>)>) -> Vec<Vec<LoweredOp>> {
    let mut kept: Vec<(Vec<f64>, Vec<LoweredOp>)> = Vec::new();
    for (point, exprs) in solutions {
        let duplicate = kept.iter().any(|(seen, _)| {
            seen.iter()
                .zip(point.iter())
                .all(|(a, b)| (a - b).abs() <= DEDUP_TOL)
        });
        if !duplicate {
            kept.push((point, exprs));
        }
    }

    // Deterministic output: sort lexicographically by numeric coordinates.
    kept.sort_by(|a, b| {
        a.0.iter()
            .zip(b.0.iter())
            .map(|(x, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .find(|o| *o != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    kept.into_iter().map(|(_, exprs)| exprs).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::coeff_from_i64;
    use crate::poly::groebner::multipoly_from_terms;

    fn t(c: i64, e: &[u32]) -> (Coeff, Vec<u32>) {
        (coeff_from_i64(c), e.to_vec())
    }

    fn poly(terms: &[(Coeff, Vec<u32>)], n: usize) -> MultiPoly {
        multipoly_from_terms(terms, n).expect("well-formed test polynomial")
    }

    fn solutions_of(polys: &[MultiPoly], n: usize) -> Vec<Vec<f64>> {
        let outcome = solve_zero_dim(polys, n, &GroebnerOpts::default()).expect("terminates");
        match outcome {
            ZeroDimOutcome::Solutions(sols) => sols
                .into_iter()
                .map(|s| s.iter().map(|e| e.eval(&[])).collect())
                .collect(),
            other => panic!("expected Solutions, got {other:?}"),
        }
    }

    #[test]
    fn circle_meets_diagonal_at_the_two_expected_points() {
        // {x² + y² = 1, x = y}  ⟹  (±1/√2, ±1/√2), same sign in both coordinates.
        let circle = poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(-1, &[0, 0])], 2);
        let diagonal = poly(&[t(1, &[1, 0]), t(-1, &[0, 1])], 2);

        let sols = solutions_of(&[circle, diagonal], 2);
        assert_eq!(sols.len(), 2, "got {sols:?}");

        let r = 1.0 / 2.0_f64.sqrt();
        // Sorted ascending, so the negative point comes first.
        assert!((sols[0][0] + r).abs() < 1e-12, "{sols:?}");
        assert!((sols[0][1] + r).abs() < 1e-12, "{sols:?}");
        assert!((sols[1][0] - r).abs() < 1e-12, "{sols:?}");
        assert!((sols[1][1] - r).abs() < 1e-12, "{sols:?}");
    }

    #[test]
    fn circle_meets_line_x_plus_y_eq_1_at_the_axis_points() {
        // {x² + y² = 1, x + y = 1} ⟹ (1, 0) and (0, 1). Exact rational solutions.
        let circle = poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(-1, &[0, 0])], 2);
        let line = poly(&[t(1, &[1, 0]), t(1, &[0, 1]), t(-1, &[0, 0])], 2);

        let sols = solutions_of(&[circle, line], 2);
        assert_eq!(sols.len(), 2, "got {sols:?}");
        assert!(
            sols.iter()
                .any(|s| (s[0] - 1.0).abs() < 1e-10 && s[1].abs() < 1e-10)
        );
        assert!(
            sols.iter()
                .any(|s| s[0].abs() < 1e-10 && (s[1] - 1.0).abs() < 1e-10)
        );
    }

    #[test]
    fn tangency_is_reported_once() {
        // {x² + y² = 1, y = 1} touches at the single point (0, 1) with multiplicity 2.
        let circle = poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(-1, &[0, 0])], 2);
        let tangent = poly(&[t(1, &[0, 1]), t(-1, &[0, 0])], 2);

        let sols = solutions_of(&[circle, tangent], 2);
        assert_eq!(sols.len(), 1, "a tangency is one point: {sols:?}");
        assert!(sols[0][0].abs() < 1e-7, "{sols:?}");
        assert!((sols[0][1] - 1.0).abs() < 1e-7, "{sols:?}");
    }

    #[test]
    fn purely_complex_intersections_yield_an_empty_real_solution_list() {
        // {x² + y² = −1, x = y} has no real points, but the ideal is NOT the unit
        // ideal — the solutions are complex. Those two outcomes must not be
        // confused with one another.
        let imaginary = poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(1, &[0, 0])], 2);
        let diagonal = poly(&[t(1, &[1, 0]), t(-1, &[0, 1])], 2);

        let outcome = solve_zero_dim(&[imaginary, diagonal], 2, &GroebnerOpts::default())
            .expect("terminates");
        match outcome {
            ZeroDimOutcome::Solutions(s) => assert!(s.is_empty(), "got {s:?}"),
            other => panic!("expected an empty real solution list, got {other:?}"),
        }
    }

    #[test]
    fn contradictory_system_is_no_solutions_not_an_empty_list() {
        // {x = 0, x = 1} — the unit ideal.
        let a = poly(&[t(1, &[1, 0])], 2);
        let b = poly(&[t(1, &[1, 0]), t(-1, &[0, 0])], 2);
        // y must be constrained too, or the system is positive-dimensional.
        let c = poly(&[t(1, &[0, 1])], 2);

        let outcome = solve_zero_dim(&[a, b, c], 2, &GroebnerOpts::default()).expect("terminates");
        assert!(
            matches!(outcome, ZeroDimOutcome::NoSolutions),
            "got {outcome:?}"
        );
    }

    #[test]
    fn underdetermined_system_is_positive_dimensional() {
        // A single circle in two variables: a curve, not a finite point set.
        let circle = poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(-1, &[0, 0])], 2);
        let outcome = solve_zero_dim(&[circle], 2, &GroebnerOpts::default()).expect("terminates");
        assert!(
            matches!(outcome, ZeroDimOutcome::PositiveDimensional),
            "got {outcome:?}"
        );
    }

    #[test]
    fn nonlinear_three_variable_system() {
        // {x + y + z = 6, x − y = 0, z² = 4} with z = ±2.
        //   z = 2  ⟹ x + y = 4, x = y ⟹ (2, 2, 2)
        //   z = −2 ⟹ x + y = 8, x = y ⟹ (4, 4, −2)
        let e1 = poly(
            &[
                t(1, &[1, 0, 0]),
                t(1, &[0, 1, 0]),
                t(1, &[0, 0, 1]),
                t(-6, &[0, 0, 0]),
            ],
            3,
        );
        let e2 = poly(&[t(1, &[1, 0, 0]), t(-1, &[0, 1, 0])], 3);
        let e3 = poly(&[t(1, &[0, 0, 2]), t(-4, &[0, 0, 0])], 3);

        let sols = solutions_of(&[e1, e2, e3], 3);
        assert_eq!(sols.len(), 2, "got {sols:?}");
        assert!(
            sols.iter().any(|s| (s[0] - 2.0).abs() < 1e-9
                && (s[1] - 2.0).abs() < 1e-9
                && (s[2] - 2.0).abs() < 1e-9),
            "missing (2,2,2): {sols:?}"
        );
        assert!(
            sols.iter().any(|s| (s[0] - 4.0).abs() < 1e-9
                && (s[1] - 4.0).abs() < 1e-9
                && (s[2] + 2.0).abs() < 1e-9),
            "missing (4,4,−2): {sols:?}"
        );
    }

    #[test]
    fn quartic_eliminant_goes_through_the_sturm_solver() {
        // {x² − y = 0, y² − x − 2 = 0}. Eliminating x gives a quartic in y that no
        // closed form handles here, so this exercises the Sturm fallback path.
        // Substituting y = x²: x⁴ − x − 2 = 0, which factors as (x − ...)(...).
        let f = poly(&[t(1, &[2, 0]), t(-1, &[0, 1])], 2);
        let g = poly(&[t(1, &[0, 2]), t(-1, &[1, 0]), t(-2, &[0, 0])], 2);

        let sols = solutions_of(&[f, g], 2);
        assert!(!sols.is_empty(), "expected at least one real solution");
        // Every returned point must genuinely satisfy the original system.
        for s in &sols {
            assert!((s[0] * s[0] - s[1]).abs() < 1e-7, "x² ≠ y at {s:?}");
            assert!(
                (s[1] * s[1] - s[0] - 2.0).abs() < 1e-7,
                "y² − x − 2 ≠ 0 at {s:?}"
            );
        }
    }

    #[test]
    fn every_returned_solution_satisfies_the_original_system() {
        // Property check across several systems: nothing unverified escapes.
        let systems: Vec<(Vec<MultiPoly>, usize)> = vec![
            (
                vec![
                    poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(-1, &[0, 0])], 2),
                    poly(&[t(1, &[1, 0]), t(-1, &[0, 1])], 2),
                ],
                2,
            ),
            (
                vec![
                    poly(&[t(1, &[2, 0]), t(-1, &[0, 1])], 2),
                    poly(&[t(1, &[0, 1]), t(-4, &[0, 0])], 2),
                ],
                2,
            ),
            (
                vec![
                    poly(&[t(1, &[1, 1]), t(-1, &[0, 0])], 2),
                    poly(&[t(1, &[1, 0]), t(-1, &[0, 1])], 2),
                ],
                2,
            ),
        ];

        for (polys, n) in systems {
            let sols = solutions_of(&polys, n);
            for s in &sols {
                for f in &polys {
                    assert!(
                        f.eval_f64(s).abs() < ACCEPT_RESIDUAL_TOL,
                        "residual {} at {s:?}",
                        f.eval_f64(s)
                    );
                }
            }
        }
    }

    #[test]
    fn cap_exhaustion_propagates_as_an_error() {
        let circle = poly(&[t(1, &[2, 0]), t(1, &[0, 2]), t(-1, &[0, 0])], 2);
        let diagonal = poly(&[t(1, &[1, 0]), t(-1, &[0, 1])], 2);
        let opts = GroebnerOpts {
            max_total_degree: 1,
            ..GroebnerOpts::default()
        };
        assert!(
            solve_zero_dim(&[circle, diagonal], 2, &opts).is_err(),
            "a cap hit must be an error, never an empty solution list"
        );
    }
}
