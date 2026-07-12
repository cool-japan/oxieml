//! Shared OxiZ encoding of EML constraints (LRA relaxation).
//!
//! Both the one-shot [`EmlSmtSolver`](super::EmlSmtSolver) (fresh OxiZ
//! `Solver` per query) and the [`IncrementalEmlSolver`](super::IncrementalEmlSolver)
//! (one live `Solver`, `push`/`pop` per query) funnel through the functions in
//! this module, so the *encoding* is bit-for-bit identical between them and the
//! only difference between the two backends is the solver lifecycle. That is
//! exactly what the differential push/pop tests exercise.
//!
//! # Encoding
//!
//! `eml(l, r) = exp(l) − ln(r)` is not expressible in LRA, so each `Eml` node is
//! replaced by two fresh real-sorted auxiliary variables `ex ≈ exp(l)` and
//! `lg ≈ ln(r)`, constrained by a **secant + tangent linear relaxation** over the
//! interval bounds of `l` and `r` obtained from interval propagation:
//!
//! * `exp` is convex, so on `l ∈ [a, b]`
//!   - the chord through `(a, e^a)` and `(b, e^b)` is an **upper** bound, and
//!   - every tangent `e^t + e^t·(l − t)` is a **lower** bound.
//! * `ln` is concave, so on `r ∈ [c, d]` with `c > 0`
//!   - the chord through `(c, ln c)` and `(d, ln d)` is a **lower** bound, and
//!   - every tangent `ln t + (1/t)·(r − t)` is an **upper** bound.
//!
//! The node's value is then the LRA term `ex − lg`.
//!
//! # Soundness
//!
//! The relaxation is a strict **over-approximation**: every real assignment that
//! satisfies the original nonlinear constraint also satisfies the relaxation
//! (with `ex = exp(l)`, `lg = ln(r)`, which lie between the tangent and secant
//! envelopes by convexity/concavity). Therefore
//!
//! > `relaxation UNSAT  ⟹  original UNSAT`
//!
//! and `Unsat` may be reported directly. The converse does **not** hold: a model
//! of the relaxation need not satisfy the original constraint, so `Sat` from OxiZ
//! is only used as a *seed* and every reported `Sat` is re-verified concretely by
//! `check_constraint` on the witness (see `super::oxiz_backend::check_sat_flow`).
//!
//! When any part of the constraint cannot be soundly encoded (unbounded interval,
//! `ln` of an interval reaching `≤ 0`, a rational that would overflow `i64`, or a
//! nested quantifier) the encoder returns `None` and the caller degrades to
//! `Unknown` — never to `Unsat`. This is the invariant that fixes Issue #1
//! (spurious `Unsat` from a real-domain `ln` of a non-positive interval).

use super::constraint::EmlConstraint;
use super::interval::{Interval, IntervalDomain};
use crate::tree::EmlNode;
use oxiz::{Solver, SolverResult, TermId, TermManager};

/// Verdict returned by the OxiZ LRA layer for one query.
pub(super) enum OxizVerdict {
    /// The relaxation is satisfiable; the payload is the model's projection onto
    /// the free variables, to be used as a *seed* (never trusted as a witness).
    Sat(Vec<f64>),
    /// The relaxation is unsatisfiable — sound, so the original is UNSAT too.
    Unsat,
    /// The query could not be encoded or OxiZ gave up.
    Unknown,
}

/// Anything that can answer "is this constraint satisfiable over `domain`?" with
/// the OxiZ LRA relaxation.
///
/// Implemented by both solver lifecycles so that `check_sat_flow` — the
/// interval-propagation / quantifier / witness-verification pipeline — is shared
/// verbatim and the *only* difference under test is `Solver` reuse.
pub(super) trait LraOracle {
    /// Check `c` over the (already propagated) `domain`.
    fn lra_check(&mut self, c: &EmlConstraint, domain: &IntervalDomain) -> OxizVerdict;
}

/// Cap on the denominator of the rational approximation of an `f64`.
///
/// Unbounded continued-fraction rationals make `Rational64` accumulate
/// denominators near `i64::MAX`, which then overflows when two rational terms are
/// multiplied (which happens in every secant/tangent constraint). Capping keeps
/// all intermediate LRA arithmetic safely inside `i64`.
const DENOM_CAP: i64 = 1_000_000;

/// Cap on the magnitude of a value we are willing to encode.
const VALUE_CAP: f64 = 1.0e12;

/// Convert `f64` to an OxiZ real term with a bounded-denominator rational
/// approximation. Returns `None` for non-finite or out-of-range values, which
/// makes the enclosing encoding fail and the query degrade to `Unknown`.
pub(super) fn float_to_term(tm: &mut TermManager, v: f64) -> Option<TermId> {
    use num_rational::Rational64;
    if !v.is_finite() || v.abs() > VALUE_CAP {
        return None;
    }
    let scaled = (v * DENOM_CAP as f64).round();
    if !scaled.is_finite() || scaled.abs() > (i64::MAX as f64) / 4.0 {
        return None;
    }
    Some(tm.mk_real(Rational64::new(scaled as i64, DENOM_CAP)))
}

/// Declare `n` fresh real-sorted variable terms for one query scope.
///
/// The `scope` tag makes the interned names — and therefore the `TermId`s, since
/// `TermManager` hash-conses — unique to this scope. See the soundness note on
/// [`IncrementalEmlSolver`](super::IncrementalEmlSolver): OxiZ 0.2.3's `pop()`
/// does *not* roll back the theory solvers' `TermId → theory-variable` caches, so
/// re-asserting a `TermId` that a previous `pop()` destroyed would hand the
/// simplex a dangling variable. Fresh names per scope make that impossible.
pub(super) fn declare_vars(tm: &mut TermManager, n: usize, scope: u64) -> Vec<TermId> {
    let real_sort = tm.sorts.real_sort;
    (0..n)
        .map(|i| tm.mk_var(&format!("x{i}@{scope}"), real_sort))
        .collect()
}

/// Assert `lo ≤ x_i ≤ hi` for every variable, at the solver's current assertion
/// level. Returns `false` if any bound is not encodable (caller ⇒ `Unknown`).
pub(super) fn assert_var_bounds(
    var_terms: &[TermId],
    domain: &IntervalDomain,
    tm: &mut TermManager,
    solver: &mut Solver,
) -> bool {
    for (i, iv) in domain.vars.iter().enumerate() {
        let Some(&x) = var_terms.get(i) else {
            return false;
        };
        let (Some(lo), Some(hi)) = (float_to_term(tm, iv.lo), float_to_term(tm, iv.hi)) else {
            return false;
        };
        let ge = tm.mk_ge(x, lo);
        let le = tm.mk_le(x, hi);
        solver.assert(ge, tm);
        solver.assert(le, tm);
    }
    true
}

/// Mutable state threaded through the recursive encoder.
pub(super) struct Encoder<'a> {
    /// One OxiZ term per free EML variable.
    pub(super) var_terms: &'a [TermId],
    /// Interval bounds used to build the linear relaxation.
    pub(super) domain: &'a IntervalDomain,
    /// Number of tangent sample points per `exp`/`ln` (≥ 1).
    pub(super) samples: usize,
    /// Monotonically increasing counter that names auxiliary variables.
    ///
    /// It is **never** reset while a `Solver` is alive, so no auxiliary `TermId`
    /// is ever asserted in two different `push`/`pop` scopes.
    pub(super) aux_counter: &'a mut u64,
}

impl Encoder<'_> {
    /// Allocate a fresh auxiliary real variable named `prefix{k}`.
    fn fresh_aux(&mut self, prefix: &str, tm: &mut TermManager) -> TermId {
        *self.aux_counter += 1;
        let real_sort = tm.sorts.real_sort;
        tm.mk_var(&format!("{prefix}{}", *self.aux_counter), real_sort)
    }
}

/// Encode an EML constraint into a boolean OxiZ term, asserting the relaxation
/// side constraints for every `Eml` node into `solver` as a side effect.
///
/// Returns `None` when the constraint is outside the encodable fragment; the
/// caller must then report `Unknown` (never `Unsat`).
pub(super) fn encode_constraint(
    c: &EmlConstraint,
    enc: &mut Encoder<'_>,
    tm: &mut TermManager,
    solver: &mut Solver,
) -> Option<TermId> {
    match c {
        EmlConstraint::EqZero(tree) => {
            let (t, zero) = encode_atom(tree, enc, tm, solver)?;
            Some(tm.mk_eq(t, zero))
        }
        EmlConstraint::GtZero(tree) => {
            let (t, zero) = encode_atom(tree, enc, tm, solver)?;
            Some(tm.mk_gt(t, zero))
        }
        EmlConstraint::GeZero(tree) => {
            let (t, zero) = encode_atom(tree, enc, tm, solver)?;
            Some(tm.mk_ge(t, zero))
        }
        EmlConstraint::LtZero(tree) => {
            let (t, zero) = encode_atom(tree, enc, tm, solver)?;
            Some(tm.mk_lt(t, zero))
        }
        EmlConstraint::LeZero(tree) => {
            let (t, zero) = encode_atom(tree, enc, tm, solver)?;
            Some(tm.mk_le(t, zero))
        }
        EmlConstraint::NeZero(tree) => {
            let (t, zero) = encode_atom(tree, enc, tm, solver)?;
            let eq_zero = tm.mk_eq(t, zero);
            Some(tm.mk_not(eq_zero))
        }
        EmlConstraint::Not(inner) => {
            let nnf = (**inner).clone().to_nnf();
            encode_constraint(&nnf, enc, tm, solver)
        }
        EmlConstraint::And(cs) => {
            if cs.is_empty() {
                return Some(tm.mk_true());
            }
            let mut encoded = Vec::with_capacity(cs.len());
            for inner in cs {
                encoded.push(encode_constraint(inner, enc, tm, solver)?);
            }
            Some(tm.mk_and(encoded))
        }
        EmlConstraint::Or(cs) => {
            if cs.is_empty() {
                return Some(tm.mk_false());
            }
            let mut encoded = Vec::with_capacity(cs.len());
            for inner in cs {
                encoded.push(encode_constraint(inner, enc, tm, solver)?);
            }
            Some(tm.mk_or(encoded))
        }
        // Quantifiers nested inside a propositional structure are outside the LRA
        // fragment. Reporting `None` (⇒ `Unknown`) is the sound answer; the
        // top-level quantifier cases are handled by the decision procedures in
        // `super::helpers` before the encoder ever runs.
        EmlConstraint::ForAll { .. } | EmlConstraint::Exists { .. } => None,
    }
}

/// Encode the tree of a comparison-with-zero atom, returning `(value, zero)`.
fn encode_atom(
    tree: &crate::tree::EmlTree,
    enc: &mut Encoder<'_>,
    tm: &mut TermManager,
    solver: &mut Solver,
) -> Option<(TermId, TermId)> {
    let (term, _) = encode_tree(&tree.root, enc, tm, solver)?;
    let zero = float_to_term(tm, 0.0)?;
    Some((term, zero))
}

/// Encode an EML subtree, returning `(term, interval)` where `term` is the LRA
/// term standing for the node's value and `interval` is a sound bound on it.
///
/// Every `Eml` node contributes two fresh auxiliary variables and their
/// secant/tangent relaxation constraints, asserted directly into `solver` at its
/// current assertion level (so `pop()` removes them together with the constraint
/// they belong to).
pub(super) fn encode_tree(
    node: &EmlNode,
    enc: &mut Encoder<'_>,
    tm: &mut TermManager,
    solver: &mut Solver,
) -> Option<(TermId, Interval)> {
    match node {
        EmlNode::One => Some((float_to_term(tm, 1.0)?, Interval::point(1.0))),
        EmlNode::Const(v) => Some((float_to_term(tm, *v)?, Interval::point(*v))),
        EmlNode::Var(i) => {
            let iv = *enc.domain.vars.get(*i)?;
            let term = *enc.var_terms.get(*i)?;
            Some((term, iv))
        }
        EmlNode::Eml { left, right } => {
            let (l_term, l_iv) = encode_tree(left, enc, tm, solver)?;
            let (r_term, r_iv) = encode_tree(right, enc, tm, solver)?;

            // Guard: `exp` needs a bounded argument; `ln` additionally needs a
            // strictly positive, bounded one. Anything else is *indeterminate*,
            // not infeasible — bail out to `Unknown` rather than encode a
            // constraint that could produce a spurious `Unsat` (Issue #1).
            if !l_iv.lo.is_finite() || !l_iv.hi.is_finite() || l_iv.is_empty() {
                return None;
            }
            if !r_iv.lo.is_finite() || !r_iv.hi.is_finite() || r_iv.is_empty() || r_iv.lo <= 0.0 {
                return None;
            }

            let ex = enc.fresh_aux("__ex", tm);
            let lg = enc.fresh_aux("__ln", tm);
            let n = enc.samples.max(1);

            // ---- exp(left) relaxation (exp is convex) ----
            let exp_lo = float_to_term(tm, l_iv.lo.exp())?;
            let exp_hi = float_to_term(tm, l_iv.hi.exp())?;
            let bound_lo = tm.mk_ge(ex, exp_lo);
            let bound_hi = tm.mk_le(ex, exp_hi);
            solver.assert(bound_lo, tm);
            solver.assert(bound_hi, tm);

            if l_iv.width() > 0.0 {
                // Secant (upper): ex ≤ exp(a) + slope·(l − a).
                let slope = (l_iv.hi.exp() - l_iv.lo.exp()) / l_iv.width();
                let slope_term = float_to_term(tm, slope)?;
                let a = float_to_term(tm, l_iv.lo)?;
                let diff = tm.mk_sub(l_term, a);
                let prod = tm.mk_mul([slope_term, diff]);
                let rhs = tm.mk_add([exp_lo, prod]);
                let secant = tm.mk_le(ex, rhs);
                solver.assert(secant, tm);

                // Tangents (lower): ex ≥ exp(t) + exp(t)·(l − t).
                for k in 0..n {
                    let t = sample_point(l_iv, k, n);
                    let exp_t = float_to_term(tm, t.exp())?;
                    let t_term = float_to_term(tm, t)?;
                    let l_minus_t = tm.mk_sub(l_term, t_term);
                    let slope_part = tm.mk_mul([exp_t, l_minus_t]);
                    let rhs = tm.mk_add([exp_t, slope_part]);
                    let tangent = tm.mk_ge(ex, rhs);
                    solver.assert(tangent, tm);
                }
            }

            // ---- ln(right) relaxation (ln is concave) ----
            let ln_lo = float_to_term(tm, r_iv.lo.ln())?;
            let ln_hi = float_to_term(tm, r_iv.hi.ln())?;
            let bound_lo = tm.mk_ge(lg, ln_lo);
            let bound_hi = tm.mk_le(lg, ln_hi);
            solver.assert(bound_lo, tm);
            solver.assert(bound_hi, tm);

            if r_iv.width() > 0.0 {
                // Secant (lower): lg ≥ ln(c) + slope·(r − c).
                let slope = (r_iv.hi.ln() - r_iv.lo.ln()) / r_iv.width();
                let slope_term = float_to_term(tm, slope)?;
                let c = float_to_term(tm, r_iv.lo)?;
                let diff = tm.mk_sub(r_term, c);
                let prod = tm.mk_mul([slope_term, diff]);
                let rhs = tm.mk_add([ln_lo, prod]);
                let secant = tm.mk_ge(lg, rhs);
                solver.assert(secant, tm);

                // Tangents (upper): lg ≤ ln(t) + (1/t)·(r − t).
                for k in 0..n {
                    let t = sample_point(r_iv, k, n);
                    if t <= 0.0 {
                        continue;
                    }
                    let ln_t = float_to_term(tm, t.ln())?;
                    let inv_t = float_to_term(tm, 1.0 / t)?;
                    let t_term = float_to_term(tm, t)?;
                    let r_minus_t = tm.mk_sub(r_term, t_term);
                    let slope_part = tm.mk_mul([inv_t, r_minus_t]);
                    let rhs = tm.mk_add([ln_t, slope_part]);
                    let tangent = tm.mk_le(lg, rhs);
                    solver.assert(tangent, tm);
                }
            }

            let result = tm.mk_sub(ex, lg);
            let result_iv =
                Interval::new(l_iv.lo.exp() - r_iv.hi.ln(), l_iv.hi.exp() - r_iv.lo.ln());
            Some((result, result_iv))
        }
    }
}

/// The `k`-th of `n` evenly spaced tangent points across `iv`.
fn sample_point(iv: Interval, k: usize, n: usize) -> f64 {
    if n == 1 {
        iv.midpoint()
    } else {
        iv.lo + (iv.width() * k as f64) / ((n - 1) as f64)
    }
}

/// Run `check()` and, on `Sat`, project the model onto the free variables.
///
/// **Must be called before the enclosing `pop()`** — OxiZ's `pop()` does not
/// clear `model`, but the model refers to terms whose SAT variables the pop
/// invalidates, so reading it afterwards would be meaningless.
pub(super) fn check_and_extract(
    var_terms: &[TermId],
    domain: &IntervalDomain,
    tm: &mut TermManager,
    solver: &mut Solver,
) -> OxizVerdict {
    match solver.check(tm) {
        SolverResult::Unsat => OxizVerdict::Unsat,
        SolverResult::Unknown => OxizVerdict::Unknown,
        SolverResult::Sat => {
            use oxiz::core::TermKind;
            let seed: Vec<f64> = match solver.model() {
                Some(model) => var_terms
                    .iter()
                    .enumerate()
                    .map(|(i, &var_id)| {
                        model
                            .get(var_id)
                            .and_then(|val_id| tm.get(val_id))
                            .and_then(|term| match term.kind {
                                TermKind::RealConst(ref r) => {
                                    let v = (*r.numer() as f64) / (*r.denom() as f64);
                                    v.is_finite().then_some(v)
                                }
                                _ => None,
                            })
                            .unwrap_or_else(|| domain.vars[i].midpoint())
                    })
                    .collect(),
                None => domain.vars.iter().map(Interval::midpoint).collect(),
            };
            OxizVerdict::Sat(seed)
        }
    }
}
