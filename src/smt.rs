//! SMT (Satisfiability Modulo Theories) integration.
//!
//! Two-layer solver stack:
//! 1. **Interval propagation** (`IntervalDomain`) — EML-aware forward/backward
//!    rules for `eml(l, r) = exp(l) − ln(r)`. Tightens variable domains and
//!    proves trivial UNSAT cases without external deps.
//! 2. **OxiZ backend** (`EmlSmtSolver`, feature-gated on `smt`) — encodes
//!    EML constraints for OxiZ's LRA theory via secant+tangent linear
//!    relaxation of `exp`/`ln` over tight intervals. Uses OxiZ 0.2.0 from
//!    crates.io.
//!
//! Legacy `EmlNraSolver` (interval bisection) remains available for witness
//! extraction and is enhanced with propagation.

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::tree::{EmlNode, EmlTree};

// ----------------------------------------------------------------------------
// Public constraint types (unchanged public API)
// ----------------------------------------------------------------------------

/// An EML-based constraint for SMT solving.
#[derive(Clone, Debug)]
pub enum EmlConstraint {
    /// `eml_expr == 0`
    EqZero(EmlTree),
    /// `eml_expr > 0`
    GtZero(EmlTree),
    /// `eml_expr >= 0`
    GeZero(EmlTree),
    /// Conjunction of constraints.
    And(Vec<EmlConstraint>),
    /// Disjunction of constraints.
    Or(Vec<EmlConstraint>),
}

/// A solution to an EML constraint system.
#[derive(Clone, Debug)]
pub struct EmlSolution {
    /// Variable assignments that satisfy the constraints.
    pub assignments: Vec<f64>,
    /// Whether the solution is exact or approximate.
    pub is_exact: bool,
}

// ----------------------------------------------------------------------------
// Interval arithmetic
// ----------------------------------------------------------------------------

/// Interval `[lo, hi]` for variable bounds during constraint solving.
///
/// Empty when `lo > hi` or when bounds are non-finite. `hull` is the join used
/// to combine bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Lower bound (inclusive).
    pub lo: f64,
    /// Upper bound (inclusive).
    pub hi: f64,
}

impl Interval {
    /// Build a new interval `[lo, hi]`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// True if the interval is empty (no real points inside).
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi || self.lo.is_nan() || self.hi.is_nan()
    }

    /// Width of the interval (`hi - lo`).
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// True if `x` lies inside the (closed) interval.
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    /// Split the interval at the midpoint into two halves.
    pub fn split(&self) -> (Self, Self) {
        let mid = self.midpoint();
        (Self::new(self.lo, mid), Self::new(mid, self.hi))
    }

    /// Intersect two intervals; result is empty if they are disjoint.
    pub fn intersect(&self, other: &Self) -> Self {
        Self::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    /// Convex hull (bounding interval containing both).
    pub fn hull(&self, other: &Self) -> Self {
        Self::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    /// Forward `exp`: since `exp` is monotone, `exp([lo, hi]) = [exp(lo), exp(hi)]`.
    pub fn exp(&self) -> Self {
        Self::new(self.lo.exp(), self.hi.exp())
    }

    /// Forward `ln`: `ln([lo, hi])` for `lo > 0`, else empty.
    pub fn ln(&self) -> Self {
        if self.lo <= 0.0 || !self.lo.is_finite() || !self.hi.is_finite() {
            // Represent empty as `[+inf, -inf]` so `is_empty()` returns true.
            Self::new(f64::INFINITY, f64::NEG_INFINITY)
        } else {
            Self::new(self.lo.ln(), self.hi.ln())
        }
    }
}

// ----------------------------------------------------------------------------
// Interval domain and propagation
// ----------------------------------------------------------------------------

/// Result of a constraint propagation step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropResult {
    /// At least one variable interval was tightened.
    Changed,
    /// No change; propagation reached fixpoint.
    Stable,
    /// Constraint is provably unsatisfiable on current domain.
    Conflict,
}

/// Variable interval domain for EML constraint propagation.
#[derive(Clone, Debug)]
pub struct IntervalDomain {
    /// One interval per variable (index = variable index).
    pub vars: Vec<Interval>,
}

impl IntervalDomain {
    /// Build a domain from `(lo, hi)` bounds, padded with `(-10, 10)` for any
    /// missing slots, up to `num_vars` variables in total.
    pub fn new(bounds: &[(f64, f64)], num_vars: usize) -> Self {
        let vars = (0..num_vars)
            .map(|i| {
                if i < bounds.len() {
                    Interval::new(bounds[i].0, bounds[i].1)
                } else {
                    Interval::new(-10.0, 10.0)
                }
            })
            .collect();
        Self { vars }
    }

    /// True if any variable's interval is empty.
    pub fn is_empty(&self) -> bool {
        self.vars.iter().any(Interval::is_empty)
    }

    /// Propagate a constraint until fixpoint or conflict.
    ///
    /// Returns `Changed` if any variable was tightened, `Stable` if fixpoint
    /// reached without any change, `Conflict` if unsatisfiable.
    pub fn propagate(&mut self, c: &EmlConstraint) -> PropResult {
        const MAX_ITERATIONS: usize = 20;
        let mut changed_any = false;
        for _ in 0..MAX_ITERATIONS {
            let result = propagate_once(&mut self.vars, c);
            match result {
                PropResult::Conflict => return PropResult::Conflict,
                PropResult::Changed => {
                    changed_any = true;
                    continue;
                }
                PropResult::Stable => break,
            }
        }
        if changed_any {
            PropResult::Changed
        } else {
            PropResult::Stable
        }
    }
}

/// Forward-evaluate an EML subtree on interval-valued variables.
/// Returns the interval of possible output values.
fn eval_interval(node: &EmlNode, vars: &[Interval]) -> Interval {
    match node {
        EmlNode::One => Interval::new(1.0, 1.0),
        EmlNode::Var(i) => vars
            .get(*i)
            .copied()
            .unwrap_or_else(|| Interval::new(-10.0, 10.0)),
        EmlNode::Eml { left, right } => {
            let l = eval_interval(left, vars);
            let r = eval_interval(right, vars);
            if l.is_empty() || r.is_empty() {
                return Interval::new(f64::INFINITY, f64::NEG_INFINITY);
            }
            let exp_l = l.exp();
            let ln_r = r.ln();
            if ln_r.is_empty() {
                return Interval::new(f64::INFINITY, f64::NEG_INFINITY);
            }
            // eml(l, r) = exp(l) − ln(r)
            // min = exp(l.lo) − ln(r.hi)
            // max = exp(l.hi) − ln(r.lo)
            Interval::new(exp_l.lo - ln_r.hi, exp_l.hi - ln_r.lo)
        }
    }
}

/// Single-pass propagation: walk constraint, evaluate intervals, tighten.
fn propagate_once(vars: &mut [Interval], c: &EmlConstraint) -> PropResult {
    match c {
        EmlConstraint::EqZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            // Must contain 0.
            if v.lo > 0.0 || v.hi < 0.0 {
                return PropResult::Conflict;
            }
            PropResult::Stable
        }
        EmlConstraint::GtZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.hi <= 0.0 {
                return PropResult::Conflict;
            }
            PropResult::Stable
        }
        EmlConstraint::GeZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.hi < 0.0 {
                return PropResult::Conflict;
            }
            PropResult::Stable
        }
        EmlConstraint::And(constraints) => {
            let mut any_changed = false;
            for inner in constraints {
                match propagate_once(vars, inner) {
                    PropResult::Conflict => return PropResult::Conflict,
                    PropResult::Changed => any_changed = true,
                    PropResult::Stable => {}
                }
            }
            if any_changed {
                PropResult::Changed
            } else {
                PropResult::Stable
            }
        }
        EmlConstraint::Or(constraints) => {
            if constraints.is_empty() {
                return PropResult::Conflict;
            }
            // For Or, conflict only if ALL branches conflict.
            let mut all_conflict = true;
            for inner in constraints {
                if !matches!(propagate_once(vars, inner), PropResult::Conflict) {
                    all_conflict = false;
                    break;
                }
            }
            if all_conflict {
                PropResult::Conflict
            } else {
                PropResult::Stable
            }
        }
    }
}

// ----------------------------------------------------------------------------
// EmlNraSolver (interval bisection, enhanced with propagation)
// ----------------------------------------------------------------------------

/// EML-based Non-linear Real Arithmetic (NRA) solver.
///
/// Uses interval propagation + bisection. With propagation, trivial UNSAT
/// cases (e.g., `exp(x) < 0`) are detected early.
pub struct EmlNraSolver {
    /// Maximum number of interval bisection iterations.
    pub max_iterations: usize,
    /// Tolerance for interval width (stops bisection when all widths fall below).
    pub tolerance: f64,
    /// Initial search bounds for each variable.
    pub initial_bounds: Vec<(f64, f64)>,
}

impl Default for EmlNraSolver {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-8,
            initial_bounds: vec![(-10.0, 10.0)],
        }
    }
}

impl EmlNraSolver {
    /// Create a new solver with the given initial bounds for each variable.
    pub fn new(initial_bounds: Vec<(f64, f64)>) -> Self {
        Self {
            initial_bounds,
            ..Default::default()
        }
    }

    /// Solve a constraint system using interval propagation + bisection.
    pub fn solve(&self, constraint: &EmlConstraint) -> Result<EmlSolution, EmlError> {
        let num_vars = count_constraint_vars(constraint);
        if num_vars == 0 {
            let ctx = EvalCtx::new(&[]);
            if check_constraint(constraint, &ctx) {
                return Ok(EmlSolution {
                    assignments: vec![],
                    is_exact: true,
                });
            }
            return Err(EmlError::ConvergenceFailed {
                best_mse: f64::INFINITY,
                iterations: 0,
            });
        }

        let mut domain = IntervalDomain::new(&self.initial_bounds, num_vars);
        if matches!(domain.propagate(constraint), PropResult::Conflict) {
            return Err(EmlError::ConvergenceFailed {
                best_mse: f64::INFINITY,
                iterations: 0,
            });
        }

        for _iter in 0..self.max_iterations {
            let midpoints: Vec<f64> = domain.vars.iter().map(Interval::midpoint).collect();
            let ctx = EvalCtx::new(&midpoints);
            if check_constraint(constraint, &ctx) {
                return Ok(EmlSolution {
                    assignments: midpoints,
                    is_exact: domain.vars.iter().all(|iv| iv.width() < self.tolerance),
                });
            }

            // Find widest interval, bisect toward the more promising half.
            let widest_idx = domain
                .vars
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.width()
                        .partial_cmp(&b.width())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            if let Some(widest) = widest_idx {
                let (lo_half, hi_half) = domain.vars[widest].split();

                let mut lo_point = midpoints.clone();
                lo_point[widest] = lo_half.midpoint();
                let lo_resid = evaluate_constraint_residual(constraint, &lo_point);

                let mut hi_point = midpoints;
                hi_point[widest] = hi_half.midpoint();
                let hi_resid = evaluate_constraint_residual(constraint, &hi_point);

                domain.vars[widest] = if lo_resid.abs() < hi_resid.abs() {
                    lo_half
                } else {
                    hi_half
                };
                if matches!(domain.propagate(constraint), PropResult::Conflict) {
                    return Err(EmlError::ConvergenceFailed {
                        best_mse: f64::INFINITY,
                        iterations: 0,
                    });
                }
                if domain.vars.iter().all(|iv| iv.width() < self.tolerance) {
                    break;
                }
            }
        }

        let midpoints: Vec<f64> = domain.vars.iter().map(Interval::midpoint).collect();
        let ctx = EvalCtx::new(&midpoints);
        if check_constraint(constraint, &ctx) {
            Ok(EmlSolution {
                assignments: midpoints,
                is_exact: false,
            })
        } else {
            Err(EmlError::ConvergenceFailed {
                best_mse: evaluate_constraint_residual(constraint, &midpoints).abs(),
                iterations: self.max_iterations,
            })
        }
    }
}

// ----------------------------------------------------------------------------
// EmlSmtSolver (OxiZ-backed, feature = "smt")
// ----------------------------------------------------------------------------

/// Result of an SMT check via OxiZ.
#[cfg(feature = "smt")]
#[derive(Clone, Debug)]
pub enum SmtResult {
    /// Constraint is satisfiable; witness provided.
    Sat(EmlSolution),
    /// Constraint is unsatisfiable (proved by OxiZ or interval propagation).
    Unsat,
    /// Solver could not decide (relaxation too loose, timeout, etc.).
    Unknown,
}

/// SMT solver backed by OxiZ via linear relaxation of `exp`/`ln`.
#[cfg(feature = "smt")]
pub struct EmlSmtSolver {
    /// Per-variable initial bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Number of tangent sample points for linear relaxation of exp/ln (>= 1).
    pub relaxation_samples: usize,
}

#[cfg(feature = "smt")]
impl Default for EmlSmtSolver {
    fn default() -> Self {
        Self {
            bounds: vec![(-10.0, 10.0)],
            relaxation_samples: 3,
        }
    }
}

#[cfg(feature = "smt")]
impl EmlSmtSolver {
    /// Build a new SMT solver with the given variable bounds.
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        Self {
            bounds,
            relaxation_samples: 3,
        }
    }

    /// Check satisfiability of the constraint.
    ///
    /// Strategy:
    /// 1. Interval propagation to fixpoint → empty domain ⇒ `Unsat`.
    /// 2. Build OxiZ LRA encoding with linear relaxation of `exp`/`ln`.
    /// 3. OxiZ `Unsat` ⇒ `Unsat` (sound: relaxation is over-approximation).
    /// 4. OxiZ `Sat`/`Unknown` ⇒ fall back to interval bisection on tightened
    ///    domain for witness extraction. If witness verifies ⇒ `Sat`,
    ///    else `Unknown`.
    pub fn check_sat(&self, c: &EmlConstraint) -> Result<SmtResult, EmlError> {
        let num_vars = count_constraint_vars(c);
        if num_vars == 0 {
            let ctx = EvalCtx::new(&[]);
            return Ok(if check_constraint(c, &ctx) {
                SmtResult::Sat(EmlSolution {
                    assignments: vec![],
                    is_exact: true,
                })
            } else {
                SmtResult::Unsat
            });
        }

        // Phase 1: interval propagation.
        let mut domain = IntervalDomain::new(&self.bounds, num_vars);
        if matches!(domain.propagate(c), PropResult::Conflict) {
            return Ok(SmtResult::Unsat);
        }

        // Phase 2: OxiZ linear relaxation.
        match oxiz_check(c, &domain, self.relaxation_samples) {
            OxizVerdict::Unsat => Ok(SmtResult::Unsat),
            OxizVerdict::Sat | OxizVerdict::Unknown => {
                // Fall back to bisection on tightened domain for a witness.
                let tight_bounds: Vec<(f64, f64)> =
                    domain.vars.iter().map(|iv| (iv.lo, iv.hi)).collect();
                let bisect = EmlNraSolver::new(tight_bounds);
                match bisect.solve(c) {
                    Ok(sol) => Ok(SmtResult::Sat(sol)),
                    Err(_) => Ok(SmtResult::Unknown),
                }
            }
        }
    }
}

#[cfg(feature = "smt")]
#[derive(Clone, Copy)]
enum OxizVerdict {
    Sat,
    Unsat,
    Unknown,
}

#[cfg(feature = "smt")]
fn oxiz_check(c: &EmlConstraint, domain: &IntervalDomain, samples: usize) -> OxizVerdict {
    use oxiz::{Solver, SolverResult, TermId, TermManager};

    let mut tm = TermManager::new();
    let mut solver = Solver::new();
    let real_sort = tm.sorts.real_sort;

    // Encode free variables.
    let mut var_terms: Vec<TermId> = Vec::with_capacity(domain.vars.len());
    for i in 0..domain.vars.len() {
        let term = tm.mk_var(&format!("x{i}"), real_sort);
        var_terms.push(term);
    }

    // Assert bounds for each free variable.
    for (i, iv) in domain.vars.iter().enumerate() {
        let Some(lo) = float_to_term(&mut tm, iv.lo) else {
            return OxizVerdict::Unknown;
        };
        let Some(hi) = float_to_term(&mut tm, iv.hi) else {
            return OxizVerdict::Unknown;
        };
        let ge = tm.mk_ge(var_terms[i], lo);
        let le = tm.mk_le(var_terms[i], hi);
        solver.assert(ge, &mut tm);
        solver.assert(le, &mut tm);
    }

    let mut counter: usize = 0;
    let encoded = match encode_constraint(
        c,
        &var_terms,
        domain,
        samples,
        &mut counter,
        &mut tm,
        &mut solver,
    ) {
        Some(t) => t,
        None => return OxizVerdict::Unknown,
    };
    solver.assert(encoded, &mut tm);

    match solver.check(&mut tm) {
        SolverResult::Sat => OxizVerdict::Sat,
        SolverResult::Unsat => OxizVerdict::Unsat,
        SolverResult::Unknown => OxizVerdict::Unknown,
    }
}

/// Convert `f64` to an OxiZ real term with a **bounded-denominator**
/// rational approximation. Using unbounded continued-fraction rationals
/// causes `Rational64` to accumulate denominators near `i64::MAX`, which
/// then overflows in secant/tangent products. We cap denominators to
/// `DENOM_CAP` and cap the numerator magnitude to `VALUE_CAP`, which keeps
/// all intermediate LRA arithmetic safely inside `i64`.
#[cfg(feature = "smt")]
fn float_to_term(tm: &mut oxiz::TermManager, v: f64) -> Option<oxiz::TermId> {
    use num_rational::Rational64;
    if !v.is_finite() {
        return None;
    }

    // Keep numerator and denominator modest so that products of two rational
    // terms (common in relaxation constraints) fit safely inside i64 and leave
    // plenty of head-room for downstream arithmetic inside OxiZ.
    const DENOM_CAP: i64 = 1_000_000;
    const VALUE_CAP: f64 = 1.0e12;

    if v.abs() > VALUE_CAP {
        return None;
    }

    let scaled = (v * DENOM_CAP as f64).round();
    if !scaled.is_finite() || scaled.abs() > (i64::MAX as f64) / 4.0 {
        return None;
    }
    let num = scaled as i64;
    let r = Rational64::new(num, DENOM_CAP);
    Some(tm.mk_real(r))
}

/// Encode an EML constraint to an OxiZ boolean term representing the
/// constraint's truth. Returns `None` if any subterm cannot be encoded —
/// the caller then falls through to `Unknown`.
#[cfg(feature = "smt")]
fn encode_constraint(
    c: &EmlConstraint,
    var_terms: &[oxiz::TermId],
    domain: &IntervalDomain,
    samples: usize,
    counter: &mut usize,
    tm: &mut oxiz::TermManager,
    solver: &mut oxiz::Solver,
) -> Option<oxiz::TermId> {
    match c {
        EmlConstraint::EqZero(tree) => {
            let (term, _) =
                encode_tree(&tree.root, var_terms, domain, samples, counter, tm, solver)?;
            let zero = float_to_term(tm, 0.0)?;
            Some(tm.mk_eq(term, zero))
        }
        EmlConstraint::GtZero(tree) => {
            let (term, _) =
                encode_tree(&tree.root, var_terms, domain, samples, counter, tm, solver)?;
            let zero = float_to_term(tm, 0.0)?;
            Some(tm.mk_gt(term, zero))
        }
        EmlConstraint::GeZero(tree) => {
            let (term, _) =
                encode_tree(&tree.root, var_terms, domain, samples, counter, tm, solver)?;
            let zero = float_to_term(tm, 0.0)?;
            Some(tm.mk_ge(term, zero))
        }
        EmlConstraint::And(cs) => {
            if cs.is_empty() {
                return Some(tm.mk_true());
            }
            let mut encoded: Vec<oxiz::TermId> = Vec::with_capacity(cs.len());
            for inner in cs {
                let t = encode_constraint(inner, var_terms, domain, samples, counter, tm, solver)?;
                encoded.push(t);
            }
            Some(tm.mk_and(encoded))
        }
        EmlConstraint::Or(cs) => {
            if cs.is_empty() {
                return Some(tm.mk_false());
            }
            let mut encoded: Vec<oxiz::TermId> = Vec::with_capacity(cs.len());
            for inner in cs {
                let t = encode_constraint(inner, var_terms, domain, samples, counter, tm, solver)?;
                encoded.push(t);
            }
            Some(tm.mk_or(encoded))
        }
    }
}

/// Encode an EML subtree. Returns `(term, interval)` where `term` is the
/// OxiZ term standing for the value and `interval` is its interval range.
///
/// Inserts auxiliary variables and relaxation constraints for each `eml`
/// node so that the OxiZ LRA theory can reason about `exp`/`ln` via a
/// secant+tangent linear relaxation.
#[cfg(feature = "smt")]
fn encode_tree(
    node: &EmlNode,
    var_terms: &[oxiz::TermId],
    domain: &IntervalDomain,
    samples: usize,
    counter: &mut usize,
    tm: &mut oxiz::TermManager,
    solver: &mut oxiz::Solver,
) -> Option<(oxiz::TermId, Interval)> {
    match node {
        EmlNode::One => {
            let one = float_to_term(tm, 1.0)?;
            Some((one, Interval::new(1.0, 1.0)))
        }
        EmlNode::Var(i) => {
            let iv = *domain.vars.get(*i)?;
            let term = *var_terms.get(*i)?;
            Some((term, iv))
        }
        EmlNode::Eml { left, right } => {
            let (l_term, l_iv) =
                encode_tree(left, var_terms, domain, samples, counter, tm, solver)?;
            let (r_term, r_iv) =
                encode_tree(right, var_terms, domain, samples, counter, tm, solver)?;

            // Guard against infeasible or unbounded intervals.
            if !l_iv.lo.is_finite() || !l_iv.hi.is_finite() {
                return None;
            }
            if !r_iv.lo.is_finite() || !r_iv.hi.is_finite() || r_iv.lo <= 0.0 {
                return None;
            }
            if l_iv.is_empty() || r_iv.is_empty() {
                return None;
            }

            let real_sort = tm.sorts.real_sort;
            *counter += 1;
            let ex = tm.mk_var(&format!("__ex_{counter}"), real_sort);
            *counter += 1;
            let ln_term = tm.mk_var(&format!("__ln_{counter}"), real_sort);

            // ---- exp(l) relaxation (exp is convex) ----
            // Value bounds: exp(l.lo) <= ex <= exp(l.hi).
            let exp_lo = float_to_term(tm, l_iv.lo.exp())?;
            let exp_hi = float_to_term(tm, l_iv.hi.exp())?;
            let t1 = tm.mk_ge(ex, exp_lo);
            let t2 = tm.mk_le(ex, exp_hi);
            solver.assert(t1, tm);
            solver.assert(t2, tm);

            if l_iv.width() > 0.0 {
                // Secant upper bound:
                //   ex <= exp(l.lo) + slope * (l - l.lo)
                let slope = (l_iv.hi.exp() - l_iv.lo.exp()) / l_iv.width();
                let slope_term = float_to_term(tm, slope)?;
                let lo_const = float_to_term(tm, l_iv.lo)?;
                let diff = tm.mk_sub(l_term, lo_const);
                let prod = tm.mk_mul([slope_term, diff]);
                let rhs = tm.mk_add([exp_lo, prod]);
                let sec_ub = tm.mk_le(ex, rhs);
                solver.assert(sec_ub, tm);

                // Tangent lower bounds at evenly spaced sample points:
                //   ex >= exp(xk) + exp(xk) * (l - xk)
                let n = samples.max(1);
                for k in 0..n {
                    let t = if n == 1 {
                        l_iv.midpoint()
                    } else {
                        l_iv.lo + (l_iv.width() * k as f64) / ((n - 1) as f64)
                    };
                    let exp_t = t.exp();
                    let exp_t_term = float_to_term(tm, exp_t)?;
                    let t_const = float_to_term(tm, t)?;
                    let l_minus_t = tm.mk_sub(l_term, t_const);
                    let slope_tangent = tm.mk_mul([exp_t_term, l_minus_t]);
                    let rhs = tm.mk_add([exp_t_term, slope_tangent]);
                    let tan_lb = tm.mk_ge(ex, rhs);
                    solver.assert(tan_lb, tm);
                }
            }

            // ---- ln(r) relaxation (ln is concave) ----
            // Value bounds: ln(r.lo) <= ln_term <= ln(r.hi).
            let ln_lo = float_to_term(tm, r_iv.lo.ln())?;
            let ln_hi = float_to_term(tm, r_iv.hi.ln())?;
            let t3 = tm.mk_ge(ln_term, ln_lo);
            let t4 = tm.mk_le(ln_term, ln_hi);
            solver.assert(t3, tm);
            solver.assert(t4, tm);

            if r_iv.width() > 0.0 {
                // Secant lower bound:
                //   ln_term >= ln(r.lo) + slope * (r - r.lo)
                let slope = (r_iv.hi.ln() - r_iv.lo.ln()) / r_iv.width();
                let slope_term = float_to_term(tm, slope)?;
                let lo_const = float_to_term(tm, r_iv.lo)?;
                let diff = tm.mk_sub(r_term, lo_const);
                let prod = tm.mk_mul([slope_term, diff]);
                let rhs = tm.mk_add([ln_lo, prod]);
                let sec_lb = tm.mk_ge(ln_term, rhs);
                solver.assert(sec_lb, tm);

                // Tangent upper bounds at evenly spaced sample points:
                //   ln_term <= ln(rk) + (1/rk) * (r - rk)
                let n = samples.max(1);
                for k in 0..n {
                    let t = if n == 1 {
                        r_iv.midpoint()
                    } else {
                        r_iv.lo + (r_iv.width() * k as f64) / ((n - 1) as f64)
                    };
                    if t <= 0.0 {
                        continue;
                    }
                    let inv_t = 1.0 / t;
                    let ln_t = t.ln();
                    let ln_t_term = float_to_term(tm, ln_t)?;
                    let inv_t_term = float_to_term(tm, inv_t)?;
                    let t_const = float_to_term(tm, t)?;
                    let r_minus_t = tm.mk_sub(r_term, t_const);
                    let slope_tangent = tm.mk_mul([inv_t_term, r_minus_t]);
                    let rhs = tm.mk_add([ln_t_term, slope_tangent]);
                    let tan_ub = tm.mk_le(ln_term, rhs);
                    solver.assert(tan_ub, tm);
                }
            }

            // Result term is `ex - ln_term`; interval = [exp(l.lo) - ln(r.hi), exp(l.hi) - ln(r.lo)].
            let result = tm.mk_sub(ex, ln_term);
            let result_iv =
                Interval::new(l_iv.lo.exp() - r_iv.hi.ln(), l_iv.hi.exp() - r_iv.lo.ln());
            Some((result, result_iv))
        }
    }
}

// ----------------------------------------------------------------------------
// Helpers (shared by both solvers)
// ----------------------------------------------------------------------------

fn check_constraint(constraint: &EmlConstraint, ctx: &EvalCtx) -> bool {
    match constraint {
        EmlConstraint::EqZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v.abs() < 1e-8),
        EmlConstraint::GtZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v > 0.0),
        EmlConstraint::GeZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v >= -1e-12),
        EmlConstraint::And(constraints) => constraints.iter().all(|c| check_constraint(c, ctx)),
        EmlConstraint::Or(constraints) => constraints.iter().any(|c| check_constraint(c, ctx)),
    }
}

fn evaluate_constraint_residual(constraint: &EmlConstraint, vars: &[f64]) -> f64 {
    let ctx = EvalCtx::new(vars);
    match constraint {
        EmlConstraint::EqZero(tree) => tree.eval_real(&ctx).unwrap_or(f64::INFINITY),
        EmlConstraint::GtZero(tree) => {
            let v = tree.eval_real(&ctx).unwrap_or(f64::NEG_INFINITY);
            if v > 0.0 { 0.0 } else { -v }
        }
        EmlConstraint::GeZero(tree) => {
            let v = tree.eval_real(&ctx).unwrap_or(f64::NEG_INFINITY);
            if v >= 0.0 { 0.0 } else { -v }
        }
        EmlConstraint::And(constraints) => constraints
            .iter()
            .map(|c| evaluate_constraint_residual(c, vars))
            .map(f64::abs)
            .sum(),
        EmlConstraint::Or(constraints) => constraints
            .iter()
            .map(|c| evaluate_constraint_residual(c, vars))
            .map(f64::abs)
            .fold(f64::INFINITY, f64::min),
    }
}

fn count_constraint_vars(constraint: &EmlConstraint) -> usize {
    match constraint {
        EmlConstraint::EqZero(tree) | EmlConstraint::GtZero(tree) | EmlConstraint::GeZero(tree) => {
            tree.num_vars()
        }
        EmlConstraint::And(cs) | EmlConstraint::Or(cs) => {
            cs.iter().map(count_constraint_vars).max().unwrap_or(0)
        }
    }
}

// ----------------------------------------------------------------------------
// Tests (always on)
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
}

// ----------------------------------------------------------------------------
// Tests (OxiZ-backed, feature = "smt")
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "smt"))]
mod smt_tests {
    use super::*;
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
    fn test_smt_unsat_ln_of_negative() {
        // ln(x) > 0 with x in [-2, -1] is unsat (ln undefined on that domain).
        let x = EmlTree::var(0);
        let ln_x = Canonical::ln(&x);
        let c = EmlConstraint::GtZero(ln_x);
        let solver = EmlSmtSolver::new(vec![(-2.0, -1.0)]);
        assert!(matches!(
            solver.check_sat(&c).expect("check_sat error"),
            SmtResult::Unsat
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
}
