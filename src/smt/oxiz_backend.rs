use super::constraint::{EmlConstraint, EmlSolution};
use super::helpers::{check_constraint, count_constraint_vars, decide_exists, decide_forall};
use super::interval::{Interval, IntervalDomain, PropResult};
use super::nra::EmlNraSolver;
use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::tree::EmlNode;

// ----------------------------------------------------------------------------
// EmlSmtSolver (OxiZ-backed, feature = "smt")
// ----------------------------------------------------------------------------

/// Result of an SMT check via OxiZ.
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
pub struct EmlSmtSolver {
    /// Per-variable initial bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Number of tangent sample points for linear relaxation of exp/ln (>= 1).
    pub relaxation_samples: usize,
}

impl Default for EmlSmtSolver {
    fn default() -> Self {
        Self {
            bounds: vec![(-10.0, 10.0)],
            relaxation_samples: 3,
        }
    }
}

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

        // Handle top-level quantifiers before LRA encoding.
        match c {
            EmlConstraint::ForAll { var, lo, hi, body } => {
                use super::helpers::QuantResult;
                return match decide_forall(*var, *lo, *hi, body, num_vars) {
                    QuantResult::True => Ok(SmtResult::Sat(EmlSolution {
                        assignments: vec![],
                        is_exact: true,
                    })),
                    QuantResult::FalseWithCounterexample { counterexample } => {
                        // Verify the counterexample confirms the body is indeed false.
                        let ctx = EvalCtx::new(&counterexample);
                        debug_assert!(
                            !check_constraint(body, &ctx),
                            "counterexample should falsify the body"
                        );
                        Ok(SmtResult::Unsat)
                    }
                    _ => Ok(SmtResult::Unknown),
                };
            }
            EmlConstraint::Exists { var, lo, hi, body } => {
                use super::helpers::QuantResult;
                return match decide_exists(*var, *lo, *hi, body, num_vars) {
                    QuantResult::TrueWithWitness(witness) => Ok(SmtResult::Sat(EmlSolution {
                        assignments: witness,
                        is_exact: false,
                    })),
                    _ => Ok(SmtResult::Unknown),
                };
            }
            _ => {}
        }

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
        if domain.propagate(c) == PropResult::Conflict {
            return Ok(SmtResult::Unsat);
        }

        // Phase 2: OxiZ linear relaxation.
        match oxiz_check(c, &domain, self.relaxation_samples) {
            OxizVerdict::Unsat => Ok(SmtResult::Unsat),
            OxizVerdict::Sat(seed) => {
                // OxiZ models the LRA relaxation (auxiliary vars for exp/ln may not equal the
                // true nonlinear values). The model is used as a starting-point seed for
                // verification; soundness comes from explicit `check_constraint` verification.
                let ctx = EvalCtx::new(&seed);
                if check_constraint(c, &ctx) {
                    return Ok(SmtResult::Sat(EmlSolution {
                        assignments: seed,
                        is_exact: true,
                    }));
                }
                // Seed didn't satisfy — try domain midpoints
                let midpoints: Vec<f64> = domain.vars.iter().map(Interval::midpoint).collect();
                let ctx = EvalCtx::new(&midpoints);
                if check_constraint(c, &ctx) {
                    return Ok(SmtResult::Sat(EmlSolution {
                        assignments: midpoints,
                        is_exact: true,
                    }));
                }
                // Fall back to bisection on tightened domain for a witness.
                let tight_bounds: Vec<(f64, f64)> =
                    domain.vars.iter().map(|iv| (iv.lo, iv.hi)).collect();
                let bisect = EmlNraSolver::new(tight_bounds);
                match bisect.solve(c) {
                    Ok(sol) => Ok(SmtResult::Sat(sol)),
                    Err(_) => Ok(SmtResult::Unknown),
                }
            }
            OxizVerdict::Unknown => {
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

enum OxizVerdict {
    Sat(Vec<f64>),
    Unsat,
    Unknown,
}

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
        SolverResult::Sat => {
            use oxiz::core::TermKind;
            let seed: Vec<f64> = if let Some(model) = solver.model() {
                var_terms
                    .iter()
                    .enumerate()
                    .map(|(i, &var_id)| {
                        model
                            .get(var_id)
                            .and_then(|val_id| tm.get(val_id))
                            .and_then(|term| {
                                if let TermKind::RealConst(ref r) = term.kind {
                                    let v = (*r.numer() as f64) / (*r.denom() as f64);
                                    if v.is_finite() {
                                        return Some(v);
                                    }
                                }
                                None
                            })
                            .unwrap_or_else(|| domain.vars[i].midpoint())
                    })
                    .collect()
            } else {
                domain.vars.iter().map(|iv| iv.midpoint()).collect()
            };
            OxizVerdict::Sat(seed)
        }
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
        EmlConstraint::LtZero(tree) => {
            let (term, _) =
                encode_tree(&tree.root, var_terms, domain, samples, counter, tm, solver)?;
            let zero = float_to_term(tm, 0.0)?;
            Some(tm.mk_lt(term, zero))
        }
        EmlConstraint::LeZero(tree) => {
            let (term, _) =
                encode_tree(&tree.root, var_terms, domain, samples, counter, tm, solver)?;
            let zero = float_to_term(tm, 0.0)?;
            Some(tm.mk_le(term, zero))
        }
        EmlConstraint::NeZero(tree) => {
            let (term, _) =
                encode_tree(&tree.root, var_terms, domain, samples, counter, tm, solver)?;
            let zero = float_to_term(tm, 0.0)?;
            let eq_zero = tm.mk_eq(term, zero);
            Some(tm.mk_not(eq_zero))
        }
        EmlConstraint::Not(inner) => {
            let nnf = (**inner).clone().to_nnf();
            encode_constraint(&nnf, var_terms, domain, samples, counter, tm, solver)
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
        // Quantifiers nested inside other constraints: conservative Unknown path via None.
        EmlConstraint::ForAll { .. } | EmlConstraint::Exists { .. } => None,
    }
}

/// Encode an EML subtree. Returns `(term, interval)` where `term` is the
/// OxiZ term standing for the value and `interval` is its interval range.
///
/// Inserts auxiliary variables and relaxation constraints for each `eml`
/// node so that the OxiZ LRA theory can reason about `exp`/`ln` via a
/// secant+tangent linear relaxation.
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
        EmlNode::Const(v) => {
            let term = float_to_term(tm, *v)?;
            Some((term, Interval::new(*v, *v)))
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
