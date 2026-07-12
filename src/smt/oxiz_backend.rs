//! One-shot OxiZ-backed EML SMT solver, and the solving pipeline shared with the
//! incremental backend.

use super::constraint::{EmlConstraint, EmlSolution};
use super::core::UnsatCore;
use super::encode::{self, Encoder, LraOracle, OxizVerdict};
use super::helpers::{check_constraint, count_constraint_vars, decide_exists, decide_forall};
use super::interval::{Interval, IntervalDomain, PropResult};
use super::maxsmt::{MaxSmtOutcome, SoftConstraint};
use super::nra::EmlNraSolver;
use crate::error::EmlError;
use crate::eval::EvalCtx;

// ----------------------------------------------------------------------------
// SmtResult
// ----------------------------------------------------------------------------

/// Result of an SMT check via OxiZ.
#[derive(Clone, Debug)]
pub enum SmtResult {
    /// Constraint is satisfiable; a **verified** witness is provided (the
    /// assignment has been evaluated against the original nonlinear constraint,
    /// not merely against the LRA relaxation).
    Sat(EmlSolution),
    /// Constraint is unsatisfiable. Sound: proved either by interval propagation
    /// or by the LRA over-approximation, both of which only ever report `Unsat`
    /// when the original problem is genuinely infeasible.
    Unsat,
    /// Solver could not decide (relaxation too loose, unencodable subterm,
    /// nested quantifier, ...). Never a guessed verdict.
    Unknown,
}

impl SmtResult {
    /// True when this is [`SmtResult::Unsat`].
    #[must_use]
    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResult::Unsat)
    }

    /// True when this is [`SmtResult::Sat`].
    #[must_use]
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResult::Sat(_))
    }

    /// The witness assignment, when satisfiable.
    #[must_use]
    pub fn witness(&self) -> Option<&[f64]> {
        match self {
            SmtResult::Sat(sol) => Some(&sol.assignments),
            _ => None,
        }
    }
}

// ----------------------------------------------------------------------------
// EmlSmtSolver (one fresh OxiZ Solver per query)
// ----------------------------------------------------------------------------

/// SMT solver backed by OxiZ via linear relaxation of `exp`/`ln`.
///
/// Each [`check_sat`](EmlSmtSolver::check_sat) call builds a fresh `TermManager`
/// and `Solver`. That is the right thing for a one-off query and the reference
/// semantics against which [`IncrementalEmlSolver`](super::IncrementalEmlSolver)
/// is differentially tested; for a hot loop over many candidate topologies, use
/// the incremental solver instead.
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
    #[must_use]
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        Self {
            bounds,
            relaxation_samples: 3,
        }
    }

    /// Check satisfiability of the constraint.
    ///
    /// Strategy:
    /// 1. Top-level quantifiers are discharged by the decision procedures in
    ///    `super::helpers`.
    /// 2. Interval propagation to fixpoint → empty domain ⇒ `Unsat`.
    /// 3. OxiZ LRA encoding with a secant+tangent relaxation of `exp`/`ln`.
    ///    `Unsat` there is sound (the relaxation over-approximates).
    /// 4. `Sat`/`Unknown` from OxiZ ⇒ the model is used only as a *seed*; the
    ///    witness is verified concretely, falling back to interval bisection on
    ///    the tightened domain. A witness that does not verify yields `Unknown`,
    ///    never a fabricated `Sat`.
    ///
    /// # Errors
    /// Propagates any [`EmlError`] raised while evaluating the constraint.
    pub fn check_sat(&self, c: &EmlConstraint) -> Result<SmtResult, EmlError> {
        let mut oracle = OneShotOracle {
            samples: self.relaxation_samples.max(1),
        };
        check_sat_flow(&mut oracle, &self.bounds, c)
    }

    /// Check satisfiability of the conjunction of `constraints`.
    ///
    /// # Errors
    /// Propagates any [`EmlError`] raised while evaluating the constraints.
    pub fn check_all(&self, constraints: &[EmlConstraint]) -> Result<SmtResult, EmlError> {
        match constraints {
            [] => Ok(SmtResult::Sat(EmlSolution {
                assignments: Vec::new(),
                is_exact: true,
            })),
            [single] => self.check_sat(single),
            many => self.check_sat(&EmlConstraint::And(many.to_vec())),
        }
    }

    /// Compute a **minimal** unsatisfiable subset (MUS) of `constraints`.
    ///
    /// Returns `Ok(None)` when the conjunction is not provably unsatisfiable.
    /// See [`super::core`] for the algorithm and its minimality guarantee.
    ///
    /// # Errors
    /// Propagates any [`EmlError`] raised while evaluating the constraints.
    pub fn unsat_core(&self, constraints: &[EmlConstraint]) -> Result<Option<UnsatCore>, EmlError> {
        super::core::minimize_unsat_core(&self.bounds, self.relaxation_samples, constraints)
    }

    /// Maximize the total weight of satisfied soft constraints subject to all
    /// `hard` constraints (MaxSMT).
    ///
    /// See [`super::maxsmt`] for the algorithms: an exact core-guided
    /// implicit-hitting-set search driven by OxiZ's RC2 MaxSAT engine under the
    /// `smt-opt` feature, and an honestly-labelled greedy fallback under base
    /// `smt`. The returned [`MaxSmtOutcome`] always states which guarantee it
    /// carries.
    ///
    /// # Errors
    /// Propagates any [`EmlError`] raised while evaluating the constraints.
    pub fn max_smt(
        &self,
        hard: &[EmlConstraint],
        soft: &[SoftConstraint],
    ) -> Result<MaxSmtOutcome, EmlError> {
        super::maxsmt::max_smt(&self.bounds, self.relaxation_samples, hard, soft)
    }
}

// ----------------------------------------------------------------------------
// The shared solving pipeline
// ----------------------------------------------------------------------------

/// Interval propagation + quantifier handling + LRA query + witness verification.
///
/// Generic over the [`LraOracle`] so that the one-shot and the incremental
/// backends run *literally the same* pipeline and differ only in how the OxiZ
/// `Solver` is obtained. Any verdict disagreement between them is therefore a
/// push/pop hygiene bug and nothing else — which is what makes the differential
/// test in `smt_tests` meaningful.
pub(super) fn check_sat_flow<O: LraOracle>(
    oracle: &mut O,
    bounds: &[(f64, f64)],
    c: &EmlConstraint,
) -> Result<SmtResult, EmlError> {
    let num_vars = count_constraint_vars(c);

    // Top-level quantifiers are decided before any LRA encoding.
    match c {
        EmlConstraint::ForAll { var, lo, hi, body } => {
            use super::helpers::QuantResult;
            return match decide_forall(*var, *lo, *hi, body, num_vars) {
                QuantResult::True => Ok(SmtResult::Sat(EmlSolution {
                    assignments: vec![],
                    is_exact: true,
                })),
                QuantResult::FalseWithCounterexample { counterexample } => {
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
    let mut domain = IntervalDomain::new(bounds, num_vars);
    if domain.propagate(c) == PropResult::Conflict {
        return Ok(SmtResult::Unsat);
    }

    // Phase 2: OxiZ linear relaxation.
    match oracle.lra_check(c, &domain) {
        OxizVerdict::Unsat => Ok(SmtResult::Unsat),
        OxizVerdict::Sat(seed) => {
            // The LRA model assigns the *auxiliary* variables freely within the
            // relaxation envelope, so it need not satisfy the true nonlinear
            // constraint. Soundness comes from verifying the seed explicitly.
            let ctx = EvalCtx::new(&seed);
            if check_constraint(c, &ctx) {
                return Ok(SmtResult::Sat(EmlSolution {
                    assignments: seed,
                    is_exact: true,
                }));
            }
            witness_from_domain(c, &domain)
        }
        OxizVerdict::Unknown => witness_from_domain(c, &domain),
    }
}

/// Try to extract a verified witness from the tightened domain: first the
/// midpoint, then interval bisection. `Unknown` when neither succeeds — an
/// undecided query is *never* upgraded to a verdict.
fn witness_from_domain(c: &EmlConstraint, domain: &IntervalDomain) -> Result<SmtResult, EmlError> {
    let midpoints: Vec<f64> = domain.vars.iter().map(Interval::midpoint).collect();
    let ctx = EvalCtx::new(&midpoints);
    if check_constraint(c, &ctx) {
        return Ok(SmtResult::Sat(EmlSolution {
            assignments: midpoints,
            is_exact: true,
        }));
    }
    let tight_bounds: Vec<(f64, f64)> = domain.vars.iter().map(|iv| (iv.lo, iv.hi)).collect();
    let bisect = EmlNraSolver::new(tight_bounds);
    match bisect.solve(c) {
        Ok(sol) => Ok(SmtResult::Sat(sol)),
        Err(_) => Ok(SmtResult::Unknown),
    }
}

// ----------------------------------------------------------------------------
// One-shot oracle: a brand-new TermManager + Solver per query
// ----------------------------------------------------------------------------

/// [`LraOracle`] that constructs a fresh OxiZ `Solver` for every query.
pub(super) struct OneShotOracle {
    /// Tangent sample count for the relaxation.
    pub(super) samples: usize,
}

impl LraOracle for OneShotOracle {
    fn lra_check(&mut self, c: &EmlConstraint, domain: &IntervalDomain) -> OxizVerdict {
        use oxiz::{Solver, TermManager};

        let mut term_manager = TermManager::new();
        let mut solver = Solver::new();
        let mut aux_counter: u64 = 0;

        let var_terms = encode::declare_vars(&mut term_manager, domain.vars.len(), 0);
        if !encode::assert_var_bounds(&var_terms, domain, &mut term_manager, &mut solver) {
            return OxizVerdict::Unknown;
        }

        let mut enc = Encoder {
            var_terms: &var_terms,
            domain,
            samples: self.samples,
            aux_counter: &mut aux_counter,
        };
        let Some(term) = encode::encode_constraint(c, &mut enc, &mut term_manager, &mut solver)
        else {
            return OxizVerdict::Unknown;
        };
        solver.assert(term, &mut term_manager);

        encode::check_and_extract(&var_terms, domain, &mut term_manager, &mut solver)
    }
}
