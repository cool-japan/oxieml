//! Minimal unsatisfiable subsets (unsat cores) over EML constraints.
//!
//! # Why not OxiZ's core?
//!
//! OxiZ 0.2.3 *has* `Solver::get_unsat_core` / `Solver::minimize_unsat_core`, but
//! neither is usable here, for two independent reasons:
//!
//! * **Granularity.** A single [`EmlConstraint`] expands into *dozens* of OxiZ
//!   assertions — one per secant/tangent lemma of every `exp`/`ln` relaxation
//!   plus the variable bounds. A core expressed over those assertions names
//!   internal relaxation lemmas, not the user's constraints, and is meaningless
//!   to the caller.
//! * **Quality.** `Solver::build_unsat_core` returns *every* named assertion
//!   verbatim (it has no assumption-based extraction wired up), and
//!   `Solver::minimize_unsat_core` then deletion-minimizes by constructing a
//!   brand-new `Solver` on every trial — which is exactly the solver churn the
//!   incremental backend exists to eliminate.
//!
//! So the core is computed at the **`EmlConstraint` level**, over the EML
//! decision procedure as a whole (interval propagation ∧ LRA relaxation ∧ witness
//! verification), and the deletion trials are driven through one live
//! [`IncrementalEmlSolver`] via `push`/`pop`.
//!
//! # Algorithm
//!
//! Standard deletion-based (a.k.a. destructive) MUS extraction:
//!
//! ```text
//! core ← {0, …, n−1}                       (verified UNSAT first)
//! for i in 0 … n−1:
//!     if i ∉ core: continue
//!     if UNSAT(core \ {i}):  core ← core \ {i}
//! return core
//! ```
//!
//! # Guarantee
//!
//! On return, for **every** `i` in the core, `core \ {i}` is *not* reported
//! `Unsat` by the oracle, while the core itself *is*. The core is therefore
//! **irreducible** (locally minimal): no proper subset of it is provably
//! unsatisfiable. It is not claimed to be a *minimum-cardinality* MUS — that is a
//! Σ₂ᵖ problem and deletion-based extraction does not solve it.
//!
//! # Honesty about incompleteness
//!
//! The EML oracle is a decision procedure only up to `Unknown`. Irreducibility is
//! therefore relative to the oracle: "`core \ {i}` is not provably UNSAT" covers
//! both "it is genuinely SAT" and "the oracle could not decide". This is stated
//! rather than papered over, and [`UnsatCore::is_verified`] records whether every
//! deletion trial came back with a *definite* `Sat` (in which case the core is a
//! true MUS of the original nonlinear problem).

use super::constraint::EmlConstraint;
use super::incremental::IncrementalEmlSolver;
use super::oxiz_backend::SmtResult;
use crate::error::EmlError;

/// A minimal (irreducible) unsatisfiable subset of a constraint list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsatCore {
    /// Indices into the original `constraints` slice, ascending.
    ///
    /// The conjunction of these constraints is unsatisfiable, and removing *any*
    /// one of them makes the remainder no longer provably unsatisfiable.
    pub indices: Vec<usize>,
    /// `true` when every deletion trial returned a definite `Sat` (so the core is
    /// a genuine minimal unsatisfiable subset of the nonlinear problem), `false`
    /// when at least one trial was `Unknown` (so minimality holds only relative
    /// to this incomplete oracle).
    pub verified_minimal: bool,
}

impl UnsatCore {
    /// Number of constraints in the core.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// True when the core is empty (only possible for an empty input).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// True when minimality was established with definite `Sat` verdicts on every
    /// deletion trial, rather than merely "the oracle stopped saying `Unsat`".
    #[must_use]
    pub fn is_verified(&self) -> bool {
        self.verified_minimal
    }
}

/// Extract an irreducible unsatisfiable subset of `constraints`.
///
/// Returns `Ok(None)` when the conjunction of `constraints` is not provably
/// unsatisfiable — either because it is satisfiable or because the oracle
/// returned `Unknown`. A core is only ever produced from a *definite* `Unsat`,
/// so a returned core is always a genuine infeasibility certificate.
///
/// Uses a single live OxiZ `Solver` for all `n + 1` oracle calls.
///
/// # Errors
/// Propagates any [`EmlError`] raised while evaluating the constraints.
pub fn minimize_unsat_core(
    bounds: &[(f64, f64)],
    relaxation_samples: usize,
    constraints: &[EmlConstraint],
) -> Result<Option<UnsatCore>, EmlError> {
    if constraints.is_empty() {
        return Ok(None);
    }

    // One solver for the whole extraction: 1 seed check + n deletion trials.
    let mut solver = IncrementalEmlSolver::with_config(
        bounds.to_vec(),
        relaxation_samples,
        // Never recycle mid-extraction: the constraint count bounds the work.
        0,
    );

    // Step 0: the full set must be *provably* unsatisfiable, else there is no
    // core to speak of. Never start deletion from an unverified candidate — that
    // is how a "core" that is not actually unsatisfiable gets fabricated.
    if !solver.check_all(constraints)?.is_unsat() {
        return Ok(None);
    }

    let mut core: Vec<usize> = (0..constraints.len()).collect();
    let mut all_trials_definite = true;

    let mut position = 0;
    while position < core.len() {
        let candidate: Vec<EmlConstraint> = core
            .iter()
            .enumerate()
            .filter(|(slot, _)| *slot != position)
            .map(|(_, &idx)| constraints[idx].clone())
            .collect();

        match solver.check_all(&candidate)? {
            // Still infeasible without it ⇒ it is not needed. Drop it and re-test
            // whatever slid into this slot.
            SmtResult::Unsat => {
                core.remove(position);
            }
            // Genuinely satisfiable without it ⇒ it is a *necessary* member.
            SmtResult::Sat(_) => {
                position += 1;
            }
            // The oracle could not decide. Keeping the constraint is the sound
            // choice (dropping it could break the core's unsatisfiability), but
            // minimality is then only relative to the oracle — record that.
            SmtResult::Unknown => {
                all_trials_definite = false;
                position += 1;
            }
        }
    }

    debug_assert!(
        !core.is_empty(),
        "a definite Unsat over a non-empty set must leave a non-empty core"
    );

    Ok(Some(UnsatCore {
        indices: core,
        verified_minimal: all_trials_definite,
    }))
}
