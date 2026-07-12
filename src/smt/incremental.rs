//! Incremental EML SMT solving: one live OxiZ `Solver`, one query per
//! `push()`/`pop()` scope.
//!
//! [`EmlSmtSolver`](super::EmlSmtSolver) builds a brand-new `TermManager` and
//! `Solver` for every `check_sat` call. In the symbolic-regression pruner that
//! happens once per candidate topology — thousands of times per run — and the
//! construction cost dominates the actual solving. [`IncrementalEmlSolver`] keeps
//! a single `Solver` alive and wraps each query in a `push`/`pop` scope instead.
//!
//! # Solver-state hygiene (the whole ball game)
//!
//! An incremental solver that leaks assertions across `pop()` silently returns
//! *wrong verdicts*: a topology that is satisfiable on its own gets reported
//! `Unsat` because a previous candidate's constraints are still in the assertion
//! stack, and the pruner throws away a viable branch. The contract this type
//! upholds is therefore exactly:
//!
//! > The assertion set that `check()` sees inside scope *k* is precisely the set
//! > a freshly constructed `Solver` would see for query *k*.
//!
//! Three facts are needed to establish it, and OxiZ 0.2.3 only gives us two:
//!
//! 1. **Clauses and assertions are rolled back.** `Solver::pop()` truncates
//!    `assertions`, unwinds its trail (removing `term → SAT var` bindings and
//!    theory constraints), and calls `sat.pop()`, which deletes every clause
//!    registered at that assertion level, and `arith.pop()` / `euf.pop()`, which
//!    undo the simplex bounds and variable allocations made inside the scope. ✔
//!
//! 2. **Nothing is asserted outside a scope.** This type never asserts at
//!    assertion level 0: `push()` happens before the first `assert`, and `pop()`
//!    is unconditional (there is no early return between them). The invariant
//!    `assertion_depth() == 0` therefore holds before and after every query, and
//!    is asserted in `debug_assert!`s and exercised by the push/pop-hygiene test. ✔
//!
//! 3. **No `TermId` is ever asserted in two different scopes.** This one OxiZ
//!    does *not* give us, and it is the subtle trap. `ArithSolver::pop()`
//!    truncates `var_to_term` and lets the simplex undo its `NewVar` allocations,
//!    but it does **not** remove the corresponding entries from its
//!    `term_to_var: HashMap<TermId, VarId>` cache. Re-asserting a `TermId` whose
//!    simplex variable a previous `pop()` destroyed would hit that stale cache
//!    entry and hand the simplex a **dangling variable id**. Because
//!    `TermManager` hash-conses, naively reusing one persistent `x0` term (or any
//!    term built from it, such as `x0 - 3`) across scopes walks straight into it.
//!
//!    We sidestep it by construction: [`encode::declare_vars`](super::encode::declare_vars)
//!    names the variables of scope *k* `x{i}@{k}` and the auxiliary `exp`/`ln`
//!    variables carry a counter that is monotone for the entire lifetime of the
//!    `Solver`. Every `TermId` asserted inside a scope is thus created inside that
//!    scope, so no stale cache lookup is even possible, and OxiZ's incomplete
//!    rollback becomes harmless. `TermManager` growth is bounded by recycling the
//!    solver every [`recycle_after`](IncrementalEmlSolver::with_recycle_after)
//!    scopes.
//!
//! The claim is not merely argued, it is *tested*. `push_pop_verdicts_equal_n_independent_check_sat`
//! runs a randomized battery of 64 topologies, over three different variable
//! boxes, through this type and through N independent `EmlSmtSolver::check_sat`
//! calls, and requires the verdicts to agree exactly;
//! `push_pop_verdicts_are_order_independent` re-runs the battery backwards on a
//! fresh instance and requires the same answers; and `push_pop_stack_hygiene`
//! interleaves a contradiction with a tautology 40 times, so that any assertion
//! surviving a `pop()` would immediately turn the tautology `Unsat`.
//!
//! # Threading
//!
//! OxiZ's `Solver` is not `Sync`, so an `IncrementalEmlSolver` cannot be shared
//! across threads. The symbolic-regression pruner keeps one per thread in a
//! `thread_local!` (see [`crate::symreg::smt_prune`]); no `R1`-style parallel
//! phase is affected.

use super::constraint::EmlConstraint;
use super::encode::{self, Encoder, LraOracle, OxizVerdict};
use super::interval::IntervalDomain;
use super::oxiz_backend::{SmtResult, check_sat_flow};
use crate::error::EmlError;
use oxiz::{Solver, TermManager};
use std::cell::{Cell, RefCell};

thread_local! {
    /// Number of OxiZ `Solver` objects this thread has constructed through
    /// [`IncrementalEmlSolver`]. Used by the reuse tests to *prove* that 50+
    /// topologies are checked against a single solver instance.
    static SOLVER_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
}

/// Number of OxiZ `Solver` objects constructed by [`IncrementalEmlSolver`] on the
/// current thread since the thread started.
///
/// The counter is thread-local because the solvers themselves are (OxiZ's
/// `Solver` is not `Sync`), which also makes it deterministic under a parallel
/// test runner. Take a baseline, run a batch of queries, and compare the delta to
/// verify that the solver is genuinely being reused.
#[must_use]
pub fn solver_construction_count() -> usize {
    SOLVER_CONSTRUCTIONS.with(Cell::get)
}

/// Default number of query scopes after which the underlying `Solver` and
/// `TermManager` are rebuilt to bound memory growth.
///
/// Every scope interns fresh terms (see the module docs), so the `TermManager`
/// grows monotonically while a `Solver` is alive. Recycling caps that. The value
/// is far above the 50-topology reuse requirement, so a pruning sweep over a
/// realistic beam never recycles mid-batch.
pub const DEFAULT_RECYCLE_AFTER: usize = 512;

/// An EML SMT solver that reuses one live OxiZ `Solver` across queries.
///
/// Same verdicts as [`EmlSmtSolver`](super::EmlSmtSolver) — the two share the
/// entire encoding and post-processing pipeline and differ only in solver
/// lifecycle — but without paying for `Solver::new()` per query.
///
/// ```
/// # #[cfg(feature = "smt")] {
/// use oxieml::smt::{EmlConstraint, IncrementalEmlSolver, SmtResult};
/// use oxieml::EmlTree;
///
/// let mut solver = IncrementalEmlSolver::new(vec![(0.5, 4.0)]);
/// // exp(x) - ln(1) = exp(x) > 0 is satisfiable.
/// let c = EmlConstraint::GtZero(EmlTree::eml(&EmlTree::var(0), &EmlTree::one()));
/// assert!(matches!(solver.check_sat(&c), Ok(SmtResult::Sat(_))));
/// // The assertion stack is empty again — every query is self-contained.
/// assert_eq!(solver.assertion_depth(), 0);
/// # }
/// ```
pub struct IncrementalEmlSolver {
    /// Per-variable initial bounds (the global box every query is solved in).
    bounds: Vec<(f64, f64)>,
    /// Number of tangent sample points for the `exp`/`ln` relaxation (≥ 1).
    relaxation_samples: usize,
    /// Rebuild the solver after this many scopes (`0` = never).
    recycle_after: usize,
    /// The live OxiZ term store.
    term_manager: TermManager,
    /// The live OxiZ solver — constructed once, reused via `push`/`pop`.
    solver: Solver,
    /// Variable terms of the scope currently being encoded.
    var_terms: Vec<oxiz::TermId>,
    /// Monotone counter naming this solver's scopes (`x{i}@{scope}`).
    scope_counter: u64,
    /// Monotone counter naming this solver's auxiliary `exp`/`ln` variables.
    aux_counter: u64,
    /// Scopes opened since the current `Solver` was constructed.
    scopes_since_build: usize,
    /// Total scopes opened over this value's whole lifetime.
    scopes_total: usize,
    /// Current `push` depth. Invariant: `0` outside `scoped_check`.
    depth: usize,
}

impl IncrementalEmlSolver {
    /// Build an incremental solver over the given per-variable bounds.
    ///
    /// Variables beyond `bounds.len()` fall back to `(-10, 10)`, matching
    /// [`IntervalDomain::new`].
    #[must_use]
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        Self::with_config(bounds, 3, DEFAULT_RECYCLE_AFTER)
    }

    /// Build an incremental solver with an explicit relaxation-sample count and
    /// recycling interval.
    ///
    /// `relaxation_samples` is clamped to at least 1. `recycle_after == 0`
    /// disables recycling entirely (the `TermManager` then grows without bound —
    /// only appropriate for short-lived solvers).
    #[must_use]
    pub fn with_config(
        bounds: Vec<(f64, f64)>,
        relaxation_samples: usize,
        recycle_after: usize,
    ) -> Self {
        SOLVER_CONSTRUCTIONS.with(|c| c.set(c.get() + 1));
        Self {
            bounds,
            relaxation_samples: relaxation_samples.max(1),
            recycle_after,
            term_manager: TermManager::new(),
            solver: Solver::new(),
            var_terms: Vec::new(),
            scope_counter: 0,
            aux_counter: 0,
            scopes_since_build: 0,
            scopes_total: 0,
            depth: 0,
        }
    }

    /// Set the number of scopes after which the solver is recycled.
    #[must_use]
    pub fn with_recycle_after(mut self, recycle_after: usize) -> Self {
        self.recycle_after = recycle_after;
        self
    }

    /// The global variable box this solver was built with.
    #[must_use]
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Current `push`/`pop` depth of the underlying OxiZ assertion stack.
    ///
    /// **Always `0`** between queries: every `check_sat` pops exactly the scopes
    /// it pushed. A non-zero value observed from the outside would mean an
    /// assertion leak, which is precisely the bug this type exists to not have.
    #[must_use]
    pub fn assertion_depth(&self) -> usize {
        self.depth
    }

    /// Number of OxiZ `push`/`pop` scopes this solver has opened.
    ///
    /// This is *not* the number of queries answered: a query that interval
    /// propagation refutes outright (or one with no free variables) is decided
    /// before the LRA layer is reached and never opens a scope. It is therefore
    /// also a useful measure of how much work the cheap layer is saving.
    #[must_use]
    pub fn lra_scopes(&self) -> usize {
        self.scopes_total
    }

    /// Check satisfiability of `c`, reusing the live solver.
    ///
    /// Verdict semantics are identical to [`EmlSmtSolver::check_sat`](super::EmlSmtSolver::check_sat):
    /// `Unsat` is sound (proved by interval propagation or by the LRA
    /// over-approximation), every `Sat` carries a witness that has been verified
    /// against the *original* nonlinear constraint, and anything undecidable is
    /// `Unknown` — never a guessed verdict.
    ///
    /// # Errors
    /// Propagates any [`EmlError`] raised while evaluating the constraint during
    /// witness verification.
    pub fn check_sat(&mut self, c: &EmlConstraint) -> Result<SmtResult, EmlError> {
        debug_assert_eq!(
            self.depth, 0,
            "assertion stack must be empty between queries"
        );
        let bounds = std::mem::take(&mut self.bounds);
        let result = check_sat_flow(self, &bounds, c);
        self.bounds = bounds;
        debug_assert_eq!(self.depth, 0, "every push must be matched by a pop");
        result
    }

    /// Check satisfiability of the conjunction of `constraints`.
    ///
    /// An empty slice is trivially satisfiable.
    ///
    /// # Errors
    /// Propagates any [`EmlError`] raised during witness verification.
    pub fn check_all(&mut self, constraints: &[EmlConstraint]) -> Result<SmtResult, EmlError> {
        match constraints {
            [] => Ok(SmtResult::Sat(super::constraint::EmlSolution {
                assignments: Vec::new(),
                is_exact: true,
            })),
            [single] => self.check_sat(single),
            many => self.check_sat(&EmlConstraint::And(many.to_vec())),
        }
    }

    /// Rebuild the underlying `Solver` and `TermManager` from scratch.
    ///
    /// Reclaims the memory held by the hash-consed terms of past scopes. Bumps
    /// [`solver_construction_count`].
    fn rebuild(&mut self) {
        SOLVER_CONSTRUCTIONS.with(|c| c.set(c.get() + 1));
        self.term_manager = TermManager::new();
        self.solver = Solver::new();
        self.var_terms.clear();
        self.aux_counter = 0;
        self.scopes_since_build = 0;
        self.depth = 0;
    }
}

impl LraOracle for IncrementalEmlSolver {
    fn lra_check(&mut self, c: &EmlConstraint, domain: &IntervalDomain) -> OxizVerdict {
        if self.recycle_after != 0 && self.scopes_since_build >= self.recycle_after {
            self.rebuild();
        }
        self.scopes_since_build += 1;
        self.scopes_total += 1;
        self.scope_counter += 1;

        // ---- open the scope ----
        self.solver.push();
        self.depth += 1;

        // Declare this scope's variables. Fresh names ⇒ fresh `TermId`s ⇒ no
        // term is ever asserted in two scopes (see the module soundness note).
        self.var_terms = encode::declare_vars(
            &mut self.term_manager,
            domain.vars.len(),
            self.scope_counter,
        );

        // ---- encode + assert, all strictly inside the scope ----
        // Every path from here to the `pop()` below must fall through, never
        // return early, or the assertion stack would leak.
        let verdict = {
            let Self {
                term_manager,
                solver,
                var_terms,
                aux_counter,
                relaxation_samples,
                ..
            } = self;

            if encode::assert_var_bounds(var_terms, domain, term_manager, solver) {
                let mut enc = Encoder {
                    var_terms,
                    domain,
                    samples: *relaxation_samples,
                    aux_counter,
                };
                match encode::encode_constraint(c, &mut enc, term_manager, solver) {
                    Some(term) => {
                        solver.assert(term, term_manager);
                        // The model must be read before the pop below.
                        encode::check_and_extract(var_terms, domain, term_manager, solver)
                    }
                    None => OxizVerdict::Unknown,
                }
            } else {
                OxizVerdict::Unknown
            }
        };

        // ---- close the scope: unconditional, exactly one pop per push ----
        self.solver.pop();
        self.depth -= 1;
        debug_assert_eq!(self.depth, 0, "scoped_check must restore depth 0");

        verdict
    }
}

impl std::fmt::Debug for IncrementalEmlSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalEmlSolver")
            .field("bounds", &self.bounds)
            .field("relaxation_samples", &self.relaxation_samples)
            .field("recycle_after", &self.recycle_after)
            .field("lra_scopes", &self.scopes_total)
            .field("assertion_depth", &self.depth)
            .finish()
    }
}

// ----------------------------------------------------------------------------
// Per-thread solver pool
// ----------------------------------------------------------------------------

thread_local! {
    /// This thread's cached solver, plus the variable box it was built for.
    static POOLED_SOLVER: RefCell<Option<IncrementalEmlSolver>> = const { RefCell::new(None) };
}

/// Run `f` against this thread's cached [`IncrementalEmlSolver`] for `bounds`.
///
/// The solver is constructed on first use and then **reused for every subsequent
/// call with the same variable box** — which is the whole point: a symbolic
/// regression sweep screens thousands of candidate topologies against one fixed
/// box, and each of them should cost a `push`/`pop`, not a `Solver::new()`.
/// A different box rebuilds the solver (checking a candidate against a stale
/// domain would be a correctness bug, not just a slow one).
///
/// OxiZ's `Solver` is not `Sync`, so the pool is per-thread by construction: no
/// locking, and parallel search workers each get their own solver.
///
/// Use [`reset_thread_local_solver`] to drop the cached solver and release its
/// OxiZ term store.
pub fn with_thread_local_solver<R>(
    bounds: &[(f64, f64)],
    f: impl FnOnce(&mut IncrementalEmlSolver) -> R,
) -> R {
    POOLED_SOLVER.with(|cell| {
        let mut slot = cell.borrow_mut();

        let stale = slot
            .as_ref()
            .is_none_or(|solver| !bounds_match(solver.bounds(), bounds));
        if stale {
            *slot = Some(IncrementalEmlSolver::new(bounds.to_vec()));
        }

        let solver = slot
            .as_mut()
            .expect("the pooled solver slot was just populated");
        f(solver)
    })
}

/// Drop this thread's pooled solver, releasing its OxiZ term store.
///
/// The next [`with_thread_local_solver`] call rebuilds it. Useful to return
/// memory once a search has finished, and to establish a deterministic
/// [`solver_construction_count`] baseline in tests.
pub fn reset_thread_local_solver() {
    POOLED_SOLVER.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Bit-exact comparison of two variable boxes. A `NaN` bound never matches, which
/// forces a rebuild — the conservative choice.
fn bounds_match(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && x.1 == y.1 && !x.0.is_nan() && !x.1.is_nan())
}
