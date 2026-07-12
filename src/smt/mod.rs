//! SMT (Satisfiability Modulo Theories) integration.
//!
//! Three-layer solver stack:
//! 1. **Interval propagation** (`IntervalDomain`) — EML-aware forward/backward
//!    rules for `eml(l, r) = exp(l) − ln(r)`. Tightens variable domains and
//!    proves trivial UNSAT cases without external deps.
//! 2. **OxiZ backend** (`EmlSmtSolver`, feature-gated on `smt`) — encodes
//!    EML constraints for OxiZ's LRA theory via secant+tangent linear
//!    relaxation of `exp`/`ln` over tight intervals. Uses OxiZ 0.2.3 from
//!    crates.io. Builds a fresh `Solver` per query.
//! 3. **Incremental backend** ([`IncrementalEmlSolver`]) — the same encoding and
//!    the same pipeline, but against **one live `Solver`**, with a `push`/`pop`
//!    scope per query. This is what the symbolic-regression pruner uses. See the
//!    [`incremental`] module docs for the solver-state hygiene argument that
//!    makes its verdicts identical to N independent `check_sat` calls, and for
//!    the OxiZ 0.2.3 `pop()` limitation it works around.
//!
//! On top of that stack:
//! * [`core`] — irreducible unsat cores (minimal infeasible subsets) of an
//!   `EmlConstraint` list.
//! * [`maxsmt`] — MaxSMT: maximize the total weight of satisfied soft
//!   constraints. Exact core-guided search driven by OxiZ's RC2 MaxSAT engine
//!   under `smt-opt`; an honestly-labelled greedy fallback under base `smt`.
//!
//! Legacy `EmlNraSolver` (interval bisection) remains available for witness
//! extraction and is enhanced with propagation.

mod constraint;
mod helpers;
mod interval;
mod nra;
#[cfg(test)]
mod tests;

#[cfg(feature = "smt")]
pub mod core;
#[cfg(feature = "smt")]
mod encode;
#[cfg(feature = "smt")]
pub mod incremental;
#[cfg(feature = "smt")]
pub mod maxsmt;
#[cfg(feature = "smt")]
mod oxiz_backend;
#[cfg(all(test, feature = "smt"))]
mod smt_tests;

pub use constraint::{EmlConstraint, EmlSolution};
pub use interval::{Interval, IntervalDomain, PropResult};
pub use nra::EmlNraSolver;

#[cfg(feature = "smt")]
pub use self::core::UnsatCore;
#[cfg(feature = "smt")]
pub use incremental::{
    DEFAULT_RECYCLE_AFTER, IncrementalEmlSolver, reset_thread_local_solver,
    solver_construction_count, with_thread_local_solver,
};
#[cfg(feature = "smt")]
pub use maxsmt::{MaxSmtOptimality, MaxSmtOutcome, MaxSmtSolution, SoftConstraint};
#[cfg(feature = "smt")]
pub use oxiz_backend::{EmlSmtSolver, SmtResult};
