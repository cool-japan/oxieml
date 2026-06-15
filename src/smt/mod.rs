//! SMT (Satisfiability Modulo Theories) integration.
//!
//! Two-layer solver stack:
//! 1. **Interval propagation** (`IntervalDomain`) — EML-aware forward/backward
//!    rules for `eml(l, r) = exp(l) − ln(r)`. Tightens variable domains and
//!    proves trivial UNSAT cases without external deps.
//! 2. **OxiZ backend** (`EmlSmtSolver`, feature-gated on `smt`) — encodes
//!    EML constraints for OxiZ's LRA theory via secant+tangent linear
//!    relaxation of `exp`/`ln` over tight intervals. Uses OxiZ 0.2.3 from
//!    crates.io.
//!
//! Legacy `EmlNraSolver` (interval bisection) remains available for witness
//! extraction and is enhanced with propagation.

mod constraint;
mod helpers;
mod interval;
mod nra;
#[cfg(feature = "smt")]
mod oxiz_backend;
#[cfg(all(test, feature = "smt"))]
mod smt_tests;
#[cfg(test)]
mod tests;

pub use constraint::{EmlConstraint, EmlSolution};
pub use interval::{Interval, IntervalDomain, PropResult};
pub use nra::EmlNraSolver;
#[cfg(feature = "smt")]
pub use oxiz_backend::{EmlSmtSolver, SmtResult};
