//! OxiEML — All elementary functions from a single binary operator.
//!
//! This crate implements the EML operator `eml(x, y) = exp(x) - ln(y)` and
//! builds uniform binary trees that represent all elementary functions using
//! only this operator and the constant `1`.
//!
//! Based on the paper: "All elementary functions from a single binary operator"
//! (arXiv:2603.21852).
//!
//! # Capabilities
//!
//! 1. **Symbolic Regression**: Discover closed-form formulas from data via
//!    gradient-based search over EML tree topologies.
//! 2. **Uniform Tree Representation**: Express any elementary function using
//!    the grammar `S → 1 | eml(S, S)`.
//!
//! # Example
//!
//! ```
//! use oxieml::{EmlTree, Canonical, EvalCtx};
//!
//! // Build exp(x) = eml(x, 1)
//! let x = EmlTree::var(0);
//! let exp_x = Canonical::exp(&x);
//!
//! // Evaluate at x = 1.0
//! let ctx = EvalCtx::new(&[1.0]);
//! let result = exp_x.eval_real(&ctx).unwrap();
//! assert!((result - std::f64::consts::E).abs() < 1e-10);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

pub mod canonical;
pub mod compile;
pub mod error;
pub mod eval;
pub mod grad;
pub mod lower;
pub mod parser;
#[cfg(feature = "simd")]
pub mod simd_eval;
pub mod simplify;
#[cfg(feature = "smt")]
pub mod smt;
pub mod symreg;
pub mod tree;

// Re-exports for convenience
pub use canonical::Canonical;
pub use error::EmlError;
pub use eval::EvalCtx;
pub use lower::LoweredOp;
pub use parser::{ParseError, parse};
pub use symreg::{SymRegConfig, SymRegEngine};
pub use tree::{EmlNode, EmlTree};

#[cfg(feature = "smt")]
pub use smt::{
    EmlConstraint, EmlNraSolver, EmlSmtSolver, EmlSolution, Interval, IntervalDomain, PropResult,
    SmtResult,
};
