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

pub mod autodiff;
pub mod canonical;
pub mod compile;
pub mod error;
pub mod eval;
pub mod grad;
pub mod integrate;
pub(crate) mod integrate_subst;
pub mod limit;
pub mod linalg;
pub mod lower;
pub mod lower_cse;
pub mod lower_grad;
pub mod lower_interval;
pub mod lower_simplify;
pub mod lower_units;
pub mod named_const;
pub mod numeric;
pub mod numeric_verified;
pub mod ode;
pub mod parser;
pub mod poly;
pub mod quadrature_nd;
pub mod series;
#[cfg(feature = "simd")]
pub mod simd_eval;
pub(crate) mod simd_vec_math;
pub mod simplify;
#[cfg(feature = "smt")]
pub mod smt;
pub mod solve;
pub mod solve_poly;
pub mod special;
pub mod symreg;
pub mod system;
#[cfg(feature = "tensorlogic")]
pub mod tensorlogic;
pub mod tree;
pub mod units;

#[cfg(feature = "scirs2")]
pub mod scirs2;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "jit")]
pub mod jit;

// Re-exports for convenience
pub use canonical::Canonical;
pub use error::EmlError;
pub use eval::EvalCtx;
pub use integrate::IntegrateResult;
pub use limit::{LimitPoint, LimitResult};
pub use lower::LoweredOp;
pub use lower_interval::IntervalLO;
pub use named_const::NamedConst;
pub use numeric::{QuadOpts, RootOpts};
pub use numeric::{lambert_w0, lambert_wm1};
pub use numeric_verified::{RootCertificate, RootStatus, VerifiedQuadOpts};
pub use ode::{OdeForm, OdeKind, OdeSolution, dsolve};
pub use parser::{ParseError, parse};
pub use poly::{Factorization, MultiPoly, Poly, PolyError};
pub use quadrature_nd::{QuadNdMethod, QuadNdOpts, quadrature_nd};
pub use solve::SolveResult;
pub use solve::{SystemSolveResult, solve_for_all, solve_linear_system};
pub use solve_poly::RootsResult;
pub use symreg::SymRegLoss;
pub use symreg::{
    DiscoveredFormula, LmConfig, MultiOutputStrategy, OptimizerKind, PdeConfig, PdeResult,
    SelectionCriterion, SharedFormula, SymRegConfig, SymRegEngine, SymRegStrategy, discover_pde,
    pareto_front,
};
pub use symreg::{
    LibraryTerm, SindyConfig, SindyEquation, SindyMode, SindyResult, discover_ode_sindy,
};
pub use system::{SystemOpts, solve_system_newton};
pub use tree::{EmlNode, EmlTree};
pub use units::{UnitError, Units};

#[cfg(feature = "smt")]
pub use smt::{
    EmlConstraint, EmlNraSolver, EmlSmtSolver, EmlSolution, Interval, IntervalDomain, PropResult,
    SmtResult,
};

#[cfg(feature = "scirs2")]
pub use scirs2::{symbolic_regression, symbolic_regression_multi, symbolic_regression_with_names};

#[cfg(feature = "jit")]
pub use jit::{JitCache, JitFn};
