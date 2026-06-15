//! Error types for the OxiEML crate.

use std::fmt;

/// Errors that can occur during EML tree operations.
#[derive(Clone, Debug, PartialEq)]
pub enum EmlError {
    /// Evaluation produced a complex result when real was expected.
    /// Contains the imaginary part magnitude.
    ComplexResult(f64),

    /// Numerical overflow during exp computation.
    /// Contains the argument that caused overflow.
    ExpOverflow(f64),

    /// Logarithm of zero or negative number in real mode.
    LnDomain(f64),

    /// Variable index out of bounds.
    /// (requested_index, num_vars)
    VarOutOfBounds(usize, usize),

    /// Input data dimension mismatch.
    /// (expected, got)
    DimensionMismatch(usize, usize),

    /// Symbolic regression failed to converge.
    ConvergenceFailed {
        /// Best MSE achieved
        best_mse: f64,
        /// Number of iterations completed
        iterations: usize,
    },

    /// NaN encountered during computation.
    NanEncountered,

    /// Empty input data.
    EmptyData,

    /// Numeric iterative method hit its iteration cap without converging.
    NonConvergence {
        /// Name of the method (e.g. "find_root", "lm_optimizer").
        method: &'static str,
        /// Number of iterations completed before giving up.
        iterations: usize,
    },

    /// Requested operation is undefined at the given point.
    /// (e.g. Taylor expansion where a derivative is non-finite at center,
    /// or quadrature over an interval containing a singularity).
    UndefinedAtPoint(f64),

    /// A numeric parameter was invalid (e.g. n_samples == 0).
    InvalidParameter(&'static str),

    /// The matrix passed to a linear solver is singular (zero pivot found).
    SingularMatrix,

    /// The matrix passed to the Cholesky solver is not positive definite.
    NotSpd,

    /// Input is outside the domain of the operation.
    OutOfDomain,

    /// Equation has no closed-form solution via implemented methods.
    NotSolvable,

    /// Grid data is too small for the requested stencil/operation.
    GridTooSmall {
        /// How many points are needed.
        needed: usize,
        /// How many were provided.
        got: usize,
    },

    /// A [`tensorlogic_ir::TLExpr`] variant has no `LoweredOp` equivalent.
    ///
    /// Produced by `crate::tensorlogic::from_tlexpr` when the input falls
    /// outside the arithmetic/transcendental subset supported by the bridge
    /// (for example, logical connectives, quantifiers, or set-theoretic ops).
    #[cfg(feature = "tensorlogic")]
    UnsupportedTlExpr(String),
}

impl fmt::Display for EmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComplexResult(im) => {
                write!(f, "complex result with |Im| = {im:.2e}")
            }
            Self::ExpOverflow(x) => {
                write!(f, "exp overflow: argument {x:.2e} exceeds limit")
            }
            Self::LnDomain(x) => {
                write!(f, "ln domain error: argument {x}")
            }
            Self::VarOutOfBounds(idx, n) => {
                write!(f, "variable index {idx} out of bounds (num_vars = {n})")
            }
            Self::DimensionMismatch(expected, got) => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::ConvergenceFailed {
                best_mse,
                iterations,
            } => {
                write!(
                    f,
                    "convergence failed after {iterations} iterations (best MSE = {best_mse:.2e})"
                )
            }
            Self::NanEncountered => write!(f, "NaN encountered during computation"),
            Self::EmptyData => write!(f, "empty input data"),
            Self::NonConvergence { method, iterations } => {
                write!(f, "{method} did not converge after {iterations} iterations")
            }
            Self::UndefinedAtPoint(x) => {
                write!(f, "operation undefined at x = {x}")
            }
            Self::InvalidParameter(msg) => {
                write!(f, "invalid parameter: {msg}")
            }
            Self::SingularMatrix => write!(f, "matrix is singular (zero pivot)"),
            Self::NotSpd => write!(f, "matrix is not symmetric positive definite"),
            Self::OutOfDomain => write!(f, "input is outside the domain of the operation"),
            Self::NotSolvable => write!(
                f,
                "equation has no closed-form solution via implemented methods"
            ),
            Self::GridTooSmall { needed, got } => {
                write!(f, "grid too small: need {needed} points, got {got}")
            }
            #[cfg(feature = "tensorlogic")]
            Self::UnsupportedTlExpr(desc) => {
                write!(f, "unsupported TLExpr variant: {desc}")
            }
        }
    }
}

impl std::error::Error for EmlError {}
