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
        }
    }
}

impl std::error::Error for EmlError {}
