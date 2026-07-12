//! Symbolic linear algebra over [`LoweredOp`] entries.
//!
//! A [`Matrix`] holds arbitrary symbolic expressions — `a`, `x·y + 1`, `exp(x)`,
//! `sin(x)/x` — and supports exact determinants, characteristic polynomials,
//! reduced row echelon form, nullspaces, ranks, inverses, eigenvalues and
//! eigenvectors.
//!
//! # The one hard problem
//!
//! Linear algebra runs on a single primitive: *is this entry zero?* Over ℝ that is
//! a comparison. Over symbolic expressions it is **undecidable** — Richardson's
//! theorem (1968) shows that no algorithm can decide whether an expression built
//! from `ℚ`, `π`, `x`, `+ − ×`, `sin` and `exp` is identically zero. Every symbolic
//! linear algebra system must therefore either lie, or be honest about the gap.
//! This one is honest, in three layers.
//!
//! ## 1. Atomize: move into a ring where zero-testing *is* exact
//!
//! Each entry is read as a polynomial over a finite set of *atoms* (the variables,
//! plus each maximal transcendental subexpression), i.e. as an element of
//! `R = ℚ[a₁ … a_m]` — an integral domain with an exact zero test. See
//! [`atoms`].
//!
//! ## 2. Prefer algorithms that never need a functional zero test
//!
//! Determinant (Bareiss, [`bareiss`]) and characteristic polynomial
//! (Faddeev–LeVerrier, [`faddeev`]) are **polynomial** in the matrix entries. Since
//! atomization is a ring homomorphism, computing them in `R` — where every zero test
//! is exact — yields results that are *provably* correct for the original
//! transcendental matrix, no matter what hidden relations the atoms satisfy. These
//! operations carry **no uncertainty at all**. That is not a lucky accident; it is
//! why those two algorithms were chosen over pivoting alternatives.
//!
//! ## 3. Where a functional zero test is unavoidable, surface the doubt
//!
//! Anything that *divides* by an entry — rref, nullspace, rank, inverse — must know
//! that the pivot is nonzero *as a function*, not merely as a polynomial in the
//! atoms (`sin²x + cos²x − 1` is a nonzero polynomial and the zero function). There
//! the [`ZeroOracle`] is consulted, and it returns one of three verdicts, never two:
//!
//! | verdict | meaning |
//! |---|---|
//! | [`ZeroVerdict::Zero`] | **proved** identically zero |
//! | [`ZeroVerdict::NonZero`] | **proved** not identically zero (exact, or a probe witness) |
//! | [`ZeroVerdict::ProbablyZero`] | **undecided** — every probe vanished, but there is no proof |
//!
//! A `ProbablyZero` is *never* silently treated as a zero. Instead:
//!
//! * the `*_certified` API returns a [`Certified<T>`] whose [`Certainty`] is
//!   [`Certainty::Conditional`], listing every [`Assumption`] the algorithm had to
//!   make, each with the expression, what was assumed about it, where, and the
//!   probing evidence;
//! * the plain API ([`Matrix::rref`], [`Matrix::inverse`], …) refuses outright with
//!   [`MatrixError::Undecidable`] rather than return a number it cannot stand behind.
//!
//! Finally [`Matrix::verify`] is the safety net: an independent numeric spot-check of
//! the determinant, the Cayley–Hamilton identity, the inverse and the nullspace at
//! seeded sample points.
//!
//! # Size cap
//!
//! Symbolic determinants have `n!` terms in the worst case, so the symbolic path is
//! capped at `n ≤ 8` ([`MAX_SYMBOLIC_DIM`]). Purely numeric matrices are exact-rational
//! and cheap, and are allowed up to [`MAX_NUMERIC_DIM`].
//!
//! # Example
//!
//! ```
//! use oxieml::matrix::Matrix;
//!
//! // Numeric determinant, exactly.
//! let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
//! assert_eq!(m.det().unwrap(), oxieml::LoweredOp::Const(-2.0));
//!
//! // Symbolic determinant: [[a, b], [c, d]] → a·d − b·c
//! let symbolic = Matrix::of_vars(2, 2).unwrap();
//! let det = symbolic.det().unwrap();
//! assert!((det.eval(&[1.0, 2.0, 3.0, 4.0]) - (1.0 * 4.0 - 2.0 * 3.0)).abs() < 1e-12);
//! ```

pub mod atoms;
mod bareiss;
mod eigen;
mod faddeev;
mod ring;
mod verify;
pub mod zero;

use std::fmt;
use std::sync::Arc;

use crate::error::EmlError;
use crate::lower::LoweredOp;
use crate::poly::{Coeff, Poly, PolyError};

pub use atoms::AtomSpace;
pub use eigen::Eigenpair;
pub use verify::VerifyReport;
pub use zero::{NonZeroProof, ProbeEvidence, ZeroOracle, ZeroProof, ZeroVerdict};

use ring::PolyMatrix;

/// Largest dimension handled on the **symbolic** path.
///
/// A symbolic `n × n` determinant has up to `n!` terms; at `n = 8` that is already
/// 40 320 products of eight symbols. Beyond this, expression swell makes the answer
/// useless even when it is computable, so the cap is enforced rather than allowing a
/// combinatorial explosion.
pub const MAX_SYMBOLIC_DIM: usize = 8;

/// Largest dimension handled on the **numeric** path.
///
/// Numeric entries stay in exact rational arithmetic, where Bareiss keeps
/// intermediate values bounded by Hadamard's inequality, so a much larger cap is
/// affordable.
pub const MAX_NUMERIC_DIM: usize = 64;

// ── Certainty ────────────────────────────────────────────────────────────────

/// What an algorithm had to *assume* about an expression it could not decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssumedRelation {
    /// The expression was assumed to be identically zero.
    IdenticallyZero,
    /// The expression was assumed **not** to be identically zero.
    NotIdenticallyZero,
}

impl fmt::Display for AssumedRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdenticallyZero => write!(f, "≡ 0"),
            Self::NotIdenticallyZero => write!(f, "≢ 0"),
        }
    }
}

/// An unproved hypothesis a computation rested on.
///
/// Every `Assumption` is a place where the zero oracle returned
/// [`ZeroVerdict::ProbablyZero`] and the algorithm nevertheless had to branch. It
/// records exactly *what* was assumed, *where*, and *on what evidence*, so the
/// result can be audited instead of trusted.
#[derive(Clone, Debug, PartialEq)]
pub struct Assumption {
    /// The expression in question.
    pub expr: LoweredOp,
    /// What was assumed about it.
    pub assumed: AssumedRelation,
    /// Where in the algorithm the assumption was made.
    pub context: String,
    /// The probing evidence behind the oracle's undecided verdict.
    pub evidence: ProbeEvidence,
}

impl fmt::Display for Assumption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ASSUMED {} {} — {} (evidence: {} of {} probes usable, max |value| = {:.3e})",
            self.expr.to_pretty(),
            self.assumed,
            self.context,
            self.evidence.usable_probes,
            self.evidence.attempted_probes,
            self.evidence.max_abs_value
        )
    }
}

/// How much a result can be trusted.
#[derive(Clone, Debug, PartialEq)]
pub enum Certainty {
    /// Every zero test the computation depended on was **proved**. The result is
    /// exact and unconditional.
    Proved,
    /// At least one zero test was undecidable and the computation proceeded under an
    /// explicit assumption. The result is correct **if and only if** every listed
    /// [`Assumption`] holds.
    Conditional(Vec<Assumption>),
}

impl Certainty {
    /// Build from a list of assumptions (empty ⇒ [`Certainty::Proved`]).
    #[must_use]
    pub fn from_assumptions(assumptions: Vec<Assumption>) -> Self {
        if assumptions.is_empty() {
            Self::Proved
        } else {
            Self::Conditional(assumptions)
        }
    }

    /// `true` only when nothing was assumed.
    #[must_use]
    pub fn is_proved(&self) -> bool {
        matches!(self, Self::Proved)
    }

    /// The assumptions (empty when proved).
    #[must_use]
    pub fn assumptions(&self) -> &[Assumption] {
        match self {
            Self::Proved => &[],
            Self::Conditional(a) => a,
        }
    }
}

/// A value together with the [`Certainty`] of the computation that produced it.
///
/// This is the type that keeps the undecidability from being swept under the rug: a
/// caller cannot get at the value without walking past the certainty.
#[derive(Clone, Debug, PartialEq)]
pub struct Certified<T> {
    /// The computed value.
    pub value: T,
    /// Whether every zero test behind it was proved.
    pub certainty: Certainty,
}

impl<T> Certified<T> {
    /// A value that rested on no assumptions.
    pub fn proved(value: T) -> Self {
        Self {
            value,
            certainty: Certainty::Proved,
        }
    }

    /// A value that rested on `assumptions` (empty ⇒ proved).
    pub fn conditional(value: T, assumptions: Vec<Assumption>) -> Self {
        Self {
            value,
            certainty: Certainty::from_assumptions(assumptions),
        }
    }

    /// `true` only when nothing was assumed.
    pub fn is_proved(&self) -> bool {
        self.certainty.is_proved()
    }

    /// The assumptions behind this value (empty when proved).
    pub fn assumptions(&self) -> &[Assumption] {
        self.certainty.assumptions()
    }

    /// Unwrap the value **only** if it was proved.
    ///
    /// This is how the plain (non-`_certified`) API refuses to launder an
    /// unprovable assumption into an unqualified answer.
    ///
    /// # Errors
    ///
    /// [`MatrixError::Undecidable`] carrying the first assumption, when the value is
    /// merely conditional.
    pub fn into_proved(self) -> Result<T, MatrixError> {
        match self.certainty {
            Certainty::Proved => Ok(self.value),
            Certainty::Conditional(mut assumptions) => {
                let first = if assumptions.is_empty() {
                    return Ok(self.value);
                } else {
                    assumptions.remove(0)
                };
                Err(MatrixError::Undecidable(Box::new(first)))
            }
        }
    }

    /// Apply `f` to the value, keeping the certainty.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Certified<U> {
        Certified {
            value: f(self.value),
            certainty: self.certainty,
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Everything that can go wrong in symbolic linear algebra.
#[derive(Clone, Debug, PartialEq)]
pub enum MatrixError {
    /// Two matrices (or a matrix and a vector) have incompatible shapes.
    DimensionMismatch {
        /// The shape that was required.
        expected: (usize, usize),
        /// The shape that was supplied.
        got: (usize, usize),
    },
    /// The operation needs a square matrix.
    NotSquare {
        /// Row count of the offending matrix.
        rows: usize,
        /// Column count of the offending matrix.
        cols: usize,
    },
    /// The matrix is **proved** singular: its determinant is identically zero.
    SingularMatrix,
    /// The matrix exceeds the dimension cap for its path.
    TooLarge {
        /// The requested dimension.
        n: usize,
        /// The cap that applies ([`MAX_SYMBOLIC_DIM`] or [`MAX_NUMERIC_DIM`]).
        cap: usize,
    },
    /// The operation needs numeric entries but the matrix is genuinely symbolic.
    ///
    /// Returned by [`Matrix::eigenvalues`] and friends: the eigenvalues of a
    /// symbolic matrix are algebraic *functions* of the symbols, not numbers, and
    /// this crate will not fabricate numbers for them.
    NotNumeric,
    /// **The honest failure.** A zero test the computation depended on could not be
    /// decided, and rather than guess, the operation refused.
    ///
    /// Use the corresponding `*_certified` method to obtain the result together with
    /// the explicit assumption, and audit it yourself (or call [`Matrix::verify`]).
    Undecidable(Box<Assumption>),
    /// A fraction-free division left a remainder.
    ///
    /// Unreachable via Bareiss / Faddeev–LeVerrier (Sylvester's identity guarantees
    /// exactness); it exists as a self-check on the ring arithmetic.
    InexactDivision,
    /// The underlying polynomial arithmetic failed.
    Poly(PolyError),
    /// An entry contains a non-finite literal (`inf` / `NaN`), which has no rational
    /// value and therefore no place in exact arithmetic.
    NonRationalConstant(f64),
    /// An expression referenced an atom outside the space it was polynomialized in.
    UnknownAtom(Box<LoweredOp>),
    /// The matrix is empty where a non-empty one is required.
    Empty,
    /// A numeric sub-algorithm (root finding, inverse iteration) failed to converge.
    NumericFailure(&'static str),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {}×{}, got {}×{}",
                expected.0, expected.1, got.0, got.1
            ),
            Self::NotSquare { rows, cols } => {
                write!(f, "matrix must be square, but is {rows}×{cols}")
            }
            Self::SingularMatrix => {
                write!(f, "matrix is singular (determinant is identically zero)")
            }
            Self::TooLarge { n, cap } => {
                write!(f, "dimension {n} exceeds the cap of {cap} for this path")
            }
            Self::NotNumeric => write!(
                f,
                "operation requires numeric entries; the eigenvalues of a symbolic matrix are \
                 algebraic functions, not numbers"
            ),
            Self::Undecidable(a) => write!(
                f,
                "symbolic zero test is undecidable — refusing to guess. {a}. Use the *_certified \
                 API to proceed under this explicit assumption, or verify() to spot-check"
            ),
            Self::InexactDivision => write!(
                f,
                "fraction-free division left a remainder (this should be unreachable; it \
                 indicates an internal inconsistency)"
            ),
            Self::Poly(e) => write!(f, "polynomial arithmetic failed: {e}"),
            Self::NonRationalConstant(c) => {
                write!(
                    f,
                    "entry contains the non-finite literal {c}, which has no rational value"
                )
            }
            Self::UnknownAtom(a) => {
                write!(f, "expression contains the unknown atom {}", a.to_pretty())
            }
            Self::Empty => write!(f, "matrix is empty"),
            Self::NumericFailure(msg) => write!(f, "numeric failure: {msg}"),
        }
    }
}

impl std::error::Error for MatrixError {}

impl From<PolyError> for MatrixError {
    fn from(e: PolyError) -> Self {
        Self::Poly(e)
    }
}

impl From<MatrixError> for EmlError {
    fn from(e: MatrixError) -> Self {
        match e {
            MatrixError::SingularMatrix => Self::SingularMatrix,
            MatrixError::DimensionMismatch { expected, got } => {
                Self::DimensionMismatch(expected.0 * expected.1, got.0 * got.1)
            }
            MatrixError::NotSquare { .. } => Self::InvalidParameter("matrix must be square"),
            MatrixError::TooLarge { .. } => {
                Self::InvalidParameter("matrix dimension exceeds the symbolic cap")
            }
            MatrixError::NotNumeric => {
                Self::InvalidParameter("operation requires numeric matrix entries")
            }
            MatrixError::Undecidable(_) => Self::NotSolvable,
            MatrixError::InexactDivision => {
                Self::InvalidParameter("fraction-free division left a remainder")
            }
            MatrixError::Poly(_) => Self::InvalidParameter("polynomial arithmetic failed"),
            MatrixError::NonRationalConstant(c) => Self::UndefinedAtPoint(c),
            MatrixError::UnknownAtom(_) => Self::InvalidParameter("unknown atom in expression"),
            MatrixError::Empty => Self::EmptyData,
            MatrixError::NumericFailure(_) => Self::NonConvergence {
                method: "matrix",
                iterations: 0,
            },
        }
    }
}

// ── Results ──────────────────────────────────────────────────────────────────

/// A characteristic polynomial `p(λ) = λⁿ + c_{n−1}λⁿ⁻¹ + … + c₀`.
#[derive(Clone, Debug, PartialEq)]
pub struct CharPoly {
    /// Coefficients in **ascending** order: `coeffs[k]` is `c_k`, and `coeffs[n]` is
    /// the constant `1`.
    pub coeffs: Vec<LoweredOp>,
}

impl CharPoly {
    /// The degree `n` (= the matrix dimension).
    #[must_use]
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Render `p(λ)` as an expression in variable index `lambda_var`.
    #[must_use]
    pub fn to_lowered(&self, lambda_var: usize) -> LoweredOp {
        let lambda = Arc::new(LoweredOp::Var(lambda_var));
        let mut acc: Option<LoweredOp> = None;
        for (k, c) in self.coeffs.iter().enumerate() {
            let term = match k {
                0 => c.clone(),
                1 => LoweredOp::Mul(Arc::new(c.clone()), Arc::clone(&lambda)),
                _ => LoweredOp::Mul(
                    Arc::new(c.clone()),
                    Arc::new(LoweredOp::Pow(
                        Arc::clone(&lambda),
                        Arc::new(LoweredOp::Const(k as f64)),
                    )),
                ),
            };
            acc = Some(match acc {
                None => term,
                Some(a) => LoweredOp::Add(Arc::new(a), Arc::new(term)),
            });
        }
        acc.unwrap_or(LoweredOp::Const(0.0)).simplify()
    }
}

/// A reduced row echelon form.
#[derive(Clone, Debug, PartialEq)]
pub struct Rref {
    /// The RREF matrix.
    pub matrix: Matrix,
    /// The pivot column of each pivot row, ascending.
    pub pivot_cols: Vec<usize>,
}

impl Rref {
    /// The rank (= the number of pivots).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.pivot_cols.len()
    }
}

// ── Matrix ───────────────────────────────────────────────────────────────────

/// A row-major matrix of symbolic expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Entries in row-major order: entry `(i, j)` is `data[i * cols + j]`.
    pub data: Vec<LoweredOp>,
}

impl Matrix {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Build from row-major `data`.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<LoweredOp>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (rows, cols),
                got: (data.len(), 1),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Build from a list of equal-length rows.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] if the rows are ragged.
    pub fn from_rows(rows: &[Vec<LoweredOp>]) -> Result<Self, MatrixError> {
        let n_rows = rows.len();
        let n_cols = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for row in rows {
            if row.len() != n_cols {
                return Err(MatrixError::DimensionMismatch {
                    expected: (n_rows, n_cols),
                    got: (n_rows, row.len()),
                });
            }
            data.extend(row.iter().cloned());
        }
        Ok(Self {
            rows: n_rows,
            cols: n_cols,
            data,
        })
    }

    /// Build a numeric matrix from row-major `f64` values.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] if `values.len() != rows * cols`.
    pub fn from_f64(rows: usize, cols: usize, values: &[f64]) -> Result<Self, MatrixError> {
        Self::new(
            rows,
            cols,
            values.iter().copied().map(LoweredOp::Const).collect(),
        )
    }

    /// The `rows × cols` matrix whose entries are the distinct variables
    /// `x₀ … x_{rows·cols−1}`, in row-major order.
    ///
    /// A convenient way to build a fully generic symbolic matrix:
    /// `Matrix::of_vars(2, 2)` is `[[x₀, x₁], [x₂, x₃]]`.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] never triggers here; the `Result` keeps the
    /// constructor family uniform.
    pub fn of_vars(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        Self::new(rows, cols, (0..rows * cols).map(LoweredOp::Var).collect())
    }

    /// The `n × n` identity.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut data = vec![LoweredOp::Const(0.0); n * n];
        for i in 0..n {
            data[i * n + i] = LoweredOp::Const(1.0);
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    /// The all-zero `rows × cols` matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![LoweredOp::Const(0.0); rows * cols],
        }
    }

    // ── Access ───────────────────────────────────────────────────────────────

    /// Entry `(i, j)`, or `None` when out of bounds.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Option<&LoweredOp> {
        if i >= self.rows || j >= self.cols {
            return None;
        }
        self.data.get(i * self.cols + j)
    }

    /// Entry `(i, j)`. Callers inside this module hold the bounds invariant.
    fn at(&self, i: usize, j: usize) -> &LoweredOp {
        debug_assert!(i < self.rows && j < self.cols);
        &self.data[i * self.cols + j]
    }

    /// Overwrite entry `(i, j)`.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] when `(i, j)` is out of bounds.
    pub fn set(&mut self, i: usize, j: usize, value: LoweredOp) -> Result<(), MatrixError> {
        if i >= self.rows || j >= self.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (i + 1, j + 1),
            });
        }
        self.data[i * self.cols + j] = value;
        Ok(())
    }

    /// `true` when the matrix is square.
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// `true` when every entry is a literal constant.
    #[must_use]
    pub fn is_numeric(&self) -> bool {
        self.data
            .iter()
            .all(|e| matches!(e, LoweredOp::Const(_) | LoweredOp::NamedConst(_)))
    }

    /// The number of distinct variables appearing anywhere in the matrix.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.data
            .iter()
            .map(LoweredOp::count_vars)
            .max()
            .unwrap_or(0)
    }

    /// Simplify every entry.
    #[must_use]
    pub fn simplify(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(LoweredOp::simplify).collect(),
        }
    }

    /// Evaluate every entry at `point`, giving a row-major `f64` matrix.
    #[must_use]
    pub fn eval_at(&self, point: &[f64]) -> Vec<f64> {
        let needed = self.n_vars();
        let mut padded: Vec<f64> = point.to_vec();
        padded.resize(padded.len().max(needed), 0.0);
        self.data.iter().map(|e| e.eval(&padded)).collect()
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self.at(i, j).clone());
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Entry-wise sum.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] on a shape mismatch.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        self.zip_with(other, |a, b| {
            LoweredOp::Add(Arc::new(a.clone()), Arc::new(b.clone())).simplify()
        })
    }

    /// Entry-wise difference.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] on a shape mismatch.
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        self.zip_with(other, |a, b| {
            LoweredOp::Sub(Arc::new(a.clone()), Arc::new(b.clone())).simplify()
        })
    }

    fn zip_with(
        &self,
        other: &Self,
        f: impl Fn(&LoweredOp, &LoweredOp) -> LoweredOp,
    ) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| f(a, b))
                .collect(),
        })
    }

    /// Matrix product.
    ///
    /// # Errors
    ///
    /// [`MatrixError::DimensionMismatch`] when `self.cols != other.rows`.
    pub fn mul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.cols, other.cols),
                got: (other.rows, other.cols),
            });
        }
        let mut data = Vec::with_capacity(self.rows * other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc: Option<LoweredOp> = None;
                for k in 0..self.cols {
                    let term = LoweredOp::Mul(
                        Arc::new(self.at(i, k).clone()),
                        Arc::new(other.at(k, j).clone()),
                    );
                    acc = Some(match acc {
                        None => term,
                        Some(a) => LoweredOp::Add(Arc::new(a), Arc::new(term)),
                    });
                }
                data.push(acc.unwrap_or(LoweredOp::Const(0.0)).simplify());
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            data,
        })
    }

    /// Multiply every entry by `scalar`.
    #[must_use]
    pub fn scale(&self, scalar: &LoweredOp) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .map(|e| LoweredOp::Mul(Arc::new(scalar.clone()), Arc::new(e.clone())).simplify())
                .collect(),
        }
    }

    // ── Bridge into the polynomial ring ──────────────────────────────────────

    /// Atomize the whole matrix into `ℚ[atoms]`.
    pub(crate) fn to_poly_matrix(&self) -> Result<(AtomSpace, PolyMatrix), MatrixError> {
        let simplified: Vec<LoweredOp> = self.data.iter().map(LoweredOp::simplify).collect();
        let space = AtomSpace::from_exprs(&simplified);
        let mut data = Vec::with_capacity(simplified.len());
        for entry in &simplified {
            data.push(space.polynomialize(entry)?);
        }
        let pm = PolyMatrix {
            rows: self.rows,
            cols: self.cols,
            num_vars: space.len(),
            data,
        };
        Ok((space, pm))
    }

    /// Enforce the dimension cap appropriate to this matrix.
    fn check_cap(&self) -> Result<(), MatrixError> {
        let n = self.rows.max(self.cols);
        let cap = if self.is_numeric() {
            MAX_NUMERIC_DIM
        } else {
            MAX_SYMBOLIC_DIM
        };
        if n > cap {
            return Err(MatrixError::TooLarge { n, cap });
        }
        Ok(())
    }

    fn require_square(&self) -> Result<usize, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(self.rows)
    }

    // ── Determinant ──────────────────────────────────────────────────────────

    /// The exact determinant, by Bareiss fraction-free elimination.
    ///
    /// **This result carries no uncertainty.** The determinant is a polynomial in the
    /// entries and atomization is a ring homomorphism, so computing it in `ℚ[atoms]`
    /// — where every zero test is exact — yields the determinant of the original
    /// matrix even when the atoms satisfy hidden transcendental relations. No zero
    /// oracle is consulted and no assumption is made. See [`bareiss`].
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotSquare`], [`MatrixError::TooLarge`], or a ring-arithmetic
    /// failure.
    pub fn det(&self) -> Result<LoweredOp, MatrixError> {
        self.require_square()?;
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let det = bareiss::bareiss_det(&pm)?;
        Ok(space.poly_to_lowered(&det).simplify())
    }

    /// The determinant as an **exact rational**, for a numeric matrix.
    ///
    /// Unlike [`Matrix::det`], nothing is rounded to `f64` on the way out.
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotNumeric`] when the matrix is genuinely symbolic.
    pub fn det_exact(&self) -> Result<Coeff, MatrixError> {
        self.require_square()?;
        self.check_cap()?;
        let (_space, pm) = self.to_poly_matrix()?;
        let det = bareiss::bareiss_det(&pm)?;
        ring::as_constant(&det).ok_or(MatrixError::NotNumeric)
    }

    /// The determinant together with the oracle's verdict on whether it is
    /// identically zero.
    ///
    /// This is the transparent form of [`Matrix::is_singular`]: it hands back both the
    /// (unconditionally correct) determinant expression and the (possibly undecided)
    /// verdict, so a caller can inspect the evidence instead of taking a `bool` on
    /// faith.
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotSquare`] or [`MatrixError::TooLarge`].
    pub fn determinant_verdict(
        &self,
        oracle: &ZeroOracle,
    ) -> Result<(LoweredOp, ZeroVerdict), MatrixError> {
        self.require_square()?;
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let det = bareiss::bareiss_det(&pm)?;
        let verdict = oracle.test_poly(&det, &space);
        Ok((space.poly_to_lowered(&det).simplify(), verdict))
    }

    /// Is the matrix singular?
    ///
    /// # Errors
    ///
    /// [`MatrixError::Undecidable`] when the oracle cannot decide whether the
    /// determinant vanishes identically. That is not a failure of this crate; it is
    /// the undecidability of the problem, reported rather than papered over.
    pub fn is_singular(&self) -> Result<bool, MatrixError> {
        let oracle = ZeroOracle::new();
        let (det, verdict) = self.determinant_verdict(&oracle)?;
        match verdict {
            ZeroVerdict::Zero(_) => Ok(true),
            ZeroVerdict::NonZero(_) => Ok(false),
            ZeroVerdict::ProbablyZero(evidence) => {
                Err(MatrixError::Undecidable(Box::new(Assumption {
                    expr: det,
                    assumed: AssumedRelation::IdenticallyZero,
                    context: "singularity test: the determinant vanished at every probe point but \
                              is not the zero polynomial in the atom ring, so it cannot be shown \
                              to be identically zero"
                        .to_string(),
                    evidence,
                })))
            }
        }
    }

    // ── Characteristic polynomial ────────────────────────────────────────────

    /// The characteristic polynomial `det(λI − A)`, by Faddeev–LeVerrier.
    ///
    /// Like [`Matrix::det`], this carries **no uncertainty**: the recurrence divides
    /// only by the integers `1 … n`, never by a matrix entry, so no symbolic zero
    /// test is involved. See [`faddeev`].
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotSquare`] or [`MatrixError::TooLarge`].
    pub fn charpoly(&self) -> Result<CharPoly, MatrixError> {
        self.require_square()?;
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let fl = faddeev::faddeev_leverrier(&pm)?;
        Ok(CharPoly {
            coeffs: fl
                .coeffs
                .iter()
                .map(|c| space.poly_to_lowered(c).simplify())
                .collect(),
        })
    }

    /// The characteristic polynomial with **exact rational** coefficients, for a
    /// numeric matrix.
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotNumeric`] when the matrix is genuinely symbolic — the
    /// charpoly then has symbolic coefficients and is not a [`Poly`].
    pub fn charpoly_exact(&self) -> Result<Poly, MatrixError> {
        self.require_square()?;
        self.check_cap()?;
        let (_space, pm) = self.to_poly_matrix()?;
        let fl = faddeev::faddeev_leverrier(&pm)?;
        eigen::exact_charpoly(&fl.coeffs)
    }

    /// `p(A)`, where `p` is the characteristic polynomial — the Cayley–Hamilton
    /// residual.
    ///
    /// By the Cayley–Hamilton theorem (valid over any commutative ring) this is the
    /// zero matrix, *exactly*, and the entries come back as literal `Const(0.0)`.
    /// A nonzero entry would mean a bug in the charpoly, the ring arithmetic or the
    /// atomization — which is what makes this a genuine probe rather than a
    /// tautology.
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotSquare`] or [`MatrixError::TooLarge`].
    pub fn cayley_hamilton_residual(&self) -> Result<Self, MatrixError> {
        self.require_square()?;
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let fl = faddeev::faddeev_leverrier(&pm)?;
        let residual = faddeev::cayley_hamilton_residual(&pm, &fl.coeffs)?;
        Ok(Self {
            rows: residual.rows,
            cols: residual.cols,
            data: residual
                .data
                .iter()
                .map(|p| space.poly_to_lowered(p).simplify())
                .collect(),
        })
    }

    // ── Inverse ──────────────────────────────────────────────────────────────

    /// The symbolic inverse, as `−M_n / c₀` from Faddeev–LeVerrier.
    ///
    /// The numerators are exact polynomials (the adjugate); the common denominator is
    /// the determinant. Nothing here is approximate.
    ///
    /// # Errors
    ///
    /// * [`MatrixError::SingularMatrix`] when the determinant is **proved** to vanish
    ///   identically.
    /// * [`MatrixError::Undecidable`] when the oracle cannot tell whether the
    ///   determinant vanishes. Inverting would require *assuming* it does not — an
    ///   assumption the evidence actively argues against — so this refuses.
    ///   [`Matrix::inverse_assuming_nonsingular`] will do it if you take that
    ///   assumption on explicitly.
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        let oracle = ZeroOracle::new();
        self.inverse_assuming_nonsingular(&oracle)?.into_proved()
    }

    /// The symbolic inverse, proceeding under the explicit assumption that the matrix
    /// is nonsingular when the oracle cannot decide.
    ///
    /// The returned [`Certified`] is [`Certainty::Proved`] when the determinant was
    /// *proved* nonzero, and [`Certainty::Conditional`] — naming the determinant and
    /// carrying the probing evidence — when it was not. The caller therefore cannot
    /// mistake a conditional inverse for a proved one.
    ///
    /// # Errors
    ///
    /// [`MatrixError::SingularMatrix`] when the determinant is proved to be
    /// identically zero. No assumption can rescue that: the inverse does not exist.
    pub fn inverse_assuming_nonsingular(
        &self,
        oracle: &ZeroOracle,
    ) -> Result<Certified<Self>, MatrixError> {
        let n = self.require_square()?;
        self.check_cap()?;
        if n == 0 {
            return Ok(Certified::proved(Self::zeros(0, 0)));
        }

        let (space, pm) = self.to_poly_matrix()?;
        let fl = faddeev::faddeev_leverrier(&pm)?;
        let c0 = fl.coeffs.first().ok_or(MatrixError::Empty)?.clone();
        let det = fl.determinant(n)?;

        let verdict = oracle.test_poly(&det, &space);
        let assumptions = match verdict {
            ZeroVerdict::Zero(_) => return Err(MatrixError::SingularMatrix),
            ZeroVerdict::NonZero(_) => Vec::new(),
            ZeroVerdict::ProbablyZero(evidence) => vec![Assumption {
                expr: space.poly_to_lowered(&det).simplify(),
                assumed: AssumedRelation::NotIdenticallyZero,
                context: "inverse: the determinant vanished at every probe point, so the matrix \
                          cannot be shown to be invertible; proceeding only because the caller \
                          asked to assume nonsingularity"
                    .to_string(),
                evidence,
            }],
        };

        // A⁻¹ = −M_n / c₀  (Cayley–Hamilton: A·M_n = −c₀·I).
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let numerator = fl.m_final.at(i, j).neg().map_err(MatrixError::Poly)?;
                data.push(bareiss::ratio_to_lowered(&numerator, &c0, &space)?);
            }
        }

        Ok(Certified::conditional(
            Self {
                rows: n,
                cols: n,
                data,
            },
            assumptions,
        ))
    }

    // ── Elimination: rref, nullspace, rank ───────────────────────────────────

    /// Reduced row echelon form.
    ///
    /// # Errors
    ///
    /// [`MatrixError::Undecidable`] when a pivot decision could not be proved. Use
    /// [`Matrix::rref_certified`] to see the assumption instead.
    pub fn rref(&self) -> Result<Rref, MatrixError> {
        let oracle = ZeroOracle::new();
        self.rref_certified(&oracle)?.into_proved()
    }

    /// Reduced row echelon form, with the assumptions made along the way.
    ///
    /// Pivots are chosen only among entries the oracle **proves** nonzero. A column
    /// whose only candidates are undecidable is declared pivot-free — which *assumes*
    /// those entries vanish identically — and every such assumption is recorded in the
    /// returned [`Certainty`].
    ///
    /// # Errors
    ///
    /// [`MatrixError::TooLarge`] or a ring-arithmetic failure.
    pub fn rref_certified(&self, oracle: &ZeroOracle) -> Result<Certified<Rref>, MatrixError> {
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let reduced = bareiss::reduced_echelon(&pm, &space, oracle)?;

        let mut data = vec![LoweredOp::Const(0.0); self.rows * self.cols];
        for (r, &pivot_col) in reduced.pivot_cols.iter().enumerate() {
            let pivot = reduced.matrix.at(r, pivot_col).clone();
            for j in 0..self.cols {
                let entry = bareiss::ratio_to_lowered(reduced.matrix.at(r, j), &pivot, &space)?;
                data[r * self.cols + j] = entry;
            }
        }

        let rref = Rref {
            matrix: Self {
                rows: self.rows,
                cols: self.cols,
                data,
            },
            pivot_cols: reduced.pivot_cols,
        };
        Ok(Certified::conditional(rref, reduced.assumptions))
    }

    /// A basis of the nullspace `{v : A v = 0}`.
    ///
    /// # Errors
    ///
    /// [`MatrixError::Undecidable`] when a pivot decision could not be proved.
    pub fn nullspace(&self) -> Result<Vec<Vec<LoweredOp>>, MatrixError> {
        let oracle = ZeroOracle::new();
        self.nullspace_certified(&oracle)?.into_proved()
    }

    /// A nullspace basis, with the assumptions made along the way.
    ///
    /// One basis vector per free column: the free variable is set to `1` and the
    /// pivot variables are back-substituted from the reduced echelon form.
    ///
    /// Note that the nullspace of a symbolic matrix is only well defined *given* the
    /// rank, and the rank depends on the pivot decisions — so a `Conditional` result
    /// here means the basis is correct only if the listed assumptions hold. Under the
    /// opposite assumption the nullspace could have a different dimension entirely.
    ///
    /// # Errors
    ///
    /// [`MatrixError::TooLarge`] or a ring-arithmetic failure.
    pub fn nullspace_certified(
        &self,
        oracle: &ZeroOracle,
    ) -> Result<Certified<Vec<Vec<LoweredOp>>>, MatrixError> {
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let reduced = bareiss::reduced_echelon(&pm, &space, oracle)?;

        let free_cols: Vec<usize> = (0..self.cols)
            .filter(|c| !reduced.pivot_cols.contains(c))
            .collect();

        let mut basis: Vec<Vec<LoweredOp>> = Vec::with_capacity(free_cols.len());
        for &free in &free_cols {
            let mut v = vec![LoweredOp::Const(0.0); self.cols];
            v[free] = LoweredOp::Const(1.0);
            // Row `r` of the reduced echelon form reads
            //     p_r · x_{c_r}  +  E[r][f] · x_f  +  (other free terms)  =  0,
            // so setting x_f = 1 and every other free variable to 0 gives
            //     x_{c_r} = −E[r][f] / p_r .
            // The numerator and the denominator must be normalised *together* (as a
            // ratio) or not at all — scaling them independently would silently change
            // the value.
            for (r, &pivot_col) in reduced.pivot_cols.iter().enumerate() {
                let pivot = reduced.matrix.at(r, pivot_col);
                let numerator = reduced
                    .matrix
                    .at(r, free)
                    .neg()
                    .map_err(MatrixError::Poly)?;
                v[pivot_col] = bareiss::ratio_to_lowered(&numerator, pivot, &space)?;
            }
            basis.push(v);
        }

        Ok(Certified::conditional(basis, reduced.assumptions))
    }

    /// The rank.
    ///
    /// # Errors
    ///
    /// [`MatrixError::Undecidable`] when a pivot decision could not be proved — the
    /// rank of a matrix with an undecidable entry is genuinely not determined.
    pub fn rank(&self) -> Result<usize, MatrixError> {
        let oracle = ZeroOracle::new();
        self.rank_certified(&oracle)?.into_proved()
    }

    /// The rank, with the assumptions made along the way.
    ///
    /// # Errors
    ///
    /// [`MatrixError::TooLarge`] or a ring-arithmetic failure.
    pub fn rank_certified(&self, oracle: &ZeroOracle) -> Result<Certified<usize>, MatrixError> {
        self.check_cap()?;
        let (space, pm) = self.to_poly_matrix()?;
        let ech = bareiss::echelon(&pm, &space, oracle)?;
        Ok(Certified::conditional(
            ech.pivot_cols.len(),
            ech.assumptions,
        ))
    }

    // ── Spectrum ─────────────────────────────────────────────────────────────

    /// All eigenvalues (with multiplicity), as the complex roots of the exact
    /// characteristic polynomial.
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotNumeric`] for a symbolic matrix: its eigenvalues are
    /// algebraic *functions* of the symbols, and beyond degree 4 they have no radical
    /// form at all (Abel–Ruffini). Factor [`Matrix::charpoly`] yourself if you want
    /// the symbolic ones.
    pub fn eigenvalues(&self) -> Result<Vec<num_complex::Complex<f64>>, MatrixError> {
        let charpoly = self.charpoly_exact()?;
        eigen::eigenvalues_of(&charpoly)
    }

    /// All eigenvalues, each with a unit eigenvector, by inverse iteration.
    ///
    /// # Errors
    ///
    /// [`MatrixError::NotNumeric`] for a symbolic matrix, or
    /// [`MatrixError::NumericFailure`] when inverse iteration cannot reach an
    /// eigenvector of acceptable residual (a severely defective matrix). A vector is
    /// never returned unless its residual is small — a plausible-looking non-solution
    /// is worse than an error.
    pub fn eigenvectors(&self) -> Result<Vec<Eigenpair>, MatrixError> {
        let n = self.require_square()?;
        self.check_cap()?;
        let values = self.eigenvalues()?;
        let (_space, pm) = self.to_poly_matrix()?;
        let entries = eigen::numeric_entries(&pm)?;
        eigen::eigenvectors_of(&entries, n, &values)
    }

    // ── Safety net ───────────────────────────────────────────────────────────

    /// Independently spot-check the symbolic results against numeric linear algebra.
    ///
    /// See [`VerifyReport`] for what is checked. This is the backstop against a wrong
    /// assumption having slipped through: it evaluates the matrix at seeded sample
    /// points and compares the symbolic determinant, Cayley–Hamilton residual,
    /// inverse and nullspace against numerically computed ones.
    ///
    /// # Errors
    ///
    /// Propagates the failure of whichever symbolic computation could not be carried
    /// out at all.
    pub fn verify(&self) -> Result<VerifyReport, MatrixError> {
        verify::verify(self, &ZeroOracle::new())
    }

    /// [`Matrix::verify`] with an explicit oracle (probe count / seed).
    ///
    /// # Errors
    ///
    /// As [`Matrix::verify`].
    pub fn verify_with(&self, oracle: &ZeroOracle) -> Result<VerifyReport, MatrixError> {
        verify::verify(self, oracle)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            let row: Vec<String> = (0..self.cols).map(|j| self.at(i, j).to_pretty()).collect();
            writeln!(f, "[{}]", row.join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_determinant_and_inverse() {
        let m = Matrix::identity(4);
        assert_eq!(m.det().expect("det"), LoweredOp::Const(1.0));
        let inv = m.inverse().expect("inverse");
        assert_eq!(inv, Matrix::identity(4));
    }

    #[test]
    fn singular_matrix_is_reported() {
        let m = Matrix::from_f64(2, 2, &[1.0, 2.0, 2.0, 4.0]).expect("dims");
        assert_eq!(m.inverse(), Err(MatrixError::SingularMatrix));
        assert!(m.is_singular().expect("decidable"));
    }

    #[test]
    fn dimension_cap_is_enforced_on_the_symbolic_path() {
        let n = MAX_SYMBOLIC_DIM + 1;
        let m = Matrix::of_vars(n, n).expect("dims");
        assert_eq!(
            m.det(),
            Err(MatrixError::TooLarge {
                n,
                cap: MAX_SYMBOLIC_DIM
            })
        );
    }

    #[test]
    fn numeric_path_allows_larger_matrices() {
        let n = MAX_SYMBOLIC_DIM + 2;
        let mut m = Matrix::identity(n);
        m.set(0, 0, LoweredOp::Const(3.0)).expect("in bounds");
        assert_eq!(m.det().expect("det"), LoweredOp::Const(3.0));
    }

    #[test]
    fn transpose_round_trips() {
        let m = Matrix::of_vars(2, 3).expect("dims");
        assert_eq!(m.transpose().transpose(), m);
    }
}
