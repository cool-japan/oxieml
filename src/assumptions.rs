//! Lightweight per-variable assumptions for assumption-gated simplification.
//!
//! Many classical simplification identities are only valid under a side
//! condition on the sign (or reality, or integrality) of a variable. The
//! canonical example is
//!
//! ```text
//! √(x²) = |x|          (for all real x)
//! √(x²) = x            (only when x ≥ 0)
//! ```
//!
//! Rewriting `√(x²)` to `x` is *unsound* in general — it changes the value at
//! every negative point — but *sound* on the sub-domain `x ≥ 0`. Rather than
//! forbidding such rewrites outright, this module lets a caller *assert* what it
//! knows about each variable; [`crate::lower::LoweredOp::simplify_with`] then
//! restricts the numeric-verification probe points to the asserted sub-domain,
//! so a gated rewrite is accepted exactly when it is value-preserving *there*.
//!
//! # Design
//!
//! * [`VarAssumption`] is a small `Copy` record of per-variable flags. Every flag
//!   defaults to `false` — the *weakest* possible knowledge (i.e. "unknown").
//!   `positive` is strictly stronger than `nonnegative`, and asserting the former
//!   implies the latter.
//! * [`Assumptions`] maps a variable index to its [`VarAssumption`]. A variable
//!   absent from the map is treated as fully unknown ([`VarAssumption::default`]).
//!
//! The engine currently reasons only about the *real* line (evaluation is `f64`),
//! so `real` and `integer` are carried for completeness and future use; the
//! sign flags (`positive`, `nonnegative`) are the ones the sqrt/abs gate reads.

use std::collections::BTreeMap;

/// Per-variable mathematical assumptions.
///
/// Each flag is an *assertion* the caller makes about a variable; the simplifier
/// never sets these itself. All flags default to `false`, meaning "nothing is
/// known" (the weakest assumption), so a default `VarAssumption` never unlocks a
/// gated rewrite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarAssumption {
    /// The variable is strictly positive (`x > 0`). Implies [`Self::nonnegative`].
    pub positive: bool,
    /// The variable is non-negative (`x ≥ 0`).
    pub nonnegative: bool,
    /// The variable is a real number (`x ∈ ℝ`).
    pub real: bool,
    /// The variable is an integer (`x ∈ ℤ`).
    pub integer: bool,
}

impl VarAssumption {
    /// A `VarAssumption` with every flag cleared (nothing known).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            positive: false,
            nonnegative: false,
            real: false,
            integer: false,
        }
    }

    /// Assert the variable is strictly positive (`x > 0`).
    ///
    /// Positivity implies non-negativity, so this also sets
    /// [`Self::nonnegative`].
    #[must_use]
    pub fn positive(mut self) -> Self {
        self.positive = true;
        self.nonnegative = true;
        self
    }

    /// Assert the variable is non-negative (`x ≥ 0`).
    #[must_use]
    pub fn nonnegative(mut self) -> Self {
        self.nonnegative = true;
        self
    }

    /// Assert the variable is real (`x ∈ ℝ`).
    #[must_use]
    pub fn real(mut self) -> Self {
        self.real = true;
        self
    }

    /// Assert the variable is an integer (`x ∈ ℤ`).
    #[must_use]
    pub fn integer(mut self) -> Self {
        self.integer = true;
        self
    }

    /// Whether the variable is known to be non-negative (`x ≥ 0`).
    ///
    /// `true` when either [`Self::nonnegative`] or the stronger
    /// [`Self::positive`] is set.
    #[must_use]
    pub const fn is_nonnegative(&self) -> bool {
        self.nonnegative || self.positive
    }

    /// Whether the variable is known to be strictly positive (`x > 0`).
    #[must_use]
    pub const fn is_positive(&self) -> bool {
        self.positive
    }
}

/// A collection of per-variable [`VarAssumption`]s, keyed by variable index.
///
/// A variable that has never been [`set`](Self::set) is treated as fully unknown
/// ([`VarAssumption::default`]). The default `Assumptions` is empty, so
/// `op.simplify_with(&Assumptions::default())` applies only the identities that
/// are valid unconditionally on the whole real line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assumptions {
    vars: BTreeMap<usize, VarAssumption>,
}

impl Assumptions {
    /// An empty assumption set (every variable unknown).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The assumptions recorded for `var`, or [`VarAssumption::default`] when the
    /// variable is unknown.
    #[must_use]
    pub fn get(&self, var: usize) -> VarAssumption {
        self.vars.get(&var).copied().unwrap_or_default()
    }

    /// Record `assumption` for `var`, replacing any previous entry.
    pub fn set(&mut self, var: usize, assumption: VarAssumption) {
        self.vars.insert(var, assumption);
    }

    /// Builder form of [`set`](Self::set): record `assumption` for `var` and
    /// return the updated set.
    #[must_use]
    pub fn with(mut self, var: usize, assumption: VarAssumption) -> Self {
        self.vars.insert(var, assumption);
        self
    }

    /// Convenience builder: assert `var` is strictly positive.
    #[must_use]
    pub fn assume_positive(self, var: usize) -> Self {
        self.with(var, VarAssumption::new().positive())
    }

    /// Convenience builder: assert `var` is non-negative.
    #[must_use]
    pub fn assume_nonnegative(self, var: usize) -> Self {
        self.with(var, VarAssumption::new().nonnegative())
    }

    /// Convenience builder: assert `var` is an integer.
    #[must_use]
    pub fn assume_integer(self, var: usize) -> Self {
        self.with(var, VarAssumption::new().integer())
    }

    /// Whether no assumptions have been recorded at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_weakest() {
        let a = VarAssumption::default();
        assert!(!a.is_nonnegative());
        assert!(!a.is_positive());
        assert!(!a.real);
        assert!(!a.integer);
    }

    #[test]
    fn positive_implies_nonnegative() {
        let a = VarAssumption::new().positive();
        assert!(a.is_positive());
        assert!(a.is_nonnegative());
    }

    #[test]
    fn nonnegative_is_not_positive() {
        let a = VarAssumption::new().nonnegative();
        assert!(!a.is_positive());
        assert!(a.is_nonnegative());
    }

    #[test]
    fn unknown_var_is_default() {
        let asm = Assumptions::new().assume_nonnegative(0);
        assert!(asm.get(0).is_nonnegative());
        assert_eq!(asm.get(1), VarAssumption::default());
        assert!(!asm.get(1).is_nonnegative());
    }

    #[test]
    fn builder_roundtrip() {
        let asm = Assumptions::new()
            .assume_positive(2)
            .with(5, VarAssumption::new().integer().nonnegative());
        assert!(asm.get(2).is_positive());
        assert!(asm.get(5).integer);
        assert!(asm.get(5).is_nonnegative());
        assert!(!asm.is_empty());
    }
}
