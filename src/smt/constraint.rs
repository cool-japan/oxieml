use crate::tree::EmlTree;
use std::fmt;

// ----------------------------------------------------------------------------
// Public constraint types (unchanged public API)
// ----------------------------------------------------------------------------

/// An EML-based constraint for SMT solving.
#[derive(Clone, Debug)]
pub enum EmlConstraint {
    /// `eml_expr == 0`
    EqZero(EmlTree),
    /// `eml_expr > 0`
    GtZero(EmlTree),
    /// `eml_expr >= 0`
    GeZero(EmlTree),
    /// `eml_expr < 0`
    LtZero(EmlTree),
    /// `eml_expr <= 0`
    LeZero(EmlTree),
    /// `eml_expr ≠ 0`
    NeZero(EmlTree),
    /// Negation of a constraint.
    Not(Box<EmlConstraint>),
    /// Conjunction of constraints.
    And(Vec<EmlConstraint>),
    /// Disjunction of constraints.
    Or(Vec<EmlConstraint>),
    /// ∀ var ∈ [lo, hi]. body
    ForAll {
        /// Variable index being universally quantified.
        var: usize,
        /// Lower bound of the variable's range.
        lo: f64,
        /// Upper bound of the variable's range.
        hi: f64,
        /// Body constraint (must hold for all values of `var` in `[lo, hi]`).
        body: Box<EmlConstraint>,
    },
    /// ∃ var ∈ [lo, hi]. body
    Exists {
        /// Variable index being existentially quantified.
        var: usize,
        /// Lower bound of the variable's range.
        lo: f64,
        /// Upper bound of the variable's range.
        hi: f64,
        /// Body constraint (must hold for some value of `var` in `[lo, hi]`).
        body: Box<EmlConstraint>,
    },
}

impl fmt::Display for EmlConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmlConstraint::EqZero(t) => write!(f, "({t} == 0)"),
            EmlConstraint::GtZero(t) => write!(f, "({t} > 0)"),
            EmlConstraint::GeZero(t) => write!(f, "({t} >= 0)"),
            EmlConstraint::LtZero(t) => write!(f, "({t} < 0)"),
            EmlConstraint::LeZero(t) => write!(f, "({t} <= 0)"),
            EmlConstraint::NeZero(t) => write!(f, "({t} != 0)"),
            EmlConstraint::Not(inner) => write!(f, "¬{inner}"),
            EmlConstraint::And(cs) => {
                write!(f, "(and")?;
                for c in cs {
                    write!(f, " {c}")?;
                }
                write!(f, ")")
            }
            EmlConstraint::Or(cs) => {
                write!(f, "(or")?;
                for c in cs {
                    write!(f, " {c}")?;
                }
                write!(f, ")")
            }
            EmlConstraint::ForAll { var, lo, hi, body } => {
                write!(f, "∀x{var}∈[{lo},{hi}].{body}")
            }
            EmlConstraint::Exists { var, lo, hi, body } => {
                write!(f, "∃x{var}∈[{lo},{hi}].{body}")
            }
        }
    }
}

/// A solution to an EML constraint system.
#[derive(Clone, Debug)]
pub struct EmlSolution {
    /// Variable assignments that satisfy the constraints.
    pub assignments: Vec<f64>,
    /// Whether the solution is exact or approximate.
    pub is_exact: bool,
}

impl EmlConstraint {
    /// `a < b`: equivalent to `LtZero(a - b)`.
    pub fn lt(a: EmlTree, b: EmlTree) -> Self {
        Self::LtZero(crate::canonical::Canonical::sub(&a, &b))
    }
    /// `a <= b`
    pub fn le(a: EmlTree, b: EmlTree) -> Self {
        Self::LeZero(crate::canonical::Canonical::sub(&a, &b))
    }
    /// `a > b`
    pub fn gt(a: EmlTree, b: EmlTree) -> Self {
        Self::GtZero(crate::canonical::Canonical::sub(&a, &b))
    }
    /// `a >= b`
    pub fn ge(a: EmlTree, b: EmlTree) -> Self {
        Self::GeZero(crate::canonical::Canonical::sub(&a, &b))
    }
    /// `a == b`
    pub fn eq(a: EmlTree, b: EmlTree) -> Self {
        Self::EqZero(crate::canonical::Canonical::sub(&a, &b))
    }
    /// `a != b`
    pub fn ne(a: EmlTree, b: EmlTree) -> Self {
        Self::NeZero(crate::canonical::Canonical::sub(&a, &b))
    }
    /// Push `Not` inward to negation-normal form (NNF).
    /// After calling this, `Not` only appears over atoms (never over And/Or).
    pub fn to_nnf(self) -> Self {
        match self {
            Self::Not(inner) => negate(*inner),
            Self::And(cs) => Self::And(cs.into_iter().map(Self::to_nnf).collect()),
            Self::Or(cs) => Self::Or(cs.into_iter().map(Self::to_nnf).collect()),
            Self::ForAll { var, lo, hi, body } => Self::ForAll {
                var,
                lo,
                hi,
                body: Box::new(body.to_nnf()),
            },
            Self::Exists { var, lo, hi, body } => Self::Exists {
                var,
                lo,
                hi,
                body: Box::new(body.to_nnf()),
            },
            other => other,
        }
    }
}

/// Negate a constraint into NNF.
fn negate(c: EmlConstraint) -> EmlConstraint {
    match c {
        EmlConstraint::EqZero(t) => EmlConstraint::NeZero(t),
        EmlConstraint::GtZero(t) => EmlConstraint::LeZero(t),
        EmlConstraint::GeZero(t) => EmlConstraint::LtZero(t),
        EmlConstraint::LtZero(t) => EmlConstraint::GeZero(t),
        EmlConstraint::LeZero(t) => EmlConstraint::GtZero(t),
        EmlConstraint::NeZero(t) => EmlConstraint::EqZero(t),
        EmlConstraint::Not(inner) => inner.to_nnf(), // double negation
        EmlConstraint::And(cs) => EmlConstraint::Or(cs.into_iter().map(negate).collect()),
        EmlConstraint::Or(cs) => EmlConstraint::And(cs.into_iter().map(negate).collect()),
        // ¬∀x∈B.φ  ⇒  ∃x∈B.¬φ
        EmlConstraint::ForAll { var, lo, hi, body } => EmlConstraint::Exists {
            var,
            lo,
            hi,
            body: Box::new(negate(*body)),
        },
        // ¬∃x∈B.φ  ⇒  ∀x∈B.¬φ
        EmlConstraint::Exists { var, lo, hi, body } => EmlConstraint::ForAll {
            var,
            lo,
            hi,
            body: Box::new(negate(*body)),
        },
    }
}
