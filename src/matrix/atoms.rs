//! Atomization: viewing an arbitrary [`LoweredOp`] as a **polynomial over opaque atoms**.
//!
//! # Why
//!
//! Every exact algorithm in this module (Bareiss, Faddeev–LeVerrier, fraction-free
//! elimination) needs its entries to live in an *integral domain* with an exact
//! zero test and an exact division. `LoweredOp` is not such a thing: it is a raw
//! expression tree containing `exp`, `ln`, `sin`, … and deciding whether two such
//! trees denote the same function is undecidable in general (Richardson's theorem).
//!
//! The way out is the classical one: **polynomialize**. Pick a finite set of
//! *atoms* `a₁ … a_m` — the variables `x_i` plus every maximal non-polynomial
//! subexpression (`exp(x)`, `sin(x+1)`, `1/(x−y)`, …) — and read each matrix entry
//! as an element of the polynomial ring
//!
//! ```text
//! R = ℚ[a₁, …, a_m]
//! ```
//!
//! `R` *is* an integral domain with an exact zero test (a polynomial is zero iff
//! all of its coefficients vanish) and exact division, so all the fraction-free
//! machinery applies verbatim.
//!
//! # What this buys, and what it does not
//!
//! Let `φ : R → (ℝⁿ → ℝ)` be the evaluation map that sends each atom to the
//! function it denotes. `φ` is a **ring homomorphism**. Two consequences, and they
//! are the two load-bearing facts of this whole module:
//!
//! 1. **Determinant / charpoly are always right.** `det` is a *polynomial* in the
//!    matrix entries, so it commutes with `φ`:
//!    `φ(det_R(A)) = det(φ(A))`. Computing the determinant in `R` — where every
//!    zero test is exact — therefore yields an expression that is *provably* the
//!    determinant of the original transcendental matrix, no matter how the atoms
//!    are related to each other. The same holds for Faddeev–LeVerrier, since it is
//!    pure ring arithmetic (plus divisions by the integers `1..n`).
//!
//! 2. **`φ` is not injective — this is where undecidability lives.** The atoms are
//!    *not* algebraically independent: `sin(x)² + cos(x)² − 1` is a nonzero element
//!    of `R` (with atoms `s = sin x`, `c = cos x` it is `s² + c² − 1`) but `φ` sends
//!    it to the zero function. Hence
//!
//!    * `p = 0` in `R`  ⟹  `φ(p) = 0`  — *sound*, this is tier 1 of the oracle;
//!    * `p ≠ 0` in `R`  ⟹̸  `φ(p) ≠ 0` — **unsound**, and no amount of cleverness
//!      fixes it. That gap is exactly what [`super::zero`] must confront.
//!
//! Any algorithm that *divides* by an entry (rref, inverse, nullspace) needs to
//! know that `φ(p) ≠ 0`, not merely that `p ≠ 0` in `R`, and so must consult the
//! zero oracle. Any algorithm that only *multiplies and adds* (det, charpoly) does
//! not, and is unconditionally exact.

use std::sync::Arc;

use crate::lower::LoweredOp;
use crate::poly::{Coeff, MultiPoly, coeff_is_zero, coeff_one, coeff_recip, f64_to_ratio};

use super::MatrixError;

/// Largest integer exponent expanded structurally by [`AtomSpace`].
///
/// `x^70` would expand into a degree-70 polynomial; beyond this bound the power is
/// kept as an opaque atom instead, which keeps the term count finite.
const MAX_STRUCTURAL_EXPONENT: f64 = 64.0;

/// The syntactic shape of a `LoweredOp` node as seen by the polynomializer.
///
/// [`AtomSpace::collect`] and [`AtomSpace::polynomialize`] are two passes over the
/// *same* classification, so they are guaranteed to agree on which subexpressions
/// become atoms.
enum Shape<'a> {
    /// A literal constant (`Const` or `NamedConst`).
    Constant(f64),
    /// A variable `x_i` — an atom in its own right.
    Variable(usize),
    /// `a + b`.
    Sum(&'a LoweredOp, &'a LoweredOp),
    /// `a − b`.
    Difference(&'a LoweredOp, &'a LoweredOp),
    /// `a · b`.
    Product(&'a LoweredOp, &'a LoweredOp),
    /// `−a`.
    Negation(&'a LoweredOp),
    /// `base^n` with `n` a non-negative integer ≤ [`MAX_STRUCTURAL_EXPONENT`].
    NaturalPower(&'a LoweredOp, u32),
    /// `base^(−n)` — becomes `recip(base)^n` with `recip(base)` an atom.
    ReciprocalPower(&'a LoweredOp, u32),
    /// `a / c` with `c` a nonzero literal constant — a plain rational scaling.
    ConstantQuotient(&'a LoweredOp, f64),
    /// `a / b` with `b` non-constant — becomes `a · recip(b)` with `recip(b)` an atom.
    ReciprocalQuotient(&'a LoweredOp, &'a LoweredOp),
    /// Anything else (`exp`, `ln`, `sin`, `x^y`, …): the whole node is one atom.
    Opaque,
}

/// Sum a slice of leaves into a **balanced** binary `Add` tree (depth `O(log n)`).
///
/// A left-leaning fold would give depth `O(n)`; for the many-term expressions this
/// module produces (determinants, adjugates) that depth is exactly what overflows
/// the stack in later recursive passes.
fn balanced_sum(leaves: &[LoweredOp]) -> LoweredOp {
    match leaves {
        [] => LoweredOp::Const(0.0),
        [only] => only.clone(),
        _ => {
            let mid = leaves.len() / 2;
            LoweredOp::Add(
                Arc::new(balanced_sum(&leaves[..mid])),
                Arc::new(balanced_sum(&leaves[mid..])),
            )
        }
    }
}

/// The reciprocal atom `1 / b`.
fn reciprocal_of(base: &LoweredOp) -> LoweredOp {
    LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(base.clone()))
}

/// Classify a node. Both atomization passes dispatch through this one function.
fn classify(expr: &LoweredOp) -> Shape<'_> {
    match expr {
        LoweredOp::Const(c) => Shape::Constant(*c),
        LoweredOp::NamedConst(nc) => Shape::Constant(nc.value()),
        LoweredOp::Var(i) => Shape::Variable(*i),
        LoweredOp::Add(a, b) => Shape::Sum(a, b),
        LoweredOp::Sub(a, b) => Shape::Difference(a, b),
        LoweredOp::Mul(a, b) => Shape::Product(a, b),
        LoweredOp::Neg(a) => Shape::Negation(a),
        LoweredOp::Pow(base, exponent) => match exponent.as_ref() {
            LoweredOp::Const(e)
                if e.fract() == 0.0 && *e >= 0.0 && *e <= MAX_STRUCTURAL_EXPONENT =>
            {
                Shape::NaturalPower(base, *e as u32)
            }
            LoweredOp::Const(e)
                if e.fract() == 0.0 && *e < 0.0 && -*e <= MAX_STRUCTURAL_EXPONENT =>
            {
                Shape::ReciprocalPower(base, (-*e) as u32)
            }
            _ => Shape::Opaque,
        },
        LoweredOp::Div(a, b) => match b.as_ref() {
            LoweredOp::Const(c) if *c != 0.0 => Shape::ConstantQuotient(a, *c),
            LoweredOp::NamedConst(nc) if nc.value() != 0.0 => {
                Shape::ConstantQuotient(a, nc.value())
            }
            LoweredOp::Const(_) | LoweredOp::NamedConst(_) => Shape::Opaque,
            _ => Shape::ReciprocalQuotient(a, b),
        },
        _ => Shape::Opaque,
    }
}

/// The finite set of atoms a family of expressions is polynomial in.
///
/// Atoms are interned in first-encounter order (row-major over the matrix), so the
/// atom indices — and therefore every polynomial and every printed result — are
/// fully deterministic.
#[derive(Clone, Debug, Default)]
pub struct AtomSpace {
    atoms: Vec<LoweredOp>,
    n_vars: usize,
}

impl AtomSpace {
    /// An empty atom space.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the atom space spanned by `exprs` (interned in slice order).
    #[must_use]
    pub fn from_exprs(exprs: &[LoweredOp]) -> Self {
        let mut space = Self::new();
        for e in exprs {
            space.collect(e);
        }
        space
    }

    /// Number of atoms — equivalently, the number of variables of the polynomial
    /// ring `ℚ[a₁ … a_m]`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// `true` when no atoms were interned (the expressions were all constants).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// The expression denoted by atom `index`.
    #[must_use]
    pub fn atom(&self, index: usize) -> Option<&LoweredOp> {
        self.atoms.get(index)
    }

    /// All atoms, in index order.
    #[must_use]
    pub fn atoms(&self) -> &[LoweredOp] {
        &self.atoms
    }

    /// `true` when atom `index` is a bare variable `x_i` rather than a
    /// transcendental (or otherwise opaque) subexpression.
    ///
    /// A polynomial supported only on variable atoms is an honest polynomial in the
    /// real variables, and its zero test is therefore *exact in both directions*.
    #[must_use]
    pub fn is_variable_atom(&self, index: usize) -> bool {
        matches!(self.atoms.get(index), Some(LoweredOp::Var(_)))
    }

    /// The number of real variables `x₀ … x_{n−1}` that appear anywhere (including
    /// *inside* opaque atoms). This is the length a probe point must have.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Intern `expr` as an atom, returning its index.
    fn intern(&mut self, expr: LoweredOp) -> usize {
        if let Some(pos) = self.atoms.iter().position(|a| *a == expr) {
            return pos;
        }
        self.n_vars = self.n_vars.max(expr.count_vars());
        self.atoms.push(expr);
        self.atoms.len() - 1
    }

    /// Index of an already-interned atom.
    fn index_of(&self, expr: &LoweredOp) -> Option<usize> {
        self.atoms.iter().position(|a| a == expr)
    }

    /// Pass 1 — walk `expr` and intern every atom it needs.
    fn collect(&mut self, expr: &LoweredOp) {
        match classify(expr) {
            Shape::Constant(_) => {}
            Shape::Variable(i) => {
                self.intern(LoweredOp::Var(i));
            }
            Shape::Sum(a, b) | Shape::Difference(a, b) | Shape::Product(a, b) => {
                self.collect(a);
                self.collect(b);
            }
            Shape::Negation(a) | Shape::ConstantQuotient(a, _) => self.collect(a),
            Shape::NaturalPower(base, _) => self.collect(base),
            Shape::ReciprocalPower(base, _) => {
                self.intern(reciprocal_of(base));
            }
            Shape::ReciprocalQuotient(a, b) => {
                self.collect(a);
                self.intern(reciprocal_of(b));
            }
            Shape::Opaque => {
                self.intern(expr.clone());
            }
        }
    }

    /// Pass 2 — read `expr` as an element of `ℚ[atoms]`.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::NonRationalConstant`] if the expression contains a
    /// non-finite literal (`inf`/`NaN`), which has no rational value, and
    /// [`MatrixError::UnknownAtom`] if `expr` contains an atom this space does not
    /// know (only possible when the space was not built from `expr`).
    pub fn polynomialize(&self, expr: &LoweredOp) -> Result<MultiPoly, MatrixError> {
        let n = self.len();
        match classify(expr) {
            Shape::Constant(c) => Ok(MultiPoly::constant(
                f64_to_ratio(c).map_err(|_| MatrixError::NonRationalConstant(c))?,
                n,
            )),
            Shape::Variable(i) => self.atom_monomial(&LoweredOp::Var(i), 1),
            Shape::Sum(a, b) => {
                let pa = self.polynomialize(a)?;
                let pb = self.polynomialize(b)?;
                pa.add(&pb).map_err(MatrixError::Poly)
            }
            Shape::Difference(a, b) => {
                let pa = self.polynomialize(a)?;
                let pb = self.polynomialize(b)?;
                pa.sub(&pb).map_err(MatrixError::Poly)
            }
            Shape::Product(a, b) => {
                let pa = self.polynomialize(a)?;
                let pb = self.polynomialize(b)?;
                pa.mul(&pb).map_err(MatrixError::Poly)
            }
            Shape::Negation(a) => self.polynomialize(a)?.neg().map_err(MatrixError::Poly),
            Shape::NaturalPower(base, e) => self
                .polynomialize(base)?
                .pow(e as usize)
                .map_err(MatrixError::Poly),
            Shape::ReciprocalPower(base, e) => self.atom_monomial(&reciprocal_of(base), e),
            Shape::ConstantQuotient(a, c) => {
                let inv =
                    coeff_recip(&f64_to_ratio(c).map_err(|_| MatrixError::NonRationalConstant(c))?)
                        .ok_or(MatrixError::NonRationalConstant(c))?;
                self.polynomialize(a)?
                    .scale(&inv)
                    .map_err(MatrixError::Poly)
            }
            Shape::ReciprocalQuotient(a, b) => {
                let pa = self.polynomialize(a)?;
                let pb = self.atom_monomial(&reciprocal_of(b), 1)?;
                pa.mul(&pb).map_err(MatrixError::Poly)
            }
            Shape::Opaque => self.atom_monomial(expr, 1),
        }
    }

    /// The monomial `atom^exponent`.
    fn atom_monomial(&self, atom: &LoweredOp, exponent: u32) -> Result<MultiPoly, MatrixError> {
        let idx = self
            .index_of(atom)
            .ok_or_else(|| MatrixError::UnknownAtom(Box::new(atom.clone())))?;
        let n = self.len();
        let mut exp = vec![0u32; n];
        exp[idx] = exponent;
        let mut poly = MultiPoly::zero(n);
        poly.terms.insert(exp, coeff_one());
        Ok(poly)
    }

    /// Substitute the atoms back and render `poly` as a `LoweredOp`.
    ///
    /// Terms are emitted in a deterministic, human-friendly order: descending total
    /// degree, then descending lexicographic exponent. The sum is assembled as a
    /// **balanced** binary tree so that a polynomial with `t` terms has depth
    /// `O(log t)` rather than `O(t)` — a determinant can have thousands of terms, and
    /// a left-leaning chain would make later `simplify`/`eval` recursions overflow
    /// the stack.
    ///
    /// The rational coefficients are rounded to `f64` here — `LoweredOp` has no
    /// exact-rational literal. Every *computation* in this module is exact; only
    /// this final rendering step rounds. Use [`super::Matrix::det_exact`] /
    /// [`super::Matrix::charpoly_exact`] when the exact rational is needed.
    #[must_use]
    pub fn poly_to_lowered(&self, poly: &MultiPoly) -> LoweredOp {
        let mut terms: Vec<(&Vec<u32>, &Coeff)> = poly
            .terms
            .iter()
            .filter(|(_, c)| !coeff_is_zero(c))
            .collect();
        if terms.is_empty() {
            return LoweredOp::Const(0.0);
        }
        terms.sort_by(|(ea, _), (eb, _)| {
            let da: u32 = ea.iter().sum();
            let db: u32 = eb.iter().sum();
            db.cmp(&da).then_with(|| eb.cmp(ea))
        });

        // Each term becomes a signed leaf (negatives wrapped in `Neg`), then the
        // leaves are summed pairwise up a balanced tree.
        let leaves: Vec<LoweredOp> = terms
            .into_iter()
            .map(|(exps, coeff)| {
                let value = crate::poly::ratio_to_f64(coeff);
                let magnitude = self.render_term(exps, value.abs());
                if value < 0.0 {
                    LoweredOp::Neg(Arc::new(magnitude))
                } else {
                    magnitude
                }
            })
            .collect();
        balanced_sum(&leaves)
    }

    /// Render `|coeff| · ∏ atomᵢ^eᵢ` (the caller supplies the sign).
    fn render_term(&self, exps: &[u32], magnitude: f64) -> LoweredOp {
        let mut factors: Vec<LoweredOp> = Vec::new();
        if magnitude != 1.0 {
            factors.push(LoweredOp::Const(magnitude));
        }
        for (idx, &e) in exps.iter().enumerate() {
            if e == 0 {
                continue;
            }
            // An out-of-range atom index cannot occur: `exps` always has length
            // `self.len()`. Falling back to `Const(1.0)` keeps this total.
            let atom = self
                .atoms
                .get(idx)
                .cloned()
                .unwrap_or(LoweredOp::Const(1.0));
            if e == 1 {
                factors.push(atom);
            } else {
                factors.push(LoweredOp::Pow(
                    Arc::new(atom),
                    Arc::new(LoweredOp::Const(f64::from(e))),
                ));
            }
        }
        let mut iter = factors.into_iter();
        let Some(first) = iter.next() else {
            return LoweredOp::Const(magnitude);
        };
        iter.fold(first, |acc, f| LoweredOp::Mul(Arc::new(acc), Arc::new(f)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    #[test]
    fn polynomial_expression_uses_only_variable_atoms() {
        // x0 * x1 + 2
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Mul(Arc::new(var(0)), Arc::new(var(1)))),
            Arc::new(LoweredOp::Const(2.0)),
        );
        let space = AtomSpace::from_exprs(std::slice::from_ref(&expr));
        assert_eq!(space.len(), 2);
        assert!(space.is_variable_atom(0));
        assert!(space.is_variable_atom(1));
        let p = space.polynomialize(&expr).expect("polynomializable");
        assert_eq!(p.terms.len(), 2);
    }

    #[test]
    fn transcendental_becomes_opaque_atom() {
        // exp(x0) + x0
        let expr = LoweredOp::Add(Arc::new(LoweredOp::Exp(Arc::new(var(0)))), Arc::new(var(0)));
        let space = AtomSpace::from_exprs(std::slice::from_ref(&expr));
        assert_eq!(space.len(), 2, "atoms: exp(x0) and x0");
        assert!(!space.is_variable_atom(0), "exp(x0) is opaque");
        assert!(space.is_variable_atom(1));
        assert_eq!(space.n_vars(), 1);
    }

    #[test]
    fn round_trip_preserves_value() {
        // (x0 - 3*x1)^2
        let inner = LoweredOp::Sub(
            Arc::new(var(0)),
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(3.0)),
                Arc::new(var(1)),
            )),
        );
        let expr = LoweredOp::Pow(Arc::new(inner), Arc::new(LoweredOp::Const(2.0)));
        let space = AtomSpace::from_exprs(std::slice::from_ref(&expr));
        let p = space.polynomialize(&expr).expect("polynomializable");
        let back = space.poly_to_lowered(&p);
        for point in [[1.0, 2.0], [-0.5, 3.25], [7.0, 0.0]] {
            let a = expr.eval(&point);
            let b = back.eval(&point);
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn division_becomes_reciprocal_atom() {
        // x0 / x1
        let expr = LoweredOp::Div(Arc::new(var(0)), Arc::new(var(1)));
        let space = AtomSpace::from_exprs(std::slice::from_ref(&expr));
        // atoms: x0, 1/x1
        assert_eq!(space.len(), 2);
        let p = space.polynomialize(&expr).expect("polynomializable");
        let back = space.poly_to_lowered(&p);
        let point = [3.0, 4.0];
        assert!((back.eval(&point) - 0.75).abs() < 1e-12);
    }
}
