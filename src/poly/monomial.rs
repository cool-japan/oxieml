//! Monomials and monomial orderings for multivariate polynomial algebra.
//!
//! A *monomial* in `n` variables `x₀, …, x_{n−1}` is a product
//! `x^α = x₀^α₀ · x₁^α₁ ⋯ x_{n−1}^α_{n−1}` and is completely described by its
//! exponent vector `α ∈ ℕⁿ`. [`Monomial`] is a thin newtype over that vector.
//!
//! # Monomial orders
//!
//! A **monomial order** `>` on ℕⁿ is a relation satisfying
//!
//! 1. `>` is a total (linear) order on ℕⁿ;
//! 2. it is *compatible with multiplication*: `α > β  ⟹  α + γ > β + γ` for all
//!    `γ ∈ ℕⁿ`;
//! 3. it is a *well-ordering*: every non-empty subset of ℕⁿ has a least element.
//!
//! Property (3) is what makes the multivariate division algorithm and
//! Buchberger's algorithm terminate: each rewriting step strictly lowers the
//! leading monomial, and an infinite strictly-decreasing chain cannot exist.
//! For orders satisfying (1) and (2), property (3) is equivalent to
//! `α ≥ 0` for all `α` (Dickson's lemma), which all three orders below satisfy.
//!
//! [`MonOrder`] provides the three standard orders, all with the convention
//! `x₀ > x₁ > … > x_{n−1}` (variable index 0 is the *most significant*).
//!
//! ## Lexicographic (`Lex`)
//!
//! `α >_lex β` iff the **leftmost** nonzero entry of `α − β` is positive.
//!
//! Lex is an *elimination order*: for any `k`, the monomials involving only
//! `x_k, …, x_{n−1}` are exactly the ones smaller than every monomial that
//! contains some `x_j` with `j < k`. This is precisely the property the
//! Elimination Theorem needs, which is why [`crate::poly::solve_system`]
//! computes its Gröbner basis under `Lex`.
//!
//! ## Graded lexicographic (`GrLex`)
//!
//! `α >_grlex β` iff `|α| > |β|`, or `|α| = |β|` and `α >_lex β`, where
//! `|α| = Σ αᵢ` is the total degree.
//!
//! ## Graded reverse lexicographic (`GrevLex`)
//!
//! `α >_grevlex β` iff `|α| > |β|`, or `|α| = |β|` and the **rightmost** nonzero
//! entry of `α − β` is **negative**.
//!
//! The double reversal is not a typo: grevlex breaks degree ties by *pushing
//! down* the monomials that use the last variables most heavily. Grevlex is
//! almost always the cheapest order for Buchberger's algorithm, but it is *not*
//! an elimination order, so it cannot be used for triangularizing a system.
//!
//! Note that lex and grevlex genuinely differ only for `n ≥ 3`: in two variables
//! with equal total degree, "leftmost difference positive" and "rightmost
//! difference negative" are the same condition.

use std::cmp::Ordering;

/// A monomial `x^α`, represented by its exponent vector `α ∈ ℕⁿ`.
///
/// The vector always has length `n` (the number of variables); a zero entry
/// means the corresponding variable does not occur.
///
/// The derived [`Ord`] is *not* one of the monomial orders — it is the ordinary
/// lexicographic order on `Vec<u32>`, used only so monomials can be `BTreeMap`
/// keys. Always compare with [`MonOrder::cmp_monomials`] when the monomial order
/// matters.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Monomial(pub Vec<u32>);

impl Monomial {
    /// Wrap an exponent vector.
    #[must_use]
    pub fn new(exponents: Vec<u32>) -> Self {
        Self(exponents)
    }

    /// The monomial `1` (all exponents zero) in `num_vars` variables.
    #[must_use]
    pub fn one(num_vars: usize) -> Self {
        Self(vec![0u32; num_vars])
    }

    /// The exponent vector.
    #[must_use]
    pub fn exponents(&self) -> &[u32] {
        &self.0
    }

    /// The number of variables (the length of the exponent vector).
    #[must_use]
    pub fn num_vars(&self) -> usize {
        self.0.len()
    }

    /// Total degree `|α| = Σ αᵢ`.
    ///
    /// Accumulated in `u64` so that a long exponent vector cannot overflow.
    #[must_use]
    pub fn total_degree(&self) -> u64 {
        total_degree(&self.0)
    }

    /// Return `true` when this is the monomial `1`.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.0.iter().all(|&e| e == 0)
    }

    /// Product `x^α · x^β = x^(α+β)`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        Self(mul_exponents(&self.0, &other.0))
    }

    /// Return `true` when `x^self` divides `x^other`, i.e. `αᵢ ≤ βᵢ` for all `i`.
    #[must_use]
    pub fn divides(&self, other: &Self) -> bool {
        divides(&self.0, &other.0)
    }

    /// Exact quotient `x^other / x^self`, or `None` when `x^self ∤ x^other`.
    #[must_use]
    pub fn div(&self, other: &Self) -> Option<Self> {
        div_exponents(&self.0, &other.0).map(Self)
    }

    /// Least common multiple: `lcm(α, β)ᵢ = max(αᵢ, βᵢ)`.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Self {
        Self(lcm_exponents(&self.0, &other.0))
    }

    /// Greatest common divisor: `gcd(α, β)ᵢ = min(αᵢ, βᵢ)`.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| (*a).min(*b))
                .collect(),
        )
    }

    /// Return `true` when the two monomials share no variable, i.e.
    /// `gcd(x^α, x^β) = 1`, equivalently `lcm(α, β) = α + β`.
    ///
    /// This is the hypothesis of **Buchberger's first (product) criterion**.
    #[must_use]
    pub fn is_coprime(&self, other: &Self) -> bool {
        is_coprime(&self.0, &other.0)
    }

    /// If this monomial is a pure power `x_i^k` with `k ≥ 1`, return `Some(i)`.
    ///
    /// Used by the Finiteness Theorem check in
    /// [`crate::poly::groebner::is_zero_dimensional`].
    #[must_use]
    pub fn pure_power_var(&self) -> Option<usize> {
        let mut found: Option<usize> = None;
        for (i, &e) in self.0.iter().enumerate() {
            if e > 0 {
                if found.is_some() {
                    return None;
                }
                found = Some(i);
            }
        }
        found
    }
}

// ── Slice-level helpers ───────────────────────────────────────────────────────
//
// `MultiPoly` stores its terms in a `BTreeMap<Vec<u32>, Coeff>`, so these
// operate directly on exponent slices to avoid wrapping/unwrapping `Monomial`
// in the inner loops of the division algorithm and Buchberger's algorithm.

/// Total degree of an exponent slice, accumulated in `u64`.
#[must_use]
pub fn total_degree(exps: &[u32]) -> u64 {
    exps.iter().map(|&e| u64::from(e)).sum()
}

/// Exponent vector of the product `x^a · x^b`.
#[must_use]
pub fn mul_exponents(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.saturating_add(*y))
        .collect()
}

/// Return `true` when `x^a` divides `x^b`.
#[must_use]
pub fn divides(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x <= y)
}

/// Exponent vector of the exact quotient `x^b / x^a`, or `None` when `x^a ∤ x^b`.
#[must_use]
pub fn div_exponents(a: &[u32], b: &[u32]) -> Option<Vec<u32>> {
    if !divides(a, b) {
        return None;
    }
    Some(a.iter().zip(b.iter()).map(|(x, y)| y - x).collect())
}

/// Exponent vector of `lcm(x^a, x^b)`.
#[must_use]
pub fn lcm_exponents(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| (*x).max(*y)).collect()
}

/// Return `true` when `x^a` and `x^b` have no variable in common.
#[must_use]
pub fn is_coprime(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| *x == 0 || *y == 0)
}

// ── Monomial orders ───────────────────────────────────────────────────────────

/// The monomial order used to pick leading terms.
///
/// All three orders use the variable convention `x₀ > x₁ > … > x_{n−1}`.
/// See the [module documentation](self) for the definitions and for why `Lex` is
/// the only one of the three that supports elimination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MonOrder {
    /// Lexicographic order. The elimination order.
    #[default]
    Lex,
    /// Graded lexicographic order: total degree first, ties broken by `Lex`.
    GrLex,
    /// Graded reverse lexicographic order: total degree first, ties broken by
    /// the *rightmost* differing exponent, *smaller wins*.
    GrevLex,
}

impl MonOrder {
    /// Compare two exponent vectors under this order.
    ///
    /// Returns [`Ordering::Greater`] when `a` is the larger monomial. Vectors of
    /// unequal length are compared by length first; in practice every polynomial
    /// keeps all its exponent vectors at the same length, so that branch is only
    /// a defensive total-order guarantee.
    #[must_use]
    pub fn cmp_exponents(self, a: &[u32], b: &[u32]) -> Ordering {
        if a.len() != b.len() {
            return a.len().cmp(&b.len());
        }
        match self {
            Self::Lex => cmp_lex(a, b),
            Self::GrLex => total_degree(a)
                .cmp(&total_degree(b))
                .then_with(|| cmp_lex(a, b)),
            Self::GrevLex => total_degree(a)
                .cmp(&total_degree(b))
                .then_with(|| cmp_revlex_tiebreak(a, b)),
        }
    }

    /// Compare two [`Monomial`]s under this order.
    #[must_use]
    pub fn cmp_monomials(self, a: &Monomial, b: &Monomial) -> Ordering {
        self.cmp_exponents(&a.0, &b.0)
    }

    /// Human-readable name, used in error messages and documentation.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Lex => "lex",
            Self::GrLex => "grlex",
            Self::GrevLex => "grevlex",
        }
    }

    /// Return `true` when this order supports the Elimination Theorem, i.e. when
    /// a Gröbner basis under it triangularizes the system.
    ///
    /// Only [`MonOrder::Lex`] does. The graded orders compare total degree first,
    /// so a high-degree monomial in the *last* variable alone can outrank a
    /// low-degree monomial containing the *first* variable — which destroys the
    /// property the Elimination Theorem relies on.
    #[must_use]
    pub fn is_elimination_order(self) -> bool {
        matches!(self, Self::Lex)
    }
}

/// Lexicographic comparison: leftmost differing exponent decides, larger wins.
fn cmp_lex(a: &[u32], b: &[u32]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match x.cmp(y) {
            Ordering::Equal => {}
            non_equal => return non_equal,
        }
    }
    Ordering::Equal
}

/// Grevlex degree-tie-break: rightmost differing exponent decides, **smaller wins**.
///
/// Equivalently: the rightmost nonzero entry of `a − b` must be negative for `a`
/// to be the greater monomial.
fn cmp_revlex_tiebreak(a: &[u32], b: &[u32]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()).rev() {
        match x.cmp(y) {
            Ordering::Equal => {}
            // Reversed on purpose: the smaller exponent in the last differing
            // position makes the monomial *greater*.
            Ordering::Less => return Ordering::Greater,
            Ordering::Greater => return Ordering::Less,
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(e: &[u32]) -> Monomial {
        Monomial::new(e.to_vec())
    }

    #[test]
    fn lex_orders_by_leftmost_difference() {
        // x0 > x1 > x2, so x0 beats any power of x1/x2.
        assert_eq!(
            MonOrder::Lex.cmp_exponents(&[1, 0, 0], &[0, 5, 7]),
            Ordering::Greater
        );
        // x0^2 x1 > x0^2  (tie on x0, then x1 decides)
        assert_eq!(
            MonOrder::Lex.cmp_exponents(&[2, 1, 0], &[2, 0, 3]),
            Ordering::Greater
        );
        assert_eq!(
            MonOrder::Lex.cmp_exponents(&[1, 2, 3], &[1, 2, 3]),
            Ordering::Equal
        );
    }

    #[test]
    fn grlex_orders_by_total_degree_first() {
        // |(0,5,7)| = 12 > |(1,0,0)| = 1, so the x0 monomial now loses.
        assert_eq!(
            MonOrder::GrLex.cmp_exponents(&[1, 0, 0], &[0, 5, 7]),
            Ordering::Less
        );
        // Equal degree 5: lex tie-break, leftmost difference wins.
        assert_eq!(
            MonOrder::GrLex.cmp_exponents(&[3, 2, 0], &[3, 1, 1]),
            Ordering::Greater
        );
    }

    #[test]
    fn grevlex_tiebreak_is_rightmost_smaller_wins() {
        // The textbook discriminating example (Cox–Little–O'Shea, Ch. 2 §2):
        //   x0 x1^2 x2^3  vs  x0^3 x1 x2   — hmm, different degrees; use degree 5.
        // α = (1,2,2), β = (3,1,1): both total degree 5.
        //   grlex: leftmost difference is index 0, 1 < 3 → α < β.
        //   grevlex: rightmost difference is index 2, 2 > 1 → α < β as well.
        assert_eq!(
            MonOrder::GrLex.cmp_exponents(&[1, 2, 2], &[3, 1, 1]),
            Ordering::Less
        );
        assert_eq!(
            MonOrder::GrevLex.cmp_exponents(&[1, 2, 2], &[3, 1, 1]),
            Ordering::Less
        );

        // The classic case where grlex and grevlex genuinely disagree:
        // α = (1,5,2), β = (4,1,3), both of total degree 8.
        //   grlex:   leftmost difference index 0: 1 < 4  → α <_grlex β.
        //   grevlex: rightmost difference index 2: 2 < 3 → α >_grevlex β.
        assert_eq!(
            MonOrder::GrLex.cmp_exponents(&[1, 5, 2], &[4, 1, 3]),
            Ordering::Less
        );
        assert_eq!(
            MonOrder::GrevLex.cmp_exponents(&[1, 5, 2], &[4, 1, 3]),
            Ordering::Greater
        );
    }

    #[test]
    fn all_orders_are_multiplicative() {
        // α > β  ⟹  α + γ > β + γ, checked exhaustively on a small box.
        let orders = [MonOrder::Lex, MonOrder::GrLex, MonOrder::GrevLex];
        let pts: Vec<Vec<u32>> = (0..3)
            .flat_map(|a| (0..3).flat_map(move |b| (0..3).map(move |c| vec![a, b, c])))
            .collect();
        for order in orders {
            for a in &pts {
                for b in &pts {
                    let base = order.cmp_exponents(a, b);
                    for g in &pts {
                        let ag = mul_exponents(a, g);
                        let bg = mul_exponents(b, g);
                        assert_eq!(
                            order.cmp_exponents(&ag, &bg),
                            base,
                            "{} not multiplicative at {a:?} {b:?} {g:?}",
                            order.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn all_orders_are_total_and_antisymmetric() {
        let orders = [MonOrder::Lex, MonOrder::GrLex, MonOrder::GrevLex];
        let pts: Vec<Vec<u32>> = (0..4)
            .flat_map(|a| (0..4).flat_map(move |b| (0..4).map(move |c| vec![a, b, c])))
            .collect();
        for order in orders {
            for a in &pts {
                for b in &pts {
                    let ab = order.cmp_exponents(a, b);
                    let ba = order.cmp_exponents(b, a);
                    assert_eq!(ab, ba.reverse(), "{} antisymmetry", order.name());
                    assert_eq!(ab == Ordering::Equal, a == b, "{} totality", order.name());
                }
            }
        }
    }

    #[test]
    fn every_monomial_is_at_least_one() {
        // The well-ordering property in the form actually used: 1 is the minimum.
        let orders = [MonOrder::Lex, MonOrder::GrLex, MonOrder::GrevLex];
        let one = vec![0u32, 0, 0];
        for order in orders {
            for a in 0..3u32 {
                for b in 0..3u32 {
                    for c in 0..3u32 {
                        let e = vec![a, b, c];
                        let expected = if e == one {
                            Ordering::Equal
                        } else {
                            Ordering::Greater
                        };
                        assert_eq!(order.cmp_exponents(&e, &one), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn divisibility_lcm_gcd_and_coprimality() {
        let a = m(&[2, 0, 1]);
        let b = m(&[1, 3, 0]);
        assert!(!a.divides(&b));
        assert!(m(&[1, 0, 0]).divides(&a));
        assert_eq!(a.lcm(&b), m(&[2, 3, 1]));
        assert_eq!(a.gcd(&b), m(&[1, 0, 0]));
        assert!(!a.is_coprime(&b));

        // x^2 and y^3 share no variable: the product criterion hypothesis.
        let x2 = m(&[2, 0]);
        let y3 = m(&[0, 3]);
        assert!(x2.is_coprime(&y3));
        assert_eq!(x2.lcm(&y3), x2.mul(&y3), "coprime ⟺ lcm = product");

        assert_eq!(a.div(&m(&[3, 0, 2])), Some(m(&[1, 0, 1])));
        assert_eq!(a.div(&b), None);
    }

    #[test]
    fn pure_power_detection() {
        assert_eq!(m(&[0, 3, 0]).pure_power_var(), Some(1));
        assert_eq!(m(&[2, 1, 0]).pure_power_var(), None);
        // The monomial 1 is not a pure power of any variable.
        assert_eq!(m(&[0, 0, 0]).pure_power_var(), None);
    }

    #[test]
    fn total_degree_and_identity() {
        assert_eq!(m(&[2, 3, 4]).total_degree(), 9);
        assert!(Monomial::one(3).is_one());
        assert_eq!(Monomial::one(3).total_degree(), 0);
    }
}
