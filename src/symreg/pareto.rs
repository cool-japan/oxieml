//! Pareto-dominance utilities for discovered formulas.
//!
//! Provides [`pareto_front`] and the `dominates` method on [`DiscoveredFormula`].

use super::DiscoveredFormula;

impl DiscoveredFormula {
    /// Returns `true` if `self` Pareto-dominates `other`.
    ///
    /// `self` dominates `other` when it is at least as good on every objective
    /// (MSE and complexity) and strictly better on at least one.
    pub fn dominates(&self, other: &DiscoveredFormula) -> bool {
        self.mse <= other.mse
            && self.complexity <= other.complexity
            && (self.mse < other.mse || self.complexity < other.complexity)
    }
}

/// Extract the Pareto-optimal subset from a slice of discovered formulas.
///
/// A formula F is Pareto-optimal if no other formula G dominates it
/// (i.e., G has both lower-or-equal MSE **and** lower-or-equal complexity,
/// with at least one strictly lower).
///
/// The returned vector is sorted by complexity ascending (simplest first).
/// If `formulas` is empty, returns an empty vector.
/// Time complexity: O(n²) — acceptable for the formula counts typical of EML.
pub fn pareto_front(formulas: &[DiscoveredFormula]) -> Vec<DiscoveredFormula> {
    let mut front: Vec<DiscoveredFormula> = formulas
        .iter()
        .filter(|candidate| !formulas.iter().any(|other| other.dominates(candidate)))
        .cloned()
        .collect();

    front.sort_by(|a, b| {
        a.complexity.cmp(&b.complexity).then_with(|| {
            a.mse
                .partial_cmp(&b.mse)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    front
}

/// Returns `true` if `a` dominates `b` with respect to complexity and a custom
/// objective function `objective`.
///
/// Dominance: `a.complexity ≤ b.complexity` AND `objective(a) < objective(b)`.
pub fn dominates_by<F>(a: &DiscoveredFormula, b: &DiscoveredFormula, objective: F) -> bool
where
    F: Fn(&DiscoveredFormula) -> f64,
{
    a.complexity <= b.complexity && objective(a) < objective(b)
}

/// Pareto-optimal subset ranked by (complexity, AIC) or (complexity, BIC).
///
/// A formula F is on the IC-Pareto front if no other formula G has both
/// lower-or-equal complexity and strictly lower IC value.
///
/// The returned vector is sorted by complexity ascending. If `formulas` is
/// empty, returns an empty vector.
pub fn pareto_front_ic(formulas: &[DiscoveredFormula], use_bic: bool) -> Vec<DiscoveredFormula> {
    let ic_of = |f: &DiscoveredFormula| if use_bic { f.bic } else { f.aic };

    let mut front: Vec<DiscoveredFormula> = formulas
        .iter()
        .filter(|candidate| {
            !formulas
                .iter()
                .any(|other| dominates_by(other, candidate, ic_of))
        })
        .cloned()
        .collect();

    front.sort_by_key(|a| a.complexity);
    front
}
