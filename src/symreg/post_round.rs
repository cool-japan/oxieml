//! Post-Adam constant rounding and named-constant extraction.
//!
//! After Adam optimisation converges, we optionally snap free constants to:
//! 1. **Integer rounding** — parameters within 0.02 of an integer are rounded.
//!    Controlled by [`SymRegConfig::integer_rounding`].
//! 2. **Named-constant extraction** — candidate set: π, e, √2, simple rationals
//!    with denominator ≤ 12 (Stern-Brocot). Acceptance criterion:
//!    `new_mse ≤ (1 + eps) * current_mse`.
//!    Controlled by [`SymRegConfig::constant_extraction`].

use crate::grad::ParameterizedEmlTree;
use crate::lower::LoweredOp;
use crate::tree::EmlTree;

use super::constants::{
    bake_params_into_lowered, extract_named_constants, extract_named_constants_params,
};
use super::topology::{compute_mse_parameterized, try_integer_rounding};

/// Apply integer rounding to `best_params` and accept if MSE stays within 1%.
///
/// Returns the updated `(params, mse)` pair. If rounding degrades MSE by
/// more than 1%, the original values are returned unchanged.
pub(super) fn try_post_adam_rounding(
    topology: &EmlTree,
    best_params: Vec<f64>,
    best_mse: f64,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> (Vec<f64>, f64) {
    let rounded = try_integer_rounding(&best_params);
    let mut ptree_rounded = ParameterizedEmlTree::from_topology(topology, 1.0);
    ptree_rounded.params = rounded;
    let rounded_mse = compute_mse_parameterized(&ptree_rounded, inputs, targets);
    if let Some(rmse) = rounded_mse {
        if rmse <= best_mse * 1.01 {
            return (ptree_rounded.params, rmse);
        }
    }
    (best_params, best_mse)
}

/// Bake learned parameters into the lowered form and optionally extract named
/// constants (π, e, √2, simple rationals).
///
/// Returns `(final_lowered_op, final_params, final_mse)`.
///
/// When `constant_extraction` is `None` the parameters are returned unchanged
/// and `final_mse == best_mse`; the lowered op is just the fitted tree.
///
/// When it is `Some(eps)` the extraction runs in two stages:
///
/// 1. **Parameters** — [`extract_named_constants_params`] snaps the fitted
///    values themselves. This is the stage that decides the model, so an
///    accepted constant lands in `final_params` (and hence in
///    `DiscoveredFormula::params`/`::eml_tree`) together with its MSE.
/// 2. **Lowered form** — [`extract_named_constants`] then runs over the
///    lowered/simplified op purely to render those values with their symbolic
///    names (`π` instead of `3.141593`), including constants the simplifier
///    folded out of several parameters. Its MSE is deliberately discarded:
///    the reported `final_mse` always belongs to `final_params`, i.e. to the
///    tree the caller is handed.
pub(super) fn try_extract_named_constants(
    topology: &EmlTree,
    best_params: &[f64],
    best_mse: f64,
    constant_extraction: Option<f64>,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> (LoweredOp, Vec<f64>, f64) {
    let Some(eps) = constant_extraction else {
        let baked = bake_params_into_lowered(topology, best_params).simplify();
        return (baked, best_params.to_vec(), best_mse);
    };

    let (snapped_params, snapped_mse) =
        extract_named_constants_params(topology, best_params, best_mse, eps, inputs, targets);
    let baked = bake_params_into_lowered(topology, &snapped_params).simplify();
    let (display_op, _display_mse) =
        extract_named_constants(baked, snapped_mse, eps, inputs, targets);
    (display_op, snapped_params, snapped_mse)
}
