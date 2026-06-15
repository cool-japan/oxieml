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

use super::constants::{bake_params_into_lowered, extract_named_constants};
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
/// Returns `(final_lowered_op, final_mse)`. If `constant_extraction` is `None`,
/// the baked-but-not-extracted form is returned.
pub(super) fn try_extract_named_constants(
    topology: &EmlTree,
    best_params: &[f64],
    best_mse: f64,
    constant_extraction: Option<f64>,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> (LoweredOp, f64) {
    let baked = bake_params_into_lowered(topology, best_params);
    let baked_simplified = baked.simplify();

    if let Some(eps) = constant_extraction {
        extract_named_constants(baked_simplified, best_mse, eps, inputs, targets)
    } else {
        (baked_simplified, best_mse)
    }
}
