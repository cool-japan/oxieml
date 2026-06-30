//! Named-constants extraction helpers for symbolic regression.
//!
//! After Adam optimisation, free parameters are real-valued floats. This module
//! provides post-processing that recognises well-known mathematical constants
//! (π, e, √2, simple rationals) and substitutes them when doing so does not
//! significantly degrade MSE.

use crate::lower::LoweredOp;
use crate::named_const::NamedConst;
use crate::tree::EmlTree;
use std::sync::Arc;

/// Bake learned parameters into a lowered operation tree.
///
/// Walks the EML topology and the lowered op tree in tandem, substituting
/// each `Const(1.0)` that originated from a `One` node with `Const(param[i])`.
/// This produces a `LoweredOp` tree that reflects the actual values discovered
/// by the Adam optimizer — suitable for pretty-printing and constants extraction.
///
/// # Strategy
///
/// `lower_node` consumes `EmlNode::One` nodes in various patterns:
/// - `eml(x, One)` → `Exp(x)` (the `One` is hidden inside the pattern)
/// - `eml(One, One)` → `Const(e)` (both Ones collapsed to e)
/// - etc.
///
/// Because patterns swallow `One` nodes before we can intercept them, this
/// function uses a post-lowering substitution: it scans the resulting
/// `LoweredOp` tree for `Const(1.0)` nodes and replaces them with the
/// corresponding parameter value in post-order.
///
/// The substitution is best-effort: if the lowering collapsed several `One`
/// nodes into a single constant (e.g., `Const(e)`), those values are not
/// individually accessible. In that case, the `Const(e)` is left as-is.
///
/// This is adequate for the constants-extraction use-case, where we scan
/// all `Const` positions regardless of their origin.
pub(super) fn bake_params_into_lowered(topology: &EmlTree, params: &[f64]) -> LoweredOp {
    if params.is_empty() {
        return topology.lower();
    }

    // If the topology has Const leaves, use the substitute-first approach which
    // correctly handles both One and Const as parameters.
    // When no Const leaves exist (the common case), use the legacy approach
    // to preserve bit-identical behavior.
    if topology.count_const_leaves() > 0 {
        let mut idx = 0usize;
        let new_root = substitute_params(&topology.root, params, &mut idx);
        let new_tree = EmlTree::from_node(new_root);
        return new_tree.lower();
    }

    // Legacy path: lower first, then substitute Const(1.0) with param values.
    let lowered = topology.lower();
    let mut param_idx = 0usize;
    replace_const_one(&lowered, params, &mut param_idx)
}

/// Bake learned parameters directly into an `EmlTree`, replacing each `One`/
/// `Const` leaf (in left-to-right traversal order) with the corresponding
/// fitted value from `params`.
///
/// This is the `EmlTree`-level counterpart to [`bake_params_into_lowered`]:
/// callers that need a self-contained, directly-evaluable `EmlTree` (rather
/// than a `LoweredOp`) for a [`crate::symreg::DiscoveredFormula`] should use
/// this instead of returning the bare, unparameterized topology.
pub(super) fn bake_params_into_tree(topology: &EmlTree, params: &[f64]) -> EmlTree {
    if params.is_empty() {
        return topology.clone();
    }
    let mut idx = 0usize;
    let new_root = substitute_params(&topology.root, params, &mut idx);
    EmlTree::from_node(new_root)
}

fn substitute_params(
    node: &crate::tree::EmlNode,
    params: &[f64],
    idx: &mut usize,
) -> std::sync::Arc<crate::tree::EmlNode> {
    use crate::tree::EmlNode;
    match node {
        EmlNode::One => {
            let p = params.get(*idx).copied().unwrap_or(1.0);
            *idx += 1;
            std::sync::Arc::new(EmlNode::Const(p))
        }
        EmlNode::Const(v) => {
            let p = params.get(*idx).copied().unwrap_or(*v);
            *idx += 1;
            std::sync::Arc::new(EmlNode::Const(p))
        }
        EmlNode::Var(i) => std::sync::Arc::new(EmlNode::Var(*i)),
        EmlNode::Eml { left, right } => {
            let l = substitute_params(left, params, idx);
            let r = substitute_params(right, params, idx);
            std::sync::Arc::new(EmlNode::Eml { left: l, right: r })
        }
    }
}

fn replace_const_one(op: &LoweredOp, params: &[f64], idx: &mut usize) -> LoweredOp {
    match op {
        LoweredOp::Const(c) if (*c - 1.0).abs() < 1e-15 => {
            if *idx < params.len() {
                let p = params[*idx];
                *idx += 1;
                LoweredOp::Const(p)
            } else {
                op.clone()
            }
        }
        LoweredOp::Const(_) | LoweredOp::Var(_) | LoweredOp::NamedConst(_) => op.clone(),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(replace_const_one(a, params, idx))),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(replace_const_one(a, params, idx)),
            Arc::new(replace_const_one(b, params, idx)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(replace_const_one(a, params, idx)),
            Arc::new(replace_const_one(b, params, idx)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(replace_const_one(a, params, idx)),
            Arc::new(replace_const_one(b, params, idx)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            Arc::new(replace_const_one(a, params, idx)),
            Arc::new(replace_const_one(b, params, idx)),
        ),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            Arc::new(replace_const_one(a, params, idx)),
            Arc::new(replace_const_one(b, params, idx)),
        ),
    }
}

/// Candidate named constants for constants extraction, ordered by priority.
fn named_const_candidates() -> Vec<(f64, NamedConst)> {
    vec![
        (std::f64::consts::PI, NamedConst::Pi),
        (-std::f64::consts::PI, NamedConst::NegPi),
        (std::f64::consts::E, NamedConst::E),
        (-std::f64::consts::E, NamedConst::NegE),
        (std::f64::consts::SQRT_2, NamedConst::Sqrt2),
        (-std::f64::consts::SQRT_2, NamedConst::NegSqrt2),
        (0.5, NamedConst::Half),
        (-0.5, NamedConst::NegHalf),
        (1.0 / 3.0, NamedConst::Third),
        (0.25, NamedConst::Quarter),
        (2.0 * std::f64::consts::PI, NamedConst::TwoPi),
        (std::f64::consts::PI / 2.0, NamedConst::PiHalf),
        (3.0_f64.sqrt(), NamedConst::Sqrt3),
        (std::f64::consts::E * std::f64::consts::E, NamedConst::ESq),
        ((1.0 + 5.0_f64.sqrt()) / 2.0, NamedConst::Phi),
        (2.0_f64.ln(), NamedConst::Ln2),
        (-2.0, NamedConst::NegTwo),
    ]
}

/// Substitute the `target_idx`-th `Const` node (in post-order traversal) with
/// `replacement`, returning the modified tree.
fn substitute_const(
    op: &LoweredOp,
    target_idx: usize,
    replacement: &LoweredOp,
    current_idx: &mut usize,
) -> LoweredOp {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => {
            let this_idx = *current_idx;
            *current_idx += 1;
            if this_idx == target_idx {
                replacement.clone()
            } else {
                op.clone()
            }
        }
        LoweredOp::Var(_) => op.clone(),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(substitute_const(
            a,
            target_idx,
            replacement,
            current_idx,
        ))),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(substitute_const(a, target_idx, replacement, current_idx)),
            Arc::new(substitute_const(b, target_idx, replacement, current_idx)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(substitute_const(a, target_idx, replacement, current_idx)),
            Arc::new(substitute_const(b, target_idx, replacement, current_idx)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(substitute_const(a, target_idx, replacement, current_idx)),
            Arc::new(substitute_const(b, target_idx, replacement, current_idx)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            Arc::new(substitute_const(a, target_idx, replacement, current_idx)),
            Arc::new(substitute_const(b, target_idx, replacement, current_idx)),
        ),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            Arc::new(substitute_const(a, target_idx, replacement, current_idx)),
            Arc::new(substitute_const(b, target_idx, replacement, current_idx)),
        ),
    }
}

/// Count `Const` and `NamedConst` nodes in post-order.
fn count_const_nodes(op: &LoweredOp) -> usize {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => 1,
        LoweredOp::Var(_) => 0,
        LoweredOp::Neg(a)
        | LoweredOp::Exp(a)
        | LoweredOp::Ln(a)
        | LoweredOp::Sin(a)
        | LoweredOp::Cos(a)
        | LoweredOp::Tan(a)
        | LoweredOp::Sinh(a)
        | LoweredOp::Cosh(a)
        | LoweredOp::Tanh(a)
        | LoweredOp::Arcsin(a)
        | LoweredOp::Arccos(a)
        | LoweredOp::Arctan(a)
        | LoweredOp::Arcsinh(a)
        | LoweredOp::Arccosh(a)
        | LoweredOp::Arctanh(a)
        | LoweredOp::Erf(a)
        | LoweredOp::LGamma(a)
        | LoweredOp::Digamma(a)
        | LoweredOp::Trigamma(a)
        | LoweredOp::Ei(a)
        | LoweredOp::Si(a)
        | LoweredOp::Ci(a) => count_const_nodes(a),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => count_const_nodes(a) + count_const_nodes(b),
    }
}

/// Evaluate MSE of a `LoweredOp` tree directly against data.
fn eval_lowered_mse(op: &LoweredOp, inputs: &[Vec<f64>], targets: &[f64]) -> f64 {
    let ops = op.to_oxiblas_ops();
    let mut total = 0.0;
    let mut count = 0usize;
    for (input, &target) in inputs.iter().zip(targets) {
        let val = LoweredOp::eval_ops(&ops, input);
        if val.is_finite() {
            total += (val - target) * (val - target);
            count += 1;
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        total / count as f64
    }
}

/// Run greedy named-constants extraction on a `LoweredOp` tree.
///
/// For each `Const` position (in post-order), tries each candidate constant
/// and accepts the substitution if `new_mse ≤ (1 + eps) * current_mse`.
/// Returns the (possibly modified) tree and its MSE.
pub(super) fn extract_named_constants(
    op: LoweredOp,
    initial_mse: f64,
    eps: f64,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> (LoweredOp, f64) {
    let candidates = named_const_candidates();
    let mut current = op;
    let mut current_mse = initial_mse;

    let n_consts = count_const_nodes(&current);
    for const_idx in 0..n_consts {
        // Find the best candidate for this position
        let mut best_candidate: Option<(LoweredOp, f64)> = None;

        for (cand_val, cand_nc) in &candidates {
            let replacement = LoweredOp::NamedConst(cand_nc.clone());
            let candidate_tree = substitute_const(&current, const_idx, &replacement, &mut 0);
            let new_mse = eval_lowered_mse(&candidate_tree, inputs, targets);
            if new_mse <= (1.0 + eps) * current_mse {
                // Prefer the candidate with lowest MSE
                let accept = match &best_candidate {
                    None => true,
                    Some((_, prev_mse)) => new_mse < *prev_mse,
                };
                if accept {
                    // Only accept if it's actually "closer" than current raw value
                    // (avoid replacing a well-fit constant with a distant named one)
                    fn get_const_val(
                        op: &LoweredOp,
                        target_idx: usize,
                        ctr: &mut usize,
                    ) -> Option<f64> {
                        match op {
                            LoweredOp::Const(c) => {
                                let i = *ctr;
                                *ctr += 1;
                                if i == target_idx { Some(*c) } else { None }
                            }
                            LoweredOp::NamedConst(nc) => {
                                let i = *ctr;
                                *ctr += 1;
                                if i == target_idx {
                                    Some(nc.value())
                                } else {
                                    None
                                }
                            }
                            LoweredOp::Var(_) => None,
                            LoweredOp::Neg(a)
                            | LoweredOp::Exp(a)
                            | LoweredOp::Ln(a)
                            | LoweredOp::Sin(a)
                            | LoweredOp::Cos(a)
                            | LoweredOp::Tan(a)
                            | LoweredOp::Sinh(a)
                            | LoweredOp::Cosh(a)
                            | LoweredOp::Tanh(a)
                            | LoweredOp::Arcsin(a)
                            | LoweredOp::Arccos(a)
                            | LoweredOp::Arctan(a)
                            | LoweredOp::Arcsinh(a)
                            | LoweredOp::Arccosh(a)
                            | LoweredOp::Arctanh(a)
                            | LoweredOp::Erf(a)
                            | LoweredOp::LGamma(a)
                            | LoweredOp::Digamma(a)
                            | LoweredOp::Trigamma(a)
                            | LoweredOp::Ei(a)
                            | LoweredOp::Si(a)
                            | LoweredOp::Ci(a) => get_const_val(a, target_idx, ctr),
                            LoweredOp::Add(a, b)
                            | LoweredOp::Sub(a, b)
                            | LoweredOp::Mul(a, b)
                            | LoweredOp::Div(a, b)
                            | LoweredOp::Pow(a, b) => get_const_val(a, target_idx, ctr)
                                .or_else(|| get_const_val(b, target_idx, ctr)),
                        }
                    }
                    let orig_val = {
                        let mut ctr = 0usize;
                        get_const_val(&current, const_idx, &mut ctr).unwrap_or(f64::NAN)
                    };
                    // Accept only when the candidate is genuinely close to the
                    // original constant value (within 5%).
                    let close_enough =
                        (cand_val - orig_val).abs() <= 0.05 * orig_val.abs().max(1e-12);
                    if close_enough {
                        best_candidate = Some((candidate_tree, new_mse));
                    }
                }
            }
        }

        if let Some((new_tree, new_mse)) = best_candidate {
            current = new_tree;
            current_mse = new_mse;
        }
    }

    (current, current_mse)
}

/// Attempt to snap a float value to the nearest named constant.
///
/// Returns `Some(nc)` when `v` is within 2% of a named constant's value,
/// `None` otherwise.
pub fn snap_to_named_const(v: f64) -> Option<NamedConst> {
    for (cand_val, nc) in named_const_candidates() {
        if cand_val.abs() > 1e-12 && (v - cand_val).abs() <= 0.02 * cand_val.abs() {
            return Some(nc);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_snapping_twopi() {
        // 2π ≈ 6.283185; use the constant directly to avoid approx_constant warning.
        let nc = snap_to_named_const(2.0 * std::f64::consts::PI);
        assert_eq!(nc, Some(NamedConst::TwoPi));
    }

    #[test]
    fn test_composite_snapping_no_snap() {
        let nc = snap_to_named_const(3.5);
        assert_eq!(nc, None);
    }

    #[test]
    fn test_composite_snapping_pihalf() {
        // π/2 ≈ 1.5708; use the constant directly to avoid approx_constant warning.
        let nc = snap_to_named_const(std::f64::consts::FRAC_PI_2);
        assert_eq!(nc, Some(NamedConst::PiHalf));
    }
}
