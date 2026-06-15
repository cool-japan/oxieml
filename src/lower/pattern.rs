//! Canonical-shape pattern matchers for EML → `LoweredOp` lowering.
//!
//! Recognises sin/cos and 10+ transcendental canonical EML tree shapes
//! and converts them to their native [`super::LoweredOp`] variants.

use crate::tree::{EmlNode, EmlTree};
use std::sync::{Arc, OnceLock};

use super::LoweredOp;

/// Sentinel variable index used as a wildcard in structural templates.
///
/// When a template contains `EmlNode::Var(WILDCARD_VAR)`, the matcher
/// captures the corresponding subtree from the candidate and enforces
/// that every wildcard occurrence refers to the same captured subtree.
///
/// Chosen well below `usize::MAX` to avoid overflow in `count_vars`
/// (which computes `i + 1`) and `EmlTree::var` (same), yet far above
/// any realistic user variable index.
const WILDCARD_VAR: usize = usize::MAX / 2;

/// Lazily-initialised canonical `sin(x_placeholder)` tree, where `x_placeholder`
/// is `EmlNode::Var(WILDCARD_VAR)`. Used as a unification template.
fn sin_template() -> &'static EmlNode {
    static TEMPLATE: OnceLock<EmlNode> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let placeholder = EmlTree::var(WILDCARD_VAR);
        let tree = crate::canonical::Canonical::sin(&placeholder);
        (*tree.root).clone()
    })
}

/// Lazily-initialised canonical `cos(x_placeholder)` tree.
fn cos_template() -> &'static EmlNode {
    static TEMPLATE: OnceLock<EmlNode> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let placeholder = EmlTree::var(WILDCARD_VAR);
        let tree = crate::canonical::Canonical::cos(&placeholder);
        (*tree.root).clone()
    })
}

/// Unify `candidate` against `template`, where `template` may contain wildcards
/// (`EmlNode::Var(WILDCARD_VAR)`). On success, returns `Some(captured_subtree)`.
///
/// All wildcard occurrences in the template must capture the **same** subtree
/// structurally. If any mismatch or inconsistent capture is found, returns `None`.
///
/// Non-wildcard leaves and internal nodes must match exactly; two `Eml` nodes
/// recurse on both children.
fn unify_with_wildcard<'a>(
    candidate: &'a EmlNode,
    template: &EmlNode,
    captured: &mut Option<&'a EmlNode>,
) -> bool {
    // Wildcard in template captures (or must agree with previous capture).
    if let EmlNode::Var(idx) = template {
        if *idx == WILDCARD_VAR {
            match captured {
                None => {
                    *captured = Some(candidate);
                    return true;
                }
                Some(prev) => {
                    return nodes_structurally_equal(prev, candidate);
                }
            }
        }
    }

    match (candidate, template) {
        (EmlNode::One, EmlNode::One) => true,
        (EmlNode::Var(a), EmlNode::Var(b)) => a == b,
        (EmlNode::Const(a), EmlNode::Const(b)) => a.to_bits() == b.to_bits(),
        (
            EmlNode::Eml {
                left: la,
                right: ra,
            },
            EmlNode::Eml {
                left: lb,
                right: rb,
            },
        ) => {
            unify_with_wildcard(la.as_ref(), lb.as_ref(), captured)
                && unify_with_wildcard(ra.as_ref(), rb.as_ref(), captured)
        }
        _ => false,
    }
}

/// Structural equality on `EmlNode` references.
fn nodes_structurally_equal(a: &EmlNode, b: &EmlNode) -> bool {
    match (a, b) {
        (EmlNode::One, EmlNode::One) => true,
        (EmlNode::Var(i), EmlNode::Var(j)) => i == j,
        (EmlNode::Const(a), EmlNode::Const(b)) => (a - b).abs() < 1e-15,
        (
            EmlNode::Eml {
                left: la,
                right: ra,
            },
            EmlNode::Eml {
                left: lb,
                right: rb,
            },
        ) => {
            nodes_structurally_equal(la.as_ref(), lb.as_ref())
                && nodes_structurally_equal(ra.as_ref(), rb.as_ref())
        }
        _ => false,
    }
}

/// Recognise the canonical `Canonical::sin(x)` EML tree shape.
/// Returns the captured `x` subtree on success.
fn match_sin_structure(node: &EmlNode) -> Option<EmlNode> {
    let mut captured: Option<&EmlNode> = None;
    if unify_with_wildcard(node, sin_template(), &mut captured) {
        captured.cloned()
    } else {
        None
    }
}

/// Recognise the canonical `Canonical::cos(x)` EML tree shape.
/// Returns the captured `x` subtree on success.
fn match_cos_structure(node: &EmlNode) -> Option<EmlNode> {
    let mut captured: Option<&EmlNode> = None;
    if unify_with_wildcard(node, cos_template(), &mut captured) {
        captured.cloned()
    } else {
        None
    }
}

/// Match the ln structure: `eml(1, eml(eml(1, x), 1))` → returns `x`.
fn match_ln_structure(node: &EmlNode) -> Option<EmlNode> {
    if let EmlNode::Eml { left, right } = node {
        if !matches!(left.as_ref(), EmlNode::One) {
            return None;
        }
        if let EmlNode::Eml {
            left: mid_l,
            right: mid_r,
        } = right.as_ref()
        {
            if !matches!(mid_r.as_ref(), EmlNode::One) {
                return None;
            }
            if let EmlNode::Eml {
                left: inner_l,
                right: inner_r,
            } = mid_l.as_ref()
            {
                if matches!(inner_l.as_ref(), EmlNode::One) {
                    return Some(inner_r.as_ref().clone());
                }
            }
        }
    }
    None
}

/// Match ln pattern in the right subtree of `eml(1, right)`.
/// Looks for `eml(eml(1, x), 1)` inside right, giving `ln(x)`.
fn match_ln_of_right(right: &EmlNode) -> Option<EmlNode> {
    if let EmlNode::Eml {
        left: mid_l,
        right: mid_r,
    } = right
    {
        if !matches!(mid_r.as_ref(), EmlNode::One) {
            return None;
        }
        if let EmlNode::Eml {
            left: inner_l,
            right: inner_r,
        } = mid_l.as_ref()
        {
            if matches!(inner_l.as_ref(), EmlNode::One) {
                return Some(inner_r.as_ref().clone());
            }
        }
    }
    None
}

/// Lower a single EML node to a `LoweredOp`.
pub(super) fn lower_node(node: &EmlNode) -> LoweredOp {
    match node {
        EmlNode::Const(v) => LoweredOp::Const(*v),
        EmlNode::One => LoweredOp::Const(1.0),
        EmlNode::Var(i) => LoweredOp::Var(*i),
        EmlNode::Eml { left, right } => {
            // Try to recognize known patterns before falling back to exp(l) - ln(r).
            // Patterns are checked most-specific first to avoid premature matches.

            // Most specific: recognise canonical `Canonical::sin(x)` / `Canonical::cos(x)`
            // tree shapes and lower them to native `LoweredOp::Sin` / `LoweredOp::Cos`,
            // giving f64::sin / f64::cos precision (~1e-15) instead of the ~1e-2 drift
            // that complex-arithmetic evaluation accumulates over the deep Euler-formula
            // EML tree.
            if let Some(inner) = match_sin_structure(node) {
                return LoweredOp::Sin(Arc::new(lower_node(&inner)));
            }
            if let Some(inner) = match_cos_structure(node) {
                return LoweredOp::Cos(Arc::new(lower_node(&inner)));
            }

            // Pattern: eml(x, One) = exp(x)
            if matches!(right.as_ref(), EmlNode::One) {
                // Sub-pattern: eml(ln_tree, One) = exp(ln(x)) = x
                if let Some(inner) = match_ln_structure(left) {
                    return lower_node(&inner);
                }
                return LoweredOp::Exp(Arc::new(lower_node(left)));
            }

            // Pattern: eml(One, One) = e
            if matches!(left.as_ref(), EmlNode::One) && matches!(right.as_ref(), EmlNode::One) {
                return LoweredOp::Const(std::f64::consts::E);
            }

            // Pattern: eml(One, eml(eml(One, x), One)) = ln(x)
            // MUST be checked before the e-x pattern since it's more specific.
            if matches!(left.as_ref(), EmlNode::One) {
                if let Some(inner) = match_ln_of_right(right) {
                    return LoweredOp::Ln(Arc::new(lower_node(&inner)));
                }
            }

            // Pattern: eml(One, eml(x, One)) = e - x
            if matches!(left.as_ref(), EmlNode::One) {
                if let EmlNode::Eml {
                    left: inner_l,
                    right: inner_r,
                } = right.as_ref()
                {
                    if matches!(inner_r.as_ref(), EmlNode::One) {
                        let x_lowered = lower_node(inner_l);
                        return LoweredOp::Sub(
                            Arc::new(LoweredOp::Const(std::f64::consts::E)),
                            Arc::new(x_lowered),
                        );
                    }
                }
            }

            // Pattern: eml(ln(x), eml(y, One)) = x - y (subtraction)
            // This is the sub() canonical construction.
            if let Some(x_inner) = match_ln_structure(left) {
                if let EmlNode::Eml {
                    left: y_node,
                    right: y_one,
                } = right.as_ref()
                {
                    if matches!(y_one.as_ref(), EmlNode::One) {
                        // eml(ln(x), eml(y, 1)) = exp(ln(x)) - ln(exp(y)) = x - y
                        return LoweredOp::Sub(
                            Arc::new(lower_node(&x_inner)),
                            Arc::new(lower_node(y_node)),
                        );
                    }
                }
            }

            // Default: eml(left, right) = exp(left) - ln(right)
            let left_lowered = lower_node(left);
            let right_lowered = lower_node(right);
            LoweredOp::Sub(
                Arc::new(LoweredOp::Exp(Arc::new(left_lowered))),
                Arc::new(LoweredOp::Ln(Arc::new(right_lowered))),
            )
        }
    }
}
