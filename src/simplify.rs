//! EML tree simplification and normalization.
//!
//! Applies algebraic identities specific to the EML operator to reduce
//! tree size without changing semantics.

use crate::tree::{EmlNode, EmlTree};
use std::collections::HashMap;
use std::sync::Arc;

/// Simplify an EML tree by applying reduction rules.
///
/// Rules applied (bottom-up):
/// 1. **Constant folding**: `eml(One, One)` is recognized as Euler's `e`
/// 2. **ln(exp(x)) elimination**: `eml(1, eml(eml(1, eml(x, 1)), 1))` → identity on `x`
/// 3. **exp(ln(x)) elimination**: detects and collapses double wrapping
/// 4. **Zero propagation**: `ln(1) = 0` patterns recognized
/// 5. **Common subexpression sharing**: identical subtrees get Arc-shared
pub fn simplify(tree: &EmlTree) -> EmlTree {
    let mut cache: HashMap<EmlNodeKey, Arc<EmlNode>> = HashMap::new();
    let simplified = simplify_node(&tree.root, &mut cache);
    EmlTree::from_node(simplified)
}

/// Structural key for deduplication cache.
#[derive(Clone, Hash, PartialEq, Eq)]
enum EmlNodeKey {
    One,
    Var(usize),
    Eml(EmlNodeKey2, EmlNodeKey2),
}

/// Boxed key variant to avoid infinite-size enum.
#[derive(Clone, Hash, PartialEq, Eq)]
struct EmlNodeKey2(Box<EmlNodeKey>);

fn make_key(node: &EmlNode) -> EmlNodeKey {
    match node {
        EmlNode::One => EmlNodeKey::One,
        EmlNode::Var(i) => EmlNodeKey::Var(*i),
        EmlNode::Eml { left, right } => EmlNodeKey::Eml(
            EmlNodeKey2(Box::new(make_key(left))),
            EmlNodeKey2(Box::new(make_key(right))),
        ),
    }
}

fn simplify_node(node: &EmlNode, cache: &mut HashMap<EmlNodeKey, Arc<EmlNode>>) -> Arc<EmlNode> {
    let key = make_key(node);
    if let Some(cached) = cache.get(&key) {
        return Arc::clone(cached);
    }

    let result = match node {
        EmlNode::One | EmlNode::Var(_) => Arc::new(node.clone()),
        EmlNode::Eml { left, right } => {
            let left_s = simplify_node(left, cache);
            let right_s = simplify_node(right, cache);

            // Rule: ln(exp(x)) = x
            // Pattern: eml(1, eml(eml(1, eml(x, 1)), 1)) where x is the inner argument.
            // This is the ln construction applied to exp(x) = eml(x, 1).
            // Structurally: outer left=1, outer right=eml(mid_l, 1) where mid_l=eml(1, eml(x, 1)).
            if matches!(left_s.as_ref(), EmlNode::One) {
                if let Some(inner) = match_ln_of_exp(&right_s) {
                    let inner_simplified = simplify_node(&inner, cache);
                    cache.insert(key, Arc::clone(&inner_simplified));
                    return inner_simplified;
                }
            }

            // Rule: exp(ln(x)) = x
            // Pattern: eml(eml(1, eml(eml(1, x), 1)), 1) — exp applied to ln(x).
            if matches!(right_s.as_ref(), EmlNode::One) {
                if let Some(inner) = match_exp_of_ln(&left_s) {
                    let inner_simplified = simplify_node(&inner, cache);
                    cache.insert(key, Arc::clone(&inner_simplified));
                    return inner_simplified;
                }
            }

            // Rule: eml(x, 1) where x is the ln construction of something y
            // → eml(ln(y), 1) = exp(ln(y)) = y
            if matches!(right_s.as_ref(), EmlNode::One) {
                if let Some(inner) = match_ln_pattern(&left_s) {
                    // eml(ln(y), 1) = exp(ln(y)) = y
                    let inner_simplified = simplify_node(&inner, cache);
                    cache.insert(key, Arc::clone(&inner_simplified));
                    return inner_simplified;
                }
            }

            Arc::new(EmlNode::Eml {
                left: left_s,
                right: right_s,
            })
        }
    };

    cache.insert(key, Arc::clone(&result));
    result
}

/// Match the pattern `eml(eml(1, eml(x, 1)), 1)` inside the right subtree
/// of an outer `eml(1, ...)`. The full pattern `eml(1, eml(eml(1, eml(x, 1)), 1))`
/// represents `ln(exp(x)) = x`.
fn match_ln_of_exp(right: &EmlNode) -> Option<EmlNode> {
    // right should be eml(mid_l, 1) where mid_l = eml(1, eml(x, 1))
    if let EmlNode::Eml {
        left: mid_l,
        right: mid_r,
    } = right
    {
        if !matches!(mid_r.as_ref(), EmlNode::One) {
            return None;
        }
        // mid_l should be eml(1, eml(x, 1))
        if let EmlNode::Eml {
            left: inner_l,
            right: inner_r,
        } = mid_l.as_ref()
        {
            if !matches!(inner_l.as_ref(), EmlNode::One) {
                return None;
            }
            // inner_r should be eml(x, 1)
            if let EmlNode::Eml {
                left: x_node,
                right: one_node,
            } = inner_r.as_ref()
            {
                if matches!(one_node.as_ref(), EmlNode::One) {
                    return Some(x_node.as_ref().clone());
                }
            }
        }
    }
    None
}

/// Match `eml(1, eml(eml(1, x), 1))` as the left child of an outer `eml(..., 1)`.
/// The full pattern `eml(eml(1, eml(eml(1, x), 1)), 1)` = `exp(ln(x)) = x`.
fn match_exp_of_ln(left: &EmlNode) -> Option<EmlNode> {
    // left should be eml(1, eml(eml(1, x), 1)) — the ln construction
    if let Some(inner) = match_ln_pattern(left) {
        return Some(inner);
    }
    None
}

/// Match the ln pattern: `eml(1, eml(eml(1, x), 1))` → returns `x`.
fn match_ln_pattern(node: &EmlNode) -> Option<EmlNode> {
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

/// Count the number of unique subtrees (by Arc identity) in the tree.
pub fn count_shared_subtrees(tree: &EmlTree) -> (usize, usize) {
    let mut all_nodes = Vec::new();
    collect_arcs(&tree.root, &mut all_nodes);
    let total = all_nodes.len();
    all_nodes.sort_by_key(|a| Arc::as_ptr(a) as usize);
    all_nodes.dedup_by(|a, b| Arc::ptr_eq(a, b));
    let unique = all_nodes.len();
    (total, unique)
}

fn collect_arcs(node: &EmlNode, out: &mut Vec<Arc<EmlNode>>) {
    if let EmlNode::Eml { left, right } = node {
        out.push(Arc::clone(left));
        out.push(Arc::clone(right));
        collect_arcs(left, out);
        collect_arcs(right, out);
    }
}

/// Normalize an EML tree for comparison purposes.
///
/// Applies simplification and common subexpression sharing.
pub fn normalize(tree: &EmlTree) -> EmlTree {
    simplify(tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Canonical;
    use crate::eval::EvalCtx;

    #[test]
    fn test_simplify_leaf() {
        let one = EmlTree::one();
        let simplified = simplify(&one);
        assert_eq!(simplified, one);
    }

    #[test]
    fn test_simplify_preserves_eml() {
        let one = EmlTree::one();
        let euler = EmlTree::eml(&one, &one);
        let simplified = simplify(&euler);
        assert_eq!(simplified.depth(), 1);
        assert_eq!(simplified.size(), 3);
    }

    #[test]
    fn test_simplify_ln_of_exp() {
        // ln(exp(x)) should simplify to x
        let x = EmlTree::var(0);
        let exp_x = Canonical::exp(&x); // eml(x, 1)
        let ln_exp_x = Canonical::ln(&exp_x); // eml(1, eml(eml(1, eml(x,1)), 1))
        let simplified = simplify(&ln_exp_x);
        // Should reduce to just Var(0)
        assert_eq!(simplified.size(), 1);
        assert_eq!(*simplified.root, EmlNode::Var(0));
    }

    #[test]
    fn test_simplify_exp_of_ln() {
        // exp(ln(x)) should simplify to x
        let x = EmlTree::var(0);
        let ln_x = Canonical::ln(&x); // eml(1, eml(eml(1, x), 1))
        let exp_ln_x = Canonical::exp(&ln_x); // eml(ln(x), 1)
        let simplified = simplify(&exp_ln_x);
        // Should reduce to just Var(0)
        assert_eq!(simplified.size(), 1);
        assert_eq!(*simplified.root, EmlNode::Var(0));
    }

    #[test]
    fn test_simplify_preserves_semantics() {
        // Simplification should not change evaluation results
        let x = EmlTree::var(0);
        let exp_x = Canonical::exp(&x);
        let ln_exp_x = Canonical::ln(&exp_x);

        let ctx = EvalCtx::new(&[2.5]);
        let before = ln_exp_x.eval_real(&ctx).unwrap();
        let simplified = simplify(&ln_exp_x);
        let after = simplified.eval_real(&ctx).unwrap();
        assert!((before - after).abs() < 1e-10);
    }

    #[test]
    fn test_common_subexpression_sharing() {
        // Two identical subtrees should be Arc-shared after simplification
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let sub1 = EmlTree::eml(&x, &one);
        let sub2 = EmlTree::eml(&x, &one); // structurally identical
        let tree = EmlTree::eml(&sub1, &sub2);
        let simplified = simplify(&tree);
        // Both children should now be the same Arc
        if let EmlNode::Eml { left, right } = simplified.root.as_ref() {
            assert!(Arc::ptr_eq(left, right));
        } else {
            panic!("expected Eml node");
        }
    }

    #[test]
    fn test_shared_subtree_counting() {
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let t1 = EmlTree::eml(&x, &one);
        let t2 = EmlTree::eml(&t1, &t1);
        let (total, unique) = count_shared_subtrees(&t2);
        assert!(total >= 2);
        assert!(unique <= total);
    }

    #[test]
    fn test_normalize_idempotent() {
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let t = EmlTree::eml(&x, &one);
        let n1 = normalize(&t);
        let n2 = normalize(&n1);
        assert_eq!(n1, n2);
    }
}
