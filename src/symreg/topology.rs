//! Topology enumeration and related helper functions for symbolic regression.
//!
//! This module contains:
//! - Tree topology enumeration (Catalan-number based)
//! - Topology deduplication via structural hashing
//! - MSE computation helpers for topology evaluation
//! - Interval-feasibility pre-filter for pruning

use std::sync::Arc;

use crate::eval::EvalCtx;
use crate::grad::ParameterizedEmlTree;
use crate::tree::{EmlNode, EmlTree};

/// Enumerate all EML tree topologies up to given depth with given number of variables.
///
/// Follows the Catalan-number-based enumeration of binary trees.
pub fn enumerate_topologies(max_depth: usize, num_vars: usize) -> Vec<EmlTree> {
    let mut topologies = Vec::new();
    let leaves = build_leaves(num_vars, None);

    for depth in 0..=max_depth {
        enumerate_at_depth(depth, &leaves, &mut topologies);
    }

    topologies
}

/// Enumerate topologies with optional `Const` leaf (activated by `enable_const_leaf`).
pub(super) fn enumerate_topologies_gated(
    max_depth: usize,
    num_vars: usize,
    const_leaf: Option<f64>,
) -> Vec<EmlTree> {
    let mut topologies = Vec::new();
    let leaves = build_leaves(num_vars, const_leaf);
    for depth in 0..=max_depth {
        enumerate_at_depth(depth, &leaves, &mut topologies);
    }
    topologies
}

/// Deduplicate topologies that lower+simplify to the same structural form.
///
/// Many EML topologies are semantically equivalent (commutative/associative
/// reorderings, algebraic identities like `exp(ln(x)) = x`). We apply
/// EML-level simplification first (to collapse patterns like `ln(exp(x))`
/// and `exp(ln(x))` directly on the tree), then lower to a conventional
/// op tree and run the arithmetic simplifier. We compute a structural
/// hash over the resulting `LoweredOp` and keep the first representative
/// of each hash bucket.
///
/// Note: EML is non-commutative and non-associative, so tree rearrangements
/// generally produce genuinely distinct functions. Structural dedup thus
/// catches only the cases where simplification rules collapse different
/// trees to identical forms. Deeper reduction would require evaluation-based
/// fingerprinting, which is out of scope here.
pub fn dedupe_by_semantics(topologies: Vec<EmlTree>) -> Vec<EmlTree> {
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut seen: HashSet<u64> = HashSet::new();
    topologies
        .into_iter()
        .filter(|t| {
            let eml_simplified = crate::simplify::simplify(t);
            let simplified = eml_simplified.lower().simplify();
            let mut hasher = DefaultHasher::new();
            simplified.structural_hash(&mut hasher);
            seen.insert(hasher.finish())
        })
        .collect()
}

/// Build leaf nodes: One and Var(0), Var(1), ...
pub(super) fn build_leaves(num_vars: usize, const_leaf: Option<f64>) -> Vec<Arc<EmlNode>> {
    let mut leaves = vec![Arc::new(EmlNode::One)];
    for i in 0..num_vars {
        leaves.push(Arc::new(EmlNode::Var(i)));
    }
    if let Some(v) = const_leaf {
        leaves.push(Arc::new(EmlNode::Const(v)));
    }
    leaves
}

/// Enumerate trees at exactly the given depth (duplicate-free).
///
/// A tree has exact depth `d` iff at least one child has depth `d-1`.
/// We enumerate three disjoint cases:
/// 1. Both children at exactly depth `d-1`
/// 2. Left at depth `d-1`, right at depth `< d-1`
/// 3. Left at depth `< d-1`, right at depth `d-1`
fn enumerate_at_depth(depth: usize, leaves: &[Arc<EmlNode>], out: &mut Vec<EmlTree>) {
    if depth == 0 {
        for leaf in leaves {
            out.push(EmlTree::from_node(Arc::clone(leaf)));
        }
        return;
    }

    let at_max = enumerate_at_depth_nodes(depth - 1, leaves);

    // Nodes strictly below max depth
    let mut below_max = Vec::new();
    for d in 0..(depth - 1) {
        below_max.extend(enumerate_at_depth_nodes(d, leaves));
    }

    // Case 1: both at max-1 depth
    for left in &at_max {
        for right in &at_max {
            out.push(EmlTree::from_node(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            })));
        }
    }

    // Case 2: left at max, right strictly below
    for left in &at_max {
        for right in &below_max {
            out.push(EmlTree::from_node(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            })));
        }
    }

    // Case 3: left strictly below, right at max
    for left in &below_max {
        for right in &at_max {
            out.push(EmlTree::from_node(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            })));
        }
    }
}

/// Enumerate node Arcs at exactly the given depth.
fn enumerate_at_depth_nodes(depth: usize, leaves: &[Arc<EmlNode>]) -> Vec<Arc<EmlNode>> {
    if depth == 0 {
        return leaves.to_vec();
    }

    let at_max = enumerate_at_depth_nodes(depth - 1, leaves);
    let mut below_max = Vec::new();
    for d in 0..(depth - 1) {
        below_max.extend(enumerate_at_depth_nodes(d, leaves));
    }

    let mut out = Vec::new();

    // Both at max
    for left in &at_max {
        for right in &at_max {
            out.push(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            }));
        }
    }

    // Left at max, right below
    for left in &at_max {
        for right in &below_max {
            out.push(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            }));
        }
    }

    // Left below, right at max
    for left in &below_max {
        for right in &at_max {
            out.push(Arc::new(EmlNode::Eml {
                left: Arc::clone(left),
                right: Arc::clone(right),
            }));
        }
    }

    out
}

/// Try rounding parameters to nearest integers.
pub(super) fn try_integer_rounding(params: &[f64]) -> Vec<f64> {
    params
        .iter()
        .map(|&p| {
            let rounded = p.round();
            if (p - rounded).abs() < 0.1 {
                rounded
            } else {
                p
            }
        })
        .collect()
}

/// Compute MSE for a tree without parameters (all One nodes = 1.0).
pub(super) fn compute_mse_direct(
    tree: &EmlTree,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;

    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        match tree.eval_real(&ctx) {
            Ok(val) if val.is_finite() => {
                total += (val - target).powi(2);
                count += 1;
            }
            _ => {}
        }
    }

    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

/// Check whether a topology can plausibly produce outputs in `[target_lo, target_hi]`.
///
/// Substitutes each `LoweredOp::Const(1.0)` (which represents a free parameter)
/// with a wide interval `[-PARAM_BOUND, PARAM_BOUND]` and evaluates the resulting
/// interval tree.  If the output interval overlaps the target range and is
/// non-NaN the topology is considered feasible.
///
/// Variables get intervals derived from the observed input data.
pub(super) fn topology_interval_feasible(
    topology: &EmlTree,
    input_intervals: &[crate::lower_interval::IntervalLO],
    target_lo: f64,
    target_hi: f64,
) -> bool {
    use crate::lower::LoweredOp;
    use crate::lower_interval::IntervalLO;

    /// Wide bound for unconstrained free parameters.
    const PARAM_BOUND: f64 = 1_000.0;
    let param_interval = IntervalLO::new(-PARAM_BOUND, PARAM_BOUND);

    // Replace every Const(1.0) in the lowered tree with the wide parameter interval.
    fn widen_params(op: &LoweredOp, param_iv: &IntervalLO) -> LoweredOp {
        match op {
            LoweredOp::Const(c) if (*c - 1.0).abs() < 1e-15 => {
                // Represent as a Const(-PARAM_BOUND) sentinel only for structure;
                // we will evaluate via a custom evaluator below that intercepts it.
                // Instead, build a fake two-node tree: use Exp(Const(ln(PARAM_BOUND)))
                // which evaluates to PARAM_BOUND — but that misrepresents negatives.
                // Better: widen ALL Const(1.0) nodes by replacing them with a
                // special sentinel and evaluating manually.
                // Simplest correct approach: we actually call eval_interval on the
                // *original* tree, but need to intercept the Const-1.0 nodes.
                // To avoid a custom eval, we just keep them as Const(1.0) and
                // note that the caller must widen the result after the fact.
                // ACTUAL APPROACH: use a dummy variable index past num_vars that
                // maps to `param_iv` in the vars slice — but that requires passing
                // the count. Instead, substitute Const(1.0) with the min of the
                // param_interval so the tree is purely constant for structure
                // but the real feasibility is evaluated below.
                //
                // Cleanest: build a recursive interval evaluator here that mirrors
                // eval_interval but expands Const(1.0) → param_iv.
                let _ = param_iv;
                op.clone()
            }
            _ => op.clone(),
        }
    }
    // (Suppress the helper; we implement the inline evaluator directly below.)
    let _ = widen_params;

    // Inline interval evaluator that expands Const(1.0) → param_interval.
    fn eval_interval_with_params<'a>(
        op: &'a LoweredOp,
        vars: &[IntervalLO],
        param_iv: &IntervalLO,
    ) -> IntervalLO {
        use crate::lower_interval::IntervalLO;

        let recurse = |child: &'a LoweredOp| eval_interval_with_params(child, vars, param_iv);
        let recurse2 = |a: &'a LoweredOp, b: &'a LoweredOp| {
            (
                eval_interval_with_params(a, vars, param_iv),
                eval_interval_with_params(b, vars, param_iv),
            )
        };

        match op {
            LoweredOp::Const(c) if (*c - 1.0).abs() < 1e-15 => *param_iv,
            LoweredOp::Const(c) => IntervalLO::point(*c),
            LoweredOp::NamedConst(nc) => IntervalLO::point(nc.value()),
            LoweredOp::Var(i) => vars.get(*i).copied().unwrap_or_else(IntervalLO::nan),
            LoweredOp::Neg(x) => {
                let ix = recurse(x);
                IntervalLO::new(-ix.hi, -ix.lo)
            }
            LoweredOp::Add(a, b) => {
                let (ia, ib) = recurse2(a, b);
                IntervalLO::new(ia.lo + ib.lo, ia.hi + ib.hi)
            }
            LoweredOp::Sub(a, b) => {
                let (ia, ib) = recurse2(a, b);
                IntervalLO::new(ia.lo - ib.hi, ia.hi - ib.lo)
            }
            LoweredOp::Mul(a, b) => {
                let (ia, ib) = recurse2(a, b);
                let p = [ia.lo * ib.lo, ia.lo * ib.hi, ia.hi * ib.lo, ia.hi * ib.hi];
                IntervalLO::new(
                    p.iter().copied().fold(f64::INFINITY, f64::min),
                    p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            }
            LoweredOp::Div(a, b) => {
                let (ia, ib) = recurse2(a, b);
                if ib.lo <= 0.0 && ib.hi >= 0.0 {
                    return IntervalLO::full();
                }
                let recip = IntervalLO::new(1.0 / ib.hi, 1.0 / ib.lo);
                let p = [
                    ia.lo * recip.lo,
                    ia.lo * recip.hi,
                    ia.hi * recip.lo,
                    ia.hi * recip.hi,
                ];
                IntervalLO::new(
                    p.iter().copied().fold(f64::INFINITY, f64::min),
                    p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            }
            LoweredOp::Exp(x) => {
                let ix = recurse(x);
                // Clamp the exponent to avoid overflow: exp(709) ≈ 8e307
                let lo_exp = ix.lo.max(-PARAM_BOUND).exp();
                let hi_exp = ix.hi.min(709.0).exp();
                IntervalLO::new(lo_exp, hi_exp)
            }
            LoweredOp::Ln(x) => {
                let ix = recurse(x);
                if ix.lo > 0.0 {
                    IntervalLO::new(ix.lo.ln(), ix.hi.ln())
                } else {
                    IntervalLO::nan()
                }
            }
            LoweredOp::Sin(_) | LoweredOp::Cos(_) => IntervalLO::new(-1.0, 1.0),
            LoweredOp::Tan(_) => IntervalLO::full(),
            LoweredOp::Sinh(x) => {
                let ix = recurse(x);
                IntervalLO::new(ix.lo.sinh(), ix.hi.sinh())
            }
            LoweredOp::Cosh(x) => {
                let ix = recurse(x);
                let lo_val = if ix.lo <= 0.0 && 0.0 <= ix.hi {
                    1.0
                } else {
                    ix.lo.cosh().min(ix.hi.cosh())
                };
                IntervalLO::new(lo_val, ix.lo.cosh().max(ix.hi.cosh()))
            }
            LoweredOp::Tanh(x) => {
                let ix = recurse(x);
                IntervalLO::new(ix.lo.tanh(), ix.hi.tanh())
            }
            LoweredOp::Arcsin(x) | LoweredOp::Arccos(x) => {
                let ix = recurse(x);
                if ix.lo < -1.0 || ix.hi > 1.0 {
                    IntervalLO::nan()
                } else {
                    IntervalLO::new(ix.lo.asin(), ix.hi.asin())
                }
            }
            LoweredOp::Arctan(x) | LoweredOp::Arctanh(x) | LoweredOp::Arcsinh(x) => {
                let ix = recurse(x);
                IntervalLO::new(ix.lo.atan(), ix.hi.atan())
            }
            LoweredOp::Arccosh(x) => {
                let ix = recurse(x);
                if ix.hi < 1.0 {
                    IntervalLO::nan()
                } else {
                    let lo_c = ix.lo.max(1.0);
                    IntervalLO::new(lo_c.acosh(), ix.hi.acosh())
                }
            }
            LoweredOp::Pow(a, b) => {
                // Use exp(b*ln(a)) for general case; return full if base straddles zero.
                let (ia, ib) = recurse2(a, b);
                if ia.lo <= 0.0 {
                    return IntervalLO::full();
                }
                let ln_base = IntervalLO::new(ia.lo.ln(), ia.hi.ln());
                let p = [
                    ib.lo * ln_base.lo,
                    ib.lo * ln_base.hi,
                    ib.hi * ln_base.lo,
                    ib.hi * ln_base.hi,
                ];
                let mul_lo = p.iter().copied().fold(f64::INFINITY, f64::min);
                let mul_hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                IntervalLO::new(mul_lo.exp(), mul_hi.min(709.0).exp())
            }
            LoweredOp::Erf(_) => IntervalLO::new(-1.0, 1.0),
            LoweredOp::LGamma(_)
            | LoweredOp::Digamma(_)
            | LoweredOp::Trigamma(_)
            | LoweredOp::Ei(_)
            | LoweredOp::Si(_)
            | LoweredOp::Ci(_) => IntervalLO::full(),
        }
    }

    let lowered = topology.lower();
    let out_iv = eval_interval_with_params(&lowered, input_intervals, &param_interval);

    // NaN output → definitely infeasible
    if out_iv.lo.is_nan() || out_iv.hi.is_nan() {
        return false;
    }
    // [-inf, +inf] → always feasible (no information)
    if out_iv.lo.is_infinite() && out_iv.hi.is_infinite() {
        return true;
    }
    // Check overlap with [target_lo, target_hi]
    out_iv.hi >= target_lo && out_iv.lo <= target_hi
}

/// Compute MSE for a parameterized tree.
pub(super) fn compute_mse_parameterized(
    ptree: &ParameterizedEmlTree,
    inputs: &[Vec<f64>],
    targets: &[f64],
) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;

    for (input, &target) in inputs.iter().zip(targets) {
        let ctx = EvalCtx::new(input);
        match ptree.forward(&ctx) {
            Ok(val) if val.is_finite() => {
                total += (val - target).powi(2);
                count += 1;
            }
            _ => {}
        }
    }

    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}
