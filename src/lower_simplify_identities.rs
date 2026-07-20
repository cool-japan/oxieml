//! Opt-in trigonometric / exponential-logarithmic simplify identities, gated by
//! a lightweight [`Assumptions`] layer.
//!
//! [`crate::lower::LoweredOp::simplify`] applies only *unconditional* algebraic
//! simplification (constant folding, polynomial canonicalisation). This module
//! adds a second, richer layer reached through
//! [`LoweredOp::simplify_with`](crate::lower::LoweredOp::simplify_with) and
//! [`LoweredOp::simplify_with_flags`](crate::lower::LoweredOp::simplify_with_flags).
//! The base `simplify` is left byte-for-byte unchanged, so every existing
//! canonical form is preserved; the new identities live entirely on top of it.
//!
//! # Two classes of rewrite
//!
//! **Contracting identities are on by default** (they never enlarge the tree):
//!
//! | identity                              | rewrite                              | validity            |
//! |---------------------------------------|--------------------------------------|---------------------|
//! | Pythagorean (circular)                | `sin²u + cos²u → 1`                   | all real `u`        |
//! | Pythagorean (hyperbolic)              | `cosh²u − sinh²u → 1`                 | all real `u`        |
//! | Pythagorean (tangent/secant)          | `1 + tan²u → 1/cos²u` (`= sec²u`)     | `cos u ≠ 0`         |
//! | radical of a square (assumption-gated)| `√(u²) → u`                          | only when `u ≥ 0`   |
//!
//! **Expanding identities are opt-in** through [`RewriteFlags`] (off by default,
//! so canonical forms of existing expressions are untouched):
//!
//! | flag              | rewrite                                        | validity     |
//! |-------------------|------------------------------------------------|--------------|
//! | `product_to_sum`  | `sin A cos B → ½[sin(A+B)+sin(A−B)]`, …         | all real     |
//! | `double_angle`    | `sin 2u → 2 sin u cos u`, `cos 2u → cos²u−sin²u`| all real     |
//! | `half_angle`      | `sin²u → (1−cos 2u)/2`, `cos²u → (1+cos 2u)/2`  | all real     |
//! | `log_expand`      | `ln(a·b) → ln a + ln b`                         | `a,b > 0`    |
//! | `exp_expand`      | `exp(a+b) → exp a · exp b`                      | all real     |
//! | `log_combine`     | `ln a + ln b → ln(a·b)`                         | `a,b > 0`    |
//! | `exp_combine`     | `exp a · exp b → exp(a+b)`                      | all real     |
//! | `pow_simp`        | `xᵃ · xᵇ → xᵃ⁺ᵇ`                               | positive base|
//!
//! # Numeric-verification gate (never emit an unverified rewrite)
//!
//! Every candidate rewrite is re-checked against the original at the
//! [`PROBE_POINTS`](crate::lower_simplify) — exactly the discipline the base
//! simplifier and the rational rewrites use. The twist here is that the probe
//! points are **mapped through the caller's assumptions**: a variable asserted
//! non-negative is probed only at non-negative values, so an assumption-gated
//! identity such as `√(u²) → u` is accepted precisely on the asserted
//! sub-domain and rejected everywhere else (where the correct answer is `|u|`).
//!
//! A contracting rewrite is additionally required not to increase
//! [`node_count`](crate::lower_simplify); an expanding rewrite is required to
//! change the tree at all. Any candidate that fails its gate is discarded and
//! the original subtree is kept.

use crate::assumptions::Assumptions;
use crate::integrate_trig::product_to_sum;
use crate::lower::LoweredOp;
use crate::lower_simplify::{PROBE_POINTS, node_count, ops_struct_hash};
use crate::rewrite::logexp::{
    expcombine_tree, expexpand_tree, logcombine_tree, logexpand_tree, powsimp_tree,
};
use std::sync::Arc;

/// Bound on contracting-identity fixpoint iterations (guarantees termination).
const MAX_CONTRACT_ITERS: usize = 12;
/// Bound on expand/contract alternation passes (guarantees termination even when
/// two directly-inverse expanding flags are enabled together).
const MAX_EXPAND_PASSES: usize = 6;
/// Absolute tolerance used when recognising integer/half-integer exponents.
const EXP_EPS: f64 = 1e-12;

/// Which opt-in *expanding* identities the simplifier may apply.
///
/// Every field defaults to `false`, so [`RewriteFlags::default`] leaves only the
/// contracting identities active and never changes the canonical form of an
/// expression that the base [`simplify`](crate::lower::LoweredOp::simplify) would
/// produce (beyond the always-on Pythagorean collapses). Construct either with
/// the field-update syntax (`RewriteFlags { product_to_sum: true,
/// ..RewriteFlags::default() }`) or with the `with_*` builder methods.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RewriteFlags {
    /// Rewrite a product of `sin`/`cos` into a sum (`sin A cos B →
    /// ½[sin(A+B)+sin(A−B)]`, etc.).
    pub product_to_sum: bool,
    /// Expand a double angle (`sin 2u → 2 sin u cos u`, `cos 2u → cos²u − sin²u`,
    /// and the hyperbolic analogues).
    pub double_angle: bool,
    /// Power-reduce a squared `sin`/`cos` (half-angle: `sin²u → (1−cos 2u)/2`,
    /// `cos²u → (1+cos 2u)/2`).
    pub half_angle: bool,
    /// Split a logarithm of a product/quotient/power (`ln(a·b) → ln a + ln b`).
    pub log_expand: bool,
    /// Split an exponential of a sum/difference (`exp(a+b) → exp a · exp b`).
    pub exp_expand: bool,
    /// Combine a sum of logarithms (`ln a + ln b → ln(a·b)`).
    pub log_combine: bool,
    /// Combine a product of exponentials (`exp a · exp b → exp(a+b)`).
    pub exp_combine: bool,
    /// Merge powers of a common base (`xᵃ · xᵇ → xᵃ⁺ᵇ`).
    pub pow_simp: bool,
}

impl RewriteFlags {
    /// No opt-in rewrites (identical to [`RewriteFlags::default`]).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            product_to_sum: false,
            double_angle: false,
            half_angle: false,
            log_expand: false,
            exp_expand: false,
            log_combine: false,
            exp_combine: false,
            pow_simp: false,
        }
    }

    /// Enable every angle-expanding and log/exp-*splitting* rewrite
    /// (`product_to_sum`, `double_angle`, `half_angle`, `log_expand`,
    /// `exp_expand`). The *combining* directions are intentionally left off so
    /// this preset does not fight itself.
    #[must_use]
    pub const fn all_expanding() -> Self {
        Self {
            product_to_sum: true,
            double_angle: true,
            half_angle: true,
            log_expand: true,
            exp_expand: true,
            log_combine: false,
            exp_combine: false,
            pow_simp: false,
        }
    }

    /// Builder: enable [`product_to_sum`](Self::product_to_sum).
    #[must_use]
    pub const fn with_product_to_sum(mut self) -> Self {
        self.product_to_sum = true;
        self
    }

    /// Builder: enable [`double_angle`](Self::double_angle).
    #[must_use]
    pub const fn with_double_angle(mut self) -> Self {
        self.double_angle = true;
        self
    }

    /// Builder: enable [`half_angle`](Self::half_angle).
    #[must_use]
    pub const fn with_half_angle(mut self) -> Self {
        self.half_angle = true;
        self
    }

    /// Builder: enable [`log_expand`](Self::log_expand).
    #[must_use]
    pub const fn with_log_expand(mut self) -> Self {
        self.log_expand = true;
        self
    }

    /// Builder: enable [`exp_expand`](Self::exp_expand).
    #[must_use]
    pub const fn with_exp_expand(mut self) -> Self {
        self.exp_expand = true;
        self
    }

    /// Builder: enable [`log_combine`](Self::log_combine).
    #[must_use]
    pub const fn with_log_combine(mut self) -> Self {
        self.log_combine = true;
        self
    }

    /// Builder: enable [`exp_combine`](Self::exp_combine).
    #[must_use]
    pub const fn with_exp_combine(mut self) -> Self {
        self.exp_combine = true;
        self
    }

    /// Builder: enable [`pow_simp`](Self::pow_simp).
    #[must_use]
    pub const fn with_pow_simp(mut self) -> Self {
        self.pow_simp = true;
        self
    }

    /// Whether any opt-in rewrite is enabled.
    const fn any(self) -> bool {
        self.product_to_sum
            || self.double_angle
            || self.half_angle
            || self.log_expand
            || self.exp_expand
            || self.log_combine
            || self.exp_combine
            || self.pow_simp
    }
}

/// Simplify `root` with assumption-gated identities and the opt-in `flags`.
///
/// This is the shared implementation behind
/// [`LoweredOp::simplify_with`](crate::lower::LoweredOp::simplify_with) and
/// [`LoweredOp::simplify_with_flags`](crate::lower::LoweredOp::simplify_with_flags).
///
/// Pipeline:
/// 1. run the unchanged base [`simplify`](crate::lower::LoweredOp::simplify);
/// 2. drive the always-on contracting identities to a fixpoint;
/// 3. if any expanding flag is set, alternate one expanding pass with a
///    re-contraction, using structural-hash cycle detection to terminate.
pub(crate) fn simplify_with_flags(
    root: &LoweredOp,
    assumptions: &Assumptions,
    flags: RewriteFlags,
) -> LoweredOp {
    let mut cur = contracting_fixpoint(root.simplify(), assumptions);

    if flags.any() {
        // Structural hashes we have already produced — used to break out of a
        // fixpoint or of an oscillation between two mutually-inverse rules.
        let mut seen: Vec<u64> = vec![ops_struct_hash(&cur)];
        for _ in 0..MAX_EXPAND_PASSES {
            let expanded = expand_pass(&cur, assumptions, flags);
            let contracted = contracting_fixpoint(expanded.simplify(), assumptions);
            let hash = ops_struct_hash(&contracted);
            if seen.contains(&hash) {
                // Converged (hash == current) or entered a cycle — stop.
                break;
            }
            seen.push(hash);
            cur = contracted;
        }
    }

    cur
}

// ── Contracting layer (always on) ──────────────────────────────────────────────

/// Iterate [`apply_contracting_pass`] (followed by a base `simplify` to fold any
/// constants the rewrite exposed) until the structural hash stabilises.
fn contracting_fixpoint(mut cur: LoweredOp, assumptions: &Assumptions) -> LoweredOp {
    for _ in 0..MAX_CONTRACT_ITERS {
        let next = apply_contracting_pass(&cur, assumptions).simplify();
        if ops_struct_hash(&next) == ops_struct_hash(&cur) {
            break;
        }
        cur = next;
    }
    cur
}

/// One bottom-up pass of the always-on contracting identities.
fn apply_contracting_pass(op: &LoweredOp, assumptions: &Assumptions) -> LoweredOp {
    let node = map_children(op, |child| apply_contracting_pass(child, assumptions));

    // √(u²) → u  (only when u is asserted non-negative; verified on that domain).
    if let Some(candidate) = rule_sqrt_of_square(&node, assumptions) {
        if accept_contracting(&node, &candidate, assumptions) {
            return candidate;
        }
    }
    // sin²u+cos²u → 1 and cosh²u−sinh²u → 1 (both strictly contracting).
    if let Some(candidate) = collapse_pythagorean(&node) {
        if accept_contracting(&node, &candidate, assumptions) {
            return candidate;
        }
    }
    // 1 + tan²u → 1/cos²u (kept separate: it is size-neutral, not strictly
    // shrinking, so it earns its own node-count gate).
    if let Some(candidate) = collapse_tan_identity(&node) {
        if accept_contracting(&node, &candidate, assumptions) {
            return candidate;
        }
    }
    node
}

// ── Expanding layer (opt-in) ────────────────────────────────────────────────────

/// One pass of the opt-in expanding identities selected by `flags`.
fn expand_pass(op: &LoweredOp, assumptions: &Assumptions, flags: RewriteFlags) -> LoweredOp {
    let mut cur = expand_trig_pass(op, assumptions, flags);
    if flags.log_expand {
        cur = gated_whole(&cur, logexpand_tree, assumptions);
    }
    if flags.exp_expand {
        cur = gated_whole(&cur, expexpand_tree, assumptions);
    }
    if flags.log_combine {
        cur = gated_whole(&cur, logcombine_tree, assumptions);
    }
    if flags.exp_combine {
        cur = gated_whole(&cur, expcombine_tree, assumptions);
    }
    if flags.pow_simp {
        cur = gated_whole(&cur, powsimp_tree, assumptions);
    }
    cur
}

/// Bottom-up pass of the trigonometric expanding rules (product-to-sum,
/// double-angle, half-angle/power-reduction).
fn expand_trig_pass(op: &LoweredOp, assumptions: &Assumptions, flags: RewriteFlags) -> LoweredOp {
    let mut best = map_children(op, |child| expand_trig_pass(child, assumptions, flags));
    if flags.product_to_sum {
        if let Some(candidate) = rule_product_to_sum(&best) {
            if accept_expanding(&best, &candidate, assumptions) {
                best = candidate;
            }
        }
    }
    if flags.double_angle {
        if let Some(candidate) = rule_double_angle(&best) {
            if accept_expanding(&best, &candidate, assumptions) {
                best = candidate;
            }
        }
    }
    if flags.half_angle {
        if let Some(candidate) = rule_power_reduction(&best) {
            if accept_expanding(&best, &candidate, assumptions) {
                best = candidate;
            }
        }
    }
    best
}

/// Apply a whole-tree rewrite `f` and keep it only if it changes the tree while
/// preserving its value on the asserted domain.
fn gated_whole<F: Fn(&LoweredOp) -> LoweredOp>(
    op: &LoweredOp,
    f: F,
    assumptions: &Assumptions,
) -> LoweredOp {
    let candidate = f(op);
    if accept_expanding(op, &candidate, assumptions) {
        candidate
    } else {
        op.clone()
    }
}

// ── Individual rewrite rules ────────────────────────────────────────────────────

/// `sin A · cos B → ½[…]` and the other three product-to-sum forms.
///
/// Delegates to the shared [`product_to_sum`] helper (also used by the
/// trigonometric-integration engine), which returns `None` unless both factors
/// are bare `sin`/`cos` nodes.
fn rule_product_to_sum(op: &LoweredOp) -> Option<LoweredOp> {
    match op {
        LoweredOp::Mul(a, b) => product_to_sum(a, b),
        _ => None,
    }
}

/// Double-angle expansion for `sin`/`cos`/`sinh`/`cosh` of `2u`.
///
/// Uses the standard identities
/// ```text
/// sin 2u = 2 sin u cos u ,   cos 2u = cos²u − sin²u ,
/// sinh 2u = 2 sinh u cosh u , cosh 2u = cosh²u + sinh²u ,
/// ```
/// each valid for every real `u`.
fn rule_double_angle(op: &LoweredOp) -> Option<LoweredOp> {
    match op {
        LoweredOp::Sin(arg) => {
            let u = as_double(arg)?;
            Some(mul(cst(2.0), mul(sinf(u.clone()), cosf(u))))
        }
        LoweredOp::Cos(arg) => {
            let u = as_double(arg)?;
            Some(sub(powf(cosf(u.clone()), 2.0), powf(sinf(u), 2.0)))
        }
        LoweredOp::Sinh(arg) => {
            let u = as_double(arg)?;
            Some(mul(cst(2.0), mul(sinhf(u.clone()), coshf(u))))
        }
        LoweredOp::Cosh(arg) => {
            let u = as_double(arg)?;
            Some(add(powf(coshf(u.clone()), 2.0), powf(sinhf(u), 2.0)))
        }
        _ => None,
    }
}

/// Half-angle / power-reduction for a squared `sin`/`cos`.
///
/// ```text
/// sin²u = (1 − cos 2u) / 2 ,   cos²u = (1 + cos 2u) / 2 ,
/// ```
/// both valid for every real `u`.
fn rule_power_reduction(op: &LoweredOp) -> Option<LoweredOp> {
    let (kind, u) = square_head(op)?;
    let two_u = mul(cst(2.0), u);
    match kind {
        Trig::Sin => Some(divi(sub(cst(1.0), cosf(two_u)), cst(2.0))),
        Trig::Cos => Some(divi(add(cst(1.0), cosf(two_u)), cst(2.0))),
        Trig::Tan | Trig::Sinh | Trig::Cosh => None,
    }
}

/// `√(u²) → u`, gated on `u ≥ 0`.
///
/// # Mathematics
///
/// For every real `u`, `√(u²) = |u|`. The rewrite to the *unsigned* `u` is
/// therefore sound only on the sub-domain `u ≥ 0`; on `u < 0` the correct value
/// is `−u`. We emit the candidate `u` only when `u` is *provably* non-negative
/// under the caller's assumptions (see [`is_nonnegative_expr`]); the numeric
/// gate then re-verifies value-equality using probe points restricted to that
/// same non-negative domain. When `u`'s sign is unknown the rule yields `None`
/// and the expression is left in its `√(u²)` (i.e. `|u|`) form.
fn rule_sqrt_of_square(op: &LoweredOp, assumptions: &Assumptions) -> Option<LoweredOp> {
    let half_base = match op {
        LoweredOp::Pow(base, exp) if is_const_near(exp, 0.5) => base,
        _ => return None,
    };
    let inner_base = match half_base.as_ref() {
        LoweredOp::Pow(b, e) if is_const_near(e, 2.0) => b.as_ref(),
        // u·u form of a square.
        LoweredOp::Mul(a, b) if a == b => a.as_ref(),
        _ => return None,
    };
    if is_nonnegative_expr(inner_base, assumptions) {
        Some(inner_base.clone())
    } else {
        None
    }
}

/// Conservative structural test for `op ≥ 0` under `assumptions`.
///
/// Returns `true` only for shapes that are provably non-negative on the reals;
/// unknown cases return `false` (the safe direction — a spurious `false` merely
/// declines a rewrite). The final numeric gate is the ultimate arbiter, so this
/// need not be complete, only *sound*.
fn is_nonnegative_expr(op: &LoweredOp, assumptions: &Assumptions) -> bool {
    match op {
        LoweredOp::Const(c) => *c >= -EXP_EPS,
        LoweredOp::NamedConst(nc) => nc.value() >= -EXP_EPS,
        LoweredOp::Var(i) => assumptions.get(*i).is_nonnegative(),
        // exp(x) > 0 and cosh(x) ≥ 1 for all real x.
        LoweredOp::Exp(_) | LoweredOp::Cosh(_) => true,
        LoweredOp::Pow(base, exp) => {
            if let LoweredOp::Const(ev) = exp.as_ref() {
                let is_int = ev.fract().abs() < EXP_EPS;
                let is_even = is_int && (ev * 0.5).fract().abs() < EXP_EPS;
                if is_even {
                    // real^{even integer} ≥ 0.
                    return true;
                }
            }
            // (non-negative base)^{anything real} ≥ 0.
            is_nonnegative_expr(base, assumptions)
        }
        LoweredOp::Mul(a, b) | LoweredOp::Add(a, b) => {
            is_nonnegative_expr(a, assumptions) && is_nonnegative_expr(b, assumptions)
        }
        _ => false,
    }
}

// ── Pythagorean collapse over flattened sums ────────────────────────────────────

/// A single additive term `coeff · core`, where `core` carries no leading numeric
/// factor.
struct Term {
    coeff: f64,
    core: LoweredOp,
}

/// The five trig/hyperbolic heads a squared term can have.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Trig {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
}

/// Collapse `sin²u+cos²u → 1` and `cosh²u−sinh²u → 1` inside a sum.
///
/// # Mathematics
///
/// The circular Pythagorean identity `sin²u + cos²u = 1` and its hyperbolic
/// counterpart `cosh²u − sinh²u = 1` hold for every real `u`. Working on a
/// flattened list of signed terms makes the collapse independent of the exact
/// associativity/ordering the polynomial canonicaliser produced. A matched pair
/// contributes `min` of the two coefficients to the constant term (so
/// `3 sin²u + cos²u → 1 + 2 sin²u`, etc.). Returns `None` when nothing matched.
fn collapse_pythagorean(node: &LoweredOp) -> Option<LoweredOp> {
    if !is_sum_like(node) {
        return None;
    }
    let (mut terms, mut konst) = flatten(node);
    let mut changed = false;
    loop {
        let a = collapse_sin_cos(&mut terms, &mut konst);
        let b = collapse_cosh_sinh(&mut terms, &mut konst);
        if a || b {
            changed = true;
        } else {
            break;
        }
    }
    changed.then(|| rebuild_sum(terms, konst))
}

/// Collapse `1 + tan²u → 1/cos²u` (`= sec²u`) inside a sum.
///
/// # Mathematics
///
/// `1 + tan²u = sec²u = 1/cos²u` wherever `cos u ≠ 0`. Because this rewrite is
/// only size-*neutral* (it trades an additive `1` and a `tan²` for a reciprocal
/// `cos²`), it is applied separately from the strictly-shrinking Pythagorean
/// collapses and is subject to the same node-count gate. The value gate skips
/// any probe where `cos u = 0` (both sides diverge there).
fn collapse_tan_identity(node: &LoweredOp) -> Option<LoweredOp> {
    if !is_sum_like(node) {
        return None;
    }
    let (mut terms, mut konst) = flatten(node);
    if collapse_tan_sec(&mut terms, &mut konst) {
        Some(rebuild_sum(terms, konst))
    } else {
        None
    }
}

/// Whether `op`'s outermost node is additive (so flattening is meaningful).
fn is_sum_like(op: &LoweredOp) -> bool {
    matches!(
        op,
        LoweredOp::Add(_, _) | LoweredOp::Sub(_, _) | LoweredOp::Neg(_)
    )
}

/// Flatten an additive tree into signed terms plus an accumulated constant.
fn flatten(op: &LoweredOp) -> (Vec<Term>, f64) {
    let mut terms = Vec::new();
    let mut konst = 0.0;
    flatten_into(op, 1.0, &mut terms, &mut konst);
    (terms, konst)
}

fn flatten_into(op: &LoweredOp, sign: f64, terms: &mut Vec<Term>, konst: &mut f64) {
    match op {
        LoweredOp::Add(a, b) => {
            flatten_into(a, sign, terms, konst);
            flatten_into(b, sign, terms, konst);
        }
        LoweredOp::Sub(a, b) => {
            flatten_into(a, sign, terms, konst);
            flatten_into(b, -sign, terms, konst);
        }
        LoweredOp::Neg(a) => flatten_into(a, -sign, terms, konst),
        LoweredOp::Const(c) => *konst += sign * c,
        _ => {
            let (coeff, core) = split_leading_const(op);
            terms.push(Term {
                coeff: sign * coeff,
                core,
            });
        }
    }
}

/// Peel a single leading numeric factor off a multiplicative term.
fn split_leading_const(op: &LoweredOp) -> (f64, LoweredOp) {
    match op {
        LoweredOp::Mul(a, b) => {
            if let LoweredOp::Const(c) = a.as_ref() {
                return (*c, (**b).clone());
            }
            if let LoweredOp::Const(c) = b.as_ref() {
                return (*c, (**a).clone());
            }
            (1.0, op.clone())
        }
        LoweredOp::Neg(a) => {
            let (c, core) = split_leading_const(a);
            (-c, core)
        }
        _ => (1.0, op.clone()),
    }
}

/// Rebuild an additive expression from signed terms and a constant.
fn rebuild_sum(terms: Vec<Term>, konst: f64) -> LoweredOp {
    let mut acc: Option<LoweredOp> = if konst.abs() > 1e-15 {
        Some(cst(konst))
    } else {
        None
    };
    for term in terms {
        if term.coeff.abs() < 1e-15 {
            continue;
        }
        let piece = if (term.coeff - 1.0).abs() < 1e-15 {
            term.core
        } else if (term.coeff + 1.0).abs() < 1e-15 {
            neg(term.core)
        } else {
            mul(cst(term.coeff), term.core)
        };
        acc = Some(match acc {
            None => piece,
            Some(prev) => add(prev, piece),
        });
    }
    acc.unwrap_or_else(|| cst(0.0))
}

/// Recognise a squared trig/hyperbolic term `head(u)²`, returning `(head, u)`.
///
/// Handles both the `Pow(head(u), 2)` encoding the polynomial canonicaliser
/// emits and the raw `head(u) · head(u)` product form.
fn square_head(core: &LoweredOp) -> Option<(Trig, LoweredOp)> {
    let base = match core {
        LoweredOp::Pow(b, e) if is_const_near(e, 2.0) => b.as_ref(),
        LoweredOp::Mul(a, b) if a == b => a.as_ref(),
        _ => return None,
    };
    let (kind, u) = match base {
        LoweredOp::Sin(u) => (Trig::Sin, u),
        LoweredOp::Cos(u) => (Trig::Cos, u),
        LoweredOp::Tan(u) => (Trig::Tan, u),
        LoweredOp::Sinh(u) => (Trig::Sinh, u),
        LoweredOp::Cosh(u) => (Trig::Cosh, u),
        _ => return None,
    };
    Some((kind, (**u).clone()))
}

/// Find indices `(i, j)` with `terms[i] = a(u)²` and `terms[j] = b(u)²` sharing
/// the same argument `u`.
fn find_square_pair(terms: &[Term], a: Trig, b: Trig) -> Option<(usize, usize)> {
    for (i, ti) in terms.iter().enumerate() {
        let ui = match square_head(&ti.core) {
            Some((kind, u)) if kind == a => u,
            _ => continue,
        };
        for (j, tj) in terms.iter().enumerate() {
            if j == i {
                continue;
            }
            if let Some((kind, uj)) = square_head(&tj.core) {
                if kind == b && uj == ui {
                    return Some((i, j));
                }
            }
        }
    }
    None
}

/// One `sin²u + cos²u → 1` collapse (general coefficient); `true` if applied.
fn collapse_sin_cos(terms: &mut Vec<Term>, konst: &mut f64) -> bool {
    if let Some((i, j)) = find_square_pair(terms, Trig::Sin, Trig::Cos) {
        // Indices came from `find_square_pair` over `terms` and are in-bounds.
        if terms[i].coeff > 1e-12 && terms[j].coeff > 1e-12 {
            let m = terms[i].coeff.min(terms[j].coeff);
            terms[i].coeff -= m;
            terms[j].coeff -= m;
            *konst += m;
            terms.retain(|t| t.coeff.abs() > 1e-12);
            return true;
        }
    }
    false
}

/// One `cosh²u − sinh²u → 1` collapse (general coefficient); `true` if applied.
fn collapse_cosh_sinh(terms: &mut Vec<Term>, konst: &mut f64) -> bool {
    if let Some((i, j)) = find_square_pair(terms, Trig::Cosh, Trig::Sinh) {
        // cosh² must appear with a positive coefficient, sinh² with a negative.
        if terms[i].coeff > 1e-12 && terms[j].coeff < -1e-12 {
            let m = terms[i].coeff.min(-terms[j].coeff);
            terms[i].coeff -= m;
            terms[j].coeff += m;
            *konst += m;
            terms.retain(|t| t.coeff.abs() > 1e-12);
            return true;
        }
    }
    false
}

/// One `1 + tan²u → 1/cos²u` collapse; `true` if applied.
fn collapse_tan_sec(terms: &mut Vec<Term>, konst: &mut f64) -> bool {
    if *konst <= 1e-12 {
        return false;
    }
    let idx = terms
        .iter()
        .position(|t| matches!(square_head(&t.core), Some((Trig::Tan, _))) && t.coeff > 1e-12);
    let i = match idx {
        Some(i) => i,
        None => return false,
    };
    let u = match square_head(&terms[i].core) {
        Some((_, u)) => u,
        None => return false,
    };
    let m = konst.min(terms[i].coeff);
    if m <= 1e-12 {
        return false;
    }
    *konst -= m;
    terms[i].coeff -= m;
    let sec2 = divi(cst(1.0), powf(cosf(u), 2.0));
    terms.retain(|t| t.coeff.abs() > 1e-12);
    terms.push(Term {
        coeff: m,
        core: sec2,
    });
    true
}

// ── Acceptance gates ────────────────────────────────────────────────────────────

/// Accept a contracting rewrite: value-preserving on the asserted domain and not
/// larger than the original.
fn accept_contracting(orig: &LoweredOp, candidate: &LoweredOp, assumptions: &Assumptions) -> bool {
    node_count(candidate) <= node_count(orig) && verify_domain(orig, candidate, assumptions)
}

/// Accept an expanding rewrite: it must actually change the tree and stay
/// value-preserving on the asserted domain.
fn accept_expanding(orig: &LoweredOp, candidate: &LoweredOp, assumptions: &Assumptions) -> bool {
    ops_struct_hash(orig) != ops_struct_hash(candidate)
        && verify_domain(orig, candidate, assumptions)
}

/// Check that `candidate` agrees with `orig` at the [`PROBE_POINTS`], with each
/// variable's probe value mapped into its asserted domain.
///
/// Non-finite samples (poles, out-of-domain logs) are skipped; at least two
/// finite agreements are required so a rewrite is never accepted purely because
/// both sides were undefined everywhere probed.
fn verify_domain(orig: &LoweredOp, candidate: &LoweredOp, assumptions: &Assumptions) -> bool {
    let n = orig.count_vars().max(candidate.count_vars()).max(1);
    let len = PROBE_POINTS.len();
    let mut agreed = 0usize;
    for k in 0..len {
        let vars: Vec<f64> = (0..n)
            .map(|i| {
                let base = PROBE_POINTS.get((k + i) % len).copied().unwrap_or(1.0);
                domain_value(i, base, assumptions)
            })
            .collect();
        let ov = orig.eval(&vars);
        let cv = candidate.eval(&vars);
        if !ov.is_finite() || !cv.is_finite() {
            continue;
        }
        let tol = 1e-9 * (1.0 + ov.abs());
        if (ov - cv).abs() > tol {
            return false;
        }
        agreed += 1;
    }
    agreed >= 2
}

/// Map a raw probe scalar for variable `var` into its asserted domain.
///
/// * `nonnegative` / `positive`: take the absolute value (and keep it strictly
///   positive for `positive`), so a gated `√(u²) → u` is validated only where
///   `u ≥ 0`.
/// * `integer`: round to the nearest integer.
///
/// A variable with no assumptions keeps the raw probe (including the negative
/// probe points), which is what makes unconditional identities get tested across
/// the sign change.
fn domain_value(var: usize, base: f64, assumptions: &Assumptions) -> f64 {
    let a = assumptions.get(var);
    let mut v = base;
    if a.integer {
        v = v.round();
    }
    if a.is_positive() {
        if a.integer {
            if v < 1.0 {
                v = 1.0;
            }
        } else {
            v = v.abs();
            if v < 1e-3 {
                v = 0.5;
            }
        }
    } else if a.is_nonnegative() {
        v = v.abs();
    }
    v
}

// ── Small tree constructors ─────────────────────────────────────────────────────

fn cst(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Arc::new(a), Arc::new(b))
}
fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Sub(Arc::new(a), Arc::new(b))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Arc::new(a), Arc::new(b))
}
fn divi(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Arc::new(a), Arc::new(b))
}
fn neg(a: LoweredOp) -> LoweredOp {
    LoweredOp::Neg(Arc::new(a))
}
fn powf(base: LoweredOp, exp: f64) -> LoweredOp {
    LoweredOp::Pow(Arc::new(base), Arc::new(LoweredOp::Const(exp)))
}
fn sinf(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sin(Arc::new(a))
}
fn cosf(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cos(Arc::new(a))
}
fn sinhf(a: LoweredOp) -> LoweredOp {
    LoweredOp::Sinh(Arc::new(a))
}
fn coshf(a: LoweredOp) -> LoweredOp {
    LoweredOp::Cosh(Arc::new(a))
}

/// Whether `op` is a constant within [`EXP_EPS`] of `v`.
fn is_const_near(op: &LoweredOp, v: f64) -> bool {
    matches!(op, LoweredOp::Const(c) if (*c - v).abs() < EXP_EPS)
}

/// `2u` detector: returns `u` when `arg` is `2·u`, `u·2` or `u+u`.
fn as_double(arg: &LoweredOp) -> Option<LoweredOp> {
    match arg {
        LoweredOp::Mul(a, b) => {
            if is_const_near(a, 2.0) {
                return Some((**b).clone());
            }
            if is_const_near(b, 2.0) {
                return Some((**a).clone());
            }
            None
        }
        LoweredOp::Add(a, b) if a == b => Some((**a).clone()),
        _ => None,
    }
}

/// Rebuild `op` with `f` applied to each direct child.
fn map_children<F: Fn(&LoweredOp) -> LoweredOp>(op: &LoweredOp, f: F) -> LoweredOp {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(f(a))),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(f(a))),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(f(a))),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(f(a))),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(f(a))),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(f(a))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(f(a))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(f(a))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(f(a))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(f(a))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(f(a))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(f(a))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(f(a))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(f(a))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(f(a))),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(f(a))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(f(a))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(f(a))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(f(a))),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(f(a))),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(f(a))),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(f(a))),
        LoweredOp::Add(a, b) => LoweredOp::Add(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Div(a, b) => LoweredOp::Div(Arc::new(f(a)), Arc::new(f(b))),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(Arc::new(f(a)), Arc::new(f(b))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    #[test]
    fn sin2_plus_cos2_collapses_to_one() {
        let x = var(0);
        let expr = add(powf(sinf(x.clone()), 2.0), powf(cosf(x), 2.0));
        let simplified = expr.simplify_with(&Assumptions::default());
        assert!(
            matches!(&simplified, LoweredOp::Const(c) if (*c - 1.0).abs() < 1e-12),
            "sin²x + cos²x should be 1, got {simplified}"
        );
    }

    #[test]
    fn cosh2_minus_sinh2_collapses_to_one() {
        let x = var(0);
        let expr = sub(powf(coshf(x.clone()), 2.0), powf(sinhf(x), 2.0));
        let simplified = expr.simplify_with(&Assumptions::default());
        assert!(
            matches!(&simplified, LoweredOp::Const(c) if (*c - 1.0).abs() < 1e-12),
            "cosh²x − sinh²x should be 1, got {simplified}"
        );
    }

    #[test]
    fn one_plus_tan2_becomes_reciprocal_cos2() {
        let x = var(0);
        let expr = add(cst(1.0), powf(LoweredOp::Tan(Arc::new(x)), 2.0));
        let simplified = expr.simplify_with(&Assumptions::default());
        // Should be 1/cos²x (sec²x), value-equal to 1 + tan²x.
        for probe in [0.3_f64, 0.7, 1.3, -0.5] {
            let want = 1.0 + probe.tan().powi(2);
            let got = simplified.eval(&[probe]);
            assert!(
                (got - want).abs() < 1e-9,
                "sec² mismatch at {probe}: got {got}, want {want}"
            );
        }
        assert!(
            !matches!(&simplified, LoweredOp::Add(_, _)),
            "1 + tan²x should have been rewritten, got {simplified}"
        );
    }

    #[test]
    fn sqrt_of_square_needs_nonnegative_assumption() {
        // √(x²) = Pow(Pow(x, 2), 0.5).
        let x = var(0);
        let expr = powf(powf(x, 2.0), 0.5);

        // Without an assumption the sign of x is unknown → left as |x| (unchanged).
        let unchanged = expr.simplify_with(&Assumptions::default());
        assert!(
            !matches!(&unchanged, LoweredOp::Var(0)),
            "√(x²) must not collapse to x without an assumption, got {unchanged}"
        );
        // The unchanged form still evaluates to |x| (differs from x at x<0).
        assert!((unchanged.eval(&[-2.0]) - 2.0).abs() < 1e-9);

        // With x ≥ 0 the collapse to x is sound and must fire.
        let asm = Assumptions::new().assume_nonnegative(0);
        let collapsed = expr.simplify_with(&asm);
        assert_eq!(collapsed, LoweredOp::Var(0), "√(x²) with x≥0 should be x");
    }

    #[test]
    fn product_to_sum_is_opt_in() {
        // sin(x0) · cos(x1): distinct arguments give a clean sum when enabled.
        let expr = mul(sinf(var(0)), cosf(var(1)));

        let default = expr.simplify_with(&Assumptions::default());
        assert!(
            matches!(&default, LoweredOp::Mul(_, _)),
            "product-to-sum must NOT fire by default, got {default}"
        );

        let flags = RewriteFlags::default().with_product_to_sum();
        let enabled = expr.simplify_with_flags(&Assumptions::default(), flags);
        assert_ne!(
            ops_struct_hash(&default),
            ops_struct_hash(&enabled),
            "product-to-sum should change the tree when enabled"
        );
        // Value is preserved by the expansion.
        for (a, b) in [(0.3_f64, 1.1_f64), (-0.5, 0.7)] {
            let want = a.sin() * b.cos();
            let got = enabled.eval(&[a, b]);
            assert!((got - want).abs() < 1e-9, "value changed: {got} vs {want}");
        }
    }

    #[test]
    fn double_angle_expands_when_enabled() {
        // sin(2x) → 2 sin x cos x, only with the flag.
        let expr = sinf(mul(cst(2.0), var(0)));
        let flags = RewriteFlags::default().with_double_angle();
        let enabled = expr.simplify_with_flags(&Assumptions::default(), flags);
        for probe in [0.3_f64, 0.7, -0.5, 1.3] {
            let want = (2.0 * probe).sin();
            assert!((enabled.eval(&[probe]) - want).abs() < 1e-9);
        }
        // The bare 2x argument must be gone.
        assert!(!matches!(&enabled, LoweredOp::Sin(_)));
    }

    #[test]
    fn default_leaves_plain_polynomials_identical_to_simplify() {
        // A tree with no trig-Pythagorean / sqrt pattern must be byte-identical
        // to what the base `simplify` produces.
        let x = Arc::new(LoweredOp::Var(0));
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Mul(Arc::clone(&x), Arc::clone(&x))),
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(2.0)),
                Arc::clone(&x),
            )),
        );
        let base = expr.simplify();
        let with = expr.simplify_with(&Assumptions::default());
        assert_eq!(
            ops_struct_hash(&base),
            ops_struct_hash(&with),
            "simplify_with(default) diverged from simplify: {base} vs {with}"
        );
    }

    #[test]
    fn log_combine_flag_merges_logs() {
        // ln(x) + ln(y) → ln(x·y) under the log_combine flag.
        let expr = add(
            LoweredOp::Ln(Arc::new(var(0))),
            LoweredOp::Ln(Arc::new(var(1))),
        );
        let flags = RewriteFlags::default().with_log_combine();
        let combined = expr.simplify_with_flags(&Assumptions::default(), flags);
        for (a, b) in [(1.5_f64, 2.0_f64), (0.5, 3.0)] {
            let want = a.ln() + b.ln();
            assert!((combined.eval(&[a, b]) - want).abs() < 1e-9);
        }
    }
}
