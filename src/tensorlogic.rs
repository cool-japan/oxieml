//! TensorLogic IR bridge (feature: `tensorlogic`).
//!
//! Converts OxiEML's [`LoweredOp`] to and from [`tensorlogic_ir::TLExpr`],
//! and exports EML canonical identities as
//! [`tensorlogic_ir::RewriteRule`]s.
//!
//! # Variable Convention
//!
//! The bridge represents a `LoweredOp::Var(i)` as a `TLExpr::Pred` with
//! name `"x{i}"` and a single `Term::var("x{i}")` argument. This keeps the
//! variable addressable through `tensorlogic-ir`'s predicate-centric pattern
//! matcher while remaining trivially reversible.
//!
//! # Negation Round-Trip
//!
//! [`to_tlexpr`] lowers `LoweredOp::Neg(a)` as `TLExpr::Sub(0.0, a)` because
//! `TLExpr` has no dedicated unary negation variant. Consequently a
//! `Neg`-containing expression does not round-trip *structurally* through
//! [`from_tlexpr`] — it returns an equivalent `LoweredOp::Sub` instead.
//! Numerical evaluation is preserved.
//!
//! ## TensorLogic training pipeline
//!
//! The high-level entry point for TL training integration is
//! [`formulas_to_tl_weighted_rules`]: it converts a ranked set of discovered
//! formulas into `TLExpr::WeightedRule` soft-prior constraints ready for a
//! TensorLogic training system to ingest.
//!
//! ```ignore
//! use oxieml::tensorlogic;
//! let constraints = tensorlogic::formulas_to_tl_weighted_rules(&formulas, &weights)?;
//! // On the TL side (in tensorlogic-train, which holds the SciRS2 dep):
//! // trainer.add_soft_priors(constraints);
//! ```
//!
//! No OxiEML→SciRS2 dependency is introduced. This module depends only on
//! `tensorlogic-ir` (SciRS2-free: deps are serde, serde_json, oxicode, chrono, thiserror).
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "tensorlogic")]
//! # {
//! use oxieml::lower::LoweredOp;
//! use oxieml::tensorlogic::{from_tlexpr, to_tlexpr};
//!
//! let op = LoweredOp::Add(
//!     std::sync::Arc::new(LoweredOp::Const(3.0)),
//!     std::sync::Arc::new(LoweredOp::Var(0)),
//! );
//! let tl = to_tlexpr(&op);
//! let round_trip = from_tlexpr(&tl).expect("bridge round-trip");
//! assert_eq!(op, round_trip);
//! # }
//! ```

use crate::error::EmlError;
use crate::lower::LoweredOp;
use std::sync::Arc;
use tensorlogic_ir::{Pattern, RewriteRule, TLExpr, Term};

/// Build the `TLExpr::Pred` used to represent `LoweredOp::Var(i)`.
///
/// The name is `x{i}` and the single argument is `Term::var("x{i}")`.
fn var_pred(i: usize) -> TLExpr {
    let name = format!("x{i}");
    TLExpr::pred(name.clone(), vec![Term::var(name)])
}

/// If `expr` is the predicate produced by [`var_pred`], return the variable
/// index. Returns `None` for any other shape (including predicates whose
/// name does not parse as `x<index>`).
fn match_var_pred(expr: &TLExpr) -> Option<usize> {
    if let TLExpr::Pred { name, args } = expr {
        if args.len() == 1 {
            if let Some(rest) = name.strip_prefix('x') {
                if let Ok(idx) = rest.parse::<usize>() {
                    // Verify the single arg is the matching variable term.
                    if let Term::Var(arg_name) = &args[0] {
                        if arg_name == name {
                            return Some(idx);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Convert an OxiEML [`LoweredOp`] tree into a TensorLogic [`TLExpr`].
///
/// Variables become `TLExpr::Pred { name: "x{i}", args: [Term::var("x{i}")] }`.
/// Arithmetic and transcendental ops map directly onto their `TLExpr`
/// counterparts. `LoweredOp::Ln` maps to `TLExpr::Log`. `LoweredOp::Neg(a)`
/// is encoded as `TLExpr::Sub(Constant(0.0), a)` because `TLExpr` lacks a
/// dedicated unary negation.
pub fn to_tlexpr(op: &LoweredOp) -> TLExpr {
    match op {
        LoweredOp::Const(v) => TLExpr::Constant(*v),
        LoweredOp::NamedConst(nc) => TLExpr::Constant(nc.value()),
        LoweredOp::Var(i) => var_pred(*i),
        LoweredOp::Add(a, b) => TLExpr::add(to_tlexpr(a), to_tlexpr(b)),
        LoweredOp::Sub(a, b) => TLExpr::sub(to_tlexpr(a), to_tlexpr(b)),
        LoweredOp::Mul(a, b) => TLExpr::mul(to_tlexpr(a), to_tlexpr(b)),
        LoweredOp::Div(a, b) => TLExpr::div(to_tlexpr(a), to_tlexpr(b)),
        LoweredOp::Pow(a, b) => TLExpr::pow(to_tlexpr(a), to_tlexpr(b)),
        LoweredOp::Exp(a) => TLExpr::exp(to_tlexpr(a)),
        LoweredOp::Ln(a) => TLExpr::log(to_tlexpr(a)),
        LoweredOp::Sin(a) => TLExpr::sin(to_tlexpr(a)),
        LoweredOp::Cos(a) => TLExpr::cos(to_tlexpr(a)),
        LoweredOp::Neg(a) => TLExpr::sub(TLExpr::Constant(0.0), to_tlexpr(a)),
        // Tan has a native TLExpr variant
        LoweredOp::Tan(a) => TLExpr::tan(to_tlexpr(a)),
        // Sinh: expand as (exp(x) - exp(-x)) / 2
        LoweredOp::Sinh(a) => {
            let x = to_tlexpr(a);
            let neg_x = TLExpr::sub(TLExpr::Constant(0.0), x.clone());
            TLExpr::div(
                TLExpr::sub(TLExpr::exp(x), TLExpr::exp(neg_x)),
                TLExpr::Constant(2.0),
            )
        }
        // Cosh: expand as (exp(x) + exp(-x)) / 2
        LoweredOp::Cosh(a) => {
            let x = to_tlexpr(a);
            let neg_x = TLExpr::sub(TLExpr::Constant(0.0), x.clone());
            TLExpr::div(
                TLExpr::add(TLExpr::exp(x), TLExpr::exp(neg_x)),
                TLExpr::Constant(2.0),
            )
        }
        // Tanh: expand as sinh(x) / cosh(x) via their expansions
        LoweredOp::Tanh(a) => {
            let x = to_tlexpr(a);
            let neg_x = TLExpr::sub(TLExpr::Constant(0.0), x.clone());
            let sinh_x = TLExpr::div(
                TLExpr::sub(TLExpr::exp(x.clone()), TLExpr::exp(neg_x.clone())),
                TLExpr::Constant(2.0),
            );
            let cosh_x = TLExpr::div(
                TLExpr::add(TLExpr::exp(x), TLExpr::exp(neg_x)),
                TLExpr::Constant(2.0),
            );
            TLExpr::div(sinh_x, cosh_x)
        }
        // Arcsinh: expand as ln(x + sqrt(x^2 + 1))
        LoweredOp::Arcsinh(a) => {
            let x = to_tlexpr(a);
            let x_sq = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
            let inner = TLExpr::add(x_sq, TLExpr::Constant(1.0));
            TLExpr::log(TLExpr::add(x, TLExpr::sqrt(inner)))
        }
        // Arccosh: expand as ln(x + sqrt(x^2 - 1))
        LoweredOp::Arccosh(a) => {
            let x = to_tlexpr(a);
            let x_sq = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
            let inner = TLExpr::sub(x_sq, TLExpr::Constant(1.0));
            TLExpr::log(TLExpr::add(x, TLExpr::sqrt(inner)))
        }
        // Arctanh: expand as (ln(1+x) - ln(1-x)) / 2
        LoweredOp::Arctanh(a) => {
            let x = to_tlexpr(a);
            let one_plus_x = TLExpr::add(TLExpr::Constant(1.0), x.clone());
            let one_minus_x = TLExpr::sub(TLExpr::Constant(1.0), x);
            TLExpr::div(
                TLExpr::sub(TLExpr::log(one_plus_x), TLExpr::log(one_minus_x)),
                TLExpr::Constant(2.0),
            )
        }
        // Arctan, Arcsin, Arccos: no native TLExpr variant and no clean real-valued
        // closed-form expansion in {exp, log, sqrt} without complex arithmetic.
        // Return NaN sentinel to avoid breaking infallible callers.
        LoweredOp::Arctan(_) | LoweredOp::Arcsin(_) | LoweredOp::Arccos(_) => {
            TLExpr::Constant(f64::NAN)
        }
        // Special functions: no TLExpr native variant, return NaN sentinel
        LoweredOp::Erf(_)
        | LoweredOp::LGamma(_)
        | LoweredOp::Digamma(_)
        | LoweredOp::Trigamma(_)
        | LoweredOp::Ei(_)
        | LoweredOp::Si(_)
        | LoweredOp::Ci(_) => TLExpr::Constant(f64::NAN),
    }
}

/// Convert a TensorLogic [`TLExpr`] back into an OxiEML [`LoweredOp`].
///
/// Supports the arithmetic/transcendental subset emitted by [`to_tlexpr`]
/// plus any syntactically-equivalent expression constructed directly. Any
/// variant outside that subset yields
/// [`EmlError::UnsupportedTlExpr`].
///
/// Note that `TLExpr::Sub(Constant(0.0), a)` is *not* collapsed back into
/// `LoweredOp::Neg`; it becomes `LoweredOp::Sub(Const(0.0), from(a))`.
/// Numerically identical; structurally distinct.
pub fn from_tlexpr(expr: &TLExpr) -> Result<LoweredOp, EmlError> {
    match expr {
        TLExpr::Constant(v) => Ok(LoweredOp::Const(*v)),
        TLExpr::Pred { name, args } => match match_var_pred(expr) {
            Some(idx) => Ok(LoweredOp::Var(idx)),
            None => Err(EmlError::UnsupportedTlExpr(format!(
                "Pred {{ name: {name:?}, arity: {} }} does not match the \
                 OxiEML variable convention `x<usize>`",
                args.len()
            ))),
        },
        TLExpr::Add(a, b) => Ok(LoweredOp::Add(
            Arc::new(from_tlexpr(a)?),
            Arc::new(from_tlexpr(b)?),
        )),
        TLExpr::Sub(a, b) => Ok(LoweredOp::Sub(
            Arc::new(from_tlexpr(a)?),
            Arc::new(from_tlexpr(b)?),
        )),
        TLExpr::Mul(a, b) => Ok(LoweredOp::Mul(
            Arc::new(from_tlexpr(a)?),
            Arc::new(from_tlexpr(b)?),
        )),
        TLExpr::Div(a, b) => Ok(LoweredOp::Div(
            Arc::new(from_tlexpr(a)?),
            Arc::new(from_tlexpr(b)?),
        )),
        TLExpr::Pow(a, b) => Ok(LoweredOp::Pow(
            Arc::new(from_tlexpr(a)?),
            Arc::new(from_tlexpr(b)?),
        )),
        TLExpr::Exp(a) => Ok(LoweredOp::Exp(Arc::new(from_tlexpr(a)?))),
        TLExpr::Log(a) => Ok(LoweredOp::Ln(Arc::new(from_tlexpr(a)?))),
        TLExpr::Sin(a) => Ok(LoweredOp::Sin(Arc::new(from_tlexpr(a)?))),
        TLExpr::Cos(a) => Ok(LoweredOp::Cos(Arc::new(from_tlexpr(a)?))),
        TLExpr::Tan(a) => Ok(LoweredOp::Tan(Arc::new(from_tlexpr(a)?))),
        other => Err(EmlError::UnsupportedTlExpr(describe_variant(other))),
    }
}

/// Human-readable tag identifying a `TLExpr` variant for error reporting.
///
/// We intentionally avoid pretty-printing the whole expression (that would
/// recurse into arbitrary subtrees and is not what `UnsupportedTlExpr`
/// needs to communicate).
fn describe_variant(expr: &TLExpr) -> String {
    let tag = match expr {
        TLExpr::Pred { .. } => "Pred",
        TLExpr::And(_, _) => "And",
        TLExpr::Or(_, _) => "Or",
        TLExpr::Not(_) => "Not",
        TLExpr::Exists { .. } => "Exists",
        TLExpr::ForAll { .. } => "ForAll",
        TLExpr::Imply(_, _) => "Imply",
        TLExpr::Score(_) => "Score",
        TLExpr::Add(_, _) => "Add",
        TLExpr::Sub(_, _) => "Sub",
        TLExpr::Mul(_, _) => "Mul",
        TLExpr::Div(_, _) => "Div",
        TLExpr::Pow(_, _) => "Pow",
        TLExpr::Mod(_, _) => "Mod",
        TLExpr::Min(_, _) => "Min",
        TLExpr::Max(_, _) => "Max",
        TLExpr::Abs(_) => "Abs",
        TLExpr::Floor(_) => "Floor",
        TLExpr::Ceil(_) => "Ceil",
        TLExpr::Round(_) => "Round",
        TLExpr::Sqrt(_) => "Sqrt",
        TLExpr::Exp(_) => "Exp",
        TLExpr::Log(_) => "Log",
        TLExpr::Sin(_) => "Sin",
        TLExpr::Cos(_) => "Cos",
        TLExpr::Tan(_) => "Tan",
        TLExpr::Eq(_, _) => "Eq",
        TLExpr::Lt(_, _) => "Lt",
        TLExpr::Gt(_, _) => "Gt",
        TLExpr::Lte(_, _) => "Lte",
        TLExpr::Gte(_, _) => "Gte",
        TLExpr::IfThenElse { .. } => "IfThenElse",
        TLExpr::Constant(_) => "Constant",
        TLExpr::Aggregate { .. } => "Aggregate",
        TLExpr::Let { .. } => "Let",
        TLExpr::Box(_) => "Box",
        TLExpr::Diamond(_) => "Diamond",
        TLExpr::Next(_) => "Next",
        TLExpr::Eventually(_) => "Eventually",
        TLExpr::Always(_) => "Always",
        TLExpr::Until { .. } => "Until",
        TLExpr::TNorm { .. } => "TNorm",
        TLExpr::TCoNorm { .. } => "TCoNorm",
        TLExpr::FuzzyNot { .. } => "FuzzyNot",
        TLExpr::FuzzyImplication { .. } => "FuzzyImplication",
        TLExpr::SoftExists { .. } => "SoftExists",
        TLExpr::SoftForAll { .. } => "SoftForAll",
        TLExpr::WeightedRule { .. } => "WeightedRule",
        TLExpr::ProbabilisticChoice { .. } => "ProbabilisticChoice",
        TLExpr::Release { .. } => "Release",
        TLExpr::WeakUntil { .. } => "WeakUntil",
        TLExpr::StrongRelease { .. } => "StrongRelease",
        TLExpr::Lambda { .. } => "Lambda",
        TLExpr::Apply { .. } => "Apply",
        TLExpr::SetMembership { .. } => "SetMembership",
        TLExpr::SetUnion { .. } => "SetUnion",
        TLExpr::SetIntersection { .. } => "SetIntersection",
        TLExpr::SetDifference { .. } => "SetDifference",
        TLExpr::SetCardinality { .. } => "SetCardinality",
        TLExpr::EmptySet => "EmptySet",
        TLExpr::SetComprehension { .. } => "SetComprehension",
        TLExpr::CountingExists { .. } => "CountingExists",
        TLExpr::CountingForAll { .. } => "CountingForAll",
        TLExpr::ExactCount { .. } => "ExactCount",
        TLExpr::Majority { .. } => "Majority",
        TLExpr::LeastFixpoint { .. } => "LeastFixpoint",
        TLExpr::GreatestFixpoint { .. } => "GreatestFixpoint",
        TLExpr::Nominal { .. } => "Nominal",
        TLExpr::At { .. } => "At",
        TLExpr::Somewhere { .. } => "Somewhere",
        TLExpr::Everywhere { .. } => "Everywhere",
        TLExpr::AllDifferent { .. } => "AllDifferent",
        TLExpr::GlobalCardinality { .. } => "GlobalCardinality",
        TLExpr::Abducible { .. } => "Abducible",
        TLExpr::Explain { .. } => "Explain",
        TLExpr::SymbolLiteral(_) => "SymbolLiteral",
        TLExpr::Match { .. } => "Match",
    };
    tag.to_string()
}

/// Canonical rewrite rules for OxiEML algebraic identities.
///
/// Returns a `Vec<RewriteRule>` encoding the following ten algebraic
/// identities as pattern/template pairs usable by
/// `tensorlogic_ir::RewriteSystem`:
///
/// 1. `exp(log(x)) → x`
/// 2. `log(exp(x)) → x`
/// 3. `neg(neg(x)) → x`
/// 4. `0 + x → x`
/// 5. `x + 0 → x`
/// 6. `x * 1 → x`
/// 7. `1 * x → x`
/// 8. `x / 1 → x`
/// 9. `x ^ 0 → 1`
/// 10. `x ^ 1 → x`
///
/// Each rule's `template` closure safely retrieves the matched binding
/// via `HashMap::get`. Because the template is only invoked after a
/// successful pattern match, the binding is guaranteed to exist; a
/// `NaN` sentinel is returned as a defensive fallback.
///
/// [`canonical_simplify`] applies the same identities directly to a
/// `TLExpr` tree and is the recommended choice when an in-place
/// simplification pass is preferred over declarative rule lists.
pub fn canonical_rewrite_rules() -> Vec<RewriteRule> {
    /// Defensive fallback for when a binding is unexpectedly missing.
    /// Pattern matching guarantees the key exists, so this should never
    /// be reached in practice.
    fn bound(bindings: &std::collections::HashMap<String, TLExpr>, key: &str) -> TLExpr {
        bindings
            .get(key)
            .cloned()
            .unwrap_or(TLExpr::Constant(f64::NAN))
    }

    vec![
        // 1. exp(log(x)) → x
        RewriteRule {
            pattern: Pattern::exp(Pattern::log(Pattern::var("x"))),
            template: |b| bound(b, "x"),
            name: Some("exp_log_inverse".to_string()),
        },
        // 2. log(exp(x)) → x
        RewriteRule {
            pattern: Pattern::log(Pattern::exp(Pattern::var("x"))),
            template: |b| bound(b, "x"),
            name: Some("log_exp_inverse".to_string()),
        },
        // 3. neg(neg(x)) → x
        RewriteRule {
            pattern: Pattern::neg(Pattern::neg(Pattern::var("x"))),
            template: |b| bound(b, "x"),
            name: Some("double_negation".to_string()),
        },
        // 4. 0 + x → x
        RewriteRule {
            pattern: Pattern::add(Pattern::constant(0.0), Pattern::var("x")),
            template: |b| bound(b, "x"),
            name: Some("zero_add_left".to_string()),
        },
        // 5. x + 0 → x
        RewriteRule {
            pattern: Pattern::add(Pattern::var("x"), Pattern::constant(0.0)),
            template: |b| bound(b, "x"),
            name: Some("zero_add_right".to_string()),
        },
        // 6. x * 1 → x
        RewriteRule {
            pattern: Pattern::mul(Pattern::var("x"), Pattern::constant(1.0)),
            template: |b| bound(b, "x"),
            name: Some("one_mul_right".to_string()),
        },
        // 7. 1 * x → x
        RewriteRule {
            pattern: Pattern::mul(Pattern::constant(1.0), Pattern::var("x")),
            template: |b| bound(b, "x"),
            name: Some("one_mul_left".to_string()),
        },
        // 8. x / 1 → x
        RewriteRule {
            pattern: Pattern::div(Pattern::var("x"), Pattern::constant(1.0)),
            template: |b| bound(b, "x"),
            name: Some("div_by_one".to_string()),
        },
        // 9. x ^ 0 → 1
        RewriteRule {
            pattern: Pattern::pow(Pattern::var("_x"), Pattern::constant(0.0)),
            template: |_b| TLExpr::Constant(1.0),
            name: Some("pow_zero".to_string()),
        },
        // 10. x ^ 1 → x
        RewriteRule {
            pattern: Pattern::pow(Pattern::var("x"), Pattern::constant(1.0)),
            template: |b| bound(b, "x"),
            name: Some("pow_one".to_string()),
        },
    ]
}

/// Absolute tolerance used when comparing an `f64` constant against `0.0`
/// or `1.0` while looking for algebraic identity shortcuts.
///
/// Structural rewrites only fire when the `TLExpr::Constant(c)` is literally
/// (to within rounding noise) the identity element for the surrounding
/// operator, so the tolerance is intentionally tight.
const CANONICAL_EPS: f64 = 1e-15;

/// Returns `true` if `e` is a `TLExpr::Constant` numerically equal to `0.0`
/// (within [`CANONICAL_EPS`]).
fn is_const_zero(e: &TLExpr) -> bool {
    matches!(e, TLExpr::Constant(c) if c.abs() < CANONICAL_EPS)
}

/// Returns `true` if `e` is a `TLExpr::Constant` numerically equal to `1.0`
/// (within [`CANONICAL_EPS`]).
fn is_const_one(e: &TLExpr) -> bool {
    matches!(e, TLExpr::Constant(c) if (c - 1.0).abs() < CANONICAL_EPS)
}

/// Apply a single bottom-up pass of canonical rewrites to `expr`.
///
/// Children are simplified first, then local identities are checked at the
/// current node. The catch-all arm intentionally leaves non-arithmetic
/// variants (predicates, logical connectives, quantifiers, ...) alone so
/// that [`canonical_simplify`] is a pure arithmetic/transcendental rewriter
/// on the subset of `TLExpr` produced by [`to_tlexpr`].
fn simplify_one_pass(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::Add(a, b) => {
            let a = simplify_one_pass(a);
            let b = simplify_one_pass(b);
            if let (TLExpr::Constant(ac), TLExpr::Constant(bc)) = (&a, &b) {
                return TLExpr::Constant(ac + bc);
            }
            if is_const_zero(&a) {
                return b;
            }
            if is_const_zero(&b) {
                return a;
            }
            TLExpr::add(a, b)
        }
        TLExpr::Sub(a, b) => {
            let a = simplify_one_pass(a);
            let b = simplify_one_pass(b);
            if let (TLExpr::Constant(ac), TLExpr::Constant(bc)) = (&a, &b) {
                return TLExpr::Constant(ac - bc);
            }
            // Double negation: Sub(0, Sub(0, x)) -> x.
            if is_const_zero(&a) {
                if let TLExpr::Sub(inner_a, inner_x) = &b {
                    if is_const_zero(inner_a) {
                        return (**inner_x).clone();
                    }
                }
            }
            // x - 0 -> x. `0 - x` stays structural (no unary Neg variant).
            if is_const_zero(&b) {
                return a;
            }
            TLExpr::sub(a, b)
        }
        TLExpr::Mul(a, b) => {
            let a = simplify_one_pass(a);
            let b = simplify_one_pass(b);
            if let (TLExpr::Constant(ac), TLExpr::Constant(bc)) = (&a, &b) {
                return TLExpr::Constant(ac * bc);
            }
            if is_const_zero(&a) || is_const_zero(&b) {
                return TLExpr::Constant(0.0);
            }
            if is_const_one(&a) {
                return b;
            }
            if is_const_one(&b) {
                return a;
            }
            TLExpr::mul(a, b)
        }
        TLExpr::Div(a, b) => {
            let a = simplify_one_pass(a);
            let b = simplify_one_pass(b);
            if let (TLExpr::Constant(ac), TLExpr::Constant(bc)) = (&a, &b) {
                if bc.abs() > CANONICAL_EPS {
                    let v = ac / bc;
                    if v.is_finite() {
                        return TLExpr::Constant(v);
                    }
                }
            }
            if is_const_one(&b) {
                return a;
            }
            TLExpr::div(a, b)
        }
        TLExpr::Pow(a, b) => {
            let a = simplify_one_pass(a);
            let b = simplify_one_pass(b);
            // x^0 -> 1 (note: 0^0 is left as 1 by convention here; matches
            // `f64::powf` which returns 1.0 for that case).
            if is_const_zero(&b) {
                return TLExpr::Constant(1.0);
            }
            if is_const_one(&b) {
                return a;
            }
            if let (TLExpr::Constant(ac), TLExpr::Constant(bc)) = (&a, &b) {
                let v = ac.powf(*bc);
                if v.is_finite() {
                    return TLExpr::Constant(v);
                }
            }
            TLExpr::pow(a, b)
        }
        TLExpr::Exp(a) => {
            let a = simplify_one_pass(a);
            // exp(log(x)) -> x.
            if let TLExpr::Log(inner) = &a {
                return (**inner).clone();
            }
            if let TLExpr::Constant(c) = &a {
                let v = c.exp();
                if v.is_finite() {
                    return TLExpr::Constant(v);
                }
            }
            TLExpr::exp(a)
        }
        TLExpr::Log(a) => {
            let a = simplify_one_pass(a);
            // log(exp(x)) -> x.
            if let TLExpr::Exp(inner) = &a {
                return (**inner).clone();
            }
            if let TLExpr::Constant(c) = &a {
                if *c > 0.0 {
                    let v = c.ln();
                    if v.is_finite() {
                        return TLExpr::Constant(v);
                    }
                }
            }
            TLExpr::log(a)
        }
        TLExpr::Sin(a) => {
            let a = simplify_one_pass(a);
            if let TLExpr::Constant(c) = &a {
                let v = c.sin();
                if v.is_finite() {
                    return TLExpr::Constant(v);
                }
            }
            TLExpr::sin(a)
        }
        TLExpr::Cos(a) => {
            let a = simplify_one_pass(a);
            if let TLExpr::Constant(c) = &a {
                let v = c.cos();
                if v.is_finite() {
                    return TLExpr::Constant(v);
                }
            }
            TLExpr::cos(a)
        }
        // Leaf (`Constant`, `Pred`) and anything outside the arithmetic /
        // transcendental subset is left untouched. This matches the
        // documented behaviour — non-arithmetic variants (logical
        // connectives, quantifiers, temporal operators, ...) are not
        // rewritten by [`canonical_simplify`].
        _ => expr.clone(),
    }
}

/// Simplify a `TLExpr` by applying OxiEML's canonical algebraic identities
/// until fixpoint.
///
/// This fills the role of [`canonical_rewrite_rules`] for use cases where
/// direct in-place simplification is preferred over declarative rule lists.
/// Currently recognizes:
///
/// - `exp(log(x)) → x`
/// - `log(exp(x)) → x`
/// - `x + 0 → x`, `0 + x → x`
/// - `x - 0 → x`, `0 - x → -x` (kept as `Sub(0, x)`)
/// - `x * 1 → x`, `1 * x → x`
/// - `x * 0 → 0`, `0 * x → 0`
/// - `x / 1 → x`
/// - `x ^ 0 → 1`, `x ^ 1 → x`
/// - Constant-constant folding for `+`, `-`, `*`, `/`, `exp`, `log`, `sin`,
///   `cos`, `pow` (folds only happen when the result is finite, and
///   `log`/`exp` additionally guard the input domain).
/// - Double-negation via `Sub(0, Sub(0, x)) → x`.
///
/// Only the arithmetic and transcendental subset produced by [`to_tlexpr`]
/// is rewritten. Predicates, logical connectives, quantifiers, temporal
/// operators, and all other `TLExpr` variants are returned unchanged, even
/// if they contain simplifiable sub-expressions — this preserves the
/// invariant that [`canonical_simplify`] is side-effect-free with respect
/// to non-EML structure.
///
/// Iterates to a fixpoint; returns the simplified expression.
pub fn canonical_simplify(expr: &TLExpr) -> TLExpr {
    let mut current = expr.clone();
    loop {
        let next = simplify_one_pass(&current);
        if current == next {
            return next;
        }
        current = next;
    }
}

/// Convert a slice of [`crate::symreg::DiscoveredFormula`] values into
/// [`TLExpr::WeightedRule`] expressions.
///
/// Each formula is paired with the corresponding weight: `weights[i]` is passed
/// verbatim to [`crate::symreg::DiscoveredFormula::to_tl_weighted_rule`]. The weight
/// semantics (likelihood, scalar loss multiplier, etc.) are defined by the consuming
/// training system; OxiEML passes them through unchanged.
///
/// # Errors
///
/// Returns [`crate::EmlError::DimensionMismatch`] if `formulas.len() != weights.len()`.
///
/// # Example — producing soft TL training priors
///
/// ```ignore
/// // Discover formulas, rank by MSE, assign inverse-MSE weights
/// let formulas = engine.discover(&inputs, &targets, 1)?;
/// let weights: Vec<f64> = formulas.iter().map(|f| 1.0 / (1.0 + f.mse)).collect();
/// let constraints = tensorlogic::formulas_to_tl_weighted_rules(&formulas, &weights)?;
/// // Pass `constraints` into tensorlogic-train as soft priors.
/// ```
pub fn formulas_to_tl_weighted_rules(
    formulas: &[crate::symreg::DiscoveredFormula],
    weights: &[f64],
) -> Result<Vec<TLExpr>, crate::EmlError> {
    if formulas.len() != weights.len() {
        return Err(crate::EmlError::DimensionMismatch(
            formulas.len(),
            weights.len(),
        ));
    }
    Ok(formulas
        .iter()
        .zip(weights.iter().copied())
        .map(|(f, w)| f.to_tl_weighted_rule(w))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_roundtrip() {
        let op = LoweredOp::Const(std::f64::consts::PI);
        let tl = to_tlexpr(&op);
        assert!(matches!(tl, TLExpr::Constant(v) if (v - std::f64::consts::PI).abs() < 1e-15));
        let back = from_tlexpr(&tl).expect("const round-trip");
        assert_eq!(op, back);
    }

    #[test]
    fn var_pred_shape() {
        let op = LoweredOp::Var(7);
        let tl = to_tlexpr(&op);
        let back = from_tlexpr(&tl).expect("var round-trip");
        assert_eq!(op, back);
        match tl {
            TLExpr::Pred { name, args } => {
                assert_eq!(name, "x7");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Term::Var(n) if n == "x7"));
            }
            other => panic!("expected Pred, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_variant_rejected() {
        let tl = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("a")]),
            TLExpr::pred("Q", vec![Term::var("b")]),
        );
        let err = from_tlexpr(&tl).expect_err("And must be rejected");
        match err {
            EmlError::UnsupportedTlExpr(desc) => {
                assert!(desc.contains("And"), "got {desc}");
            }
            other => panic!("unexpected error variant {other:?}"),
        }
    }

    #[test]
    fn stray_pred_name_rejected() {
        let tl = TLExpr::pred("foo", vec![Term::var("a")]);
        let err = from_tlexpr(&tl).expect_err("non-x predicate must be rejected");
        assert!(matches!(err, EmlError::UnsupportedTlExpr(_)));
    }

    #[test]
    fn canonical_rewrite_rules_has_ten_rules() {
        let rules = canonical_rewrite_rules();
        assert_eq!(rules.len(), 10);
    }

    #[test]
    fn canonical_simplify_exp_log_eliminates() {
        // exp(log(x)) -> x, with x represented as a variable predicate so
        // the simplifier cannot accidentally fold through `Constant`.
        let x = var_pred(0);
        let expr = TLExpr::exp(TLExpr::log(x.clone()));
        assert_eq!(canonical_simplify(&expr), x);

        // log(exp(x)) -> x, the dual direction.
        let expr2 = TLExpr::log(TLExpr::exp(x.clone()));
        assert_eq!(canonical_simplify(&expr2), x);
    }

    #[test]
    fn canonical_simplify_folds_const_arithmetic() {
        // (2 + 3) * 4 - 1 -> 19, entirely by constant folding.
        let expr = TLExpr::sub(
            TLExpr::mul(
                TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
                TLExpr::Constant(4.0),
            ),
            TLExpr::Constant(1.0),
        );
        let simplified = canonical_simplify(&expr);
        match simplified {
            TLExpr::Constant(v) => assert!(
                (v - 19.0).abs() < 1e-12,
                "expected Constant(19.0), got Constant({v})"
            ),
            other => panic!("expected folded Constant, got {other:?}"),
        }
    }

    #[test]
    fn canonical_simplify_removes_zero_add() {
        // x + 0 -> x, 0 + x -> x.
        let x = var_pred(0);
        let a = TLExpr::add(x.clone(), TLExpr::Constant(0.0));
        assert_eq!(canonical_simplify(&a), x);

        let b = TLExpr::add(TLExpr::Constant(0.0), x.clone());
        assert_eq!(canonical_simplify(&b), x);

        // x - 0 -> x.
        let c = TLExpr::sub(x.clone(), TLExpr::Constant(0.0));
        assert_eq!(canonical_simplify(&c), x);
    }

    #[test]
    fn canonical_simplify_removes_one_mul() {
        // 1 * x -> x, x * 1 -> x.
        let x = var_pred(0);
        let a = TLExpr::mul(TLExpr::Constant(1.0), x.clone());
        assert_eq!(canonical_simplify(&a), x);

        let b = TLExpr::mul(x.clone(), TLExpr::Constant(1.0));
        assert_eq!(canonical_simplify(&b), x);

        // x * 0 -> 0, 0 * x -> 0.
        let z = TLExpr::mul(x.clone(), TLExpr::Constant(0.0));
        assert_eq!(canonical_simplify(&z), TLExpr::Constant(0.0));

        let z2 = TLExpr::mul(TLExpr::Constant(0.0), x.clone());
        assert_eq!(canonical_simplify(&z2), TLExpr::Constant(0.0));

        // x / 1 -> x.
        let d = TLExpr::div(x.clone(), TLExpr::Constant(1.0));
        assert_eq!(canonical_simplify(&d), x);

        // x ^ 0 -> 1, x ^ 1 -> x.
        let p0 = TLExpr::pow(x.clone(), TLExpr::Constant(0.0));
        assert_eq!(canonical_simplify(&p0), TLExpr::Constant(1.0));

        let p1 = TLExpr::pow(x.clone(), TLExpr::Constant(1.0));
        assert_eq!(canonical_simplify(&p1), x);
    }

    #[test]
    fn canonical_simplify_double_neg() {
        // Sub(0, Sub(0, x)) -> x.
        let x = var_pred(0);
        let expr = TLExpr::sub(
            TLExpr::Constant(0.0),
            TLExpr::sub(TLExpr::Constant(0.0), x.clone()),
        );
        assert_eq!(canonical_simplify(&expr), x);

        // Verify that single negation `0 - x` is *not* unnecessarily
        // changed (it is already in canonical form per the documented
        // contract).
        let single = TLExpr::sub(TLExpr::Constant(0.0), x.clone());
        assert_eq!(canonical_simplify(&single), single);
    }

    #[test]
    fn canonical_simplify_leaves_complex_untouched() {
        // An `And` of two arithmetic sub-expressions: each sub-expression
        // *could* be simplified, but the simplifier is intentionally
        // limited to the arithmetic/transcendental subset so predicates
        // and logical connectives are passed through unchanged.
        let logic = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("a")]),
            TLExpr::pred("Q", vec![Term::var("b")]),
        );
        assert_eq!(canonical_simplify(&logic), logic);

        // A pure predicate is also untouched.
        let p = TLExpr::pred("R", vec![Term::var("z")]);
        assert_eq!(canonical_simplify(&p), p);

        // A bare variable predicate is also untouched (it's a leaf for the
        // simplifier).
        let x = var_pred(3);
        assert_eq!(canonical_simplify(&x), x);
    }
}
