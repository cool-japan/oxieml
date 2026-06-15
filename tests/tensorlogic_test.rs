//! Integration tests for the `tensorlogic` feature.
//!
//! Build and round-trip `LoweredOp` expressions through
//! `oxieml::tensorlogic::{to_tlexpr, from_tlexpr}` and compare numerical
//! evaluation at random points so both the syntactic and semantic contracts
//! of the bridge are exercised.

#![cfg(feature = "tensorlogic")]

use oxieml::lower::LoweredOp;
use oxieml::tensorlogic::{canonical_rewrite_rules, canonical_simplify, from_tlexpr, to_tlexpr};
use rand::{Rng, RngExt};
use std::sync::Arc;
use tensorlogic_ir::{TLExpr, Term};

/// Maximum allowed absolute error between the original and the
/// round-tripped `LoweredOp` when evaluated at the same sample points.
const EVAL_TOLERANCE: f64 = 1e-12;

/// Sample count for the random evaluation-equivalence tests.
const NUM_SAMPLES: usize = 10;

/// Draw a sample point in a numerically-safe range that keeps `ln` and
/// `exp` well-defined (positive x, bounded magnitude).
fn safe_sample(rng: &mut impl Rng) -> (f64, f64) {
    // x in [0.2, 3.0] keeps exp(x) modest and ln(x) finite.
    // y in [-2.0, 2.0] covers negative values for subtraction tests.
    let x = rng.random_range(0.2..3.0);
    let y = rng.random_range(-2.0..2.0);
    (x, y)
}

/// Build `LoweredOp` representing `exp(x) - y`.
fn build_exp_x_minus_y() -> LoweredOp {
    LoweredOp::Sub(
        Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Var(0)))),
        Arc::new(LoweredOp::Var(1)),
    )
}

/// Build `LoweredOp` representing `(x + 1) * (y - 2)`.
fn build_add_mul() -> LoweredOp {
    LoweredOp::Mul(
        Arc::new(LoweredOp::Add(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Const(1.0)),
        )),
        Arc::new(LoweredOp::Sub(
            Arc::new(LoweredOp::Var(1)),
            Arc::new(LoweredOp::Const(2.0)),
        )),
    )
}

#[test]
fn roundtrip_exp_x_minus_y() {
    let op = build_exp_x_minus_y();
    let tl = to_tlexpr(&op);
    let restored = from_tlexpr(&tl).expect("round-trip must succeed for supported subset");

    let mut rng = rand::rng();
    for _ in 0..NUM_SAMPLES {
        let (x, y) = safe_sample(&mut rng);
        let vars = [x, y];
        let before = op.eval(&vars);
        let after = restored.eval(&vars);
        assert!(
            (before - after).abs() < EVAL_TOLERANCE,
            "roundtrip mismatch at (x={x}, y={y}): before={before} after={after}"
        );
    }
}

#[test]
fn roundtrip_add_mul() {
    let op = build_add_mul();
    let tl = to_tlexpr(&op);
    let restored = from_tlexpr(&tl).expect("round-trip must succeed for supported subset");

    // Structural equality is expected here because there are no `Neg`
    // nodes in this tree — every LoweredOp variant maps 1:1 to a TLExpr
    // variant and back.
    assert_eq!(op, restored);

    let mut rng = rand::rng();
    for _ in 0..NUM_SAMPLES {
        let (x, y) = safe_sample(&mut rng);
        let vars = [x, y];
        let before = op.eval(&vars);
        let after = restored.eval(&vars);
        assert!(
            (before - after).abs() < EVAL_TOLERANCE,
            "roundtrip mismatch at (x={x}, y={y}): before={before} after={after}"
        );
    }
}

#[test]
fn tlexpr_has_correct_shape() {
    // LoweredOp::Add(Const(3.0), Var(0)) must lower to
    //   TLExpr::Add(TLExpr::Constant(3.0), TLExpr::Pred { name: "x0", ... })
    let op = LoweredOp::Add(Arc::new(LoweredOp::Const(3.0)), Arc::new(LoweredOp::Var(0)));
    let tl = to_tlexpr(&op);

    let (lhs, rhs) = match &tl {
        TLExpr::Add(a, b) => (a.as_ref(), b.as_ref()),
        other => panic!("expected TLExpr::Add, got {other:?}"),
    };

    match lhs {
        TLExpr::Constant(v) => assert!(
            (v - 3.0).abs() < 1e-15,
            "expected Constant(3.0), got Constant({v})"
        ),
        other => panic!("expected Constant on LHS, got {other:?}"),
    }

    match rhs {
        TLExpr::Pred { name, args } => {
            assert_eq!(name, "x0", "predicate name for Var(0) must be `x0`");
            assert_eq!(args.len(), 1, "variable pred must have exactly one arg");
            match &args[0] {
                Term::Var(arg_name) => assert_eq!(arg_name, "x0"),
                other => panic!("expected Term::Var, got {other:?}"),
            }
        }
        other => panic!("expected Pred on RHS, got {other:?}"),
    }
}

#[test]
fn neg_roundtrips_numerically_via_zero_sub() {
    // `LoweredOp::Neg(x)` encodes as `TLExpr::Sub(0, x)`, so the
    // structural round-trip gives `LoweredOp::Sub(Const(0.0), x)` — which
    // is numerically identical but not structurally equal. We verify the
    // numeric contract here.
    let op = LoweredOp::Neg(Arc::new(LoweredOp::Var(0)));
    let tl = to_tlexpr(&op);
    let restored = from_tlexpr(&tl).expect("neg encodes via Sub(0, _)");

    let mut rng = rand::rng();
    for _ in 0..NUM_SAMPLES {
        let (x, _) = safe_sample(&mut rng);
        let vars = [x];
        let before = op.eval(&vars);
        let after = restored.eval(&vars);
        assert!(
            (before - after).abs() < EVAL_TOLERANCE,
            "neg numeric mismatch at x={x}: before={before} after={after}"
        );
    }
}

#[test]
fn canonical_rewrite_rules_returns_ten_rules() {
    let rules = canonical_rewrite_rules();
    assert_eq!(rules.len(), 10, "expected 10 canonical rewrite rules");
}

#[test]
fn canonical_simplify_on_roundtrip_expression() {
    // `LoweredOp::Add(Var(0), Const(0.0))` lowers to
    //   `TLExpr::Add(x0_pred, Constant(0.0))`
    // which should simplify to just the predicate for Var(0).
    let op = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(0.0)));
    let tl = to_tlexpr(&op);
    let simplified = canonical_simplify(&tl);

    // Build the expected TLExpr by hand — the predicate for Var(0).
    let expected = TLExpr::pred("x0", vec![Term::var("x0")]);
    assert_eq!(simplified, expected);

    // And the simplified TLExpr should round-trip back to just `Var(0)`.
    let restored = from_tlexpr(&simplified).expect("simplified expr round-trips");
    assert_eq!(restored, LoweredOp::Var(0));

    // Verify numerical equivalence too.
    let mut rng = rand::rng();
    for _ in 0..NUM_SAMPLES {
        let (x, _) = safe_sample(&mut rng);
        let vars = [x];
        let before = op.eval(&vars);
        let after = restored.eval(&vars);
        assert!(
            (before - after).abs() < EVAL_TOLERANCE,
            "simplified mismatch at x={x}: before={before} after={after}"
        );
    }
}

#[test]
fn canonical_simplify_exp_log_through_bridge() {
    // exp(ln(x)) should collapse to just x after `canonical_simplify`,
    // exercising the full pipeline LoweredOp -> TLExpr -> simplify.
    let op = LoweredOp::Exp(Arc::new(LoweredOp::Ln(Arc::new(LoweredOp::Var(0)))));
    let tl = to_tlexpr(&op);
    let simplified = canonical_simplify(&tl);
    let restored = from_tlexpr(&simplified).expect("simplified exp(ln(x)) round-trips");
    assert_eq!(restored, LoweredOp::Var(0));
}

// ---------------------------------------------------------------------------
// Rewrite-rule pattern matching + template application tests
// ---------------------------------------------------------------------------

/// Helper: build a `TLExpr::Pred` for variable index `i`, matching the
/// convention used by `to_tlexpr`.
fn test_var(i: usize) -> TLExpr {
    let name = format!("x{i}");
    TLExpr::pred(name.clone(), vec![Term::var(name)])
}

#[test]
fn rewrite_rule_exp_log_inverse() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[0]; // exp(log(x)) → x
    assert_eq!(rule.name.as_deref(), Some("exp_log_inverse"));

    let x = test_var(0);
    let expr = TLExpr::exp(TLExpr::log(x.clone()));
    let bindings = rule
        .pattern
        .matches(&expr)
        .expect("pattern must match exp(log(x))");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_log_exp_inverse() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[1]; // log(exp(x)) → x
    assert_eq!(rule.name.as_deref(), Some("log_exp_inverse"));

    let x = test_var(1);
    let expr = TLExpr::log(TLExpr::exp(x.clone()));
    let bindings = rule
        .pattern
        .matches(&expr)
        .expect("pattern must match log(exp(x))");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_double_negation() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[2]; // neg(neg(x)) → x
    assert_eq!(rule.name.as_deref(), Some("double_negation"));

    // neg is encoded as Sub(0, _) in TLExpr
    let x = test_var(0);
    let expr = TLExpr::sub(
        TLExpr::Constant(0.0),
        TLExpr::sub(TLExpr::Constant(0.0), x.clone()),
    );
    let bindings = rule
        .pattern
        .matches(&expr)
        .expect("pattern must match neg(neg(x))");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_zero_add_left() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[3]; // 0 + x → x
    assert_eq!(rule.name.as_deref(), Some("zero_add_left"));

    let x = test_var(2);
    let expr = TLExpr::add(TLExpr::Constant(0.0), x.clone());
    let bindings = rule.pattern.matches(&expr).expect("pattern must match 0+x");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_zero_add_right() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[4]; // x + 0 → x
    assert_eq!(rule.name.as_deref(), Some("zero_add_right"));

    let x = test_var(3);
    let expr = TLExpr::add(x.clone(), TLExpr::Constant(0.0));
    let bindings = rule.pattern.matches(&expr).expect("pattern must match x+0");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_one_mul_right() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[5]; // x * 1 → x
    assert_eq!(rule.name.as_deref(), Some("one_mul_right"));

    let x = test_var(0);
    let expr = TLExpr::mul(x.clone(), TLExpr::Constant(1.0));
    let bindings = rule.pattern.matches(&expr).expect("pattern must match x*1");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_one_mul_left() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[6]; // 1 * x → x
    assert_eq!(rule.name.as_deref(), Some("one_mul_left"));

    let x = test_var(0);
    let expr = TLExpr::mul(TLExpr::Constant(1.0), x.clone());
    let bindings = rule.pattern.matches(&expr).expect("pattern must match 1*x");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_div_by_one() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[7]; // x / 1 → x
    assert_eq!(rule.name.as_deref(), Some("div_by_one"));

    let x = test_var(0);
    let expr = TLExpr::div(x.clone(), TLExpr::Constant(1.0));
    let bindings = rule.pattern.matches(&expr).expect("pattern must match x/1");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rule_pow_zero() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[8]; // x ^ 0 → 1
    assert_eq!(rule.name.as_deref(), Some("pow_zero"));

    let x = test_var(5);
    let expr = TLExpr::pow(x, TLExpr::Constant(0.0));
    let bindings = rule.pattern.matches(&expr).expect("pattern must match x^0");
    assert!(bindings.contains_key("_x"), "var '_x' must be bound");
    let result = (rule.template)(&bindings);
    assert_eq!(result, TLExpr::Constant(1.0));
}

#[test]
fn rewrite_rule_pow_one() {
    let rules = canonical_rewrite_rules();
    let rule = &rules[9]; // x ^ 1 → x
    assert_eq!(rule.name.as_deref(), Some("pow_one"));

    let x = test_var(0);
    let expr = TLExpr::pow(x.clone(), TLExpr::Constant(1.0));
    let bindings = rule.pattern.matches(&expr).expect("pattern must match x^1");
    assert_eq!(bindings.get("x"), Some(&x));
    let result = (rule.template)(&bindings);
    assert_eq!(result, x);
}

#[test]
fn rewrite_rules_do_not_match_wrong_expressions() {
    let rules = canonical_rewrite_rules();
    let x = test_var(0);

    // exp(log(x)) rule should NOT match log(exp(x))
    assert!(
        rules[0]
            .pattern
            .matches(&TLExpr::log(TLExpr::exp(x.clone())))
            .is_none()
    );

    // 0+x rule should NOT match 1+x
    assert!(
        rules[3]
            .pattern
            .matches(&TLExpr::add(TLExpr::Constant(1.0), x.clone()))
            .is_none()
    );

    // x*1 rule should NOT match x*2
    assert!(
        rules[5]
            .pattern
            .matches(&TLExpr::mul(x.clone(), TLExpr::Constant(2.0)))
            .is_none()
    );
}

// ── DiscoveredFormula TL adapter tests ──────────────────────────────────────

mod df_tl_adapter {
    use super::*;
    use oxieml::Canonical;
    use oxieml::symreg::DiscoveredFormula;
    use oxieml::tensorlogic::{self, canonical_simplify};

    fn make_formula() -> DiscoveredFormula {
        let tree = Canonical::nat(1);
        DiscoveredFormula {
            eml_tree: tree,
            mse: 0.0,
            complexity: 1,
            score: 0.0,
            pretty: "1".to_string(),
            params: vec![],
            cv_mse: None,
            aic: 0.0,
            bic: 0.0,
            param_intervals: None,
        }
    }

    fn make_three_formulas() -> Vec<DiscoveredFormula> {
        (0..3).map(|_| make_formula()).collect()
    }

    #[test]
    fn discoveredformula_to_tlexpr_matches_lowered_simplified_path() {
        let f = make_formula();
        let expected = tensorlogic::to_tlexpr(&f.eml_tree.lower().simplify());
        assert_eq!(f.to_tlexpr(), expected);
    }

    #[test]
    fn to_tl_weighted_rule_shape_carries_weight_verbatim() {
        let f = make_formula();
        let tl = f.to_tl_weighted_rule(0.42);
        match tl {
            TLExpr::WeightedRule { weight, .. } => {
                assert!((weight - 0.42).abs() < f64::EPSILON);
            }
            other => panic!("expected WeightedRule, got {other:?}"),
        }
    }

    #[test]
    fn to_tl_weighted_equation_shape_lhs_pred_eq_rhs_formula() {
        let f = make_formula();
        let tl = f.to_tl_weighted_equation("y", 1.0);
        match tl {
            TLExpr::WeightedRule { weight, rule } => {
                assert!((weight - 1.0).abs() < f64::EPSILON);
                match *rule {
                    TLExpr::Eq(lhs, rhs) => {
                        match *lhs {
                            TLExpr::Pred { name, ref args } => {
                                assert_eq!(name, "y");
                                assert_eq!(args.len(), 1);
                                assert_eq!(args[0], Term::var("y"));
                            }
                            other => panic!("expected Pred on lhs, got {other:?}"),
                        }
                        let f2 = make_formula();
                        assert_eq!(*rhs, f2.to_tlexpr());
                    }
                    other => panic!("expected Eq inside WeightedRule, got {other:?}"),
                }
            }
            other => panic!("expected WeightedRule, got {other:?}"),
        }
    }

    #[test]
    fn formulas_to_tl_weighted_rules_pairs_in_order() {
        let formulas = make_three_formulas();
        let weights = vec![0.1_f64, 0.5, 1.0];
        let result = tensorlogic::formulas_to_tl_weighted_rules(&formulas, &weights)
            .expect("should succeed");
        assert_eq!(result.len(), 3);
        for (i, rule) in result.iter().enumerate() {
            match rule {
                TLExpr::WeightedRule { weight, .. } => {
                    assert!(
                        (weight - weights[i]).abs() < f64::EPSILON,
                        "weight at index {i} mismatch: got {weight}, want {}",
                        weights[i]
                    );
                }
                other => panic!("expected WeightedRule at index {i}, got {other:?}"),
            }
        }
    }

    #[test]
    fn formulas_to_tl_weighted_rules_length_mismatch_returns_dimension_mismatch() {
        let formulas = make_three_formulas();
        let weights = vec![0.1_f64, 0.5]; // 2 != 3
        match tensorlogic::formulas_to_tl_weighted_rules(&formulas, &weights) {
            Err(oxieml::EmlError::DimensionMismatch(a, b)) => {
                assert_eq!(a, 3);
                assert_eq!(b, 2);
            }
            other => panic!("expected DimensionMismatch(3,2), got {other:?}"),
        }
    }

    #[test]
    fn tl_weighted_rule_passes_through_canonical_simplify_unchanged() {
        let f = make_formula();
        let rule = f.to_tl_weighted_rule(0.5);
        // canonical_simplify applies the 10 rewrite rules (exp/log inverses,
        // double-negation, identity elements). WeightedRule is not matched by
        // any of those rules, so the wrapper round-trips intact.
        let simplified = canonical_simplify(&rule);
        // We compare the WeightedRule weight field directly; inner formula may or
        // may not be simplified (canonicalise recurses but Const(1) is already minimal).
        match (&rule, &simplified) {
            (TLExpr::WeightedRule { weight: w1, .. }, TLExpr::WeightedRule { weight: w2, .. }) => {
                assert!((w1 - w2).abs() < f64::EPSILON);
            }
            _ => panic!("expected both to be WeightedRule"),
        }
    }
}
