//! Integration tests for the dimensional-analysis / unit-aware regression feature.
//!
//! Covers:
//! - [`oxieml::units::Units`] algebra (mul, div, pow)
//! - [`oxieml::LoweredOp::check_units`] (all rule cases)
//! - [`oxieml::SymRegConfig::unit_filter`] integration with `discover_exhaustive`

use oxieml::units::{UnitError, Units};
use oxieml::{LoweredOp, SymRegConfig, SymRegEngine};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Units algebra
// ---------------------------------------------------------------------------

#[test]
fn dimensionless_const_passes() {
    let result = LoweredOp::Const(2.0).check_units(&[]);
    assert_eq!(result, Ok(Units::DIMENSIONLESS));
}

#[test]
fn var_returns_its_units() {
    let result = LoweredOp::Var(0).check_units(&[Units::METER]);
    assert_eq!(result, Ok(Units::METER));
}

#[test]
fn add_compatible_units_ok() {
    // x0 [m] + x1 [m] → [m]
    let expr = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::METER, Units::METER]);
    assert_eq!(result, Ok(Units::METER));
}

#[test]
fn add_incompatible_units_err() {
    // x0 [m] + x1 [s] → error
    let expr = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::METER, Units::SECOND]);
    assert!(matches!(
        result,
        Err(UnitError::IncompatibleAddSub { left, right })
            if left == Units::METER && right == Units::SECOND
    ));
}

#[test]
fn mul_combines_units() {
    // x0 [m] * x1 [s] → [m·s]
    let expr = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::METER, Units::SECOND]);
    let expected = Units::new([1, 0, 1, 0, 0, 0, 0]);
    assert_eq!(result, Ok(expected));
}

#[test]
fn div_subtracts_units() {
    // x0 [m] / x1 [s] → [m/s]  = [1,0,-1,0,0,0,0]
    let expr = LoweredOp::Div(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::METER, Units::SECOND]);
    let expected = Units::new([1, 0, -1, 0, 0, 0, 0]);
    assert_eq!(result, Ok(expected));
}

#[test]
fn pow_integer_scales_units() {
    // x0 [m] ^ 2 → [m²]  = [2,0,0,0,0,0,0]
    let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)));
    let result = expr.check_units(&[Units::METER]);
    let expected = Units::new([2, 0, 0, 0, 0, 0, 0]);
    assert_eq!(result, Ok(expected));
}

#[test]
fn pow_rational_exponent_with_units_ok() {
    // x0 [m] ^ 0.5 → m^(1/2)  (rationalized from 0.5)
    let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(0.5)));
    let result = expr.check_units(&[Units::METER]);
    assert!(
        result.is_ok(),
        "m^0.5 should succeed (rational 1/2), got {:?}",
        result
    );
    let expected = Units::METER.sqrt();
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn exp_requires_dimensionless() {
    // exp(x0 [m]) → error: argument must be dimensionless
    let expr = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    let result = expr.check_units(&[Units::METER]);
    assert!(
        matches!(
            result,
            Err(UnitError::NonDimensionlessArgument { op: "exp", .. })
        ),
        "expected NonDimensionlessArgument(exp), got {result:?}"
    );
}

#[test]
fn exp_dimensionless_ok() {
    // exp(x0 [dimensionless]) → dimensionless
    let expr = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    let result = expr.check_units(&[Units::DIMENSIONLESS]);
    assert_eq!(result, Ok(Units::DIMENSIONLESS));
}

#[test]
fn newton_formula_checks() {
    // F = m * (v/t): Mul(Var(0), Div(Var(1), Var(2)))
    // var_units = [kg, m/s, s]  →  kg * (m/s / s) = kg * m/s² = N
    let expr = LoweredOp::Mul(
        Arc::new(LoweredOp::Var(0)),
        Arc::new(LoweredOp::Div(
            Arc::new(LoweredOp::Var(1)),
            Arc::new(LoweredOp::Var(2)),
        )),
    );
    let velocity = Units::METER.div(&Units::SECOND); // m/s
    let var_units = [Units::KILOGRAM, velocity, Units::SECOND];
    let result = expr.check_units(&var_units);
    assert_eq!(
        result,
        Ok(Units::NEWTON),
        "F = m*(v/t) should have units of Newton"
    );
}

#[test]
fn unit_filter_reduces_symreg_search() {
    // Scenario: y = x0 (identity mapping), var 0 has units of METER.
    // With unit_filter = Some(([METER], METER)), topologies that output METER
    // are retained; those that output dimensionless, SECOND, etc. are dropped.
    //
    // This test verifies:
    //   (a) unit_filter doesn't crash and returns at least one formula;
    //   (b) the filtered run returns fewer or equal formulas than the unfiltered run;
    //   (c) every returned formula is dimensionally consistent (output units == METER).
    let inputs: Vec<Vec<f64>> = (1..=10).map(|i| vec![i as f64 * 0.5]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0]).collect(); // y = x0

    let config_filtered = SymRegConfig {
        max_depth: 2,
        max_iter: 200,
        num_restarts: 1,
        unit_filter: Some((vec![Units::METER], Units::METER)),
        ..SymRegConfig::default()
    };

    let config_unfiltered = SymRegConfig {
        max_depth: 2,
        max_iter: 200,
        num_restarts: 1,
        ..SymRegConfig::default()
    };

    let engine_filtered = SymRegEngine::new(config_filtered);
    let engine_unfiltered = SymRegEngine::new(config_unfiltered);

    let filtered = engine_filtered
        .discover(&inputs, &targets, 1)
        .expect("unit-filtered discover should succeed");
    let unfiltered = engine_unfiltered
        .discover(&inputs, &targets, 1)
        .expect("unfiltered discover should succeed");

    // (a) unit filter must still yield at least one formula (the identity Var(0) is METER)
    assert!(
        !filtered.is_empty(),
        "unit filter should still yield formulas: the identity topology Var(0) has units METER"
    );

    // (b) filtered run should produce no more formulas than the unfiltered run
    assert!(
        filtered.len() <= unfiltered.len(),
        "unit filter should reduce or equal the formula count: filtered={}, unfiltered={}",
        filtered.len(),
        unfiltered.len()
    );

    // (c) every returned formula must have output units == METER
    for formula in &filtered {
        let lowered = formula.eml_tree.lower().simplify();
        let units = lowered.check_units(&[Units::METER]);
        assert_eq!(
            units,
            Ok(Units::METER),
            "formula '{}' should have METER output units, got {units:?}",
            formula.pretty
        );
    }
}

#[test]
fn units_to_string_meter_per_second() {
    let v = Units::METER.div(&Units::SECOND);
    let s = v.to_string();
    assert!(s.contains('m'), "expected 'm' in unit string '{s}'");
    assert!(s.contains('s'), "expected 's' in unit string '{s}'");
}

#[test]
fn var_index_out_of_range_err() {
    // Var(5) with only 1 unit supplied → VarIndexOutOfRange
    let result = LoweredOp::Var(5).check_units(&[Units::METER]);
    assert!(
        matches!(
            result,
            Err(UnitError::VarIndexOutOfRange {
                index: 5,
                n_vars: 1
            })
        ),
        "expected VarIndexOutOfRange, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Additional checks: all transcendentals, negation, Sub, NamedConst
// ---------------------------------------------------------------------------

#[test]
fn sub_compatible_units_ok() {
    let expr = LoweredOp::Sub(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::KILOGRAM, Units::KILOGRAM]);
    assert_eq!(result, Ok(Units::KILOGRAM));
}

#[test]
fn neg_preserves_units() {
    let expr = LoweredOp::Neg(Arc::new(LoweredOp::Var(0)));
    let result = expr.check_units(&[Units::METER]);
    assert_eq!(result, Ok(Units::METER));
}

#[test]
fn ln_requires_dimensionless() {
    let expr = LoweredOp::Ln(Arc::new(LoweredOp::Var(0)));
    let result = expr.check_units(&[Units::SECOND]);
    assert!(
        matches!(
            result,
            Err(UnitError::NonDimensionlessArgument { op: "ln", .. })
        ),
        "expected NonDimensionlessArgument(ln), got {result:?}"
    );
}

#[test]
fn sin_requires_dimensionless() {
    let expr = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let result = expr.check_units(&[Units::AMPERE]);
    assert!(
        matches!(
            result,
            Err(UnitError::NonDimensionlessArgument { op: "sin", .. })
        ),
        "expected NonDimensionlessArgument(sin), got {result:?}"
    );
}

#[test]
fn pow_with_symbolic_exponent_and_units_err() {
    // x0 [m] ^ x1 [dimensionless but non-const] → NonRationalPower
    let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::METER, Units::DIMENSIONLESS]);
    assert!(
        matches!(result, Err(UnitError::NonRationalPower { .. })),
        "expected NonRationalPower for symbolic exponent with dimensioned base, got {result:?}"
    );
}

#[test]
fn pow_dimensionless_base_any_exponent_ok() {
    // dimensionless ^ x1 [dimensionless] → DIMENSIONLESS
    let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    let result = expr.check_units(&[Units::DIMENSIONLESS, Units::DIMENSIONLESS]);
    assert_eq!(result, Ok(Units::DIMENSIONLESS));
}

#[test]
fn named_const_is_dimensionless() {
    use oxieml::NamedConst;
    let expr = LoweredOp::NamedConst(NamedConst::Pi);
    let result = expr.check_units(&[]);
    assert_eq!(result, Ok(Units::DIMENSIONLESS));
}

#[test]
fn unit_filter_none_is_default() {
    let config = SymRegConfig::default();
    assert!(
        config.unit_filter.is_none(),
        "unit_filter must default to None"
    );
}

#[test]
fn test_rexp_from_int_preserves_integer_behavior() {
    use oxieml::units::Rexp;
    let r = Rexp::from_int(3);
    assert_eq!(r.num, 3);
    assert_eq!(r.den, 1);
    assert!(r.is_integer());
    assert_eq!(r.to_i8(), Some(3));
}

#[test]
fn test_rexp_sqrt_of_area_is_length() {
    // sqrt(m²) = m (1/2 * 2 = 1)
    let m2 = Units::METER.pow_int(2).expect("no overflow");
    let m = m2.sqrt();
    assert_eq!(m.0[0].to_i8(), Some(1), "sqrt(m²) should give m^1");
}

#[test]
fn test_rexp_sqrt_of_meter_is_half_power() {
    use oxieml::units::Rexp;
    let m_half = Units::METER.sqrt();
    assert_eq!(
        m_half.0[0],
        Rexp::from_ratio(1, 2),
        "sqrt(m) should be m^(1/2)"
    );
}

#[test]
fn test_rexp_mul_half_plus_half_is_one() {
    // m^(1/2) * m^(1/2) = m
    let m_half = Units::METER.sqrt();
    let m = m_half.mul(&m_half);
    assert_eq!(m.0[0].to_i8(), Some(1), "m^(1/2) * m^(1/2) = m");
}

#[test]
fn test_legacy_integer_ops_preserved() {
    let m = Units::METER;
    let m2 = m.pow_int(2).expect("no overflow");
    assert_eq!(m2.0[0].to_i8(), Some(2));
    let m_back = m2.div(&m);
    assert_eq!(m_back.0[0].to_i8(), Some(1));
}

#[test]
fn test_units_display_rational() {
    let m_half = Units::METER.sqrt();
    let s = m_half.to_string();
    assert!(
        s.contains("(1/2)"),
        "rational display should show (1/2), got: {s}"
    );
}

#[test]
fn test_units_new_from_int_array() {
    let v = Units::new([1, 0, -1, 0, 0, 0, 0]);
    assert_eq!(v.try_into_int_exps(), Some([1i8, 0, -1, 0, 0, 0, 0]));
}

#[test]
fn test_with_units_convenience() {
    let config = SymRegConfig::default().with_units(vec![Units::METER], Units::METER);
    assert!(config.unit_filter.is_some());
    let (var_units, target) = config.unit_filter.unwrap();
    assert_eq!(var_units[0], Units::METER);
    assert_eq!(target, Units::METER);
}
