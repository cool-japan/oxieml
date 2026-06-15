use oxieml::SolveResult;
use oxieml::lower::LoweredOp;
use std::sync::Arc;

fn is_closed(r: &SolveResult) -> bool {
    matches!(r, SolveResult::Closed(_))
}

fn closed_eval(r: SolveResult, vars: &[f64]) -> f64 {
    match r {
        SolveResult::Closed(op) => op.eval(vars),
        SolveResult::Residual(_) => f64::NAN,
    }
}

#[test]
fn solve_linear() {
    // 2*x + 3 == 7 → x == 2
    let expr = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Const(2.0)),
            Arc::new(LoweredOp::Var(0)),
        )),
        Arc::new(LoweredOp::Const(3.0)),
    );
    let rhs = LoweredOp::Const(7.0);
    let result = expr.solve_for(0, &rhs);
    assert!(is_closed(&result));
    let val = closed_eval(result, &[]);
    assert!((val - 2.0).abs() < 1e-10, "expected 2.0, got {val}");
}

#[test]
fn solve_exp() {
    // exp(x) == e → x == 1
    let expr = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    let rhs = LoweredOp::Const(std::f64::consts::E);
    let result = expr.solve_for(0, &rhs);
    assert!(is_closed(&result));
    let val = closed_eval(result, &[]);
    assert!((val - 1.0).abs() < 1e-10, "expected 1.0, got {val}");
}

#[test]
fn solve_nested() {
    // 3 * exp(x + 1) == 3e → exp(x+1) == e → x+1 == 1 → x == 0
    let expr = LoweredOp::Mul(
        Arc::new(LoweredOp::Const(3.0)),
        Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Add(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Const(1.0)),
        )))),
    );
    let rhs = LoweredOp::Const(3.0 * std::f64::consts::E);
    let result = expr.solve_for(0, &rhs);
    assert!(is_closed(&result));
    let val = closed_eval(result, &[]);
    assert!((val - 0.0).abs() < 1e-9, "expected 0.0, got {val}");
}

#[test]
fn solve_sin() {
    // sin(x) == 0.5 → x == arcsin(0.5) == π/6
    let expr = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
    let rhs = LoweredOp::Const(0.5);
    let result = expr.solve_for(0, &rhs);
    assert!(is_closed(&result));
    let val = closed_eval(result, &[]);
    let expected = (0.5_f64).asin();
    assert!(
        (val - expected).abs() < 1e-10,
        "expected {expected}, got {val}"
    );
}

#[test]
fn solve_power() {
    // x^2 == 9 → x == 9^(1/2) == 3
    let expr = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)));
    let rhs = LoweredOp::Const(9.0);
    let result = expr.solve_for(0, &rhs);
    assert!(is_closed(&result));
    let val = closed_eval(result, &[]);
    assert!((val - 3.0).abs() < 1e-10, "expected 3.0, got {val}");
}

#[test]
fn solve_unsolvable_returns_residual() {
    // x + sin(x) == 1: not algebraically solvable
    let expr = LoweredOp::Add(
        Arc::new(LoweredOp::Var(0)),
        Arc::new(LoweredOp::Sin(Arc::new(LoweredOp::Var(0)))),
    );
    let rhs = LoweredOp::Const(1.0);
    let result = expr.solve_for(0, &rhs);
    assert!(matches!(result, SolveResult::Residual(_)));
}

#[test]
fn contains_var_basic() {
    let expr = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(1.0)));
    assert!(expr.contains_var(0));
    assert!(!expr.contains_var(1));
}

#[test]
fn solve_division() {
    // x / 4 == 3 → x == 12
    let expr = LoweredOp::Div(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(4.0)));
    let result = expr.solve_for(0, &LoweredOp::Const(3.0));
    assert!(is_closed(&result));
    let val = closed_eval(result, &[]);
    assert!((val - 12.0).abs() < 1e-10, "expected 12.0, got {val}");
}
