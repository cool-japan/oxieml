//! Symbolic equation solving for `LoweredOp` expression trees.
//!
//! Given an equation `f(x) == rhs`, [`LoweredOp::solve_for`] attempts to derive
//! a closed-form expression for the target variable by recursively applying
//! algebraic inversion rules (inverse operations for +, -, *, /, exp, ln, trig, pow).
//!
//! When no closed-form inversion is possible (e.g. `x + sin(x) == 1`) the method
//! returns the residual `f - rhs` which can be used for numeric root-finding.

use crate::lower::LoweredOp;
use std::sync::Arc;

/// Result of symbolically solving `f(x) == rhs` for a target variable.
#[derive(Debug, Clone)]
pub enum SolveResult {
    /// Closed-form solution: the expression for the target variable.
    ///
    /// Evaluate this with the *other* variables' values to get the target
    /// variable's value. For example, solving `2*x + 3 == 7` yields
    /// `Closed(Const(2.0))`.
    Closed(LoweredOp),

    /// Could not find a closed-form solution.
    ///
    /// The residual `f(x) - rhs` can be passed to a numeric root-finder.
    /// For example, `x + sin(x) == 1` is not algebraically invertible and
    /// yields `Residual(x + sin(x) - 1)`.
    Residual(LoweredOp),
}

impl LoweredOp {
    /// Returns `true` if this expression tree contains `Var(var)` anywhere.
    ///
    /// Used by [`solve_for`](Self::solve_for) to determine which branch of a
    /// binary node contains the target variable before attempting inversion.
    pub fn contains_var(&self, var: usize) -> bool {
        match self {
            Self::Var(i) => *i == var,
            Self::Const(_) | Self::NamedConst(_) => false,
            // Unary nodes: check the child.
            Self::Neg(x)
            | Self::Sin(x)
            | Self::Cos(x)
            | Self::Tan(x)
            | Self::Sinh(x)
            | Self::Cosh(x)
            | Self::Tanh(x)
            | Self::Arcsin(x)
            | Self::Arccos(x)
            | Self::Arctan(x)
            | Self::Arcsinh(x)
            | Self::Arccosh(x)
            | Self::Arctanh(x)
            | Self::Exp(x)
            | Self::Ln(x)
            | Self::Erf(x)
            | Self::LGamma(x)
            | Self::Digamma(x)
            | Self::Trigamma(x)
            | Self::Ei(x)
            | Self::Si(x)
            | Self::Ci(x) => x.contains_var(var),
            // Binary nodes: check either branch.
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b) => a.contains_var(var) || b.contains_var(var),
        }
    }

    /// Symbolically solve `self == rhs` for the given target variable.
    ///
    /// Applies algebraic inversion rules recursively. Returns
    /// [`SolveResult::Closed`] when a closed-form solution can be derived, or
    /// [`SolveResult::Residual`]`(self − rhs)` when no algebraic inversion is
    /// known (e.g. both operands contain the target variable, or the function
    /// has no elementary inverse in a simple form).
    ///
    /// Only the *principal branch* is returned for multi-valued inverses (e.g.
    /// `arcsin` for `sin`).
    ///
    /// # Example
    ///
    /// ```
    /// use oxieml::lower::LoweredOp;
    /// use oxieml::SolveResult;
    ///
    /// // Solve 2*x + 3 == 7  →  x == 2
    /// let expr = LoweredOp::Add(
    ///     std::sync::Arc::new(LoweredOp::Mul(
    ///         std::sync::Arc::new(LoweredOp::Const(2.0)),
    ///         std::sync::Arc::new(LoweredOp::Var(0)),
    ///     )),
    ///     std::sync::Arc::new(LoweredOp::Const(3.0)),
    /// );
    /// let rhs = LoweredOp::Const(7.0);
    /// let result = expr.solve_for(0, &rhs);
    /// assert!(matches!(result, SolveResult::Closed(_)));
    /// if let SolveResult::Closed(solution) = result {
    ///     assert!((solution.eval(&[]) - 2.0).abs() < 1e-10);
    /// }
    /// ```
    pub fn solve_for(&self, target_var: usize, rhs: &LoweredOp) -> SolveResult {
        self.solve_inner(target_var, rhs.clone())
    }

    /// Recursive implementation of `solve_for`.
    ///
    /// Invariant at each call: we are solving `self == rhs` for `target_var`.
    /// The method rewrites the equation, moving nodes from `self` to `rhs`
    /// (applying the inverse operation) until `self` is exactly `Var(target_var)`.
    fn solve_inner(&self, target_var: usize, rhs: LoweredOp) -> SolveResult {
        match self {
            // ---- Base cases -----------------------------------------------

            // Isolated the variable: return the accumulated rhs as the solution.
            Self::Var(i) if *i == target_var => SolveResult::Closed(rhs.simplify()),

            // This subtree does not contain the target variable at all.
            // The equation degenerates to a constant identity; return residual.
            _ if !self.contains_var(target_var) => SolveResult::Residual(
                LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
            ),

            // ---- Unary operators ------------------------------------------

            // −x == rhs  →  x == −rhs
            Self::Neg(x) => x.solve_inner(target_var, LoweredOp::Neg(Arc::new(rhs)).simplify()),

            // exp(x) == rhs  →  x == ln(rhs)
            Self::Exp(x) => x.solve_inner(target_var, LoweredOp::Ln(Arc::new(rhs)).simplify()),

            // ln(x) == rhs  →  x == exp(rhs)
            Self::Ln(x) => x.solve_inner(target_var, LoweredOp::Exp(Arc::new(rhs)).simplify()),

            // sin(x) == rhs  →  x == arcsin(rhs)  (principal branch)
            Self::Sin(x) => x.solve_inner(target_var, LoweredOp::Arcsin(Arc::new(rhs)).simplify()),

            // cos(x) == rhs  →  x == arccos(rhs)  (principal branch)
            Self::Cos(x) => x.solve_inner(target_var, LoweredOp::Arccos(Arc::new(rhs)).simplify()),

            // tan(x) == rhs  →  x == arctan(rhs)
            Self::Tan(x) => x.solve_inner(target_var, LoweredOp::Arctan(Arc::new(rhs)).simplify()),

            // sinh(x) == rhs  →  x == arcsinh(rhs)
            Self::Sinh(x) => {
                x.solve_inner(target_var, LoweredOp::Arcsinh(Arc::new(rhs)).simplify())
            }

            // cosh(x) == rhs  →  x == arccosh(rhs)  (principal branch, rhs ≥ 1)
            Self::Cosh(x) => {
                x.solve_inner(target_var, LoweredOp::Arccosh(Arc::new(rhs)).simplify())
            }

            // tanh(x) == rhs  →  x == arctanh(rhs)
            Self::Tanh(x) => {
                x.solve_inner(target_var, LoweredOp::Arctanh(Arc::new(rhs)).simplify())
            }

            // arcsin(x) == rhs  →  x == sin(rhs)
            Self::Arcsin(x) => x.solve_inner(target_var, LoweredOp::Sin(Arc::new(rhs)).simplify()),

            // arccos(x) == rhs  →  x == cos(rhs)
            Self::Arccos(x) => x.solve_inner(target_var, LoweredOp::Cos(Arc::new(rhs)).simplify()),

            // arctan(x) == rhs  →  x == tan(rhs)
            Self::Arctan(x) => x.solve_inner(target_var, LoweredOp::Tan(Arc::new(rhs)).simplify()),

            // arcsinh(x) == rhs  →  x == sinh(rhs)
            Self::Arcsinh(x) => {
                x.solve_inner(target_var, LoweredOp::Sinh(Arc::new(rhs)).simplify())
            }

            // arccosh(x) == rhs  →  x == cosh(rhs)
            Self::Arccosh(x) => {
                x.solve_inner(target_var, LoweredOp::Cosh(Arc::new(rhs)).simplify())
            }

            // arctanh(x) == rhs  →  x == tanh(rhs)
            Self::Arctanh(x) => {
                x.solve_inner(target_var, LoweredOp::Tanh(Arc::new(rhs)).simplify())
            }

            // ---- Binary operators -----------------------------------------

            // a + b == rhs
            Self::Add(a, b) => {
                if a.contains_var(target_var) && !b.contains_var(target_var) {
                    // a == rhs − b
                    a.solve_inner(
                        target_var,
                        LoweredOp::Sub(Arc::new(rhs), Arc::clone(b)).simplify(),
                    )
                } else if b.contains_var(target_var) && !a.contains_var(target_var) {
                    // b == rhs − a
                    b.solve_inner(
                        target_var,
                        LoweredOp::Sub(Arc::new(rhs), Arc::clone(a)).simplify(),
                    )
                } else {
                    // Both contain target_var — cannot invert algebraically.
                    SolveResult::Residual(
                        LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
                    )
                }
            }

            // a − b == rhs
            Self::Sub(a, b) => {
                if a.contains_var(target_var) && !b.contains_var(target_var) {
                    // a == rhs + b
                    a.solve_inner(
                        target_var,
                        LoweredOp::Add(Arc::new(rhs), Arc::clone(b)).simplify(),
                    )
                } else if b.contains_var(target_var) && !a.contains_var(target_var) {
                    // −b == rhs − a  →  b == a − rhs
                    b.solve_inner(
                        target_var,
                        LoweredOp::Sub(Arc::clone(a), Arc::new(rhs)).simplify(),
                    )
                } else {
                    SolveResult::Residual(
                        LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
                    )
                }
            }

            // a * b == rhs
            Self::Mul(a, b) => {
                if a.contains_var(target_var) && !b.contains_var(target_var) {
                    // a == rhs / b
                    a.solve_inner(
                        target_var,
                        LoweredOp::Div(Arc::new(rhs), Arc::clone(b)).simplify(),
                    )
                } else if b.contains_var(target_var) && !a.contains_var(target_var) {
                    // b == rhs / a
                    b.solve_inner(
                        target_var,
                        LoweredOp::Div(Arc::new(rhs), Arc::clone(a)).simplify(),
                    )
                } else {
                    SolveResult::Residual(
                        LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
                    )
                }
            }

            // a / b == rhs
            Self::Div(a, b) => {
                if a.contains_var(target_var) && !b.contains_var(target_var) {
                    // a == rhs * b
                    a.solve_inner(
                        target_var,
                        LoweredOp::Mul(Arc::new(rhs), Arc::clone(b)).simplify(),
                    )
                } else if b.contains_var(target_var) && !a.contains_var(target_var) {
                    // a / b == rhs  →  b == a / rhs
                    b.solve_inner(
                        target_var,
                        LoweredOp::Div(Arc::clone(a), Arc::new(rhs)).simplify(),
                    )
                } else {
                    SolveResult::Residual(
                        LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
                    )
                }
            }

            // base^exp == rhs
            Self::Pow(base, exp) => {
                if base.contains_var(target_var) && !exp.contains_var(target_var) {
                    // base == rhs^(1/exp)
                    let inv_exp = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::clone(exp));
                    base.solve_inner(
                        target_var,
                        LoweredOp::Pow(Arc::new(rhs), Arc::new(inv_exp)).simplify(),
                    )
                } else if exp.contains_var(target_var) && !base.contains_var(target_var) {
                    // base^exp == rhs  →  exp == ln(rhs) / ln(base)
                    let ln_rhs = LoweredOp::Ln(Arc::new(rhs));
                    let ln_base = LoweredOp::Ln(Arc::clone(base));
                    exp.solve_inner(
                        target_var,
                        LoweredOp::Div(Arc::new(ln_rhs), Arc::new(ln_base)).simplify(),
                    )
                } else {
                    SolveResult::Residual(
                        LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
                    )
                }
            }

            // Fallback for any variant not covered above (e.g. Var(i) where i != target_var
            // that somehow passed the contains_var guard — should not occur in practice).
            _ => SolveResult::Residual(
                LoweredOp::Sub(Arc::new(self.clone()), Arc::new(rhs)).simplify(),
            ),
        }
    }
}

impl SolveResult {
    /// Resolve to a numeric value.
    ///
    /// * [`SolveResult::Closed`] — evaluates the closed-form expression
    ///   with `bindings`, injecting `x0` as the value for `var`.
    ///   Returns `Err(EmlError::NanEncountered)` if the result is non-finite.
    /// * [`SolveResult::Residual`]`(g)` — calls `g.find_root(var, bindings, x0)`.
    pub fn solve_numeric(
        &self,
        var: usize,
        bindings: &crate::eval::EvalCtx,
        x0: f64,
    ) -> Result<f64, crate::error::EmlError> {
        match self {
            Self::Closed(expr) => {
                let v = crate::numeric::eval_at_pub(expr, var, bindings, x0);
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(crate::error::EmlError::NanEncountered)
                }
            }
            Self::Residual(g) => g.find_root(var, bindings, x0),
        }
    }
}

// ── C4: solve_for_all, solve_linear_system ────────────────────────────────────

use crate::error::EmlError;
use crate::poly::MultiPoly;
use crate::solve_poly::{RootsResult, solve_polynomial, try_lambert_w_solve};

/// All real roots of `expr = rhs` in the given variable.
pub fn solve_for_all(
    expr: &LoweredOp,
    rhs: &LoweredOp,
    var: usize,
) -> Result<RootsResult, EmlError> {
    use crate::poly::Poly;
    let f = LoweredOp::Sub(Arc::new(expr.clone()), Arc::new(rhs.clone())).simplify();

    if let Ok(poly) = Poly::from_lowered(&f, var) {
        if poly.degree().is_some_and(|d| d >= 1) {
            return solve_polynomial(&poly, var);
        }
        return Ok(RootsResult { roots: vec![] });
    }

    if let Some(result) = try_lambert_w_solve(&f, var) {
        return Ok(result);
    }

    match expr.solve_for(var, rhs) {
        SolveResult::Closed(op) => Ok(RootsResult { roots: vec![op] }),
        SolveResult::Residual(_) => Err(EmlError::NotSolvable),
    }
}

/// Result of solving a linear system of equations.
#[derive(Debug, Clone)]
pub enum SystemSolveResult {
    /// A unique solution was found: `vars[i] = values[i]`.
    Unique(Vec<LoweredOp>),
    /// The system is underdetermined (infinitely many solutions).
    Underdetermined,
    /// The system is inconsistent (no solution exists).
    Inconsistent,
    /// The system contains nonlinear terms that are not handled.
    Nonlinear,
}

/// Solve a linear system `exprs[i] = 0` for variables `vars`.
pub fn solve_linear_system(
    exprs: &[LoweredOp],
    vars: &[usize],
) -> Result<SystemSolveResult, EmlError> {
    let m = exprs.len();
    let n = vars.len();

    if m == 0 || n == 0 {
        return Ok(SystemSolveResult::Underdetermined);
    }

    let max_var = vars.iter().max().copied().unwrap_or(0) + 1;

    let mut a_mat: Vec<f64> = vec![0.0; m * n];
    let mut b_vec: Vec<f64> = vec![0.0; m];

    for (i, expr) in exprs.iter().enumerate() {
        let mp = match MultiPoly::from_lowered(expr, max_var) {
            Ok(p) => p,
            Err(_) => return Ok(SystemSolveResult::Nonlinear),
        };

        for (exp_vec, coeff) in &mp.terms {
            let coeff_f64 = *coeff.numer() as f64 / *coeff.denom() as f64;
            let total_deg: u32 = vars
                .iter()
                .map(|&v| if v < exp_vec.len() { exp_vec[v] } else { 0 })
                .sum();
            if total_deg > 1 {
                return Ok(SystemSolveResult::Nonlinear);
            }
            let has_other_vars = exp_vec
                .iter()
                .enumerate()
                .any(|(vi, &e)| e > 0 && !vars.contains(&vi));
            if has_other_vars {
                return Ok(SystemSolveResult::Nonlinear);
            }
            if total_deg == 1 {
                for (j, &v) in vars.iter().enumerate() {
                    let e = if v < exp_vec.len() { exp_vec[v] } else { 0 };
                    if e == 1 {
                        a_mat[i * n + j] += coeff_f64;
                        break;
                    }
                }
            } else {
                b_vec[i] -= coeff_f64;
            }
        }
    }

    if m != n {
        if m < n {
            return Ok(SystemSolveResult::Underdetermined);
        }
        return Ok(SystemSolveResult::Nonlinear);
    }

    let mut a_work = a_mat.clone();
    let mut b_work = b_vec.clone();

    match crate::linalg::solve_lu(&mut a_work, &mut b_work, n) {
        Ok(()) => {
            let x = &b_work;
            let mut consistent = true;
            for i in 0..m {
                let mut row_sum = 0.0;
                for j in 0..n {
                    row_sum += a_mat[i * n + j] * x[j];
                }
                if (row_sum - b_vec[i]).abs() > 1e-6 {
                    consistent = false;
                    break;
                }
            }
            if !consistent {
                return Ok(SystemSolveResult::Inconsistent);
            }
            let solutions: Vec<LoweredOp> = x.iter().map(|&v| LoweredOp::Const(v)).collect();
            Ok(SystemSolveResult::Unique(solutions))
        }
        Err(EmlError::SingularMatrix) => {
            let b_nonzero = b_vec.iter().any(|&v| v.abs() > 1e-10);
            if b_nonzero {
                Ok(SystemSolveResult::Inconsistent)
            } else {
                Ok(SystemSolveResult::Underdetermined)
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod c4_tests {
    use super::*;
    use std::sync::Arc;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    #[test]
    fn test_solve_quadratic_golden_ratio() {
        // x² + x - 1 = 0 => roots are (-1 ± √5)/2
        // root_minus = (-1 - √5)/2 ≈ -1.618
        // root_plus  = (-1 + √5)/2 ≈ +0.618
        let x2 = LoweredOp::Pow(Arc::new(var(0)), Arc::new(c(2.0)));
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Add(Arc::new(x2), Arc::new(var(0)))),
            Arc::new(c(-1.0)),
        );
        let result = solve_for_all(&expr, &c(0.0), 0).unwrap();
        assert_eq!(result.roots.len(), 2);
        let r0 = result.roots[0].eval(&[]);
        let r1 = result.roots[1].eval(&[]);
        let root_minus = (-1.0 - 5.0_f64.sqrt()) / 2.0;
        let root_plus = (-1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(
            (r0 - root_minus).abs() < 1e-10,
            "r0={r0}, expected {root_minus}"
        );
        assert!(
            (r1 - root_plus).abs() < 1e-10,
            "r1={r1}, expected {root_plus}"
        );
    }

    #[test]
    fn test_solve_quadratic_two_roots() {
        // x² - 2 = 0 => roots ±√2
        let x2 = LoweredOp::Pow(Arc::new(var(0)), Arc::new(c(2.0)));
        let expr = LoweredOp::Sub(Arc::new(x2), Arc::new(c(2.0)));
        let result = solve_for_all(&expr, &c(0.0), 0).unwrap();
        assert_eq!(
            result.roots.len(),
            2,
            "expected 2 roots, got {:?}",
            result.roots.len()
        );
        let sqrt2 = 2.0_f64.sqrt();
        let vals: Vec<f64> = result.roots.iter().map(|r| r.eval(&[])).collect();
        assert!(vals.iter().any(|&v| (v + sqrt2).abs() < 1e-10));
        assert!(vals.iter().any(|&v| (v - sqrt2).abs() < 1e-10));
    }

    #[test]
    fn test_solve_cubic_three_roots() {
        // x³ - x = 0 => roots {-1, 0, 1}
        let x3 = LoweredOp::Pow(Arc::new(var(0)), Arc::new(c(3.0)));
        let expr = LoweredOp::Sub(Arc::new(x3), Arc::new(var(0)));
        let result = solve_for_all(&expr, &c(0.0), 0).unwrap();
        assert!(
            result.roots.len() >= 3,
            "expected >=3 roots, got {:?}",
            result.roots.len()
        );
        let vals: Vec<f64> = result.roots.iter().map(|r| r.eval(&[])).collect();
        assert!(vals.iter().any(|&v| v.abs() < 1e-6), "missing root at 0");
        assert!(
            vals.iter().any(|&v| (v - 1.0).abs() < 1e-6),
            "missing root at 1"
        );
        assert!(
            vals.iter().any(|&v| (v + 1.0).abs() < 1e-6),
            "missing root at -1"
        );
    }

    #[test]
    fn test_lambert_w_basic() {
        use crate::numeric::lambert_w0;
        assert!((lambert_w0(std::f64::consts::E).unwrap() - 1.0).abs() < 1e-10);
        let neg_inv_e = -1.0 / std::f64::consts::E;
        assert!((lambert_w0(neg_inv_e).unwrap() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_lambert_w_out_of_domain() {
        use crate::error::EmlError;
        use crate::numeric::lambert_w0;
        let result = lambert_w0(-1.0);
        assert!(matches!(result, Err(EmlError::OutOfDomain)));
    }

    #[test]
    fn test_solve_x_exp_x() {
        // x·eˣ = 1 => x = W₀(1) ≈ 0.5671
        let x_exp_x = LoweredOp::Mul(Arc::new(var(0)), Arc::new(LoweredOp::Exp(Arc::new(var(0)))));
        let result = solve_for_all(&x_exp_x, &c(1.0), 0).unwrap();
        assert!(!result.roots.is_empty(), "expected at least one root");
        let root = result.roots[0].eval(&[]);
        assert!((root * root.exp() - 1.0).abs() < 1e-8, "root={root}");
    }

    #[test]
    fn test_solve_linear_system_2x2() {
        // x + y = 3, x - y = 1 => x=2, y=1
        let expr0 = LoweredOp::Sub(
            Arc::new(LoweredOp::Add(Arc::new(var(0)), Arc::new(var(1)))),
            Arc::new(c(3.0)),
        );
        let expr1 = LoweredOp::Sub(
            Arc::new(LoweredOp::Sub(Arc::new(var(0)), Arc::new(var(1)))),
            Arc::new(c(1.0)),
        );
        let result = solve_linear_system(&[expr0, expr1], &[0, 1]).unwrap();
        match result {
            SystemSolveResult::Unique(sols) => {
                assert_eq!(sols.len(), 2);
                let x = sols[0].eval(&[]);
                let y = sols[1].eval(&[]);
                assert!((x - 2.0).abs() < 1e-10, "x={x}");
                assert!((y - 1.0).abs() < 1e-10, "y={y}");
            }
            other => panic!("expected Unique, got {:?}", other),
        }
    }

    #[test]
    fn test_solve_linear_system_inconsistent() {
        // x + y = 1, x + y = 2 => inconsistent
        let expr0 = LoweredOp::Sub(
            Arc::new(LoweredOp::Add(Arc::new(var(0)), Arc::new(var(1)))),
            Arc::new(c(1.0)),
        );
        let expr1 = LoweredOp::Sub(
            Arc::new(LoweredOp::Add(Arc::new(var(0)), Arc::new(var(1)))),
            Arc::new(c(2.0)),
        );
        let result = solve_linear_system(&[expr0, expr1], &[0, 1]).unwrap();
        assert!(matches!(result, SystemSolveResult::Inconsistent));
    }

    #[test]
    fn test_solve_linear_1d() {
        // 2x - 6 = 0 => x = 3
        let expr = LoweredOp::Sub(
            Arc::new(LoweredOp::Mul(Arc::new(c(2.0)), Arc::new(var(0)))),
            Arc::new(c(6.0)),
        );
        let result = solve_for_all(&expr, &c(0.0), 0).unwrap();
        assert!(!result.roots.is_empty());
        let root = result.roots[0].eval(&[]);
        assert!((root - 3.0).abs() < 1e-10, "root={root}");
    }
}
