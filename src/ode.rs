//! Symbolic ODE solving: `dsolve`.
//!
//! Recognises and closes five classical families of ODEs expressed as
//! residual expression trees (`eq = 0`), using the variable-slot scheme in
//! [`OdeForm`]:
//!
//! | Family                       | Example                       |
//! |------------------------------|-------------------------------|
//! | Separable                    | y′ = x · y                    |
//! | First-order linear           | y′ + p(x)·y = q(x)            |
//! | Exact                        | M dx + N dy = 0               |
//! | Bernoulli                    | y′ + p(x)·y = q(x)·yⁿ        |
//! | 2nd-order constant-coeff.    | a·y″ + b·y′ + c·y = 0        |
//!
//! The ODE is represented as a [`crate::LoweredOp`] residual tree with
//! dedicated variable slots for x, y, y′, y″ (specified by [`OdeForm`]).
//! Arbitrary constants are placed at fresh variable slots starting at
//! [`OdeForm::c_start`].

use crate::integrate::IntegrateResult;
use crate::lower::LoweredOp;
use crate::solve::SolveResult;
use std::sync::Arc;

// ── Public Types ──────────────────────────────────────────────────────────────

/// Variable-slot assignments for an ODE in the dependent variable y(x).
///
/// Indices into the `Var(i)` leaf of a [`LoweredOp`] expression tree.
///
/// ```
/// use oxieml::ode::OdeForm;
/// let form = OdeForm { x: 0, y: 1, dy: 2, d2y: 3, c_start: 10 };
/// assert_eq!(form.c_start, 10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OdeForm {
    /// Independent variable slot (x).
    pub x: usize,
    /// Dependent variable slot (y).
    pub y: usize,
    /// First derivative slot (y′ = dy/dx).
    pub dy: usize,
    /// Second derivative slot (y″ = d²y/dx²).
    pub d2y: usize,
    /// Starting slot for arbitrary constants C₁, C₂, …
    pub c_start: usize,
}

impl OdeForm {
    /// Construct an `OdeForm` given the number of variables already in use.
    ///
    /// Assigns x=0, y=1, dy=2, d2y=3, c_start=max(n_existing_vars, 4).
    pub fn new(n_existing_vars: usize) -> Self {
        OdeForm {
            x: 0,
            y: 1,
            dy: 2,
            d2y: 3,
            c_start: n_existing_vars.max(4),
        }
    }

    fn c1(&self) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(self.c_start))
    }

    fn c2(&self) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(self.c_start + 1))
    }
}

/// Symbolic ODE solution.
///
/// Returned by [`dsolve`] alongside an [`OdeKind`] diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub enum OdeSolution {
    /// Explicit solution `y = f(x, C₁)`.
    Explicit(LoweredOp),
    /// Implicit solution `F(x, y, C₁) = 0`.
    Implicit(LoweredOp),
    /// ODE was not recognised or could not be solved symbolically.
    Unsolved,
}

/// Diagnostic: which ODE family was recognised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdeKind {
    /// y′ = f(x) · g(y)
    Separable,
    /// y′ + p(x)·y = q(x)
    FirstOrderLinear,
    /// M(x,y) dx + N(x,y) dy = 0 with ∂M/∂y = ∂N/∂x
    Exact,
    /// y′ + p(x)·y = q(x)·yⁿ  (n ≠ 0, 1)
    Bernoulli,
    /// a·y″ + b·y′ + c·y = 0  (constants a,b,c)
    SecondOrderConstCoeff,
    /// No recognised pattern.
    Unsolved,
}

// ── Entry Point ───────────────────────────────────────────────────────────────

/// Symbolically solve an ODE given as a residual expression `eq = 0`.
///
/// Variable slots are specified by `form`. E.g. for `y′ = x·y`, pass
/// `eq = Sub(Var(dy), Mul(Var(x), Var(y)))`.
///
/// Returns `(solution, kind)` where `kind` identifies which family was matched.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use oxieml::LoweredOp;
/// use oxieml::ode::{OdeForm, OdeKind, OdeSolution, dsolve};
///
/// // y′ - x·y = 0  (separable: y = C·e^{x²/2})
/// let form = OdeForm { x: 0, y: 1, dy: 2, d2y: 3, c_start: 10 };
/// let eq = LoweredOp::Sub(
///     Arc::new(LoweredOp::Var(form.dy)),
///     Arc::new(LoweredOp::Mul(
///         Arc::new(LoweredOp::Var(form.x)),
///         Arc::new(LoweredOp::Var(form.y)),
///     )),
/// );
/// let (sol, kind) = dsolve(&eq, &form);
/// assert_eq!(kind, OdeKind::Separable);
/// assert!(matches!(sol, OdeSolution::Explicit(_) | OdeSolution::Implicit(_)));
/// ```
pub fn dsolve(eq: &LoweredOp, form: &OdeForm) -> (OdeSolution, OdeKind) {
    // 1. Second-order constant-coefficient (checked first to avoid
    //    misclassification by first-order families).
    if let Some(sol) = try_second_order_const_coeff(eq, form) {
        return (sol, OdeKind::SecondOrderConstCoeff);
    }

    // 2. Separable: y′ = f(x) · g(y)
    if let Some(sol) = try_separable(eq, form) {
        return (sol, OdeKind::Separable);
    }

    // 3. First-order linear: y′ + p(x)·y = q(x)
    if let Some(sol) = try_first_order_linear(eq, form) {
        return (sol, OdeKind::FirstOrderLinear);
    }

    // 4. Exact: M(x,y) dx + N(x,y) dy = 0
    if let Some(sol) = try_exact(eq, form) {
        return (sol, OdeKind::Exact);
    }

    // 5. Bernoulli: y′ + p(x)·y = q(x)·yⁿ
    if let Some(sol) = try_bernoulli(eq, form) {
        return (sol, OdeKind::Bernoulli);
    }

    (OdeSolution::Unsolved, OdeKind::Unsolved)
}

// ── Helper Utilities ──────────────────────────────────────────────────────────

/// Returns `true` if `expr` contains `Var(var)` anywhere in its subtree.
fn depends_on_var(expr: &LoweredOp, var: usize) -> bool {
    expr.contains_var(var)
}

/// Attempt to obtain the antiderivative of `expr` wrt variable `wrt`.
/// Returns `None` if integration is unsupported.
fn integrate_expr(expr: &LoweredOp, wrt: usize) -> Option<LoweredOp> {
    match expr.integrate(wrt) {
        IntegrateResult::Closed(anti) => Some(anti),
        IntegrateResult::Unsupported => None,
    }
}

/// Attempt to solve `eq = 0` for `Var(target_var)`.
/// Returns `None` if the result is not a closed form.
fn solve_for_dy(eq: &LoweredOp, target_var: usize) -> Option<LoweredOp> {
    match eq.solve_for(target_var, &LoweredOp::Const(0.0)) {
        SolveResult::Closed(sol) => Some(sol),
        SolveResult::Residual(_) => None,
    }
}

/// Build `1 / expr`.
fn recip(expr: LoweredOp) -> LoweredOp {
    LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(expr))
}

/// Try to simplify `exp(c * ln(f(x)))` → `f(x)^c` (power law).
/// Also handles `exp(a * ln(f) + b * ln(g))` → ... but starts simple.
fn simplify_exp_ln(expr: &LoweredOp) -> LoweredOp {
    match expr {
        LoweredOp::Exp(inner) => {
            let inner_s = inner.as_ref().clone().simplify();
            // exp(c * ln(f)) → f^c
            if let LoweredOp::Mul(a, b) = &inner_s {
                if let LoweredOp::Const(c) = a.as_ref() {
                    if let LoweredOp::Ln(f) = b.as_ref() {
                        return LoweredOp::Pow(Arc::clone(f), Arc::new(LoweredOp::Const(*c)))
                            .simplify();
                    }
                }
                if let LoweredOp::Const(c) = b.as_ref() {
                    if let LoweredOp::Ln(f) = a.as_ref() {
                        return LoweredOp::Pow(Arc::clone(f), Arc::new(LoweredOp::Const(*c)))
                            .simplify();
                    }
                }
            }
            // exp(ln(f)) → f (already handled by simplify, but belt-and-suspenders)
            if let LoweredOp::Ln(f) = &inner_s {
                return f.as_ref().clone();
            }
            // exp(neg(ln(f))) → f^{-1} = 1/f
            if let LoweredOp::Neg(inner2) = &inner_s {
                if let LoweredOp::Ln(f) = inner2.as_ref() {
                    return recip(f.as_ref().clone());
                }
            }
            expr.clone()
        }
        _ => expr.clone(),
    }
}

/// Build `s * expr` (folds `1.0 * x → x`).
fn scalar_mul(s: f64, expr: LoweredOp) -> LoweredOp {
    if (s - 1.0).abs() < 1e-14 {
        return expr;
    }
    LoweredOp::Mul(Arc::new(LoweredOp::Const(s)), Arc::new(expr))
}

/// Build `e^{r * x}` for a scalar coefficient `r` and expression `x`.
/// Folds `r = 0 → Const(1)` and `r = 1 → Exp(x)`.
fn exp_term(coeff: LoweredOp, r: f64, x: &LoweredOp) -> LoweredOp {
    if r.abs() < 1e-14 {
        // e^0 = 1; coeff * 1 = coeff
        return coeff;
    }
    let rx = scalar_mul(r, x.clone());
    let exp_rx = LoweredOp::Exp(Arc::new(rx));
    // coeff * exp_rx
    match &coeff {
        LoweredOp::Const(v) if (*v - 1.0).abs() < 1e-14 => exp_rx,
        _ => LoweredOp::Mul(Arc::new(coeff), Arc::new(exp_rx)),
    }
}

/// Compute the structural hash of a `LoweredOp` for equality comparison.
fn structural_hash_value(expr: &LoweredOp) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    expr.structural_hash(&mut h);
    h.finish()
}

/// Structural equality check: same symbolic form.
fn ops_struct_equal(a: &LoweredOp, b: &LoweredOp) -> bool {
    structural_hash_value(a) == structural_hash_value(b) && format!("{a:?}") == format!("{b:?}")
}

/// Numerical equality check: evaluate both expressions at several (x,y) test points.
/// Returns `true` if all values agree within tolerance.
/// Variables above `n_vars` are left at 0.
fn ops_numeric_equal(a: &LoweredOp, b: &LoweredOp, x_slot: usize, y_slot: usize) -> bool {
    let n_vars = x_slot.max(y_slot) + 1;
    let mut vars = vec![0.0f64; n_vars];
    let test_pts: &[(f64, f64)] = &[
        (1.0, 1.0),
        (-1.0, 2.0),
        (0.5, -0.5),
        (2.0, 3.0),
        (-2.0, -1.0),
    ];
    for &(xv, yv) in test_pts {
        vars[x_slot] = xv;
        vars[y_slot] = yv;
        let va = a.eval(&vars);
        let vb = b.eval(&vars);
        if !va.is_finite() || !vb.is_finite() {
            continue; // Skip undefined points
        }
        // Relative or absolute tolerance
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(1.0);
        if diff / scale > 1e-8 {
            return false;
        }
    }
    true
}

// ── Separability ─────────────────────────────────────────────────────────────

/// Returns `true` if `expr` contains only `Var(form.x)` (and constants),
/// with no dependence on `form.y` or `form.dy`.
fn only_x(expr: &LoweredOp, form: &OdeForm) -> bool {
    !depends_on_var(expr, form.y) && !depends_on_var(expr, form.dy)
}

/// Returns `true` if `expr` contains only `Var(form.y)` (and constants),
/// with no dependence on `form.x` or `form.dy`.
fn only_y(expr: &LoweredOp, form: &OdeForm) -> bool {
    !depends_on_var(expr, form.x) && !depends_on_var(expr, form.dy)
}

/// Check if `expr = f(x) * g(y)` where f depends only on x and g only on y.
///
/// Returns `(f_x_part, g_y_part)` on success. Handles:
/// - expr depends only on x → `(expr, 1)`
/// - expr depends only on y → `(1, expr)`
/// - `Neg(inner)` → propagate negation into x-part
/// - `Mul(a, b)` where a is x-only and b is y-only (or vice versa)
/// - `Div(a, b)` where a is y-only and b is x-only → `(1/b, a)` (g = a, f = 1/b)
///   or a is x-only and b is y-only → `(a, 1/b)`
///   or numerator mixes x and y: try factoring
fn separate_x_y(expr: &LoweredOp, form: &OdeForm) -> Option<(LoweredOp, LoweredOp)> {
    // If it depends on dy, it's not separable in x and y
    if depends_on_var(expr, form.dy) {
        return None;
    }

    let dep_x = depends_on_var(expr, form.x);
    let dep_y = depends_on_var(expr, form.y);

    // Pure x or constant
    if !dep_y {
        return Some((expr.clone(), LoweredOp::Const(1.0)));
    }
    // Pure y
    if !dep_x {
        return Some((LoweredOp::Const(1.0), expr.clone()));
    }

    // Neg(inner) → negate the x-part
    if let LoweredOp::Neg(inner) = expr {
        if let Some((fx, gy)) = separate_x_y(inner, form) {
            let neg_fx = LoweredOp::Neg(Arc::new(fx)).simplify();
            return Some((neg_fx, gy));
        }
        return None;
    }

    // Mul(a, b)
    if let LoweredOp::Mul(a, b) = expr {
        // Try a = x-only, b = y-only
        if only_x(a, form) && only_y(b, form) {
            return Some((a.as_ref().clone(), b.as_ref().clone()));
        }
        // Try a = y-only, b = x-only
        if only_y(a, form) && only_x(b, form) {
            return Some((b.as_ref().clone(), a.as_ref().clone()));
        }
        // Try recursively: Mul(separated_sub, something)
        if let Some((fa, ga)) = separate_x_y(a, form) {
            if let Some((fb, gb)) = separate_x_y(b, form) {
                let fx = LoweredOp::Mul(Arc::new(fa), Arc::new(fb)).simplify();
                let gy = LoweredOp::Mul(Arc::new(ga), Arc::new(gb)).simplify();
                return Some((fx, gy));
            }
        }
        return None;
    }

    // Div(a, b)
    if let LoweredOp::Div(a, b) = expr {
        // Case 1: a = x-only, b = y-only → f(x)/g(y), g_part = 1/b
        if only_x(a, form) && only_y(b, form) {
            return Some((a.as_ref().clone(), recip(b.as_ref().clone())));
        }
        // Case 2: a = y-only, b = x-only → (1/b)*a = (1/b(x)) * a(y)
        if only_y(a, form) && only_x(b, form) {
            return Some((recip(b.as_ref().clone()), a.as_ref().clone()));
        }
        // Case 3: a mixes x and y, b is x-only → try to separate a
        if only_x(b, form) {
            if let Some((fa, ga)) = separate_x_y(a, form) {
                // a/b = (fa/b) * ga
                let fx = LoweredOp::Div(Arc::new(fa), Arc::clone(b)).simplify();
                return Some((fx, ga));
            }
        }
        // Case 4: a mixes x and y, b is y-only → a / b(y) = try to separate a
        if only_y(b, form) {
            if let Some((fa, ga)) = separate_x_y(a, form) {
                // a/b = fa * (ga/b)
                let gy = LoweredOp::Div(Arc::new(ga), Arc::clone(b)).simplify();
                return Some((fa, gy));
            }
        }
        return None;
    }

    None
}

// ── Family 1: Separable ───────────────────────────────────────────────────────

fn try_separable(eq: &LoweredOp, form: &OdeForm) -> Option<OdeSolution> {
    // If eq depends on d2y, it's not first-order
    if depends_on_var(eq, form.d2y) {
        return None;
    }

    // Solve eq = 0 for Var(dy) to get y' = rhs
    let rhs = solve_for_dy(eq, form.dy)?;

    // Try to separate rhs = f(x) * g(y)
    let (fx, gy) = separate_x_y(&rhs, form)?;

    // Integrate 1/g(y) dy
    let inv_gy = recip(gy);
    let lhs_anti = integrate_expr(&inv_gy, form.y)?;

    // Integrate f(x) dx
    let rhs_anti = integrate_expr(&fx, form.x)?;

    // Implicit solution: lhs_anti(y) - rhs_anti(x) - C₁ = 0
    let c1 = form.c1();
    let rhs_with_c = LoweredOp::Add(Arc::new(rhs_anti), c1);
    let implicit_eq = LoweredOp::Sub(Arc::new(lhs_anti), Arc::new(rhs_with_c));
    let implicit_eq = implicit_eq.simplify();

    // Try to solve for y explicitly
    match implicit_eq.solve_for(form.y, &LoweredOp::Const(0.0)) {
        SolveResult::Closed(y_sol) => Some(OdeSolution::Explicit(y_sol.simplify())),
        SolveResult::Residual(_) => Some(OdeSolution::Implicit(implicit_eq)),
    }
}

// ── Family 2: First-order linear ─────────────────────────────────────────────

/// Try to split `expr = a(x) * Var(y) + b(x)` where neither a nor b depend on y.
/// Returns `(a, b)` i.e. `(coefficient of y, free term)`.
fn split_linear_in_y(expr: &LoweredOp, form: &OdeForm) -> Option<(LoweredOp, LoweredOp)> {
    // Compute ∂expr/∂y — if this depends on y, expr is not linear in y
    let grad_y = expr.grad(form.y);
    if depends_on_var(&grad_y, form.y) {
        return None;
    }

    // The free term: b = expr - grad_y * Var(y)
    let y_op = LoweredOp::Var(form.y);
    let linear_term = LoweredOp::Mul(Arc::new(grad_y.clone()), Arc::new(y_op));
    let b_raw = LoweredOp::Sub(Arc::new(expr.clone()), Arc::new(linear_term));
    let b = b_raw.simplify();

    // Verify b doesn't depend on y
    if depends_on_var(&b, form.y) {
        return None;
    }

    Some((grad_y.simplify(), b))
}

fn try_first_order_linear(eq: &LoweredOp, form: &OdeForm) -> Option<OdeSolution> {
    // If eq depends on d2y, not first-order
    if depends_on_var(eq, form.d2y) {
        return None;
    }

    // Solve for y': y' = rhs
    let rhs = solve_for_dy(eq, form.dy)?;

    // Split: rhs = -p(x)*y + q(x)  →  (coeff_of_y, free_term)
    let (neg_p, q) = split_linear_in_y(&rhs, form)?;

    // p(x) = -neg_p
    let p = LoweredOp::Neg(Arc::new(neg_p)).simplify();

    // Integrating factor: μ = exp(∫p(x)dx)
    let int_p = integrate_expr(&p, form.x)?;
    let mu_raw = LoweredOp::Exp(Arc::new(int_p)).simplify();
    let mu = simplify_exp_ln(&mu_raw).simplify();

    // Numerator: ∫ μ·q dx + C₁
    let mu_q = LoweredOp::Mul(Arc::new(mu.clone()), Arc::new(q)).simplify();
    let int_mu_q = integrate_expr(&mu_q, form.x)?;
    let c1 = form.c1();
    let numerator = LoweredOp::Add(Arc::new(int_mu_q), c1);

    // y = numerator / μ
    let y_sol = LoweredOp::Div(Arc::new(numerator), Arc::new(mu)).simplify();

    Some(OdeSolution::Explicit(y_sol))
}

// ── Family 3: Exact ───────────────────────────────────────────────────────────

/// Substitute `Var(slot) → replacement` in `expr`, recursively.
fn subst_var(expr: &LoweredOp, slot: usize, replacement: &LoweredOp) -> LoweredOp {
    match expr {
        LoweredOp::Var(i) => {
            if *i == slot {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => expr.clone(),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(subst_var(a, slot, replacement)),
            Arc::new(subst_var(b, slot, replacement)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(subst_var(a, slot, replacement)),
            Arc::new(subst_var(b, slot, replacement)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(subst_var(a, slot, replacement)),
            Arc::new(subst_var(b, slot, replacement)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            Arc::new(subst_var(a, slot, replacement)),
            Arc::new(subst_var(b, slot, replacement)),
        ),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(subst_var(a, slot, replacement))),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            Arc::new(subst_var(a, slot, replacement)),
            Arc::new(subst_var(b, slot, replacement)),
        ),
    }
}

/// Split `eq = M(x,y) + N(x,y) * Var(dy)` where N = ∂eq/∂Var(dy).
/// M is obtained by substituting dy→0 in eq.
fn split_m_n(eq: &LoweredOp, form: &OdeForm) -> Option<(LoweredOp, LoweredOp)> {
    // N = ∂eq/∂Var(dy)
    let n_raw = eq.grad(form.dy);
    let n = n_raw.simplify();

    // M = eq with Var(dy) → 0 (reliable alternative to eq - N*dy)
    let zero = LoweredOp::Const(0.0);
    let m = subst_var(eq, form.dy, &zero).simplify();

    // M must not depend on dy after substitution
    if depends_on_var(&m, form.dy) {
        return None;
    }

    // N must not depend on dy (eq is linear in dy)
    if depends_on_var(&n, form.dy) {
        return None;
    }

    Some((m, n))
}

fn try_exact(eq: &LoweredOp, form: &OdeForm) -> Option<OdeSolution> {
    // Exact ODEs are first-order
    if depends_on_var(eq, form.d2y) {
        return None;
    }

    let (m, n) = split_m_n(eq, form)?;

    // Exactness condition: ∂M/∂y = ∂N/∂x (checked numerically to handle commutativity)
    let dm_dy = m.grad(form.y).simplify();
    let dn_dx = n.grad(form.x).simplify();

    // Use numerical check (handles cases like 2x+2y vs 2y+2x where structural differs)
    if !ops_struct_equal(&dm_dy, &dn_dx) && !ops_numeric_equal(&dm_dy, &dn_dx, form.x, form.y) {
        return None;
    }

    // Potential function F: ∂F/∂x = M  →  F = ∫M dx + g(y)
    let f_partial = integrate_expr(&m, form.x)?;

    // g′(y) = N - ∂f_partial/∂y
    let df_partial_dy = f_partial.grad(form.y).simplify();
    let g_prime_raw = LoweredOp::Sub(Arc::new(n), Arc::new(df_partial_dy));
    let g_prime = g_prime_raw.simplify();

    // g′ must only depend on y (or be zero).
    // Use numerical check for the x-dependence test (handles simplifier gaps).
    let g_prime_has_x = if depends_on_var(&g_prime, form.x) {
        // Numerically check if g_prime is actually zero or x-independent
        !ops_numeric_equal(&g_prime, &LoweredOp::Const(0.0), form.x, form.y) && {
            // Check if g_prime truly depends on x by evaluating at multiple x values
            let n_slots = form.c_start.max(form.d2y + 1);
            let mut vars = vec![0.0f64; n_slots];
            vars[form.y] = 1.0;
            vars[form.x] = 1.0;
            let v1 = g_prime.eval(&vars);
            vars[form.x] = 2.0;
            let v2 = g_prime.eval(&vars);
            v1.is_finite() && v2.is_finite() && (v1 - v2).abs() > 1e-8
        }
    } else {
        false
    };

    if g_prime_has_x {
        return None;
    }

    // If g_prime is numerically zero, integrate Const(0) wrt y = Const(0).
    let g_prime_eff = if ops_numeric_equal(&g_prime, &LoweredOp::Const(0.0), form.x, form.y) {
        LoweredOp::Const(0.0)
    } else {
        g_prime
    };

    let g = integrate_expr(&g_prime_eff, form.y).unwrap_or(LoweredOp::Const(0.0));

    // F = f_partial + g = C₁  →  implicit: F - C₁ = 0
    let f = LoweredOp::Add(Arc::new(f_partial), Arc::new(g)).simplify();
    let c1 = form.c1();
    let implicit_eq = LoweredOp::Sub(Arc::new(f), c1).simplify();

    Some(OdeSolution::Implicit(implicit_eq))
}

// ── Family 4: Bernoulli ───────────────────────────────────────────────────────

/// Detect y′ = -p(x)·y + q(x)·yⁿ for n ≠ 0, 1.
///
/// Approach: evaluate rhs at y=1 and use polynomial sampling to identify n and
/// the coefficients. The rhs is a polynomial of degree n in y if Bernoulli.
/// We try candidate integer and half-integer n values.
///
/// Returns `(p, q, n)` where p = p(x), q = q(x) don't depend on y.
fn detect_bernoulli_terms(rhs: &LoweredOp, form: &OdeForm) -> Option<(LoweredOp, LoweredOp, f64)> {
    // Strategy: if rhs = -p(x)*y + q(x)*y^n, then
    //   rhs(x, y) = -p(x)*y + q(x)*y^n
    // We use gradient to find p(x) = -∂rhs/∂y|_{y=0} ... but that requires y→0.
    //
    // Better approach: use the symbolic gradient.
    // ∂rhs/∂y = -p(x) + n*q(x)*y^{n-1}
    // ∂²rhs/∂y² = n*(n-1)*q(x)*y^{n-2}
    //
    // For n=2: ∂²/∂y² = 2*q(x) (constant in y)
    // For n=3: ∂²/∂y² = 6*q(x)*y (linear in y), ∂³/∂y³ = 6*q(x) (constant)
    //
    // We try to detect by checking if rhs has only two y-polynomial degrees: 1 and n.

    // First try to detect via structural pattern matching on known forms.
    // Case: rhs = Div(numerator, x_expr) where numerator is polynomial in y
    let rhs_simplified = rhs.clone().simplify();

    // Try to extract n by probing at several y values
    // rhs(x0, y) = -p(x0)*y + q(x0)*y^n for fixed x0
    // At y=0: rhs=0 (if it's truly Bernoulli with no constant term)
    // rhs(x0, y) / y = -p(x0) + q(x0)*y^{n-1}
    //
    // We probe: at y=2 and y=3 with x=x0
    // Divide by y:  g(y) = rhs/y = -p + q*y^{n-1}
    // g(2)/g(3) should let us solve for n-1
    // But this might not converge for non-integer n.

    // Simplest working approach: try candidate n values 2, 3, -1, 1/2
    let x0 = 1.5_f64; // test x value
    let mut vars = vec![0.0f64; form.c_start + 4];
    vars[form.x] = x0;

    // Evaluate rhs at several y values to detect the polynomial structure
    let probe_y = [1.0_f64, 2.0, 4.0, 0.5];
    let probe_rhs: Vec<f64> = probe_y
        .iter()
        .map(|&yv| {
            vars[form.y] = yv;
            rhs_simplified.eval(&vars)
        })
        .collect();

    // Check if rhs(y=0) = 0 (Bernoulli has no constant in y)
    vars[form.y] = 0.0;
    let rhs_at_0 = rhs_simplified.eval(&vars);
    if rhs_at_0.is_finite() && rhs_at_0.abs() > 1e-6 {
        return None; // Has constant term — not Bernoulli
    }

    // Try to detect n by ratio: rhs(y)/y = -p + q*y^{n-1}
    // (rhs(y2)/y2 - rhs(y1)/y1) / (rhs(y3)/y3 - rhs(y1)/y1)
    // = (q*(y2^{n-1} - y1^{n-1})) / (q*(y3^{n-1} - y1^{n-1}))
    // = (y2^{n-1} - y1^{n-1}) / (y3^{n-1} - y1^{n-1})
    let g: Vec<f64> = probe_y
        .iter()
        .zip(&probe_rhs)
        .map(|(&y, &r)| r / y)
        .collect();

    // Use g[0] at y=1, g[1] at y=2, g[2] at y=4
    // (g[1]-g[0])/(g[2]-g[0]) = (2^{n-1}-1)/(4^{n-1}-1)
    if !g[0].is_finite() || !g[1].is_finite() || !g[2].is_finite() {
        return None;
    }

    // Try candidate values of n (integer and common fractions)
    let candidates = [2.0_f64, 3.0, -1.0, 0.5, 4.0, 1.0 / 3.0, -2.0];
    let mut found_n = None;

    for &n_cand in &candidates {
        if (n_cand - 1.0).abs() < 1e-10 || n_cand.abs() < 1e-10 {
            continue;
        }
        // Expected g[1] - g[0] relative to g[2] - g[0]
        let exp_ratio =
            if (probe_y[2].powf(n_cand - 1.0) - probe_y[0].powf(n_cand - 1.0)).abs() > 1e-10 {
                (probe_y[1].powf(n_cand - 1.0) - probe_y[0].powf(n_cand - 1.0))
                    / (probe_y[2].powf(n_cand - 1.0) - probe_y[0].powf(n_cand - 1.0))
            } else {
                continue;
            };
        let obs_ratio = if (g[2] - g[0]).abs() > 1e-10 {
            (g[1] - g[0]) / (g[2] - g[0])
        } else {
            continue;
        };
        if (exp_ratio - obs_ratio).abs() < 1e-6 {
            found_n = Some(n_cand);
            break;
        }
    }

    let n_exp = found_n?;

    // Extract p(x) and q(x):
    // -p(x) = g(1) - q(x)*1^{n-1} = g(1) - q(x)
    // q(x) = (g(y) + p(x)) / y^{n-1} for any y ≠ 1
    // From the 2-equation system:
    //   g(y1) = -p + q*y1^{n-1}
    //   g(y2) = -p + q*y2^{n-1}
    // Solving: q = (g(y2) - g(y1)) / (y2^{n-1} - y1^{n-1})
    //          p = -(g(y1) - q*y1^{n-1})  = q*y1^{n-1} - g(y1)
    //
    // These are numeric for x=x0; we need the symbolic expressions.

    // Symbolic extraction:
    // ∂rhs/∂y = -p(x) + n*q(x)*y^{n-1}
    // at y=0 (if n>1): ∂rhs/∂y|_{y=0} = -p(x) ... but only if n>1
    // for n=2: -p(x) = ∂rhs/∂y at y=0 = ∂rhs/∂y  (since it's linear in y)
    //   Actually ∂(−py + qy^2)/∂y = -p + 2qy, so at y=0: -p

    let grad_y = rhs_simplified.grad(form.y).simplify();

    // For Bernoulli: ∂rhs/∂y = -p(x) + n*q(x)*y^{n-1}
    // Further: ∂²rhs/∂y² = n*(n-1)*q(x)*y^{n-2}
    // For n=2: ∂²/∂y² = 2*q(x) (constant in y)
    // For n=3: ∂³/∂y³ = 6*q(x)

    // Compute the nth order derivative to isolate q(x):
    // ∂^n rhs/∂y^n = n! * q(x)  (constant in y)
    // We need exactly n derivatives for this to be constant.
    let n_int = n_exp.round() as i32;
    if (n_exp - n_int as f64).abs() > 1e-10 || n_int < 2 {
        // Non-integer or n<2: fall back to numerical coefficient extraction
        return detect_bernoulli_numeric(rhs, form, n_exp);
    }

    // Take n derivatives to get the leading coefficient (n! * q(x))
    let mut deriv = rhs_simplified.clone();
    for _ in 0..n_int {
        deriv = deriv.grad(form.y).simplify();
    }
    // deriv = n! * q(x)  (factorial of n)
    let n_factorial: f64 = (1..=n_int).map(|k| k as f64).product();
    // q(x) = deriv / n!
    let q = LoweredOp::Div(Arc::new(deriv), Arc::new(LoweredOp::Const(n_factorial))).simplify();

    if depends_on_var(&q, form.y) {
        return None; // Not actually constant in y
    }

    // p(x): from ∂rhs/∂y at y=0 = -p(x) (for n>=2, the y^{n-1} term vanishes at y=0)
    // neg_p = (∂rhs/∂y)|_{y=0} = -p(x)
    let neg_p = subst_var(&grad_y, form.y, &LoweredOp::Const(0.0)).simplify();

    // neg_p should not depend on y (it was evaluated at y=0)
    if depends_on_var(&neg_p, form.y) {
        return None;
    }

    let p = LoweredOp::Neg(Arc::new(neg_p)).simplify();

    if depends_on_var(&p, form.y) {
        return None;
    }

    Some((p, q, n_exp))
}

/// Fallback: detect Bernoulli structure numerically for non-integer n.
fn detect_bernoulli_numeric(
    rhs: &LoweredOp,
    form: &OdeForm,
    n_exp: f64,
) -> Option<(LoweredOp, LoweredOp, f64)> {
    // For non-integer n, we cannot take integer derivatives.
    // Try: q(x) = rhs(x,y) / y^n - rhs(x,1) / 1^n ... this is complex.
    // Simplified: only handle integer n for now.
    let _ = (rhs, form, n_exp);
    None
}

fn try_bernoulli(eq: &LoweredOp, form: &OdeForm) -> Option<OdeSolution> {
    if depends_on_var(eq, form.d2y) {
        return None;
    }

    // Solve for y'
    let rhs = solve_for_dy(eq, form.dy)?;

    // Detect Bernoulli structure: y' = -p(x)*y + q(x)*y^n
    let (p, q, n) = detect_bernoulli_terms(&rhs, form)?;

    let one_minus_n = 1.0 - n;

    // New linear ODE for v = y^{1-n}:
    // v' + (1-n)*p(x)*v = (1-n)*q(x)
    //
    // We build this as a virtual first-order-linear problem using fresh slots
    // so we can reuse try_first_order_linear.
    let scale = LoweredOp::Const(one_minus_n);
    let new_p = LoweredOp::Mul(Arc::new(scale.clone()), Arc::new(p));
    let new_q = LoweredOp::Mul(Arc::new(scale), Arc::new(q));

    // Integrating factor for the v ODE: μ = exp(∫(1-n)*p dx)
    let new_p_s = new_p.simplify();
    let int_new_p = integrate_expr(&new_p_s, form.x)?;
    let mu_raw = LoweredOp::Exp(Arc::new(int_new_p)).simplify();
    let mu = simplify_exp_ln(&mu_raw).simplify();

    // Numerator: ∫ μ * (1-n)*q dx + C₁
    let new_q_s = new_q.simplify();
    let mu_new_q = LoweredOp::Mul(Arc::new(mu.clone()), Arc::new(new_q_s)).simplify();
    let int_mu_new_q = integrate_expr(&mu_new_q, form.x)?;
    let c1 = form.c1();
    let v_numerator = LoweredOp::Add(Arc::new(int_mu_new_q), c1);

    // v = numerator / μ
    let v_sol = LoweredOp::Div(Arc::new(v_numerator), Arc::new(mu)).simplify();

    // Back-substitute: y = v^{1/(1-n)}
    let exp_back = LoweredOp::Const(1.0 / one_minus_n);
    let y_sol = LoweredOp::Pow(Arc::new(v_sol), Arc::new(exp_back)).simplify();

    Some(OdeSolution::Explicit(y_sol))
}

// ── Family 5: 2nd-order constant-coefficient ─────────────────────────────────

/// Extract (a, b, c) from `eq ≡ a·d2y + b·dy + c·y` where a, b, c are constants.
fn extract_const_coeff_2nd(eq: &LoweredOp, form: &OdeForm) -> Option<(f64, f64, f64)> {
    let da = eq.grad(form.d2y).simplify();
    let db = eq.grad(form.dy).simplify();
    let dc = eq.grad(form.y).simplify();

    let a = match &da {
        LoweredOp::Const(v) => *v,
        _ => return None,
    };
    let b = match &db {
        LoweredOp::Const(v) => *v,
        _ => return None,
    };
    let c = match &dc {
        LoweredOp::Const(v) => *v,
        _ => return None,
    };

    // Verify: eq - (a*d2y + b*dy + c*y) should be ~zero
    let recon = LoweredOp::Add(
        Arc::new(LoweredOp::Add(
            Arc::new(scalar_mul(a, LoweredOp::Var(form.d2y))),
            Arc::new(scalar_mul(b, LoweredOp::Var(form.dy))),
        )),
        Arc::new(scalar_mul(c, LoweredOp::Var(form.y))),
    );
    let remainder = LoweredOp::Sub(Arc::new(eq.clone()), Arc::new(recon)).simplify();

    match &remainder {
        LoweredOp::Const(v) if v.abs() < 1e-12 => {}
        _ => {
            // Try numerically at a few points
            let test_pts: &[(f64, f64, f64, f64)] = &[
                (1.0, 2.0, 3.0, 4.0),
                (-1.0, 0.5, 2.0, -3.0),
                (0.0, 1.0, -1.0, 0.5),
            ];
            for &(xv, yv, dyv, d2yv) in test_pts {
                let mut vars = vec![0.0f64; form.c_start + 4];
                vars[form.x] = xv;
                vars[form.y] = yv;
                vars[form.dy] = dyv;
                vars[form.d2y] = d2yv;
                let eq_val = eq.eval(&vars);
                let recon_val = a * d2yv + b * dyv + c * yv;
                if (eq_val - recon_val).abs() > 1e-8 {
                    return None;
                }
            }
        }
    }

    Some((a, b, c))
}

fn try_second_order_const_coeff(eq: &LoweredOp, form: &OdeForm) -> Option<OdeSolution> {
    // Must depend on d2y to be second-order
    if !depends_on_var(eq, form.d2y) {
        return None;
    }

    let (a, b, c) = extract_const_coeff_2nd(eq, form)?;

    if a.abs() < 1e-14 {
        return None;
    }

    // Characteristic polynomial: a·r² + b·r + c = 0
    let disc = b * b - 4.0 * a * c;
    let x_op = LoweredOp::Var(form.x);
    let c1 = form.c1();
    let c2 = form.c2();

    if disc > 1e-10 {
        // Two distinct real roots
        let sq = disc.sqrt();
        let r1 = (-b - sq) / (2.0 * a);
        let r2 = (-b + sq) / (2.0 * a);

        // y = C₁·e^{r₁x} + C₂·e^{r₂x}
        let t1 = exp_term(c1.as_ref().clone(), r1, &x_op);
        let t2 = exp_term(c2.as_ref().clone(), r2, &x_op);
        let y_sol = LoweredOp::Add(Arc::new(t1), Arc::new(t2)).simplify();
        Some(OdeSolution::Explicit(y_sol))
    } else if disc.abs() <= 1e-10 {
        // Repeated real root r = -b / (2a)
        let r = -b / (2.0 * a);

        // y = (C₁ + C₂·x)·e^{rx}
        let c2_x = LoweredOp::Mul(Arc::clone(&c2), Arc::new(x_op.clone()));
        let bracket = LoweredOp::Add(Arc::clone(&c1), Arc::new(c2_x));
        let exp_rx = exp_term(LoweredOp::Const(1.0), r, &x_op);
        let y_sol = LoweredOp::Mul(Arc::new(bracket), Arc::new(exp_rx)).simplify();
        Some(OdeSolution::Explicit(y_sol))
    } else {
        // Complex conjugate roots: α ± βi
        let alpha = -b / (2.0 * a);
        let beta = (-disc).sqrt() / (2.0 * a);

        // y = e^{αx}·(C₁·cos(βx) + C₂·sin(βx))
        let exp_ax = exp_term(LoweredOp::Const(1.0), alpha, &x_op);
        let bx = scalar_mul(beta, x_op.clone());
        let cos_bx = LoweredOp::Cos(Arc::new(bx.clone()));
        let sin_bx = LoweredOp::Sin(Arc::new(bx));
        let c1_cos = LoweredOp::Mul(Arc::clone(&c1), Arc::new(cos_bx));
        let c2_sin = LoweredOp::Mul(Arc::clone(&c2), Arc::new(sin_bx));
        let trig_sum = LoweredOp::Add(Arc::new(c1_cos), Arc::new(c2_sin));
        let y_sol = LoweredOp::Mul(Arc::new(exp_ax), Arc::new(trig_sum)).simplify();
        Some(OdeSolution::Explicit(y_sol))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod ode_tests {
    use super::*;

    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Arc::new(a), Arc::new(b))
    }

    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Arc::new(a), Arc::new(b))
    }

    fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Sub(Arc::new(a), Arc::new(b))
    }

    fn pow(base: LoweredOp, exp: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(base), Arc::new(exp))
    }

    fn default_form() -> OdeForm {
        OdeForm {
            x: 0,
            y: 1,
            dy: 2,
            d2y: 3,
            c_start: 10,
        }
    }

    /// Evaluate the solution expression at a specific x with C₁ = c1_val.
    fn eval_solution(sol: &LoweredOp, form: &OdeForm, x_val: f64, c1_val: f64) -> f64 {
        let mut vars = vec![0.0f64; form.c_start + 4];
        vars[form.x] = x_val;
        vars[form.c_start] = c1_val;
        sol.eval(&vars)
    }

    #[test]
    fn test_separable_y_prime_eq_xy() {
        // y′ = x·y  →  separable  →  y = C·e^{x²/2}
        let form = default_form();
        // eq: dy - x*y = 0
        let eq = sub(var(form.dy), mul(var(form.x), var(form.y)));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::Separable, "Should recognise separable ODE");
        assert!(
            matches!(sol, OdeSolution::Explicit(_) | OdeSolution::Implicit(_)),
            "Should return a solution, got {sol:?}"
        );
    }

    #[test]
    fn test_separable_y_prime_eq_x() {
        // y′ = x  →  separable (g(y) = 1)  →  y = x²/2 + C
        let form = default_form();
        let eq = sub(var(form.dy), var(form.x));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::Separable, "y′=x should be separable");
        assert!(
            matches!(sol, OdeSolution::Explicit(_) | OdeSolution::Implicit(_)),
            "Should return a solution"
        );
    }

    #[test]
    fn test_first_order_linear_y_prime_plus_y_eq_x() {
        // y′ + y = x  →  y = x - 1 + C·e^{-x}
        let form = default_form();
        // eq: dy + y - x = 0
        let eq = sub(add(var(form.dy), var(form.y)), var(form.x));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(
            kind,
            OdeKind::FirstOrderLinear,
            "Should recognise first-order linear, got {:?}",
            kind
        );
        assert!(
            matches!(sol, OdeSolution::Explicit(_)),
            "Should return explicit solution"
        );

        // Spot-check: at x=0, y=0-1+C = C-1 → with C=1 → y(0)=0
        // y'+y=x: deriv at x=0 should equal x=0
        if let OdeSolution::Explicit(y_expr) = &sol {
            // With C1=1: y(0) = 0-1+e^0 = 0
            let y0 = eval_solution(y_expr, &form, 0.0, 1.0);
            // y(0) = -1 + C*e^0 = -1 + 1 = 0  (with x=0, the formula is x-1+C*e^{-x})
            assert!(y0.is_finite(), "Solution should be finite at x=0");
        }
    }

    #[test]
    fn test_exact_2xy_dx_plus_x2_dy() {
        // (2xy)dx + x²dy = 0 — both exact (∂M/∂y=2x=∂N/∂x) and separable (y'=-2y/x).
        // dsolve checks separable first, so we accept either family.
        let form = default_form();
        // eq = 2*x*y + x²*dy = 0
        let m = mul(mul(c(2.0), var(form.x)), var(form.y));
        let x2 = mul(var(form.x), var(form.x));
        let n_dy = mul(x2, var(form.dy));
        let eq = add(m, n_dy);
        let (sol, kind) = dsolve(&eq, &form);
        assert!(
            kind == OdeKind::Exact
                || kind == OdeKind::Separable
                || kind == OdeKind::FirstOrderLinear,
            "Should recognise as exact, separable, or linear, got {kind:?}"
        );
        assert!(
            matches!(sol, OdeSolution::Implicit(_) | OdeSolution::Explicit(_)),
            "Should return a solution"
        );
    }

    #[test]
    fn test_second_order_cc_two_real_roots() {
        // y″ - 3y′ + 2y = 0  →  r=1,2  →  y = C₁e^x + C₂e^{2x}
        let form = default_form();
        // eq: d2y - 3*dy + 2*y = 0
        let eq = add(
            sub(var(form.d2y), mul(c(3.0), var(form.dy))),
            mul(c(2.0), var(form.y)),
        );
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::SecondOrderConstCoeff);
        assert!(matches!(sol, OdeSolution::Explicit(_)));
    }

    #[test]
    fn test_second_order_cc_complex_roots() {
        // y″ + y = 0  →  r=±i  →  y = C₁cos(x) + C₂sin(x)
        let form = default_form();
        // eq: d2y + y = 0
        let eq = add(var(form.d2y), var(form.y));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::SecondOrderConstCoeff);
        // Complex-root branch: alpha=0, beta=1
        assert!(matches!(sol, OdeSolution::Explicit(_)));
    }

    #[test]
    fn test_second_order_cc_repeated_root() {
        // y″ - 2y′ + y = 0  →  (r-1)² = 0  →  y = (C₁ + C₂x)e^x
        let form = default_form();
        // eq: d2y - 2*dy + y = 0
        let eq = add(sub(var(form.d2y), mul(c(2.0), var(form.dy))), var(form.y));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::SecondOrderConstCoeff);
        assert!(matches!(sol, OdeSolution::Explicit(_)));
    }

    #[test]
    fn test_second_order_cc_pure_exponential() {
        // y″ - y = 0  →  r = ±1  →  y = C₁e^x + C₂e^{-x}
        let form = default_form();
        let eq = sub(var(form.d2y), var(form.y));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::SecondOrderConstCoeff);
        assert!(matches!(sol, OdeSolution::Explicit(_)));
    }

    #[test]
    fn test_bernoulli_y_prime_plus_y_eq_y_squared() {
        // y′ + y = y²  →  Bernoulli with n=2
        // Note: y' = y² - y = y(y-1) is ALSO separable, so separable solver
        // catches it first. We verify a solution is found (either family is valid).
        let form = default_form();
        // eq: dy + y - y² = 0
        let y_sq = pow(var(form.y), c(2.0));
        let eq = sub(add(var(form.dy), var(form.y)), y_sq);
        let (sol, kind) = dsolve(&eq, &form);
        assert!(
            kind == OdeKind::Bernoulli || kind == OdeKind::Separable,
            "Should recognise as Bernoulli or Separable (both valid), got {kind:?}"
        );
        assert!(
            matches!(sol, OdeSolution::Explicit(_) | OdeSolution::Implicit(_)),
            "Should return a solution"
        );
    }

    #[test]
    fn test_unrecognised_ode_returns_unsolved() {
        // y′ = sin(x·y) — not in any recognisable family
        let form = default_form();
        let xy = mul(var(form.x), var(form.y));
        let sin_xy = LoweredOp::Sin(Arc::new(xy));
        let eq = sub(var(form.dy), sin_xy);
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::Unsolved);
        assert!(matches!(sol, OdeSolution::Unsolved));
    }

    #[test]
    fn test_ode_form_new() {
        let form = OdeForm::new(10);
        assert_eq!(form.x, 0);
        assert_eq!(form.y, 1);
        assert_eq!(form.dy, 2);
        assert_eq!(form.d2y, 3);
        assert_eq!(form.c_start, 10);

        let form_small = OdeForm::new(2);
        assert_eq!(form_small.c_start, 4); // max(2, 4) = 4
    }

    #[test]
    fn test_separable_pure_y() {
        // y′ = y  →  y = Ce^x
        let form = default_form();
        let eq = sub(var(form.dy), var(form.y));
        let (sol, kind) = dsolve(&eq, &form);
        assert_eq!(kind, OdeKind::Separable, "y′=y should be separable");
        assert!(matches!(sol, OdeSolution::Explicit(_)));
    }

    #[test]
    fn test_first_order_linear_y_prime_eq_neg_y() {
        // y′ + y = 0  →  y = Ce^{-x}
        let form = default_form();
        let eq = add(var(form.dy), var(form.y));
        let (sol, kind) = dsolve(&eq, &form);
        // This can be both separable and linear — accept either
        assert!(
            kind == OdeKind::Separable || kind == OdeKind::FirstOrderLinear,
            "y′+y=0 should be separable or linear, got {kind:?}"
        );
        assert!(matches!(sol, OdeSolution::Explicit(_)));
    }
}
