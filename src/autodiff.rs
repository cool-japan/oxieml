//! Automatic differentiation: forward-mode (JVP) and reverse-mode (VJP).
//!
//! Implements dual-number forward-mode AD and a recursive reverse-mode sweep
//! for `LoweredOp` expression trees.

use crate::lower::LoweredOp;

// ──────────────────────────────────────────────────────────────────────────────
// Dual numbers for forward-mode AD
// ──────────────────────────────────────────────────────────────────────────────

/// A dual number `re + du·ε` where `ε² = 0`.
///
/// Used for forward-mode automatic differentiation (JVP).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dual {
    re: f64,
    du: f64,
}

impl Dual {
    fn new(re: f64, du: f64) -> Self {
        Self { re, du }
    }
}

impl std::ops::Add for Dual {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.du + other.du)
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.du - other.du)
    }
}

impl std::ops::Mul for Dual {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        // (a + b·ε)(c + d·ε) = a·c + (a·d + b·c)·ε
        Self::new(self.re * other.re, self.re * other.du + self.du * other.re)
    }
}

impl std::ops::Div for Dual {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        // (a + b·ε) / (c + d·ε) ≈ a/c + (b·c - a·d)/(c²)·ε
        let re = self.re / other.re;
        let du = (self.du * other.re - self.re * other.du) / (other.re * other.re);
        Self::new(re, du)
    }
}

impl std::ops::Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.du)
    }
}

/// Evaluate `expr` in dual-number arithmetic.
fn eval_dual(expr: &LoweredOp, vars: &[Dual]) -> Dual {
    match expr {
        LoweredOp::Const(c) => Dual::new(*c, 0.0),
        LoweredOp::NamedConst(nc) => Dual::new(nc.value(), 0.0),
        LoweredOp::Var(i) => vars.get(*i).copied().unwrap_or(Dual::new(0.0, 0.0)),
        LoweredOp::Neg(x) => -eval_dual(x, vars),
        LoweredOp::Add(a, b) => eval_dual(a, vars) + eval_dual(b, vars),
        LoweredOp::Sub(a, b) => eval_dual(a, vars) - eval_dual(b, vars),
        LoweredOp::Mul(a, b) => eval_dual(a, vars) * eval_dual(b, vars),
        LoweredOp::Div(a, b) => eval_dual(a, vars) / eval_dual(b, vars),
        LoweredOp::Exp(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.exp();
            Dual::new(re, re * xd.du)
        }
        LoweredOp::Ln(x) => {
            let xd = eval_dual(x, vars);
            Dual::new(xd.re.ln(), xd.du / xd.re)
        }
        LoweredOp::Sin(x) => {
            let xd = eval_dual(x, vars);
            Dual::new(xd.re.sin(), xd.re.cos() * xd.du)
        }
        LoweredOp::Cos(x) => {
            let xd = eval_dual(x, vars);
            Dual::new(xd.re.cos(), -xd.re.sin() * xd.du)
        }
        LoweredOp::Tan(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.tan();
            let cos = xd.re.cos();
            Dual::new(re, xd.du / (cos * cos))
        }
        LoweredOp::Sinh(x) => {
            let xd = eval_dual(x, vars);
            Dual::new(xd.re.sinh(), xd.re.cosh() * xd.du)
        }
        LoweredOp::Cosh(x) => {
            let xd = eval_dual(x, vars);
            Dual::new(xd.re.cosh(), xd.re.sinh() * xd.du)
        }
        LoweredOp::Tanh(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.tanh();
            Dual::new(re, (1.0 - re * re) * xd.du)
        }
        LoweredOp::Arcsin(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.asin();
            let du = xd.du / (1.0 - xd.re * xd.re).sqrt();
            Dual::new(re, du)
        }
        LoweredOp::Arccos(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.acos();
            let du = -xd.du / (1.0 - xd.re * xd.re).sqrt();
            Dual::new(re, du)
        }
        LoweredOp::Arctan(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.atan();
            let du = xd.du / (1.0 + xd.re * xd.re);
            Dual::new(re, du)
        }
        LoweredOp::Arcsinh(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.asinh();
            let du = xd.du / (xd.re * xd.re + 1.0).sqrt();
            Dual::new(re, du)
        }
        LoweredOp::Arccosh(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.acosh();
            let du = xd.du / (xd.re * xd.re - 1.0).sqrt();
            Dual::new(re, du)
        }
        LoweredOp::Arctanh(x) => {
            let xd = eval_dual(x, vars);
            let re = xd.re.atanh();
            let du = xd.du / (1.0 - xd.re * xd.re);
            Dual::new(re, du)
        }
        LoweredOp::Erf(x) => {
            let xd = eval_dual(x, vars);
            let deriv = std::f64::consts::FRAC_2_SQRT_PI * (-xd.re * xd.re).exp();
            Dual::new(crate::special::erf(xd.re), deriv * xd.du)
        }
        LoweredOp::LGamma(x) => {
            let xd = eval_dual(x, vars);
            Dual::new(
                crate::special::lgamma(xd.re),
                crate::special::digamma(xd.re) * xd.du,
            )
        }
        LoweredOp::Digamma(x) => {
            let xd = eval_dual(x, vars);
            // trigamma not implemented; du = 0 (correct only for constant)
            Dual::new(crate::special::digamma(xd.re), 0.0)
        }
        LoweredOp::Trigamma(x) => {
            let xd = eval_dual(x, vars);
            // tetragamma not implemented; du = 0 (correct only for constant)
            Dual::new(crate::special::trigamma(xd.re), 0.0)
        }
        LoweredOp::Ei(x) => {
            let xd = eval_dual(x, vars);
            let re = crate::special::ei(xd.re);
            let du = if xd.re != 0.0 {
                xd.re.exp() / xd.re * xd.du
            } else {
                0.0
            };
            Dual::new(re, du)
        }
        LoweredOp::Si(x) => {
            let xd = eval_dual(x, vars);
            let re = crate::special::si(xd.re);
            let du = if xd.re != 0.0 {
                xd.re.sin() / xd.re * xd.du
            } else {
                xd.du
            };
            Dual::new(re, du)
        }
        LoweredOp::Ci(x) => {
            let xd = eval_dual(x, vars);
            let re = crate::special::ci(xd.re);
            let du = if xd.re != 0.0 {
                xd.re.cos() / xd.re * xd.du
            } else {
                0.0
            };
            Dual::new(re, du)
        }
        LoweredOp::Pow(base, exp) => {
            let bd = eval_dual(base, vars);
            let ed = eval_dual(exp, vars);
            // d/dt base(t)^exp(t) = base^exp * (exp' * ln(base) + exp * base' / base)
            let re = bd.re.powf(ed.re);
            let du = re * (ed.du * bd.re.ln() + ed.re * bd.du / bd.re);
            Dual::new(re, du)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Reverse-mode AD helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Reverse-mode AD: propagate adjoints back through the expression tree.
///
/// `grad[i]` accumulates ∂L/∂x_i.
fn reverse_sweep(expr: &LoweredOp, adj: f64, vars: &[f64], grad: &mut Vec<f64>) {
    match expr {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => {}
        LoweredOp::Var(i) => {
            if *i < grad.len() {
                grad[*i] += adj;
            }
        }
        LoweredOp::Neg(x) => {
            reverse_sweep(x, -adj, vars, grad);
        }
        LoweredOp::Add(a, b) => {
            reverse_sweep(a, adj, vars, grad);
            reverse_sweep(b, adj, vars, grad);
        }
        LoweredOp::Sub(a, b) => {
            reverse_sweep(a, adj, vars, grad);
            reverse_sweep(b, -adj, vars, grad);
        }
        LoweredOp::Mul(a, b) => {
            let va = a.eval(vars);
            let vb = b.eval(vars);
            reverse_sweep(a, adj * vb, vars, grad);
            reverse_sweep(b, adj * va, vars, grad);
        }
        LoweredOp::Div(a, b) => {
            let va = a.eval(vars);
            let vb = b.eval(vars);
            reverse_sweep(a, adj / vb, vars, grad);
            reverse_sweep(b, -adj * va / (vb * vb), vars, grad);
        }
        LoweredOp::Exp(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj * vx.exp(), vars, grad);
        }
        LoweredOp::Ln(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj / vx, vars, grad);
        }
        LoweredOp::Sin(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj * vx.cos(), vars, grad);
        }
        LoweredOp::Cos(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, -adj * vx.sin(), vars, grad);
        }
        LoweredOp::Tan(x) => {
            let vx = x.eval(vars);
            let cos = vx.cos();
            reverse_sweep(x, adj / (cos * cos), vars, grad);
        }
        LoweredOp::Sinh(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj * vx.cosh(), vars, grad);
        }
        LoweredOp::Cosh(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj * vx.sinh(), vars, grad);
        }
        LoweredOp::Tanh(x) => {
            let vx = x.eval(vars);
            let th = vx.tanh();
            reverse_sweep(x, adj * (1.0 - th * th), vars, grad);
        }
        LoweredOp::Arcsin(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj / (1.0 - vx * vx).sqrt(), vars, grad);
        }
        LoweredOp::Arccos(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, -adj / (1.0 - vx * vx).sqrt(), vars, grad);
        }
        LoweredOp::Arctan(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj / (1.0 + vx * vx), vars, grad);
        }
        LoweredOp::Arcsinh(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj / (vx * vx + 1.0).sqrt(), vars, grad);
        }
        LoweredOp::Arccosh(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj / (vx * vx - 1.0).sqrt(), vars, grad);
        }
        LoweredOp::Arctanh(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj / (1.0 - vx * vx), vars, grad);
        }
        LoweredOp::Erf(x) => {
            let vx = x.eval(vars);
            let deriv = std::f64::consts::FRAC_2_SQRT_PI * (-vx * vx).exp();
            reverse_sweep(x, adj * deriv, vars, grad);
        }
        LoweredOp::LGamma(x) => {
            let vx = x.eval(vars);
            reverse_sweep(x, adj * crate::special::digamma(vx), vars, grad);
        }
        LoweredOp::Digamma(_x) => {
            // trigamma not implemented; no gradient propagated
        }
        LoweredOp::Trigamma(_x) => {
            // tetragamma not implemented; no gradient propagated
        }
        LoweredOp::Ei(x) => {
            let vx = x.eval(vars);
            if vx != 0.0 {
                reverse_sweep(x, adj * vx.exp() / vx, vars, grad);
            }
        }
        LoweredOp::Si(x) => {
            let vx = x.eval(vars);
            if vx != 0.0 {
                reverse_sweep(x, adj * vx.sin() / vx, vars, grad);
            }
        }
        LoweredOp::Ci(x) => {
            let vx = x.eval(vars);
            if vx != 0.0 {
                reverse_sweep(x, adj * vx.cos() / vx, vars, grad);
            }
        }
        LoweredOp::Pow(base, exp) => {
            let vb = base.eval(vars);
            let ve = exp.eval(vars);
            let pow_val = vb.powf(ve);
            // ∂/∂base: ve * vb^(ve-1) * adj
            if vb != 0.0 {
                reverse_sweep(base, adj * ve * vb.powf(ve - 1.0), vars, grad);
                // ∂/∂exp: vb^ve * ln(vb) * adj
                reverse_sweep(exp, adj * pow_val * vb.ln(), vars, grad);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public LoweredOp methods
// ──────────────────────────────────────────────────────────────────────────────

impl LoweredOp {
    /// Forward-mode AD (JVP): compute value and directional derivative simultaneously.
    ///
    /// Returns `(f(vars), df/dt)` where `df/dt = Σ_i (∂f/∂x_i) · tangents[i]`.
    /// A single forward pass of cost O(|ops|) — one scalar dual evaluation.
    ///
    /// # Example
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    ///
    /// // f(x, y) = x * y; at (3, 5), tangent (1, 0) → value=15, jvp=5
    /// let expr = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
    /// let (val, dval) = expr.jvp(&[3.0, 5.0], &[1.0, 0.0]);
    /// assert!((val - 15.0).abs() < 1e-12);
    /// assert!((dval - 5.0).abs() < 1e-12);
    /// ```
    pub fn jvp(&self, vars: &[f64], tangents: &[f64]) -> (f64, f64) {
        let dual_vars: Vec<Dual> = vars
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let t = tangents.get(i).copied().unwrap_or(0.0);
                Dual::new(v, t)
            })
            .collect();
        let result = eval_dual(self, &dual_vars);
        (result.re, result.du)
    }

    /// Reverse-mode AD (VJP): compute value and full gradient in one sweep.
    ///
    /// Returns `(f(vars), grad)` where `grad[i] = ∂f/∂x_i`.
    /// Cost: O(|ops|) — a forward eval pass followed by a backward adjoint sweep.
    ///
    /// # Example
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    ///
    /// // f(x, y) = x² + y²; at (2, 3) → grad = (4, 6)
    /// let expr = LoweredOp::Add(
    ///     Arc::new(LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)))),
    ///     Arc::new(LoweredOp::Pow(Arc::new(LoweredOp::Var(1)), Arc::new(LoweredOp::Const(2.0)))),
    /// );
    /// let (val, grad) = expr.vjp(&[2.0, 3.0]);
    /// assert!((val - 13.0).abs() < 1e-12);
    /// assert!((grad[0] - 4.0).abs() < 1e-10);
    /// assert!((grad[1] - 6.0).abs() < 1e-10);
    /// ```
    pub fn vjp(&self, vars: &[f64]) -> (f64, Vec<f64>) {
        let n = self.count_vars();
        let mut grad = vec![0.0_f64; n];
        let val = self.eval(vars);
        // Reverse sweep with initial adjoint = 1.0 (∂L/∂output = 1)
        reverse_sweep(self, 1.0, vars, &mut grad);
        (val, grad)
    }
}

#[cfg(test)]
mod tests {
    use crate::lower::LoweredOp;
    use std::sync::Arc;

    #[test]
    fn test_jvp_mul() {
        // f(x, y) = x * y; jvp at (3, 5), tangent (1, 0) → (15, 5)
        let expr = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
        let (val, dval) = expr.jvp(&[3.0, 5.0], &[1.0, 0.0]);
        assert!((val - 15.0).abs() < 1e-12);
        assert!((dval - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_vjp_x2_y2() {
        // f(x, y) = x² + y²; vjp at (2, 3) → grad = (4, 6)
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Pow(
                Arc::new(LoweredOp::Var(0)),
                Arc::new(LoweredOp::Const(2.0)),
            )),
            Arc::new(LoweredOp::Pow(
                Arc::new(LoweredOp::Var(1)),
                Arc::new(LoweredOp::Const(2.0)),
            )),
        );
        let (val, grad) = expr.vjp(&[2.0, 3.0]);
        assert!((val - 13.0).abs() < 1e-12);
        assert!((grad[0] - 4.0).abs() < 1e-10, "grad[0]={}", grad[0]);
        assert!((grad[1] - 6.0).abs() < 1e-10, "grad[1]={}", grad[1]);
    }

    #[test]
    fn test_jvp_exp() {
        // f(x) = exp(x); at x=1.0, tangent=1.0 → dval = exp(1.0)
        let expr = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
        let (val, dval) = expr.jvp(&[1.0], &[1.0]);
        let e = std::f64::consts::E;
        assert!((val - e).abs() < 1e-12);
        assert!((dval - e).abs() < 1e-12);
    }

    #[test]
    fn test_vjp_sin() {
        // f(x) = sin(x); grad = cos(x)
        let expr = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
        let x = 0.7;
        let (val, grad) = expr.vjp(&[x]);
        assert!((val - x.sin()).abs() < 1e-12);
        assert!((grad[0] - x.cos()).abs() < 1e-12);
    }
}
