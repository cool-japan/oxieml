//! Symbolic differentiation for the lowered IR.
//!
//! Implements partial derivatives, Jacobian, and Hessian for [`LoweredOp`]
//! trees via the chain rule and standard calculus identities.

use crate::lower::LoweredOp;
use std::sync::Arc;

impl LoweredOp {
    /// Symbolic partial derivative of this operation tree with respect to
    /// variable `wrt`.
    ///
    /// Applies standard calculus rules (sum, product, quotient, chain) to
    /// every variant of [`LoweredOp`] and returns a new `LoweredOp`
    /// representing the derivative. The result is post-processed via
    /// [`LoweredOp::simplify`] so that constant folding and 0/1 identities
    /// collapse trivial subterms.
    ///
    /// # Variables
    ///
    /// Variables are indexed from 0. `Var(i).grad(v)` is `Const(1.0)` when
    /// `i == v` and `Const(0.0)` otherwise.
    ///
    /// # `Pow`
    ///
    /// The general power rule is used (both base and exponent may depend on
    /// any variable). Concretely,
    /// `d/dx base^expo = base^expo · (expo'·ln(base) + expo·base'/base)`.
    /// For constant exponents this simplifies to the familiar
    /// `n·base^(n-1)·base'` after [`LoweredOp::simplify`] — but because the
    /// current simplifier does not perform algebraic cancellation, the
    /// surface form may keep the generic shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxieml::LoweredOp;
    ///
    /// // f(x, y) = x * y, df/dx = y
    /// let op = LoweredOp::Mul(
    ///     std::sync::Arc::new(LoweredOp::Var(0)),
    ///     std::sync::Arc::new(LoweredOp::Var(1)),
    /// );
    /// let df_dx = op.grad(0);
    /// assert!((df_dx.eval(&[3.0, 5.0]) - 5.0).abs() < 1e-12);
    /// ```
    pub fn grad(&self, wrt: usize) -> Self {
        let shared = raw_grad(self, wrt).simplify().cse();
        Arc::try_unwrap(shared).unwrap_or_else(|a| (*a).clone())
    }

    /// Count the number of distinct variable indices present in this tree.
    ///
    /// Returns `max(i) + 1` over all `Var(i)` nodes, or `0` if no `Var`
    /// nodes exist. This gives the minimum variable vector length required
    /// for a valid [`eval`](Self::eval) call.
    pub fn count_vars(&self) -> usize {
        match self {
            Self::Const(_) | Self::NamedConst(_) => 0,
            Self::Var(i) => i + 1,
            Self::Neg(x)
            | Self::Exp(x)
            | Self::Ln(x)
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
            | Self::Erf(x)
            | Self::LGamma(x)
            | Self::Digamma(x)
            | Self::Trigamma(x)
            | Self::Ei(x)
            | Self::Si(x)
            | Self::Ci(x) => x.count_vars(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b) => a.count_vars().max(b.count_vars()),
        }
    }

    /// Compute the vector of partial derivatives `[∂f/∂x0, ∂f/∂x1, …]`.
    ///
    /// Calls [`grad`](Self::grad) for each index `0..count_vars()` and
    /// simplifies each result.
    pub fn grad_all(&self) -> Vec<Self> {
        let n = self.count_vars();
        (0..n).map(|i| self.grad(i)).collect()
    }

    /// Return the Jacobian row for this scalar expression with exactly
    /// `n_vars` columns.
    ///
    /// If `n_vars > count_vars()` the vector is padded with `Const(0.0)`.
    /// If `n_vars < count_vars()` the vector is truncated.
    pub fn jacobian(&self, n_vars: usize) -> Vec<Self> {
        let mut grads = self.grad_all();
        while grads.len() < n_vars {
            grads.push(Self::Const(0.0));
        }
        grads.truncate(n_vars);
        grads
    }

    /// Compute the Hessian matrix of second-order partial derivatives.
    ///
    /// Returns an `n_vars × n_vars` matrix where `H[i][j] = ∂²f / ∂xi ∂xj`.
    /// Only the upper triangle is computed (O(n²·|tree|) complexity), then
    /// mirrored to the lower triangle exploiting Schwarz's symmetry theorem.
    pub fn hessian(&self, n_vars: usize) -> Vec<Vec<Self>> {
        let jac = self.jacobian(n_vars);
        // Collect upper-triangle (i, j, entry) tuples first, then assign.
        // This avoids the needless-range-loop lint that fires when `j` is
        // used both for `grad(j)` and for double-indexing `h[i][j]`/`h[j][i]`.
        let upper: Vec<(usize, usize, Self)> = jac
            .iter()
            .enumerate()
            .flat_map(|(i, jac_row)| (i..n_vars).map(move |j| (i, j, jac_row.grad(j))))
            .collect();
        let mut h = vec![vec![Self::Const(0.0); n_vars]; n_vars];
        for (i, j, entry) in upper {
            // Exploit Schwarz symmetry: H[i][j] == H[j][i].
            h[i][j] = entry.clone();
            h[j][i] = entry;
        }
        h
    }

    /// Symbolic nth derivative d^n f / dx_wrt^n.
    ///
    /// Applies [`grad`](Self::grad) `n` times, each time returning a simplified
    /// expression. Complexity grows exponentially for symbolic expressions, so
    /// keep `n` small (typically ≤ 6).
    ///
    /// # Example
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    ///
    /// // d^4/dx^4 exp(x) = exp(x)
    /// let expr = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
    /// let d4 = expr.nth_derivative(0, 4).unwrap();
    /// let val = d4.eval(&[1.5]);
    /// let expected = (1.5_f64).exp();
    /// assert!((val - expected).abs() < 1e-8);
    /// ```
    pub fn nth_derivative(&self, wrt: usize, n: usize) -> Result<Self, crate::error::EmlError> {
        let mut result = self.clone();
        for _ in 0..n {
            result = result.grad(wrt);
        }
        Ok(result)
    }

    /// Mixed partial derivative `∂^k f / ∂x_{vars[0]} … ∂x_{vars[k-1]}`.
    ///
    /// Applies [`grad`](Self::grad) once for each variable index in `vars`,
    /// in the given order. For smooth functions the order does not matter
    /// (Schwarz's theorem).
    ///
    /// # Example
    ///
    /// ```
    /// use oxieml::LoweredOp;
    /// use std::sync::Arc;
    ///
    /// // f(x,y) = x² * y; ∂²f/∂x∂y = 2x; at x=3 → 6
    /// let expr = LoweredOp::Mul(
    ///     Arc::new(LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)))),
    ///     Arc::new(LoweredOp::Var(1)),
    /// );
    /// let mp = expr.mixed_partial(&[0, 1]);
    /// assert!((mp.eval(&[3.0, 1.0]) - 6.0).abs() < 1e-8);
    /// ```
    pub fn mixed_partial(&self, vars: &[usize]) -> Self {
        let mut result = self.clone();
        for &wrt in vars {
            result = result.grad(wrt);
        }
        result
    }
}

/// Build the raw (un-simplified) symbolic derivative of `op` with respect to
/// variable `wrt`.
///
/// Callers should always route through [`LoweredOp::grad`], which applies
/// [`LoweredOp::simplify`] on the result. This helper exists so the rewrite
/// rules are easy to read and test in isolation.
pub(crate) fn raw_grad(op: &LoweredOp, wrt: usize) -> LoweredOp {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => LoweredOp::Const(0.0),
        LoweredOp::Var(i) => {
            if *i == wrt {
                LoweredOp::Const(1.0)
            } else {
                LoweredOp::Const(0.0)
            }
        }
        LoweredOp::Add(a, b) => {
            LoweredOp::Add(Arc::new(raw_grad(a, wrt)), Arc::new(raw_grad(b, wrt)))
        }
        LoweredOp::Sub(a, b) => {
            LoweredOp::Sub(Arc::new(raw_grad(a, wrt)), Arc::new(raw_grad(b, wrt)))
        }
        LoweredOp::Mul(a, b) => {
            // (a·b)' = a'·b + a·b'
            let da = raw_grad(a, wrt);
            let db = raw_grad(b, wrt);
            LoweredOp::Add(
                Arc::new(LoweredOp::Mul(Arc::new(da), Arc::clone(b))),
                Arc::new(LoweredOp::Mul(Arc::clone(a), Arc::new(db))),
            )
        }
        LoweredOp::Div(a, b) => {
            // (a/b)' = (a'·b - a·b') / (b·b)
            let da = raw_grad(a, wrt);
            let db = raw_grad(b, wrt);
            let num = LoweredOp::Sub(
                Arc::new(LoweredOp::Mul(Arc::new(da), Arc::clone(b))),
                Arc::new(LoweredOp::Mul(Arc::clone(a), Arc::new(db))),
            );
            let denom = LoweredOp::Mul(Arc::clone(b), Arc::clone(b));
            LoweredOp::Div(Arc::new(num), Arc::new(denom))
        }
        LoweredOp::Exp(a) => {
            // d/dx exp(f) = exp(f) · f'
            let da = raw_grad(a, wrt);
            LoweredOp::Mul(Arc::new(LoweredOp::Exp(Arc::clone(a))), Arc::new(da))
        }
        LoweredOp::Ln(a) => {
            // d/dx ln(f) = f' / f
            let da = raw_grad(a, wrt);
            LoweredOp::Div(Arc::new(da), Arc::clone(a))
        }
        LoweredOp::Sin(a) => {
            // d/dx sin(f) = cos(f) · f'
            let da = raw_grad(a, wrt);
            LoweredOp::Mul(Arc::new(LoweredOp::Cos(Arc::clone(a))), Arc::new(da))
        }
        LoweredOp::Cos(a) => {
            // d/dx cos(f) = -sin(f) · f'
            let da = raw_grad(a, wrt);
            LoweredOp::Neg(Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Sin(Arc::clone(a))),
                Arc::new(da),
            )))
        }
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(raw_grad(a, wrt))),
        LoweredOp::Pow(base, expo) => {
            // General power rule via exp-log rewriting:
            //   d/dx base^expo
            //     = base^expo · (expo' · ln(base) + expo · base' / base)
            let base_grad = raw_grad(base, wrt);
            let expo_grad = raw_grad(expo, wrt);
            let bracket = LoweredOp::Add(
                Arc::new(LoweredOp::Mul(
                    Arc::new(expo_grad),
                    Arc::new(LoweredOp::Ln(Arc::clone(base))),
                )),
                Arc::new(LoweredOp::Div(
                    Arc::new(LoweredOp::Mul(Arc::clone(expo), Arc::new(base_grad))),
                    Arc::clone(base),
                )),
            );
            LoweredOp::Mul(
                Arc::new(LoweredOp::Pow(Arc::clone(base), Arc::clone(expo))),
                Arc::new(bracket),
            )
        }
        LoweredOp::Tan(a) => {
            // d/dx tan(f) = (1 + tan²(f)) · f'
            let da = raw_grad(a, wrt);
            let tan_sq = LoweredOp::Mul(
                Arc::new(LoweredOp::Tan(Arc::clone(a))),
                Arc::new(LoweredOp::Tan(Arc::clone(a))),
            );
            let one_plus_tan_sq = LoweredOp::Add(Arc::new(LoweredOp::Const(1.0)), Arc::new(tan_sq));
            LoweredOp::Mul(Arc::new(one_plus_tan_sq), Arc::new(da))
        }
        LoweredOp::Sinh(a) => {
            // d/dx sinh(f) = cosh(f) · f'
            let da = raw_grad(a, wrt);
            LoweredOp::Mul(Arc::new(LoweredOp::Cosh(Arc::clone(a))), Arc::new(da))
        }
        LoweredOp::Cosh(a) => {
            // d/dx cosh(f) = sinh(f) · f'
            let da = raw_grad(a, wrt);
            LoweredOp::Mul(Arc::new(LoweredOp::Sinh(Arc::clone(a))), Arc::new(da))
        }
        LoweredOp::Tanh(a) => {
            // d/dx tanh(f) = (1 - tanh²(f)) · f'
            let da = raw_grad(a, wrt);
            let tanh_sq = LoweredOp::Pow(
                Arc::new(LoweredOp::Tanh(Arc::clone(a))),
                Arc::new(LoweredOp::Const(2.0)),
            );
            let one_minus_tanh_sq =
                LoweredOp::Sub(Arc::new(LoweredOp::Const(1.0)), Arc::new(tanh_sq));
            LoweredOp::Mul(Arc::new(one_minus_tanh_sq), Arc::new(da))
        }
        LoweredOp::Arcsin(a) => {
            // d/dx arcsin(f) = 1 / sqrt(1 - f²) · f'
            let da = raw_grad(a, wrt);
            let f_sq = LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0)));
            let one_minus_fsq = LoweredOp::Sub(Arc::new(LoweredOp::Const(1.0)), Arc::new(f_sq));
            let denom = LoweredOp::Pow(Arc::new(one_minus_fsq), Arc::new(LoweredOp::Const(0.5)));
            let deriv = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(denom));
            LoweredOp::Mul(Arc::new(deriv), Arc::new(da))
        }
        LoweredOp::Arccos(a) => {
            // d/dx arccos(f) = -1 / sqrt(1 - f²) · f'
            let da = raw_grad(a, wrt);
            let f_sq = LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0)));
            let one_minus_fsq = LoweredOp::Sub(Arc::new(LoweredOp::Const(1.0)), Arc::new(f_sq));
            let denom = LoweredOp::Pow(Arc::new(one_minus_fsq), Arc::new(LoweredOp::Const(0.5)));
            let neg_deriv = LoweredOp::Neg(Arc::new(LoweredOp::Div(
                Arc::new(LoweredOp::Const(1.0)),
                Arc::new(denom),
            )));
            LoweredOp::Mul(Arc::new(neg_deriv), Arc::new(da))
        }
        LoweredOp::Arctan(a) => {
            // d/dx arctan(f) = 1 / (1 + f²) · f'
            let da = raw_grad(a, wrt);
            let f_sq = LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0)));
            let one_plus_fsq = LoweredOp::Add(Arc::new(LoweredOp::Const(1.0)), Arc::new(f_sq));
            let deriv = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(one_plus_fsq));
            LoweredOp::Mul(Arc::new(deriv), Arc::new(da))
        }
        LoweredOp::Arcsinh(a) => {
            // d/dx arcsinh(f) = 1 / sqrt(1 + f²) · f'
            let da = raw_grad(a, wrt);
            let f_sq = LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0)));
            let one_plus_fsq = LoweredOp::Add(Arc::new(LoweredOp::Const(1.0)), Arc::new(f_sq));
            let denom = LoweredOp::Pow(Arc::new(one_plus_fsq), Arc::new(LoweredOp::Const(0.5)));
            let deriv = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(denom));
            LoweredOp::Mul(Arc::new(deriv), Arc::new(da))
        }
        LoweredOp::Arccosh(a) => {
            // d/dx arccosh(f) = 1 / sqrt(f² - 1) · f'
            let da = raw_grad(a, wrt);
            let f_sq = LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0)));
            let fsq_minus_one = LoweredOp::Sub(Arc::new(f_sq), Arc::new(LoweredOp::Const(1.0)));
            let denom = LoweredOp::Pow(Arc::new(fsq_minus_one), Arc::new(LoweredOp::Const(0.5)));
            let deriv = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(denom));
            LoweredOp::Mul(Arc::new(deriv), Arc::new(da))
        }
        LoweredOp::Arctanh(a) => {
            // d/dx arctanh(f) = 1 / (1 - f²) · f'
            let da = raw_grad(a, wrt);
            let f_sq = LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0)));
            let one_minus_fsq = LoweredOp::Sub(Arc::new(LoweredOp::Const(1.0)), Arc::new(f_sq));
            let deriv = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(one_minus_fsq));
            LoweredOp::Mul(Arc::new(deriv), Arc::new(da))
        }
        // d/dx erf(f) = (2/√π) e^{-f²} · f'
        LoweredOp::Erf(a) => {
            let da = raw_grad(a, wrt);
            let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
            let factor = LoweredOp::Mul(
                Arc::new(LoweredOp::Const(two_over_sqrt_pi)),
                Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Neg(Arc::new(
                    LoweredOp::Pow(Arc::clone(a), Arc::new(LoweredOp::Const(2.0))),
                ))))),
            );
            LoweredOp::Mul(Arc::new(factor), Arc::new(da))
        }
        // d/dx lgamma(f) = digamma(f) · f'
        LoweredOp::LGamma(a) => {
            let da = raw_grad(a, wrt);
            LoweredOp::Mul(Arc::new(LoweredOp::Digamma(Arc::clone(a))), Arc::new(da))
        }
        // d/dx digamma(f) = trigamma(f) · f' — trigamma not yet implemented, return 0
        LoweredOp::Digamma(_a) => {
            // Trigamma (ψ¹) is not yet implemented; return 0 as placeholder.
            LoweredOp::Const(0.0)
        }
        // d/dx trigamma(f) = tetragamma(f) · f' — tetragamma not yet implemented, return 0
        LoweredOp::Trigamma(_a) => {
            // Tetragamma (ψ²) is not yet implemented; return 0 as placeholder.
            LoweredOp::Const(0.0)
        }
        // d/dx Ei(f) = e^f/f · f'
        LoweredOp::Ei(a) => {
            let da = raw_grad(a, wrt);
            let factor = LoweredOp::Div(Arc::new(LoweredOp::Exp(Arc::clone(a))), Arc::clone(a));
            LoweredOp::Mul(Arc::new(factor), Arc::new(da))
        }
        // d/dx Si(f) = sin(f)/f · f'
        LoweredOp::Si(a) => {
            let da = raw_grad(a, wrt);
            let factor = LoweredOp::Div(Arc::new(LoweredOp::Sin(Arc::clone(a))), Arc::clone(a));
            LoweredOp::Mul(Arc::new(factor), Arc::new(da))
        }
        // d/dx Ci(f) = cos(f)/f · f'
        LoweredOp::Ci(a) => {
            let da = raw_grad(a, wrt);
            let factor = LoweredOp::Div(Arc::new(LoweredOp::Cos(Arc::clone(a))), Arc::clone(a));
            LoweredOp::Mul(Arc::new(factor), Arc::new(da))
        }
    }
}
