//! Numeric algorithms: root-finding, quadrature.
//!
//! Provides Newton–Brent root-finding and adaptive Simpson quadrature
//! for `LoweredOp` expression trees.

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::lower::LoweredOp;

// Tolerance constants
const ROOT_TOL: f64 = 1e-12;
const ROOT_MAX_ITER: usize = 100;
const QUAD_TOL: f64 = 1e-10;
const QUAD_MAX_DEPTH: usize = 50;
const QUAD_MIN_DEPTH: usize = 2;

/// Tuning options for Newton–Brent root-finding.
#[derive(Clone, Copy, Debug)]
pub struct RootOpts {
    /// Absolute tolerance for both |f(x)| and |Δx|. Default: 1e-12.
    pub tol: f64,
    /// Maximum Newton iterations before switching to Brent. Default: 100.
    pub max_iter: usize,
}

impl Default for RootOpts {
    fn default() -> Self {
        Self {
            tol: ROOT_TOL,
            max_iter: ROOT_MAX_ITER,
        }
    }
}

/// Tuning options for adaptive Simpson quadrature.
#[derive(Clone, Copy, Debug)]
pub struct QuadOpts {
    /// Absolute error tolerance. Default: 1e-10.
    pub tol: f64,
    /// Maximum recursion depth. Default: 50.
    pub max_depth: usize,
    /// Minimum recursion depth (forces at least a few splits). Default: 2.
    pub min_depth: usize,
}

impl Default for QuadOpts {
    fn default() -> Self {
        Self {
            tol: QUAD_TOL,
            max_depth: QUAD_MAX_DEPTH,
            min_depth: QUAD_MIN_DEPTH,
        }
    }
}

/// Evaluate `expr` with variable `var` set to `x`, other vars from `bindings`.
/// Pads the var slice to avoid out-of-bounds panics.
pub(crate) fn eval_at_pub(expr: &LoweredOp, var: usize, bindings: &EvalCtx, x: f64) -> f64 {
    let base = bindings.as_slice();
    let needed = (var + 1).max(base.len());
    let mut vars: Vec<f64> = base.to_vec();
    while vars.len() < needed {
        vars.push(0.0);
    }
    vars[var] = x;
    expr.eval(&vars)
}

/// Evaluate `expr` with variable `var` set to `x`, other vars from `bindings`.
/// Pads the var slice to avoid out-of-bounds panics.
fn eval_at(expr: &LoweredOp, var: usize, bindings: &EvalCtx, x: f64) -> f64 {
    eval_at_pub(expr, var, bindings, x)
}

/// Run Brent's method on the bracket [p, q] where f(p)*f(q) <= 0.
fn brent_method(
    expr: &LoweredOp,
    var: usize,
    bindings: &EvalCtx,
    p: f64,
    q: f64,
    tol: f64,
) -> Result<f64, EmlError> {
    let fp = eval_at(expr, var, bindings, p);
    let fq = eval_at(expr, var, bindings, q);

    // Ensure a has the larger absolute value (Brent convention: |f(b)| <= |f(a)|)
    let (mut a, mut b, mut fa, mut fb) = if fq.abs() <= fp.abs() {
        (p, q, fp, fq)
    } else {
        (q, p, fq, fp)
    };

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0_f64;

    for _ in 0..200 {
        if fb.abs() < tol {
            return Ok(b);
        }
        if (b - a).abs() < tol {
            return Ok(b);
        }

        // Attempt inverse-quadratic interpolation or secant
        let mut s = if (fa - fc).abs() > f64::EPSILON * fa.abs()
            && (fb - fc).abs() > f64::EPSILON * fb.abs()
        {
            // Inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant
            b - fb * (b - a) / (fb - fa)
        };

        // Conditions under which we fall back to bisection
        let mid3 = (3.0 * a + b) / 4.0;
        let cond1 = !((mid3 < s && s < b) || (mid3 > s && s > b));
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = eval_at(expr, var, bindings, s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure |f(b)| <= |f(a)|
        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }

    Ok(b)
}

impl LoweredOp {
    /// Find a root of `self` with respect to variable `var`, starting from `x0`,
    /// using the provided options.
    ///
    /// Uses Newton's method first, then falls back to Brent's method if Newton
    /// does not converge or encounters a zero derivative.
    pub fn find_root_opts(
        &self,
        var: usize,
        bindings: &EvalCtx,
        x0: f64,
        opts: RootOpts,
    ) -> Result<f64, EmlError> {
        let tol = opts.tol;
        let max_iter = opts.max_iter;
        let df = self.grad(var);

        // --- Newton phase ---
        let mut xc = x0;
        for _ in 0..max_iter {
            let fx = eval_at(self, var, bindings, xc);
            if fx.abs() < tol {
                return Ok(xc);
            }
            let dfx = eval_at(&df, var, bindings, xc);
            if dfx == 0.0 || !dfx.is_finite() || !fx.is_finite() {
                break;
            }
            let step = fx / dfx;
            let xn = xc - step;
            if !xn.is_finite() {
                break;
            }
            if (xn - xc).abs() < tol && fx.abs() < tol.sqrt() {
                return Ok(xn);
            }
            xc = xn;
        }

        // --- Bracket search ---
        // Try widening window from x0, probing both sides
        let mut bracket: Option<(f64, f64)> = None;
        let f0 = eval_at(self, var, bindings, x0);
        if f0 == 0.0 {
            return Ok(x0);
        }

        for step in 1..=60_usize {
            let w = 0.5 * (step as f64);
            let p = x0 - w;
            let q = x0 + w;
            let fp = eval_at(self, var, bindings, p);
            let fq = eval_at(self, var, bindings, q);

            if fp == 0.0 {
                return Ok(p);
            }
            if fq == 0.0 {
                return Ok(q);
            }

            if fp.is_finite() && fq.is_finite() && fp * fq < 0.0 {
                bracket = Some((p, q));
                break;
            }
            // Also check if f0 and either probe form a bracket
            if f0.is_finite() && fp.is_finite() && f0 * fp < 0.0 {
                bracket = Some((x0, p));
                break;
            }
            if f0.is_finite() && fq.is_finite() && f0 * fq < 0.0 {
                bracket = Some((x0, q));
                break;
            }
        }

        match bracket {
            Some((p, q)) => brent_method(self, var, bindings, p, q, tol),
            None => Err(EmlError::NonConvergence {
                method: "find_root",
                iterations: max_iter,
            }),
        }
    }

    /// Find a root of `self` with respect to variable `var`, starting from `x0`.
    ///
    /// Uses default [`RootOpts`].
    pub fn find_root(&self, var: usize, bindings: &EvalCtx, x0: f64) -> Result<f64, EmlError> {
        self.find_root_opts(var, bindings, x0, RootOpts::default())
    }

    /// Find all roots in the interval `[a, b]` by sampling `n_samples` sub-intervals
    /// and applying Brent's method on each sign-changing pair.
    ///
    /// Returns a sorted, deduplicated `Vec<f64>`.
    pub fn find_roots_in(
        &self,
        var: usize,
        bindings: &EvalCtx,
        a: f64,
        b: f64,
        n_samples: usize,
    ) -> Result<Vec<f64>, EmlError> {
        if n_samples == 0 {
            return Err(EmlError::InvalidParameter("n_samples must be > 0"));
        }
        let dedup_tol = 1e-10_f64;
        let n = n_samples + 1;
        let mut roots: Vec<f64> = Vec::new();

        // Sample points
        let points: Vec<f64> = (0..n)
            .map(|i| a + (b - a) * (i as f64) / (n_samples as f64))
            .collect();
        let fvals: Vec<f64> = points
            .iter()
            .map(|&x| eval_at(self, var, bindings, x))
            .collect();

        for i in 0..(n - 1) {
            let xi = points[i];
            let xi1 = points[i + 1];
            let fi = fvals[i];
            let fi1 = fvals[i + 1];

            if fi == 0.0 {
                roots.push(xi);
            } else if fi.is_finite() && fi1.is_finite() && fi * fi1 < 0.0 {
                // Sign change: use Brent on this sub-interval
                match brent_method(self, var, bindings, xi, xi1, 1e-12) {
                    Ok(root) => roots.push(root),
                    Err(_) => {
                        // Best effort: add midpoint
                        roots.push((xi + xi1) / 2.0);
                    }
                }
            }
        }
        // Check the last point
        if let Some(&last_f) = fvals.last() {
            if last_f == 0.0 {
                if let Some(&last_x) = points.last() {
                    roots.push(last_x);
                }
            }
        }

        // Sort and deduplicate within dedup_tol
        roots.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        roots.dedup_by(|a_r, b_r| (*a_r - *b_r).abs() <= dedup_tol);

        Ok(roots)
    }

    /// Compute the definite integral of `self` with respect to variable `var`
    /// from `a` to `b` using adaptive Simpson quadrature with the provided options.
    pub fn quadrature_opts(
        &self,
        var: usize,
        bindings: &EvalCtx,
        a: f64,
        b: f64,
        opts: QuadOpts,
    ) -> Result<f64, EmlError> {
        if a == b {
            return Ok(0.0);
        }

        let (lo, hi, sign) = if a > b {
            (b, a, -1.0_f64)
        } else {
            (a, b, 1.0_f64)
        };

        let ctx = SimpsonCtx {
            expr: self,
            var,
            bindings,
            min_depth: opts.min_depth,
            max_depth: opts.max_depth,
        };

        let fa = ctx.eval_f(lo)?;
        let m = (lo + hi) / 2.0;
        let fm = ctx.eval_f(m)?;
        let fb = ctx.eval_f(hi)?;
        let whole = (hi - lo) / 6.0 * (fa + 4.0 * fm + fb);

        let iv = SimpsonInterval {
            a: lo,
            m,
            b: hi,
            fa,
            fm,
            fb,
            whole,
        };
        let result = adaptive_simpson_recurse(&ctx, iv, opts.tol, 0)?;

        Ok(sign * result)
    }

    /// Compute the definite integral of `self` with respect to variable `var`
    /// from `a` to `b` using adaptive Simpson quadrature with default options.
    pub fn quadrature(
        &self,
        var: usize,
        bindings: &EvalCtx,
        a: f64,
        b: f64,
    ) -> Result<f64, EmlError> {
        self.quadrature_opts(var, bindings, a, b, QuadOpts::default())
    }
}

/// Immutable evaluation context bundled for the adaptive Simpson recursion.
struct SimpsonCtx<'a> {
    expr: &'a LoweredOp,
    var: usize,
    bindings: &'a EvalCtx,
    min_depth: usize,
    max_depth: usize,
}

impl SimpsonCtx<'_> {
    fn eval_f(&self, x: f64) -> Result<f64, EmlError> {
        let v = eval_at(self.expr, self.var, self.bindings, x);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(EmlError::UndefinedAtPoint(x))
        }
    }
}

/// Mutable per-interval state threaded through the adaptive Simpson recursion.
struct SimpsonInterval {
    a: f64,
    m: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: f64,
}

/// Recursive adaptive Simpson quadrature.
///
/// Carries pre-computed function values to avoid redundant evaluations.
fn adaptive_simpson_recurse(
    ctx: &SimpsonCtx<'_>,
    iv: SimpsonInterval,
    tol: f64,
    depth: usize,
) -> Result<f64, EmlError> {
    let SimpsonInterval {
        a,
        m,
        b,
        fa,
        fm,
        fb,
        whole,
    } = iv;
    let lm = (a + m) / 2.0;
    let rm = (m + b) / 2.0;
    let flm = ctx.eval_f(lm)?;
    let frm = ctx.eval_f(rm)?;

    let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);

    if depth >= ctx.max_depth {
        return Ok(left + right);
    }

    if depth >= ctx.min_depth && (left + right - whole).abs() <= 15.0 * tol {
        return Ok(left + right + (left + right - whole) / 15.0);
    }

    let half_tol = tol / 2.0;
    let left_half = adaptive_simpson_recurse(
        ctx,
        SimpsonInterval {
            a,
            m: lm,
            b: m,
            fa,
            fm: flm,
            fb: fm,
            whole: left,
        },
        half_tol,
        depth + 1,
    )?;
    let right_half = adaptive_simpson_recurse(
        ctx,
        SimpsonInterval {
            a: m,
            m: rm,
            b,
            fa: fm,
            fm: frm,
            fb,
            whole: right,
        },
        half_tol,
        depth + 1,
    )?;

    Ok(left_half + right_half)
}

/// Lambert W₀ function (principal branch): solves W·eᵂ = x for x ≥ -1/e.
///
/// Uses Halley's method with a carefully chosen initial guess.
/// Returns `Err(EmlError::OutOfDomain)` for x < -1/e - tolerance.
/// Returns `Err(EmlError::NonConvergence)` if iteration does not converge.
pub fn lambert_w0(x: f64) -> Result<f64, EmlError> {
    const NEG_INV_E: f64 = -0.367_879_441_171_442_32; // -1/e
    const TOL: f64 = 1e-12;
    const MAX_ITER: usize = 50;

    if x < NEG_INV_E - TOL {
        return Err(EmlError::OutOfDomain);
    }

    // Initial guess: choose a region-aware starting point.
    let mut w = if (x - NEG_INV_E).abs() < 1e-6 {
        // Near branch point: use the Puiseux-series approximation.
        let t = (2.0 * (x * std::f64::consts::E + 1.0)).max(0.0).sqrt();
        -1.0 + t
    } else if x < 0.0 {
        // Negative side (-1/e, 0): series x - x² + 3/2 x³ is a safe start.
        let x2 = x * x;
        x - x2 + 1.5 * x2 * x
    } else if x < 1.0 {
        // (0, 1): ln(1+x) is a stable, monotone underestimate.
        (1.0 + x).ln()
    } else if x < std::f64::consts::E {
        // [1, e): interpolate between W(1)≈0.567 and W(e)=1.
        let frac = (x - 1.0) / (std::f64::consts::E - 1.0);
        0.567_143_29 + frac * (1.0 - 0.567_143_29)
    } else {
        // x ≥ e: asymptotic approximation ln(x) - ln(ln(x)).
        let lx = x.ln();
        let llx = lx.ln();
        lx - llx
    };

    for _ in 0..MAX_ITER {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        if f.abs() < TOL {
            return Ok(w);
        }
        let fp = ew * (w + 1.0);
        let fpp = ew * (w + 2.0);
        // Halley's method: step = 2·f·f′ / (2·f′² − f·f′′)
        let denom = 2.0 * fp * fp - f * fpp;
        if denom.abs() < 1e-30 {
            break;
        }
        let step = 2.0 * f * fp / denom;
        w -= step;
        if step.abs() < TOL {
            return Ok(w);
        }
    }

    let ew = w.exp();
    if (w * ew - x).abs() < 1e-8 {
        Ok(w)
    } else {
        Err(EmlError::NonConvergence {
            method: "lambert_w0",
            iterations: MAX_ITER,
        })
    }
}

/// Lambert W₋₁ function (secondary branch): solves W·eᵂ = x for -1/e ≤ x < 0.
///
/// Returns `Err(EmlError::OutOfDomain)` for x ≥ 0 or x < -1/e.
/// Returns `Err(EmlError::NonConvergence)` if iteration does not converge.
pub fn lambert_wm1(x: f64) -> Result<f64, EmlError> {
    const NEG_INV_E: f64 = -0.367_879_441_171_442_32;
    const TOL: f64 = 1e-12;
    const MAX_ITER: usize = 50;

    if !(NEG_INV_E - TOL..0.0).contains(&x) {
        return Err(EmlError::OutOfDomain);
    }

    let lnx = (-x).ln();
    let mut w = if (-x).ln().is_finite() && (-lnx).ln().is_finite() {
        lnx - (-lnx).ln()
    } else {
        -2.0
    };

    for _ in 0..MAX_ITER {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        if f.abs() < TOL {
            return Ok(w);
        }
        let fp = ew * (w + 1.0);
        let fpp = ew * (w + 2.0);
        // Halley's method: step = 2·f·f′ / (2·f′² − f·f′′)
        let denom = 2.0 * fp * fp - f * fpp;
        if denom.abs() < 1e-30 {
            break;
        }
        let step = 2.0 * f * fp / denom;
        w -= step;
        if step.abs() < TOL {
            return Ok(w);
        }
    }

    let ew = w.exp();
    if (w * ew - x).abs() < 1e-8 {
        Ok(w)
    } else {
        Err(EmlError::NonConvergence {
            method: "lambert_wm1",
            iterations: MAX_ITER,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn const_op(c: f64) -> LoweredOp {
        LoweredOp::Const(c)
    }

    fn var_op(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    fn bindings(vals: &[f64]) -> EvalCtx {
        EvalCtx::new(vals)
    }

    // x^2 - 2 as a LoweredOp
    fn x_sq_minus_2() -> LoweredOp {
        LoweredOp::Sub(
            Arc::new(LoweredOp::Mul(Arc::new(var_op(0)), Arc::new(var_op(0)))),
            Arc::new(const_op(2.0)),
        )
    }

    // cos(x) - x
    fn cos_x_minus_x() -> LoweredOp {
        LoweredOp::Sub(
            Arc::new(LoweredOp::Cos(Arc::new(var_op(0)))),
            Arc::new(var_op(0)),
        )
    }

    // exp(x) - 2
    fn exp_x_minus_2() -> LoweredOp {
        LoweredOp::Sub(
            Arc::new(LoweredOp::Exp(Arc::new(var_op(0)))),
            Arc::new(const_op(2.0)),
        )
    }

    // cos(x)
    fn cos_x() -> LoweredOp {
        LoweredOp::Cos(Arc::new(var_op(0)))
    }

    // sin(x)
    fn sin_x() -> LoweredOp {
        LoweredOp::Sin(Arc::new(var_op(0)))
    }

    // x^2
    fn x_sq() -> LoweredOp {
        LoweredOp::Mul(Arc::new(var_op(0)), Arc::new(var_op(0)))
    }

    // 1/(1+x^2)
    fn one_over_one_plus_x_sq() -> LoweredOp {
        LoweredOp::Div(
            Arc::new(const_op(1.0)),
            Arc::new(LoweredOp::Add(
                Arc::new(const_op(1.0)),
                Arc::new(LoweredOp::Mul(Arc::new(var_op(0)), Arc::new(var_op(0)))),
            )),
        )
    }

    // x (for testing reversed bounds)
    fn x_op() -> LoweredOp {
        var_op(0)
    }

    // x^3 (for FTC test)
    fn x_cubed() -> LoweredOp {
        LoweredOp::Mul(
            Arc::new(LoweredOp::Mul(Arc::new(var_op(0)), Arc::new(var_op(0)))),
            Arc::new(var_op(0)),
        )
    }

    #[test]
    fn test_find_root_x_sq_minus_2_pos() {
        let expr = x_sq_minus_2();
        let ctx = bindings(&[]);
        let root = expr.find_root(0, &ctx, 1.0).expect("should converge");
        assert!(
            (root - 2.0_f64.sqrt()).abs() < 1e-10,
            "expected sqrt(2) ≈ {}, got {}",
            2.0_f64.sqrt(),
            root
        );
    }

    #[test]
    fn test_find_root_x_sq_minus_2_neg() {
        let expr = x_sq_minus_2();
        let ctx = bindings(&[]);
        let root = expr.find_root(0, &ctx, -1.0).expect("should converge");
        assert!(
            (root + 2.0_f64.sqrt()).abs() < 1e-10,
            "expected -sqrt(2) ≈ {}, got {}",
            -2.0_f64.sqrt(),
            root
        );
    }

    #[test]
    fn test_find_root_cos_x_minus_x() {
        // Fixed point of cos, approximately 0.73909
        let expr = cos_x_minus_x();
        let ctx = bindings(&[]);
        let root = expr.find_root(0, &ctx, 1.0).expect("should converge");
        assert!(
            (root - 0.739_085_133).abs() < 1e-6,
            "expected ~0.73909, got {}",
            root
        );
    }

    #[test]
    fn test_find_root_exp_x_minus_2() {
        // ln(2) ≈ 0.69315
        let expr = exp_x_minus_2();
        let ctx = bindings(&[]);
        let root = expr.find_root(0, &ctx, 0.0).expect("should converge");
        assert!(
            (root - 2.0_f64.ln()).abs() < 1e-10,
            "expected ln(2) ≈ {}, got {}",
            2.0_f64.ln(),
            root
        );
    }

    #[test]
    fn test_find_root_cos_brent_fallback() {
        // cos(x) has zero derivative at x=0, so Newton stalls.
        // Brent fallback must still find *a* root: cos(x)=0 at ±π/2+kπ.
        // Starting from x0=0 the algorithm may reach either -π/2 or +π/2 —
        // we only check that the residual is below tolerance.
        let expr = cos_x();
        let ctx = bindings(&[]);
        let root = expr
            .find_root(0, &ctx, 0.0)
            .expect("should converge via Brent");
        let residual = eval_at(&expr, 0, &ctx, root).abs();
        assert!(
            residual < 1e-8,
            "expected cos(root) ≈ 0, got cos({root}) = {residual}"
        );
    }

    #[test]
    fn test_find_roots_in_sin() {
        // sin(x) roots in [0, 3.2]: near 0 and near π
        let expr = sin_x();
        let ctx = bindings(&[]);
        let roots = expr
            .find_roots_in(0, &ctx, 0.0, 3.2, 100)
            .expect("should succeed");
        // Expect at least 2 roots
        assert!(
            roots.len() >= 2,
            "expected at least 2 roots, got {:?}",
            roots
        );
        // First root near 0
        assert!(
            roots[0].abs() < 1e-6,
            "first root should be near 0, got {}",
            roots[0]
        );
        // Second root near π
        let pi_approx = std::f64::consts::PI;
        let near_pi = roots.iter().any(|&r| (r - pi_approx).abs() < 1e-6);
        assert!(near_pi, "expected a root near π, roots = {:?}", roots);
    }

    #[test]
    fn test_find_roots_in_zero_samples() {
        let expr = sin_x();
        let ctx = bindings(&[]);
        let result = expr.find_roots_in(0, &ctx, 0.0, 3.2, 0);
        assert!(
            matches!(result, Err(EmlError::InvalidParameter(_))),
            "expected InvalidParameter, got {:?}",
            result
        );
    }

    #[test]
    fn test_quadrature_sin_0_pi() {
        // ∫₀^π sin(x) dx = 2.0
        let expr = sin_x();
        let ctx = bindings(&[]);
        let result = expr
            .quadrature(0, &ctx, 0.0, std::f64::consts::PI)
            .expect("should succeed");
        assert!((result - 2.0).abs() < 1e-8, "expected 2.0, got {}", result);
    }

    #[test]
    fn test_quadrature_x_sq_0_1() {
        // ∫₀¹ x² dx = 1/3
        let expr = x_sq();
        let ctx = bindings(&[]);
        let result = expr.quadrature(0, &ctx, 0.0, 1.0).expect("should succeed");
        assert!(
            (result - 1.0 / 3.0).abs() < 1e-9,
            "expected 1/3, got {}",
            result
        );
    }

    #[test]
    fn test_quadrature_arctan_integral() {
        // ∫₋₁¹ 1/(1+x²) dx = π/2
        let expr = one_over_one_plus_x_sq();
        let ctx = bindings(&[]);
        let result = expr.quadrature(0, &ctx, -1.0, 1.0).expect("should succeed");
        let pi_half = std::f64::consts::PI / 2.0;
        assert!(
            (result - pi_half).abs() < 1e-8,
            "expected π/2 ≈ {pi_half}, got {result}"
        );
    }

    #[test]
    fn test_quadrature_reversed_bounds() {
        // ∫₁₀ x dx = -∫₀₁ x dx = -0.5
        let expr = x_op();
        let ctx = bindings(&[]);
        let forward = expr.quadrature(0, &ctx, 0.0, 1.0).expect("should succeed");
        let backward = expr.quadrature(0, &ctx, 1.0, 0.0).expect("should succeed");
        assert!(
            (forward + backward).abs() < 1e-12,
            "forward={forward}, backward={backward}, sum should be 0"
        );
        assert!(
            (backward + 0.5).abs() < 1e-12,
            "∫₁₀ x dx should be -0.5, got {backward}"
        );
    }

    #[test]
    fn test_quadrature_same_bounds() {
        let expr = x_sq();
        let ctx = bindings(&[]);
        let result = expr.quadrature(0, &ctx, 1.0, 1.0).expect("should succeed");
        assert_eq!(result, 0.0, "same bounds should give 0");
    }

    #[test]
    fn test_ftc_property() {
        // FTC: ∫₀¹ f'(x) dx = f(1) - f(0) for f(x) = x³
        // f'(x) = 3x²
        let f_expr = x_cubed();
        let df_expr = f_expr.grad(0);
        let ctx = bindings(&[]);

        let integral = df_expr
            .quadrature(0, &ctx, 0.0, 1.0)
            .expect("should succeed");
        let f1 = eval_at(&f_expr, 0, &ctx, 1.0);
        let f0 = eval_at(&f_expr, 0, &ctx, 0.0);
        let expected = f1 - f0; // = 1 - 0 = 1

        assert!(
            (integral - expected).abs() < 1e-9,
            "FTC: ∫₀¹ 3x² dx = {integral}, expected f(1)-f(0) = {expected}"
        );
    }
}
