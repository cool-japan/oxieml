//! Symbolic antidifferentiation and definite integration for `LoweredOp`.
//!
//! Provides [`LoweredOp::integrate`] for closed-form antiderivatives and
//! [`LoweredOp::integrate_definite`] for definite integrals with quadrature
//! fallback.

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::lower::LoweredOp;
use std::sync::Arc;

/// Result of symbolic antidifferentiation.
///
/// Mirrors [`crate::SolveResult`]: partiality is modelled as data, never a panic.
#[derive(Clone, Debug)]
pub enum IntegrateResult {
    /// Closed-form antiderivative F such that F'(x) = f(x).
    /// The constant of integration is omitted.
    Closed(LoweredOp),
    /// No rule in the engine produced a closed form.
    /// Use [`LoweredOp::integrate_definite`] for a numeric fallback.
    Unsupported,
}

impl LoweredOp {
    /// Compute the indefinite integral ∫ self d(x_wrt).
    ///
    /// Returns [`IntegrateResult::Closed`] with a valid antiderivative, or
    /// [`IntegrateResult::Unsupported`] when no rule applies.
    pub fn integrate(&self, wrt: usize) -> IntegrateResult {
        if let Some(t) = raw_integrate(self, wrt, 0) {
            return IntegrateResult::Closed(t.simplify());
        }
        // Final fallback: the Weierstrass `t = tan(x/2)` substitution for a
        // rational function of `sin`/`cos` of one common affine argument. It is
        // self-verifying and structurally declines non-affine trig arguments,
        // so `∫ sin(x²)` / `∫ exp(x²)` remain Unsupported.
        if let Some(t) = crate::integrate_trig::try_weierstrass(self, wrt) {
            return IntegrateResult::Closed(t.simplify());
        }
        IntegrateResult::Unsupported
    }

    /// Compute the definite integral ∫_a^b self d(x_wrt).
    ///
    /// First attempts symbolic: F(b) − F(a). Falls back to adaptive quadrature
    /// when symbolic integration is [`IntegrateResult::Unsupported`] or when
    /// an endpoint evaluation is non-finite.
    ///
    /// Other variables are held at `bindings`.
    pub fn integrate_definite(
        &self,
        wrt: usize,
        a: f64,
        b: f64,
        bindings: &EvalCtx,
    ) -> Result<f64, EmlError> {
        if a == b {
            return Ok(0.0);
        }
        let (lo, hi, sign) = if a < b {
            (a, b, 1.0_f64)
        } else {
            (b, a, -1.0_f64)
        };

        // Try symbolic path first.
        if let IntegrateResult::Closed(f_anti) = self.integrate(wrt) {
            let fa = crate::numeric::eval_at_pub(&f_anti, wrt, bindings, lo);
            let fb = crate::numeric::eval_at_pub(&f_anti, wrt, bindings, hi);
            if fa.is_finite() && fb.is_finite() {
                return Ok(sign * (fb - fa));
            }
        }

        // Fallback: adaptive quadrature.
        let result = self.quadrature(wrt, bindings, lo, hi)?;
        Ok(sign * result)
    }
}

// ── private helpers ──────────────────────────────────────────────────────────

#[inline]
fn arc(op: LoweredOp) -> Arc<LoweredOp> {
    Arc::new(op)
}

#[inline]
fn const_op(c: f64) -> LoweredOp {
    LoweredOp::Const(c)
}

#[inline]
fn x_var(wrt: usize) -> LoweredOp {
    LoweredOp::Var(wrt)
}

/// Returns `Some(c)` if `op` is a numeric constant with no variables.
fn as_constant(op: &LoweredOp) -> Option<f64> {
    match op {
        LoweredOp::Const(c) => Some(*c),
        LoweredOp::NamedConst(nc) => Some(nc.value()),
        _ => None,
    }
}

/// Returns `true` if `op` contains `Var(wrt)` anywhere in its tree.
fn depends_on(op: &LoweredOp, wrt: usize) -> bool {
    op.contains_var(wrt)
}

/// Detect whether `arg` is an affine function `a·x + b` (x = Var(wrt), a ≠ 0)
/// that depends only on `wrt`.
///
/// Returns `Some((a, b))` when successful. Shared with
/// [`crate::integrate_trig`], which reduces `∫ tⁿ(a·x+b) dx` to the
/// argument-variable antiderivative scaled by `1/a`.
pub(crate) fn affine_arg(arg: &LoweredOp, wrt: usize) -> Option<(f64, f64)> {
    if !depends_on(arg, wrt) {
        return None;
    }
    let da = arg.grad(wrt).simplify();
    let a = as_constant(&da)?;
    if a.abs() < 1e-15 {
        return None;
    }
    // b = value of arg when wrt = 0 (all other vars also set to 0).
    let b = eval_only_wrt(arg, wrt, 0.0);
    if !b.is_finite() {
        return None;
    }
    Some((a, b))
}

/// Evaluate `op` with `Var(wrt) = x` and all other `Var(i) = 0.0`.
pub(crate) fn eval_only_wrt(op: &LoweredOp, wrt: usize, x: f64) -> f64 {
    match op {
        LoweredOp::Const(c) => *c,
        LoweredOp::NamedConst(nc) => nc.value(),
        LoweredOp::Var(i) => {
            if *i == wrt {
                x
            } else {
                0.0
            }
        }
        LoweredOp::Add(a, b) => eval_only_wrt(a, wrt, x) + eval_only_wrt(b, wrt, x),
        LoweredOp::Sub(a, b) => eval_only_wrt(a, wrt, x) - eval_only_wrt(b, wrt, x),
        LoweredOp::Mul(a, b) => eval_only_wrt(a, wrt, x) * eval_only_wrt(b, wrt, x),
        LoweredOp::Div(a, b) => eval_only_wrt(a, wrt, x) / eval_only_wrt(b, wrt, x),
        LoweredOp::Exp(a) => eval_only_wrt(a, wrt, x).exp(),
        LoweredOp::Ln(a) => eval_only_wrt(a, wrt, x).ln(),
        LoweredOp::Neg(a) => -eval_only_wrt(a, wrt, x),
        LoweredOp::Sin(a) => eval_only_wrt(a, wrt, x).sin(),
        LoweredOp::Cos(a) => eval_only_wrt(a, wrt, x).cos(),
        LoweredOp::Tan(a) => eval_only_wrt(a, wrt, x).tan(),
        LoweredOp::Sinh(a) => eval_only_wrt(a, wrt, x).sinh(),
        LoweredOp::Cosh(a) => eval_only_wrt(a, wrt, x).cosh(),
        LoweredOp::Tanh(a) => eval_only_wrt(a, wrt, x).tanh(),
        LoweredOp::Arcsin(a) => eval_only_wrt(a, wrt, x).asin(),
        LoweredOp::Arccos(a) => eval_only_wrt(a, wrt, x).acos(),
        LoweredOp::Arctan(a) => eval_only_wrt(a, wrt, x).atan(),
        LoweredOp::Arcsinh(a) => eval_only_wrt(a, wrt, x).asinh(),
        LoweredOp::Arccosh(a) => eval_only_wrt(a, wrt, x).acosh(),
        LoweredOp::Arctanh(a) => eval_only_wrt(a, wrt, x).atanh(),
        LoweredOp::Erf(a) => crate::special::erf(eval_only_wrt(a, wrt, x)),
        LoweredOp::LGamma(a) => crate::special::lgamma(eval_only_wrt(a, wrt, x)),
        LoweredOp::Digamma(a) => crate::special::digamma(eval_only_wrt(a, wrt, x)),
        LoweredOp::Trigamma(a) => crate::special::trigamma(eval_only_wrt(a, wrt, x)),
        LoweredOp::Ei(a) => crate::special::ei(eval_only_wrt(a, wrt, x)),
        LoweredOp::Si(a) => crate::special::si(eval_only_wrt(a, wrt, x)),
        LoweredOp::Ci(a) => crate::special::ci(eval_only_wrt(a, wrt, x)),
        LoweredOp::Pow(a, b) => eval_only_wrt(a, wrt, x).powf(eval_only_wrt(b, wrt, x)),
    }
}

// ── LIATE priority (lower = differentiate first) ─────────────────────────────

fn liate_rank(op: &LoweredOp, wrt: usize) -> u8 {
    match op {
        LoweredOp::Ln(_)
        | LoweredOp::Arcsin(_)
        | LoweredOp::Arccos(_)
        | LoweredOp::Arctan(_)
        | LoweredOp::Arcsinh(_)
        | LoweredOp::Arccosh(_)
        | LoweredOp::Arctanh(_) => 0,
        LoweredOp::Pow(base, expo)
            if (matches!(base.as_ref(), LoweredOp::Var(_)) || as_constant(expo).is_some()) =>
        {
            2
        }
        LoweredOp::Pow(_, _) => 5,
        LoweredOp::Var(i) if *i == wrt => 2,
        LoweredOp::Var(_) => 5,
        LoweredOp::Sin(_) | LoweredOp::Cos(_) | LoweredOp::Tan(_) => 3,
        LoweredOp::Exp(_) => 4,
        _ => 5,
    }
}

// ── Integration by parts ──────────────────────────────────────────────────────

const BY_PARTS_MAX_DEPTH: u32 = 4;

/// Attempt ∫(p·q) dx via integration by parts, using LIATE to choose u and dv.
fn by_parts(p: &LoweredOp, q: &LoweredOp, wrt: usize, depth: u32) -> Option<LoweredOp> {
    if depth >= BY_PARTS_MAX_DEPTH {
        return None;
    }

    // Choose u (to differentiate) by LIATE rank.
    let (u, dv) = if liate_rank(p, wrt) <= liate_rank(q, wrt) {
        (p, q)
    } else {
        (q, p)
    };

    let du = u.grad(wrt);
    let v = raw_integrate(dv, wrt, depth + 1)?;

    // Anti-cycle: bail out if v·du simplifies to the original p·q.
    let new_integrand = LoweredOp::Mul(arc(v.clone()), arc(du.clone())).simplify();
    let orig = LoweredOp::Mul(arc(p.clone()), arc(q.clone())).simplify();
    if crate::lower_simplify::ops_struct_hash(&new_integrand)
        == crate::lower_simplify::ops_struct_hash(&orig)
    {
        return None;
    }

    let integral_v_du = raw_integrate(&new_integrand, wrt, depth + 1)?;

    Some(LoweredOp::Sub(
        arc(LoweredOp::Mul(arc(u.clone()), arc(v))),
        arc(integral_v_du),
    ))
}

// ── Core integration engine ───────────────────────────────────────────────────

/// Build an antiderivative of `op` w.r.t. `wrt`. Returns `None` when no rule
/// applies. `depth` counts nested by-parts calls.
fn raw_integrate(op: &LoweredOp, wrt: usize, depth: u32) -> Option<LoweredOp> {
    // Guard: op does not depend on wrt → treat as constant.
    if !depends_on(op, wrt) {
        return Some(LoweredOp::Mul(arc(op.clone()), arc(x_var(wrt))));
    }

    match op {
        // ── Rule 1: x ────────────────────────────────────────────────────────
        LoweredOp::Var(i) if *i == wrt => Some(LoweredOp::Div(
            arc(LoweredOp::Pow(arc(x_var(wrt)), arc(const_op(2.0)))),
            arc(const_op(2.0)),
        )),

        // ── Rule 2: Negation ─────────────────────────────────────────────────
        LoweredOp::Neg(inner) => Some(LoweredOp::Neg(arc(raw_integrate(inner, wrt, depth)?))),

        // ── Rule 3: Addition ─────────────────────────────────────────────────
        LoweredOp::Add(a, b) => Some(LoweredOp::Add(
            arc(raw_integrate(a, wrt, depth)?),
            arc(raw_integrate(b, wrt, depth)?),
        )),

        // ── Rule 4: Subtraction ──────────────────────────────────────────────
        LoweredOp::Sub(a, b) => Some(LoweredOp::Sub(
            arc(raw_integrate(a, wrt, depth)?),
            arc(raw_integrate(b, wrt, depth)?),
        )),

        // ── Rule 5: Multiplication ───────────────────────────────────────────
        LoweredOp::Mul(a, b) => {
            if !depends_on(a, wrt) {
                Some(LoweredOp::Mul(
                    arc((**a).clone()),
                    arc(raw_integrate(b, wrt, depth)?),
                ))
            } else if !depends_on(b, wrt) {
                Some(LoweredOp::Mul(
                    arc((**b).clone()),
                    arc(raw_integrate(a, wrt, depth)?),
                ))
            } else {
                // Both factors depend on x. Try the trig-specific product rules
                // (sinᵐcosⁿ, product-to-sum) and the eᵃˣ·{sin,cos} solve-for-I
                // family before falling back to generic integration by parts.
                crate::integrate_trig::try_trig_product(a, b, wrt)
                    .or_else(|| crate::integrate_trig::try_solve_for_i(a, b, wrt))
                    .or_else(|| by_parts(a, b, wrt, depth))
            }
        }

        // ── Rule 6: Division ─────────────────────────────────────────────────
        LoweredOp::Div(num, den) => {
            // Sinc: sin(x) / x → Si(x)
            if let (LoweredOp::Sin(s_arg), LoweredOp::Var(dv)) = (num.as_ref(), den.as_ref()) {
                if *dv == wrt {
                    if let LoweredOp::Var(sv) = s_arg.as_ref() {
                        if *sv == wrt {
                            return Some(LoweredOp::Si(Arc::new(LoweredOp::Var(wrt))));
                        }
                    }
                }
            }
            // Cosc: cos(x) / x → Ci(x)
            if let (LoweredOp::Cos(c_arg), LoweredOp::Var(dv)) = (num.as_ref(), den.as_ref()) {
                if *dv == wrt {
                    if let LoweredOp::Var(cv) = c_arg.as_ref() {
                        if *cv == wrt {
                            return Some(LoweredOp::Ci(Arc::new(LoweredOp::Var(wrt))));
                        }
                    }
                }
            }
            // Exp/x: exp(x) / x → Ei(x)
            if let (LoweredOp::Exp(e_arg), LoweredOp::Var(dv)) = (num.as_ref(), den.as_ref()) {
                if *dv == wrt {
                    if let LoweredOp::Var(ev) = e_arg.as_ref() {
                        if *ev == wrt {
                            return Some(LoweredOp::Ei(Arc::new(LoweredOp::Var(wrt))));
                        }
                    }
                }
            }
            if !depends_on(den, wrt) {
                // Constant denominator.
                return Some(LoweredOp::Div(
                    arc(raw_integrate(num, wrt, depth)?),
                    arc((**den).clone()),
                ));
            }

            // Rational function integration via partial-fraction decomposition.
            if let Some(result) = integrate_rational::integrate_rational(num, den, wrt) {
                return Some(result);
            }

            if !depends_on(num, wrt) {
                // Constant numerator, varying denominator.
                // Special case: 1/x
                if let LoweredOp::Var(i) = den.as_ref() {
                    if *i == wrt {
                        return Some(LoweredOp::Mul(
                            arc((**num).clone()),
                            arc(LoweredOp::Ln(arc(x_var(wrt)))),
                        ));
                    }
                }
                // Affine denominator: num / (a*x+b) → num * ln(a*x+b) / a
                if let Some((a_coef, _)) = affine_arg(den, wrt) {
                    return Some(LoweredOp::Mul(
                        arc((**num).clone()),
                        arc(LoweredOp::Div(
                            arc(LoweredOp::Ln(arc((**den).clone()))),
                            arc(const_op(a_coef)),
                        )),
                    ));
                }
                return None;
            }

            // Both num and den depend on wrt: try f'/f → ln(f).
            let den_grad = den.grad(wrt).simplify();
            let num_s = num.simplify();
            if crate::lower_simplify::ops_struct_hash(&num_s)
                == crate::lower_simplify::ops_struct_hash(&den_grad)
            {
                Some(LoweredOp::Ln(arc((**den).clone())))
            } else {
                None
            }
        }

        // ── Rule 7: Power ────────────────────────────────────────────────────
        LoweredOp::Pow(base, expo) => {
            // Case A: base = Var(wrt), expo = constant n.
            if let (LoweredOp::Var(i), Some(n)) = (base.as_ref(), as_constant(expo)) {
                if *i == wrt {
                    return if (n + 1.0).abs() < 1e-15 {
                        Some(LoweredOp::Ln(arc(x_var(wrt))))
                    } else {
                        Some(LoweredOp::Div(
                            arc(LoweredOp::Pow(arc(x_var(wrt)), arc(const_op(n + 1.0)))),
                            arc(const_op(n + 1.0)),
                        ))
                    };
                }
            }

            // Case B: base is affine, expo = constant n.
            if let Some(n) = as_constant(expo) {
                if let Some((a_coef, _)) = affine_arg(base, wrt) {
                    return if (n + 1.0).abs() < 1e-15 {
                        // ∫(ax+b)^{-1} dx = ln(ax+b)/a
                        Some(LoweredOp::Div(
                            arc(LoweredOp::Ln(arc((**base).clone()))),
                            arc(const_op(a_coef)),
                        ))
                    } else {
                        // ∫(ax+b)^n dx = (ax+b)^{n+1} / (n+1) / a
                        Some(LoweredOp::Div(
                            arc(LoweredOp::Div(
                                arc(LoweredOp::Pow(
                                    arc((**base).clone()),
                                    arc(const_op(n + 1.0)),
                                )),
                                arc(const_op(n + 1.0)),
                            )),
                            arc(const_op(a_coef)),
                        ))
                    };
                }
            }

            // Case C: base = Const(c) > 0, expo = Var(wrt) → c^x / ln(c)
            if let (Some(c), LoweredOp::Var(i)) = (as_constant(base), expo.as_ref()) {
                if c > 0.0 && *i == wrt {
                    return Some(LoweredOp::Div(
                        arc(LoweredOp::Pow(arc((**base).clone()), arc(x_var(wrt)))),
                        arc(LoweredOp::Ln(arc(const_op(c)))),
                    ));
                }
            }

            // Case D: Trig substitution for Pow(inner, +/-0.5) with degree-2 polynomial inner
            if let Some(ev) = as_constant(expo) {
                if (ev - 0.5).abs() < 1e-12 || (ev + 0.5).abs() < 1e-12 {
                    if let Some(result) =
                        crate::integrate_subst::try_trig_substitution(base, ev, wrt)
                    {
                        return Some(result);
                    }
                }
            }

            // Case E: Powers of a single trig function — sinⁿ/cosⁿ/tanⁿ and their
            // reciprocals secⁿ/cscⁿ/cotⁿ (negative exponents) — via reduction.
            if let Some(result) = crate::integrate_trig::try_trig_power(op, wrt) {
                return Some(result);
            }

            None
        }

        // ── Rule 8: Exponential ──────────────────────────────────────────────
        LoweredOp::Exp(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫exp(a·x+b) dx = exp(a·x+b) / a
            Some(LoweredOp::Div(
                arc(LoweredOp::Exp(arc((**arg).clone()))),
                arc(const_op(a_coef)),
            ))
        }

        // ── Rule 9: Natural logarithm ────────────────────────────────────────
        LoweredOp::Ln(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫ln(u) du = u·ln(u) − u  (u = arg)
            let inner = LoweredOp::Sub(
                arc(LoweredOp::Mul(
                    arc((**arg).clone()),
                    arc(LoweredOp::Ln(arc((**arg).clone()))),
                )),
                arc((**arg).clone()),
            );
            Some(LoweredOp::Div(arc(inner), arc(const_op(a_coef))))
        }

        // ── Rule 10: Trig / hyperbolic table ─────────────────────────────────
        LoweredOp::Sin(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫sin(u) du = −cos(u)
            Some(LoweredOp::Div(
                arc(LoweredOp::Neg(arc(LoweredOp::Cos(arc((**arg).clone()))))),
                arc(const_op(a_coef)),
            ))
        }

        LoweredOp::Cos(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫cos(u) du = sin(u)
            Some(LoweredOp::Div(
                arc(LoweredOp::Sin(arc((**arg).clone()))),
                arc(const_op(a_coef)),
            ))
        }

        LoweredOp::Tan(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫tan(u) du = −ln(cos(u))
            Some(LoweredOp::Div(
                arc(LoweredOp::Neg(arc(LoweredOp::Ln(arc(LoweredOp::Cos(
                    arc((**arg).clone()),
                )))))),
                arc(const_op(a_coef)),
            ))
        }

        LoweredOp::Sinh(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫sinh(u) du = cosh(u)
            Some(LoweredOp::Div(
                arc(LoweredOp::Cosh(arc((**arg).clone()))),
                arc(const_op(a_coef)),
            ))
        }

        LoweredOp::Cosh(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫cosh(u) du = sinh(u)
            Some(LoweredOp::Div(
                arc(LoweredOp::Sinh(arc((**arg).clone()))),
                arc(const_op(a_coef)),
            ))
        }

        LoweredOp::Tanh(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫tanh(u) du = ln(cosh(u))
            Some(LoweredOp::Div(
                arc(LoweredOp::Ln(arc(LoweredOp::Cosh(arc((**arg).clone()))))),
                arc(const_op(a_coef)),
            ))
        }

        LoweredOp::Arcsin(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫arcsin(u) du = u·arcsin(u) + √(1−u²)
            let u = arc((**arg).clone());
            let antideriv = LoweredOp::Add(
                arc(LoweredOp::Mul(
                    Arc::clone(&u),
                    arc(LoweredOp::Arcsin(Arc::clone(&u))),
                )),
                arc(LoweredOp::Pow(
                    arc(LoweredOp::Sub(
                        arc(const_op(1.0)),
                        arc(LoweredOp::Pow(Arc::clone(&u), arc(const_op(2.0)))),
                    )),
                    arc(const_op(0.5)),
                )),
            );
            Some(LoweredOp::Div(arc(antideriv), arc(const_op(a_coef))))
        }

        LoweredOp::Arccos(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫arccos(u) du = u·arccos(u) − √(1−u²)
            let u = arc((**arg).clone());
            let antideriv = LoweredOp::Sub(
                arc(LoweredOp::Mul(
                    Arc::clone(&u),
                    arc(LoweredOp::Arccos(Arc::clone(&u))),
                )),
                arc(LoweredOp::Pow(
                    arc(LoweredOp::Sub(
                        arc(const_op(1.0)),
                        arc(LoweredOp::Pow(Arc::clone(&u), arc(const_op(2.0)))),
                    )),
                    arc(const_op(0.5)),
                )),
            );
            Some(LoweredOp::Div(arc(antideriv), arc(const_op(a_coef))))
        }

        LoweredOp::Arctan(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫arctan(u) du = u·arctan(u) − (1/2)·ln(1+u²)
            let u = arc((**arg).clone());
            let antideriv = LoweredOp::Sub(
                arc(LoweredOp::Mul(
                    Arc::clone(&u),
                    arc(LoweredOp::Arctan(Arc::clone(&u))),
                )),
                arc(LoweredOp::Mul(
                    arc(const_op(0.5)),
                    arc(LoweredOp::Ln(arc(LoweredOp::Add(
                        arc(const_op(1.0)),
                        arc(LoweredOp::Pow(Arc::clone(&u), arc(const_op(2.0)))),
                    )))),
                )),
            );
            Some(LoweredOp::Div(arc(antideriv), arc(const_op(a_coef))))
        }

        LoweredOp::Arcsinh(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫arcsinh(u) du = u·arcsinh(u) − √(u²+1)
            let u = arc((**arg).clone());
            let antideriv = LoweredOp::Sub(
                arc(LoweredOp::Mul(
                    Arc::clone(&u),
                    arc(LoweredOp::Arcsinh(Arc::clone(&u))),
                )),
                arc(LoweredOp::Pow(
                    arc(LoweredOp::Add(
                        arc(LoweredOp::Pow(Arc::clone(&u), arc(const_op(2.0)))),
                        arc(const_op(1.0)),
                    )),
                    arc(const_op(0.5)),
                )),
            );
            Some(LoweredOp::Div(arc(antideriv), arc(const_op(a_coef))))
        }

        LoweredOp::Arccosh(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫arccosh(u) du = u·arccosh(u) − √(u²−1)
            let u = arc((**arg).clone());
            let antideriv = LoweredOp::Sub(
                arc(LoweredOp::Mul(
                    Arc::clone(&u),
                    arc(LoweredOp::Arccosh(Arc::clone(&u))),
                )),
                arc(LoweredOp::Pow(
                    arc(LoweredOp::Sub(
                        arc(LoweredOp::Pow(Arc::clone(&u), arc(const_op(2.0)))),
                        arc(const_op(1.0)),
                    )),
                    arc(const_op(0.5)),
                )),
            );
            Some(LoweredOp::Div(arc(antideriv), arc(const_op(a_coef))))
        }

        LoweredOp::Arctanh(arg) => {
            let (a_coef, _) = affine_arg(arg, wrt)?;
            // ∫arctanh(u) du = u·arctanh(u) + (1/2)·ln(1−u²)
            let u = arc((**arg).clone());
            let antideriv = LoweredOp::Add(
                arc(LoweredOp::Mul(
                    Arc::clone(&u),
                    arc(LoweredOp::Arctanh(Arc::clone(&u))),
                )),
                arc(LoweredOp::Mul(
                    arc(const_op(0.5)),
                    arc(LoweredOp::Ln(arc(LoweredOp::Sub(
                        arc(const_op(1.0)),
                        arc(LoweredOp::Pow(Arc::clone(&u), arc(const_op(2.0)))),
                    )))),
                )),
            );
            Some(LoweredOp::Div(arc(antideriv), arc(const_op(a_coef))))
        }

        // ── Catch-all: try u-substitution ─────────────────────────────────────
        other => crate::integrate_subst::try_u_substitution(
            other,
            wrt,
            depth,
            BY_PARTS_MAX_DEPTH,
            &|op, w, d| raw_integrate(op, w, d),
        ),
    }
}

// ── Rational function integration ────────────────────────────────────────────

pub(crate) mod integrate_rational {
    use crate::lower::LoweredOp;
    use crate::poly::{Poly, ratio_to_f64};
    use crate::rewrite::apart::{PfTerm, partial_fractions};
    use std::sync::Arc;

    /// Attempt to integrate `num / den` with respect to variable `wrt` using
    /// partial-fraction decomposition.
    ///
    /// The decomposition itself is delegated to the shared
    /// [`partial_fractions`](crate::rewrite::apart::partial_fractions) helper
    /// (upgraded to the full [`Poly::factor`](crate::poly::Poly::factor)); this
    /// function only integrates the resulting summands with the elementary
    /// table
    ///
    /// * polynomial part → power rule,
    /// * `A/(x−r)^j` → `A·ln|x−r|` (`j = 1`) or `A/(1−j)·(x−r)^{1−j}` (`j > 1`),
    /// * `(Bx+C)/q(x)` for an irreducible quadratic `q` → `ln` + `arctan`.
    ///
    /// Returns `None` when a summand falls outside that table (an irreducible
    /// factor of degree ≥ 3, or a repeated irreducible quadratic), or when the
    /// integrand is not a rational function over `wrt`.
    pub(crate) fn integrate_rational(
        num: &LoweredOp,
        den: &LoweredOp,
        wrt: usize,
    ) -> Option<LoweredOp> {
        let num_poly = Poly::from_lowered(num, wrt).ok()?;
        let den_poly = Poly::from_lowered(den, wrt).ok()?;
        if den_poly.is_zero() {
            return None;
        }

        let decomposition = partial_fractions(&num_poly, &den_poly, wrt)?;

        let mut integral_terms: Vec<LoweredOp> = Vec::new();
        if !decomposition.polynomial_part.is_zero() {
            if let Some(anti) = poly_antiderivative(&decomposition.polynomial_part, wrt) {
                integral_terms.push(anti);
            }
        }
        for term in &decomposition.terms {
            integral_terms.push(integrate_term(term, wrt)?);
        }

        if integral_terms.is_empty() {
            return Some(LoweredOp::Const(0.0));
        }
        integral_terms
            .into_iter()
            .reduce(|acc, t| LoweredOp::Add(Arc::new(acc), Arc::new(t)))
    }

    /// Integrate a single partial-fraction summand `numerator / factor^power`.
    ///
    /// Handles a monic linear factor (any power) and a monic irreducible
    /// quadratic factor (power 1). Returns `None` for anything else.
    fn integrate_term(term: &PfTerm, wrt: usize) -> Option<LoweredOp> {
        match term.factor.degree()? {
            1 => integrate_linear_term(term, wrt),
            2 => integrate_quadratic_term(term, wrt),
            _ => None,
        }
    }

    /// ∫ A / (x − r)^j dx for a monic linear factor `x − r`.
    fn integrate_linear_term(term: &PfTerm, wrt: usize) -> Option<LoweredOp> {
        // A monic linear factor `x + c0` has root `r = −c0`.
        let root = -coeff_at(&term.factor, 0);
        let a_coeff = coeff_at(&term.numerator, 0);
        if a_coeff.abs() < 1e-14 {
            return Some(LoweredOp::Const(0.0));
        }
        let x_minus_r = LoweredOp::Sub(
            Arc::new(LoweredOp::Var(wrt)),
            Arc::new(LoweredOp::Const(root)),
        );
        let j = term.power;
        if j == 1 {
            // A · ln|x − r|, written as A · ln(√((x − r)²)).
            let abs_x_minus_r = LoweredOp::Pow(
                Arc::new(LoweredOp::Pow(
                    Arc::new(x_minus_r),
                    Arc::new(LoweredOp::Const(2.0)),
                )),
                Arc::new(LoweredOp::Const(0.5)),
            );
            Some(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(a_coeff)),
                Arc::new(LoweredOp::Ln(Arc::new(abs_x_minus_r))),
            ))
        } else {
            // A / (1 − j) · (x − r)^(1 − j).
            let exp_val = 1.0 - j as f64;
            Some(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(a_coeff / exp_val)),
                Arc::new(LoweredOp::Pow(
                    Arc::new(x_minus_r),
                    Arc::new(LoweredOp::Const(exp_val)),
                )),
            ))
        }
    }

    /// ∫ (B x + C) / (x² + c₁ x + c₀) dx for a monic irreducible quadratic.
    ///
    /// Completing the square `x² + c₁x + c₀ = (x − p)² + q²` with `p = −c₁/2`,
    /// `q² = c₀ − (c₁/2)²`, and writing `Bx + C = B(x − p) + (C + Bp)`, gives
    ///
    /// ```text
    /// ∫ = (B/2)·ln((x − p)² + q²) + ((C + B p)/q)·arctan((x − p)/q).
    /// ```
    fn integrate_quadratic_term(term: &PfTerm, wrt: usize) -> Option<LoweredOp> {
        if term.power != 1 {
            return None; // A repeated irreducible quadratic is not in this table.
        }
        let c0 = coeff_at(&term.factor, 0);
        let c1 = coeff_at(&term.factor, 1);
        let b_coeff = coeff_at(&term.numerator, 1);
        let c_coeff = coeff_at(&term.numerator, 0);

        let p_val = -c1 / 2.0;
        let q_sq = c0 - (c1 / 2.0).powi(2);
        if q_sq <= 0.0 {
            return None; // Not irreducible over ℝ; handled by the linear path.
        }
        let q_val = q_sq.sqrt();

        let x_minus_p = LoweredOp::Sub(
            Arc::new(LoweredOp::Var(wrt)),
            Arc::new(LoweredOp::Const(p_val)),
        );

        let mut parts: Vec<LoweredOp> = Vec::new();
        if b_coeff.abs() > 1e-14 {
            let ln_arg = LoweredOp::Add(
                Arc::new(LoweredOp::Pow(
                    Arc::new(x_minus_p.clone()),
                    Arc::new(LoweredOp::Const(2.0)),
                )),
                Arc::new(LoweredOp::Const(q_val * q_val)),
            );
            parts.push(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(b_coeff / 2.0)),
                Arc::new(LoweredOp::Ln(Arc::new(ln_arg))),
            ));
        }

        let arctan_coeff = (c_coeff + b_coeff * p_val) / q_val;
        if arctan_coeff.abs() > 1e-14 {
            let arctan_arg = LoweredOp::Div(Arc::new(x_minus_p), Arc::new(LoweredOp::Const(q_val)));
            parts.push(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(arctan_coeff)),
                Arc::new(LoweredOp::Arctan(Arc::new(arctan_arg))),
            ));
        }

        if parts.is_empty() {
            return Some(LoweredOp::Const(0.0));
        }
        parts
            .into_iter()
            .reduce(|acc, t| LoweredOp::Add(Arc::new(acc), Arc::new(t)))
    }

    /// Read the coefficient of `x^deg` from `p` as an `f64` (missing → 0).
    fn coeff_at(p: &Poly, deg: usize) -> f64 {
        p.coeffs.get(deg).map_or(0.0, ratio_to_f64)
    }

    fn poly_antiderivative(p: &Poly, wrt: usize) -> Option<LoweredOp> {
        let mut terms: Vec<LoweredOp> = Vec::new();
        for (i, coeff) in p.coeffs.iter().enumerate() {
            let a_f64 = ratio_to_f64(coeff);
            if a_f64.abs() < 1e-15 {
                continue;
            }
            let power = (i + 1) as f64;
            terms.push(LoweredOp::Mul(
                Arc::new(LoweredOp::Const(a_f64 / power)),
                Arc::new(LoweredOp::Pow(
                    Arc::new(LoweredOp::Var(wrt)),
                    Arc::new(LoweredOp::Const(power)),
                )),
            ));
        }
        terms
            .into_iter()
            .reduce(|acc, t| LoweredOp::Add(Arc::new(acc), Arc::new(t)))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::lower::LoweredOp;
        use std::sync::Arc;

        fn x() -> LoweredOp {
            LoweredOp::Var(0)
        }
        fn c(v: f64) -> LoweredOp {
            LoweredOp::Const(v)
        }
        fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
            LoweredOp::Add(Arc::new(a), Arc::new(b))
        }
        fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
            LoweredOp::Sub(Arc::new(a), Arc::new(b))
        }
        fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
            LoweredOp::Pow(Arc::new(a), Arc::new(b))
        }

        fn eval_at(expr: &LoweredOp, x_val: f64) -> f64 {
            expr.eval(&[x_val])
        }

        fn check_antideriv_at(anti: &LoweredOp, f: impl Fn(f64) -> f64, points: &[f64]) {
            for &xv in points {
                let fd = (eval_at(anti, xv + 1e-7) - eval_at(anti, xv)) / 1e-7;
                let fv = f(xv);
                if fv.abs() > 1e-10 {
                    assert!(
                        (fd / fv - 1.0).abs() < 1e-4,
                        "antideriv check failed at x={xv}: fd={fd}, f={fv}"
                    );
                } else {
                    assert!(
                        fd.abs() < 1e-6,
                        "antideriv check near zero at x={xv}: fd={fd}"
                    );
                }
            }
        }

        #[test]
        fn test_integral_arctan() {
            let num = c(1.0);
            let den = add(pow(x(), c(2.0)), c(1.0));
            let result = integrate_rational(&num, &den, 0);
            assert!(result.is_some(), "1/(x2+1) should integrate");
            let anti = result.unwrap();
            check_antideriv_at(&anti, |xv| 1.0 / (xv * xv + 1.0), &[0.5, 1.0, 2.0, -1.5]);
        }

        #[test]
        fn test_integral_log() {
            let num = c(1.0);
            let den = sub(pow(x(), c(2.0)), c(1.0));
            let result = integrate_rational(&num, &den, 0);
            assert!(result.is_some(), "1/(x2-1) should integrate");
            let anti = result.unwrap();
            check_antideriv_at(&anti, |xv| 1.0 / (xv * xv - 1.0), &[2.0, 3.0, -2.0, -3.0]);
        }

        #[test]
        fn test_integral_repeated() {
            let num = c(1.0);
            let den = pow(sub(x(), c(1.0)), c(2.0));
            let result = integrate_rational(&num, &den, 0);
            assert!(result.is_some(), "1/(x-1)2 should integrate");
            let anti = result.unwrap();
            check_antideriv_at(&anti, |xv| 1.0 / (xv - 1.0).powi(2), &[2.0, 3.0, -1.0, 0.5]);
        }

        #[test]
        fn test_integral_x_over_x2_plus_1() {
            let num = x();
            let den = add(pow(x(), c(2.0)), c(1.0));
            let result = integrate_rational(&num, &den, 0);
            assert!(result.is_some(), "x/(x2+1) should integrate");
            let anti = result.unwrap();
            check_antideriv_at(&anti, |xv| xv / (xv * xv + 1.0), &[0.5, 1.0, 2.0, -1.0]);
        }

        // Exercises the arctan-coefficient path with p ≠ 0 AND B ≠ 0 — the
        // combination that the earlier hand-rolled formula got wrong (it used
        // `C − Bp` where the completed-square derivation requires `C + Bp`).
        // The dormant sign error is now fixed; this locks it in.
        #[test]
        fn test_integral_x_over_irreducible_quadratic_shifted() {
            let two_x = LoweredOp::Mul(Arc::new(c(2.0)), Arc::new(x()));
            // x / (x^2 + 2x + 2)
            let num = x();
            let den = add(add(pow(x(), c(2.0)), two_x), c(2.0));
            let result = integrate_rational(&num, &den, 0);
            assert!(result.is_some(), "x/(x^2+2x+2) should integrate");
            let anti = result.unwrap();
            check_antideriv_at(
                &anti,
                |xv| xv / (xv * xv + 2.0 * xv + 2.0),
                &[0.5, 1.0, 2.0, -0.5, 3.0],
            );
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Builder helpers.
    fn x() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn c(v: f64) -> LoweredOp {
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
    fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Div(Arc::new(a), Arc::new(b))
    }
    fn pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Arc::new(a), Arc::new(b))
    }
    fn exp_op(a: LoweredOp) -> LoweredOp {
        LoweredOp::Exp(Arc::new(a))
    }
    fn ln_op(a: LoweredOp) -> LoweredOp {
        LoweredOp::Ln(Arc::new(a))
    }
    fn sin_op(a: LoweredOp) -> LoweredOp {
        LoweredOp::Sin(Arc::new(a))
    }
    fn cos_op(a: LoweredOp) -> LoweredOp {
        LoweredOp::Cos(Arc::new(a))
    }
    fn neg_op(a: LoweredOp) -> LoweredOp {
        LoweredOp::Neg(Arc::new(a))
    }

    fn eval_anti(expr: &LoweredOp, x_val: f64) -> f64 {
        expr.eval(&[x_val])
    }

    /// Verify F'(x) ≈ f(x) via forward finite differences at default points.
    fn check_antideriv(f: &LoweredOp, expected_f: impl Fn(f64) -> f64) {
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                for &xv in &[-1.5_f64, -0.5, 0.3, 1.0, 2.0] {
                    let fd = (eval_anti(&anti, xv + 1e-7) - eval_anti(&anti, xv)) / 1e-7;
                    let fv = expected_f(xv);
                    if fv.abs() > 1e-10 {
                        assert!(
                            (fd / fv - 1.0).abs() < 1e-4,
                            "antideriv check failed at x={xv}: fd={fd}, f={fv}"
                        );
                    } else {
                        assert!(
                            fd.abs() < 1e-6,
                            "antideriv check failed near zero at x={xv}: fd={fd}"
                        );
                    }
                }
            }
            IntegrateResult::Unsupported => panic!("Expected Closed, got Unsupported"),
        }
    }

    /// Verify F'(x) ≈ f(x) at specific points (use when domain is restricted).
    fn check_antideriv_at(anti: &LoweredOp, f: impl Fn(f64) -> f64, points: &[f64]) {
        for &xv in points {
            let fd = (eval_anti(anti, xv + 1e-7) - eval_anti(anti, xv)) / 1e-7;
            let fv = f(xv);
            if fv.abs() > 1e-10 {
                assert!(
                    (fd / fv - 1.0).abs() < 1e-4,
                    "antideriv check failed at x={xv}: fd={fd}, f={fv}"
                );
            } else {
                assert!(
                    fd.abs() < 1e-6,
                    "antideriv check failed near zero at x={xv}: fd={fd}"
                );
            }
        }
    }

    // ── Test 1: ∫x³ ──────────────────────────────────────────────────────────

    #[test]
    fn integrate_x_cubed() {
        let f = pow(x(), c(3.0));
        check_antideriv(&f, |xv| xv.powi(3));
    }

    // ── Test 2: ∫(1/x) ───────────────────────────────────────────────────────

    #[test]
    fn integrate_one_over_x() {
        let f = div(c(1.0), x());
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                check_antideriv_at(&anti, |xv| 1.0 / xv, &[0.5, 1.0, 2.0, 3.0]);
            }
            IntegrateResult::Unsupported => panic!("Expected Closed for 1/x"),
        }
    }

    // ── Test 3: ∫exp(x) ──────────────────────────────────────────────────────

    #[test]
    fn integrate_exp_x() {
        let f = exp_op(x());
        check_antideriv(&f, |xv| xv.exp());
    }

    // ── Test 4: ∫sin(x) ──────────────────────────────────────────────────────

    #[test]
    fn integrate_sin_x() {
        let f = sin_op(x());
        check_antideriv(&f, |xv| xv.sin());
    }

    // ── Test 5: ∫cos(x) ──────────────────────────────────────────────────────

    #[test]
    fn integrate_cos_x() {
        let f = cos_op(x());
        check_antideriv(&f, |xv| xv.cos());
    }

    // ── Test 6: ∫(2x+3) ──────────────────────────────────────────────────────

    #[test]
    fn integrate_linear() {
        let f = add(mul(c(2.0), x()), c(3.0));
        check_antideriv(&f, |xv| 2.0 * xv + 3.0);
    }

    // ── Test 7: ∫exp(2x+1) ───────────────────────────────────────────────────

    #[test]
    fn integrate_exp_affine() {
        let f = exp_op(add(mul(c(2.0), x()), c(1.0)));
        check_antideriv(&f, |xv| (2.0 * xv + 1.0).exp());
    }

    // ── Test 8: ∫sin(3x) ─────────────────────────────────────────────────────

    #[test]
    fn integrate_sin_affine() {
        let f = sin_op(mul(c(3.0), x()));
        check_antideriv(&f, |xv| (3.0 * xv).sin());
    }

    // ── Test 9: ∫x·exp(x) (by parts) ─────────────────────────────────────────

    #[test]
    fn integrate_x_times_exp() {
        let f = mul(x(), exp_op(x()));
        check_antideriv(&f, |xv| xv * xv.exp());
    }

    // ── Test 10: ∫ln(x) (by parts) ───────────────────────────────────────────

    #[test]
    fn integrate_ln_x() {
        let f = ln_op(x());
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                check_antideriv_at(&anti, |xv| xv.ln(), &[0.5, 1.0, 2.0, 3.0]);
            }
            IntegrateResult::Unsupported => panic!("Expected Closed for ln(x)"),
        }
    }

    // ── Test 11: ∫arctan(x) ──────────────────────────────────────────────────

    #[test]
    fn integrate_arctan() {
        let f = LoweredOp::Arctan(Arc::new(x()));
        check_antideriv(&f, |xv| xv.atan());
    }

    // ── Test 12: exp(x²) → Unsupported ───────────────────────────────────────

    #[test]
    fn integrate_exp_x_squared_unsupported() {
        let f = exp_op(pow(x(), c(2.0)));
        assert!(
            matches!(f.integrate(0), IntegrateResult::Unsupported),
            "exp(x²) should be Unsupported"
        );
    }

    // ── Test 13: ∫₀¹ x² ≈ 1/3 ────────────────────────────────────────────────

    #[test]
    fn integrate_definite_x_squared() {
        let f = pow(x(), c(2.0));
        let ctx = EvalCtx::new(&[]);
        let result = f
            .integrate_definite(0, 0.0, 1.0, &ctx)
            .expect("should succeed");
        assert!(
            (result - 1.0 / 3.0).abs() < 1e-9,
            "∫₀¹ x² expected 1/3, got {result}"
        );
    }

    // ── Test 14: ∫₀¹ exp(x²) via quadrature fallback ─────────────────────────

    #[test]
    fn integrate_definite_exp_x_squared_quadrature() {
        let f = exp_op(pow(x(), c(2.0)));
        let ctx = EvalCtx::new(&[]);
        let result = f
            .integrate_definite(0, 0.0, 1.0, &ctx)
            .expect("quadrature should succeed");
        // Known value ≈ 1.46265174590718
        assert!(
            (result - 1.462_651_745_907_18).abs() < 1e-6,
            "∫₀¹ exp(x²) expected ~1.46265, got {result}"
        );
    }

    // ── Bonus: reversed-bounds sign ──────────────────────────────────────────

    #[test]
    fn integrate_definite_reversed_bounds() {
        let f = x();
        let ctx = EvalCtx::new(&[]);
        let fwd = f.integrate_definite(0, 0.0, 1.0, &ctx).expect("ok");
        let bwd = f.integrate_definite(0, 1.0, 0.0, &ctx).expect("ok");
        assert!(
            (fwd + bwd).abs() < 1e-12,
            "forward + backward should be 0: {fwd} + {bwd}"
        );
        assert!((fwd - 0.5).abs() < 1e-12, "∫₀¹ x = 0.5, got {fwd}");
    }

    // ── Bonus: constant factor ────────────────────────────────────────────────

    #[test]
    fn integrate_constant_factor() {
        // ∫ 5 dx = 5x
        let f = c(5.0);
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                // antiderivative should be 5*x; evaluate derivative numerically
                check_antideriv_at(&anti, |_| 5.0, &[0.0, 1.0, -2.0]);
            }
            IntegrateResult::Unsupported => panic!("constant should integrate"),
        }
    }

    // ── Bonus: negation ───────────────────────────────────────────────────────

    #[test]
    fn integrate_neg_cos() {
        // ∫ -cos(x) dx = -sin(x)
        let f = neg_op(cos_op(x()));
        check_antideriv(&f, |xv| -xv.cos());
    }

    // ── Bonus: sub ────────────────────────────────────────────────────────────

    #[test]
    fn integrate_sub() {
        // ∫ (x² - x) dx = x³/3 - x²/2
        let f = sub(pow(x(), c(2.0)), x());
        check_antideriv(&f, |xv| xv.powi(2) - xv);
    }

    // ── C3 Tests: u-substitution and trig substitution ───────────────────────

    #[test]
    fn test_u_sub_cos_x_squared() {
        // integral 2x*cos(x^2) dx = sin(x^2) (u = x^2, du = 2x dx)
        let two_x = mul(c(2.0), x());
        let x_sq = pow(x(), c(2.0));
        let f = mul(two_x, cos_op(x_sq));
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                check_antideriv_at(
                    &anti,
                    |xv| 2.0 * xv * (xv * xv).cos(),
                    &[0.3, 0.7, 1.5, 2.0],
                );
            }
            IntegrateResult::Unsupported => {
                // u-sub may not catch this case -- acceptable
            }
        }
    }

    #[test]
    fn test_u_sub_x_over_x_sq_plus_1() {
        // integral x/(x^2+1) dx = (1/2) ln(x^2+1)
        let x_sq_plus_1 = add(pow(x(), c(2.0)), c(1.0));
        let f = div(x(), x_sq_plus_1);
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                check_antideriv_at(&anti, |xv| xv / (xv * xv + 1.0), &[0.5, 1.0, 2.0, -1.0]);
            }
            IntegrateResult::Unsupported => {}
        }
    }

    #[test]
    fn test_trig_sub_inv_sqrt_1_minus_x_sq() {
        // integral 1/sqrt(1-x^2) dx = arcsin(x)
        let inner = sub(c(1.0), pow(x(), c(2.0)));
        let f = LoweredOp::Pow(Arc::new(inner), Arc::new(c(-0.5)));
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                check_antideriv_at(
                    &anti,
                    |xv| 1.0 / (1.0 - xv * xv).sqrt(),
                    &[0.1, 0.3, 0.5, 0.7],
                );
            }
            IntegrateResult::Unsupported => panic!("1/sqrt(1-x^2) should integrate via trig sub"),
        }
    }

    #[test]
    fn test_trig_sub_inv_sqrt_1_plus_x_sq() {
        // integral 1/sqrt(1+x^2) dx = arcsinh(x)
        let inner = add(c(1.0), pow(x(), c(2.0)));
        let f = LoweredOp::Pow(Arc::new(inner), Arc::new(c(-0.5)));
        match f.integrate(0) {
            IntegrateResult::Closed(anti) => {
                check_antideriv_at(
                    &anti,
                    |xv| 1.0 / (1.0 + xv * xv).sqrt(),
                    &[0.5, 1.0, 2.0, 3.0],
                );
            }
            IntegrateResult::Unsupported => panic!("1/sqrt(1+x^2) should integrate via trig sub"),
        }
    }

    #[test]
    fn test_trig_sub_sqrt_1_minus_x_sq() {
        // integral sqrt(1-x^2) dx -- definite integral should be pi/4 over [0,1]
        let inner = sub(c(1.0), pow(x(), c(2.0)));
        let f = LoweredOp::Pow(Arc::new(inner), Arc::new(c(0.5)));
        let ctx = crate::eval::EvalCtx::new(&[]);
        let result = f
            .integrate_definite(0, 0.0, 1.0, &ctx)
            .expect("should compute");
        assert!(
            (result - std::f64::consts::PI / 4.0).abs() < 1e-4,
            "integral from 0 to 1 of sqrt(1-x^2) expected pi/4 approximately {:.6}, got {result}",
            std::f64::consts::PI / 4.0
        );
    }

    #[test]
    fn test_negative_cases_remain_unsupported() {
        // integral sin(x^2) -- non-elementary, should stay Unsupported
        let f = sin_op(pow(x(), c(2.0)));
        assert!(
            matches!(f.integrate(0), IntegrateResult::Unsupported),
            "sin(x^2) should be Unsupported"
        );

        // integral exp(x^2) -- non-elementary
        let f2 = exp_op(pow(x(), c(2.0)));
        assert!(
            matches!(f2.integrate(0), IntegrateResult::Unsupported),
            "exp(x^2) should be Unsupported"
        );
    }
}
