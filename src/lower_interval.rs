//! Interval arithmetic for over-approximate evaluation of [`LoweredOp`] trees.
//!
//! This module provides [`IntervalLO`], a closed real interval `[lo, hi]`, and
//! implements [`LoweredOp::eval_interval`] which propagates interval bounds
//! through every operation variant.

use crate::lower::LoweredOp;

/// A closed interval `[lo, hi]` for over-approximating interval arithmetic.
///
/// Used by [`LoweredOp::eval_interval`] to propagate bounds through
/// expression trees. NaN sentinels (`lo = hi = NaN`) represent undefined
/// or empty results (e.g., `ln` of a non-positive interval).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntervalLO {
    /// Lower bound of the interval.
    pub lo: f64,
    /// Upper bound of the interval.
    pub hi: f64,
}

impl IntervalLO {
    /// Construct an interval with given bounds.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Construct a degenerate point interval `[v, v]`.
    pub fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    /// The universal interval `[-∞, +∞]`.
    pub fn full() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// Sentinel NaN interval representing an undefined or out-of-domain result.
    pub fn nan() -> Self {
        Self {
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }

    /// Returns `true` if the interval is empty (lo > hi).
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// Width of the interval (`hi - lo`).
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Returns `true` if `x` lies within `[lo, hi]`.
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// Smallest interval enclosing both `self` and `other`.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Largest interval contained in both `self` and `other`.
    pub fn intersect(&self, other: &Self) -> Self {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        Self { lo, hi }
    }
}

impl LoweredOp {
    /// Over-approximate interval evaluation of this expression tree.
    ///
    /// Returns an [`IntervalLO`] that is guaranteed to contain the true
    /// result for every point-wise variable assignment within `vars`.
    /// Uses standard interval arithmetic rules for each operation.
    pub fn eval_interval(&self, vars: &[IntervalLO]) -> IntervalLO {
        match self {
            Self::Const(c) => IntervalLO::point(*c),
            Self::NamedConst(nc) => IntervalLO::point(nc.value()),
            Self::Var(i) => vars.get(*i).copied().unwrap_or_else(IntervalLO::nan),
            Self::Neg(x) => {
                let ix = x.eval_interval(vars);
                IntervalLO {
                    lo: -ix.hi,
                    hi: -ix.lo,
                }
            }
            Self::Add(a, b) => {
                let ia = a.eval_interval(vars);
                let ib = b.eval_interval(vars);
                IntervalLO {
                    lo: ia.lo + ib.lo,
                    hi: ia.hi + ib.hi,
                }
            }
            Self::Sub(a, b) => {
                let ia = a.eval_interval(vars);
                let ib = b.eval_interval(vars);
                IntervalLO {
                    lo: ia.lo - ib.hi,
                    hi: ia.hi - ib.lo,
                }
            }
            Self::Mul(a, b) => {
                let ia = a.eval_interval(vars);
                let ib = b.eval_interval(vars);
                let p = [ia.lo * ib.lo, ia.lo * ib.hi, ia.hi * ib.lo, ia.hi * ib.hi];
                IntervalLO {
                    lo: p.iter().copied().fold(f64::INFINITY, f64::min),
                    hi: p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                }
            }
            Self::Div(a, b) => {
                let ia = a.eval_interval(vars);
                let ib = b.eval_interval(vars);
                if ib.lo <= 0.0 && ib.hi >= 0.0 {
                    return IntervalLO::full();
                }
                // Multiply by reciprocal [1/hi, 1/lo]
                let recip = IntervalLO {
                    lo: 1.0 / ib.hi,
                    hi: 1.0 / ib.lo,
                };
                let p = [
                    ia.lo * recip.lo,
                    ia.lo * recip.hi,
                    ia.hi * recip.lo,
                    ia.hi * recip.hi,
                ];
                IntervalLO {
                    lo: p.iter().copied().fold(f64::INFINITY, f64::min),
                    hi: p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                }
            }
            Self::Pow(base, exp) => eval_interval_pow(base, exp, vars),
            Self::Exp(x) => {
                let ix = x.eval_interval(vars);
                IntervalLO {
                    lo: ix.lo.exp(),
                    hi: ix.hi.exp(),
                }
            }
            Self::Ln(x) => {
                let ix = x.eval_interval(vars);
                if ix.lo > 0.0 {
                    IntervalLO {
                        lo: ix.lo.ln(),
                        hi: ix.hi.ln(),
                    }
                } else {
                    IntervalLO::nan()
                }
            }
            Self::Sin(x) => eval_interval_sin(x, vars),
            Self::Cos(x) => eval_interval_cos(x, vars),
            Self::Tan(x) => {
                let ix = x.eval_interval(vars);
                let half_pi = std::f64::consts::FRAC_PI_2;
                let pi = std::f64::consts::PI;
                let k_lo = ((ix.lo - half_pi) / pi).ceil() as i64;
                let k_hi = ((ix.hi - half_pi) / pi).floor() as i64;
                if k_lo <= k_hi {
                    return IntervalLO::full();
                }
                IntervalLO {
                    lo: ix.lo.tan(),
                    hi: ix.hi.tan(),
                }
            }
            Self::Sinh(x) => {
                let ix = x.eval_interval(vars);
                IntervalLO {
                    lo: ix.lo.sinh(),
                    hi: ix.hi.sinh(),
                }
            }
            Self::Cosh(x) => {
                let ix = x.eval_interval(vars);
                let lo_val = if ix.lo <= 0.0 && 0.0 <= ix.hi {
                    1.0
                } else {
                    ix.lo.cosh().min(ix.hi.cosh())
                };
                let hi_val = ix.lo.cosh().max(ix.hi.cosh());
                IntervalLO {
                    lo: lo_val,
                    hi: hi_val,
                }
            }
            Self::Tanh(x) => {
                let ix = x.eval_interval(vars);
                IntervalLO {
                    lo: ix.lo.tanh(),
                    hi: ix.hi.tanh(),
                }
            }
            Self::Arcsin(x) => {
                let ix = x.eval_interval(vars);
                if ix.lo < -1.0 || ix.hi > 1.0 {
                    IntervalLO::nan()
                } else {
                    IntervalLO {
                        lo: ix.lo.asin(),
                        hi: ix.hi.asin(),
                    }
                }
            }
            Self::Arccos(x) => {
                let ix = x.eval_interval(vars);
                if ix.lo < -1.0 || ix.hi > 1.0 {
                    IntervalLO::nan()
                } else {
                    // arccos is decreasing: lo maps to larger value
                    IntervalLO {
                        lo: ix.hi.acos(),
                        hi: ix.lo.acos(),
                    }
                }
            }
            Self::Arctan(x) => {
                let ix = x.eval_interval(vars);
                IntervalLO {
                    lo: ix.lo.atan(),
                    hi: ix.hi.atan(),
                }
            }
            Self::Arcsinh(x) => {
                let ix = x.eval_interval(vars);
                IntervalLO {
                    lo: ix.lo.asinh(),
                    hi: ix.hi.asinh(),
                }
            }
            Self::Arccosh(x) => {
                let ix = x.eval_interval(vars);
                if ix.hi < 1.0 {
                    return IntervalLO::nan();
                }
                let lo_clamped = ix.lo.max(1.0);
                IntervalLO {
                    lo: lo_clamped.acosh(),
                    hi: ix.hi.acosh(),
                }
            }
            Self::Arctanh(x) => {
                let ix = x.eval_interval(vars);
                if ix.lo <= -1.0 || ix.hi >= 1.0 {
                    IntervalLO {
                        lo: f64::NEG_INFINITY,
                        hi: f64::INFINITY,
                    }
                } else {
                    IntervalLO {
                        lo: ix.lo.atanh(),
                        hi: ix.hi.atanh(),
                    }
                }
            }
            Self::Erf(x) => {
                let ix = x.eval_interval(vars);
                // erf is monotone increasing: erf([lo, hi]) = [erf(lo), erf(hi)]
                IntervalLO {
                    lo: crate::special::erf(ix.lo),
                    hi: crate::special::erf(ix.hi),
                }
            }
            Self::LGamma(x) => {
                // LGamma is complex; return full interval conservatively
                let _ix = x.eval_interval(vars);
                IntervalLO::full()
            }
            Self::Digamma(x) => {
                let _ix = x.eval_interval(vars);
                IntervalLO::full()
            }
            Self::Trigamma(x) => {
                let _ix = x.eval_interval(vars);
                IntervalLO::full()
            }
            Self::Ei(x) => {
                let ix = x.eval_interval(vars);
                let lo = crate::special::ei(ix.lo);
                let hi = crate::special::ei(ix.hi);
                IntervalLO {
                    lo: lo.min(hi),
                    hi: lo.max(hi),
                }
            }
            Self::Si(x) => {
                // Si is bounded by [-π/2 - ε, π/2 + ε]
                let _ix = x.eval_interval(vars);
                IntervalLO {
                    lo: -(std::f64::consts::FRAC_PI_2 + 1e-10),
                    hi: std::f64::consts::FRAC_PI_2 + 1e-10,
                }
            }
            Self::Ci(x) => {
                // Ci is unbounded for small x (goes to -∞)
                let _ix = x.eval_interval(vars);
                IntervalLO::full()
            }
        }
    }
}

/// Interval evaluation of `base^exp`.
///
/// Handles integer exponents via direct power (preserving monotonicity/shape),
/// and general (non-integer) exponents via `exp(exp_interval * ln(base_interval))`.
fn eval_interval_pow(base: &LoweredOp, exp: &LoweredOp, vars: &[IntervalLO]) -> IntervalLO {
    let ibase = base.eval_interval(vars);
    let iexp = exp.eval_interval(vars);
    // Integer exponent fast path
    if let LoweredOp::Const(e) = exp {
        let floor_e = e.floor();
        if (*e - floor_e).abs() < 1e-15 && *e >= 0.0 && *e <= 20.0 {
            let n = *e as u32;
            if n == 0 {
                return IntervalLO::point(1.0);
            }
            if n == 1 {
                return ibase;
            }
            if n.is_multiple_of(2) {
                // Even power: U-shaped, minimum at zero
                if ibase.lo >= 0.0 {
                    return IntervalLO {
                        lo: ibase.lo.powi(n as i32),
                        hi: ibase.hi.powi(n as i32),
                    };
                } else if ibase.hi <= 0.0 {
                    return IntervalLO {
                        lo: ibase.hi.powi(n as i32),
                        hi: ibase.lo.powi(n as i32),
                    };
                } else {
                    // Straddles zero: min is 0, max is max(|lo|, |hi|)^n
                    return IntervalLO {
                        lo: 0.0,
                        hi: ibase.lo.abs().max(ibase.hi.abs()).powi(n as i32),
                    };
                }
            } else {
                // Odd power: monotone increasing
                return IntervalLO {
                    lo: ibase.lo.powi(n as i32),
                    hi: ibase.hi.powi(n as i32),
                };
            }
        }
    }
    // Non-integer or non-small exponent: use exp(exp * ln(base))
    if ibase.lo <= 0.0 {
        return IntervalLO::nan();
    }
    let ln_base = IntervalLO {
        lo: ibase.lo.ln(),
        hi: ibase.hi.ln(),
    };
    // Full 4-corner cross-product of iexp × ln_base
    let p = [
        iexp.lo * ln_base.lo,
        iexp.lo * ln_base.hi,
        iexp.hi * ln_base.lo,
        iexp.hi * ln_base.hi,
    ];
    let mul_lo = p.iter().copied().fold(f64::INFINITY, f64::min);
    let mul_hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    IntervalLO {
        lo: mul_lo.exp(),
        hi: mul_hi.exp(),
    }
}

/// Interval evaluation of `sin(x)`.
///
/// Checks critical points `π/2 + kπ` where `sin` achieves ±1.
/// Guard against huge intervals first (width ≥ 2π → return [-1, 1]).
fn eval_interval_sin(x: &LoweredOp, vars: &[IntervalLO]) -> IntervalLO {
    let ix = x.eval_interval(vars);
    if ix.hi - ix.lo >= 2.0 * std::f64::consts::PI {
        return IntervalLO { lo: -1.0, hi: 1.0 };
    }
    let mut vals = vec![ix.lo.sin(), ix.hi.sin()];
    let half_pi = std::f64::consts::FRAC_PI_2;
    let pi = std::f64::consts::PI;
    let k_lo = ((ix.lo - half_pi) / pi).ceil() as i64;
    let k_hi = ((ix.hi - half_pi) / pi).floor() as i64;
    for k in k_lo..=k_hi {
        let crit = half_pi + k as f64 * pi;
        if ix.lo <= crit && crit <= ix.hi {
            vals.push(crit.sin());
        }
    }
    IntervalLO {
        lo: vals.iter().copied().fold(f64::INFINITY, f64::min),
        hi: vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Interval evaluation of `cos(x)`.
///
/// Checks critical points `kπ` where `cos` achieves ±1.
/// Guard against huge intervals first (width ≥ 2π → return [-1, 1]).
fn eval_interval_cos(x: &LoweredOp, vars: &[IntervalLO]) -> IntervalLO {
    let ix = x.eval_interval(vars);
    if ix.hi - ix.lo >= 2.0 * std::f64::consts::PI {
        return IntervalLO { lo: -1.0, hi: 1.0 };
    }
    let mut vals = vec![ix.lo.cos(), ix.hi.cos()];
    let pi = std::f64::consts::PI;
    let k_lo = (ix.lo / pi).ceil() as i64;
    let k_hi = (ix.hi / pi).floor() as i64;
    for k in k_lo..=k_hi {
        let crit = k as f64 * pi;
        if ix.lo <= crit && crit <= ix.hi {
            vals.push(crit.cos());
        }
    }
    IntervalLO {
        lo: vals.iter().copied().fold(f64::INFINITY, f64::min),
        hi: vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}
