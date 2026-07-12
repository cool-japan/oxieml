//! Laurent series expansion about a (possibly singular) center.
//!
//! # Mathematics
//!
//! A function `f` that is analytic in a punctured disc `0 < |z − c| < R` admits
//! a **Laurent series**
//!
//! ```text
//!     f(z) = Σ_{k = −∞}^{∞} a_k (z − c)^k
//!          = ( a_{−m}/(z−c)^m + … + a_{−1}/(z−c) )   [principal part]
//!          + ( a_0 + a_1 (z−c) + a_2 (z−c)^2 + … )   [regular / analytic part].
//! ```
//!
//! The lowest power present, `lo_pow = −m` (with `a_{−m} ≠ 0`), classifies the
//! isolated singularity at `c`:
//!
//! * `lo_pow ≥ 0`  — `c` is a regular point (or a zero); the series is an
//!   ordinary Taylor series.
//! * `lo_pow = −m < 0` with finitely many terms — `c` is a **pole of order
//!   `m`**.
//! * infinitely many negative powers — `c` is an **essential singularity**
//!   (e.g. `exp(1/x)`, `sin(1/x)` at `0`); no finite Laurent expansion exists,
//!   reported as [`EmlError::EssentialSingularity`].
//! * a logarithmic or fractional-power term (`ln(z−c)`, `(z−c)^{1/2}`) — `c` is
//!   a **branch point**; the function is multi-valued, no single-valued Laurent
//!   series exists, reported as [`EmlError::BranchPoint`].
//!
//! # Algorithm
//!
//! Expansion is performed by **exact truncated power-/Laurent-series algebra**
//! (not numerical differentiation, which is noisy at high order). Every node of
//! the expression tree is mapped to a truncated series in the local coordinate
//! `t = z − c`, and the tree is folded bottom-up with series `+ − × ÷`,
//! composition (`exp`, `ln`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`) and
//! powers. A pole is produced naturally when a division introduces a negative
//! leading power; the pole *order* is then simply `−lo_pow`.
//!
//! Each series carries an **absolute precision** `prec`: it is known exactly for
//! every power `< prec`, and `O(t^prec)` beyond. Precision is propagated
//! rigorously through the algebra:
//!
//! * `a ± b`  →  `prec = min(prec_a, prec_b)`,
//! * `a · b`  →  `prec = min(prec_a + lo_b, prec_b + lo_a)`,
//! * `1 / b`  →  computed to the relative precision available in `b`,
//!
//! which is exactly the standard big-`O` bookkeeping for formal power series.
//!
//! Every result is **cross-checked numerically**: the reconstructed series is
//! evaluated at fixed probe points near `c` and compared against a direct
//! evaluation of the original expression. A disagreement returns an error
//! rather than a fabricated series (honesty guard). The detectable pole order is
//! capped at `m ≤ 10`; a deeper apparent pole is reported as an error instead of
//! looping.
//!
//! Expansions in one variable treat every *other* variable `Var(j), j ≠ wrt` as
//! the constant `0`, matching the evaluation convention used elsewhere in the
//! crate (see `crate::limit`).

use crate::error::EmlError;
use crate::lower::LoweredOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Maximum pole order the expansion will report; deeper apparent poles are
/// treated as noise / unsupported and returned as an error (spec cap `m ≤ 10`).
const MAX_POLE_ORDER: i32 = 10;

/// Sentinel absolute precision meaning "known exactly to all orders" (used for
/// polynomials and constants). Kept far above any working budget so ordinary
/// saturating arithmetic never demotes it below the working range.
const EXACT: i32 = i32::MAX / 4;

/// Relative magnitude below which a leading coefficient is treated as zero when
/// stripping leading zeros to determine the true lowest power. The series
/// algebra is exact up to f64 rounding, so this only absorbs accumulated
/// round-off, and the pole order is additionally cross-checked numerically.
const LEAD_TOL: f64 = 1e-9;

/// Probe offsets (both sides of the center) for the numeric cross-check.
const CROSS_CHECK_OFFSETS: [f64; 4] = [1e-2, -1e-2, 5e-3, -5e-3];

/// Relative tolerance for the numeric cross-check of a reconstructed series.
const CROSS_CHECK_TOL: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Public Laurent series type
// ---------------------------------------------------------------------------

/// A truncated Laurent series about `center` in the variable that produced it.
///
/// `coeffs[i]` is the coefficient of `(z − center)^(lo_pow + i)`. The lowest
/// power present is `lo_pow` (which may be negative for a pole). See the
/// [module documentation](crate::series_laurent) for the underlying
/// mathematics.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    /// The expansion center `c`.
    pub center: f64,
    /// The exponent of the leading (lowest-power) term, `coeffs[0]`.
    pub lo_pow: i32,
    /// Coefficients in ascending power order starting at `lo_pow`.
    pub coeffs: Vec<f64>,
}

impl Series {
    /// The order of the pole at `center`: `max(0, −lo_pow)`.
    ///
    /// Returns `0` when the center is a regular point (or a zero of the
    /// function).
    #[must_use]
    pub fn pole_order(&self) -> u32 {
        (-self.lo_pow).max(0) as u32
    }

    /// Evaluate the truncated series at `x`.
    ///
    /// Computes `Σ_i coeffs[i] · (x − center)^(lo_pow + i)`. Evaluating exactly
    /// at `x == center` with a negative `lo_pow` yields a non-finite value, as
    /// expected at a pole.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let t = x - self.center;
        let mut acc = 0.0_f64;
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }
            let power = self.lo_pow + i as i32;
            acc += coeff * t.powi(power);
        }
        acc
    }

    /// Reconstruct a [`LoweredOp`] expression tree equal to this truncated
    /// series: `Σ_i coeffs[i] · (z − center)^(lo_pow + i)` where `z` is
    /// `Var(wrt)`.
    #[must_use]
    pub fn to_lowered(&self, wrt: usize) -> LoweredOp {
        let base = if self.center == 0.0 {
            LoweredOp::Var(wrt)
        } else {
            LoweredOp::Sub(
                Arc::new(LoweredOp::Var(wrt)),
                Arc::new(LoweredOp::Const(self.center)),
            )
        };

        let mut terms: Vec<LoweredOp> = Vec::new();
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }
            let power = self.lo_pow + i as i32;
            let power_term = match power {
                0 => LoweredOp::Const(1.0),
                1 => base.clone(),
                _ => LoweredOp::Pow(
                    Arc::new(base.clone()),
                    Arc::new(LoweredOp::Const(f64::from(power))),
                ),
            };
            let term = if (coeff - 1.0).abs() < f64::EPSILON && power != 0 {
                power_term
            } else if power == 0 {
                LoweredOp::Const(coeff)
            } else {
                LoweredOp::Mul(Arc::new(LoweredOp::Const(coeff)), Arc::new(power_term))
            };
            terms.push(term);
        }

        match terms
            .into_iter()
            .reduce(|acc, t| LoweredOp::Add(Arc::new(acc), Arc::new(t)))
        {
            Some(op) => op.simplify(),
            None => LoweredOp::Const(0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal working series (carries an absolute precision)
// ---------------------------------------------------------------------------

/// A working truncated Laurent series in `t = z − center` used by the algebra.
///
/// `coeffs[i]` is the coefficient of `t^(lo_pow + i)`. The series is known
/// *exactly* for every power `< prec` and `O(t^prec)` beyond. Powers below
/// `lo_pow` and powers in `[lo_pow + coeffs.len(), prec)` are exactly zero.
#[derive(Clone, Debug)]
struct SeriesW {
    lo_pow: i32,
    coeffs: Vec<f64>,
    prec: i32,
}

impl SeriesW {
    /// The exclusive upper bound of the stored powers, `lo_pow + coeffs.len()`.
    fn hi(&self) -> i32 {
        self.lo_pow + self.coeffs.len() as i32
    }

    /// Coefficient of `t^power` (0 if outside the stored range).
    fn coeff_at(&self, power: i32) -> f64 {
        if power < self.lo_pow {
            return 0.0;
        }
        let idx = (power - self.lo_pow) as usize;
        self.coeffs.get(idx).copied().unwrap_or(0.0)
    }

    /// The exactly-zero series.
    fn zero() -> Self {
        Self {
            lo_pow: 0,
            coeffs: Vec::new(),
            prec: EXACT,
        }
    }

    /// A constant (exact to all orders).
    fn constant(value: f64) -> Self {
        if value == 0.0 {
            return Self::zero();
        }
        Self {
            lo_pow: 0,
            coeffs: vec![value],
            prec: EXACT,
        }
    }

    /// Whether every stored coefficient is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0.0)
    }

    /// Drop trailing exactly-zero coefficients (keeps `prec` unchanged; the
    /// dropped tail is exactly zero up to `prec`).
    fn trim_trailing(&mut self) {
        while self.coeffs.last() == Some(&0.0) {
            self.coeffs.pop();
        }
    }

    /// The largest coefficient magnitude (for relative zero tests).
    fn max_abs(&self) -> f64 {
        self.coeffs.iter().fold(0.0_f64, |m, &c| m.max(c.abs()))
    }

    /// Strip leading coefficients that are zero to within `LEAD_TOL · max`,
    /// raising `lo_pow` so that `coeffs[0]` becomes the true leading term.
    /// Leaves a genuine zero series untouched.
    fn strip_leading(&mut self) {
        let scale = self.max_abs();
        if scale == 0.0 {
            return;
        }
        let tol = LEAD_TOL * scale;
        let mut drop = 0usize;
        while drop < self.coeffs.len() && self.coeffs[drop].abs() <= tol {
            drop += 1;
        }
        if drop > 0 {
            self.coeffs.drain(0..drop);
            self.lo_pow += drop as i32;
        }
        self.trim_trailing();
    }

    /// Discard every power `≥ new_prec` and lower `prec` accordingly.
    fn cap_prec(&mut self, new_prec: i32) {
        if new_prec < self.prec {
            self.prec = new_prec;
        }
        // Drop stored coefficients at or above prec.
        if self.hi() > self.prec {
            let keep = (self.prec - self.lo_pow).max(0) as usize;
            if keep < self.coeffs.len() {
                self.coeffs.truncate(keep);
            }
        }
        self.trim_trailing();
    }
}

// ---------------------------------------------------------------------------
// Series algebra
// ---------------------------------------------------------------------------

fn sw_neg(a: &SeriesW) -> SeriesW {
    SeriesW {
        lo_pow: a.lo_pow,
        coeffs: a.coeffs.iter().map(|&c| -c).collect(),
        prec: a.prec,
    }
}

fn sw_scale(a: &SeriesW, s: f64) -> SeriesW {
    if s == 0.0 {
        return SeriesW::zero();
    }
    SeriesW {
        lo_pow: a.lo_pow,
        coeffs: a.coeffs.iter().map(|&c| c * s).collect(),
        prec: a.prec,
    }
}

/// Multiply by `t^k` (shift powers up by `k`).
fn sw_shift(a: &SeriesW, k: i32) -> SeriesW {
    SeriesW {
        lo_pow: a.lo_pow + k,
        coeffs: a.coeffs.clone(),
        prec: a.prec.saturating_add(k),
    }
}

fn sw_add(a: &SeriesW, b: &SeriesW) -> SeriesW {
    if a.is_zero() {
        return b.clone();
    }
    if b.is_zero() {
        return a.clone();
    }
    let prec = a.prec.min(b.prec);
    let lo = a.lo_pow.min(b.lo_pow);
    let hi = a.hi().max(b.hi()).min(prec);
    if hi <= lo {
        let mut z = SeriesW::zero();
        z.prec = prec;
        return z;
    }
    let len = (hi - lo) as usize;
    let mut coeffs = vec![0.0_f64; len];
    for power in lo..hi {
        coeffs[(power - lo) as usize] = a.coeff_at(power) + b.coeff_at(power);
    }
    let mut out = SeriesW {
        lo_pow: lo,
        coeffs,
        prec,
    };
    out.trim_trailing();
    out
}

fn sw_sub(a: &SeriesW, b: &SeriesW) -> SeriesW {
    sw_add(a, &sw_neg(b))
}

/// Subtract a scalar constant (exact) from the series.
fn sw_sub_const(a: &SeriesW, value: f64) -> SeriesW {
    sw_sub(a, &SeriesW::constant(value))
}

fn sw_mul(a: &SeriesW, b: &SeriesW) -> SeriesW {
    if a.is_zero() || b.is_zero() {
        return SeriesW::zero();
    }
    let prec = a
        .prec
        .saturating_add(b.lo_pow)
        .min(b.prec.saturating_add(a.lo_pow));
    let lo = a.lo_pow + b.lo_pow;
    let conv_len = a.coeffs.len() + b.coeffs.len() - 1;
    let mut coeffs = vec![0.0_f64; conv_len];
    for (i, &av) in a.coeffs.iter().enumerate() {
        if av == 0.0 {
            continue;
        }
        for (j, &bv) in b.coeffs.iter().enumerate() {
            coeffs[i + j] += av * bv;
        }
    }
    let mut out = SeriesW {
        lo_pow: lo,
        coeffs,
        prec,
    };
    out.cap_prec(prec);
    out.trim_trailing();
    out
}

/// Reciprocal `1 / b`, computed to at most `budget` relative terms.
///
/// Uses the standard power-series inversion recurrence on the normalised
/// (leading-coefficient-one) part after dividing out the leading power `t^{lo}`.
/// The reciprocal of `b = b_ℓ t^ℓ (1 + w)` is `b_ℓ^{-1} t^{-ℓ} (1 + w)^{-1}`,
/// so the pole/zero structure is inverted, which is precisely how a division
/// creates a pole.
fn sw_recip(b: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let mut bb = b.clone();
    bb.strip_leading();
    if bb.is_zero() {
        // Denominator vanishes to every computed order: cannot form a series.
        return Err(EmlError::NotSolvable);
    }
    let lead = bb.coeffs[0];
    let bl = bb.lo_pow;
    // Relative precision available in b, capped by the working budget.
    let rel_known = bb.prec.saturating_sub(bl);
    let rel = rel_known.min(budget).max(1);
    let rel_usize = rel as usize;

    // Normalised hat-series coefficients b̂_0 = lead, b̂_1, …
    let bhat = |k: usize| -> f64 {
        if k < bb.coeffs.len() {
            bb.coeffs[k]
        } else {
            0.0
        }
    };

    let mut q = vec![0.0_f64; rel_usize];
    q[0] = 1.0 / lead;
    for k in 1..rel_usize {
        let mut acc = 0.0_f64;
        for i in 1..=k {
            acc += bhat(i) * q[k - i];
        }
        q[k] = -acc / lead;
    }

    let mut out = SeriesW {
        lo_pow: -bl,
        coeffs: q,
        prec: (-bl).saturating_add(rel),
    };
    out.trim_trailing();
    Ok(out)
}

fn sw_div(a: &SeriesW, b: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    if a.is_zero() {
        return Ok(SeriesW::zero());
    }
    let r = sw_recip(b, budget)?;
    Ok(sw_mul(a, &r))
}

/// Integer power `base^n` for `n ≥ 0`, truncated at absolute precision
/// `prec_cap`.
fn sw_pow_uint(base: &SeriesW, n: u32, prec_cap: i32) -> SeriesW {
    let mut result = SeriesW::constant(1.0);
    let mut acc = base.clone();
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            result = sw_mul(&result, &acc);
            result.cap_prec(prec_cap);
        }
        e >>= 1;
        if e > 0 {
            acc = sw_mul(&acc, &acc);
            acc.cap_prec(prec_cap);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Transcendental compositions (arg with lo ≥ 1 handled by power series)
// ---------------------------------------------------------------------------

/// `exp(w)` for a series `w` with `lo_pow ≥ 1`, to `budget` relative terms.
fn sw_exp_small(w: &SeriesW, budget: i32) -> SeriesW {
    // Σ_{k≥0} w^k / k!.  Since lo(w) ≥ 1, w^k contributes only at powers ≥ k,
    // so k < budget suffices for `budget` significant terms.
    let mut acc = SeriesW::constant(1.0);
    let mut term = SeriesW::constant(1.0);
    for k in 1..budget {
        term = sw_mul(&term, w);
        term = sw_scale(&term, 1.0 / f64::from(k));
        term.cap_prec(budget);
        if term.is_zero() {
            break;
        }
        acc = sw_add(&acc, &term);
    }
    acc.cap_prec(budget);
    acc
}

/// `ln(1 + w)` for a series `w` with `lo_pow ≥ 1`, to `budget` relative terms.
fn sw_ln1p_small(w: &SeriesW, budget: i32) -> SeriesW {
    // Σ_{k≥1} (−1)^{k+1} w^k / k.
    let mut acc = SeriesW::zero();
    let mut term = SeriesW::constant(1.0);
    for k in 1..budget {
        term = sw_mul(&term, w);
        term.cap_prec(budget);
        if term.is_zero() {
            break;
        }
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        acc = sw_add(&acc, &sw_scale(&term, sign / f64::from(k)));
    }
    acc.cap_prec(budget);
    acc
}

/// `(sin(w), cos(w))` for a series `w` with `lo_pow ≥ 1`.
fn sw_sincos_small(w: &SeriesW, budget: i32) -> (SeriesW, SeriesW) {
    let mut sin = SeriesW::zero();
    let mut cos = SeriesW::constant(1.0);
    // term tracks w^n / n!; alternate its contribution to sin/cos with sign.
    let mut term = SeriesW::constant(1.0);
    for n in 1..budget {
        term = sw_mul(&term, w);
        term = sw_scale(&term, 1.0 / f64::from(n));
        term.cap_prec(budget);
        if term.is_zero() {
            break;
        }
        match n % 4 {
            1 => sin = sw_add(&sin, &term),
            2 => cos = sw_sub(&cos, &term),
            3 => sin = sw_sub(&sin, &term),
            _ => cos = sw_add(&cos, &term),
        }
    }
    sin.cap_prec(budget);
    cos.cap_prec(budget);
    (sin, cos)
}

/// `(sinh(w), cosh(w))` for a series `w` with `lo_pow ≥ 1`.
fn sw_sinhcosh_small(w: &SeriesW, budget: i32) -> (SeriesW, SeriesW) {
    let mut sinh = SeriesW::zero();
    let mut cosh = SeriesW::constant(1.0);
    let mut term = SeriesW::constant(1.0);
    for n in 1..budget {
        term = sw_mul(&term, w);
        term = sw_scale(&term, 1.0 / f64::from(n));
        term.cap_prec(budget);
        if term.is_zero() {
            break;
        }
        if n % 2 == 1 {
            sinh = sw_add(&sinh, &term);
        } else {
            cosh = sw_add(&cosh, &term);
        }
    }
    sinh.cap_prec(budget);
    cosh.cap_prec(budget);
    (sinh, cosh)
}

/// Split a series into (value at center `a0`, remainder `w` with `lo ≥ 1`),
/// returning `None` when the series has a genuine pole (`lo < 0`).
fn split_constant(u: &SeriesW) -> Option<(f64, SeriesW)> {
    let mut norm = u.clone();
    norm.strip_leading();
    if norm.is_zero() {
        return Some((0.0, SeriesW::zero()));
    }
    if norm.lo_pow < 0 {
        return None;
    }
    let a0 = u.coeff_at(0);
    let w = sw_sub_const(u, a0);
    Some((a0, w))
}

fn sw_exp(u: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let Some((a0, w)) = split_constant(u) else {
        return Err(EmlError::EssentialSingularity);
    };
    let scale = a0.exp();
    if !scale.is_finite() {
        return Err(EmlError::ExpOverflow(a0));
    }
    let e = sw_exp_small(&w, budget);
    Ok(sw_scale(&e, scale))
}

fn sw_ln(u: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let mut norm = u.clone();
    norm.strip_leading();
    if norm.is_zero() {
        return Err(EmlError::LnDomain(0.0));
    }
    if norm.lo_pow != 0 {
        // ln of a zero or a pole carries an ℓ·ln(t) term → branch point.
        return Err(EmlError::BranchPoint);
    }
    let lead = norm.coeffs[0];
    if lead <= 0.0 {
        return Err(EmlError::LnDomain(lead));
    }
    let w = sw_sub_const(&sw_scale(u, 1.0 / lead), 1.0);
    let series = sw_ln1p_small(&w, budget);
    Ok(sw_add(&SeriesW::constant(lead.ln()), &series))
}

fn sw_sin(u: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let Some((a0, w)) = split_constant(u) else {
        return Err(EmlError::EssentialSingularity);
    };
    let (sin_w, cos_w) = sw_sincos_small(&w, budget);
    Ok(sw_add(
        &sw_scale(&cos_w, a0.sin()),
        &sw_scale(&sin_w, a0.cos()),
    ))
}

fn sw_cos(u: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let Some((a0, w)) = split_constant(u) else {
        return Err(EmlError::EssentialSingularity);
    };
    let (sin_w, cos_w) = sw_sincos_small(&w, budget);
    Ok(sw_sub(
        &sw_scale(&cos_w, a0.cos()),
        &sw_scale(&sin_w, a0.sin()),
    ))
}

fn sw_sinh(u: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let Some((a0, w)) = split_constant(u) else {
        return Err(EmlError::EssentialSingularity);
    };
    let (sinh_w, cosh_w) = sw_sinhcosh_small(&w, budget);
    Ok(sw_add(
        &sw_scale(&cosh_w, a0.sinh()),
        &sw_scale(&sinh_w, a0.cosh()),
    ))
}

fn sw_cosh(u: &SeriesW, budget: i32) -> Result<SeriesW, EmlError> {
    let Some((a0, w)) = split_constant(u) else {
        return Err(EmlError::EssentialSingularity);
    };
    let (sinh_w, cosh_w) = sw_sinhcosh_small(&w, budget);
    Ok(sw_add(
        &sw_scale(&cosh_w, a0.cosh()),
        &sw_scale(&sinh_w, a0.sinh()),
    ))
}

// ---------------------------------------------------------------------------
// Expression-tree → series
// ---------------------------------------------------------------------------

/// Whether `op` structurally depends on `Var(wrt)`.
fn depends_on(op: &LoweredOp, wrt: usize) -> bool {
    match op {
        LoweredOp::Var(i) => *i == wrt,
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => false,
        LoweredOp::Neg(x)
        | LoweredOp::Exp(x)
        | LoweredOp::Ln(x)
        | LoweredOp::Sin(x)
        | LoweredOp::Cos(x)
        | LoweredOp::Tan(x)
        | LoweredOp::Sinh(x)
        | LoweredOp::Cosh(x)
        | LoweredOp::Tanh(x)
        | LoweredOp::Arcsin(x)
        | LoweredOp::Arccos(x)
        | LoweredOp::Arctan(x)
        | LoweredOp::Arcsinh(x)
        | LoweredOp::Arccosh(x)
        | LoweredOp::Arctanh(x)
        | LoweredOp::Erf(x)
        | LoweredOp::LGamma(x)
        | LoweredOp::Digamma(x)
        | LoweredOp::Trigamma(x)
        | LoweredOp::Ei(x)
        | LoweredOp::Si(x)
        | LoweredOp::Ci(x) => depends_on(x, wrt),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => depends_on(a, wrt) || depends_on(b, wrt),
    }
}

/// Evaluate an expression that does not depend on `wrt` to a scalar constant.
fn const_value(op: &LoweredOp, wrt: usize) -> f64 {
    let needed = (wrt + 1).max(op.count_vars()).max(1);
    let vars = vec![0.0_f64; needed];
    op.eval(&vars)
}

/// Series of `base^r` for a *constant* real exponent `r`.
fn series_pow_const(base: &SeriesW, r: f64, budget: i32) -> Result<SeriesW, EmlError> {
    // Integer exponent: exact via repeated multiplication / reciprocal.
    let rounded = r.round();
    if (r - rounded).abs() < 1e-12 {
        let n = rounded as i64;
        if n == 0 {
            return Ok(SeriesW::constant(1.0));
        }
        let prec_cap =
            budget.saturating_add(base.lo_pow.abs().saturating_mul(n.unsigned_abs() as i32));
        if n > 0 {
            return Ok(sw_pow_uint(base, n as u32, prec_cap));
        }
        let pos = sw_pow_uint(base, (-n) as u32, prec_cap);
        return sw_recip(&pos, budget);
    }

    // Non-integer exponent: base = lead · t^ℓ · (1 + w).
    let mut norm = base.clone();
    norm.strip_leading();
    if norm.is_zero() {
        // 0^r : 0 for r > 0, otherwise undefined.
        if r > 0.0 {
            return Ok(SeriesW::zero());
        }
        return Err(EmlError::NotSolvable);
    }
    let lead = norm.coeffs[0];
    let ell = norm.lo_pow;
    let t_power = f64::from(ell) * r;
    let t_power_round = t_power.round();
    if (t_power - t_power_round).abs() > 1e-9 {
        // Fractional power of t ⇒ algebraic branch point.
        return Err(EmlError::BranchPoint);
    }
    if lead < 0.0 {
        // Non-integer real power of a negative leading value ⇒ complex.
        return Err(EmlError::ComplexResult(0.0));
    }
    // (1 + w)^r via the binomial series, w with lo ≥ 1.
    let w = sw_sub_const(&sw_shift(&sw_scale(base, 1.0 / lead), -ell), 1.0);
    let mut acc = SeriesW::constant(1.0);
    let mut coeff = 1.0_f64;
    let mut term = SeriesW::constant(1.0);
    for j in 1..budget {
        coeff *= (r - f64::from(j - 1)) / f64::from(j);
        term = sw_mul(&term, &w);
        term.cap_prec(budget);
        if term.is_zero() {
            break;
        }
        acc = sw_add(&acc, &sw_scale(&term, coeff));
    }
    acc.cap_prec(budget);
    let shifted = sw_shift(&acc, t_power_round as i32);
    Ok(sw_scale(&shifted, lead.powf(r)))
}

/// Compute the truncated Laurent series of `op` about `center` in `Var(wrt)`,
/// carrying `budget` relative terms through transcendental leaves.
fn series_of(op: &LoweredOp, wrt: usize, center: f64, budget: i32) -> Result<SeriesW, EmlError> {
    match op {
        LoweredOp::Const(v) => Ok(SeriesW::constant(*v)),
        LoweredOp::NamedConst(nc) => Ok(SeriesW::constant(nc.value())),
        LoweredOp::Var(i) => {
            if *i == wrt {
                // z = center + t
                let mut s = SeriesW {
                    lo_pow: 0,
                    coeffs: vec![center, 1.0],
                    prec: EXACT,
                };
                s.strip_leading();
                Ok(s)
            } else {
                // Other variables are held at 0 (univariate convention).
                Ok(SeriesW::zero())
            }
        }
        LoweredOp::Neg(x) => Ok(sw_neg(&series_of(x, wrt, center, budget)?)),
        LoweredOp::Add(a, b) => Ok(sw_add(
            &series_of(a, wrt, center, budget)?,
            &series_of(b, wrt, center, budget)?,
        )),
        LoweredOp::Sub(a, b) => Ok(sw_sub(
            &series_of(a, wrt, center, budget)?,
            &series_of(b, wrt, center, budget)?,
        )),
        LoweredOp::Mul(a, b) => Ok(sw_mul(
            &series_of(a, wrt, center, budget)?,
            &series_of(b, wrt, center, budget)?,
        )),
        LoweredOp::Div(a, b) => sw_div(
            &series_of(a, wrt, center, budget)?,
            &series_of(b, wrt, center, budget)?,
            budget,
        ),
        LoweredOp::Exp(x) => sw_exp(&series_of(x, wrt, center, budget)?, budget),
        LoweredOp::Ln(x) => sw_ln(&series_of(x, wrt, center, budget)?, budget),
        LoweredOp::Sin(x) => sw_sin(&series_of(x, wrt, center, budget)?, budget),
        LoweredOp::Cos(x) => sw_cos(&series_of(x, wrt, center, budget)?, budget),
        LoweredOp::Tan(x) => {
            let arg = series_of(x, wrt, center, budget)?;
            let sin = sw_sin(&arg, budget)?;
            let cos = sw_cos(&arg, budget)?;
            sw_div(&sin, &cos, budget)
        }
        LoweredOp::Sinh(x) => sw_sinh(&series_of(x, wrt, center, budget)?, budget),
        LoweredOp::Cosh(x) => sw_cosh(&series_of(x, wrt, center, budget)?, budget),
        LoweredOp::Tanh(x) => {
            let arg = series_of(x, wrt, center, budget)?;
            let sinh = sw_sinh(&arg, budget)?;
            let cosh = sw_cosh(&arg, budget)?;
            sw_div(&sinh, &cosh, budget)
        }
        LoweredOp::Pow(base, exp) => {
            let base_s = series_of(base, wrt, center, budget)?;
            if depends_on(exp, wrt) {
                // a^b = exp(b · ln a).
                let ln_a = sw_ln(&base_s, budget)?;
                let exp_s = series_of(exp, wrt, center, budget)?;
                let prod = sw_mul(&exp_s, &ln_a);
                sw_exp(&prod, budget)
            } else {
                let r = const_value(exp, wrt);
                if !r.is_finite() {
                    return Err(EmlError::NotSolvable);
                }
                series_pow_const(&base_s, r, budget)
            }
        }
        // Series expansion is not implemented for these functions; callers fall
        // back to numeric methods rather than fabricating coefficients.
        LoweredOp::Arcsin(_)
        | LoweredOp::Arccos(_)
        | LoweredOp::Arctan(_)
        | LoweredOp::Arcsinh(_)
        | LoweredOp::Arccosh(_)
        | LoweredOp::Arctanh(_)
        | LoweredOp::Erf(_)
        | LoweredOp::LGamma(_)
        | LoweredOp::Digamma(_)
        | LoweredOp::Trigamma(_)
        | LoweredOp::Ei(_)
        | LoweredOp::Si(_)
        | LoweredOp::Ci(_) => Err(EmlError::NotSolvable),
    }
}

// ---------------------------------------------------------------------------
// Numeric cross-check
// ---------------------------------------------------------------------------

fn eval_at_wrt(op: &LoweredOp, wrt: usize, x: f64) -> f64 {
    let needed = (wrt + 1).max(op.count_vars()).max(1);
    let mut vars = vec![0.0_f64; needed];
    vars[wrt] = x;
    op.eval(&vars)
}

/// Verify a reconstructed series against the original expression at fixed probe
/// points near the center. Returns an error if they disagree beyond tolerance.
fn cross_check(orig: &LoweredOp, series: &Series, wrt: usize) -> Result<(), EmlError> {
    let mut checked = 0usize;
    for &h in &CROSS_CHECK_OFFSETS {
        let x = series.center + h;
        let fx = eval_at_wrt(orig, wrt, x);
        if !fx.is_finite() {
            continue;
        }
        let sx = series.eval(x);
        if !sx.is_finite() {
            return Err(EmlError::NotSolvable);
        }
        let scale = fx.abs().max(sx.abs()).max(1.0);
        if (fx - sx).abs() > CROSS_CHECK_TOL * scale {
            return Err(EmlError::NotSolvable);
        }
        checked += 1;
    }
    if checked == 0 {
        // The function was non-finite at every probe point: cannot corroborate.
        return Err(EmlError::UndefinedAtPoint(series.center));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API on LoweredOp
// ---------------------------------------------------------------------------

impl LoweredOp {
    /// Compute the Laurent series of `self` about `center` with respect to
    /// variable `wrt`, retaining every term up to and including power `order`.
    ///
    /// Unlike [`taylor`](Self::taylor), this succeeds at isolated singular
    /// centers by producing the principal part (negative powers) of a pole. See
    /// the [module documentation](crate::series_laurent) for the mathematics.
    ///
    /// # Errors
    ///
    /// - [`EmlError::BranchPoint`] if `center` is a branch point (e.g.
    ///   `ln(x).laurent(0, 0.0, n)` or `sqrt(x)` at `0`).
    /// - [`EmlError::EssentialSingularity`] if `center` is an essential
    ///   singularity (e.g. `exp(1/x)`, `sin(1/x)` at `0`).
    /// - [`EmlError::InvalidParameter`] if the apparent pole order exceeds the
    ///   cap `m ≤ 10`.
    /// - [`EmlError::LnDomain`] / [`EmlError::ComplexResult`] for real-domain
    ///   violations (log of a non-positive value, non-integer power of a
    ///   negative leading value).
    /// - [`EmlError::NotSolvable`] if a series cannot be built for a node
    ///   (unsupported special function) or the numeric cross-check fails.
    pub fn laurent(&self, wrt: usize, center: f64, order: usize) -> Result<Series, EmlError> {
        let simplified = self.simplify();
        let order_i = order as i32;

        // Working budgets, escalating if cancellation eats the precision.
        let budgets = [order_i + 24, order_i + 48, order_i + 96];
        let mut last_err: Option<EmlError> = None;

        for &budget in &budgets {
            let raw = match series_of(&simplified, wrt, center, budget) {
                Ok(s) => s,
                // Structural errors (branch point, essential, domain) will not
                // improve with more terms: report immediately.
                Err(
                    e @ (EmlError::BranchPoint
                    | EmlError::EssentialSingularity
                    | EmlError::LnDomain(_)
                    | EmlError::ComplexResult(_)),
                ) => return Err(e),
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };

            let mut norm = raw;
            norm.strip_leading();

            if -norm.lo_pow > MAX_POLE_ORDER {
                return Err(EmlError::InvalidParameter(
                    "pole order exceeds the cap m ≤ 10",
                ));
            }

            // Enough precision to reach the requested order?
            if norm.prec <= order_i {
                last_err = Some(EmlError::NotSolvable);
                continue;
            }

            let series = finalize(&norm, center, order_i);
            cross_check(&simplified, &series, wrt)?;
            return Ok(series);
        }

        Err(last_err.unwrap_or(EmlError::NotSolvable))
    }
}

/// Convert a normalised working series to a public [`Series`] truncated to
/// powers `≤ order`.
fn finalize(norm: &SeriesW, center: f64, order: i32) -> Series {
    let mut coeffs: Vec<f64> = Vec::new();
    let mut lo = norm.lo_pow;
    if norm.lo_pow > order {
        // Nothing survives truncation: represent as the zero function.
        return Series {
            center,
            lo_pow: 0,
            coeffs: vec![0.0],
        };
    }
    for (i, &c) in norm.coeffs.iter().enumerate() {
        let power = norm.lo_pow + i as i32;
        if power > order {
            break;
        }
        coeffs.push(c);
    }
    // Strip trailing zeros but always keep the leading coefficient.
    while coeffs.len() > 1 && coeffs.last() == Some(&0.0) {
        coeffs.pop();
    }
    if coeffs.is_empty() {
        coeffs.push(0.0);
        lo = 0;
    }
    Series {
        center,
        lo_pow: lo,
        coeffs,
    }
}

/// The verdict of a two-sided limit implied by a Laurent series about the
/// (finite) point. Used by [`crate::limit`].
///
/// Returns `None` when the series does not settle the limit (should not occur
/// for a valid finite-order Laurent series).
pub(crate) fn series_limit_verdict(series: &Series) -> Option<crate::limit::LimitResult> {
    use crate::limit::LimitResult;
    // Strip leading zeros to find the true lowest power / leading sign.
    let mut lo = series.lo_pow;
    let mut lead = 0.0_f64;
    for &c in &series.coeffs {
        if c != 0.0 {
            lead = c;
            break;
        }
        lo += 1;
    }
    if lead == 0.0 {
        // Series is identically zero to the retained order.
        return Some(LimitResult::Finite(0.0));
    }
    if lo >= 1 {
        return Some(LimitResult::Finite(0.0));
    }
    if lo == 0 {
        return Some(LimitResult::Finite(lead));
    }
    // lo < 0: a pole. Two-sided behaviour depends on parity of the power.
    if lo % 2 == 0 {
        Some(if lead > 0.0 {
            LimitResult::PosInf
        } else {
            LimitResult::NegInf
        })
    } else {
        Some(LimitResult::DoesNotExist)
    }
}

/// Compute the Laurent series and return the implied two-sided limit at the
/// finite point `center`. Returns `None` when no finite-order series exists
/// (branch point, essential singularity, unsupported node, cross-check
/// failure), leaving the caller to fall back to other strategies.
pub(crate) fn limit_from_series(
    op: &LoweredOp,
    wrt: usize,
    center: f64,
) -> Option<crate::limit::LimitResult> {
    let series = op.laurent(wrt, center, 2).ok()?;
    series_limit_verdict(&series)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var() -> LoweredOp {
        LoweredOp::Var(0)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn div(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Div(Arc::new(a), Arc::new(b))
    }
    fn sin(a: LoweredOp) -> LoweredOp {
        LoweredOp::Sin(Arc::new(a))
    }
    fn ln(a: LoweredOp) -> LoweredOp {
        LoweredOp::Ln(Arc::new(a))
    }
    fn exp(a: LoweredOp) -> LoweredOp {
        LoweredOp::Exp(Arc::new(a))
    }

    // ------------------------------------------------------------------
    // Spec bullet: laurent(1/x, 0) leading x⁻¹
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_one_over_x() {
        let expr = div(c(1.0), var());
        let s = expr.laurent(0, 0.0, 3).expect("laurent(1/x)");
        assert_eq!(s.lo_pow, -1, "leading power should be x⁻¹");
        assert!(
            (s.coeffs[0] - 1.0).abs() < 1e-9,
            "leading coefficient should be 1, got {}",
            s.coeffs[0]
        );
        assert_eq!(s.pole_order(), 1);
        // Higher coefficients are all zero: 1/x is its own Laurent series.
        for &co in &s.coeffs[1..] {
            assert!(co.abs() < 1e-9, "unexpected non-zero tail coeff {co}");
        }
    }

    // ------------------------------------------------------------------
    // Spec bullet: laurent(1/sin x, 0) → x⁻¹ + x/6
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_one_over_sin() {
        let expr = div(c(1.0), sin(var()));
        let s = expr.laurent(0, 0.0, 3).expect("laurent(1/sin x)");
        assert_eq!(s.lo_pow, -1, "1/sin x has a simple pole at 0");
        // coeffs are for powers −1, 0, 1, …  ⇒  x⁻¹ + 0 + x/6 + …
        assert!(
            (s.coeffs[0] - 1.0).abs() < 1e-9,
            "x⁻¹ coeff: {}",
            s.coeffs[0]
        );
        assert!(
            s.coeffs[1].abs() < 1e-9,
            "x⁰ coeff should be 0: {}",
            s.coeffs[1]
        );
        assert!(
            (s.coeffs[2] - 1.0 / 6.0).abs() < 1e-9,
            "x¹ coeff should be 1/6, got {}",
            s.coeffs[2]
        );
    }

    // ------------------------------------------------------------------
    // Spec bullet: laurent(ln x, 0) → Err(BranchPoint)
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_ln_branch_point() {
        let expr = ln(var());
        let result = expr.laurent(0, 0.0, 3);
        assert!(
            matches!(result, Err(EmlError::BranchPoint)),
            "ln(x) at 0 must be a branch point, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // Essential singularity: exp(1/x) at 0
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_exp_inv_x_essential() {
        let expr = exp(div(c(1.0), var()));
        let result = expr.laurent(0, 0.0, 3);
        assert!(
            matches!(result, Err(EmlError::EssentialSingularity)),
            "exp(1/x) at 0 must be an essential singularity, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // sin(1/x) at 0 is an essential singularity too.
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_sin_inv_x_essential() {
        let expr = sin(div(c(1.0), var()));
        let result = expr.laurent(0, 0.0, 3);
        assert!(
            matches!(result, Err(EmlError::EssentialSingularity)),
            "sin(1/x) at 0 must be an essential singularity, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // Regular Taylor case still works via laurent: sin(x)/x → 1 − x²/6 + …
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_sinc_regular() {
        let expr = div(sin(var()), var());
        let s = expr.laurent(0, 0.0, 4).expect("laurent(sin x / x)");
        assert_eq!(s.lo_pow, 0, "sin x / x is regular at 0");
        assert!(
            (s.coeffs[0] - 1.0).abs() < 1e-9,
            "constant term 1: {}",
            s.coeffs[0]
        );
        assert!(
            (s.coeffs[2] + 1.0 / 6.0).abs() < 1e-9,
            "x² coeff should be −1/6, got {}",
            s.coeffs[2]
        );
    }

    // ------------------------------------------------------------------
    // Double pole: 1/x² at 0 → lo_pow = −2.
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_double_pole() {
        let expr = div(c(1.0), LoweredOp::Pow(Arc::new(var()), Arc::new(c(2.0))));
        let s = expr.laurent(0, 0.0, 2).expect("laurent(1/x²)");
        assert_eq!(s.lo_pow, -2);
        assert_eq!(s.pole_order(), 2);
        assert!((s.coeffs[0] - 1.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // sqrt(x) at 0 is an algebraic branch point.
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_sqrt_branch_point() {
        let expr = LoweredOp::Pow(Arc::new(var()), Arc::new(c(0.5)));
        let result = expr.laurent(0, 0.0, 3);
        assert!(
            matches!(result, Err(EmlError::BranchPoint)),
            "sqrt(x) at 0 must be a branch point, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // Series reconstruction round-trips through eval.
    // ------------------------------------------------------------------
    #[test]
    fn test_series_eval_matches() {
        let expr = div(c(1.0), sin(var()));
        let s = expr.laurent(0, 0.0, 5).expect("laurent");
        // Away from the pole the truncated series approximates 1/sin x well.
        for &x in &[0.2_f64, -0.2, 0.1, -0.1] {
            let approx = s.eval(x);
            let exact = 1.0 / x.sin();
            assert!(
                (approx - exact).abs() < 1e-4,
                "series eval mismatch at {x}: {approx} vs {exact}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Taylor about a non-zero center via laurent: expand 1/(x-2) about 0.
    // ------------------------------------------------------------------
    #[test]
    fn test_laurent_geometric_regular() {
        // 1/(1 − x) = 1 + x + x² + … about 0.
        let expr = div(c(1.0), LoweredOp::Sub(Arc::new(c(1.0)), Arc::new(var())));
        let s = expr.laurent(0, 0.0, 4).expect("laurent(1/(1-x))");
        assert_eq!(s.lo_pow, 0);
        for &co in &s.coeffs {
            assert!((co - 1.0).abs() < 1e-9, "each coeff should be 1, got {co}");
        }
    }
}
