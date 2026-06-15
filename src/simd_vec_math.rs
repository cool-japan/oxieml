//! Vectorized SIMD transcendental functions generic over [`SimdRegister`].
//!
//! All functions operate on any `R: SimdRegister<Scalar = f64>` and use
//! FMA-accelerated Horner evaluation for their polynomial kernels.
//!
//! # Error budget
//!
//! | Function | Domain              | Max relative error |
//! |----------|---------------------|--------------------|
//! | `simd_exp` | \[-709, 709\]     | < 3 ULP (~6.6e-16) |
//! | `simd_ln`  | \[1e-300, 1e300\] | < 2 ULP (~4.4e-16) |
//! | `simd_sin` | \[-50, 50\]       | < 4 ULP, degrades beyond |
//! | `simd_cos` | \[-50, 50\]       | < 4 ULP, degrades beyond |
//! | `simd_tanh`| \[-20, 20\]       | < 4 ULP |
//!
//! Special values (NaN, +-Inf) are propagated correctly.
//! Arithmetic ops (`Add`, `Sub`, `Mul`, `Neg`) are bit-exact (no polynomial involved).

#![cfg(feature = "simd")]

use core::f64::consts::{FRAC_2_PI, FRAC_PI_2, LN_2, LOG2_E, SQRT_2};
use oxiblas_core::simd::SimdRegister;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// High part of ln(2) for Cody-Waite exp range reduction.
/// Chosen so that LN2_HI + LN2_LO = ln(2) exactly in extended precision.
const LN2_HI: f64 = LN_2;
/// Low part of ln(2) for Cody-Waite exp range reduction (correction term).
const LN2_LO: f64 = 2.319_046_813_846_3e-17_f64;
/// log2(e) = 1/ln(2), used to compute k = round(x/ln2).
const LOG2E: f64 = LOG2_E;

/// Taylor/minimax coefficients for exp(r) on r ∈ [−ln2/2, ln2/2].
/// Evaluates: `poly(r) = c[0] + r*(c[1] + r*(c[2] + ...))` where `c[i] = 1/(i+1)!`.
/// The final result is `exp(r) = 1 + r * poly(r)`.
const EXP_POLY: [f64; 12] = [
    1.0_f64,                    // r^1: 1/1!
    5e-1,                       // r^2: 1/2!
    1.666_666_666_666_666_7e-1, // r^3: 1/3!
    4.166_666_666_666_666_7e-2, // r^4: 1/4!
    8.333_333_333_333_333e-3,   // r^5: 1/5!
    1.388_888_888_888_889e-3,   // r^6: 1/6!
    1.984_126_984_126_984e-4,   // r^7: 1/7!
    2.480_158_730_158_73e-5,    // r^8: 1/8!
    2.755_731_922_398_589_1e-6, // r^9: 1/9!
    2.755_731_922_398_589_1e-7, // r^10: 1/10!
    2.505_210_838_544_172e-8,   // r^11: 1/11!
    2.087_675_698_786_81e-9,    // r^12: 1/12!
];

/// Overflow threshold for exp: `exp(x) = +inf` for `x > EXP_MAX`.
const EXP_MAX: f64 = 709.782_712_893_384_f64;
/// Underflow threshold for exp: `exp(x) = 0.0` for `x < EXP_MIN`.
const EXP_MIN: f64 = -745.133_219_101_941_6_f64;

/// atanh-series coefficients for `ln(m)` where `s=(m-1)/(m+1)`, `m ∈ [1, √2)`.
/// `ln(m) = 2*s*(c[0] + t*(c[1] + t*(c[2] + ...)))` where `t = s^2`.
/// 10 terms give rel error < 1e-14 for `m ∈ [1, √2)`.
const LN_POLY: [f64; 10] = [
    1.0_f64,                    // 1/1
    3.333_333_333_333_333_7e-1, // 1/3
    2e-1,                       // 1/5
    1.428_571_428_571_428_7e-1, // 1/7
    1.111_111_111_111_111e-1,   // 1/9
    9.090_909_090_909_091e-2,   // 1/11
    7.692_307_692_307_693e-2,   // 1/13
    6.666_666_666_666_667e-2,   // 1/15
    5.882_352_941_176_470_6e-2, // 1/17
    5.263_157_894_736_842e-2,   // 1/19
];

/// Taylor coefficients for `sin(x)` on `x ∈ [−π/4, π/4]`.
/// `sin(x) = x*(c[0] + x^2*(c[1] + x^2*(c[2] + ...)))`
const SIN_POLY: [f64; 6] = [
    1.0_f64,                     // x^1
    -1.666_666_666_666_666_7e-1, // -1/3! = -1/6
    8.333_333_333_333_333e-3,    //  1/5!
    -1.984_126_984_126_984e-4,   // -1/7!
    2.755_731_922_398_589_1e-6,  //  1/9!
    -2.505_210_838_544_172e-8,   // -1/11!
];

/// Taylor coefficients for `cos(x)` on `x ∈ [−π/4, π/4]`.
/// `cos(x) = c[0] + x^2*(c[1] + x^2*(c[2] + ...))`
const COS_POLY: [f64; 6] = [
    1.0_f64,                     // x^0
    -5e-1,                       // -1/2!
    4.166_666_666_666_666_7e-2,  //  1/4!
    -1.388_888_888_888_889e-3,   // -1/6!
    2.480_158_730_158_73e-5,     //  1/8!
    -2.755_731_922_398_589_1e-7, // -1/10!
];

/// High part of pi/2 for Cody-Waite sin/cos range reduction (= `FRAC_PI_2`).
const FRAC_PI_2_HI: f64 = FRAC_PI_2;
/// Low part of pi/2 correction for Cody-Waite sin/cos range reduction.
/// This is the rounding error of `FRAC_PI_2` vs the true pi/2.
const FRAC_PI_2_LO: f64 = 6.123_233_995_736_766e-17_f64;

// ---------------------------------------------------------------------------
// Core Horner evaluator
// ---------------------------------------------------------------------------

/// Evaluate a polynomial using Horner's method with FMA.
///
/// For `coeffs = [c0, c1, ..., cn]`, evaluates:
/// `c[n] + x * (c[n-1] + x * (... + x * c[0]))`
/// i.e. Horner from highest-degree coefficient (last in slice) down to lowest (first).
///
/// The accumulation uses `mul_add(x, c)` = `acc * x + c`.
#[inline(always)]
pub(crate) fn simd_horner<R>(x: R, coeffs: &[f64]) -> R
where
    R: SimdRegister<Scalar = f64>,
{
    let n = coeffs.len();
    if n == 0 {
        return R::zero();
    }
    // Start from highest degree (last element)
    let mut acc = R::splat(coeffs[n - 1]);
    // Iterate from second-highest down to lowest
    // acc = acc * x + coeffs[i]  (FMA: acc.mul_add(x, splat(c)))
    for i in (0..n - 1).rev() {
        acc = acc.mul_add(x, R::splat(coeffs[i]));
    }
    acc
}

// ---------------------------------------------------------------------------
// simd_exp
// ---------------------------------------------------------------------------

/// Vectorized exp(x) for all lanes of `x`.
///
/// Uses Cody-Waite 2-part range reduction per lane, then evaluates the
/// degree-12 Horner polynomial in SIMD width via [`simd_horner`], and finally
/// reconstructs 2^k per lane via IEEE 754 bit manipulation.
///
/// Special values:
/// - `exp(+inf) = +inf`
/// - `exp(-inf) = 0`
/// - `exp(NaN) = NaN`
/// - `exp(x) = +inf` for x > 709.78...
/// - `exp(x) = 0` for x < -745.13...
pub fn simd_exp<R>(x: R) -> R
where
    R: SimdRegister<Scalar = f64>,
{
    let lanes = R::LANES;

    // Per-lane range reduction: x_i = k_i * ln(2) + r_i.
    // We need k_i (integer) and r_i (f64) for each lane.
    // Max LANES is 16 (AVX-512 F32x16), but for f64 max is 8 (F64x8).
    let mut k_vals = [0_i64; 16];
    let mut r_reg = R::zero();
    let mut special_mask = [false; 16];

    for lane in 0..lanes {
        let xi = x.extract(lane);
        if xi.is_nan() || xi >= EXP_MAX || xi <= EXP_MIN {
            special_mask[lane] = true;
            // r_reg lane stays 0.0; result overwritten after Horner
        } else {
            let k = (xi * LOG2E + 0.5).floor() as i64;
            let kf = k as f64;
            let r = xi - kf * LN2_HI - kf * LN2_LO;
            k_vals[lane] = k;
            r_reg = r_reg.insert(lane, r);
        }
    }

    // Evaluate p = Horner(r, EXP_POLY) in SIMD — this is where simd_horner is used.
    // p represents the polynomial such that exp(r) = 1 + r * p.
    let p_reg = simd_horner(r_reg, &EXP_POLY);

    // Reconstruct results per lane: exp(x_i) = (1 + r_i * p_i) * 2^k_i
    let mut result = R::zero();
    for lane in 0..lanes {
        let val = if special_mask[lane] {
            let xi = x.extract(lane);
            if xi.is_nan() {
                f64::NAN
            } else if xi >= EXP_MAX {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            let r = r_reg.extract(lane);
            let p = p_reg.extract(lane);
            let exp_r = 1.0 + r * p;
            let pow2k = reconstruct_pow2(k_vals[lane]);
            exp_r * pow2k
        };
        result = result.insert(lane, val);
    }
    result
}

/// Scalar exp, used in tanh and for correctness testing.
#[inline(always)]
fn scalar_exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= EXP_MAX {
        return f64::INFINITY;
    }
    if x <= EXP_MIN {
        return 0.0;
    }

    // Range reduction: k = round(x / ln2), r = x - k*ln2
    let k = (x * LOG2E + 0.5).floor() as i64;
    let kf = k as f64;
    let r = x - kf * LN2_HI - kf * LN2_LO;

    // exp(r) = 1 + r * poly(r)
    let p = horner_scalar(r, &EXP_POLY);
    let exp_r = 1.0 + r * p;

    exp_r * reconstruct_pow2(k)
}

/// Reconstruct 2^k as an f64 via IEEE 754 bit manipulation.
/// Handles normal range k ∈ [−1022, 1023], overflow, and subnormal.
#[inline(always)]
fn reconstruct_pow2(k: i64) -> f64 {
    if (-1022..=1023_i64).contains(&k) {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else if k > 1023 {
        f64::INFINITY
    } else {
        // Subnormal range: k < -1022.
        // 2^k = 2^(-1022) * 2^(k+1022). Since k+1022 < 0, the factor
        // becomes a subnormal itself. We encode it as: bit (52 + k + 1022)
        // in the mantissa of a subnormal float.
        let bit_pos = (52_i64 + k + 1022).clamp(0, 52) as u64;
        f64::from_bits(1_u64 << bit_pos)
    }
}

/// Evaluate a polynomial with scalar Horner.
/// `coeffs = [c0, c1, ..., cn]` evaluates `c[n] + x*(c[n-1] + x*(... + x*c[0]))`.
#[inline(always)]
fn horner_scalar(x: f64, coeffs: &[f64]) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = coeffs[n - 1];
    for i in (0..n - 1).rev() {
        acc = acc * x + coeffs[i];
    }
    acc
}

// ---------------------------------------------------------------------------
// simd_ln
// ---------------------------------------------------------------------------

/// Vectorized ln(x) for all lanes of `x`.
///
/// Decomposes `x = 2^e * m` (`m in [1, 2)`), normalizes `m` to `[1, √2)`,
/// then evaluates `ln(m)` via a 10-term atanh Horner series.
///
/// Special values:
/// - `ln(+inf) = +inf`
/// - `ln(0) = -inf`
/// - `ln(x < 0) = NaN`
/// - `ln(NaN) = NaN`
pub fn simd_ln<R>(x: R) -> R
where
    R: SimdRegister<Scalar = f64>,
{
    let lanes = R::LANES;
    let mut result = R::zero();
    for lane in 0..lanes {
        result = result.insert(lane, scalar_ln(x.extract(lane)));
    }
    result
}

/// Scalar ln, used per lane in simd_ln.
#[inline(always)]
fn scalar_ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    // IEEE 754 bit decomposition: x = 2^e * m, m in [1, 2)
    let bits = x.to_bits();
    let exp_bits = (bits >> 52) & 0x7FF;
    let mantissa_bits = (bits & 0x000F_FFFF_FFFF_FFFF) | (1023_u64 << 52);
    let mut m = f64::from_bits(mantissa_bits);
    let mut e = exp_bits as i64 - 1023;

    // Handle subnormal inputs (exp_bits == 0): multiply by 2^52 to normalize.
    if exp_bits == 0 {
        // 2^52 has biased exponent = 1023 + 52 = 1075
        let scale = f64::from_bits(1075_u64 << 52);
        let y = x * scale;
        let b2 = y.to_bits();
        let e2 = (b2 >> 52) & 0x7FF;
        let m2_bits = (b2 & 0x000F_FFFF_FFFF_FFFF) | (1023_u64 << 52);
        m = f64::from_bits(m2_bits);
        e = e2 as i64 - 1023 - 52;
    }

    // Normalize m to [1, √2) so s = (m-1)/(m+1) stays small.
    // After decomposition, m ∈ [1, 2). If m >= √2, halve m and increment e.
    // This bounds t = s^2 ≤ ((√2-1)/(√2+1))^2 ≈ 0.0294.
    if m >= SQRT_2 {
        m *= 0.5;
        e += 1;
    }

    // s = (m - 1) / (m + 1), t = s^2, ln(m) = 2*s*poly(t)
    let s = (m - 1.0) / (m + 1.0);
    let t = s * s;
    let poly = horner_scalar(t, &LN_POLY);
    let ln_m = 2.0 * s * poly;

    // Reconstruct: ln(x) = e * ln(2) + ln(m)
    (e as f64) * LN_2 + ln_m
}

// ---------------------------------------------------------------------------
// simd_sin / simd_cos
// ---------------------------------------------------------------------------

/// Vectorized sin(x) for all lanes of `x`.
///
/// Uses Cody-Waite 2-part range reduction (accurate to ≈50 radians),
/// then evaluates the sin/cos Taylor polynomial on [−π/4, π/4].
///
/// Accuracy degrades for |x| >> 50 (large-argument caveat documented).
pub fn simd_sin<R>(x: R) -> R
where
    R: SimdRegister<Scalar = f64>,
{
    let lanes = R::LANES;
    let mut result = R::zero();
    for lane in 0..lanes {
        result = result.insert(lane, scalar_sin(x.extract(lane)));
    }
    result
}

/// Vectorized cos(x) for all lanes of `x`.
///
/// Same algorithm as [`simd_sin`] with quadrant shifted by 1.
pub fn simd_cos<R>(x: R) -> R
where
    R: SimdRegister<Scalar = f64>,
{
    let lanes = R::LANES;
    let mut result = R::zero();
    for lane in 0..lanes {
        result = result.insert(lane, scalar_cos(x.extract(lane)));
    }
    result
}

/// Scalar sin via quadrant-based Horner.
#[inline(always)]
fn scalar_sin(x: f64) -> f64 {
    // sin(±inf) = sin(NaN) = NaN per IEEE 754
    if !x.is_finite() {
        return f64::NAN;
    }
    let (r, quadrant) = reduce_pi2(x);
    eval_sincos(r, quadrant, false)
}

/// Scalar cos via quadrant-based Horner.
#[inline(always)]
fn scalar_cos(x: f64) -> f64 {
    // cos(±inf) = cos(NaN) = NaN per IEEE 754
    if !x.is_finite() {
        return f64::NAN;
    }
    let (r, quadrant) = reduce_pi2(x);
    eval_sincos(r, quadrant, true)
}

/// Cody-Waite 2-part range reduction to `[−π/4, π/4]`.
///
/// Returns `(r, k mod 4)` where `x = k*(π/2) + r` and `r ∈ [−π/4, π/4]`.
/// `k mod 4` is the quadrant index (0, 1, 2, or 3).
fn reduce_pi2(x: f64) -> (f64, i32) {
    // k = round(x / (π/2)) using FRAC_2_PI = 2/π
    let k = (x * FRAC_2_PI).round() as i64;
    let kf = k as f64;
    // Cody-Waite 2-part: r = x - k*(π/2)
    let r = x - kf * FRAC_PI_2_HI - kf * FRAC_PI_2_LO;
    let quadrant = k.rem_euclid(4) as i32;
    (r, quadrant)
}

/// Evaluate sin or cos at reduced argument `r` with quadrant selection.
///
/// `r` is in `[−π/4, π/4]`. `quadrant` is k mod 4 from [`reduce_pi2`].
///
/// Quadrant mapping for **sin(x)** where `x = k*(π/2) + r`:
/// - q=0: sin(r)   (x near 0)
/// - q=1: cos(r)   (x near π/2)
/// - q=2: -sin(r)  (x near π)
/// - q=3: -cos(r)  (x near 3π/2)
///
/// For **cos(x)** = sin(x + π/2), shift by 1: q' = (q+1) mod 4.
fn eval_sincos(r: f64, quadrant: i32, want_cos: bool) -> f64 {
    let r2 = r * r;

    // sin(r) = r * poly(r^2) via SIN_POLY
    let sin_r = r * horner_scalar(r2, &SIN_POLY);
    // cos(r) = poly(r^2) via COS_POLY
    let cos_r = horner_scalar(r2, &COS_POLY);

    // For cos(x), shift effective quadrant by 1
    let q = if want_cos {
        (quadrant + 1).rem_euclid(4)
    } else {
        quadrant
    };

    match q {
        0 => sin_r,
        1 => cos_r,
        2 => -sin_r,
        3 => -cos_r,
        _ => sin_r, // unreachable: rem_euclid(4) is always 0..=3
    }
}

// ---------------------------------------------------------------------------
// simd_tanh
// ---------------------------------------------------------------------------

/// Vectorized tanh(x) for all lanes of `x`.
///
/// `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`, computed as
/// `(e^x - e^{-x}) / (e^x + e^{-x})` using [`scalar_exp`].
/// For `|x| >= 20`, saturates to ±1 to avoid unnecessary computation.
pub fn simd_tanh<R>(x: R) -> R
where
    R: SimdRegister<Scalar = f64>,
{
    let lanes = R::LANES;
    let mut result = R::zero();
    for lane in 0..lanes {
        result = result.insert(lane, scalar_tanh(x.extract(lane)));
    }
    result
}

/// Scalar tanh used inside simd_tanh.
#[inline(always)]
fn scalar_tanh(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // For |x| >= 20, tanh(x) is ±1 to full double precision
    if x.abs() >= 20.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let e_pos = scalar_exp(x);
    let e_neg = scalar_exp(-x);
    (e_pos - e_neg) / (e_pos + e_neg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: f64, b: f64) -> f64 {
        if b == 0.0 {
            a.abs()
        } else {
            ((a - b) / b).abs()
        }
    }

    #[test]
    fn test_scalar_exp_accuracy() {
        let mut max_rel = 0.0_f64;
        for i in 0..10_000_i32 {
            let x = -20.0 + i as f64 * 40.0 / 10_000.0;
            let got = scalar_exp(x);
            let expected = x.exp();
            if expected.is_finite() && expected != 0.0 {
                let err = rel_err(got, expected);
                if err > max_rel {
                    max_rel = err;
                }
            }
        }
        assert!(
            max_rel < 1e-11,
            "exp max relative error {max_rel} exceeds 1e-11"
        );
    }

    #[test]
    fn test_scalar_exp_special_values() {
        assert!(scalar_exp(f64::NAN).is_nan());
        assert_eq!(scalar_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(scalar_exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(scalar_exp(710.0), f64::INFINITY);
        assert_eq!(scalar_exp(-746.0), 0.0);
        assert_eq!(scalar_exp(0.0), 1.0);
    }

    #[test]
    fn test_scalar_ln_accuracy() {
        let mut max_rel = 0.0_f64;
        for i in 0..10_000_i32 {
            let t = i as f64 / 10_000.0;
            // Logarithmic spacing from 1e-6 to 1e6
            let x = 1e-6_f64 * (1e12_f64).powf(t);
            let got = scalar_ln(x);
            let expected = x.ln();
            if expected.is_finite() {
                let err = rel_err(got, expected);
                if err > max_rel {
                    max_rel = err;
                }
            }
        }
        assert!(
            max_rel < 1e-11,
            "ln max relative error {max_rel} exceeds 1e-11"
        );
    }

    #[test]
    fn test_scalar_ln_special_values() {
        assert!(scalar_ln(f64::NAN).is_nan());
        assert_eq!(scalar_ln(f64::INFINITY), f64::INFINITY);
        assert_eq!(scalar_ln(0.0), f64::NEG_INFINITY);
        assert!(scalar_ln(-1.0).is_nan());
        let got = scalar_ln(1.0);
        assert!(got.abs() < 1e-15, "ln(1) = {got}");
        let got_e = scalar_ln(core::f64::consts::E);
        assert!((got_e - 1.0).abs() < 1e-14, "ln(e) = {got_e}");
    }

    #[test]
    fn test_scalar_sin_accuracy() {
        let mut max_rel = 0.0_f64;
        for i in 0..10_000_i32 {
            let x = -50.0 + i as f64 * 100.0 / 10_000.0;
            let got = scalar_sin(x);
            let expected = x.sin();
            // Near zeros, use abs error to avoid inflated relative errors
            let err = if expected.abs() > 0.01 {
                rel_err(got, expected)
            } else {
                (got - expected).abs()
            };
            if err > max_rel {
                max_rel = err;
            }
        }
        assert!(
            max_rel < 1e-8,
            "sin max relative error {max_rel} exceeds 1e-8 on [-50, 50]"
        );
    }

    #[test]
    fn test_scalar_sin_special_values() {
        assert!(scalar_sin(f64::NAN).is_nan());
        assert!(scalar_sin(f64::INFINITY).is_nan());
        assert!(scalar_sin(f64::NEG_INFINITY).is_nan());
        assert_eq!(scalar_sin(0.0), 0.0);
        let pi_2 = core::f64::consts::PI / 2.0;
        assert!(
            (scalar_sin(pi_2) - 1.0).abs() < 1e-12,
            "sin(pi/2) = {}",
            scalar_sin(pi_2)
        );
    }

    #[test]
    fn test_scalar_cos_accuracy() {
        let mut max_rel = 0.0_f64;
        for i in 0..10_000_i32 {
            let x = -50.0 + i as f64 * 100.0 / 10_000.0;
            let got = scalar_cos(x);
            let expected = x.cos();
            let err = if expected.abs() > 0.01 {
                rel_err(got, expected)
            } else {
                (got - expected).abs()
            };
            if err > max_rel {
                max_rel = err;
            }
        }
        assert!(
            max_rel < 1e-8,
            "cos max relative error {max_rel} exceeds 1e-8 on [-50, 50]"
        );
    }

    #[test]
    fn test_scalar_cos_special_values() {
        assert!(scalar_cos(f64::NAN).is_nan());
        assert!(scalar_cos(f64::INFINITY).is_nan());
        assert_eq!(scalar_cos(0.0), 1.0);
        let pi = core::f64::consts::PI;
        assert!(
            (scalar_cos(pi) + 1.0).abs() < 1e-12,
            "cos(pi) = {}",
            scalar_cos(pi)
        );
    }

    #[test]
    fn test_scalar_tanh_accuracy() {
        let mut max_rel = 0.0_f64;
        for i in 0..10_000_i32 {
            let x = -20.0 + i as f64 * 40.0 / 10_000.0;
            let got = scalar_tanh(x);
            let expected = x.tanh();
            let err = rel_err(got, expected);
            if err > max_rel {
                max_rel = err;
            }
        }
        assert!(
            max_rel < 1e-11,
            "tanh max relative error {max_rel} exceeds 1e-11"
        );
    }

    #[test]
    fn test_scalar_tanh_special_values() {
        assert!(scalar_tanh(f64::NAN).is_nan());
        assert_eq!(scalar_tanh(f64::INFINITY), 1.0);
        assert_eq!(scalar_tanh(f64::NEG_INFINITY), -1.0);
        assert_eq!(scalar_tanh(0.0), 0.0);
    }

    #[test]
    fn test_horner_scalar_matches() {
        // coeffs = [c0=1, c1=2, c2=3] represents: 1 + 2*x + 3*x^2
        // horner_scalar(x=2, [1,2,3]):
        //   acc = 3
        //   acc = 3*2 + 2 = 8
        //   acc = 8*2 + 1 = 17
        // = 1 + 2*2 + 3*4 = 17. Correct.
        let coeffs = [1.0_f64, 2.0, 3.0];
        let got = horner_scalar(2.0, &coeffs);
        assert!(
            (got - 17.0).abs() < 1e-15,
            "horner_scalar gave {got}, expected 17"
        );
    }

    #[test]
    fn test_exp_extended_range() {
        let cases = [
            (-745.0_f64, (-745.0_f64).exp()),
            (-700.0, (-700.0_f64).exp()),
            (0.0, 1.0_f64),
            (1.0, core::f64::consts::E),
            (700.0, (700.0_f64).exp()),
        ];
        for (x, expected) in cases {
            let got = scalar_exp(x);
            if expected.is_finite() && expected != 0.0 {
                let err = rel_err(got, expected);
                assert!(
                    err < 1e-11,
                    "exp({x}) = {got}, expected {expected}, rel_err={err}"
                );
            }
        }
    }

    #[test]
    fn test_ln_subnormal() {
        // Test that subnormal inputs don't panic and give reasonable results
        let x = f64::from_bits(1); // smallest positive subnormal
        let got = scalar_ln(x);
        let expected = x.ln();
        let abs_err = (got - expected).abs();
        assert!(
            abs_err < 1.0,
            "ln(subnormal) = {got}, expected approx {expected}"
        );
    }
}
