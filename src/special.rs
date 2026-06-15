//! Special mathematical functions (erf, lgamma, digamma, Ei, Si, Ci).
//! All implementations are pure Rust with no external dependencies.

use std::f64::consts;

/// Error function erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x.signum();
    }
    let ax = x.abs();
    let result = if ax < 1e-8 {
        ax * consts::FRAC_2_SQRT_PI
    } else if ax < 5.0 {
        // Taylor series: erf(x) = (2/√π) Σ_{n=0}^∞ (-1)^n x^{2n+1} / (n!(2n+1))
        // Term t_0 = x, t_n = t_{n-1} * (-x^2) * (2n-1) / (n * (2n+1))
        let xsq = ax * ax;
        let mut term = ax; // (-1)^n * x^{2n+1} / n!  at n=0: x
        let mut sum = ax / 1.0_f64; // t_0 / (2*0+1) = x
        for n in 1..100usize {
            term *= -xsq / n as f64;
            let contrib = term / (2 * n + 1) as f64;
            sum += contrib;
            if contrib.abs() < sum.abs().max(1e-300) * f64::EPSILON {
                break;
            }
        }
        consts::FRAC_2_SQRT_PI * sum
    } else {
        let erfc_val = erfc(ax);
        1.0 - erfc_val
    };
    if x < 0.0 { -result } else { result }
}

/// Complementary error function erfc(x) = 1 - erf(x)
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x > 27.0 {
        return 0.0;
    }
    if x < 0.5 {
        return 1.0 - erf(x);
    }
    // Numerical Recipes approximation (accurate to ~1.2e-7 for the coefficients below)
    // Use Chebyshev approach from Abramowitz & Stegun 7.1.26
    let t = 1.0 / (1.0 + 0.5 * x);
    let poly = t
        * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_94)))))))));
    poly.exp() * t
}

/// Natural log of the Gamma function.
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        if x.fract() == 0.0 {
            return f64::INFINITY;
        }
        // Reflection: Γ(x)Γ(1-x) = π/sin(πx)
        let lnpi = consts::PI.ln();
        let sinpix = (consts::PI * x).sin().abs();
        return lnpi - sinpix.ln() - lgamma(1.0 - x);
    }
    if x < 0.5 {
        let lnpi = consts::PI.ln();
        let sinpix = (consts::PI * x).sin().abs();
        return lnpi - sinpix.ln() - lgamma(1.0 - x);
    }
    // Lanczos g=7, n=9 (Godfrey 2001)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_9,
        -0.138_571_095_265_72,
        9.984_369_578_019_57e-6,
        1.505_632_735_149_31e-7,
    ];
    let xm1 = x - 1.0;
    let mut a = C[0];
    for (k, &c) in C[1..].iter().enumerate() {
        a += c / (xm1 + (k as f64) + 1.0);
    }
    let t = xm1 + G + 0.5;
    (2.0 * consts::PI).sqrt().ln() + a.ln() + (xm1 + 0.5) * t.ln() - t
}

/// Digamma function ψ(x) = d/dx ln Γ(x)
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.5 {
        // Reflection: ψ(x) = ψ(1-x) - π/tan(πx)
        return digamma(1.0 - x) - consts::PI / (consts::PI * x).tan();
    }
    let mut x = x;
    let mut result = 0.0;
    // Shift x > 6 for asymptotic accuracy
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion
    let xinv = 1.0 / x;
    let xinv2 = xinv * xinv;
    result += x.ln()
        - 0.5 * xinv
        - xinv2
            * (1.0 / 12.0
                - xinv2
                    * (1.0 / 120.0
                        - xinv2 * (1.0 / 252.0 - xinv2 * (1.0 / 240.0 - xinv2 * (1.0 / 132.0)))));
    result
}

/// Trigamma function ψ¹(x) = d²/dx² ln Γ(x) = Σ_{k=0}^∞ 1/(x+k)²
///
/// Returns +∞ for non-positive integers, NaN for NaN input.
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Poles at non-positive integers
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    // Reflection formula: ψ¹(x) + ψ¹(1-x) = π²/sin²(πx)
    if x < 0.5 {
        let pi_x = std::f64::consts::PI * x;
        let s = pi_x.sin();
        return std::f64::consts::PI * std::f64::consts::PI / (s * s) - trigamma(1.0 - x);
    }
    // Recurrence: ψ¹(x) = ψ¹(x+1) + 1/x²  — shift until x >= 6
    if x < 6.0 {
        return trigamma(x + 1.0) + 1.0 / (x * x);
    }
    // Asymptotic expansion (x >= 6): 1/x + 1/(2x²) + Σ B_{2n}/x^{2n+1}
    // B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    inv_x
        + 0.5 * inv_x2
        + inv_x2
            * inv_x
            * (1.0 / 6.0
                + inv_x2
                    * (-1.0 / 30.0
                        + inv_x2 * (1.0 / 42.0 + inv_x2 * (-1.0 / 30.0 + inv_x2 * (5.0 / 66.0)))))
}

/// Exponential integral Ei(x).
/// Ei(x) = P.V. ∫_{-∞}^x e^t/t dt for x ≠ 0
pub fn ei(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -e1(-x);
    }
    if x > 40.0 {
        // Asymptotic: Ei(x) ≈ e^x/x * Σ n!/x^n
        let ex_over_x = x.exp() / x;
        let mut series = 1.0_f64;
        let mut term = 1.0_f64;
        for n in 1..30usize {
            term *= n as f64 / x;
            if term.abs() > 1e50 {
                break;
            }
            series += term;
            if term.abs() < series.abs() * 1e-15 {
                break;
            }
        }
        return ex_over_x * series;
    }
    // Series: Ei(x) = γ + ln(x) + Σ_{n=1}^∞ x^n/(n·n!)
    const EULER: f64 = 0.577_215_664_901_532_9;
    let mut series = 0.0_f64;
    let mut term = x;
    for n in 1..200usize {
        series += term / n as f64;
        term *= x / (n + 1) as f64;
        if term.abs() < (series.abs().max(1.0)) * 1e-16 {
            break;
        }
    }
    EULER + x.abs().ln() + series
}

/// E₁(x) = ∫_x^∞ e^{-t}/t dt for x > 0
fn e1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x <= 1.0 {
        // Series: E₁(x) = -γ - ln(x) - Σ_{n=1}^∞ (-x)^n/(n·n!)
        const EULER: f64 = 0.577_215_664_901_532_9;
        let mut series = 0.0_f64;
        let mut term = -x;
        for n in 1..200usize {
            series += term / n as f64;
            term *= -x / (n + 1) as f64;
            if term.abs() < (series.abs().max(1e-100)) * 1e-16 {
                break;
            }
        }
        return -EULER - x.ln() - series;
    }
    // x > 1: Lentz continued fraction for E₁
    e1_cf(x)
}

/// Continued fraction for E₁(x) via Lentz's algorithm (for x > 1).
fn e1_cf(x: f64) -> f64 {
    let tiny = 1e-300_f64;
    let mut f;
    let mut c;
    let mut d;

    d = 1.0 / x.max(tiny);
    f = d;
    c = 1.0 / tiny;

    for n in 1i64..200 {
        let half_n = (n + 1) / 2;
        let a = half_n as f64;
        let b = if n % 2 == 1 { 1.0 } else { x };
        d = 1.0 / (b + a * d);
        c = b + a / c;
        f *= c * d;
        if (c * d - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x).exp() * f
}

/// Sine integral Si(x) = ∫₀ˣ sin(t)/t dt
pub fn si(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let (si_val, _) = si_ci_internal(x.abs());
    if x < 0.0 { -si_val } else { si_val }
}

/// Cosine integral Ci(x) = γ + ln(x) + ∫₀ˣ (cos(t)-1)/t dt  (x > 0)
pub fn ci(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let (_, ci_val) = si_ci_internal(x);
    ci_val
}

/// Compute both Si(x) and Ci(x) for x >= 0.
fn si_ci_internal(x: f64) -> (f64, f64) {
    const EULER: f64 = 0.577_215_664_901_532_9;

    if x < 1e-8 {
        let ci_val = if x > 0.0 {
            EULER + x.ln()
        } else {
            f64::NEG_INFINITY
        };
        return (x, ci_val);
    }

    if x < 4.0 {
        // Power series for Si and Ci
        let xsq = x * x;
        // Si(x) = Σ_{n=0}^∞ (-1)^n x^{2n+1} / ((2n+1) * (2n+1)!)
        // Let t_n = (-1)^n x^{2n+1} / (2n+1)!   (without the extra (2n+1) denominator)
        // Recurrence: t_n = t_{n-1} * (-x^2) / ((2n) * (2n+1))
        // Contribution at n: t_n / (2n+1)
        let mut si_t = x; // t_0 = x / 1! = x
        let mut si_acc = x; // contribution at n=0: t_0 / 1 = x
        for n in 1..100usize {
            si_t *= -xsq / ((2 * n) as f64 * (2 * n + 1) as f64);
            let contrib = si_t / (2 * n + 1) as f64;
            si_acc += contrib;
            if contrib.abs() < si_acc.abs().max(1e-300) * f64::EPSILON {
                break;
            }
        }
        // Ci(x) = γ + ln(x) + Σ_{n=1}^∞ (-1)^n x^{2n}/((2n)(2n)!)
        // u_n = (-1)^n x^{2n} / (2n)!,  u_n = u_{n-1} * (-x^2)/((2n-1)*(2n))
        // contribution_n = u_n / (2n)
        // u_1 = -x^2/2!, contribution_1 = -x^2/(2*2) = -x^2/4
        let mut u_n = -xsq / 2.0; // u_1 = -x^2 / 2!
        let mut ci_sum = u_n / 2.0; // contribution at n=1: u_1/(2*1) = -x^2/4
        for n in 2..100usize {
            u_n *= -xsq / ((2 * n - 1) as f64 * (2 * n) as f64);
            let contrib = u_n / (2 * n) as f64;
            ci_sum += contrib;
            if contrib.abs() < ci_sum.abs().max(1e-300) * f64::EPSILON {
                break;
            }
        }
        let ci_val = EULER + x.ln() + ci_sum;
        return (si_acc, ci_val);
    }

    // x >= 4: auxiliary functions f, g
    // Si(x) = π/2 - f(x)cos(x) - g(x)sin(x)
    // Ci(x) = f(x)sin(x) - g(x)cos(x)
    let (f_val, g_val) = si_ci_aux(x);
    let sinx = x.sin();
    let cosx = x.cos();
    let si_val = consts::FRAC_PI_2 - f_val * cosx - g_val * sinx;
    let ci_val = f_val * sinx - g_val * cosx;
    (si_val, ci_val)
}

/// Auxiliary functions f(x) and g(x) for Si/Ci using asymptotic series.
fn si_ci_aux(x: f64) -> (f64, f64) {
    let xsq = x * x;
    // f ~ 1/x (1 - 2!/x^2 + 4!/x^4 - ...)
    // g ~ 1/x^2 (1 - 3!/x^2 + 5!/x^4 - ...)
    let mut f_sum = 1.0_f64;
    let mut g_sum = 1.0_f64;
    let mut f_term = 1.0_f64;
    let mut g_term = 1.0_f64;
    for n in 1..40usize {
        let even = (2 * n) as f64;
        let odd = (2 * n - 1) as f64;
        f_term *= -odd * even / xsq;
        g_term *= -(even) * (even + 1.0) / xsq;
        if f_term.abs() >= f_sum.abs() {
            break;
        }
        f_sum += f_term;
        if g_term.abs() < g_sum.abs() {
            g_sum += g_term;
        }
    }
    (f_sum / x, g_sum / xsq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_known_values() {
        assert!(erf(0.0).abs() < 1e-15);
        assert!((erf(1.0) - 0.842_700_792_949_715).abs() < 1e-10);
        assert!((erf(-1.0) + 0.842_700_792_949_715).abs() < 1e-10);
        assert!((erf(2.0) - 0.995_322_265_004_329).abs() < 1e-8);
    }

    #[test]
    fn test_erf_odd() {
        for &x in &[0.1_f64, 0.5, 1.0, 2.0, 3.0] {
            assert!((erf(x) + erf(-x)).abs() < 1e-12, "erf not odd at x={x}");
        }
    }

    #[test]
    fn test_lgamma_known() {
        assert!(lgamma(1.0).abs() < 1e-12);
        assert!(lgamma(2.0).abs() < 1e-12);
        assert!((lgamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
        assert!((lgamma(0.5) - 0.5 * consts::PI.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_lgamma_recurrence() {
        for &x in &[1.0_f64, 2.0, 3.0, 5.0] {
            let lhs = lgamma(x + 1.0);
            let rhs = lgamma(x) + x.ln();
            assert!((lhs - rhs).abs() < 1e-10, "lgamma recurrence at x={x}");
        }
    }

    #[test]
    fn test_digamma_known() {
        const EULER: f64 = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + EULER).abs() < 1e-10);
    }

    #[test]
    fn test_digamma_recurrence() {
        for &x in &[1.0_f64, 2.0, 3.0, 0.5] {
            let diff = digamma(x + 1.0) - digamma(x);
            assert!(
                (diff - 1.0 / x).abs() < 1e-10,
                "digamma recurrence at x={x}: got {diff}"
            );
        }
    }

    #[test]
    fn test_ei_known() {
        assert!((ei(1.0) - 1.895_117_816_355_937).abs() < 1e-8);
    }

    #[test]
    fn test_si_known() {
        assert!((si(1.0) - 0.946_083_070_367_183).abs() < 1e-8);
    }

    #[test]
    fn test_si_odd() {
        for &x in &[0.5_f64, 1.0, 2.0] {
            assert!((si(x) + si(-x)).abs() < 1e-10, "Si not odd at x={x}");
        }
    }

    #[test]
    fn test_ci_positive() {
        assert!((ci(1.0) - 0.337_403_922_900_968).abs() < 1e-8);
        assert!(ci(-1.0).is_nan());
        assert!(ci(0.0).is_nan());
    }

    #[test]
    fn test_trigamma_known_values() {
        use std::f64::consts::PI;
        // ψ¹(1) = π²/6
        let expected_1 = PI * PI / 6.0;
        assert!(
            (trigamma(1.0) - expected_1).abs() < 1e-10,
            "trigamma(1) = {}, expected {}",
            trigamma(1.0),
            expected_1
        );

        // ψ¹(2) = π²/6 - 1
        let expected_2 = PI * PI / 6.0 - 1.0;
        assert!(
            (trigamma(2.0) - expected_2).abs() < 1e-10,
            "trigamma(2) = {}, expected {}",
            trigamma(2.0),
            expected_2
        );

        // ψ¹(1/2) = π²/2
        let expected_half = PI * PI / 2.0;
        assert!(
            (trigamma(0.5) - expected_half).abs() < 1e-10,
            "trigamma(0.5) = {}, expected {}",
            trigamma(0.5),
            expected_half
        );
    }

    #[test]
    fn test_trigamma_recurrence() {
        // ψ¹(x) - ψ¹(x+1) = 1/x²
        for &x in &[1.0f64, 2.0, 3.0, 0.5, 1.5, 2.5] {
            let diff = trigamma(x) - trigamma(x + 1.0);
            let expected = 1.0 / (x * x);
            assert!(
                (diff - expected).abs() < 1e-10,
                "recurrence failed at x={}: {} != {}",
                x,
                diff,
                expected
            );
        }
    }

    #[test]
    fn test_trigamma_reflection() {
        use std::f64::consts::PI;
        // ψ¹(x) + ψ¹(1-x) = π²/sin²(πx)
        for &x in &[0.3f64, 0.25, 0.1, 0.4] {
            let lhs = trigamma(x) + trigamma(1.0 - x);
            let s = (PI * x).sin();
            let rhs = PI * PI / (s * s);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "reflection failed at x={}: {} != {}",
                x,
                lhs,
                rhs
            );
        }
    }

    #[test]
    fn test_trigamma_pole() {
        assert!(trigamma(0.0).is_infinite());
        assert!(trigamma(-1.0).is_infinite());
        assert!(trigamma(-2.0).is_infinite());
    }

    #[test]
    fn test_trigamma_nan() {
        assert!(trigamma(f64::NAN).is_nan());
    }
}
