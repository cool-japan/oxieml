//! Named mathematical constants for the `LoweredOp` IR.
//!
//! [`NamedConst`] represents well-known mathematical constants (π, e, √2, etc.)
//! as explicit enum variants rather than opaque `f64` values. This enables
//! pretty-printing (`π` instead of `3.141592...`), LaTeX rendering (`\pi`),
//! and constants extraction after Adam optimisation.

/// A well-known mathematical constant recognized by the constants-extraction pass.
///
/// Created only by [`crate::symreg::SymRegEngine`]'s post-Adam extraction step;
/// never produced by lowering alone. Constant folding in [`crate::lower::LoweredOp::simplify`]
/// reduces `NamedConst` back to `Const(value())` when combined with other constants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NamedConst {
    /// π ≈ 3.141 592 653 589 793
    Pi,
    /// Euler's number e ≈ 2.718 281 828 459 045
    E,
    /// √2 ≈ 1.414 213 562 373 095
    Sqrt2,
    /// −π
    NegPi,
    /// −e
    NegE,
    /// −√2
    NegSqrt2,
    /// 1/2
    Half,
    /// −1/2
    NegHalf,
    /// 1/3
    Third,
    /// 1/4
    Quarter,
    /// 2π ≈ 6.283
    TwoPi,
    /// π/2 ≈ 1.5708
    PiHalf,
    /// √3 ≈ 1.7321
    Sqrt3,
    /// e² ≈ 7.3891
    ESq,
    /// Golden ratio φ = (1+√5)/2 ≈ 1.6180
    Phi,
    /// ln(2) ≈ 0.6931
    Ln2,
    /// −2
    NegTwo,
}

impl NamedConst {
    /// Numeric value of this constant.
    pub fn value(&self) -> f64 {
        match self {
            Self::Pi => std::f64::consts::PI,
            Self::E => std::f64::consts::E,
            Self::Sqrt2 => std::f64::consts::SQRT_2,
            Self::NegPi => -std::f64::consts::PI,
            Self::NegE => -std::f64::consts::E,
            Self::NegSqrt2 => -std::f64::consts::SQRT_2,
            Self::Half => 0.5,
            Self::NegHalf => -0.5,
            Self::Third => 1.0 / 3.0,
            Self::Quarter => 0.25,
            Self::TwoPi => 2.0 * std::f64::consts::PI,
            Self::PiHalf => std::f64::consts::PI / 2.0,
            Self::Sqrt3 => 3.0_f64.sqrt(),
            Self::ESq => std::f64::consts::E * std::f64::consts::E,
            Self::Phi => (1.0 + 5.0_f64.sqrt()) / 2.0,
            Self::Ln2 => 2.0_f64.ln(),
            Self::NegTwo => -2.0,
        }
    }

    /// Human-readable symbol for display in `to_pretty()`.
    pub fn to_pretty(&self) -> &'static str {
        match self {
            Self::Pi => "π",
            Self::E => "e",
            Self::Sqrt2 => "√2",
            Self::NegPi => "(-π)",
            Self::NegE => "(-e)",
            Self::NegSqrt2 => "(-√2)",
            Self::Half => "(1/2)",
            Self::NegHalf => "(-1/2)",
            Self::Third => "(1/3)",
            Self::Quarter => "(1/4)",
            Self::TwoPi => "2π",
            Self::PiHalf => "(π/2)",
            Self::Sqrt3 => "√3",
            Self::ESq => "e²",
            Self::Phi => "φ",
            Self::Ln2 => "ln(2)",
            Self::NegTwo => "(-2)",
        }
    }

    /// LaTeX representation.
    pub fn to_latex(&self) -> &'static str {
        match self {
            Self::Pi => r"\pi",
            Self::E => "e",
            Self::Sqrt2 => r"\sqrt{2}",
            Self::NegPi => r"(-\pi)",
            Self::NegE => "(-e)",
            Self::NegSqrt2 => r"(-\sqrt{2})",
            Self::Half => r"\frac{1}{2}",
            Self::NegHalf => r"-\frac{1}{2}",
            Self::Third => r"\frac{1}{3}",
            Self::Quarter => r"\frac{1}{4}",
            Self::TwoPi => r"2\pi",
            Self::PiHalf => r"\frac{\pi}{2}",
            Self::Sqrt3 => r"\sqrt{3}",
            Self::ESq => r"e^{2}",
            Self::Phi => r"\varphi",
            Self::Ln2 => r"\ln(2)",
            Self::NegTwo => "(-2)",
        }
    }
}
