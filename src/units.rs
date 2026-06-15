//! Dimensional analysis for unit-aware symbolic regression.
//!
//! Provides [`Units`] — a 7-element exponent vector representing SI base units —
//! and [`UnitError`] for dimensional consistency violations. Integrates with
//! [`crate::lower::LoweredOp::check_units`] to enable hard pruning of
//! dimensionally-inadmissible topologies during symbolic regression.
//!
//! # SI base dimensions
//!
//! Index | Symbol | Quantity
//! ------|--------|----------
//! 0     | m      | Length
//! 1     | kg     | Mass
//! 2     | s      | Time
//! 3     | A      | Electric current
//! 4     | K      | Thermodynamic temperature
//! 5     | mol    | Amount of substance
//! 6     | cd     | Luminous intensity
//!
//! # Example
//!
//! ```
//! use oxieml::units::Units;
//!
//! // velocity: m/s
//! let velocity = Units::METER.div(&Units::SECOND);
//! assert_eq!(velocity.try_into_int_exps(), Some([1i8, 0, -1, 0, 0, 0, 0]));
//!
//! // kinetic energy: kg·m²/s²  ≡ JOULE
//! let ke = Units::KILOGRAM.mul(&Units::METER).mul(&Units::METER).div(&Units::SECOND).div(&Units::SECOND);
//! assert_eq!(ke, Units::JOULE);
//! ```

use std::fmt;

// -----------------------------------------------------------------------
// gcd helpers (needed for Rexp normalization)
// -----------------------------------------------------------------------

fn gcd_u16(mut a: u16, mut b: u16) -> u16 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a }
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a }
}

// -----------------------------------------------------------------------
// Rexp: rational exponent
// -----------------------------------------------------------------------

/// A rational exponent for SI base dimensions.
/// Normalized: `gcd(num.abs(), den) == 1`, `den > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rexp {
    /// Numerator of the rational exponent.
    pub num: i16,
    /// Denominator of the rational exponent (always positive, ≥ 1).
    pub den: i16,
}

impl Rexp {
    /// The additive identity: `0/1`.
    pub const ZERO: Self = Rexp { num: 0, den: 1 };
    /// The multiplicative identity: `1/1`.
    pub const ONE: Self = Rexp { num: 1, den: 1 };

    /// Construct a normalized `Rexp`. Returns `None` if `den <= 0`.
    pub fn new(num: i16, den: i16) -> Option<Self> {
        if den <= 0 {
            return None;
        }
        if num == 0 {
            return Some(Self::ZERO);
        }
        let g = gcd_u16(num.unsigned_abs(), den.unsigned_abs()) as i16;
        Some(Rexp {
            num: num / g,
            den: den / g,
        })
    }

    /// Construct from a known-valid ratio (panics only if den <= 0, which
    /// is only called from `rationalize_f64` with controlled inputs).
    pub fn from_ratio(num: i16, den: i16) -> Self {
        Self::new(num, den).expect("Rexp::from_ratio: den must be positive")
    }

    /// Construct from an integer exponent.
    pub fn from_int(n: i8) -> Self {
        Rexp {
            num: n as i16,
            den: 1,
        }
    }

    /// Convert to `f64` (numerator / denominator).
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Return `true` if the denominator is 1 (i.e. an integer exponent).
    pub fn is_integer(self) -> bool {
        self.den == 1
    }

    /// Try to convert to `i8`. Returns `None` if the denominator is not 1 or the value is out of `i8` range.
    pub fn to_i8(self) -> Option<i8> {
        if self.den == 1 && self.num >= -128 && self.num <= 127 {
            Some(self.num as i8)
        } else {
            None
        }
    }

    fn add_raw(self, rhs: Self) -> Self {
        let num = self.num as i32 * rhs.den as i32 + rhs.num as i32 * self.den as i32;
        let den = self.den as i32 * rhs.den as i32;
        if num == 0 {
            return Self::ZERO;
        }
        let g = gcd_u32(num.unsigned_abs(), den.unsigned_abs()) as i32;
        let g = if g == 0 { 1 } else { g };
        Rexp {
            num: (num / g).clamp(-32767, 32767) as i16,
            den: (den / g).clamp(1, 32767) as i16,
        }
    }

    fn rexp_add(self, rhs: Self) -> Self {
        self.add_raw(rhs)
    }

    fn rexp_sub(self, rhs: Self) -> Self {
        self.add_raw(Rexp {
            num: -rhs.num,
            den: rhs.den,
        })
    }

    /// Multiply by an integer scalar: `self * s`.
    pub fn mul_int(self, s: i32) -> Self {
        if s == 0 {
            return Self::ZERO;
        }
        let num = self.num as i32 * s;
        let g = gcd_u32(num.unsigned_abs(), self.den as u32) as i32;
        let g = if g == 0 { 1 } else { g };
        Rexp {
            num: (num / g).clamp(-32767, 32767) as i16,
            den: (self.den as i32 / g) as i16,
        }
    }

    /// Multiply two rational exponents: `self * rhs`.
    pub fn mul_rexp(self, rhs: Self) -> Self {
        let num = self.num as i32 * rhs.num as i32;
        let den = self.den as i32 * rhs.den as i32;
        if num == 0 {
            return Self::ZERO;
        }
        let g = gcd_u32(num.unsigned_abs(), den.unsigned_abs()) as i32;
        let g = if g == 0 { 1 } else { g };
        Rexp {
            num: (num / g).clamp(-32767, 32767) as i16,
            den: (den / g).clamp(1, 32767) as i16,
        }
    }
}

impl std::ops::Add for Rexp {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.rexp_add(rhs)
    }
}

impl std::ops::Sub for Rexp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.rexp_sub(rhs)
    }
}

impl fmt::Display for Rexp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({}/{})", self.num, self.den)
        }
    }
}

// -----------------------------------------------------------------------
// const helper for building Units constants
// -----------------------------------------------------------------------

const fn rexp_int(n: i8) -> Rexp {
    Rexp {
        num: n as i16,
        den: 1,
    }
}

// -----------------------------------------------------------------------
// Units
// -----------------------------------------------------------------------

/// SI unit represented as a 7-element rational exponent vector `[m, kg, s, A, K, mol, cd]`.
///
/// Positive exponents appear in the numerator, negative in the denominator.
/// The arithmetic follows the standard rules:
///
/// - Multiplication: element-wise addition of exponents (`m·s → [1,0,1,0,0,0,0]`).
/// - Division: element-wise subtraction of exponents (`m/s → [1,0,-1,0,0,0,0]`).
/// - Integer power: element-wise scaling (`m² → [2,0,0,0,0,0,0]`).
/// - Rational power: element-wise rational scaling (`m^(1/2)` via `sqrt()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Units(pub [Rexp; 7]);

impl Units {
    // -----------------------------------------------------------------------
    // SI base units
    // -----------------------------------------------------------------------

    /// Dimensionless quantity (no physical unit).
    pub const DIMENSIONLESS: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Metre (length).
    pub const METER: Self = Self([
        Rexp::ONE,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Kilogram (mass).
    pub const KILOGRAM: Self = Self([
        Rexp::ZERO,
        Rexp::ONE,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Second (time).
    pub const SECOND: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ONE,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Ampere (electric current).
    pub const AMPERE: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ONE,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Kelvin (thermodynamic temperature).
    pub const KELVIN: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ONE,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Mole (amount of substance).
    pub const MOL: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ONE,
        Rexp::ZERO,
    ]);
    /// Candela (luminous intensity).
    pub const CANDELA: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ONE,
    ]);

    // -----------------------------------------------------------------------
    // Derived units
    // -----------------------------------------------------------------------

    /// Newton (force): kg·m·s⁻².
    pub const NEWTON: Self = Self([
        rexp_int(1),
        rexp_int(1),
        rexp_int(-2),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Joule (energy): kg·m²·s⁻².
    pub const JOULE: Self = Self([
        rexp_int(2),
        rexp_int(1),
        rexp_int(-2),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Watt (power): kg·m²·s⁻³.
    pub const WATT: Self = Self([
        rexp_int(2),
        rexp_int(1),
        rexp_int(-3),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Pascal (pressure): kg·m⁻¹·s⁻².
    pub const PASCAL: Self = Self([
        rexp_int(-1),
        rexp_int(1),
        rexp_int(-2),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Hertz (frequency): s⁻¹.
    pub const HERTZ: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        rexp_int(-1),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Coulomb (electric charge): A·s.
    pub const COULOMB: Self = Self([
        Rexp::ZERO,
        Rexp::ZERO,
        rexp_int(1),
        rexp_int(1),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Volt (electric potential): kg·m²·s⁻³·A⁻¹.
    pub const VOLT: Self = Self([
        rexp_int(2),
        rexp_int(1),
        rexp_int(-3),
        rexp_int(-1),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);
    /// Ohm (electric resistance): kg·m²·s⁻³·A⁻².
    pub const OHM: Self = Self([
        rexp_int(2),
        rexp_int(1),
        rexp_int(-3),
        rexp_int(-2),
        Rexp::ZERO,
        Rexp::ZERO,
        Rexp::ZERO,
    ]);

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Construct a `Units` value from a raw 7-element integer exponent array.
    #[inline]
    pub fn new(exps: [i8; 7]) -> Self {
        Self([
            Rexp::from_int(exps[0]),
            Rexp::from_int(exps[1]),
            Rexp::from_int(exps[2]),
            Rexp::from_int(exps[3]),
            Rexp::from_int(exps[4]),
            Rexp::from_int(exps[5]),
            Rexp::from_int(exps[6]),
        ])
    }

    /// Construct from integer exponents array (alias for `new`).
    #[inline]
    pub fn from_int_exps(exps: [i8; 7]) -> Self {
        Self::new(exps)
    }

    /// Try to convert back to integer exponents (for backward-compat checks).
    /// Returns `None` if any exponent is non-integer.
    pub fn try_into_int_exps(&self) -> Option<[i8; 7]> {
        let mut result = [0i8; 7];
        for (i, r) in self.0.iter().enumerate() {
            result[i] = r.to_i8()?;
        }
        Some(result)
    }

    /// Return `true` when all exponents are zero (i.e. dimensionless).
    #[inline]
    pub fn is_dimensionless(&self) -> bool {
        self.0.iter().all(|r| *r == Rexp::ZERO)
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    /// Multiply units: element-wise addition of exponents.
    ///
    /// Equivalent to multiplying two quantities with these units.
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = [Rexp::ZERO; 7];
        for (dst, (a, b)) in result.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *dst = *a + *b;
        }
        Self(result)
    }

    /// Divide units: element-wise subtraction of exponents.
    ///
    /// Equivalent to dividing a quantity with `self` units by one with `other` units.
    #[inline]
    pub fn div(&self, other: &Self) -> Self {
        let mut result = [Rexp::ZERO; 7];
        for (dst, (a, b)) in result.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *dst = *a - *b;
        }
        Self(result)
    }

    /// Raise to an integer power: multiply each exponent by `n`.
    ///
    /// Returns `Err` when exponent overflow occurs.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::ExponentOverflow`] when any exponent would overflow.
    pub fn pow_int(&self, n: i32) -> Result<Self, UnitError> {
        let mut result = [Rexp::ZERO; 7];
        for (i, (dst, r)) in result.iter_mut().zip(self.0.iter()).enumerate() {
            let scaled_num = r.num as i64 * n as i64;
            // For integer base exponents (den==1), enforce i8 range for backward compatibility.
            // For rational base exponents (den!=1), enforce i16 range.
            let overflow = if r.den == 1 {
                !(-128..=127_i64).contains(&scaled_num)
            } else {
                !(-32767..=32767_i64).contains(&scaled_num)
            };
            if overflow {
                return Err(UnitError::ExponentOverflow {
                    dimension: i,
                    base_exp: r.to_i8().unwrap_or(0),
                    power: n,
                });
            }
            *dst = r.mul_int(n);
        }
        Ok(Self(result))
    }

    /// Raise to a rational power: each exponent multiplied by `num/den`.
    ///
    /// # Errors
    ///
    /// Returns [`UnitError::ExponentOverflow`] if `den <= 0`.
    pub fn pow_rational(&self, num: i16, den: i16) -> Result<Self, UnitError> {
        if den <= 0 {
            return Err(UnitError::ExponentOverflow {
                dimension: 0,
                base_exp: 0,
                power: num as i32,
            });
        }
        let r = Rexp::from_ratio(num, den);
        let mut out = [Rexp::ZERO; 7];
        for (dst, src) in out.iter_mut().zip(self.0.iter()) {
            *dst = src.mul_rexp(r);
        }
        Ok(Self(out))
    }

    /// Square root: each exponent multiplied by 1/2.
    pub fn sqrt(&self) -> Self {
        self.pow_rational(1, 2).unwrap_or(*self)
    }

    // -----------------------------------------------------------------------
    // Symbol helpers
    // -----------------------------------------------------------------------

    /// Symbol string for dimension `idx` (0 = m, 1 = kg, 2 = s, …).
    fn dim_symbol(idx: usize) -> &'static str {
        match idx {
            0 => "m",
            1 => "kg",
            2 => "s",
            3 => "A",
            4 => "K",
            5 => "mol",
            6 => "cd",
            _ => "?",
        }
    }
}

// -----------------------------------------------------------------------
// Display — produces strings like "m¹·kg²·s⁻²" or "m^(1/2)" for rationals
// -----------------------------------------------------------------------

fn format_superscript(n: i16) -> String {
    match n {
        1 => String::new(),
        -1 => "\u{207B}\u{00B9}".to_string(),
        2 => "\u{00B2}".to_string(),
        3 => "\u{00B3}".to_string(),
        -2 => "\u{207B}\u{00B2}".to_string(),
        -3 => "\u{207B}\u{00B3}".to_string(),
        _ => {
            let sign = if n < 0 { "\u{207B}" } else { "" };
            let digits: String = n
                .unsigned_abs()
                .to_string()
                .chars()
                .map(|c| match c {
                    '0' => '\u{2070}',
                    '1' => '\u{00B9}',
                    '2' => '\u{00B2}',
                    '3' => '\u{00B3}',
                    '4' => '\u{2074}',
                    '5' => '\u{2075}',
                    '6' => '\u{2076}',
                    '7' => '\u{2077}',
                    '8' => '\u{2078}',
                    '9' => '\u{2079}',
                    other => other,
                })
                .collect();
            format!("{sign}{digits}")
        }
    }
}

impl fmt::Display for Units {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }

        let mut first = true;
        for (i, r) in self.0.iter().enumerate() {
            if *r == Rexp::ZERO {
                continue;
            }
            if !first {
                write!(f, "\u{00B7}")?; // · middle dot
            }
            first = false;
            let sym = Self::dim_symbol(i);
            if r.den == 1 {
                // Integer exponent: use superscripts like before
                write!(f, "{}{}", sym, format_superscript(r.num))?;
            } else {
                // Rational exponent
                write!(f, "{}^({}/{})", sym, r.num, r.den)?;
            }
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during dimensional analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnitError {
    /// `Add` or `Sub` applied to operands with incompatible units.
    IncompatibleAddSub {
        /// Units of the left operand.
        left: Units,
        /// Units of the right operand.
        right: Units,
    },
    /// Transcendental function (`exp`, `ln`, `sin`, …) applied to a
    /// non-dimensionless argument.
    NonDimensionlessArgument {
        /// Name of the offending operation.
        op: &'static str,
        /// Actual units of the argument.
        got: Units,
    },
    /// `Pow` with a non-dimensionless base and a non-rational exponent.
    NonRationalPower {
        /// Units of the base expression.
        base_units: Units,
    },
    /// Variable index `index` was out of range for a `var_units` slice of
    /// length `n_vars`.
    VarIndexOutOfRange {
        /// The offending variable index.
        index: usize,
        /// Length of the `var_units` slice.
        n_vars: usize,
    },
    /// Integer exponent scaling would overflow the representable range.
    ExponentOverflow {
        /// Zero-based dimension index.
        dimension: usize,
        /// Base exponent value.
        base_exp: i8,
        /// Requested integer power.
        power: i32,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleAddSub { left, right } => {
                write!(f, "add/sub unit mismatch: {left} ≠ {right}")
            }
            Self::NonDimensionlessArgument { op, got } => {
                write!(f, "{op} requires a dimensionless argument, got {got}")
            }
            Self::NonRationalPower { base_units } => {
                write!(
                    f,
                    "Pow with dimensioned base ({base_units}) requires a rational-constant exponent"
                )
            }
            Self::VarIndexOutOfRange { index, n_vars } => {
                write!(
                    f,
                    "variable index {index} out of range (var_units has {n_vars} entries)"
                )
            }
            Self::ExponentOverflow {
                dimension,
                base_exp,
                power,
            } => {
                write!(
                    f,
                    "exponent overflow for dimension {dimension}: {base_exp} × {power} exceeds range"
                )
            }
        }
    }
}

impl std::error::Error for UnitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensionless_is_zero_vector() {
        assert!(Units::DIMENSIONLESS.is_dimensionless());
        assert_eq!(Units::DIMENSIONLESS.try_into_int_exps(), Some([0i8; 7]));
    }

    #[test]
    fn named_units_correct() {
        assert_eq!(
            Units::METER.try_into_int_exps(),
            Some([1i8, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            Units::KILOGRAM.try_into_int_exps(),
            Some([0i8, 1, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            Units::SECOND.try_into_int_exps(),
            Some([0i8, 0, 1, 0, 0, 0, 0])
        );
    }

    #[test]
    fn mul_adds_exponents() {
        let result = Units::METER.mul(&Units::SECOND);
        assert_eq!(result.try_into_int_exps(), Some([1i8, 0, 1, 0, 0, 0, 0]));
    }

    #[test]
    fn div_subtracts_exponents() {
        let result = Units::METER.div(&Units::SECOND);
        assert_eq!(result.try_into_int_exps(), Some([1i8, 0, -1, 0, 0, 0, 0]));
    }

    #[test]
    fn pow_int_scales_exponents() {
        let result = Units::METER.pow_int(3).expect("no overflow");
        assert_eq!(result.try_into_int_exps(), Some([3i8, 0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn pow_int_overflow_returns_err() {
        let result = Units::METER.pow_int(200);
        assert!(result.is_err());
    }

    #[test]
    fn display_meter_per_second() {
        let v = Units::METER.div(&Units::SECOND);
        let s = v.to_string();
        assert!(s.contains('m'), "expected 'm' in '{s}'");
        assert!(s.contains('s'), "expected 's' in '{s}'");
    }

    #[test]
    fn display_dimensionless() {
        assert_eq!(Units::DIMENSIONLESS.to_string(), "1");
    }

    #[test]
    fn derived_newton() {
        // F = kg·m·s⁻²
        let newton = Units::KILOGRAM.mul(&Units::METER).mul(
            &Units::SECOND
                .pow_int(-2)
                .expect("pow_int(-2) does not overflow"),
        );
        assert_eq!(newton, Units::NEWTON);
    }

    #[test]
    fn rexp_from_int_roundtrip() {
        let r = Rexp::from_int(3);
        assert_eq!(r.num, 3);
        assert_eq!(r.den, 1);
        assert!(r.is_integer());
        assert_eq!(r.to_i8(), Some(3));
    }

    #[test]
    fn sqrt_of_m2_is_m() {
        let m2 = Units::METER.pow_int(2).expect("no overflow");
        let m = m2.sqrt();
        assert_eq!(m.0[0].to_i8(), Some(1), "sqrt(m²) should give m^1");
    }

    #[test]
    fn sqrt_of_meter_is_half_power() {
        let m_half = Units::METER.sqrt();
        assert_eq!(
            m_half.0[0],
            Rexp::from_ratio(1, 2),
            "sqrt(m) should be m^(1/2)"
        );
    }

    #[test]
    fn mul_half_plus_half_is_one() {
        let m_half = Units::METER.sqrt();
        let m = m_half.mul(&m_half);
        assert_eq!(m.0[0].to_i8(), Some(1), "m^(1/2) * m^(1/2) = m");
    }

    #[test]
    fn legacy_integer_ops_preserved() {
        let m = Units::METER;
        let m2 = m.pow_int(2).expect("no overflow");
        assert_eq!(m2.0[0].to_i8(), Some(2));
        let m_back = m2.div(&m);
        assert_eq!(m_back.0[0].to_i8(), Some(1));
    }

    #[test]
    fn display_rational_exponent() {
        let m_half = Units::METER.sqrt();
        let s = m_half.to_string();
        assert!(
            s.contains("(1/2)"),
            "rational display should show (1/2), got: {s}"
        );
    }

    #[test]
    fn new_from_int_array() {
        let v = Units::new([1, 0, -1, 0, 0, 0, 0]);
        assert_eq!(v.try_into_int_exps(), Some([1i8, 0, -1, 0, 0, 0, 0]));
    }
}
