//! Dimensional analysis extension for [`LoweredOp`].
//!
//! This module provides [`LoweredOp::check_units`], which traverses a lowered
//! expression tree and checks dimensional consistency against a caller-supplied
//! map from variable index to physical [`Units`].
//!
//! The rules implemented here are:
//!
//! - **Constants / named constants** — always dimensionless.
//! - **Variables** — the unit is looked up in `var_units[i]`.
//! - **Negation** — passes units through unchanged.
//! - **Add / Sub** — both operands must have identical units; result has those units.
//! - **Mul** — units multiply (exponent-wise addition).
//! - **Div** — units divide (exponent-wise subtraction).
//! - **Pow** — if the base is dimensionless the result is dimensionless;
//!   otherwise the exponent must be a rational constant (integer or fraction),
//!   and the result is `base_units^(num/den)`.
//! - **Transcendentals** (`exp`, `ln`, `sin`, `cos`, … all 14) — argument
//!   must be dimensionless; result is dimensionless.

use crate::lower::LoweredOp;
use crate::units::{UnitError, Units};

impl LoweredOp {
    /// Check dimensional consistency and infer the output units of this
    /// expression.
    ///
    /// `var_units[i]` gives the physical units of variable `i`.  Pass an empty
    /// slice when the expression contains no variables.
    ///
    /// # Returns
    ///
    /// `Ok(units)` — the expression is dimensionally consistent and its output
    /// has these units.
    ///
    /// `Err(UnitError)` — at least one dimensional rule was violated.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxieml::{LoweredOp, units::Units};
    ///
    /// // x * t  where x has units of metres and t has units of seconds → m·s
    /// let expr = LoweredOp::Mul(
    ///     std::sync::Arc::new(LoweredOp::Var(0)),
    ///     std::sync::Arc::new(LoweredOp::Var(1)),
    /// );
    /// let var_units = [Units::METER, Units::SECOND];
    /// assert_eq!(
    ///     expr.check_units(&var_units),
    ///     Ok(Units::METER.mul(&Units::SECOND))
    /// );
    /// ```
    pub fn check_units(&self, var_units: &[Units]) -> Result<Units, UnitError> {
        match self {
            // Constants are always dimensionless.
            Self::Const(_) | Self::NamedConst(_) => Ok(Units::DIMENSIONLESS),

            // Variables carry user-supplied units.
            Self::Var(i) => {
                if *i >= var_units.len() {
                    Err(UnitError::VarIndexOutOfRange {
                        index: *i,
                        n_vars: var_units.len(),
                    })
                } else {
                    Ok(var_units[*i])
                }
            }

            // Negation preserves units.
            Self::Neg(x) => x.check_units(var_units),

            // Add / Sub: both operands must have the same units.
            Self::Add(a, b) | Self::Sub(a, b) => {
                let ua = a.check_units(var_units)?;
                let ub = b.check_units(var_units)?;
                if ua != ub {
                    Err(UnitError::IncompatibleAddSub {
                        left: ua,
                        right: ub,
                    })
                } else {
                    Ok(ua)
                }
            }

            // Mul: units multiply (exponent-wise add).
            Self::Mul(a, b) => Ok(a.check_units(var_units)?.mul(&b.check_units(var_units)?)),

            // Div: units divide (exponent-wise subtract).
            Self::Div(a, b) => Ok(a.check_units(var_units)?.div(&b.check_units(var_units)?)),

            // Pow: dimensionless base → dimensionless result; dimensioned base
            // requires a rational-constant exponent.
            Self::Pow(base, exp) => {
                let base_units = base.check_units(var_units)?;
                let exp_units = exp.check_units(var_units)?;

                // Exponent must always be dimensionless.
                if !exp_units.is_dimensionless() {
                    return Err(UnitError::NonDimensionlessArgument {
                        op: "Pow(exponent)",
                        got: exp_units,
                    });
                }

                // Dimensionless base: result is dimensionless regardless of exponent.
                if base_units.is_dimensionless() {
                    return Ok(Units::DIMENSIONLESS);
                }

                // Dimensioned base: rationalize the exponent and apply pow_rational.
                match exp.as_ref() {
                    Self::Const(n) => {
                        let r = rationalize_f64(*n, 12);
                        base_units
                            .pow_rational(r.num, r.den)
                            .map_err(|_| UnitError::NonRationalPower { base_units })
                    }
                    Self::NamedConst(nc) => {
                        let v = nc.value();
                        let r = rationalize_f64(v, 12);
                        base_units
                            .pow_rational(r.num, r.den)
                            .map_err(|_| UnitError::NonRationalPower { base_units })
                    }
                    _ => Err(UnitError::NonRationalPower { base_units }),
                }
            }

            // Transcendental functions: argument must be dimensionless, output is dimensionless.
            transcendental => {
                // Extract the single inner operand and determine the op name.
                let (inner, op_name): (&Self, &'static str) = match transcendental {
                    Self::Exp(x) => (x.as_ref(), "exp"),
                    Self::Ln(x) => (x.as_ref(), "ln"),
                    Self::Sin(x) => (x.as_ref(), "sin"),
                    Self::Cos(x) => (x.as_ref(), "cos"),
                    Self::Tan(x) => (x.as_ref(), "tan"),
                    Self::Sinh(x) => (x.as_ref(), "sinh"),
                    Self::Cosh(x) => (x.as_ref(), "cosh"),
                    Self::Tanh(x) => (x.as_ref(), "tanh"),
                    Self::Arcsin(x) => (x.as_ref(), "arcsin"),
                    Self::Arccos(x) => (x.as_ref(), "arccos"),
                    Self::Arctan(x) => (x.as_ref(), "arctan"),
                    Self::Arcsinh(x) => (x.as_ref(), "arcsinh"),
                    Self::Arccosh(x) => (x.as_ref(), "arccosh"),
                    Self::Arctanh(x) => (x.as_ref(), "arctanh"),
                    Self::Erf(x) => (x.as_ref(), "erf"),
                    Self::LGamma(x) => (x.as_ref(), "lgamma"),
                    Self::Digamma(x) => (x.as_ref(), "digamma"),
                    Self::Ei(x) => (x.as_ref(), "Ei"),
                    Self::Si(x) => (x.as_ref(), "Si"),
                    Self::Ci(x) => (x.as_ref(), "Ci"),
                    // All other variants are already handled above.
                    _ => unreachable!("all non-transcendental variants handled before this arm"),
                };

                let ux = inner.check_units(var_units)?;
                if !ux.is_dimensionless() {
                    Err(UnitError::NonDimensionlessArgument {
                        op: op_name,
                        got: ux,
                    })
                } else {
                    Ok(Units::DIMENSIONLESS)
                }
            }
        }
    }
}

/// Approximate `x` as a rational `num/den` with `den <= max_den`.
///
/// Uses exhaustive denominator search — small `max_den` keeps this fast.
/// The result is always normalized (gcd(num.abs(), den) == 1, den > 0).
pub(crate) fn rationalize_f64(x: f64, max_den: i16) -> crate::units::Rexp {
    use crate::units::Rexp;
    if !x.is_finite() {
        return Rexp::ZERO;
    }
    let sign: i16 = if x < 0.0 { -1 } else { 1 };
    let x_abs = x.abs();

    // Check if exactly integer
    let integer_part = x_abs.floor() as i16;
    if (x_abs - integer_part as f64).abs() < 1e-10 {
        return Rexp {
            num: sign * integer_part,
            den: 1,
        };
    }

    let mut best_num = sign * integer_part;
    let mut best_den: i16 = 1;
    let mut best_err = (x_abs - integer_part as f64).abs();

    for den in 1i16..=max_den {
        let num_f = x_abs * den as f64;
        let num = num_f.round() as i16;
        if num < 0 {
            continue;
        }
        let err = (x_abs - num as f64 / den as f64).abs();
        if err < best_err {
            best_err = err;
            best_num = sign * num;
            best_den = den;
        }
        if best_err < 1e-12 {
            break;
        }
    }

    // Normalize
    Rexp::from_ratio(best_num, best_den)
}
