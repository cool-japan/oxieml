//! Display, LaTeX export, pretty-printing, evaluation, and structural hashing
//! for [`super::LoweredOp`].

use std::fmt;

use super::LoweredOp;

impl LoweredOp {
    /// Compute a structural hash of this tree.
    ///
    /// Used by the symbolic regression pruner to detect semantically equivalent
    /// topologies after lowering + simplification.
    ///
    /// **f64 note**: constants are hashed as `c.to_bits()` (a `u64`) since
    /// `f64` does not implement `Hash`.
    pub fn structural_hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::hash::Hash;
        match self {
            Self::Const(c) => {
                0u8.hash(state);
                c.to_bits().hash(state);
            }
            Self::NamedConst(nc) => {
                // Hash as if it were the equivalent Const so structural
                // deduplication treats NamedConst(Pi) == Const(PI).
                0u8.hash(state);
                nc.value().to_bits().hash(state);
            }
            Self::Var(i) => {
                1u8.hash(state);
                i.hash(state);
            }
            Self::Add(a, b) => {
                a.structural_hash(state);
                b.structural_hash(state);
                2u8.hash(state);
            }
            Self::Sub(a, b) => {
                a.structural_hash(state);
                b.structural_hash(state);
                3u8.hash(state);
            }
            Self::Mul(a, b) => {
                a.structural_hash(state);
                b.structural_hash(state);
                4u8.hash(state);
            }
            Self::Div(a, b) => {
                a.structural_hash(state);
                b.structural_hash(state);
                5u8.hash(state);
            }
            Self::Exp(a) => {
                a.structural_hash(state);
                6u8.hash(state);
            }
            Self::Ln(a) => {
                a.structural_hash(state);
                7u8.hash(state);
            }
            Self::Sin(a) => {
                a.structural_hash(state);
                8u8.hash(state);
            }
            Self::Cos(a) => {
                a.structural_hash(state);
                9u8.hash(state);
            }
            Self::Pow(a, b) => {
                a.structural_hash(state);
                b.structural_hash(state);
                10u8.hash(state);
            }
            Self::Neg(a) => {
                a.structural_hash(state);
                11u8.hash(state);
            }
            Self::Tan(a) => {
                a.structural_hash(state);
                12u8.hash(state);
            }
            Self::Sinh(a) => {
                a.structural_hash(state);
                13u8.hash(state);
            }
            Self::Cosh(a) => {
                a.structural_hash(state);
                14u8.hash(state);
            }
            Self::Tanh(a) => {
                a.structural_hash(state);
                15u8.hash(state);
            }
            Self::Arcsin(a) => {
                a.structural_hash(state);
                16u8.hash(state);
            }
            Self::Arccos(a) => {
                a.structural_hash(state);
                17u8.hash(state);
            }
            Self::Arctan(a) => {
                a.structural_hash(state);
                18u8.hash(state);
            }
            Self::Arcsinh(a) => {
                a.structural_hash(state);
                19u8.hash(state);
            }
            Self::Arccosh(a) => {
                a.structural_hash(state);
                20u8.hash(state);
            }
            Self::Arctanh(a) => {
                a.structural_hash(state);
                21u8.hash(state);
            }
            Self::Erf(a) => {
                a.structural_hash(state);
                22u8.hash(state);
            }
            Self::LGamma(a) => {
                a.structural_hash(state);
                23u8.hash(state);
            }
            Self::Digamma(a) => {
                a.structural_hash(state);
                24u8.hash(state);
            }
            Self::Trigamma(a) => {
                a.structural_hash(state);
                28u8.hash(state);
            }
            Self::Ei(a) => {
                a.structural_hash(state);
                25u8.hash(state);
            }
            Self::Si(a) => {
                a.structural_hash(state);
                26u8.hash(state);
            }
            Self::Ci(a) => {
                a.structural_hash(state);
                27u8.hash(state);
            }
        }
    }

    /// Convert to a human-readable mathematical expression string.
    pub fn to_pretty(&self) -> String {
        format!("{self}")
    }

    /// Convert to a LaTeX math expression string.
    ///
    /// Produces valid LaTeX for use inside `$...$` or `\[...\]` math mode.
    ///
    /// # Examples
    /// ```
    /// use oxieml::LoweredOp;
    /// let expr = LoweredOp::Div(
    ///     std::sync::Arc::new(LoweredOp::Const(1.0)),
    ///     std::sync::Arc::new(LoweredOp::Var(0)),
    /// );
    /// assert_eq!(expr.to_latex(), r"\frac{1}{x_{0}}");
    /// ```
    pub fn to_latex(&self) -> String {
        fn render(op: &LoweredOp, top_level: bool) -> String {
            match op {
                LoweredOp::NamedConst(nc) => nc.to_latex().to_string(),
                LoweredOp::Const(c) => {
                    if (*c - std::f64::consts::E).abs() < 1e-15 {
                        "e".to_string()
                    } else if (*c - std::f64::consts::PI).abs() < 1e-15 {
                        r"\pi".to_string()
                    } else if (*c - std::f64::consts::TAU).abs() < 1e-15 {
                        r"2\pi".to_string()
                    } else if (*c - (-1.0_f64)).abs() < 1e-15 {
                        "-1".to_string()
                    } else if (c - c.round()).abs() < 1e-10 && c.abs() < 1e15 {
                        format!("{}", *c as i64)
                    } else {
                        format!("{c:.6}")
                    }
                }
                LoweredOp::Var(i) => format!("x_{{{i}}}"),
                LoweredOp::Add(a, b) => {
                    let inner = format!("{} + {}", render(a, false), render(b, false));
                    if top_level {
                        inner
                    } else {
                        format!("({inner})")
                    }
                }
                LoweredOp::Sub(a, b) => {
                    let inner = format!("{} - {}", render(a, false), render(b, false));
                    if top_level {
                        inner
                    } else {
                        format!("({inner})")
                    }
                }
                LoweredOp::Mul(a, b) => {
                    let inner = format!(r"{} \cdot {}", render(a, false), render(b, false));
                    if top_level {
                        inner
                    } else {
                        format!("({inner})")
                    }
                }
                LoweredOp::Div(a, b) => {
                    format!(r"\frac{{{}}}{{{}}}", render(a, true), render(b, true))
                }
                LoweredOp::Exp(a) => {
                    let arg = render(a, true);
                    format!("e^{{{arg}}}")
                }
                LoweredOp::Ln(a) => {
                    format!(r"\ln\left({}\right)", render(a, true))
                }
                LoweredOp::Sin(a) => {
                    format!(r"\sin\left({}\right)", render(a, true))
                }
                LoweredOp::Cos(a) => {
                    format!(r"\cos\left({}\right)", render(a, true))
                }
                LoweredOp::Pow(base, exp) => {
                    let b = render(base, false);
                    let e = render(exp, true);
                    format!("{b}^{{{e}}}")
                }
                LoweredOp::Neg(a) => {
                    let inner = render(a, false);
                    format!("-{inner}")
                }
                LoweredOp::Tan(a) => {
                    format!(r"\tan{{{}}}", render(a, true))
                }
                LoweredOp::Sinh(a) => {
                    format!(r"\sinh{{{}}}", render(a, true))
                }
                LoweredOp::Cosh(a) => {
                    format!(r"\cosh{{{}}}", render(a, true))
                }
                LoweredOp::Tanh(a) => {
                    format!(r"\tanh{{{}}}", render(a, true))
                }
                LoweredOp::Arcsin(a) => {
                    format!(r"\arcsin{{{}}}", render(a, true))
                }
                LoweredOp::Arccos(a) => {
                    format!(r"\arccos{{{}}}", render(a, true))
                }
                LoweredOp::Arctan(a) => {
                    format!(r"\arctan{{{}}}", render(a, true))
                }
                LoweredOp::Arcsinh(a) => {
                    format!(r"\operatorname{{arcsinh}}{{{}}}", render(a, true))
                }
                LoweredOp::Arccosh(a) => {
                    format!(r"\operatorname{{arccosh}}{{{}}}", render(a, true))
                }
                LoweredOp::Arctanh(a) => {
                    format!(r"\operatorname{{arctanh}}{{{}}}", render(a, true))
                }
                LoweredOp::Erf(a) => {
                    format!(r"\operatorname{{erf}}\!\left({}\right)", render(a, true))
                }
                LoweredOp::LGamma(a) => format!(r"\ln\Gamma\!\left({}\right)", render(a, true)),
                LoweredOp::Digamma(a) => format!(r"\psi\!\left({}\right)", render(a, true)),
                LoweredOp::Trigamma(a) => {
                    format!(r"\psi^{{(1)}}\!\left({}\right)", render(a, true))
                }
                LoweredOp::Ei(a) => {
                    format!(r"\operatorname{{Ei}}\!\left({}\right)", render(a, true))
                }
                LoweredOp::Si(a) => {
                    format!(r"\operatorname{{Si}}\!\left({}\right)", render(a, true))
                }
                LoweredOp::Ci(a) => {
                    format!(r"\operatorname{{Ci}}\!\left({}\right)", render(a, true))
                }
            }
        }
        render(self, true)
    }

    /// Evaluate the lowered operation tree with the given variable values.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        match self {
            Self::Const(c) => *c,
            Self::NamedConst(nc) => nc.value(),
            Self::Var(i) => vars[*i],
            Self::Add(a, b) => a.eval(vars) + b.eval(vars),
            Self::Sub(a, b) => a.eval(vars) - b.eval(vars),
            Self::Mul(a, b) => a.eval(vars) * b.eval(vars),
            Self::Div(a, b) => a.eval(vars) / b.eval(vars),
            Self::Exp(a) => a.eval(vars).exp(),
            Self::Ln(a) => a.eval(vars).ln(),
            Self::Sin(a) => a.eval(vars).sin(),
            Self::Cos(a) => a.eval(vars).cos(),
            Self::Pow(a, b) => a.eval(vars).powf(b.eval(vars)),
            Self::Neg(a) => -a.eval(vars),
            Self::Tan(a) => a.eval(vars).tan(),
            Self::Sinh(a) => a.eval(vars).sinh(),
            Self::Cosh(a) => a.eval(vars).cosh(),
            Self::Tanh(a) => a.eval(vars).tanh(),
            Self::Arcsin(a) => a.eval(vars).asin(),
            Self::Arccos(a) => a.eval(vars).acos(),
            Self::Arctan(a) => a.eval(vars).atan(),
            Self::Arcsinh(a) => a.eval(vars).asinh(),
            Self::Arccosh(a) => a.eval(vars).acosh(),
            Self::Arctanh(a) => a.eval(vars).atanh(),
            Self::Erf(a) => crate::special::erf(a.eval(vars)),
            Self::LGamma(a) => crate::special::lgamma(a.eval(vars)),
            Self::Digamma(a) => crate::special::digamma(a.eval(vars)),
            Self::Trigamma(a) => crate::special::trigamma(a.eval(vars)),
            Self::Ei(a) => crate::special::ei(a.eval(vars)),
            Self::Si(a) => crate::special::si(a.eval(vars)),
            Self::Ci(a) => crate::special::ci(a.eval(vars)),
        }
    }
}

impl fmt::Display for LoweredOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedConst(nc) => write!(f, "{}", nc.to_pretty()),
            Self::Const(c) => {
                if (*c - std::f64::consts::E).abs() < 1e-15 {
                    write!(f, "e")
                } else if (*c - std::f64::consts::PI).abs() < 1e-15 {
                    write!(f, "π")
                } else if (c - c.round()).abs() < 1e-10 && c.abs() < 1e15 {
                    write!(f, "{}", *c as i64)
                } else {
                    write!(f, "{c:.6}")
                }
            }
            Self::Var(i) => write!(f, "x{i}"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Sub(a, b) => write!(f, "({a} - {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Div(a, b) => write!(f, "({a} / {b})"),
            Self::Exp(a) => write!(f, "exp({a})"),
            Self::Ln(a) => write!(f, "ln({a})"),
            Self::Sin(a) => write!(f, "sin({a})"),
            Self::Cos(a) => write!(f, "cos({a})"),
            Self::Pow(a, b) => write!(f, "({a})^({b})"),
            Self::Neg(a) => write!(f, "-{a}"),
            Self::Tan(a) => write!(f, "tan({a})"),
            Self::Sinh(a) => write!(f, "sinh({a})"),
            Self::Cosh(a) => write!(f, "cosh({a})"),
            Self::Tanh(a) => write!(f, "tanh({a})"),
            Self::Arcsin(a) => write!(f, "arcsin({a})"),
            Self::Arccos(a) => write!(f, "arccos({a})"),
            Self::Arctan(a) => write!(f, "arctan({a})"),
            Self::Arcsinh(a) => write!(f, "arcsinh({a})"),
            Self::Arccosh(a) => write!(f, "arccosh({a})"),
            Self::Arctanh(a) => write!(f, "arctanh({a})"),
            Self::Erf(a) => write!(f, "erf({a})"),
            Self::LGamma(a) => write!(f, "lgamma({a})"),
            Self::Digamma(a) => write!(f, "digamma({a})"),
            Self::Trigamma(a) => write!(f, "trigamma({a})"),
            Self::Ei(a) => write!(f, "Ei({a})"),
            Self::Si(a) => write!(f, "Si({a})"),
            Self::Ci(a) => write!(f, "Ci({a})"),
        }
    }
}
