//! Lowering EML trees to standard mathematical operations.
//!
//! The EML representation is optimal for *discovery* (uniform search space)
//! but inefficient for *execution* (a single multiplication requires 41+ nodes).
//! Lowering converts EML trees to conventional operation trees for efficient
//! evaluation and human-readable output.

use crate::tree::{EmlNode, EmlTree};
use std::fmt;

/// A conventional mathematical operation tree.
///
/// Produced by lowering an EML tree. Supports efficient evaluation
/// and pretty-printing.
#[derive(Clone, Debug, PartialEq)]
pub enum LoweredOp {
    /// Constant value.
    Const(f64),
    /// Input variable.
    Var(usize),
    /// Addition.
    Add(Box<LoweredOp>, Box<LoweredOp>),
    /// Subtraction.
    Sub(Box<LoweredOp>, Box<LoweredOp>),
    /// Multiplication.
    Mul(Box<LoweredOp>, Box<LoweredOp>),
    /// Division.
    Div(Box<LoweredOp>, Box<LoweredOp>),
    /// Exponential function.
    Exp(Box<LoweredOp>),
    /// Natural logarithm.
    Ln(Box<LoweredOp>),
    /// Sine.
    Sin(Box<LoweredOp>),
    /// Cosine.
    Cos(Box<LoweredOp>),
    /// Power.
    Pow(Box<LoweredOp>, Box<LoweredOp>),
    /// Negation.
    Neg(Box<LoweredOp>),
}

/// Flat post-order instruction for stack-machine evaluation.
///
/// Produced by [`LoweredOp::to_oxiblas_ops`]. Consumed by scalar or
/// SIMD batch evaluators. Post-order means leaves come before operators:
/// `a + b` encodes as `[Const(a), Const(b), Add]`.
#[derive(Clone, Debug, PartialEq)]
pub enum OxiOp {
    /// Push a constant value.
    Const(f64),
    /// Push variable `vars[i]`.
    Var(usize),
    /// Pop two, push sum.
    Add,
    /// Pop two (a, b), push a - b.
    Sub,
    /// Pop two, push product.
    Mul,
    /// Pop two (a, b), push a / b.
    Div,
    /// Pop one, push negation.
    Neg,
    /// Pop one, push exp.
    Exp,
    /// Pop one, push ln.
    Ln,
    /// Pop one, push sin.
    Sin,
    /// Pop one, push cos.
    Cos,
    /// Pop two (base, exp), push base^exp.
    Pow,
}

impl EmlTree {
    /// Lower an EML tree to a conventional operation tree.
    ///
    /// Recognizes common EML patterns (exp, ln, arithmetic) and
    /// converts them to their standard equivalents. Unrecognized
    /// subtrees are lowered as literal `exp(left) - ln(right)`.
    pub fn lower(&self) -> LoweredOp {
        lower_node(&self.root)
    }
}

/// Lower a single EML node to a `LoweredOp`.
fn lower_node(node: &EmlNode) -> LoweredOp {
    match node {
        EmlNode::One => LoweredOp::Const(1.0),
        EmlNode::Var(i) => LoweredOp::Var(*i),
        EmlNode::Eml { left, right } => {
            // Try to recognize known patterns before falling back to exp(l) - ln(r).
            // Patterns are checked most-specific first to avoid premature matches.

            // Pattern: eml(x, One) = exp(x)
            if matches!(right.as_ref(), EmlNode::One) {
                // Sub-pattern: eml(ln_tree, One) = exp(ln(x)) = x
                if let Some(inner) = match_ln_structure(left) {
                    return lower_node(&inner);
                }
                return LoweredOp::Exp(Box::new(lower_node(left)));
            }

            // Pattern: eml(One, One) = e
            if matches!(left.as_ref(), EmlNode::One) && matches!(right.as_ref(), EmlNode::One) {
                return LoweredOp::Const(std::f64::consts::E);
            }

            // Pattern: eml(One, eml(eml(One, x), One)) = ln(x)
            // MUST be checked before the e-x pattern since it's more specific.
            if matches!(left.as_ref(), EmlNode::One) {
                if let Some(inner) = match_ln_of_right(right) {
                    return LoweredOp::Ln(Box::new(lower_node(&inner)));
                }
            }

            // Pattern: eml(One, eml(x, One)) = e - x
            if matches!(left.as_ref(), EmlNode::One) {
                if let EmlNode::Eml {
                    left: inner_l,
                    right: inner_r,
                } = right.as_ref()
                {
                    if matches!(inner_r.as_ref(), EmlNode::One) {
                        let x_lowered = lower_node(inner_l);
                        return LoweredOp::Sub(
                            Box::new(LoweredOp::Const(std::f64::consts::E)),
                            Box::new(x_lowered),
                        );
                    }
                }
            }

            // Pattern: eml(ln(x), eml(y, One)) = x - y (subtraction)
            // This is the sub() canonical construction.
            if let Some(x_inner) = match_ln_structure(left) {
                if let EmlNode::Eml {
                    left: y_node,
                    right: y_one,
                } = right.as_ref()
                {
                    if matches!(y_one.as_ref(), EmlNode::One) {
                        // eml(ln(x), eml(y, 1)) = exp(ln(x)) - ln(exp(y)) = x - y
                        return LoweredOp::Sub(
                            Box::new(lower_node(&x_inner)),
                            Box::new(lower_node(y_node)),
                        );
                    }
                }
            }

            // Default: eml(left, right) = exp(left) - ln(right)
            let left_lowered = lower_node(left);
            let right_lowered = lower_node(right);
            LoweredOp::Sub(
                Box::new(LoweredOp::Exp(Box::new(left_lowered))),
                Box::new(LoweredOp::Ln(Box::new(right_lowered))),
            )
        }
    }
}

/// Match the ln structure: `eml(1, eml(eml(1, x), 1))` → returns `x`.
fn match_ln_structure(node: &EmlNode) -> Option<EmlNode> {
    if let EmlNode::Eml { left, right } = node {
        if !matches!(left.as_ref(), EmlNode::One) {
            return None;
        }
        if let EmlNode::Eml {
            left: mid_l,
            right: mid_r,
        } = right.as_ref()
        {
            if !matches!(mid_r.as_ref(), EmlNode::One) {
                return None;
            }
            if let EmlNode::Eml {
                left: inner_l,
                right: inner_r,
            } = mid_l.as_ref()
            {
                if matches!(inner_l.as_ref(), EmlNode::One) {
                    return Some(inner_r.as_ref().clone());
                }
            }
        }
    }
    None
}

/// Match ln pattern in the right subtree of `eml(1, right)`.
/// Looks for `eml(eml(1, x), 1)` inside right, giving `ln(x)`.
fn match_ln_of_right(right: &EmlNode) -> Option<EmlNode> {
    if let EmlNode::Eml {
        left: mid_l,
        right: mid_r,
    } = right
    {
        if !matches!(mid_r.as_ref(), EmlNode::One) {
            return None;
        }
        if let EmlNode::Eml {
            left: inner_l,
            right: inner_r,
        } = mid_l.as_ref()
        {
            if matches!(inner_l.as_ref(), EmlNode::One) {
                return Some(inner_r.as_ref().clone());
            }
        }
    }
    None
}

impl LoweredOp {
    /// Flatten this tree into a post-order instruction list for stack-machine evaluation.
    ///
    /// The returned slice can be fed to [`Self::eval_ops`] for scalar evaluation
    /// or to `simd_eval::eval_batch_simd` for SIMD-accelerated batch evaluation.
    pub fn to_oxiblas_ops(&self) -> Vec<OxiOp> {
        let mut ops = Vec::new();
        self.collect_ops(&mut ops);
        ops
    }

    fn collect_ops(&self, ops: &mut Vec<OxiOp>) {
        match self {
            Self::Const(c) => ops.push(OxiOp::Const(*c)),
            Self::Var(i) => ops.push(OxiOp::Var(*i)),
            Self::Add(a, b) => {
                a.collect_ops(ops);
                b.collect_ops(ops);
                ops.push(OxiOp::Add);
            }
            Self::Sub(a, b) => {
                a.collect_ops(ops);
                b.collect_ops(ops);
                ops.push(OxiOp::Sub);
            }
            Self::Mul(a, b) => {
                a.collect_ops(ops);
                b.collect_ops(ops);
                ops.push(OxiOp::Mul);
            }
            Self::Div(a, b) => {
                a.collect_ops(ops);
                b.collect_ops(ops);
                ops.push(OxiOp::Div);
            }
            Self::Exp(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Exp);
            }
            Self::Ln(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Ln);
            }
            Self::Sin(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Sin);
            }
            Self::Cos(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Cos);
            }
            Self::Pow(a, b) => {
                a.collect_ops(ops);
                b.collect_ops(ops);
                ops.push(OxiOp::Pow);
            }
            Self::Neg(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Neg);
            }
        }
    }

    /// Evaluate a flat instruction list over scalar variable values.
    ///
    /// Runs a stack machine: push leaves, pop operands for each operator.
    /// Returns `f64::NAN` for stack underflow (malformed instruction sequence).
    pub fn eval_ops(ops: &[OxiOp], vars: &[f64]) -> f64 {
        let mut stack: Vec<f64> = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                OxiOp::Const(c) => stack.push(*c),
                OxiOp::Var(i) => {
                    stack.push(vars.get(*i).copied().unwrap_or(f64::NAN));
                }
                OxiOp::Add => {
                    let b = stack.pop().unwrap_or(f64::NAN);
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a + b);
                }
                OxiOp::Sub => {
                    let b = stack.pop().unwrap_or(f64::NAN);
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a - b);
                }
                OxiOp::Mul => {
                    let b = stack.pop().unwrap_or(f64::NAN);
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a * b);
                }
                OxiOp::Div => {
                    let b = stack.pop().unwrap_or(f64::NAN);
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a / b);
                }
                OxiOp::Neg => {
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(-a);
                }
                OxiOp::Exp => {
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a.exp());
                }
                OxiOp::Ln => {
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a.ln());
                }
                OxiOp::Sin => {
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a.sin());
                }
                OxiOp::Cos => {
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a.cos());
                }
                OxiOp::Pow => {
                    let b = stack.pop().unwrap_or(f64::NAN);
                    let a = stack.pop().unwrap_or(f64::NAN);
                    stack.push(a.powf(b));
                }
            }
        }
        stack.pop().unwrap_or(f64::NAN)
    }

    /// Evaluate a batch of data points using the flat IR. Uses SIMD when the
    /// `simd` feature is enabled; otherwise delegates to scalar evaluation.
    ///
    /// Returns a `Vec<f64>` of the same length as `data`. Unlike
    /// [`crate::eval::EvalCtx`]-based evaluation, NaN/inf propagate silently
    /// (no `Result` wrapping) — the IR layer treats them as valid f64 values.
    pub fn eval_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
        let ops = self.to_oxiblas_ops();
        #[cfg(feature = "simd")]
        {
            crate::simd_eval::eval_batch_simd(&ops, data)
        }
        #[cfg(not(feature = "simd"))]
        {
            Self::eval_batch_scalar_from_ops(&ops, data)
        }
    }

    /// Scalar batch evaluation over a pre-built flat IR slice.
    ///
    /// Exposed as `pub` so the `simd_eval` stub and SIMD remainder path can
    /// delegate to it without re-encoding the tree.
    pub fn eval_batch_scalar_from_ops(ops: &[OxiOp], data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|row| Self::eval_ops(ops, row)).collect()
    }

    /// Scalar batch evaluation building the flat IR internally.
    pub fn eval_batch_scalar(&self, data: &[Vec<f64>]) -> Vec<f64> {
        let ops = self.to_oxiblas_ops();
        Self::eval_batch_scalar_from_ops(&ops, data)
    }

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
        }
    }

    /// Convert to a human-readable mathematical expression string.
    pub fn to_pretty(&self) -> String {
        format!("{self}")
    }

    /// Evaluate the lowered operation tree with the given variable values.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        match self {
            Self::Const(c) => *c,
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
        }
    }

    /// Simplify the lowered operation tree.
    ///
    /// Applies constant folding and algebraic simplifications.
    pub fn simplify(&self) -> Self {
        match self {
            Self::Add(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                // 0 + x = x
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return b_s;
                    }
                }
                // x + 0 = x
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return a_s;
                    }
                }
                // const + const
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c + b_c);
                }
                Self::Add(Box::new(a_s), Box::new(b_s))
            }
            Self::Sub(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                // x - 0 = x
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return a_s;
                    }
                }
                // 0 - x = -x
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return Self::Neg(Box::new(b_s));
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c - b_c);
                }
                Self::Sub(Box::new(a_s), Box::new(b_s))
            }
            Self::Mul(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                // 0 * x = 0
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return Self::Const(0.0);
                    }
                }
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return Self::Const(0.0);
                    }
                }
                // 1 * x = x
                if let Self::Const(c) = &a_s {
                    if (*c - 1.0).abs() < 1e-15 {
                        return b_s;
                    }
                }
                if let Self::Const(c) = &b_s {
                    if (*c - 1.0).abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c * b_c);
                }
                Self::Mul(Box::new(a_s), Box::new(b_s))
            }
            Self::Div(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                // x / 1 = x
                if let Self::Const(c) = &b_s {
                    if (*c - 1.0).abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    if b_c.abs() > 1e-15 {
                        return Self::Const(a_c / b_c);
                    }
                }
                Self::Div(Box::new(a_s), Box::new(b_s))
            }
            Self::Exp(a) => {
                let a_s = a.simplify();
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return Self::Const(1.0); // exp(0) = 1
                    }
                }
                // exp(ln(x)) = x
                if let Self::Ln(inner) = &a_s {
                    return *inner.clone();
                }
                Self::Exp(Box::new(a_s))
            }
            Self::Ln(a) => {
                let a_s = a.simplify();
                if let Self::Const(c) = &a_s {
                    if (*c - 1.0).abs() < 1e-15 {
                        return Self::Const(0.0); // ln(1) = 0
                    }
                }
                // ln(exp(x)) = x
                if let Self::Exp(inner) = &a_s {
                    return *inner.clone();
                }
                Self::Ln(Box::new(a_s))
            }
            Self::Neg(a) => {
                let a_s = a.simplify();
                if let Self::Const(c) = &a_s {
                    return Self::Const(-c);
                }
                // neg(neg(x)) = x
                if let Self::Neg(inner) = &a_s {
                    return *inner.clone();
                }
                Self::Neg(Box::new(a_s))
            }
            Self::Pow(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                // x^0 = 1
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return Self::Const(1.0);
                    }
                    // x^1 = x
                    if (*c - 1.0).abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c.powf(*b_c));
                }
                Self::Pow(Box::new(a_s), Box::new(b_s))
            }
            Self::Sin(a) => Self::Sin(Box::new(a.simplify())),
            Self::Cos(a) => Self::Cos(Box::new(a.simplify())),
            Self::Const(_) | Self::Var(_) => self.clone(),
        }
    }
}

impl fmt::Display for LoweredOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_one() {
        let t = EmlTree::one();
        let lowered = t.lower();
        assert_eq!(lowered, LoweredOp::Const(1.0));
    }

    #[test]
    fn test_lower_var() {
        let t = EmlTree::var(0);
        let lowered = t.lower();
        assert_eq!(lowered, LoweredOp::Var(0));
    }

    #[test]
    fn test_lower_exp() {
        // eml(x, 1) → exp(x)
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let lowered = exp_x.lower();
        assert_eq!(lowered, LoweredOp::Exp(Box::new(LoweredOp::Var(0))));
    }

    #[test]
    fn test_lower_e_minus_x() {
        // eml(1, eml(x, 1)) → e - x
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let e_minus_x = EmlTree::eml(&one, &exp_x);
        let lowered = e_minus_x.lower();
        assert_eq!(
            lowered,
            LoweredOp::Sub(
                Box::new(LoweredOp::Const(std::f64::consts::E)),
                Box::new(LoweredOp::Var(0)),
            )
        );
    }

    #[test]
    fn test_lower_ln() {
        // eml(1, eml(eml(1, x), 1)) → ln(x)
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let inner = EmlTree::eml(&one, &x); // eml(1, x)
        let middle = EmlTree::eml(&inner, &one); // eml(eml(1,x), 1)
        let ln_x = EmlTree::eml(&one, &middle); // eml(1, eml(eml(1,x), 1))
        let lowered = ln_x.lower();
        assert_eq!(lowered, LoweredOp::Ln(Box::new(LoweredOp::Var(0))));
    }

    #[test]
    fn test_lowered_eval() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(3.0)));
        assert!((op.eval(&[2.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_pretty_print() {
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        assert_eq!(op.to_pretty(), "(x0 * x1)");
    }

    #[test]
    fn test_simplify_exp_ln() {
        // exp(ln(x)) → x
        let op = LoweredOp::Exp(Box::new(LoweredOp::Ln(Box::new(LoweredOp::Var(0)))));
        let simplified = op.simplify();
        assert_eq!(simplified, LoweredOp::Var(0));
    }

    #[test]
    fn test_simplify_constants() {
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Const(2.0)),
            Box::new(LoweredOp::Const(3.0)),
        );
        let simplified = op.simplify();
        assert_eq!(simplified, LoweredOp::Const(5.0));
    }

    #[test]
    fn test_to_oxiblas_ops_roundtrip() {
        use crate::Canonical;
        // exp(x)
        let x = crate::tree::EmlTree::var(0);
        let exp_x = Canonical::exp(&x);
        let lowered = exp_x.lower();
        let ops = lowered.to_oxiblas_ops();
        let result = LoweredOp::eval_ops(&ops, &[1.5_f64]);
        assert!(
            (result - 1.5_f64.exp()).abs() < 1e-12,
            "exp roundtrip failed: {result}"
        );

        // ln(x)
        let ln_x = Canonical::ln(&x);
        let lowered_ln = ln_x.lower();
        let ops_ln = lowered_ln.to_oxiblas_ops();
        let result_ln = LoweredOp::eval_ops(&ops_ln, &[2.0_f64]);
        assert!(
            (result_ln - 2.0_f64.ln()).abs() < 1e-12,
            "ln roundtrip failed: {result_ln}"
        );

        // sin(x) — directly construct LoweredOp::Sin to test the Sin opcode
        // (Canonical::sin uses complex arithmetic and requires complex evaluation)
        let lowered_sin = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let ops_sin = lowered_sin.to_oxiblas_ops();
        let result_sin = LoweredOp::eval_ops(&ops_sin, &[std::f64::consts::PI / 6.0]);
        assert!(
            (result_sin - 0.5_f64).abs() < 1e-9,
            "sin roundtrip failed: {result_sin}"
        );
    }

    #[test]
    fn test_eval_batch_scalar_matches_eval() {
        use crate::Canonical;
        let x = crate::tree::EmlTree::var(0);
        let exp_x = Canonical::exp(&x);
        let lowered = exp_x.lower();

        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.05]).collect();
        let batch_results = lowered.eval_batch_scalar(&data);
        assert_eq!(batch_results.len(), 100);
        for (row, result) in data.iter().zip(batch_results.iter()) {
            let expected = lowered.eval(row);
            assert!(
                (result - expected).abs() < 1e-12,
                "mismatch at x={}: got {result}, expected {expected}",
                row[0]
            );
        }
    }

    #[test]
    fn test_structural_hash_differs() {
        use crate::Canonical;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let x = crate::tree::EmlTree::var(0);
        let exp_x = Canonical::exp(&x).lower().simplify();
        let ln_x = Canonical::ln(&x).lower().simplify();

        let mut h1 = DefaultHasher::new();
        exp_x.structural_hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        ln_x.structural_hash(&mut h2);
        assert_ne!(
            h1.finish(),
            h2.finish(),
            "exp and ln should have different structural hashes"
        );
    }

    #[test]
    fn test_structural_hash_same_for_equiv() {
        use crate::Canonical;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let x = crate::tree::EmlTree::var(0);
        let exp_x1 = Canonical::exp(&x).lower().simplify();
        let exp_x2 = Canonical::exp(&x).lower().simplify();

        let mut h1 = DefaultHasher::new();
        exp_x1.structural_hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        exp_x2.structural_hash(&mut h2);
        assert_eq!(
            h1.finish(),
            h2.finish(),
            "identical trees should have the same structural hash"
        );
    }
}
