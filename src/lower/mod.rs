//! Lowering EML trees to standard mathematical operations.
//!
//! The EML representation is optimal for *discovery* (uniform search space)
//! but inefficient for *execution* (a single multiplication requires 41+ nodes).
//! Lowering converts EML trees to conventional operation trees for efficient
//! evaluation and human-readable output.

pub mod display;
pub mod oxiblas;
pub mod pattern;

pub use crate::lower_interval::IntervalLO;
pub use oxiblas::OxiOp;

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::named_const::NamedConst;
use crate::tree::EmlTree;
use std::sync::Arc;

/// A conventional mathematical operation tree.
///
/// Produced by lowering an EML tree. Supports efficient evaluation
/// and pretty-printing.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LoweredOp {
    /// Constant value.
    Const(f64),
    /// Input variable.
    Var(usize),
    /// Addition.
    Add(Arc<LoweredOp>, Arc<LoweredOp>),
    /// Subtraction.
    Sub(Arc<LoweredOp>, Arc<LoweredOp>),
    /// Multiplication.
    Mul(Arc<LoweredOp>, Arc<LoweredOp>),
    /// Division.
    Div(Arc<LoweredOp>, Arc<LoweredOp>),
    /// Exponential function.
    Exp(Arc<LoweredOp>),
    /// Natural logarithm.
    Ln(Arc<LoweredOp>),
    /// Sine.
    Sin(Arc<LoweredOp>),
    /// Cosine.
    Cos(Arc<LoweredOp>),
    /// Power.
    Pow(Arc<LoweredOp>, Arc<LoweredOp>),
    /// Negation.
    Neg(Arc<LoweredOp>),
    /// Tangent.
    Tan(Arc<LoweredOp>),
    /// Hyperbolic sine.
    Sinh(Arc<LoweredOp>),
    /// Hyperbolic cosine.
    Cosh(Arc<LoweredOp>),
    /// Hyperbolic tangent.
    Tanh(Arc<LoweredOp>),
    /// Inverse sine (arcsine).
    Arcsin(Arc<LoweredOp>),
    /// Inverse cosine (arccosine).
    Arccos(Arc<LoweredOp>),
    /// Inverse tangent (arctangent).
    Arctan(Arc<LoweredOp>),
    /// Inverse hyperbolic sine.
    Arcsinh(Arc<LoweredOp>),
    /// Inverse hyperbolic cosine.
    Arccosh(Arc<LoweredOp>),
    /// Inverse hyperbolic tangent.
    Arctanh(Arc<LoweredOp>),
    /// A named mathematical constant (π, e, √2, …).
    ///
    /// Created only by the constants-extraction pass in [`crate::symreg`];
    /// never emitted by lowering. Constant-folds down to `Const(value())` on
    /// the first `simplify` call that encounters it in a binary/unary context.
    NamedConst(NamedConst),
    /// Error function erf(x).
    Erf(Arc<LoweredOp>),
    /// Natural log of the Gamma function.
    LGamma(Arc<LoweredOp>),
    /// Digamma function ψ(x) = d/dx ln Γ(x).
    Digamma(Arc<LoweredOp>),
    /// Trigamma function ψ¹(x) = d/dx ψ(x) = d²/dx² ln Γ(x).
    Trigamma(Arc<LoweredOp>),
    /// Exponential integral Ei(x).
    Ei(Arc<LoweredOp>),
    /// Sine integral Si(x).
    Si(Arc<LoweredOp>),
    /// Cosine integral Ci(x).
    Ci(Arc<LoweredOp>),
}

impl EmlTree {
    /// Lower an EML tree to a conventional operation tree.
    ///
    /// Recognizes common EML patterns (exp, ln, arithmetic) and
    /// converts them to their standard equivalents. Unrecognized
    /// subtrees are lowered as literal `exp(left) - ln(right)`.
    pub fn lower(&self) -> LoweredOp {
        pattern::lower_node(&self.root)
    }

    /// Evaluate the tree at real-valued variables **via the lowered IR**.
    ///
    /// Unlike [`EmlTree::eval_real`], which walks the raw EML tree through
    /// complex arithmetic (and accumulates ~1e-2 precision drift on deep
    /// constructions such as `Canonical::sin(x)`), this method first lowers
    /// the tree (recognising `sin`/`cos`/arithmetic patterns), simplifies
    /// the lowered IR, and evaluates through the `OxiOp` stack machine.
    ///
    /// Because the stack machine dispatches directly to `f64::sin`/`f64::cos`
    /// when the lowering recognised a trig pattern, the result attains
    /// full `f64` precision (~1e-15).
    ///
    /// # Errors
    /// Returns `Err(EmlError::NanEncountered)` if the IR evaluates to NaN.
    pub fn eval_real_lowered(&self, ctx: &EvalCtx) -> Result<f64, EmlError> {
        let cse_root = self.lower().simplify().cse();
        let (ops, n_slots) = cse_root.to_oxiblas_ops_shared();
        let result = LoweredOp::eval_ops_shared(&ops, ctx.as_slice(), n_slots);
        if result.is_nan() {
            return Err(EmlError::NanEncountered);
        }
        Ok(result)
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
        assert_eq!(lowered, LoweredOp::Exp(Arc::new(LoweredOp::Var(0))));
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
                Arc::new(LoweredOp::Const(std::f64::consts::E)),
                Arc::new(LoweredOp::Var(0)),
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
        assert_eq!(lowered, LoweredOp::Ln(Arc::new(LoweredOp::Var(0))));
    }

    #[test]
    fn test_lowered_eval() {
        let op = LoweredOp::Add(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(3.0)));
        assert!((op.eval(&[2.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_pretty_print() {
        let op = LoweredOp::Mul(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Var(1)));
        assert_eq!(op.to_pretty(), "(x0 * x1)");
    }

    #[test]
    fn test_simplify_exp_ln() {
        // exp(ln(x)) → x
        let op = LoweredOp::Exp(Arc::new(LoweredOp::Ln(Arc::new(LoweredOp::Var(0)))));
        let simplified = op.simplify();
        assert_eq!(simplified, LoweredOp::Var(0));
    }

    #[test]
    fn test_simplify_constants() {
        let op = LoweredOp::Add(
            Arc::new(LoweredOp::Const(2.0)),
            Arc::new(LoweredOp::Const(3.0)),
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
        let lowered_sin = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
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

    #[test]
    fn latex_var() {
        assert_eq!(LoweredOp::Var(0).to_latex(), "x_{0}");
        assert_eq!(LoweredOp::Var(3).to_latex(), "x_{3}");
    }

    #[test]
    fn latex_const_pi() {
        assert_eq!(LoweredOp::Const(std::f64::consts::PI).to_latex(), r"\pi");
    }

    #[test]
    fn latex_const_e() {
        assert_eq!(LoweredOp::Const(std::f64::consts::E).to_latex(), "e");
    }

    #[test]
    fn latex_const_integer() {
        assert_eq!(LoweredOp::Const(2.0).to_latex(), "2");
        assert_eq!(LoweredOp::Const(-1.0).to_latex(), "-1");
    }

    #[test]
    fn latex_div() {
        let op = LoweredOp::Div(Arc::new(LoweredOp::Const(1.0)), Arc::new(LoweredOp::Var(0)));
        assert_eq!(op.to_latex(), r"\frac{1}{x_{0}}");
    }

    #[test]
    fn latex_exp() {
        let op = LoweredOp::Exp(Arc::new(LoweredOp::Var(0)));
        assert_eq!(op.to_latex(), r"e^{x_{0}}");
    }

    #[test]
    fn latex_ln() {
        let op = LoweredOp::Ln(Arc::new(LoweredOp::Var(0)));
        assert_eq!(op.to_latex(), r"\ln\left(x_{0}\right)");
    }

    #[test]
    fn latex_sin_cos() {
        let op = LoweredOp::Sin(Arc::new(LoweredOp::Var(0)));
        assert_eq!(op.to_latex(), r"\sin\left(x_{0}\right)");
        let op2 = LoweredOp::Cos(Arc::new(LoweredOp::Var(0)));
        assert_eq!(op2.to_latex(), r"\cos\left(x_{0}\right)");
    }

    #[test]
    fn latex_pow() {
        let op = LoweredOp::Pow(Arc::new(LoweredOp::Var(0)), Arc::new(LoweredOp::Const(2.0)));
        assert_eq!(op.to_latex(), "x_{0}^{2}");
    }

    #[test]
    fn latex_neg() {
        let op = LoweredOp::Neg(Arc::new(LoweredOp::Var(0)));
        assert_eq!(op.to_latex(), "-x_{0}");
    }

    #[test]
    fn latex_mul() {
        let op = LoweredOp::Mul(Arc::new(LoweredOp::Const(2.0)), Arc::new(LoweredOp::Var(0)));
        assert_eq!(op.to_latex(), r"2 \cdot x_{0}");
    }

    #[test]
    fn latex_composite() {
        let op = LoweredOp::Div(
            Arc::new(LoweredOp::Sin(Arc::new(LoweredOp::Var(0)))),
            Arc::new(LoweredOp::Cos(Arc::new(LoweredOp::Var(0)))),
        );
        let latex = op.to_latex();
        assert!(latex.contains(r"\frac"));
        assert!(latex.contains(r"\sin"));
        assert!(latex.contains(r"\cos"));
    }
}
