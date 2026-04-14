//! Numerical evaluation of EML trees.
//!
//! Provides real, complex, and batch evaluation using a stack-machine
//! post-order traversal to avoid recursion depth issues on deep trees.

use crate::error::EmlError;
use crate::tree::{EmlNode, EmlTree};
use num_complex::Complex64;

/// Evaluation context holding variable bindings.
#[derive(Clone, Debug)]
pub struct EvalCtx {
    vars: Vec<f64>,
}

/// Threshold for exp argument to prevent overflow (f64::MAX ≈ 1.8e308, exp(709.78) ≈ MAX).
const EXP_CLAMP: f64 = 709.0;

/// Threshold for imaginary part to consider result as real.
const IMAG_THRESHOLD: f64 = 1e-12;

/// Minimum batch size to trigger parallel evaluation (avoids rayon overhead on small batches).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
const PARALLEL_BATCH_THRESHOLD: usize = 128;

impl EvalCtx {
    /// Create a new evaluation context with the given variable values.
    pub fn new(vars: &[f64]) -> Self {
        Self {
            vars: vars.to_vec(),
        }
    }

    /// Get the value of variable at the given index.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.vars.get(index).copied()
    }

    /// Number of variables in this context.
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }
}

impl EmlTree {
    /// Evaluate the tree with real-valued inputs.
    ///
    /// Internally uses complex arithmetic when needed (e.g., `ln` of negative
    /// numbers). Returns the real part if the imaginary part is below threshold,
    /// otherwise returns `Err(EmlError::ComplexResult)`.
    pub fn eval_real(&self, ctx: &EvalCtx) -> Result<f64, EmlError> {
        let complex_vars: Vec<Complex64> =
            ctx.vars.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let result = self.eval_complex(&complex_vars)?;

        if result.im.abs() < IMAG_THRESHOLD {
            let re = result.re;
            if re.is_nan() {
                return Err(EmlError::NanEncountered);
            }
            Ok(re)
        } else {
            Err(EmlError::ComplexResult(result.im.abs()))
        }
    }

    /// Evaluate the tree with complex-valued inputs.
    ///
    /// Uses a stack-machine post-order traversal for numerical stability
    /// and to avoid stack overflow on deeply nested trees.
    pub fn eval_complex(&self, vars: &[Complex64]) -> Result<Complex64, EmlError> {
        // Build a flattened post-order instruction list, then evaluate with a stack.
        let mut instructions = Vec::new();
        flatten_postorder(&self.root, &mut instructions);

        let mut stack: Vec<Complex64> = Vec::with_capacity(instructions.len());

        for inst in &instructions {
            match inst {
                Instruction::PushOne => {
                    stack.push(Complex64::new(1.0, 0.0));
                }
                Instruction::PushVar(idx) => {
                    let idx = *idx;
                    if idx >= vars.len() {
                        return Err(EmlError::VarOutOfBounds(idx, vars.len()));
                    }
                    stack.push(vars[idx]);
                }
                Instruction::Eml => {
                    let right = stack.pop().ok_or(EmlError::NanEncountered)?;
                    let left = stack.pop().ok_or(EmlError::NanEncountered)?;
                    let result = eml_complex(left, right)?;
                    stack.push(result);
                }
            }
        }

        debug_assert_eq!(stack.len(), 1);
        Ok(stack[0])
    }

    /// Evaluate the tree on a batch of data points.
    ///
    /// Each element of `data` is a vector of variable values for one data point.
    /// Returns a vector of results, one per data point.
    /// When the `parallel` feature is enabled and `data.len() >= 128`,
    /// evaluation is distributed across rayon threads.
    pub fn eval_batch(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, EmlError> {
        if data.is_empty() {
            return Err(EmlError::EmptyData);
        }

        let mut instructions = Vec::new();
        flatten_postorder(&self.root, &mut instructions);

        #[cfg(feature = "parallel")]
        if data.len() >= PARALLEL_BATCH_THRESHOLD {
            use rayon::prelude::*;
            return data
                .par_iter()
                .map(|point| eval_point(&instructions, point))
                .collect::<Result<Vec<f64>, EmlError>>();
        }

        // Sequential path (default, or when batch is small)
        data.iter()
            .map(|point| eval_point(&instructions, point))
            .collect::<Result<Vec<f64>, EmlError>>()
    }
}

/// Internal instruction for the stack-machine evaluator.
#[derive(Clone, Debug)]
enum Instruction {
    PushOne,
    PushVar(usize),
    Eml,
}

/// Evaluate a single data point using a pre-built instruction list.
fn eval_point(instructions: &[Instruction], point: &[f64]) -> Result<f64, EmlError> {
    let complex_vars: Vec<Complex64> = point.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    let mut stack: Vec<Complex64> = Vec::with_capacity(instructions.len());

    for inst in instructions {
        match inst {
            Instruction::PushOne => {
                stack.push(Complex64::new(1.0, 0.0));
            }
            Instruction::PushVar(idx) => {
                let idx = *idx;
                if idx >= complex_vars.len() {
                    return Err(EmlError::VarOutOfBounds(idx, complex_vars.len()));
                }
                stack.push(complex_vars[idx]);
            }
            Instruction::Eml => {
                let right = stack.pop().ok_or(EmlError::NanEncountered)?;
                let left = stack.pop().ok_or(EmlError::NanEncountered)?;
                let result = eml_complex(left, right)?;
                stack.push(result);
            }
        }
    }

    let result = stack[0];
    if result.im.abs() < IMAG_THRESHOLD {
        if result.re.is_nan() {
            return Err(EmlError::NanEncountered);
        }
        Ok(result.re)
    } else {
        Err(EmlError::ComplexResult(result.im.abs()))
    }
}

/// Flatten an EML tree into post-order instructions.
fn flatten_postorder(node: &EmlNode, out: &mut Vec<Instruction>) {
    match node {
        EmlNode::One => out.push(Instruction::PushOne),
        EmlNode::Var(idx) => out.push(Instruction::PushVar(*idx)),
        EmlNode::Eml { left, right } => {
            flatten_postorder(left, out);
            flatten_postorder(right, out);
            out.push(Instruction::Eml);
        }
    }
}

/// Compute `eml(left, right) = exp(left) - ln(right)` for complex values.
fn eml_complex(left: Complex64, right: Complex64) -> Result<Complex64, EmlError> {
    // Clamp real part of left to prevent exp overflow
    let clamped_left = if left.re > EXP_CLAMP {
        Complex64::new(EXP_CLAMP, left.im)
    } else if left.re < -EXP_CLAMP {
        Complex64::new(-EXP_CLAMP, left.im)
    } else {
        left
    };

    let exp_part = clamped_left.exp();
    let ln_part = right.ln();

    let result = exp_part - ln_part;

    if result.re.is_nan() || result.im.is_nan() {
        return Err(EmlError::NanEncountered);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::EmlTree;

    #[test]
    fn test_eval_one() {
        let t = EmlTree::one();
        let ctx = EvalCtx::new(&[]);
        // One evaluates... well, One is a leaf, not eml.
        // We need to handle the fact that One by itself is just the constant 1.
        // But eval_real goes through eval_complex which handles leaves.
        let result = t.eval_real(&ctx).unwrap();
        assert!((result - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_eval_var() {
        let t = EmlTree::var(0);
        let ctx = EvalCtx::new(&[2.71]);
        let result = t.eval_real(&ctx).unwrap();
        assert!((result - 2.71).abs() < 1e-15);
    }

    #[test]
    fn test_eval_exp() {
        // eml(x, 1) = exp(x) - ln(1) = exp(x)
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let ctx = EvalCtx::new(&[1.0]);
        let result = exp_x.eval_real(&ctx).unwrap();
        assert!((result - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_eval_euler() {
        // eml(1, 1) = exp(1) - ln(1) = e
        let one = EmlTree::one();
        let euler = EmlTree::eml(&one, &one);
        let ctx = EvalCtx::new(&[]);
        let result = euler.eval_real(&ctx).unwrap();
        assert!((result - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_eval_batch() {
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let results = exp_x.eval_batch(&data).unwrap();
        assert!((results[0] - 1.0).abs() < 1e-10);
        assert!((results[1] - std::f64::consts::E).abs() < 1e-10);
        assert!((results[2] - (2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_var_out_of_bounds() {
        let t = EmlTree::var(5);
        let ctx = EvalCtx::new(&[1.0]);
        assert!(matches!(
            t.eval_real(&ctx),
            Err(EmlError::VarOutOfBounds(5, 1))
        ));
    }

    #[test]
    fn test_eval_batch_parallel() {
        // 200 points > PARALLEL_BATCH_THRESHOLD (128); exercises the parallel path when feature on.
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let data: Vec<Vec<f64>> = (0..200).map(|i| vec![i as f64 * 0.01]).collect();
        let results = exp_x.eval_batch(&data).unwrap();
        assert_eq!(results.len(), 200);
        for (i, &r) in results.iter().enumerate() {
            let expected = (i as f64 * 0.01_f64).exp();
            assert!(
                (r - expected).abs() < 1e-8,
                "index {i}: got {r}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_eval_batch_parallel_error_short_circuit() {
        // A batch containing a var-out-of-bounds row should propagate the error.
        let t = EmlTree::var(5);
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = t.eval_batch(&data);
        assert!(matches!(result, Err(EmlError::VarOutOfBounds(5, 1))));
    }
}
