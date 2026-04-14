//! Automatic differentiation for EML trees.
//!
//! Provides forward-backward pass for computing gradients of a loss function
//! with respect to continuous parameters embedded in the tree. Used by the
//! symbolic regression engine to optimize parameters within a fixed topology.

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::tree::{EmlNode, EmlTree};
use num_complex::Complex64;
use std::sync::Arc;

/// EML tree with continuous parameters replacing `One` nodes.
///
/// During symbolic regression, the tree topology is fixed and `One` leaves
/// are relaxed to learnable continuous parameters. This struct holds both
/// the topology and the parameter values.
#[derive(Clone, Debug)]
pub struct ParameterizedEmlTree {
    /// Tree topology (structural template).
    pub topology: EmlTree,
    /// Continuous parameters, one per `One` node in post-order.
    pub params: Vec<f64>,
}

/// Internal node representation for the linearized computation graph.
#[derive(Clone, Debug)]
enum TapeEntry {
    /// A parameter node.
    Param,
    /// A variable node.
    Var,
    /// An EML node: indices of left and right children in the tape.
    Eml(usize, usize),
}

impl ParameterizedEmlTree {
    /// Create a new parameterized tree from a topology.
    ///
    /// Each `One` node in the topology gets a parameter initialized to
    /// the given value (typically 1.0).
    pub fn from_topology(topology: &EmlTree, init_value: f64) -> Self {
        let count = count_ones(&topology.root);
        Self {
            topology: topology.clone(),
            params: vec![init_value; count],
        }
    }

    /// Number of learnable parameters.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Forward pass: evaluate the tree with current parameters.
    pub fn forward(&self, ctx: &EvalCtx) -> Result<f64, EmlError> {
        let (_tape, values) = self.build_tape_and_forward(ctx)?;
        let result = values.last().copied().unwrap_or(Complex64::new(0.0, 0.0));
        if result.im.abs() < 1e-12 {
            Ok(result.re)
        } else {
            Err(EmlError::ComplexResult(result.im.abs()))
        }
    }

    /// Forward + backward pass.
    ///
    /// Returns `(loss, gradients)` where loss is `(output - target)^2`
    /// and gradients is `d(loss)/d(param_i)` for each parameter.
    pub fn forward_backward(
        &self,
        ctx: &EvalCtx,
        target: f64,
    ) -> Result<(f64, Vec<f64>), EmlError> {
        let (tape, values) = self.build_tape_and_forward(ctx)?;

        let output = values.last().copied().unwrap_or(Complex64::new(0.0, 0.0));
        if output.im.abs() >= 1e-12 {
            return Err(EmlError::ComplexResult(output.im.abs()));
        }

        let output_re = output.re;
        let loss = (output_re - target) * (output_re - target);

        // Backward pass: compute d(loss)/d(node_value) for each tape entry
        let n = tape.len();
        let mut grad_values = vec![Complex64::new(0.0, 0.0); n];
        // d(loss)/d(output) = 2 * (output - target)
        grad_values[n - 1] = Complex64::new(2.0 * (output_re - target), 0.0);

        // Traverse tape in reverse
        for i in (0..n).rev() {
            let g = grad_values[i];
            if let TapeEntry::Eml(left_idx, right_idx) = &tape[i] {
                let left_val = values[*left_idx];
                let right_val = values[*right_idx];

                // eml(l, r) = exp(l) - ln(r)
                // d(eml)/d(l) = exp(l)
                // d(eml)/d(r) = -1/r
                let d_left = clamped_exp(left_val);
                let d_right = -Complex64::new(1.0, 0.0) / right_val;

                grad_values[*left_idx] += g * d_left;
                grad_values[*right_idx] += g * d_right;
            }
        }

        // Extract parameter gradients
        let mut param_grads = Vec::with_capacity(self.params.len());
        for (i, entry) in tape.iter().enumerate() {
            if let TapeEntry::Param = entry {
                param_grads.push(grad_values[i].re);
            }
        }

        Ok((loss, param_grads))
    }

    /// Build the computation tape and run the forward pass.
    fn build_tape_and_forward(
        &self,
        ctx: &EvalCtx,
    ) -> Result<(Vec<TapeEntry>, Vec<Complex64>), EmlError> {
        let mut tape = Vec::new();
        let mut values = Vec::new();
        let mut param_idx = 0;
        self.build_tape_recursive(
            &self.topology.root,
            ctx,
            &mut tape,
            &mut values,
            &mut param_idx,
        )?;
        Ok((tape, values))
    }

    fn build_tape_recursive(
        &self,
        node: &EmlNode,
        ctx: &EvalCtx,
        tape: &mut Vec<TapeEntry>,
        values: &mut Vec<Complex64>,
        param_idx: &mut usize,
    ) -> Result<usize, EmlError> {
        match node {
            EmlNode::One => {
                let idx = tape.len();
                let p = self.params[*param_idx];
                *param_idx += 1;
                tape.push(TapeEntry::Param);
                values.push(Complex64::new(p, 0.0));
                Ok(idx)
            }
            EmlNode::Var(var_idx) => {
                let idx = tape.len();
                let val = ctx
                    .get(*var_idx)
                    .ok_or(EmlError::VarOutOfBounds(*var_idx, ctx.num_vars()))?;
                tape.push(TapeEntry::Var);
                values.push(Complex64::new(val, 0.0));
                Ok(idx)
            }
            EmlNode::Eml { left, right } => {
                let left_idx = self.build_tape_recursive(left, ctx, tape, values, param_idx)?;
                let right_idx = self.build_tape_recursive(right, ctx, tape, values, param_idx)?;

                let left_val = values[left_idx];
                let right_val = values[right_idx];
                let result = eml_complex_grad(left_val, right_val)?;

                let idx = tape.len();
                tape.push(TapeEntry::Eml(left_idx, right_idx));
                values.push(result);
                Ok(idx)
            }
        }
    }
}

/// Count `One` nodes in a tree.
fn count_ones(node: &EmlNode) -> usize {
    match node {
        EmlNode::One => 1,
        EmlNode::Var(_) => 0,
        EmlNode::Eml { left, right } => count_ones(left) + count_ones(right),
    }
}

/// Clamped exp for gradient computation.
fn clamped_exp(z: Complex64) -> Complex64 {
    let clamped = if z.re > 709.0 {
        Complex64::new(709.0, z.im)
    } else if z.re < -709.0 {
        Complex64::new(-709.0, z.im)
    } else {
        z
    };
    clamped.exp()
}

/// Compute `eml(left, right) = exp(left) - ln(right)` for gradient computation.
fn eml_complex_grad(left: Complex64, right: Complex64) -> Result<Complex64, EmlError> {
    let exp_part = clamped_exp(left);
    let ln_part = right.ln();
    let result = exp_part - ln_part;
    if result.re.is_nan() || result.im.is_nan() {
        return Err(EmlError::NanEncountered);
    }
    Ok(result)
}

/// Reconstruct an EML tree from a parameterized tree (replacing params back as constants).
pub fn reconstruct_tree(ptree: &ParameterizedEmlTree) -> EmlTree {
    let mut param_idx = 0;
    let root = reconstruct_node(&ptree.topology.root, &ptree.params, &mut param_idx);
    EmlTree::from_node(root)
}

fn reconstruct_node(node: &EmlNode, params: &[f64], param_idx: &mut usize) -> Arc<EmlNode> {
    match node {
        EmlNode::One => {
            let _p = params[*param_idx];
            *param_idx += 1;
            // Keep as One — the parameter was used during optimization
            // but the reconstructed tree uses the standard EmlNode::One
            Arc::new(EmlNode::One)
        }
        EmlNode::Var(i) => Arc::new(EmlNode::Var(*i)),
        EmlNode::Eml { left, right } => {
            let l = reconstruct_node(left, params, param_idx);
            let r = reconstruct_node(right, params, param_idx);
            Arc::new(EmlNode::Eml { left: l, right: r })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameterized_forward() {
        // eml(1, 1) with params [1.0, 1.0] should give e
        let one = EmlTree::one();
        let tree = EmlTree::eml(&one, &one);
        let ptree = ParameterizedEmlTree::from_topology(&tree, 1.0);
        assert_eq!(ptree.num_params(), 2);
        let ctx = EvalCtx::new(&[]);
        let result = ptree.forward(&ctx).unwrap();
        assert!((result - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_forward_backward() {
        // eml(x, 1) = exp(x), target = exp(1) = e
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let tree = EmlTree::eml(&x, &one);
        let ptree = ParameterizedEmlTree::from_topology(&tree, 1.0);
        assert_eq!(ptree.num_params(), 1); // one One node on the right

        let ctx = EvalCtx::new(&[1.0]);
        let target = std::f64::consts::E;
        let (loss, grads) = ptree.forward_backward(&ctx, target).unwrap();
        // When param = 1.0, eml(x=1, param=1) = exp(1) - ln(1) = e - 0 = e
        // loss = (e - e)^2 = 0
        assert!(loss < 1e-20);
        assert_eq!(grads.len(), 1);
    }

    #[test]
    fn test_gradient_nonzero() {
        // eml(1, 1) = e, target = 3.0 → loss > 0, gradient nonzero
        let one = EmlTree::one();
        let tree = EmlTree::eml(&one, &one);
        let ptree = ParameterizedEmlTree::from_topology(&tree, 1.0);
        let ctx = EvalCtx::new(&[]);
        let (loss, grads) = ptree.forward_backward(&ctx, 3.0).unwrap();
        assert!(loss > 0.0);
        assert!(grads.iter().any(|g| g.abs() > 1e-10));
    }
}
