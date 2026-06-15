//! Post-order flat-IR emission and stack-machine evaluation for `LoweredOp`.
//!
//! Produces the `OxiOp` `Vec` consumed by scalar and SIMD batch evaluators.

use super::LoweredOp;
use std::collections::HashMap;
use std::sync::Arc;

/// Pop the top value from the stack, returning `f64::NAN` on underflow.
///
/// In debug builds a `debug_assert!` fires immediately on underflow, providing
/// a clear panic message instead of silently propagating NaN.
#[inline(always)]
fn pop_or_nan(stack: &mut Vec<f64>) -> f64 {
    debug_assert!(!stack.is_empty(), "OxiOp stack underflow — malformed IR");
    stack.pop().unwrap_or(f64::NAN)
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
    /// Pop one, push tan.
    Tan,
    /// Pop one, push sinh.
    Sinh,
    /// Pop one, push cosh.
    Cosh,
    /// Pop one, push tanh.
    Tanh,
    /// Pop one, push arcsin (asin).
    Arcsin,
    /// Pop one, push arccos (acos).
    Arccos,
    /// Pop one, push arctan (atan).
    Arctan,
    /// Pop one, push arcsinh (asinh).
    Arcsinh,
    /// Pop one, push arccosh (acosh).
    Arccosh,
    /// Pop one, push arctanh (atanh).
    Arctanh,
    /// Pop one, push erf.
    Erf,
    /// Pop one, push lgamma.
    LGamma,
    /// Pop one, push digamma.
    Digamma,
    /// Pop one, push trigamma.
    Trigamma,
    /// Pop one, push Ei.
    Ei,
    /// Pop one, push Si.
    Si,
    /// Pop one, push Ci.
    Ci,
    /// Peek the top of the stack and copy it into slot `k` (does NOT pop).
    Store(usize),
    /// Push the value cached in slot `k` onto the stack.
    Load(usize),
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

    pub(super) fn collect_ops(&self, ops: &mut Vec<OxiOp>) {
        match self {
            Self::Const(c) => ops.push(OxiOp::Const(*c)),
            Self::NamedConst(nc) => ops.push(OxiOp::Const(nc.value())),
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
            Self::Tan(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Tan);
            }
            Self::Sinh(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Sinh);
            }
            Self::Cosh(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Cosh);
            }
            Self::Tanh(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Tanh);
            }
            Self::Arcsin(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Arcsin);
            }
            Self::Arccos(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Arccos);
            }
            Self::Arctan(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Arctan);
            }
            Self::Arcsinh(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Arcsinh);
            }
            Self::Arccosh(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Arccosh);
            }
            Self::Arctanh(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Arctanh);
            }
            Self::Erf(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Erf);
            }
            Self::LGamma(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::LGamma);
            }
            Self::Digamma(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Digamma);
            }
            Self::Trigamma(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Trigamma);
            }
            Self::Ei(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Ei);
            }
            Self::Si(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Si);
            }
            Self::Ci(a) => {
                a.collect_ops(ops);
                ops.push(OxiOp::Ci);
            }
        }
    }

    /// Evaluate a flat instruction list over scalar variable values.
    ///
    /// Runs a stack machine: push leaves, pop operands for each operator.
    /// Returns `f64::NAN` for stack underflow (malformed instruction sequence).
    /// In debug builds, stack underflow (malformed IR) additionally triggers a
    /// `debug_assert!` panic; release builds preserve the silent fallback.
    pub fn eval_ops(ops: &[OxiOp], vars: &[f64]) -> f64 {
        let mut stack: Vec<f64> = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                OxiOp::Const(c) => stack.push(*c),
                OxiOp::Var(i) => {
                    stack.push(vars.get(*i).copied().unwrap_or(f64::NAN));
                }
                OxiOp::Add => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a + b);
                }
                OxiOp::Sub => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a - b);
                }
                OxiOp::Mul => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a * b);
                }
                OxiOp::Div => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a / b);
                }
                OxiOp::Neg => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(-a);
                }
                OxiOp::Exp => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.exp());
                }
                OxiOp::Ln => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.ln());
                }
                OxiOp::Sin => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.sin());
                }
                OxiOp::Cos => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.cos());
                }
                OxiOp::Pow => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.powf(b));
                }
                OxiOp::Tan => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.tan());
                }
                OxiOp::Sinh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.sinh());
                }
                OxiOp::Cosh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.cosh());
                }
                OxiOp::Tanh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.tanh());
                }
                OxiOp::Arcsin => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.asin());
                }
                OxiOp::Arccos => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.acos());
                }
                OxiOp::Arctan => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.atan());
                }
                OxiOp::Arcsinh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.asinh());
                }
                OxiOp::Arccosh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.acosh());
                }
                OxiOp::Arctanh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.atanh());
                }
                OxiOp::Erf => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::erf(a));
                }
                OxiOp::LGamma => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::lgamma(a));
                }
                OxiOp::Digamma => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::digamma(a));
                }
                OxiOp::Trigamma => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::trigamma(a));
                }
                OxiOp::Ei => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::ei(a));
                }
                OxiOp::Si => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::si(a));
                }
                OxiOp::Ci => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::ci(a));
                }
                // Store/Load are only emitted by to_oxiblas_ops_shared.
                // eval_ops does not use slots; encountering them here indicates
                // a caller is mixing shared-IR with the non-slot evaluator.
                // Preserve NaN-propagation semantics in release builds.
                OxiOp::Store(_) => {
                    debug_assert!(
                        false,
                        "OxiOp::Store in eval_ops — use eval_ops_shared instead"
                    );
                }
                OxiOp::Load(_) => {
                    debug_assert!(
                        false,
                        "OxiOp::Load in eval_ops — use eval_ops_shared instead"
                    );
                    stack.push(f64::NAN);
                }
            }
        }
        pop_or_nan(&mut stack)
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

    /// Two-pass sharing-aware codegen. Returns `(ops, n_slots)`.
    ///
    /// Pass 1: census — count parent-reference count per node (by Arc pointer).
    /// Nodes referenced ≥ 2 times earn a slot. Assign dense slot indices.
    ///
    /// Pass 2: emit — recursive post-order emit. If a node is shared and already
    /// emitted, push `Load(slot)` only. If shared and first-emission, emit its
    /// subtree normally then append `Store(slot)`.
    ///
    /// **Strict-generalisation invariant**: for a pure tree (no sharing), the output
    /// is byte-identical to `to_oxiblas_ops()` — zero `Store`, zero `Load`.
    pub fn to_oxiblas_ops_shared(&self) -> (Vec<OxiOp>, usize) {
        let root = Arc::new(self.clone());
        // Pass 1: count references per raw pointer.
        let mut refcount: HashMap<*const LoweredOp, u32> = HashMap::new();
        census(&root, &mut refcount);
        // Assign slot indices to nodes with refcount >= 2.
        let mut slot_for: HashMap<*const LoweredOp, usize> = HashMap::new();
        let mut next_slot = 0usize;
        for (&ptr, &count) in &refcount {
            if count >= 2 {
                slot_for.insert(ptr, next_slot);
                next_slot += 1;
            }
        }
        let n_slots = next_slot;
        // Pass 2: emit ops.
        let mut ops = Vec::new();
        let mut emitted: HashMap<*const LoweredOp, usize> = HashMap::new();
        emit_shared(&root, &slot_for, &mut emitted, &mut ops);
        (ops, n_slots)
    }

    /// Stack-machine evaluator with a slot register file (for shared nodes).
    ///
    /// This is the companion to `to_oxiblas_ops_shared`. `n_slots` must match
    /// what `to_oxiblas_ops_shared` returned.
    pub fn eval_ops_shared(ops: &[OxiOp], vars: &[f64], n_slots: usize) -> f64 {
        let mut stack: Vec<f64> = Vec::with_capacity(ops.len());
        let mut slots: Vec<f64> = vec![f64::NAN; n_slots];
        for op in ops {
            match op {
                OxiOp::Store(k) => {
                    debug_assert!(
                        !stack.is_empty(),
                        "OxiOp::Store({k}) stack underflow — malformed IR"
                    );
                    let v = stack.last().copied().unwrap_or(f64::NAN);
                    if let Some(slot) = slots.get_mut(*k) {
                        *slot = v;
                    } else {
                        debug_assert!(
                            false,
                            "OxiOp::Store({k}) slot index out of range (n_slots={n_slots})"
                        );
                    }
                }
                OxiOp::Load(k) => {
                    let v = slots.get(*k).copied().unwrap_or(f64::NAN);
                    debug_assert!(
                        *k < n_slots,
                        "OxiOp::Load({k}) slot index out of range (n_slots={n_slots})"
                    );
                    stack.push(v);
                }
                OxiOp::Const(c) => stack.push(*c),
                OxiOp::Var(i) => {
                    stack.push(vars.get(*i).copied().unwrap_or(f64::NAN));
                }
                OxiOp::Add => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a + b);
                }
                OxiOp::Sub => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a - b);
                }
                OxiOp::Mul => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a * b);
                }
                OxiOp::Div => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a / b);
                }
                OxiOp::Neg => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(-a);
                }
                OxiOp::Exp => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.exp());
                }
                OxiOp::Ln => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.ln());
                }
                OxiOp::Sin => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.sin());
                }
                OxiOp::Cos => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.cos());
                }
                OxiOp::Pow => {
                    let b = pop_or_nan(&mut stack);
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.powf(b));
                }
                OxiOp::Tan => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.tan());
                }
                OxiOp::Sinh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.sinh());
                }
                OxiOp::Cosh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.cosh());
                }
                OxiOp::Tanh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.tanh());
                }
                OxiOp::Arcsin => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.asin());
                }
                OxiOp::Arccos => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.acos());
                }
                OxiOp::Arctan => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.atan());
                }
                OxiOp::Arcsinh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.asinh());
                }
                OxiOp::Arccosh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.acosh());
                }
                OxiOp::Arctanh => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(a.atanh());
                }
                OxiOp::Erf => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::erf(a));
                }
                OxiOp::LGamma => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::lgamma(a));
                }
                OxiOp::Digamma => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::digamma(a));
                }
                OxiOp::Trigamma => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::trigamma(a));
                }
                OxiOp::Ei => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::ei(a));
                }
                OxiOp::Si => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::si(a));
                }
                OxiOp::Ci => {
                    let a = pop_or_nan(&mut stack);
                    stack.push(crate::special::ci(a));
                }
            }
        }
        pop_or_nan(&mut stack)
    }
}

/// Count Arc pointer references for each node in the DAG.
///
/// Stops recursing into a subtree once its count exceeds 1 — this keeps
/// traversal O(|DAG|) rather than O(|tree|) even on already-shared input.
fn census(node: &Arc<LoweredOp>, counts: &mut HashMap<*const LoweredOp, u32>) {
    let ptr = Arc::as_ptr(node);
    let c = counts.entry(ptr).or_insert(0);
    *c += 1;
    if *c > 1 {
        // Already visited children on the first traversal — don't recurse again.
        return;
    }
    match node.as_ref() {
        LoweredOp::Const(_) | LoweredOp::Var(_) | LoweredOp::NamedConst(_) => {}
        LoweredOp::Neg(a)
        | LoweredOp::Exp(a)
        | LoweredOp::Ln(a)
        | LoweredOp::Sin(a)
        | LoweredOp::Cos(a)
        | LoweredOp::Tan(a)
        | LoweredOp::Sinh(a)
        | LoweredOp::Cosh(a)
        | LoweredOp::Tanh(a)
        | LoweredOp::Arcsin(a)
        | LoweredOp::Arccos(a)
        | LoweredOp::Arctan(a)
        | LoweredOp::Arcsinh(a)
        | LoweredOp::Arccosh(a)
        | LoweredOp::Arctanh(a)
        | LoweredOp::Erf(a)
        | LoweredOp::LGamma(a)
        | LoweredOp::Digamma(a)
        | LoweredOp::Trigamma(a)
        | LoweredOp::Ei(a)
        | LoweredOp::Si(a)
        | LoweredOp::Ci(a) => {
            census(a, counts);
        }
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            census(a, counts);
            census(b, counts);
        }
    }
}

/// Emit post-order instructions for `node`, reusing slots for shared subtrees.
///
/// If `node` is shared (in `slot_for`) and already emitted, push `Load(slot)`
/// only. On first emission of a shared node, emit children recursively then
/// the node's own op, then `Store(slot)` (peek — does NOT pop).
fn emit_shared(
    node: &Arc<LoweredOp>,
    slot_for: &HashMap<*const LoweredOp, usize>,
    emitted: &mut HashMap<*const LoweredOp, usize>,
    ops: &mut Vec<OxiOp>,
) {
    let ptr = Arc::as_ptr(node);
    // If this is a shared node that was already emitted, just Load from its slot.
    if let Some(&slot) = emitted.get(&ptr) {
        ops.push(OxiOp::Load(slot));
        return;
    }
    // Emit children first (post-order).
    match node.as_ref() {
        LoweredOp::Const(_) | LoweredOp::Var(_) | LoweredOp::NamedConst(_) => {}
        LoweredOp::Neg(a)
        | LoweredOp::Exp(a)
        | LoweredOp::Ln(a)
        | LoweredOp::Sin(a)
        | LoweredOp::Cos(a)
        | LoweredOp::Tan(a)
        | LoweredOp::Sinh(a)
        | LoweredOp::Cosh(a)
        | LoweredOp::Tanh(a)
        | LoweredOp::Arcsin(a)
        | LoweredOp::Arccos(a)
        | LoweredOp::Arctan(a)
        | LoweredOp::Arcsinh(a)
        | LoweredOp::Arccosh(a)
        | LoweredOp::Arctanh(a)
        | LoweredOp::Erf(a)
        | LoweredOp::LGamma(a)
        | LoweredOp::Digamma(a)
        | LoweredOp::Trigamma(a)
        | LoweredOp::Ei(a)
        | LoweredOp::Si(a)
        | LoweredOp::Ci(a) => {
            emit_shared(a, slot_for, emitted, ops);
        }
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            emit_shared(a, slot_for, emitted, ops);
            emit_shared(b, slot_for, emitted, ops);
        }
    }
    // Emit this node's own op (same as collect_ops).
    match node.as_ref() {
        LoweredOp::Const(c) => ops.push(OxiOp::Const(*c)),
        LoweredOp::NamedConst(nc) => ops.push(OxiOp::Const(nc.value())),
        LoweredOp::Var(i) => ops.push(OxiOp::Var(*i)),
        LoweredOp::Add(_, _) => ops.push(OxiOp::Add),
        LoweredOp::Sub(_, _) => ops.push(OxiOp::Sub),
        LoweredOp::Mul(_, _) => ops.push(OxiOp::Mul),
        LoweredOp::Div(_, _) => ops.push(OxiOp::Div),
        LoweredOp::Exp(_) => ops.push(OxiOp::Exp),
        LoweredOp::Ln(_) => ops.push(OxiOp::Ln),
        LoweredOp::Sin(_) => ops.push(OxiOp::Sin),
        LoweredOp::Cos(_) => ops.push(OxiOp::Cos),
        LoweredOp::Pow(_, _) => ops.push(OxiOp::Pow),
        LoweredOp::Neg(_) => ops.push(OxiOp::Neg),
        LoweredOp::Tan(_) => ops.push(OxiOp::Tan),
        LoweredOp::Sinh(_) => ops.push(OxiOp::Sinh),
        LoweredOp::Cosh(_) => ops.push(OxiOp::Cosh),
        LoweredOp::Tanh(_) => ops.push(OxiOp::Tanh),
        LoweredOp::Arcsin(_) => ops.push(OxiOp::Arcsin),
        LoweredOp::Arccos(_) => ops.push(OxiOp::Arccos),
        LoweredOp::Arctan(_) => ops.push(OxiOp::Arctan),
        LoweredOp::Arcsinh(_) => ops.push(OxiOp::Arcsinh),
        LoweredOp::Arccosh(_) => ops.push(OxiOp::Arccosh),
        LoweredOp::Arctanh(_) => ops.push(OxiOp::Arctanh),
        LoweredOp::Erf(_) => ops.push(OxiOp::Erf),
        LoweredOp::LGamma(_) => ops.push(OxiOp::LGamma),
        LoweredOp::Digamma(_) => ops.push(OxiOp::Digamma),
        LoweredOp::Trigamma(_) => ops.push(OxiOp::Trigamma),
        LoweredOp::Ei(_) => ops.push(OxiOp::Ei),
        LoweredOp::Si(_) => ops.push(OxiOp::Si),
        LoweredOp::Ci(_) => ops.push(OxiOp::Ci),
    }
    // If this node is shared, peek and store.
    if let Some(&slot) = slot_for.get(&ptr) {
        ops.push(OxiOp::Store(slot));
        emitted.insert(ptr, slot);
    }
}
