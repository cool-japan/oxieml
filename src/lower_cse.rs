//! Common-subexpression elimination (CSE) for `LoweredOp` trees.
//!
//! Hash-consing pass that converts a tree into a maximally-shared DAG.
//! Every pair of structurally-identical subtrees is collapsed to a single
//! shared `Arc<LoweredOp>` allocation.

use crate::lower::LoweredOp;
use crate::lower_simplify::ops_struct_hash;
use std::collections::HashMap;
use std::sync::Arc;

/// Intern table for hash-consing `LoweredOp` nodes.
///
/// Entirely local to a single `cse()` call — never shared across threads.
#[derive(Default)]
struct CseInterner {
    /// Pointer-identity memo: maps an already-visited node's raw pointer to its
    /// canonical Arc.  Makes traversal O(DAG-size) even on already-shared input.
    visited: HashMap<*const LoweredOp, Arc<LoweredOp>>,
    /// Structural-hash intern table.  `Vec` handles hash collisions; `PartialEq`
    /// guards against false positives (two distinct-value subtrees sharing a hash).
    table: HashMap<u64, Vec<Arc<LoweredOp>>>,
}

impl CseInterner {
    fn intern(&mut self, node: &Arc<LoweredOp>) -> Arc<LoweredOp> {
        // 1. Pointer memo — if we already interned this exact Arc, return its canonical.
        let ptr = Arc::as_ptr(node);
        if let Some(c) = self.visited.get(&ptr) {
            return Arc::clone(c);
        }

        // 2. Post-order: rebuild this node with every child replaced by its canonical Arc.
        let rebuilt = self.rebuild(node);

        // 3. Intern the rebuilt node by structural hash.
        let h = ops_struct_hash(&rebuilt);
        let canonical = {
            let bucket = self.table.entry(h).or_default();
            // Scan for a structurally-equal existing entry (collision guard).
            if let Some(existing) = bucket.iter().find(|e| ***e == rebuilt) {
                Arc::clone(existing)
            } else {
                let fresh = Arc::new(rebuilt);
                bucket.push(Arc::clone(&fresh));
                fresh
            }
        };

        // 4. Record in pointer memo before returning.
        self.visited.insert(ptr, Arc::clone(&canonical));
        canonical
    }

    fn rebuild(&mut self, node: &Arc<LoweredOp>) -> LoweredOp {
        // For leaves, clone directly.  For composite nodes, intern each child Arc.
        match node.as_ref() {
            LoweredOp::Const(_) | LoweredOp::Var(_) | LoweredOp::NamedConst(_) => (**node).clone(),
            LoweredOp::Add(a, b) => LoweredOp::Add(self.intern(a), self.intern(b)),
            LoweredOp::Sub(a, b) => LoweredOp::Sub(self.intern(a), self.intern(b)),
            LoweredOp::Mul(a, b) => LoweredOp::Mul(self.intern(a), self.intern(b)),
            LoweredOp::Div(a, b) => LoweredOp::Div(self.intern(a), self.intern(b)),
            LoweredOp::Pow(a, b) => LoweredOp::Pow(self.intern(a), self.intern(b)),
            LoweredOp::Neg(a) => LoweredOp::Neg(self.intern(a)),
            LoweredOp::Exp(a) => LoweredOp::Exp(self.intern(a)),
            LoweredOp::Ln(a) => LoweredOp::Ln(self.intern(a)),
            LoweredOp::Sin(a) => LoweredOp::Sin(self.intern(a)),
            LoweredOp::Cos(a) => LoweredOp::Cos(self.intern(a)),
            LoweredOp::Tan(a) => LoweredOp::Tan(self.intern(a)),
            LoweredOp::Sinh(a) => LoweredOp::Sinh(self.intern(a)),
            LoweredOp::Cosh(a) => LoweredOp::Cosh(self.intern(a)),
            LoweredOp::Tanh(a) => LoweredOp::Tanh(self.intern(a)),
            LoweredOp::Arcsin(a) => LoweredOp::Arcsin(self.intern(a)),
            LoweredOp::Arccos(a) => LoweredOp::Arccos(self.intern(a)),
            LoweredOp::Arctan(a) => LoweredOp::Arctan(self.intern(a)),
            LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(self.intern(a)),
            LoweredOp::Arccosh(a) => LoweredOp::Arccosh(self.intern(a)),
            LoweredOp::Arctanh(a) => LoweredOp::Arctanh(self.intern(a)),
            LoweredOp::Erf(a) => LoweredOp::Erf(self.intern(a)),
            LoweredOp::LGamma(a) => LoweredOp::LGamma(self.intern(a)),
            LoweredOp::Digamma(a) => LoweredOp::Digamma(self.intern(a)),
            LoweredOp::Trigamma(a) => LoweredOp::Trigamma(self.intern(a)),
            LoweredOp::Ei(a) => LoweredOp::Ei(self.intern(a)),
            LoweredOp::Si(a) => LoweredOp::Si(self.intern(a)),
            LoweredOp::Ci(a) => LoweredOp::Ci(self.intern(a)),
        }
    }
}

impl LoweredOp {
    /// Hash-cons this expression tree into a maximally-shared DAG.
    ///
    /// Every pair of structurally-identical subtrees is collapsed to a single
    /// shared `Arc<LoweredOp>` allocation.  The result is semantically equivalent
    /// to `self` (eval-identical for all inputs).
    ///
    /// Cost: O(|DAG|) traversal (pointer-identity memo prevents re-traversal of
    /// already-shared nodes).
    pub fn cse(&self) -> Arc<LoweredOp> {
        let root = Arc::new(self.clone());
        let mut interner = CseInterner::default();
        interner.intern(&root)
    }
}
