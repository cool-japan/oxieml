//! Algebraic simplification for the lowered IR.
//!
//! Applies constant folding, algebraic identities, and polynomial canonicalization
//! to [`LoweredOp`] trees. Also provides canonical pattern recognition
//! (e.g. `sin(x)/cos(x) → tan(x)`).

use crate::lower::LoweredOp;
use crate::poly::MultiPoly;
use std::sync::Arc;

/// Compute a u64 structural hash of a `LoweredOp` node using `DefaultHasher`.
///
/// Used internally for structural-equality checks in the simplifier's
/// canonical pattern recognisers (e.g. `sin(x)/cos(x) → tan(x)`).
pub(crate) fn ops_struct_hash(op: &LoweredOp) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    op.structural_hash(&mut h);
    h.finish()
}

/// Count the number of AST nodes in a `LoweredOp` tree.
pub(crate) fn node_count(op: &LoweredOp) -> usize {
    match op {
        LoweredOp::Const(_) | LoweredOp::Var(_) | LoweredOp::NamedConst(_) => 1,
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
        | LoweredOp::Ci(a) => 1 + node_count(a),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => 1 + node_count(a) + node_count(b),
    }
}

/// Return the number of distinct variable indices used in `op` (= max Var index + 1).
fn count_vars(op: &LoweredOp) -> usize {
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) => 0,
        LoweredOp::Var(i) => i + 1,
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
        | LoweredOp::Ci(a) => count_vars(a),
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => count_vars(a).max(count_vars(b)),
    }
}

/// Return `true` if the outermost node is polynomial-eligible.
fn is_poly_eligible_node(op: &LoweredOp) -> bool {
    match op {
        LoweredOp::Const(_)
        | LoweredOp::NamedConst(_)
        | LoweredOp::Var(_)
        | LoweredOp::Add(..)
        | LoweredOp::Sub(..)
        | LoweredOp::Mul(..)
        | LoweredOp::Neg(..) => true,
        LoweredOp::Pow(_, exp) => {
            if let LoweredOp::Const(e) = exp.as_ref() {
                *e >= 0.0 && e.fract() == 0.0 && *e <= 100.0
            } else {
                false
            }
        }
        _ => false,
    }
}

fn extract_atoms(
    op: &LoweredOp,
    base_n_vars: usize,
    atoms: &mut Vec<(u64, LoweredOp)>,
) -> LoweredOp {
    if !is_poly_eligible_node(op) {
        let h = ops_struct_hash(op);
        if let Some(idx) = atoms.iter().position(|(ah, _)| *ah == h) {
            return LoweredOp::Var(base_n_vars + idx);
        }
        let idx = atoms.len();
        atoms.push((h, op.clone()));
        return LoweredOp::Var(base_n_vars + idx);
    }
    match op {
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => op.clone(),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(extract_atoms(a, base_n_vars, atoms)),
            Arc::new(extract_atoms(b, base_n_vars, atoms)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(extract_atoms(a, base_n_vars, atoms)),
            Arc::new(extract_atoms(b, base_n_vars, atoms)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(extract_atoms(a, base_n_vars, atoms)),
            Arc::new(extract_atoms(b, base_n_vars, atoms)),
        ),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(extract_atoms(a, base_n_vars, atoms))),
        LoweredOp::Pow(base, exp) => LoweredOp::Pow(
            Arc::new(extract_atoms(base, base_n_vars, atoms)),
            Arc::clone(exp),
        ),
        _ => op.clone(),
    }
}

fn substitute_atoms(template: &LoweredOp, base: usize, atoms: &[(u64, LoweredOp)]) -> LoweredOp {
    match template {
        LoweredOp::Var(i) if *i >= base => {
            let idx = i - base;
            if idx < atoms.len() {
                atoms[idx].1.clone()
            } else {
                template.clone()
            }
        }
        LoweredOp::Const(_) | LoweredOp::NamedConst(_) | LoweredOp::Var(_) => template.clone(),
        LoweredOp::Add(a, b) => LoweredOp::Add(
            Arc::new(substitute_atoms(a, base, atoms)),
            Arc::new(substitute_atoms(b, base, atoms)),
        ),
        LoweredOp::Sub(a, b) => LoweredOp::Sub(
            Arc::new(substitute_atoms(a, base, atoms)),
            Arc::new(substitute_atoms(b, base, atoms)),
        ),
        LoweredOp::Mul(a, b) => LoweredOp::Mul(
            Arc::new(substitute_atoms(a, base, atoms)),
            Arc::new(substitute_atoms(b, base, atoms)),
        ),
        LoweredOp::Div(a, b) => LoweredOp::Div(
            Arc::new(substitute_atoms(a, base, atoms)),
            Arc::new(substitute_atoms(b, base, atoms)),
        ),
        LoweredOp::Neg(a) => LoweredOp::Neg(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Pow(a, b) => LoweredOp::Pow(
            Arc::new(substitute_atoms(a, base, atoms)),
            Arc::new(substitute_atoms(b, base, atoms)),
        ),
        LoweredOp::Exp(a) => LoweredOp::Exp(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Ln(a) => LoweredOp::Ln(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Sin(a) => LoweredOp::Sin(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Cos(a) => LoweredOp::Cos(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Tan(a) => LoweredOp::Tan(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Sinh(a) => LoweredOp::Sinh(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Cosh(a) => LoweredOp::Cosh(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Tanh(a) => LoweredOp::Tanh(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Arcsin(a) => LoweredOp::Arcsin(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Arccos(a) => LoweredOp::Arccos(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Arctan(a) => LoweredOp::Arctan(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Arcsinh(a) => LoweredOp::Arcsinh(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Arccosh(a) => LoweredOp::Arccosh(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Arctanh(a) => LoweredOp::Arctanh(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Erf(a) => LoweredOp::Erf(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::LGamma(a) => LoweredOp::LGamma(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Digamma(a) => LoweredOp::Digamma(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Trigamma(a) => LoweredOp::Trigamma(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Ei(a) => LoweredOp::Ei(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Si(a) => LoweredOp::Si(Arc::new(substitute_atoms(a, base, atoms))),
        LoweredOp::Ci(a) => LoweredOp::Ci(Arc::new(substitute_atoms(a, base, atoms))),
    }
}

const PROBE_POINTS: [f64; 5] = [0.7, 1.3, 2.1, -0.5, 0.3];
const MAX_POLY_TERMS: usize = 50;

pub(crate) fn canonicalize_poly(op: &LoweredOp) -> LoweredOp {
    let real_var_count = count_vars(op);
    let mut atoms: Vec<(u64, LoweredOp)> = Vec::new();
    let rewritten = extract_atoms(op, real_var_count, &mut atoms);
    let n_vars = real_var_count + atoms.len();

    let mp = match MultiPoly::from_lowered(&rewritten, n_vars) {
        Ok(p) => p,
        Err(_) => return op.clone(),
    };

    if mp.terms.len() > MAX_POLY_TERMS {
        return op.clone();
    }

    let canonical_template = mp.to_lowered();
    let canonical = substitute_atoms(&canonical_template, real_var_count, &atoms);

    if node_count(&canonical) > node_count(op) {
        return op.clone();
    }

    let n_eval = real_var_count.max(1);
    for &probe in &PROBE_POINTS {
        let vars: Vec<f64> = vec![probe; n_eval];
        let orig_val = op.eval(&vars);
        let new_val = canonical.eval(&vars);
        if !orig_val.is_finite() || !new_val.is_finite() {
            continue;
        }
        let rel_tol = 1e-9 * (1.0 + orig_val.abs());
        if (orig_val - new_val).abs() > rel_tol {
            return op.clone();
        }
    }

    canonical
}

impl LoweredOp {
    /// Simplify the lowered operation tree using algebraic identities,
    /// constant folding, and polynomial canonicalization.
    pub fn simplify(&self) -> Self {
        let inner = self.simplify_inner();
        let canonical = canonicalize_poly(&inner);
        canonical.simplify_inner()
    }

    fn simplify_inner(&self) -> Self {
        match self {
            Self::Add(a, b) => {
                let a_s = a.simplify_inner();
                let b_s = b.simplify_inner();
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return b_s;
                    }
                }
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c + b_c);
                }
                if let Self::Neg(inner) = &b_s {
                    return Self::Sub(Arc::new(a_s), Arc::clone(inner));
                }
                Self::Add(Arc::new(a_s), Arc::new(b_s))
            }
            Self::Sub(a, b) => {
                let a_s = a.simplify_inner();
                let b_s = b.simplify_inner();
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return Self::Neg(Arc::new(b_s));
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c - b_c);
                }
                if let Self::Neg(inner) = &b_s {
                    return Self::Add(Arc::new(a_s), Arc::clone(inner));
                }
                Self::Sub(Arc::new(a_s), Arc::new(b_s))
            }
            Self::Mul(a, b) => {
                let a_s = a.simplify_inner();
                let b_s = b.simplify_inner();
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
                if let Self::Const(c) = &a_s {
                    if (*c + 1.0).abs() < 1e-15 {
                        return Self::Neg(Arc::new(b_s));
                    }
                }
                if let Self::Const(c) = &b_s {
                    if (*c + 1.0).abs() < 1e-15 {
                        return Self::Neg(Arc::new(a_s));
                    }
                }
                Self::Mul(Arc::new(a_s), Arc::new(b_s))
            }
            Self::Div(a, b) => {
                let a_s = a.simplify_inner();
                let b_s = b.simplify_inner();
                if let Self::Const(c) = &b_s {
                    if (*c - 1.0).abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let (Self::Sin(sa), Self::Cos(ca)) = (&a_s, &b_s) {
                    if ops_struct_hash(sa) == ops_struct_hash(ca) {
                        return Self::Tan(Arc::clone(sa));
                    }
                }
                if let (Self::Sinh(sa), Self::Cosh(ca)) = (&a_s, &b_s) {
                    if ops_struct_hash(sa) == ops_struct_hash(ca) {
                        return Self::Tanh(Arc::clone(sa));
                    }
                }
                if let Self::Const(d) = &b_s {
                    if (*d - 2.0).abs() < 1e-15 {
                        if let Self::Sub(sub_a, sub_b) = &a_s {
                            if let (Self::Exp(ea), Self::Exp(eb)) = (sub_a.as_ref(), sub_b.as_ref())
                            {
                                if let Self::Neg(neg_inner) = eb.as_ref() {
                                    if ops_struct_hash(ea) == ops_struct_hash(neg_inner) {
                                        return Self::Sinh(Arc::clone(ea));
                                    }
                                }
                            }
                        }
                    }
                }
                if let Self::Const(d) = &b_s {
                    if (*d - 2.0).abs() < 1e-15 {
                        if let Self::Add(add_a, add_b) = &a_s {
                            if let (Self::Exp(ea), Self::Exp(eb)) = (add_a.as_ref(), add_b.as_ref())
                            {
                                if let Self::Neg(neg_inner) = eb.as_ref() {
                                    if ops_struct_hash(ea) == ops_struct_hash(neg_inner) {
                                        return Self::Cosh(Arc::clone(ea));
                                    }
                                }
                            }
                        }
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c / b_c);
                }
                Self::Div(Arc::new(a_s), Arc::new(b_s))
            }
            Self::Exp(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    if c.abs() < 1e-15 {
                        return Self::Const(1.0);
                    }
                    return Self::Const(c.exp());
                }
                if let Self::Ln(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Exp(Arc::new(a_s))
            }
            Self::Ln(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    if (*c - 1.0).abs() < 1e-15 {
                        return Self::Const(0.0);
                    }
                    return Self::Const(c.ln());
                }
                if let Self::Exp(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Ln(Arc::new(a_s))
            }
            Self::Neg(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(-c);
                }
                if let Self::Neg(inner) = &a_s {
                    return (**inner).clone();
                }
                if let Self::Sub(lhs, rhs) = &a_s {
                    return Self::Sub(Arc::clone(rhs), Arc::clone(lhs));
                }
                Self::Neg(Arc::new(a_s))
            }
            Self::Pow(a, b) => {
                let a_s = a.simplify_inner();
                let b_s = b.simplify_inner();
                if let Self::Const(c) = &b_s {
                    if c.abs() < 1e-15 {
                        return Self::Const(1.0);
                    }
                    if (*c - 1.0).abs() < 1e-15 {
                        return a_s;
                    }
                }
                if let (Self::Const(a_c), Self::Const(b_c)) = (&a_s, &b_s) {
                    return Self::Const(a_c.powf(*b_c));
                }
                Self::Pow(Arc::new(a_s), Arc::new(b_s))
            }
            Self::Sin(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.sin());
                }
                if let Self::Arcsin(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Sin(Arc::new(a_s))
            }
            Self::Cos(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.cos());
                }
                if let Self::Arccos(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Cos(Arc::new(a_s))
            }
            Self::Tan(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.tan());
                }
                if let Self::Arctan(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Tan(Arc::new(a_s))
            }
            Self::Sinh(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.sinh());
                }
                if let Self::Arcsinh(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Sinh(Arc::new(a_s))
            }
            Self::Cosh(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.cosh());
                }
                if let Self::Arccosh(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Cosh(Arc::new(a_s))
            }
            Self::Tanh(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.tanh());
                }
                if let Self::Arctanh(inner) = &a_s {
                    return (**inner).clone();
                }
                Self::Tanh(Arc::new(a_s))
            }
            Self::Arcsin(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.asin());
                }
                Self::Arcsin(Arc::new(a_s))
            }
            Self::Arccos(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.acos());
                }
                Self::Arccos(Arc::new(a_s))
            }
            Self::Arctan(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.atan());
                }
                Self::Arctan(Arc::new(a_s))
            }
            Self::Arcsinh(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.asinh());
                }
                Self::Arcsinh(Arc::new(a_s))
            }
            Self::Arccosh(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.acosh());
                }
                Self::Arccosh(Arc::new(a_s))
            }
            Self::Arctanh(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(c) = &a_s {
                    return Self::Const(c.atanh());
                }
                Self::Arctanh(Arc::new(a_s))
            }
            Self::Erf(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::erf(*v));
                }
                Self::Erf(Arc::new(a_s))
            }
            Self::LGamma(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::lgamma(*v));
                }
                Self::LGamma(Arc::new(a_s))
            }
            Self::Digamma(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::digamma(*v));
                }
                Self::Digamma(Arc::new(a_s))
            }
            Self::Trigamma(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::trigamma(*v));
                }
                Self::Trigamma(Arc::new(a_s))
            }
            Self::Ei(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::ei(*v));
                }
                Self::Ei(Arc::new(a_s))
            }
            Self::Si(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::si(*v));
                }
                Self::Si(Arc::new(a_s))
            }
            Self::Ci(a) => {
                let a_s = a.simplify_inner();
                if let Self::Const(v) = &a_s {
                    return Self::Const(crate::special::ci(*v));
                }
                Self::Ci(Arc::new(a_s))
            }
            Self::Const(_) | Self::Var(_) | Self::NamedConst(_) => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn nc(op: &LoweredOp) -> usize {
        node_count(op)
    }

    #[test]
    fn test_like_terms_polynomial() {
        let x = LoweredOp::Var(0);
        let two = LoweredOp::Const(2.0);
        let two_x = LoweredOp::Mul(Arc::new(two), Arc::new(x.clone()));
        let expr = LoweredOp::Add(Arc::new(x), Arc::new(two_x));
        let simplified = expr.simplify();
        assert!(
            (simplified.eval(&[1.0]) - 3.0).abs() < 1e-10,
            "x+2x should evaluate to 3 at x=1, got {}",
            simplified.eval(&[1.0])
        );
        assert!(
            nc(&simplified) <= nc(&expr),
            "simplified should not be larger: {} > {}",
            nc(&simplified),
            nc(&expr)
        );
    }

    #[test]
    fn test_like_terms_with_transcendental() {
        let x = LoweredOp::Var(0);
        let sin_x = LoweredOp::Sin(Arc::new(x.clone()));
        let two = LoweredOp::Const(2.0);
        let two_sin_x = LoweredOp::Mul(Arc::new(two), Arc::new(sin_x.clone()));
        let expr = LoweredOp::Add(Arc::new(sin_x), Arc::new(two_sin_x));
        let simplified = expr.simplify();
        let probe = std::f64::consts::FRAC_PI_4;
        let expected = 3.0 * probe.sin();
        let got = simplified.eval(&[probe]);
        assert!(
            (got - expected).abs() < 1e-9,
            "sin(x)+2*sin(x) should equal 3*sin(x) at π/4, got {got} expected {expected}"
        );
        assert!(
            nc(&simplified) <= nc(&expr),
            "simplified should not be larger"
        );
    }

    #[test]
    fn test_idempotence_polynomial() {
        let x = Arc::new(LoweredOp::Var(0));
        let x2 = LoweredOp::Mul(Arc::clone(&x), Arc::clone(&x));
        let two_x = LoweredOp::Mul(Arc::new(LoweredOp::Const(2.0)), Arc::clone(&x));
        let one = LoweredOp::Const(1.0);
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Add(Arc::new(x2), Arc::new(two_x))),
            Arc::new(one),
        );
        let s1 = expr.simplify();
        let s2 = s1.simplify();
        assert_eq!(
            ops_struct_hash(&s1),
            ops_struct_hash(&s2),
            "simplify must be idempotent: s1={s1} s2={s2}",
        );
    }

    #[test]
    fn test_value_preservation_at_probes() {
        let x = Arc::new(LoweredOp::Var(0));
        let xp1 = LoweredOp::Add(Arc::clone(&x), Arc::new(LoweredOp::Const(1.0)));
        let xm1 = LoweredOp::Sub(Arc::clone(&x), Arc::new(LoweredOp::Const(1.0)));
        let expr = LoweredOp::Mul(Arc::new(xp1), Arc::new(xm1));
        let simplified = expr.simplify();
        for &probe in &PROBE_POINTS {
            let orig = expr.eval(&[probe]);
            let got = simplified.eval(&[probe]);
            assert!(
                (orig - got).abs() < 1e-9 * (1.0 + orig.abs()),
                "value changed at probe {probe}: orig={orig} got={got}"
            );
        }
    }

    #[test]
    fn test_never_worse_constant() {
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Const(1.0)),
            Arc::new(LoweredOp::Const(2.0)),
        );
        let simplified = expr.simplify();
        assert!(
            nc(&simplified) <= nc(&expr),
            "simplified constant should not be larger"
        );
        assert!(
            (simplified.eval(&[0.0]) - 3.0).abs() < 1e-12,
            "1+2 should equal 3"
        );
    }

    #[test]
    fn test_sin_over_cos_becomes_tan() {
        let x = Arc::new(LoweredOp::Var(0));
        let sin_x = LoweredOp::Sin(Arc::clone(&x));
        let cos_x = LoweredOp::Cos(Arc::clone(&x));
        let expr = LoweredOp::Div(Arc::new(sin_x), Arc::new(cos_x));
        let simplified = expr.simplify();
        assert!(
            matches!(&simplified, LoweredOp::Tan(_)),
            "sin(x)/cos(x) should become Tan, got: {simplified}",
        );
    }
}
