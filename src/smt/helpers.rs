use super::constraint::EmlConstraint;
use crate::eval::EvalCtx;

// ----------------------------------------------------------------------------
// Helpers (shared by both solvers)
// ----------------------------------------------------------------------------

pub(super) fn check_constraint(constraint: &EmlConstraint, ctx: &EvalCtx) -> bool {
    match constraint {
        EmlConstraint::EqZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v.abs() < 1e-8),
        EmlConstraint::GtZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v > 0.0),
        EmlConstraint::GeZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v >= -1e-12),
        EmlConstraint::LtZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v < 0.0),
        EmlConstraint::LeZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v <= 1e-12),
        EmlConstraint::NeZero(tree) => tree.eval_real(ctx).is_ok_and(|v| v.abs() > 1e-12),
        EmlConstraint::Not(inner) => !check_constraint(inner, ctx),
        EmlConstraint::And(constraints) => constraints.iter().all(|c| check_constraint(c, ctx)),
        EmlConstraint::Or(constraints) => constraints.iter().any(|c| check_constraint(c, ctx)),
        // Conservative evaluation via sampling at endpoints and midpoint.
        EmlConstraint::ForAll { var, lo, hi, body } => {
            let mid = (lo + hi) / 2.0;
            [*lo, mid, *hi].iter().all(|&sample| {
                let mut vars: Vec<f64> = ctx.as_slice().to_vec();
                if *var < vars.len() {
                    vars[*var] = sample;
                }
                let inner_ctx = EvalCtx::new(&vars);
                check_constraint(body, &inner_ctx)
            })
        }
        EmlConstraint::Exists { var, lo, hi, body } => {
            let mid = (lo + hi) / 2.0;
            [*lo, mid, *hi].iter().any(|&sample| {
                let mut vars: Vec<f64> = ctx.as_slice().to_vec();
                if *var < vars.len() {
                    vars[*var] = sample;
                }
                let inner_ctx = EvalCtx::new(&vars);
                check_constraint(body, &inner_ctx)
            })
        }
    }
}

pub(super) fn evaluate_constraint_residual(constraint: &EmlConstraint, vars: &[f64]) -> f64 {
    let ctx = EvalCtx::new(vars);
    match constraint {
        EmlConstraint::EqZero(tree) => tree.eval_real(&ctx).unwrap_or(f64::INFINITY).abs(),
        EmlConstraint::GtZero(tree) => {
            let v = tree.eval_real(&ctx).unwrap_or(f64::NEG_INFINITY);
            if v > 0.0 { 0.0 } else { -v }
        }
        EmlConstraint::GeZero(tree) => {
            let v = tree.eval_real(&ctx).unwrap_or(f64::NEG_INFINITY);
            if v >= 0.0 { 0.0 } else { -v }
        }
        EmlConstraint::LtZero(tree) => {
            let v = tree.eval_real(&ctx).unwrap_or(f64::INFINITY);
            if v < 0.0 { 0.0 } else { v.abs() }
        }
        EmlConstraint::LeZero(tree) => {
            let v = tree.eval_real(&ctx).unwrap_or(f64::INFINITY);
            if v <= 0.0 { 0.0 } else { v.abs() }
        }
        EmlConstraint::NeZero(tree) => tree.eval_real(&ctx).unwrap_or(0.0).abs(),
        EmlConstraint::Not(inner) => {
            let r = evaluate_constraint_residual(inner, vars);
            if r < 1e-12 { 1.0 } else { 0.0 }
        }
        EmlConstraint::And(constraints) => constraints
            .iter()
            .map(|c| evaluate_constraint_residual(c, vars))
            .map(f64::abs)
            .sum(),
        EmlConstraint::Or(constraints) => constraints
            .iter()
            .map(|c| evaluate_constraint_residual(c, vars))
            .map(f64::abs)
            .fold(f64::INFINITY, f64::min),
        // For quantifiers: sample and aggregate residuals.
        EmlConstraint::ForAll { var, lo, hi, body } => {
            let mid = (lo + hi) / 2.0;
            [*lo, mid, *hi]
                .iter()
                .map(|&sample| {
                    let mut v = vars.to_vec();
                    if *var < v.len() {
                        v[*var] = sample;
                    }
                    evaluate_constraint_residual(body, &v)
                })
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
        }
        EmlConstraint::Exists { var, lo, hi, body } => {
            let mid = (lo + hi) / 2.0;
            [*lo, mid, *hi]
                .iter()
                .map(|&sample| {
                    let mut v = vars.to_vec();
                    if *var < v.len() {
                        v[*var] = sample;
                    }
                    evaluate_constraint_residual(body, &v)
                })
                .map(f64::abs)
                .fold(f64::INFINITY, f64::min)
        }
    }
}

pub(super) fn count_constraint_vars(constraint: &EmlConstraint) -> usize {
    match constraint {
        EmlConstraint::EqZero(tree)
        | EmlConstraint::GtZero(tree)
        | EmlConstraint::GeZero(tree)
        | EmlConstraint::LtZero(tree)
        | EmlConstraint::LeZero(tree)
        | EmlConstraint::NeZero(tree) => tree.num_vars(),
        EmlConstraint::Not(inner) => count_constraint_vars(inner),
        EmlConstraint::And(cs) | EmlConstraint::Or(cs) => {
            cs.iter().map(count_constraint_vars).max().unwrap_or(0)
        }
        EmlConstraint::ForAll { var, body, .. } | EmlConstraint::Exists { var, body, .. } => {
            // The quantified variable `var` is bound; the body may reference it
            // plus any free variables in the context.
            count_constraint_vars(body).max(var + 1)
        }
    }
}

// ----------------------------------------------------------------------------
// Quantifier decision procedures (J1)
// ----------------------------------------------------------------------------

/// Epsilon for near-zero detection in NeZero propagation.
pub(super) const NEZERO_EPS: f64 = 1e-12;

/// Result of a quantifier decision procedure.
#[derive(Debug)]
pub enum QuantResult {
    /// The constraint is universally true over the given box.
    True,
    /// The constraint is existentially satisfied; witness point provided.
    TrueWithWitness(Vec<f64>),
    /// The universal constraint is falsified; the field holds the counterexample.
    FalseWithCounterexample {
        /// A point that falsifies the body under the universally quantified variable.
        counterexample: Vec<f64>,
    },
    /// The procedure could not decide.
    Unknown,
}

/// Decide ∀ var ∈ [lo, hi]. body (conservative, honest).
///
/// Returns `True` if interval propagation of ¬body yields Conflict over the box.
/// Returns `FalseWithCounterexample` if a sample falsifies body.
/// Returns `Unknown` otherwise.
pub(crate) fn decide_forall(
    var: usize,
    lo: f64,
    hi: f64,
    body: &EmlConstraint,
    num_vars: usize,
) -> QuantResult {
    use super::interval::{IntervalDomain, PropResult};

    // Try interval refutation: propagate ¬body over [var ∈ [lo, hi]].
    let negated = negate_constraint(body.clone());
    let mut bounds: Vec<(f64, f64)> = vec![(-1e10, 1e10); num_vars];
    if var < num_vars {
        bounds[var] = (lo, hi);
    }
    let mut domain = IntervalDomain::new(&bounds, num_vars);
    if domain.propagate(&negated) == PropResult::Conflict {
        return QuantResult::True;
    }

    // Sample a few candidate counterexamples.
    let mid = (lo + hi) / 2.0;
    let samples = [lo, hi, mid, lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)];
    for &sample in &samples {
        let mut point = vec![0.0f64; num_vars];
        if var < num_vars {
            point[var] = sample;
        }
        let ctx = EvalCtx::new(&point);
        if !check_constraint(body, &ctx) {
            return QuantResult::FalseWithCounterexample {
                counterexample: point,
            };
        }
    }

    QuantResult::Unknown
}

/// Decide ∃ var ∈ [lo, hi]. body.
///
/// Returns `TrueWithWitness` if a verified witness point is found.
/// Returns `Unknown` otherwise (never returns `False` — sound over-approximation).
pub(crate) fn decide_exists(
    var: usize,
    lo: f64,
    hi: f64,
    body: &EmlConstraint,
    num_vars: usize,
) -> QuantResult {
    let mid = (lo + hi) / 2.0;
    let samples = [mid, lo, hi, lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)];
    for &sample in &samples {
        let mut point = vec![0.0f64; num_vars];
        if var < num_vars {
            point[var] = sample;
        }
        let ctx = EvalCtx::new(&point);
        if check_constraint(body, &ctx) {
            return QuantResult::TrueWithWitness(point);
        }
    }
    QuantResult::Unknown
}

/// NNF-negate a constraint (delegates to the private `negate` in constraint.rs
/// via the public `to_nnf` + `Not` wrapper).
fn negate_constraint(c: EmlConstraint) -> EmlConstraint {
    EmlConstraint::Not(Box::new(c)).to_nnf()
}
