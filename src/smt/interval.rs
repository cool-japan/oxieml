use super::constraint::EmlConstraint;
use crate::tree::EmlNode;

// ----------------------------------------------------------------------------
// Interval arithmetic
// ----------------------------------------------------------------------------

/// Interval `[lo, hi]` for variable bounds during constraint solving.
///
/// Empty when `lo > hi` or when bounds are non-finite. `hull` is the join used
/// to combine bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Lower bound (inclusive).
    pub lo: f64,
    /// Upper bound (inclusive).
    pub hi: f64,
}

impl Interval {
    /// Build a new interval `[lo, hi]`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Build a point interval `[v, v]`.
    pub fn point(v: f64) -> Self {
        Self::new(v, v)
    }

    /// True if the interval is empty (no real points inside).
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi || self.lo.is_nan() || self.hi.is_nan()
    }

    /// Width of the interval (`hi - lo`).
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// True if `x` lies inside the (closed) interval.
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    /// Split the interval at the midpoint into two halves.
    pub fn split(&self) -> (Self, Self) {
        let mid = self.midpoint();
        (Self::new(self.lo, mid), Self::new(mid, self.hi))
    }

    /// Intersect two intervals; result is empty if they are disjoint.
    pub fn intersect(&self, other: &Self) -> Self {
        Self::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    /// Convex hull (bounding interval containing both).
    pub fn hull(&self, other: &Self) -> Self {
        Self::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    /// Forward `exp`: since `exp` is monotone, `exp([lo, hi]) = [exp(lo), exp(hi)]`.
    pub fn exp(&self) -> Self {
        Self::new(self.lo.exp(), self.hi.exp())
    }

    /// Forward `ln`: `ln([lo, hi])` for `lo > 0`, else empty.
    pub fn ln(&self) -> Self {
        if self.lo <= 0.0 || !self.lo.is_finite() || !self.hi.is_finite() {
            // Represent empty as `[+inf, -inf]` so `is_empty()` returns true.
            Self::new(f64::INFINITY, f64::NEG_INFINITY)
        } else {
            Self::new(self.lo.ln(), self.hi.ln())
        }
    }

    /// Inverse of `exp`: given output interval for exp(x), compute x's interval.
    /// Since exp is strictly monotone, this is just `ln([lo, hi])`.
    pub fn exp_inv(&self) -> Self {
        self.ln()
    }

    /// Inverse of `ln`: given output interval for ln(x), compute x's interval.
    /// Since ln is strictly monotone, this is just `exp([lo, hi])`.
    pub fn ln_inv(&self) -> Self {
        self.exp()
    }
}

// ----------------------------------------------------------------------------
// Interval domain and propagation
// ----------------------------------------------------------------------------

/// Result of a constraint propagation step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropResult {
    /// At least one variable interval was tightened.
    Changed,
    /// No change; propagation reached fixpoint.
    Stable,
    /// Constraint is provably unsatisfiable on current domain.
    Conflict,
}

/// Variable interval domain for EML constraint propagation.
#[derive(Clone, Debug)]
pub struct IntervalDomain {
    /// One interval per variable (index = variable index).
    pub vars: Vec<Interval>,
}

impl IntervalDomain {
    /// Build a domain from `(lo, hi)` bounds, padded with `(-10, 10)` for any
    /// missing slots, up to `num_vars` variables in total.
    pub fn new(bounds: &[(f64, f64)], num_vars: usize) -> Self {
        let vars = (0..num_vars)
            .map(|i| {
                if i < bounds.len() {
                    Interval::new(bounds[i].0, bounds[i].1)
                } else {
                    Interval::new(-10.0, 10.0)
                }
            })
            .collect();
        Self { vars }
    }

    /// True if any variable's interval is empty.
    pub fn is_empty(&self) -> bool {
        self.vars.iter().any(Interval::is_empty)
    }

    /// Propagate a constraint until fixpoint or conflict.
    ///
    /// Returns `Changed` if any variable was tightened, `Stable` if fixpoint
    /// reached without any change, `Conflict` if unsatisfiable.
    pub fn propagate(&mut self, c: &EmlConstraint) -> PropResult {
        const MAX_ITERATIONS: usize = 20;
        let mut changed_any = false;
        for _ in 0..MAX_ITERATIONS {
            let result = propagate_once(&mut self.vars, c);
            match result {
                PropResult::Conflict => return PropResult::Conflict,
                PropResult::Changed => {
                    changed_any = true;
                    continue;
                }
                PropResult::Stable => break,
            }
        }
        if changed_any {
            PropResult::Changed
        } else {
            PropResult::Stable
        }
    }
}

/// Forward-evaluate an EML subtree on interval-valued variables.
/// Returns the interval of possible output values.
fn eval_interval(node: &EmlNode, vars: &[Interval]) -> Interval {
    match node {
        EmlNode::One => Interval::new(1.0, 1.0),
        EmlNode::Const(v) => Interval::new(*v, *v),
        EmlNode::Var(i) => vars
            .get(*i)
            .copied()
            .unwrap_or_else(|| Interval::new(-10.0, 10.0)),
        EmlNode::Eml { left, right } => {
            let l = eval_interval(left, vars);
            let r = eval_interval(right, vars);
            if l.is_empty() || r.is_empty() {
                return Interval::new(f64::INFINITY, f64::NEG_INFINITY);
            }
            let exp_l = l.exp();
            let ln_r = r.ln();
            if ln_r.is_empty() {
                return Interval::new(f64::INFINITY, f64::NEG_INFINITY);
            }
            // eml(l, r) = exp(l) − ln(r)
            // min = exp(l.lo) − ln(r.hi)
            // max = exp(l.hi) − ln(r.lo)
            Interval::new(exp_l.lo - ln_r.hi, exp_l.hi - ln_r.lo)
        }
    }
}

/// HC4-revise style backward propagation: given the output constraint for a node,
/// tighten the variable domains.
fn backward_propagate(
    node: &EmlNode,
    vars: &mut [Interval],
    out_constraint: Interval,
) -> PropResult {
    if out_constraint.is_empty() {
        return PropResult::Conflict;
    }
    match node {
        EmlNode::Var(i) => {
            let idx = *i;
            let old = vars
                .get(idx)
                .copied()
                .unwrap_or_else(|| Interval::new(-10.0, 10.0));
            let new_iv = old.intersect(&out_constraint);
            if new_iv.is_empty() {
                return PropResult::Conflict;
            }
            if new_iv.lo > old.lo + 1e-12 || new_iv.hi < old.hi - 1e-12 {
                if idx < vars.len() {
                    vars[idx] = new_iv;
                }
                PropResult::Changed
            } else {
                PropResult::Stable
            }
        }
        EmlNode::One => {
            if !out_constraint.contains(1.0) {
                PropResult::Conflict
            } else {
                PropResult::Stable
            }
        }
        EmlNode::Const(v) => {
            if !out_constraint.contains(*v) {
                PropResult::Conflict
            } else {
                PropResult::Stable
            }
        }
        EmlNode::Eml { left, right } => {
            // out = exp(left) - ln(right)
            let left_iv = eval_interval(left, vars);
            let right_iv = eval_interval(right, vars);

            if left_iv.is_empty() || right_iv.is_empty() {
                return PropResult::Conflict;
            }

            let exp_l = left_iv.exp();
            let ln_r = right_iv.ln();

            if ln_r.is_empty() {
                return PropResult::Conflict;
            }

            // Forward output: [exp_l.lo - ln_r.hi, exp_l.hi - ln_r.lo]
            let forward_out = Interval::new(exp_l.lo - ln_r.hi, exp_l.hi - ln_r.lo);
            let out_c = out_constraint.intersect(&forward_out);
            if out_c.is_empty() {
                return PropResult::Conflict;
            }

            // Back-propagate to exp(left):
            // exp(left) = out + ln(right)
            // → exp(left) ∈ [out_c.lo + ln_r.lo, out_c.hi + ln_r.hi]
            let exp_l_back = Interval::new(out_c.lo + ln_r.lo, out_c.hi + ln_r.hi);
            let exp_l_c = exp_l.intersect(&exp_l_back);
            if exp_l_c.is_empty() {
                return PropResult::Conflict;
            }
            // left = ln(exp_l_c) (monotone inverse)
            let left_c = left_iv.intersect(&exp_l_c.exp_inv());

            // Back-propagate to ln(right):
            // ln(right) = exp(left) - out
            // → ln(right) ∈ [exp_l.lo - out_c.hi, exp_l.hi - out_c.lo]
            let ln_r_back = Interval::new(exp_l.lo - out_c.hi, exp_l.hi - out_c.lo);
            let ln_r_c = ln_r.intersect(&ln_r_back);
            if ln_r_c.is_empty() {
                return PropResult::Conflict;
            }
            // right = exp(ln_r_c) (monotone inverse)
            let right_c = right_iv.intersect(&ln_r_c.ln_inv());

            // Recurse
            let r_left = backward_propagate(left, vars, left_c);
            let r_right = backward_propagate(right, vars, right_c);

            match (r_left, r_right) {
                (PropResult::Conflict, _) | (_, PropResult::Conflict) => PropResult::Conflict,
                (PropResult::Changed, _) | (_, PropResult::Changed) => PropResult::Changed,
                _ => PropResult::Stable,
            }
        }
    }
}

/// Single-pass propagation: walk constraint, evaluate intervals, tighten.
fn propagate_once(vars: &mut [Interval], c: &EmlConstraint) -> PropResult {
    match c {
        EmlConstraint::EqZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.lo > 0.0 || v.hi < 0.0 {
                return PropResult::Conflict;
            }
            // Backward: output must be exactly 0
            backward_propagate(&tree.root, vars, Interval::new(0.0, 0.0))
        }
        EmlConstraint::GtZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.hi <= 0.0 {
                return PropResult::Conflict;
            }
            // Backward: output must be > 0; tighten lower bound
            let eps = f64::EPSILON * 4.0;
            let out_c = Interval::new(v.lo.max(eps), v.hi);
            backward_propagate(&tree.root, vars, out_c)
        }
        EmlConstraint::GeZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.hi < 0.0 {
                return PropResult::Conflict;
            }
            // Backward: output must be >= 0
            let out_c = Interval::new(v.lo.max(0.0), v.hi);
            backward_propagate(&tree.root, vars, out_c)
        }
        EmlConstraint::LtZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.lo >= 0.0 {
                return PropResult::Conflict;
            }
            let eps = f64::EPSILON * 4.0;
            let out_c = Interval::new(v.lo, v.hi.min(-eps));
            if out_c.is_empty() {
                return PropResult::Conflict;
            }
            backward_propagate(&tree.root, vars, out_c)
        }
        EmlConstraint::LeZero(tree) => {
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            if v.lo > 0.0 {
                return PropResult::Conflict;
            }
            let out_c = Interval::new(v.lo, v.hi.min(0.0));
            if out_c.is_empty() {
                return PropResult::Conflict;
            }
            backward_propagate(&tree.root, vars, out_c)
        }
        EmlConstraint::NeZero(tree) => {
            use super::helpers::NEZERO_EPS;
            let v = eval_interval(&tree.root, vars);
            if v.is_empty() {
                return PropResult::Conflict;
            }
            // Point interval at or near zero → Conflict.
            if (v.hi - v.lo) < NEZERO_EPS && v.lo.abs() < NEZERO_EPS {
                return PropResult::Conflict;
            }
            // Try to nudge bare Var(i) lower/upper bound away from zero.
            if let crate::tree::EmlNode::Var(i) = &*tree.root {
                let idx = *i;
                if let Some(iv) = vars.get(idx).copied() {
                    // Lower bound sits at zero, interval extends above — nudge up.
                    if iv.lo == 0.0 && iv.hi > 0.0 {
                        vars[idx] = Interval::new(0.0_f64.next_up(), iv.hi);
                        return PropResult::Changed;
                    }
                    // Upper bound sits at zero, interval extends below — nudge down.
                    if iv.hi == 0.0 && iv.lo < 0.0 {
                        vars[idx] = Interval::new(iv.lo, 0.0_f64.next_down());
                        return PropResult::Changed;
                    }
                }
            }
            // 0 in the strict interior, or non-Var tree — can't split a single interval.
            PropResult::Stable
        }
        EmlConstraint::Not(inner) => {
            let nnf = (**inner).clone().to_nnf();
            propagate_once(vars, &nnf)
        }
        EmlConstraint::And(constraints) => {
            let mut any_changed = false;
            for inner in constraints {
                match propagate_once(vars, inner) {
                    PropResult::Conflict => return PropResult::Conflict,
                    PropResult::Changed => any_changed = true,
                    PropResult::Stable => {}
                }
            }
            if any_changed {
                PropResult::Changed
            } else {
                PropResult::Stable
            }
        }
        EmlConstraint::Or(constraints) => {
            if constraints.is_empty() {
                return PropResult::Conflict;
            }
            // Save original domains so we never widen beyond them.
            let original: Vec<Interval> = vars.to_vec();

            // Propagate each branch independently, collect feasible survivors.
            let mut survivors: Vec<Vec<Interval>> = Vec::with_capacity(constraints.len());
            for inner in constraints {
                let mut branch = original.clone();
                let result = propagate_once(&mut branch, inner);
                if result != PropResult::Conflict {
                    survivors.push(branch);
                }
            }

            match survivors.len() {
                0 => PropResult::Conflict,
                1 => {
                    // Exactly one feasible branch: adopt it.
                    let branch = survivors.remove(0);
                    let mut any_changed = false;
                    for (i, (new_iv, orig_iv)) in branch.iter().zip(original.iter()).enumerate() {
                        if i < vars.len() {
                            // Intersect with original to be sound.
                            let tightened = new_iv.intersect(orig_iv);
                            if !tightened.is_empty()
                                && (tightened.lo > orig_iv.lo + 1e-12
                                    || tightened.hi < orig_iv.hi - 1e-12)
                            {
                                vars[i] = tightened;
                                any_changed = true;
                            }
                        }
                    }
                    if any_changed {
                        PropResult::Changed
                    } else {
                        PropResult::Stable
                    }
                }
                _ => {
                    // Multiple feasible branches: hull ∩ original — NEVER widen.
                    let mut any_changed = false;
                    for i in 0..vars.len() {
                        // Hull of the survivors for this variable.
                        let hull = survivors
                            .iter()
                            .filter_map(|b| b.get(i).copied())
                            .reduce(|acc, iv| acc.hull(&iv));
                        if let Some(hull_iv) = hull {
                            // Intersect hull with original to never widen.
                            let tightened = hull_iv.intersect(&original[i]);
                            if !tightened.is_empty()
                                && (tightened.lo > original[i].lo + 1e-12
                                    || tightened.hi < original[i].hi - 1e-12)
                            {
                                vars[i] = tightened;
                                any_changed = true;
                            }
                        }
                    }
                    if any_changed {
                        PropResult::Changed
                    } else {
                        PropResult::Stable
                    }
                }
            }
        }
        // Quantifiers: no outer tightening in v1 — sound conservative default.
        EmlConstraint::ForAll { .. } => PropResult::Stable,
        EmlConstraint::Exists { .. } => PropResult::Stable,
    }
}
