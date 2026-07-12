//! MaxSMT over EML constraints: maximize the total weight of satisfied *soft*
//! constraints subject to all *hard* constraints.
//!
//! # Two engines, two honest guarantees
//!
//! | feature   | engine                                                | guarantee                     |
//! |-----------|-------------------------------------------------------|-------------------------------|
//! | `smt-opt` | core-guided implicit hitting set + exact inner search  | [`MaxSmtOptimality::Optimal`] |
//! | `smt`     | weight-ordered greedy                                 | [`MaxSmtOptimality::Maximal`] |
//!
//! The greedy fallback is a *real* greedy algorithm, not a fake optimum: it
//! returns a **maximal** selection (no further soft constraint can be added
//! without losing feasibility) and says so in the returned [`MaxSmtOptimality`].
//! It can and does miss the maximum — with soft constraints of weights `3, 2, 2`
//! where the `3` is mutually exclusive with both `2`s, greedy takes the `3`
//! (total 3) while the optimum is `2 + 2 = 4`. Build with `--features smt-opt` if
//! you need the optimum. That is exactly the `s1_maxsmt` test instance, and the
//! two feature configurations are asserted to give the two different answers.
//!
//! # What OxiZ actually provides, and what it does not
//!
//! Two findings about `oxiz` 0.2.3 shaped this module. Both are stated plainly
//! rather than papered over, because both would otherwise have turned into silent
//! fabrications:
//!
//! * **`oxiz::opt::MaxSmtSolver` is a stub.** `check_hard_satisfiable` and
//!   `solve_core_guided` return `MaxSmtResult::Unknown` without ever consulting an
//!   SMT solver. Building on it would yield a plausible-looking API that optimizes
//!   nothing, so it is not used.
//! * **`oxiz::opt::Rc2Solver` is real, but its weighted optimum is unreliable.**
//!   It is a genuine SAT-backed core-guided MaxSAT solver, yet on *overlapping
//!   weighted cores* it reports the wrong optimum — see [`best_selection`] for the
//!   concrete instance, produced by this very loop, where it claims a cost of 4
//!   for an instance whose optimum cost is 3, with a self-consistent model to
//!   match. It is therefore used as an **incumbent** (a lower bound that prunes the
//!   search), never as an optimality certificate. The certificate comes from an
//!   exact branch-and-bound.
//!
//! # The exact algorithm (`smt-opt`)
//!
//! Feasibility of a soft-constraint subset is **downward closed**: if `T` is
//! infeasible with the hard constraints then so is every superset of `T`.
//! MaxSMT is therefore a maximum-weight independent-set problem over an implicit
//! hypergraph of infeasible subsets, and the implicit-hitting-set (MaxHS/IHS)
//! scheme solves it exactly with a SAT/UNSAT oracle:
//!
//! ```text
//! K ← ∅                                   // known infeasible cores
//! loop:
//!     T ← argmax { w(T) : T contains no core in K }  // ← exact, RC2-warm-started
//!     if SMT(hard ∧ T) is SAT:  return T             // optimal
//!     else:                     K ← K ∪ { MUS(T) }   // ← deletion-based core
//! ```
//!
//! **Optimality proof.** Every feasible `T′` contains no core (cores are
//! infeasible, feasibility is downward closed), so `T′` is a candidate for the
//! MaxSAT argmax and hence `w(T′) ≤ w(T)`. When the loop returns, `T` is itself
//! feasible. Therefore `w(T) = max { w(T′) : T′ feasible } = OPT`. ∎
//!
//! **Termination.** Each iteration adds a core that the current `T` fails to hit,
//! so that `T` is excluded from every later argmax. There are finitely many
//! subsets, so the loop terminates. (A defensive iteration cap is nevertheless
//! enforced; hitting it downgrades the answer to
//! [`MaxSmtOptimality::Feasible`] rather than lying about optimality.)
//!
//! Every returned selection is validated *concretely*: the reported `satisfied`
//! set is recomputed by evaluating each soft constraint at the returned witness,
//! so a soft constraint is only ever claimed satisfied if it demonstrably is.

use super::constraint::EmlConstraint;
use super::helpers::{check_constraint, count_constraint_vars};
use super::incremental::IncrementalEmlSolver;
use super::oxiz_backend::SmtResult;
use crate::error::EmlError;
use crate::eval::EvalCtx;

/// Defensive cap on core-guided iterations. Reaching it never corrupts the
/// answer, it only prevents claiming optimality.
#[cfg(feature = "smt-opt")]
const MAX_CORE_ITERATIONS: usize = 4096;

/// A soft constraint: satisfy it if you can, and it is worth `weight` if you do.
#[derive(Clone, Debug)]
pub struct SoftConstraint {
    /// The constraint.
    pub constraint: EmlConstraint,
    /// Weight (benefit) of satisfying it. Zero-weight softs are legal but
    /// contribute nothing to the objective.
    pub weight: u64,
}

impl SoftConstraint {
    /// A soft constraint with the given weight.
    #[must_use]
    pub fn new(constraint: EmlConstraint, weight: u64) -> Self {
        Self { constraint, weight }
    }

    /// A soft constraint with unit weight.
    #[must_use]
    pub fn unit(constraint: EmlConstraint) -> Self {
        Self::new(constraint, 1)
    }
}

/// What is actually guaranteed about a [`MaxSmtSolution`].
///
/// Read this before trusting the weight — the three variants are genuinely
/// different promises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxSmtOptimality {
    /// **Proved maximum weight.** No feasible selection has a higher weight.
    /// Only produced by the exact core-guided search (`smt-opt`).
    Optimal,
    /// **Maximal, not proved maximum.** Every soft constraint outside the
    /// selection was individually tried and made the selection infeasible, so
    /// nothing can be *added* — but a different, heavier selection may exist.
    /// This is what the greedy fallback under base `smt` returns.
    Maximal,
    /// **Feasible only.** The selection is verified feasible, but neither
    /// maximality nor optimality could be established because the oracle returned
    /// `Unknown` somewhere, or the iteration cap was reached.
    Feasible,
}

/// A verified MaxSMT solution.
#[derive(Clone, Debug)]
pub struct MaxSmtSolution {
    /// Witness assignment satisfying every hard constraint and every soft
    /// constraint listed in `satisfied`. Verified by concrete evaluation.
    pub assignments: Vec<f64>,
    /// Indices into the `soft` slice of the soft constraints that *actually hold*
    /// at `assignments`, ascending. Recomputed from the witness, not assumed.
    pub satisfied: Vec<usize>,
    /// Sum of the weights of `satisfied`.
    pub weight: u64,
    /// What is guaranteed about `weight`. **Check this.**
    pub optimality: MaxSmtOptimality,
}

/// Outcome of a MaxSMT query.
#[derive(Clone, Debug)]
pub enum MaxSmtOutcome {
    /// The hard constraints are satisfiable and this is the best selection found.
    Solved(MaxSmtSolution),
    /// The hard constraints are unsatisfiable on their own — no solution exists,
    /// whatever the soft constraints say.
    Unsat,
    /// The oracle could not decide the hard constraints.
    Unknown,
}

/// Solve a MaxSMT instance.
///
/// Dispatches to the exact core-guided search when built with `smt-opt`, and to
/// the greedy fallback otherwise. See the module docs for the guarantees.
///
/// # Errors
/// Propagates any [`EmlError`] raised while evaluating the constraints.
pub fn max_smt(
    bounds: &[(f64, f64)],
    relaxation_samples: usize,
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
) -> Result<MaxSmtOutcome, EmlError> {
    let mut solver = IncrementalEmlSolver::with_config(bounds.to_vec(), relaxation_samples, 0);

    // The hard constraints must be feasible on their own, or nothing is.
    let base = match solver.check_all(hard)? {
        SmtResult::Unsat => return Ok(MaxSmtOutcome::Unsat),
        SmtResult::Unknown => return Ok(MaxSmtOutcome::Unknown),
        SmtResult::Sat(sol) => sol,
    };

    if soft.is_empty() {
        let assignments = pad_witness(base.assignments, hard, soft, bounds);
        return Ok(MaxSmtOutcome::Solved(MaxSmtSolution {
            assignments,
            satisfied: Vec::new(),
            weight: 0,
            optimality: MaxSmtOptimality::Optimal,
        }));
    }

    #[cfg(feature = "smt-opt")]
    {
        solve_core_guided(&mut solver, bounds, hard, soft)
    }
    #[cfg(not(feature = "smt-opt"))]
    {
        solve_greedy(&mut solver, bounds, hard, soft)
    }
}

// ----------------------------------------------------------------------------
// Shared helpers
// ----------------------------------------------------------------------------

/// Build the conjunction `hard ∧ { soft[i] : i ∈ selection }`.
fn conjunction(
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
    selection: &[usize],
) -> Vec<EmlConstraint> {
    let mut out = Vec::with_capacity(hard.len() + selection.len());
    out.extend(hard.iter().cloned());
    out.extend(selection.iter().map(|&i| soft[i].constraint.clone()));
    out
}

/// Widest variable index touched by any hard or soft constraint.
fn total_vars(hard: &[EmlConstraint], soft: &[SoftConstraint]) -> usize {
    hard.iter()
        .map(count_constraint_vars)
        .chain(soft.iter().map(|s| count_constraint_vars(&s.constraint)))
        .max()
        .unwrap_or(0)
}

/// Extend a witness to cover every variable mentioned anywhere in the instance.
///
/// A witness for `hard ∧ T` only spans the variables those constraints touch. To
/// evaluate the *remaining* soft constraints at it we need values for their
/// variables too; we use the midpoint of each variable's box. This never breaks
/// `hard ∧ T` — the added coordinates are, by construction, ones none of those
/// constraints reference.
fn pad_witness(
    mut assignments: Vec<f64>,
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
    bounds: &[(f64, f64)],
) -> Vec<f64> {
    let needed = total_vars(hard, soft);
    while assignments.len() < needed {
        let i = assignments.len();
        let (lo, hi) = bounds.get(i).copied().unwrap_or((-10.0, 10.0));
        assignments.push((lo + hi) / 2.0);
    }
    assignments
}

/// The soft constraints that *actually hold* at `witness`, and their total weight.
///
/// This is what makes a reported selection trustworthy: nothing is claimed
/// satisfied on the strength of an SMT model, only on the strength of evaluating
/// the constraint at the concrete witness.
fn satisfied_at(witness: &[f64], soft: &[SoftConstraint]) -> (Vec<usize>, u64) {
    let ctx = EvalCtx::new(witness);
    let indices: Vec<usize> = (0..soft.len())
        .filter(|&i| check_constraint(&soft[i].constraint, &ctx))
        .collect();
    let weight = indices
        .iter()
        .fold(0u64, |acc, &i| acc.saturating_add(soft[i].weight));
    (indices, weight)
}

// ----------------------------------------------------------------------------
// Greedy fallback (base `smt`)
// ----------------------------------------------------------------------------

/// Weight-ordered greedy MaxSMT.
///
/// Considers the soft constraints in decreasing weight order (ties broken by
/// index, so the result is deterministic) and commits to each one that keeps the
/// accumulated selection feasible.
///
/// **This is not an optimal algorithm and does not pretend to be.** It returns a
/// *maximal* selection — every rejected soft constraint was individually shown to
/// be infeasible against the selection as it stood when tried — and labels the
/// result [`MaxSmtOptimality::Maximal`]. A rejection made early can cost more
/// weight later than it saved: the classic counterexample is weights `3, 2, 2`
/// where the `3` conflicts with both `2`s (greedy: 3, optimum: 4).
///
/// If any trial comes back `Unknown` the label drops to
/// [`MaxSmtOptimality::Feasible`], since a rejected constraint may then have been
/// perfectly satisfiable.
fn solve_greedy(
    solver: &mut IncrementalEmlSolver,
    bounds: &[(f64, f64)],
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
) -> Result<MaxSmtOutcome, EmlError> {
    let mut order: Vec<usize> = (0..soft.len()).collect();
    order.sort_by(|&a, &b| soft[b].weight.cmp(&soft[a].weight).then(a.cmp(&b)));

    let mut selection: Vec<usize> = Vec::new();
    let mut best_witness: Option<Vec<f64>> = None;
    let mut every_trial_definite = true;

    for candidate in order {
        let mut trial = selection.clone();
        trial.push(candidate);
        trial.sort_unstable();

        match solver.check_all(&conjunction(hard, soft, &trial))? {
            SmtResult::Sat(sol) => {
                selection = trial;
                best_witness = Some(sol.assignments);
            }
            SmtResult::Unsat => {}
            SmtResult::Unknown => every_trial_definite = false,
        }
    }

    // Re-derive a witness if no soft constraint was ever accepted.
    let witness = match best_witness {
        Some(w) => w,
        None => match solver.check_all(hard)? {
            SmtResult::Sat(sol) => sol.assignments,
            // The hard constraints were already proved satisfiable above, so this
            // is unreachable in practice; degrade honestly rather than panic.
            _ => return Ok(MaxSmtOutcome::Unknown),
        },
    };

    let assignments = pad_witness(witness, hard, soft, bounds);
    let (satisfied, weight) = satisfied_at(&assignments, soft);

    Ok(MaxSmtOutcome::Solved(MaxSmtSolution {
        assignments,
        satisfied,
        weight,
        optimality: if every_trial_definite {
            MaxSmtOptimality::Maximal
        } else {
            MaxSmtOptimality::Feasible
        },
    }))
}

// ----------------------------------------------------------------------------
// Exact core-guided search (`smt-opt`, OxiZ RC2)
// ----------------------------------------------------------------------------

/// Exact MaxSMT by implicit hitting sets, with OxiZ's RC2 as the inner MaxSAT
/// engine. See the module docs for the optimality proof.
#[cfg(feature = "smt-opt")]
fn solve_core_guided(
    solver: &mut IncrementalEmlSolver,
    bounds: &[(f64, f64)],
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
) -> Result<MaxSmtOutcome, EmlError> {
    // Known infeasible subsets ("cores"): each is a set of soft indices that
    // cannot all hold together with `hard`.
    let mut cores: Vec<Vec<usize>> = Vec::new();

    for _ in 0..MAX_CORE_ITERATIONS {
        // Maximum-weight selection that hits none of the known cores.
        let Some(candidate) = best_selection(soft, &cores) else {
            // The exact inner search ran out of budget, so optimality is
            // unprovable. Fall back honestly rather than claim it anyway.
            return degrade_to_greedy(solver, bounds, hard, soft);
        };
        let upper_bound = selection_weight(soft, &candidate);

        match solver.check_all(&conjunction(hard, soft, &candidate))? {
            SmtResult::Sat(sol) => {
                let assignments = pad_witness(sol.assignments, hard, soft, bounds);
                let (satisfied, weight) = satisfied_at(&assignments, soft);
                debug_assert!(
                    weight >= upper_bound,
                    "the witness must satisfy at least the selection it was solved for"
                );
                // `candidate` is feasible, and every feasible selection hits no
                // core and so was a candidate for the argmax that produced it.
                // Hence no feasible selection is heavier: this is the optimum.
                return Ok(MaxSmtOutcome::Solved(MaxSmtSolution {
                    assignments,
                    satisfied,
                    weight,
                    optimality: MaxSmtOptimality::Optimal,
                }));
            }
            SmtResult::Unsat => {
                // Shrink the infeasible candidate to an irreducible core, so the
                // clause we learn excludes as much of the search space as
                // possible.
                let core = shrink_core(solver, hard, soft, &candidate)?;
                if core.is_empty() {
                    // Would mean `hard` alone is infeasible — already refuted by
                    // the caller. Bail out rather than spin.
                    return degrade_to_greedy(solver, bounds, hard, soft);
                }
                cores.push(core);
            }
            SmtResult::Unknown => {
                // Undecidable candidate: no core to learn, no progress possible.
                return degrade_to_greedy(solver, bounds, hard, soft);
            }
        }
    }

    degrade_to_greedy(solver, bounds, hard, soft)
}

/// Total weight of a selection.
#[cfg(feature = "smt-opt")]
fn selection_weight(soft: &[SoftConstraint], selection: &[usize]) -> u64 {
    selection
        .iter()
        .fold(0u64, |acc, &i| acc.saturating_add(soft[i].weight))
}

/// The exact search could not finish. Return the greedy solution, explicitly
/// downgraded to [`MaxSmtOptimality::Feasible`] — we must not claim maximality
/// either, because the reason we are here is that the oracle went `Unknown`
/// somewhere.
#[cfg(feature = "smt-opt")]
fn degrade_to_greedy(
    solver: &mut IncrementalEmlSolver,
    bounds: &[(f64, f64)],
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
) -> Result<MaxSmtOutcome, EmlError> {
    Ok(match solve_greedy(solver, bounds, hard, soft)? {
        MaxSmtOutcome::Solved(mut sol) => {
            sol.optimality = MaxSmtOptimality::Feasible;
            MaxSmtOutcome::Solved(sol)
        }
        other => other,
    })
}

/// Deletion-based shrinking of an infeasible selection to an irreducible core.
///
/// Identical in spirit to [`super::core::minimize_unsat_core`], but works over
/// *soft indices* against a fixed hard background, reusing the same live solver.
#[cfg(feature = "smt-opt")]
fn shrink_core(
    solver: &mut IncrementalEmlSolver,
    hard: &[EmlConstraint],
    soft: &[SoftConstraint],
    selection: &[usize],
) -> Result<Vec<usize>, EmlError> {
    let mut core = selection.to_vec();
    let mut position = 0;
    while position < core.len() {
        let mut trial = core.clone();
        trial.remove(position);
        if solver
            .check_all(&conjunction(hard, soft, &trial))?
            .is_unsat()
        {
            core.remove(position);
        } else {
            position += 1;
        }
    }
    Ok(core)
}

/// Maximum-weight subset of the soft constraints that contains no known core.
///
/// This is the argmax the hitting-set loop needs, and it must be **exact** — the
/// entire optimality proof rests on it.
///
/// # Why OxiZ's RC2 cannot certify it (measured, not assumed)
///
/// `oxiz::opt::Rc2Solver` is a real SAT-backed core-guided MaxSAT solver (unlike
/// `oxiz::opt::MaxSmtSolver`, which is a stub that returns `Unknown`), but in
/// 0.2.3 its **weighted** cost accounting is wrong when cores overlap. On the
/// instance actually produced by this loop —
///
/// ```text
/// soft:  b0 (w=3), b1 (w=2), b2 (w=2)
/// hard:  (¬b0 ∨ ¬b2), (¬b0 ∨ ¬b1)
/// ```
///
/// the optimum is `{b1, b2}` at weight 4 (cost 3), but RC2 reports `cost() == 4`
/// and a model selecting only `{b0}` (weight 3). Its `cost()` is *self-consistent*
/// with that model, so even cross-checking `weight == total − cost` would have
/// happily certified a suboptimal answer as optimal. Trusting it would have made
/// [`MaxSmtOptimality::Optimal`] a lie.
///
/// # What we do instead
///
/// RC2 is still used, in the role its answer *can* be trusted for: it produces a
/// good **incumbent** (a feasible lower bound). The incumbent is validated to be
/// core-free and then handed to an exact branch-and-bound that proves optimality
/// by exhaustion. A strong incumbent is exactly what makes the branch-and-bound
/// cheap — it prunes the search immediately — so RC2 earns its keep without
/// having to be right.
///
/// Returns `None` only if the search budget is exhausted, in which case the caller
/// refuses to claim optimality rather than guessing.
#[cfg(feature = "smt-opt")]
fn best_selection(soft: &[SoftConstraint], cores: &[Vec<usize>]) -> Option<Vec<usize>> {
    // No cores learned yet ⇒ "select everything" is trivially the maximum.
    if cores.is_empty() {
        return Some((0..soft.len()).collect());
    }

    // Incumbent from RC2: a lower bound, never an optimality certificate.
    let incumbent = rc2_incumbent(soft, cores)
        .filter(|sel| is_core_free(sel, cores))
        .unwrap_or_default();
    let incumbent_weight = selection_weight(soft, &incumbent);

    let mut search = BranchAndBound::new(soft, cores, incumbent, incumbent_weight);
    if search.explore(0) {
        Some(search.best)
    } else {
        None
    }
}

/// Node budget for the exact search. Exhausting it costs the `Optimal` label, not
/// correctness.
#[cfg(feature = "smt-opt")]
const MAX_SEARCH_NODES: usize = 500_000;

/// Exact branch-and-bound for the maximum-weight core-free subset.
///
/// Branches on the soft constraints in decreasing weight order (which makes the
/// bound bite early) and prunes whenever *every* remaining constraint together
/// with the current selection still cannot beat the incumbent. Inclusion is
/// rejected the moment it would complete a known core, so only core-free
/// selections are ever scored.
#[cfg(feature = "smt-opt")]
struct BranchAndBound<'a> {
    soft: &'a [SoftConstraint],
    cores: &'a [Vec<usize>],
    /// Soft indices in decreasing-weight order.
    order: Vec<usize>,
    /// `suffix[k]` = total weight of `order[k..]` — the optimistic completion.
    suffix: Vec<u64>,
    /// `included[i]` = soft `i` is in the partial selection.
    included: Vec<bool>,
    current: Vec<usize>,
    current_weight: u64,
    best: Vec<usize>,
    best_weight: u64,
    budget: usize,
}

#[cfg(feature = "smt-opt")]
impl<'a> BranchAndBound<'a> {
    fn new(
        soft: &'a [SoftConstraint],
        cores: &'a [Vec<usize>],
        best: Vec<usize>,
        best_weight: u64,
    ) -> Self {
        let n = soft.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| soft[b].weight.cmp(&soft[a].weight).then(a.cmp(&b)));

        let mut suffix = vec![0u64; n + 1];
        for k in (0..n).rev() {
            suffix[k] = suffix[k + 1].saturating_add(soft[order[k]].weight);
        }

        Self {
            soft,
            cores,
            order,
            suffix,
            included: vec![false; n],
            current: Vec::new(),
            current_weight: 0,
            best,
            best_weight,
            budget: MAX_SEARCH_NODES,
        }
    }

    /// Explore from position `k` in the branching order. Returns `false` when the
    /// node budget ran out (the result is then not provably optimal).
    fn explore(&mut self, k: usize) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.budget -= 1;

        if k == self.order.len() {
            if self.current_weight > self.best_weight {
                self.best_weight = self.current_weight;
                self.best = self.current.clone();
                self.best.sort_unstable();
            }
            return true;
        }

        // Optimistic bound: even taking every remaining constraint cannot beat the
        // incumbent, so this whole subtree is worthless.
        if self
            .current_weight
            .saturating_add(self.suffix[k])
            .le(&self.best_weight)
        {
            return true;
        }

        let i = self.order[k];

        // Branch 1: include `i`, unless doing so completes a known core.
        if !self.completes_core(i) {
            self.included[i] = true;
            self.current.push(i);
            self.current_weight = self.current_weight.saturating_add(self.soft[i].weight);

            let ok = self.explore(k + 1);

            self.current_weight -= self.soft[i].weight;
            self.current.pop();
            self.included[i] = false;

            if !ok {
                return false;
            }
        }

        // Branch 2: exclude `i`.
        self.explore(k + 1)
    }

    /// True when adding `i` to the current partial selection would complete a core
    /// (i.e. every other member of some core is already included).
    fn completes_core(&self, i: usize) -> bool {
        self.cores
            .iter()
            .any(|core| core.contains(&i) && core.iter().all(|&j| j == i || self.included[j]))
    }
}

/// A good core-free selection from OxiZ's RC2 MaxSAT engine, used purely as an
/// incumbent (lower bound) for the exact search above.
///
/// Stratification is switched off: `Rc2Config::default()` solves each weight level
/// independently without carrying the higher levels' commitments down, which makes
/// its answer even less useful. RC2's model is then completed to maximality —
/// RC2 only constrains its relaxation variables, so it may leave a selector false
/// where setting it true would violate nothing — which is what turns it into a
/// genuinely strong bound.
///
/// Nothing here is trusted: the caller re-checks that the result is core-free, and
/// the branch-and-bound can only improve on it.
#[cfg(feature = "smt-opt")]
fn rc2_incumbent(soft: &[SoftConstraint], cores: &[Vec<usize>]) -> Option<Vec<usize>> {
    use oxiz::opt::{MaxSatResult, Rc2Config, Rc2Solver, SoftId, Weight};
    use oxiz::sat::{LBool, Lit, Var};

    let n = soft.len();
    let mut rc2 = Rc2Solver::with_config(Rc2Config {
        stratified: false,
        ..Rc2Config::default()
    });

    for core in cores {
        // Cores are never empty (the caller checks), so this is never the empty
        // clause and the hard part can never be trivially unsatisfiable.
        let clause: Vec<Lit> = core
            .iter()
            .map(|&i| u32::try_from(i).ok().map(|v| Lit::neg(Var::new(v))))
            .collect::<Option<Vec<Lit>>>()?;
        rc2.add_hard(clause);
    }

    for (i, s) in soft.iter().enumerate() {
        let idx = u32::try_from(i).ok()?;
        rc2.add_soft(
            SoftId::from(idx),
            [Lit::pos(Var::new(idx))],
            Weight::from(s.weight),
        );
    }

    match rc2.solve() {
        Ok(MaxSatResult::Optimal | MaxSatResult::Satisfiable) => {}
        _ => return None,
    }

    let model = rc2.get_model();
    let mut selection: Vec<usize> = (0..n)
        .filter(|&i| matches!(model.get(i), Some(LBool::True)))
        .collect();
    if !is_core_free(&selection, cores) {
        return None;
    }

    // Complete RC2's model to a maximal core-free selection (heaviest first).
    let mut spare: Vec<usize> = (0..n).filter(|i| !selection.contains(i)).collect();
    spare.sort_by(|&a, &b| soft[b].weight.cmp(&soft[a].weight).then(a.cmp(&b)));
    for i in spare {
        let mut trial = selection.clone();
        trial.push(i);
        trial.sort_unstable();
        if is_core_free(&trial, cores) {
            selection = trial;
        }
    }

    Some(selection)
}

/// True when `selection` contains no known core as a subset.
#[cfg(feature = "smt-opt")]
fn is_core_free(selection: &[usize], cores: &[Vec<usize>]) -> bool {
    !cores
        .iter()
        .any(|core| core.iter().all(|i| selection.contains(i)))
}
