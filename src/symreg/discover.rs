//! Single-output symbolic regression discovery strategies.
//!
//! Implements [`SymRegEngine::discover`], the strategy dispatcher, and the
//! three concrete search strategies:
//! - [`SymRegEngine::discover_exhaustive`] — full topology enumeration.
//! - [`SymRegEngine::discover_beam`] — beam search with surrogate scoring.
//! - [`SymRegEngine::discover_mcts`] — Monte-Carlo tree search (bridges to `mcts.rs`).
//!
//! Also contains the shared optimization primitives:
//! - [`SymRegEngine::optimize_topology`] — single-topology Adam loop.
//! - [`SymRegEngine::optimize_and_finalize`] — batch finalization + optional CV.
//! - [`SymRegEngine::k_fold_cv`] — k-fold cross-validation.

use rand::RngExt;
use rand::SeedableRng;

use crate::error::EmlError;
use crate::eval::EvalCtx;
use crate::grad::ParameterizedEmlTree;
use crate::tree::EmlTree;

use super::dedupe_by_semantics;
use super::loss::{
    SymRegLoss, huber_grad_factor, huber_loss, trimmed_mse, trimmed_mse_grad_factor,
};
use super::post_round::{try_extract_named_constants, try_post_adam_rounding};
use super::topology::{compute_mse_direct, topology_interval_feasible};
use super::{
    DiscoveredFormula, OptimizerKind, SelectionCriterion, SymRegConfig, SymRegEngine,
    SymRegStrategy,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type Rng = rand::rngs::StdRng;

/// Derive a per-topology seed from a master seed using SplitMix64 mixing.
///
/// Guarantees that each topology gets a statistically independent RNG stream
/// even when master seeds are close together.
pub(super) fn derive_seed(master: u64, topology_idx: u64) -> u64 {
    // SplitMix64 step applied twice (once per input) then XOR-folded.
    let mix = |mut z: u64| -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    mix(master).wrapping_add(mix(topology_idx))
}

/// Build a seeded or OS-random RNG for topology `salt`.
pub(super) fn make_rng(seed: Option<u64>, salt: u64) -> Rng {
    match seed {
        Some(s) => Rng::seed_from_u64(derive_seed(s, salt)),
        None => rand::make_rng::<Rng>(),
    }
}

/// Access a single row from a row-major flat Jacobian buffer.
///
/// `jac_buf` holds `valid_count * n_params` entries in row-major order.
/// Returns a slice of length `n_params` for row `row`.
#[inline]
fn jac_row(jac_buf: &[f64], row: usize, n_params: usize) -> &[f64] {
    &jac_buf[row * n_params..(row + 1) * n_params]
}

/// Compute Akaike (AIC) and Bayesian (BIC) information criteria.
///
/// Formulas: `AIC = n · ln(MSE) + 2k` and `BIC = n · ln(MSE) + k · ln(n)`,
/// where `n` is the sample count and `k` is the number of free parameters.
pub(super) fn compute_aic_bic(mse: f64, n_data: usize, n_params: usize) -> (f64, f64) {
    let n = n_data as f64;
    let k = n_params as f64;
    let rss_per_n = mse.max(f64::MIN_POSITIVE);
    let aic = n * rss_per_n.ln() + 2.0 * k;
    let bic = n * rss_per_n.ln() + k * n.max(1.0).ln();
    (aic, bic)
}

impl SymRegEngine {
    /// Discover closed-form formulas from input-output data.
    ///
    /// Dispatches to [`Self::discover_exhaustive`] or [`Self::discover_beam`]
    /// according to [`SymRegConfig::strategy`].
    ///
    /// - `inputs`: each row is one data point's variable values
    /// - `targets`: corresponding output values
    /// - `num_vars`: number of input variables
    ///
    /// Returns formulas sorted by score (best first).
    pub fn discover(
        &self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        num_vars: usize,
    ) -> Result<Vec<DiscoveredFormula>, EmlError> {
        match self.config.strategy {
            SymRegStrategy::Exhaustive => self.discover_exhaustive(inputs, targets, num_vars),
            SymRegStrategy::Beam { width } => self.discover_beam(inputs, targets, num_vars, width),
            SymRegStrategy::Mcts {
                iterations,
                exploration,
            } => self.discover_mcts(inputs, targets, num_vars, iterations, exploration),
            SymRegStrategy::Evolutionary {
                population,
                generations,
                tournament_size,
                crossover_rate,
                mutation_rate,
                elitism,
            } => {
                let best = super::evolution::run_evolutionary(
                    inputs,
                    targets,
                    &self.config,
                    population,
                    generations,
                    tournament_size,
                    crossover_rate,
                    mutation_rate,
                    elitism,
                    1,
                    0,
                    0,
                )?;
                Ok(vec![best])
            }
            SymRegStrategy::Islands {
                n_islands,
                migration_interval,
                migrants,
                population,
                generations,
                tournament_size,
                crossover_rate,
                mutation_rate,
                elitism,
            } => {
                let best = super::evolution::run_evolutionary(
                    inputs,
                    targets,
                    &self.config,
                    population,
                    generations,
                    tournament_size,
                    crossover_rate,
                    mutation_rate,
                    elitism,
                    n_islands,
                    migration_interval,
                    migrants,
                )?;
                Ok(vec![best])
            }
        }
    }

    /// Discover formulas using exhaustive topology enumeration.
    ///
    /// Evaluates every distinct topology up to `max_depth` with full Adam
    /// optimisation.  Use [`Self::discover_beam`] when the topology space is
    /// too large to enumerate exhaustively.
    pub fn discover_exhaustive(
        &self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        num_vars: usize,
    ) -> Result<Vec<DiscoveredFormula>, EmlError> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(EmlError::EmptyData);
        }
        if inputs.len() != targets.len() {
            return Err(EmlError::DimensionMismatch(inputs.len(), targets.len()));
        }

        // Phase 1: Enumerate all topologies up to max_depth
        let const_leaf = if self.config.enable_const_leaf {
            Some(self.config.const_leaf_init)
        } else {
            None
        };
        let topologies = super::topology::enumerate_topologies_gated(
            self.config.max_depth,
            num_vars,
            const_leaf,
        );

        // Phase 1b: Prune semantically-equivalent topologies
        let topologies = dedupe_by_semantics(topologies);

        // Phase 1c (optional): Interval-arithmetic pre-filter.
        let topologies = if self.config.interval_pruning {
            use crate::lower_interval::IntervalLO;

            let input_intervals: Vec<IntervalLO> = (0..num_vars)
                .map(|j| {
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for row in inputs.iter() {
                        if let Some(&v) = row.get(j) {
                            if v < lo {
                                lo = v;
                            }
                            if v > hi {
                                hi = v;
                            }
                        }
                    }
                    if lo.is_finite() && hi.is_finite() {
                        IntervalLO::new(lo, hi)
                    } else {
                        IntervalLO::full()
                    }
                })
                .collect();

            let target_lo = targets.iter().copied().fold(f64::INFINITY, f64::min);
            let target_hi = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let threshold = self.config.interval_pruning_depth_threshold;

            topologies
                .into_iter()
                .filter(|topo| {
                    if topo.depth() < threshold {
                        return true;
                    }
                    if !topology_interval_feasible(topo, &input_intervals, target_lo, target_hi) {
                        return false;
                    }
                    // Additional SMT-backed pruning (opt-in)
                    #[cfg(feature = "smt")]
                    {
                        use crate::smt::Interval;
                        let smt_vars: Vec<Interval> = input_intervals
                            .iter()
                            .map(|iv| Interval::new(iv.lo, iv.hi))
                            .collect();
                        let constraint = crate::smt::EmlConstraint::GeZero(topo.clone());
                        if self.config.smt_prune_solver {
                            let depth = topo.depth();
                            let min_d = self.config.interval_pruning_depth_threshold;
                            if super::smt_prune::solver_prune(&constraint, &smt_vars, min_d, depth)
                            {
                                return false;
                            }
                        } else if self.config.smt_prune
                            && super::smt_prune::interval_prune(&constraint, &smt_vars)
                        {
                            return false;
                        }
                    }
                    true
                })
                .collect()
        } else {
            topologies
        };

        // Phase 1d (optional): Dimensional-analysis pre-filter.
        let topologies = if let Some((ref var_units, target_units)) = self.config.unit_filter {
            topologies
                .into_iter()
                .filter(|topo| {
                    let lowered = topo.lower().simplify();
                    matches!(lowered.check_units(var_units), Ok(u) if u == target_units)
                })
                .collect()
        } else {
            topologies
        };

        self.optimize_and_finalize(topologies, inputs, targets)
    }

    /// Discover formulas using beam search.
    ///
    /// Two-phase approach:
    /// 1. **Surrogate pass**: Run each topology with a cheap budget (few Adam steps,
    ///    single restart) to get a quick score estimate.
    /// 2. **Full pass**: Keep only the top `width` candidates by surrogate MSE and
    ///    run full Adam optimisation on those.
    pub fn discover_beam(
        &self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        num_vars: usize,
        width: usize,
    ) -> Result<Vec<DiscoveredFormula>, EmlError> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(EmlError::EmptyData);
        }
        if inputs.len() != targets.len() {
            return Err(EmlError::DimensionMismatch(inputs.len(), targets.len()));
        }

        // Phase 1: enumerate and deduplicate (same as exhaustive)
        let const_leaf = if self.config.enable_const_leaf {
            Some(self.config.const_leaf_init)
        } else {
            None
        };
        let topologies = super::topology::enumerate_topologies_gated(
            self.config.max_depth,
            num_vars,
            const_leaf,
        );
        let topologies = dedupe_by_semantics(topologies);

        // Phase 1b (optional): Dimensional-analysis pre-filter (mirrors exhaustive path).
        let topologies = if let Some((ref var_units, target_units)) = self.config.unit_filter {
            topologies
                .into_iter()
                .filter(|topo| {
                    let lowered = topo.lower().simplify();
                    matches!(lowered.check_units(var_units), Ok(u) if u == target_units)
                })
                .collect()
        } else {
            topologies
        };

        // Phase 2: surrogate pass with cheap Adam budget.
        let surrogate_iters = self.config.max_iter.clamp(10, 50);
        let surrogate_config = SymRegConfig {
            max_iter: surrogate_iters,
            num_restarts: 1,
            cv_folds: None,
            ..self.config.clone()
        };
        let surrogate_engine = SymRegEngine::new(surrogate_config);

        #[cfg(feature = "parallel")]
        let mut surrogate_scores: Vec<(usize, f64)> = topologies
            .par_iter()
            .enumerate()
            .filter_map(|(i, topo)| {
                surrogate_engine
                    .optimize_topology(topo, inputs, targets, i)
                    .map(|f| (i, f.mse))
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let mut surrogate_scores: Vec<(usize, f64)> = topologies
            .iter()
            .enumerate()
            .filter_map(|(i, topo)| {
                surrogate_engine
                    .optimize_topology(topo, inputs, targets, i)
                    .map(|f| (i, f.mse))
            })
            .collect();

        // Phase 3: truncate to top `width` by surrogate MSE
        surrogate_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let effective_width = width.max(1);
        surrogate_scores.truncate(effective_width);

        let mut keep_indices: Vec<usize> = surrogate_scores.iter().map(|&(i, _)| i).collect();
        keep_indices.sort_unstable();
        let beam_topologies: Vec<EmlTree> = keep_indices
            .iter()
            .filter_map(|&i| topologies.get(i).cloned())
            .collect();

        // Phase 4: full Adam on the beam candidates
        self.optimize_and_finalize(beam_topologies, inputs, targets)
    }

    /// Bridge: run MCTS topology search.
    fn discover_mcts(
        &self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        num_vars: usize,
        iterations: usize,
        exploration: f64,
    ) -> Result<Vec<DiscoveredFormula>, EmlError> {
        super::mcts::run_mcts(self, inputs, targets, num_vars, iterations, exploration)
    }

    /// Shared finalization: optimize topologies, sort, optionally cross-validate.
    pub(super) fn optimize_and_finalize(
        &self,
        topologies: Vec<EmlTree>,
        inputs: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<Vec<DiscoveredFormula>, EmlError> {
        #[cfg(feature = "parallel")]
        let mut formulas: Vec<DiscoveredFormula> = topologies
            .par_iter()
            .enumerate()
            .filter_map(|(i, topology)| self.optimize_topology(topology, inputs, targets, i))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let mut formulas: Vec<DiscoveredFormula> = topologies
            .iter()
            .enumerate()
            .filter_map(|(i, topology)| self.optimize_topology(topology, inputs, targets, i))
            .collect();

        // Sort by score — use complexity and structural hash as tiebreakers
        formulas.sort_by(|a, b| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hasher;
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.complexity.cmp(&b.complexity))
                .then_with(|| {
                    let hash_of = |f: &DiscoveredFormula| {
                        let mut h = DefaultHasher::new();
                        f.eml_tree.lower().simplify().structural_hash(&mut h);
                        h.finish()
                    };
                    hash_of(a).cmp(&hash_of(b))
                })
        });

        // Optional: k-fold cross-validation
        if let Some(k) = self.config.cv_folds {
            let k = k.clamp(2, inputs.len());
            for formula in &mut formulas {
                formula.cv_mse = Some(super::cv::k_fold_cv(
                    self,
                    &formula.eml_tree,
                    &formula.params,
                    inputs,
                    targets,
                    k,
                ));
            }
            formulas.sort_by(|a, b| {
                let a_score = a.cv_mse.unwrap_or(a.score);
                let b_score = b.cv_mse.unwrap_or(b.score);
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Re-sort by IC if requested
        match self.config.selection {
            SelectionCriterion::Score => {}
            SelectionCriterion::Aic => formulas.sort_by(|a, b| {
                a.aic
                    .partial_cmp(&b.aic)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            SelectionCriterion::Bic => formulas.sort_by(|a, b| {
                a.bic
                    .partial_cmp(&b.bic)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
        }

        // Optional: bootstrap uncertainty quantification on top-k formulas
        if self.config.bootstrap_samples > 0 {
            let top_k = self.config.uq_top_k.min(formulas.len());
            for formula in &mut formulas[..top_k] {
                formula.param_intervals = super::uncertainty::compute_bootstrap_intervals(
                    self,
                    formula,
                    inputs,
                    targets,
                    self.config.bootstrap_samples,
                    self.config.confidence_level,
                    self.config.seed,
                );
            }
        } else if self.config.uq_analytic
            && matches!(self.config.optimizer, OptimizerKind::LevenbergMarquardt)
        {
            // Analytic (Laplace) UQ: only meaningful for LM optimizer
            let top_k = self.config.uq_top_k.min(formulas.len());
            for formula in &mut formulas[..top_k] {
                if formula.params.is_empty() {
                    continue;
                }
                formula.param_intervals = super::optimize_lm::jac_residuals_for_uq(
                    &formula.eml_tree,
                    &formula.params,
                    inputs,
                    targets,
                )
                .and_then(|(residuals, jac, n_data, n_params)| {
                    super::uncertainty::compute_analytic_intervals(
                        &jac,
                        &residuals,
                        &formula.params,
                        n_data,
                        n_params,
                        self.config.confidence_level,
                    )
                });
            }
        }

        Ok(formulas)
    }

    /// Optimize parameters for a single topology — dispatches to Adam or LM.
    pub(super) fn optimize_topology(
        &self,
        topology: &EmlTree,
        inputs: &[Vec<f64>],
        targets: &[f64],
        topology_idx: usize,
    ) -> Option<DiscoveredFormula> {
        match self.config.optimizer {
            OptimizerKind::Adam => {
                self.optimize_topology_adam(topology, inputs, targets, topology_idx)
            }
            OptimizerKind::LevenbergMarquardt => super::optimize_lm::optimize_topology_lm(
                self,
                topology,
                inputs,
                targets,
                topology_idx,
            ),
        }
    }

    /// Optimize parameters using the Adam optimizer (renamed internal entry point).
    pub(super) fn optimize_topology_adam(
        &self,
        topology: &EmlTree,
        inputs: &[Vec<f64>],
        targets: &[f64],
        topology_idx: usize,
    ) -> Option<DiscoveredFormula> {
        let mut best_mse = f64::INFINITY;
        let mut best_params = Vec::new();
        let mut rng = make_rng(self.config.seed, topology_idx as u64);

        for _ in 0..self.config.num_restarts {
            let mut ptree = ParameterizedEmlTree::from_topology(topology, 1.0);

            for p in &mut ptree.params {
                *p = 1.0 + rng.random_range(-0.5..0.5);
            }

            let n_params = ptree.num_params();
            if n_params == 0 {
                let mse = compute_mse_direct(topology, inputs, targets);
                if let Some(mse) = mse {
                    if mse < best_mse {
                        best_mse = mse;
                        best_params = vec![];
                    }
                }
                break;
            }

            let mut m = vec![0.0_f64; n_params];
            let mut v = vec![0.0_f64; n_params];
            let beta1 = 0.9;
            let beta2 = 0.999;
            let epsilon = 1e-8;
            let lr = self.config.learning_rate;

            // Pre-allocated buffers — reused across iterations to avoid per-iter alloc churn.
            let mut residuals: Vec<f64> = Vec::with_capacity(inputs.len());
            // Row-major flat Jacobian: valid_count rows × n_params columns.
            let mut jac_buf: Vec<f64> = Vec::with_capacity(inputs.len() * n_params.max(1));
            // One-row scratch buffer for forward_with_jacobian_into.
            let mut jac_tmp: Vec<f64> = Vec::with_capacity(n_params);
            // Gradient accumulator — zeroed at the top of each iteration.
            let mut tg: Vec<f64> = vec![0.0_f64; n_params];

            let mut converged = false;

            for t in 1..=self.config.max_iter {
                residuals.clear();
                jac_buf.clear();
                tg.iter_mut().for_each(|x| *x = 0.0);

                for (input, &target) in inputs.iter().zip(targets) {
                    let ctx = EvalCtx::new(input);
                    match ptree.forward_with_jacobian_into(&ctx, &mut jac_tmp) {
                        Ok(out) if out.is_finite() => {
                            residuals.push(out - target);
                            jac_buf.extend_from_slice(&jac_tmp);
                        }
                        _ => {}
                    }
                }

                let valid_count = residuals.len();
                if valid_count == 0 {
                    break;
                }

                let total_loss = match &self.config.loss {
                    SymRegLoss::Mse => {
                        let tloss: f64 = residuals.iter().map(|r| r * r).sum();
                        for (row_idx, r) in residuals.iter().enumerate() {
                            let jac = jac_row(&jac_buf, row_idx, n_params);
                            for (tg_i, &j) in tg.iter_mut().zip(jac) {
                                if j.is_finite() {
                                    *tg_i += 2.0 * r * j;
                                }
                            }
                        }
                        tloss
                    }
                    SymRegLoss::Huber { delta } => {
                        let d = *delta;
                        let tloss = huber_loss(&residuals, d) * valid_count as f64;
                        for (row_idx, r) in residuals.iter().enumerate() {
                            let gf = 2.0 * huber_grad_factor(*r, d);
                            let jac = jac_row(&jac_buf, row_idx, n_params);
                            for (tg_i, &j) in tg.iter_mut().zip(jac) {
                                if j.is_finite() {
                                    *tg_i += gf * j;
                                }
                            }
                        }
                        tloss
                    }
                    SymRegLoss::TrimmedMse { alpha } => {
                        let a = *alpha;
                        let tloss = trimmed_mse(&residuals, a) * valid_count as f64;
                        for (row_idx, r) in residuals.iter().enumerate() {
                            let gf = 2.0 * trimmed_mse_grad_factor(*r, &residuals, a);
                            let jac = jac_row(&jac_buf, row_idx, n_params);
                            for (tg_i, &j) in tg.iter_mut().zip(jac) {
                                if j.is_finite() {
                                    *tg_i += gf * j;
                                }
                            }
                        }
                        tloss
                    }
                };

                let n_f = valid_count as f64;
                let mse = total_loss / n_f;

                if mse < self.config.tolerance {
                    best_mse = mse;
                    best_params = ptree.params.clone();
                    converged = true;
                    break;
                }

                for i in 0..n_params {
                    let g = tg[i] / n_f;
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                    let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                    let v_hat = v[i] / (1.0 - beta2.powi(t as i32));
                    ptree.params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }

                if mse < best_mse {
                    best_mse = mse;
                    best_params = ptree.params.clone();
                }
            }

            if converged {
                break;
            }
        }

        // Phase 4: Integer rounding
        if self.config.integer_rounding && !best_params.is_empty() {
            let (rounded_params, rounded_mse) =
                try_post_adam_rounding(topology, best_params, best_mse, inputs, targets);
            best_params = rounded_params;
            best_mse = rounded_mse;
        }

        if !best_mse.is_finite() || best_mse > 1e10 {
            return None;
        }

        let complexity = topology.size();

        let (final_op, final_mse) = try_extract_named_constants(
            topology,
            &best_params,
            best_mse,
            self.config.constant_extraction,
            inputs,
            targets,
        );

        let pretty = final_op.to_pretty();

        let (aic, bic) = compute_aic_bic(final_mse, inputs.len(), best_params.len());

        Some(DiscoveredFormula {
            eml_tree: topology.clone(),
            mse: final_mse,
            complexity,
            score: final_mse + self.config.complexity_penalty * complexity as f64,
            pretty,
            params: best_params,
            cv_mse: None,
            aic,
            bic,
            param_intervals: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::symreg::{SymRegConfig, SymRegEngine};

    #[test]
    fn test_const_leaf_legacy_mode_no_panic() {
        let cfg = SymRegConfig {
            enable_const_leaf: false,
            ..SymRegConfig::quick()
        };
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<f64> = x.iter().map(|r| 3.7 * r[0] * r[0]).collect();
        let engine = SymRegEngine::new(cfg);
        let result = engine.discover(&x, &y, 1);
        assert!(result.is_ok());
    }
}
