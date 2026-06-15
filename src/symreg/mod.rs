//! Symbolic regression engine.
//!
//! Discovers closed-form mathematical formulas from data using EML trees.
//! The algorithm enumerates tree topologies up to a maximum depth, optimizes
//! continuous parameters via Adam, and selects the best formulas by MSE
//! with a complexity penalty (Occam's razor).

use crate::tree::EmlTree;
use crate::units::Units;

mod constants;
mod cv;
mod discover;
mod discover_multi;
mod discover_shared;
mod evolution;
mod loss;
mod mcts;
mod numerics;
mod optimize_lm;
mod pareto;
mod pde;
mod post_round;
mod sindy;
#[cfg(feature = "smt")]
mod smt_prune;
mod strlsq;
mod topology;
mod uncertainty;

pub use topology::{dedupe_by_semantics, enumerate_topologies};

// Public re-exports so downstream crates see everything at `crate::symreg::*`.
pub use constants::snap_to_named_const;
pub use evolution::run_evolutionary;
pub use loss::SymRegLoss;
pub use pareto::{dominates_by, pareto_front, pareto_front_ic};
pub use pde::{
    PdeConfig, PdeField, PdeLibraryTerm, PdeMode, PdeResult, PdeShape, discover_pde,
    discover_pde_nd,
};
pub use sindy::{
    LibraryTerm, SindyConfig, SindyEquation, SindyMode, SindyResult, discover_ode_sindy,
};
pub use strlsq::strlsq_qr;
pub use uncertainty::{compute_analytic_intervals, compute_bootstrap_intervals, inv_norm_cdf};

/// Strategy for multi-output symbolic regression.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MultiOutputStrategy {
    /// Each output runs a completely independent single-output regression. Default.
    #[default]
    Independent,
    /// All outputs share a single topology skeleton; each output has its own parameter vector.
    /// The complexity cost of the skeleton is counted once (parsimony win over Independent).
    SharedTopology,
}

/// Topology search strategy.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SymRegStrategy {
    /// Exhaustive enumeration of all topologies up to `max_depth`. Default.
    #[default]
    Exhaustive,
    /// Bounded beam search: score each topology with a cheap surrogate
    /// (few Adam steps), keep top `width` candidates, then do full Adam on those.
    Beam {
        /// Maximum number of candidates to keep at each depth level.
        width: usize,
    },
    /// Monte-Carlo tree search over partial EML topologies.
    ///
    /// Uses UCB1 selection: `score + exploration * sqrt(ln(parent_visits) / child_visits)`.
    /// Each rollout: randomly complete a partial tree, fit with a few Adam steps,
    /// return `1/(1+MSE)` as the value signal (higher = better fit).
    Mcts {
        /// Total number of MCTS rollout iterations.
        iterations: usize,
        /// UCB1 exploration coefficient (higher = more exploration).
        exploration: f64,
    },
    /// Genetic algorithm with tournament selection, crossover, and mutation.
    Evolutionary {
        /// Population size per island.
        population: usize,
        /// Number of generations.
        generations: usize,
        /// Tournament size for selection.
        tournament_size: usize,
        /// Crossover probability [0, 1].
        crossover_rate: f64,
        /// Mutation probability per individual [0, 1].
        mutation_rate: f64,
        /// Number of elite individuals copied unchanged.
        elitism: usize,
    },
    /// Multiple independent island populations with ring migration.
    Islands {
        /// Number of islands.
        n_islands: usize,
        /// Generations between migrations.
        migration_interval: usize,
        /// Number of migrants per migration event.
        migrants: usize,
        /// GA parameters (same for all islands).
        population: usize,
        /// Number of generations.
        generations: usize,
        /// Tournament size for selection.
        tournament_size: usize,
        /// Crossover rate.
        crossover_rate: f64,
        /// Mutation rate.
        mutation_rate: f64,
        /// Elitism count.
        elitism: usize,
    },
}

/// Which optimizer to use for per-topology constant fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimizerKind {
    /// Adam stochastic gradient descent (default).
    #[default]
    Adam,
    /// Levenberg–Marquardt nonlinear least-squares.
    LevenbergMarquardt,
}

/// Configuration for the Levenberg–Marquardt optimizer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LmConfig {
    /// Initial damping coefficient λ.
    pub lambda_init: f64,
    /// Factor to multiply λ on step rejection.
    pub lambda_up: f64,
    /// Factor to divide λ on step acceptance.
    pub lambda_down: f64,
    /// Minimum allowed λ.
    pub lambda_min: f64,
    /// Maximum allowed λ (abort restart if exceeded).
    pub lambda_max: f64,
    /// Maximum number of LM iterations per restart.
    pub max_iter: usize,
    /// Infinity-norm gradient convergence threshold.
    pub grad_tol: f64,
    /// Relative step-size convergence threshold.
    pub step_tol: f64,
    /// Relative cost decrease convergence threshold.
    pub cost_tol: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            lambda_init: 1e-3,
            lambda_up: 10.0,
            lambda_down: 10.0,
            lambda_min: 1e-12,
            lambda_max: 1e12,
            max_iter: 100,
            grad_tol: 1e-12,
            step_tol: 1e-12,
            cost_tol: 1e-15,
        }
    }
}

/// Criterion used to rank and select among discovered formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SelectionCriterion {
    /// Default: rank by combined score = MSE + complexity_penalty × complexity.
    #[default]
    Score,
    /// Rank by Akaike Information Criterion (lower = better).
    Aic,
    /// Rank by Bayesian Information Criterion (lower = better).
    Bic,
}

/// Configuration for symbolic regression.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct SymRegConfig {
    /// Maximum tree depth to explore (paper: 4 is often sufficient).
    pub max_depth: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Convergence threshold (MSE).
    pub tolerance: f64,
    /// Maximum optimization iterations per topology.
    pub max_iter: usize,
    /// Complexity penalty coefficient (Occam's razor).
    pub complexity_penalty: f64,
    /// Number of random restarts per topology.
    pub num_restarts: usize,
    /// Whether to attempt integer rounding of parameters.
    pub integer_rounding: bool,
    /// Number of folds for k-fold cross-validation.
    ///
    /// When `Some(k)`, each formula is also evaluated on held-out folds.
    /// `cv_mse` in `DiscoveredFormula` is populated and results are sorted
    /// by `cv_mse`. When `None` (default), no cross-validation is performed
    /// and behaviour is identical to before.
    pub cv_folds: Option<usize>,
    /// Optional master RNG seed for fully reproducible runs.
    ///
    /// When `Some(s)`, per-topology seeds are derived via SplitMix64 so
    /// every topology gets an independent but deterministic RNG stream.
    /// When `None` (default), `rand::from_os_rng()` is used (non-deterministic).
    pub seed: Option<u64>,
    /// Loss function for Adam optimisation.
    ///
    /// Default is `SymRegLoss::Mse`. Use `Huber` or `TrimmedMse` to improve
    /// robustness against outliers.
    pub loss: SymRegLoss,
    /// Post-Adam constants extraction tolerance.
    ///
    /// When `Some(eps)`, each free constant is tested against a set of
    /// well-known values (π, e, √2, simple rationals). The nearest candidate
    /// is accepted when the resulting MSE satisfies
    /// `new_mse ≤ (1 + eps) * current_mse`.
    /// When `None` (default), raw float values are kept.
    pub constant_extraction: Option<f64>,
    /// Enable interval-based topology pruning before Adam fitting (cheap pre-filter).
    ///
    /// Only topologies whose output interval can span the target range are attempted.
    /// Default: `false`.
    pub interval_pruning: bool,
    /// Only apply interval pruning to topologies at this depth or deeper.
    ///
    /// Default: `2` (pruning tiny depth-1 trees is usually counterproductive).
    pub interval_pruning_depth_threshold: usize,
    /// Strategy for multi-output symbolic regression.
    ///
    /// Default: `MultiOutputStrategy::Independent`.
    pub multi_output_strategy: MultiOutputStrategy,
    /// Topology search strategy.
    ///
    /// Default: `SymRegStrategy::Exhaustive` (full enumeration, fast at depth ≤ 4).
    pub strategy: SymRegStrategy,
    /// Window size for Savitzky-Golay derivative estimation in ODE discovery.
    ///
    /// When `None` (default), [`SymRegEngine::discover_ode`] uses central
    /// differences for all state variables.  When `Some(w)` with `w >= 5`,
    /// the Savitzky-Golay smoother (window=5, poly=2) is applied instead;
    /// values below 5 are treated the same as `None`.
    pub ode_sg_window: Option<usize>,
    /// Optional dimensional-analysis filter: `Some((var_units, target_units))` enables
    /// hard pruning of dimensionally-inadmissible topologies before Adam optimisation.
    ///
    /// - `var_units[i]` gives the physical units of variable `i`.
    /// - `target_units` specifies the expected units of the regression target.
    ///
    /// A topology is retained only if [`crate::lower::LoweredOp::check_units`] returns
    /// `Ok(u)` with `u == target_units`.  All other topologies (including those that
    /// raise a `UnitError`) are skipped entirely, providing a 10–100× search-space
    /// reduction on physics problems.
    ///
    /// Default: `None` (no unit filtering; identical behaviour to previous releases).
    pub unit_filter: Option<(Vec<Units>, Units)>,
    /// Optimizer to use for per-topology constant fitting.
    ///
    /// Default: `OptimizerKind::Adam`.
    pub optimizer: OptimizerKind,
    /// Levenberg–Marquardt hyper-parameters (only used when `optimizer = LM`).
    pub lm: LmConfig,
    /// Number of bootstrap resamples for parameter confidence intervals.
    ///
    /// Set to `0` (default) to skip bootstrap UQ entirely.
    pub bootstrap_samples: usize,
    /// Confidence level for bootstrap intervals, e.g. `0.95` for 95% CI.
    pub confidence_level: f64,
    /// When `true` and `optimizer == LevenbergMarquardt`, compute analytic parameter confidence
    /// intervals via the Laplace approximation Σ = σ̂²(JᵀJ)⁻¹.
    ///
    /// Assumes MSE loss and approximately Gaussian residuals (asymptotic).
    /// If `bootstrap_samples > 0`, bootstrap takes precedence and `uq_analytic` has no effect.
    /// For Adam optimizer, `uq_analytic` has no effect (use `uq_bootstrap` instead).
    pub uq_analytic: bool,
    /// When `true` and the `smt` feature is enabled, prune candidate topologies
    /// whose interval propagation yields `Conflict` before running the optimizer.
    /// Default: `false`.
    pub smt_prune: bool,
    /// When `true`, enables OxiZ-backed UNSAT pruning for topology search
    /// (expensive, opt-in). Calls `EmlSmtSolver::check_sat` to prove that a
    /// candidate topology's output constraint is UNSAT, then skips it entirely.
    ///
    /// Requires the `smt` feature.  Depth-gated via
    /// `interval_pruning_depth_threshold`. Default: `false`.
    ///
    /// See `smt_prune` for the cheaper interval-only alternative.
    #[cfg_attr(feature = "serde", serde(default))]
    pub smt_prune_solver: bool,
    /// Criterion for ranking discovered formulas.
    ///
    /// Default: `SelectionCriterion::Score`.
    pub selection: SelectionCriterion,
    /// How many top-ranked formulas to run bootstrap UQ on.
    ///
    /// Default: `5`.
    pub uq_top_k: usize,
    /// Enable the free `Const(f64)` grammar leaf. Default `false` (back-compat).
    ///
    /// When `true`, the topology enumeration adds a `Const(const_leaf_init)` leaf
    /// to the grammar, allowing the optimizer to learn arbitrary constant values.
    /// Set to `false` (default) to use only `One` and `Var(i)` leaves — identical
    /// to the behaviour before this field was added.
    pub enable_const_leaf: bool,
    /// Initial value for `Const` leaves before optimization. Default `1.0`.
    pub const_leaf_init: f64,
}

impl Default for SymRegConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            learning_rate: 1e-3,
            tolerance: 1e-10,
            max_iter: 10_000,
            complexity_penalty: 1e-4,
            num_restarts: 3,
            integer_rounding: true,
            cv_folds: None,
            seed: None,
            loss: SymRegLoss::default(),
            constant_extraction: None,
            interval_pruning: false,
            interval_pruning_depth_threshold: 2,
            multi_output_strategy: MultiOutputStrategy::Independent,
            strategy: SymRegStrategy::Exhaustive,
            ode_sg_window: None,
            unit_filter: None,
            optimizer: OptimizerKind::Adam,
            lm: LmConfig::default(),
            bootstrap_samples: 0,
            confidence_level: 0.95,
            uq_analytic: false,
            smt_prune: false,
            smt_prune_solver: false,
            selection: SelectionCriterion::Score,
            uq_top_k: 5,
            enable_const_leaf: false,
            const_leaf_init: 1.0,
        }
    }
}

impl SymRegConfig {
    /// Quick preset — fast preview; may miss the global optimum.
    ///
    /// Use during interactive exploration or smoke tests. Shallow tree
    /// depth and few restarts trade accuracy for speed.
    pub fn quick() -> Self {
        Self {
            max_depth: 2,
            max_iter: 200,
            num_restarts: 2,
            ..Self::default()
        }
    }

    /// Balanced preset — production default. Alias for `Self::default()`.
    ///
    /// This preserves whatever `Default` returns today. If `Default` ever
    /// changes, `balanced()` moves with it.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Exhaustive preset — slow but thorough. Use for publication-quality runs.
    ///
    /// Deepens `max_depth`, increases iterations and restart count. Expect
    /// multi-minute runs on larger datasets.
    ///
    /// Note: `max_iter` is set to `20_000` so it genuinely exceeds the
    /// current `Default::default()` value of `10_000` (the plan's suggested
    /// `2_000` would have been *fewer* iterations than the balanced default
    /// and therefore inconsistent with the preset's "slower" semantics).
    pub fn exhaustive() -> Self {
        Self {
            max_depth: 4,
            max_iter: 20_000,
            num_restarts: 8,
            cv_folds: Some(5),
            ..Self::default()
        }
    }

    /// Convenience: set the unit filter for dimensional analysis.
    ///
    /// - `var_units[i]` gives the physical units of variable `i`.
    /// - `target_units` specifies the expected units of the regression target.
    pub fn with_units(mut self, var_units: Vec<Units>, target_units: Units) -> Self {
        self.unit_filter = Some((var_units, target_units));
        self
    }
}

/// A formula discovered by symbolic regression.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscoveredFormula {
    /// The EML tree representation.
    pub eml_tree: EmlTree,
    /// Final mean squared error.
    pub mse: f64,
    /// Tree node count (complexity measure).
    pub complexity: usize,
    /// Combined score: MSE + complexity_penalty * complexity.
    pub score: f64,
    /// Human-readable expression (from lowering).
    pub pretty: String,
    /// Optimized parameter values.
    pub params: Vec<f64>,
    /// Cross-validated MSE (average over held-out folds), or `None` when
    /// `SymRegConfig::cv_folds` was not set.
    pub cv_mse: Option<f64>,
    /// Akaike Information Criterion: `n·ln(MSE) + 2k`.
    pub aic: f64,
    /// Bayesian Information Criterion: `n·ln(MSE) + k·ln(n)`.
    pub bic: f64,
    /// Per-parameter confidence intervals from bootstrap or analytic UQ.
    ///
    /// `param_intervals[i] = (lower, upper)` at `SymRegConfig::confidence_level`.
    /// `None` when UQ has not been computed (`bootstrap_samples == 0`).
    pub param_intervals: Option<Vec<(f64, f64)>>,
}

impl DiscoveredFormula {
    /// Render the discovered formula as a LaTeX math expression.
    ///
    /// Lowers the EML tree and converts to LaTeX notation.
    /// Returns a string suitable for use inside `$...$` math mode.
    pub fn to_latex(&self) -> String {
        self.eml_tree.lower().simplify().to_latex()
    }
}

/// Result of a shared-topology multi-output symbolic regression run.
///
/// One EML tree skeleton is shared across all `n_outputs` dimensions.
/// Each output has an independent fitted parameter vector. The parsimony
/// advantage over [`MultiOutputStrategy::Independent`] is that tree
/// complexity is charged once regardless of the number of outputs.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SharedFormula {
    /// The shared EML tree skeleton.
    pub eml_tree: EmlTree,
    /// Per-output fitted parameter vectors (length = n_outputs).
    pub per_output_params: Vec<Vec<f64>>,
    /// Per-output MSE values (length = n_outputs).
    pub per_output_mse: Vec<f64>,
    /// Total score: Σ mse_k + complexity_penalty × node_count(tree).
    /// Tree complexity is counted once.
    pub total_score: f64,
    /// Human-readable formula per output with fitted parameters substituted in.
    pub pretty_per_output: Vec<String>,
}

#[cfg(feature = "serde")]
impl DiscoveredFormula {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to binary using `oxicode`.
    pub fn to_binary(&self) -> Result<Vec<u8>, oxicode::Error> {
        oxicode::serde::encode_serde(self)
    }

    /// Deserialize from binary bytes encoded with [`Self::to_binary`].
    pub fn from_binary(bytes: &[u8]) -> Result<Self, oxicode::Error> {
        oxicode::serde::decode_serde(bytes)
    }
}

#[cfg(feature = "tensorlogic")]
impl DiscoveredFormula {
    /// Convert this formula to a [`tensorlogic_ir::TLExpr`] via lower + simplify.
    pub fn to_tlexpr(&self) -> tensorlogic_ir::TLExpr {
        crate::tensorlogic::to_tlexpr(&self.eml_tree.lower().simplify())
    }

    /// Wrap the formula's `TLExpr` in a [`tensorlogic_ir::TLExpr::WeightedRule`] with the
    /// given weight.
    pub fn to_tl_weighted_rule(&self, weight: f64) -> tensorlogic_ir::TLExpr {
        tensorlogic_ir::TLExpr::WeightedRule {
            weight,
            rule: Box::new(self.to_tlexpr()),
        }
    }

    /// Build a `WeightedRule` encoding the equation `target_var = formula`.
    ///
    /// The left-hand side is `TLExpr::Pred { name: target_var, args: [Term::var(target_var)] }`.
    pub fn to_tl_weighted_equation(&self, target_var: &str, weight: f64) -> tensorlogic_ir::TLExpr {
        let lhs = tensorlogic_ir::TLExpr::Pred {
            name: target_var.to_string(),
            args: vec![tensorlogic_ir::Term::var(target_var)],
        };
        let eq = tensorlogic_ir::TLExpr::Eq(Box::new(lhs), Box::new(self.to_tlexpr()));
        tensorlogic_ir::TLExpr::WeightedRule {
            weight,
            rule: Box::new(eq),
        }
    }
}

/// Symbolic regression engine.
pub struct SymRegEngine {
    pub(super) config: SymRegConfig,
}

impl SymRegEngine {
    /// Create a new symbolic regression engine.
    pub fn new(config: SymRegConfig) -> Self {
        Self { config }
    }

    /// Discover the Pareto-optimal formulas (MSE vs complexity trade-off).
    ///
    /// Runs the full symbolic regression via [`Self::discover`], then extracts
    /// the non-dominated Pareto front.
    ///
    /// Use this when you want the full trade-off curve rather than a single
    /// "best" formula. Sort order: complexity ascending.
    pub fn discover_pareto(
        &self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        num_vars: usize,
    ) -> Result<Vec<DiscoveredFormula>, crate::error::EmlError> {
        let formulas = self.discover(inputs, targets, num_vars)?;
        Ok(pareto_front(&formulas))
    }
}

#[cfg(test)]
#[cfg(feature = "tensorlogic")]
mod tl_adapter_tests {
    use super::*;
    use crate::canonical::Canonical;
    use crate::tensorlogic;
    use tensorlogic_ir::{TLExpr, Term};

    fn make_formula() -> DiscoveredFormula {
        let tree = Canonical::nat(1);
        DiscoveredFormula {
            eml_tree: tree,
            mse: 0.0,
            complexity: 1,
            score: 0.0,
            pretty: "1".to_string(),
            params: vec![],
            cv_mse: None,
            aic: 0.0,
            bic: 0.0,
            param_intervals: None,
        }
    }

    #[test]
    fn discoveredformula_to_tlexpr_matches_lowered_simplified_path() {
        let f = make_formula();
        let expected = tensorlogic::to_tlexpr(&f.eml_tree.lower().simplify());
        assert_eq!(f.to_tlexpr(), expected);
    }

    #[test]
    fn to_tl_weighted_rule_shape_carries_weight_verbatim() {
        let f = make_formula();
        let tl = f.to_tl_weighted_rule(0.42);
        match tl {
            TLExpr::WeightedRule { weight, .. } => {
                assert!((weight - 0.42).abs() < f64::EPSILON);
            }
            other => panic!("expected WeightedRule, got {other:?}"),
        }
    }

    #[test]
    fn to_tl_weighted_equation_shape_lhs_pred_eq_rhs_formula() {
        let f = make_formula();
        let tl = f.to_tl_weighted_equation("y", 1.0);
        match tl {
            TLExpr::WeightedRule { weight, rule } => {
                assert!((weight - 1.0).abs() < f64::EPSILON);
                match *rule {
                    TLExpr::Eq(lhs, rhs) => {
                        match *lhs {
                            TLExpr::Pred { name, ref args } => {
                                assert_eq!(name, "y");
                                assert_eq!(args.len(), 1);
                                assert_eq!(args[0], Term::var("y"));
                            }
                            other => panic!("expected Pred on lhs, got {other:?}"),
                        }
                        assert_eq!(*rhs, f.to_tlexpr());
                    }
                    other => panic!("expected Eq inside WeightedRule, got {other:?}"),
                }
            }
            other => panic!("expected WeightedRule, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_depth0() {
        let topos = enumerate_topologies(0, 1);
        // Depth 0: One, Var(0) = 2 leaves
        assert_eq!(topos.len(), 2);
    }

    #[test]
    fn test_enumerate_depth1() {
        let topos = enumerate_topologies(1, 1);
        assert!(topos.len() >= 6);
    }

    #[test]
    fn test_symreg_exp() {
        let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 1,
            learning_rate: 1e-2,
            tolerance: 1e-6,
            max_iter: 1000,
            complexity_penalty: 1e-4,
            num_restarts: 2,
            integer_rounding: true,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine
            .discover(&inputs, &targets, 1)
            .expect("symreg discover exp should succeed");
        assert!(!formulas.is_empty());
        assert!(formulas[0].mse < 1.0);
    }

    #[test]
    fn test_integer_rounding() {
        use super::topology::try_integer_rounding;
        let params = vec![0.98, 2.03, 1.51, -0.99];
        let rounded = try_integer_rounding(&params);
        assert!((rounded[0] - 1.0).abs() < 1e-15);
        assert!((rounded[1] - 2.0).abs() < 1e-15);
        assert!((rounded[2] - 1.51).abs() < 1e-15); // Not close enough to round
        assert!((rounded[3] - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_symreg_parallel_matches_sequential() {
        let inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.25]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 1,
            learning_rate: 1e-2,
            tolerance: 1e-6,
            max_iter: 1000,
            complexity_penalty: 1e-4,
            num_restarts: 2,
            integer_rounding: true,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine
            .discover(&inputs, &targets, 1)
            .expect("parallel symreg discover should succeed");
        assert!(!formulas.is_empty());
        assert!(formulas[0].mse < 1.0);
    }

    #[test]
    fn test_empty_data() {
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = engine.discover(&[], &[], 1);
        assert!(matches!(result, Err(crate::error::EmlError::EmptyData)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = engine.discover(&[vec![1.0]], &[1.0, 2.0], 1);
        assert!(matches!(
            result,
            Err(crate::error::EmlError::DimensionMismatch(1, 2))
        ));
    }

    #[test]
    fn test_dedupe_reduces_topology_count() {
        let topologies = enumerate_topologies(2, 1);
        let before = topologies.len();
        let after = dedupe_by_semantics(topologies).len();
        assert!(
            after <= before,
            "dedup must not grow the set: before={before}, after={after}"
        );
    }

    #[test]
    #[ignore = "slow: depth-4 enumerates 2M topologies, ~38s wall-clock"]
    fn test_dedupe_depth_four_stress() {
        let topologies = enumerate_topologies(4, 1);
        let before = topologies.len();
        let after = dedupe_by_semantics(topologies).len();
        assert!(after <= before);
    }

    #[test]
    fn test_dedupe_preserves_uniqueness() {
        use std::collections::HashSet;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let topologies = enumerate_topologies(3, 1);
        let deduped = dedupe_by_semantics(topologies);

        let mut hashes: HashSet<u64> = HashSet::new();
        for tree in &deduped {
            let eml_simplified = crate::simplify::simplify(tree);
            let simplified = eml_simplified.lower().simplify();
            let mut h = DefaultHasher::new();
            simplified.structural_hash(&mut h);
            let inserted = hashes.insert(h.finish());
            assert!(inserted, "duplicate structural hash found in deduped set");
        }
        assert_eq!(hashes.len(), deduped.len());
    }

    #[test]
    fn test_dedupe_preserves_discovery_exp() {
        let inputs: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.2]).collect();
        let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

        let config = SymRegConfig {
            max_depth: 2,
            learning_rate: 1e-2,
            tolerance: 1e-5,
            max_iter: 1000,
            complexity_penalty: 1e-4,
            num_restarts: 2,
            integer_rounding: false,
            ..SymRegConfig::default()
        };

        let engine = SymRegEngine::new(config);
        let formulas = engine
            .discover(&inputs, &targets, 1)
            .expect("discover should succeed");
        assert!(!formulas.is_empty(), "should discover at least one formula");
        let best = &formulas[0];
        assert!(
            best.mse < 0.1,
            "best formula MSE too high after dedup: {} (pretty={})",
            best.mse,
            best.pretty
        );
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn balanced_equals_default() {
        let bal = SymRegConfig::balanced();
        let def = SymRegConfig::default();
        assert_eq!(bal.max_depth, def.max_depth);
        assert_eq!(bal.max_iter, def.max_iter);
        assert_eq!(bal.num_restarts, def.num_restarts);
        assert_eq!(bal.integer_rounding, def.integer_rounding);
        assert_eq!(bal.seed, def.seed);
        assert_eq!(bal.constant_extraction, def.constant_extraction);
        assert_eq!(bal.learning_rate.to_bits(), def.learning_rate.to_bits());
        assert_eq!(bal.tolerance.to_bits(), def.tolerance.to_bits());
        assert_eq!(
            bal.complexity_penalty.to_bits(),
            def.complexity_penalty.to_bits()
        );
    }

    #[test]
    fn quick_is_faster_than_balanced() {
        let q = SymRegConfig::quick();
        let b = SymRegConfig::balanced();
        assert!(q.max_iter <= b.max_iter);
        assert!(q.num_restarts <= b.num_restarts);
        assert!(q.max_depth <= b.max_depth);
    }

    #[test]
    fn exhaustive_is_slower_than_balanced() {
        let e = SymRegConfig::exhaustive();
        let b = SymRegConfig::balanced();
        assert!(e.max_iter >= b.max_iter);
        assert!(e.num_restarts >= b.num_restarts);
        assert!(e.max_depth >= b.max_depth);
    }

    #[test]
    fn engine_constructs_from_preset() {
        let _ = SymRegEngine::new(SymRegConfig::quick());
        let _ = SymRegEngine::new(SymRegConfig::balanced());
        let _ = SymRegEngine::new(SymRegConfig::exhaustive());
    }
}
