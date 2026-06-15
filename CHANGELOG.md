# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-06-15

### Added

**Automatic Differentiation (`src/autodiff.rs`)**

- `LoweredOp::jvp(vars, tangents) -> (f64, f64)` — forward-mode AD via dual numbers (JVP)
- `LoweredOp::vjp(vars) -> (f64, Vec<f64>)` — reverse-mode AD sweep (VJP / backprop)

**Special Mathematical Functions (`src/special.rs`)**

- Pure-Rust special functions with no external dependencies: `erf(x)`, `erfc(x)`, `lgamma(x)`, `digamma(x)`, `trigamma(x)`, `ei(x)` (exponential integral), `si(x)` (sine integral), `ci(x)` (cosine integral)
- New `LoweredOp` variants: `Erf`, `LGamma`, `Digamma`, `Trigamma`, `Ei`, `Si`, `Ci` — all lowerable, evaluable, and differentiable

**Symbolic Integration (`src/integrate.rs`, `src/integrate_subst.rs`)**

- `LoweredOp::integrate(wrt) -> IntegrateResult` — indefinite integral with closed-form rules; returns `IntegrateResult::Closed(expr)` or `IntegrateResult::Unsupported`
- `LoweredOp::integrate_definite(wrt, a, b, bindings) -> Result<f64, EmlError>` — definite integral: tries symbolic F(b)−F(a) first, falls back to adaptive Simpson quadrature
- `IntegrateResult` enum re-exported from crate root

**Limit Computation (`src/limit.rs`)**

- `LoweredOp::limit(wrt, point) -> LimitResult` — numeric probing with L'Hôpital's rule for 0/0 and ∞/∞ indeterminate forms
- `LimitPoint` enum (`Finite(f64)`, `PosInf`, `NegInf`) and `LimitResult` enum re-exported from crate root

**Symbolic ODE Solving (`src/ode.rs`)**

- `dsolve(eq, form) -> (OdeSolution, OdeKind)` — symbolic ODE solver recognising 5 families: separable, first-order linear, exact, Bernoulli, second-order constant-coefficient
- `OdeForm` — variable-slot assignments for x, y, y′, y″, and arbitrary constants
- `OdeSolution`, `OdeKind` enums re-exported from crate root

**Taylor/Maclaurin Series (`src/series.rs`)**

- Order-n Taylor and Maclaurin polynomial approximations via iterated symbolic differentiation

**Polynomial Algebra (`src/poly/`)**

- `Poly` — dense univariate polynomial over exact `Ratio<i64>` coefficients with GCD, pseudo-division, differentiation, evaluation, degree
- `Poly::sturm_sequence()`, `Poly::count_real_roots(a, b)`, `Poly::isolate_real_roots(lo, hi, tol)` — Sturm's theorem for certified real-root isolation
- `Poly::factor() -> Result<Factorization, PolyError>` — Yun's square-free decomposition + rational root theorem
- `MultiPoly` — sparse multivariate polynomial via `BTreeMap<Vec<u32>, Ratio<i64>>` with add, sub, mul, pow, GCD, `from_lowered`, `to_lowered`, `eval_f64`
- `PolyError`, `Factorization` enums/structs re-exported from crate root

**Dense Linear Algebra (`src/linalg/`)**

- `qr(a, m, n) -> Result<QrFactors, EmlError>`, `q_from_qr(factors) -> Vec<f64>` — Householder QR factorisation
- `svd(a, m, n) -> Result<SvdResult, EmlError>` — singular value decomposition
- `solve_spd_cholesky`, `solve_lu`, `solve_normal_equations` — linear system solvers
- `invert_spd(a, n) -> Result<Vec<f64>, EmlError>` — SPD matrix inversion
- `solve_least_squares(a, b, m, n)`, `pinv(a, m, n, rcond)` — least-squares and Moore-Penrose pseudoinverse
- `jtj`, `jtj_marquardt`, `jtr` — normal-equation builders for Levenberg-Marquardt

**Interval Arithmetic (`src/lower_interval.rs`)**

- `IntervalLO` — closed interval `[lo, hi]` with NaN sentinel for undefined results
- `LoweredOp::eval_interval(var_intervals) -> IntervalLO` — over-approximate interval evaluation through every op variant
- Re-exported from `crate::lower`

**Dimensional Analysis (`src/lower_units.rs`)**

- `LoweredOp::check_units(var_units) -> Result<Units, UnitError>` — checks dimensional consistency of expression trees against caller-supplied per-variable units

**Numeric Root-finding and Quadrature (`src/numeric.rs`)**

- `LoweredOp::find_root(var, bindings, x0)` and `::find_root_opts` — Newton–Brent root-finding
- `LoweredOp::find_roots_in(var, bindings, a, b, n_seeds)` — multi-root scan
- `LoweredOp::quadrature(wrt, a, b, bindings)` and `::quadrature_opts` — adaptive Simpson quadrature
- `lambert_w0(x)`, `lambert_wm1(x)` — Lambert W function branches (re-exported from crate root)
- `RootOpts`, `QuadOpts` config structs re-exported from crate root

**Verified Numerics (`src/numeric_verified.rs`)**

- `integrate_definite_verified(expr, wrt, a, b, bindings, opts)` — interval-arithmetic-certified definite integration
- `find_root_verified(expr, var, bindings, interval, opts) -> RootCertificate` — certified root with enclosure interval
- `RootCertificate`, `RootStatus`, `VerifiedQuadOpts` re-exported from crate root

**N-dimensional Quadrature (`src/quadrature_nd.rs`)**

- `quadrature_nd(expr, ranges, bindings, opts) -> Result<f64, EmlError>` — Gauss-Legendre N-dimensional quadrature
- `QuadNdOpts`, `QuadNdMethod` re-exported from crate root

**Higher-order and Mixed Derivatives (`src/lower_grad.rs`)**

- `LoweredOp::nth_derivative(wrt, n) -> Result<Self, EmlError>` — iterated symbolic differentiation
- `LoweredOp::mixed_partial(vars) -> Self` — mixed partial derivative ∂ⁿ/∂x₁∂x₂…

**Symbolic Regression Engine Expansions**

- `OptimizerKind` enum (`Adam`, `LM`) and `LmConfig` — Levenberg-Marquardt optimizer config alongside existing Adam
- `SelectionCriterion` enum — AIC/BIC/MDL-based model selection
- `SymRegConfig` new fields: `optimizer`, `lm`, `bootstrap_samples`, `confidence_level`, `uq_analytic`, `smt_prune`, `smt_prune_solver`, `selection`, `uq_top_k`, `enable_const_leaf`
- `SymRegConfig::with_units(var_units, target_units)` — dimensional-consistency filtering during search
- `SymRegEngine::discover_pareto(data)` — Pareto-front multi-objective search
- `SymRegEngine::discover_beam(data)` — beam search
- `SymRegEngine::discover_multi(data)` — multi-output shared-topology discovery
- `SymRegEngine::discover_ode(data, form)` / `discover_ode_sindy(data, form)` — ODE regression via SINDy
- `SymRegEngine::discover_exhaustive(data)` — exhaustive topology enumeration
- SMT-based pruning (`smt_prune.rs`) — prunes expressions whose bounds provably cannot fit data
- STRLSQ (`strlsq.rs`) — Sequential Thresholded Ridge Least-Squares sparse regression (`strlsq_qr`)
- Evolutionary search (`evolution.rs`) — genetic-algorithm topology mutation
- MCTS topology search (`mcts.rs`) — Monte-Carlo tree search over expression topologies
- SINDy (`sindy.rs`) — Sparse Identification of Nonlinear Dynamics
- PDE discovery (`pde.rs`) — partial differential equation discovery
- Uncertainty quantification (`uncertainty.rs`) — analytic interval propagation (`compute_analytic_intervals`) + bootstrap confidence intervals (`compute_bootstrap_intervals`, `inv_norm_cdf`)
- `SymRegLoss` enum — Huber loss and trimmed MSE as alternatives to squared error
- `snap_to_named_const(v)` — snap floating-point to named mathematical constants (π, e, φ, etc.)
- Pareto-front utilities: `dominates_by`, `pareto_front`, `pareto_front_ic`

**Python Bindings Expansion (`src/python/`)**

- `calculus.rs` — Python-callable `integrate`, `limit`, `series_expansion`, `dsolve`
- `numeric.rs` — Python-callable `find_root`, `quadrature`, `lambert_w0`
- `solve.rs` — Python-callable `solve_for`, `solve_system`
- `symreg.rs` — expanded Python bindings for the symbolic regression engine

### Changed

- `num-rational` promoted from optional dependency (behind `smt` feature) to required; `Ratio<i64>` is now always available for polynomial algebra
- `smt` feature no longer requires `num-rational` (removed from feature deps)
- Dependency upgrades: `oxiz` 0.2.1 → 0.2.3, `tensorlogic-ir` 0.1.0 → 0.1.1, `scirs2-core` 0.4.3 → 0.5.0, `pyo3` 0.28.3 → 0.29.0, `numpy` 0.28.0 → 0.29.0, `oxicode` 0.2.2 → 0.2.4, `wasm-bindgen` 0.2.118 → 0.2.125, `cranelift-*` 0.131 → 0.132.1
- `src/symreg/mod.rs` (1541 lines) split into focused submodules: `constants.rs`, `cv.rs`, `discover.rs`, `discover_multi.rs`, `discover_shared.rs`, `evolution.rs`, `loss.rs`, `mcts.rs`, `numerics.rs`, `optimize_lm.rs`, `pareto.rs`, `pde.rs`, `post_round.rs`, `sindy.rs`, `smt_prune.rs`, `strlsq.rs`, `topology.rs`, `uncertainty.rs`. Public API unchanged.
- `src/lower.rs` (1233 lines) split into `src/lower/` directory module: `mod.rs`, `pattern.rs`, `oxiblas.rs`, `display.rs`. Public API unchanged.
- `src/bin/oxieml.rs` (1169 lines) split into slim entry point plus 7 submodules under `src/bin/oxieml/`: `args.rs`, `evaluate.rs`, `format.rs`, `generate.rs`, `grad.rs`, `lower.rs`, `symreg.rs`. CLI behaviour unchanged.
- `src/smt.rs` (1024 lines) split into `src/smt/` with `constraint.rs`, `interval.rs`, `nra.rs`, `oxiz_backend.rs`, `helpers.rs`, `tests.rs`, `smt_tests.rs`. Public API preserved.
- Corrected stale `[0.1.1]` changelog wording for `canonical_rewrite_rules()`: it returns 10 real `RewriteRule` instances, not a placeholder `vec![]`.

### Performance

- Adam optimiser loop in `src/symreg/discover.rs`: per-iteration buffers hoisted above the loop and reused via `clear()`/refill — eliminates dominant allocator-pressure on hot symbolic-regression paths
- `LoweredOp` recursive-enum variants now store children as `Arc<LoweredOp>` instead of `Box<LoweredOp>` — hot-path AST clones in `grad`, `grad_all`, `jacobian`, `hessian`, `simplify` drop from O(|subtree|) deep clone to O(1) atomic reference-count bump
- `LoweredOp::cse(&self) -> Arc<LoweredOp>` hash-consing pass: converts expression tree into maximally-shared DAG via dual-map `CseInterner` (pointer-identity memo + structural-hash intern table)
- `eval_real_lowered` routes through `.cse()` → `to_oxiblas_ops_shared()` — sharing-aware stack machine exploits repeated subexpressions
- `LoweredOp::grad(wrt)` applies `.cse()` to simplified gradient; removes redundant second `.simplify()` previously added by `grad_all`/`hessian`

### Breaking Changes

- External code constructing `LoweredOp` variants directly (e.g. `LoweredOp::Add(Box::new(...), Box::new(...))`) must switch inner children from `Box::new(...)` to `Arc::new(...)`. Code using `canonical::*` constructor helpers is unaffected.
- `OxiOp` gains two new variants: `Store(usize)` (peek top of stack into slot k, does NOT pop) and `Load(usize)` (push cached value from slot k). Exhaustive matches must add arms for these variants.

### Internal

- `ParameterizedEmlTree::forward_with_jacobian_into(&self, ctx, jac_out: &mut Vec<f64>)` in `src/grad.rs` — callers may supply a pre-allocated Jacobian buffer; the existing `forward_with_jacobian` is a thin wrapper
- `LoweredOp::to_oxiblas_ops_shared() -> (Vec<OxiOp>, usize)` — two-pass sharing-aware emitter; pure trees produce byte-identical output to `to_oxiblas_ops()` with `n_slots == 0`
- `LoweredOp::eval_ops_shared(ops, vars, n_slots)` — scalar evaluator with `slots: Vec<f64>` register file for `Store`/`Load`
- SIMD path (`src/simd_eval.rs`): derives `n_slots` by scanning for max `Store(k)|Load(k)` index; runs parallel `slots: Vec<SimdVec>` register file
- JIT path (`src/jit.rs`): `Store(k)` peeks top SSA `Value` into `slot_values: HashMap<usize, Value>`; `Load(k)` clones it back
- `pop_or_nan` helper in `src/lower/oxiblas.rs` wraps `debug_assert!(!stack.is_empty())` around pop — production semantics unchanged; debug builds panic with actionable message
- New test files: `tests/lower_cse_test.rs` (8 tests), `tests/oxiblas_shared_test.rs` (7 tests)

### Dependencies

- `oxiz`: 0.2.1 → 0.2.3
- `tensorlogic-ir`: 0.1.0 → 0.1.1
- `scirs2-core`: 0.4.3 → 0.5.0
- `pyo3`: 0.28.3 → 0.29.0
- `numpy`: 0.28.0 → 0.29.0
- `oxicode`: 0.2.2 → 0.2.4
- `wasm-bindgen`: 0.2.118 → 0.2.125
- `cranelift-*`: 0.131 → 0.132.1

## [0.1.1] - 2026-05-03

### Added

- Trig precision: `src/lower.rs` now detects the canonical `Canonical::sin(x)`
  and `Canonical::cos(x)` EML tree shapes and lowers them directly to
  `LoweredOp::Sin(x)` / `LoweredOp::Cos(x)`, giving 0.0 error vs `f64::sin`/
  `f64::cos` on canonical trig trees.
- `EmlTree::eval_real_lowered(&EvalCtx) -> Result<f64, EmlError>` convenience
  that routes through lowering → pattern recognition → `LoweredOp::Sin`/`Cos`
  → `f64::sin`/`cos` precision.
- `LoweredOp::grad(wrt: usize) -> LoweredOp` symbolic differentiation with
  chain, product, quotient, and general `Pow(base, expo)` (via exp-log) rules.
  Result is `.simplify()`'d for clean pretty-printing. 13 cross-check tests
  against central-difference numerical derivatives in
  `tests/lowered_grad_test.rs`.
- Optional `tensorlogic` feature gated on `tensorlogic-ir`. Exposes
  `to_tlexpr(&LoweredOp) -> TLExpr`,
  `from_tlexpr(&TLExpr) -> Result<LoweredOp, EmlError>`, and
  `canonical_simplify(&TLExpr) -> TLExpr` (algebraic identities + const
  folding). `Neg(x)` lowers to `Sub(Const(0), x)` since `TLExpr` has no unary
  negation. `canonical_rewrite_rules()` now returns 10 `RewriteRule` instances
  covering exp/log inverses, double negation, and identity elements (`0+x`,
  `x*1`, `x/1`, `x^0`, `x^1`).
- `EmlError::UnsupportedTlExpr(String)` variant (feature-gated) for
  non-arithmetic `TLExpr` nodes encountered during conversion.
- CLI: `--grad`/`-d <wrt>` subcommand in `src/bin/oxieml.rs` that parses an
  EML expression, lowers it, and prints both the expression and its symbolic
  derivative with respect to `x{wrt}`.
- Symbolic regression examples: `examples/physics_pipeline.rs` (projectile
  motion, 3-var end-to-end symreg → compile → batch eval),
  `examples/pendulum.rs` (T = 2π√(L/g), 1-var), and
  `examples/harmonic_oscillator.rs` (x(t) = A·cos(ωt), 3-var).
- `benches/trig_bench.rs` criterion benchmarks comparing `eval_real` (raw
  tree walk) vs `eval_real_lowered` (lowered stack-machine) for sin/cos/exp/
  composite on 1000 points each.
- 13 symbolic-gradient tests in `tests/lowered_grad_test.rs`, 13 trig
  precision tests in `tests/trig_precision_test.rs`, and 8 TensorLogic
  bridge tests in `tests/tensorlogic_test.rs`.
- CLI `--symreg` / `-s` subcommand: runs `SymRegEngine::discover` on
  whitespace-separated data from stdin or `--file` and prints top-K
  ranked formulas with MSE, complexity, and score. Supports forwarding
  flags for every `SymRegConfig` field (`--max-depth`, `--max-iter`,
  `--learning-rate`, `--tolerance`, `--complexity-penalty`,
  `--num-restarts`) plus `--vars` and `--top`. Integration tests in
  `tests/cli_symreg_test.rs` via `assert_cmd` + `predicates`
  (new `[dev-dependencies]`).
- `SymRegConfig::quick()`, `::balanced()`, `::exhaustive()` ergonomic
  preset constructors. `balanced()` aliases `Default::default()`;
  `quick()` shortens search for fast iteration; `exhaustive()` deepens
  for publication-quality runs.

### Changed

- Bumped version 0.1.0 → 0.1.1 (branch-driven).
- `LoweredOp::simplify` now handles Sin/Cos/Exp/Ln constant folding (guarded
  for finite ln arg and positive domain), `Mul(Const(-1), x) → Neg(x)`,
  `Add(a, Neg(b)) → Sub(a, b)`, `Sub(a, Neg(b)) → Add(a, b)`,
  `Neg(Sub(a, b)) → Sub(b, a)`, and guards `Pow(Const(b), Const(e))` folding
  against non-finite results.
- `src/bin/oxieml.rs` now dispatches to `run_grad()` when `--grad`/`-d` is
  supplied; default path remains the evaluator.

### Internal

- `EvalCtx::as_slice() -> &[f64]` accessor for SIMD / batch evaluators.

### Notes

- No SciRS2 integration in this release; deferred pending OxiEML's final
  crate-placement decision.

## [0.1.0] - 2026-04-14

### Added
- EML operator `eml(x, y) = exp(x) - ln(y)` implementation
- Uniform binary tree representation for all elementary functions
- Tree evaluation: real (`eval_real`) and complex (`eval_complex`) modes
- Batch evaluation with optional SIMD acceleration (`simd` feature)
- Parallel batch evaluation (`parallel` feature via rayon)
- Expression simplification and normalization
- Canonical form derivations (exp, ln, neg, add, sub, mul, div, pow, sqrt, abs, trig, hyperbolic, inverse trig/hyperbolic)
- Symbolic differentiation (gradient computation)
- Expression compiler with lowered IR for fast evaluation
- S-expression parser supporting `E(...)` and `eml(...)` notation
- SMT constraint solving via OxiZ (`smt` feature)
- Symbolic regression engine for discovering EML formulas from data
- CLI tool (`oxieml`) with eval, simplify, lower, grad, parse, symreg, and smt commands
- Comprehensive test suite (173 tests)
- Criterion benchmarks for evaluation and symbolic regression

[0.1.2]: https://github.com/cool-japan/oxieml/releases/tag/v0.1.2
