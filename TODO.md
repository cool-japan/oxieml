# OxiEML TODO

## Phylogenetic Tree (Paper Figure 1) — Canonical Constructions

All functions from the paper's phylogenetic tree are implemented in `src/canonical.rs`:

- [x] **Core**: `eml`, `1`
- [x] **Basic**: `exp`, `ln`, `e` (Euler's number)
- [x] **Arithmetic**: `add`, `sub`, `mul`, `div`, `neg`
- [x] **Powers**: `pow`, `square`, `sqrt`, `reciprocal`, `abs`
- [x] **Trig**: `sin`, `cos`, `tan`
- [x] **Inverse trig**: `arcsin`, `arccos`, `arctan` (via complex logarithms)
- [x] **Hyperbolic**: `sinh`, `cosh`, `tanh`
- [x] **Inverse hyperbolic**: `arcsinh`, `arccosh`, `arctanh`
- [x] **Constants**: `pi` (iπ), `zero` (= ln(1)), `neg_one`, `neg_two`, `imag_unit`, `nat(n)`

## CLI Tool

- [x] **`src/parser.rs`** — recursive descent parser for `E(x,y)` / `eml(x,y)` notation.
- [x] **`src/bin/oxieml.rs`** — CLI evaluator with constant matching, complex eval, lowering; `--help`/`-V` flags.
- [x] **End-to-end verification** — user's 193-node depth-34 EML expression evaluates correctly to π.

## Code Quality

- [x] **Clean canonical doc comments** — replaced 250+ lines of derivation scratch-work with concise per-function docs in `src/canonical.rs`.
- [x] **Codegen / grad cleanup** — fixed trailing space in `Neg` codegen (`src/compile.rs`); pruned unused tape indices and prefixed unused locals in `src/grad.rs`.

## Functionality Implemented

- [x] **Real `simplify`** — `ln(exp(x))→x`, `exp(ln(x))→x`, structural-hash CSE/dedup (`src/simplify.rs`).
- [x] **Lower-pattern recognition** — subtraction `eml(ln(x), eml(y, 1)) → x − y`, exp-of-ln elimination, ln structural matching (`src/lower.rs`).
- [x] **Disjoint topology enumeration** — symreg generates each EML topology exactly once via three-case split (both/left/right at max depth) in `src/symreg.rs`.
- [x] **`canonical::zero()`** — added `0 = ln(1)` constructor.

## Lowered IR & Evaluation

- [x] **`LoweredOp::to_oxiblas_ops()` flat IR** — post-order `OxiOp` enum consumed by scalar/SIMD evaluator (`src/lower.rs`).
- [x] **`simd` feature** — real SIMD via `oxiblas-core` 0.2.1, runtime aarch64/x86_64 dispatch to `F64x2`/`F64x4`; combines with `parallel` for SIMD-per-worker (`src/simd_eval.rs`).
- [x] **`parallel` feature** — rayon `par_iter` batch eval, threshold 128 (scalar) / 512 (SIMD).
- [x] **Sin/cos precision** — `lower.rs` pattern-matches canonical sin/cos shapes → `LoweredOp::Sin`/`Cos`; `eval_real_lowered()` gives true `f64::sin` precision (0.0 vs ~1e-14 tree-walk error).

## Symbolic Tooling

- [x] **`LoweredOp::grad(wrt)`** — chain/product/quotient/`Pow`-via-exp-log rules, returns simplified `LoweredOp`; 13 tests cross-check against central differences (`src/lower.rs`).
- [x] **LaTeX export** — `LoweredOp::to_latex()` and `DiscoveredFormula::to_latex()` (π/e detection, `\frac`, `e^{·}`, `x_{i}`, etc.).
- [x] **`compile_to_rust` / `compile_to_closure` / `compile_to_rust_batch`** — codegen including parallel `_batch` form (`src/compile.rs`).

## SMT / Constraint Solving (`smt` feature)

- [x] **`IntervalDomain`** — always-on forward exp/ln propagation with conflict detection.
- [x] **`EmlSmtSolver`** — OxiZ 0.2 LRA backend via secant + tangent linear relaxation; proves UNSAT on cases interval bisection alone misses (e.g. `ln(x) > 0` on negative domain).
- [x] **`EmlNraSolver`** — interval-bisection fallback; constraints: `EqZero`, `GtZero`, `GeZero`, `And`, `Or`.

## Symbolic Regression

- [x] **`SymRegEngine::discover` / `discover_pareto`** — Adam optimizer, k-fold CV (`SymRegConfig.cv_folds`), Pareto front, parallel topology eval, depth-limited enumeration.
- [x] **Pareto-front API** — `discover_pareto()`, free `pareto_front(&[DiscoveredFormula])`, `DiscoveredFormula::dominates(&other)`.
- [x] **K-fold cross-validation** — deterministic shuffle, `cv_mse: Option<f64>`, no-param topologies skip CV.
- [x] **Presets** — `SymRegConfig::{quick, balanced, exhaustive}()`.
- [x] **Pruning** — `dedupe_by_semantics` via `lower().simplify().structural_hash`; correct but limited (~0.0002% reduction at depth 4 because EML is non-commutative).
- [x] **CLI `--symreg`** — stdin / `--file` data, top-K ranked output, every `SymRegConfig` field exposed; 4 integration tests via `assert_cmd` + `predicates`.

## Examples & Benchmarks

- [x] **`examples/physics_pipeline.rs`** — projectile-motion data → `discover` → lower → `compile_to_rust_batch` → batch eval on held-out set.
- [x] **`examples/pendulum.rs`** — 1-var, T = 2π√(L/g).
- [x] **`examples/harmonic_oscillator.rs`** — 3-var, x(t) = A·cos(ωt).
- [x] **`benches/eval_bench.rs`, `benches/trig_bench.rs`** — criterion comparisons of `eval_real` vs `eval_real_lowered` over sin/cos/exp/composite, 1000 points each.

## SciRS2 Adapter (`scirs2` feature)

- [x] **`src/scirs2.rs`** — `symbolic_regression(Array2, Array1, config)` and `_with_names` variant; row-major conversion under the hood; feature-gated optional `scirs2-core` dep.

## TensorLogic Integration (`tensorlogic` feature)

EML's uniform rewriting and TensorLogic's logic-to-tensor compilation are natural counterparts: OxiEML discovers closed-form formulas from data; TensorLogic compiles logical rules into einsum graphs for neurosymbolic AI. Connecting them gives a **data-driven formula discovery → neurosymbolic prior** pipeline.

**Dependency strategy (cycle-safe):** OxiEML may become a SciRS2 subcrate; TensorLogic's execution layer (`tensorlogic-scirs-backend`, `tensorlogic-train`) depends on SciRS2. To avoid cycles, OxiEML depends **only** on `tensorlogic-ir` — the engine-agnostic AST/IR layer with **zero SciRS2 dependencies** (verified: serde, serde_json, oxicode, chrono, thiserror only).

```
SciRS2 ─may contain→ OxiEML ─optional→ tensorlogic-ir  (no SciRS2 dep)
                                              │
TensorLogic ─depends→ SciRS2    tensorlogic-ir is SciRS2-free ✓
```

No cycle. The `tensorlogic-compiler` and `tensorlogic-adapters` crates are also SciRS2-free and may be used if needed.

- [x] **`to_tlexpr` / `from_tlexpr`** — bidirectional `LoweredOp ↔ TLExpr` mapping for the arithmetic/transcendental subset; `Neg` encoded as `Sub(0, x)`; logic-only `TLExpr` nodes return `EmlError::UnsupportedTlExpr` (`src/tensorlogic.rs`).
- [x] **`canonical_rewrite_rules()`** — 10 real `RewriteRule` instances over `tensorlogic_ir::Pattern` (exp/log inverses, double negation, identity elements `0+x`, `x*1`, `x/1`, `x^0`, `x^1`).
- [x] **Soft-prior export** — `DiscoveredFormula::{to_tlexpr, to_tl_weighted_rule, to_tl_weighted_equation}` + free `formulas_to_tl_weighted_rules`; reuses the same `lower().simplify()` chain as `to_latex` so the printed pretty form, LaTeX, and TLExpr stay in lock-step (9 tests, `src/tensorlogic.rs`, `src/symreg.rs`, `tests/tensorlogic_test.rs`).
- [x] **Core path stays in OxiEML** — tree eval, OxiOp stack machine, `simd_eval` have no dependency on `tensorlogic-compiler`, `tensorlogic-scirs-backend`, or `tensorlogic-train` in any feature set (verified 2026-04-15).

---

## Completed in v0.1.1 (2026-05-03)

- [x] **Extended LoweredOp variants for transcendentals** (implemented 2026-04-27) — add `Tan`, `Sinh`, `Cosh`, `Tanh`, `Arcsin`, `Arccos`, `Arctan`, `Arcsinh`, `Arccosh`, `Arctanh` and recognize their canonical EML shapes during lowering. `[medium]`
  - **Why:** Today these functions exist as `Canonical` constructions but lower to raw `Exp`/`Ln` forests, blowing up node counts, hurting `to_latex` / `to_pretty` readability, and forcing `grad` to differentiate through the desugaring instead of using the closed-form derivative. This is the single biggest gap in lowered-IR expressiveness.
  - **Design:** Extend the `LoweredOp` enum (one variant per function); update every consumer — `eval`, `eval_batch`, `simplify` (idempotent shape-preserving rules: `tanh(arctanh(x))→x`, etc.), `grad` (closed-form derivatives, e.g. `d/dx tanh(x) = 1 − tanh²(x)`), `to_latex`, `to_pretty`, `to_oxiblas_ops` (new `OxiOp` variants), `structural_hash`. Add canonical-shape pattern recognizers in `src/lower.rs` mirroring the existing sin/cos detectors. Provide a feature-flag-free fallback path: if the IR/SIMD backend doesn't yet implement a variant, expand inline to the existing exp/ln equivalent so behavior is preserved.
  - **Files:** `src/lower.rs`, `src/eval.rs`, `src/simplify.rs`, `src/compile.rs`, `src/simd_eval.rs`, `src/tensorlogic.rs`.
  - **Tests:** Round-trip canonical→lower→pretty assertions, grad central-difference cross-checks for each new variant, SIMD scalar/vector parity, LaTeX golden strings.
  - **Risk:** Variant explosion in matchers — every `match LoweredOp { ... }` gains 10 arms; missing one in a non-exhaustive site silently regresses. Mitigation: keep the enum non-`#[non_exhaustive]` so the compiler enforces exhaustiveness.

- [x] **Serde serialization for EmlTree / LoweredOp / DiscoveredFormula** (implemented 2026-04-27) — feature-gated `serde` support so formulas survive disk and process boundaries. `[medium]`
  - **Why:** Symbolic-regression runs are expensive; saving the Pareto front to JSON / oxicode is a research-workflow basic. Also unblocks Python bindings (cross-language transport) and reproducible experiment artifacts.
  - **Design:** Add optional `serde = "1"` and (for binary) `oxicode` deps gated on `serde`. Derive `Serialize` / `Deserialize` on `EmlTree`, `EmlNode`, `LoweredOp`, `DiscoveredFormula`, `SymRegConfig`. Use `#[serde(rename_all = "snake_case")]` and an explicit version tag (`#[serde(tag = "v")]`) to keep file format upgradable. Ship `EmlTree::to_json` / `from_json` convenience methods; binary path uses oxicode (NOT bincode — COOLJAPAN policy).
  - **Files:** `Cargo.toml`, `src/tree.rs`, `src/lower.rs`, `src/symreg.rs`, `src/lib.rs`, new `tests/serde_test.rs`.
  - **Tests:** Round-trip equality for each type (deep-nested tree, every `LoweredOp` variant, `DiscoveredFormula` with `cv_mse: Some` and `None`); golden JSON snapshot to detect accidental schema breaks.
  - **Risk:** Schema lock-in — once published, field renames are breaking. Mitigation: explicit version tag from day one, `#[serde(default)]` on additive fields.

- [x] **Constant folding in `simplify`** (implemented 2026-04-27) — fold `Const(a)` op `Const(b)` → `Const(a op b)` for `Add`/`Sub`/`Mul`/`Div`/`Pow`/`Exp`/`Ln`/`Neg` (and the new transcendentals). `[small]`
  - **Why:** Surprisingly absent today: `Add(Const(2.0), Const(3.0))` survives `simplify`. After the canonical-construction path runs, lowered trees often contain folded subexpressions that should collapse to a single constant before printing or grad. Free correctness/readability win.
  - **Files:** `src/simplify.rs`. **Tests:** golden strings for each fold; assert idempotence (`simplify(simplify(x)) == simplify(x)`). **Risk:** NaN/Inf handling — folding `Ln(Const(-1.0))` must produce a non-finite that doesn't poison surrounding logic.

- [x] **`LoweredOp::jacobian(n_vars)` and `LoweredOp::grad_all()`** (implemented 2026-04-27) — convenience + shared-subexpression batch gradient. `[medium]`
  - **Why:** Calling `grad(wrt)` in a loop recomputes shared subexpressions n times. Reverse-mode-style sharing is strictly faster and matches research-library expectations. Jacobian is the obvious wrapper.
  - **Design:** `pub fn jacobian(&self, n_vars: usize) -> Vec<LoweredOp>` is a thin wrapper that calls `grad_all`. `grad_all` performs one structural pass building a CSE-aware adjoint table keyed by structural hash, returning `Vec<LoweredOp>` of length `n_vars`. Reuse the existing `simplify` cache to deduplicate identical adjoint expressions across outputs. Document complexity: O(|tree|·n) worst case, often much better with CSE.
  - **Files:** `src/lower.rs`. **Tests:** Cross-check each Jacobian column against `grad(i)` and against finite differences for random expressions; assert `grad_all` is faster than the loop on a 100-node tree (perf smoke test, not strict bench). **Risk:** Subtle numerical drift if simplification orders differ from per-call `grad` — pin it via golden tests.

- [x] **Symbolic Hessian** (implemented 2026-04-27) — `LoweredOp::hessian(n_vars) -> Vec<Vec<LoweredOp>>` for second-order derivatives. `[small]`
  - **Why:** Newton-style optimization, curvature-based model selection, and physics applications all consume Hessians. Cheap to add once `jacobian` exists.
  - **Design:** `hessian` = jacobian of jacobian, exploiting Schwarz symmetry (`H[i][j] == H[j][i]`) to compute only the upper triangle and mirror. Each entry simplified.
  - **Files:** `src/lower.rs`. **Tests:** Symmetry check, central-difference cross-check on quadratic and trig benchmarks. **Risk:** Tree-size blow-up on deep expressions; document the O(n²·|tree|) growth and recommend `simplify` after each entry.

- [x] **Interval arithmetic on `LoweredOp`** (implemented 2026-04-27) — `LoweredOp::eval_interval(&[Interval]) -> Interval` for over-box evaluation. `[medium]`
  - **Why:** Today's `IntervalDomain` lives in `smt.rs` and operates on the EML tree level. Lifting it to `LoweredOp` (where named ops have tight monotonicity/convexity properties) tightens bounds substantially and unlocks reliable range analysis for the symbolic-regression scoring path (e.g. reject candidates whose output range cannot contain target observations).
  - **Design:** New `Interval { lo: f64, hi: f64 }` lightweight struct (or reuse the one in `smt.rs` if shape allows). Standard rounding-aware interval rules per op; transcendentals use monotone-region splits (e.g. `sin` over a box that crosses π/2). Generalize `IntervalDomain` so both consumers share the underlying ops.
  - **Files:** `src/lower.rs`, `src/smt.rs` (refactor shared), new `tests/interval_test.rs`.
  - **Tests:** Containment property (point eval ∈ interval eval) for random inputs; tight-bound check on monotone functions; SMT/interval cross-validation that the lowered-IR interval is no looser than the tree-level one.
  - **Risk:** Rounding-mode portability — `f64` rounding control is platform-flavoured; document that we use directed rounding only where available and conservative widening elsewhere.

- [x] **Noise-robust loss functions for symreg** (implemented 2026-04-27) — Huber and trimmed MSE alongside the current MSE objective. `[medium]`
  - **Why:** Real-world observational data has outliers. MSE alone gives outliers quadratic leverage and biases topology selection toward formulas that explain the noise. Huber and α-trimmed MSE are the standard fixes and don't require user-supplied noise models.
  - **Design:** New `enum SymRegLoss { Mse, Huber { delta: f64 }, TrimmedMse { alpha: f64 } }` exposed via `SymRegConfig.loss`. Adam optimizer differentiates Huber analytically (piecewise quadratic / linear); trimmed MSE drops the top `α·n` residuals before averaging. Pareto front uses the same configured loss for `mse` field but renames it to `loss` in the public API; keep an `mse` alias for one minor.
  - **Files:** `src/symreg.rs`, CLI flag `--loss huber:0.1` etc. in `src/bin/oxieml.rs`.
  - **Tests:** Synthetic data with a 10% outlier contamination; Huber/trimmed should recover the underlying formula at lower complexity than MSE.
  - **Risk:** Adam convergence with non-smooth trimmed loss can stall — use a smooth approximation (e.g. soft-trim via sigmoid weight) for the gradient step and the exact form for scoring.

- [x] **Constants extraction post-Adam** (implemented 2026-04-27) — round optimized scalars to π / e / simple rationals when MSE doesn't worsen by ε. `[medium]`
  - **Why:** Adam returns `2.99998…` for what is plainly `3` and `3.14159…` for what is plainly `π`. Reporting raw floats in published formulas is ugly and obscures the discovery. This is a high-perceived-quality win for almost no cost.
  - **Design:** After Adam termination, for each free constant try a candidate set `{0, ±1, ±1/2, ±1/3, ±1/4, ±π, ±e, ±√2, …, simple rationals via Stern-Brocot up to denominator 12}`. Accept the rounded value if the resulting MSE on the training set is within `(1 + ε)·current_mse` (default ε = 1e-3, configurable). Iterate constants left-to-right (greedy) — full combinatorial search is exponential. Mark rounded constants so `to_latex` can render `\pi` / `\frac{1}{2}` symbolically.
  - **Files:** `src/symreg.rs`, `src/lower.rs` (named-constant marker on `Const`).
  - **Tests:** Pendulum example should report `T = 2π√(L/g)` not `T = 6.2831·√(L/g)`; MSE-tolerance test ensures we never accept a worse fit.
  - **Risk:** Greedy rounding can get stuck — document and provide `--no-constant-rounding` escape hatch.

- [x] **Beam search topology exploration** (implemented 2026-04-27) — replace exhaustive enumeration with bounded beam search at depths > 4. `[medium]`
  - **Why:** Exhaustive enumeration is the right default at depth ≤ 4 (that's where it terminates in seconds) but is intractable at depth 5+ where most physically interesting formulas live. Beam search gives a tunable depth-vs-breadth tradeoff that's deterministic, parallel-friendly, and well-understood.
  - **Design:** New `enum SymRegStrategy { Exhaustive, Beam { width: usize } }` in `SymRegConfig`. Beam: at each depth d, score every candidate topology by a cheap surrogate (e.g. lowered-node-count + pre-fit residual after 5 Adam steps), keep top-`width`, expand only those. Use `IntervalDomain::eval_interval` to drop candidates whose output range can't span the target range. Beam runs inside the same parallel `par_iter` skeleton as exhaustive.
  - **Files:** `src/symreg.rs`, CLI `--strategy beam:64`.
  - **Tests:** Beam at width = ∞ matches exhaustive; beam at width = 1 is greedy and deterministic; depth-6 beam terminates within budget on Pendulum example.
  - **Risk:** Surrogate misranks — a topology that fits poorly at 5 Adam steps may shine at 200. Mitigation: warm-start the survivors longer in the final pass.

- [x] **RNG seed in `SymRegConfig`** (implemented 2026-04-27) — explicit `seed: Option<u64>` for fully reproducible runs. `[trivial]`
  - **Why:** Adam initialization, k-fold shuffle, and any future stochastic strategy all currently draw from `rand::thread_rng()`. Published research results must be reproducible — without a seed they aren't.
  - **Files:** `src/symreg.rs`. **Tests:** Two `discover()` calls with same seed produce byte-identical Pareto fronts. **Risk:** Parallel determinism with `rayon` — must use seeded per-topology RNGs derived from the master seed (e.g. `SplitMix64`), not a single shared RNG.

- [x] **CLI `--format` (pretty / latex / json) and `--output <file>`** (implemented 2026-04-27) — structured output for piping into LaTeX docs, notebooks, or downstream tools. `[small]`
  - **Why:** Today the CLI prints a fixed human-readable form. Researchers want LaTeX for papers, JSON for scripting, and writing to a file is necessary on Windows where `>` shell redirection is awkward.
  - **Design:** Add `--format pretty|latex|json` (default `pretty`) and `--output <path>`; JSON requires the `serde` feature. Apply uniformly to `--eval`, `--lower`, and `--symreg` subcommands.
  - **Files:** `src/bin/oxieml.rs`, `tests/cli_format_test.rs`. **Tests:** Each format on a known formula matches a golden string. **Risk:** None significant.

- [x] **Property-based grad tests via proptest** (implemented 2026-04-27) — random valid trees compared against central differences. `[small]`
  - **Why:** Existing 13 grad tests are hand-picked. proptest will surface adversarial expressions (near-singularities, deep nesting, mixed unary/binary chains) the suite doesn't cover.
  - **Design:** `proptest` strategy generating `LoweredOp` trees up to depth 6 with bounded constant range; for each, sample 5 random points, compare `grad(i).eval(point)` vs central difference, tolerance `max(1e-5·|expected|, 1e-7)`. Skip points where the function is non-differentiable (e.g. `|x|` at 0). Add to `tests/`.
  - **Files:** `Cargo.toml` (`proptest` dev-dep), `tests/grad_proptest.rs`.
  - **Risk:** Flaky tests due to numerical edge cases — pin `cases = 1024`, persist failed seeds, document tolerance.

- [x] **Multi-output symbolic regression** (implemented 2026-04-27) — `discover_multi(features, targets: Array2)` returning vector-valued formulas. `[medium]`
  - **Why:** Many physics problems are multi-output (position + velocity, Lorenz system, etc.). The scalar-output API forces users into N independent runs that miss shared structure.
  - **Design:** Add `SymRegEngine::discover_multi(features, targets) -> Vec<Vec<DiscoveredFormula>>`. Two strategies behind `SymRegConfig.multi_output`: (a) **independent** — N parallel scalar runs (cheap, no sharing); (b) **shared-topology** — co-evolve topologies and have each output use its own constants only (forces a common functional skeleton, useful when outputs are physically related). Pareto front per output.
  - **Files:** `src/symreg.rs`, `src/scirs2.rs` (ndarray adapter).
  - **Tests:** Synthetic Lorenz dataset; independent matches three single-output runs; shared-topology recovers the structural similarity.
  - **Risk:** Shared-topology mode dramatically raises search cost — keep `independent` as default.

- [x] **Python bindings via PyO3** (implemented 2026-04-27) (feature `python`) — expose `EmlTree`, `LoweredOp`, `SymRegEngine` to Python with maturin packaging. `[large]`
  - **Why:** Python is the lingua franca of scientific computing; without it, OxiEML is invisible to the SciRS2 + NumRS2 user base and to the broader symbolic-regression community (PySR, gplearn, etc.). High distribution leverage.
  - **Design:** New `python` feature, `Cargo.toml` adds `pyo3` (latest, `extension-module` feature) under that gate. New `src/python.rs` exporting `PyEmlTree`, `PyLoweredOp`, `PySymRegEngine`, `PySymRegConfig`, `PyDiscoveredFormula`. Use `numpy` crate for `ndarray::Array2 ↔ numpy.ndarray` zero-copy interop. Ship `pyproject.toml` + `pypi-publish.yml` (allowed by COOLJAPAN policy). Wheels built via maturin on `manylinux2014`, `macos-arm64`, `windows`. No GIL release on the hot loop initially — measure first.
  - **Files:** `Cargo.toml`, `src/python.rs`, `pyproject.toml`, `.github/workflows/pypi-publish.yml`, `python/oxieml/__init__.py`, `python/tests/test_basic.py`.
  - **Tests:** Python-side pytest suite mirroring the Rust integration tests; CI builds wheels for all three platforms.
  - **Risk:** Build-system complexity (maturin + multi-platform wheels), CPython ABI churn between minor versions. Mitigation: pin `abi3-py39` for forward compatibility, gate any non-`abi3` features.

- [x] **Constraint-guided pruning in symreg via `EmlSmtSolver`** (implemented 2026-04-27) — drop topologies that cannot fit the training domain (UNSAT). `[research]`
  - **Why:** OxiEML uniquely has both a symreg engine and an SMT solver in the same crate. Using interval/SMT to prove `∀x ∈ training_box, f(x) ≠ y(x)` for a candidate topology lets us drop entire branches before paying for Adam fitting. This is a real research angle nobody else can run.
  - **Design:** Before optimization, abstract each topology to its `LoweredOp` skeleton with constants treated as free variables. Encode `∃c. ∀(x,y) ∈ training. |f(x; c) − y| ≤ ε` as an SMT query over the constants; if UNSAT, skip. Use the new `LoweredOp::eval_interval` for cheap pre-filtering before the expensive SMT call. Threshold: only invoke SMT when interval-only filtering is inconclusive.
  - **Files:** `src/symreg.rs`, `src/smt.rs`. **Tests:** Synthetic dataset where half the topology space is provably infeasible; pruning reduces fit-time by ≥ 30% with no quality loss. **Risk:** SMT call cost can exceed Adam fit cost on small topologies — gate behind `SymRegConfig.smt_pruning: bool` (default false) and a depth threshold.

---

## Future: Research Directions

- [x] **ODE / PDE discovery (SINDy-style)** — given a trajectory `x(t)`, discover `f` such that `dx/dt = f(x)`. `[research]` (implemented 2026-04-27)
  - Differentiate the trajectory numerically (Savitzky-Golay or central differences), feed `(x_i, ẋ_i)` pairs into the symreg engine. Multi-output mode handles vector ODEs. Extension to PDEs requires a discretization story (method-of-lines on a grid). Open questions: noise robustness, sparsity-promoting regularizers, conserved-quantity constraints encoded via SMT.

- [x] **Dimensional analysis / unit-aware regression** — annotate variables with SI units; reject formulas that violate unit consistency. `[research]` (implemented 2026-04-28)
  - Ship a small `Units` algebra (length, time, mass, …) as exponent vectors. Each `LoweredOp` carries a unit signature; `Add`/`Sub` require unit equality, `exp`/`ln` require dimensionless argument, `Pow` requires rational exponent for non-dimensionless base. Integrates with symreg as a hard pruning filter — entire topology branches drop if they're dimensionally inadmissible. This is a 10-100× search-space reduction on physics problems and produces formulas that are unit-checked by construction.

- [x] **JIT compilation of `OxiOp` sequences via cranelift** — pure-Rust JIT for hot evaluation paths. `[research]` (implemented 2026-04-28)
  - Cranelift is pure Rust and a natural fit. Generate IR from the post-order `OxiOp` sequence; emit machine code at first call, cache by structural hash. Expected win: 5-20× over the interpreter on long batches, beating even SIMD on irregular workloads. Risk: cranelift is large; gate behind a `jit` feature so default builds stay lean. Cross-platform parity (aarch64, x86_64, riscv64) needs validation.
  - Implemented: `src/jit.rs` with `JitFn::compile` (OxiOp→Cranelift IR→native code) and `JitCache` (FNV-1a hash keyed LRU-less cache, Mutex-guarded). All 22 OxiOp variants handled; transcendental functions use extern C bindings to libm. 13 integration tests in `tests/jit_test.rs` covering const, vars, arithmetic, exp, sin, cos, neg, div, pow, complex, cache parity, hash stability, and empty-ops error. 0 warnings, 429/429 tests pass.

- [x] **MCTS topology search** — Monte-Carlo tree search over the topology space. `[research]` (implemented 2026-04-28)
  - Beam search is deterministic and breadth-limited; MCTS adds exploration via UCB1 over partially-built trees. State = partial EML tree, action = expand a leaf, value = achieved fit MSE after a short Adam fit. Prior work (DSO, AlphaSymPy) shows promising results. OxiEML angle: combine MCTS rollout pruning with our SMT/interval constraint propagation — a unique combination.
  - Implemented: `src/symreg/mcts.rs` (508 lines) with `PartialNode` recursive enum (Hole/One/Var/Eml), UCB1 selection, leftmost-HOLE expansion, random rollout completion, `1/(1+MSE)` reward. `SymRegStrategy::Mcts { iterations, exploration }` variant added; dispatch via `discover_mcts` bridge in `mod.rs`. Interval pruning hook integrated. 5 integration tests in `tests/symreg_mcts_test.rs`. 434/434 tests pass, 0 clippy warnings.

- [x] **Symbolic equation solving** — given `f(x) = g(y)`, derive `y = h(x)` when invertible. `[research]` (implemented 2026-04-27)
  - Builds on `canonical_rewrite_rules`. For each operator that has a known inverse on its monotone region (`exp`/`ln`, `sin`/`arcsin` with branch cuts, `pow` with appropriate domain), implement a `solve_for(target_var, rhs) -> SolveResult` pass. Returns either a closed-form solution or a residual `f − g` for numeric solving. Useful for closing the loop on discovered formulas (solve for any variable).

- [x] **WASM target + npm package** — `wasm32-unknown-unknown` build with a TypeScript-typed JS API. `[research]` (implemented 2026-04-27)
  - Browser-deployable symbolic regression has obvious educational and demo value (live formula discovery in a notebook UI). Pure-Rust default features make this tractable; SIMD requires `wasm32-bleeding-edge`-level toolchain support. Ship via `npm-publish.yml` (allowed). Open question: does the symreg search complete in interactive time on WASM, or do we need to expose async / web-worker hooks?
  - Implemented via `src/wasm.rs` behind `#[cfg(feature = "wasm")]`; exposes `WasmSymRegConfig`, `WasmDiscoveredFormula`, `WasmSymRegEngine` to JS/TS via `wasm-bindgen = 0.2.121`. `package.json` provides `wasm-pack` build scripts for bundler/node/web targets. CI via `.github/workflows/npm-publish.yml`.

---

## Non-goals

- Compiling full pre-lowering EML trees to einsum — the uniform binary tree is too deep and repetitive for efficient tensor contraction.
- Replacing OxiEML's own simplify/lower pipeline with TensorLogic's compiler.
- Running EML evaluation through the TensorLogic executor.
- Depending on any TensorLogic crate that transitively pulls in SciRS2 (`tensorlogic-compiler` / `-scirs-backend` / `-train`).
- C / Fortran in default features — Pure Rust Policy. C/Fortran-bearing dependencies allowed only behind explicit feature gates.
- `f32` numeric precision — OxiEML targets scientific/research workloads where `f64` is the floor. Not in scope unless a concrete user demands it.
- `bincode`, `flate2`, `zstd`, `zip`, `rustfft`, `Z3`, `openblas`, or any non-OxiOxiZ/OxiBLAS/OxiFFT/OxiARC/oxicode equivalent — COOLJAPAN policy, no exceptions.
- C FFI surface — Python via PyO3 (Upcoming) and WASM (Future) cover external embedding needs.
- `unwrap()` in production code — No-unwrap policy.

---

## v0.1.2 (audit-driven)

- [x] **Fix stale `CHANGELOG.md:29` wording** (planned 2026-05-11)
  - **Goal:** `CHANGELOG.md` accurately describes the v0.1.1 state of `canonical_rewrite_rules()`. Lines 24–30 and 69–72 describe the function as a stub returning `vec![]` pending upstream `Pattern` enum extension — but `TODO.md:89` records 10 real `RewriteRule` instances now implemented.
  - **Design:** Rewrite the v0.1.1 "Added" entry for the `tensorlogic` feature so it describes the 10-rule implementation (exp/log inverses, double negation, identity elements `0+x`, `x*1`, `x/1`, `x^0`, `x^1`). Remove the "Changed" stub-explanation entry at lines 69–72 if now redundant. Keep release date `2026-05-03` intact. Do NOT add a `[0.1.2]` section.
  - **Files:** `CHANGELOG.md`
  - **Prerequisites:** None
  - **Tests:** None (doc fix). Existing `tests/tensorlogic_test.rs` validates actual behavior.
  - **Risk:** Zero

- [x] **Pre-emptive split of `src/symreg/mod.rs` (1541 lines → submodules)** (planned 2026-05-11)
  - **Goal:** Split `src/symreg/mod.rs` into focused submodules under `src/symreg/`, preserving the entire public API and all existing tests. No file in `src/symreg/` exceeds ~600 lines after split.
  - **Design:** Use `splitrs` to seed; refine by hand. Target layout (existing `constants.rs`, `mcts.rs`, `numerics.rs`, `topology.rs` stay untouched):
    - `mod.rs` (<300 lines) — re-exports + `SymRegEngine` + `SymRegConfig` + presets + strategy/loss enums
    - `discover.rs` — `SymRegEngine::discover()` scalar path, Exhaustive/Beam dispatch
    - `discover_multi.rs` — `SymRegEngine::discover_multi()`, Independent/SharedTopology
    - `pareto.rs` — `discover_pareto()`, `DiscoveredFormula::dominates`, free `pareto_front()`
    - `loss.rs` — `SymRegLoss` enum + MSE/Huber/TrimmedMse math
    - `post_round.rs` — post-Adam constant rounding to π/e/simple rationals
  - **Files:** `src/symreg/mod.rs` (shrinks) + 5 new files under `src/symreg/`
  - **Prerequisites:** None (`splitrs` installed)
  - **Tests:** All `tests/symreg*.rs` pass; `cargo nextest run --all-features` green; `cargo clippy --all-features --all-targets -- -D warnings` clean; `cargo doc --all-features --no-deps` succeeds
  - **Risk:** Re-export drift (mitigation: enumerate public surface first, mirror in new `mod.rs`); visibility downgrade (compiler flags); Adam closure lifetimes (keep literal, no abstraction)

- [x] **Pre-emptive split of `src/lower.rs` (1233 lines → submodules)** (planned 2026-05-11)
  - **Goal:** Split `src/lower.rs` into `src/lower/` directory module with one submodule per responsibility. Preserves public surface and all dependent tests.
  - **Design:** Use `splitrs` to seed; refine by hand. Target layout (existing `lower_grad.rs`, `lower_interval.rs`, `lower_simplify.rs`, `lower_units.rs` stay top-level):
    - `lower/mod.rs` (slim) — re-exports + `LoweredOp` enum + `structural_hash` + `simplify` entry
    - `lower/pattern.rs` — sin/cos + 10 transcendental canonical-shape pattern matchers
    - `lower/grad.rs` — `grad(wrt)`, `grad_all()`, `jacobian(n_vars)`, `hessian(n_vars)`
    - `lower/oxiblas.rs` — `to_oxiblas_ops` post-order flat-IR emission
  - **Files:** `src/lower.rs` (deleted; replaced by `src/lower/mod.rs`) + 3 new submodules under `src/lower/`
  - **Prerequisites:** None
  - **Tests:** `tests/lowered_grad_test.rs`, `tests/lower_transcendental_test.rs`, `tests/grad_proptest.rs`, `tests/grad_utilities_test.rs`, `tests/trig_precision_test.rs` all pass; clippy clean
  - **Risk:** `lower::structural_hash` visibility (mitigation: explicit `pub use`); submodule circularity (mitigation: `LoweredOp` enum stays in `mod.rs`, submodules `use super::LoweredOp`)

---

## v0.1.2 (round 2 — deferred splits)

- [x] **Pre-emptive split of `src/bin/oxieml.rs` (1169 lines → submodules)** (planned 2026-05-13)
  - **Goal:** Reduce `src/bin/oxieml.rs` to a slim entry point + arg dispatch, with per-mode logic in `src/bin/oxieml/`. Public CLI behavior unchanged. No Cargo manifest change.
  - **Design:** Declare `mod <submod>;` in `bin/oxieml.rs`; submodules live in `src/bin/oxieml/`:
    - `mod.rs` / `oxieml.rs` (slim ~250 lines) — `main()`, arg dispatch, `print_usage`, `print_help`, `usage_text`
    - `format.rs` — `KNOWN_CONSTANTS`, `OutputFormat`, `output_path`, `write_output`, `json_escape_str`
    - `evaluate.rs` — `run_evaluate_fmt`
    - `generate.rs` — `run_generate`, `try_generate`, `parse_func_call`, `parse_arg`, `strip_func`, `print_known_functions`
    - `lower.rs` — `run_lower`
    - `grad.rs` — `run_grad`
    - `symreg.rs` — `run_symreg`, `format_symreg_results`, `parse_dataset`, `get_symreg_data`, `parse_named_usize/f64`, `parse_strategy`
    - `args.rs` — `get_input`, `parse_var_assignments`, `count_variables`, `check_known_constants`, `check_known_constants_labeled`
  - **Files:** `src/bin/oxieml.rs` (shrinks) + 7 new files under `src/bin/oxieml/`
  - **Prerequisites:** None
  - **Tests:** All `tests/cli_*.rs` pass; `cargo nextest run --all-features` green; `cargo clippy --all-features --all-targets -- -D warnings` clean
  - **Risk:** Bin auto-discovery (no `main.rs` in subdir); visibility cascade (compiler catches); shared `json_escape_str` in `format.rs`

- [x] **Pre-emptive split of `src/smt.rs` (1024 lines → submodules)** (planned 2026-05-13)
  - **Goal:** Convert `src/smt.rs` → `src/smt/` directory module, one submodule per layer. Public API (`EmlConstraint`, `EmlSolution`, `Interval`, `PropResult`, `IntervalDomain`, `EmlNraSolver`, `SmtResult`, `EmlSmtSolver`) preserved exactly.
  - **Design:**
    - `smt/mod.rs` — module doc + `pub use` re-exports of all public types
    - `smt/constraint.rs` — `EmlConstraint`, `EmlSolution`
    - `smt/interval.rs` — `Interval`, `PropResult`, `IntervalDomain`, `eval_interval`, `propagate_once`
    - `smt/nra.rs` — legacy `EmlNraSolver` (bisection + propagation)
    - `smt/oxiz_backend.rs` — all `#[cfg(feature = "smt")]` items (`SmtResult`, `EmlSmtSolver`, `OxizVerdict`, `oxiz_check`, `float_to_term`, `encode_constraint`, `encode_tree`)
    - `smt/helpers.rs` — `check_constraint`, `evaluate_constraint_residual`, `count_constraint_vars` (shared, `pub(super)`)
    - `smt/tests.rs` — always-on `mod tests`
    - `smt/smt_tests.rs` — OxiZ-backed `mod smt_tests` (`cfg(all(test, feature = "smt"))`)
  - **Files:** `src/smt.rs` (deleted; replaced by `src/smt/mod.rs`) + 7 new files under `src/smt/`
  - **Prerequisites:** None
  - **Tests:** `cargo nextest run --all-features` green; `cargo nextest run` (no smt feature) green; clippy clean
  - **Risk:** Feature-gate propagation (re-export `SmtResult`/`EmlSmtSolver` with `#[cfg(feature = "smt")]` in `mod.rs`); cross-module ref `oxiz_backend → nra` via `super::nra`; test module access to `pub(super)` helpers; delete `smt.rs` before `cargo check`

---

## v0.1.2 (round 3 — quality improvements)

- [x] **Adam optimizer allocation churn fix in `src/symreg/discover.rs`** (planned 2026-05-14)
  - **Goal:** Eliminate per-iteration heap allocations in the Adam loop. Each iteration currently allocates 3+ Vecs (outputs, residuals, gradient) plus one inner Vec<f64> Jacobian per training row. Pre-allocate once per restart, reuse via clear()+push().
  - **Design:**
    - Hoist `outputs`, `residuals`, `tg` out of the `for t` loop; `clear()`+`resize()` at top of each iteration
    - Add `forward_with_jacobian_into(&self, ctx, jac_out: &mut Vec<f64>) -> Result<f64, EmlError>` in `src/grad.rs`; delegate `forward_with_jacobian` to it
    - Replace `outputs_and_jacs: Vec<(f64, Vec<f64>)>` with parallel flat buffers: `outputs: Vec<f64>` and `jac_buf: Vec<f64>` (row-major); local helper `jac_row(jac, i, n_params)` for indexing
  - **Files:** `src/grad.rs`, `src/symreg/discover.rs`, `tests/symreg_alloc_parity_test.rs` (new)
  - **Prerequisites:** None
  - **Tests:** Numerical-parity test (fixed seed, assert top-1 formula MSE+params bit-identical to pre-refactor snapshot); `forward_with_jacobian_into` unit test asserting equal output to `forward_with_jacobian`; all existing tests pass
  - **Risk:** Numerical drift (mitigated by parity test); flat Jacobian off-by-one (mitigated by `jac_row` helper + parity test); borrow checker on shared `tg` buffer (borrows non-overlapping)

- [x] **`eval_ops` debug-time correctness check in `src/lower/oxiblas.rs`** (planned 2026-05-14)
  - **Goal:** Catch malformed-IR stack underflows in debug builds. Current 23× `stack.pop().unwrap_or(f64::NAN)` silently masks stack underflow as a math NaN, making it indistinguishable from legitimate NaN (e.g. ln(-1)). Production semantics unchanged; debug panics on underflow.
  - **Design:** Add `#[inline(always)] fn pop_or_nan(stack: &mut Vec<f64>) -> f64 { debug_assert!(!stack.is_empty(), "OxiOp stack underflow — malformed IR"); stack.pop().unwrap_or(f64::NAN) }`. Replace all 23 `stack.pop().unwrap_or(f64::NAN)` sites with `pop_or_nan(&mut stack)`.
  - **Files:** `src/lower/oxiblas.rs`, `tests/oxiop_malformed_ir_test.rs` (new)
  - **Prerequisites:** None
  - **Tests:** `#[cfg(debug_assertions)] #[should_panic(expected = "stack underflow")]` test with a deliberately-truncated OxiOp sequence; all existing tests pass
  - **Risk:** False-positive debug panics if any existing test passes malformed OxiOp sequences (none found in audit); zero-cost in release builds

- [x] **TODO.md cosmetic drift cleanup** (planned 2026-05-14)
  - **Goal:** Fix two stale references in the "Future: Research Directions" section.
  - **Design:** Line 206: `invert(target_var)` → `solve_for(target_var, rhs) -> SolveResult`. Line 210: `wasm-bindgen = 0.2.118` → `wasm-bindgen = 0.2.121`.
  - **Files:** `TODO.md`
  - **Prerequisites:** None
  - **Tests:** None (doc-only)
  - **Risk:** None

---

## v0.1.2 (round 4 — Arc migration + CHANGELOG sync)

- [x] **`Arc<LoweredOp>` migration** (planned 2026-05-14)
  - **Goal:** Replace `Box<LoweredOp>` with `Arc<LoweredOp>` in all 20 recursive-enum variant slots in `src/lower/mod.rs`. Public API (`grad`, `simplify`, `eval`, `to_oxiblas_ops`, `to_latex`, etc.) unchanged in name and signature. Hot-path AST clones in `grad`/`grad_all`/`jacobian`/`hessian`/`simplify` rewrites drop from O(|tree|) deep-recursive clone to O(1) atomic reference-count bump — the dominant clone sites are `src/lower_grad.rs` (25 `.clone()` calls) and `src/lower_simplify.rs` (17 `.clone()` calls).
  - **Design:** (1) Enum definition at `src/lower/mod.rs:27` — swap 20 `Box<LoweredOp>` slots to `Arc<LoweredOp>`, add `use std::sync::Arc;`. (2) Construction sites: `Box::new(LoweredOp::...)` → `Arc::new(LoweredOp::...)` across `src/lower_grad.rs`, `src/lower_simplify.rs`, `src/lower/pattern.rs`, `src/simplify.rs`, `src/canonical.rs`, `src/solve.rs`, `src/symreg/constants.rs`, `src/symreg/topology.rs`, `src/tensorlogic.rs`, `src/compile.rs`. (3) Clone sites: `(**ptr).clone()` deep-clone patterns → `Arc::clone(ptr)` O(1) in `lower_grad.rs` and `lower_simplify.rs`. (4) Move-out edge cases (`*box_value` → compiler-surfaced → switch to `(*arc).clone()` or reference). (5) Out-of-scope: structural-hash-keyed Arc CSE dedup (follow-up round).
  - **Files:** `src/lower/mod.rs`, `src/lower_grad.rs`, `src/lower_simplify.rs`, `src/lower/pattern.rs`, `src/simplify.rs`, plus other LoweredOp construction sites as needed; `tests/arc_sharing_test.rs` (new).
  - **Tests:** Behavioral parity: full 438-test all-features suite must remain green byte-identically. New `tests/arc_sharing_test.rs` asserts `Arc::strong_count > 1` after cloning a shared subtree — proves structural sharing, not deep-copying.
  - **Risk:** Move-out incompatibility (`*box_value` → use `(*arc).clone()` or restructure; compiler surfaces every case); public-API breaking change for external direct constructors (pre-1.0, documented in CHANGELOG); ~5ns atomic overhead on construction (dominated by O(|tree|)→O(1) clone wins).

- [x] **CHANGELOG.md sync for v0.1.2 rounds 2, 3, 4** (planned 2026-05-14)
  - **Goal:** `CHANGELOG.md` accurately documents all v0.1.2 changes. Currently missing: round-2 splits (`src/bin/oxieml.rs` 1169-line split, `src/smt.rs` 1024-line split), round-3 quality fixes (Adam alloc-churn fix + `forward_with_jacobian_into`, `pop_or_nan` debug-assert helper), and the round-4 Arc migration.
  - **Design:** Append to the existing v0.1.2 `[Unreleased]` section (no new round-N header — CHANGELOG is user-facing). Add `### Performance` bullets (Adam alloc churn fix + Arc migration), `### Breaking Changes` bullet (Box → Arc variant API), `### Internal` bullets (bin/oxieml split, smt split, pop_or_nan helper). Match existing CHANGELOG style; no commit.
  - **Files:** `CHANGELOG.md`
  - **Tests:** None (doc-only)
  - **Risk:** None

## v0.1.2 (round 5 — CSE hash-consing + sharing-aware codegen + CHANGELOG sync)

- [x] CSE hash-consing + sharing-aware codegen + integration (planned 2026-05-15)
  - **Goal:** Three coordinated capabilities: (1) `LoweredOp::cse(&self) -> Arc<LoweredOp>` hash-consing pass via a `CseInterner` dual-map; (2) `OxiOp::Store(usize)` / `Load(usize)` opcodes + `to_oxiblas_ops_shared` two-pass emitter (in-degree census → emit-with-slots); (3) integration: `eval_real_lowered` rewired to `.cse() → to_oxiblas_ops_shared()`, `grad` gains terminal `.cse()` + `Arc::try_unwrap`, double-simplify removed from `grad_all`/`hessian`.
  - **Design:** `CseInterner` holds `visited: HashMap<*const LoweredOp, Arc<LoweredOp>>` (pointer-identity O(DAG-size) memo) and `table: HashMap<u64, Vec<Arc<LoweredOp>>>` (structural-hash intern table with `PartialEq` collision guard). Two-pass emitter in `to_oxiblas_ops_shared`: pass 1 counts in-degrees per `Arc::as_ptr`; pass 2 emits each shared node once with `OxiOp::Store(slot)` (peek semantics, does NOT pop) and subsequent refs with `OxiOp::Load(slot)`. `eval_ops` gains `let mut slots = vec![f64::NAN; n_shared];`. SIMD path mirrors scalar with `slots: Vec<SimdVec>`.
  - **Files:** `src/lower_cse.rs` (NEW), `src/lib.rs`, `src/lower/oxiblas.rs`, `src/simd_eval.rs`, `src/lower/mod.rs`, `src/lower_grad.rs`, `tests/lower_cse_test.rs` (NEW), `tests/oxiblas_shared_test.rs` (NEW).
  - **Tests:** CSE sharing (`Arc::ptr_eq`), behavioural parity, idempotence, `NamedConst(Pi)`/`Const(PI)` not merged, input-DAG bounded traversal, `±0.0` edge case, grad→shared DAG, `grad_all` parity; strict-generalisation invariant (no-sharing tree → byte-identical op vec), `Store`/`Load` slot correctness, shared-DAG → shorter op vec, deep grad → fewer ops, SIMD parity.
  - **Risk:** Store semantics drift mitigated by SIMD-parity test; hash collision mitigated by `PartialEq` guard; `OxiOp` variant addition documented in CHANGELOG.

- [x] CHANGELOG.md sync for round-5 CSE work (planned 2026-05-15)
  - **Goal:** Append CSE hash-consing + sharing-aware codegen + double-simplify-removal bullets, plus `OxiOp` breaking-change note, to the existing `[0.1.2]` section in `CHANGELOG.md`.
  - **Files:** `CHANGELOG.md` only.
  - **Prerequisites:** Item 1 (CSE implementation) must complete first — the wording references actual method/opcode names from the implementation.

## v0.1.3 (planned 2026-06-13)

- [x] **H3 — Polynomial algebra module split** (implemented 2026-06-14) — Split `src/poly.rs` into `src/poly/` directory module with `mod.rs`, `univariate.rs`, `factor.rs`, `sturm.rs`, `multivariate.rs`, `tests.rs`. Added `Factorization` struct, Yun's square-free decomposition, Kronecker splitting, resultant, discriminant, content/primitive-part. All tests pass.

- [x] **A4 — Numeric root-finder + adaptive quadrature + `solve_numeric` wiring** (planned 2026-06-13)
  - **Goal:** Close the documented `SolveResult::Residual` loop. `find_root(var, &EvalCtx, x0)` (Newton via the existing symbolic derivative + bracketed Brent fallback), `find_roots_in(var, &EvalCtx, a, b, n_samples)`, adaptive-Simpson `quadrature(var, &EvalCtx, a, b)`, and `SolveResult::solve_numeric(var, &EvalCtx, x0)`. Foundation slice — everything in Milestone A depends on its `EmlError` variants and (for A1) `quadrature`.
  - **Design:** Newton uses `self.grad(var)` (computed once) with `|f|<tol ∧ |Δx|<tol` convergence, `max_iter=100`; on divergence / `df≈0` switch to outward sign-change bracket search + Brent (guaranteed convergence). `find_roots_in` grids `[a,b]`, brackets sign changes, refines via Brent, sorts+dedups. Adaptive Simpson with Richardson error (`tol=1e-10`, `max_depth=50`, `min_depth=2`); non-finite sample ⇒ `UndefinedAtPoint`. Single `eval_at` funnel pads the var vector. `RootOpts`/`QuadOpts` configs. New `EmlError` variants: `NonConvergence{method,iterations}`, `UndefinedAtPoint(f64)`, `InvalidParameter(&'static str)`.
  - **Files:** `src/numeric.rs` (new ~420 lines), `src/error.rs` (3 new variants + Display), `src/solve.rs` (`solve_numeric` ~30 lines), `src/lib.rs` (module decl + `RootOpts`/`QuadOpts` re-exports).
  - **Prerequisites:** none — ships first.
  - **Tests:** roots of `x²−2`(±√2), `cos x`(π/2), `eˣ−2`(ln2), `x³−x`(−1,0,1); Newton→Brent fallback; `find_roots_in(sin,[0,3π])`→{0,π,2π,3π}; quadrature ∫₀^π sin=2, ∫₀¹x²=⅓, ∫_{-1}^1 1/(1+x²)=π/2, reversed bounds, singular⇒`UndefinedAtPoint`; FTC property test; `solve_numeric` `Closed`+`Residual`(x+sin x=1≈0.511, x·eˣ=1≈0.567).
  - **Risk:** Newton divergence → Brent fallback; quadrature on improper integrals → `UndefinedAtPoint`.

- [x] **A1 — Symbolic integration** `LoweredOp::integrate(wrt) -> IntegrateResult` (planned 2026-06-13)
  - **Goal:** Sound antiderivative engine mirroring `raw_grad`. `enum IntegrateResult { Closed(LoweredOp), Unsupported }`. `integrate_definite(wrt, a, b, &EvalCtx)` computes `F(b)−F(a)`, falling back to adaptive quadrature when `Unsupported` or endpoint non-finite.
  - **Design:** `raw_integrate(op, wrt, depth) -> Option<LoweredOp>`: constant guard; linearity (`Add`/`Sub`/`Neg`); power rule; `∫exp`, `∫1/x=ln`, full trig/hyperbolic+inverse table; linear u-substitution (detect affine arg `a·x+b` via `arg.grad(wrt)` folding to `Const(a)` ⇒ antiderivative `G(a·x+b)/a`); bounded LIATE integration-by-parts (`depth≤4`, structural anti-cycle check); `f'/f⇒ln` via structural-hash equality. Soundness-by-construction: when unsure return `None`⇒`Unsupported`.
  - **Files:** `src/integrate.rs` (new ~650 lines), `src/lib.rs` (mod decl + re-export `IntegrateResult`).
  - **Prerequisites:** A4 (quadrature + `EmlError` variants).
  - **Tests:** unit table (∫x³, ∫1/x, ∫eˣ, ∫sin, ∫tan, u-sub ∫e^{2x+1}, by-parts ∫x·eˣ/∫x·sin x/∫x·ln x/∫arctan x); `Unsupported` cases; `integrate_definite`. **Property oracle** (`tests/integrate_proptest.rs`, ~1024 cases): `integrate(f).grad()` must numerically recover `f` (skip `Unsupported`).
  - **Risk:** silent unsoundness → killed by property test; by-parts non-termination → depth cap + anti-cycle.

- [x] **A2 — Taylor / Maclaurin series** `LoweredOp::taylor(wrt, center, order) -> Result<LoweredOp>` (planned 2026-06-13)
  - **Goal:** Order-n Taylor polynomial `Σ f⁽ⁿ⁾(center)/n! · (x−center)ⁿ` as a `LoweredOp`. `maclaurin(wrt, order) = taylor(wrt, 0, order)`.
  - **Design:** repeated `grad` + `eval` at center; `factorial_f64` with `order≤170` cap (`InvalidParameter`); non-finite coefficient ⇒ `UndefinedAtPoint(center)`; assemble terms, fold with `Add`, `simplify`. Univariate contract: other vars padded to 0.
  - **Files:** `src/series.rs` (new ~300 lines), `src/lib.rs` module decl.
  - **Prerequisites:** A4 (`EmlError` variants).
  - **Tests:** coefficient checks (exp, sin, cos, ln(1+x), 1/(1−x)); truncation accuracy; nonzero center; edge cases (`ln(x)` Maclaurin → `UndefinedAtPoint`, factorial overflow).
  - **Risk:** factorial overflow / singular center → capped + `Result`.

- [x] **A3 — Limits** `LoweredOp::limit(wrt, LimitPoint) -> LimitResult` (planned 2026-06-13)
  - **Goal:** `enum LimitPoint { Finite(f64), PosInf, NegInf }`, `enum LimitResult { Finite(f64), PosInf, NegInf, DoesNotExist, Indeterminate }`. Numeric probing primary oracle; L'Hôpital symbolic accelerator.
  - **Design:** finite point — direct substitution + two-sided `h`-ladder `{1e-2…1e-8}` with Cauchy-stability + oscillation guard; detect `0/0`/`∞/∞` on top-level `Div` ⇒ L'Hôpital (cap `LHOPITAL_MAX=8`). ±∞ via structural substitution `x→1/t`, `t→0±`.
  - **Files:** `src/limit.rs` (new ~480 lines), `src/lib.rs` re-exports.
  - **Prerequisites:** A4 (`EmlError` variants).
  - **Tests:** `sin x/x→1`, `(1−cos x)/x²→½`, `(eˣ−1)/x→1`, `(1+1/x)^x→e` at +∞; `1/x→DoesNotExist`, `sin(1/x)→DoesNotExist`; cap → `Indeterminate`.
  - **Risk:** numeric misclassification → two-sided ladder + oscillation guard; L'Hôpital fires only on confirmed indeterminate `Div`.

- [x] **B1 — Levenberg–Marquardt least-squares constant fitter** (planned 2026-06-13)
  - **Goal:** `enum OptimizerKind { Adam (default), LevenbergMarquardt }` on `SymRegConfig` + `LmConfig`. LM reuses the already-computed per-row Jacobian, solves `(JᵀJ + λ·diag(JᵀJ))δ = −Jᵀr`, multi-criterion convergence. Fewer iterations, sharper constants.
  - **Design:** pure-Rust SPD solver in `src/linalg.rs` (Cholesky + LU fallback, `solve_normal_equations`, `invert_spd`, `jtj`/`jtr`; `#[cfg(feature="simd")]` swaps inner dot via oxiblas_core). Marquardt diagonal scaling; damping loop; same restart/RNG as Adam. Refactor `discover.rs`: rename Adam body, add dispatcher + shared `finalize_formula` tail (rounding, named-constants, AIC/BIC fill, optional UQ). `cv.rs` optimizer-aware `refit_fold`.
  - **Files:** `src/linalg.rs` (new ~550), `src/symreg/optimize_lm.rs` (new ~450), touch `mod.rs`, `discover.rs`, `cv.rs`, `error.rs` (`SingularMatrix`, `NotSpd`), `lib.rs`.
  - **Prerequisites:** `src/linalg.rs` lands first; both B1 and B2 depend on it.
  - **Tests:** linalg units; LM recovers `a·exp(b·x)` and `a·sin(b·x)+c` to `<1e-4` in `<30` iters; LM MSE ≤ Adam's (same seed); determinism; `n_params==0`; IRLS+outlier; pass under default and `--features simd`.
  - **Risk:** pure-Rust-default → solver has zero oxiblas dep; rank-deficient → Marquardt damping + LU fallback.

- [x] **B2 — PDE discovery** `SymRegEngine::discover_pde(field, dx, dt, &PdeConfig) -> Result<PdeResult>` (planned 2026-06-13)
  - **Goal:** Discover `u_t = N(u, u_x, u_xx, …)` from gridded spatiotemporal data via STRidge (PDE-FIND). Makes the README's false "ODE/PDE discovery" claim true.
  - **Design:** extend `numerics.rs` with spatial finite-difference stencils (1st/2nd order, 2nd/4th accuracy). Build candidate library `Θ = [1, u, u², u_x, u·u_x, u_xx, …]`; target `u_t` via existing time-derivative fns; boundary-trim; L2-normalize columns. STRidge = repeated SPD ridge solves on shrinking support — direct reuse of `solve_normal_equations`/`jtj`/`jtr`. Assemble `PdeResult { terms, coefficients, pretty, mse, latex, coef_intervals }`.
  - **Files:** `src/symreg/pde.rs` (new ~650), `numerics.rs` (spatial stencils), `mod.rs` (`PdeConfig`/`PdeResult`+`discover_pde`), `lib.rs`, `error.rs` (`GridTooSmall{needed,got}`).
  - **Prerequisites:** `src/linalg.rs` (B-S1).
  - **Tests:** stencil exactness; heat eqn `u_t=α·u_xx` (α=0.1) → only `u_xx`≈0.1; Burgers' `u_t=−u·u_x+ν·u_xx` → `u·u_x`≈−1, `u_xx`≈0.05; 1% noise+SG; `GridTooSmall` error; doc-test.
  - **Risk:** noise amplification → SG+boundary trim+normalization+ridge.

- [x] **B3 — Uncertainty quantification (bootstrap + analytic covariance CIs)** (planned 2026-06-13)
  - **Goal:** `param_intervals: Option<Vec<(f64,f64)>>` on `DiscoveredFormula`. Config: `bootstrap_samples: usize` (0=off), `confidence_level: f64` (0.95), `uq_analytic: bool` (LM only).
  - **Design:** Bootstrap (any optimizer): SplitMix64-seeded data resampling, refit via `refit_fold`, percentile CIs; run on top-k finalists post-ranking. Analytic (LM only): `σ²·(JᵀJ)⁻¹` via `invert_spd`, `θ̂ ± z·se` with pure-Rust `inv_norm_cdf` (Acklam).
  - **Files:** `src/symreg/uncertainty.rs` (new ~400), touch `mod.rs` (field+config+both struct-literal sites), `discover.rs` (`finalize_formula`+post-ranking UQ pass).
  - **Prerequisites:** bootstrap path independent; analytic path depends on B1.
  - **Tests:** fixed-seed coverage (~95% in [0.85,1.0] band); analytic-vs-bootstrap agreement ≤20%; determinism; `inv_norm_cdf(0.975)≈1.95996`; UQ only on finalists.
  - **Risk:** bootstrap cost → top-k gate + warm start.

- [x] **B4 — Information criteria (AIC/BIC) + IC-based selection** (planned 2026-06-13)
  - **Goal:** `aic`/`bic` fields on `DiscoveredFormula` (always populated). `enum SelectionCriterion { Score (default), Aic, Bic }`. `pareto_front_ic`/`dominates_by` added.
  - **Design:** `AIC=n·ln(RSS/n)+2k`, `BIC=n·ln(RSS/n)+k·ln(n)`, `k`=fitted-constant count. Clamp `RSS/n` to tiny floor. Compute in `finalize_formula`. Backward-compatible defaults.
  - **Files:** touch `mod.rs` (fields+`SelectionCriterion`), `discover.rs` (`finalize_formula`), `pareto.rs` (`pareto_front_ic`/`dominates_by`).
  - **Prerequisites:** none; co-owns `DiscoveredFormula` struct change with B3.
  - **Tests:** formula correctness; nested-model penalty; ranking flip under `selection=Bic`; `pareto_front_ic`; perfect-fit guard; existing `pareto_front` snapshot unchanged.
  - **Risk:** `ln(0)` → floor clamp; regressions → defaults unchanged.

### Extended scope (round 2)

- [x] **C1 — Polynomial core (`Poly` + sparse `MultiPoly`)** (planned 2026-06-13)
  - **Goal:** New `src/poly.rs` exposing exact polynomial algebra usable by C2/C4/C5.
    `Poly { coeffs: Vec<Ratio<i64>> }` with `from_lowered(op,wrt)`/`to_lowered(wrt)` (Horner),
    `eval`/`eval_f64`, `add`/`sub`/`mul`/`scale`/`neg`, `div_rem` (Euclidean), `gcd` (monic),
    `diff`, `square_free` (Yun), `rational_roots` (rational-root theorem), `isolate_real_roots`
    (Sturm → disjoint brackets), `PolyError{NotPolynomial,CoeffOverflow,DivByZero}`. Plus a sparse
    `MultiPoly { terms: BTreeMap<Vec<u32>, Ratio<i64>> }` for C5.
  - **Design:** Exact `Ratio<i64>` coefficients (not f64): GCD/Yun need a decidable "remainder == 0"
    test that f64 cannot give. The f64-only policy governs the IR/eval layer — `to_lowered` always
    emits `Const(f64)`, so the IR stays pure f64; only symbolic-algebra internals are rational. All
    arithmetic via `i64::checked_*` + `Ratio::new`; overflow ⇒ `CoeffOverflow` (callers degrade to
    numeric, never panic). `from_lowered` is recursive descent. `square_free` = Yun's algorithm.
    Roots: exact `rational_roots` first, then Sturm sequence isolation; f64 polish delegated to
    `numeric::find_root`.
  - **Files:** new `src/poly.rs`; `src/lib.rs`; `Cargo.toml` (un-gate `num-rational` from `smt` to
    unconditional — keeps `default` pure Rust).
  - **Prerequisites:** none (foundation; gates C2, C4-poly, C5).
  - **Tests:** round-trip `from_lowered∘to_lowered`; `div_rem` identity; `gcd((x²−1),(x−1))==(x−1)`;
    Yun square-free; `rational_roots(x²−1)={±1}` exact; `isolate_real_roots(x²−2)` brackets ±√2;
    overflow ⇒ `Err(CoeffOverflow)` no panic.
  - **Risk:** `Ratio<i64>` overflow → checked arithmetic + `CoeffOverflow`. Sturm edge cases → run
    `square_free` first. Keep `MultiPoly` lean; split if nearing 2000 lines.

- [x] **C2 — Partial-fraction decomposition + rational-function integration** (planned 2026-06-13)
  - **Goal:** `∫P(x)/Q(x)dx` returns `Closed` for all rationals whose Q factors into real
    linear/quadratic pieces. New `src/integrate_rational.rs` (`pub(crate) fn integrate_rational`),
    wired into the Div arm of `raw_integrate` (`integrate.rs:~277`).
  - **Design:** Guard: try `Poly::from_lowered` on num+den; fall through on failure (back-compat).
    `div_rem` if degree of num ≥ den. Factor Q via `square_free` + `rational_roots` + complete-the-square
    for irreducible quadratics. Two-tier decompose: Heaviside cover-up for simple linear factors;
    rational linear system for repeated/quadratic. Integrate term-by-term with recursive reduction
    formula for `((·)²+q²)^k`.
  - **Files:** new `src/integrate_rational.rs`; touched `src/integrate.rs`, `src/lib.rs`.
  - **Prerequisites:** C1.
  - **Tests:** `∫1/(x²+1)→arctan`; `∫1/(x²−1)→½ln|·|`; `∫(x³+1)/(x²−1)` FTC vs quadrature;
    `∫1/(x−1)²→−1/(x−1)`; `∫1/(x²+1)²` via reduction; 512-case property test.
  - **Risk:** wrong real/complex split → use only negative-discriminant degree-2 cofactors. Float
    residue solve → exact rational elimination for small systems.

- [x] **C3 — General u-substitution + trig substitution; structural-hash equality** (planned 2026-06-13)
  - **Goal:** Recognize `∫f(g(x))·g'(x)dx` for general g; trig substitution for `∫√(a²−x²)` etc.
    Replace fragile `format!`-string equalities at `integrate.rs:213,313` with `ops_struct_hash`.
  - **Design:** Hash-eq cleanup first (behavior-neutral). General u-sub: enumerate subtree candidates,
    compute g', substitute g→u, integrate, accept only if symbolic grad numerically matches at ≥5+1
    guard probe points. Trig-sub: detect `Pow(inner,±0.5)` with degree-2 inner, emit arcsin/arcsinh/
    arccosh forms — each verified by grad-recover probe.
  - **Files:** touched `src/integrate.rs`; split `src/integrate_subst.rs` if nearing 2000 lines.
  - **Prerequisites:** C1 (trig-sub only). Hash-eq + u-sub: none.
  - **Tests:** `∫2x·cos(x²)→sin(x²)`; `∫x/(x²+1)→½ln(x²+1)`; `∫√(1−x²)` grad-recovers; arcsin/
    arcsinh; negative tests `∫sin x²`, `∫e^{x²}` stay `Unsupported`.
  - **Risk:** u-sub false positives → 6th guard point. Singular probes → skip non-finite.

- [x] **C4 — Equation-solving depth (quadratic/cubic, Lambert-W, systems)** (planned 2026-06-13)
  - **Goal:** Solve `x²+x−1=0`, `x³−x=0`, `x·eˣ=k`, linear systems. Extend `SolveResult` with
    `Roots(Vec<LoweredOp>)`; new `solve_for_all`, `solve_system → SystemSolveResult`; pure-Rust
    `lambert_w0`/`lambert_wm1` in `numeric.rs`.
  - **Design:** Back-compat: `Roots` only from opt-in `solve_for_all`. Quadratic formula on exact
    `Ratio<i64>` coefficients; Cardano (Viète form for 3-real, cube-root form for 1-real); Lambert-W
    via Halley iteration; every emitted solution verified by back-substitution. `solve_system` via
    `MultiPoly` + `linalg::solve_lu`.
  - **Files:** touched `src/solve.rs`, `src/numeric.rs`, `src/lib.rs`; split `src/solve_poly.rs` if
    nearing 2000 lines.
  - **Prerequisites:** C1. Lambert-W is independent.
  - **Tests:** golden-ratio roots; `x³−x→{−1,0,1}`; `W(e)=1,W(−1/e)=−1`; `x·eˣ=2` back-subs;
    `solve_system([x+y=3,x−y=1])→Unique([2,1])`.
  - **Risk:** enum change → same-commit match updates. Cubic Δ≈0 → tolerance band. Complex roots →
    documented `Residual`.

- [x] **C5 — Canonicalizing simplifier (like-term collection / normal form)** (planned 2026-06-13)
  - **Goal:** `simplify` collects like terms: `x+2x→3x`, `(x+1)(x−1)→x²−1`, `1/x+1/(x+1)→(2x+1)/(x²+x)`.
    Idempotent, value-preserving, no public API change.
  - **Design:** New `canonicalize_poly(op)` after existing structural rules. Extract via `MultiPoly`,
    re-emit in canonical monomial order. Never-worse node-count guard ensures idempotence. Value
    preservation: evaluation fingerprint gate (≥5 sample points). Transcendental subtrees are opaque
    atoms. Common-denominator via `Poly::gcd`-based LCM. Bail on >N monomials.
  - **Files:** touched `src/lower_simplify.rs`; split `src/lower_canonical.rs` if nearing 2000 lines.
  - **Prerequisites:** C1.
  - **Tests:** `x+2x→3*x`; `sin x+2 sin x→3 sin x`; `1/x+1/(x+1)` equal-valued; idempotence +
    value-preservation proptests (512 trees each).
  - **Risk:** value-changing rewrite → evaluation-fingerprint gate. Non-termination → monotone
    node-count guard.

- [x] **D1 — Free `Const` grammar leaf + composite-constant recognition** (planned 2026-06-13)
  - **Goal:** Real `EmlNode::Const(f64)` leaf. `SymRegConfig` gains `enable_const_leaf: bool`
    (default `false`) + `const_leaf_init: f64`. Composite constants: 2π, π/2, √3, φ, `q·κ`. Pendulum
    data recovers `T=2π√(L/g)`.
  - **Design:** Architecture (a): real leaf variant (not affine envelopes). ~12 touch sites —
    compiler-enforced via non-`#[non_exhaustive]` `EmlNode`. `Const` arm in `lower/pattern.rs`
    ordered BEFORE One-swallowing patterns. Composite recognition via Stern-Brocot/continued fractions
    (q·κ within 2% + MSE guard). Determinism via already-seeded RNG.
  - **Files:** touched `tree.rs`, `grad.rs`, `eval.rs`, `lower/pattern.rs`, `simplify.rs`,
    `symreg/{topology,mcts,constants,discover,mod}.rs`, `named_const.rs`, `parser.rs`,
    `smt/{interval,oxiz_backend}.rs`, `bin/oxieml/args.rs`.
  - **Prerequisites:** none (foundation for D2/D3).
  - **Tests:** pendulum recovers `2π√(L/g)` with `NamedConst::TwoPi`; `y=3.7x²` recovers free
    `Const(3.7)`; composite snapping; legacy mode bit-identical at seed 42; serde round-trip.
  - **Risk:** touches every match site → compiler-enumerated exhaustiveness. Search blowup → gated
    off by default.

- [x] **D2 — Evolutionary search + parallel island populations** (planned 2026-06-13)
  - **Goal:** New `SymRegStrategy::Evolutionary{…}` and `Islands{…}`, dispatched into new
    `src/symreg/evolution.rs`. Bit-identical across runs at a fixed seed.
  - **Design:** Genome = `Arc<EmlNode>`. Fitness cached by structural-hash. Tournament selection;
    crossover = Arc-subtree swap; mutation = point/subtree-regrow/Const-jitter. Islands via rayon.
    Determinism: per-island `derive_seed` + lockstep barrier sync so migration is index-keyed, never
    rayon-arrival-order-keyed.
  - **Files:** new `src/symreg/evolution.rs`; touched `symreg/mod.rs`, `symreg/discover.rs`.
  - **Prerequisites:** independent; recommended after D1.
  - **Tests:** bit-identical populations at seed S across two runs; thread-count-invariant
    (RAYON_NUM_THREADS=1 vs 4); recovers `sin x+x²` in 50 gens; crossover/mutation ≤ max_depth.
  - **Risk:** rayon non-determinism → generation barrier + index-keyed migration is the guard.

- [x] **D3 — Coupled multi-output (`SharedTopology`)** (planned 2026-06-13)
  - **Goal:** Implement deferred `MultiOutputStrategy::SharedTopology` (`mod.rs:40`). One shared
    skeleton, per-output fitted params. New `SharedFormula`.
  - **Design:** One `EmlTree` + `Vec<Vec<f64>>` per-output params. Score by joint objective
    `Σ wₖ·lossₖ + penalty·size(T)` — size counted once. Layers on existing topology search.
    `discover_ode` with SharedTopology recovers Lorenz cross-terms once.
  - **Files:** new `src/symreg/discover_shared.rs`; touched `symreg/mod.rs`, `symreg/discover_multi.rs`.
  - **Prerequisites:** after D1.
  - **Tests:** Lorenz shared bilinear cross-term; per-output params within 5%; total complexity <
    Σ Independent; single-output degenerates to Independent.
  - **Risk:** dissimilar outputs → keep Independent default. Fit cost → prune via Beam/MCTS first.

- [x] **D4 — SINDy upgrade: STLSQ-library + weak-form + ensemble** (planned 2026-06-13)
  - **Goal:** `SindyConfig{library,threshold,…}`, `SymRegEngine::discover_ode_sindy`. Generalize
    `discover_pde` to user-supplied library. Weak-form for noise robustness. Ensemble bootstrap.
  - **Design:** Configurable library of `LoweredOp` evaluators. STLSQ: refactor `pde.rs::strridge`
    into shared `src/symreg/strlsq.rs` over `linalg::{jtj,jtr,solve_normal_equations}`. Weak-form:
    integrate against compactly-supported test functions via `numeric::quadrature`. Ensemble:
    bootstrap-resample Θ rows, vote on support ≥ quorum, median coefficients.
  - **Files:** new `src/symreg/sindy.rs` + `strlsq.rs`; touched `symreg/pde.rs`, `symreg/mod.rs`,
    `symreg/discover_multi.rs`.
  - **Prerequisites:** independent of D1–D3.
  - **Tests:** STLSQ recovers Lorenz in <1/10 full-SR runtime; weak-form recovers Burgers under 5%
    noise; ensemble selects 7-term support at quorum 0.8; generalized `discover_pde` regression.
  - **Risk:** library blowup → caps + Units filter. Extracting strridge → explicit heat-eqn regression
    guard.

- [x] **E1 — Verified interval integration + Krawczyk root-finding** (planned 2026-06-13)
  - **Goal:** New `integrate_definite_verified` (guaranteed enclosure) and `find_root_verified →
    RootCertificate{enclosure, status: RootStatus{UniqueExists,NoRoot,Indeterminate}}`. Activates
    dead `IntervalLO` / `eval_interval`.
  - **Design:** Verified quadrature: mean-value Taylor form `width·f(mᵢ) + ½width²·[−1,1]·|f'(box)|`,
    adaptive bisection. Verified root-finding: interval Newton operator `N(X)=m−f(m)/F'(X)`;
    Krawczyk fallback when `0∈F'(X)`.
  - **Files:** new `src/numeric_verified.rs`; touched `src/lib.rs`.
  - **Prerequisites:** none (default-feature).
  - **Tests:** `∫₀¹x²` enclosure ∋ 1/3, width <1e-6; Krawczyk certifies unique root of `x²−2`;
    `x²+1` on [−1,1] ⇒ `NoRoot`; enclosure always brackets `quadrature` result.
  - **Risk:** `eval_interval` round-to-nearest → widen endpoints by `next_down()/next_up()`. NaN ⇒
    `Indeterminate`.

- [x] **E2 — Multivariate Newton systems + multidimensional quadrature** (planned 2026-06-13)
  - **Goal:** `solve_system(fs,x0,opts) -> Result<Vec<f64>>`; thin `System` type;
    `quadrature_nd(vars,lo,hi,opts)`. First numeric consumer of symbolic `jacobian`.
  - **Design:** Symbolic Jacobian via `fᵢ.jacobian(n)`, solve `J·Δ=−F` via `linalg::solve_lu`,
    Armijo backtracking line search. Tensor-product Gauss–Legendre for n≤4; Monte-Carlo for higher n.
  - **Files:** new `src/system.rs` + `src/quadrature_nd.rs`; touched `src/lib.rs`.
  - **Prerequisites:** none (default-feature).
  - **Tests:** 2×2 Newton solves `{x²+y²=1,x−y=0}`; line-search converges where undamped diverges;
    tensor-Gauss `∫₀¹∫₀¹(x+y)=1` to 1e-10; MC volume within 3σ.
  - **Risk:** non-convergence → `max_iter` + line search ⇒ `NonConvergence`. Curse of dimensionality
    → `n≤TENSOR_MAX_DIM` guard.

- [x] **E3 — Forward-mode AD (dual numbers) + JVP/VJP + nth-derivative** (planned 2026-06-13)
  - **Goal:** `jvp(x,tangents)->(f64,f64)`, `vjp(x)->(f64,Vec<f64>)`, `nth_derivative(wrt,n)`,
    `mixed_partial(&[usize])`.
  - **Design:** `Dual{re,du}` with transcendental rules; dual-valued `eval_ops` over the `OxiOp` flat
    IR for JVP (zero symbolic blowup). VJP = reverse sweep over Wengert list. `nth_derivative` /
    `mixed_partial` fold `grad` with `simplify().cse()`.
  - **Files:** new `src/autodiff.rs`; touched `src/lower_grad.rs`, `src/lib.rs`.
  - **Prerequisites:** none (default-feature).
  - **Tests:** `jvp(x·y)` at (3,5),v=(1,0)→(15,5); directional deriv matches central-difference;
    `vjp(x²+y²)=(2x,2y)`; `nth_derivative(exp,0,4)` evals to exp.
  - **Risk:** NaN propagation consistent with `eval_ops` silent-NaN policy. Symbolic swell → `cse()`
    each step.

- [x] **F1 — Trustworthy SMT SAT models (verified witnesses)** (planned 2026-06-13)
  - **Goal:** `SmtResult::Sat` carries a verified model. Stop merging `OxizVerdict::Sat` with
    `Unknown` (`oxiz_backend.rs:82`).
  - **Design:** Split merged arm. On Sat: attempt model extraction; else refinement-to-witness via
    damped Newton (reuse E2) inside propagated box; verify via `helpers::check_constraint`. Only
    passing point ⇒ `Sat{is_exact:true}`; else `Unknown`.
  - **Files:** touched `smt/oxiz_backend.rs`, `smt/nra.rs`.
  - **Prerequisites:** `smt` feature; best after E2.
  - **Tests:** SAT model for `exp(x)>0` verified by direct eval; every `Sat` across suite re-verifies
    (invariant test); UNSAT path unchanged.
  - **Risk:** OxiZ model extraction non-ergonomic → refinement is the workhorse. Newton non-convergence
    → bisection → `Unknown`.

- [x] **F2 — Real backward interval tightening + wire SMT into symreg pruning** (delivered 2026-06-13)
  - **Goal:** `interval.rs::propagate_once` actually narrows (writes tightened intervals back;
    `PropResult::Changed` becomes reachable). `SymRegConfig.smt_prune` (default off) wires symreg
    pruning via `EmlSmtSolver`, making the README claim true.
  - **Design:** HC4-revise / interval constraint propagation. After forward sweep, constrain per atom,
    invert each operator to tighten children: for `eml(l,r)`, invert exp/ln; intersect child intervals.
    Iterate to fixpoint. Symreg wiring: `smt_prune` flag → prune topologies that propagate to Conflict.
  - **Files:** touched `smt/interval.rs`, `smt/mod.rs`, `symreg/mod.rs`, `symreg/discover.rs`,
    `symreg/mcts.rs`.
  - **Prerequisites:** backward tightening = default-feature; symreg SMT wiring = `smt`.
  - **Tests:** backward-tighten narrows `x∈[−10,10]` under `x²<4` to ⊆[−2,2]; `propagate` returns
    `Changed` then `Stable`; `smt_prune=true` finds same/better formula; flag-off ⇒ byte-identical.
  - **Risk:** soundness — backward rules only INTERSECT, use outward rounding. Property test: no
    satisfying point discarded.

- [x] **Issue #1 — SMT soundness: spurious `Unsat` from real-domain `ln` of a non-positive interval** (fixed 2026-06-25, post-release follow-up; [#1](https://github.com/cool-japan/oxieml/issues/1))
  - **Goal:** `EmlSmtSolver::check_sat` must never return `Unsat` for a satisfiable constraint. The Phase-1 interval propagator turned `Interval::ln` of a non-positive interval (empty) into a `Conflict`, but EML's `Canonical::sub`/`ln` constructions legitimately evaluate such an intermediate `ln` in the complex domain — so the real interval layer must treat it as INDETERMINATE, never infeasible.
  - **Design:** `eval_interval(node, vars) -> Option<Interval>` returns `None` (indeterminate) when `ln` reaches `<= 0`/non-finite; the six atomic arms map `None -> PropResult::Stable`; `backward_propagate`'s `Eml` arm guards the `ln`-operand and skips back-substitution rather than conflicting. `Interval::ln`'s empty sentinel is unchanged. Net effect: strictly fewer (only sound) `Unsat`/`Conflict` verdicts; `ln(x)>0` on negative domains now returns `Unknown`.
  - **Files:** `src/smt/interval.rs` (the fix), `src/smt/smt_tests.rs` (2 tests corrected to assert sound results), `tests/smt_issue1_const_ln_operand_test.rs` (new regression — interval-layer anchor + 7 issue cases with verified witnesses).
  - **Tests:** interval propagation never spuriously `Conflict`s on `{le,ge,lt,gt,eq}(f,0)` for `f=exp(x0)-1`; all seven issue cases sound (g-cases decidable, f-cases never `Unsat` and `Sat` with a re-verified witness).
  - **Risk:** precision trade-off (some previously-`Unsat` cases now `Unknown`) accepted — `Unknown` is always sound, a false `Unsat` is not.

- [x] **F3 — Richer constraint language** (planned 2026-06-13)
  - **Goal:** Extend `EmlConstraint` with `Not(Box<…>)`, `LtZero`/`LeZero`/`NeZero`, binary
    tree-vs-tree `Lt`/`Le`/`Gt`/`Ge`/`Eq`/`Ne`. Quantifier-free. NNF pass.
  - **Design:** Constructor helpers build `a−b` difference trees. Thread through `helpers.rs`,
    `interval.rs`, `oxiz_backend.rs`, `nra.rs`. Internal NNF pass so `Not` survives only over atoms.
  - **Files:** touched `smt/{constraint,helpers,interval,nra,oxiz_backend}.rs`.
  - **Prerequisites:** `Not`/relation eval = default-feature; OxiZ encoding = `smt`; backward
    tightening of new atoms after F2.
  - **Tests:** `LtZero(exp x)` UNSAT; `Lt(a,b)≡LtZero(a−b)`; `Not(GtZero t)≡LeZero t` (NNF);
    exhaustive-match compiles.
  - **Risk:** strict vs non-strict under over-approximation → document conservative returns. UNSAT
    stays sound.

- [x] **G1 — Documentation-honesty pass** (planned 2026-06-13)
  - **Goal:** (1) test count 434→~529; (2) `eval_complex` labelled as public API; (3) clarify ODE
    vs PDE APIs; (4) SMT-pruning claim marked "planned" (or "delivered" after F2); (5) `uq_analytic`
    documented as reserved/no-op.
  - **Design:** README + doc-comment edits only. Re-measure `grep -rc '#[test]'` at edit time.
  - **Files:** touched `README.md`; optional doc notes on `symreg/mod.rs`, `eval.rs`.
  - **Prerequisites:** mostly standalone; SMT-pruning sentence after F2; test-count edit last.
  - **Tests:** `grep -c '#[test]'` matches README count; `eval_complex` doctest compiles; `cargo doc`
    warning-free.
  - **Risk:** test count drifts as E/F items add tests → count edit last.

### Extended scope (round 3)

- [x] **H3 — Polynomial factorization + true multivariate GCD + split poly.rs** (implemented 2026-06-14)
  - **Goal:** `Poly::factor()` returns content × irreducible factors with multiplicities (bounded, honest scope); `MultiPoly::gcd` becomes a true recursive GCD replacing the var-0 projection approximation; `Poly` gains `content`/`primitive_part`/`resultant`/`discriminant`; and poly.rs (1562 lines) is split with `splitrs` into a sub-2000-line `src/poly/` directory.
  - **Design:** Univariate factorization over ℚ: content/primitive split (`Ratio<i64>`) → full Yun square-free decomposition → rational-root linear factors → irreducible-quadratic detection + bounded Kronecker trial-split (degree ≤6). Never emits a wrong factorization. True multivariate GCD via Brown's-spirit recursive PRS: `gcd = content_gcd · primitive_gcd` via subresultant PRS, base case `Poly::gcd`. Needs new `MultiPoly::div_rem`, `degree_in`/`leading_coeff_in`/`as_univariate_in`. Resultant/discriminant via subresultant PRS. splitrs split (mandatory): `src/poly.rs` → `src/poly/{mod,univariate,factor,sturm,multivariate,tests}.rs`; each target <~900 lines; `pub use poly::{MultiPoly,Poly,PolyError}` stays byte-identical.
  - **Files:** src/poly.rs → src/poly/{mod,univariate,factor,sturm,multivariate,tests}.rs; src/lib.rs.
  - **Prerequisites:** existing `gcd`/`div_rem`/`diff`/`rational_roots`/`square_free`/`scale`; `splitrs`.
  - **Tests:** `x²−1→(x−1)(x+1)`; `(x−1)²(x+1)` mult 2; `x²+1→irreducible`; `x⁴−1→(x−1)(x+1)(x²+1)`; multivariate gcd divides both at random points; `res(x²−1,x−1)=0`; `disc(x²+bx+c)=b²−4c`; `Π factors^m·content == original`; full existing poly suite passes post-split.
  - **Risk:** i64 overflow → `checked_*`→`CoeffOverflow`; Brown recursion bounded by num_vars; Kronecker has hard caps + early-out.

- [x] **H1 — Special-function core (erf, lgamma/digamma, Ei, Si, Ci)** (completed 2026-06-15)
  - **Goal:** erf, Γ (via lgamma), Ei, Si, Ci become first-class `LoweredOp`/`OxiOp` variants — evaluable to ~1e-15 in pure Rust, differentiable, simplifiable, printable, JIT/SIMD-compilable — and the five blocking integrals (`∫e^{−x²}`, `∫sin x/x`, `∫cos x/x`, `∫e^x/x`, `∫1/ln x`) return `IntegrateResult::Closed`.
  - **Design:** Add 6 unary variants `Erf/LGamma/Digamma/Ei/Si/Ci` to `LoweredOp` (src/lower/mod.rs) and mirror in `OxiOp` (src/lower/oxiblas.rs). New `src/special.rs`: erf via W.J. Cody minimax; lgamma via Lanczos (g=7,n=9) + reflection; digamma via recurrence+asymptotic; ei via series/continued-fraction; si/ci via Maclaurin/auxiliary-f,g. Grad: `Erf'=2/√π·e^{−a²}·a'`, `Si'=sin a/a·a'`, `Ci'=cos a/a·a'`, `Ei'=e^a/a·a'`, `LGamma'=Digamma·a'`; Digamma' (trigamma) is documented as out-of-scope. Dispatch across ~17 sites: eval, oxiblas, autodiff, simd_eval, jit, lower_simplify, lower_interval, structural_hash, to_latex/Display, compile.rs, symreg/topology.rs, tensorlogic. Integration table: `∫e^{−x²}→(√π/2)Erf`, `∫sin x/x→Si`, `∫cos x/x→Ci`, `∫e^x/x→Ei`, `∫1/ln x→Ei(ln x)`.
  - **Files:** src/special.rs (new); src/lower/{mod,oxiblas,display}.rs, src/lower_grad.rs, src/lower_simplify.rs, src/lower_cse.rs, src/lower_interval.rs, src/autodiff.rs, src/integrate.rs, src/integrate_subst.rs, src/solve.rs, src/compile.rs, src/jit.rs, src/simd_eval.rs, src/symreg/{constants,topology}.rs, src/tensorlogic.rs, src/canonical.rs, src/lib.rs.
  - **Prerequisites:** H3 (resultant/GCD for rational integration). No new deps.
  - **Tests:** special.rs values vs references to 1e-12; properties (erf odd, lgamma(n+1)=ln n!); finite-difference grad checks; 5 newly-closing integrals; oxiblas round-trip; JIT erf matches scalar. `integrate_exp_x_squared_unsupported` tests flip to Closed — update not delete.
  - **Risk:** ~17 compiler-enforced match sites all land in one pass. Digamma 2nd-derivative gap → loud doc. Si/Ci seam → continuity test at crossover.

- [x] **H4 — Complex & higher-degree roots** (planned 2026-06-14)
  - **Goal:** Negative-discriminant quadratics, quartics (Ferrari), and general degree-n polynomials report their complete real+complex root set via a new `ComplexRoots` API; `lambert_wm1` wired in for the second real branch; existing real-root APIs stay back-compatible.
  - **Design:** Keep `RootsResult` as-is; add `pub struct ComplexRoots { roots: Vec<Complex<f64>> }` + `solve_polynomial_complex`. Quadratic: conjugate pair on disc<0; Cubic: one-real-two-complex via complex cube roots; Quartic: Ferrari (depress → resolvent cubic → two quadratics); General-n: Durand-Kerner with square-free pre-division + Newton polish. Wire `lambert_wm1` for `x·eˣ=k` on `−1/e≤k<0`. Complex stays out of `LoweredOp`; add `solve_for_all_complex` in solve.rs.
  - **Files:** src/solve_poly.rs, src/solve.rs, src/lib.rs. Uses existing `num-complex`. Soft-reuses H3.
  - **Prerequisites:** `num-complex 0.4` (present); `lambert_w0/wm1`. Soft: H3.
  - **Tests:** `x²+1→±i` (was dropped); `x³+1`; `x⁴−1→±1,±i`; `x⁴+1`; DK `x⁵−1→5th roots` (residual <1e-9); `lambert_wm1`: `x·eˣ=−0.2→two roots`; all existing `c4_tests` unchanged; property: DK roots reconstruct the polynomial. Real API stays empty for `x²+1`.
  - **Risk:** DK non-convergence → square-free pre-division + capped iters → `Err(NonConvergence)`. Ferrari near-degenerate instability → tolerance branch + cross-check vs DK.

- [x] **H2 — Symbolic ODE solving `dsolve`** (implemented 2026-06-15)
  - **Goal:** A new `dsolve` recognises and closes separable, first-order-linear, exact, Bernoulli, and 2nd-order linear constant-coefficient ODEs, returning closed-form or implicit solutions with an explicit arbitrary constant.
  - **Design:** New `src/ode.rs`. Derivatives passed as variable slots via `pub struct OdeForm { y, dy, d2y, x }` + `dsolve_form(eq, &OdeForm)`. `pub enum OdeSolution { Explicit(LoweredOp), Implicit(LoweredOp), Unsolved }`. Families: separable, first-order linear (μ=exp(∫p)), exact (M.grad(y)≡N.grad(x), numerical equality check for commutativity), Bernoulli (v=y^{1−n}, polynomial-sampling n detection), 2nd-order const-coeff (characteristic Poly via grad). Arbitrary constant = fresh `Var(max_var+1)`. Added `subst_var` helper and `simplify_exp_ln` for `exp(c*ln(f))→f^c`.
  - **Files:** src/ode.rs (new, ~1000 lines), src/lib.rs. Read-only reuse of integrate.rs, solve.rs, lower_grad.rs, lower_simplify.rs.
  - **Prerequisites:** H4 (soft — 2nd-order complex case degrades to `Unsolved` without it).
  - **Tests:** separable `y'=xy⇒y=Ce^{x²/2}`; linear `y'+y=x⇒y=x−1+Ce^{−x}`; exact `(2xy)dx+x²dy=0`; Bernoulli `y'+y=y²`; 2nd-order `y''−3y'+2y=0`, `y''+y=0`, repeated `y''−2y'+y=0`; residual check; `y'=sin(xy)⇒Unsolved`. 13 ODE unit tests, 2 doctests. All 607 tests pass (0 warnings).

- [x] **I3 — QR + SVD + rank-revealing least squares in linalg** (implemented 2026-06-15)
  - **Goal:** `linalg` exposes pure-Rust `solve_least_squares` (Householder-QR, no κ-squaring), `pinv` (Jacobi-SVD pseudo-inverse), and `qr`/`svd` primitives, all `Result`-returning. Existing tests still pass.
  - **Design:** Split `linalg.rs` (510 lines) → `src/linalg/{mod,decomp,solve,builders}.rs` (re-export all pub paths). Householder QR storing compact reflectors + betas. One-sided Jacobi SVD: sweep column pairs to orthogonality, σ=‖col‖, sort descending; cap max_sweeps=60. `pinv(a,m,n,rcond)` = `V·diag(1/σ trunc)·Uᵀ` (rcond 1e-12). Opt-in QR path in strlsq (default stays normal-equations).
  - **Files:** src/linalg.rs → src/linalg/{mod,decomp,solve,builders}.rs; src/symreg/strlsq.rs.
  - **Prerequisites:** existing `cholesky_factor`/`solve_normal_equations`/`EmlError::{SingularMatrix,NotSpd}`.
  - **Tests:** `Q·R≈A`, `QᵀQ≈I`; `U·Σ·Vᵀ≈A` on rectangular + rank-deficient; `solve_least_squares` beats normal-equations on Hilbert-like design; `pinv` satisfies four Moore–Penrose conditions to 1e-9; strlsq/pde regression.
  - **Risk:** Jacobi non-convergence → cap sweeps, `SingularMatrix` only on NaN/Inf. Split: re-export all symbols, `cargo nextest --no-run` after split before logic.

- [x] **I1 — Analytic (Laplace/Hessian) uncertainty quantification** (planned 2026-06-14)
  - **Goal:** The dead `uq_analytic` flag (symreg/mod.rs:274) becomes live for the LM optimizer: top-k LM-fitted formulas receive `param_intervals` from the Laplace covariance `Σ=σ̂²(JᵀJ)⁻¹`.
  - **Design:** New `compute_analytic_intervals` in uncertainty.rs: rebuild tree, re-assemble J + residuals (factor `assemble_jac_residuals` from optimize_lm.rs into `pub(super)`), `σ̂²=RSS/(n−k)`, `Σ=σ̂²·invert_spd(JᵀJ)` with `pinv` fallback on NotSpd, CIs `θ̂±z·√diagΣ` reusing existing `inv_norm_cdf` (Acklam, uncertainty.rs:30). Wire into discover.rs::optimize_and_finalize (~454-468): bootstrap precedence; analytic when `uq_analytic && optimizer==LevenbergMarquardt`.
  - **Files:** src/symreg/uncertainty.rs, src/symreg/discover.rs, src/symreg/optimize_lm.rs, src/symreg/mod.rs (doc only).
  - **Prerequisites:** I3's `pinv` (soft — without it use `invert_spd`, return `None` on NotSpd).
  - **Tests:** `uq_analytic=true` → `param_intervals.is_some()`, finite, true coeff inside CI; analytic-vs-bootstrap agreement (half-widths within 2×, both bracket truth); `n−k≤0`→`None`; Adam+`uq_analytic`→`None`.
  - **Risk:** singular JᵀJ → pinv + non-negative-diag guard → `None`. Bootstrap precedence branch kept first.

- [x] **I2 — Multi-dimensional PDE discovery** (implemented 2026-06-15)
  - **Goal:** `discover_pde` generalizes from 1-D/fixed-6-term to 2-D/3-D spatial grids, a user-extensible library, mixed/multi-axis derivatives, and an optional weak-form mode. The `engine` arg is honored. 1-D heat-equation test passes unchanged.
  - **Design:** `PdeField{data, shape:{D1/D2/D3}}` with per-axis `dx`. 1-D signature is a thin wrapper to `discover_pde_nd`. Derivatives via `nth_derivative_1d` (orders 1–3) + `apply_axis_derivative`; mixed = sequential 1-D stencils. `PdeLibraryTerm{label,latex,factors}`; `PdeConfig.library:Option` (None→default_library_1d byte-identical). `PdeMode::WeakForm` via tensor-product Hann windows. Honor `engine`.
  - **Files:** src/symreg/pde.rs (split into pde/{mod,library,weak}.rs if nearing 2000), src/symreg/numerics.rs, src/symreg/mod.rs.
  - **Prerequisites:** `strlsq`, existing `first_/second_derivative_1d`; I3 QR (soft).
  - **Tests:** regression heat-equation + `grid_too_small`; `nth_derivative_1d` order 3 on cubic; mixed `u_xy`; 2-D heat recovers both coefficients; 1-D advection; weak-form under noise; default n-D library with `spatial_dims=1` == legacy 6-term.
  - **Risk:** n-D library combinatorial blowup → `max_deriv_order` + degree caps. All new `PdeConfig` fields `Option`/defaulted.

- [x] **I4 — Units-aware search inside MCTS/GA + rational dimensions** (implemented 2026-06-15)
  - **Goal:** `unit_filter` prunes during MCTS expansion/rollout and GA construction/mutation/crossover; `units.rs` supports rational exponents (`m^(1/2)`). Integer-dimension behavior is bit-for-bit preserved.
  - **Design:** Replace `Units([i8;7])` with `Units([Rexp;7])` where `Rexp{num:i16,den:i16}` is an in-crate normalized rational; `mul`/`div` become rational add/sub; `pow_rational`/`sqrt`; keep integer constructors + `from_int_exps`/`try_into_int_exps`. lower_units.rs Pow arm: rationalize Const exponent (continued fraction den≤12). MCTS: optimistic `partial_units_feasible` (holes=⊤) wired into `legal_actions` + `complete_random`, gated on `Some(unit_filter)`, post-hoc `check_units` backstop. GA: reject inadmissible children in `random_tree`/`mutate_*`/`crossover` (bounded retry → fall back to parent). Add `SymRegConfig::with_units`.
  - **Files:** src/units.rs, src/lower_units.rs, src/symreg/{mcts,evolution,topology}.rs, src/symreg/mod.rs.
  - **Prerequisites:** existing `LoweredOp::check_units` (reused verbatim).
  - **Tests:** `METER.pow_rational(1/2)=m^(1/2)`; `m^(1/2)·m^(1/2)=m`; legacy integer ops byte-identical; `x^0.5` over `m²`→`m`; MCTS/GA with `var_units`: every formula passes `check_units`; MCTS evaluates fewer candidates; `unit_filter=None` → identical seeded output.
  - **Risk:** Units type change is wide-blast → full `cargo nextest` after type change before MCTS/GA. Bounded GA retries → fall back to parent, never infinite-loop.

- [x] **J1 — Bounded quantifiers (∀/∃ over box domains)** (planned 2026-06-14)
  - **Goal:** Add `ForAll`/`Exists` over box domains to `EmlConstraint`, decided by interval-refutation (∀) and witness-search (∃), with conservative `Unknown`.
  - **Design:** New variants `ForAll{var,lo,hi,body}` / `Exists{...}` on `EmlConstraint` (constraint.rs), not `#[non_exhaustive]`. NNF/negate: `¬∀x∈B.φ⇒∃x∈B.¬φ`. `decide_quantifier` in helpers.rs: ∀ TRUE when `IntervalDomain::propagate(¬body)` yields Conflict; false sample ⇒ FALSE; else Unknown. ∃ TRUE by witness search + `check_constraint`; else Unknown. interval.rs `propagate_once` gets ForAll/Exists arms returning Stable. oxiz_backend.rs decides top-level ∀/∃ before LRA, returns None for nested.
  - **Files:** src/smt/constraint.rs, src/smt/helpers.rs, src/smt/interval.rs, src/smt/oxiz_backend.rs.
  - **Prerequisites:** none. Do J1 + J3 in one pass over interval.rs/helpers.rs.
  - **Tests:** `∀x∈[−5,5].exp(x)>0⇒Sat`; `∀x∈[−2,−1].ln(x)` defined ⇒ counterexample; `∃x∈[0,3].exp(x)=2⇒Sat witness≈0.693`; `∃x∈[5,6].exp(x)=2⇒Unknown`; NNF round-trip; all existing QF tests still compile.
  - **Risk:** Unknown is the soundness escape hatch. Do J1+J3 together (same match sites).

- [x] **J3 — Disjunction-hull tightening + NeZero disequality splitting** (planned 2026-06-14)
  - **Goal:** `Or` propagation tightens to the union-hull of feasible branches; `NeZero` reasons near a point (not only exact `[0,0]`). Sound, conservative, no regression.
  - **Design:** Or (interval.rs:392-409): clone `vars` per branch, single `propagate_once`, collect survivors — none⇒Conflict; one⇒adopt; >1⇒per-var `hull(survivors) ∩ original`, tighten only if strictly inside. NeZero (interval.rs:363-372): point `[c,c]` with `|c|<eps` ⇒ Conflict; bare `Var(i)` x≠0 with endpoint=0 ⇒ nudge `f64::next_up/next_down`; 0 interior ⇒ Stable (documented). `NEZERO_EPS` const in helpers.rs.
  - **Files:** src/smt/interval.rs, src/smt/helpers.rs.
  - **Prerequisites:** J1 (do together, same interval.rs pass).
  - **Tests:** `Or[[0,3],[1,2]]`-style → lo tightened; one infeasible branch collapses; all-infeasible → Conflict; `x≠0` on `[0,5]` → lo=next_up(0); on `[0,0]` → Conflict; on `[−5,5]` bare Var → Stable; fixpoint terminates (MAX_ITERATIONS=20).
  - **Risk:** Hull MUST intersect with old (no widening). next_up/down nudge is idempotent.

- [x] **J2 — OxiZ model extraction + wire EmlSmtSolver into symreg** (planned 2026-06-14)
  - **Goal:** (a) Extract a model from OxiZ on Sat and use it as a refinement seed. (b) Add a real `check_sat`-backed UNSAT pruning path in symreg so README:248 becomes TRUE.
  - **Design:** (a) On Sat read `solver.model()` (oxiz 0.2.3), map `var_terms[i]`→f64 (missing→midpoint). `OxizVerdict::Sat{seed}`; verify via `check_constraint`, else refine with `solve_system_newton` from seed, else fall back to bisection. OxiZ models only the LRA relaxation — seed is a starting point; soundness via verify+refine. (b) `SymRegConfig.smt_prune_solver:bool` (default false). New `src/symreg/smt_prune.rs` (deduplicates discover.rs:234-247/mcts.rs:429-441) under `#[cfg(feature="smt")]` calls `check_sat` to UNSAT-prune. Depth-gated, opt-in.
  - **Files:** src/smt/oxiz_backend.rs, src/symreg/smt_prune.rs (new), src/symreg/discover.rs, src/symreg/mcts.rs, src/symreg/mod.rs.
  - **Prerequisites:** none; no dep bump (oxiz 0.2.3 has `model()`). Gates M1 SMT wording.
  - **Tests:** `eml(x,1)=5` on `[−10,10]` → seed finite near ln5; UNSAT still UNSAT; `smt_prune_solver` result ⊆ `smt_prune=false`, ≥1 topology pruned, winner unchanged.
  - **Risk:** Rational overflow → finite-guard + midpoint fallback. Per-topology `check_sat` slow → opt-in + depth-gated.

- [x] **K1 — Vectorized SIMD transcendentals + AVX-512 F64x8** (planned 2026-06-14)
  - **Goal:** Replace per-lane scalar exp/ln/sin/cos with pure-Rust SIMD polynomial approximations (~1e-13..1e-14 rel); add F64x8/AVX-512 path if oxiblas-core exposes it. Scalar fallback kept. Honest precision note.
  - **Design:** New `src/simd_vec_math.rs` generic over `SimdRegister<Scalar=f64>`. exp: range-reduce k=round(x/ln2), Cody-Waite 2-part remainder, degree-~10 minimax Horner, reconstruct 2^k. ln: mantissa/exponent bit-split + atanh-series Horner. sin/cos: Payne-Hanek-lite + minimax on [−π/4,π/4]. tanh/sinh/cosh from SIMD exp. Priority: exp, ln, sin, cos, tanh. Wire into simd_eval.rs (163-283), relax transcendental test tolerance to ~1e-11. AVX-512: if oxiblas-core exposes F64x8, map Simd512⇒F64x8; else DO NOT hand-roll intrinsics. Fix false "bit-exact" claim at simd_eval.rs:17-19.
  - **Files:** src/simd_vec_math.rs (new), src/simd_eval.rs.
  - **Prerequisites:** verify oxiblas-core `SimdRegister` surface before locking kernels.
  - **Tests:** rel-error < tol on dense grids (exp x∈[−20,20]; ln x∈[1e-6,1e6]; sin/cos x∈[−50,50]); Add..Neg <1e-15; NaN/inf parity; F64x8 `matches_scalar` if added.
  - **Risk:** Large-arg sin/cos correctness → partial vectorization + documented error budget. F64x8 depends on upstream type that may not exist — plan correct either way.

- [x] **L1 — Expand Python + WASM bindings + dependency bumps** (planned 2026-06-14)
  - **Goal:** Bump pyo3 0.28.3→0.29 + numpy 0.28→0.29, cranelift-* 0.131→0.132, wasm-bindgen 0.2.121→0.2.125; expose high-value APIs to Python/WASM; add WASM `exhaustive()`. Pure-Rust `default = []` unchanged.
  - **Design:** Cargo.toml bumps first (oxiz stays 0.2.3). pyo3 0.28→0.29: adjust GIL-release rename + `from_py_object` after `cargo check --features python`. Split python.rs → `src/python/` (mod.rs keeps `_core` ABI name). Add thin wrappers: integrate/integrate_definite, limit, solve/solve_for_all, quadrature_nd, verified root/integration, units, series/taylor, parser, discover_multi/ode/pde, lambert_w0/wm1; widen `PySymRegConfig`. WASM: add `exhaustive()`, widen config, curated in-browser subset; skip discover_pde/units in WASM v1. CI: only edit pypi-publish.yml/npm-publish.yml in place if needed — no new yaml.
  - **Files:** Cargo.toml (4 bumps); src/python/ (split dir); src/wasm.rs; src/lib.rs.
  - **Prerequisites:** dep bumps GATE binding work — bump + `cargo check` first.
  - **Tests:** `cargo check/test --features python` + `--features wasm` post-bump; `cargo build --target wasm32-unknown-unknown --features wasm`; WASM parity (quick/balanced/exhaustive); `cargo tree` no new default-feature deps.
  - **Risk:** pyo3 0.28→0.29 + numpy lockstep is the main breakage. cranelift 0.131→0.132 may rename InstBuilder methods. Splitting python.rs must keep `_core` module name.

- [x] **M1 — README & doc honesty round 2 (DOC-ONLY, DO LAST)** (planned 2026-06-14)
  - **Goal:** Doc-only: smt/mod.rs version string; README SMT-pruning wording (gated on J2); new Round-3 public API sections; test count fixed last by re-measuring with grep.
  - **Design:** smt/mod.rs:9 `"OxiZ 0.2.0"→"0.2.3"`. README ~217 "not yet ergonomic" becomes false after J2(a). README:248 gated on J2(b): if `smt_prune_solver` wired → document interval-only vs OxiZ-UNSAT distinction; else → reword to "via interval propagation (IntervalDomain), not the full SMT solver". Either way README becomes TRUE. New-API subsections for special functions, dsolve, QR/SVD, quantifiers, analytic UQ. Verify each API exists via grep before writing. Test count ABSOLUTE LAST: re-run `grep -rc '#[test]' src tests | awk -F: '{s+=$2} END{print s}'` and update both README occurrences.
  - **Files:** README.md, src/smt/mod.rs; doc comments in new modules.
  - **Prerequisites:** LAST overall. SMT wording gated by J2. `cargo doc --all-features` warning-free.
  - **Tests (meta):** `grep -c '#[test]'` total == both README numbers; `cargo doc --all-features` no warnings; every newly-documented API name matches a `pub fn/pub struct` via grep.
  - **Risk:** Test count drifts as J1/J2/J3/K1/L1 add tests → re-measure at edit time.

### Extended scope (round 4)

- [~] **N1 — BigRational/BigInt coefficient backend for `poly/`** (planned 2026-06-15)
  - **Goal:** Every `poly/` op (`add/sub/mul/pow/div_rem/gcd/resultant/discriminant/content/square_free/rational_roots`) is exact and **never returns `CoeffOverflow`** for any in-ℚ input; `resultant`/`discriminant` return exact arbitrary-precision values; `f64_to_ratio` rationalizes with no denom≤1000 cap.
  - **Design:** Switch the coefficient type directly to `num_rational::BigRational` (type alias `pub type Coeff = BigRational` in poly/mod.rs). Delete the four `checked_*` helpers. Keep `PolyError::CoeffOverflow` variant but stop constructing it. `f64_to_ratio` becomes continued-fraction/Stern–Brocot rationalization. Add `Poly::from_int_coeffs(&[i64])`/`from_ratios(Vec<BigRational>)` constructors.
  - **Files:** Cargo.toml (add `num-bigint = "0.4.6"`, `num-integer = "0.1.46"`); src/poly/{mod,univariate,multivariate,factor,sturm,tests}.rs; src/integrate.rs, src/solve_poly.rs, src/integrate_subst.rs.
  - **Prerequisites:** none (foundational; blocks N2, N3).
  - **Tests:** overflow elimination `[(10^18)x+1]^3` succeeds; exact `res(x²−2,x²−3)==1`, `disc(x²+bx+c)==b²−4c` for `b=10^10`; `f64_to_ratio` for `1/3`, `355/113`, `0.1→1/10`, `NaN→Err`; full existing poly/solve suite green.
  - **Risk:** public field type change mitigated by ergonomic constructors + 2 raw-site updates; perf acceptable (poly/ is not the symreg hot path).

- [~] **N2 — Modern univariate factorization over ℚ (Cantor–Zassenhaus + Hensel + Zassenhaus)** (planned 2026-06-15)
  - **Goal:** `Poly::factor()` returns the full irreducible factorization over ℚ for **arbitrary degree** (replacing deg≤6 Kronecker), correct on Swinnerton-Dyer / cyclotomic / many-factor inputs, polynomial-time except documented worst-case recombination.
  - **Design:** New `src/poly/modular.rs` (`ModPoly`: GF(p)/ℤ/pᵏ arithmetic via `i128`). New `src/poly/factor_zassenhaus.rs`: prime selection → distinct-degree factorization → Cantor–Zassenhaus EDF → multifactor Hensel lifting (quadratic Newton, Bézout cofactors) → Mignotte bound → Zassenhaus recombination. Delete Kronecker helpers.
  - **Files:** src/poly/modular.rs (new), src/poly/factor_zassenhaus.rs (new), src/poly/factor.rs, src/poly/mod.rs, src/poly/tests.rs.
  - **Prerequisites:** **N1** (bignum for `pᵏ` lift growth + overflow-free trial division).
  - **Tests:** `x²−2` irreducible; SD(2,3) `x⁴−10x²+1` irreducible; `(x²+1)(x²+x+1)(x³−2)` → all three; cyclotomics `Φ₈`, `x⁶−1`; degree-10 `∏(x−k)`; property re-multiply == original.
  - **Risk:** recombination blow-up → subset cap + fallback; EDF randomness → seeded in tests; prime exhaustion → fall back to `[f]` with doc note.

- [~] **N3 — Gröbner bases (Buchberger) + multivariate nonlinear system solving** (planned 2026-06-15)
  - **Goal:** Correct **reduced Gröbner basis** under lex/grlex/grevlex; **ideal membership**; nonlinear system solving so `{x²+y²=1, x=y}` returns actual roots instead of `SystemSolveResult::Nonlinear`.
  - **Design:** New `src/poly/monomial.rs` (`Monomial(Vec<u32>)` + `MonOrder{Lex,GrLex,GrevLex}`). Extend multivariate.rs with order-aware `leading_term`/`reduce`/`s_polynomial`. New `src/poly/groebner.rs` (Buchberger + both criteria + minimalize + reduce). New `src/poly/solve_system.rs` (`solve_zero_dim` via lex elimination + back-substitution). New `SystemSolveResult::NonlinearSolutions(Vec<Vec<LoweredOp>>)` variant in solve.rs.
  - **Files:** src/poly/{monomial,groebner,solve_system}.rs (new), src/poly/multivariate.rs, src/poly/mod.rs, src/solve.rs, src/lib.rs, src/python/solve.rs.
  - **Prerequisites:** **N1** (coefficient blow-up needs bignum).
  - **Tests:** `⟨x²+y²−1, x−y⟩` lex GB → solutions `(±1/√2,±1/√2)`; ideal membership (`xy∈⟨x,y⟩` true, `1∈⟨x,y⟩` false); Product-criterion prunes `⟨x²,y³⟩`; end-to-end nonlinear solve; linear regression intact.
  - **Risk:** doubly-exponential worst case → degree/pair caps + documented timeout; new `SystemSolveResult` arm is compiler-enforced.

- [~] **O1 — Symbolic linear algebra (`src/matrix.rs` new)** (planned 2026-06-15)
  - **Goal:** `Matrix` over symbolic `LoweredOp` entries with **Bareiss fraction-free exact determinant**, rref/nullspace, **Faddeev–LeVerrier charpoly** (det+inverse as by-products), eigenvalues, eigenvectors, symbolic inverse.
  - **Design:** `Matrix{rows,cols,data:Vec<LoweredOp>}` row-major. Symbolic zero-test 3-tier oracle (structural simplify → Schwartz–Zippel probing → ProbablyZero). Bareiss exact det routed through `MultiPoly` exact division. Faddeev–LeVerrier charpoly (only integer divisions). `verify()` safety net. Cap symbolic path at n≤8. Split into `src/matrix/` if > 2000 lines.
  - **Files:** src/matrix.rs (or src/matrix/ package), src/lib.rs.
  - **Prerequisites:** independent (entries are `LoweredOp`); benefits from N1/N3 for over-ℚ det.
  - **Tests:** numeric det `[[1,2],[3,4]]→−2`; symbolic `[[a,b],[c,d]]→ad−bc`; Bareiss 3×3 Vandermonde; rref/nullspace; charpoly + Cayley–Hamilton probe; eigenvalues `{±i}`; singular `→Err(SingularMatrix)`; `A·inv(A)≈I`.
  - **Risk:** symbolic zero undecidable for transcendentals → 3-tier oracle + `verify()` net; expression swell → simplify each step + n≤8 cap.

- [~] **P1 — Symbolic summation: Gosper + telescoping + Faulhaber** (planned 2026-06-15)
  - **Goal:** `sum_indefinite(term,k)` / `sum_definite(term,k,lo,hi)` → `SumResult` with closed forms for hypergeometric terms; honest `NotHypergeometric`/`NotClosedForm`.
  - **Design:** New `src/summation/{mod,gosper,faulhaber,ratio}.rs`. Full Gosper-Petkovšek a/b/c decomposition + degree-bounded key polynomial via `gaussian_eliminate`. Faulhaber via Bernoulli numbers. Dispatch: polynomial→Faulhaber; constant-ratio→geometric; rational-ratio→Gosper; else `NotHypergeometric`.
  - **Files:** src/summation/ (new), src/lib.rs, src/integrate.rs (relocate `gaussian_eliminate`).
  - **Prerequisites:** Poly (`div_rem`/`gcd`/`resultant`/`rational_roots`/`diff`), `gaussian_eliminate`. N1 strengthens exactness.
  - **Tests:** `Σk=n(n+1)/2`, `Σk²`, `Σk³`, `Σ2ᵏ=2^{n+1}−1`, `Σk·k!=(n+1)!−1` (Gosper), `Σ1/(k(k+1))=n/(n+1)` (telescope); property verify at 6 probes; `Σ1/k→NotClosedForm`.
  - **Risk:** i64 overflow → checked/f64 fallback (eased by N1); no IR variant → zero match-arm churn.

- [~] **P2 — Algebraic rewriting API: expand/factor/collect/apart/together/powsimp/logcombine** (planned 2026-06-15)
  - **Goal:** Seven never-panic `impl LoweredOp` methods: `expand`, `factor`, `collect(var)`, `apart` (partial fractions), `together`, `powsimp`, `logcombine`.
  - **Design:** New `src/rewrite/{mod,expand,factor,collect,apart,logexp}.rs`. `expand` via `MultiPoly` round-trip with local `EXPAND_MAX_TERMS` (~4096). `apart` = shared `partial_fractions` helper lifted from integrate.rs:651-890, upgraded to `Poly::factor`. `integrate_rational` refactored to call shared `partial_fractions`. `logcombine` delegates to `logexp.rs`.
  - **Files:** src/rewrite/ (new), src/lib.rs, src/integrate.rs (refactor + relocate `gaussian_eliminate`).
  - **Prerequisites:** **N2** `Poly::factor` (for `factor`/`apart`); PFD core from integrate.rs.
  - **Tests:** `expand((x+1)²)`, `factor(x²−1)`, `apart(1/((x²+1)(x²+4)))` succeeds (was None), apart→together round-trip, `powsimp(x²·x³)=x⁵`, `logcombine(ln2+ln3)=ln6`; 4 rational-integral tests unchanged.
  - **Risk:** `apart` refactor regression → same numeric-verify gate; expand blow-up → local cap.

- [~] **P3 — Trig/exp-log simplify identities + lightweight assumptions** (planned 2026-06-15)
  - **Goal:** simplify gains opt-in Pythagorean (`sin²+cos²→1`, `1+tan²→1/cos²`, `cosh²−sinh²→1`), double/half-angle, product-to-sum, log/exp-combine — gated by an **Assumptions** layer. Existing simplify tests green.
  - **Design:** New `src/assumptions.rs` (`VarAssumption{positive,nonnegative,real,integer}`, `Assumptions`) + `LoweredOp::simplify_with(&Assumptions)`. `apply_identities` runs inside simplify, value-checked at `PROBE_POINTS`. Expanding rules are opt-in via `RewriteFlags`. Split trig collector into `src/lower_simplify_identities.rs`.
  - **Files:** src/assumptions.rs (new), src/rewrite/logexp.rs (shared with P2), src/lower_simplify_identities.rs (new), src/lower_simplify.rs, src/lib.rs.
  - **Prerequisites:** `PROBE_POINTS`/`node_count`; P2 `logexp.rs` (shared).
  - **Tests:** `sin²x+cos²x→1`; `cosh²−sinh²→1`; product-to-sum opt-in; `√(x²)` with `nonnegative`; regression: existing lower_simplify tests unchanged.
  - **Risk:** identities perturbing canonical form → expanding rules opt-in + value-check every commit.

- [~] **P4 — Integration depth: trig powers/products + Weierstrass + by-parts solve-for-I** (planned 2026-06-15)
  - **Goal:** `∫sinⁿ/cosⁿ/tanⁿ/secⁿ`, `∫sinᵐcosⁿ`, `∫sin ax cos bx`, `∫R(sin,cos)` via Weierstrass `t=tan(x/2)`, and `∫eˣsin x` solve-for-I. Tests asserting `Unsupported` for these flip to `Closed`.
  - **Design:** New `src/integrate_trig.rs`. Reduction formulas (cap n≤20). Weierstrass: predicate → substitute → `partial_fractions`/`integrate_rational` (P2) → back-substitute. By-parts solve-for-I replaces cycle-hash bail (integrate.rs:213-220).
  - **Files:** src/integrate_trig.rs (new), src/integrate.rs, src/lib.rs.
  - **Prerequisites:** **P2** shared `partial_fractions`; product-to-sum table shared with P3.
  - **Tests (flip Unsupported→Closed):** `∫sin²x=x/2−sin2x/4`, `∫tan²x=tan x−x`, `∫1/(2+cos x)` (Weierstrass), `∫eˣsin x=eˣ(sin x−cos x)/2`. **`∫sin(x²)`/`∫exp(x²)` MUST stay Unsupported.**
  - **Risk:** update Unsupported-asserting tests; solve-for-I sign errors caught by numeric-verify.

- [~] **P5 — Laurent series + indeterminate-form L'Hôpital + series-based limits** (planned 2026-06-15)
  - **Goal:** `laurent(wrt,center,order)` so singular centers no longer hard-error; limit normalizes `0·∞,∞−∞,0⁰,1^∞,∞⁰`; honest `Indeterminate`/`DoesNotExist`.
  - **Design:** New `src/series_laurent.rs` (`Series{center,lo_pow:i32,coeffs}`). Pole-order detection + regular-part Taylor expansion. `normalize_indeterminate` in limit.rs. `limit_via_series` ordered L'Hôpital → series → numeric probing. Branch points → `Err(BranchPoint)`/`Err(EssentialSingularity)`.
  - **Files:** src/series_laurent.rs (new), src/series.rs, src/limit.rs, src/lib.rs (new `EmlError` variants).
  - **Prerequisites:** existing `taylor`; **P2 `together`** (∞−∞).
  - **Tests:** `laurent(1/x,0)` leading `x⁻¹`; `laurent(1/sin x,0)→x⁻¹+x/6`; `lim x·ln x=0` (0·∞); `lim(1+1/x)^x=e` (1^∞); `lim sin(1/x)→DoesNotExist`; `laurent(ln x,0)→Err(BranchPoint)`.
  - **Risk:** pole-order noise → symbolic cross-check + cap m≤10; existing `test_taylor_undefined_at_point` stays valid.

- [~] **R1 — Parallel fitness evaluation (root-parallel MCTS + intra-island GA fitness)** (planned 2026-06-15)
  - **Goal:** Under `feature=parallel`, MCTS rollouts and intra-island GA fitness run on the rayon pool; with fixed `config.seed` results are **byte-identical** to single-thread.
  - **Design:** MCTS root-parallel: batch simulation phase via `chunk.par_iter().map(rollout_once)`, backprop in deterministic chunk order. GA two-phase map-then-merge: read-only cache lookup before parallel region, `par_iter` `optimize_topology`, write back in ascending-slot order. All `par_iter` `#[cfg(feature="parallel")]` with sequential twin.
  - **Files:** src/symreg/mcts.rs (extract `rollout_once`, chunked parallel sim), src/symreg/evolution.rs (two-phase parallel fitness).
  - **Prerequisites:** none (rayon present).
  - **Tests:** parallel==sequential bit-for-bit (seed=42); thread-count invariance (`RAYON_NUM_THREADS=1` vs `8`).
  - **Risk:** non-associative float → never reduce floats across tasks; verified `optimize_topology` has no global state.

- [~] **R2 — Proper NSGA-II multi-objective symbolic regression** (planned 2026-06-15)
  - **Goal:** Selectable NSGA-II mode returning full non-dominated front annotated with (rank, crowding). Existing `pareto_front`/`discover_pareto` kept.
  - **Design:** New `src/symreg/nsga2.rs`. Fast-nondominated-sort O(MN²) + crowding distance + crowded-comparison + (μ+λ) selection. Expose `discover_nsga2(...)->Vec<RankedFormula>`.
  - **Files:** src/symreg/nsga2.rs (new), src/symreg/mod.rs.
  - **Prerequisites:** R1 recommended, not required.
  - **Tests:** textbook fast-nondominated-sort ranks; rank-0 == `pareto_front` on same pool; crowding boundary `+∞`; determinism parallel==sequential.
  - **Risk:** crowding div-by-zero → explicit skip; NaN MSE → treat as worst.

- [~] **R3 — JIT batch/SIMD evaluation path** (planned 2026-06-15)
  - **Goal:** `JitFn::call_batch(rows,n_rows,stride,out)` emits a Cranelift internal loop (one compile, N rows); `call_batch_parallel` splits via rayon. Scalar `call` unchanged. Transcendentals stay scalar host calls.
  - **Design:** Tier 1: compile `fn(in_ptr,n_rows,stride,out_ptr)` reusing `emit_op` with per-iteration `row_base`; `header/body/exit` block loop. Tier 2 (optional): vectorize only `{Const,Var,Add,Sub,Mul,Div,Neg,int-Pow}` via `is_vectorizable`.
  - **Files:** src/jit.rs (batch codegen + `call_batch`/`call_batch_parallel`).
  - **Prerequisites:** none. Coordinate with T3 (both touch jit.rs).
  - **Tests:** `call_batch` == per-row `call` (1000 rows, 0 ULP); parallel == batch (thread-invariant); malformed stride/len asserts.
  - **Risk:** Cranelift block-sealing → strict `seal_block` discipline; n_rows∈{0,1,2,large} test.

- [~] **S1 — Incremental SMT (solver reuse + push/pop) + MaxSMT + unsat-core** (planned 2026-06-15)
  - **Goal:** Symreg pruner reuses one live OxiZ `Solver` via push/pop; `EmlSmtSolver::max_smt` maximizes satisfied soft constraints; `unsat_core` returns a minimal infeasible subset.
  - **Design:** New `IncrementalEmlSolver` (persistent `TermManager`+`Solver`+`var_terms`); per candidate `push()`→encode→assert→check→`pop()`. MaxSMT via `oxiz::opt` RC2 behind `smt-opt = ["smt","oxiz/optimization"]` feature + pure greedy fallback under base `smt`. Unsat-core via `minimize_unsat_core`. New `smt/{incremental,maxsmt,core,encode}.rs`.
  - **Files:** src/smt/{incremental,maxsmt,core}.rs (new), src/smt/oxiz_backend.rs, src/smt/mod.rs, src/symreg/smt_prune.rs, Cargo.toml (`smt-opt`).
  - **Prerequisites:** none. **USER NOTE:** `smt-opt` feature enlarges build (pulls `oxiz-opt`); base incremental/push-pop needs no new feature.
  - **Tests:** push/pop verdicts == N independent `check_sat`; push/pop hygiene; unsat-core minimality; MaxSMT optimum; solver-construction-count == 1 over ≥50 topologies.
  - **Risk:** OxiZ `Solver` not `Sync` → stays in sequential phase (no R1 conflict).

- [~] **T1 — Full Python + WASM API coverage** (planned 2026-06-15)
  - **Goal:** `oxieml._core` (ABI name preserved) exposes thin `unwrap`-free wrappers over unexposed public surface including Round-4 APIs (NSGA-II, incremental/MaxSMT, JIT batch, trigamma, summation, matrix). WASM exposes curated browser subset.
  - **Design:** `map_err`/`js_err` helpers. New `src/python/{jit,smt,verified,units,pareto,discovery,algebra}.rs` (each < 300 lines), each `add_class` cfg-gated on both `python` and its capability feature. WASM excludes JIT/SMT/PDE/SINDy (cranelift + bundle size).
  - **Files:** src/python/{jit,smt,verified,units,pareto,discovery,algebra}.rs (new), src/python/mod.rs, src/wasm/ (split mod/symreg/algebra).
  - **Prerequisites:** **LAST** — depends on R2/R3/S1/T3 public APIs existing.
  - **Tests:** `Python::with_gil` smoke per `Py*` class; `wasm-bindgen-test` round-trips; Python import smoke.
  - **Risk:** ABI break → keep `_core` + `abi3` tags; `unwrap` creep → clippy `-D warnings`.

- [~] **T2 — clap-based CLI + REPL + subcommands** (planned 2026-06-15)
  - **Goal:** `src/bin/oxieml` uses clap 4 (derive) with subcommands `eval, lower, grad, integrate, solve, simplify, symreg, smt, series, limit, repl`. REPL default is FFI-free (std stdin). Raw-mode editing is opt-in `repl-rich`. Legacy flags preserved via shim.
  - **Design:** `clap = { version = "4", features = ["derive"] }` behind `cli` feature (`[[bin]] required-features=["cli"]`). Legacy shim detects old flags before clap parsing. Default `repl` = std `read_line` loop. `repl-rich` → crossterm/reedline (libc/termios FFI, OS-syscall boundary, not C/C++/Fortran — kept out of default build).
  - **Files:** src/bin/oxieml.rs, src/bin/oxieml/{repl,integrate,solve,simplify,series,limit,smt}.rs (new), Cargo.toml.
  - **Prerequisites:** S1 helps `smt` subcommand.
  - **Tests:** `assert_cmd`: `eval "E(1,1)"`, legacy `--lower` shim, piped REPL `echo "E(1,1)\n:quit" | oxieml repl`, `--help` exit 0.
  - **Risk:** backward-compat → legacy shim + CHANGELOG; clap `cli`-gated so lib purity intact; no new CI yaml.

- [x] **T3 — Trigamma (ψ¹) derivative — close the silent grad-through-digamma gap** (planned 2026-06-15)
  - **Goal:** `trigamma(x)` in special.rs (pure Rust); new `LoweredOp::Trigamma` (+ `OxiOp`) across ~22 match sites; `d/dx digamma(f)=trigamma(f)·f'` wired into lower_grad.rs:384, autodiff.rs:158/297. **CORRECTNESS fix** — run SOLO (only Round-4 core-enum change).
  - **Design:** NaN/pole guards; reflection `ψ¹(x)=π²/sin²(πx)−ψ¹(1−x)` for x<0.5; recurrence up-shift while x<6; asymptotic `1/x+1/2x²+ΣB₂ₙ/x^{2n+1}` to ~1e-12. ~22 sites each clone the adjacent `Digamma` arm. Tetragamma ψ² documented as honest gap (explicit comment, not silent 0).
  - **Files:** src/special.rs + ~22 match sites across lower/mod.rs, lower/oxiblas.rs, lower_grad.rs:384 (the fix), autodiff.rs:158+297, lower_simplify.rs, lower/display.rs, lower_interval.rs, lower_units.rs, compile.rs, jit.rs, limit.rs, integrate.rs, integrate_subst.rs, ode.rs, solve.rs, tensorlogic.rs, simd_eval.rs, symreg/constants.rs, symreg/topology.rs.
  - **Prerequisites:** none — **land solo** before R1/R3 (both also touch jit.rs/simd_eval.rs).
  - **Tests:** `trigamma(1)=π²/6≈1.6449340668`; recurrence `ψ¹(x)−ψ¹(x+1)=1/x²`; reflection identity; **the fix**: `Digamma(Var0).grad(0)` matches central-difference at x∈{1.5,2.5,3.0} (currently returns 0); JIT/SIMD parity; `lgamma''=trigamma`.
  - **Risk:** missing match arm → compile error (non-exhaustive forces completeness); tetragamma documented honestly; no `#[allow]`, no unwrap.
