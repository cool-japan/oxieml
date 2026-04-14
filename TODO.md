# OxiEML TODO

## Phylogenetic Tree (Paper Figure 1) — Canonical Constructions

All functions from the paper's phylogenetic tree are now implemented:

- [x] **Core**: eml, 1
- [x] **Basic**: exp, ln, e (Euler's number)
- [x] **Arithmetic**: add (+), sub (-), mul (*), div (/), neg (-x)
- [x] **Powers**: pow (x^y), square (x^2), sqrt, reciprocal (1/x), abs
- [x] **Trig**: sin, cos, tan
- [x] **Inverse trig**: arcsin, arccos, arctan (via complex logarithms)
- [x] **Hyperbolic**: sinh, cosh, tanh
- [x] **Inverse hyperbolic**: arcsinh, arccosh, arctanh
- [x] **Constants**: pi (iπ), zero (0), neg_one (-1), neg_two (-2), imag_unit (i), nat(n)

## CLI Tool

- [x] **parser.rs**: Recursive descent parser for `E(x,y)` / `eml(x,y)` notation.
- [x] **src/bin/oxieml.rs**: CLI evaluator with constant matching, complex eval, lowering.
- [x] **--help / --version flags**: Standard `--help`/`-h` and `--version`/`-V` CLI flags.
- [x] **Verified**: User's 193-node depth-34 EML expression correctly evaluates to π.

## Code Quality

- [x] **canonical.rs: Clean up doc comments** — Replaced 250+ lines of scratch-work
      derivations with clean 5-15 line explanations per function.
- [x] **compile.rs:116: Trailing space in Neg codegen** — Fixed.
- [x] **grad.rs:58: Unused tape in forward()** — Fixed (`_tape` prefix).
      TapeEntry variants simplified (removed unused indices).

## Functionality Implemented

- [x] **simplify.rs: Real simplification** — Implemented: ln(exp(x))→x, exp(ln(x))→x,
      common subexpression sharing via structural hashing, cache-based dedup.
- [x] **lower.rs: Expanded pattern recognition** — Added: subtraction pattern
      `eml(ln(x), eml(y, 1))→x-y`, exp-of-ln elimination, ln structure matching.
- [x] **symreg.rs: Fixed duplicate topology generation** — Rewrote enumeration with
      three disjoint cases (both-at-max, left-at-max, right-at-max) eliminating all duplicates.
- [x] **canonical.rs: Added zero() constructor** — `0 = ln(1)` as EML tree.

## Remaining Enhancements

- [x] **lower.rs: LoweredOp::to_oxiblas_ops()** — Flat post-order IR (`OxiOp`
      enum) in lower.rs; consumed by SIMD evaluator and scalar stack machine.

- [x] **simd feature** — Real SIMD via oxiblas-core 0.2.1. Runtime dispatches
      to F64x2/F64x4 per arch. Combines with `parallel` for SIMD-per-worker.

- [x] **parallel feature** — Rayon-based parallel evaluation via `parallel` feature flag. `eval_batch` uses `par_iter` for batches ≥ 128 points.

- [x] **smt.rs: Constraint propagation** — `IntervalDomain` with forward
      exp/ln propagation and conflict detection (always-on). `EmlSmtSolver`
      with OxiZ 0.2.0 LRA backend via secant/tangent linear relaxation
      (feature-gated `smt`). Can now prove UNSAT for cases interval bisection
      alone cannot (e.g., `ln(x) > 0` on negative domain).

- [x] **symreg.rs: Parallel topology evaluation** — Implemented via `#[cfg(feature = "parallel")]` rayon `par_iter` in `discover()`. Each topology optimization runs on its own rayon worker with thread-local RNG.

- [x] **symreg.rs: Pruning heuristics** — `dedupe_by_semantics` uses
      `lower().simplify()` + `structural_hash`. NOTE: EML non-commutative
      → only catches `exp(ln(x))=x` simplifications, ~0.0002% reduction at
      depth 4. Correct, but structural dedup is fundamentally limited here.

- [x] **compile.rs: Multi-point codegen** — `compile_to_rust_batch()` emits a `_batch(data: &[Vec<f64>]) -> Vec<f64>` function using `par_iter` (when `parallel` feature is active) or sequential `iter`.

- [ ] **sin/cos precision** — Current trig implementations produce deep trees
      that may lose precision through numerical noise. Could improve by using
      more direct complex evaluation paths.

- [ ] **SciRS2 integration** — Blueprint specifies a `symbolic_regression()`
      adapter that takes `scirs2::DataFrame` input/target columns and returns
      `Vec<DiscoveredFormula>` with lowered pretty-printed expressions.

- [ ] **Physics crate data pipeline** — Blueprint envisions oxiphysics /
      oxiphoton / oxigrid / spintronics feeding simulation data into
      `SymRegEngine::discover()`, lowered to `LoweredOp`, then emitted via
      `compile_to_rust()` or `to_oxiblas_ops()` for production evaluation.

## Planned: TensorLogic Integration

**Status:** Not implemented — design phase.

EML's uniform rewriting system and TensorLogic's logic-to-tensor compilation
are natural counterparts. OxiEML discovers closed-form formulas from data;
TensorLogic compiles logical rules into einsum graphs for neurosymbolic AI.
Connecting them enables a **data-driven formula discovery → neurosymbolic
pipeline** workflow.

**Dependency strategy (cycle-safe):**

OxiEML may become a SciRS2 subcrate in the future, while TensorLogic's
execution layer (`tensorlogic-scirs-backend`, `tensorlogic-train`) depends on
SciRS2. To avoid circular dependencies, OxiEML depends **only** on
`tensorlogic-ir` — the engine-agnostic AST/IR layer that has **zero SciRS2
dependencies** (verified: deps are serde, serde_json, oxicode, chrono,
thiserror only).

```
SciRS2 ─may contain→ OxiEML ─optional→ tensorlogic-ir  (no SciRS2 dep)
                                              │
TensorLogic ─depends→ SciRS2    tensorlogic-ir is SciRS2-free ✓
```

No cycle. The `tensorlogic-compiler` and `tensorlogic-adapters` crates are
also SciRS2-free and may be used if needed.

**Scope (thin, optional, feature-gated):**

- [ ] **`tensorlogic` feature (optional)** — Depends on `tensorlogic-ir` only.
      `LoweredOp` → `TLExpr` conversion: export discovered formulas as
      TensorLogic logical rules (predicates over arithmetic expressions).
      The lowered IR (Exp/Ln/Add/Sub/Mul/Div) maps directly to `TLExpr`
      arithmetic nodes.

- [ ] **Rewrite rule export** — Register EML canonical identities
      (`exp(ln(x)) → x`, `ln(exp(x)) → x`, `eml(0, eml(eml(0, x), 0)) → x`,
      etc.) as TensorLogic rewrite rules, enabling the TL compiler to exploit
      EML algebraic simplifications during einsum graph optimization.

- [ ] **TLExpr → EmlTree** (reverse direction) — Parse a `TLExpr` arithmetic
      sub-expression back into an EML tree for further symbolic regression
      refinement or EML-native simplification/lowering.

- [ ] **SymReg → TL training pipeline** — This integration lives on the
      **TensorLogic side** (not in OxiEML), since `tensorlogic-train` already
      depends on SciRS2. TensorLogic would add an optional `oxieml` feature
      to accept `DiscoveredFormula` as training constraints. No OxiEML→SciRS2
      dependency is introduced.

- [ ] The core computational path (tree evaluation, OxiOp stack machine,
      simd_eval) stays entirely inside oxieml. No dependency on
      `tensorlogic-compiler`, `tensorlogic-scirs-backend`, or
      `tensorlogic-train` in any feature set.

**Non-goals:**
- Compiling full EML trees (pre-lowering) to einsum — the uniform binary tree
  structure is too deep and repetitive for efficient tensor contraction.
- Replacing OxiEML's own simplify/lower pipeline with TensorLogic's compiler.
- Running EML evaluation through the TensorLogic executor.
- Depending on any TensorLogic crate that transitively pulls in SciRS2.
