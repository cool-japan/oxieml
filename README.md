# OxiEML

**All elementary functions from a single binary operator.**

A Pure Rust crate that implements the EML operator `eml(x, y) = exp(x) - ln(y)`
and builds uniform binary trees expressing **all elementary functions** using only
this operator and the constant `1`.

Based on [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — *"All elementary
functions from a single binary operator"* by Andrzej Odrzywolek (Jagiellonian
University, Institute of Theoretical Physics).

## Key Capabilities

1. **Uniform Tree Representation** — Every elementary function (exp, ln, sin, cos,
   +, -, *, /, ^, sqrt, abs, ...) is expressed via the grammar `S -> 1 | eml(S, S)`.

2. **Symbolic Regression** — Discover closed-form mathematical formulas from
   input/output data using gradient-based search over EML tree topologies.

3. **Lowering & Code Generation** — Convert discovered EML trees to standard
   operation trees for efficient evaluation, pretty-printing, and Rust code emission.

4. **CLI Tool** — Parse, evaluate, and generate EML expressions from the command line.

5. **SMT Integration** — Constraint solving via EML tree interval narrowing
   (feature-gated for oxiz integration).

## CLI Tool

The `oxieml` CLI can evaluate EML expressions, generate EML from function names,
and verify claims about mathematical constants.

```bash
# Evaluate an EML expression
oxieml "E(1, 1)"
#=> MATCH: e (Euler's number) = 2.718281828459045

# Generate EML from a function/constant name
oxieml -g pi
#=> E(1,E(E(1,E(E(1,E(E(1,E(1,E(1,1))),1)),E(E(1,1),1))),1))
#=> MATCH: Im ~ pi (diff = 0.00e0)

oxieml -g e
#=> E(1,1)

oxieml -g sin x0=0.5
#=> Result: 0.4794255386042034

# Evaluate with variables
oxieml "E(x0, 1)" x0=2.0
#=> Result: 7.38905609893065  (= exp(2))

# Read from file
oxieml --file expression.txt

# List all available functions and constants
oxieml -l

# Show help / version
oxieml --help
oxieml --version
```

If the input is not a valid EML expression, the CLI auto-detects function names:

```bash
oxieml pi          # same as: oxieml -g pi
oxieml sin         # generates sin(x0) template
```

## Quick Start (Library)

```rust
use oxieml::{EmlTree, Canonical, EvalCtx};

// Build exp(x) = eml(x, 1)
let x = EmlTree::var(0);
let exp_x = Canonical::exp(&x);

// Evaluate at x = 1.0 -> e
let ctx = EvalCtx::new(&[1.0]);
let result = exp_x.eval_real(&ctx).unwrap();
assert!((result - std::f64::consts::E).abs() < 1e-10);

// Euler's number: eml(1, 1) = exp(1) - ln(1) = e
let e = Canonical::euler();
println!("{}", e); // "eml(1, 1)"

// Negation, addition, multiplication — all from eml and 1
let y = EmlTree::var(1);
let sum = Canonical::add(&x, &y);
let product = Canonical::mul(&x, &y);

// Lower to standard operations for efficient evaluation
let lowered = exp_x.lower();
println!("{}", lowered.to_pretty()); // "exp(x0)"
let fast_result = lowered.eval(&[1.0]);

// Generate Rust source code
let code = oxieml::compile::compile_to_rust(&exp_x, "my_exp");
println!("{code}");
```

## Parser

Parse EML expressions from strings and convert back:

```rust
use oxieml::parser::{parse, to_compact_string};

// Parse E(x, y) notation
let tree = parse("E(E(1, 1), 1)").unwrap();
assert_eq!(tree.depth(), 2);

// Also accepts eml(x, y) notation
let tree = parse("eml(E(1, x0), 1)").unwrap();

// Convert back to compact string
let compact = to_compact_string(&tree);
assert_eq!(parse(&compact).unwrap(), tree); // roundtrip
```

## Symbolic Regression

```rust
use oxieml::symreg::{SymRegConfig, SymRegEngine};

// Generate data from an unknown function
let inputs: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1]).collect();
let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();

let config = SymRegConfig {
    max_depth: 2,
    learning_rate: 1e-2,
    tolerance: 1e-8,
    ..Default::default()
};

let engine = SymRegEngine::new(config);
let formulas = engine.discover(&inputs, &targets, 1).unwrap();

println!("Best formula: {}", formulas[0].pretty);
println!("MSE: {:.2e}", formulas[0].mse);
```

## SMT / Constraint Solving

With the `smt` feature, oxieml integrates [OxiZ](https://crates.io/crates/oxiz) 0.2
as a backend for deciding EML constraints. The solver uses **interval propagation**
(EML-aware forward/backward rules for exp/ln) followed by **linear relaxation**
(secant + tangent bounds) for OxiZ's LRA theory.

```rust,ignore
use oxieml::{EmlTree, Canonical, EmlConstraint, EmlSmtSolver, SmtResult};

// Constraint: exp(x) > 0 — trivially true for all x
let x = EmlTree::var(0);
let one = EmlTree::one();
let exp_x = EmlTree::eml(&x, &one);
let c = EmlConstraint::GtZero(exp_x);

let solver = EmlSmtSolver::new(vec![(-10.0, 10.0)]);
match solver.check_sat(&c).unwrap() {
    SmtResult::Sat(sol) => println!("SAT: x = {}", sol.assignments[0]),
    SmtResult::Unsat => println!("UNSAT — impossible"),
    SmtResult::Unknown => println!("unknown"),
}
```

The `EmlSmtSolver` can prove **UNSAT** for cases the legacy `EmlNraSolver`
(interval bisection) cannot — e.g., `ln(x) > 0` with `x ∈ [-2, -1]` (ln
undefined for non-positive reals). It falls back to bisection on OxiZ-tightened
bounds to extract concrete SAT witnesses, since extracting real-valued models
from OxiZ 0.2 is not yet ergonomic.

Enable with:

```toml
[dependencies]
oxieml = { version = "0.1", features = ["smt"] }
```

The `IntervalDomain` type is always available (no feature) for lightweight
propagation use-cases.

## Canonical Constructions (Complete Phylogenetic Tree)

All functions from the paper's phylogenetic tree (Figure 1) are implemented:

### Table 1: Basic Operations

| Function    | EML Construction               | Depth |
|-------------|--------------------------------|-------|
| `exp(x)`    | `eml(x, 1)`                   | 1     |
| `e`         | `eml(1, 1)`                   | 1     |
| `ln(x)`     | `eml(1, eml(eml(1, x), 1))`   | 3     |
| `-x`        | via `(e-x) - e` composition   | 6     |
| `0`         | `ln(1)`                        | 3     |

### Table 2: Arithmetic

| Function    | EML Construction               | Depth |
|-------------|--------------------------------|-------|
| `x + y`     | `sub(x, neg(y))`              | ~12   |
| `x - y`     | `eml(ln(x), eml(y, 1))`       | ~7    |
| `x * y`     | `exp(ln(x) + ln(y))`          | ~14   |
| `x / y`     | `exp(ln(x) - ln(y))`          | ~14   |
| `x ^ y`     | `exp(y * ln(x))`              | ~18   |
| `1/x`       | `exp(-ln(x))`                 | ~10   |
| `x^2`       | `pow(x, 2)`                   | deep  |

### Table 3: Trigonometric

| Function      | EML Construction                       | Depth |
|---------------|----------------------------------------|-------|
| `pi` (iπ)     | `ln(-1)` in complex domain            | 9     |
| `sin(x)`      | `(exp(ix) - exp(-ix)) / 2i`           | ~52   |
| `cos(x)`      | `(exp(ix) + exp(-ix)) / 2`            | ~52   |
| `tan(x)`      | `sin(x) / cos(x)`                     | deep  |

### Table 4: Inverse Trigonometric

| Function      | EML Construction                              |
|---------------|-----------------------------------------------|
| `arcsin(x)`   | `-i * ln(ix + sqrt(1 - x^2))`                |
| `arccos(x)`   | `-i * ln(x + i * sqrt(1 - x^2))`             |
| `arctan(x)`   | `(-i/2) * ln((1 + ix) / (1 - ix))`           |

### Table 5: Hyperbolic

| Function    | EML Construction                |
|-------------|---------------------------------|
| `sinh(x)`   | `(exp(x) - exp(-x)) / 2`      |
| `cosh(x)`   | `(exp(x) + exp(-x)) / 2`      |
| `tanh(x)`   | `sinh(x) / cosh(x)`           |

### Table 6: Inverse Hyperbolic

| Function      | EML Construction                        |
|---------------|-----------------------------------------|
| `arcsinh(x)`  | `ln(x + sqrt(x^2 + 1))`               |
| `arccosh(x)`  | `ln(x + sqrt(x^2 - 1))`               |
| `arctanh(x)`  | `(1/2) * ln((1 + x) / (1 - x))`       |

### Table 7: Other Functions & Constants

| Function    | EML Construction         |
|-------------|--------------------------|
| `sqrt(x)`   | `x^0.5`                 |
| `abs(x)`    | `sqrt(x^2)`             |
| `nat(n)`    | `1 + 1 + ... + 1`       |
| `-1`        | `neg(1)`                |
| `-2`        | `neg(nat(2))`           |
| `i`         | `exp(iπ/2)`             |

## Architecture

```
Discovery Phase              Execution Phase
─────────────────           ──────────────────
EML tree space     lower()  Standard ops
S -> 1 | eml(S,S) -------> Add/Sub/Mul/Exp/Ln...
     |                           |
     | Adam optimizer            | to_pretty()
     | (symreg)                  | compile_to_rust()
     |                           | eval()
  DiscoveredFormula         Fast evaluation

     parse()                to_compact_string()
"E(1,1)" -----> EmlTree ---------> "E(1,1)"
                   |
                   | -g pi / -g sin
                   |
              CLI evaluation & constant matching
```

## Module Overview

| Module        | Purpose                                           |
|---------------|---------------------------------------------------|
| `tree`        | `EmlNode`/`EmlTree` — Arc-shared uniform binary trees |
| `eval`        | Stack-machine evaluation (real, complex, batch)   |
| `grad`        | Automatic differentiation for parameter optimization |
| `canonical`   | Complete phylogenetic tree: 30+ elementary functions |
| `parser`      | Parse `E(x,y)` / `eml(x,y)` notation, roundtrip  |
| `simplify`    | EML tree algebraic simplification + CSE            |
| `lower`       | EML -> standard operation trees + pretty-print    |
| `compile`     | EML -> Rust source code generation                |
| `symreg`      | Symbolic regression engine (topology enum + Adam) |
| `smt`         | [feature: smt] Constraint solving (interval propagation + OxiZ LRA via linear relaxation) |
| `simd_eval`   | [feature: simd] SIMD batch evaluation via oxiblas-core |
| `error`       | Error types                                       |

## Features

```toml
[features]
default = []
smt = ["dep:oxiz", "dep:num-rational"]   # OxiZ 0.2 backend + interval propagation
simd = ["dep:oxiblas-core"]    # SIMD batch evaluation (F64x2/F64x4 via oxiblas-core 0.2)
parallel = ["dep:rayon"]       # Rayon-based parallel discovery & batch eval
```

Combine `simd,parallel` for SIMD-per-worker batch evaluation.

## Performance

Measured on Apple M1 (8-core, NEON 128-bit), M1 MacBook Air, 2026-04:

**Speedup from `parallel` feature** (RAYON_NUM_THREADS=1 → 8):

| Workload | 1 thread | 8 threads | Speedup |
|---|---|---|---|
| `eval_batch` 10K points (exp tree walk) | 436 µs | 235 µs | **1.85×** |
| `lowered_eval_batch` 100K points (SIMD) | 2.71 ms | 682 µs | **3.97×** |
| `symreg_discover` (topology optimization) | 73.7 ms | 17.3 ms | **4.26×** |

**Speedup from `simd` feature** (10K-point batch, LoweredOp IR):

| Variant | time | Speedup |
|---|---|---|
| Scalar stack machine | 159.8 µs | 1.0× |
| SIMD (F64x2 NEON via oxiblas-core) | 57.0 µs | **2.80×** |

Parallelism helps most for coarse-grained work (symreg topology optimization).
SIMD gives ~2.8× on batch evaluation regardless of batch size. Combining both
scales near-linearly on large batches (100K+ points).

## Design Decisions

- **`Arc<EmlNode>`** — O(1) subtree sharing during symbolic regression
- **Stack-machine evaluator** — Post-order traversal avoids recursion overflow
  on deep trees (sin alone needs 543 nodes)
- **Complex64 internally** — Trig functions and π require `ln(-1) = iπ`;
  complex eval is an internal detail, API is real-valued
- **Discovery vs execution separation** — EML trees for search, lowered ops for speed
- **Parser roundtrip** — `parse(to_compact_string(tree)) == tree`
- **Pure Rust, zero FFI** — Deps: `num-complex`, `rand`;
  optional: `rayon` (parallel), `oxiblas-core` (simd), `oxiz` + `num-rational` (smt)

## Test Coverage

173 tests covering:
- All canonical constructions (Tables 1-7)
- Evaluation: real, complex, batch, error cases
- Parser: roundtrip, E/eml notation, error handling
- Symbolic regression: exp, constant, linear discovery
- Lowering + simplification: pattern matching, constant folding
- SIMD/parallel equivalence (scalar vs SIMD vs rayon results match to 1e-12)
- Integration: EML <-> lower <-> compile consistency
- SMT/constraint solving: interval propagation, OxiZ backend SAT/UNSAT, witness verification

```bash
cargo nextest run --all-features    # 173 tests
cargo clippy --all-targets --all-features -- -D warnings   # zero warnings
cargo bench --features simd,parallel                       # criterion benchmarks
```

## References

- Paper: Andrzej Odrzywolek, *"All elementary functions from a single binary operator"*,
  [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (v2: 2026-04-04),
  Jagiellonian University, Institute of Theoretical Physics

## COOLJAPAN Ecosystem

OxiEML is part of the **COOLJAPAN Pure Rust Ecosystem** — one of the largest pure-Rust sovereignty stacks in existence, comprising 660 crates, ~26M SLoC, and 350,000+ passing tests across 50+ production-grade libraries. All projects enforce `fail0 + Clippy0` with zero C/Fortran dependencies by default.

### Core Projects

| Domain | Project | Description |
|--------|---------|-------------|
| Scientific Computing | [SciRS2](https://github.com/cool-japan/scirs) | Complete NumPy/SciPy/scikit-learn replacement (3M SLoC) |
| Scientific Computing | [NumRS2](https://github.com/cool-japan/numrs) | High-performance numerical computing in Rust |
| Scientific Computing | [QuantRS2](https://github.com/cool-japan/quantrs) | Full quantum computing framework |
| Deep Learning | [ToRSh](https://github.com/cool-japan/torsh) | PyTorch-compatible framework with native sharding |
| LLM | [OxiBonsai](https://github.com/cool-japan/oxibonsai) | Pure Rust 1-Bit LLM inference engine for PrismML Bonsai models |
| GPU (CUDA) | [OxiCUDA](https://github.com/cool-japan/oxicuda) | NVIDIA CUDA Toolkit with type-safe, memory-safe Rust code |
| Media & CV | [OxiMedia](https://github.com/cool-japan/oximedia) | FFmpeg + OpenCV replacement (106 crates) |
| Geospatial | [OxiGDAL](https://github.com/cool-japan/oxigdal) | Pure Rust GDAL replacement (cloud-native, full CRS & formats) |
| Semantic Web | [OxiRS](https://github.com/cool-japan/oxirs) | SPARQL 1.2, GraphQL, Digital Twin (Apache Jena replacement) |
| Physics | [OxiPhysics](https://github.com/cool-japan/oxiphysics) | Unified physics engine — Bullet/OpenFOAM/LAMMPS/CalculiX replacement |
| Formal Verification | [OxiLean](https://github.com/cool-japan/oxilean) | Memory-safe interactive theorem prover (Lean 4 inspired) |
| Formal Verification | [OxiZ](https://github.com/cool-japan/oxiz) | High-performance SMT solver (Z3 replacement) |
| Legal Technology | [Legalis-RS](https://github.com/cool-japan/legalis-rs) | Legal statute parser, analyzer & simulator |
| Digital Humans | [OxiHuman](https://github.com/cool-japan/oxihuman) | Privacy-first parametric human body generator (WASM/WebGPU) |
| Signal Processing | [Kizzasi](https://github.com/cool-japan/kizzasi) | Rust-native AGSP for continuous audio, sensor, robotics & video streams |
| Tensor Logic | [TensorLogic](https://github.com/cool-japan/tensorlogic) | Logical rules → tensor equations (einsum graphs) with DSL + IR |
| Math | **OxiEML** | All elementary functions from a single binary operator (this crate) |

Full project list & latest releases → [cooljapan.tech](https://cooljapan.tech/) · [GitHub](https://github.com/cool-japan)

## Sponsorship

OxiEML is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

The COOLJAPAN Ecosystem represents one of the largest Pure Rust scientific computing efforts in existence — spanning 50+ projects, 650+ crates, and millions of lines of Rust code across scientific computing, machine learning, quantum computing, geospatial analysis, legal technology, multimedia processing, and more. Every line is written and maintained by a small dedicated team committed to a C/Fortran-free future for scientific software.

If you find OxiEML or any COOLJAPAN project useful, please consider sponsoring to support continued development.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and expand the COOLJAPAN ecosystem (50+ projects, 650+ crates)
- Keep the entire stack 100% Pure Rust — no C/Fortran/system library dependencies
- Develop production-grade alternatives to OpenCV, FFmpeg, SciPy, NumPy, scikit-learn, PyTorch, TensorFlow, GDAL, and more
- Provide long-term support, security updates, and documentation
- Fund research into novel Rust-native algorithms and optimizations

## License

Apache-2.0

2026 COOLJAPAN OU (Team KitaSan)
