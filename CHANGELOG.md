# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
