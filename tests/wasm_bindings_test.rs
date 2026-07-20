//! WASM round-trip tests for the curated browser API (Round-4 coverage).
//!
//! Gated to `target_arch = "wasm32"` only, so this file compiles to nothing on
//! the host target and the host `--all-features` gate never tries to run it.
//!
//! Note on harness: the task's suggested `wasm_bindgen_test` macro comes from
//! the `wasm-bindgen-test` crate, which is **not** a dependency of this
//! project (checked `Cargo.toml`/`Cargo.lock`); per the no-new-dependency
//! policy this file uses plain `#[test]` functions instead. They exercise the
//! `#[wasm_bindgen]`-annotated items as ordinary Rust (valid regardless of
//! target — wasm-bindgen's glue is additive), which is functionally
//! equivalent for the stated goal of "excluded from the host gate, runs under
//! a wasm target". Actually *executing* them still needs a wasm32 test
//! runner (e.g. `wasm-bindgen-test-runner` or a configured
//! `wasm32-unknown-unknown` runner); note also that `wasm32-unknown-unknown`
//! builds of this crate currently fail independently, before reaching this
//! file, because of an unconditional `rand`/`getrandom` backend-selection gap
//! unrelated to these bindings (see the final report).
//!
//! Expression strings use the crate's actual (minimal) parser grammar —
//! `1 | x<N> | E(a,b) | eml(a,b)` — not infix algebraic notation (see
//! `src/parser.rs` and `tests/python_bindings_test.rs`'s module docs).
#![cfg(target_arch = "wasm32")]

use oxieml::wasm::algebra::{
    WasmMatrix, expand_wasm, factor_wasm, simplify_wasm, sum_definite_wasm, sum_indefinite_wasm,
};
use oxieml::wasm::symreg::{WasmSymRegConfig, WasmSymRegEngine};
use oxieml::wasm::{integrate_definite_wasm, parse_and_eval, solve_for_all_wasm, to_latex_wasm};

#[test]
fn parse_and_eval_identity() {
    let value = parse_and_eval("x0", &[3.5]).expect("parse_and_eval");
    assert!((value - 3.5).abs() < 1e-12);
}

#[test]
fn to_latex_round_trip() {
    let latex = to_latex_wasm("E(x0,1)").expect("to_latex_wasm");
    assert!(!latex.is_empty());
}

#[test]
fn integrate_definite_identity() {
    // integral of x dx from 0 to 1 = 0.5
    let value = integrate_definite_wasm("x0", 0, 0.0, 1.0).expect("integrate_definite_wasm");
    assert!((value - 0.5).abs() < 1e-6);
}

#[test]
fn solve_for_all_returns_json_array() {
    let json = solve_for_all_wasm("x0", 0).expect("solve_for_all_wasm");
    assert!(json.starts_with('[') && json.ends_with(']'));
}

#[test]
fn algebra_rewrite_smoke() {
    assert!(!expand_wasm("E(x0,1)").expect("expand_wasm").is_empty());
    assert!(!factor_wasm("E(x0,1)").expect("factor_wasm").is_empty());
    assert!(!simplify_wasm("x0").expect("simplify_wasm").is_empty());
}

#[test]
fn summation_json_round_trip() {
    let indefinite = sum_indefinite_wasm("x0", 0).expect("sum_indefinite_wasm");
    assert!(indefinite.contains("\"kind\":\"closed\""));

    let definite = sum_definite_wasm("x0", 0, 1.0, 3.0).expect("sum_definite_wasm");
    assert!(definite.contains("\"kind\":\"closed\""));
}

#[test]
fn wasm_matrix_small_ops() {
    let m = WasmMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("WasmMatrix::new");
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    assert!(!m.det().expect("det").is_empty());
    assert_eq!(m.rank().expect("rank"), 2);
    assert!(!m.charpoly(0).expect("charpoly").is_empty());
}

#[test]
fn symreg_engine_discover_and_nsga2() {
    let mut config = WasmSymRegConfig::quick();
    config.set_max_formulas(5);
    let engine = WasmSymRegEngine::new(&config);

    // y = 2 * x  (4 samples, 1 feature)
    let x_flat = [1.0, 2.0, 3.0, 4.0];
    let y_flat = [2.0, 4.0, 6.0, 8.0];

    let formulas = engine
        .discover(&x_flat, &y_flat, 4, 1)
        .expect("WasmSymRegEngine::discover");
    assert!(!formulas.is_empty());
    let _ = formulas[0].pretty();
    let _ = formulas[0].to_latex();

    let ranked = engine
        .discover_nsga2(&x_flat, &y_flat, 4, 1, 6, 2)
        .expect("WasmSymRegEngine::discover_nsga2");
    assert!(!ranked.is_empty());
    let _ = ranked[0].rank();
    let _ = ranked[0].crowding();
    let _ = ranked[0].to_latex();
}

#[test]
fn symreg_engine_rejects_mismatched_shapes() {
    let config = WasmSymRegConfig::quick();
    let engine = WasmSymRegEngine::new(&config);
    let result = engine.discover(&[1.0, 2.0, 3.0], &[1.0, 2.0], 3, 1);
    assert!(result.is_err());
}
