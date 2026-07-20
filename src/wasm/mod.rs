//! WebAssembly bindings for OxiEML via wasm-bindgen.
//!
//! # JavaScript usage
//! ```javascript
//! import init, { WasmSymRegConfig, WasmSymRegEngine } from './oxieml_wasm.js';
//! await init();
//! const config = WasmSymRegConfig.quick();
//! config.max_depth = 2;
//! const engine = new WasmSymRegEngine(config);
//! // X: flat row-major array [x00, x01, ..., xij] for n_samples × n_features
//! // y: flat array [y0, y1, ..., yn]
//! const formulas = engine.discover(X_flat, y_flat, n_samples, n_features);
//! for (const f of formulas) {
//!     console.log(f.pretty, f.mse);
//! }
//! ```
//!
//! # Module layout
//!
//! - This module: general-purpose parse/eval/solve utilities.
//! - [`symreg`]: symbolic regression (config, engine, discovered/ranked formulas).
//! - [`algebra`]: a curated subset of the Round-4 algebra API — rewrite rules,
//!   summation, and small symbolic matrices. JIT, SMT, PDE, and SINDy are
//!   intentionally excluded from the WASM surface (see [`algebra`]'s docs).

pub mod algebra;
pub mod symreg;

pub use algebra::{WasmMatrix, expand_wasm, factor_wasm, simplify_wasm};
pub use algebra::{sum_definite_wasm, sum_indefinite_wasm};
pub use symreg::{WasmDiscoveredFormula, WasmRankedFormula, WasmSymRegConfig, WasmSymRegEngine};

use wasm_bindgen::prelude::*;

/// Convert any displayable error into the `JsValue` wasm-bindgen expects.
pub(crate) fn js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Parse an expression string and evaluate it at the given variable values.
///
/// `vars` is a flat f64 slice where `vars[i]` is the value for variable `i`.
#[wasm_bindgen]
pub fn parse_and_eval(expr_str: &str, vars: &[f64]) -> Result<f64, JsValue> {
    let tree = crate::parse(expr_str).map_err(js_err)?;
    let lowered = tree.lower().simplify();
    Ok(lowered.eval(vars))
}

/// Convert an expression string to a LaTeX representation.
#[wasm_bindgen]
pub fn to_latex_wasm(expr_str: &str) -> Result<String, JsValue> {
    let tree = crate::parse(expr_str).map_err(js_err)?;
    Ok(tree.lower().simplify().to_latex())
}

/// Numerically evaluate a definite integral ∫_lo^hi f(x) dx.
///
/// `var` is the 0-based index of the integration variable.
#[wasm_bindgen]
pub fn integrate_definite_wasm(
    expr_str: &str,
    var: usize,
    lo: f64,
    hi: f64,
) -> Result<f64, JsValue> {
    let tree = crate::parse(expr_str).map_err(js_err)?;
    let lowered = tree.lower().simplify();
    let ctx = crate::EvalCtx::new(&[]);
    lowered
        .integrate_definite(var, lo, hi, &ctx)
        .map_err(js_err)
}

/// Find all symbolic solutions of `expr_str = 0` for the given variable.
///
/// Returns a JSON array of LaTeX strings, e.g. `["x","\\frac{1}{2}"]`.
#[wasm_bindgen]
pub fn solve_for_all_wasm(expr_str: &str, var: usize) -> Result<String, JsValue> {
    let tree = crate::parse(expr_str).map_err(js_err)?;
    let lowered = tree.lower().simplify();
    let zero = crate::LoweredOp::Const(0.0);
    let result = crate::solve_for_all(&lowered, &zero, var).map_err(js_err)?;
    // Manually build JSON (serde_json is not available in wasm feature).
    let parts: Vec<String> = result
        .roots
        .iter()
        .map(|r| format!("{:?}", r.to_latex()))
        .collect();
    Ok(format!("[{}]", parts.join(",")))
}
