//! WASM bindings for algebraic rewriting, summation, and small symbolic
//! matrices — a curated browser subset of the Round-4 algebra API.
//!
//! JIT, SMT, PDE, and SINDy discovery are intentionally **excluded** from the
//! WASM surface: Cranelift native codegen does not target `wasm32`, and the
//! solver/PDE dependencies would bloat the bundle for capabilities a browser
//! build rarely needs.

use wasm_bindgen::prelude::*;

use super::js_err;

/// Parse `expr_str` into a lowered, simplified expression tree.
fn parse_lowered(expr_str: &str) -> Result<crate::LoweredOp, JsValue> {
    let tree = crate::parse(expr_str).map_err(js_err)?;
    Ok(tree.lower().simplify())
}

/// Expand products and integer powers into a canonical sum of monomials;
/// returns LaTeX.
#[wasm_bindgen]
pub fn expand_wasm(expr_str: &str) -> Result<String, JsValue> {
    Ok(parse_lowered(expr_str)?.expand().to_latex())
}

/// Factor a univariate polynomial into irreducibles over ℚ; returns LaTeX.
#[wasm_bindgen]
pub fn factor_wasm(expr_str: &str) -> Result<String, JsValue> {
    Ok(parse_lowered(expr_str)?.factor().to_latex())
}

/// Parse, simplify, and render `expr_str` as LaTeX.
#[wasm_bindgen]
pub fn simplify_wasm(expr_str: &str) -> Result<String, JsValue> {
    Ok(parse_lowered(expr_str)?.to_latex())
}

/// Render a [`crate::SumResult`] as a JSON string: `{"kind":"closed","latex":"…"}`
/// or `{"kind":"not_hypergeometric"}` / `{"kind":"not_closed_form"}`.
///
/// Built manually (no `serde_json`, which is not guaranteed available under
/// the `wasm` feature alone) — same approach as `solve_for_all_wasm`.
fn sum_result_to_json(result: crate::SumResult) -> String {
    match result {
        crate::SumResult::Closed(op) => {
            format!("{{\"kind\":\"closed\",\"latex\":{:?}}}", op.to_latex())
        }
        crate::SumResult::NotHypergeometric => "{\"kind\":\"not_hypergeometric\"}".to_string(),
        crate::SumResult::NotClosedForm => "{\"kind\":\"not_closed_form\"}".to_string(),
    }
}

/// Indefinite sum `Σ term(k)` over variable `k`. Returns a JSON string; see
/// [`sum_result_to_json`] for the shape.
#[wasm_bindgen]
pub fn sum_indefinite_wasm(expr_str: &str, k: usize) -> Result<String, JsValue> {
    let term = parse_lowered(expr_str)?;
    Ok(sum_result_to_json(term.sum_indefinite(k)))
}

/// Definite sum `Σ_{k=lo}^{hi} term(k)` for numeric bounds `lo`/`hi`. Same
/// return convention as [`sum_indefinite_wasm`].
#[wasm_bindgen]
pub fn sum_definite_wasm(expr_str: &str, k: usize, lo: f64, hi: f64) -> Result<String, JsValue> {
    let term = parse_lowered(expr_str)?;
    let lo_op = crate::LoweredOp::Const(lo);
    let hi_op = crate::LoweredOp::Const(hi);
    Ok(sum_result_to_json(term.sum_definite(k, &lo_op, &hi_op)))
}

/// A small symbolic matrix for the browser: numeric construction plus
/// determinant, characteristic polynomial, and rank.
///
/// Eigen-decomposition, inversion, and RREF are intentionally left off this
/// curated surface; use the Python bindings' `Matrix` for the full API.
#[wasm_bindgen]
pub struct WasmMatrix {
    inner: crate::matrix::Matrix,
}

#[wasm_bindgen]
impl WasmMatrix {
    /// Build a numeric matrix from a flat row-major `f64` array.
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, values: &[f64]) -> Result<WasmMatrix, JsValue> {
        let inner = crate::matrix::Matrix::from_f64(rows, cols, values).map_err(js_err)?;
        Ok(Self { inner })
    }

    /// Exact determinant, as LaTeX.
    pub fn det(&self) -> Result<String, JsValue> {
        Ok(self.inner.det().map_err(js_err)?.to_latex())
    }

    /// Characteristic polynomial `p(λ)`, as LaTeX in variable index `lambda_var`.
    pub fn charpoly(&self, lambda_var: usize) -> Result<String, JsValue> {
        Ok(self
            .inner
            .charpoly()
            .map_err(js_err)?
            .to_lowered(lambda_var)
            .to_latex())
    }

    /// Matrix rank.
    pub fn rank(&self) -> Result<usize, JsValue> {
        self.inner.rank().map_err(js_err)
    }

    /// Number of rows.
    #[wasm_bindgen(getter)]
    pub fn nrows(&self) -> usize {
        self.inner.rows
    }

    /// Number of columns.
    #[wasm_bindgen(getter)]
    pub fn ncols(&self) -> usize {
        self.inner.cols
    }
}
