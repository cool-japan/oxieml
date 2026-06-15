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

use wasm_bindgen::prelude::*;

// --- WasmSymRegConfig ---

/// Symbolic regression configuration exposed to JavaScript.
///
/// Wraps [`crate::symreg::SymRegConfig`] and exposes each tunable field
/// as a JS-accessible getter/setter pair.
#[wasm_bindgen]
pub struct WasmSymRegConfig {
    inner: crate::symreg::SymRegConfig,
    /// Maximum number of formulas to return from `discover`.
    max_formulas: usize,
}

#[wasm_bindgen]
impl WasmSymRegConfig {
    /// Create a new configuration with default values.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::default(),
            max_formulas: 10,
        }
    }

    /// Create a quick (low-depth, few-restart) preset for fast interactive use.
    pub fn quick() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::quick(),
            max_formulas: 10,
        }
    }

    /// Create the balanced (default) preset.
    pub fn balanced() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::balanced(),
            max_formulas: 10,
        }
    }

    /// Create an exhaustive (slow but thorough) configuration preset.
    pub fn exhaustive() -> Self {
        Self {
            inner: crate::symreg::SymRegConfig::exhaustive(),
            max_formulas: 10,
        }
    }

    /// Maximum tree depth to explore.
    #[wasm_bindgen(getter)]
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth
    }

    /// Set maximum tree depth to explore.
    #[wasm_bindgen(setter)]
    pub fn set_max_depth(&mut self, v: usize) {
        self.inner.max_depth = v;
    }

    /// Maximum number of formulas returned from `discover`.
    #[wasm_bindgen(getter)]
    pub fn max_formulas(&self) -> usize {
        self.max_formulas
    }

    /// Set the maximum number of formulas returned from `discover`.
    #[wasm_bindgen(setter)]
    pub fn set_max_formulas(&mut self, v: usize) {
        self.max_formulas = v;
    }

    /// Maximum number of Adam optimiser iterations per topology.
    #[wasm_bindgen(getter)]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    /// Set the maximum number of Adam optimiser iterations per topology.
    #[wasm_bindgen(setter)]
    pub fn set_max_iter(&mut self, v: usize) {
        self.inner.max_iter = v;
    }

    /// Optional RNG seed for reproducible runs.
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    /// Set the optional RNG seed for reproducible runs.
    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, v: Option<u64>) {
        self.inner.seed = v;
    }
}

impl Default for WasmSymRegConfig {
    fn default() -> Self {
        Self::new()
    }
}

// --- WasmDiscoveredFormula ---

/// A formula discovered by symbolic regression, exposed to JavaScript.
///
/// Wraps [`crate::symreg::DiscoveredFormula`] and provides JS-accessible
/// getters plus utility methods for LaTeX rendering and point evaluation.
#[wasm_bindgen]
pub struct WasmDiscoveredFormula {
    inner: crate::symreg::DiscoveredFormula,
}

#[wasm_bindgen]
impl WasmDiscoveredFormula {
    /// Human-readable expression string.
    #[wasm_bindgen(getter)]
    pub fn pretty(&self) -> String {
        self.inner.pretty.clone()
    }

    /// Final mean squared error on the training data.
    #[wasm_bindgen(getter)]
    pub fn mse(&self) -> f64 {
        self.inner.mse
    }

    /// Tree node count used as a complexity measure.
    #[wasm_bindgen(getter)]
    pub fn complexity(&self) -> usize {
        self.inner.complexity
    }

    /// Combined score: `mse + complexity_penalty * complexity`.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.inner.score
    }

    /// Render the formula as a LaTeX math expression.
    pub fn to_latex(&self) -> String {
        self.inner.eml_tree.lower().simplify().to_latex()
    }

    /// Evaluate the formula at a single point.
    ///
    /// `xs` must contain one value per input feature.
    pub fn eval(&self, xs: &[f64]) -> f64 {
        self.inner.eml_tree.lower().simplify().eval(xs)
    }
}

// --- WasmSymRegEngine ---

/// Symbolic regression engine exposed to JavaScript.
///
/// Wraps [`crate::symreg::SymRegEngine`] and exposes the `discover` method
/// accepting flat row-major float64 arrays for ergonomic use from JS/TS.
#[wasm_bindgen]
pub struct WasmSymRegEngine {
    config: crate::symreg::SymRegConfig,
    max_formulas: usize,
}

#[wasm_bindgen]
impl WasmSymRegEngine {
    /// Create a new engine from the supplied configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(config: &WasmSymRegConfig) -> Self {
        Self {
            config: config.inner.clone(),
            max_formulas: config.max_formulas,
        }
    }

    /// Discover symbolic formulas from data.
    ///
    /// - `x_flat`: row-major float64 array of shape `(n_samples, n_features)`
    /// - `y_flat`: float64 array of length `n_samples`
    /// - `n_samples`: number of data points
    /// - `n_features`: number of input features
    ///
    /// Returns an array of [`WasmDiscoveredFormula`] objects sorted by score
    /// (best first), truncated to at most `config.max_formulas` entries.
    pub fn discover(
        &self,
        x_flat: &[f64],
        y_flat: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<WasmDiscoveredFormula>, JsValue> {
        if x_flat.len() != n_samples * n_features {
            return Err(JsValue::from_str(&format!(
                "x_flat.len()={} but n_samples*n_features={}",
                x_flat.len(),
                n_samples * n_features
            )));
        }
        if y_flat.len() != n_samples {
            return Err(JsValue::from_str(&format!(
                "y_flat.len()={} but n_samples={}",
                y_flat.len(),
                n_samples
            )));
        }

        // Convert row-major flat array to Vec<Vec<f64>> where each inner Vec
        // is one sample's features — matching the `discover(inputs, targets, num_vars)`
        // signature of `SymRegEngine`.
        let inputs: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| x_flat[i * n_features..(i + 1) * n_features].to_vec())
            .collect();
        let targets: Vec<f64> = y_flat.to_vec();

        let engine = crate::symreg::SymRegEngine::new(self.config.clone());
        engine
            .discover(&inputs, &targets, n_features)
            .map_err(|e| JsValue::from_str(&e.to_string()))
            .map(|mut formulas| {
                formulas.truncate(self.max_formulas);
                formulas
                    .into_iter()
                    .map(|f| WasmDiscoveredFormula { inner: f })
                    .collect()
            })
    }
}

// ---------------------------------------------------------------------------
// Free utility functions
// ---------------------------------------------------------------------------

/// Parse an expression string and evaluate it at the given variable values.
///
/// `vars` is a flat f64 slice where `vars[i]` is the value for variable `i`.
#[wasm_bindgen]
pub fn parse_and_eval(expr_str: &str, vars: &[f64]) -> Result<f64, JsValue> {
    let tree = crate::parse(expr_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lowered = tree.lower().simplify();
    Ok(lowered.eval(vars))
}

/// Convert an expression string to a LaTeX representation.
#[wasm_bindgen]
pub fn to_latex_wasm(expr_str: &str) -> Result<String, JsValue> {
    let tree = crate::parse(expr_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
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
    let tree = crate::parse(expr_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lowered = tree.lower().simplify();
    let ctx = crate::EvalCtx::new(&[]);
    lowered
        .integrate_definite(var, lo, hi, &ctx)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Find all symbolic solutions of `expr_str = 0` for the given variable.
///
/// Returns a JSON array of LaTeX strings, e.g. `["x","\\frac{1}{2}"]`.
#[wasm_bindgen]
pub fn solve_for_all_wasm(expr_str: &str, var: usize) -> Result<String, JsValue> {
    let tree = crate::parse(expr_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lowered = tree.lower().simplify();
    let zero = crate::LoweredOp::Const(0.0);
    let result = crate::solve_for_all(&lowered, &zero, var)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    // Manually build JSON (serde_json is not available in wasm feature).
    let parts: Vec<String> = result
        .roots
        .iter()
        .map(|r| format!("{:?}", r.to_latex()))
        .collect();
    Ok(format!("[{}]", parts.join(",")))
}
