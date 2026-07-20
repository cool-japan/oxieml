//! WASM bindings for symbolic regression: config, engine, and discovered/ranked
//! formulas.

use wasm_bindgen::prelude::*;

use super::js_err;

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

// --- WasmRankedFormula ---

/// A formula annotated with its NSGA-II non-domination rank and crowding
/// distance, exposed to JavaScript.
///
/// Wraps [`crate::RankedFormula`].
#[wasm_bindgen]
pub struct WasmRankedFormula {
    inner: crate::RankedFormula,
}

#[wasm_bindgen]
impl WasmRankedFormula {
    /// Human-readable expression string.
    #[wasm_bindgen(getter)]
    pub fn pretty(&self) -> String {
        self.inner.formula.pretty.clone()
    }

    /// Final mean squared error on the training data.
    #[wasm_bindgen(getter)]
    pub fn mse(&self) -> f64 {
        self.inner.formula.mse
    }

    /// Tree node count used as a complexity measure.
    #[wasm_bindgen(getter)]
    pub fn complexity(&self) -> usize {
        self.inner.formula.complexity
    }

    /// Non-domination rank (`0` = Pareto front).
    #[wasm_bindgen(getter)]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    /// Crowding distance within the formula's own front (may be `+Infinity`).
    #[wasm_bindgen(getter)]
    pub fn crowding(&self) -> f64 {
        self.inner.crowding
    }

    /// Render the formula as a LaTeX math expression.
    pub fn to_latex(&self) -> String {
        self.inner.formula.eml_tree.lower().simplify().to_latex()
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
        let (inputs, targets) = flatten_xy(x_flat, y_flat, n_samples, n_features)?;

        let engine = crate::symreg::SymRegEngine::new(self.config.clone());
        engine
            .discover(&inputs, &targets, n_features)
            .map_err(js_err)
            .map(|mut formulas| {
                formulas.truncate(self.max_formulas);
                formulas
                    .into_iter()
                    .map(|f| WasmDiscoveredFormula { inner: f })
                    .collect()
            })
    }

    /// Discover formulas via the full NSGA-II multi-objective (MSE,
    /// complexity) search, returning the **whole final population** annotated
    /// with `(rank, crowding)`; filter `rank === 0` for the Pareto front.
    ///
    /// `population`/`generations` configure the evolutionary loop (see
    /// [`crate::Nsga2Config`]); the search-space shape still comes from this
    /// engine's `config`.
    pub fn discover_nsga2(
        &self,
        x_flat: &[f64],
        y_flat: &[f64],
        n_samples: usize,
        n_features: usize,
        population: usize,
        generations: usize,
    ) -> Result<Vec<WasmRankedFormula>, JsValue> {
        let (inputs, targets) = flatten_xy(x_flat, y_flat, n_samples, n_features)?;

        let nsga2_config = crate::Nsga2Config {
            population,
            generations,
            ..crate::Nsga2Config::default()
        };
        let engine = crate::symreg::SymRegEngine::new(self.config.clone());
        engine
            .discover_nsga2(&inputs, &targets, n_features, &nsga2_config)
            .map_err(js_err)
            .map(|ranked| {
                ranked
                    .into_iter()
                    .map(|f| WasmRankedFormula { inner: f })
                    .collect()
            })
    }
}

/// Validate and convert a flat row-major `(x_flat, y_flat)` pair into the
/// nested `Vec<Vec<f64>>` / `Vec<f64>` shape [`crate::symreg::SymRegEngine`]
/// expects.
fn flatten_xy(
    x_flat: &[f64],
    y_flat: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), JsValue> {
    if x_flat.len() != n_samples * n_features {
        return Err(js_err(format!(
            "x_flat.len()={} but n_samples*n_features={}",
            x_flat.len(),
            n_samples * n_features
        )));
    }
    if y_flat.len() != n_samples {
        return Err(js_err(format!(
            "y_flat.len()={} but n_samples={}",
            y_flat.len(),
            n_samples
        )));
    }
    let inputs: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| x_flat[i * n_features..(i + 1) * n_features].to_vec())
        .collect();
    Ok((inputs, y_flat.to_vec()))
}
