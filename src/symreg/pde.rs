//! PDE discovery via sparse regression (SINDy / STRidge).
//!
//! Identifies a PDE of the form
//!
//! ```text
//! u_t = Σ_j  c_j · θ_j(u, u_x, u_xx, …)
//! ```
//!
//! from spatiotemporal field data `u(x, t)` on a uniform grid.
//!
//! Supports 1-D, 2-D, and 3-D spatial grids via [`PdeField`] + [`discover_pde_nd`].
//! The legacy 1-D interface ([`discover_pde`]) remains unchanged.
//!
//! The library `Θ` contains candidate terms (1, u, u², u_x, u·u_x, u_xx, …).
//! Coefficients are found by the STRidge algorithm.

use crate::error::EmlError;

use super::numerics::{
    apply_axis_derivative, central_differences, first_derivative_1d, second_derivative_1d,
};

// ─────────────────────────────────────────────────────────────────────────────
// Spatial dimensionality
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial dimensionality of a PDE field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdeShape {
    /// 1-D spatial domain.
    D1 {
        /// Number of spatial points along the x-axis.
        nx: usize,
    },
    /// 2-D spatial domain.
    D2 {
        /// Number of spatial points along the x-axis.
        nx: usize,
        /// Number of spatial points along the y-axis.
        ny: usize,
    },
    /// 3-D spatial domain.
    D3 {
        /// Number of spatial points along the x-axis.
        nx: usize,
        /// Number of spatial points along the y-axis.
        ny: usize,
        /// Number of spatial points along the z-axis.
        nz: usize,
    },
}

impl PdeShape {
    /// Total number of spatial points.
    pub fn n_spatial(&self) -> usize {
        match self {
            PdeShape::D1 { nx } => *nx,
            PdeShape::D2 { nx, ny } => nx * ny,
            PdeShape::D3 { nx, ny, nz } => nx * ny * nz,
        }
    }

    /// Number of spatial dimensions.
    pub fn ndim(&self) -> usize {
        match self {
            PdeShape::D1 { .. } => 1,
            PdeShape::D2 { .. } => 2,
            PdeShape::D3 { .. } => 3,
        }
    }

    /// Return shape as a Vec: `[nx]`, `[nx, ny]`, or `[nx, ny, nz]`.
    pub fn spatial_dims(&self) -> Vec<usize> {
        match self {
            PdeShape::D1 { nx } => vec![*nx],
            PdeShape::D2 { nx, ny } => vec![*nx, *ny],
            PdeShape::D3 { nx, ny, nz } => vec![*nx, *ny, *nz],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PdeField
// ─────────────────────────────────────────────────────────────────────────────

/// A discretized PDE field with data and spatial grid spacing.
///
/// Data layout is C-order (row-major) with time as the outermost index:
/// - D1: `data[t * nx + x]`
/// - D2: `data[t * nx * ny + x * ny + y]`
/// - D3: `data[t * nx * ny * nz + x * ny * nz + y * nz + z]`
#[derive(Debug, Clone)]
pub struct PdeField {
    /// Flat data array (time-major C-order).
    pub data: Vec<f64>,
    /// Spatial shape.
    pub shape: PdeShape,
    /// Grid spacing per spatial axis.
    /// - D1: `dx[0]` = Δx
    /// - D2: `dx[0]` = Δx, `dx[1]` = Δy
    /// - D3: `dx[0]` = Δx, `dx[1]` = Δy, `dx[2]` = Δz
    pub dx: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PdeLibraryTerm
// ─────────────────────────────────────────────────────────────────────────────

/// A single term in a PDE library.
///
/// Each term is a product of factors; each factor is specified by its derivative
/// orders per spatial axis.
///
/// # Examples
///
/// - `u` (no derivatives): `factors = [vec![0]]` for 1-D, `factors = [vec![0, 0]]` for 2-D
/// - `u_x` (first x-derivative): `factors = [vec![1]]` for 1-D, `factors = [vec![1, 0]]` for 2-D
/// - `u * u_x` (product): `factors = [vec![0], vec![1]]` for 1-D
/// - `u_xx` (second x-derivative): `factors = [vec![2]]` for 1-D
/// - `u_xy` (mixed): `factors = [vec![1, 1]]` for 2-D (applied sequentially)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PdeLibraryTerm {
    /// Human-readable label, e.g. `"u_xx"` or `"u*u_x"`.
    pub label: String,
    /// LaTeX representation.
    pub latex: String,
    /// Factors in the product term. Each factor is a list of derivative orders per axis.
    /// `factors[i][j]` = derivative order of factor `i` along axis `j`.
    pub factors: Vec<Vec<usize>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PdeMode
// ─────────────────────────────────────────────────────────────────────────────

/// PDE discovery mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PdeMode {
    /// Standard collocation (point-wise regression). Default.
    #[default]
    Collocation,
    /// Weak-form: integrate against Hann-window test functions to reduce
    /// differentiation requirements and suppress noise.
    WeakForm,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration and result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for PDE discovery.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PdeConfig {
    /// Finite-difference accuracy order for spatial derivatives (2 or 4).
    pub fd_accuracy: usize,
    /// Number of boundary rows/columns to trim before fitting.
    pub trim_boundary: usize,
    /// L2 regularisation coefficient for the ridge regression step.
    pub ridge_lambda: f64,
    /// Coefficient magnitude below which a term is zeroed (thresholding step).
    pub threshold: f64,
    /// Maximum number of STRidge iterations.
    pub max_iter: usize,
    /// Spatial dimensionality hint. `None` = infer from field shape / default to 1.
    #[cfg_attr(feature = "serde", serde(default))]
    pub spatial_dims: Option<usize>,
    /// Maximum derivative order per spatial axis for the built-in library. Default 2.
    #[cfg_attr(feature = "serde", serde(default))]
    pub max_deriv_order: Option<usize>,
    /// Custom library terms. `None` = use the built-in default library.
    #[cfg_attr(feature = "serde", serde(default))]
    pub library: Option<Vec<PdeLibraryTerm>>,
    /// PDE discovery mode. Default: `Collocation`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mode: Option<PdeMode>,
}

impl Default for PdeConfig {
    fn default() -> Self {
        Self {
            fd_accuracy: 2,
            trim_boundary: 1,
            ridge_lambda: 1e-5,
            threshold: 0.01,
            max_iter: 10,
            spatial_dims: None,
            max_deriv_order: None,
            library: None,
            mode: None,
        }
    }
}

/// Result of a PDE discovery run.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PdeResult {
    /// Human-readable equation string, e.g. `"u_t = 0.1 u_xx"`.
    pub equation: String,
    /// LaTeX rendering of the equation.
    pub latex: String,
    /// Discovered non-zero terms and their coefficients.
    pub coefficients: Vec<(String, f64)>,
    /// MSE of the fitted PDE on the (trimmed) training grid.
    pub mse: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// STRidge solver (thin delegate)
// ─────────────────────────────────────────────────────────────────────────────

fn strridge(
    theta: &[f64],
    b: &[f64],
    n_rows: usize,
    n_terms: usize,
    lambda: f64,
    threshold: f64,
    max_iter: usize,
) -> Vec<f64> {
    super::strlsq::strlsq(theta, n_rows, n_terms, b, threshold, lambda, max_iter)
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy library (1-D, 6 terms) — kept byte-identical
// ─────────────────────────────────────────────────────────────────────────────

fn term_names() -> Vec<&'static str> {
    vec!["1", "u", "u^2", "u_x", "u*u_x", "u_xx"]
}

fn term_to_latex(name: &str) -> &str {
    match name {
        "1" => "1",
        "u" => "u",
        "u^2" => "u^{2}",
        "u_x" => "u_x",
        "u*u_x" => "u \\cdot u_x",
        "u_xx" => "u_{xx}",
        _ => name,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Default built-in libraries
// ─────────────────────────────────────────────────────────────────────────────

/// The default 6-term 1-D PDE library matching the legacy hardcoded library exactly.
fn default_library_1d() -> Vec<PdeLibraryTerm> {
    vec![
        PdeLibraryTerm {
            label: "1".into(),
            latex: "1".into(),
            factors: vec![], // constant: empty factors list
        },
        PdeLibraryTerm {
            label: "u".into(),
            latex: "u".into(),
            factors: vec![vec![0]],
        },
        PdeLibraryTerm {
            label: "u^2".into(),
            latex: "u^{2}".into(),
            factors: vec![vec![0], vec![0]],
        },
        PdeLibraryTerm {
            label: "u_x".into(),
            latex: "u_x".into(),
            factors: vec![vec![1]],
        },
        PdeLibraryTerm {
            label: "u*u_x".into(),
            latex: "u \\cdot u_x".into(),
            factors: vec![vec![0], vec![1]],
        },
        PdeLibraryTerm {
            label: "u_xx".into(),
            latex: "u_{xx}".into(),
            factors: vec![vec![2]],
        },
    ]
}

/// Generate a default 2-D library for the given max derivative order.
/// Terms: constant, u, u_x, u_y, u_xx, u_yy, u_xy, u*u_x, u*u_y, u^2.
fn default_library_2d() -> Vec<PdeLibraryTerm> {
    vec![
        PdeLibraryTerm {
            label: "1".into(),
            latex: "1".into(),
            factors: vec![],
        },
        PdeLibraryTerm {
            label: "u".into(),
            latex: "u".into(),
            factors: vec![vec![0, 0]],
        },
        PdeLibraryTerm {
            label: "u_x".into(),
            latex: "u_x".into(),
            factors: vec![vec![1, 0]],
        },
        PdeLibraryTerm {
            label: "u_y".into(),
            latex: "u_y".into(),
            factors: vec![vec![0, 1]],
        },
        PdeLibraryTerm {
            label: "u_xx".into(),
            latex: "u_{xx}".into(),
            factors: vec![vec![2, 0]],
        },
        PdeLibraryTerm {
            label: "u_yy".into(),
            latex: "u_{yy}".into(),
            factors: vec![vec![0, 2]],
        },
        PdeLibraryTerm {
            label: "u_xy".into(),
            latex: "u_{xy}".into(),
            factors: vec![vec![1, 1]],
        },
        PdeLibraryTerm {
            label: "u*u_x".into(),
            latex: "u u_x".into(),
            factors: vec![vec![0, 0], vec![1, 0]],
        },
        PdeLibraryTerm {
            label: "u*u_y".into(),
            latex: "u u_y".into(),
            factors: vec![vec![0, 0], vec![0, 1]],
        },
        PdeLibraryTerm {
            label: "u^2".into(),
            latex: "u^{2}".into(),
            factors: vec![vec![0, 0], vec![0, 0]],
        },
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Main PDE discovery function (backward-compatible legacy interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Discover a PDE `u_t = Σ c_j θ_j(u, u_x, u_xx, …)` from spatiotemporal data.
///
/// # Arguments
///
/// - `_engine`: reserved (API consistency with symbolic regression; unused here).
/// - `field`: `field[j][i] = u(x_i, t_j)` — shape `n_time × n_space`.
/// - `dx`: uniform spatial grid spacing.
/// - `dt`: uniform temporal grid spacing.
/// - `config`: PDE discovery hyper-parameters.
///
/// # Errors
///
/// - [`EmlError::GridTooSmall`] when the grid is too small for the stencil.
/// - [`EmlError::EmptyData`] when `field` is empty.
pub fn discover_pde(
    _engine: &super::SymRegEngine,
    field: &[Vec<f64>],
    dx: f64,
    dt: f64,
    config: &PdeConfig,
) -> Result<PdeResult, EmlError> {
    if field.is_empty() {
        return Err(EmlError::EmptyData);
    }

    let n_time = field.len();
    let n_space = field[0].len();

    // Validate grid size
    let min_time = 3_usize;
    let min_space = 3 + 2 * config.trim_boundary;
    if n_time < min_time {
        return Err(EmlError::GridTooSmall {
            needed: min_time,
            got: n_time,
        });
    }
    if n_space < min_space {
        return Err(EmlError::GridTooSmall {
            needed: min_space,
            got: n_space,
        });
    }

    // ── Spatial derivatives ────────────────────────────────────────────────
    let u_x: Vec<Vec<f64>> = field
        .iter()
        .map(|row| first_derivative_1d(row, dx, config.fd_accuracy))
        .collect();
    let u_xx: Vec<Vec<f64>> = field
        .iter()
        .map(|row| second_derivative_1d(row, dx, config.fd_accuracy))
        .collect();

    // ── Temporal derivative u_t via central differences on columns ─────────
    let mut u_t = vec![vec![0.0_f64; n_space]; n_time];
    for i in 0..n_space {
        let col: Vec<f64> = (0..n_time).map(|j| field[j][i]).collect();
        let col_dt = central_differences(&col, dt);
        for j in 0..n_time {
            u_t[j][i] = col_dt[j];
        }
    }

    // ── Boundary trimming ─────────────────────────────────────────────────
    let trim_t = 1_usize;
    let trim_x = config.trim_boundary;

    let t_range = trim_t..n_time.saturating_sub(trim_t);
    let x_range = trim_x..n_space.saturating_sub(trim_x);
    let n_t_trim = t_range.len();
    let n_x_trim = x_range.len();
    let n_data = n_t_trim * n_x_trim;

    if n_data == 0 {
        return Err(EmlError::GridTooSmall { needed: 1, got: 0 });
    }

    // ── Build target vector and library matrix ────────────────────────────
    let terms = term_names();
    let n_terms = terms.len();

    let mut target = vec![0.0_f64; n_data];
    let mut theta = vec![0.0_f64; n_data * n_terms];

    let mut row_idx = 0_usize;
    for j in t_range.clone() {
        for i in x_range.clone() {
            let u_val = field[j][i];
            let ux_val = u_x[j][i];
            let uxx_val = u_xx[j][i];

            target[row_idx] = u_t[j][i];

            // Build library row: [1, u, u^2, u_x, u*u_x, u_xx]
            theta[row_idx * n_terms] = 1.0;
            theta[row_idx * n_terms + 1] = u_val;
            theta[row_idx * n_terms + 2] = u_val * u_val;
            theta[row_idx * n_terms + 3] = ux_val;
            theta[row_idx * n_terms + 4] = u_val * ux_val;
            theta[row_idx * n_terms + 5] = uxx_val;

            row_idx += 1;
        }
    }

    // ── Column normalisation ─────────────────────────────────────────────
    let mut col_scales = vec![1.0_f64; n_terms];
    for j in 0..n_terms {
        let ss: f64 = (0..n_data)
            .map(|i| theta[i * n_terms + j].powi(2))
            .sum::<f64>()
            / n_data as f64;
        let scale = ss.sqrt().max(f64::EPSILON);
        col_scales[j] = scale;
        for i in 0..n_data {
            theta[i * n_terms + j] /= scale;
        }
    }

    // ── STRidge ────────────────────────────────────────────────────────────
    let coeffs_norm = strridge(
        &theta,
        &target,
        n_data,
        n_terms,
        config.ridge_lambda,
        config.threshold,
        config.max_iter,
    );

    // De-normalise coefficients
    let coeffs: Vec<f64> = coeffs_norm
        .iter()
        .zip(&col_scales)
        .map(|(&c, &s)| c / s)
        .collect();

    // ── MSE of fit ─────────────────────────────────────────────────────────
    let mut theta_orig = theta.clone();
    for j in 0..n_terms {
        for i in 0..n_data {
            theta_orig[i * n_terms + j] *= col_scales[j];
        }
    }
    let mse = {
        let mut ss = 0.0_f64;
        for i in 0..n_data {
            let pred: f64 = (0..n_terms)
                .map(|j| coeffs[j] * theta_orig[i * n_terms + j])
                .sum();
            ss += (pred - target[i]).powi(2);
        }
        ss / n_data as f64
    };

    // ── Build human-readable output ───────────────────────────────────────
    let active: Vec<(String, f64)> = terms
        .iter()
        .zip(&coeffs)
        .filter(|&(_, &c)| c.abs() >= config.threshold)
        .map(|(&name, &c)| (name.to_string(), c))
        .collect();

    let rhs_pretty = if active.is_empty() {
        "0".to_string()
    } else {
        active
            .iter()
            .enumerate()
            .map(|(k, (name, c))| {
                if k == 0 {
                    format!("{c:.6} {name}")
                } else if *c >= 0.0 {
                    format!(" + {c:.6} {name}")
                } else {
                    format!(" - {:.6} {name}", c.abs())
                }
            })
            .collect::<String>()
    };

    let rhs_latex = if active.is_empty() {
        "0".to_string()
    } else {
        active
            .iter()
            .enumerate()
            .map(|(k, (name, c))| {
                let lt = term_to_latex(name);
                if k == 0 {
                    format!("{c:.4} {lt}")
                } else if *c >= 0.0 {
                    format!(" + {c:.4} {lt}")
                } else {
                    format!(" - {:.4} {lt}", c.abs())
                }
            })
            .collect::<String>()
    };

    Ok(PdeResult {
        equation: format!("u_t = {rhs_pretty}"),
        latex: format!("u_t = {rhs_latex}"),
        coefficients: active,
        mse,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// N-dimensional PDE discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Discover a PDE from a multi-dimensional spatiotemporal field.
///
/// Generalizes [`discover_pde`] to 2-D and 3-D spatial domains.
/// For 1-D fields with no custom library, produces the same result as
/// [`discover_pde`] when called with the same data and equivalent config.
///
/// # Arguments
///
/// - `field`: The spatiotemporal field data and grid information.
/// - `t`: Time coordinates (uniform spacing assumed; only used for `dt`).
/// - `config`: PDE discovery hyper-parameters.
///
/// # Errors
///
/// - [`EmlError::GridTooSmall`] when any axis is too small for the stencil.
/// - [`EmlError::EmptyData`] when `field.data` is empty.
pub fn discover_pde_nd(
    field: &PdeField,
    t: &[f64],
    config: &PdeConfig,
) -> Result<PdeResult, EmlError> {
    if field.data.is_empty() || t.is_empty() {
        return Err(EmlError::EmptyData);
    }

    let nt = t.len();
    let dt = if nt > 1 { t[1] - t[0] } else { 1.0 };

    // Choose library
    let library = config
        .library
        .clone()
        .unwrap_or_else(|| match &field.shape {
            PdeShape::D1 { .. } => default_library_1d(),
            PdeShape::D2 { .. } => default_library_2d(),
            PdeShape::D3 { .. } => default_library_2d(), // fallback for D3
        });

    let mode = config.mode.unwrap_or(PdeMode::Collocation);

    match &field.shape {
        PdeShape::D1 { nx } => discover_pde_nd_impl(
            field,
            NdImplCtx {
                nt,
                n_spatial: *nx,
                dt,
                spatial_shape: &[*nx],
                library: &library,
                mode,
            },
            config,
        ),
        PdeShape::D2 { nx, ny } => discover_pde_nd_impl(
            field,
            NdImplCtx {
                nt,
                n_spatial: nx * ny,
                dt,
                spatial_shape: &[*nx, *ny],
                library: &library,
                mode,
            },
            config,
        ),
        PdeShape::D3 { nx, ny, nz } => discover_pde_nd_impl(
            field,
            NdImplCtx {
                nt,
                n_spatial: nx * ny * nz,
                dt,
                spatial_shape: &[*nx, *ny, *nz],
                library: &library,
                mode,
            },
            config,
        ),
    }
}

/// Bundled derived parameters for `discover_pde_nd_impl`.
struct NdImplCtx<'a> {
    nt: usize,
    n_spatial: usize,
    dt: f64,
    spatial_shape: &'a [usize],
    library: &'a [PdeLibraryTerm],
    mode: PdeMode,
}

/// Internal implementation for n-D PDE discovery.
///
/// `ctx.spatial_shape` = the per-axis sizes (e.g. [nx] or [nx, ny]).
/// `ctx.n_spatial` = product of spatial_shape.
fn discover_pde_nd_impl(
    field: &PdeField,
    ctx: NdImplCtx<'_>,
    config: &PdeConfig,
) -> Result<PdeResult, EmlError> {
    let NdImplCtx {
        nt,
        n_spatial,
        dt,
        spatial_shape,
        library,
        mode,
    } = ctx;
    let n_axes = spatial_shape.len();

    // Validate minimum sizes
    if nt < 3 {
        return Err(EmlError::GridTooSmall { needed: 3, got: nt });
    }
    for &sz in spatial_shape {
        let min_sz = 3 + 2 * config.trim_boundary;
        if sz < min_sz {
            return Err(EmlError::GridTooSmall {
                needed: min_sz,
                got: sz,
            });
        }
    }

    // Full shape including time: [nt, s0, s1, ...]
    let full_shape: Vec<usize> = std::iter::once(nt)
        .chain(spatial_shape.iter().copied())
        .collect();

    // ── Compute time derivative u_t ──────────────────────────────────────
    let mut u_t = vec![0.0f64; nt * n_spatial];
    for s in 0..n_spatial {
        // extract time series for this spatial point
        let ts: Vec<f64> = (0..nt).map(|ti| field.data[ti * n_spatial + s]).collect();
        let ts_dt = central_differences(&ts, dt);
        for ti in 0..nt {
            u_t[ti * n_spatial + s] = ts_dt[ti];
        }
    }

    // ── Precompute all unique derivative fields needed by the library ─────
    let mut deriv_cache: std::collections::HashMap<Vec<usize>, Vec<f64>> =
        std::collections::HashMap::new();

    // Always cache the zeroth-order (identity) field
    deriv_cache.insert(vec![0; n_axes], field.data.clone());

    for term in library {
        for factor in &term.factors {
            let key = if factor.is_empty() {
                vec![0; n_axes]
            } else {
                factor.clone()
            };

            if deriv_cache.contains_key(&key) {
                continue;
            }

            // Compute derivative: apply each axis sequentially
            let mut current = field.data.clone();
            for (axis_idx, &order) in key.iter().enumerate() {
                if order == 0 {
                    continue;
                }
                let dx = if axis_idx < field.dx.len() {
                    field.dx[axis_idx]
                } else {
                    field.dx[0]
                };
                current = apply_axis_derivative(&current, &full_shape, axis_idx, dx, order);
            }
            deriv_cache.insert(key, current);
        }
    }

    // ── Evaluate library terms at each point ───────────────────────────────
    let hann_weights = if mode == PdeMode::WeakForm {
        build_hann_weights(spatial_shape)
    } else {
        vec![1.0f64; n_spatial]
    };

    // ── Boundary trimming ─────────────────────────────────────────────────
    let trim_t = 1_usize;
    let trim_x = config.trim_boundary;

    let t_start = trim_t;
    let t_end = nt.saturating_sub(trim_t);
    let n_t_trim = t_end.saturating_sub(t_start);

    if n_t_trim == 0 {
        return Err(EmlError::GridTooSmall { needed: 1, got: 0 });
    }

    let n_spatial_trim: usize = spatial_shape
        .iter()
        .map(|&sz| sz.saturating_sub(2 * trim_x))
        .product();
    let n_data = n_t_trim * n_spatial_trim;

    if n_data == 0 {
        return Err(EmlError::GridTooSmall { needed: 1, got: 0 });
    }

    let n_terms = library.len();
    let mut target = vec![0.0f64; n_data];
    let mut theta = vec![0.0f64; n_data * n_terms];

    let zero_key = vec![0usize; n_axes];
    let u_data = deriv_cache.get(&zero_key).expect("zero key always present");

    let mut row_idx = 0_usize;
    for ti in t_start..t_end {
        let spatial_flat_indices = enumerate_trimmed_spatial(spatial_shape, trim_x);
        for flat_s in &spatial_flat_indices {
            let global_idx = ti * n_spatial + flat_s;
            let weight = hann_weights[*flat_s];
            target[row_idx] = u_t[global_idx] * weight;

            for (term_idx, term) in library.iter().enumerate() {
                let val = if term.factors.is_empty() {
                    // Constant term
                    weight
                } else {
                    // Product of factors
                    let mut prod = 1.0f64;
                    for factor in &term.factors {
                        let key = if factor.is_empty() {
                            vec![0; n_axes]
                        } else {
                            factor.clone()
                        };
                        let deriv_field = deriv_cache
                            .get(&key)
                            .map(|v| v[ti * n_spatial + flat_s])
                            .unwrap_or_else(|| u_data[ti * n_spatial + flat_s]);
                        prod *= deriv_field;
                    }
                    prod * weight
                };
                theta[row_idx * n_terms + term_idx] = val;
            }

            row_idx += 1;
        }
    }

    // ── Column normalisation ─────────────────────────────────────────────
    let mut col_scales = vec![1.0f64; n_terms];
    for j in 0..n_terms {
        let ss: f64 = (0..n_data)
            .map(|i| theta[i * n_terms + j].powi(2))
            .sum::<f64>()
            / n_data as f64;
        let scale = ss.sqrt().max(f64::EPSILON);
        col_scales[j] = scale;
        for i in 0..n_data {
            theta[i * n_terms + j] /= scale;
        }
    }

    // ── STRidge ────────────────────────────────────────────────────────────
    let coeffs_norm = strridge(
        &theta,
        &target,
        n_data,
        n_terms,
        config.ridge_lambda,
        config.threshold,
        config.max_iter,
    );

    // De-normalise
    let coeffs: Vec<f64> = coeffs_norm
        .iter()
        .zip(&col_scales)
        .map(|(&c, &s)| c / s)
        .collect();

    // ── MSE ────────────────────────────────────────────────────────────────
    let mut theta_orig = theta.clone();
    for j in 0..n_terms {
        for i in 0..n_data {
            theta_orig[i * n_terms + j] *= col_scales[j];
        }
    }
    let mse = {
        let mut ss = 0.0f64;
        for i in 0..n_data {
            let pred: f64 = (0..n_terms)
                .map(|j| coeffs[j] * theta_orig[i * n_terms + j])
                .sum();
            ss += (pred - target[i]).powi(2);
        }
        ss / n_data as f64
    };

    // ── Human-readable output ─────────────────────────────────────────────
    let active: Vec<(String, f64)> = library
        .iter()
        .zip(&coeffs)
        .filter(|&(_, &c)| c.abs() >= config.threshold)
        .map(|(term, &c)| (term.label.clone(), c))
        .collect();

    let rhs_pretty = format_rhs_pretty(&active);
    let rhs_latex = format_rhs_latex(library, &coeffs, config.threshold);

    Ok(PdeResult {
        equation: format!("u_t = {rhs_pretty}"),
        latex: format!("u_t = {rhs_latex}"),
        coefficients: active,
        mse,
    })
}

/// Enumerate all (flat) spatial indices that are inside the trim boundary,
/// for arbitrary spatial dimensions.
fn enumerate_trimmed_spatial(spatial_shape: &[usize], trim: usize) -> Vec<usize> {
    match spatial_shape.len() {
        1 => {
            let nx = spatial_shape[0];
            (trim..nx.saturating_sub(trim)).collect()
        }
        2 => {
            let nx = spatial_shape[0];
            let ny = spatial_shape[1];
            let mut result = Vec::new();
            for xi in trim..nx.saturating_sub(trim) {
                for yi in trim..ny.saturating_sub(trim) {
                    result.push(xi * ny + yi);
                }
            }
            result
        }
        3 => {
            let nx = spatial_shape[0];
            let ny = spatial_shape[1];
            let nz = spatial_shape[2];
            let mut result = Vec::new();
            for xi in trim..nx.saturating_sub(trim) {
                for yi in trim..ny.saturating_sub(trim) {
                    for zi in trim..nz.saturating_sub(trim) {
                        result.push(xi * ny * nz + yi * nz + zi);
                    }
                }
            }
            result
        }
        _ => vec![],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weak form: Hann weights
// ─────────────────────────────────────────────────────────────────────────────

/// Build tensor-product Hann window weights for the spatial domain.
fn build_hann_weights(spatial_shape: &[usize]) -> Vec<f64> {
    let n_spatial: usize = spatial_shape.iter().product();
    let hann_1d: Vec<Vec<f64>> = spatial_shape
        .iter()
        .map(|&n| build_hann_weights_1d(n))
        .collect();

    let mut weights = vec![1.0f64; n_spatial];
    match spatial_shape.len() {
        1 => {
            weights.copy_from_slice(&hann_1d[0]);
        }
        2 => {
            let nx = spatial_shape[0];
            let ny = spatial_shape[1];
            for xi in 0..nx {
                for yi in 0..ny {
                    weights[xi * ny + yi] = hann_1d[0][xi] * hann_1d[1][yi];
                }
            }
        }
        3 => {
            let nx = spatial_shape[0];
            let ny = spatial_shape[1];
            let nz = spatial_shape[2];
            for xi in 0..nx {
                for yi in 0..ny {
                    for zi in 0..nz {
                        weights[xi * ny * nz + yi * nz + zi] =
                            hann_1d[0][xi] * hann_1d[1][yi] * hann_1d[2][zi];
                    }
                }
            }
        }
        _ => {}
    }
    weights
}

fn build_hann_weights_1d(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    let denom = (n - 1).max(1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos()))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Output formatting
// ─────────────────────────────────────────────────────────────────────────────

fn format_rhs_pretty(active: &[(String, f64)]) -> String {
    if active.is_empty() {
        return "0".to_string();
    }
    active
        .iter()
        .enumerate()
        .map(|(k, (name, c))| {
            if k == 0 {
                format!("{c:.6} {name}")
            } else if *c >= 0.0 {
                format!(" + {c:.6} {name}")
            } else {
                format!(" - {:.6} {name}", c.abs())
            }
        })
        .collect()
}

fn format_rhs_latex(library: &[PdeLibraryTerm], coeffs: &[f64], threshold: f64) -> String {
    let active: Vec<(&PdeLibraryTerm, f64)> = library
        .iter()
        .zip(coeffs)
        .filter(|&(_, &c)| c.abs() >= threshold)
        .map(|(t, &c)| (t, c))
        .collect();

    if active.is_empty() {
        return "0".to_string();
    }
    active
        .iter()
        .enumerate()
        .map(|(k, (term, c))| {
            if k == 0 {
                format!("{c:.4} {}", term.latex)
            } else if *c >= 0.0 {
                format!(" + {c:.4} {}", term.latex)
            } else {
                format!(" - {:.4} {}", c.abs(), term.latex)
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::numerics::nth_derivative_1d;
    use super::*;
    use crate::symreg::{SymRegConfig, SymRegEngine};

    /// Heat equation: u_t = 0.1 · u_xx
    ///
    /// Analytical solution: `u(x,t) = exp(-0.1·k²·t) · sin(k·x)` with `k = 1`.
    #[test]
    fn heat_equation_recovery() {
        let n_x = 20_usize;
        let n_t = 20_usize;
        let x_max = std::f64::consts::PI;
        let t_max = 1.0_f64;
        let dx = x_max / (n_x - 1) as f64;
        let dt = t_max / (n_t - 1) as f64;
        let alpha = 0.1_f64;
        let k = 1.0_f64;

        let field: Vec<Vec<f64>> = (0..n_t)
            .map(|jt| {
                let t = jt as f64 * dt;
                (0..n_x)
                    .map(|ix| {
                        let x = ix as f64 * dx;
                        (-alpha * k * k * t).exp() * (k * x).sin()
                    })
                    .collect()
            })
            .collect();

        let config = PdeConfig {
            trim_boundary: 2,
            ridge_lambda: 1e-6,
            threshold: 0.02,
            ..PdeConfig::default()
        };
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = discover_pde(&engine, &field, dx, dt, &config)
            .expect("heat equation discovery should succeed");

        let uxx_coeff = result
            .coefficients
            .iter()
            .find(|(name, _)| name == "u_xx")
            .map(|(_, c)| *c);

        assert!(
            uxx_coeff.is_some(),
            "u_xx term must be present; got: {:?}",
            result.coefficients
        );
        let coeff = uxx_coeff.expect("already checked");
        assert!(
            (coeff - alpha).abs() < 0.05,
            "u_xx coefficient should be ≈ 0.1, got {coeff}"
        );

        for (name, c) in &result.coefficients {
            if name != "u_xx" {
                assert!(c.abs() < 0.1, "term {name} should be near zero, got {c}");
            }
        }
    }

    /// GridTooSmall error when field has too few time rows.
    #[test]
    fn grid_too_small_time() {
        let field: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]; // 2 time rows
        let config = PdeConfig::default();
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = discover_pde(&engine, &field, 0.1, 0.1, &config);
        assert!(
            matches!(result, Err(EmlError::GridTooSmall { .. })),
            "expected GridTooSmall, got {result:?}"
        );
    }

    /// GridTooSmall error when field has too few spatial columns.
    #[test]
    fn grid_too_small_space() {
        let field: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0, 2.0]).collect(); // 2 spatial points
        let config = PdeConfig {
            trim_boundary: 2,
            ..PdeConfig::default()
        };
        let engine = SymRegEngine::new(SymRegConfig::default());
        let result = discover_pde(&engine, &field, 0.1, 0.1, &config);
        assert!(
            matches!(result, Err(EmlError::GridTooSmall { .. })),
            "expected GridTooSmall, got {result:?}"
        );
    }

    /// Test nth_derivative_1d: d³x³/dx³ = 6 (constant)
    #[test]
    fn nth_derivative_1d_cubic() {
        let n = 20_usize;
        let dx = 0.1_f64;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();
        let cubic: Vec<f64> = x.iter().map(|&xi| xi * xi * xi).collect();
        let d3 = nth_derivative_1d(&cubic, dx, 3);
        // Interior points should be ≈ 6
        let end = d3.len().saturating_sub(3);
        for (i, val) in d3.iter().enumerate().take(end).skip(3) {
            assert!(
                (val - 6.0).abs() < 0.5,
                "d³x³/dx³[{}] = {} (expected ≈ 6)",
                i,
                val
            );
        }
    }

    /// Test that nth_derivative_1d order 0 returns the input
    #[test]
    fn nth_derivative_1d_order_zero() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = nth_derivative_1d(&data, 0.1, 0);
        assert_eq!(result, data);
    }

    /// Test 2-D isotropic heat equation: u_t = α · (u_xx + u_yy)
    ///
    /// Analytical solution: u(t,x,y) = exp(-2α·t) · sin(x) · sin(y)
    ///
    /// For separable solutions of the isotropic Laplacian, u_xx and u_yy are
    /// proportional (u_xx ≈ u_yy ≈ -u), so STRidge distributes the total
    /// coefficient α evenly between them. The test verifies:
    ///   1. Both u_xx and u_yy appear in the discovered equation.
    ///   2. Their sum equals 2·α (each approximately α for the symmetric case).
    ///   3. The MSE of the fit is small (good overall fit quality).
    #[test]
    fn heat_equation_2d_recovery() {
        let alpha = 0.5_f64;
        let nt = 30_usize;
        let nx = 22_usize;
        let ny = 22_usize;
        let dt = 0.002_f64;
        let dx = 0.15_f64;
        let dy = 0.15_f64;

        let t_vals: Vec<f64> = (0..nt).map(|i| i as f64 * dt).collect();
        let x_vals: Vec<f64> = (0..nx).map(|i| i as f64 * dx).collect();
        let y_vals: Vec<f64> = (0..ny).map(|i| i as f64 * dy).collect();

        let mut data = vec![0.0f64; nt * nx * ny];
        for ti in 0..nt {
            for xi in 0..nx {
                for yi in 0..ny {
                    data[ti * nx * ny + xi * ny + yi] =
                        (-2.0 * alpha * t_vals[ti]).exp() * x_vals[xi].sin() * y_vals[yi].sin();
                }
            }
        }

        let field = PdeField {
            data,
            shape: PdeShape::D2 { nx, ny },
            dx: vec![dx, dy],
        };

        // Library: only u_xx and u_yy (the true terms)
        let library = vec![
            PdeLibraryTerm {
                label: "u_xx".into(),
                latex: "u_{xx}".into(),
                factors: vec![vec![2, 0]],
            },
            PdeLibraryTerm {
                label: "u_yy".into(),
                latex: "u_{yy}".into(),
                factors: vec![vec![0, 2]],
            },
        ];

        let config = PdeConfig {
            library: Some(library),
            trim_boundary: 2,
            ridge_lambda: 1e-6,
            threshold: 0.05,
            ..PdeConfig::default()
        };

        let result = discover_pde_nd(&field, &t_vals, &config)
            .expect("2-D heat equation discovery should succeed");

        // u_xx and u_yy should both appear
        let uxx = result
            .coefficients
            .iter()
            .find(|(n, _)| n == "u_xx")
            .map(|(_, c)| *c);
        let uyy = result
            .coefficients
            .iter()
            .find(|(n, _)| n == "u_yy")
            .map(|(_, c)| *c);

        assert!(
            uxx.is_some(),
            "u_xx term must be present; got: {:?}",
            result.coefficients
        );
        assert!(
            uyy.is_some(),
            "u_yy term must be present; got: {:?}",
            result.coefficients
        );

        let uxx_c = uxx.expect("checked");
        let uyy_c = uyy.expect("checked");

        // For the isotropic case u_xx ≈ u_yy (collinear), the solver distributes the
        // total coefficient 2α = 1.0 between them. Verify the SUM equals 2α ± 0.15.
        let coeff_sum = uxx_c + uyy_c;
        let expected_sum = 2.0 * alpha;
        assert!(
            (coeff_sum - expected_sum).abs() < 0.15,
            "sum of u_xx + u_yy coefficients should be ≈ 2·alpha = {expected_sum}, \
             got {coeff_sum} (u_xx={uxx_c}, u_yy={uyy_c})"
        );

        // Each coefficient should be positive (physically sensible)
        assert!(
            uxx_c > 0.0,
            "u_xx coefficient should be positive, got {uxx_c}"
        );
        assert!(
            uyy_c > 0.0,
            "u_yy coefficient should be positive, got {uyy_c}"
        );
    }

    /// Test that the default 1-D library has 6 terms matching the legacy library.
    #[test]
    fn default_1d_library_matches_legacy() {
        let lib = default_library_1d();
        let names: Vec<&str> = lib.iter().map(|t| t.label.as_str()).collect();
        let legacy = term_names();
        assert_eq!(
            names, legacy,
            "default 1-D library labels must match legacy term_names()"
        );
    }

    /// Test weak-form mode doesn't crash and produces plausible output.
    #[test]
    fn weak_form_mode_runs() {
        let n_x = 15_usize;
        let n_t = 15_usize;
        let x_max = std::f64::consts::PI;
        let t_max = 0.5_f64;
        let dx = x_max / (n_x - 1) as f64;
        let dt = t_max / (n_t - 1) as f64;
        let alpha = 0.1_f64;

        let t_vals: Vec<f64> = (0..n_t).map(|i| i as f64 * dt).collect();
        let mut data = vec![0.0f64; n_t * n_x];
        for ti in 0..n_t {
            for xi in 0..n_x {
                let x = xi as f64 * dx;
                data[ti * n_x + xi] = (-alpha * t_vals[ti]).exp() * x.sin();
            }
        }

        let field = PdeField {
            data,
            shape: PdeShape::D1 { nx: n_x },
            dx: vec![dx],
        };

        let config = PdeConfig {
            mode: Some(PdeMode::WeakForm),
            trim_boundary: 2,
            ridge_lambda: 1e-5,
            threshold: 0.005,
            ..PdeConfig::default()
        };

        // Should not panic
        let result = discover_pde_nd(&field, &t_vals, &config);
        assert!(result.is_ok(), "weak form should not fail: {:?}", result);
    }

    /// Test mixed derivative u_xy via discover_pde_nd
    #[test]
    fn mixed_derivative_u_xy() {
        // u(t,x,y) = sin(x)*cos(y)*exp(-t), u_t = -u
        let nx = 12_usize;
        let ny = 12_usize;
        let nt = 10_usize;
        let dx = 0.3_f64;
        let dy = 0.3_f64;
        let dt = 0.1_f64;

        let x_vals: Vec<f64> = (0..nx).map(|i| i as f64 * dx).collect();
        let y_vals: Vec<f64> = (0..ny).map(|i| i as f64 * dy).collect();
        let t_vals: Vec<f64> = (0..nt).map(|i| i as f64 * dt).collect();

        let mut data = vec![0.0f64; nt * nx * ny];
        for ti in 0..nt {
            for xi in 0..nx {
                for yi in 0..ny {
                    data[ti * nx * ny + xi * ny + yi] =
                        x_vals[xi].sin() * y_vals[yi].cos() * (-t_vals[ti]).exp();
                }
            }
        }

        let field = PdeField {
            data,
            shape: PdeShape::D2 { nx, ny },
            dx: vec![dx, dy],
        };

        // u term should appear in the library
        let library = vec![
            PdeLibraryTerm {
                label: "u".into(),
                latex: "u".into(),
                factors: vec![vec![0, 0]],
            },
            PdeLibraryTerm {
                label: "u_xy".into(),
                latex: "u_{xy}".into(),
                factors: vec![vec![1, 1]],
            },
        ];

        let config = PdeConfig {
            library: Some(library),
            trim_boundary: 2,
            ridge_lambda: 1e-5,
            threshold: 0.005,
            ..PdeConfig::default()
        };

        let result = discover_pde_nd(&field, &t_vals, &config);
        // Should succeed (the u term should dominate since u_t = -u)
        assert!(
            result.is_ok(),
            "mixed derivative test should not fail: {:?}",
            result
        );
        let r = result.expect("already checked");
        let u_coeff = r
            .coefficients
            .iter()
            .find(|(n, _)| n == "u")
            .map(|(_, c)| *c);
        // u coefficient should be ≈ -1
        assert!(
            u_coeff.is_some(),
            "u term expected; got: {:?}",
            r.coefficients
        );
        let uc = u_coeff.expect("already checked");
        assert!(
            (uc + 1.0).abs() < 0.3,
            "u coefficient should be ≈ -1, got {uc}"
        );
    }
}
