//! Multi-output and ODE symbolic regression discovery.
//!
//! Implements:
//! - [`SymRegEngine::discover_multi`] — independent per-output regression.
//! - [`SymRegEngine::discover_ode`] — ODE discovery via numerical differentiation.

use crate::error::EmlError;

use super::discover_shared::{run_shared_topology, shared_to_multi_result};

use super::numerics;
use super::{DiscoveredFormula, MultiOutputStrategy, SymRegEngine};

impl SymRegEngine {
    /// Discover formulas for multiple outputs independently.
    ///
    /// Each output column is treated as a separate single-output regression
    /// problem (following [`MultiOutputStrategy::Independent`]).
    ///
    /// - `inputs`: each row is one data point's variable values (n_samples × n_vars)
    /// - `targets`: one `Vec<f64>` per output, each of length `n_samples`
    /// - `num_vars`: number of input variables
    ///
    /// Returns one `Vec<DiscoveredFormula>` per output, sorted by score.
    pub fn discover_multi(
        &self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        num_vars: usize,
    ) -> Result<Vec<Vec<DiscoveredFormula>>, EmlError> {
        if inputs.is_empty() {
            return Err(EmlError::EmptyData);
        }
        if targets.is_empty() {
            return Err(EmlError::EmptyData);
        }
        for col in targets.iter() {
            if col.len() != inputs.len() {
                return Err(EmlError::DimensionMismatch(inputs.len(), col.len()));
            }
        }

        match &self.config.multi_output_strategy {
            MultiOutputStrategy::Independent => targets
                .iter()
                .map(|col| self.discover(inputs, col, num_vars))
                .collect(),
            MultiOutputStrategy::SharedTopology => {
                let n_outputs = targets.len();
                let shared = run_shared_topology(self, inputs, targets, num_vars);
                Ok(shared_to_multi_result(
                    shared,
                    self.config.complexity_penalty,
                    n_outputs,
                ))
            }
        }
    }

    /// Discover ODEs `dx_k/dt = f_k(x)` from trajectory data using numerical
    /// differentiation followed by symbolic regression.
    ///
    /// # Arguments
    ///
    /// - `trajectory`: one `Vec<f64>` per state variable, each of length
    ///   `n_timesteps`.  All slices must have the same length.
    /// - `dt`: uniform time step between consecutive observations.
    ///
    /// # Returns
    ///
    /// One `Vec<DiscoveredFormula>` per state variable, sorted by score
    /// (best first), giving candidate expressions for `dx_k/dt`.
    ///
    /// # Errors
    ///
    /// Returns [`EmlError::DimensionMismatch`] when `trajectory` is empty,
    /// when variable slices disagree in length, or when fewer than 3 time
    /// steps are provided (interior-point differentiation requires at least
    /// one interior point).
    pub fn discover_ode(
        &self,
        trajectory: &[Vec<f64>],
        dt: f64,
    ) -> Result<Vec<Vec<DiscoveredFormula>>, EmlError> {
        if trajectory.is_empty() {
            return Err(EmlError::EmptyData);
        }
        let n_timesteps = trajectory[0].len();
        for var in trajectory.iter() {
            if var.len() != n_timesteps {
                return Err(EmlError::DimensionMismatch(n_timesteps, var.len()));
            }
        }
        if n_timesteps < 3 {
            return Err(EmlError::DimensionMismatch(3, n_timesteps));
        }

        let n_vars = trajectory.len();

        let derivatives: Vec<Vec<f64>> = trajectory
            .iter()
            .map(|x| match self.config.ode_sg_window {
                Some(w) if w >= 5 => numerics::savitzky_golay_derivative(x, dt),
                _ => numerics::central_differences(x, dt),
            })
            .collect();

        let n_interior = n_timesteps - 2;

        let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_interior);
        for t in 1..n_timesteps - 1 {
            features.push(trajectory.iter().map(|x| x[t]).collect());
        }

        let targets: Vec<Vec<f64>> = derivatives
            .iter()
            .map(|dx| dx[1..n_timesteps - 1].to_vec())
            .collect();

        self.discover_multi(&features, &targets, n_vars)
    }

    /// Discover ODEs using SINDy sparse regression.
    ///
    /// Convenience wrapper that delegates to [`super::sindy::discover_ode_sindy`].
    pub fn discover_ode_sindy(
        &self,
        trajectory: &[Vec<f64>],
        dt: f64,
        cfg: &super::sindy::SindyConfig,
    ) -> Result<super::sindy::SindyResult, crate::error::EmlError> {
        super::sindy::discover_ode_sindy(trajectory, dt, cfg)
    }
}
