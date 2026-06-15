use super::constraint::{EmlConstraint, EmlSolution};
use super::helpers::{check_constraint, count_constraint_vars, evaluate_constraint_residual};
use super::interval::{Interval, IntervalDomain, PropResult};
use crate::error::EmlError;
use crate::eval::EvalCtx;

// ----------------------------------------------------------------------------
// EmlNraSolver (interval bisection, enhanced with propagation)
// ----------------------------------------------------------------------------

/// EML-based Non-linear Real Arithmetic (NRA) solver.
///
/// Uses interval propagation + bisection. With propagation, trivial UNSAT
/// cases (e.g., `exp(x) < 0`) are detected early.
pub struct EmlNraSolver {
    /// Maximum number of interval bisection iterations.
    pub max_iterations: usize,
    /// Tolerance for interval width (stops bisection when all widths fall below).
    pub tolerance: f64,
    /// Initial search bounds for each variable.
    pub initial_bounds: Vec<(f64, f64)>,
}

impl Default for EmlNraSolver {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-8,
            initial_bounds: vec![(-10.0, 10.0)],
        }
    }
}

impl EmlNraSolver {
    /// Create a new solver with the given initial bounds for each variable.
    pub fn new(initial_bounds: Vec<(f64, f64)>) -> Self {
        Self {
            initial_bounds,
            ..Default::default()
        }
    }

    /// Solve a constraint system using interval propagation + bisection.
    pub fn solve(&self, constraint: &EmlConstraint) -> Result<EmlSolution, EmlError> {
        let num_vars = count_constraint_vars(constraint);
        if num_vars == 0 {
            let ctx = EvalCtx::new(&[]);
            if check_constraint(constraint, &ctx) {
                return Ok(EmlSolution {
                    assignments: vec![],
                    is_exact: true,
                });
            }
            return Err(EmlError::ConvergenceFailed {
                best_mse: f64::INFINITY,
                iterations: 0,
            });
        }

        let mut domain = IntervalDomain::new(&self.initial_bounds, num_vars);
        if domain.propagate(constraint) == PropResult::Conflict {
            return Err(EmlError::ConvergenceFailed {
                best_mse: f64::INFINITY,
                iterations: 0,
            });
        }

        for _iter in 0..self.max_iterations {
            let midpoints: Vec<f64> = domain.vars.iter().map(Interval::midpoint).collect();
            let ctx = EvalCtx::new(&midpoints);
            if check_constraint(constraint, &ctx) {
                return Ok(EmlSolution {
                    assignments: midpoints,
                    is_exact: domain.vars.iter().all(|iv| iv.width() < self.tolerance),
                });
            }

            // Find widest interval, bisect toward the more promising half.
            let widest_idx = domain
                .vars
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.width()
                        .partial_cmp(&b.width())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            if let Some(widest) = widest_idx {
                let (lo_half, hi_half) = domain.vars[widest].split();

                let mut lo_point = midpoints.clone();
                lo_point[widest] = lo_half.midpoint();
                let lo_resid = evaluate_constraint_residual(constraint, &lo_point);

                let mut hi_point = midpoints;
                hi_point[widest] = hi_half.midpoint();
                let hi_resid = evaluate_constraint_residual(constraint, &hi_point);

                domain.vars[widest] = if lo_resid.abs() < hi_resid.abs() {
                    lo_half
                } else {
                    hi_half
                };
                if domain.propagate(constraint) == PropResult::Conflict {
                    return Err(EmlError::ConvergenceFailed {
                        best_mse: f64::INFINITY,
                        iterations: 0,
                    });
                }
                if domain.vars.iter().all(|iv| iv.width() < self.tolerance) {
                    break;
                }
            }
        }

        let midpoints: Vec<f64> = domain.vars.iter().map(Interval::midpoint).collect();
        let ctx = EvalCtx::new(&midpoints);
        if check_constraint(constraint, &ctx) {
            Ok(EmlSolution {
                assignments: midpoints,
                is_exact: false,
            })
        } else {
            Err(EmlError::ConvergenceFailed {
                best_mse: evaluate_constraint_residual(constraint, &midpoints).abs(),
                iterations: self.max_iterations,
            })
        }
    }
}
