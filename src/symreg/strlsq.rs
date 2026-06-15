//! Sequential Thresholded Ridge Least-Squares (STLSQ) solver.
//!
//! Provides the [`strlsq`] function used by SINDy and PDE discovery.

use crate::linalg;

/// Solve `Θ w ≈ y` via Sequential Thresholded Ridge regression (STLSQ).
///
/// Algorithm:
/// 1. Ridge regression: `w = (ΘᵀΘ + λI)⁻¹Θᵀy`.
/// 2. Repeat up to `max_iter`:
///    a. Zero-out coefficients with `|w_i| < threshold`.
///    b. Re-fit ridge on the remaining (active) terms.
///    c. Stop early when the active support does not change.
/// 3. Final threshold pass.
///
/// # Arguments
///
/// - `theta`: row-major feature matrix, shape `n_rows × n_cols`.
/// - `n_rows`: number of data rows.
/// - `n_cols`: number of library terms.
/// - `target`: target vector of length `n_rows`.
/// - `threshold`: coefficient magnitude below which a term is zeroed.
/// - `lambda`: L2 regularisation coefficient (ridge).
/// - `max_iter`: maximum number of thresholding iterations.
///
/// Returns the coefficient vector of length `n_cols`.
pub fn strlsq(
    theta: &[f64],
    n_rows: usize,
    n_cols: usize,
    target: &[f64],
    threshold: f64,
    lambda: f64,
    max_iter: usize,
) -> Vec<f64> {
    if n_rows == 0 || n_cols == 0 {
        return vec![0.0; n_cols];
    }

    // Initial ridge regression
    let a = linalg::jtj(theta, n_rows, n_cols, lambda);
    let mut rhs = linalg::jtr(theta, target, n_rows, n_cols);
    if linalg::solve_normal_equations(&a, &mut rhs, n_cols).is_err() {
        return vec![0.0; n_cols];
    }
    let mut coeffs = rhs;

    let mut prev_support: Vec<usize> = Vec::new();

    for _iter in 0..max_iter {
        let support: Vec<usize> = (0..n_cols)
            .filter(|&j| coeffs[j].abs() >= threshold)
            .collect();

        if support.is_empty() {
            coeffs.fill(0.0);
            break;
        }

        if support == prev_support {
            break;
        }
        prev_support = support.clone();

        // Build sub-system on active support
        let n_active = support.len();
        let mut theta_sub = vec![0.0_f64; n_rows * n_active];
        for (col, &term_idx) in support.iter().enumerate() {
            for row in 0..n_rows {
                theta_sub[row * n_active + col] = theta[row * n_cols + term_idx];
            }
        }
        let a_sub = linalg::jtj(&theta_sub, n_rows, n_active, lambda);
        let mut rhs_sub = linalg::jtr(&theta_sub, target, n_rows, n_active);
        if linalg::solve_normal_equations(&a_sub, &mut rhs_sub, n_active).is_err() {
            coeffs.fill(0.0);
            break;
        }
        coeffs.fill(0.0);
        for (col, &term_idx) in support.iter().enumerate() {
            coeffs[term_idx] = rhs_sub[col];
        }
    }

    // Final threshold pass
    for c in &mut coeffs {
        if c.abs() < threshold {
            *c = 0.0;
        }
    }
    coeffs
}

/// QR-based STLSQ variant: uses `solve_least_squares` instead of normal equations.
/// Better numerical conditioning for ill-conditioned feature matrices.
///
/// Same API as `strlsq` — drop-in replacement with better numerics.
pub fn strlsq_qr(
    theta: &[f64],
    n_rows: usize,
    n_cols: usize,
    target: &[f64],
    threshold: f64,
    lambda: f64,
    max_iter: usize,
) -> Vec<f64> {
    use crate::linalg::solve_least_squares;

    if n_rows == 0 || n_cols == 0 {
        return vec![0.0; n_cols];
    }

    let sqrt_lambda = lambda.sqrt();

    let build_augmented = |theta_sub: &[f64], n_r: usize, n_c: usize| -> (Vec<f64>, Vec<f64>) {
        let mut a_aug = vec![0.0f64; (n_r + n_c) * n_c];
        for i in 0..n_r {
            for j in 0..n_c {
                a_aug[i * n_c + j] = theta_sub[i * n_c + j];
            }
        }
        for i in 0..n_c {
            a_aug[(n_r + i) * n_c + i] = sqrt_lambda;
        }
        let mut b_aug = vec![0.0f64; n_r + n_c];
        b_aug[..n_r].copy_from_slice(&target[..n_r]);
        (a_aug, b_aug)
    };

    let aug_rows = n_rows + n_cols;
    let (a_aug, b_aug) = build_augmented(theta, n_rows, n_cols);
    let mut coeffs = match solve_least_squares(&a_aug, &b_aug, aug_rows, n_cols) {
        Ok(x) => x,
        Err(_) => return vec![0.0; n_cols],
    };

    let mut prev_support: Vec<usize> = Vec::new();

    for _iter in 0..max_iter {
        let support: Vec<usize> = (0..n_cols)
            .filter(|&j| coeffs[j].abs() >= threshold)
            .collect();

        if support.is_empty() {
            coeffs.fill(0.0);
            break;
        }

        if support == prev_support {
            break;
        }
        prev_support = support.clone();

        let n_active = support.len();
        let mut theta_sub = vec![0.0_f64; n_rows * n_active];
        for (col, &term_idx) in support.iter().enumerate() {
            for row in 0..n_rows {
                theta_sub[row * n_active + col] = theta[row * n_cols + term_idx];
            }
        }

        let (a_aug, b_aug) = build_augmented(&theta_sub, n_rows, n_active);
        let rhs_sub = match solve_least_squares(&a_aug, &b_aug, n_rows + n_active, n_active) {
            Ok(x) => x,
            Err(_) => {
                coeffs.fill(0.0);
                break;
            }
        };

        coeffs.fill(0.0);
        for (col, &term_idx) in support.iter().enumerate() {
            coeffs[term_idx] = rhs_sub[col];
        }
    }

    for c in &mut coeffs {
        if c.abs() < threshold {
            *c = 0.0;
        }
    }
    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strlsq_exact_recovery() {
        // y = 2x + 3x^2. Library: [x, x^2, x^3, sin(x)]
        // Expect coefficients [2, 3, 0, 0] (up to threshold tolerance)
        let n = 30;
        let xs: Vec<f64> = (0..n)
            .map(|i| -1.5 + 3.0 * i as f64 / (n - 1) as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 3.0 * x * x).collect();

        // Build feature matrix: [x, x^2, x^3, sin(x)]
        let mut theta = vec![0.0_f64; n * 4];
        for (row, &x) in xs.iter().enumerate() {
            theta[row * 4] = x;
            theta[row * 4 + 1] = x * x;
            theta[row * 4 + 2] = x * x * x;
            theta[row * 4 + 3] = x.sin();
        }

        let coeffs = strlsq(&theta, n, 4, &ys, 0.05, 1e-6, 20);

        assert!(
            (coeffs[0] - 2.0).abs() < 0.1,
            "x coefficient should be ~2, got {}",
            coeffs[0]
        );
        assert!(
            (coeffs[1] - 3.0).abs() < 0.1,
            "x^2 coefficient should be ~3, got {}",
            coeffs[1]
        );
        assert!(
            coeffs[2].abs() < 0.05,
            "x^3 coefficient should be ~0, got {}",
            coeffs[2]
        );
        assert!(
            coeffs[3].abs() < 0.05,
            "sin(x) coefficient should be ~0, got {}",
            coeffs[3]
        );
    }
}
