//! Normal-equation builder functions.

/// Build `JᵀJ + λ I` from a row-major Jacobian (identity damping).
pub fn jtj(jac: &[f64], n_rows: usize, n_params: usize, lambda: f64) -> Vec<f64> {
    let mut result = vec![0.0_f64; n_params * n_params];

    for p in 0..n_params {
        for q in p..n_params {
            let mut sum = 0.0_f64;
            for i in 0..n_rows {
                sum += jac[i * n_params + p] * jac[i * n_params + q];
            }
            result[p * n_params + q] = sum;
            result[q * n_params + p] = sum;
        }
        result[p * n_params + p] += lambda;
    }

    result
}

/// Build `JᵀJ + λ · diag(JᵀJ)` using Marquardt diagonal scaling.
pub fn jtj_marquardt(jac: &[f64], n_rows: usize, n_params: usize, lambda: f64) -> Vec<f64> {
    let mut result = vec![0.0_f64; n_params * n_params];

    for p in 0..n_params {
        for q in p..n_params {
            let mut sum = 0.0_f64;
            for i in 0..n_rows {
                sum += jac[i * n_params + p] * jac[i * n_params + q];
            }
            result[p * n_params + q] = sum;
            result[q * n_params + p] = sum;
        }
    }

    for p in 0..n_params {
        let diag = result[p * n_params + p];
        let damping = if diag.abs() < f64::EPSILON {
            lambda
        } else {
            lambda * diag
        };
        result[p * n_params + p] += damping;
    }

    result
}

/// Compute `Jᵀ r` from a row-major Jacobian and a residual vector.
pub fn jtr(jac: &[f64], r: &[f64], n_rows: usize, n_params: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n_params];
    for p in 0..n_params {
        let mut sum = 0.0_f64;
        for i in 0..n_rows {
            sum += jac[i * n_params + p] * r[i];
        }
        result[p] = sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_jtj_values() {
        let j = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let result = jtj(&j, 3, 2, 0.0);
        assert!(approx_eq(result[0], 2.0, EPS), "[0,0] = {}", result[0]);
        assert!(approx_eq(result[1], 1.0, EPS), "[0,1] = {}", result[1]);
        assert!(approx_eq(result[2], 1.0, EPS), "[1,0] = {}", result[2]);
        assert!(approx_eq(result[3], 2.0, EPS), "[1,1] = {}", result[3]);
    }

    #[test]
    fn test_jtj_lambda_damping() {
        let j = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let result = jtj(&j, 3, 2, 0.5);
        assert!(approx_eq(result[0], 2.5, EPS), "[0,0] = {}", result[0]);
        assert!(approx_eq(result[1], 1.0, EPS), "[0,1] = {}", result[1]);
        assert!(approx_eq(result[2], 1.0, EPS), "[1,0] = {}", result[2]);
        assert!(approx_eq(result[3], 2.5, EPS), "[1,1] = {}", result[3]);
    }

    #[test]
    fn test_jtr() {
        let j = [1.0_f64, 2.0, 3.0, 4.0];
        let r = [1.0_f64, 1.0];
        let result = jtr(&j, &r, 2, 2);
        assert!(approx_eq(result[0], 4.0, EPS), "result[0] = {}", result[0]);
        assert!(approx_eq(result[1], 6.0, EPS), "result[1] = {}", result[1]);
    }

    #[test]
    fn test_jtj_marquardt() {
        let j = [2.0_f64, 0.0, 0.0, 3.0];
        let result = jtj_marquardt(&j, 2, 2, 1.0);
        assert!(approx_eq(result[0], 8.0, EPS), "[0,0] = {}", result[0]);
        assert!(approx_eq(result[1], 0.0, EPS), "[0,1] = {}", result[1]);
        assert!(approx_eq(result[2], 0.0, EPS), "[1,0] = {}", result[2]);
        assert!(approx_eq(result[3], 18.0, EPS), "[1,1] = {}", result[3]);
    }

    #[test]
    fn test_jtj_marquardt_zero_diag_fallback() {
        let j = [0.0_f64, 1.0];
        let result = jtj_marquardt(&j, 1, 2, 2.0);
        assert!(approx_eq(result[0], 2.0, EPS), "[0,0] = {}", result[0]);
        assert!(approx_eq(result[3], 3.0, EPS), "[1,1] = {}", result[3]);
    }
}
