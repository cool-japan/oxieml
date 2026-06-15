//! QR and SVD decompositions (pure-Rust, row-major storage).

use crate::error::EmlError;

/// Compact QR factorization of matrix A (m×n row-major, m ≥ n).
#[derive(Debug, Clone)]
pub struct QrFactors {
    /// Packed factorization data: upper triangle is R, lower off-diagonal stores Householder vectors.
    pub data: Vec<f64>,
    /// Householder scaling coefficients β_j = 2 / (v_j^T v_j).
    pub betas: Vec<f64>,
    /// Number of rows in the original matrix.
    pub m: usize,
    /// Number of columns in the original matrix.
    pub n: usize,
}

/// Householder QR factorization of A (m×n row-major, m ≥ n).
pub fn qr(a: &[f64], m: usize, n: usize) -> Result<QrFactors, EmlError> {
    if a.len() != m * n {
        return Err(EmlError::DimensionMismatch(m * n, a.len()));
    }
    if m < n {
        return Err(EmlError::DimensionMismatch(m, n));
    }
    let k = m.min(n);
    let mut data = a.to_vec();
    let mut betas = vec![0.0f64; k];

    for j in 0..k {
        let col_len = m - j;
        let mut v: Vec<f64> = (0..col_len).map(|i| data[(j + i) * n + j]).collect();

        let norm_x = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_x < 1e-15 {
            continue;
        }

        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_x;

        let v_sq_sum: f64 = v.iter().map(|x| x * x).sum();
        if v_sq_sum < 1e-30 {
            continue;
        }
        let beta = 2.0 / v_sq_sum;

        for jj in j..n {
            let w: f64 = v
                .iter()
                .enumerate()
                .map(|(i, vi)| vi * data[(j + i) * n + jj])
                .sum::<f64>();
            for i in 0..col_len {
                data[(j + i) * n + jj] -= beta * w * v[i];
            }
        }

        // Store v[i]/v[0] for i > 0 so the implicit leading component is 1.
        // Also store beta * v[0]^2 so reconstruction with normalized v' works as:
        //   H = I - beta' * v' * v'^T  where v' has v'[0]=1.
        let v0 = v[0];
        betas[j] = beta * v0 * v0;
        for i in 1..col_len {
            data[(j + i) * n + j] = v[i] / v0;
        }
    }

    Ok(QrFactors { data, betas, m, n })
}

/// Reconstruct the full explicit Q matrix (m×m row-major) from compact QR factors.
pub fn q_from_qr(factors: &QrFactors) -> Vec<f64> {
    let m = factors.m;
    let k = m.min(factors.n);
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    for j in (0..k).rev() {
        let col_len = m - j;
        let mut v = vec![1.0f64];
        for i in 1..col_len {
            v.push(factors.data[(j + i) * factors.n + j]);
        }
        let beta = factors.betas[j];
        for jj in 0..m {
            let w: f64 = v
                .iter()
                .enumerate()
                .map(|(i, vi)| vi * q[(j + i) * m + jj])
                .sum::<f64>();
            for i in 0..col_len {
                q[(j + i) * m + jj] -= beta * w * v[i];
            }
        }
    }
    q
}

/// Apply Q^T to a vector b (length m) in-place: b := Q^T b
pub(crate) fn apply_qt_to_vec(factors: &QrFactors, b: &mut [f64]) {
    let k = factors.m.min(factors.n);
    for j in 0..k {
        let col_len = factors.m - j;
        let mut v = vec![1.0f64];
        for i in 1..col_len {
            v.push(factors.data[(j + i) * factors.n + j]);
        }
        let beta = factors.betas[j];
        let w: f64 = v
            .iter()
            .enumerate()
            .map(|(i, vi)| vi * b[j + i])
            .sum::<f64>();
        for i in 0..col_len {
            b[j + i] -= beta * w * v[i];
        }
    }
}

/// SVD result: A ≈ U * diag(s) * V^T
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors, row-major m×m orthogonal matrix.
    pub u: Vec<f64>,
    /// Singular values in descending order, length min(m, n).
    pub s: Vec<f64>,
    /// Right singular vectors, row-major n×n orthogonal matrix.
    pub v: Vec<f64>,
    /// Number of rows in the original matrix.
    pub m: usize,
    /// Number of columns in the original matrix.
    pub n: usize,
}

/// Jacobi SVD of a square matrix B (n×n, column-major storage).
///
/// Uses the symmetric one-sided Jacobi algorithm on B^T B to compute eigenvalues,
/// then recovers U and V. This correctly handles rank-deficient matrices because
/// the symmetric eigenproblem B^T B is always solvable even when columns of B are parallel.
///
/// Returns `(u_cols, s, v_cols)` where both u_cols and v_cols are column-major n×n orthogonal.
fn jacobi_svd_square(b_cols: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Build C = B^T B (n×n symmetric, row-major for eigensolver)
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let dot: f64 = (0..n).map(|k| b_cols[k + i * n] * b_cols[k + j * n]).sum();
            c[i * n + j] = dot;
            c[j * n + i] = dot;
        }
    }

    // Jacobi eigendecomposition of the symmetric matrix C = V * diag(lambda) * V^T.
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    const MAX_SWEEPS: usize = 100;
    const TOL: f64 = 1e-14;

    for _sweep in 0..MAX_SWEEPS {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let cpq = c[p * n + q];
                if cpq.abs() <= TOL * (c[p * n + p] * c[q * n + q]).abs().sqrt() {
                    continue;
                }
                converged = false;

                // Symmetric Jacobi rotation to zero c[p,q]
                let theta = (c[q * n + q] - c[p * n + p]) / (2.0 * cpq);
                let t = if theta.abs() < 1e15 {
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    sign / (theta.abs() + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (2.0 * theta)
                };
                let cos_a = 1.0 / (1.0 + t * t).sqrt();
                let sin_a = t * cos_a;

                // Update C: symmetric update for (p,q) rotation
                let cpp = c[p * n + p];
                let cqq = c[q * n + q];
                c[p * n + p] = cpp - t * cpq;
                c[q * n + q] = cqq + t * cpq;
                c[p * n + q] = 0.0;
                c[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let crp = c[r * n + p];
                        let crq = c[r * n + q];
                        let new_crp = cos_a * crp - sin_a * crq;
                        let new_crq = sin_a * crp + cos_a * crq;
                        c[r * n + p] = new_crp;
                        c[p * n + r] = new_crp;
                        c[r * n + q] = new_crq;
                        c[q * n + r] = new_crq;
                    }
                }

                // Accumulate V
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = cos_a * vip - sin_a * viq;
                    v[i * n + q] = sin_a * vip + cos_a * viq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Eigenvalues of B^T B are lambda[j] = c[j * n + j]; singular values = sqrt(max(0, lambda))
    let singular_vals: Vec<f64> = (0..n).map(|j| c[j * n + j].max(0.0).sqrt()).collect();

    // Compute U: U = B * V * diag(1/sigma) [column j of U = B * v[:,j] / sigma_j]
    // b_cols is column-major n×n, v is row-major n×n.
    let mut u_cols = vec![0.0f64; n * n];
    for j in 0..n {
        let sigma = singular_vals[j];
        if sigma > 1e-15 {
            // u_col_j = B * v_col_j / sigma
            // B in column-major: b_cols[i + p*n] = B[i,p]
            // v in row-major: v[i * n + j] = V[i,j] = v_col_j[i]
            for i in 0..n {
                let mut s = 0.0f64;
                for p in 0..n {
                    s += b_cols[i + p * n] * v[p * n + j];
                }
                u_cols[i + j * n] = s / sigma;
            }
        }
        // else u_col_j stays zero (rank-deficient direction)
    }

    (u_cols, singular_vals, v)
}

/// SVD of A (m×n row-major, m ≥ n) via R-SVD: first QR, then Jacobi on R.
///
/// Returns `SvdResult` with singular values in descending order.
/// `u` is m×m, `s` has length n, `v` is n×n (all row-major).
pub fn svd(a: &[f64], m: usize, n: usize) -> Result<SvdResult, EmlError> {
    if a.len() != m * n {
        return Err(EmlError::DimensionMismatch(m * n, a.len()));
    }
    if m < n {
        return Err(EmlError::DimensionMismatch(m, n));
    }

    // Step 1: QR decomposition of A.  A = Q_full * [R; 0]
    let qr_f = qr(a, m, n)?;

    // Extract the n×n upper-triangular R from qr_f.data (row-major, stride n).
    let mut r_cols = vec![0.0f64; n * n]; // column-major n×n
    for i in 0..n {
        for j in i..n {
            r_cols[i + j * n] = qr_f.data[i * n + j];
        }
    }

    // Step 2: Jacobi SVD on the n×n matrix R.
    // After Jacobi sweeps: R ≈ U_r * diag(s) * V_r^T (column-major U_r, V_r).
    let (u_r_cols, singular_vals, v_r_row) = jacobi_svd_square(&r_cols, n);

    // Step 3: Compute Q_full (m×m row-major) from QR factors.
    let q_full = q_from_qr(&qr_f);

    // Step 4: Sort singular values descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a_idx, &b_idx| singular_vals[b_idx].total_cmp(&singular_vals[a_idx]));

    let s_sorted: Vec<f64> = order.iter().map(|&i| singular_vals[i]).collect();

    // Step 5: Reorder U_r and V_r according to sort order, then combine with Q.
    // U = Q_full * U_r_reordered (m×m * m×n => but U_r is n×n; pad to m×n first)
    // u_r is column-major n×n; we need row-major m×m for full U.
    // Actually the final U columns are: Q_full * u_r[:,j] (m×1 each), for j in sorted order.
    let mut u_sorted = vec![0.0f64; m * m];
    for (new_j, &old_j) in order.iter().enumerate() {
        // u_r column old_j (in column-major storage): u_r_cols[i + old_j * n] for i in 0..n
        // We want Q_full (m×m row-major) * this n-vector (padded to m with zeros) = column new_j of U
        for i in 0..m {
            let mut s = 0.0f64;
            for k in 0..n {
                // Q_full[i, k] = q_full[i * m + k]
                s += q_full[i * m + k] * u_r_cols[k + old_j * n];
            }
            u_sorted[i * m + new_j] = s;
        }
    }
    // Remaining columns of U (m×m) are in the null space of A; fill with Gram-Schmidt if needed.
    // For the purposes of pinv and least-squares we only use the first n columns, so fill the
    // remaining columns with the last (m-n) columns of Q_full unchanged.
    for new_j in n..m {
        let q_col = new_j; // Use Q columns n..m as the remaining U columns
        for i in 0..m {
            u_sorted[i * m + new_j] = q_full[i * m + q_col];
        }
    }

    // V is V_r reordered (row-major n×n).
    let mut v_sorted = vec![0.0f64; n * n];
    for (new_j, &old_j) in order.iter().enumerate() {
        for i in 0..n {
            v_sorted[i * n + new_j] = v_r_row[i * n + old_j];
        }
    }

    if s_sorted.iter().any(|s| !s.is_finite()) {
        return Err(EmlError::SingularMatrix);
    }

    Ok(SvdResult {
        u: u_sorted,
        s: s_sorted,
        v: v_sorted,
        m,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_rm(a: &[f64], p: usize, q: usize, b: &[f64], r: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; p * r];
        for i in 0..p {
            for k in 0..q {
                for j in 0..r {
                    c[i * r + j] += a[i * q + k] * b[k * r + j];
                }
            }
        }
        c
    }

    fn transpose_rm(a: &[f64], p: usize, q: usize) -> Vec<f64> {
        let mut t = vec![0.0f64; q * p];
        for i in 0..p {
            for j in 0..q {
                t[j * p + i] = a[i * q + j];
            }
        }
        t
    }

    #[test]
    fn test_qr_qt_q_is_identity() {
        let a = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = 3;
        let n = 2;
        let factors = qr(&a, m, n).unwrap();
        let q = q_from_qr(&factors);
        let qt = transpose_rm(&q, m, m);
        let qtq = matmul_rm(&qt, m, m, &q, m);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * m + j] - expected).abs() < 1e-10,
                    "Q^T Q [{i},{j}] = {} (expected {})",
                    qtq[i * m + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_q_r_equals_a() {
        let a = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = 3;
        let n = 2;
        let factors = qr(&a, m, n).unwrap();
        let q = q_from_qr(&factors);
        let mut r_full = vec![0.0f64; m * n];
        for i in 0..n {
            for j in i..n {
                r_full[i * n + j] = factors.data[i * n + j];
            }
        }
        let qr_prod = matmul_rm(&q, m, m, &r_full, n);
        for (idx, (v1, v2)) in qr_prod.iter().zip(a.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-9,
                "QR[{idx}] = {} != A[{idx}] = {}",
                v1,
                v2
            );
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = vec![
            1.0_f64, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ];
        let m = 4;
        let n = 3;
        let svd_r = svd(&a, m, n).unwrap();
        let mut diag_s = vec![0.0f64; m * n];
        for i in 0..n {
            diag_s[i * n + i] = svd_r.s[i];
        }
        let u_diag = matmul_rm(&svd_r.u, m, m, &diag_s, n);
        let vt = transpose_rm(&svd_r.v, n, n);
        let recon = matmul_rm(&u_diag, m, n, &vt, n);
        for (idx, (v1, v2)) in recon.iter().zip(a.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "SVD recon[{idx}] = {} != A[{idx}] = {}",
                v1,
                v2
            );
        }
    }

    #[test]
    fn test_svd_rank_deficient_small_sigma() {
        let a = vec![1.0_f64, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        let m = 3;
        let n = 3;
        let svd_r = svd(&a, m, n).unwrap();
        assert!(svd_r.s[1].abs() < 1e-8, "s[1] = {}", svd_r.s[1]);
        assert!(svd_r.s[2].abs() < 1e-8, "s[2] = {}", svd_r.s[2]);
    }

    #[test]
    fn test_qr_dimension_mismatch() {
        let a = vec![1.0_f64; 5];
        assert!(qr(&a, 3, 2).is_err());
    }

    #[test]
    fn test_svd_dimension_mismatch() {
        let a = vec![1.0_f64; 5];
        assert!(svd(&a, 3, 2).is_err());
    }
}
