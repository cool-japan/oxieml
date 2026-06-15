//! Linear system solvers (Cholesky, LU, least-squares, pseudo-inverse).

use super::decomp::{apply_qt_to_vec, qr, svd};
use crate::error::EmlError;

fn cholesky_factor(a: &mut [f64], n: usize) -> Result<(), EmlError> {
    let scale = (0..n).map(|k| a[k * n + k].abs()).fold(1.0_f64, f64::max);

    for j in 0..n {
        let mut sum = a[j * n + j];
        for k in 0..j {
            sum -= a[j * n + k] * a[j * n + k];
        }
        if sum <= 1e-14 * scale {
            return Err(EmlError::NotSpd);
        }
        a[j * n + j] = sum.sqrt();

        let diag_j = a[j * n + j];
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / diag_j;
        }
    }
    Ok(())
}

fn cholesky_solve_with_factor(l: &[f64], b: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Solve the symmetric positive-definite system `A x = b` via Cholesky.
pub fn solve_spd_cholesky(a: &mut [f64], b: &mut [f64], n: usize) -> Result<(), EmlError> {
    cholesky_factor(a, n)?;
    cholesky_solve_with_factor(a, b, n);
    Ok(())
}

/// Solve the linear system `A x = b` via LU decomposition with partial row pivoting.
pub fn solve_lu(a: &mut [f64], b: &mut [f64], n: usize) -> Result<(), EmlError> {
    for k in 0..n {
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return Err(EmlError::SingularMatrix);
        }

        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
            b.swap(k, max_row);
        }

        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            for j in (k + 1)..n {
                let updated = a[i * n + j] - factor * a[k * n + j];
                a[i * n + j] = updated;
            }
        }
    }

    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s;
    }

    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }

    Ok(())
}

/// Solve `A x = b` by attempting Cholesky first; fall back to LU.
pub fn solve_normal_equations(a: &[f64], b: &mut [f64], n: usize) -> Result<(), EmlError> {
    let mut a_chol = a.to_vec();
    match solve_spd_cholesky(&mut a_chol, b, n) {
        Ok(()) => Ok(()),
        Err(EmlError::NotSpd) => {
            let mut a_lu = a.to_vec();
            solve_lu(&mut a_lu, b, n)
        }
        Err(e) => Err(e),
    }
}

/// Compute the inverse of a symmetric positive-definite matrix via Cholesky.
pub fn invert_spd(a: &[f64], n: usize) -> Result<Vec<f64>, EmlError> {
    let mut l = a.to_vec();
    cholesky_factor(&mut l, n)?;

    let mut inv = vec![0.0_f64; n * n];
    for j in 0..n {
        let mut col = vec![0.0_f64; n];
        col[j] = 1.0_f64;
        cholesky_solve_with_factor(&l, &mut col, n);
        for i in 0..n {
            inv[i * n + j] = col[i];
        }
    }
    Ok(inv)
}

/// Back-substitution for upper triangular system R x = b.
pub(crate) fn back_sub_upper_from_qr(data: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut x = b[..n].to_vec();
    for j in (0..n).rev() {
        let r_jj = data[j * n + j];
        if r_jj.abs() < 1e-15 {
            return None;
        }
        x[j] /= r_jj;
        for i in 0..j {
            x[i] -= data[i * n + j] * x[j];
        }
    }
    Some(x)
}

/// Solve least-squares min||Ax - b|| using Householder QR.
pub fn solve_least_squares(a: &[f64], b: &[f64], m: usize, n: usize) -> Result<Vec<f64>, EmlError> {
    if a.len() != m * n {
        return Err(EmlError::DimensionMismatch(m * n, a.len()));
    }
    if b.len() != m {
        return Err(EmlError::DimensionMismatch(m, b.len()));
    }
    let factors = qr(a, m, n)?;
    let mut rhs = b.to_vec();
    apply_qt_to_vec(&factors, &mut rhs);
    back_sub_upper_from_qr(&factors.data, &rhs, n).ok_or(EmlError::SingularMatrix)
}

/// Moore-Penrose pseudo-inverse of A (m×n row-major).
/// Returns n×m pseudo-inverse as a row-major n×m matrix.
pub fn pinv(a: &[f64], m: usize, n: usize, rcond: Option<f64>) -> Result<Vec<f64>, EmlError> {
    if a.len() != m * n {
        return Err(EmlError::DimensionMismatch(m * n, a.len()));
    }
    let svd_r = svd(a, m, n)?;
    let s0 = svd_r.s.first().copied().unwrap_or(0.0);
    let thresh = rcond.unwrap_or(1e-12) * s0;

    let mut result = vec![0.0f64; n * m];
    for j in 0..svd_r.s.len() {
        if svd_r.s[j] <= thresh {
            continue;
        }
        let inv_s = 1.0 / svd_r.s[j];
        for i in 0..n {
            let v_ij = svd_r.v[i * n + j];
            for k in 0..m {
                let u_kj = svd_r.u[k * m + j];
                result[i * m + k] += inv_s * v_ij * u_kj;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_cholesky_solve() {
        let mut a = [4.0_f64, 2.0, 2.0, 3.0];
        let mut b = [6.0_f64, 5.0];
        solve_spd_cholesky(&mut a, &mut b, 2).expect("Cholesky solve should succeed");
        assert!(approx_eq(b[0], 1.0, EPS), "x[0] = {}", b[0]);
        assert!(approx_eq(b[1], 1.0, EPS), "x[1] = {}", b[1]);
    }

    #[test]
    fn test_lu_solve() {
        let mut a = [4.0_f64, 2.0, 2.0, 3.0];
        let mut b = [6.0_f64, 5.0];
        solve_lu(&mut a, &mut b, 2).expect("LU solve should succeed");
        assert!(approx_eq(b[0], 1.0, EPS), "x[0] = {}", b[0]);
        assert!(approx_eq(b[1], 1.0, EPS), "x[1] = {}", b[1]);
    }

    #[test]
    fn test_solve_normal_equations_round_trip() {
        use crate::linalg::builders::{jtj, jtr};
        let j = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = [1.0_f64, 1.0, 2.0];
        let a = jtj(&j, 3, 2, 0.0);
        let mut b = jtr(&j, &y, 3, 2);
        solve_normal_equations(&a, &mut b, 2).expect("solve_normal_equations should succeed");
        assert!(approx_eq(b[0], 1.0, EPS), "x[0] = {}", b[0]);
        assert!(approx_eq(b[1], 1.0, EPS), "x[1] = {}", b[1]);
    }

    #[test]
    fn test_invert_spd() {
        let a = [2.0_f64, 1.0, 1.0, 2.0];
        let inv = invert_spd(&a, 2).expect("invert_spd should succeed");

        assert!(approx_eq(inv[0], 2.0 / 3.0, EPS), "inv[0,0] = {}", inv[0]);
        assert!(approx_eq(inv[1], -1.0 / 3.0, EPS), "inv[0,1] = {}", inv[1]);
        assert!(approx_eq(inv[2], -1.0 / 3.0, EPS), "inv[1,0] = {}", inv[2]);
        assert!(approx_eq(inv[3], 2.0 / 3.0, EPS), "inv[1,1] = {}", inv[3]);

        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += a[i * n + k] * inv[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(s, expected, 1e-10),
                    "A*A\u{207b}\u{00b9}[{i},{j}] = {s} (expected {expected})"
                );
            }
        }
    }

    #[test]
    fn test_not_spd() {
        let mut a = [-1.0_f64, 0.0, 0.0, 1.0];
        let mut b = [0.0_f64, 0.0];
        let result = solve_spd_cholesky(&mut a, &mut b, 2);
        assert_eq!(result, Err(EmlError::NotSpd));
    }

    #[test]
    fn test_singular_matrix() {
        let mut a = [1.0_f64, 2.0, 2.0, 4.0];
        let mut b = [1.0_f64, 2.0];
        let result = solve_lu(&mut a, &mut b, 2);
        assert_eq!(result, Err(EmlError::SingularMatrix));
    }

    #[test]
    fn test_3x3_cholesky() {
        #[rustfmt::skip]
        let mut a = [
            4.0_f64, 2.0, 0.0,
            2.0,     5.0, 2.0,
            0.0,     2.0, 5.0,
        ];
        let mut b = [6.0_f64, 9.0, 7.0];
        solve_spd_cholesky(&mut a, &mut b, 3).expect("3x3 Cholesky should succeed");
        assert!(approx_eq(b[0], 1.0, EPS), "x[0] = {}", b[0]);
        assert!(approx_eq(b[1], 1.0, EPS), "x[1] = {}", b[1]);
        assert!(approx_eq(b[2], 1.0, EPS), "x[2] = {}", b[2]);
    }

    #[test]
    fn test_3x3_lu() {
        #[rustfmt::skip]
        let mut a = [
            2.0_f64, 1.0, 1.0,
            4.0,     3.0, 3.0,
            8.0,     7.0, 9.0,
        ];
        let mut b = [1.0_f64, 1.0, 1.0];
        solve_lu(&mut a, &mut b, 3).expect("3x3 LU should succeed");
        #[rustfmt::skip]
        let a_orig = [
            2.0_f64, 1.0, 1.0,
            4.0,     3.0, 3.0,
            8.0,     7.0, 9.0,
        ];
        for i in 0..3 {
            let mut s = 0.0_f64;
            for j in 0..3 {
                s += a_orig[i * 3 + j] * b[j];
            }
            let b_orig = 1.0_f64;
            assert!(approx_eq(s, b_orig, 1e-9), "residual[{i}] = {s}");
        }
    }
}

#[cfg(test)]
mod tests_qr_svd {
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
    fn test_solve_least_squares_exact() {
        let a = vec![1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = solve_least_squares(&a, &b, 3, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-8, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-8, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_pinv_moore_penrose_cond1() {
        let a = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = 3;
        let n = 2;
        let ap = pinv(&a, m, n, None).unwrap();
        let aap = matmul_rm(&a, m, n, &ap, m);
        let aapa = matmul_rm(&aap, m, m, &a, n);
        for (v1, v2) in aapa.iter().zip(a.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "cond1 failed: {} != {}", v1, v2);
        }
    }

    #[test]
    fn test_pinv_hermitian_projectors() {
        let a = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0];
        let m = 3;
        let n = 2;
        let ap = pinv(&a, m, n, None).unwrap();
        let aap = matmul_rm(&a, m, n, &ap, m);
        let aap_t = transpose_rm(&aap, m, m);
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (aap[i * m + j] - aap_t[i * m + j]).abs() < 1e-6,
                    "(A A^+)[{i},{j}] is not symmetric"
                );
            }
        }
    }

    #[test]
    fn test_solve_least_squares_overdetermined() {
        let a = vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = solve_least_squares(&a, &b, 4, 2).unwrap();
        assert!(x[0].is_finite(), "x[0] must be finite");
        assert!(x[1].is_finite(), "x[1] must be finite");
    }
}
