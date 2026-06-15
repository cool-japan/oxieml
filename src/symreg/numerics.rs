//! Numerical differentiation helpers for ODE discovery.
//!
//! These are private utilities used by [`super::SymRegEngine::discover_ode`]
//! to estimate time-derivatives from trajectory data before running symbolic
//! regression on the `(x, dx/dt)` pairs.

/// Estimate `dx/dt` via central differences.
///
/// Interior points use the central-difference formula
/// `(x[i+1] - x[i-1]) / (2*dt)`.
/// The first endpoint uses forward differences `(x[1] - x[0]) / dt`
/// and the last endpoint uses backward differences `(x[n-1] - x[n-2]) / dt`.
///
/// Returns a `Vec` of length equal to `x.len()`.
pub(super) fn central_differences(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut dx = vec![0.0; n];
    // Endpoints: forward / backward
    dx[0] = (x[1] - x[0]) / dt;
    dx[n - 1] = (x[n - 1] - x[n - 2]) / dt;
    // Interior: central
    for i in 1..n - 1 {
        dx[i] = (x[i + 1] - x[i - 1]) / (2.0 * dt);
    }
    dx
}

/// Savitzky-Golay smoothed derivative with window=5, polynomial degree=2.
///
/// Falls back to [`central_differences`] when `n < 5`.
///
/// The SG derivative coefficients for window=5, poly=2 are
/// `[-2, -1, 0, 1, 2]`, normalised by `10 * dt`.
pub(super) fn savitzky_golay_derivative(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    if n < 5 {
        return central_differences(x, dt);
    }
    let coeffs = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let norm = 10.0 * dt;
    // Fill all positions with central differences first (handles endpoints)
    let mut dx = central_differences(x, dt);
    // Overwrite interior positions (indices 2..n-2) with SG filter
    for i in 2..n - 2 {
        dx[i] = coeffs
            .iter()
            .enumerate()
            .map(|(k, &c)| c * x[i + k - 2])
            .sum::<f64>()
            / norm;
    }
    dx
}

/// Compute the first spatial derivative using finite differences.
///
/// - `accuracy = 2`: standard central differences with one-sided edges
///   (`interior: (u[i+1]-u[i-1])/(2·dx)`, left/right edges: 2nd-order one-sided).
/// - `accuracy = 4`: 5-point central stencil `(-u[i+2]+8u[i+1]-8u[i-1]+u[i-2])/(12dx)`
///   for interior, falling back to 3-point near the boundary.
///
/// Returns a `Vec<f64>` of the same length as `field`.
pub(super) fn first_derivative_1d(field: &[f64], dx: f64, accuracy: usize) -> Vec<f64> {
    let n = field.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut out = vec![0.0_f64; n];

    // Boundary (2nd-order one-sided stencils for both endpoints)
    out[0] = (-3.0 * field[0] + 4.0 * field[1] - field[2.min(n - 1)]) / (2.0 * dx);
    out[n - 1] =
        (3.0 * field[n - 1] - 4.0 * field[n - 2] + field[n.saturating_sub(3)]) / (2.0 * dx);

    if accuracy >= 4 && n >= 5 {
        // 5-point interior stencil, fall back to 3-point near edges
        for i in 1..n - 1 {
            if i >= 2 && i + 2 < n {
                // Full 5-point
                out[i] = (-field[i + 2] + 8.0 * field[i + 1] - 8.0 * field[i - 1] + field[i - 2])
                    / (12.0 * dx);
            } else {
                // 3-point central
                out[i] = (field[i + 1] - field[i - 1]) / (2.0 * dx);
            }
        }
    } else {
        // Standard central differences
        for i in 1..n - 1 {
            out[i] = (field[i + 1] - field[i - 1]) / (2.0 * dx);
        }
    }

    out
}

/// Compute the second spatial derivative using finite differences.
///
/// - `accuracy = 2`: standard 3-point stencil `(u[i+1]-2u[i]+u[i-1])/dx²`.
/// - `accuracy = 4`: 5-point stencil `(-u[i+2]+16u[i+1]-30u[i]+16u[i-1]-u[i-2])/(12dx²)`.
///
/// Boundary points always use the 3-point stencil (extrapolating one phantom point
/// via reflection to maintain the same array size).
pub(super) fn second_derivative_1d(field: &[f64], dx: f64, accuracy: usize) -> Vec<f64> {
    let n = field.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let dx2 = dx * dx;
    let mut out = vec![0.0_f64; n];

    // Boundary: use one-sided 3-point via reflection
    out[0] = (field[2.min(n - 1)] - 2.0 * field[1.min(n - 1)] + field[0]) / dx2;
    out[n - 1] =
        (field[n - 1] - 2.0 * field[n.saturating_sub(2)] + field[n.saturating_sub(3)]) / dx2;

    if accuracy >= 4 && n >= 5 {
        for i in 1..n - 1 {
            if i >= 2 && i + 2 < n {
                out[i] = (-field[i + 2] + 16.0 * field[i + 1] - 30.0 * field[i]
                    + 16.0 * field[i - 1]
                    - field[i - 2])
                    / (12.0 * dx2);
            } else {
                out[i] = (field[i + 1] - 2.0 * field[i] + field[i - 1]) / dx2;
            }
        }
    } else {
        for i in 1..n - 1 {
            out[i] = (field[i + 1] - 2.0 * field[i] + field[i - 1]) / dx2;
        }
    }

    out
}

/// Compute the n-th order derivative of a 1-D array using central finite differences.
/// Supported orders: 0, 1, 2, 3. Higher orders use repeated first-order.
/// Returns a vector of the same length.
pub(super) fn nth_derivative_1d(data: &[f64], dx: f64, order: usize) -> Vec<f64> {
    match order {
        0 => data.to_vec(),
        1 => first_derivative_1d(data, dx, 2),
        2 => second_derivative_1d(data, dx, 2),
        3 => {
            let n = data.len();
            if n < 2 {
                return vec![0.0; n];
            }
            let dx3 = 2.0 * dx * dx * dx;
            (0..n)
                .map(|i| {
                    let im2 = data[i.saturating_sub(2)];
                    let im1 = data[i.saturating_sub(1)];
                    let ip1 = if i + 1 < n { data[i + 1] } else { data[n - 1] };
                    let ip2 = if i + 2 < n { data[i + 2] } else { data[n - 1] };
                    (-im2 + 2.0 * im1 - 2.0 * ip1 + ip2) / dx3
                })
                .collect()
        }
        _ => {
            let mut result = data.to_vec();
            for _ in 0..order {
                result = first_derivative_1d(&result, dx, 2);
            }
            result
        }
    }
}

/// Apply a 1-D derivative along a specific axis of a multi-dimensional array.
/// `data` is stored in C order (row-major): [nt, nx] for 1-spatial, [nt, nx, ny] for 2-spatial.
/// `full_shape` gives ALL sizes including time: [nt, nx] or [nt, nx, ny].
/// `axis` is the spatial axis index (0 = x, 1 = y).
/// `dx` is the grid spacing along that axis.
/// `order` is the derivative order.
pub(super) fn apply_axis_derivative(
    data: &[f64],
    full_shape: &[usize], // [nt, nx] or [nt, nx, ny]
    axis: usize,          // spatial axis (0=x, 1=y)
    dx: f64,
    order: usize,
) -> Vec<f64> {
    if full_shape.len() < 2 {
        return data.to_vec();
    }
    let nt = full_shape[0];
    let spatial_shape = &full_shape[1..];
    let n_spatial: usize = spatial_shape.iter().product();
    let mut result = vec![0.0f64; data.len()];

    for t in 0..nt {
        let time_offset = t * n_spatial;
        if spatial_shape.len() == 1 {
            // 1-D spatial: just differentiate the slice
            let nx = spatial_shape[0];
            let slice = &data[time_offset..time_offset + nx];
            let deriv = nth_derivative_1d(slice, dx, order);
            result[time_offset..time_offset + nx].copy_from_slice(&deriv);
        } else if spatial_shape.len() == 2 {
            // 2-D spatial: shape = [nt, nx, ny]
            let nx = spatial_shape[0];
            let ny = spatial_shape[1];
            if axis == 0 {
                // x-axis: for each y, extract fiber along x, differentiate, write back
                for yi in 0..ny {
                    let fiber: Vec<f64> =
                        (0..nx).map(|xi| data[time_offset + xi * ny + yi]).collect();
                    let deriv = nth_derivative_1d(&fiber, dx, order);
                    for (xi, &d) in deriv.iter().enumerate() {
                        result[time_offset + xi * ny + yi] = d;
                    }
                }
            } else {
                // y-axis: for each x, extract fiber along y, differentiate, write back
                for xi in 0..nx {
                    let fiber: Vec<f64> =
                        (0..ny).map(|yi| data[time_offset + xi * ny + yi]).collect();
                    let deriv = nth_derivative_1d(&fiber, dx, order);
                    for (yi, &d) in deriv.iter().enumerate() {
                        result[time_offset + xi * ny + yi] = d;
                    }
                }
            }
        } else if spatial_shape.len() == 3 {
            // 3-D spatial: shape = [nt, nx, ny, nz]
            let nx = spatial_shape[0];
            let ny = spatial_shape[1];
            let nz = spatial_shape[2];
            match axis {
                0 => {
                    for yi in 0..ny {
                        for zi in 0..nz {
                            let fiber: Vec<f64> = (0..nx)
                                .map(|xi| data[time_offset + xi * ny * nz + yi * nz + zi])
                                .collect();
                            let deriv = nth_derivative_1d(&fiber, dx, order);
                            for (xi, &d) in deriv.iter().enumerate() {
                                result[time_offset + xi * ny * nz + yi * nz + zi] = d;
                            }
                        }
                    }
                }
                1 => {
                    for xi in 0..nx {
                        for zi in 0..nz {
                            let fiber: Vec<f64> = (0..ny)
                                .map(|yi| data[time_offset + xi * ny * nz + yi * nz + zi])
                                .collect();
                            let deriv = nth_derivative_1d(&fiber, dx, order);
                            for (yi, &d) in deriv.iter().enumerate() {
                                result[time_offset + xi * ny * nz + yi * nz + zi] = d;
                            }
                        }
                    }
                }
                _ => {
                    // axis 2 = z
                    for xi in 0..nx {
                        for yi in 0..ny {
                            let fiber: Vec<f64> = (0..nz)
                                .map(|zi| data[time_offset + xi * ny * nz + yi * nz + zi])
                                .collect();
                            let deriv = nth_derivative_1d(&fiber, dx, order);
                            for (zi, &d) in deriv.iter().enumerate() {
                                result[time_offset + xi * ny * nz + yi * nz + zi] = d;
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_differences_linear() {
        // x(t) = 2t → dx/dt = 2 everywhere
        let dt = 0.1_f64;
        let n = 10;
        let x: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 * dt).collect();
        let dx = central_differences(&x, dt);
        for &d in &dx {
            assert!((d - 2.0).abs() < 1e-10, "expected 2.0, got {d}");
        }
    }

    #[test]
    fn central_differences_short_slice() {
        // Single element → returns [0.0]
        let dx = central_differences(&[42.0], 0.1);
        assert_eq!(dx, vec![0.0]);
    }

    #[test]
    fn savitzky_golay_falls_back_for_small_n() {
        // n=4 < 5 → should behave identically to central_differences
        let dt = 0.1_f64;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sg = savitzky_golay_derivative(&x, dt);
        let cd = central_differences(&x, dt);
        for (a, b) in sg.iter().zip(cd.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: sg={a}, cd={b}");
        }
    }

    #[test]
    fn savitzky_golay_linear_recovers_exact_slope() {
        // For a linear signal, SG deriv equals exact slope everywhere
        let dt = 0.1_f64;
        let n = 20;
        let x: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 * dt).collect();
        let dx = savitzky_golay_derivative(&x, dt);
        for &d in &dx {
            assert!((d - 3.0).abs() < 1e-9, "expected 3.0, got {d}");
        }
    }
}
