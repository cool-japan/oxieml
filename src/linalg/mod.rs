//! Pure-Rust dense linear algebra for the LM optimizer and PDE discovery.
//!
//! Provides:
//! - Normal-equation builders: [`jtj`], [`jtj_marquardt`], [`jtr`]
//! - SPD and general linear solvers: [`solve_spd_cholesky`], [`solve_lu`],
//!   [`solve_normal_equations`]
//! - SPD matrix inversion: [`invert_spd`]
//! - QR and SVD decompositions: [`qr`], [`svd`], [`q_from_qr`]
//! - Least-squares and pseudo-inverse: [`solve_least_squares`], [`pinv`]
//!
//! All matrices are row-major and indexed as `a[i * n + j]` for row `i`, column `j`.

pub mod builders;
pub mod decomp;
pub mod solve;

pub use builders::{jtj, jtj_marquardt, jtr};
pub use decomp::{QrFactors, SvdResult, q_from_qr, qr, svd};
pub use solve::{
    invert_spd, pinv, solve_least_squares, solve_lu, solve_normal_equations, solve_spd_cholesky,
};
