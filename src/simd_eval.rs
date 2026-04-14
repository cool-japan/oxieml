//! SIMD-accelerated batch evaluation of flat `OxiOp` instruction sequences.
//!
//! Only compiled when the `simd` feature is enabled (which activates oxiblas-core).
//!
//! # Architecture dispatch
//!
//! - **AArch64**: NEON `F64x2` (2 lanes) is always available; we also use `F64x4`
//!   (emulated 4 lanes) if `detect_simd_level()` reports Simd256+ (future SVE).
//! - **x86_64**: runtime dispatch via `detect_simd_level()`:
//!   - `Simd256`/`Simd512` → `F64x4` (AVX2, 4 lanes)
//!   - `Simd128` → `F64x2Sse` (SSE, 2 lanes)
//!   - `Scalar` → scalar fallback
//! - **Other architectures**: scalar fallback.
//!
//! # Transcendental operations
//!
//! `oxiblas_core::simd::SimdRegister` does not expose `exp`/`ln`/`sin`/`cos`.
//! For those ops we extract each lane to scalar, compute with `f64::exp()` etc.,
//! then re-insert. Pure Rust, bit-exact, correct.

use crate::lower::{LoweredOp, OxiOp};
use oxiblas_core::simd::{SimdLevel, SimdRegister, detect_simd_level};

#[cfg(target_arch = "aarch64")]
use oxiblas_core::simd::aarch64::{F64x2, F64x4 as F64x4Aarch};

#[cfg(target_arch = "x86_64")]
use oxiblas_core::simd::x86_64::{F64x2Sse, F64x4};

/// Minimum number of data rows before activating rayon parallelism (simd+parallel).
#[cfg(feature = "parallel")]
const PARALLEL_SIMD_THRESHOLD: usize = 512;

/// Evaluate a batch of data points over the flat instruction list using SIMD.
///
/// Called by [`LoweredOp::eval_batch`] when the `simd` feature is active.
/// Dispatches to the best available SIMD width at runtime.
pub fn eval_batch_simd(ops: &[OxiOp], data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    #[cfg(target_arch = "aarch64")]
    {
        match detect_simd_level() {
            SimdLevel::Simd256 | SimdLevel::Simd512 => {
                eval_chunks_dispatch::<F64x4Aarch>(ops, data)
            }
            SimdLevel::Simd128 => eval_chunks_dispatch::<F64x2>(ops, data),
            SimdLevel::Scalar => LoweredOp::eval_batch_scalar_from_ops(ops, data),
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        match detect_simd_level() {
            SimdLevel::Simd256 | SimdLevel::Simd512 => eval_chunks_dispatch::<F64x4>(ops, data),
            SimdLevel::Simd128 => eval_chunks_dispatch::<F64x2Sse>(ops, data),
            SimdLevel::Scalar => LoweredOp::eval_batch_scalar_from_ops(ops, data),
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        LoweredOp::eval_batch_scalar_from_ops(ops, data)
    }
}

/// Dispatch to parallel or sequential SIMD evaluation based on feature flags and batch size.
fn eval_chunks_dispatch<V>(ops: &[OxiOp], data: &[Vec<f64>]) -> Vec<f64>
where
    V: SimdRegister<Scalar = f64>,
{
    #[cfg(feature = "parallel")]
    {
        if data.len() >= PARALLEL_SIMD_THRESHOLD {
            return eval_chunks_parallel::<V>(ops, data);
        }
    }
    eval_chunks::<V>(ops, data)
}

/// Sequential SIMD evaluation: process data in groups of `LANES`, scalar remainder.
fn eval_chunks<V>(ops: &[OxiOp], data: &[Vec<f64>]) -> Vec<f64>
where
    V: SimdRegister<Scalar = f64>,
{
    let lanes = V::LANES;
    let n = data.len();
    let full_chunks = n / lanes;

    let mut results = Vec::with_capacity(n);

    for chunk_idx in 0..full_chunks {
        let base = chunk_idx * lanes;
        let reg = eval_simd_chunk::<V>(ops, data, base);
        for i in 0..lanes {
            results.push(reg.extract(i));
        }
    }

    let remainder_start = full_chunks * lanes;
    for row in data.iter().take(n).skip(remainder_start) {
        results.push(LoweredOp::eval_ops(ops, row));
    }

    results
}

/// Evaluate exactly `LANES` rows simultaneously using SIMD.
fn eval_simd_chunk<V>(ops: &[OxiOp], data: &[Vec<f64>], base: usize) -> V
where
    V: SimdRegister<Scalar = f64>,
{
    let lanes = V::LANES;
    let mut stack: Vec<V> = Vec::with_capacity(ops.len());

    for op in ops {
        match op {
            OxiOp::Const(c) => stack.push(V::splat(*c)),
            OxiOp::Var(i) => {
                let mut reg = V::zero();
                for lane in 0..lanes {
                    let val = data[base + lane].get(*i).copied().unwrap_or(f64::NAN);
                    reg = reg.insert(lane, val);
                }
                stack.push(reg);
            }
            OxiOp::Add => {
                let b = stack.pop().unwrap_or_else(V::zero);
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(a.add(b));
            }
            OxiOp::Sub => {
                let b = stack.pop().unwrap_or_else(V::zero);
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(a.sub(b));
            }
            OxiOp::Mul => {
                let b = stack.pop().unwrap_or_else(V::zero);
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(a.mul(b));
            }
            OxiOp::Div => {
                let b = stack.pop().unwrap_or_else(V::zero);
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(a.div(b));
            }
            OxiOp::Neg => {
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(V::zero().sub(a));
            }
            OxiOp::Exp => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).exp());
                }
                stack.push(reg);
            }
            OxiOp::Ln => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).ln());
                }
                stack.push(reg);
            }
            OxiOp::Sin => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).sin());
                }
                stack.push(reg);
            }
            OxiOp::Cos => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).cos());
                }
                stack.push(reg);
            }
            OxiOp::Pow => {
                let b = stack.pop().unwrap_or_else(V::zero);
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).powf(b.extract(lane)));
                }
                stack.push(reg);
            }
        }
    }

    stack.pop().unwrap_or_else(V::zero)
}

/// Parallel SIMD evaluation: split into rayon chunks, each runs `eval_chunks`.
///
/// Only active when both `simd` and `parallel` features are enabled.
#[cfg(feature = "parallel")]
fn eval_chunks_parallel<V>(ops: &[OxiOp], data: &[Vec<f64>]) -> Vec<f64>
where
    V: SimdRegister<Scalar = f64>,
{
    use rayon::prelude::*;
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = data.len().div_ceil(num_threads).max(V::LANES);
    data.par_chunks(chunk_size)
        .flat_map_iter(|chunk| eval_chunks::<V>(ops, chunk))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Canonical;

    fn exp_lowered() -> crate::lower::LoweredOp {
        let x = crate::tree::EmlTree::var(0);
        Canonical::exp(&x).lower()
    }

    #[test]
    fn test_eval_batch_simd_matches_scalar() {
        let lowered = exp_lowered();
        let ops = lowered.to_oxiblas_ops();

        let data: Vec<Vec<f64>> = (0..256).map(|i| vec![i as f64 * 0.01]).collect();
        let simd_results = eval_batch_simd(&ops, &data);
        let scalar_results = LoweredOp::eval_batch_scalar_from_ops(&ops, &data);

        assert_eq!(simd_results.len(), scalar_results.len());
        for (i, (s, r)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-12,
                "row {i}: SIMD={s} scalar={r} diff={}",
                (s - r).abs()
            );
        }
    }

    #[test]
    fn test_eval_batch_simd_transcendentals() {
        use crate::lower::LoweredOp as L;
        // sin(x) + cos(x) built directly as LoweredOp
        let lowered = L::Add(
            Box::new(L::Sin(Box::new(L::Var(0)))),
            Box::new(L::Cos(Box::new(L::Var(0)))),
        );
        let ops = lowered.to_oxiblas_ops();
        let data: Vec<Vec<f64>> = (0..128).map(|i| vec![i as f64 * 0.05]).collect();

        let simd_results = eval_batch_simd(&ops, &data);
        let scalar_results = LoweredOp::eval_batch_scalar_from_ops(&ops, &data);

        assert_eq!(simd_results.len(), 128);
        for (i, (s, r)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-12,
                "sin+cos row {i}: SIMD={s} scalar={r}"
            );
        }
    }

    #[test]
    fn test_eval_batch_simd_remainder() {
        let lowered = exp_lowered();
        let ops = lowered.to_oxiblas_ops();

        let data: Vec<Vec<f64>> = (0..7).map(|i| vec![i as f64 * 0.3]).collect();
        let simd_results = eval_batch_simd(&ops, &data);
        let scalar_results = LoweredOp::eval_batch_scalar_from_ops(&ops, &data);

        assert_eq!(simd_results.len(), 7);
        for (i, (s, r)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!((s - r).abs() < 1e-12, "remainder row {i}: {s} vs {r}");
        }
    }

    #[test]
    fn test_eval_batch_simd_empty() {
        let lowered = exp_lowered();
        let ops = lowered.to_oxiblas_ops();
        let results = eval_batch_simd(&ops, &[]);
        assert!(results.is_empty());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_eval_batch_simd_parallel_matches_scalar() {
        let lowered = exp_lowered();
        let ops = lowered.to_oxiblas_ops();

        let data: Vec<Vec<f64>> = (0..1000).map(|i| vec![i as f64 * 0.001]).collect();
        let simd_results = eval_batch_simd(&ops, &data);
        let scalar_results = LoweredOp::eval_batch_scalar_from_ops(&ops, &data);

        assert_eq!(simd_results.len(), 1000);
        for (i, (s, r)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!((s - r).abs() < 1e-12, "parallel row {i}: {s} vs {r}");
        }
    }
}
