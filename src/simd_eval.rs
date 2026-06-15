//! SIMD-accelerated batch evaluation of flat `OxiOp` instruction sequences.
//!
//! Only compiled when the `simd` feature is enabled (which activates oxiblas-core).
//!
//! # Architecture dispatch
//!
//! - **AArch64**: NEON `F64x2` (2 lanes) is always available; we also use `F64x4`
//!   (emulated 4 lanes) if `detect_simd_level()` reports Simd256+ (future SVE).
//! - **x86_64**: runtime dispatch via `detect_simd_level()`:
//!   - `Simd512` → `F64x8` (AVX-512, 8 lanes) if oxiblas-core exposes it
//!   - `Simd256` → `F64x4` (AVX2, 4 lanes)
//!   - `Simd128` → `F64x2Sse` (SSE, 2 lanes)
//!   - `Scalar` → scalar fallback
//! - **Other architectures**: scalar fallback.
//!
//! # Transcendental operations
//!
//! `oxiblas_core::simd::SimdRegister` does not expose `exp`/`ln`/`sin`/`cos`.
//! `exp`, `ln`, `sin`, `cos`, and `tanh` use polynomial kernels from `crate::simd_vec_math`
//! (degree-12 Horner, Cody-Waite range reduction) with relative error < 1e-11.
//! Other transcendentals fall back to per-lane scalar extraction.

use crate::lower::{LoweredOp, OxiOp};
use oxiblas_core::simd::{SimdLevel, SimdRegister, detect_simd_level};

#[cfg(target_arch = "aarch64")]
use oxiblas_core::simd::aarch64::{F64x2, F64x4 as F64x4Aarch};

#[cfg(target_arch = "x86_64")]
use oxiblas_core::simd::x86_64::{F64x2Sse, F64x4, F64x8};

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
            SimdLevel::Simd512 => eval_chunks_dispatch::<F64x8>(ops, data),
            SimdLevel::Simd256 => eval_chunks_dispatch::<F64x4>(ops, data),
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
    // Derive n_slots from the ops sequence — avoids cascading API churn.
    let n_slots = ops
        .iter()
        .filter_map(|op| match op {
            OxiOp::Store(k) | OxiOp::Load(k) => Some(*k + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let mut slots: Vec<V> = vec![V::splat(f64::NAN); n_slots];

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
                stack.push(crate::simd_vec_math::simd_exp(a));
            }
            OxiOp::Ln => {
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(crate::simd_vec_math::simd_ln(a));
            }
            OxiOp::Sin => {
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(crate::simd_vec_math::simd_sin(a));
            }
            OxiOp::Cos => {
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(crate::simd_vec_math::simd_cos(a));
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
            OxiOp::Tan => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).tan());
                }
                stack.push(reg);
            }
            OxiOp::Sinh => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).sinh());
                }
                stack.push(reg);
            }
            OxiOp::Cosh => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).cosh());
                }
                stack.push(reg);
            }
            OxiOp::Tanh => {
                let a = stack.pop().unwrap_or_else(V::zero);
                stack.push(crate::simd_vec_math::simd_tanh(a));
            }
            OxiOp::Arcsin => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).asin());
                }
                stack.push(reg);
            }
            OxiOp::Arccos => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).acos());
                }
                stack.push(reg);
            }
            OxiOp::Arctan => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).atan());
                }
                stack.push(reg);
            }
            OxiOp::Arcsinh => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).asinh());
                }
                stack.push(reg);
            }
            OxiOp::Arccosh => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).acosh());
                }
                stack.push(reg);
            }
            OxiOp::Arctanh => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, a.extract(lane).atanh());
                }
                stack.push(reg);
            }
            OxiOp::Erf => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::erf(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::LGamma => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::lgamma(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::Digamma => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::digamma(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::Trigamma => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::trigamma(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::Ei => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::ei(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::Si => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::si(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::Ci => {
                let a = stack.pop().unwrap_or_else(V::zero);
                let mut reg = V::zero();
                for lane in 0..lanes {
                    reg = reg.insert(lane, crate::special::ci(a.extract(lane)));
                }
                stack.push(reg);
            }
            OxiOp::Store(k) => {
                // Peek top of SIMD stack into slot k (does NOT pop).
                let top = stack.last().copied().unwrap_or_else(|| V::splat(f64::NAN));
                if let Some(slot) = slots.get_mut(*k) {
                    *slot = top;
                }
            }
            OxiOp::Load(k) => {
                let v = slots.get(*k).copied().unwrap_or_else(|| V::splat(f64::NAN));
                stack.push(v);
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
                (s - r).abs() < 1e-11,
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
            std::sync::Arc::new(L::Sin(std::sync::Arc::new(L::Var(0)))),
            std::sync::Arc::new(L::Cos(std::sync::Arc::new(L::Var(0)))),
        );
        let ops = lowered.to_oxiblas_ops();
        let data: Vec<Vec<f64>> = (0..128).map(|i| vec![i as f64 * 0.05]).collect();

        let simd_results = eval_batch_simd(&ops, &data);
        let scalar_results = LoweredOp::eval_batch_scalar_from_ops(&ops, &data);

        assert_eq!(simd_results.len(), 128);
        for (i, (s, r)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            // Polynomial kernels have ~1e-11 relative error vs f64::sin/cos.
            // Near-cancellation zones (where sin(x)+cos(x)≈0) amplify relative error;
            // use absolute error tolerance of 2e-10 (≈ ULP budget for the sum).
            assert!(
                (s - r).abs() < 2e-10,
                "sin+cos row {i}: SIMD={s} scalar={r} abs_err={}",
                (s - r).abs()
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
            assert!((s - r).abs() < 1e-11, "remainder row {i}: {s} vs {r}");
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
            assert!((s - r).abs() < 1e-11, "parallel row {i}: {s} vs {r}");
        }
    }
}
