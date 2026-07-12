//! Integration tests for the JIT **batch** evaluation path (`R3`).
//!
//! The contract under test is a bit-exactness contract, not an approximate one:
//!
//! * `JitFn::call_batch` must reproduce, for every row, the exact `f64` bit
//!   pattern that `JitFn::call` produces for that row alone (0 ULP).
//! * `JitFn::call_batch_parallel` must reproduce `call_batch` bit-for-bit,
//!   independently of how many rayon worker threads happen to be running.
//! * Malformed `stride` / length combinations must come back as an honest
//!   `Err(JitBatchError)` — never as an out-of-bounds access.
//!
//! Every comparison therefore goes through [`f64::to_bits`], which distinguishes
//! `+0.0` from `-0.0` and refuses to call two different NaNs equal.
//!
//! All tests are gated behind `#[cfg(feature = "jit")]`.

#![cfg(feature = "jit")]

use oxieml::JitFn;
use oxieml::jit::JitBatchError;
use oxieml::lower::OxiOp;

// ─── deterministic data generation ───────────────────────────────────────────

/// SplitMix64 — a tiny, dependency-free, fully deterministic PRNG so that every
/// run of these tests sees exactly the same bytes.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[-4.0, 4.0)`, which keeps `ln`/`sqrt`-style ops in a range
    /// where both finite results and NaNs occur.
    fn next_f64(&mut self) -> f64 {
        let mantissa = (self.next_u64() >> 11) as f64;
        let unit = mantissa / ((1_u64 << 53) as f64);
        unit * 8.0 - 4.0
    }
}

/// Build a row-major `n_rows × stride` matrix of pseudo-random values.
fn make_rows(n_rows: usize, stride: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    let len = n_rows.saturating_sub(1) * stride + stride;
    (0..len).map(|_| rng.next_f64()).collect()
}

// ─── expression fixtures ─────────────────────────────────────────────────────

/// `(ops, n_vars, expected_is_batch_vectorized, label)`.
type Fixture = (Vec<OxiOp>, usize, bool, &'static str);

/// A spread of expressions covering both batch code paths:
/// the `f64x2` vector loop (pure arithmetic, `Pow`, `Store`/`Load`) and the
/// scalar loop (anything containing a transcendental host call).
fn fixtures() -> Vec<Fixture> {
    vec![
        // ── vectorizable: pure arithmetic ────────────────────────────────────
        // x0 * x0 + 3.5 * x1 - x2 / (x1 + 1.25)
        (
            vec![
                OxiOp::Var(0),
                OxiOp::Var(0),
                OxiOp::Mul,
                OxiOp::Const(3.5),
                OxiOp::Var(1),
                OxiOp::Mul,
                OxiOp::Add,
                OxiOp::Var(2),
                OxiOp::Var(1),
                OxiOp::Const(1.25),
                OxiOp::Add,
                OxiOp::Div,
                OxiOp::Sub,
            ],
            3,
            true,
            "x0^2 + 3.5*x1 - x2/(x1+1.25)",
        ),
        // -(x0 - x1) * (x0 + x1)
        (
            vec![
                OxiOp::Var(0),
                OxiOp::Var(1),
                OxiOp::Sub,
                OxiOp::Neg,
                OxiOp::Var(0),
                OxiOp::Var(1),
                OxiOp::Add,
                OxiOp::Mul,
            ],
            2,
            true,
            "-(x0-x1)*(x0+x1)",
        ),
        // ── vectorizable: Pow stays a per-lane scalar host call ──────────────
        // x0^3 + x1^x0
        (
            vec![
                OxiOp::Var(0),
                OxiOp::Const(3.0),
                OxiOp::Pow,
                OxiOp::Var(1),
                OxiOp::Var(0),
                OxiOp::Pow,
                OxiOp::Add,
            ],
            2,
            true,
            "x0^3 + x1^x0",
        ),
        // ── vectorizable: Store/Load (what lower_cse emits) ──────────────────
        // t = x0 + x1;  t * t - t
        (
            vec![
                OxiOp::Var(0),
                OxiOp::Var(1),
                OxiOp::Add,
                OxiOp::Store(0),
                OxiOp::Load(0),
                OxiOp::Mul,
                OxiOp::Load(0),
                OxiOp::Sub,
            ],
            2,
            true,
            "t=x0+x1; t*t - t",
        ),
        // ── vectorizable: constant only, zero variables ──────────────────────
        (vec![OxiOp::Const(-2.5)], 0, true, "const -2.5"),
        // ── vectorizable: identity ───────────────────────────────────────────
        (vec![OxiOp::Var(0)], 1, true, "x0"),
        // ── NOT vectorizable: transcendental host calls ──────────────────────
        // sin(x0) * exp(x1) + ln(x2)      (ln of a negative row yields NaN)
        (
            vec![
                OxiOp::Var(0),
                OxiOp::Sin,
                OxiOp::Var(1),
                OxiOp::Exp,
                OxiOp::Mul,
                OxiOp::Var(2),
                OxiOp::Ln,
                OxiOp::Add,
            ],
            3,
            false,
            "sin(x0)*exp(x1) + ln(x2)",
        ),
        // tanh(x0 * x1) - erf(x0)
        (
            vec![
                OxiOp::Var(0),
                OxiOp::Var(1),
                OxiOp::Mul,
                OxiOp::Tanh,
                OxiOp::Var(0),
                OxiOp::Erf,
                OxiOp::Sub,
            ],
            2,
            false,
            "tanh(x0*x1) - erf(x0)",
        ),
        // lgamma(x0) + arctan(x1) * cos(x0)
        (
            vec![
                OxiOp::Var(0),
                OxiOp::LGamma,
                OxiOp::Var(1),
                OxiOp::Arctan,
                OxiOp::Var(0),
                OxiOp::Cos,
                OxiOp::Mul,
                OxiOp::Add,
            ],
            2,
            false,
            "lgamma(x0) + arctan(x1)*cos(x0)",
        ),
    ]
}

// ─── bit-exact comparison helpers ────────────────────────────────────────────

/// Evaluate every row with the *scalar* entry point — the reference the batch
/// path must reproduce bit-for-bit.
fn scalar_reference(f: &JitFn, rows: &[f64], n_rows: usize, stride: usize) -> Vec<f64> {
    let n_vars = f.n_vars();
    (0..n_rows)
        .map(|i| {
            let start = i * stride;
            f.call(&rows[start..start + n_vars])
        })
        .collect()
}

/// Assert two `f64` slices are identical **bit for bit** (0 ULP).
///
/// Comparing raw bits rather than values is deliberate: it makes `-0.0` differ
/// from `+0.0` and makes two NaNs with different payloads a failure, which is
/// precisely the strictness the batch path promises.
fn assert_bits_eq(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: row {i} differs — batch={g} (0x{:016x}) scalar={w} (0x{:016x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

// ─── Tier 1: call_batch == per-row call, 0 ULP ───────────────────────────────

/// The headline requirement: 1000 rows, every expression, 0 ULP.
#[test]
fn call_batch_matches_per_row_call_bit_exactly() {
    const N_ROWS: usize = 1000;

    for (ops, n_vars, _, label) in fixtures() {
        let f = JitFn::compile(&ops, n_vars).expect("compile");
        // A stride strictly larger than n_vars: rows carry a trailing column the
        // expression must not read (e.g. the regression target).
        let stride = f.n_vars() + 2;
        let rows = make_rows(N_ROWS, stride, 0x0E01_0001);

        let want = scalar_reference(&f, &rows, N_ROWS, stride);

        let mut got = vec![f64::NAN; N_ROWS];
        f.call_batch(&rows, N_ROWS, stride, &mut got)
            .expect("call_batch");

        assert_bits_eq(&got, &want, label);
    }
}

/// Densely packed rows (`stride == n_vars`) must work exactly the same.
#[test]
fn call_batch_dense_stride_matches_per_row_call() {
    const N_ROWS: usize = 1000;

    for (ops, n_vars, _, label) in fixtures() {
        let f = JitFn::compile(&ops, n_vars).expect("compile");
        let stride = f.n_vars();
        if stride == 0 {
            // A zero-variable expression has no rows to pack; covered by the
            // dedicated zero-variable test below.
            continue;
        }
        let rows = make_rows(N_ROWS, stride, 0x0E01_0002);

        let want = scalar_reference(&f, &rows, N_ROWS, stride);

        let mut got = vec![f64::NAN; N_ROWS];
        f.call_batch(&rows, N_ROWS, stride, &mut got)
            .expect("call_batch");

        assert_bits_eq(&got, &want, label);
    }
}

// ─── n_rows ∈ {0, 1, 2, large} — the block-sealing / loop-boundary cases ─────

/// Row counts that exercise every corner of both generated loops: the empty
/// batch (loop never entered), the single row (vector loop skipped entirely,
/// scalar epilogue does all the work), the exact row pair (one vector
/// iteration, empty epilogue), odd/even counts around it, and a large batch.
#[test]
fn call_batch_row_count_edge_cases() {
    const ROW_COUNTS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 63, 64, 65, 1023, 1024, 4096];

    for (ops, n_vars, _, label) in fixtures() {
        let f = JitFn::compile(&ops, n_vars).expect("compile");
        let stride = f.n_vars() + 1;

        for &n_rows in ROW_COUNTS {
            let rows = make_rows(n_rows, stride, 0x0E01_0003 ^ (n_rows as u64));

            let want = scalar_reference(&f, &rows, n_rows, stride);

            let mut got = vec![f64::NAN; n_rows];
            f.call_batch(&rows, n_rows, stride, &mut got)
                .expect("call_batch");

            assert_bits_eq(&got, &want, &format!("{label} @ n_rows={n_rows}"));
        }
    }
}

/// An empty batch must not write a single element of `out`, and must not care
/// what `rows` contains (the generated loop exits on its first bounds test).
#[test]
fn call_batch_zero_rows_leaves_output_untouched() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Mul];
    let f = JitFn::compile(&ops, 2).expect("compile");

    const SENTINEL: f64 = -12345.678;
    let mut out = vec![SENTINEL; 8];

    // Empty input slice, empty batch: nothing is read, nothing is written.
    f.call_batch(&[], 0, 2, &mut out).expect("call_batch");
    for (i, v) in out.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            SENTINEL.to_bits(),
            "out[{i}] was written by a zero-row batch"
        );
    }

    // Same, with a non-empty input and an oversized `out`.
    let rows = make_rows(4, 2, 0x0E01_0004);
    f.call_batch(&rows, 0, 2, &mut out).expect("call_batch");
    for (i, v) in out.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            SENTINEL.to_bits(),
            "out[{i}] was written by a zero-row batch"
        );
    }
}

/// `out` may be longer than `n_rows`; the tail must stay untouched.
#[test]
fn call_batch_writes_exactly_n_rows() {
    let ops = vec![OxiOp::Var(0), OxiOp::Const(2.0), OxiOp::Mul];
    let f = JitFn::compile(&ops, 1).expect("compile");

    const SENTINEL: f64 = 9.5;
    const N_ROWS: usize = 5;
    let rows: Vec<f64> = (0..N_ROWS).map(|i| i as f64).collect();

    let mut out = vec![SENTINEL; N_ROWS + 3];
    f.call_batch(&rows, N_ROWS, 1, &mut out)
        .expect("call_batch");

    for (i, v) in out.iter().enumerate().take(N_ROWS) {
        assert_eq!(v.to_bits(), (2.0 * i as f64).to_bits(), "row {i}");
    }
    for (i, v) in out.iter().enumerate().skip(N_ROWS) {
        assert_eq!(
            v.to_bits(),
            SENTINEL.to_bits(),
            "out[{i}] past n_rows was overwritten"
        );
    }
}

/// A zero-variable expression: `n_vars == 0` means the row pointer is never
/// dereferenced, so an empty input and a zero stride are both legitimate.
#[test]
fn call_batch_zero_variable_expression() {
    let ops = vec![OxiOp::Const(-2.5), OxiOp::Const(4.0), OxiOp::Div];
    let f = JitFn::compile(&ops, 0).expect("compile");
    assert_eq!(f.n_vars(), 0);

    const N_ROWS: usize = 1000;
    let mut out = vec![f64::NAN; N_ROWS];
    f.call_batch(&[], N_ROWS, 0, &mut out)
        .expect("call_batch with no variables");

    let want = f.call(&[]);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(v.to_bits(), want.to_bits(), "row {i}");
    }
}

/// A single row makes `stride` irrelevant (it is only ever multiplied by 0), so
/// even a zero stride must be accepted and produce the scalar result.
#[test]
fn call_batch_single_row_ignores_stride() {
    let ops = vec![
        OxiOp::Var(0),
        OxiOp::Var(1),
        OxiOp::Var(2),
        OxiOp::Add,
        OxiOp::Add,
    ];
    let f = JitFn::compile(&ops, 3).expect("compile");

    let rows = [1.5, 2.25, -0.75];
    let want = f.call(&rows);

    for stride in [0_usize, 1, 3, 17] {
        let mut out = [f64::NAN];
        f.call_batch(&rows, 1, stride, &mut out)
            .unwrap_or_else(|e| panic!("call_batch with stride {stride}: {e}"));
        assert_eq!(out[0].to_bits(), want.to_bits(), "stride {stride}");
    }
}

// ─── special values: NaN, ±Inf, ±0.0, denormals ─────────────────────────────

/// Non-finite and edge-case inputs must round-trip through the batch path with
/// the *same bits* as the scalar path — including NaN payloads and signed zero.
#[test]
fn call_batch_special_values_bit_exact() {
    let specials = [
        0.0_f64,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 2.0, // subnormal
        f64::MAX,
        f64::EPSILON,
    ];

    for (ops, n_vars, _, label) in fixtures() {
        let f = JitFn::compile(&ops, n_vars).expect("compile");
        let stride = f.n_vars().max(1);

        // Every ordered pair of specials, laid out one pair per row (padded).
        let mut rows: Vec<f64> = Vec::new();
        for &a in &specials {
            for &b in &specials {
                for slot in 0..stride {
                    rows.push(match slot {
                        0 => a,
                        1 => b,
                        _ => a * 0.5 + b,
                    });
                }
            }
        }
        let n_rows = specials.len() * specials.len();

        let want = scalar_reference(&f, &rows, n_rows, stride);

        let mut got = vec![0.0; n_rows];
        f.call_batch(&rows, n_rows, stride, &mut got)
            .expect("call_batch");

        assert_bits_eq(&got, &want, &format!("{label} (special values)"));
    }
}

// ─── the vectorization gate ──────────────────────────────────────────────────

/// `is_batch_vectorized` must agree with the documented op subset: arithmetic,
/// `Pow` and `Store`/`Load` vectorize; anything with a transcendental host call
/// does not.  (This is a performance property only — both paths are asserted
/// bit-exact above — but a silent regression to the scalar loop should be loud.)
#[test]
fn call_batch_vectorization_gate() {
    for (ops, n_vars, expect_vectorized, label) in fixtures() {
        let f = JitFn::compile(&ops, n_vars).expect("compile");
        assert_eq!(
            f.is_batch_vectorized(),
            expect_vectorized,
            "is_batch_vectorized() mismatch for {label}"
        );
    }
}

// ─── malformed arguments: honest errors, never UB ────────────────────────────

/// `rows` shorter than the last row it must supply.
#[test]
fn call_batch_rejects_short_rows() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Add];
    let f = JitFn::compile(&ops, 2).expect("compile");

    // 10 rows of stride 2 need (10 - 1) * 2 + 2 == 20 elements; give 19.
    let rows = vec![1.0; 19];
    let mut out = vec![0.0; 10];

    let err = f
        .call_batch(&rows, 10, 2, &mut out)
        .expect_err("short rows must be rejected");
    assert_eq!(err, JitBatchError::RowsTooShort { got: 19, need: 20 });

    // The output must not have been touched.
    assert!(out.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
}

/// `out` too short to hold one result per row.
#[test]
fn call_batch_rejects_short_output() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Add];
    let f = JitFn::compile(&ops, 2).expect("compile");

    let rows = vec![1.0; 20];
    let mut out = vec![0.0; 9];

    let err = f
        .call_batch(&rows, 10, 2, &mut out)
        .expect_err("short output must be rejected");
    assert_eq!(err, JitBatchError::OutTooShort { got: 9, need: 10 });
}

/// A stride narrower than the row itself: consecutive rows would overlap, so it
/// cannot be a row stride.
#[test]
fn call_batch_rejects_stride_smaller_than_n_vars() {
    let ops = vec![
        OxiOp::Var(0),
        OxiOp::Var(1),
        OxiOp::Var(2),
        OxiOp::Add,
        OxiOp::Add,
    ];
    let f = JitFn::compile(&ops, 3).expect("compile");

    let rows = vec![1.0; 4096];
    let mut out = vec![0.0; 10];

    let err = f
        .call_batch(&rows, 10, 2, &mut out)
        .expect_err("stride < n_vars must be rejected");
    assert_eq!(
        err,
        JitBatchError::StrideTooSmall {
            stride: 2,
            n_vars: 3
        }
    );
}

/// A stride so large that the row footprint cannot describe any allocation.
#[test]
fn call_batch_rejects_length_overflow() {
    let ops = vec![OxiOp::Var(0), OxiOp::Neg];
    let f = JitFn::compile(&ops, 1).expect("compile");

    let rows = vec![1.0; 8];
    let mut out = vec![0.0; 3];
    let stride = usize::MAX / 2 + 1; // (3 - 1) * stride overflows usize

    let err = f
        .call_batch(&rows, 3, stride, &mut out)
        .expect_err("overflowing footprint must be rejected");
    assert_eq!(
        err,
        JitBatchError::LengthOverflow {
            n_rows: 3,
            stride,
            n_vars: 1
        }
    );
}

/// An under-reported `n_vars` at compile time must not become an out-of-bounds
/// read at batch time: `compile` raises `n_vars` to the largest `Var(i)` it
/// sees, and the batch validator uses that raised value.
#[test]
fn call_batch_uses_effective_n_vars_for_validation() {
    // Claims 1 variable, actually reads Var(0) and Var(2).
    let ops = vec![OxiOp::Var(0), OxiOp::Var(2), OxiOp::Mul];
    let f = JitFn::compile(&ops, 1).expect("compile");
    assert_eq!(f.n_vars(), 3, "n_vars must be raised to 1 + max Var index");

    let rows = vec![1.0; 6];
    let mut out = vec![0.0; 3];

    // 3 rows of stride 2 would need (3 - 1) * 2 + 3 == 7 elements, but the
    // stride is also below the real n_vars — that is caught first.
    let err = f
        .call_batch(&rows, 3, 2, &mut out)
        .expect_err("under-reported n_vars must still be validated");
    assert_eq!(
        err,
        JitBatchError::StrideTooSmall {
            stride: 2,
            n_vars: 3
        }
    );
}

// ─── parallel path ───────────────────────────────────────────────────────────

/// `call_batch_parallel` must agree with the per-row scalar `call` bit for bit,
/// at every row count — including the ones that straddle the chunk boundary.
///
/// This is also the test that [`call_batch_parallel_is_thread_count_invariant`]
/// re-runs in a child process under different `RAYON_NUM_THREADS` settings, so
/// keep it self-contained.
#[cfg(feature = "parallel")]
#[test]
fn call_batch_parallel_matches_per_row_call_bit_exactly() {
    const ROW_COUNTS: &[usize] = &[0, 1, 2, 3, 63, 64, 65, 127, 128, 1000, 5000];

    for (ops, n_vars, _, label) in fixtures() {
        let f = JitFn::compile(&ops, n_vars).expect("compile");
        let stride = f.n_vars() + 2;

        for &n_rows in ROW_COUNTS {
            let rows = make_rows(n_rows, stride, 0x0E01_0005 ^ (n_rows as u64));

            let want = scalar_reference(&f, &rows, n_rows, stride);

            // Serial batch and parallel batch must both reproduce it exactly.
            let mut serial = vec![f64::NAN; n_rows];
            f.call_batch(&rows, n_rows, stride, &mut serial)
                .expect("call_batch");
            assert_bits_eq(&serial, &want, &format!("{label} serial @ {n_rows}"));

            let mut parallel = vec![f64::NAN; n_rows];
            f.call_batch_parallel(&rows, n_rows, stride, &mut parallel)
                .expect("call_batch_parallel");
            assert_bits_eq(&parallel, &want, &format!("{label} parallel @ {n_rows}"));
            assert_bits_eq(
                &parallel,
                &serial,
                &format!("{label} parallel vs serial @ {n_rows}"),
            );
        }
    }
}

/// Thread invariance: re-execute this very test binary with `RAYON_NUM_THREADS`
/// pinned to 1 and to 8, running only the parallel bit-exactness test above.
///
/// Rayon reads `RAYON_NUM_THREADS` when its global pool is first built, i.e.
/// once per process — so a child process is the only honest way to vary it.
/// Both children assert equality against the *same* scalar reference, so their
/// success proves the outputs are identical to each other as well.
#[cfg(feature = "parallel")]
#[test]
fn call_batch_parallel_is_thread_count_invariant() {
    use std::process::Command;

    let exe = std::env::current_exe().expect("current_exe");

    for threads in ["1", "8"] {
        let status = Command::new(&exe)
            .arg("call_batch_parallel_matches_per_row_call_bit_exactly")
            .arg("--exact")
            .arg("--test-threads=1")
            .env("RAYON_NUM_THREADS", threads)
            .status()
            .unwrap_or_else(|e| panic!("spawning the test binary with {threads} threads: {e}"));

        assert!(
            status.success(),
            "call_batch_parallel produced different results with RAYON_NUM_THREADS={threads}"
        );
    }
}

/// The parallel entry point validates its arguments exactly like the serial one
/// — before it hands any pointer to rayon.
#[cfg(feature = "parallel")]
#[test]
fn call_batch_parallel_rejects_malformed_arguments() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Add];
    let f = JitFn::compile(&ops, 2).expect("compile");

    let rows = vec![1.0; 19];
    let mut out = vec![0.0; 10];
    assert_eq!(
        f.call_batch_parallel(&rows, 10, 2, &mut out)
            .expect_err("short rows"),
        JitBatchError::RowsTooShort { got: 19, need: 20 }
    );

    let rows = vec![1.0; 20];
    let mut out = vec![0.0; 9];
    assert_eq!(
        f.call_batch_parallel(&rows, 10, 2, &mut out)
            .expect_err("short output"),
        JitBatchError::OutTooShort { got: 9, need: 10 }
    );

    let rows = vec![1.0; 64];
    let mut out = vec![0.0; 10];
    assert_eq!(
        f.call_batch_parallel(&rows, 10, 1, &mut out)
            .expect_err("stride < n_vars"),
        JitBatchError::StrideTooSmall {
            stride: 1,
            n_vars: 2
        }
    );
}

/// An empty parallel batch must be a no-op, not a rayon panic on a zero-sized
/// chunk.
#[cfg(feature = "parallel")]
#[test]
fn call_batch_parallel_zero_rows_is_a_no_op() {
    let ops = vec![OxiOp::Var(0), OxiOp::Exp];
    let f = JitFn::compile(&ops, 1).expect("compile");

    const SENTINEL: f64 = 42.0;
    let mut out = vec![SENTINEL; 4];
    f.call_batch_parallel(&[], 0, 1, &mut out)
        .expect("empty parallel batch");
    assert!(out.iter().all(|v| v.to_bits() == SENTINEL.to_bits()));
}
