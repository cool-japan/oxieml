//! JIT compilation of `OxiOp` sequences via Cranelift.
//!
//! This module provides [`JitFn`] and [`JitCache`] for compiling post-order
//! `OxiOp` stacks to native machine code through Cranelift's JIT backend.
//!
//! Every [`JitFn`] carries **two** entry points, produced by a single
//! compilation of a single Cranelift [`JITModule`]:
//!
//! * a *scalar* entry point `fn(*const f64, usize) -> f64` driven by
//!   [`JitFn::call`], and
//! * a *batch* entry point `fn(*const f64, usize, usize, *mut f64)` driven by
//!   [`JitFn::call_batch`] (and, with the `parallel` feature,
//!   [`JitFn::call_batch_parallel`]).  The batch entry point contains the row
//!   loop **inside** the generated machine code, so a batch of `n_rows` rows
//!   costs one native call instead of `n_rows` calls.
//!
//! Both entry points emit the *same* IEEE-754 operations for a given
//! `OxiOp` sequence — `fadd`/`fsub`/`fmul`/`fdiv`/`fneg` and scalar host calls
//! for transcendentals — so `call_batch` is **bit-exact** (0 ULP) with
//! per-row `call`.  See [`JitFn::call_batch`] for the full argument contract.
//!
//! The module is gated behind the `jit` feature and has **zero impact** on
//! default builds.

#[cfg(feature = "jit")]
mod inner {
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::types::{F64, F64X2};
    use cranelift_codegen::ir::{
        AbiParam, BlockArg, Function, InstBuilder, MemFlagsData, Signature, Type, UserFuncName,
        Value,
    };
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{FuncId, Linkage, Module};

    use crate::lower::OxiOp;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// Size of one `f64` in bytes, as an `Imm64` operand for `imul_imm`.
    const F64_BYTES: i64 = 8;

    /// Size of one `f64` in bytes, as an `Offset32` operand for `load`/`store`.
    const F64_OFFSET: i32 = 8;

    /// Boxed, thread-safe error type used throughout this module.
    type JitResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

    /// Raw ABI type of the compiled scalar entry point.
    ///
    /// `fn(vars_ptr: *const f64, vars_len: usize) -> f64`
    type ScalarFnPtr = unsafe extern "C" fn(*const f64, usize) -> f64;

    /// Raw ABI type of the compiled batch entry point.
    ///
    /// `fn(in_ptr: *const f64, n_rows: usize, stride: usize, out_ptr: *mut f64)`
    ///
    /// `stride` is measured in **elements** (`f64`s), not bytes; the generated
    /// code converts it to bytes internally.  Row `i` occupies
    /// `in_ptr[i * stride .. i * stride + n_vars]` and its result is written to
    /// `out_ptr[i]`.
    type BatchFnPtr = unsafe extern "C" fn(*const f64, usize, usize, *mut f64);

    // ─── helpers ────────────────────────────────────────────────────────────

    /// Build an `f64 → f64` ABI signature for external math functions.
    fn sig_f64_to_f64(call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(F64));
        sig.returns.push(AbiParam::new(F64));
        sig
    }

    /// Build an `(f64, f64) → f64` ABI signature for `pow`.
    fn sig_f64_f64_to_f64(call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(F64));
        sig.params.push(AbiParam::new(F64));
        sig.returns.push(AbiParam::new(F64));
        sig
    }

    /// FNV-1a hash for an `OxiOp` slice — no external dependency.
    pub fn ops_hash(ops: &[OxiOp]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = FNV_OFFSET;
        for op in ops {
            // Serialize each op to a small byte sequence and feed into FNV-1a.
            let bytes: [u8; 9] = encode_op(op);
            for byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }

    /// Encode an `OxiOp` to a fixed-size byte array for hashing.
    ///
    /// Layout: `[discriminant (1 byte)] [payload (8 bytes)]`.
    fn encode_op(op: &OxiOp) -> [u8; 9] {
        let mut out = [0u8; 9];
        match op {
            OxiOp::Const(c) => {
                out[0] = 0;
                out[1..9].copy_from_slice(&c.to_bits().to_le_bytes());
            }
            OxiOp::Var(i) => {
                out[0] = 1;
                out[1..9].copy_from_slice(&(*i as u64).to_le_bytes());
            }
            OxiOp::Add => out[0] = 2,
            OxiOp::Sub => out[0] = 3,
            OxiOp::Mul => out[0] = 4,
            OxiOp::Div => out[0] = 5,
            OxiOp::Neg => out[0] = 6,
            OxiOp::Exp => out[0] = 7,
            OxiOp::Ln => out[0] = 8,
            OxiOp::Sin => out[0] = 9,
            OxiOp::Cos => out[0] = 10,
            OxiOp::Pow => out[0] = 11,
            OxiOp::Tan => out[0] = 12,
            OxiOp::Sinh => out[0] = 13,
            OxiOp::Cosh => out[0] = 14,
            OxiOp::Tanh => out[0] = 15,
            OxiOp::Arcsin => out[0] = 16,
            OxiOp::Arccos => out[0] = 17,
            OxiOp::Arctan => out[0] = 18,
            OxiOp::Arcsinh => out[0] = 19,
            OxiOp::Arccosh => out[0] = 20,
            OxiOp::Arctanh => out[0] = 21,
            OxiOp::Erf => out[0] = 24,
            OxiOp::LGamma => out[0] = 25,
            OxiOp::Digamma => out[0] = 26,
            OxiOp::Ei => out[0] = 27,
            OxiOp::Si => out[0] = 28,
            OxiOp::Ci => out[0] = 29,
            OxiOp::Trigamma => out[0] = 30,
            OxiOp::Store(k) => {
                out[0] = 22;
                out[1..9].copy_from_slice(&(*k as u64).to_le_bytes());
            }
            OxiOp::Load(k) => {
                out[0] = 23;
                out[1..9].copy_from_slice(&(*k as u64).to_le_bytes());
            }
        }
        out
    }

    // ─── batch argument validation ──────────────────────────────────────────

    /// Why a [`JitFn::call_batch`] / [`JitFn::call_batch_parallel`] request was
    /// rejected.
    ///
    /// Every variant describes a *malformed* argument combination that would
    /// otherwise make the generated code read or write out of bounds.  The
    /// arguments are fully validated **before** any `unsafe` code runs, so a
    /// malformed call is an honest `Err`, never undefined behaviour.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum JitBatchError {
        /// `stride < n_vars` while `n_rows > 1`: consecutive rows would overlap,
        /// so `stride` cannot be a row stride of an `n_rows × n_vars` matrix.
        StrideTooSmall {
            /// The stride (in `f64` elements) that was supplied.
            stride: usize,
            /// The number of variable slots the compiled function reads.
            n_vars: usize,
        },
        /// The input buffer is shorter than the last row it must supply.
        RowsTooShort {
            /// `rows.len()` as supplied.
            got: usize,
            /// `(n_rows - 1) * stride + n_vars`, the number of elements needed.
            need: usize,
        },
        /// The output buffer cannot hold one `f64` per row.
        OutTooShort {
            /// `out.len()` as supplied.
            got: usize,
            /// `n_rows`, the number of elements needed.
            need: usize,
        },
        /// `(n_rows - 1) * stride + n_vars` overflows `usize`, so the requested
        /// footprint cannot describe any real allocation.
        LengthOverflow {
            /// Number of rows requested.
            n_rows: usize,
            /// Row stride (in `f64` elements) requested.
            stride: usize,
            /// Number of variable slots the compiled function reads.
            n_vars: usize,
        },
    }

    impl std::fmt::Display for JitBatchError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::StrideTooSmall { stride, n_vars } => write!(
                    f,
                    "JitFn batch: stride {stride} is smaller than n_vars {n_vars}; \
                     rows would overlap"
                ),
                Self::RowsTooShort { got, need } => write!(
                    f,
                    "JitFn batch: input slice has {got} elements but {need} are required"
                ),
                Self::OutTooShort { got, need } => write!(
                    f,
                    "JitFn batch: output slice has {got} elements but {need} are required"
                ),
                Self::LengthOverflow {
                    n_rows,
                    stride,
                    n_vars,
                } => write!(
                    f,
                    "JitFn batch: (n_rows - 1) * stride + n_vars overflows usize \
                     (n_rows={n_rows}, stride={stride}, n_vars={n_vars})"
                ),
            }
        }
    }

    impl std::error::Error for JitBatchError {}

    // ─── JitFn ──────────────────────────────────────────────────────────────

    /// A JIT-compiled function over a flat `&[f64]` variable slice.
    ///
    /// Created by [`JitFn::compile`]; called by [`JitFn::call`] (one row) or
    /// [`JitFn::call_batch`] (many rows, one native call).
    /// Holds the [`JITModule`] alive so the generated code is not reclaimed.
    pub struct JitFn {
        /// Scalar entry point: `fn(vars_ptr, vars_len) -> f64`.
        fn_ptr: ScalarFnPtr,
        /// Batch entry point: `fn(in_ptr, n_rows, stride, out_ptr)`.
        batch_fn_ptr: BatchFnPtr,
        /// Keep the JIT module alive so the code mapping is not freed.
        _module: JITModule,
        /// Minimum number of variables required.
        n_vars: usize,
        /// Whether the batch entry point uses the two-lane (`f64x2`) loop.
        batch_vectorized: bool,
    }

    // SAFETY: After `finalize_definitions`, the `JITModule` no longer mutates
    // any shared state — the compiled machine code is mapped read-execute and
    // all mutable metadata is consumed.  The function pointer is just a
    // read-only reference into that immutable mapping, so sharing across
    // threads is safe.
    unsafe impl Send for JitFn {}
    // SAFETY: `JitFn::call` only reads from the vars slice through the function
    // pointer; no interior mutability is exposed, so concurrent reads from
    // multiple threads are safe.
    unsafe impl Sync for JitFn {}

    impl JitFn {
        /// Compile an `OxiOp` post-order sequence to native machine code.
        ///
        /// Two entry points are emitted into a single [`JITModule`] and
        /// finalized together (one compile):
        ///
        /// * `__oxi_jit_eval` — the scalar entry point used by [`JitFn::call`].
        /// * `__oxi_jit_eval_batch` — the batch entry point used by
        ///   [`JitFn::call_batch`], which contains the row loop in generated
        ///   code.  If [`JitFn::is_batch_vectorized`] reports `true` the batch
        ///   loop processes two rows per iteration in `f64x2` lanes.
        ///
        /// `n_vars` is the number of `Var(i)` slots; the pointer arithmetic in
        /// the emitted code accesses indices `0..n_vars` without bounds checks,
        /// so callers **must** pass at least `n_vars` elements.  The value is
        /// raised to `max(n_vars, 1 + max Var(i))` so that an under-reported
        /// `n_vars` can never turn into an out-of-bounds read.
        pub fn compile(ops: &[OxiOp], n_vars: usize) -> JitResult<Self> {
            // Compute the minimum n_vars required by scanning for Var(i) ops.
            // The effective count is the max of the caller-supplied hint and the
            // largest index seen in the op sequence plus one.  This prevents
            // out-of-bounds memory access when the caller under-reports n_vars.
            let actual_n_vars = ops
                .iter()
                .filter_map(|op| {
                    if let OxiOp::Var(i) = op {
                        Some(i + 1)
                    } else {
                        None
                    }
                })
                .max()
                .unwrap_or(0);
            let effective_n_vars = actual_n_vars.max(n_vars);

            // ── JIT module ─────────────────────────────────────────────────
            // `JITBuilder::new` internally calls `cranelift_native::builder()`
            // and sets the `use_colocated_libcalls = false` / `is_pic = false`
            // flags that are required for JIT operation.
            let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
                .map_err(|e| format!("JITBuilder::new: {e}"))?;

            // Register math symbols explicitly so the dynamic linker resolves
            // them even on platforms where libm symbols are not in the default
            // dynamic-link search path (e.g., musl-libc static builds).
            jit_builder.symbol("exp", f64::exp as *const u8);
            jit_builder.symbol("log", (f64::ln as fn(f64) -> f64) as *const u8);
            jit_builder.symbol("sin", f64::sin as *const u8);
            jit_builder.symbol("cos", f64::cos as *const u8);
            jit_builder.symbol("pow", f64::powf as *const u8);
            jit_builder.symbol("tan", f64::tan as *const u8);
            jit_builder.symbol("sinh", f64::sinh as *const u8);
            jit_builder.symbol("cosh", f64::cosh as *const u8);
            jit_builder.symbol("tanh", f64::tanh as *const u8);
            jit_builder.symbol("asin", f64::asin as *const u8);
            jit_builder.symbol("acos", f64::acos as *const u8);
            jit_builder.symbol("atan", f64::atan as *const u8);
            jit_builder.symbol("asinh", f64::asinh as *const u8);
            jit_builder.symbol("acosh", f64::acosh as *const u8);
            jit_builder.symbol("atanh", f64::atanh as *const u8);
            jit_builder.symbol("oxieml_erf", crate::special::erf as *const u8);
            jit_builder.symbol("oxieml_lgamma", crate::special::lgamma as *const u8);
            jit_builder.symbol("oxieml_digamma", crate::special::digamma as *const u8);
            jit_builder.symbol("oxieml_ei", crate::special::ei as *const u8);
            jit_builder.symbol("oxieml_si", crate::special::si as *const u8);
            jit_builder.symbol("oxieml_ci", crate::special::ci as *const u8);
            jit_builder.symbol("oxieml_trigamma", crate::special::trigamma as *const u8);

            let mut module = JITModule::new(jit_builder);

            // Query the default call convention from the module's ISA.
            let call_conv = module.target_config().default_call_conv;

            // ── Declare the scalar entry point ──────────────────────────────
            // Signature: fn(ptr: *const f64, len: usize) -> f64
            let ptr_type = module.target_config().pointer_type();
            let mut main_sig = Signature::new(call_conv);
            // ptr (*const f64) — represented as a pointer-sized integer in IR
            main_sig.params.push(AbiParam::new(ptr_type));
            // len (usize) — pointer-sized integer
            main_sig.params.push(AbiParam::new(ptr_type));
            main_sig.returns.push(AbiParam::new(F64));

            let main_func_id = module
                .declare_function("__oxi_jit_eval", Linkage::Local, &main_sig)
                .map_err(|e| format!("declare_function: {e}"))?;

            // ── Declare the batch entry point ───────────────────────────────
            // Signature: fn(in_ptr: *const f64, n_rows: usize, stride: usize,
            //               out_ptr: *mut f64)
            // All four parameters are pointer-sized integers in IR; there is no
            // return value — results are written straight into `out_ptr`.
            let mut batch_sig = Signature::new(call_conv);
            for _ in 0..4 {
                batch_sig.params.push(AbiParam::new(ptr_type));
            }

            let batch_func_id = module
                .declare_function("__oxi_jit_eval_batch", Linkage::Local, &batch_sig)
                .map_err(|e| format!("declare_function (batch): {e}"))?;

            // ── Declare external math functions ─────────────────────────────
            let extern_ids = declare_extern_fns(&mut module, call_conv)?;

            // ── Build the scalar entry point ────────────────────────────────
            let mut ctx = module.make_context();
            ctx.func = Function::with_name_signature(
                UserFuncName::user(0, main_func_id.as_u32()),
                main_sig,
            );

            {
                let mut fb_ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                // `entry_block` has no predecessors, so it can be sealed at once.
                builder.seal_block(entry_block);

                let vars_ptr_val = builder.block_params(entry_block)[0];

                let result =
                    emit_row_scalar(ops, &mut builder, vars_ptr_val, &extern_ids, &mut module)?;

                builder.ins().return_(&[result]);
                builder.finalize();
            }

            module
                .define_function(main_func_id, &mut ctx)
                .map_err(|e| format!("define_function: {e}"))?;
            module.clear_context(&mut ctx);

            // ── Build the batch entry point ─────────────────────────────────
            // The vectorized loop is only legal for the arithmetic subset; see
            // `is_vectorizable` for why transcendentals are excluded.
            let batch_vectorized = is_vectorizable(ops);

            ctx.func = Function::with_name_signature(
                UserFuncName::user(0, batch_func_id.as_u32()),
                batch_sig,
            );

            {
                let mut fb_ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

                if batch_vectorized {
                    build_batch_vector_loop(ops, &mut builder, ptr_type, &extern_ids, &mut module)?;
                } else {
                    build_batch_scalar_loop(ops, &mut builder, ptr_type, &extern_ids, &mut module)?;
                }

                builder.finalize();
            }

            module
                .define_function(batch_func_id, &mut ctx)
                .map_err(|e| format!("define_function (batch): {e}"))?;
            module.clear_context(&mut ctx);

            // ── Compile ─────────────────────────────────────────────────────
            module
                .finalize_definitions()
                .map_err(|e| format!("finalize_definitions: {e}"))?;

            let raw_ptr = module.get_finalized_function(main_func_id);
            let raw_batch_ptr = module.get_finalized_function(batch_func_id);

            // SAFETY: `get_finalized_function` returns a pointer into the
            // module's read-execute code mapping, which stays valid for as long
            // as `module` (moved into `self` below) is alive.  The transmuted
            // signature is exactly the `Signature` the function was declared and
            // defined with, and the module uses the platform's default C calling
            // convention (`module.target_config().default_call_conv`), so the
            // `extern "C"` fn types match the generated prologue/epilogue.
            let fn_ptr: ScalarFnPtr = unsafe { std::mem::transmute(raw_ptr) };
            // SAFETY: as above, for the four-argument batch entry point.
            let batch_fn_ptr: BatchFnPtr = unsafe { std::mem::transmute(raw_batch_ptr) };

            Ok(Self {
                fn_ptr,
                batch_fn_ptr,
                _module: module,
                n_vars: effective_n_vars,
                batch_vectorized,
            })
        }

        /// Call the JIT-compiled function with the given variable slice.
        ///
        /// # Panics
        ///
        /// Panics if `vars.len() < self.n_vars`.
        pub fn call(&self, vars: &[f64]) -> f64 {
            assert!(
                vars.len() >= self.n_vars,
                "JitFn::call: need {} vars, got {}",
                self.n_vars,
                vars.len()
            );
            // SAFETY: Compiled code only reads `vars[0..n_vars]` via the raw pointer.
            unsafe { (self.fn_ptr)(vars.as_ptr(), vars.len()) }
        }

        /// The minimum number of variable slots this function requires.
        pub fn n_vars(&self) -> usize {
            self.n_vars
        }

        /// Whether the batch entry point was compiled as a two-lane (`f64x2`)
        /// loop.
        ///
        /// This is a pure performance property: the vectorized and scalar batch
        /// loops are **bit-exact** with each other and with [`JitFn::call`].
        /// It is exposed mainly so that tests and benchmarks can assert which
        /// code path a given `OxiOp` sequence took.
        pub fn is_batch_vectorized(&self) -> bool {
            self.batch_vectorized
        }

        /// Validate the batch arguments against the compiled function's
        /// `n_vars`.
        ///
        /// On success the following hold, which is exactly what the generated
        /// code needs in order to stay inside both allocations:
        ///
        /// * every read `in_ptr[i * stride + j]` with `i < n_rows`,
        ///   `j < n_vars` is `< rows_len`;
        /// * every write `out_ptr[i]` with `i < n_rows` is `< out_len`;
        /// * `(n_rows - 1) * stride + n_vars` does not overflow `usize`, hence
        ///   none of the byte offsets computed in IR (`* 8`) can overflow
        ///   `isize` either, because they are bounded by the byte size of a
        ///   live Rust allocation.
        fn validate_batch(
            &self,
            rows_len: usize,
            n_rows: usize,
            stride: usize,
            out_len: usize,
        ) -> Result<(), JitBatchError> {
            // An empty batch touches no memory: the generated loop exits on its
            // very first bounds test.  Nothing to validate.
            if n_rows == 0 {
                return Ok(());
            }

            // With two or more rows, `stride` really is a *row* stride: rows
            // must not overlap, otherwise `stride` is not describing the matrix
            // the caller thinks it is.  (With a single row the stride is never
            // multiplied by anything but zero, so it is irrelevant.)
            if n_rows > 1 && stride < self.n_vars {
                return Err(JitBatchError::StrideTooSmall {
                    stride,
                    n_vars: self.n_vars,
                });
            }

            // Footprint of the *last* row: it starts at (n_rows - 1) * stride
            // and extends n_vars elements.  Checked arithmetic, because a
            // caller-supplied stride can be arbitrarily large.
            let need = (n_rows - 1)
                .checked_mul(stride)
                .and_then(|base| base.checked_add(self.n_vars))
                .ok_or(JitBatchError::LengthOverflow {
                    n_rows,
                    stride,
                    n_vars: self.n_vars,
                })?;

            if rows_len < need {
                return Err(JitBatchError::RowsTooShort {
                    got: rows_len,
                    need,
                });
            }
            if out_len < n_rows {
                return Err(JitBatchError::OutTooShort {
                    got: out_len,
                    need: n_rows,
                });
            }
            Ok(())
        }

        /// Evaluate the compiled expression for `n_rows` rows in **one** native
        /// call, writing `out[i]` for each row `i`.
        ///
        /// `rows` is a row-major matrix: row `i` supplies the variable slots
        /// `rows[i * stride .. i * stride + n_vars()]`.  `stride` is measured in
        /// `f64` **elements**, so a densely packed `n_rows × n_vars` matrix uses
        /// `stride == n_vars()`; a larger stride lets the caller carry extra
        /// columns (e.g. the target value) alongside the features.
        ///
        /// The row loop lives inside the generated machine code, so the cost of
        /// a batch is one call plus `n_rows` loop iterations, rather than
        /// `n_rows` calls.  The emitted arithmetic is identical to the one
        /// [`JitFn::call`] emits, therefore:
        ///
        /// ```text
        /// out[i].to_bits() == f.call(&rows[i * stride ..]).to_bits()
        /// ```
        ///
        /// for every row — a **0 ULP** guarantee, not an approximation.
        /// Transcendentals are scalar host calls in both paths, and no
        /// fast-math / FP-contraction rewrites are enabled in Cranelift.
        ///
        /// # Errors
        ///
        /// Returns [`JitBatchError`] if `rows` / `out` are too short for the
        /// requested `n_rows` and `stride`, if `stride < n_vars()` while
        /// `n_rows > 1` (overlapping rows), or if the requested footprint
        /// overflows `usize`.  Arguments are fully validated before any
        /// generated code runs, so a malformed request is always a clean `Err`
        /// and never undefined behaviour.
        ///
        /// Only `out[..n_rows]` is written; any trailing elements are left
        /// untouched.
        pub fn call_batch(
            &self,
            rows: &[f64],
            n_rows: usize,
            stride: usize,
            out: &mut [f64],
        ) -> Result<(), JitBatchError> {
            self.validate_batch(rows.len(), n_rows, stride, out.len())?;

            // SAFETY:
            // * `validate_batch` has just proved that the generated loop only
            //   reads `rows[i * stride + j]` for `i < n_rows`, `j < n_vars`,
            //   all of which are within `rows.len()`, and only writes
            //   `out[i]` for `i < n_rows`, all within `out.len()`.
            // * `rows` and `out` come from a `&[f64]` and a `&mut [f64]`, so
            //   Rust's borrow checker guarantees they do not alias; the
            //   generated code therefore cannot observe a torn read.
            // * Both pointers are non-null and `f64`-aligned (slice pointers
            //   always are, even for empty slices), which is what the emitted
            //   `load`/`store` instructions assume.  `n_rows == 0` passes
            //   dangling-but-aligned pointers, which the loop never
            //   dereferences because its first bounds test fails.
            // * The compiled code is `extern "C"`, does not unwind, does not
            //   retain the pointers past the call, and only mutates through
            //   `out_ptr`.
            unsafe {
                (self.batch_fn_ptr)(rows.as_ptr(), n_rows, stride, out.as_mut_ptr());
            }
            Ok(())
        }

        /// Rayon-parallel [`JitFn::call_batch`]: identical contract, identical
        /// results.
        ///
        /// Rows are split into **contiguous** chunks, each chunk is evaluated by
        /// a single call into the same compiled batch entry point, and every row
        /// is written by exactly one task.  No floating-point value is ever
        /// reduced or combined across tasks, so the output is bit-for-bit
        /// independent of the number of rayon threads and of the chunking:
        ///
        /// ```text
        /// call_batch_parallel(..) ≡ call_batch(..) ≡ per-row call(..)   (0 ULP)
        /// ```
        ///
        /// # Errors
        ///
        /// Same as [`JitFn::call_batch`].
        #[cfg(feature = "parallel")]
        pub fn call_batch_parallel(
            &self,
            rows: &[f64],
            n_rows: usize,
            stride: usize,
            out: &mut [f64],
        ) -> Result<(), JitBatchError> {
            self.validate_batch(rows.len(), n_rows, stride, out.len())?;
            if n_rows == 0 {
                return Ok(());
            }

            let chunk_rows = parallel_chunk_rows(n_rows);
            // A plain `fn` pointer is `Copy + Send + Sync`; copying it out of
            // `self` keeps the closure's captures trivially thread-safe.
            let batch_fn = self.batch_fn_ptr;

            out[..n_rows]
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_index, out_chunk)| {
                    let first_row = chunk_index * chunk_rows;
                    let chunk_n_rows = out_chunk.len();

                    // `first_row <= n_rows - 1`, and `validate_batch` proved
                    // that `(n_rows - 1) * stride` does not overflow and is
                    // `<= rows.len() - n_vars`, so this product neither
                    // overflows nor leaves the allocation.
                    let row_offset = first_row * stride;

                    // SAFETY:
                    // * `row_offset <= rows.len()`, so `add` stays inside the
                    //   allocation (or one past its end, which is allowed).
                    // * The sub-batch reads at most
                    //   `row_offset + (chunk_n_rows - 1) * stride + n_vars`
                    //   `== (first_row + chunk_n_rows - 1) * stride + n_vars`
                    //   `<= (n_rows - 1) * stride + n_vars <= rows.len()`
                    //   elements, i.e. exactly the region `validate_batch`
                    //   checked.
                    // * `par_chunks_mut` hands out pairwise-disjoint `&mut`
                    //   chunks, so no two tasks write the same `out` element,
                    //   and `rows` is only ever read.
                    // * The remaining invariants (alignment, no aliasing
                    //   between `rows` and `out`, `extern "C"`, no unwinding)
                    //   are the same as for `call_batch`.
                    unsafe {
                        let in_ptr = rows.as_ptr().add(row_offset);
                        batch_fn(in_ptr, chunk_n_rows, stride, out_chunk.as_mut_ptr());
                    }
                });

            Ok(())
        }
    }

    /// Minimum number of rows per rayon task.
    ///
    /// Below this, task-scheduling overhead dominates the actual evaluation.
    /// It has no effect on the *values* produced — rows are independent — only
    /// on how they are scheduled.
    #[cfg(feature = "parallel")]
    const MIN_PARALLEL_CHUNK_ROWS: usize = 64;

    /// Choose a contiguous chunk size for [`JitFn::call_batch_parallel`].
    ///
    /// One chunk per worker thread (rows are uniform cost, so work-stealing
    /// buys nothing), floored at [`MIN_PARALLEL_CHUNK_ROWS`] and never larger
    /// than the batch itself.
    #[cfg(feature = "parallel")]
    fn parallel_chunk_rows(n_rows: usize) -> usize {
        debug_assert!(n_rows > 0, "parallel_chunk_rows requires a non-empty batch");
        let threads = rayon::current_num_threads().max(1);
        let per_thread = n_rows.div_ceil(threads);
        if per_thread < MIN_PARALLEL_CHUNK_ROWS {
            MIN_PARALLEL_CHUNK_ROWS.min(n_rows)
        } else {
            per_thread
        }
    }

    // ─── external function registry ─────────────────────────────────────────

    /// Names and arities for all external math functions we may need.
    struct ExternIds {
        exp: FuncId,
        log: FuncId,
        sin: FuncId,
        cos: FuncId,
        pow: FuncId,
        tan: FuncId,
        sinh: FuncId,
        cosh: FuncId,
        tanh: FuncId,
        asin: FuncId,
        acos: FuncId,
        atan: FuncId,
        asinh: FuncId,
        acosh: FuncId,
        atanh: FuncId,
        erf: FuncId,
        lgamma: FuncId,
        digamma_fn: FuncId,
        trigamma_fn: FuncId,
        ei_fn: FuncId,
        si_fn: FuncId,
        ci_fn: FuncId,
    }

    /// Declare all external math functions in the module.
    fn declare_extern_fns(
        module: &mut JITModule,
        call_conv: CallConv,
    ) -> Result<ExternIds, Box<dyn std::error::Error + Send + Sync>> {
        let s1 = sig_f64_to_f64(call_conv);
        let s2 = sig_f64_f64_to_f64(call_conv);

        macro_rules! decl1 {
            ($name:expr) => {
                module
                    .declare_function($name, Linkage::Import, &s1)
                    .map_err(|e| format!("declare {}: {e}", $name))?
            };
        }

        Ok(ExternIds {
            exp: decl1!("exp"),
            log: decl1!("log"),
            sin: decl1!("sin"),
            cos: decl1!("cos"),
            pow: module
                .declare_function("pow", Linkage::Import, &s2)
                .map_err(|e| format!("declare pow: {e}"))?,
            tan: decl1!("tan"),
            sinh: decl1!("sinh"),
            cosh: decl1!("cosh"),
            tanh: decl1!("tanh"),
            asin: decl1!("asin"),
            acos: decl1!("acos"),
            atan: decl1!("atan"),
            asinh: decl1!("asinh"),
            acosh: decl1!("acosh"),
            atanh: decl1!("atanh"),
            erf: decl1!("oxieml_erf"),
            lgamma: decl1!("oxieml_lgamma"),
            digamma_fn: decl1!("oxieml_digamma"),
            trigamma_fn: decl1!("oxieml_trigamma"),
            ei_fn: decl1!("oxieml_ei"),
            si_fn: decl1!("oxieml_si"),
            ci_fn: decl1!("oxieml_ci"),
        })
    }

    // ─── IR emission ────────────────────────────────────────────────────────

    /// Emit Cranelift IR for one `OxiOp`, operating on the `vstack`.
    fn emit_op(
        op: &OxiOp,
        builder: &mut FunctionBuilder<'_>,
        vstack: &mut Vec<cranelift_codegen::ir::Value>,
        vars_ptr: cranelift_codegen::ir::Value,
        extern_ids: &ExternIds,
        module: &mut JITModule,
        slot_values: &mut HashMap<usize, cranelift_codegen::ir::Value>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match op {
            OxiOp::Const(c) => {
                let v = builder.ins().f64const(*c);
                vstack.push(v);
            }
            OxiOp::Var(i) => {
                // Load f64 from `vars_ptr + i * 8`.
                let offset = i32::try_from(*i * 8)
                    .map_err(|_| format!("Var index {i} too large for i32 offset"))?;
                let v = builder
                    .ins()
                    .load(F64, MemFlagsData::trusted(), vars_ptr, offset);
                vstack.push(v);
            }
            OxiOp::Add => {
                let b = vstack.pop().ok_or("stack underflow at Add")?;
                let a = vstack.pop().ok_or("stack underflow at Add")?;
                vstack.push(builder.ins().fadd(a, b));
            }
            OxiOp::Sub => {
                let b = vstack.pop().ok_or("stack underflow at Sub")?;
                let a = vstack.pop().ok_or("stack underflow at Sub")?;
                vstack.push(builder.ins().fsub(a, b));
            }
            OxiOp::Mul => {
                let b = vstack.pop().ok_or("stack underflow at Mul")?;
                let a = vstack.pop().ok_or("stack underflow at Mul")?;
                vstack.push(builder.ins().fmul(a, b));
            }
            OxiOp::Div => {
                let b = vstack.pop().ok_or("stack underflow at Div")?;
                let a = vstack.pop().ok_or("stack underflow at Div")?;
                vstack.push(builder.ins().fdiv(a, b));
            }
            OxiOp::Neg => {
                let a = vstack.pop().ok_or("stack underflow at Neg")?;
                vstack.push(builder.ins().fneg(a));
            }
            OxiOp::Exp => {
                let a = vstack.pop().ok_or("stack underflow at Exp")?;
                let result = call_extern1(builder, module, extern_ids.exp, a)?;
                vstack.push(result);
            }
            OxiOp::Ln => {
                let a = vstack.pop().ok_or("stack underflow at Ln")?;
                let result = call_extern1(builder, module, extern_ids.log, a)?;
                vstack.push(result);
            }
            OxiOp::Sin => {
                let a = vstack.pop().ok_or("stack underflow at Sin")?;
                let result = call_extern1(builder, module, extern_ids.sin, a)?;
                vstack.push(result);
            }
            OxiOp::Cos => {
                let a = vstack.pop().ok_or("stack underflow at Cos")?;
                let result = call_extern1(builder, module, extern_ids.cos, a)?;
                vstack.push(result);
            }
            OxiOp::Pow => {
                let b = vstack.pop().ok_or("stack underflow at Pow")?;
                let a = vstack.pop().ok_or("stack underflow at Pow")?;
                let result = call_extern2(builder, module, extern_ids.pow, a, b)?;
                vstack.push(result);
            }
            OxiOp::Tan => {
                let a = vstack.pop().ok_or("stack underflow at Tan")?;
                let result = call_extern1(builder, module, extern_ids.tan, a)?;
                vstack.push(result);
            }
            OxiOp::Sinh => {
                let a = vstack.pop().ok_or("stack underflow at Sinh")?;
                let result = call_extern1(builder, module, extern_ids.sinh, a)?;
                vstack.push(result);
            }
            OxiOp::Cosh => {
                let a = vstack.pop().ok_or("stack underflow at Cosh")?;
                let result = call_extern1(builder, module, extern_ids.cosh, a)?;
                vstack.push(result);
            }
            OxiOp::Tanh => {
                let a = vstack.pop().ok_or("stack underflow at Tanh")?;
                let result = call_extern1(builder, module, extern_ids.tanh, a)?;
                vstack.push(result);
            }
            OxiOp::Arcsin => {
                let a = vstack.pop().ok_or("stack underflow at Arcsin")?;
                let result = call_extern1(builder, module, extern_ids.asin, a)?;
                vstack.push(result);
            }
            OxiOp::Arccos => {
                let a = vstack.pop().ok_or("stack underflow at Arccos")?;
                let result = call_extern1(builder, module, extern_ids.acos, a)?;
                vstack.push(result);
            }
            OxiOp::Arctan => {
                let a = vstack.pop().ok_or("stack underflow at Arctan")?;
                let result = call_extern1(builder, module, extern_ids.atan, a)?;
                vstack.push(result);
            }
            OxiOp::Arcsinh => {
                let a = vstack.pop().ok_or("stack underflow at Arcsinh")?;
                let result = call_extern1(builder, module, extern_ids.asinh, a)?;
                vstack.push(result);
            }
            OxiOp::Arccosh => {
                let a = vstack.pop().ok_or("stack underflow at Arccosh")?;
                let result = call_extern1(builder, module, extern_ids.acosh, a)?;
                vstack.push(result);
            }
            OxiOp::Arctanh => {
                let a = vstack.pop().ok_or("stack underflow at Arctanh")?;
                let result = call_extern1(builder, module, extern_ids.atanh, a)?;
                vstack.push(result);
            }
            OxiOp::Erf => {
                let a = vstack.pop().ok_or("stack underflow at Erf")?;
                let result = call_extern1(builder, module, extern_ids.erf, a)?;
                vstack.push(result);
            }
            OxiOp::LGamma => {
                let a = vstack.pop().ok_or("stack underflow at LGamma")?;
                let result = call_extern1(builder, module, extern_ids.lgamma, a)?;
                vstack.push(result);
            }
            OxiOp::Digamma => {
                let a = vstack.pop().ok_or("stack underflow at Digamma")?;
                let result = call_extern1(builder, module, extern_ids.digamma_fn, a)?;
                vstack.push(result);
            }
            OxiOp::Trigamma => {
                let a = vstack.pop().ok_or("stack underflow at Trigamma")?;
                let result = call_extern1(builder, module, extern_ids.trigamma_fn, a)?;
                vstack.push(result);
            }
            OxiOp::Ei => {
                let a = vstack.pop().ok_or("stack underflow at Ei")?;
                let result = call_extern1(builder, module, extern_ids.ei_fn, a)?;
                vstack.push(result);
            }
            OxiOp::Si => {
                let a = vstack.pop().ok_or("stack underflow at Si")?;
                let result = call_extern1(builder, module, extern_ids.si_fn, a)?;
                vstack.push(result);
            }
            OxiOp::Ci => {
                let a = vstack.pop().ok_or("stack underflow at Ci")?;
                let result = call_extern1(builder, module, extern_ids.ci_fn, a)?;
                vstack.push(result);
            }
            OxiOp::Store(k) => {
                // Peek top of vstack (does NOT pop) and record the SSA value in slot k.
                let top = *vstack.last().ok_or("stack underflow at Store")?;
                slot_values.insert(*k, top);
            }
            OxiOp::Load(k) => {
                // Push the SSA value previously stored in slot k.
                let v = slot_values
                    .get(k)
                    .copied()
                    .ok_or_else(|| format!("OxiOp::Load({k}) before Store — malformed IR"))?;
                vstack.push(v);
            }
        }
        Ok(())
    }

    /// Emit a call to a unary external `f64 → f64` function.
    fn call_extern1(
        builder: &mut FunctionBuilder<'_>,
        module: &mut JITModule,
        func_id: FuncId,
        arg: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, Box<dyn std::error::Error + Send + Sync>> {
        let func_ref = module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, &[arg]);
        let results = builder.inst_results(call);
        results
            .first()
            .copied()
            .ok_or_else(|| "extern f64→f64 call returned no value".into())
    }

    /// Emit a call to a binary external `(f64, f64) → f64` function.
    fn call_extern2(
        builder: &mut FunctionBuilder<'_>,
        module: &mut JITModule,
        func_id: FuncId,
        arg0: cranelift_codegen::ir::Value,
        arg1: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, Box<dyn std::error::Error + Send + Sync>> {
        let func_ref = module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, &[arg0, arg1]);
        let results = builder.inst_results(call);
        results
            .first()
            .copied()
            .ok_or_else(|| "extern (f64,f64)→f64 call returned no value".into())
    }

    // ─── batch code generation ──────────────────────────────────────────────

    /// Emit the whole `OxiOp` sequence for **one** row and return the SSA value
    /// holding its `f64` result.
    ///
    /// `row_base` is the address of element `0` of that row; `Var(i)` becomes a
    /// load from `row_base + i * 8`.  Passing the function's `vars_ptr`
    /// parameter yields the scalar entry point; passing a loop-carried
    /// `in_ptr + i * stride * 8` yields one iteration of the batch loop — the
    /// two are the *same* instruction sequence, which is what makes
    /// [`JitFn::call_batch`] bit-exact with [`JitFn::call`].
    fn emit_row_scalar(
        ops: &[OxiOp],
        builder: &mut FunctionBuilder<'_>,
        row_base: Value,
        extern_ids: &ExternIds,
        module: &mut JITModule,
    ) -> JitResult<Value> {
        // Operand stack of the post-order machine, holding SSA values.
        let mut vstack: Vec<Value> = Vec::new();
        // Slot register file: maps a slot index to an SSA value (Store/Load).
        let mut slot_values: HashMap<usize, Value> = HashMap::new();

        for op in ops {
            emit_op(
                op,
                builder,
                &mut vstack,
                row_base,
                extern_ids,
                module,
                &mut slot_values,
            )?;
        }

        let result = vstack
            .pop()
            .ok_or("OxiOp sequence produced empty stack — malformed ops")?;
        Ok(result)
    }

    /// Common prologue of both batch loops: bind the four ABI parameters and
    /// pre-compute the loop-invariant byte stride.
    ///
    /// Returns `(in_ptr, n_rows, stride_bytes, out_ptr)`.  The caller must
    /// already have switched to — and sealed — `entry_block`.
    fn batch_prologue(
        builder: &mut FunctionBuilder<'_>,
        entry_block: cranelift_codegen::ir::Block,
    ) -> (Value, Value, Value, Value) {
        let params = builder.block_params(entry_block);
        let (in_ptr, n_rows, stride, out_ptr) = (params[0], params[1], params[2], params[3]);

        // `stride` arrives in elements; the IR needs bytes.  A wrapping `imul`
        // is harmless here: `JitFn::validate_batch` guarantees that whenever
        // `stride_bytes` is actually *used* (i.e. `n_rows > 1`, so the loop
        // index can be non-zero) the product `i * stride * 8` is bounded by the
        // byte size of a live Rust allocation and hence cannot overflow.  For
        // `n_rows <= 1` the index is always zero and the product is zero
        // regardless of what `stride_bytes` happens to contain.
        let stride_bytes = builder.ins().imul_imm(stride, F64_BYTES);

        (in_ptr, n_rows, stride_bytes, out_ptr)
    }

    /// Address of row `i`: `in_ptr + i * stride_bytes`.
    fn row_address(
        builder: &mut FunctionBuilder<'_>,
        in_ptr: Value,
        index: Value,
        stride_bytes: Value,
    ) -> Value {
        let byte_offset = builder.ins().imul(index, stride_bytes);
        builder.ins().iadd(in_ptr, byte_offset)
    }

    /// Address of output element `i`: `out_ptr + i * 8`.
    fn out_address(builder: &mut FunctionBuilder<'_>, out_ptr: Value, index: Value) -> Value {
        let byte_offset = builder.ins().imul_imm(index, F64_BYTES);
        builder.ins().iadd(out_ptr, byte_offset)
    }

    /// Tier 1 — emit the scalar batch entry point: one row per loop iteration.
    ///
    /// Generated control-flow graph:
    ///
    /// ```text
    ///   entry(in_ptr, n_rows, stride, out_ptr):     ; no predecessors
    ///       stride_bytes = stride * 8
    ///       jump header(0)
    ///
    ///   header(i):                                  ; preds: entry, body
    ///       brif (i <u n_rows), body(i), exit
    ///
    ///   body(i):                                    ; preds: header
    ///       row   = in_ptr + i * stride_bytes
    ///       value = <ops, with Var(j) := load [row + 8j]>
    ///       store value -> [out_ptr + 8i]
    ///       jump header(i + 1)                      ; back-edge
    ///
    ///   exit:                                       ; preds: header
    ///       return
    /// ```
    ///
    /// **Block-sealing discipline** (the one thing Cranelift will not forgive):
    ///
    /// * `entry` has no predecessors → sealed the moment it is switched to.
    /// * `body` and `exit` are reachable *only* from the `brif` in `header`, so
    ///   they are sealed immediately after that `brif` is emitted — at which
    ///   point their (single) predecessor is known and no further branch to
    ///   them will ever be added.
    /// * `header` has two predecessors, `entry` and the back-edge at the end of
    ///   `body`.  It is therefore sealed **after** `body`'s terminating `jump`
    ///   has been emitted, and not one instruction earlier.
    ///
    /// The loop index is carried as an explicit block parameter (a hand-written
    /// φ) on both `header` and `body`, so no `cranelift_frontend::Variable` is
    /// involved and the SSA construction cannot silently depend on seal order.
    ///
    /// Row bases are recomputed as `in_ptr + i * stride_bytes` rather than
    /// accumulated with `row += stride_bytes`, so that `stride == 0` and
    /// `n_rows <= 1` degrade to exactly `in_ptr` and cannot drift.
    fn build_batch_scalar_loop(
        ops: &[OxiOp],
        builder: &mut FunctionBuilder<'_>,
        ptr_type: Type,
        extern_ids: &ExternIds,
        module: &mut JITModule,
    ) -> JitResult<()> {
        let entry_block = builder.create_block();
        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.append_block_params_for_function_params(entry_block);
        // Loop-carried index `i`.
        builder.append_block_param(header_block, ptr_type);
        builder.append_block_param(body_block, ptr_type);

        // ── entry ───────────────────────────────────────────────────────────
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block); // no predecessors
        let (in_ptr, n_rows, stride_bytes, out_ptr) = batch_prologue(builder, entry_block);
        let zero = builder.ins().iconst(ptr_type, 0);
        builder.ins().jump(header_block, &[BlockArg::Value(zero)]);

        // ── header ──────────────────────────────────────────────────────────
        builder.switch_to_block(header_block);
        let index = builder.block_params(header_block)[0];
        // Unsigned compare: `n_rows` is a `usize` and the index starts at 0, so
        // a signed compare could only differ for values that cannot fit in any
        // real allocation anyway — but unsigned is the honest semantics.
        let keep_going = builder.ins().icmp(IntCC::UnsignedLessThan, index, n_rows);
        builder.ins().brif(
            keep_going,
            body_block,
            &[BlockArg::Value(index)],
            exit_block,
            &[],
        );
        // `body` and `exit` are now fully connected: `header` is their only
        // predecessor and it has just been terminated.
        builder.seal_block(body_block);
        builder.seal_block(exit_block);

        // ── body ────────────────────────────────────────────────────────────
        builder.switch_to_block(body_block);
        let index = builder.block_params(body_block)[0];
        let row_base = row_address(builder, in_ptr, index, stride_bytes);

        let result = emit_row_scalar(ops, builder, row_base, extern_ids, module)?;

        let out_slot = out_address(builder, out_ptr, index);
        builder
            .ins()
            .store(MemFlagsData::trusted(), result, out_slot, 0);

        let next_index = builder.ins().iadd_imm(index, 1);
        builder
            .ins()
            .jump(header_block, &[BlockArg::Value(next_index)]);
        // The back-edge exists now, so `header`'s predecessor set is complete.
        builder.seal_block(header_block);

        // ── exit ────────────────────────────────────────────────────────────
        builder.switch_to_block(exit_block);
        builder.ins().return_(&[]);

        Ok(())
    }

    /// The two row base addresses processed by one iteration of the vector loop.
    #[derive(Clone, Copy)]
    struct RowPair {
        /// Base address of the row that lands in lane 0.
        lane0: Value,
        /// Base address of the row that lands in lane 1.
        lane1: Value,
    }

    /// Tier 2 — is every op in `ops` legal inside the two-lane `f64x2` loop?
    ///
    /// The gate admits exactly the ops whose vector form is **bit-identical** to
    /// the scalar form, which is the hard requirement of
    /// [`JitFn::call_batch`]:
    ///
    /// * `Const`, `Var`, `Store`, `Load` — pure data movement.
    /// * `Add`, `Sub`, `Mul`, `Div`, `Neg` — IEEE-754 is defined lane-wise, so
    ///   `fadd.f64x2` computes the very same result as two `fadd.f64`s.  (No
    ///   FP-contraction or reassociation is performed by Cranelift, and packed
    ///   and scalar SSE/NEON operations share the same rounding mode and
    ///   denormal handling.)
    /// * `Pow` — kept as **scalar host calls**, one per lane, with the lanes
    ///   extracted and repacked around them.  This is what preserves 0 ULP:
    ///   expanding `x^n` into repeated multiplication would *not* reproduce
    ///   libm's `pow`, so it is deliberately not done, and the surrounding
    ///   arithmetic still vectorizes.
    ///
    /// The remaining transcendentals (`Exp`, `Ln`, `Sin`, …, `Erf`, `LGamma`,
    /// …) are excluded: they are host calls with no vector counterpart, so a
    /// vector loop containing them would pay for lane extraction and repacking
    /// on *every* op without vectorizing anything.  Such sequences use the
    /// Tier 1 scalar loop, which is exactly as fast for them.
    fn is_vectorizable(ops: &[OxiOp]) -> bool {
        ops.iter().all(|op| {
            matches!(
                op,
                OxiOp::Const(_)
                    | OxiOp::Var(_)
                    | OxiOp::Add
                    | OxiOp::Sub
                    | OxiOp::Mul
                    | OxiOp::Div
                    | OxiOp::Neg
                    | OxiOp::Pow
                    | OxiOp::Store(_)
                    | OxiOp::Load(_)
            )
        })
    }

    /// Build an `f64x2` from two scalars: `lane0` → lane 0, `lane1` → lane 1.
    fn pack_lanes(builder: &mut FunctionBuilder<'_>, lane0: Value, lane1: Value) -> Value {
        let vector = builder.ins().scalar_to_vector(F64X2, lane0);
        builder.ins().insertlane(vector, lane1, 1_u8)
    }

    /// Emit one vectorized `OxiOp`, operating on a stack of `f64x2` values.
    ///
    /// Lane `k` of every value on `vstack` holds the computation for
    /// `rows.lane{k}`, so the two rows are evaluated in lock-step.  Only the ops
    /// accepted by [`is_vectorizable`] may reach here; anything else is a
    /// codegen bug and is reported as an error rather than silently mis-emitted.
    fn emit_op_vector(
        op: &OxiOp,
        builder: &mut FunctionBuilder<'_>,
        vstack: &mut Vec<Value>,
        rows: RowPair,
        extern_ids: &ExternIds,
        module: &mut JITModule,
        slot_values: &mut HashMap<usize, Value>,
    ) -> JitResult<()> {
        match op {
            OxiOp::Const(c) => {
                let scalar = builder.ins().f64const(*c);
                let splatted = builder.ins().splat(F64X2, scalar);
                vstack.push(splatted);
            }
            OxiOp::Var(i) => {
                // Rows are `stride` elements apart, so the two lanes come from
                // two *separate* scalar loads — a packed load would be a gather.
                // The arithmetic that follows is what gets vectorized.
                let offset = i32::try_from(*i * 8)
                    .map_err(|_| format!("Var index {i} too large for i32 offset"))?;
                let a = builder
                    .ins()
                    .load(F64, MemFlagsData::trusted(), rows.lane0, offset);
                let b = builder
                    .ins()
                    .load(F64, MemFlagsData::trusted(), rows.lane1, offset);
                let packed = pack_lanes(builder, a, b);
                vstack.push(packed);
            }
            OxiOp::Add => {
                let b = vstack.pop().ok_or("stack underflow at Add")?;
                let a = vstack.pop().ok_or("stack underflow at Add")?;
                vstack.push(builder.ins().fadd(a, b));
            }
            OxiOp::Sub => {
                let b = vstack.pop().ok_or("stack underflow at Sub")?;
                let a = vstack.pop().ok_or("stack underflow at Sub")?;
                vstack.push(builder.ins().fsub(a, b));
            }
            OxiOp::Mul => {
                let b = vstack.pop().ok_or("stack underflow at Mul")?;
                let a = vstack.pop().ok_or("stack underflow at Mul")?;
                vstack.push(builder.ins().fmul(a, b));
            }
            OxiOp::Div => {
                let b = vstack.pop().ok_or("stack underflow at Div")?;
                let a = vstack.pop().ok_or("stack underflow at Div")?;
                vstack.push(builder.ins().fdiv(a, b));
            }
            OxiOp::Neg => {
                let a = vstack.pop().ok_or("stack underflow at Neg")?;
                vstack.push(builder.ins().fneg(a));
            }
            OxiOp::Pow => {
                // Transcendental: stays a scalar host call, once per lane.
                let b = vstack.pop().ok_or("stack underflow at Pow")?;
                let a = vstack.pop().ok_or("stack underflow at Pow")?;
                let base0 = builder.ins().extractlane(a, 0_u8);
                let base1 = builder.ins().extractlane(a, 1_u8);
                let exp0 = builder.ins().extractlane(b, 0_u8);
                let exp1 = builder.ins().extractlane(b, 1_u8);
                let r0 = call_extern2(builder, module, extern_ids.pow, base0, exp0)?;
                let r1 = call_extern2(builder, module, extern_ids.pow, base1, exp1)?;
                let packed = pack_lanes(builder, r0, r1);
                vstack.push(packed);
            }
            OxiOp::Store(k) => {
                let top = *vstack.last().ok_or("stack underflow at Store")?;
                slot_values.insert(*k, top);
            }
            OxiOp::Load(k) => {
                let v = slot_values
                    .get(k)
                    .copied()
                    .ok_or_else(|| format!("OxiOp::Load({k}) before Store — malformed IR"))?;
                vstack.push(v);
            }
            other => {
                return Err(format!(
                    "emit_op_vector: {other:?} has no vector form; \
                     is_vectorizable() must not admit it"
                )
                .into());
            }
        }
        Ok(())
    }

    /// Emit the whole `OxiOp` sequence for **two** rows at once, returning the
    /// `f64x2` value whose lane `k` is the result for `rows.lane{k}`.
    fn emit_row_pair_vector(
        ops: &[OxiOp],
        builder: &mut FunctionBuilder<'_>,
        rows: RowPair,
        extern_ids: &ExternIds,
        module: &mut JITModule,
    ) -> JitResult<Value> {
        let mut vstack: Vec<Value> = Vec::new();
        let mut slot_values: HashMap<usize, Value> = HashMap::new();

        for op in ops {
            emit_op_vector(
                op,
                builder,
                &mut vstack,
                rows,
                extern_ids,
                module,
                &mut slot_values,
            )?;
        }

        let result = vstack
            .pop()
            .ok_or("OxiOp sequence produced empty stack — malformed ops")?;
        Ok(result)
    }

    /// Tier 2 — emit the vectorized batch entry point: **two** rows per
    /// iteration of the main loop, plus a scalar epilogue for the odd row.
    ///
    /// Generated control-flow graph:
    ///
    /// ```text
    ///   entry(in_ptr, n_rows, stride, out_ptr):     ; no predecessors
    ///       stride_bytes = stride * 8
    ///       pair_end     = n_rows & !1              ; largest even <= n_rows
    ///       jump vheader(0)
    ///
    ///   vheader(i):                                 ; preds: entry, vbody
    ///       brif (i <u pair_end), vbody(i), theader(i)
    ///
    ///   vbody(i):                                   ; preds: vheader
    ///       lane0 = in_ptr + i * stride_bytes
    ///       lane1 = lane0 + stride_bytes
    ///       v     = <ops, in f64x2>
    ///       store extractlane(v, 0) -> [out_ptr + 8i]
    ///       store extractlane(v, 1) -> [out_ptr + 8i + 8]
    ///       jump vheader(i + 2)                     ; back-edge
    ///
    ///   theader(i):                                 ; preds: vheader, tbody
    ///       brif (i <u n_rows), tbody(i), exit
    ///
    ///   tbody(i):                                   ; preds: theader
    ///       <the Tier 1 body, one row>              ; runs at most once
    ///       jump theader(i + 1)                     ; back-edge
    ///
    ///   exit:                                       ; preds: theader
    ///       return
    /// ```
    ///
    /// **Block-sealing discipline**, same rules as Tier 1 but with two loops:
    ///
    /// * `entry` — sealed on entry (no predecessors).
    /// * `vbody` — sealed right after `vheader`'s `brif` (its only predecessor).
    /// * `vheader` — sealed only after `vbody`'s back-edge exists.
    /// * `tbody` and `exit` — sealed right after `theader`'s `brif`.
    /// * `theader` — has *two* predecessors (`vheader`'s else-edge and `tbody`'s
    ///   back-edge), so it is sealed last, after `tbody` is terminated.  It is
    ///   filled while still unsealed, which is legal precisely because the index
    ///   is an explicit block parameter rather than a `Variable`.
    ///
    /// Results are written with two scalar stores rather than one 16-byte vector
    /// store: `out` is only `f64`-aligned and lane-to-byte order is
    /// target-endian, so scalar stores are both portable and alignment-safe,
    /// while costing one extra instruction per row pair.
    ///
    /// Row counts of 0, 1 and 2 are the interesting boundaries: `n_rows = 0`
    /// gives `pair_end = 0` and both loops fall straight through to `exit`;
    /// `n_rows = 1` gives `pair_end = 0`, so the single row is handled entirely
    /// by the scalar epilogue; `n_rows = 2` gives `pair_end = 2`, so one vector
    /// iteration runs and the epilogue does nothing.
    fn build_batch_vector_loop(
        ops: &[OxiOp],
        builder: &mut FunctionBuilder<'_>,
        ptr_type: Type,
        extern_ids: &ExternIds,
        module: &mut JITModule,
    ) -> JitResult<()> {
        let entry_block = builder.create_block();
        let vector_header = builder.create_block();
        let vector_body = builder.create_block();
        let tail_header = builder.create_block();
        let tail_body = builder.create_block();
        let exit_block = builder.create_block();

        builder.append_block_params_for_function_params(entry_block);
        for block in [vector_header, vector_body, tail_header, tail_body] {
            builder.append_block_param(block, ptr_type);
        }

        // ── entry ───────────────────────────────────────────────────────────
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block); // no predecessors
        let (in_ptr, n_rows, stride_bytes, out_ptr) = batch_prologue(builder, entry_block);

        // `pair_end = n_rows & !1` — clearing the low bit yields the largest
        // even number <= n_rows, i.e. the first index the vector loop must not
        // process.  `-2` is `…1110` in two's complement for any integer width.
        let pair_end = builder.ins().band_imm(n_rows, -2);

        let zero = builder.ins().iconst(ptr_type, 0);
        builder.ins().jump(vector_header, &[BlockArg::Value(zero)]);

        // ── vector header ───────────────────────────────────────────────────
        builder.switch_to_block(vector_header);
        let index = builder.block_params(vector_header)[0];
        let has_pair = builder.ins().icmp(IntCC::UnsignedLessThan, index, pair_end);
        builder.ins().brif(
            has_pair,
            vector_body,
            &[BlockArg::Value(index)],
            tail_header,
            &[BlockArg::Value(index)],
        );
        // `vector_body` is reachable only from here, so it can be sealed.
        // `tail_header` cannot: `tail_body` will branch back into it.
        builder.seal_block(vector_body);

        // ── vector body ─────────────────────────────────────────────────────
        builder.switch_to_block(vector_body);
        let index = builder.block_params(vector_body)[0];
        let lane0 = row_address(builder, in_ptr, index, stride_bytes);
        // Row `i + 1` is exactly one stride past row `i`.
        let lane1 = builder.ins().iadd(lane0, stride_bytes);

        let packed =
            emit_row_pair_vector(ops, builder, RowPair { lane0, lane1 }, extern_ids, module)?;

        let result0 = builder.ins().extractlane(packed, 0_u8);
        let result1 = builder.ins().extractlane(packed, 1_u8);
        let out_slot = out_address(builder, out_ptr, index);
        builder
            .ins()
            .store(MemFlagsData::trusted(), result0, out_slot, 0);
        // `out_slot + 8` is `out[i + 1]`; `i` is even and `i + 1 < n_rows`
        // because `i < pair_end <= n_rows` and `pair_end` is even.
        builder
            .ins()
            .store(MemFlagsData::trusted(), result1, out_slot, F64_OFFSET);

        let next_index = builder.ins().iadd_imm(index, 2);
        builder
            .ins()
            .jump(vector_header, &[BlockArg::Value(next_index)]);
        // The back-edge exists now, so `vector_header` is complete.
        builder.seal_block(vector_header);

        // ── tail header ─────────────────────────────────────────────────────
        // Still unsealed at this point — `tail_body`'s back-edge is not emitted
        // yet.  Filling an unsealed block is fine: the loop index is a block
        // parameter, not a `Variable`, so no φ has to be inferred.
        builder.switch_to_block(tail_header);
        let index = builder.block_params(tail_header)[0];
        let has_row = builder.ins().icmp(IntCC::UnsignedLessThan, index, n_rows);
        builder.ins().brif(
            has_row,
            tail_body,
            &[BlockArg::Value(index)],
            exit_block,
            &[],
        );
        builder.seal_block(tail_body);
        builder.seal_block(exit_block);

        // ── tail body ───────────────────────────────────────────────────────
        // Runs at most once (`n_rows - pair_end` is 0 or 1) and emits exactly
        // the Tier 1 / scalar-`call` instruction sequence, which is why the odd
        // last row is bit-exact with the rows handled in pairs.
        builder.switch_to_block(tail_body);
        let index = builder.block_params(tail_body)[0];
        let row_base = row_address(builder, in_ptr, index, stride_bytes);

        let result = emit_row_scalar(ops, builder, row_base, extern_ids, module)?;

        let out_slot = out_address(builder, out_ptr, index);
        builder
            .ins()
            .store(MemFlagsData::trusted(), result, out_slot, 0);

        let next_index = builder.ins().iadd_imm(index, 1);
        builder
            .ins()
            .jump(tail_header, &[BlockArg::Value(next_index)]);
        builder.seal_block(tail_header);

        // ── exit ────────────────────────────────────────────────────────────
        builder.switch_to_block(exit_block);
        builder.ins().return_(&[]);

        Ok(())
    }

    // ─── JitCache ───────────────────────────────────────────────────────────

    /// Thread-safe cache mapping a structural hash of an `OxiOp` sequence to
    /// a compiled [`JitFn`].
    ///
    /// On first call for a given sequence, the function is compiled and the
    /// result cached. Subsequent calls return the same [`Arc<JitFn>`].
    pub struct JitCache {
        cache: Mutex<HashMap<u64, Arc<JitFn>>>,
    }

    impl JitCache {
        /// Create an empty cache.
        pub fn new() -> Self {
            Self {
                cache: Mutex::new(HashMap::new()),
            }
        }

        /// Return a compiled [`JitFn`] for `ops`, compiling on first use.
        ///
        /// # Errors
        ///
        /// Returns an error if compilation fails or the cache lock is poisoned.
        pub fn get_or_compile(
            &self,
            ops: &[OxiOp],
            n_vars: usize,
        ) -> Result<Arc<JitFn>, Box<dyn std::error::Error + Send + Sync>> {
            let key = ops_hash(ops);

            {
                let guard = self
                    .cache
                    .lock()
                    .map_err(|e| format!("JitCache lock poisoned: {e}"))?;
                if let Some(f) = guard.get(&key) {
                    return Ok(Arc::clone(f));
                }
            }

            let compiled = Arc::new(JitFn::compile(ops, n_vars)?);

            let mut guard = self
                .cache
                .lock()
                .map_err(|e| format!("JitCache lock poisoned (post-compile): {e}"))?;
            // Another thread may have compiled while we were waiting — keep whichever arrived first.
            let entry = guard.entry(key).or_insert_with(|| Arc::clone(&compiled));
            Ok(Arc::clone(entry))
        }

        /// Number of cached entries.
        pub fn len(&self) -> usize {
            self.cache.lock().map(|g| g.len()).unwrap_or(0)
        }

        /// Whether the cache is empty.
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    impl Default for JitCache {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ─── public re-exports ───────────────────────────────────────────────────────

#[cfg(feature = "jit")]
pub use inner::{JitBatchError, JitCache, JitFn, ops_hash};
