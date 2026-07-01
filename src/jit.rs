//! JIT compilation of `OxiOp` sequences via Cranelift.
//!
//! This module provides [`JitFn`] and [`JitCache`] for compiling post-order
//! `OxiOp` stacks to native machine code through Cranelift's JIT backend.
//! The compiled function has the signature `fn(*const f64, usize) -> f64`.
//!
//! The module is gated behind the `jit` feature and has **zero impact** on
//! default builds.

#[cfg(feature = "jit")]
mod inner {
    use cranelift_codegen::ir::types::F64;
    use cranelift_codegen::ir::{
        AbiParam, Function, InstBuilder, MemFlagsData, Signature, UserFuncName,
    };
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{FuncId, Linkage, Module};

    use crate::lower::OxiOp;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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

    // ─── JitFn ──────────────────────────────────────────────────────────────

    /// A JIT-compiled function over a flat `&[f64]` variable slice.
    ///
    /// Created by [`JitFn::compile`]; called by [`JitFn::call`].
    /// Holds the [`JITModule`] alive so the generated code is not reclaimed.
    pub struct JitFn {
        /// Unsafe function pointer: `fn(vars_ptr: *const f64, vars_len: usize) -> f64`.
        fn_ptr: unsafe extern "C" fn(*const f64, usize) -> f64,
        /// Keep the JIT module alive so the code mapping is not freed.
        _module: JITModule,
        /// Minimum number of variables required.
        n_vars: usize,
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
        /// `n_vars` is the number of `Var(i)` slots; the pointer arithmetic in
        /// the emitted code accesses indices `0..n_vars` without bounds checks,
        /// so callers **must** pass at least `n_vars` elements.
        pub fn compile(
            ops: &[OxiOp],
            n_vars: usize,
        ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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

            // ── Declare the main function ───────────────────────────────────
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

            // ── Declare external math functions ─────────────────────────────
            let extern_ids = declare_extern_fns(&mut module, call_conv)?;

            // ── Build the IR ────────────────────────────────────────────────
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
                builder.seal_block(entry_block);

                let vars_ptr_val = builder.block_params(entry_block)[0];

                // Stack for cranelift Values
                let mut vstack: Vec<cranelift_codegen::ir::Value> = Vec::new();
                // Slot register file: maps slot index to SSA Value (for Store/Load).
                let mut slot_values: HashMap<usize, cranelift_codegen::ir::Value> = HashMap::new();

                for op in ops {
                    emit_op(
                        op,
                        &mut builder,
                        &mut vstack,
                        vars_ptr_val,
                        &extern_ids,
                        &mut module,
                        &mut slot_values,
                    )?;
                }

                let result = vstack
                    .pop()
                    .ok_or("OxiOp sequence produced empty stack — malformed ops")?;

                builder.ins().return_(&[result]);
                builder.finalize();
            }

            // ── Compile ─────────────────────────────────────────────────────
            module
                .define_function(main_func_id, &mut ctx)
                .map_err(|e| format!("define_function: {e}"))?;
            module.clear_context(&mut ctx);
            module
                .finalize_definitions()
                .map_err(|e| format!("finalize_definitions: {e}"))?;

            let raw_ptr = module.get_finalized_function(main_func_id);
            // SAFETY: The pointer is valid as long as `module` is alive.
            let fn_ptr: unsafe extern "C" fn(*const f64, usize) -> f64 =
                unsafe { std::mem::transmute(raw_ptr) };

            Ok(Self {
                fn_ptr,
                _module: module,
                n_vars: effective_n_vars,
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
pub use inner::{JitCache, JitFn, ops_hash};
