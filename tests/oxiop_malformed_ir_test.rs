//! Tests for OxiOp malformed-IR detection in debug builds.

#[cfg(debug_assertions)]
mod debug_only {
    use oxieml::lower::{LoweredOp, OxiOp};

    #[test]
    #[should_panic(expected = "stack underflow")]
    fn eval_ops_underflow_panics_in_debug() {
        // A bare Add with no operands on the stack — underflow on first pop.
        let ops = vec![OxiOp::Add];
        LoweredOp::eval_ops(&ops, &[]);
    }
}
