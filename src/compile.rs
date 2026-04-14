//! Compile EML trees to Rust source code.
//!
//! Generates standalone Rust functions from EML trees (via lowering),
//! enabling zero-overhead evaluation of discovered formulas.

use crate::lower::LoweredOp;
use crate::tree::EmlTree;

/// Compile an EML tree into a Rust function source code string.
///
/// The generated function takes a slice of `f64` variables and returns `f64`.
///
/// # Example output
/// ```text
/// fn discovered_formula(vars: &[f64]) -> f64 {
///     let x0 = vars[0];
///     let x1 = vars[1];
///     (x0 * x1) + x0.exp()
/// }
/// ```
pub fn compile_to_rust(tree: &EmlTree, fn_name: &str) -> String {
    let lowered = tree.lower();
    let simplified = lowered.simplify();
    compile_lowered_to_rust(&simplified, fn_name, tree.num_vars())
}

/// Compile a `LoweredOp` tree into a Rust function source code string.
pub fn compile_lowered_to_rust(op: &LoweredOp, fn_name: &str, num_vars: usize) -> String {
    let mut code = String::new();

    code.push_str(&format!("fn {fn_name}(vars: &[f64]) -> f64 {{\n"));

    // Bind variables
    for i in 0..num_vars {
        code.push_str(&format!("    let x{i} = vars[{i}];\n"));
    }

    if num_vars > 0 {
        code.push('\n');
    }

    // Generate expression
    let expr = emit_rust_expr(op);
    code.push_str(&format!("    {expr}\n"));
    code.push_str("}\n");

    code
}

/// Compile an EML tree into a Rust closure source code string.
pub fn compile_to_closure(tree: &EmlTree) -> String {
    let lowered = tree.lower();
    let simplified = lowered.simplify();
    let num_vars = tree.num_vars();

    let mut code = String::from("|vars: &[f64]| -> f64 {\n");

    for i in 0..num_vars {
        code.push_str(&format!("    let x{i} = vars[{i}];\n"));
    }

    if num_vars > 0 {
        code.push('\n');
    }

    let expr = emit_rust_expr(&simplified);
    code.push_str(&format!("    {expr}\n"));
    code.push('}');

    code
}

/// Compile an EML tree into a pair of Rust functions: a single-point evaluator
/// and a batch evaluator.
///
/// The batch function signature is:
/// ```text
/// fn {fn_name}_batch(data: &[Vec<f64>]) -> Vec<f64>
/// ```
///
/// When oxieml is compiled with the `parallel` feature, the generated batch
/// function uses `rayon::prelude::*` for parallel evaluation. Otherwise it uses
/// a sequential iterator.
///
/// # Note
/// The generated batch function body references `{fn_name}` (the single-point
/// function) which must be in scope when the generated code is compiled.
pub fn compile_to_rust_batch(tree: &EmlTree, fn_name: &str) -> String {
    let single_point = compile_to_rust(tree, fn_name);

    let batch_body = if cfg!(feature = "parallel") {
        format!(
            "fn {fn_name}_batch(data: &[Vec<f64>]) -> Vec<f64> {{\n\
             use rayon::prelude::*;\n\
             data.par_iter().map(|pt| {fn_name}(pt)).collect()\n\
             }}\n"
        )
    } else {
        format!(
            "fn {fn_name}_batch(data: &[Vec<f64>]) -> Vec<f64> {{\n\
             data.iter().map(|pt| {fn_name}(pt)).collect()\n\
             }}\n"
        )
    };

    format!("{single_point}\n{batch_body}")
}

/// Emit a Rust expression string for a `LoweredOp`.
fn emit_rust_expr(op: &LoweredOp) -> String {
    match op {
        LoweredOp::Const(c) => {
            if (c - std::f64::consts::E).abs() < 1e-15 {
                "std::f64::consts::E".to_string()
            } else if (c - std::f64::consts::PI).abs() < 1e-15 {
                "std::f64::consts::PI".to_string()
            } else if (c - c.round()).abs() < 1e-10 && c.abs() < 1e15 {
                format!("{}_f64", *c as i64)
            } else {
                format!("{c:.15e}_f64")
            }
        }
        LoweredOp::Var(i) => format!("x{i}"),
        LoweredOp::Add(a, b) => {
            format!("({} + {})", emit_rust_expr(a), emit_rust_expr(b))
        }
        LoweredOp::Sub(a, b) => {
            format!("({} - {})", emit_rust_expr(a), emit_rust_expr(b))
        }
        LoweredOp::Mul(a, b) => {
            format!("({} * {})", emit_rust_expr(a), emit_rust_expr(b))
        }
        LoweredOp::Div(a, b) => {
            format!("({} / {})", emit_rust_expr(a), emit_rust_expr(b))
        }
        LoweredOp::Exp(a) => {
            format!("({}).exp()", emit_rust_expr(a))
        }
        LoweredOp::Ln(a) => {
            format!("({}).ln()", emit_rust_expr(a))
        }
        LoweredOp::Sin(a) => {
            format!("({}).sin()", emit_rust_expr(a))
        }
        LoweredOp::Cos(a) => {
            format!("({}).cos()", emit_rust_expr(a))
        }
        LoweredOp::Pow(a, b) => {
            format!("({}).powf({})", emit_rust_expr(a), emit_rust_expr(b))
        }
        LoweredOp::Neg(a) => {
            format!("(-({}))", emit_rust_expr(a))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_exp() {
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let code = compile_to_rust(&exp_x, "exp_fn");
        assert!(code.contains("fn exp_fn"));
        assert!(code.contains("x0"));
        assert!(code.contains(".exp()"));
    }

    #[test]
    fn test_compile_euler() {
        let one = EmlTree::one();
        let e = EmlTree::eml(&one, &one);
        let code = compile_to_rust(&e, "euler_fn");
        assert!(code.contains("fn euler_fn"));
        // The lowered form should contain E constant
        assert!(code.contains("E") || code.contains("exp"));
    }

    #[test]
    fn test_compile_closure() {
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let code = compile_to_closure(&exp_x);
        assert!(code.contains("|vars: &[f64]| -> f64"));
    }

    #[test]
    fn test_compile_no_vars() {
        let one = EmlTree::one();
        let code = compile_to_rust(&one, "const_fn");
        assert!(code.contains("fn const_fn"));
        assert!(!code.contains("let x"));
    }

    #[test]
    fn test_compile_to_rust_batch() {
        let x = EmlTree::var(0);
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(&x, &one);
        let code = compile_to_rust_batch(&exp_x, "exp_fn");

        // Single-point function is present
        assert!(code.contains("fn exp_fn(vars: &[f64]) -> f64"));
        // Batch function is present
        assert!(code.contains("fn exp_fn_batch(data: &[Vec<f64>]) -> Vec<f64>"));
        // Uses collect()
        assert!(code.contains(".collect()"));

        // When parallel feature is active, the batch body uses par_iter
        #[cfg(feature = "parallel")]
        assert!(code.contains("par_iter"));
        // When parallel feature is inactive, the batch body uses iter
        #[cfg(not(feature = "parallel"))]
        assert!(code.contains("data.iter()"));
    }
}
