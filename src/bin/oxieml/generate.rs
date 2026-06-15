use super::args::{check_known_constants, check_known_constants_labeled, count_variables};
use oxieml::canonical::Canonical;
use oxieml::eval::EvalCtx;
use oxieml::parser::to_compact_string;
use oxieml::tree::EmlTree;

pub(super) fn run_generate(expr: &str, vars: &[f64]) {
    let tree = match try_generate(expr) {
        Some(t) => t,
        None => {
            eprintln!("Unknown function or constant: \"{expr}\"");
            eprintln!();
            eprintln!("Use --list to see all available functions.");
            std::process::exit(1);
        }
    };

    let compact = to_compact_string(&tree);

    println!("=== OxiEML Generator ===\n");
    println!("Function: {expr}");
    println!("Depth:    {}", tree.depth());
    println!("Size:     {} nodes", tree.size());
    println!();
    println!("EML expression:");
    println!("{compact}");
    println!();

    // Also show the eml(...) notation
    let display = format!("{tree}");
    if display.len() <= 500 {
        println!("eml notation:");
        println!("{display}");
        println!();
    }

    // Evaluate if no variables or variables are provided
    let num_vars = count_variables(&tree);
    if num_vars == 0 {
        // Constant — evaluate directly
        let ctx = EvalCtx::new(&[]);
        println!("--- Evaluation ---");
        match tree.eval_real(&ctx) {
            Ok(val) => {
                println!("  Result: {val}");
                println!("  Result (full precision): {val:.17e}");
                println!();
                check_known_constants(val);
            }
            Err(_) => {
                // Try complex
                match tree.eval_complex(&[]) {
                    Ok(z) => {
                        println!("  Complex result: {} + {}i", z.re, z.im);
                        if z.im.abs() > 1e-10 {
                            check_known_constants_labeled("  Im", z.im);
                        }
                        if z.re.abs() > 1e-10 {
                            check_known_constants_labeled("  Re", z.re);
                        }
                    }
                    Err(e) => println!("  Evaluation failed: {e}"),
                }
            }
        }
    } else if !vars.is_empty() {
        // Variables provided — evaluate
        let ctx = EvalCtx::new(vars);
        println!("--- Evaluation ---");
        print!("  Variables: ");
        for (i, v) in vars.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("x{i} = {v}");
        }
        println!();
        match tree.eval_real(&ctx) {
            Ok(val) => {
                println!("  Result: {val}");
                println!("  Result (full precision): {val:.17e}");
                println!();
                check_known_constants(val);
            }
            Err(e) => println!("  Evaluation failed: {e}"),
        }
    } else {
        println!("(Provide variable values to evaluate, e.g., x0=1.5)");
    }
}

/// Try to parse a function/constant name and build the corresponding EML tree.
pub(super) fn try_generate(expr: &str) -> Option<EmlTree> {
    let expr = expr.trim();

    // Constants (no arguments)
    match expr {
        "pi" | "π" => return Some(Canonical::pi()),
        "e" | "euler" => return Some(Canonical::euler()),
        "0" | "zero" => return Some(Canonical::zero()),
        "i" | "imag" => return Some(Canonical::imag_unit()),
        "-1" | "neg_one" => return Some(Canonical::neg_one()),
        "-2" | "neg_two" => return Some(Canonical::neg_two()),
        _ => {}
    }

    // nat(N) — natural number
    if let Some(inner) = strip_func(expr, "nat") {
        if let Ok(n) = inner.parse::<u64>() {
            if n >= 1 {
                return Some(Canonical::nat(n));
            }
        }
        return None;
    }

    // Unary functions: func(arg)
    // First try to extract (func_name, arg_string)
    if let Some((func, arg_str)) = parse_func_call(expr) {
        let arg = parse_arg(arg_str)?;
        return match func {
            "exp" => Some(Canonical::exp(&arg)),
            "ln" | "log" => Some(Canonical::ln(&arg)),
            "neg" => Some(Canonical::neg(&arg)),
            "sin" => Some(Canonical::sin(&arg)),
            "cos" => Some(Canonical::cos(&arg)),
            "tan" => Some(Canonical::tan(&arg)),
            "arcsin" | "asin" => Some(Canonical::arcsin(&arg)),
            "arccos" | "acos" => Some(Canonical::arccos(&arg)),
            "arctan" | "atan" => Some(Canonical::arctan(&arg)),
            "sinh" => Some(Canonical::sinh(&arg)),
            "cosh" => Some(Canonical::cosh(&arg)),
            "tanh" => Some(Canonical::tanh(&arg)),
            "arcsinh" | "asinh" => Some(Canonical::arcsinh(&arg)),
            "arccosh" | "acosh" => Some(Canonical::arccosh(&arg)),
            "arctanh" | "atanh" => Some(Canonical::arctanh(&arg)),
            "sqrt" | "√" => Some(Canonical::sqrt(&arg)),
            "abs" => Some(Canonical::abs(&arg)),
            "square" => Some(Canonical::square(&arg)),
            "reciprocal" | "inv" => Some(Canonical::reciprocal(&arg)),
            _ => None,
        };
    }

    // Bare function name → default to x0 as argument
    let x0 = EmlTree::var(0);
    match expr {
        "exp" => Some(Canonical::exp(&x0)),
        "ln" | "log" => Some(Canonical::ln(&x0)),
        "neg" => Some(Canonical::neg(&x0)),
        "sin" => Some(Canonical::sin(&x0)),
        "cos" => Some(Canonical::cos(&x0)),
        "tan" => Some(Canonical::tan(&x0)),
        "arcsin" | "asin" => Some(Canonical::arcsin(&x0)),
        "arccos" | "acos" => Some(Canonical::arccos(&x0)),
        "arctan" | "atan" => Some(Canonical::arctan(&x0)),
        "sinh" => Some(Canonical::sinh(&x0)),
        "cosh" => Some(Canonical::cosh(&x0)),
        "tanh" => Some(Canonical::tanh(&x0)),
        "arcsinh" | "asinh" => Some(Canonical::arcsinh(&x0)),
        "arccosh" | "acosh" => Some(Canonical::arccosh(&x0)),
        "arctanh" | "atanh" => Some(Canonical::arctanh(&x0)),
        "sqrt" => Some(Canonical::sqrt(&x0)),
        "abs" => Some(Canonical::abs(&x0)),
        "square" => Some(Canonical::square(&x0)),
        "reciprocal" | "inv" => Some(Canonical::reciprocal(&x0)),
        _ => {
            // Try binary: "add", "sub", etc. with default x0, x1
            let x1 = EmlTree::var(1);
            match expr {
                "add" => Some(Canonical::add(&x0, &x1)),
                "sub" => Some(Canonical::sub(&x0, &x1)),
                "mul" => Some(Canonical::mul(&x0, &x1)),
                "div" => Some(Canonical::div(&x0, &x1)),
                "pow" => Some(Canonical::pow(&x0, &x1)),
                _ => None,
            }
        }
    }
}

/// Parse "func(args)" → ("func", "args")
fn parse_func_call(expr: &str) -> Option<(&str, &str)> {
    let open = expr.find('(')?;
    if !expr.ends_with(')') {
        return None;
    }
    let func = expr[..open].trim();
    let inner = &expr[open + 1..expr.len() - 1];
    Some((func, inner.trim()))
}

/// Parse a function argument: "x0", "x1", "1", "e", "pi", or nested function
fn parse_arg(s: &str) -> Option<EmlTree> {
    let s = s.trim();

    // Variable: x0, x1, ...
    if let Some(idx_str) = s.strip_prefix('x') {
        if let Ok(idx) = idx_str.parse::<usize>() {
            return Some(EmlTree::var(idx));
        }
    }

    // Constant
    match s {
        "1" => return Some(EmlTree::one()),
        "e" | "euler" => return Some(Canonical::euler()),
        "pi" | "π" => return Some(Canonical::pi()),
        "0" | "zero" => return Some(Canonical::zero()),
        _ => {}
    }

    // Number literal
    if let Ok(n) = s.parse::<u64>() {
        if n >= 1 {
            return Some(Canonical::nat(n));
        }
    }

    // Nested function call
    if let Some((func, inner)) = parse_func_call(s) {
        let inner_arg = parse_arg(inner)?;
        return match func {
            "exp" => Some(Canonical::exp(&inner_arg)),
            "ln" | "log" => Some(Canonical::ln(&inner_arg)),
            "neg" => Some(Canonical::neg(&inner_arg)),
            "sin" => Some(Canonical::sin(&inner_arg)),
            "cos" => Some(Canonical::cos(&inner_arg)),
            "tan" => Some(Canonical::tan(&inner_arg)),
            "sqrt" => Some(Canonical::sqrt(&inner_arg)),
            "square" => Some(Canonical::square(&inner_arg)),
            _ => None,
        };
    }

    None
}

/// Extract inner string from "func(inner)"
fn strip_func<'a>(expr: &'a str, func: &str) -> Option<&'a str> {
    let rest = expr.strip_prefix(func)?;
    let rest = rest.strip_prefix('(')?;
    let rest = rest.strip_suffix(')')?;
    Some(rest.trim())
}

pub(super) fn print_known_functions() {
    println!("=== Available Functions & Constants ===\n");
    println!("Constants:");
    println!("  pi, π          iπ (use in trig constructions)");
    println!("  e, euler       Euler's number (2.71828...)");
    println!("  0, zero        Zero = ln(1)");
    println!("  -1, neg_one    Negative one");
    println!("  -2, neg_two    Negative two");
    println!("  i, imag        Imaginary unit = exp(iπ/2)");
    println!("  nat(N)         Natural number N (1, 2, 3, ...)");
    println!();
    println!("Unary functions (default arg: x0):");
    println!("  exp             exp(x) = eml(x, 1)");
    println!("  ln, log         ln(x)");
    println!("  neg             -x");
    println!("  sqrt            √x");
    println!("  square          x²");
    println!("  abs             |x|");
    println!("  reciprocal, inv 1/x");
    println!();
    println!("Trigonometric:");
    println!("  sin, cos, tan");
    println!("  arcsin/asin, arccos/acos, arctan/atan");
    println!();
    println!("Hyperbolic:");
    println!("  sinh, cosh, tanh");
    println!("  arcsinh/asinh, arccosh/acosh, arctanh/atanh");
    println!();
    println!("Binary functions (default args: x0, x1):");
    println!("  add             x + y");
    println!("  sub             x - y");
    println!("  mul             x * y");
    println!("  div             x / y");
    println!("  pow             x ^ y");
    println!();
    println!("Examples:");
    println!("  oxieml -g pi");
    println!("  oxieml -g e");
    println!("  oxieml -g sin             # sin(x0) template");
    println!("  oxieml -g \"sin(x0)\" x0=0.5");
    println!("  oxieml -g \"exp(x0)\" x0=1.0");
    println!("  oxieml -g \"sqrt(x0)\" x0=4.0");
    println!("  oxieml -g nat(5)");
}
