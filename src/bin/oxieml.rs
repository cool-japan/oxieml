//! OxiEML CLI — Parse, evaluate, and generate EML expressions.
//!
//! Usage:
//!   oxieml "E(1, 1)"                     # Evaluate EML expression
//!   oxieml -g pi                          # Generate EML for π
//!   oxieml -g "sin(x0)" x0=0.5           # Generate & evaluate sin
//!   oxieml --file expression.txt          # Read from file
//!   echo "E(1, 1)" | oxieml              # Read from stdin

use oxieml::canonical::Canonical;
use oxieml::eval::EvalCtx;
use oxieml::parser::{parse, to_compact_string};
use oxieml::tree::EmlTree;
use std::io::IsTerminal;
use std::io::Read;

/// Known mathematical constants to check against.
const KNOWN_CONSTANTS: &[(&str, f64)] = &[
    ("e (Euler's number)", std::f64::consts::E),
    ("pi", std::f64::consts::PI),
    ("tau (2*pi)", std::f64::consts::TAU),
    ("ln(2)", std::f64::consts::LN_2),
    ("ln(10)", std::f64::consts::LN_10),
    ("sqrt(2)", std::f64::consts::SQRT_2),
    ("1/sqrt(2)", std::f64::consts::FRAC_1_SQRT_2),
    ("1/pi", std::f64::consts::FRAC_1_PI),
    ("2/pi", std::f64::consts::FRAC_2_PI),
    ("2/sqrt(pi)", std::f64::consts::FRAC_2_SQRT_PI),
    ("pi/2", std::f64::consts::FRAC_PI_2),
    ("pi/3", std::f64::consts::FRAC_PI_3),
    ("pi/4", std::f64::consts::FRAC_PI_4),
    ("pi/6", std::f64::consts::FRAC_PI_6),
    ("pi/8", std::f64::consts::FRAC_PI_8),
    ("log2(e)", std::f64::consts::LOG2_E),
    ("log10(e)", std::f64::consts::LOG10_E),
    ("golden ratio (phi)", 1.618_033_988_749_895),
    ("0", 0.0),
    ("1", 1.0),
    ("2", 2.0),
    ("3", 3.0),
    ("-1", -1.0),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --help / -h
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    // --version / -V
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("oxieml {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // Check for --gen / -g flag (generate mode)
    if let Some(pos) = args.iter().position(|a| a == "--gen" || a == "-g") {
        let expr = args.get(pos + 1).unwrap_or_else(|| {
            eprintln!("Error: --gen requires a function/constant name");
            print_usage();
            std::process::exit(1);
        });
        let vars = parse_var_assignments(&args);
        run_generate(expr, &vars);
        return;
    }

    // Check for --list / -l flag (list all known functions)
    if args.iter().any(|a| a == "--list" || a == "-l") {
        print_known_functions();
        return;
    }

    let input = match get_input(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            std::process::exit(1);
        }
    };

    let input = input.trim();
    if input.is_empty() {
        eprintln!("Error: empty input");
        print_usage();
        std::process::exit(1);
    }

    // Try EML parse first; if it fails, try as a generate request
    match parse(input) {
        Ok(tree) => {
            let vars = parse_var_assignments(&args);
            run_evaluate(&tree, input, &vars);
        }
        Err(parse_err) => {
            // Maybe the user typed a function name like "pi" or "sin(x0)"
            let vars = parse_var_assignments(&args);
            if try_generate(input).is_some() {
                run_generate(input, &vars);
            } else {
                eprintln!("Parse error: {parse_err}");
                eprintln!();
                eprintln!("Hint: Use -g to generate EML from a function name:");
                eprintln!("  oxieml -g pi");
                eprintln!("  oxieml -g \"sin(x0)\"");
                std::process::exit(1);
            }
        }
    }
}

// ================================================================
// Generate mode: function/constant name → EML expression
// ================================================================

fn run_generate(expr: &str, vars: &[f64]) {
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
fn try_generate(expr: &str) -> Option<EmlTree> {
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

fn print_known_functions() {
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

// ================================================================
// Evaluate mode: EML expression → result
// ================================================================

fn run_evaluate(tree: &EmlTree, input: &str, vars: &[f64]) {
    println!("=== OxiEML Expression Evaluator ===\n");

    if input.len() > 200 {
        println!("Input: {}... ({} chars)", &input[..200], input.len());
    } else {
        println!("Input: {input}");
    }
    println!();

    println!("--- Tree Statistics ---");
    println!("  Depth: {}", tree.depth());
    println!("  Size (nodes): {}", tree.size());
    println!("  Variables used: {}", count_variables(tree));
    println!();

    let compact = to_compact_string(tree);
    if compact.len() <= 200 {
        println!("Compact: {compact}");
        println!();
    }

    let ctx = EvalCtx::new(vars);
    println!("--- Real Evaluation ---");
    if !vars.is_empty() {
        print!("  Variables: ");
        for (i, v) in vars.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("x{i} = {v}");
        }
        println!();
    }

    match tree.eval_real(&ctx) {
        Ok(val) => {
            println!("  Result: {val}");
            println!("  Result (full precision): {val:.17e}");
            println!();
            check_known_constants(val);
        }
        Err(e) => {
            println!("  Real evaluation failed: {e}");
            println!();
        }
    }

    println!("--- Complex Evaluation ---");
    let complex_vars: Vec<num_complex::Complex64> = vars
        .iter()
        .map(|&v| num_complex::Complex64::new(v, 0.0))
        .collect();

    match tree.eval_complex(&complex_vars) {
        Ok(z) => {
            println!("  Result: {} + {}i", z.re, z.im);
            println!("  |z| = {}", z.norm());
            println!("  arg(z) = {} rad", z.arg());
            println!();

            if z.im.abs() > 1e-10 {
                println!("  --- Imaginary part analysis ---");
                check_known_constants_labeled("  Im(result)", z.im);
                if z.re.abs() > 1e-10 {
                    check_known_constants_labeled("  Re(result)", z.re);
                }
            }
        }
        Err(e) => {
            println!("  Complex evaluation failed: {e}");
        }
    }

    println!();
    println!("--- Lowered Form ---");
    let lowered = tree.lower();
    let lowered_str = format!("{lowered}");
    if lowered_str.len() <= 500 {
        println!("  {lowered_str}");
    } else {
        println!(
            "  (expression too large to display, {} chars)",
            lowered_str.len()
        );
    }

    println!();
    println!("--- Lowered Evaluation ---");
    let lowered_val = lowered.eval(vars);
    println!("  Result: {lowered_val}");
    println!("  Result (full precision): {lowered_val:.17e}");
}

// ================================================================
// Input handling
// ================================================================

fn get_input(args: &[String]) -> Result<String, String> {
    if let Some(pos) = args.iter().position(|a| a == "--file" || a == "-f") {
        let path = args.get(pos + 1).ok_or("--file requires a path argument")?;
        return std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read file '{path}': {e}"));
    }

    for arg in args.iter().skip(1) {
        if !arg.contains('=') && !arg.starts_with('-') {
            return Ok(arg.clone());
        }
    }

    if std::io::stdin().is_terminal() {
        return Err("no expression provided".to_string());
    }

    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    Ok(buf)
}

fn parse_var_assignments(args: &[String]) -> Vec<f64> {
    let mut vars: Vec<(usize, f64)> = Vec::new();

    for arg in args.iter().skip(1) {
        if let Some(eq_pos) = arg.find('=') {
            let name = &arg[..eq_pos];
            let val_str = &arg[eq_pos + 1..];
            if let Some(idx_str) = name.strip_prefix('x') {
                if let (Ok(idx), Ok(val)) = (idx_str.parse::<usize>(), val_str.parse::<f64>()) {
                    vars.push((idx, val));
                }
            }
        }
    }

    if vars.is_empty() {
        return Vec::new();
    }

    let max_idx = vars.iter().map(|(i, _)| *i).max().unwrap_or(0);
    let mut result = vec![0.0; max_idx + 1];
    for (idx, val) in vars {
        result[idx] = val;
    }
    result
}

fn count_variables(tree: &oxieml::EmlTree) -> usize {
    let mut max_var: Option<usize> = None;
    for node in tree.iter_postorder() {
        if let oxieml::EmlNode::Var(idx) = node {
            match max_var {
                None => max_var = Some(*idx),
                Some(m) if *idx > m => max_var = Some(*idx),
                _ => {}
            }
        }
    }
    match max_var {
        None => 0,
        Some(m) => m + 1,
    }
}

fn check_known_constants(val: f64) {
    println!("  --- Constant matching ---");
    let mut found = false;

    for &(name, constant) in KNOWN_CONSTANTS {
        let diff = (val - constant).abs();
        if diff < 1e-10 {
            println!("  MATCH: {name} = {constant}");
            println!("         difference = {diff:.2e}");
            found = true;
        } else if diff < 1e-4 {
            println!("  CLOSE: {name} = {constant}");
            println!("         difference = {diff:.2e}");
            found = true;
        }
    }

    for &(name, constant) in KNOWN_CONSTANTS {
        if constant == 0.0 {
            continue;
        }
        let diff = (val - (-constant)).abs();
        if diff < 1e-10 {
            println!("  MATCH: -{name} = {}", -constant);
            println!("         difference = {diff:.2e}");
            found = true;
        }
    }

    for n in 2..=10 {
        let n_f = n as f64;
        let diff = (val - n_f).abs();
        if diff < 1e-10 {
            println!("  MATCH: {n}");
            println!("         difference = {diff:.2e}");
            found = true;
        }
    }

    if !found {
        println!("  No known constant matches found.");
    }
}

fn check_known_constants_labeled(label: &str, val: f64) {
    for &(name, constant) in KNOWN_CONSTANTS {
        let diff = (val - constant).abs();
        if diff < 1e-4 {
            let quality = if diff < 1e-10 { "MATCH" } else { "CLOSE" };
            println!("  {quality}: {label} ~ {name} (diff = {diff:.2e})");
        }
        if constant != 0.0 {
            let diff_neg = (val - (-constant)).abs();
            if diff_neg < 1e-4 {
                let quality = if diff_neg < 1e-10 { "MATCH" } else { "CLOSE" };
                println!("  {quality}: {label} ~ -{name} (diff = {diff_neg:.2e})");
            }
        }
    }
}

fn print_usage() {
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  oxieml \"E(1, 1)\"                     # Evaluate EML expression");
    eprintln!("  oxieml \"E(x0, 1)\" x0=2.0             # With variable bindings");
    eprintln!("  oxieml -g pi                           # Generate EML for π");
    eprintln!("  oxieml -g sin                          # Generate EML for sin(x0)");
    eprintln!("  oxieml -g \"sin(x0)\" x0=0.5            # Generate & evaluate");
    eprintln!("  oxieml -l                              # List all functions");
    eprintln!("  oxieml --help                          # Show this help");
    eprintln!("  oxieml --version                       # Show version");
    eprintln!("  oxieml --file expression.txt           # Read from file");
    eprintln!("  echo \"E(1, 1)\" | oxieml               # Read from stdin");
    eprintln!();
    eprintln!("Notation:");
    eprintln!("  1         The constant 1");
    eprintln!("  x0, x1    Variables");
    eprintln!("  E(a, b)   The EML operator: exp(a) - ln(b)");
    eprintln!("  eml(a, b) Alternative notation for E(a, b)");
}
