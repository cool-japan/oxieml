use super::format::KNOWN_CONSTANTS;
use oxieml::{EmlNode, EmlTree};
use std::io::IsTerminal;
use std::io::Read;

pub(super) fn get_input(args: &[String]) -> Result<String, String> {
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

pub(super) fn parse_var_assignments(args: &[String]) -> Vec<f64> {
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

/// Parse `xN=value` tokens out of clap's trailing-var-arg positionals.
///
/// clap subcommands collect trailing `x0=1.5`-style tokens into a plain
/// `Vec<String>` with no leading program-name placeholder. This wraps that
/// slice with a dummy element 0 so [`parse_var_assignments`] (written against
/// raw `std::env::args()`, which always has the binary path at index 0) can be
/// reused verbatim for both the legacy shim and the new clap subcommands.
pub(super) fn vars_from_extra(extra: &[String]) -> Vec<f64> {
    let mut synthetic = Vec::with_capacity(extra.len() + 1);
    synthetic.push(String::new());
    synthetic.extend_from_slice(extra);
    parse_var_assignments(&synthetic)
}

/// Pad a variable-binding vector with trailing zeros so it has at least
/// `needed` elements.
///
/// [`oxieml::LoweredOp::eval`] indexes `vars[i]` directly with no bounds
/// check, and several `LoweredOp` combinators (`integrate_definite`,
/// `solve_numeric`) only pad relative to a single `wrt`/`var` index, not the
/// full variable count of an arbitrary expression. Subcommands that build an
/// `EvalCtx` from user-supplied `xN=value` bindings call this first so a
/// sparsely-bound multi-variable expression cannot panic the binary.
pub(super) fn pad_vars(vars: &[f64], needed: usize) -> Vec<f64> {
    if vars.len() >= needed {
        return vars.to_vec();
    }
    let mut padded = vars.to_vec();
    padded.resize(needed, 0.0);
    padded
}

pub(super) fn count_variables(tree: &EmlTree) -> usize {
    let mut max_var: Option<usize> = None;
    for node in tree.iter_postorder() {
        if let EmlNode::Var(idx) = node {
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

pub(super) fn check_known_constants(val: f64) {
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

pub(super) fn check_known_constants_labeled(label: &str, val: f64) {
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
