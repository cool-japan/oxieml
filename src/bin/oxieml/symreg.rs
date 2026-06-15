use super::format::{OutputFormat, json_escape_str, write_output};
use std::io::IsTerminal;
use std::io::Read;

pub(super) fn run_symreg(
    args: &[String],
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use oxieml::symreg::{SymRegConfig, SymRegEngine};

    // Required: --vars N
    let num_vars = parse_named_usize(args, "--vars")?
        .ok_or_else(|| "--symreg requires --vars <N> (N >= 1)".to_string())?;
    if num_vars == 0 {
        return Err("--vars must be at least 1".into());
    }

    // Optional: --top K
    let top_k = parse_named_usize(args, "--top")?.unwrap_or(3);
    if top_k == 0 {
        return Err("--top must be at least 1".into());
    }

    // Build SymRegConfig, forwarding optional flags.
    let mut config = SymRegConfig::default();
    if let Some(v) = parse_named_usize(args, "--max-depth")? {
        config.max_depth = v;
    }
    if let Some(v) = parse_named_usize(args, "--max-iter")? {
        config.max_iter = v;
    }
    if let Some(v) = parse_named_f64(args, "--learning-rate")? {
        config.learning_rate = v;
    }
    if let Some(v) = parse_named_f64(args, "--tolerance")? {
        config.tolerance = v;
    }
    if let Some(v) = parse_named_f64(args, "--complexity-penalty")? {
        config.complexity_penalty = v;
    }
    if let Some(v) = parse_named_usize(args, "--num-restarts")? {
        config.num_restarts = v;
    }

    // Optional: --strategy exhaustive | beam:<N>
    if let Some(pos) = args.iter().position(|a| a == "--strategy") {
        let val = args
            .get(pos + 1)
            .ok_or("--strategy requires a value: exhaustive or beam:<N>")?;
        config.strategy = parse_strategy(val)?;
    }

    // Read dataset text from --file or stdin.
    let text = get_symreg_data(args)?;
    let (inputs, targets) = parse_dataset(&text, num_vars)?;
    if inputs.is_empty() {
        return Err("no data: dataset is empty".into());
    }

    let engine = SymRegEngine::new(config);
    let formulas = engine
        .discover(&inputs, &targets, num_vars)
        .map_err(|e| format!("symreg failed: {e}"))?;

    if formulas.is_empty() {
        return Err("no formulas discovered".into());
    }

    let limit = top_k.min(formulas.len());

    let content = format_symreg_results(&formulas[..limit], fmt);
    write_output(&content, out)
}

/// Format the top-K discovered formulas according to the requested output format.
fn format_symreg_results(formulas: &[oxieml::DiscoveredFormula], fmt: &OutputFormat) -> String {
    match fmt {
        OutputFormat::Pretty => {
            let mut buf = String::new();
            for (i, f) in formulas.iter().enumerate() {
                buf.push_str(&format!(
                    "Rank {}: {}   mse={:.4}   complexity={}   score={:.4}\n",
                    i + 1,
                    f.pretty,
                    f.mse,
                    f.complexity,
                    f.score
                ));
            }
            buf
        }
        OutputFormat::Latex => {
            let mut buf = String::new();
            for (i, f) in formulas.iter().enumerate() {
                let latex = f.to_latex();
                buf.push_str(&format!(
                    "Rank {}: $${}$$   mse={:.4}   complexity={}\n",
                    i + 1,
                    latex,
                    f.mse,
                    f.complexity
                ));
            }
            buf
        }
        OutputFormat::Json => {
            // Hand-rolled JSON — no serde.
            let mut buf = String::new();
            buf.push_str("{\"version\":1,\"formulas\":[");
            for (i, f) in formulas.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                let pretty_escaped = json_escape_str(&f.pretty);
                let latex = f.to_latex();
                let latex_escaped = json_escape_str(&latex);
                buf.push_str(&format!(
                    "{{\"rank\":{rank},\"mse\":{mse},\"complexity\":{complexity},\"score\":{score},\"pretty\":\"{pretty_escaped}\",\"latex\":\"{latex_escaped}\"}}",
                    rank = i + 1,
                    mse = f.mse,
                    complexity = f.complexity,
                    score = f.score,
                ));
            }
            buf.push_str("]}\n");
            buf
        }
    }
}

/// Parse whitespace-separated numeric dataset text.
///
/// - Lines starting with `#` or blank lines are skipped.
/// - Every other line must contain exactly `num_vars + 1` f64 values.
/// - First `num_vars` values become the input row; the last is the target.
fn parse_dataset(text: &str, num_vars: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();
    let expected = num_vars + 1;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|tok| {
                tok.parse::<f64>()
                    .map_err(|_| format!("line {}: invalid number '{}'", lineno + 1, tok))
            })
            .collect::<Result<Vec<f64>, String>>()?;
        if values.len() != expected {
            return Err(format!(
                "line {}: expected {} floats ({} vars + 1 target), got {}",
                lineno + 1,
                expected,
                num_vars,
                values.len()
            ));
        }
        let target = values[num_vars];
        let row: Vec<f64> = values[..num_vars].to_vec();
        inputs.push(row);
        targets.push(target);
    }

    Ok((inputs, targets))
}

/// Read symreg dataset text from `--file <path>` or stdin (no positional fallback).
fn get_symreg_data(args: &[String]) -> Result<String, String> {
    if let Some(pos) = args.iter().position(|a| a == "--file" || a == "-f") {
        let path = args.get(pos + 1).ok_or("--file requires a path argument")?;
        return std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read file '{path}': {e}"));
    }

    if std::io::stdin().is_terminal() {
        return Err("no data: provide a dataset via --file <path> or stdin".to_string());
    }

    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    Ok(buf)
}

/// Look up `--name <value>` and parse as `usize`. Returns `None` if the flag is absent.
fn parse_named_usize(args: &[String], name: &str) -> Result<Option<usize>, String> {
    let Some(pos) = args.iter().position(|a| a == name) else {
        return Ok(None);
    };
    let val = args
        .get(pos + 1)
        .ok_or_else(|| format!("{name} requires a value"))?;
    val.parse::<usize>()
        .map(Some)
        .map_err(|_| format!("{name}: expected non-negative integer, got '{val}'"))
}

/// Parse `--strategy` value: `"exhaustive"` or `"beam:<N>"`.
fn parse_strategy(val: &str) -> Result<oxieml::symreg::SymRegStrategy, String> {
    use oxieml::symreg::SymRegStrategy;
    if val == "exhaustive" {
        return Ok(SymRegStrategy::Exhaustive);
    }
    if let Some(n_str) = val.strip_prefix("beam:") {
        let width = n_str.parse::<usize>().map_err(|_| {
            format!("--strategy beam:<N>: expected positive integer, got '{n_str}'")
        })?;
        if width == 0 {
            return Err("--strategy beam:<N>: N must be at least 1".to_string());
        }
        return Ok(SymRegStrategy::Beam { width });
    }
    Err(format!(
        "--strategy: unknown value '{val}'; expected 'exhaustive' or 'beam:<N>' (e.g. beam:10)"
    ))
}

/// Look up `--name <value>` and parse as `f64`. Returns `None` if the flag is absent.
fn parse_named_f64(args: &[String], name: &str) -> Result<Option<f64>, String> {
    let Some(pos) = args.iter().position(|a| a == name) else {
        return Ok(None);
    };
    let val = args
        .get(pos + 1)
        .ok_or_else(|| format!("{name} requires a value"))?;
    val.parse::<f64>()
        .map(Some)
        .map_err(|_| format!("{name}: expected floating-point number, got '{val}'"))
}
