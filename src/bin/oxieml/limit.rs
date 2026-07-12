//! `oxieml limit` — one-sided/two-sided limit computation.

use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::parser::parse;
use oxieml::{LimitPoint, LimitResult};

/// Parse a limit point argument: `inf`/`+inf`/`infinity` for `+∞`,
/// `-inf`/`-infinity` for `-∞`, or any finite `f64` literal.
fn parse_limit_point(s: &str) -> Result<LimitPoint, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Ok(LimitPoint::PosInf),
        "-inf" | "-infinity" => Ok(LimitPoint::NegInf),
        other => other
            .parse::<f64>()
            .map(LimitPoint::Finite)
            .map_err(|_| format!("--point: expected 'inf', '-inf', or a finite number, got '{s}'")),
    }
}

/// Run the `limit` subcommand: `lim_{x_wrt -> point} expr`, via
/// [`oxieml::LoweredOp::limit`]. Other variables in `expr` are held at `0`
/// (per `LoweredOp::limit`'s own contract — it takes no external bindings).
pub(super) fn run_limit(
    expr_str: &str,
    wrt: usize,
    point_str: &str,
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let point = parse_limit_point(point_str)?;
    let tree = parse(expr_str).map_err(|e| format!("parse error: {e}"))?;
    let lowered = tree.lower().simplify();

    let result = lowered.limit(wrt, point);

    let (kind, value): (&str, Option<f64>) = match result {
        LimitResult::Finite(v) => ("finite", Some(v)),
        LimitResult::PosInf => ("pos_inf", None),
        LimitResult::NegInf => ("neg_inf", None),
        LimitResult::DoesNotExist => ("does_not_exist", None),
        LimitResult::Indeterminate => ("indeterminate", None),
    };

    let point_desc = match point {
        LimitPoint::Finite(v) => format!("{v}"),
        LimitPoint::PosInf => "+inf".to_string(),
        LimitPoint::NegInf => "-inf".to_string(),
    };

    let content = match fmt {
        OutputFormat::Pretty => match value {
            Some(v) => format!("lim x{wrt} -> {point_desc}: {expr_str} = {v}\n"),
            None => format!("lim x{wrt} -> {point_desc}: {expr_str} = {kind}\n"),
        },
        OutputFormat::Latex => {
            let rhs = match value {
                Some(v) => format!("{v}"),
                None => kind.to_string(),
            };
            format!("$$\\lim_{{x_{{{wrt}}} \\to {point_desc}}} = {rhs}$$\n")
        }
        OutputFormat::Json => {
            let value_field = match value {
                Some(v) => format!("{v}"),
                None => "null".to_string(),
            };
            let point_escaped = json_escape_str(&point_desc);
            format!(
                "{{\"version\":1,\"wrt\":{wrt},\"point\":\"{point_escaped}\",\"kind\":\"{kind}\",\"value\":{value_field}}}\n"
            )
        }
    };

    write_output(&content, out)
}
