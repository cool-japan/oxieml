//! `oxieml series` — Taylor/Maclaurin series expansion.

use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::parser::parse;

/// Run the `series` subcommand: the order-`order` Taylor polynomial of `expr`
/// about `center` with respect to variable `wrt`, via
/// [`oxieml::LoweredOp::taylor`].
pub(super) fn run_series(
    expr_str: &str,
    wrt: usize,
    center: f64,
    order: usize,
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tree = parse(expr_str).map_err(|e| format!("parse error: {e}"))?;
    let lowered = tree.lower().simplify();

    let poly = lowered.taylor(wrt, center, order)?;
    let pretty = poly.to_pretty();
    let latex = poly.to_latex();

    let content = match fmt {
        OutputFormat::Pretty => format!("{pretty}\n"),
        OutputFormat::Latex => format!("$${latex}$$\n"),
        OutputFormat::Json => {
            let pretty_escaped = json_escape_str(&pretty);
            let latex_escaped = json_escape_str(&latex);
            format!(
                "{{\"version\":1,\"wrt\":{wrt},\"center\":{center},\"order\":{order},\"pretty\":\"{pretty_escaped}\",\"latex\":\"{latex_escaped}\"}}\n"
            )
        }
    };

    write_output(&content, out)
}
