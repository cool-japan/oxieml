//! `oxieml simplify` — tree-level EML algebraic simplification.
//!
//! Distinct from the `lower`/`eval` subcommands, which run
//! [`oxieml::LoweredOp::simplify`] on the *lowered* IR: this subcommand runs
//! [`oxieml::simplify::simplify`] on the *original* EML tree first (folding
//! `ln(exp(x))`, `exp(ln(x))`, and structurally-shared subexpressions at the
//! `eml(l, r)` level), then lowers+simplifies the result for display so every
//! output format (`pretty`/`latex`/`json`) stays consistent with the rest of
//! the CLI.

use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::parser::parse;

pub(super) fn run_simplify(
    expr_str: &str,
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tree = parse(expr_str).map_err(|e| format!("parse error: {e}"))?;
    let size_before = tree.size();
    let depth_before = tree.depth();

    let simplified_tree = oxieml::simplify::simplify(&tree);
    let size_after = simplified_tree.size();
    let depth_after = simplified_tree.depth();

    let lowered = simplified_tree.lower().simplify();
    let pretty = lowered.to_pretty();
    let latex = lowered.to_latex();

    let content = match fmt {
        OutputFormat::Pretty => format!(
            "{pretty}\n(tree-level simplify: {size_before} -> {size_after} nodes, depth {depth_before} -> {depth_after})\n"
        ),
        OutputFormat::Latex => format!("$${latex}$$\n"),
        OutputFormat::Json => {
            let pretty_escaped = json_escape_str(&pretty);
            let latex_escaped = json_escape_str(&latex);
            format!(
                "{{\"version\":1,\"pretty\":\"{pretty_escaped}\",\"latex\":\"{latex_escaped}\",\"size_before\":{size_before},\"size_after\":{size_after},\"depth_before\":{depth_before},\"depth_after\":{depth_after}}}\n"
            )
        }
    };

    write_output(&content, out)
}
