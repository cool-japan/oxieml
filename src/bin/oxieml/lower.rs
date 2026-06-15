use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::parser::parse;

pub(super) fn run_lower(
    expr_str: &str,
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tree = parse(expr_str).map_err(|e| format!("parse error: {e}"))?;
    let lowered = tree.lower().simplify();
    let pretty = lowered.to_pretty();
    let latex = lowered.to_latex();

    let content = match fmt {
        OutputFormat::Pretty => format!("{pretty}\n"),
        OutputFormat::Latex => format!("$${latex}$$\n"),
        OutputFormat::Json => {
            // Hand-rolled JSON — no serde dependency.
            let pretty_escaped = json_escape_str(&pretty);
            let latex_escaped = json_escape_str(&latex);
            format!(
                "{{\"version\":1,\"formulas\":[{{\"pretty\":\"{pretty_escaped}\",\"latex\":\"{latex_escaped}\"}}]}}\n"
            )
        }
    };

    write_output(&content, out)
}
