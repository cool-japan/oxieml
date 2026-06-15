use super::args::count_variables;
use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::eval::EvalCtx;
use oxieml::parser::to_compact_string;
use oxieml::tree::EmlTree;

pub(super) fn run_evaluate_fmt(
    tree: &EmlTree,
    input: &str,
    vars: &[f64],
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let lowered = tree.lower().simplify();
    let pretty = lowered.to_pretty();
    let latex = lowered.to_latex();

    let content = match fmt {
        OutputFormat::Pretty => {
            // Classic verbose output piped to a single string.
            let mut buf = String::new();
            buf.push_str("=== OxiEML Expression Evaluator ===\n\n");

            if input.len() > 200 {
                buf.push_str(&format!(
                    "Input: {}... ({} chars)\n\n",
                    &input[..200],
                    input.len()
                ));
            } else {
                buf.push_str(&format!("Input: {input}\n\n"));
            }

            buf.push_str("--- Tree Statistics ---\n");
            buf.push_str(&format!("  Depth: {}\n", tree.depth()));
            buf.push_str(&format!("  Size (nodes): {}\n", tree.size()));
            buf.push_str(&format!("  Variables used: {}\n\n", count_variables(tree)));

            let compact = to_compact_string(tree);
            if compact.len() <= 200 {
                buf.push_str(&format!("Compact: {compact}\n\n"));
            }

            let ctx = EvalCtx::new(vars);
            buf.push_str("--- Real Evaluation ---\n");
            if !vars.is_empty() {
                buf.push_str("  Variables: ");
                for (i, v) in vars.iter().enumerate() {
                    if i > 0 {
                        buf.push_str(", ");
                    }
                    buf.push_str(&format!("x{i} = {v}"));
                }
                buf.push('\n');
            }
            match tree.eval_real(&ctx) {
                Ok(val) => {
                    buf.push_str(&format!("  Result: {val}\n"));
                    buf.push_str(&format!("  Result (full precision): {val:.17e}\n\n"));
                }
                Err(e) => {
                    buf.push_str(&format!("  Real evaluation failed: {e}\n\n"));
                }
            }

            buf.push_str("--- Lowered Form ---\n");
            if pretty.len() <= 500 {
                buf.push_str(&format!("  {pretty}\n\n"));
            } else {
                buf.push_str(&format!(
                    "  (expression too large to display, {} chars)\n\n",
                    pretty.len()
                ));
            }

            buf.push_str("--- Lowered Evaluation ---\n");
            let lowered_val = lowered.eval(vars);
            buf.push_str(&format!("  Result: {lowered_val}\n"));
            buf.push_str(&format!("  Result (full precision): {lowered_val:.17e}\n"));
            buf
        }
        OutputFormat::Latex => {
            format!("$${latex}$$\n")
        }
        OutputFormat::Json => {
            let val = lowered.eval(vars);
            let pretty_escaped = json_escape_str(&pretty);
            let latex_escaped = json_escape_str(&latex);
            format!(
                "{{\"version\":1,\"result\":{val},\"pretty\":\"{pretty_escaped}\",\"latex\":\"{latex_escaped}\"}}\n"
            )
        }
    };

    write_output(&content, out)
}
