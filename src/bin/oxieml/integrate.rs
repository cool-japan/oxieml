//! `oxieml integrate` — symbolic antiderivatives and definite integrals.

use super::args::pad_vars;
use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::eval::EvalCtx;
use oxieml::integrate::IntegrateResult;
use oxieml::parser::parse;

/// Run the `integrate` subcommand.
///
/// With `bounds = None`, computes the closed-form antiderivative via
/// [`oxieml::LoweredOp::integrate`]. With `bounds = Some((a, b))`, computes the
/// definite integral via [`oxieml::LoweredOp::integrate_definite`] (symbolic
/// evaluation of the antiderivative at the endpoints, falling back to adaptive
/// quadrature when no closed form exists).
pub(super) fn run_integrate(
    expr_str: &str,
    wrt: usize,
    bounds: Option<(f64, f64)>,
    vars: &[f64],
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tree = parse(expr_str).map_err(|e| format!("parse error: {e}"))?;
    let lowered = tree.lower().simplify();

    let content = match bounds {
        Some((a, b)) => {
            let padded = pad_vars(vars, lowered.count_vars().max(wrt + 1));
            let ctx = EvalCtx::new(&padded);
            let value = lowered.integrate_definite(wrt, a, b, &ctx)?;
            match fmt {
                OutputFormat::Pretty => {
                    format!("integral of {expr_str} d x{wrt} from {a} to {b} = {value}\n")
                }
                OutputFormat::Latex => format!("$$\\int_{{{a}}}^{{{b}}} = {value}$$\n"),
                OutputFormat::Json => format!(
                    "{{\"version\":1,\"kind\":\"definite\",\"wrt\":{wrt},\"a\":{a},\"b\":{b},\"value\":{value}}}\n"
                ),
            }
        }
        None => match lowered.integrate(wrt) {
            IntegrateResult::Closed(antideriv) => {
                let pretty = antideriv.to_pretty();
                let latex = antideriv.to_latex();
                match fmt {
                    OutputFormat::Pretty => format!("{pretty}\n"),
                    OutputFormat::Latex => format!("$${latex}$$\n"),
                    OutputFormat::Json => {
                        let pretty_escaped = json_escape_str(&pretty);
                        let latex_escaped = json_escape_str(&latex);
                        format!(
                            "{{\"version\":1,\"kind\":\"indefinite\",\"wrt\":{wrt},\"pretty\":\"{pretty_escaped}\",\"latex\":\"{latex_escaped}\"}}\n"
                        )
                    }
                }
            }
            IntegrateResult::Unsupported => {
                return Err(format!(
                    "no closed-form antiderivative found for d/dx{wrt} of \"{expr_str}\"; \
                     pass --from and --to for a numeric (quadrature) definite integral"
                )
                .into());
            }
        },
    };

    write_output(&content, out)
}
