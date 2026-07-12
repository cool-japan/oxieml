//! `oxieml solve` — symbolic equation solving (`lhs == rhs`, target variable).

use super::args::pad_vars;
use super::format::{OutputFormat, json_escape_str, write_output};
use oxieml::SolveResult;
use oxieml::eval::EvalCtx;
use oxieml::parser::parse;

/// Run the `solve` subcommand: solve `lhs == rhs` for `Var(target_var)`.
///
/// Always prints the symbolic result — either a closed-form expression
/// ([`SolveResult::Closed`]) or the residual `lhs - rhs`
/// ([`SolveResult::Residual`]) when no algebraic inversion applies. When
/// `x0_guess` is provided, also resolves a numeric value via
/// [`SolveResult::solve_numeric`] (direct evaluation for `Closed`, Newton
/// root-finding from the given initial guess for `Residual`).
pub(super) fn run_solve(
    lhs_str: &str,
    rhs_str: &str,
    target_var: usize,
    x0_guess: Option<f64>,
    vars: &[f64],
    fmt: &OutputFormat,
    out: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let lhs = parse(lhs_str)
        .map_err(|e| format!("parse error (lhs): {e}"))?
        .lower()
        .simplify();
    let rhs = parse(rhs_str)
        .map_err(|e| format!("parse error (rhs): {e}"))?
        .lower()
        .simplify();

    let result = lhs.solve_for(target_var, &rhs);

    let (kind, symbolic, latex) = match &result {
        SolveResult::Closed(sol) => ("closed", sol.to_pretty(), sol.to_latex()),
        SolveResult::Residual(res) => ("residual", res.to_pretty(), res.to_latex()),
    };

    let numeric: Option<f64> = match x0_guess {
        Some(x0) => {
            let needed = lhs.count_vars().max(rhs.count_vars()).max(target_var + 1);
            let padded = pad_vars(vars, needed);
            let ctx = EvalCtx::new(&padded);
            Some(result.solve_numeric(target_var, &ctx, x0)?)
        }
        None => None,
    };

    let content = match fmt {
        OutputFormat::Pretty => {
            let mut buf = format!("x{target_var} ({kind}) = {symbolic}\n");
            if let Some(v) = numeric {
                buf.push_str(&format!("numeric: x{target_var} = {v}\n"));
            }
            buf
        }
        OutputFormat::Latex => {
            let mut buf = format!("$$x_{{{target_var}}} = {latex}$$\n");
            if let Some(v) = numeric {
                buf.push_str(&format!("$$x_{{{target_var}}} \\approx {v}$$\n"));
            }
            buf
        }
        OutputFormat::Json => {
            let symbolic_escaped = json_escape_str(&symbolic);
            let latex_escaped = json_escape_str(&latex);
            let numeric_field = match numeric {
                Some(v) => format!("{v}"),
                None => "null".to_string(),
            };
            format!(
                "{{\"version\":1,\"kind\":\"{kind}\",\"var\":{target_var},\"pretty\":\"{symbolic_escaped}\",\"latex\":\"{latex_escaped}\",\"numeric\":{numeric_field}}}\n"
            )
        }
    };

    write_output(&content, out)
}
