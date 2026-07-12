//! `oxieml smt` — SMT-check one or more EML constraints (feature `smt`).
//!
//! Constraints are written as `"<lhs> <op> <rhs>"` where `<op>` is one of
//! `<`, `<=`, `>`, `>=`, `==`, `!=`, and `<lhs>`/`<rhs>` are ordinary EML
//! expressions (the same `E(a,b)` / `eml(a,b)` / `xN` / numeric-literal
//! grammar every other subcommand accepts).

use super::args::count_variables;
use oxieml::parser::parse;
use oxieml::smt::{EmlConstraint, EmlSmtSolver, SmtResult};

/// Split `"<lhs> <op> <rhs>"` into a parsed [`EmlConstraint`] plus the number
/// of variables referenced (`max(var index) + 1`, or `0` if none).
///
/// Two-character operators are checked before their one-character prefixes
/// (`>=`/`<=` before `>`/`<`) so the split point is never taken inside a
/// two-character operator.
fn parse_constraint(s: &str) -> Result<(EmlConstraint, usize), String> {
    const OPS: &[&str] = &[">=", "<=", "==", "!=", "<", ">"];
    let (pos, op) = OPS
        .iter()
        .find_map(|op| s.find(op).map(|pos| (pos, *op)))
        .ok_or_else(|| {
            format!(
                "constraint \"{s}\" has no comparison operator \
                 (expected one of <, <=, >, >=, ==, !=)"
            )
        })?;

    let lhs_str = s[..pos].trim();
    let rhs_str = s[pos + op.len()..].trim();

    let lhs = parse(lhs_str).map_err(|e| format!("parse error (lhs of \"{s}\"): {e}"))?;
    let rhs = parse(rhs_str).map_err(|e| format!("parse error (rhs of \"{s}\"): {e}"))?;

    let num_vars = count_variables(&lhs).max(count_variables(&rhs));

    let constraint = match op {
        "<" => EmlConstraint::lt(lhs, rhs),
        "<=" => EmlConstraint::le(lhs, rhs),
        ">" => EmlConstraint::gt(lhs, rhs),
        ">=" => EmlConstraint::ge(lhs, rhs),
        "==" => EmlConstraint::eq(lhs, rhs),
        "!=" => EmlConstraint::ne(lhs, rhs),
        other => unreachable!("OPS is exhaustive, got '{other}'"),
    };

    Ok((constraint, num_vars))
}

/// Run the `smt` subcommand: check satisfiability of the conjunction of
/// `constraint_strs` via [`oxieml::smt::EmlSmtSolver::check_all`]. When the
/// result is `Unsat` and more than one constraint was given, optionally
/// extract a minimal unsatisfiable subset via
/// [`oxieml::smt::EmlSmtSolver::unsat_core`] (built on the incremental S1
/// solver stack).
pub(super) fn run_smt(
    constraint_strs: &[String],
    lo: f64,
    hi: f64,
    show_unsat_core: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if constraint_strs.is_empty() {
        return Err("smt requires at least one constraint, e.g. \"E(x0,1) > 1\"".into());
    }
    if lo >= hi {
        return Err(format!("--lo/--hi: expected lo < hi, got lo={lo}, hi={hi}").into());
    }

    let mut constraints = Vec::with_capacity(constraint_strs.len());
    let mut max_vars = 0usize;
    for s in constraint_strs {
        let (c, nv) = parse_constraint(s)?;
        max_vars = max_vars.max(nv);
        constraints.push(c);
    }

    let num_vars = max_vars.max(1);
    let solver = EmlSmtSolver::new(vec![(lo, hi); num_vars]);
    let result = solver.check_all(&constraints)?;

    match &result {
        SmtResult::Sat(solution) => {
            println!("sat");
            print!("witness:");
            for (i, v) in solution.assignments.iter().enumerate() {
                print!(" x{i}={v}");
            }
            println!();
            println!("exact: {}", solution.is_exact);
        }
        SmtResult::Unsat => {
            println!("unsat");
            if show_unsat_core && constraints.len() > 1 {
                match solver.unsat_core(&constraints)? {
                    Some(core) => {
                        print!("unsat core (input indices):");
                        for i in &core.indices {
                            print!(" {i}");
                        }
                        println!();
                        println!("verified minimal: {}", core.is_verified());
                    }
                    None => println!(
                        "(conjunction is unsat, but the oracle could not isolate a minimal core)"
                    ),
                }
            }
        }
        SmtResult::Unknown => println!("unknown"),
    }

    Ok(())
}
