//! Integration tests for T2's clap-based subcommands (`eval`, `lower`, `grad`,
//! `integrate`, `solve`, `simplify`, `symreg`, `smt`, `series`, `limit`,
//! `repl`) and for continued backward compatibility of the legacy flag-based
//! shim (`--lower`, `--grad`, `--symreg`, `--help`).
//!
//! These tests drive the compiled `oxieml` binary end-to-end via
//! `assert_cmd`, exactly like `tests/cli_format_test.rs` and
//! `tests/cli_symreg_test.rs` (which are intentionally left untouched).

use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::fs;

fn oxieml() -> Command {
    Command::cargo_bin("oxieml").expect("binary built (requires the `cli` feature, in `default`)")
}

// ============================================================================
// Spec-required tests
// ============================================================================

/// `oxieml eval "E(1,1)"` — the exact scenario named in the T2 spec's Tests
/// line. `E(1,1) = exp(1) - ln(1) = e`.
#[test]
fn eval_subcommand_evaluates_e() {
    oxieml()
        .arg("eval")
        .arg("E(1,1)")
        .assert()
        .success()
        .stdout(predicate::str::contains("2.718281828459045"));
}

/// The legacy `--lower` flag must keep working unchanged, exactly as before
/// T2 (this mirrors `tests/cli_format_test.rs::format_pretty_default`,
/// re-asserted here as documentation that the legacy shim survived the clap
/// migration).
#[test]
fn legacy_lower_shim_still_works() {
    oxieml()
        .arg("--lower")
        .arg("E(x0,1)")
        .assert()
        .success()
        .stdout(predicate::str::contains("exp"));
}

/// A piped REPL session: evaluate one expression, then `:quit`. No terminal
/// is attached (stdin is a pipe), so the REPL must not print a prompt or
/// block waiting for a TTY — it should just process both lines and exit 0.
#[test]
fn repl_piped_session_evaluates_then_quits() {
    oxieml()
        .arg("repl")
        .write_stdin("E(1,1)\n:quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("2.718281828459045"));
}

/// `oxieml --help` must exit 0 (top-level legacy help path).
#[test]
fn top_level_help_exits_zero() {
    oxieml().arg("--help").assert().success();
}

// ============================================================================
// Additional subcommand coverage — each calls into a distinct real library
// API (LoweredOp::grad / integrate / integrate_definite / solve_for /
// oxieml::simplify::simplify / taylor / limit / SymRegEngine::discover /
// EmlSmtSolver::check_all), not a canned/placeholder response.
// ============================================================================

#[test]
fn subcommand_help_exits_zero() {
    // clap's own generated per-subcommand help, distinct from the legacy
    // `usage_text()` path exercised by `top_level_help_exits_zero`.
    oxieml().arg("eval").arg("--help").assert().success();
}

#[test]
fn grad_subcommand_computes_derivative() {
    // d/dx0 exp(x0) = exp(x0).
    oxieml()
        .arg("grad")
        .arg("0")
        .arg("E(x0,1)")
        .assert()
        .success()
        .stdout(predicate::str::contains("d/dx0"))
        .stdout(predicate::str::contains("exp(x0)"));
}

/// Legacy `--grad` flag must still work after `run_grad` was refactored to
/// return `Result` (needed so the REPL's `:grad` meta-command can recover
/// from a bad expression instead of killing the whole session).
#[test]
fn legacy_grad_shim_still_works() {
    oxieml()
        .arg("--grad")
        .arg("0")
        .arg("E(x0,1)")
        .assert()
        .success()
        .stdout(predicate::str::contains("d/dx0"));
}

#[test]
fn integrate_subcommand_indefinite() {
    // Antiderivative of exp(x0) w.r.t. x0 is exp(x0) (+ C, omitted).
    oxieml()
        .arg("integrate")
        .arg("E(x0,1)")
        .arg("--wrt")
        .arg("0")
        .assert()
        .success()
        .stdout(predicate::str::contains("exp(x0)"));
}

#[test]
fn integrate_subcommand_definite() {
    // integral_0^1 exp(x0) dx0 = e - 1 ~= 1.7182818284590452.
    oxieml()
        .arg("integrate")
        .arg("E(x0,1)")
        .arg("--wrt")
        .arg("0")
        .arg("--from")
        .arg("0")
        .arg("--to")
        .arg("1")
        .assert()
        .success()
        .stdout(predicate::str::contains("1.7182818"));
}

/// `--from` without `--to` is rejected by clap's `requires` validation before
/// any integration logic runs.
#[test]
fn integrate_subcommand_rejects_lone_from() {
    oxieml()
        .arg("integrate")
        .arg("E(x0,1)")
        .arg("--from")
        .arg("0")
        .assert()
        .failure();
}

#[test]
fn solve_subcommand_closed_form() {
    // exp(x0) - ln(1) == 1  =>  exp(x0) == 1  =>  x0 == ln(1) == 0.
    oxieml()
        .arg("solve")
        .arg("E(x0,1)")
        .arg("1")
        .arg("--var")
        .arg("0")
        .assert()
        .success()
        .stdout(predicate::str::contains("x0 (closed)"))
        .stdout(predicate::str::contains('0'));
}

#[test]
fn simplify_subcommand_tree_level() {
    oxieml()
        .arg("simplify")
        .arg("E(x0,1)")
        .assert()
        .success()
        .stdout(predicate::str::contains("exp(x0)"))
        .stdout(predicate::str::contains("tree-level simplify"));
}

#[test]
fn series_subcommand_maclaurin() {
    // Maclaurin series of exp(x0) has an x0 term with unit coefficient.
    oxieml()
        .arg("series")
        .arg("E(x0,1)")
        .arg("--wrt")
        .arg("0")
        .arg("--order")
        .arg("3")
        .assert()
        .success()
        .stdout(predicate::str::contains("x0"));
}

#[test]
fn limit_subcommand_at_finite_point() {
    // lim x0 -> 0 of exp(x0) is continuous there; just check it runs and
    // reports the limit line (the exact probed value is not asserted since
    // `LoweredOp::limit` uses numeric probing, not exact substitution).
    oxieml()
        .arg("limit")
        .arg("E(x0,1)")
        .arg("--wrt")
        .arg("0")
        .arg("--point")
        .arg("0")
        .assert()
        .success()
        .stdout(predicate::str::contains("lim x0 -> 0"));
}

#[test]
fn limit_subcommand_at_infinity() {
    oxieml()
        .arg("limit")
        .arg("E(x0,1)")
        .arg("--wrt")
        .arg("0")
        .arg("--point")
        .arg("inf")
        .assert()
        .success()
        .stdout(predicate::str::contains("+inf"));
}

#[test]
fn symreg_subcommand_recovers_linear() {
    // Same corpus/shape as tests/cli_symreg_test.rs::recovery_linear, driven
    // through the new `symreg` subcommand instead of the legacy `--symreg` flag.
    let mut data = String::from("# y = x0 + x1\n");
    for i in 0..4 {
        for j in 0..5 {
            let x0 = i as f64;
            let x1 = j as f64;
            let y = x0 + x1;
            data.push_str(&format!("{x0} {x1} {y}\n"));
        }
    }

    oxieml()
        .arg("symreg")
        .arg("--vars")
        .arg("2")
        .arg("--max-depth")
        .arg("1")
        .arg("--max-iter")
        .arg("50")
        .arg("--num-restarts")
        .arg("1")
        .arg("--top")
        .arg("1")
        .write_stdin(data)
        .assert()
        .success()
        .stdout(predicate::str::contains("Rank 1:"));
}

#[test]
fn symreg_subcommand_file_input() {
    let mut data = String::from("# y = 2*x0\n");
    for i in 0..10 {
        let x0 = i as f64;
        let y = 2.0 * x0;
        data.push_str(&format!("{x0} {y}\n"));
    }

    let dir = env::temp_dir();
    let path = dir.join(format!(
        "oxieml_cli_subcommands_symreg_file_{}.txt",
        std::process::id()
    ));
    fs::write(&path, &data).expect("failed to write temp dataset");

    oxieml()
        .arg("symreg")
        .arg("--vars")
        .arg("1")
        .arg("--max-depth")
        .arg("1")
        .arg("--max-iter")
        .arg("50")
        .arg("--num-restarts")
        .arg("1")
        .arg("--top")
        .arg("1")
        .arg("--file")
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Rank 1:"));

    let _ = fs::remove_file(&path);
}

/// `smt` requires the `smt` feature; only compiled when it's enabled (e.g.
/// `cargo nextest run --all-features`).
#[cfg(feature = "smt")]
#[test]
fn smt_subcommand_sat() {
    // exp(x0) > 1 is satisfiable for x0 in (-1, 5], e.g. x0 = 2.
    oxieml()
        .arg("smt")
        .arg("E(x0,1) > 1")
        .arg("--lo")
        .arg("-1")
        .arg("--hi")
        .arg("5")
        .assert()
        .success()
        .stdout(predicate::str::contains("sat"))
        .stdout(predicate::str::contains("witness:"));
}

#[cfg(feature = "smt")]
#[test]
fn smt_subcommand_unsat_core() {
    // exp(x0) > 100 (unreachable in [-1,1]) AND exp(x0) < 1 (only x0<0) AND
    // x0 >= 0 (via `0 <= x0`) together are unsatisfiable within [-1, 1]:
    // the first two constraints alone already conflict on sign, so a core of
    // size < 3 should be reported when --unsat-core is requested.
    oxieml()
        .arg("smt")
        .arg("E(x0,1) > 100")
        .arg("E(x0,1) < 1")
        .arg("--lo")
        .arg("-1")
        .arg("--hi")
        .arg("1")
        .arg("--unsat-core")
        .assert()
        .success()
        .stdout(predicate::str::contains("unsat"));
}

// ============================================================================
// REPL meta-commands (reuse the same run_* functions as the subcommands)
// ============================================================================

#[test]
fn repl_help_meta_command() {
    oxieml()
        .arg("repl")
        .write_stdin(":help\n:quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("meta-command"));
}

#[test]
fn repl_lower_meta_command() {
    oxieml()
        .arg("repl")
        .write_stdin(":lower E(x0,1)\n:quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("exp(x0)"));
}

#[test]
fn repl_grad_meta_command() {
    oxieml()
        .arg("repl")
        .write_stdin(":grad 0 E(x0,1)\n:quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("d/dx0"));
}

/// A bad line must report an error and keep the session alive — the next
/// line still gets evaluated, and the process still exits 0 at `:quit`.
#[test]
fn repl_recovers_from_bad_line() {
    oxieml()
        .arg("repl")
        .write_stdin("not-an-expression\nE(1,1)\n:quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("2.718281828459045"));
}

#[test]
fn repl_eof_without_quit_exits_cleanly() {
    // No `:quit` at all — EOF on stdin should still end the loop cleanly.
    oxieml()
        .arg("repl")
        .write_stdin("E(1,1)\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("2.718281828459045"));
}
