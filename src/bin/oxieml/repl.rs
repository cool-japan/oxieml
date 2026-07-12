//! `oxieml repl` — interactive read-eval-print loop.
//!
//! This is the **default, FFI-free** REPL required by T2: it reads lines via
//! `std::io::Stdin`'s ordinary buffered `read_line`, with no raw-mode
//! terminal handling and therefore no libc/termios FFI. A richer
//! line-editing experience (history, cursor movement, `Ctrl-R` search) would
//! need a crate like `reedline`/`crossterm` behind an opt-in `repl-rich`
//! feature per the spec — that feature is **not implemented** here (see the
//! project's task-tracking notes and the T2 completion report): this REPL is
//! complete and clean on its own, and adding a half-built raw-mode path on
//! top of it was judged worse than shipping the plain version honestly.
//!
//! Bare input is evaluated as an EML expression (optionally followed by
//! whitespace-separated `xN=value` bindings, exactly like the `eval`
//! subcommand). Lines starting with `:` are meta-commands that dispatch to
//! the same `run_*` functions the other subcommands use — see [`print_help`].

use super::args::{pad_vars, vars_from_extra};
use super::format::OutputFormat;
#[cfg(feature = "smt")]
use super::smt;
use super::{grad, integrate, limit, lower, series, simplify, solve};
use oxieml::parser::parse;
use std::io::{self, BufRead, IsTerminal, Write};

const PROMPT: &str = "oxieml> ";

/// Run the REPL to completion (until `:quit`/EOF), reading from `stdin` and
/// writing to `stdout`/`stderr`.
///
/// A malformed line (bad parse, wrong arity) reports an error on that line
/// only and keeps the loop running — one typo should never kill the session.
pub(super) fn run_repl() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let interactive = stdin.is_terminal();
    let mut locked = stdin.lock();
    let mut stdout = io::stdout();

    loop {
        if interactive {
            print!("{PROMPT}");
            let _ = stdout.flush();
        }

        let mut line = String::new();
        let bytes_read = match locked.read_line(&mut line) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Error: failed to read from stdin: {e}");
                break;
            }
        };
        if bytes_read == 0 {
            if interactive {
                println!();
            }
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            ":quit" | ":q" | ":exit" => break,
            ":help" | ":h" => {
                print_help();
                continue;
            }
            _ => {}
        }

        if let Some(rest) = trimmed.strip_prefix(':') {
            if let Err(e) = dispatch_meta(rest) {
                eprintln!("Error: {e}");
            }
            continue;
        }

        eval_line(trimmed);
    }

    Ok(())
}

fn print_help() {
    println!("OxiEML REPL — enter an EML expression to evaluate it, or a meta-command:");
    println!();
    println!("  <expr> [x0=v0 x1=v1 ...]               evaluate an EML expression");
    println!("  :lower <expr>                           lower & simplify, print pretty form");
    println!("  :simplify <expr>                        tree-level simplify");
    println!("  :grad <idx> <expr>                      symbolic d/dx_idx");
    println!("  :integrate <wrt> <expr>                 indefinite integral w.r.t. x_wrt");
    println!(
        "  :limit <wrt> <point> <expr>              lim x_wrt -> point (point: number, inf, -inf)"
    );
    println!("  :series <wrt> <center> <order> <expr>   Taylor series about center");
    println!("  :solve <var> <lhs> <rhs>                 solve lhs == rhs for x_var");
    #[cfg(feature = "smt")]
    println!(
        "  :smt <lhs> <op> <rhs>                    SMT-check a constraint (op: < <= > >= == !=)"
    );
    println!("  :help, :h                                show this help");
    println!("  :quit, :q, :exit                         exit the REPL");
}

/// Evaluate a bare (non-`:`-prefixed) line: first whitespace token is the
/// expression, remaining tokens are `xN=value` bindings.
fn eval_line(trimmed: &str) {
    let mut tokens = trimmed.split_whitespace();
    let Some(expr) = tokens.next() else {
        return;
    };
    let extra: Vec<String> = tokens.map(str::to_string).collect();
    let vars = vars_from_extra(&extra);

    match parse(expr) {
        Ok(tree) => {
            let lowered = tree.lower().simplify();
            let padded = pad_vars(&vars, lowered.count_vars());
            println!("{}", lowered.eval(&padded));
        }
        Err(e) => eprintln!("Parse error: {e}"),
    }
}

/// Dispatch a `:command args...` meta-command.
fn dispatch_meta(rest: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut parts = rest.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("").trim();
    let cmd_args = parts.next().unwrap_or("").trim();

    match cmd {
        "lower" => lower::run_lower(cmd_args, &OutputFormat::Pretty, &None),
        "simplify" => simplify::run_simplify(cmd_args, &OutputFormat::Pretty, &None),
        "grad" => meta_grad(cmd_args),
        "integrate" => meta_integrate(cmd_args),
        "limit" => meta_limit(cmd_args),
        "series" => meta_series(cmd_args),
        "solve" => meta_solve(cmd_args),
        #[cfg(feature = "smt")]
        "smt" => smt::run_smt(&[cmd_args.to_string()], -10.0, 10.0, false),
        other => Err(format!("unknown meta-command ':{other}' (try :help)").into()),
    }
}

fn meta_grad(cmd_args: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = cmd_args.split_whitespace();
    let idx_str = tokens.next().ok_or(":grad requires <idx> <expr>")?;
    let expr = tokens.next().ok_or(":grad requires <idx> <expr>")?;
    let wrt: usize = idx_str
        .parse()
        .map_err(|_| format!(":grad: invalid index '{idx_str}'"))?;
    grad::run_grad(expr, wrt, &[])
}

fn meta_integrate(cmd_args: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = cmd_args.split_whitespace();
    let wrt_str = tokens.next().ok_or(":integrate requires <wrt> <expr>")?;
    let expr = tokens.next().ok_or(":integrate requires <wrt> <expr>")?;
    let wrt: usize = wrt_str
        .parse()
        .map_err(|_| format!(":integrate: invalid index '{wrt_str}'"))?;
    integrate::run_integrate(expr, wrt, None, &[], &OutputFormat::Pretty, &None)
}

fn meta_limit(cmd_args: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = cmd_args.split_whitespace();
    let wrt_str = tokens
        .next()
        .ok_or(":limit requires <wrt> <point> <expr>")?;
    let point_str = tokens
        .next()
        .ok_or(":limit requires <wrt> <point> <expr>")?;
    let expr = tokens
        .next()
        .ok_or(":limit requires <wrt> <point> <expr>")?;
    let wrt: usize = wrt_str
        .parse()
        .map_err(|_| format!(":limit: invalid index '{wrt_str}'"))?;
    limit::run_limit(expr, wrt, point_str, &OutputFormat::Pretty, &None)
}

fn meta_series(cmd_args: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = cmd_args.split_whitespace();
    let msg = ":series requires <wrt> <center> <order> <expr>";
    let wrt_str = tokens.next().ok_or(msg)?;
    let center_str = tokens.next().ok_or(msg)?;
    let order_str = tokens.next().ok_or(msg)?;
    let expr = tokens.next().ok_or(msg)?;
    let wrt: usize = wrt_str
        .parse()
        .map_err(|_| format!(":series: invalid index '{wrt_str}'"))?;
    let center: f64 = center_str
        .parse()
        .map_err(|_| format!(":series: invalid center '{center_str}'"))?;
    let order: usize = order_str
        .parse()
        .map_err(|_| format!(":series: invalid order '{order_str}'"))?;
    series::run_series(expr, wrt, center, order, &OutputFormat::Pretty, &None)
}

fn meta_solve(cmd_args: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = cmd_args.split_whitespace();
    let msg = ":solve requires <var> <lhs> <rhs>";
    let var_str = tokens.next().ok_or(msg)?;
    let lhs = tokens.next().ok_or(msg)?;
    let rhs = tokens.next().ok_or(msg)?;
    let var: usize = var_str
        .parse()
        .map_err(|_| format!(":solve: invalid index '{var_str}'"))?;
    solve::run_solve(lhs, rhs, var, None, &[], &OutputFormat::Pretty, &None)
}
