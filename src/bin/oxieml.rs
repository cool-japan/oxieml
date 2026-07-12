//! OxiEML CLI — Parse, evaluate, and generate EML expressions.
//!
//! # New (clap-based) interface — subcommands
//!
//!   oxieml eval "E(1, 1)"                       # Evaluate an EML expression
//!   oxieml eval "E(x0,1)" x0=2.0                 # With variable bindings
//!   oxieml lower "E(x0,1)" --format latex        # Lower & simplify
//!   oxieml grad 0 "E(x0,1)"                      # Symbolic d/dx0
//!   oxieml integrate "E(x0,1)" --wrt 0           # Indefinite integral
//!   oxieml integrate "E(x0,1)" --wrt 0 --from 0 --to 1   # Definite integral
//!   oxieml solve "E(x0,1)" "E(1,1)" --var 0      # Solve exp(x0) == e
//!   oxieml simplify "E(x0,1)"                    # Tree-level simplify
//!   oxieml symreg --vars 1 --file data.txt       # Discover a formula
//!   oxieml smt "E(x0,1) > 1" --lo -5 --hi 5      # SMT-check (needs `smt` feature)
//!   oxieml series "E(x0,1)" --wrt 0 --order 4    # Taylor series
//!   oxieml limit "E(x0,1)" --wrt 0 --point inf   # Limit as x0 -> +inf
//!   oxieml repl                                  # Interactive REPL (std stdin)
//!
//! # Legacy interface — preserved via shim (unchanged, exercised by
//! # `tests/cli_format_test.rs` / `tests/cli_symreg_test.rs`)
//!
//!   oxieml "E(1, 1)"                     # Evaluate EML expression
//!   oxieml -g pi                          # Generate EML for π
//!   oxieml -g "sin(x0)" x0=0.5           # Generate & evaluate sin
//!   oxieml --file expression.txt          # Read from file
//!   echo "E(1, 1)" | oxieml              # Read from stdin
//!   oxieml --lower "E(x0,1)" --format latex   # Print LaTeX lowered form
//!   oxieml --lower "E(x0,1)" --format json    # Print JSON lowered form
//!
//! Dispatch rule: if the first argument is exactly one of the 11 subcommand
//! keywords above, the clap parser runs; otherwise every argument is handled
//! by the legacy shim exactly as before (`--lower`, `--gen`/`-g`, `--grad`/
//! `-d`, `--list`/`-l`, `--symreg`/`-s`, bare expression, stdin, `--help`/
//! `-h`, `--version`/`-V`). The two syntaxes cannot collide: legacy flags all
//! start with `-`, subcommands never do.

#[path = "oxieml/args.rs"]
mod args;
#[path = "oxieml/evaluate.rs"]
mod evaluate;
#[path = "oxieml/format.rs"]
mod format;
#[path = "oxieml/generate.rs"]
mod generate;
#[path = "oxieml/grad.rs"]
mod grad;
#[path = "oxieml/integrate.rs"]
mod integrate;
#[path = "oxieml/limit.rs"]
mod limit;
#[path = "oxieml/lower.rs"]
mod lower;
#[path = "oxieml/repl.rs"]
mod repl;
#[path = "oxieml/series.rs"]
mod series;
#[path = "oxieml/simplify.rs"]
mod simplify;
#[cfg(feature = "smt")]
#[path = "oxieml/smt.rs"]
mod smt;
#[path = "oxieml/solve.rs"]
mod solve;
#[path = "oxieml/symreg.rs"]
mod symreg;

use args::{get_input, parse_var_assignments};
use clap::{Args, Parser, Subcommand};
use evaluate::run_evaluate_fmt;
use format::{OutputArgs, OutputFormat, output_path};
use generate::{print_known_functions, run_generate, try_generate};
use oxieml::parser::parse;

/// First-argument keywords that route to the clap-based parser instead of
/// the legacy shim. Must stay in lock-step with the `Commands` variant names
/// below (clap renders each variant in kebab-case; every name here is a
/// single lowercase word, so kebab-case is a no-op).
const SUBCOMMANDS: &[&str] = &[
    "eval",
    "lower",
    "grad",
    "integrate",
    "solve",
    "simplify",
    "symreg",
    "smt",
    "series",
    "limit",
    "repl",
];

fn main() {
    let raw_args: Vec<String> = std::env::args().collect();

    if raw_args
        .get(1)
        .is_some_and(|a| SUBCOMMANDS.contains(&a.as_str()))
    {
        run_cli(raw_args);
        return;
    }

    run_legacy(&raw_args);
}

// ============================================================================
// clap-based subcommand interface (T2)
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "oxieml",
    version,
    about = "EML operator CLI: all elementary functions from exp(x) - ln(y).",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Evaluate an EML expression numerically.
    Eval(EvalArgs),
    /// Lower & simplify an EML expression; print pretty/latex/json.
    Lower(LowerArgs),
    /// Symbolic partial derivative d/dx_idx of an EML expression.
    Grad(GradArgs),
    /// Symbolic antiderivative, or definite integral with --from/--to.
    Integrate(IntegrateArgs),
    /// Solve lhs == rhs for a target variable.
    Solve(SolveArgs),
    /// Tree-level algebraic simplification (distinct from lowered-IR simplify).
    Simplify(SimplifyArgs),
    /// Discover closed-form formulas from tabular data.
    Symreg(SymregArgs),
    /// SMT-check one or more EML constraints.
    #[cfg(feature = "smt")]
    Smt(SmtArgs),
    /// Taylor/Maclaurin series expansion.
    Series(SeriesArgs),
    /// One- or two-sided limit computation.
    Limit(LimitArgs),
    /// Interactive read-eval-print loop (FFI-free, std stdin/stdout).
    Repl(ReplArgs),
}

#[derive(Args, Debug)]
struct EvalArgs {
    /// EML expression, e.g. "E(1,1)" or "E(x0,1)".
    expr: String,
    /// Variable bindings as xN=value (e.g. x0=2.0).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    vars: Vec<String>,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug)]
struct LowerArgs {
    /// EML expression to lower & simplify.
    expr: String,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug)]
struct GradArgs {
    /// Index of the variable to differentiate with respect to.
    wrt: usize,
    /// EML expression.
    expr: String,
    /// Optional bindings (xN=value) for a numeric evaluation of the derivative.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    vars: Vec<String>,
}

#[derive(Args, Debug)]
struct IntegrateArgs {
    /// EML expression to integrate.
    expr: String,
    /// Variable index to integrate with respect to.
    #[arg(long, default_value_t = 0)]
    wrt: usize,
    /// Lower bound of a definite integral (requires --to).
    #[arg(long, requires = "to", allow_hyphen_values = true)]
    from: Option<f64>,
    /// Upper bound of a definite integral (requires --from).
    #[arg(long, requires = "from", allow_hyphen_values = true)]
    to: Option<f64>,
    /// Bindings (xN=value) for variables other than --wrt (definite integrals only).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    vars: Vec<String>,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug)]
struct SolveArgs {
    /// Left-hand side EML expression.
    lhs: String,
    /// Right-hand side EML expression.
    rhs: String,
    /// Target variable index to solve for.
    #[arg(long)]
    var: usize,
    /// Initial guess for numeric root-finding (Newton fallback when no closed form exists).
    #[arg(long, allow_hyphen_values = true)]
    x0: Option<f64>,
    /// Bindings (xN=value) for variables other than --var (used only with --x0).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    vars: Vec<String>,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug)]
struct SimplifyArgs {
    /// EML expression to simplify.
    expr: String,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug)]
struct SymregArgs {
    /// Number of input variables per data row.
    #[arg(long = "vars")]
    num_vars: usize,
    /// Number of top formulas to print.
    #[arg(long, default_value_t = 3)]
    top: usize,
    /// Maximum tree depth to explore.
    #[arg(long)]
    max_depth: Option<usize>,
    /// Maximum optimization iterations per topology.
    #[arg(long)]
    max_iter: Option<usize>,
    /// Adam learning rate.
    #[arg(long)]
    learning_rate: Option<f64>,
    /// Convergence tolerance (MSE).
    #[arg(long)]
    tolerance: Option<f64>,
    /// Occam's razor coefficient.
    #[arg(long)]
    complexity_penalty: Option<f64>,
    /// Random restarts per topology.
    #[arg(long)]
    num_restarts: Option<usize>,
    /// Search strategy: "exhaustive" (default) or "beam:<N>".
    #[arg(long)]
    strategy: Option<String>,
    /// Dataset file path (defaults to stdin).
    #[arg(long, short = 'f')]
    file: Option<std::path::PathBuf>,
    #[command(flatten)]
    output: OutputArgs,
}

/// SMT constraint-checking arguments (feature `smt`).
#[cfg(feature = "smt")]
#[derive(Args, Debug)]
struct SmtArgs {
    /// One or more constraints "<lhs> <op> <rhs>", op in < <= > >= == !=.
    #[arg(required = true)]
    constraints: Vec<String>,
    /// Lower bound applied to every variable's search interval.
    #[arg(long, default_value_t = -10.0, allow_hyphen_values = true)]
    lo: f64,
    /// Upper bound applied to every variable's search interval.
    #[arg(long, default_value_t = 10.0, allow_hyphen_values = true)]
    hi: f64,
    /// On Unsat with more than one constraint, also compute a minimal unsat core.
    #[arg(long)]
    unsat_core: bool,
}

#[derive(Args, Debug)]
struct SeriesArgs {
    /// EML expression to expand.
    expr: String,
    /// Variable index to expand with respect to.
    #[arg(long, default_value_t = 0)]
    wrt: usize,
    /// Expansion center (0.0 = Maclaurin series).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    center: f64,
    /// Polynomial order (<= 170).
    #[arg(long)]
    order: usize,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug)]
struct LimitArgs {
    /// EML expression.
    expr: String,
    /// Variable index the limit is taken over.
    #[arg(long, default_value_t = 0)]
    wrt: usize,
    /// Limit point: a finite number, "inf"/"+inf", or "-inf".
    #[arg(long, allow_hyphen_values = true)]
    point: String,
    #[command(flatten)]
    output: OutputArgs,
}

/// No fields today; kept as a struct (rather than a unit variant) so a future
/// `repl-rich` opt-in flag can be added without an enum-shape break.
#[derive(Args, Debug)]
struct ReplArgs {}

fn run_cli(raw_args: Vec<String>) {
    let cli = Cli::parse_from(raw_args);

    let result: Result<(), Box<dyn std::error::Error>> = match cli.command {
        Commands::Eval(a) => dispatch_eval(a),
        Commands::Lower(a) => lower::run_lower(&a.expr, &a.output.format(), &a.output.output()),
        Commands::Grad(a) => dispatch_grad(a),
        Commands::Integrate(a) => dispatch_integrate(a),
        Commands::Solve(a) => dispatch_solve(a),
        Commands::Simplify(a) => {
            simplify::run_simplify(&a.expr, &a.output.format(), &a.output.output())
        }
        Commands::Symreg(a) => dispatch_symreg(a),
        #[cfg(feature = "smt")]
        Commands::Smt(a) => smt::run_smt(&a.constraints, a.lo, a.hi, a.unsat_core),
        Commands::Series(a) => series::run_series(
            &a.expr,
            a.wrt,
            a.center,
            a.order,
            &a.output.format(),
            &a.output.output(),
        ),
        Commands::Limit(a) => limit::run_limit(
            &a.expr,
            a.wrt,
            &a.point,
            &a.output.format(),
            &a.output.output(),
        ),
        Commands::Repl(_) => repl::run_repl(),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn dispatch_eval(a: EvalArgs) -> Result<(), Box<dyn std::error::Error>> {
    let tree = parse(&a.expr).map_err(|e| format!("parse error: {e}"))?;
    let vars = args::vars_from_extra(&a.vars);
    run_evaluate_fmt(
        &tree,
        &a.expr,
        &vars,
        &a.output.format(),
        &a.output.output(),
    )
}

fn dispatch_grad(a: GradArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vars = args::vars_from_extra(&a.vars);
    grad::run_grad(&a.expr, a.wrt, &vars)
}

fn dispatch_integrate(a: IntegrateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bounds = match (a.from, a.to) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        _ => None,
    };
    let vars = args::vars_from_extra(&a.vars);
    integrate::run_integrate(
        &a.expr,
        a.wrt,
        bounds,
        &vars,
        &a.output.format(),
        &a.output.output(),
    )
}

fn dispatch_solve(a: SolveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vars = args::vars_from_extra(&a.vars);
    solve::run_solve(
        &a.lhs,
        &a.rhs,
        a.var,
        a.x0,
        &vars,
        &a.output.format(),
        &a.output.output(),
    )
}

/// Rebuild a legacy-style argv slice so [`symreg::run_symreg`] — shared with
/// the `--symreg` legacy shim — can be reused verbatim instead of duplicating
/// dataset parsing and engine invocation here.
fn dispatch_symreg(a: SymregArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut synthetic: Vec<String> = vec![
        String::new(),
        "--vars".to_string(),
        a.num_vars.to_string(),
        "--top".to_string(),
        a.top.to_string(),
    ];
    if let Some(v) = a.max_depth {
        synthetic.push("--max-depth".to_string());
        synthetic.push(v.to_string());
    }
    if let Some(v) = a.max_iter {
        synthetic.push("--max-iter".to_string());
        synthetic.push(v.to_string());
    }
    if let Some(v) = a.learning_rate {
        synthetic.push("--learning-rate".to_string());
        synthetic.push(v.to_string());
    }
    if let Some(v) = a.tolerance {
        synthetic.push("--tolerance".to_string());
        synthetic.push(v.to_string());
    }
    if let Some(v) = a.complexity_penalty {
        synthetic.push("--complexity-penalty".to_string());
        synthetic.push(v.to_string());
    }
    if let Some(v) = a.num_restarts {
        synthetic.push("--num-restarts".to_string());
        synthetic.push(v.to_string());
    }
    if let Some(v) = a.strategy {
        synthetic.push("--strategy".to_string());
        synthetic.push(v);
    }
    if let Some(p) = a.file {
        synthetic.push("--file".to_string());
        synthetic.push(p.to_string_lossy().into_owned());
    }
    symreg::run_symreg(&synthetic, &a.output.format(), &a.output.output())
}

// ============================================================================
// Legacy shim (unchanged behavior, preserved for backward compatibility)
// ============================================================================

fn run_legacy(args: &[String]) {
    // --help / -h
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // --version / -V
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("oxieml {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // --format / --output are global flags consumed by subcommands.
    let fmt = match OutputFormat::from_args(args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let out = match output_path(args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    // --lower flag: lower an EML expression and format the result
    if let Some(pos) = args.iter().position(|a| a == "--lower") {
        let expr_str = match args.get(pos + 1) {
            Some(s) => s.clone(),
            None => {
                eprintln!("Error: --lower requires an expression argument");
                print_usage();
                std::process::exit(1);
            }
        };
        if let Err(e) = lower::run_lower(&expr_str, &fmt, &out) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
        return;
    }

    // Check for --gen / -g flag (generate mode)
    if let Some(pos) = args.iter().position(|a| a == "--gen" || a == "-g") {
        let expr = args.get(pos + 1).unwrap_or_else(|| {
            eprintln!("Error: --gen requires a function/constant name");
            print_usage();
            std::process::exit(1);
        });
        let vars = parse_var_assignments(args);
        run_generate(expr, &vars);
        return;
    }

    // Check for --grad / -d flag (symbolic gradient)
    if let Some(pos) = args.iter().position(|a| a == "--grad" || a == "-d") {
        let wrt_str = match args.get(pos + 1) {
            Some(s) => s,
            None => {
                eprintln!("Error: --grad requires a variable index (e.g., --grad 0)");
                print_usage();
                std::process::exit(1);
            }
        };
        let wrt = match wrt_str.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                eprintln!(
                    "Error: --grad requires a non-negative integer variable index, got '{wrt_str}'"
                );
                std::process::exit(1);
            }
        };
        let expr = match args.get(pos + 2) {
            Some(s) => s.clone(),
            None => {
                eprintln!("Error: --grad <idx> requires an expression argument");
                print_usage();
                std::process::exit(1);
            }
        };
        let vars = parse_var_assignments(args);
        if let Err(e) = grad::run_grad(&expr, wrt, &vars) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
        return;
    }

    // Check for --list / -l flag (list all known functions)
    if args.iter().any(|a| a == "--list" || a == "-l") {
        print_known_functions();
        return;
    }

    // Check for --symreg / -s flag (symbolic regression)
    if args.iter().any(|a| a == "--symreg" || a == "-s") {
        if let Err(e) = symreg::run_symreg(args, &fmt, &out) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
        return;
    }

    let input = match get_input(args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            std::process::exit(1);
        }
    };

    let input = input.trim();
    if input.is_empty() {
        eprintln!("Error: empty input");
        print_usage();
        std::process::exit(1);
    }

    // Try EML parse first; if it fails, try as a generate request
    match parse(input) {
        Ok(tree) => {
            let vars = parse_var_assignments(args);
            if let Err(e) = run_evaluate_fmt(&tree, input, &vars, &fmt, &out) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Err(parse_err) => {
            // Maybe the user typed a function name like "pi" or "sin(x0)"
            let vars = parse_var_assignments(args);
            if try_generate(input).is_some() {
                run_generate(input, &vars);
            } else {
                eprintln!("Parse error: {parse_err}");
                eprintln!();
                eprintln!("Hint: Use -g to generate EML from a function name:");
                eprintln!("  oxieml -g pi");
                eprintln!("  oxieml -g \"sin(x0)\"");
                std::process::exit(1);
            }
        }
    }
}

fn usage_text() -> &'static str {
    concat!(
        "\n",
        "Usage:\n",
        "  oxieml \"E(1, 1)\"                     # Evaluate EML expression\n",
        "  oxieml \"E(x0, 1)\" x0=2.0             # With variable bindings\n",
        "  oxieml -g pi                           # Generate EML for π\n",
        "  oxieml -g sin                          # Generate EML for sin(x0)\n",
        "  oxieml -g \"sin(x0)\" x0=0.5            # Generate & evaluate\n",
        "  oxieml --lower \"E(x0,1)\"              # Lower & print expression\n",
        "  oxieml --lower \"E(x0,1)\" --format latex  # LaTeX output\n",
        "  oxieml --lower \"E(x0,1)\" --format json   # JSON output\n",
        "  oxieml --grad 0 \"E(x0, 1)\"            # Symbolic derivative of exp(x0)\n",
        "  oxieml -d 0 \"E(x0, 1)\" x0=2.0         # Derivative + numerical value\n",
        "  oxieml -l                              # List all functions\n",
        "  oxieml --help                          # Show this help\n",
        "  oxieml --version                       # Show version\n",
        "  oxieml --file expression.txt           # Read from file\n",
        "  echo \"E(1, 1)\" | oxieml               # Read from stdin\n",
        "  oxieml --symreg --vars 1 --file data.txt  # Discover formula from data\n",
        "\n",
        "Flags:\n",
        "  --gen  <name>, -g <name>    Generate EML tree for a named function/constant\n",
        "  --lower <expr>              Lower & simplify an EML expression\n",
        "  --grad <idx>,  -d <idx>     Compute symbolic partial derivative w.r.t. variable <idx>\n",
        "                              of the given expression (via lowered IR + simplify)\n",
        "  --list, -l                  List all available functions/constants\n",
        "  --file <path>, -f <path>    Read expression (or dataset, with --symreg) from file\n",
        "  --help, -h                  Show this help\n",
        "  --version, -V               Show version\n",
        "\n",
        "Output flags (apply to --lower, --symreg, and default eval mode):\n",
        "  --format <fmt>              Output format: pretty (default), latex, json\n",
        "  --output <path>             Write output to file instead of stdout\n",
        "\n",
        "Symbolic regression (--symreg / -s):\n",
        "  Discover closed-form formulas from tabular data. Data is read from\n",
        "  --file <path> or stdin. Lines starting with '#' and blank lines are\n",
        "  skipped. Each remaining line must contain exactly <vars>+1 whitespace-\n",
        "  separated f64 values: x0 x1 ... x(N-1) target.\n",
        "\n",
        "  --symreg, -s                Enable symbolic regression mode\n",
        "  --vars <N>                  (required) Number of input variables per row\n",
        "  --top <K>                   Number of formulas to print (default 3)\n",
        "\n",
        "  Forwarding flags (all optional, fall back to SymRegConfig::default()):\n",
        "  --max-depth <usize>         Maximum tree depth to explore\n",
        "  --max-iter <usize>          Maximum optimization iterations per topology\n",
        "  --learning-rate <f64>       Adam learning rate\n",
        "  --tolerance <f64>           Convergence tolerance (MSE)\n",
        "  --complexity-penalty <f64>  Occam's razor coefficient\n",
        "  --num-restarts <usize>      Random restarts per topology\n",
        "  --strategy <s>              Search strategy: exhaustive (default) or beam:<N>\n",
        "                              e.g. --strategy beam:20 keeps top 20 candidates\n",
        "\n",
        "Notation:\n",
        "  1         The constant 1\n",
        "  x0, x1    Variables\n",
        "  E(a, b)   The EML operator: exp(a) - ln(b)\n",
        "  eml(a, b) Alternative notation for E(a, b)\n",
        "\n",
        "New (clap) subcommands — see `oxieml <subcommand> --help`:\n",
        "  eval, lower, grad, integrate, solve, simplify, symreg, smt, series, limit, repl\n",
        "  oxieml eval \"E(1,1)\"                    # Same as legacy default eval\n",
        "  oxieml repl                              # Interactive REPL (std stdin, FFI-free)\n",
        "  echo \"E(1,1)\\n:quit\" | oxieml repl      # Piped REPL session"
    )
}

fn print_usage() {
    eprintln!("{}", usage_text());
}

fn print_help() {
    println!("{}", usage_text());
}
