//! OxiEML CLI — Parse, evaluate, and generate EML expressions.
//!
//! Usage:
//!   oxieml "E(1, 1)"                     # Evaluate EML expression
//!   oxieml -g pi                          # Generate EML for π
//!   oxieml -g "sin(x0)" x0=0.5           # Generate & evaluate sin
//!   oxieml --file expression.txt          # Read from file
//!   echo "E(1, 1)" | oxieml              # Read from stdin
//!   oxieml --lower "E(x0,1)" --format latex   # Print LaTeX lowered form
//!   oxieml --lower "E(x0,1)" --format json    # Print JSON lowered form

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
#[path = "oxieml/lower.rs"]
mod lower;
#[path = "oxieml/symreg.rs"]
mod symreg;

use args::{get_input, parse_var_assignments};
use evaluate::run_evaluate_fmt;
use format::{OutputFormat, output_path};
use generate::{print_known_functions, run_generate, try_generate};
use grad::run_grad;
use lower::run_lower;
use oxieml::parser::parse;
use symreg::run_symreg;

fn main() {
    let args: Vec<String> = std::env::args().collect();

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
    let fmt = match OutputFormat::from_args(&args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let out = match output_path(&args) {
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
        if let Err(e) = run_lower(&expr_str, &fmt, &out) {
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
        let vars = parse_var_assignments(&args);
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
        let vars = parse_var_assignments(&args);
        run_grad(&expr, wrt, &vars);
        return;
    }

    // Check for --list / -l flag (list all known functions)
    if args.iter().any(|a| a == "--list" || a == "-l") {
        print_known_functions();
        return;
    }

    // Check for --symreg / -s flag (symbolic regression)
    if args.iter().any(|a| a == "--symreg" || a == "-s") {
        if let Err(e) = run_symreg(&args, &fmt, &out) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
        return;
    }

    let input = match get_input(&args) {
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
            let vars = parse_var_assignments(&args);
            if let Err(e) = run_evaluate_fmt(&tree, input, &vars, &fmt, &out) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        Err(parse_err) => {
            // Maybe the user typed a function name like "pi" or "sin(x0)"
            let vars = parse_var_assignments(&args);
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
        "  eml(a, b) Alternative notation for E(a, b)",
    )
}

fn print_usage() {
    eprintln!("{}", usage_text());
}

fn print_help() {
    println!("{}", usage_text());
}
