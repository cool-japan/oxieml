/// Known mathematical constants to check against.
pub(super) const KNOWN_CONSTANTS: &[(&str, f64)] = &[
    ("e (Euler's number)", std::f64::consts::E),
    ("pi", std::f64::consts::PI),
    ("tau (2*pi)", std::f64::consts::TAU),
    ("ln(2)", std::f64::consts::LN_2),
    ("ln(10)", std::f64::consts::LN_10),
    ("sqrt(2)", std::f64::consts::SQRT_2),
    ("1/sqrt(2)", std::f64::consts::FRAC_1_SQRT_2),
    ("1/pi", std::f64::consts::FRAC_1_PI),
    ("2/pi", std::f64::consts::FRAC_2_PI),
    ("2/sqrt(pi)", std::f64::consts::FRAC_2_SQRT_PI),
    ("pi/2", std::f64::consts::FRAC_PI_2),
    ("pi/3", std::f64::consts::FRAC_PI_3),
    ("pi/4", std::f64::consts::FRAC_PI_4),
    ("pi/6", std::f64::consts::FRAC_PI_6),
    ("pi/8", std::f64::consts::FRAC_PI_8),
    ("log2(e)", std::f64::consts::LOG2_E),
    ("log10(e)", std::f64::consts::LOG10_E),
    ("golden ratio (phi)", 1.618_033_988_749_895),
    ("0", 0.0),
    ("1", 1.0),
    ("2", 2.0),
    ("3", 3.0),
    ("-1", -1.0),
];

/// Selects the output representation for results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum OutputFormat {
    /// Human-readable mathematical notation (default).
    Pretty,
    /// LaTeX math-mode expression.
    Latex,
    /// Hand-rolled JSON with a stable `version:1` envelope.
    Json,
}

impl OutputFormat {
    /// Parse `--format <value>` from the argument list.
    ///
    /// Returns `Ok(Pretty)` when the flag is absent (default).
    pub(super) fn from_args(args: &[String]) -> Result<Self, String> {
        let Some(pos) = args.iter().position(|a| a == "--format") else {
            return Ok(Self::Pretty);
        };
        let val = args
            .get(pos + 1)
            .ok_or_else(|| "--format requires a value: pretty, latex, or json".to_string())?;
        match val.as_str() {
            "pretty" => Ok(Self::Pretty),
            "latex" => Ok(Self::Latex),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "--format: unknown value '{other}'; expected pretty, latex, or json"
            )),
        }
    }
}

/// Parse `--output <path>` from the argument list.
///
/// Returns `Ok(None)` when the flag is absent (stdout).
pub(super) fn output_path(args: &[String]) -> Result<Option<std::path::PathBuf>, String> {
    let Some(pos) = args.iter().position(|a| a == "--output") else {
        return Ok(None);
    };
    let val = args
        .get(pos + 1)
        .ok_or_else(|| "--output requires a file path".to_string())?;
    Ok(Some(std::path::PathBuf::from(val)))
}

/// Write `content` to `path` when `Some`, or to stdout when `None`.
pub(super) fn write_output(
    content: &str,
    path: &Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    match path {
        None => {
            print!("{content}");
            Ok(())
        }
        Some(p) => std::fs::write(p, content).map_err(Into::into),
    }
}

/// Escape a string for embedding in a JSON string literal.
///
/// Handles the characters that are mandatory escapes per RFC 8259:
/// `"`, `\`, and the control characters U+0000–U+001F.
pub(super) fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}
