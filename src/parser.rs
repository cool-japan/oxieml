//! Parser for EML expression notation.
//!
//! Parses expressions in `E(x, y)` or `eml(x, y)` notation into `EmlTree`.
//!
//! # Grammar
//!
//! ```text
//! expr     = "1" | var | eml_call
//! eml_call = ("E" | "eml") "(" expr "," expr ")"
//! var      = "x" DIGIT+
//! ```
//!
//! Whitespace and newlines are ignored between tokens.

use crate::tree::{EmlNode, EmlTree};
use std::sync::Arc;

/// Error from parsing an EML expression.
#[derive(Clone, Debug)]
pub struct ParseError {
    /// Position in the input string where the error occurred.
    pub position: usize,
    /// Description of the error.
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ParseError {}

/// Parse an EML expression string into an `EmlTree`.
///
/// Accepts both `E(x, y)` and `eml(x, y)` notation.
///
/// # Examples
///
/// ```
/// use oxieml::parser::parse;
///
/// let tree = parse("E(1, 1)").unwrap();
/// assert_eq!(tree.depth(), 1);
///
/// let tree = parse("eml(E(1, 1), 1)").unwrap();
/// assert_eq!(tree.depth(), 2);
/// ```
pub fn parse(input: &str) -> Result<EmlTree, ParseError> {
    let mut parser = Parser::new(input);
    let node = parser.parse_expr()?;
    parser.skip_whitespace();
    if parser.pos < parser.input.len() {
        return Err(ParseError {
            position: parser.pos,
            message: format!(
                "unexpected trailing characters: '{}'",
                &parser.input[parser.pos..parser.pos + 20.min(parser.input.len() - parser.pos)]
            ),
        });
    }
    Ok(EmlTree::from_node(node))
}

struct Parser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b' '
                || self.bytes[self.pos] == b'\n'
                || self.bytes[self.pos] == b'\r'
                || self.bytes[self.pos] == b'\t')
        {
            self.pos += 1;
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), ParseError> {
        self.skip_whitespace();
        if self.pos < self.bytes.len() && self.bytes[self.pos] == ch {
            self.pos += 1;
            Ok(())
        } else {
            let found = if self.pos < self.bytes.len() {
                format!("'{}'", self.bytes[self.pos] as char)
            } else {
                "end of input".to_string()
            };
            Err(ParseError {
                position: self.pos,
                message: format!("expected '{}', found {found}", ch as char),
            })
        }
    }

    fn parse_expr(&mut self) -> Result<Arc<EmlNode>, ParseError> {
        self.skip_whitespace();

        if self.pos >= self.bytes.len() {
            return Err(ParseError {
                position: self.pos,
                message: "unexpected end of input".to_string(),
            });
        }

        let ch = self.bytes[self.pos];

        // "1" → One
        if ch == b'1' {
            self.pos += 1;
            return Ok(Arc::new(EmlNode::One));
        }

        // "x" followed by digits → Var
        if ch == b'x' {
            return self.parse_var();
        }

        // "E" or "eml" → Eml call
        if ch == b'E' {
            // Could be "E(" or "eml("
            if self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'(' {
                // E(...)
                self.pos += 1; // skip 'E'
                return self.parse_eml_body();
            }
            if self.matches_ahead("eml") {
                self.pos += 3; // skip "eml"
                return self.parse_eml_body();
            }
            return Err(ParseError {
                position: self.pos,
                message: "expected 'E(' or 'eml('".to_string(),
            });
        }

        if ch == b'e' {
            if self.matches_ahead("eml") {
                self.pos += 3;
                return self.parse_eml_body();
            }
            return Err(ParseError {
                position: self.pos,
                message: "expected 'eml('".to_string(),
            });
        }

        Err(ParseError {
            position: self.pos,
            message: format!("unexpected character '{}'", ch as char),
        })
    }

    fn parse_eml_body(&mut self) -> Result<Arc<EmlNode>, ParseError> {
        self.expect(b'(')?;
        let left = self.parse_expr()?;
        self.expect(b',')?;
        let right = self.parse_expr()?;
        self.expect(b')')?;
        Ok(Arc::new(EmlNode::Eml { left, right }))
    }

    fn parse_var(&mut self) -> Result<Arc<EmlNode>, ParseError> {
        self.pos += 1; // skip 'x'
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(ParseError {
                position: start,
                message: "expected digit after 'x'".to_string(),
            });
        }
        let idx: usize = self.input[start..self.pos]
            .parse()
            .map_err(|_| ParseError {
                position: start,
                message: "invalid variable index".to_string(),
            })?;
        Ok(Arc::new(EmlNode::Var(idx)))
    }

    fn matches_ahead(&self, s: &str) -> bool {
        let end = self.pos + s.len();
        if end > self.bytes.len() {
            return false;
        }
        &self.input[self.pos..end] == s
    }
}

/// Format an `EmlTree` in compact `E(...)` notation.
pub fn to_compact_string(tree: &EmlTree) -> String {
    node_to_compact(&tree.root)
}

fn node_to_compact(node: &EmlNode) -> String {
    match node {
        EmlNode::One => "1".to_string(),
        EmlNode::Var(i) => format!("x{i}"),
        EmlNode::Eml { left, right } => {
            format!("E({},{})", node_to_compact(left), node_to_compact(right))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_one() {
        let tree = parse("1").unwrap();
        assert_eq!(tree.size(), 1);
        assert_eq!(*tree.root, EmlNode::One);
    }

    #[test]
    fn test_parse_var() {
        let tree = parse("x0").unwrap();
        assert_eq!(*tree.root, EmlNode::Var(0));
    }

    #[test]
    fn test_parse_eml_e_notation() {
        let tree = parse("E(1, 1)").unwrap();
        assert_eq!(tree.depth(), 1);
        assert_eq!(tree.size(), 3);
    }

    #[test]
    fn test_parse_eml_full_notation() {
        let tree = parse("eml(1, 1)").unwrap();
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn test_parse_nested() {
        let tree = parse("E(E(1, 1), 1)").unwrap();
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn test_parse_no_spaces() {
        let tree = parse("E(E(1,E(1,1)),1)").unwrap();
        assert_eq!(tree.depth(), 3);
    }

    #[test]
    fn test_parse_eval_euler() {
        use crate::eval::EvalCtx;
        let tree = parse("E(1,1)").unwrap();
        let ctx = EvalCtx::new(&[]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!((result - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_parse_eval_exp() {
        use crate::eval::EvalCtx;
        // E(x0, 1) = exp(x0)
        let tree = parse("E(x0, 1)").unwrap();
        let ctx = EvalCtx::new(&[2.0]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!((result - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_parse_eval_ln() {
        use crate::eval::EvalCtx;
        // ln(x) = E(1, E(E(1, x0), 1))
        let tree = parse("E(1, E(E(1, x0), 1))").unwrap();
        let ctx = EvalCtx::new(&[std::f64::consts::E]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_compact() {
        let tree = parse("E(E(1,1),E(x0,1))").unwrap();
        let compact = to_compact_string(&tree);
        assert_eq!(compact, "E(E(1,1),E(x0,1))");
        // Parse again
        let tree2 = parse(&compact).unwrap();
        assert_eq!(tree, tree2);
    }

    #[test]
    fn test_parse_error_empty() {
        assert!(parse("").is_err());
    }

    #[test]
    fn test_parse_error_trailing() {
        assert!(parse("1 1").is_err());
    }

    #[test]
    fn test_parse_error_unmatched() {
        assert!(parse("E(1, 1").is_err());
    }
}
