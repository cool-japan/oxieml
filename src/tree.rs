//! EML tree data structures.
//!
//! The core representation: uniform binary trees where every internal node
//! is the EML operator `eml(x, y) = exp(x) - ln(y)` and leaves are either
//! the constant `1` or input variables.

use std::fmt;
use std::sync::Arc;

/// EML tree node. All nodes share the same type — uniform binary tree.
/// `Arc` enables O(1) subtree sharing during symbolic regression.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EmlNode {
    /// Constant 1 (the only constant in the paper's grammar).
    One,

    /// Free floating-point constant (activated when `SymRegConfig.enable_const_leaf = true`).
    /// Not part of the base EML grammar; used during symbolic regression to represent
    /// learnable constants that are later snapped to named constants.
    Const(f64),

    /// Input variable referenced by index: x0, x1, ...
    Var(usize),

    /// `eml(left, right) = exp(left) - ln(right)`
    Eml {
        /// Left subtree (argument to exp).
        left: Arc<EmlNode>,
        /// Right subtree (argument to ln).
        right: Arc<EmlNode>,
    },
}

/// EML tree with metadata.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub struct EmlTree {
    /// Root node of the tree.
    pub root: Arc<EmlNode>,
    /// Number of distinct variables used in the tree.
    num_vars: usize,
}

impl EmlTree {
    /// Create a tree representing the constant 1.
    pub fn one() -> Self {
        Self {
            root: Arc::new(EmlNode::One),
            num_vars: 0,
        }
    }

    /// Create a tree representing variable `x_index`.
    pub fn var(index: usize) -> Self {
        Self {
            root: Arc::new(EmlNode::Var(index)),
            num_vars: index + 1,
        }
    }

    /// Create a tree representing `eml(left, right) = exp(left) - ln(right)`.
    pub fn eml(left: &EmlTree, right: &EmlTree) -> Self {
        Self {
            root: Arc::new(EmlNode::Eml {
                left: Arc::clone(&left.root),
                right: Arc::clone(&right.root),
            }),
            num_vars: left.num_vars.max(right.num_vars),
        }
    }

    /// Construct an `EmlTree` from a raw `Arc<EmlNode>`.
    pub fn from_node(node: Arc<EmlNode>) -> Self {
        let num_vars = count_vars(&node);
        Self {
            root: node,
            num_vars,
        }
    }

    /// Create a tree with a free constant leaf (active only when `SymRegConfig.enable_const_leaf = true`).
    pub fn const_val(v: f64) -> Self {
        Self {
            root: Arc::new(EmlNode::Const(v)),
            num_vars: 0,
        }
    }

    /// Count `Const` leaves in the tree.
    pub fn count_const_leaves(&self) -> usize {
        count_const_in_node(&self.root)
    }

    /// Number of distinct variables referenced.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Depth of the tree (leaves have depth 0).
    pub fn depth(&self) -> usize {
        node_depth(&self.root)
    }

    /// Total number of nodes in the tree.
    pub fn size(&self) -> usize {
        node_size(&self.root)
    }

    /// Iterate over all nodes in post-order (left, right, parent).
    pub fn iter_postorder(&self) -> PostOrderIter<'_> {
        let mut nodes = Vec::new();
        collect_postorder(&self.root, &mut nodes);
        PostOrderIter { nodes, index: 0 }
    }
}

impl PartialEq for EmlTree {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

impl fmt::Display for EmlTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_node(&self.root, f)
    }
}

impl fmt::Display for EmlNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_node(self, f)
    }
}

fn write_node(node: &EmlNode, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match node {
        EmlNode::One => write!(f, "1"),
        EmlNode::Const(v) => write!(f, "{v:.6}"),
        EmlNode::Var(i) => write!(f, "x{i}"),
        EmlNode::Eml { left, right } => {
            write!(f, "eml(")?;
            write_node(left, f)?;
            write!(f, ", ")?;
            write_node(right, f)?;
            write!(f, ")")
        }
    }
}

/// Post-order iterator over `EmlNode` references.
pub struct PostOrderIter<'a> {
    nodes: Vec<&'a EmlNode>,
    index: usize,
}

impl<'a> Iterator for PostOrderIter<'a> {
    type Item = &'a EmlNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.nodes.len() {
            let node = self.nodes[self.index];
            self.index += 1;
            Some(node)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.nodes.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PostOrderIter<'_> {}

fn collect_postorder<'a>(node: &'a EmlNode, out: &mut Vec<&'a EmlNode>) {
    match node {
        EmlNode::Eml { left, right } => {
            collect_postorder(left, out);
            collect_postorder(right, out);
        }
        EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => {}
    }
    out.push(node);
}

fn node_depth(node: &EmlNode) -> usize {
    match node {
        EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
        EmlNode::Eml { left, right } => 1 + node_depth(left).max(node_depth(right)),
    }
}

fn node_size(node: &EmlNode) -> usize {
    match node {
        EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 1,
        EmlNode::Eml { left, right } => 1 + node_size(left) + node_size(right),
    }
}

fn count_vars(node: &EmlNode) -> usize {
    match node {
        EmlNode::One => 0,
        EmlNode::Const(_) => 0,
        EmlNode::Var(i) => i + 1,
        EmlNode::Eml { left, right } => count_vars(left).max(count_vars(right)),
    }
}

fn count_const_in_node(node: &EmlNode) -> usize {
    match node {
        EmlNode::Const(_) => 1,
        EmlNode::One | EmlNode::Var(_) => 0,
        EmlNode::Eml { left, right } => count_const_in_node(left) + count_const_in_node(right),
    }
}

#[cfg(feature = "serde")]
impl EmlTree {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to binary using `oxicode`.
    pub fn to_binary(&self) -> Result<Vec<u8>, oxicode::Error> {
        oxicode::serde::encode_serde(self)
    }

    /// Deserialize from binary bytes encoded with [`Self::to_binary`].
    pub fn from_binary(bytes: &[u8]) -> Result<Self, oxicode::Error> {
        oxicode::serde::decode_serde(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one() {
        let t = EmlTree::one();
        assert_eq!(t.depth(), 0);
        assert_eq!(t.size(), 1);
        assert_eq!(t.num_vars(), 0);
        assert_eq!(t.to_string(), "1");
    }

    #[test]
    fn test_var() {
        let t = EmlTree::var(0);
        assert_eq!(t.depth(), 0);
        assert_eq!(t.size(), 1);
        assert_eq!(t.num_vars(), 1);
        assert_eq!(t.to_string(), "x0");
    }

    #[test]
    fn test_eml_basic() {
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let t = EmlTree::eml(&x, &one);
        assert_eq!(t.depth(), 1);
        assert_eq!(t.size(), 3);
        assert_eq!(t.num_vars(), 1);
        assert_eq!(t.to_string(), "eml(x0, 1)");
    }

    #[test]
    fn test_postorder() {
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let t = EmlTree::eml(&x, &one);
        let nodes: Vec<_> = t.iter_postorder().collect();
        assert_eq!(nodes.len(), 3);
        assert_eq!(*nodes[0], EmlNode::Var(0));
        assert_eq!(*nodes[1], EmlNode::One);
        assert!(matches!(nodes[2], EmlNode::Eml { .. }));
    }

    #[test]
    fn test_nested_depth() {
        // eml(eml(1, 1), eml(x0, 1)) → depth 2
        let one = EmlTree::one();
        let x = EmlTree::var(0);
        let inner_l = EmlTree::eml(&one, &one);
        let inner_r = EmlTree::eml(&x, &one);
        let outer = EmlTree::eml(&inner_l, &inner_r);
        assert_eq!(outer.depth(), 2);
        assert_eq!(outer.size(), 7);
    }

    #[test]
    fn test_const_leaf() {
        let c = EmlTree::const_val(3.7);
        assert_eq!(c.depth(), 0);
        assert_eq!(c.size(), 1);
        assert_eq!(c.num_vars(), 0);
        assert!(c.to_string().starts_with("3.7"));
    }
}
