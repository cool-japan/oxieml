//! Tests for `to_oxiblas_ops_shared`, `eval_ops_shared`, and the strict-generalisation
//! invariant between the sharing-aware emitter and the original tree-walk emitter.

use oxieml::LoweredOp;
use oxieml::lower::OxiOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// 1. Strict-generalisation invariant (no-sharing tree → byte-identical op vec)
// ---------------------------------------------------------------------------

#[test]
fn strict_generalisation_no_sharing_tree() {
    // Build a pure tree with NO shared Arcs.
    let tree = LoweredOp::Add(
        Arc::new(LoweredOp::Mul(
            Arc::new(LoweredOp::Var(0)),
            Arc::new(LoweredOp::Const(2.0)),
        )),
        Arc::new(LoweredOp::Sin(Arc::new(LoweredOp::Var(1)))),
    );
    let plain = tree.to_oxiblas_ops();
    let (shared, n_slots) = tree.to_oxiblas_ops_shared();

    assert_eq!(n_slots, 0, "pure tree should have zero slots");
    assert_eq!(
        plain, shared,
        "pure tree: to_oxiblas_ops_shared must be byte-identical to to_oxiblas_ops"
    );

    // Verify no Store/Load in output.
    let has_store_load = shared
        .iter()
        .any(|op| matches!(op, OxiOp::Store(_) | OxiOp::Load(_)));
    assert!(
        !has_store_load,
        "pure tree should produce no Store/Load opcodes"
    );
}

// ---------------------------------------------------------------------------
// 2. Behavioural parity for a shared DAG
// ---------------------------------------------------------------------------

#[test]
fn shared_dag_eval_parity() {
    // inner = Exp(Var(0)) — a single Arc shared in both positions of Add.
    let inner = Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Var(0))));
    let tree = LoweredOp::Add(Arc::clone(&inner), Arc::clone(&inner));

    let (ops, n_slots) = tree.to_oxiblas_ops_shared();
    for i in 0..10 {
        let x = i as f64 * 0.4 - 1.0;
        let vars = [x];
        let expected = tree.eval(&vars);
        let got = LoweredOp::eval_ops_shared(&ops, &vars, n_slots);
        assert!(
            (got - expected).abs() < 1e-14,
            "eval_ops_shared parity at x={x}: expected {expected}, got {got}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Shared node → exactly one Store + one Load, n_slots == 1
// ---------------------------------------------------------------------------

#[test]
fn shared_node_produces_store_and_load() {
    let inner = Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Var(0))));
    let tree = LoweredOp::Add(Arc::clone(&inner), Arc::clone(&inner));

    let (ops, n_slots) = tree.to_oxiblas_ops_shared();
    assert_eq!(n_slots, 1, "one shared node → n_slots == 1");

    let store_count = ops
        .iter()
        .filter(|op| matches!(op, OxiOp::Store(_)))
        .count();
    let load_count = ops.iter().filter(|op| matches!(op, OxiOp::Load(_))).count();
    assert_eq!(store_count, 1, "should emit exactly one Store");
    assert_eq!(load_count, 1, "should emit exactly one Load");
}

// ---------------------------------------------------------------------------
// 4. Fewer ops than naively-flattened for a 3x-reused subtree
// ---------------------------------------------------------------------------

#[test]
fn shared_dag_fewer_ops_than_flat() {
    // Build a tree reusing the same Mul(Var(0), Var(1)) subtree three times.
    let inner = Arc::new(LoweredOp::Mul(
        Arc::new(LoweredOp::Var(0)),
        Arc::new(LoweredOp::Var(1)),
    ));
    // Add(Add(inner, inner), inner)
    let tree = LoweredOp::Add(
        Arc::new(LoweredOp::Add(Arc::clone(&inner), Arc::clone(&inner))),
        Arc::clone(&inner),
    );

    let flat = tree.to_oxiblas_ops();
    let (shared, _n_slots) = tree.to_oxiblas_ops_shared();

    assert!(
        shared.len() < flat.len(),
        "shared-aware codegen should produce fewer ops ({}) than flat tree-walk ({})",
        shared.len(),
        flat.len()
    );
}

// ---------------------------------------------------------------------------
// 5. eval_ops_shared with n_slots=0 matches eval_ops (no Store/Load path)
// ---------------------------------------------------------------------------

#[test]
fn eval_ops_shared_zero_slots_matches_eval_ops() {
    // Build a pure tree so to_oxiblas_ops_shared returns n_slots=0.
    let tree = LoweredOp::Add(
        Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Var(0)))),
        Arc::new(LoweredOp::Const(1.0)),
    );
    let plain_ops = tree.to_oxiblas_ops();
    let (shared_ops, n_slots) = tree.to_oxiblas_ops_shared();
    assert_eq!(n_slots, 0);
    assert_eq!(plain_ops, shared_ops);

    for i in 0..8 {
        let x = i as f64 * 0.25;
        let vars = [x];
        let from_eval_ops = LoweredOp::eval_ops(&plain_ops, &vars);
        let from_shared = LoweredOp::eval_ops_shared(&shared_ops, &vars, n_slots);
        assert_eq!(
            from_eval_ops.to_bits(),
            from_shared.to_bits(),
            "eval_ops vs eval_ops_shared mismatch at x={x}"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Deep shared DAG — eval_ops_shared correct for nested sharing
// ---------------------------------------------------------------------------

#[test]
fn nested_sharing_eval_correct() {
    // leaf = Var(0)
    // level1 = Add(leaf, leaf)   (sharing leaf)
    // level2 = Mul(level1, level1)  (sharing level1)
    let leaf = Arc::new(LoweredOp::Var(0));
    let level1 = Arc::new(LoweredOp::Add(Arc::clone(&leaf), Arc::clone(&leaf)));
    let tree = LoweredOp::Mul(Arc::clone(&level1), Arc::clone(&level1));

    let (ops, n_slots) = tree.to_oxiblas_ops_shared();
    for i in 0..5 {
        let x = i as f64 + 1.0;
        let vars = [x];
        let expected = tree.eval(&vars);
        let got = LoweredOp::eval_ops_shared(&ops, &vars, n_slots);
        assert!(
            (got - expected).abs() < 1e-14,
            "nested sharing eval at x={x}: expected {expected}, got {got}"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Strict-generalisation invariant from eval side:
//    eval_ops_shared on a pure-tree op-vec (n_slots=0) == eval_ops
// ---------------------------------------------------------------------------

#[test]
fn eval_ops_shared_strict_generalisation() {
    // More complex pure tree: Tanh(Add(Var(0), Const(-1.0)))
    let tree = LoweredOp::Tanh(Arc::new(LoweredOp::Add(
        Arc::new(LoweredOp::Var(0)),
        Arc::new(LoweredOp::Const(-1.0)),
    )));
    let ops = tree.to_oxiblas_ops();
    let (shared_ops, n_slots) = tree.to_oxiblas_ops_shared();
    assert_eq!(n_slots, 0, "pure tree should have zero slots");
    assert_eq!(ops, shared_ops, "byte-identical to plain ops");

    let vars = [0.5];
    let plain = LoweredOp::eval_ops(&ops, &vars);
    let shared = LoweredOp::eval_ops_shared(&shared_ops, &vars, n_slots);
    assert_eq!(
        plain.to_bits(),
        shared.to_bits(),
        "eval_ops vs eval_ops_shared (n_slots=0) mismatch"
    );
}
