use oxieml::LoweredOp;
use std::sync::Arc;

#[test]
fn arc_clone_shares_structure_not_deep_copy() {
    // Build a shared inner subtree
    let inner = Arc::new(LoweredOp::Var(0));
    assert_eq!(Arc::strong_count(&inner), 1, "fresh Arc has refcount 1");

    // Reference inner from two branches — this is the structural sharing
    let expr = LoweredOp::Add(Arc::clone(&inner), Arc::clone(&inner));
    // inner is now referenced: the original handle + the two inside Add = 3
    assert!(
        Arc::strong_count(&inner) >= 3,
        "inner should be shared by both Add branches; strong_count={}",
        Arc::strong_count(&inner)
    );

    // Cloning the outer LoweredOp should bump inner's refcount (not deep-copy)
    let expr2 = expr.clone();
    assert!(
        Arc::strong_count(&inner) >= 5,
        "after cloning the outer Add, inner should have strong_count >= 5; got {}",
        Arc::strong_count(&inner)
    );
    drop(expr2);
    // After drop, refcount should go back
    assert!(
        Arc::strong_count(&inner) >= 3,
        "after dropping cloned expr, inner should still have strong_count >= 3; got {}",
        Arc::strong_count(&inner)
    );
}
