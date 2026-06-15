//! Integration tests for SharedTopology multi-output symbolic regression (D3).

#[test]
fn test_shared_topology_does_not_panic_single_output() {
    let features: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
    let targets: Vec<f64> = features.iter().map(|row| row[0] * 2.0 + 1.0).collect();

    let mut config = oxieml::SymRegConfig::quick();
    config.multi_output_strategy = oxieml::MultiOutputStrategy::SharedTopology;
    config.seed = Some(42);
    let engine = oxieml::SymRegEngine::new(config);

    let targets_2d = vec![targets];
    let result = engine.discover_multi(&features, &targets_2d, 1);
    assert!(
        result.is_ok(),
        "SharedTopology must not error on single output"
    );
}

#[ignore = "heavy: slow integration test, run manually"]
#[test]
fn test_shared_topology_two_outputs_returns_correct_shape() {
    let features: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.1]).collect();
    let target1: Vec<f64> = features.iter().map(|row| row[0] * 2.0).collect();
    let target2: Vec<f64> = features.iter().map(|row| row[0] * 3.0).collect();

    let mut config = oxieml::SymRegConfig::quick();
    config.multi_output_strategy = oxieml::MultiOutputStrategy::SharedTopology;
    config.seed = Some(99);
    let engine = oxieml::SymRegEngine::new(config);

    let targets_2d = vec![target1, target2];
    let result = engine
        .discover_multi(&features, &targets_2d, 1)
        .expect("SharedTopology discover_multi should succeed");

    // discover_multi returns Vec<Vec<DiscoveredFormula>>, one Vec per output.
    assert_eq!(
        result.len(),
        2,
        "should return one formula list per output (2 outputs)"
    );
}

#[test]
fn test_shared_topology_empty_inputs_returns_error() {
    let mut config = oxieml::SymRegConfig::quick();
    config.multi_output_strategy = oxieml::MultiOutputStrategy::SharedTopology;
    let engine = oxieml::SymRegEngine::new(config);

    let result = engine.discover_multi(&[], &[vec![1.0, 2.0]], 1);
    assert!(
        result.is_err(),
        "empty inputs should return EmlError::EmptyData"
    );
}

#[test]
fn test_shared_formula_struct_is_accessible() {
    // Verify SharedFormula is pub-exported from lib.rs
    let _: Option<oxieml::SharedFormula> = None;
}
