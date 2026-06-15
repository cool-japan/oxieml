use oxieml::{SymRegConfig, SymRegEngine};

#[test]
fn ode_numerical_derivative_exponential() {
    // x(t) = e^t, dx/dt = e^t
    let dt = 0.01_f64;
    let ts: Vec<f64> = (0..100).map(|i| i as f64 * dt).collect();
    let x: Vec<f64> = ts.iter().map(|&t| t.exp()).collect();

    // central difference at t=0.5 (index 50): should be close to e^0.5
    let dx = (x[51] - x[49]) / (2.0 * dt);
    let expected = (0.5_f64).exp();
    assert!(
        (dx - expected).abs() < 1e-4,
        "deriv={dx}, expected={expected}"
    );
}

#[test]
fn discover_ode_exponential_growth() {
    // x(t) = e^t → dx/dt = x
    let dt = 0.05_f64;
    let ts: Vec<f64> = (0..60).map(|i| i as f64 * dt).collect();
    let x: Vec<f64> = ts.iter().map(|&t| t.exp()).collect();

    let mut config = SymRegConfig::quick();
    config.max_depth = 2;
    config.seed = Some(42);

    let engine = SymRegEngine::new(config);
    let result = engine.discover_ode(&[x], dt).expect("discover_ode failed");

    assert_eq!(result.len(), 1, "one output per state var");
    assert!(!result[0].is_empty(), "should find at least one formula");
    // Best formula should have low MSE (dx/dt ≈ x)
    let best_mse = result[0]
        .iter()
        .map(|f| f.mse)
        .fold(f64::INFINITY, f64::min);
    assert!(best_mse < 1.0, "best MSE should be reasonable: {best_mse}");
}

#[ignore = "heavy: slow integration test, run manually"]
#[test]
fn discover_ode_two_state_variables() {
    // Simple harmonic oscillator: dx/dt = v, dv/dt = -x
    let dt = 0.05_f64;
    let ts: Vec<f64> = (0..80).map(|i| i as f64 * dt).collect();
    let x: Vec<f64> = ts.iter().map(|&t| t.cos()).collect();
    let v: Vec<f64> = ts.iter().map(|&t| -t.sin()).collect();

    let mut config = SymRegConfig::quick();
    config.max_depth = 2;
    config.seed = Some(7);

    let engine = SymRegEngine::new(config);
    let result = engine
        .discover_ode(&[x, v], dt)
        .expect("discover_ode 2d failed");

    assert_eq!(result.len(), 2, "two outputs for two state vars");
    assert!(!result[0].is_empty());
    assert!(!result[1].is_empty());
}

#[test]
fn discover_ode_rejects_empty_trajectory() {
    let config = SymRegConfig::quick();
    let engine = SymRegEngine::new(config);
    assert!(engine.discover_ode(&[], 0.01).is_err());
}

#[test]
fn discover_ode_with_sg_window() {
    let dt = 0.05_f64;
    let ts: Vec<f64> = (0..80).map(|i| i as f64 * dt).collect();
    let x: Vec<f64> = ts.iter().map(|&t| t.exp()).collect();

    let mut config = SymRegConfig::quick();
    config.max_depth = 2;
    config.ode_sg_window = Some(5);
    config.seed = Some(1);

    let engine = SymRegEngine::new(config);
    let result = engine
        .discover_ode(&[x], dt)
        .expect("discover_ode with SG failed");
    assert!(!result[0].is_empty());
}
