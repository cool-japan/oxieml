//! GIL smoke tests for the new Python bindings (Round-4 API coverage).
//!
//! Gated to compile only under `--features python` **without**
//! `python-extension`: `--all-features` also turns on `python-extension`
//! (`pyo3/extension-module`), which unlinks libpython, so a test binary that
//! calls `Python::attach` would fail to *link* under `--all-features`. This
//! `not(feature = "python-extension")` guard compiles this whole file out of
//! that gate (an empty test binary, zero tests) while still letting
//! `cargo test --features python` (no `python-extension`) build, link, and run
//! these tests with libpython linked normally.
//!
//! Expression strings here use the crate's actual (minimal) parser grammar —
//! `1 | x<N> | E(a,b) | eml(a,b)` (see `src/parser.rs`) — not infix algebraic
//! notation. `"x0"` is the identity function `f(x) = x`; `"E(x0,1)"` is
//! `eml(x0, 1) = exp(x0) - ln(1) = exp(x0)`.
#![cfg(all(test, feature = "python", not(feature = "python-extension")))]

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::Python;

use oxieml::python::PySymRegConfig;
use oxieml::python::algebra::{
    self, PyMatrix, apart_py, collect_py, expand_py, factor_py, logcombine_py, powsimp_py,
    sum_definite_py, sum_indefinite_py, together_py,
};
use oxieml::python::discovery::{
    self, PyNsga2Config, discover_nsga2_py, discover_pde_py, discover_sindy_py,
};
use oxieml::python::pareto::{discover_pareto_py, pareto_front_py};
use oxieml::python::units::check_units_py;
use oxieml::python::verified::{find_root_verified_py, integrate_definite_verified_py};

#[cfg(feature = "jit")]
use oxieml::python::jit::PyJitFn;

#[cfg(feature = "smt")]
use oxieml::python::smt::{PyIncrementalSolver, max_smt_py, minimize_unsat_core_py};

#[test]
fn matrix_construction_and_det() {
    Python::initialize();
    Python::attach(|_py| {
        let m = PyMatrix::from_f64(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("from_f64");
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        let det = m.det().expect("det");
        assert!(!det.is_empty());

        let identity = PyMatrix::identity(3);
        assert_eq!(identity.nrows(), 3);
        assert!(!identity.is_singular().expect("is_singular"));
        assert_eq!(identity.rank().expect("rank"), 3);

        let transposed = m.transpose();
        assert_eq!(transposed.nrows(), 2);
        let inv = m.inverse().expect("inverse");
        assert_eq!(inv.nrows(), 2);

        let (rows, rank) = m.rref().expect("rref");
        assert_eq!(rank, 2);
        assert_eq!(rows.len(), 2);
    });
}

#[test]
fn algebra_rewrite_and_summation_smoke() {
    Python::initialize();
    Python::attach(|_py| {
        assert!(!expand_py("E(x0,1)").expect("expand_py").is_empty());
        assert!(!factor_py("E(x0,1)").expect("factor_py").is_empty());
        assert!(!collect_py("x0", 0).expect("collect_py").is_empty());
        assert!(!apart_py("x0").expect("apart_py").is_empty());
        assert!(!together_py("x0").expect("together_py").is_empty());
        assert!(!powsimp_py("x0").expect("powsimp_py").is_empty());
        assert!(!logcombine_py("x0").expect("logcombine_py").is_empty());

        // Sigma x0 over k=0 is the triangular-number sum: a genuine closed form.
        let (kind, latex) = sum_indefinite_py("x0", 0).expect("sum_indefinite_py");
        assert_eq!(kind, "closed");
        assert!(latex.is_some());

        let (kind2, latex2) = sum_definite_py("x0", 0, 1.0, 3.0).expect("sum_definite_py");
        assert_eq!(kind2, "closed");
        assert!(latex2.is_some());
    });
}

#[test]
fn units_check_pass_through() {
    Python::initialize();
    Python::attach(|_py| {
        let out = check_units_py("x0", vec![[1i8, 0, 0, 0, 0, 0, 0]]).expect("check_units_py");
        assert_eq!(out, [1i8, 0, 0, 0, 0, 0, 0]);
    });
}

#[test]
fn verified_root_and_quadrature() {
    Python::initialize();
    Python::attach(|_py| {
        // f(x) = x has a unique, certified root at 0 in [-1, 1].
        let cert =
            find_root_verified_py("x0", 0, -1.0, 1.0, vec![], 0).expect("find_root_verified_py");
        assert_eq!(cert.status(), "unique_exists");
        let (lo, hi) = cert.enclosure();
        assert!(lo <= 0.0 && hi >= 0.0);
        let _ = cert.__repr__();

        // Integral 0..1 of x dx = 0.5, guaranteed inside the enclosure.
        let (lo, hi) = integrate_definite_verified_py("x0", 0, 0.0, 1.0, vec![], 0.0, 0)
            .expect("integrate_definite_verified_py");
        assert!(lo <= 0.5 && hi >= 0.5);
    });
}

#[test]
fn nsga2_config_defaults() {
    Python::initialize();
    Python::attach(|_py| {
        let mut cfg = PyNsga2Config::new();
        assert_eq!(cfg.population(), 48);
        assert_eq!(cfg.generations(), 20);
        cfg.set_population(6);
        cfg.set_generations(2);
        assert_eq!(cfg.population(), 6);
        let _ = cfg.__repr__();
    });
}

#[test]
fn nsga2_discover_and_pareto_front() {
    Python::initialize();
    Python::attach(|py| {
        let config = PySymRegConfig::quick();
        let mut nsga2_config = PyNsga2Config::new();
        nsga2_config.set_population(6);
        nsga2_config.set_generations(2);

        let x = PyArray2::from_vec2(py, &[vec![1.0], vec![2.0], vec![3.0], vec![4.0]])
            .expect("x array");
        let y = PyArray1::from_vec(py, vec![2.0, 4.0, 6.0, 8.0]);

        let ranked = discover_nsga2_py(py, &config, &nsga2_config, x.readonly(), y.readonly())
            .expect("discover_nsga2_py");
        assert!(!ranked.is_empty());
        let first = &ranked[0];
        let _ = first.rank();
        let _ = first.crowding();
        let _ = first.formula();
        let _ = first.__repr__();

        let x2 = PyArray2::from_vec2(py, &[vec![1.0], vec![2.0], vec![3.0], vec![4.0]])
            .expect("x2 array");
        let y2 = PyArray1::from_vec(py, vec![2.0, 4.0, 6.0, 8.0]);
        let discovered = discover_pareto_py(py, &config, x2.readonly(), y2.readonly())
            .expect("discover_pareto_py");
        assert!(!discovered.is_empty());
        let front = pareto_front_py(discovered);
        assert!(!front.is_empty());
    });
}

#[test]
fn sindy_discovers_constant_derivative() {
    Python::initialize();
    Python::attach(|_py| {
        let dt = 0.1;
        let n = 20usize;
        // x(t) = t  =>  dx/dt = 1 (constant), library = {"1"}.
        let trajectory = vec![(0..n).map(|i| i as f64 * dt).collect::<Vec<f64>>()];
        let library = vec![("1".to_string(), "1".to_string())];
        let result =
            discover_sindy_py(trajectory, dt, library, 0.05, 1e-6, 20).expect("discover_sindy_py");
        assert_eq!(result.len(), 1);
        let (coefficients, _active_terms, latex) = &result[0];
        assert_eq!(coefficients.len(), 1);
        assert!(!latex.is_empty());
    });
}

#[test]
fn pde_discovery_on_a_trivial_field() {
    Python::initialize();
    Python::attach(|_py| {
        // A spatiotemporally constant field: u_t = u_x = u_xx = 0 everywhere,
        // just verifying the plumbing runs end-to-end without erroring.
        let field = vec![vec![5.0; 6]; 4];
        let result = discover_pde_py(field, 1.0, 1.0, 1e-5, 0.01, 10).expect("discover_pde_py");
        let (_equation, latex, _coefficients, mse) = result;
        assert!(!latex.is_empty());
        assert!(mse.is_finite());
    });
}

#[cfg(feature = "jit")]
#[test]
fn jit_fn_compiles_and_evaluates() {
    Python::initialize();
    Python::attach(|_py| {
        // "E(x0,1)" = exp(x0); exp(0) = 1.
        let f = PyJitFn::new("E(x0,1)", 1).expect("JitFn::new");
        assert_eq!(f.n_vars(), 1);
        let result = f.call(vec![0.0]);
        assert!(
            (result - 1.0).abs() < 1e-9,
            "exp(0) should be 1, got {result}"
        );
        let _ = f.__repr__();
    });
}

#[cfg(feature = "smt")]
#[test]
fn incremental_solver_check_sat() {
    Python::initialize();
    Python::attach(|_py| {
        let mut solver = PyIncrementalSolver::new(vec![(-10.0, 10.0)], 3, None);
        let (status, witness) = solver.check_sat("x0", "eq").expect("check_sat");
        assert_eq!(status, "sat");
        assert!(witness.is_some());

        let (status_all, _witness_all) = solver
            .check_all(vec![("x0".to_string(), "ge".to_string())])
            .expect("check_all");
        assert_eq!(status_all, "sat");
    });
}

#[cfg(feature = "smt")]
#[test]
fn max_smt_and_unsat_core() {
    Python::initialize();
    Python::attach(|_py| {
        let (outcome, witness, _satisfied, _weight, _optimality) = max_smt_py(
            vec![(-10.0, 10.0)],
            3,
            vec![("x0".to_string(), "ge".to_string())],
            vec![("x0".to_string(), "le".to_string())],
            vec![1],
        )
        .expect("max_smt_py");
        assert_eq!(outcome, "solved");
        assert!(witness.is_some());

        let core = minimize_unsat_core_py(
            vec![(-10.0, 10.0)],
            3,
            vec![
                ("x0".to_string(), "gt".to_string()),
                ("x0".to_string(), "lt".to_string()),
            ],
        )
        .expect("minimize_unsat_core_py");
        let (indices, _verified) = core.expect("conjunction x0>0 & x0<0 is unsatisfiable");
        assert_eq!(indices.len(), 2);
    });
}

/// Touch the module paths themselves so a future refactor that accidentally
/// un-`pub`s a submodule fails to compile here rather than silently.
#[test]
fn modules_are_reachable() {
    let _ = std::any::type_name::<algebra::PyMatrix>();
    let _ = std::any::type_name::<discovery::PyNsga2Config>();
}
