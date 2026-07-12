//! R1 — determinism guarantees for parallel symbolic regression.
//!
//! Under `feature = "parallel"` the MCTS simulation phase and the intra-island
//! GA fitness evaluation run on the rayon pool. This suite pins down the three
//! properties that make that safe:
//!
//! 1. **Repeat-run bit-identity** — the same seed produces the same bits, every
//!    time, in every feature configuration.
//! 2. **Thread-count invariance** — the same seed produces the same bits under
//!    1, 2, 3, 8 and 16 rayon worker threads. This is the in-binary, hermetic
//!    form of the `RAYON_NUM_THREADS=1` vs `=8` requirement; it is strictly
//!    stronger, because it also pins the *global* pool's result to the explicit
//!    pools' result.
//! 3. **`parallel` == sequential, bit-for-bit** — the `parallel` and the
//!    non-`parallel` builds of the crate must agree. Feature selection happens at
//!    compile time, so a single test binary cannot hold both. Instead each build
//!    writes a textual fingerprint of its results to `std::env::temp_dir()` and
//!    compares it against the fingerprint left behind by the *other* build. See
//!    [`cross_build_fingerprint`] for the staleness handling.
//!
//! The map-level proof (rayon `par_iter` output == `iter` output, bit for bit)
//! lives in the crate's own unit tests, in `src/symreg/mcts.rs` and
//! `src/symreg/evolution.rs`, where the private `simulate_batch` / `map_fitness`
//! functions are reachable.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::SystemTime;

use oxieml::symreg::{DiscoveredFormula, SymRegConfig, SymRegEngine, SymRegStrategy};

/// Which build produced a fingerprint.
const BUILD_TAG: &str = if cfg!(feature = "parallel") {
    "parallel"
} else {
    "sequential"
};

/// The build tag of the *other* configuration.
const OTHER_BUILD_TAG: &str = if cfg!(feature = "parallel") {
    "sequential"
} else {
    "parallel"
};

// ─────────────────────────────────────────────────────────────────────────────
// Fingerprinting
// ─────────────────────────────────────────────────────────────────────────────

/// Render a formula list as an exact, human-readable bit-level fingerprint.
///
/// Every `f64` is rendered via [`f64::to_bits`] — never via `Display`, which
/// would round away exactly the differences we are hunting for.
fn fingerprint(formulas: &[DiscoveredFormula]) -> String {
    let mut out = String::new();
    out.push_str(&format!("n={}\n", formulas.len()));
    for (i, f) in formulas.iter().enumerate() {
        out.push_str(&format!(
            "[{i}] mse={:016x} score={:016x} aic={:016x} bic={:016x} complexity={} pretty={}\n",
            f.mse.to_bits(),
            f.score.to_bits(),
            f.aic.to_bits(),
            f.bic.to_bits(),
            f.complexity,
            f.pretty,
        ));
        for (j, p) in f.params.iter().enumerate() {
            out.push_str(&format!("[{i}].p{j}={:016x}\n", p.to_bits()));
        }
        if let Some(cv) = f.cv_mse {
            out.push_str(&format!("[{i}].cv={:016x}\n", cv.to_bits()));
        }
    }
    out
}

/// Fingerprint a single formula (the evolutionary strategies return exactly one).
fn fingerprint_one(f: &DiscoveredFormula) -> String {
    fingerprint(std::slice::from_ref(f))
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenarios — all seeded with 42, as the spec requires
// ─────────────────────────────────────────────────────────────────────────────

fn quadratic_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (1..=18).map(|i| vec![i as f64 * 0.35]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0] * x[0]).collect();
    (inputs, targets)
}

fn linear_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let inputs: Vec<Vec<f64>> = (0..16).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| 2.0 * x[0] + 1.0).collect();
    (inputs, targets)
}

/// MCTS: 48 iterations, so the batch loop runs 6 full batches of 8.
fn run_mcts_scenario() -> String {
    let (inputs, targets) = quadratic_data();
    let engine = SymRegEngine::new(SymRegConfig {
        max_depth: 2,
        max_iter: 150,
        num_restarts: 1,
        seed: Some(42),
        strategy: SymRegStrategy::Mcts {
            iterations: 48,
            exploration: 1.0,
        },
        ..SymRegConfig::default()
    });
    fingerprint(
        &engine
            .discover(&inputs, &targets, 1)
            .expect("mcts discover"),
    )
}

/// MCTS with an iteration count that leaves a short trailing batch (45 = 5·8 + 5).
fn run_mcts_ragged_scenario() -> String {
    let (inputs, targets) = quadratic_data();
    let engine = SymRegEngine::new(SymRegConfig {
        max_depth: 2,
        max_iter: 150,
        num_restarts: 1,
        seed: Some(42),
        strategy: SymRegStrategy::Mcts {
            iterations: 45,
            exploration: 1.4,
        },
        ..SymRegConfig::default()
    });
    fingerprint(&engine.discover(&inputs, &targets, 1).expect("ragged mcts"))
}

/// Single-population GA — exercises the intra-island two-phase fitness map.
fn run_evolutionary_scenario() -> String {
    let (inputs, targets) = linear_data();
    let engine = SymRegEngine::new(SymRegConfig {
        max_depth: 2,
        max_iter: 150,
        num_restarts: 1,
        seed: Some(42),
        strategy: SymRegStrategy::Evolutionary {
            population: 20,
            generations: 6,
            tournament_size: 3,
            crossover_rate: 0.7,
            mutation_rate: 0.3,
            elitism: 2,
        },
        ..SymRegConfig::default()
    });
    let formulas = engine.discover(&inputs, &targets, 1).expect("ga discover");
    fingerprint_one(formulas.first().expect("ga returns one formula"))
}

/// Island GA with ring migration — exercises island-level *and* intra-island
/// parallelism at the same time (nested rayon).
fn run_islands_scenario() -> String {
    let (inputs, targets) = linear_data();
    let engine = SymRegEngine::new(SymRegConfig {
        max_depth: 2,
        max_iter: 150,
        num_restarts: 1,
        seed: Some(42),
        strategy: SymRegStrategy::Islands {
            n_islands: 4,
            migration_interval: 2,
            migrants: 2,
            population: 14,
            generations: 6,
            tournament_size: 3,
            crossover_rate: 0.7,
            mutation_rate: 0.3,
            elitism: 1,
        },
        ..SymRegConfig::default()
    });
    let formulas = engine
        .discover(&inputs, &targets, 1)
        .expect("islands discover");
    fingerprint_one(formulas.first().expect("islands returns one formula"))
}

/// A named scenario: a stable id (used for the fingerprint filename) plus the
/// closure that runs it and returns its bit-level fingerprint.
type Scenario = (&'static str, fn() -> String);

/// All scenarios, keyed by a stable id used for the fingerprint filenames.
fn all_scenarios() -> Vec<Scenario> {
    vec![
        ("mcts", run_mcts_scenario as fn() -> String),
        ("mcts_ragged", run_mcts_ragged_scenario as fn() -> String),
        ("evolutionary", run_evolutionary_scenario as fn() -> String),
        ("islands", run_islands_scenario as fn() -> String),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Repeat-run bit-identity (runs in every feature configuration)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn seeded_runs_are_bit_identical_across_repeats() {
    for (name, scenario) in all_scenarios() {
        let first = scenario();
        let second = scenario();
        assert_eq!(
            first, second,
            "scenario `{name}` ({BUILD_TAG} build) is not reproducible across repeats"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Thread-count invariance (parallel build only)
// ─────────────────────────────────────────────────────────────────────────────

/// Every scenario must produce byte-identical output under 1, 2, 3, 8 and 16
/// rayon worker threads, and that output must also match what the ambient global
/// pool (i.e. whatever `RAYON_NUM_THREADS` says) produces.
///
/// This is the direct test of the R1 acceptance criterion "`RAYON_NUM_THREADS=1`
/// vs `8`", hoisted into the binary so it cannot be skipped by forgetting an
/// environment variable.
#[cfg(feature = "parallel")]
#[test]
fn seeded_runs_are_thread_count_invariant() {
    for (name, scenario) in all_scenarios() {
        // The ambient pool honours RAYON_NUM_THREADS.
        let ambient = scenario();

        for threads in [1usize, 2, 3, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool should build");
            let got = pool.install(scenario);
            assert_eq!(
                got, ambient,
                "scenario `{name}` produced different bits with {threads} rayon threads \
                 than with the ambient pool — a float was reduced across tasks"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. parallel == sequential, bit-for-bit, across builds
// ─────────────────────────────────────────────────────────────────────────────

/// Directory holding the cross-build fingerprints.
fn fingerprint_dir() -> PathBuf {
    std::env::temp_dir().join("oxieml_r1_determinism")
}

/// Newest modification time across the crate's `src/` tree, in whole seconds.
///
/// This is the **staleness key**. Both feature configurations compile the *same*
/// source tree, so both compute the same stamp — meaning the `parallel` build
/// will happily compare itself against a `sequential` fingerprint produced from
/// the same sources, in either order, no matter when the two binaries happened to
/// be linked.
///
/// The moment anybody edits a source file the stamp changes, the filename changes
/// with it, and fingerprints from the previous revision simply stop being found.
/// A source edit can therefore never manufacture a phantom "parallel disagrees
/// with sequential" failure — the two builds re-baseline themselves silently.
fn source_stamp() -> Option<u64> {
    fn newest(dir: &PathBuf, best: &mut SystemTime) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let meta = entry.metadata()?;
            if meta.is_dir() {
                newest(&path, best)?;
            } else if path.extension().is_some_and(|e| e == "rs") {
                if let Ok(modified) = meta.modified() {
                    if modified > *best {
                        *best = modified;
                    }
                }
            }
        }
        Ok(())
    }

    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut best = SystemTime::UNIX_EPOCH;
    newest(&src, &mut best).ok()?;
    best.duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Persist this build's fingerprint for `scenario` and, if the other build has
/// left a fingerprint from the *same source revision*, assert the two are
/// byte-identical.
///
/// Running the suite once per feature configuration — which the R1 verification
/// procedure does — therefore compares `parallel` against `sequential` on real,
/// end-to-end `discover()` output. No hard-coded golden values: those would be
/// hostage to the platform's `libm` (`exp`, `ln`) rather than to our own
/// arithmetic, and would fail on a machine with a different libc for reasons
/// that have nothing to do with rayon.
///
/// Returns `true` when a comparison actually took place.
fn cross_build_fingerprint(scenario: &str, value: &str) -> bool {
    let Some(stamp) = source_stamp() else {
        // No readable source tree (e.g. running from a packaged crate): the
        // in-crate `*_parallel_equals_sequential_bitwise` unit tests still cover
        // the parallel-vs-sequential requirement, so there is nothing to do here.
        return false;
    };

    let dir = fingerprint_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        panic!("cannot create fingerprint dir {}: {e}", dir.display());
    }

    // Compare *before* writing, so the two builds cannot race on the same file.
    let other = dir.join(format!("{scenario}.{stamp}.{OTHER_BUILD_TAG}.fp"));
    let compared = match fs::read_to_string(&other) {
        Ok(previous) => {
            assert_eq!(
                previous.trim_end(),
                value.trim_end(),
                "scenario `{scenario}`: the {BUILD_TAG} build disagrees with the \
                 {OTHER_BUILD_TAG} build bit-for-bit.\n\
                 The two builds differ only in whether `simulate_batch` / `map_fitness` / \
                 `map_islands` use `par_iter` or `iter`, so a mismatch means rayon was \
                 allowed to influence the order of a floating-point reduction."
            );
            true
        }
        Err(_) => false,
    };

    let mine = dir.join(format!("{scenario}.{stamp}.{BUILD_TAG}.fp"));
    let mut file = fs::File::create(&mine)
        .unwrap_or_else(|e| panic!("cannot write fingerprint {}: {e}", mine.display()));
    file.write_all(value.as_bytes())
        .unwrap_or_else(|e| panic!("cannot write fingerprint {}: {e}", mine.display()));

    compared
}

#[test]
fn parallel_and_sequential_builds_agree_bit_for_bit() {
    for (name, scenario) in all_scenarios() {
        let compared = cross_build_fingerprint(name, &scenario());
        // Not an assertion: on the *first* of the two feature configurations there
        // is legitimately nothing to compare against yet. Print it so a reviewer
        // running both configs can see the comparison land on the second pass.
        println!(
            "scenario `{name}` [{BUILD_TAG}]: {}",
            if compared {
                format!("compared bit-for-bit against the {OTHER_BUILD_TAG} build — identical")
            } else {
                format!("baseline written; run the {OTHER_BUILD_TAG} build to compare")
            }
        );
    }
}
