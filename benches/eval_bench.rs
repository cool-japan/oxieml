//! Benchmarks for EML tree evaluation.

use criterion::{Criterion, criterion_group, criterion_main};
use oxieml::{Canonical, EmlTree, EvalCtx};

fn bench_eval_exp(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let ctx = EvalCtx::new(&[1.5]);

    c.bench_function("eval_exp", |b| {
        b.iter(|| exp_x.eval_real(&ctx).unwrap());
    });
}

fn bench_eval_ln(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let ln_x = Canonical::ln(&x);
    let ctx = EvalCtx::new(&[2.0]);

    c.bench_function("eval_ln", |b| {
        b.iter(|| ln_x.eval_real(&ctx).unwrap());
    });
}

fn bench_eval_euler(c: &mut Criterion) {
    let e = Canonical::euler();
    let ctx = EvalCtx::new(&[]);

    c.bench_function("eval_euler", |b| {
        b.iter(|| e.eval_real(&ctx).unwrap());
    });
}

fn bench_eval_neg(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let neg_x = Canonical::neg(&x);
    let ctx = EvalCtx::new(&[3.0]);

    c.bench_function("eval_neg", |b| {
        b.iter(|| neg_x.eval_real(&ctx).unwrap());
    });
}

fn bench_eval_batch(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let data: Vec<Vec<f64>> = (0..1000).map(|i| vec![i as f64 * 0.01]).collect();

    c.bench_function("eval_batch_1000", |b| {
        b.iter(|| exp_x.eval_batch(&data).unwrap());
    });
}

fn bench_eval_batch_10000(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let data: Vec<Vec<f64>> = (0..10_000).map(|i| vec![i as f64 * 0.001]).collect();

    c.bench_function("eval_batch_10000_sequential", |b| {
        b.iter(|| exp_x.eval_batch(&data).unwrap());
    });
}

#[cfg(feature = "parallel")]
fn bench_eval_batch_10000_parallel(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let data: Vec<Vec<f64>> = (0..10_000).map(|i| vec![i as f64 * 0.001]).collect();

    c.bench_function("eval_batch_10000_parallel", |b| {
        b.iter(|| exp_x.eval_batch(&data).unwrap());
    });
}

#[cfg(feature = "parallel")]
fn bench_symreg_discover_parallel(c: &mut Criterion) {
    use oxieml::{SymRegConfig, SymRegEngine};

    let inputs: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.2]).collect();
    let targets: Vec<f64> = inputs.iter().map(|x| x[0].exp()).collect();
    let config = SymRegConfig {
        max_depth: 2,
        learning_rate: 1e-2,
        tolerance: 1e-5,
        max_iter: 500,
        complexity_penalty: 1e-4,
        num_restarts: 1,
        integer_rounding: false,
    };

    c.bench_function("symreg_discover_parallel", |b| {
        b.iter(|| {
            let engine = SymRegEngine::new(config.clone());
            engine.discover(&inputs, &targets, 1).unwrap()
        });
    });
}

fn bench_lower_and_eval(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let lowered = exp_x.lower();

    c.bench_function("lowered_eval_exp", |b| {
        b.iter(|| lowered.eval(&[1.5]));
    });
}

fn bench_tree_construction(c: &mut Criterion) {
    c.bench_function("construct_neg", |b| {
        b.iter(|| {
            let x = EmlTree::var(0);
            Canonical::neg(&x)
        });
    });
}

fn bench_lowered_eval_batch_10000_scalar(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let lowered = exp_x.lower();
    let data: Vec<Vec<f64>> = (0..10_000).map(|i| vec![i as f64 * 0.001]).collect();

    c.bench_function("lowered_eval_batch_10000_scalar", |b| {
        b.iter(|| lowered.eval_batch_scalar(&data));
    });
}

#[cfg(feature = "simd")]
fn bench_lowered_eval_batch_10000_simd(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let lowered = exp_x.lower();
    let data: Vec<Vec<f64>> = (0..10_000).map(|i| vec![i as f64 * 0.001]).collect();

    c.bench_function("lowered_eval_batch_10000_simd", |b| {
        b.iter(|| lowered.eval_batch(&data));
    });
}

#[cfg(all(feature = "simd", feature = "parallel"))]
fn bench_lowered_eval_batch_100000_simd_parallel(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let exp_x = Canonical::exp(&x);
    let lowered = exp_x.lower();
    let data: Vec<Vec<f64>> = (0..100_000).map(|i| vec![i as f64 * 0.0001]).collect();

    c.bench_function("lowered_eval_batch_100000_simd_parallel", |b| {
        b.iter(|| lowered.eval_batch(&data));
    });
}

criterion_group!(
    benches,
    bench_eval_exp,
    bench_eval_ln,
    bench_eval_euler,
    bench_eval_neg,
    bench_eval_batch,
    bench_eval_batch_10000,
    bench_lower_and_eval,
    bench_tree_construction,
    bench_lowered_eval_batch_10000_scalar,
);

#[cfg(feature = "parallel")]
criterion_group!(
    parallel_benches,
    bench_eval_batch_10000_parallel,
    bench_symreg_discover_parallel,
);

#[cfg(feature = "simd")]
criterion_group!(simd_benches, bench_lowered_eval_batch_10000_simd,);

#[cfg(all(feature = "simd", feature = "parallel"))]
criterion_group!(
    simd_parallel_benches,
    bench_lowered_eval_batch_100000_simd_parallel,
);

#[cfg(all(feature = "simd", feature = "parallel"))]
criterion_main!(
    benches,
    parallel_benches,
    simd_benches,
    simd_parallel_benches
);

#[cfg(all(feature = "simd", not(feature = "parallel")))]
criterion_main!(benches, simd_benches);

#[cfg(all(not(feature = "simd"), feature = "parallel"))]
criterion_main!(benches, parallel_benches);

#[cfg(all(not(feature = "simd"), not(feature = "parallel")))]
criterion_main!(benches);
