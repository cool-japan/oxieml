//! K-fold cross-validation for symbolic regression.
//!
//! Provides [`k_fold_cv`] which evaluates a candidate formula's generalisation
//! error by refitting on `k-1` folds and scoring on the held-out fold.

use crate::eval::EvalCtx;
use crate::grad::ParameterizedEmlTree;
use crate::tree::EmlTree;

use super::SymRegEngine;
use super::topology::compute_mse_direct;

/// Compute average held-out MSE over `k` contiguous folds.
///
/// Refits the given topology on the training portion of each fold using a
/// condensed Adam loop (budget = `engine.config.max_iter / k` steps), then
/// evaluates on the held-out slice.
///
/// When the data is too small (`n < 2` or `k <= 1`) the function falls back
/// to full-data MSE via [`compute_mse_direct`].
pub(super) fn k_fold_cv(
    engine: &SymRegEngine,
    topology: &EmlTree,
    params: &[f64],
    inputs: &[Vec<f64>],
    targets: &[f64],
    k: usize,
) -> f64 {
    let n = inputs.len();

    if n < 2 || k <= 1 {
        return compute_mse_direct(topology, inputs, targets).unwrap_or(f64::INFINITY);
    }

    let fold_iters = (engine.config.max_iter / k).clamp(1, 200);
    let lr = engine.config.learning_rate;
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let epsilon = 1e-8_f64;

    let mut total_cv_mse = 0.0;
    let mut valid_folds = 0usize;

    for fold in 0..k {
        let fold_start = (fold * n) / k;
        let fold_end = ((fold + 1) * n) / k;

        if fold_start >= fold_end {
            continue;
        }

        let train_inputs: Vec<&Vec<f64>> = inputs[..fold_start]
            .iter()
            .chain(inputs[fold_end..].iter())
            .collect();
        let train_targets: Vec<f64> = targets[..fold_start]
            .iter()
            .chain(targets[fold_end..].iter())
            .copied()
            .collect();
        let test_inputs = &inputs[fold_start..fold_end];
        let test_targets = &targets[fold_start..fold_end];

        if train_inputs.is_empty() || test_inputs.is_empty() {
            continue;
        }

        let mut ptree = ParameterizedEmlTree::from_topology(topology, 1.0);
        if ptree.params.len() == params.len() {
            ptree.params.clone_from_slice(params);
        }

        let n_params = ptree.num_params();

        if n_params > 0 {
            let mut m = vec![0.0_f64; n_params];
            let mut v = vec![0.0_f64; n_params];

            for t in 1..=fold_iters {
                let mut total_grads = vec![0.0_f64; n_params];
                let mut valid_count = 0usize;

                for (input, &target) in train_inputs.iter().zip(train_targets.iter()) {
                    let ctx = EvalCtx::new(input);
                    match ptree.forward_backward(&ctx, target) {
                        Ok((loss, grads)) if loss.is_finite() => {
                            for (tg, g) in total_grads.iter_mut().zip(&grads) {
                                if g.is_finite() {
                                    *tg += g;
                                }
                            }
                            valid_count += 1;
                        }
                        _ => {}
                    }
                }

                if valid_count == 0 {
                    break;
                }

                let n_f = valid_count as f64;
                for i in 0..n_params {
                    let g = total_grads[i] / n_f;
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                    let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                    let v_hat = v[i] / (1.0 - beta2.powi(t as i32));
                    ptree.params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        }

        let held_out_mse = if n_params == 0 {
            let test_slices: Vec<&Vec<f64>> = test_inputs.iter().collect();
            let mut total = 0.0;
            let mut cnt = 0usize;
            for (input, &target) in test_slices.iter().zip(test_targets) {
                let ctx = EvalCtx::new(input);
                if let Ok(val) = topology.eval_real(&ctx) {
                    if val.is_finite() {
                        total += (val - target).powi(2);
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                None
            } else {
                Some(total / cnt as f64)
            }
        } else {
            let test_input_vecs: Vec<&Vec<f64>> = test_inputs.iter().collect();
            let mut total = 0.0;
            let mut cnt = 0usize;
            for (input, &target) in test_input_vecs.iter().zip(test_targets) {
                let ctx = EvalCtx::new(input);
                if let Ok(val) = ptree.forward(&ctx) {
                    if val.is_finite() {
                        total += (val - target).powi(2);
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                None
            } else {
                Some(total / cnt as f64)
            }
        };

        if let Some(mse) = held_out_mse {
            total_cv_mse += mse;
            valid_folds += 1;
        }
    }

    if valid_folds == 0 {
        compute_mse_direct(topology, inputs, targets).unwrap_or(f64::INFINITY)
    } else {
        total_cv_mse / valid_folds as f64
    }
}
