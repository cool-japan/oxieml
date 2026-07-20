//! Python bindings for JIT-compiled batch evaluation (`jit` feature).
//!
//! Thin wrapper over [`crate::JitFn`]: compiles one expression to native code
//! via Cranelift, then evaluates it at a single point or over a whole numpy
//! array of rows in one native call.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// A JIT-compiled scalar/batch function over one expression, via Cranelift.
#[pyclass(name = "JitFn")]
pub struct PyJitFn {
    inner: crate::JitFn,
}

#[pymethods]
impl PyJitFn {
    /// Compile `expr_str` to native code.
    ///
    /// `n_vars` is the number of variable slots; it is automatically raised to
    /// cover every `Var(i)` actually referenced, so under-reporting it is safe.
    #[new]
    pub fn new(expr_str: &str, n_vars: usize) -> PyResult<Self> {
        let tree = crate::parse(expr_str).map_err(map_err)?;
        let lowered = tree.lower().simplify();
        let ops = lowered.to_oxiblas_ops();
        let inner = crate::JitFn::compile(&ops, n_vars).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Evaluate at a single point.
    pub fn call(&self, vars: Vec<f64>) -> f64 {
        self.inner.call(&vars)
    }

    /// Minimum number of variable slots this function reads.
    #[getter]
    pub fn n_vars(&self) -> usize {
        self.inner.n_vars()
    }

    /// Evaluate a numpy `(n_rows, n_vars)` 2-D array of rows in one native
    /// call, returning a 1-D numpy array of `n_rows` results.
    ///
    /// Bit-exact (0 ULP) with calling [`Self::call`] once per row. The GIL is
    /// released for the duration of the native call.
    pub fn call_batch<'py>(
        &self,
        py: Python<'py>,
        rows: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let arr = rows.as_array();
        let n_rows = arr.nrows();
        let n_vars = arr.ncols();
        let flat: Vec<f64> = arr.iter().copied().collect();
        let mut out = vec![0.0f64; n_rows];
        py.detach(|| self.inner.call_batch(&flat, n_rows, n_vars, &mut out))
            .map_err(map_err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Rayon-parallel [`Self::call_batch`] (requires the `parallel` feature).
    ///
    /// Identical (bit-exact) results, split across threads by contiguous row
    /// chunks; no floating-point value is ever combined across threads.
    #[cfg(feature = "parallel")]
    pub fn call_batch_parallel<'py>(
        &self,
        py: Python<'py>,
        rows: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let arr = rows.as_array();
        let n_rows = arr.nrows();
        let n_vars = arr.ncols();
        let flat: Vec<f64> = arr.iter().copied().collect();
        let mut out = vec![0.0f64; n_rows];
        py.detach(|| {
            self.inner
                .call_batch_parallel(&flat, n_rows, n_vars, &mut out)
        })
        .map_err(map_err)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!("JitFn(n_vars={})", self.inner.n_vars())
    }
}
