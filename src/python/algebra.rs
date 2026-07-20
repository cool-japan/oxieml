//! Python bindings for algebraic rewriting, summation, and symbolic matrices.
//!
//! Thin wrappers over [`crate::rewrite`], [`crate::summation`], and
//! [`crate::matrix`]. Rewrite/summation operate on expression strings and
//! return LaTeX; [`PyMatrix`] wraps [`crate::matrix::Matrix`].

use pyo3::prelude::*;

use crate::matrix::Matrix;

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// One eigenpair as `(value_re, value_im, vector, residual)`, where `vector`
/// is a list of `(re, im)` pairs. See [`PyMatrix::eigenvectors`].
type EigenpairPy = (f64, f64, Vec<(f64, f64)>, f64);

/// Parse `expr_str` into a lowered, simplified expression tree.
fn parse_lowered(expr_str: &str) -> PyResult<crate::LoweredOp> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    Ok(tree.lower().simplify())
}

/// Expand products and integer powers into a canonical sum of monomials.
#[pyfunction]
pub fn expand_py(expr_str: &str) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.expand().to_latex())
}

/// Factor a univariate polynomial into irreducibles over ℚ.
#[pyfunction]
pub fn factor_py(expr_str: &str) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.factor().to_latex())
}

/// Group a polynomial by ascending powers of variable `var`.
#[pyfunction]
pub fn collect_py(expr_str: &str, var: usize) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.collect(var).to_latex())
}

/// Partial-fraction-decompose a univariate rational function.
#[pyfunction]
pub fn apart_py(expr_str: &str) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.apart().to_latex())
}

/// Combine a sum of fractions into a single reduced fraction.
#[pyfunction]
pub fn together_py(expr_str: &str) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.together().to_latex())
}

/// Merge powers of a common base (`xᵃ·xᵇ → xᵃ⁺ᵇ`).
#[pyfunction]
pub fn powsimp_py(expr_str: &str) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.powsimp().to_latex())
}

/// Combine a sum of logarithms (`ln a + ln b → ln(ab)`).
#[pyfunction]
pub fn logcombine_py(expr_str: &str) -> PyResult<String> {
    Ok(parse_lowered(expr_str)?.logcombine().to_latex())
}

/// Convert a [`crate::SumResult`] to `(kind, latex)` for Python.
fn sum_result_to_py(result: crate::SumResult) -> (String, Option<String>) {
    match result {
        crate::SumResult::Closed(op) => ("closed".to_string(), Some(op.to_latex())),
        crate::SumResult::NotHypergeometric => ("not_hypergeometric".to_string(), None),
        crate::SumResult::NotClosedForm => ("not_closed_form".to_string(), None),
    }
}

/// Indefinite sum `Σ term(k)` over variable `k`.
///
/// Returns `(kind, latex)`: `kind` is one of ``"closed"``, ``"not_hypergeometric"``,
/// ``"not_closed_form"``; `latex` is `None` unless `kind == "closed"`.
#[pyfunction]
pub fn sum_indefinite_py(expr_str: &str, k: usize) -> PyResult<(String, Option<String>)> {
    Ok(sum_result_to_py(parse_lowered(expr_str)?.sum_indefinite(k)))
}

/// Definite sum `Σ_{k=lo}^{hi} term(k)` for numeric bounds `lo`/`hi`.
///
/// Same return convention as [`sum_indefinite_py`].
#[pyfunction]
pub fn sum_definite_py(
    expr_str: &str,
    k: usize,
    lo: f64,
    hi: f64,
) -> PyResult<(String, Option<String>)> {
    let term = parse_lowered(expr_str)?;
    let lo_op = crate::LoweredOp::Const(lo);
    let hi_op = crate::LoweredOp::Const(hi);
    Ok(sum_result_to_py(term.sum_definite(k, &lo_op, &hi_op)))
}

// ---------------------------------------------------------------------------
// PyMatrix
// ---------------------------------------------------------------------------

/// A symbolic matrix of expression strings, backed by [`crate::matrix::Matrix`].
#[pyclass(name = "Matrix", from_py_object)]
#[derive(Clone)]
pub struct PyMatrix {
    inner: Matrix,
}

#[pymethods]
impl PyMatrix {
    /// Build from a rectangular list of expression-string rows.
    #[new]
    pub fn new(rows: Vec<Vec<String>>) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut parsed_row = Vec::with_capacity(row.len());
            for cell in row {
                parsed_row.push(parse_lowered(cell)?);
            }
            parsed.push(parsed_row);
        }
        let inner = Matrix::from_rows(&parsed).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Build a numeric matrix from a flat row-major list of `f64` values.
    #[staticmethod]
    pub fn from_f64(rows: usize, cols: usize, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Matrix::from_f64(rows, cols, &values).map_err(map_err)?,
        })
    }

    /// The `n × n` identity matrix.
    #[staticmethod]
    pub fn identity(n: usize) -> Self {
        Self {
            inner: Matrix::identity(n),
        }
    }

    /// Number of rows.
    #[getter]
    pub fn nrows(&self) -> usize {
        self.inner.rows
    }

    /// Number of columns.
    #[getter]
    pub fn ncols(&self) -> usize {
        self.inner.cols
    }

    /// Exact determinant, as a LaTeX expression.
    pub fn det(&self) -> PyResult<String> {
        Ok(self.inner.det().map_err(map_err)?.to_latex())
    }

    /// Reduced row echelon form: `(rows_of_latex_cells, rank)`.
    pub fn rref(&self) -> PyResult<(Vec<Vec<String>>, usize)> {
        let reduced = self.inner.rref().map_err(map_err)?;
        let rank = reduced.rank();
        let mut out = Vec::with_capacity(reduced.matrix.rows);
        for i in 0..reduced.matrix.rows {
            let mut row = Vec::with_capacity(reduced.matrix.cols);
            for j in 0..reduced.matrix.cols {
                let cell = reduced
                    .matrix
                    .get(i, j)
                    .ok_or_else(|| map_err("rref index out of bounds"))?;
                row.push(cell.to_latex());
            }
            out.push(row);
        }
        Ok((out, rank))
    }

    /// Characteristic polynomial `p(λ)`, as LaTeX in variable index `lambda_var`.
    pub fn charpoly(&self, lambda_var: usize) -> PyResult<String> {
        Ok(self
            .inner
            .charpoly()
            .map_err(map_err)?
            .to_lowered(lambda_var)
            .to_latex())
    }

    /// Eigenvalues as a list of `(re, im)` pairs (numeric matrices only).
    pub fn eigenvalues(&self) -> PyResult<Vec<(f64, f64)>> {
        Ok(self
            .inner
            .eigenvalues()
            .map_err(map_err)?
            .into_iter()
            .map(|c| (c.re, c.im))
            .collect())
    }

    /// Eigenpairs as `(value_re, value_im, vector, residual)` tuples, where
    /// `vector` is a list of `(re, im)` pairs.
    pub fn eigenvectors(&self) -> PyResult<Vec<EigenpairPy>> {
        Ok(self
            .inner
            .eigenvectors()
            .map_err(map_err)?
            .into_iter()
            .map(|p| {
                let vector = p.vector.iter().map(|c| (c.re, c.im)).collect();
                (p.value.re, p.value.im, vector, p.residual)
            })
            .collect())
    }

    /// Matrix inverse.
    pub fn inverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.inverse().map_err(map_err)?,
        })
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    /// Matrix rank.
    pub fn rank(&self) -> PyResult<usize> {
        self.inner.rank().map_err(map_err)
    }

    /// `True` when the matrix is provably singular.
    pub fn is_singular(&self) -> PyResult<bool> {
        self.inner.is_singular().map_err(map_err)
    }

    /// Human-readable representation.
    pub fn __repr__(&self) -> String {
        format!("Matrix({}x{})", self.inner.rows, self.inner.cols)
    }
}
