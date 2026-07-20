//! Python bindings for symbolic equation solving.

use pyo3::prelude::*;

use crate::poly::{GroebnerOpts, MonOrder, MultiPoly};

fn map_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Parse `name` into a [`MonOrder`], case-insensitively.
fn parse_order(name: &str) -> PyResult<MonOrder> {
    match name.to_ascii_lowercase().as_str() {
        "lex" => Ok(MonOrder::Lex),
        "grlex" | "deglex" => Ok(MonOrder::GrLex),
        "grevlex" | "degrevlex" => Ok(MonOrder::GrevLex),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown monomial order {other:?}; expected one of: lex, grlex, grevlex"
        ))),
    }
}

/// Parse an expression string into an exact multivariate polynomial.
fn parse_multipoly(expr_str: &str, num_vars: usize) -> PyResult<MultiPoly> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    MultiPoly::from_lowered(&lowered, num_vars).map_err(map_err)
}

/// Find all real symbolic solutions of `expr_str = 0` for the given variable.
///
/// Parameters
/// ----------
/// expr_str : str
///     Expression string for the left-hand side (set equal to zero).
/// var : int
///     Variable index (0-based) to solve for.
///
/// Returns
/// -------
/// list of str
///     Each element is a LaTeX string for a symbolic root.
#[pyfunction]
pub fn solve_for_all_py(expr_str: &str, var: usize) -> PyResult<Vec<String>> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let zero = crate::LoweredOp::Const(0.0);
    let result = crate::solve_for_all(&lowered, &zero, var).map_err(map_err)?;
    Ok(result.roots.iter().map(|r| r.to_latex()).collect())
}

/// Find all complex roots of the polynomial `expr_str` in variable `var`.
///
/// Parameters
/// ----------
/// expr_str : str
///     Expression string for a polynomial expression.
/// var : int
///     Variable index (0-based) to extract the polynomial in.
///
/// Returns
/// -------
/// list of tuple[float, float]
///     Each element is `(re, im)` for a complex root.
#[pyfunction]
pub fn solve_polynomial_complex_py(expr_str: &str, var: usize) -> PyResult<Vec<(f64, f64)>> {
    let tree = crate::parse(expr_str).map_err(map_err)?;
    let lowered = tree.lower().simplify();
    let poly = crate::Poly::from_lowered(&lowered, var)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let complex_roots = crate::solve_poly::solve_polynomial_complex(&poly)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(complex_roots
        .roots
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect())
}

/// Solve a system of equations `exprs[i] = 0` for the given variables.
///
/// Linear systems are solved exactly; nonlinear **polynomial** systems are solved
/// via a lexicographic Gröbner basis (Buchberger) followed by back-substitution,
/// which returns the system's real solutions.
///
/// Parameters
/// ----------
/// expr_strs : list of str
///     One expression per equation; each is implicitly set equal to zero.
/// vars : list of int
///     Variable indices (0-based) to solve for. Their order is the elimination
///     order used by the Gröbner basis.
///
/// Returns
/// -------
/// tuple[str, list[list[float]]]
///     `(kind, solutions)`. `kind` is one of:
///
/// * ``"unique"`` — a linear system with exactly one solution (one entry).
/// * ``"nonlinear_solutions"`` — a nonlinear polynomial system with finitely
///   many solutions; the **real** ones are listed. An empty list means every
///   solution is complex.
/// * ``"inconsistent"`` — no solution exists at all.
/// * ``"underdetermined"`` — infinitely many solutions.
/// * ``"nonlinear"`` — not handled: a transcendental system, a variable
///   outside `vars`, or a Gröbner run that hit a resource cap. **No solutions
///   are ever guessed in this case.**
#[pyfunction]
pub fn solve_system_py(
    expr_strs: Vec<String>,
    vars: Vec<usize>,
) -> PyResult<(String, Vec<Vec<f64>>)> {
    let mut lowered = Vec::with_capacity(expr_strs.len());
    for s in &expr_strs {
        let tree = crate::parse(s).map_err(map_err)?;
        lowered.push(tree.lower().simplify());
    }

    let result = crate::solve_linear_system(&lowered, &vars).map_err(map_err)?;

    let eval_all =
        |sol: &[crate::LoweredOp]| -> Vec<f64> { sol.iter().map(|e| e.eval(&[])).collect() };

    Ok(match result {
        crate::SystemSolveResult::Unique(sol) => ("unique".to_string(), vec![eval_all(&sol)]),
        crate::SystemSolveResult::NonlinearSolutions(sols) => (
            "nonlinear_solutions".to_string(),
            sols.iter().map(|s| eval_all(s)).collect(),
        ),
        crate::SystemSolveResult::Inconsistent => ("inconsistent".to_string(), Vec::new()),
        crate::SystemSolveResult::Underdetermined => ("underdetermined".to_string(), Vec::new()),
        crate::SystemSolveResult::Nonlinear => ("nonlinear".to_string(), Vec::new()),
    })
}

/// Compute the reduced Gröbner basis of the ideal generated by `expr_strs`.
///
/// Parameters
/// ----------
/// expr_strs : list of str
///     Generators of the ideal; each is implicitly set equal to zero.
/// num_vars : int
///     Number of variables the polynomials live in.
/// order : str, optional
///     Monomial order: ``"lex"`` (default), ``"grlex"`` or ``"grevlex"``.
///
/// Returns
/// -------
/// list of str
///     A LaTeX string for each element of the reduced basis, sorted by leading
///     monomial (descending). The basis is canonical: it depends only on the
///     ideal and the order, not on the generators supplied.
///
/// Raises
/// ------
/// ValueError
///     If an expression is not polynomial, or if Buchberger hits a resource cap
///     (the worst case is doubly exponential). No partial basis is ever returned.
#[pyfunction]
#[pyo3(signature = (expr_strs, num_vars, order = "lex"))]
pub fn groebner_basis_py(
    expr_strs: Vec<String>,
    num_vars: usize,
    order: &str,
) -> PyResult<Vec<String>> {
    let opts = GroebnerOpts::with_order(parse_order(order)?);
    let mut gens = Vec::with_capacity(expr_strs.len());
    for s in &expr_strs {
        gens.push(parse_multipoly(s, num_vars)?);
    }
    let basis = crate::groebner_basis(&gens, &opts).map_err(map_err)?;
    Ok(basis.iter().map(|g| g.to_lowered().to_latex()).collect())
}

/// Decide whether `f` lies in the ideal generated by `gens`.
///
/// Parameters
/// ----------
/// f_str : str
///     The polynomial to test for membership.
/// gen_strs : list of str
///     Generators of the ideal.
/// num_vars : int
///     Number of variables the polynomials live in.
/// order : str, optional
///     Monomial order used internally. The answer does not depend on it.
///
/// Returns
/// -------
/// bool
///     ``True`` iff `f` is in the ideal. Decided by reducing `f` against a
///     Gröbner basis, whose remainder is zero exactly for members.
#[pyfunction]
#[pyo3(signature = (f_str, gen_strs, num_vars, order = "lex"))]
pub fn ideal_member_py(
    f_str: &str,
    gen_strs: Vec<String>,
    num_vars: usize,
    order: &str,
) -> PyResult<bool> {
    let opts = GroebnerOpts::with_order(parse_order(order)?);
    let f = parse_multipoly(f_str, num_vars)?;
    let mut gens = Vec::with_capacity(gen_strs.len());
    for s in &gen_strs {
        gens.push(parse_multipoly(s, num_vars)?);
    }
    crate::ideal_member(&f, &gens, &opts).map_err(map_err)
}
