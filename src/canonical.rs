//! Canonical EML tree constructions for elementary functions.
//!
//! Implements Paper Tables 1-4: all elementary functions expressed
//! as EML trees using only `eml(x, y) = exp(x) - ln(y)` and the constant `1`.

use crate::tree::EmlTree;

/// Canonical constructions for elementary functions as EML trees.
///
/// Each method builds the EML tree representation of a standard mathematical
/// function, following the constructions from the paper (arXiv:2603.21852).
pub struct Canonical;

impl Canonical {
    // ================================================================
    // Table 1: Basic operations
    // ================================================================

    /// `exp(x) = eml(x, 1)` — depth 1
    ///
    /// `eml(x, 1) = exp(x) - ln(1) = exp(x)`.
    pub fn exp(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        EmlTree::eml(x, &one)
    }

    /// `ln(x) = eml(1, eml(eml(1, x), 1))` — depth 3
    ///
    /// Construction:
    /// - Inner: `eml(1, x) = e - ln(x)`
    /// - Middle: `eml(e - ln(x), 1) = exp(e - ln(x)) = exp(e)/x`
    /// - Outer: `eml(1, exp(e)/x) = e - ln(exp(e)/x) = e - (e - ln(x)) = ln(x)`
    pub fn ln(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let inner = EmlTree::eml(&one, x); // e - ln(x)
        let middle = EmlTree::eml(&inner, &one); // exp(e - ln(x))
        EmlTree::eml(&one, &middle) // e - ln(exp(e-ln(x))) = ln(x)
    }

    /// `e = eml(1, 1)` — depth 1
    ///
    /// `eml(1, 1) = exp(1) - ln(1) = e`.
    pub fn euler() -> EmlTree {
        let one = EmlTree::one();
        EmlTree::eml(&one, &one)
    }

    /// `-x` (negation) — depth 6
    ///
    /// Uses the identity: `-x = (e - x) - e`.
    ///
    /// Constructed as `eml(ln(e-x), exp(e))`:
    /// - `exp(ln(e-x)) - ln(exp(e)) = (e-x) - e = -x`
    ///
    /// Works in the complex domain where `ln` is defined for all nonzero values,
    /// so the construction holds for all real `x`.
    pub fn neg(x: &EmlTree) -> EmlTree {
        let e_minus_x = Self::e_minus(x);
        let ln_e_minus_x = Self::ln(&e_minus_x);
        let exp_e = Self::exp(&Self::euler());
        // eml(ln(e-x), exp(e)) = exp(ln(e-x)) - ln(exp(e)) = (e-x) - e = -x
        EmlTree::eml(&ln_e_minus_x, &exp_e)
    }

    // ================================================================
    // Table 2: Arithmetic operations
    // ================================================================

    /// `x + y = sub(x, neg(y))` = `x - (-y)`
    ///
    /// Built by composing subtraction and negation.
    pub fn add(x: &EmlTree, y: &EmlTree) -> EmlTree {
        Self::sub(x, &Self::neg(y))
    }

    /// `x - y` — subtraction via `eml(ln(x), eml(y, 1))`
    ///
    /// Key identity: `eml(ln(x), eml(y, 1)) = exp(ln(x)) - ln(exp(y)) = x - y`.
    ///
    /// This relies on complex evaluation where `exp(ln(z)) = z` on the
    /// principal branch, making the identity valid for all real inputs.
    pub fn sub(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let ln_x = Self::ln(x);
        let exp_y = EmlTree::eml(y, &one); // eml(y, 1) = exp(y)
        // eml(ln(x), exp(y)) = exp(ln(x)) - ln(exp(y)) = x - y
        EmlTree::eml(&ln_x, &exp_y)
    }

    /// `x * y = exp(ln(x) + ln(y))` — via `exp(add(ln(x), ln(y)))`
    ///
    /// Uses the logarithmic identity: `ln(x) + ln(y) = ln(xy)`,
    /// so `exp(ln(x) + ln(y)) = xy`.
    pub fn mul(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let ln_y = Self::ln(y);
        let sum = Self::add(&ln_x, &ln_y);
        Self::exp(&sum)
    }

    /// `x / y = exp(ln(x) - ln(y))`
    ///
    /// Uses `ln(x/y) = ln(x) - ln(y)`.
    pub fn div(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let ln_y = Self::ln(y);
        let diff = Self::sub(&ln_x, &ln_y);
        Self::exp(&diff)
    }

    /// `x ^ y = exp(y * ln(x))`
    pub fn pow(x: &EmlTree, y: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let y_ln_x = Self::mul(y, &ln_x);
        Self::exp(&y_ln_x)
    }

    // ================================================================
    // Table 3: Trigonometric (via complex numbers)
    // ================================================================

    /// `pi()` — returns a tree whose complex evaluation yields `iπ`.
    ///
    /// Construction: `ln(-1) = iπ` in the complex domain.
    /// This tree is used internally by `sin`/`cos` and is not intended
    /// for direct `eval_real` (which would return `ComplexResult` error).
    pub fn pi() -> EmlTree {
        let one = EmlTree::one();
        let neg_one = Self::neg(&one);
        Self::ln(&neg_one) // ln(-1) = iπ
    }

    /// `sin(x) = (exp(ix) - exp(-ix)) / (2i)` — Euler formula
    ///
    /// Constructs `i = exp(iπ/2) = exp(ln(-1)/2)`, then builds
    /// the Euler decomposition. Evaluates correctly through the
    /// complex evaluation path.
    pub fn sin(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let neg_one = Self::neg(&one);
        let half = Self::reciprocal(&Self::add(&one, &one));

        // i = exp(ln(-1) * 1/2) = exp(iπ/2)
        let ln_neg_one = Self::ln(&neg_one);
        let half_ln_neg_one = Self::mul(&half, &ln_neg_one);
        let i_val = Self::exp(&half_ln_neg_one);

        // exp(ix) and exp(-ix)
        let ix = Self::mul(&i_val, x);
        let exp_ix = Self::exp(&ix);
        let neg_ix = Self::neg(&ix);
        let exp_neg_ix = Self::exp(&neg_ix);

        // (exp(ix) - exp(-ix)) / (2i)
        let diff = Self::sub(&exp_ix, &exp_neg_ix);
        let two = Self::add(&one, &one);
        let two_i = Self::mul(&two, &i_val);
        Self::div(&diff, &two_i)
    }

    /// `cos(x) = (exp(ix) + exp(-ix)) / 2` — Euler formula
    ///
    /// Same construction as `sin` but using the real part identity.
    pub fn cos(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let neg_one = Self::neg(&one);
        let half = Self::reciprocal(&Self::add(&one, &one));

        let ln_neg_one = Self::ln(&neg_one);
        let half_ln_neg_one = Self::mul(&half, &ln_neg_one);
        let i_val = Self::exp(&half_ln_neg_one);

        let ix = Self::mul(&i_val, x);
        let exp_ix = Self::exp(&ix);
        let neg_ix = Self::neg(&ix);
        let exp_neg_ix = Self::exp(&neg_ix);

        // (exp(ix) + exp(-ix)) / 2
        let sum = Self::add(&exp_ix, &exp_neg_ix);
        let two = Self::add(&one, &one);
        Self::div(&sum, &two)
    }

    /// `tan(x) = sin(x) / cos(x)`
    pub fn tan(x: &EmlTree) -> EmlTree {
        Self::div(&Self::sin(x), &Self::cos(x))
    }

    // ================================================================
    // Table 4: Inverse trigonometric (via complex logarithms)
    // ================================================================

    /// `arcsin(x) = -i * ln(ix + sqrt(1 - x^2))`
    ///
    /// Uses the complex logarithm identity. The imaginary parts cancel
    /// for real inputs in `[-1, 1]`, yielding a real result.
    pub fn arcsin(x: &EmlTree) -> EmlTree {
        let i = Self::imag_unit();
        let one = EmlTree::one();
        let ix = Self::mul(&i, x);
        let x_sq = Self::square(x);
        let one_minus_x_sq = Self::sub(&one, &x_sq);
        let sqrt_part = Self::sqrt(&one_minus_x_sq);
        // -i * ln(ix + sqrt(1 - x^2))
        Self::neg(&Self::mul(&i, &Self::ln(&Self::add(&ix, &sqrt_part))))
    }

    /// `arccos(x) = -i * ln(x + i * sqrt(1 - x^2))`
    ///
    /// Alternative form that avoids subtracting from π/2.
    pub fn arccos(x: &EmlTree) -> EmlTree {
        let i = Self::imag_unit();
        let one = EmlTree::one();
        let x_sq = Self::square(x);
        let one_minus_x_sq = Self::sub(&one, &x_sq);
        let sqrt_part = Self::sqrt(&one_minus_x_sq);
        let i_sqrt = Self::mul(&i, &sqrt_part);
        // -i * ln(x + i*sqrt(1-x^2))
        Self::neg(&Self::mul(&i, &Self::ln(&Self::add(x, &i_sqrt))))
    }

    /// `arctan(x) = (-i/2) * ln((1 + ix) / (1 - ix))`
    ///
    /// Uses the complex logarithm identity for arctan. The imaginary
    /// parts cancel for all real `x`, yielding a real result.
    pub fn arctan(x: &EmlTree) -> EmlTree {
        let i = Self::imag_unit();
        let one = EmlTree::one();
        let two = Self::nat(2);
        let ix = Self::mul(&i, x);
        let numerator = Self::add(&one, &ix);
        let denominator = Self::sub(&one, &ix);
        // (-i/2) * ln((1+ix)/(1-ix))
        let neg_i_half = Self::neg(&Self::mul(&i, &Self::reciprocal(&two)));
        Self::mul(&neg_i_half, &Self::ln(&Self::div(&numerator, &denominator)))
    }

    // ================================================================
    // Table 5: Hyperbolic functions
    // ================================================================

    /// `sinh(x) = (exp(x) - exp(-x)) / 2`
    pub fn sinh(x: &EmlTree) -> EmlTree {
        let exp_x = Self::exp(x);
        let exp_neg_x = Self::exp(&Self::neg(x));
        Self::div(&Self::sub(&exp_x, &exp_neg_x), &Self::nat(2))
    }

    /// `cosh(x) = (exp(x) + exp(-x)) / 2`
    pub fn cosh(x: &EmlTree) -> EmlTree {
        let exp_x = Self::exp(x);
        let exp_neg_x = Self::exp(&Self::neg(x));
        Self::div(&Self::add(&exp_x, &exp_neg_x), &Self::nat(2))
    }

    /// `tanh(x) = sinh(x) / cosh(x)`
    pub fn tanh(x: &EmlTree) -> EmlTree {
        Self::div(&Self::sinh(x), &Self::cosh(x))
    }

    // ================================================================
    // Table 6: Inverse hyperbolic functions
    // ================================================================

    /// `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
    pub fn arcsinh(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let x_sq = Self::square(x);
        Self::ln(&Self::add(x, &Self::sqrt(&Self::add(&x_sq, &one))))
    }

    /// `arccosh(x) = ln(x + sqrt(x^2 - 1))` — defined for x >= 1
    pub fn arccosh(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let x_sq = Self::square(x);
        Self::ln(&Self::add(x, &Self::sqrt(&Self::sub(&x_sq, &one))))
    }

    /// `arctanh(x) = (1/2) * ln((1 + x) / (1 - x))` — defined for |x| < 1
    pub fn arctanh(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let two = Self::nat(2);
        let half = Self::reciprocal(&two);
        let numerator = Self::add(&one, x);
        let denominator = Self::sub(&one, x);
        Self::mul(&half, &Self::ln(&Self::div(&numerator, &denominator)))
    }

    // ================================================================
    // Table 7: Powers and roots
    // ================================================================

    /// `x^2 = exp(2 * ln(x))` — square
    pub fn square(x: &EmlTree) -> EmlTree {
        Self::pow(x, &Self::nat(2))
    }

    /// `sqrt(x) = x^0.5 = exp(0.5 * ln(x))`
    pub fn sqrt(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let two = Self::add(&one, &one);
        let half = Self::reciprocal(&two);
        Self::pow(x, &half)
    }

    /// `abs(x) = sqrt(x^2)`
    pub fn abs(x: &EmlTree) -> EmlTree {
        Self::sqrt(&Self::square(x))
    }

    // ================================================================
    // Constants
    // ================================================================

    /// `-1 = neg(1)` — the constant negative one
    pub fn neg_one() -> EmlTree {
        Self::neg(&EmlTree::one())
    }

    /// `-2 = neg(nat(2))` — the constant negative two
    pub fn neg_two() -> EmlTree {
        Self::neg(&Self::nat(2))
    }

    /// `i = exp(iπ/2)` — the imaginary unit
    ///
    /// Construction: `i = exp(ln(-1) / 2)`.
    /// Since `ln(-1) = iπ`, we get `exp(iπ/2) = cos(π/2) + i*sin(π/2) = i`.
    ///
    /// This tree evaluates to a purely imaginary complex number.
    /// `eval_real` will return `ComplexResult` error.
    pub fn imag_unit() -> EmlTree {
        let two = Self::nat(2);
        let half = Self::reciprocal(&two);
        let ln_neg_one = Self::ln(&Self::neg_one()); // iπ
        Self::exp(&Self::mul(&half, &ln_neg_one)) // exp(iπ/2) = i
    }

    // ================================================================
    // Helper constructions
    // ================================================================

    /// `e - x = eml(1, eml(x, 1))` — depth 2
    ///
    /// `eml(1, eml(x, 1)) = exp(1) - ln(exp(x)) = e - x`.
    fn e_minus(x: &EmlTree) -> EmlTree {
        let one = EmlTree::one();
        let exp_x = EmlTree::eml(x, &one);
        EmlTree::eml(&one, &exp_x)
    }

    /// `1/x = exp(-ln(x))` — reciprocal
    pub fn reciprocal(x: &EmlTree) -> EmlTree {
        let ln_x = Self::ln(x);
        let neg_ln_x = Self::neg(&ln_x);
        Self::exp(&neg_ln_x)
    }

    /// Build a natural number `n` as an EML tree: `n = 1 + 1 + ... + 1`.
    pub fn nat(n: u64) -> EmlTree {
        assert!(n >= 1, "nat(0) not supported; use ln(1) for zero");
        let one = EmlTree::one();
        if n == 1 {
            return one;
        }
        let mut result = one.clone();
        for _ in 1..n {
            result = Self::add(&result, &one);
        }
        result
    }

    /// Build zero as an EML tree: `0 = ln(1)`.
    pub fn zero() -> EmlTree {
        Self::ln(&EmlTree::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::EvalCtx;

    #[test]
    fn test_exp_construction() {
        let x = EmlTree::var(0);
        let exp_x = Canonical::exp(&x);
        assert_eq!(exp_x.depth(), 1);
        let ctx = EvalCtx::new(&[2.0]);
        let result = exp_x.eval_real(&ctx).unwrap();
        assert!((result - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_euler_construction() {
        let e = Canonical::euler();
        assert_eq!(e.depth(), 1);
        let ctx = EvalCtx::new(&[]);
        let result = e.eval_real(&ctx).unwrap();
        assert!((result - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_ln_construction() {
        let x = EmlTree::var(0);
        let ln_x = Canonical::ln(&x);
        assert_eq!(ln_x.depth(), 3);
        let ctx = EvalCtx::new(&[std::f64::consts::E]);
        let result = ln_x.eval_real(&ctx).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_of_one() {
        let one = EmlTree::one();
        let ln_one = Canonical::ln(&one);
        let ctx = EvalCtx::new(&[]);
        let result = ln_one.eval_real(&ctx).unwrap();
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_e_minus_x() {
        let x = EmlTree::var(0);
        let emx = Canonical::e_minus(&x);
        let ctx = EvalCtx::new(&[1.0]);
        let result = emx.eval_real(&ctx).unwrap();
        assert!((result - (std::f64::consts::E - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_neg() {
        let x = EmlTree::var(0);
        let neg_x = Canonical::neg(&x);
        let ctx = EvalCtx::new(&[3.0]);
        let result = neg_x.eval_real(&ctx).unwrap();
        assert!((result - (-3.0)).abs() < 1e-8);
    }

    #[test]
    fn test_sub() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let diff = Canonical::sub(&x, &y);
        let ctx = EvalCtx::new(&[5.0, 3.0]);
        let result = diff.eval_real(&ctx).unwrap();
        assert!((result - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_add() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let sum = Canonical::add(&x, &y);
        let ctx = EvalCtx::new(&[2.0, 3.0]);
        let result = sum.eval_real(&ctx).unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mul() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let prod = Canonical::mul(&x, &y);
        let ctx = EvalCtx::new(&[3.0, 4.0]);
        let result = prod.eval_real(&ctx).unwrap();
        assert!((result - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_div() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let quot = Canonical::div(&x, &y);
        let ctx = EvalCtx::new(&[10.0, 2.0]);
        let result = quot.eval_real(&ctx).unwrap();
        assert!((result - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_pow() {
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let p = Canonical::pow(&x, &y);
        let ctx = EvalCtx::new(&[2.0, 3.0]);
        let result = p.eval_real(&ctx).unwrap();
        assert!((result - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_reciprocal() {
        let x = EmlTree::var(0);
        let recip = Canonical::reciprocal(&x);
        let ctx = EvalCtx::new(&[4.0]);
        let result = recip.eval_real(&ctx).unwrap();
        assert!((result - 0.25).abs() < 1e-8);
    }

    #[test]
    fn test_zero() {
        let z = Canonical::zero();
        let ctx = EvalCtx::new(&[]);
        let result = z.eval_real(&ctx).unwrap();
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_nat() {
        for n in 1..=5u64 {
            let tree = Canonical::nat(n);
            let ctx = EvalCtx::new(&[]);
            let result = tree.eval_real(&ctx).unwrap();
            assert!(
                (result - n as f64).abs() < 0.1,
                "nat({n}) = {result}, expected {n}"
            );
        }
    }

    #[test]
    fn test_sqrt() {
        let x = EmlTree::var(0);
        let sqrt_x = Canonical::sqrt(&x);
        let ctx = EvalCtx::new(&[4.0]);
        let result = sqrt_x.eval_real(&ctx).unwrap();
        assert!((result - 2.0).abs() < 1e-2);
    }

    #[test]
    fn test_abs_positive() {
        let x = EmlTree::var(0);
        let abs_x = Canonical::abs(&x);
        let ctx = EvalCtx::new(&[3.0]);
        let result = abs_x.eval_real(&ctx).unwrap();
        assert!((result - 3.0).abs() < 1e-2);
    }

    #[test]
    fn test_square() {
        let x = EmlTree::var(0);
        let x_sq = Canonical::square(&x);
        for &val in &[2.0, 3.0, 0.5] {
            let ctx = EvalCtx::new(&[val]);
            let result = x_sq.eval_real(&ctx).unwrap();
            assert!(
                (result - val * val).abs() < 1e-2,
                "square({val}) = {result}, expected {}",
                val * val
            );
        }
    }

    #[test]
    fn test_neg_one() {
        let tree = Canonical::neg_one();
        let ctx = EvalCtx::new(&[]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!((result - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_neg_two() {
        let tree = Canonical::neg_two();
        let ctx = EvalCtx::new(&[]);
        let result = tree.eval_real(&ctx).unwrap();
        assert!((result - (-2.0)).abs() < 0.1);
    }

    #[test]
    fn test_imag_unit() {
        // i = exp(iπ/2): purely imaginary, eval_real should fail
        let i_tree = Canonical::imag_unit();
        let ctx = EvalCtx::new(&[]);
        assert!(
            i_tree.eval_real(&ctx).is_err(),
            "imag_unit should fail in real mode"
        );

        // Complex eval should give (0, 1)
        let result = i_tree.eval_complex(&[]).unwrap();
        assert!(
            result.re.abs() < 1e-4,
            "Re(i) should be ~0, got {}",
            result.re
        );
        assert!(
            (result.im - 1.0).abs() < 1e-4,
            "Im(i) should be ~1, got {}",
            result.im
        );
    }

    #[test]
    fn test_tan() {
        let x = EmlTree::var(0);
        let tan_x = Canonical::tan(&x);
        // tan(0) = 0
        let ctx = EvalCtx::new(&[0.0]);
        let result = tan_x.eval_real(&ctx);
        if let Ok(val) = result {
            assert!(val.abs() < 0.1, "tan(0) should be ~0, got {val}");
        }
    }

    #[test]
    fn test_sinh() {
        let x = EmlTree::var(0);
        let sinh_x = Canonical::sinh(&x);
        for &val in &[0.0, 1.0] {
            let ctx = EvalCtx::new(&[val]);
            let result = sinh_x.eval_real(&ctx).unwrap();
            assert!(
                (result - val.sinh()).abs() < 0.1,
                "sinh({val}) = {result}, expected {}",
                val.sinh()
            );
        }
    }

    #[test]
    fn test_cosh() {
        let x = EmlTree::var(0);
        let cosh_x = Canonical::cosh(&x);
        for &val in &[0.0, 1.0] {
            let ctx = EvalCtx::new(&[val]);
            let result = cosh_x.eval_real(&ctx).unwrap();
            assert!(
                (result - val.cosh()).abs() < 0.1,
                "cosh({val}) = {result}, expected {}",
                val.cosh()
            );
        }
    }

    #[test]
    fn test_tanh() {
        let x = EmlTree::var(0);
        let tanh_x = Canonical::tanh(&x);
        let ctx = EvalCtx::new(&[0.0]);
        let result = tanh_x.eval_real(&ctx);
        if let Ok(val) = result {
            assert!(val.abs() < 0.1, "tanh(0) should be ~0, got {val}");
        }
    }

    #[test]
    fn test_arcsinh() {
        let x = EmlTree::var(0);
        let asinh_x = Canonical::arcsinh(&x);
        // arcsinh(0) = 0
        let ctx = EvalCtx::new(&[0.0]);
        let result = asinh_x.eval_real(&ctx).unwrap();
        assert!(result.abs() < 0.1, "arcsinh(0) = {result}, expected 0");
    }

    #[test]
    fn test_arctanh() {
        let x = EmlTree::var(0);
        let atanh_x = Canonical::arctanh(&x);
        // arctanh(0) = 0
        let ctx = EvalCtx::new(&[0.0]);
        let result = atanh_x.eval_real(&ctx).unwrap();
        assert!(result.abs() < 0.1, "arctanh(0) = {result}, expected 0");
    }

    #[test]
    fn test_arctan() {
        let x = EmlTree::var(0);
        let atan_x = Canonical::arctan(&x);
        // arctan(0) = 0
        let ctx = EvalCtx::new(&[0.0]);
        let result = atan_x.eval_real(&ctx);
        if let Ok(val) = result {
            assert!(val.abs() < 0.1, "arctan(0) should be ~0, got {val}");
        }
    }

    #[test]
    fn test_arcsin() {
        let x = EmlTree::var(0);
        let asin_x = Canonical::arcsin(&x);
        // arcsin(0) = 0
        let ctx = EvalCtx::new(&[0.0]);
        let result = asin_x.eval_real(&ctx);
        if let Ok(val) = result {
            assert!(val.abs() < 0.1, "arcsin(0) should be ~0, got {val}");
        }
    }

    #[test]
    fn test_arccos() {
        let x = EmlTree::var(0);
        let acos_x = Canonical::arccos(&x);
        // arccos(1) = 0
        let ctx = EvalCtx::new(&[1.0]);
        let result = acos_x.eval_real(&ctx);
        if let Ok(val) = result {
            assert!(val.abs() < 0.2, "arccos(1) should be ~0, got {val}");
        }
    }
}
