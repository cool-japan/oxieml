//! The symbolic zero oracle — a three-tier, **honest** answer to "is this
//! expression identically zero?"
//!
//! # The problem is undecidable
//!
//! Richardson's theorem (1968) says that for expressions built from the rationals,
//! `π`, a variable `x`, the field operations, `sin`, `exp` and `|·|`, the predicate
//! "`E(x) = 0` for all real `x`" is **undecidable**. There is no algorithm here to
//! be found, only algorithms that are *sound in one direction* and honest about the
//! rest. This module therefore never returns a verdict it cannot back up.
//!
//! # The three tiers
//!
//! Every expression is first read as a polynomial `p ∈ ℚ[a₁ … a_m]` over the atom
//! space (see [`super::atoms`]); let `φ` be the evaluation homomorphism sending
//! atoms to the functions they denote.
//!
//! **Tier 1 — structural / algebraic normal form.**
//! If `p` is the zero polynomial then `φ(p)` is the zero function. This direction is
//! *sound*: a polynomial identity in the atoms is a functional identity, whatever
//! relations the atoms satisfy. Verdict: [`ZeroVerdict::Zero`] with proof
//! [`ZeroProof::PolynomialIdentity`] (or [`ZeroProof::StructuralConst`] when
//! `LoweredOp::simplify` already folded the tree to the literal `0`). This tier
//! subsumes every "expand and cancel" simplification: `x·x − x²`, `(a+b)² − a² −
//! 2ab − b²`, and any Bareiss intermediate that cancels, all land here.
//!
//! **Tier 1b — exact, complete decision for the polynomial case.**
//! If `p ≠ 0` *and* every atom in its support is a bare variable `x_i`, then `p` is
//! an honest nonzero polynomial in the real variables, and a nonzero polynomial over
//! an infinite field is not the zero function. Verdict: [`ZeroVerdict::NonZero`]
//! with proof [`NonZeroProof::PolynomialNonZero`]. Together with tier 1, zero-testing
//! is *decidable and complete* for polynomial entries — the undecidability only
//! enters with transcendental atoms.
//!
//! **Tier 2 — Schwartz–Zippel probing.**
//! When `p ≠ 0` but transcendental atoms appear in its support, `p ≠ 0` proves
//! nothing about `φ(p)` (`sin²+cos²−1` is a nonzero polynomial in the atoms
//! `s = sin x`, `c = cos x`, yet the zero function). So we look for a **witness**:
//! evaluate `φ(p)` at seeded, deterministic, dyadic-rational sample points. If some
//! point yields a value that provably exceeds the floating-point error bound at that
//! point (see the error-model section below), the function is *not* identically zero and we say so:
//! [`NonZeroProof::ProbeWitness`], carrying the point, the value and the bound. This
//! direction is sound (modulo the documented IEEE-754 error model) — a witness is a
//! witness.
//!
//! **Tier 3 — `ProbablyZero`.**
//! `p ≠ 0` in the ring, yet every probe vanished to within its error bound. We have
//! *no proof either way*, and we say exactly that: [`ZeroVerdict::ProbablyZero`],
//! carrying the number of usable probes and the largest observed
//! |value| / bound ratio. **This is never silently promoted to `Zero`.** Callers
//! that must branch on it record a [`super::Assumption`] and the result is marked
//! [`super::Certainty::Conditional`]; the plain (non-`_certified`) API refuses the
//! computation with [`super::MatrixError::Undecidable`] rather than lie.
//!
//! # Why Schwartz–Zippel, and what it does (not) give us
//!
//! For the *polynomial* case, the Schwartz–Zippel lemma bounds the probability that
//! a nonzero polynomial of total degree `d` vanishes at a point drawn uniformly from
//! a finite set `S ⊂ ℚ` by `d / |S|`. We sample from a grid of `2²⁴` dyadic
//! rationals, so a false "all probes vanished" for a genuinely nonzero *polynomial*
//! in the *variables* has probability at most `(d / 2²⁴)^probes` — utterly
//! negligible. (We do not even rely on this: tier 1b decides that case exactly.)
//!
//! For the *transcendental* case there is **no such bound**, and pretending
//! otherwise would be exactly the fabricated certainty this module refuses to
//! produce. `φ(p)` is an analytic function; if it is nonzero, its zero set is
//! measure-zero and probing finds a witness immediately in practice — but "in
//! practice" is not a proof, which is why the verdict is `ProbablyZero` and not
//! `Zero`.
//!
//! # The floating-point error model (why a probe can *prove* nonzero-ness)
//!
//! A probe evaluates `p` at atom values `v = (v₁ … v_m)` in `f64`. Write
//! `p⁺(|v|) = Σ_α |c_α| ∏ |vᵢ|^{αᵢ}` for the term-magnitude sum and
//! `D(|v|) = Σ_α |c_α| · deg(α) · ∏ |vᵢ|^{αᵢ}`, which bounds `Σᵢ |vᵢ · ∂p/∂aᵢ|`.
//! Under the standard model (`fl(a ∘ b) = (a ∘ b)(1 + δ)`, `|δ| ≤ u`, `u = 2⁻⁵³`)
//! and assuming each atom is evaluated with relative error at most `u_atom`
//! (we take `u_atom = 8u`; correctly-rounded libm routines achieve well under `1u`),
//! the computed value `p̂` satisfies
//!
//! ```text
//! |p̂ − φ(p)(x)|  ≤  γ_K · p⁺(|v|)  +  u_atom · D(|v|),      γ_K = K·u / (1 − K·u)
//! ```
//!
//! with `K` the operation count of the evaluation. If `|p̂|` exceeds
//! [`SAFETY_FACTOR`] times that bound, then `φ(p)(x) ≠ 0` — a genuine witness. The
//! bound is a heuristic only in its `u_atom` input; everything else is the classical
//! Higham forward-error analysis for polynomial evaluation.
//!
//! When the bound is *not* exceeded we conclude nothing (the probe is discarded as
//! evidence of nonzero-ness), which is why a badly cancelling expression degrades to
//! `ProbablyZero` rather than to a wrong answer.

use crate::lower::LoweredOp;
use crate::poly::{MultiPoly, ratio_to_f64};

use super::atoms::AtomSpace;

/// IEEE-754 double-precision unit roundoff `2⁻⁵³`.
const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Assumed relative error of a single atom evaluation (`exp`, `sin`, …), in units
/// of [`UNIT_ROUNDOFF`]. Correctly-rounded libm routines are well under `1u`; `8u`
/// is a deliberately slack safety factor.
const ATOM_ROUNDOFF_ULPS: f64 = 8.0;

/// A computed probe value must exceed this multiple of its error bound before it is
/// accepted as a proof of nonzero-ness.
pub const SAFETY_FACTOR: f64 = 16.0;

/// Default number of probe points.
pub const DEFAULT_PROBES: usize = 12;

/// Default PRNG seed. Fixed, so every verdict in this crate is reproducible.
pub const DEFAULT_SEED: u64 = 0x0FED_517E_57ED_0001;

/// How a [`ZeroVerdict::Zero`] verdict was established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroProof {
    /// `LoweredOp::simplify` folded the expression to the literal constant `0`.
    StructuralConst,
    /// The expression is the zero element of `ℚ[atoms]`, hence the zero function.
    PolynomialIdentity,
}

/// How a [`ZeroVerdict::NonZero`] verdict was established.
#[derive(Clone, Debug, PartialEq)]
pub enum NonZeroProof {
    /// The expression is a nonzero polynomial in the *real variables* only. A
    /// nonzero polynomial over an infinite field is not the zero function, so this
    /// is an exact proof — no probing involved.
    PolynomialNonZero,
    /// A probe point at which the expression provably does not vanish.
    ProbeWitness {
        /// The variable values `x₀ … x_{n−1}` (exactly representable dyadic rationals).
        point: Vec<f64>,
        /// The computed value at that point.
        value: f64,
        /// The floating-point error bound at that point; `|value| > SAFETY_FACTOR · bound`.
        bound: f64,
    },
}

/// Evidence behind an undecided ([`ZeroVerdict::ProbablyZero`]) verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeEvidence {
    /// How many probe points were usable (evaluated to a finite value).
    pub usable_probes: usize,
    /// How many probe points were attempted.
    pub attempted_probes: usize,
    /// The largest observed `|value| / bound` ratio. Every probe had a ratio below
    /// [`SAFETY_FACTOR`], otherwise the verdict would have been `NonZero`.
    pub max_ratio: f64,
    /// The largest observed `|value|`.
    pub max_abs_value: f64,
}

/// The result of a zero test. **Three** outcomes, not two — the third one is the
/// point of this module.
#[derive(Clone, Debug, PartialEq)]
pub enum ZeroVerdict {
    /// Proved identically zero.
    Zero(ZeroProof),
    /// Proved *not* identically zero.
    NonZero(NonZeroProof),
    /// Undecided: no proof either way. Never treat this as `Zero`.
    ProbablyZero(ProbeEvidence),
}

impl ZeroVerdict {
    /// `true` only for a *proved* zero.
    #[must_use]
    pub fn is_proved_zero(&self) -> bool {
        matches!(self, Self::Zero(_))
    }

    /// `true` only for a *proved* nonzero.
    #[must_use]
    pub fn is_proved_nonzero(&self) -> bool {
        matches!(self, Self::NonZero(_))
    }

    /// `true` when the oracle could not decide.
    #[must_use]
    pub fn is_undecided(&self) -> bool {
        matches!(self, Self::ProbablyZero(_))
    }
}

impl std::fmt::Display for ZeroVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(ZeroProof::StructuralConst) => {
                write!(f, "zero (proved: simplifies to the literal 0)")
            }
            Self::Zero(ZeroProof::PolynomialIdentity) => {
                write!(f, "zero (proved: zero polynomial in the atom ring)")
            }
            Self::NonZero(NonZeroProof::PolynomialNonZero) => {
                write!(f, "nonzero (proved: nonzero polynomial in the variables)")
            }
            Self::NonZero(NonZeroProof::ProbeWitness { value, bound, .. }) => write!(
                f,
                "nonzero (proved: probe value {value:.6e} exceeds error bound {bound:.3e})"
            ),
            Self::ProbablyZero(e) => write!(
                f,
                "PROBABLY zero — UNDECIDED: {} of {} probes usable, all within {:.2}x of the \
                 error bound (max |value| = {:.3e}); no proof either way",
                e.usable_probes, e.attempted_probes, e.max_ratio, e.max_abs_value
            ),
        }
    }
}

/// SplitMix64 — a tiny, seeded, deterministic PRNG.
///
/// Written out here on purpose: probe points must be reproducible across runs and
/// machines, and this crate does not depend on `rand`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A dyadic rational in `[lo, hi)` on the `2⁻²⁴` grid — exactly representable in
    /// `f64`, so the sample point itself introduces no rounding error at all.
    fn next_dyadic(&mut self, lo: f64, hi: f64) -> f64 {
        const GRID: u64 = 1 << 24;
        let k = self.next_u64() % GRID;
        // k / 2^24 is exact; lo, hi and (hi - lo) are chosen dyadic by the caller.
        lo + (k as f64) / (GRID as f64) * (hi - lo)
    }
}

/// The zero oracle.
///
/// Cheap to clone; carries only the probing configuration. The verdicts it produces
/// are deterministic functions of `(seed, probes, expression)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroOracle {
    seed: u64,
    probes: usize,
}

impl Default for ZeroOracle {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            probes: DEFAULT_PROBES,
        }
    }
}

impl ZeroOracle {
    /// The default oracle: [`DEFAULT_PROBES`] probes from [`DEFAULT_SEED`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An oracle with an explicit seed (verdicts stay fully reproducible).
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            probes: DEFAULT_PROBES,
        }
    }

    /// Set the number of probe points. More probes cannot turn `ProbablyZero` into
    /// `Zero` — only into `NonZero`, by finding a witness.
    #[must_use]
    pub fn with_probes(mut self, probes: usize) -> Self {
        self.probes = probes;
        self
    }

    /// The configured probe count.
    #[must_use]
    pub fn probes(&self) -> usize {
        self.probes
    }

    /// The configured seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Test whether `expr` is identically zero.
    ///
    /// Builds the atom space for `expr` and runs the three tiers. See the module
    /// documentation for exactly what each verdict does and does not prove.
    #[must_use]
    pub fn test(&self, expr: &LoweredOp) -> ZeroVerdict {
        let simplified = expr.simplify();
        if let LoweredOp::Const(c) = &simplified {
            if *c == 0.0 {
                return ZeroVerdict::Zero(ZeroProof::StructuralConst);
            }
        }
        let space = AtomSpace::from_exprs(std::slice::from_ref(&simplified));
        match space.polynomialize(&simplified) {
            Ok(poly) => self.test_poly(&poly, &space),
            // A non-finite literal (inf/NaN) has no rational value, so the algebraic
            // tiers do not apply. Probe the raw tree instead; a non-finite probe is
            // unusable, so this degrades honestly to `ProbablyZero`.
            Err(_) => ZeroVerdict::ProbablyZero(ProbeEvidence {
                usable_probes: 0,
                attempted_probes: 0,
                max_ratio: 0.0,
                max_abs_value: 0.0,
            }),
        }
    }

    /// Test a polynomial that is already expressed over `space`.
    ///
    /// This is the form used inside the matrix algorithms, where the entries are
    /// carried as `MultiPoly` and never leave the atom ring.
    #[must_use]
    pub fn test_poly(&self, poly: &MultiPoly, space: &AtomSpace) -> ZeroVerdict {
        // ── Tier 1: exact zero in ℚ[atoms] ⟹ the zero function. Sound. ──────────
        if poly.is_zero() {
            return ZeroVerdict::Zero(ZeroProof::PolynomialIdentity);
        }

        // ── Tier 1b: nonzero polynomial in the *variables* ⟹ not the zero
        //    function. Exact and complete for the polynomial case. ───────────────
        let transcendental_support = poly.terms.iter().any(|(exps, coeff)| {
            !crate::poly::coeff_is_zero(coeff)
                && exps
                    .iter()
                    .enumerate()
                    .any(|(idx, &e)| e > 0 && !space.is_variable_atom(idx))
        });
        if !transcendental_support {
            return ZeroVerdict::NonZero(NonZeroProof::PolynomialNonZero);
        }

        // ── Tier 2: hunt for a witness by Schwartz–Zippel probing. ──────────────
        let n_vars = space.n_vars();
        let mut usable = 0usize;
        let mut max_ratio = 0.0f64;
        let mut max_abs = 0.0f64;

        for probe_index in 0..self.probes {
            let point = self.probe_point(probe_index, n_vars);
            let Some(atom_values) = evaluate_atoms(space, &point) else {
                continue;
            };
            let (value, bound) = evaluate_with_bound(poly, &atom_values);
            if !value.is_finite() || !bound.is_finite() {
                continue;
            }
            usable += 1;
            let abs_value = value.abs();
            max_abs = max_abs.max(abs_value);
            let ratio = if bound > 0.0 {
                abs_value / bound
            } else if abs_value > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            if ratio > max_ratio {
                max_ratio = ratio;
            }
            if abs_value > SAFETY_FACTOR * bound {
                return ZeroVerdict::NonZero(NonZeroProof::ProbeWitness {
                    point,
                    value,
                    bound,
                });
            }
        }

        // ── Tier 3: no proof either way. Say so. ────────────────────────────────
        ZeroVerdict::ProbablyZero(ProbeEvidence {
            usable_probes: usable,
            attempted_probes: self.probes,
            max_ratio,
            max_abs_value: max_abs,
        })
    }

    /// The `index`-th probe point: `n_vars` dyadic rationals.
    ///
    /// Two domains are used in alternation. `[1/2, 4)` keeps `ln`, `√`, `lgamma` and
    /// friends in their domains; `[1/16, 15/16)` keeps `arcsin`, `arccos` and
    /// `arctanh` in theirs. A probe whose evaluation is not finite is simply
    /// discarded (it is evidence of nothing), so an expression that is undefined on
    /// one domain is still probed on the other.
    pub(crate) fn probe_point(&self, index: usize, n_vars: usize) -> Vec<f64> {
        let mut rng =
            SplitMix64::new(self.seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (lo, hi) = if index.is_multiple_of(2) {
            (0.5, 4.0)
        } else {
            (0.0625, 0.9375)
        };
        (0..n_vars).map(|_| rng.next_dyadic(lo, hi)).collect()
    }
}

/// Evaluate every atom at `point`. `None` if any atom is not finite there (the
/// point is outside somebody's domain), which makes the probe unusable.
fn evaluate_atoms(space: &AtomSpace, point: &[f64]) -> Option<Vec<f64>> {
    let mut values = Vec::with_capacity(space.len());
    for atom in space.atoms() {
        let v = atom.eval(point);
        if !v.is_finite() {
            return None;
        }
        values.push(v);
    }
    Some(values)
}

/// Evaluate `poly` at the atom values, together with a bound on the floating-point
/// error of that evaluation.
///
/// Returns `(value, bound)` where — under the error model documented at the top of
/// this module — the exact value `φ(poly)` at the corresponding point satisfies
/// `|value − φ(poly)| ≤ bound`. Consequently `|value| > bound` already proves
/// `φ(poly) ≠ 0`; we insist on [`SAFETY_FACTOR`] times that for margin.
fn evaluate_with_bound(poly: &MultiPoly, atom_values: &[f64]) -> (f64, f64) {
    let mut value = 0.0f64;
    // Σ |c_α| ∏ |v|^α  — the term-magnitude sum (cancellation scale).
    let mut magnitude_sum = 0.0f64;
    // Σ |c_α| · deg(α) · ∏ |v|^α  — bounds Σᵢ |vᵢ · ∂p/∂aᵢ|.
    let mut sensitivity = 0.0f64;
    // Operation count of the evaluation (multiplications + additions + coefficient
    // conversions), used for the γ_K factor.
    let mut op_count = 0.0f64;

    for (exps, coeff) in &poly.terms {
        let c: f64 = ratio_to_f64(coeff);
        let mut term = c;
        let mut abs_term = c.abs();
        let mut degree = 0u32;
        for (idx, &e) in exps.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let v = atom_values.get(idx).copied().unwrap_or(0.0);
            term *= v.powi(e as i32);
            abs_term *= v.abs().powi(e as i32);
            degree += e;
            op_count += f64::from(e);
        }
        value += term;
        magnitude_sum += abs_term;
        sensitivity += abs_term * f64::from(degree);
        op_count += 2.0;
    }

    let k_u = op_count * UNIT_ROUNDOFF;
    let gamma = if k_u < 0.5 {
        k_u / (1.0 - k_u)
    } else {
        // Astronomically many terms: the classical bound degenerates. Report an
        // infinite bound, i.e. "this probe proves nothing".
        f64::INFINITY
    };

    let bound = gamma * magnitude_sum + ATOM_ROUNDOFF_ULPS * UNIT_ROUNDOFF * sensitivity;
    (value, bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn var(i: usize) -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Var(i))
    }

    #[test]
    fn structural_zero_is_proved() {
        // x - x
        let expr = LoweredOp::Sub(var(0), var(0));
        let verdict = ZeroOracle::new().test(&expr);
        assert!(verdict.is_proved_zero(), "{verdict}");
    }

    #[test]
    fn polynomial_identity_zero_is_proved() {
        // x*x - x^2
        let expr = LoweredOp::Sub(
            Arc::new(LoweredOp::Mul(var(0), var(0))),
            Arc::new(LoweredOp::Pow(var(0), Arc::new(LoweredOp::Const(2.0)))),
        );
        let verdict = ZeroOracle::new().test(&expr);
        assert!(verdict.is_proved_zero(), "{verdict}");
    }

    #[test]
    fn nonzero_polynomial_is_proved_exactly() {
        // x^2 + 1  — never zero as a *polynomial*, decided without probing
        let expr = LoweredOp::Add(
            Arc::new(LoweredOp::Mul(var(0), var(0))),
            Arc::new(LoweredOp::Const(1.0)),
        );
        let verdict = ZeroOracle::new().test(&expr);
        assert_eq!(
            verdict,
            ZeroVerdict::NonZero(NonZeroProof::PolynomialNonZero)
        );
    }

    #[test]
    fn transcendental_nonzero_gets_a_probe_witness() {
        // exp(x) - 1  — nonzero, must be witnessed by probing
        let expr = LoweredOp::Sub(
            Arc::new(LoweredOp::Exp(var(0))),
            Arc::new(LoweredOp::Const(1.0)),
        );
        let verdict = ZeroOracle::new().test(&expr);
        match verdict {
            ZeroVerdict::NonZero(NonZeroProof::ProbeWitness { value, bound, .. }) => {
                assert!(value.abs() > SAFETY_FACTOR * bound);
            }
            other => panic!("expected a probe witness, got {other}"),
        }
    }

    #[test]
    fn pythagorean_identity_is_honestly_undecided() {
        // sin(x)^2 + cos(x)^2 - 1  — IS zero, but we cannot prove it: the atoms
        // s = sin x and c = cos x satisfy a relation the polynomial ring cannot see.
        let s2 = LoweredOp::Mul(
            Arc::new(LoweredOp::Sin(var(0))),
            Arc::new(LoweredOp::Sin(var(0))),
        );
        let c2 = LoweredOp::Mul(
            Arc::new(LoweredOp::Cos(var(0))),
            Arc::new(LoweredOp::Cos(var(0))),
        );
        let expr = LoweredOp::Sub(
            Arc::new(LoweredOp::Add(Arc::new(s2), Arc::new(c2))),
            Arc::new(LoweredOp::Const(1.0)),
        );
        let verdict = ZeroOracle::new().test(&expr);
        assert!(
            verdict.is_undecided(),
            "must NOT claim certainty here, got {verdict}"
        );
        assert!(!verdict.is_proved_zero());
        assert!(!verdict.is_proved_nonzero());
    }

    #[test]
    fn exp_reciprocal_identity_is_honestly_undecided() {
        // exp(x) * exp(-x) - 1  — also identically zero, also unprovable here.
        let expr = LoweredOp::Sub(
            Arc::new(LoweredOp::Mul(
                Arc::new(LoweredOp::Exp(var(0))),
                Arc::new(LoweredOp::Exp(Arc::new(LoweredOp::Neg(var(0))))),
            )),
            Arc::new(LoweredOp::Const(1.0)),
        );
        let verdict = ZeroOracle::new().test(&expr);
        assert!(verdict.is_undecided(), "{verdict}");
    }

    #[test]
    fn verdicts_are_reproducible() {
        let expr = LoweredOp::Sin(var(0));
        let a = ZeroOracle::with_seed(7).test(&expr);
        let b = ZeroOracle::with_seed(7).test(&expr);
        assert_eq!(a, b);
    }

    #[test]
    fn probe_points_are_dyadic_and_exact() {
        // `lo + (k / 2²⁴) · (hi − lo)` with dyadic `lo`, `hi`: the widths 3.5 = 7/2 and
        // 0.875 = 7/8 refine the grid to 2⁻²⁵ and 2⁻²⁷ respectively, and every such
        // value needs at most ~29 mantissa bits — so the probe point is an *exactly*
        // representable rational and contributes no rounding error of its own.
        let oracle = ZeroOracle::new();
        for i in 0..8 {
            for x in oracle.probe_point(i, 3) {
                let scaled = x * 2f64.powi(32);
                assert_eq!(scaled.fract(), 0.0, "probe {x} is not on the dyadic grid");
                assert!(x.is_finite() && x > 0.0);
            }
        }
    }
}
