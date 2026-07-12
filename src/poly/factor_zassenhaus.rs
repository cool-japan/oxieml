//! Zassenhaus' algorithm: univariate factorization over ℚ for arbitrary degree.
//!
//! Given a **primitive, square-free** integer polynomial `f ∈ ℤ[x]` of degree
//! `n` with positive leading coefficient `b = lc(f)`, this module returns its
//! irreducible factors over ℚ (equivalently, by Gauss' lemma, its primitive
//! irreducible factors over ℤ). The multiplicity structure is handled one level
//! up by Yun's square-free decomposition in [`super::factor`].
//!
//! # The pipeline
//!
//! ```text
//!   f ∈ ℤ[x], primitive, square-free
//!        │
//!        ├─ 1. prime selection      p ∤ lc(f), f mod p square-free
//!        │                          (a few candidates; keep the one with the
//!        │                           fewest modular factors)
//!        │
//!        ├─ 2. distinct-degree      f̄ = Π_d g_d, g_d = product of the
//!        │     factorization (DDF)  irreducible factors of degree exactly d
//!        │
//!        ├─ 3. equal-degree         Cantor–Zassenhaus splits each g_d into its
//!        │     factorization (EDF)  r_d irreducible factors  h_1 … h_r
//!        │
//!        ├─ 4. Hensel lifting       f ≡ b · Π ĥ_i  (mod p^l),  ĥ_i monic,
//!        │     (quadratic, tree)    p^l > 2 · B  with B the Mignotte bound
//!        │
//!        └─ 5. recombination        search subsets S ⊆ {1..r} whose lifted
//!              (Zassenhaus)         product is a true integer factor of f
//! ```
//!
//! # 1. Prime selection
//!
//! A prime is *lucky* when `p ∤ lc(f)` (so the degree is preserved) and
//! `f mod p` stays square-free (so its modular factors are pairwise coprime,
//! which is what Hensel lifting needs). Since `f` is square-free over ℚ its
//! discriminant is non-zero, so only the finitely many primes dividing
//! `lc(f) · disc(f)` are unlucky and a lucky prime always exists.
//!
//! Only **odd** primes are considered. This is deliberate: the Cantor–Zassenhaus
//! split below raises a random element to the power `(p^d − 1)/2`, which needs
//! `p` odd (over `GF(2)` one has to use the additive trace map instead). Odd
//! primes also give a smaller lift exponent `l` for the same bound, so nothing
//! is lost.
//!
//! Different lucky primes give different numbers of modular factors `r`, and the
//! recombination step is exponential in `r`, so we try up to
//! [`MAX_PRIME_CANDIDATES`] lucky primes and keep the one minimising `r`.
//! `r = 1` proves irreducibility outright and short-circuits everything.
//!
//! # 2. Distinct-degree factorization
//!
//! Over `GF(p)`, `x^(p^d) − x` is the product of *all* monic irreducibles whose
//! degree divides `d`. Iterating `w ← w^p mod f` from `w = x` therefore gives
//! `w = x^(p^d) mod f`, and
//!
//! ```text
//!   g_d = gcd(f, x^(p^d) − x)
//! ```
//!
//! collects exactly the irreducible factors of degree `d` once the lower degrees
//! have been divided out. The loop stops as soon as `deg f < 2d`, at which point
//! whatever is left is irreducible.
//!
//! # 3. Cantor–Zassenhaus equal-degree splitting
//!
//! Let `g` be a product of `r > 1` distinct monic irreducibles, all of degree
//! `d`, over `GF(p)` with `p` odd. By CRT,
//! `GF(p)[x]/(g) ≅ Π_{i=1}^{r} GF(p^d)`. For a random `a`, the element
//! `a^((p^d − 1)/2)` is `±1` in each component (it squares to `a^(p^d − 1) = 1`),
//! and the signs are independent and uniform. Hence
//! `gcd(a^((p^d−1)/2) − 1, g)` is a proper factor with probability
//! `1 − 2^{1−r} ≥ 1/2`. The rare `gcd(a, g) ≠ 1` case (some component is zero)
//! is also a usable split and is checked first.
//!
//! # 4. Multifactor Hensel lifting
//!
//! Write `F = lc(f)^{-1} · f (mod p^k)`, a *monic* polynomial mod `p^k`. Its
//! modular factorization `F ≡ Π h_i (mod p)` into pairwise-coprime monic factors
//! lifts uniquely to `F ≡ Π ĥ_i (mod p^k)`.
//!
//! The lifting is done on a balanced binary tree whose leaves are the `h_i` and
//! whose internal nodes hold the product of their children together with Bézout
//! cofactors `s·g + t·h ≡ 1`. Each node is lifted by one **quadratic (Newton)
//! Hensel step** — precision `m → m'` for any `m' | m²`:
//!
//! ```text
//!   e ← F − g·h                       (mod m′),  e ≡ 0 (mod m)
//!   s·e = q·h + r                     (division by the monic h)
//!   g* ← g + t·e + q·g,   h* ← h + r  (mod m′)
//!   c  ← s·g* + t·h* − 1              (mod m′),  c ≡ 0 (mod m)
//!   s·c = u·h* + v
//!   s* ← s − v,   t* ← t − t·c − u·g* (mod m′)
//! ```
//!
//! The first line is Newton's iteration in disguise: from
//! `s·g + t·h ≡ 1 (mod m)` and `e ≡ 0 (mod m)` we get
//! `e ≡ e·(s·g + t·h) = g·(s·e) + h·(t·e) (mod m²)`, and reducing `s·e` modulo
//! the monic `h` moves the excess into the `h`-multiple, which yields
//! `e ≡ g·r + h·(t·e + q·g)`. So `g* h* ≡ F (mod m′)` with `h*` still monic and
//! `deg g* = deg g` (the high-order terms of `t·e + q·g` cancel identically,
//! because `h·(t·e + q·g) = e − g·r` has degree `< deg F`). The second half
//! repairs the Bézout relation to the new precision by the same trick, so the
//! invariant is available for the next doubling. The error therefore goes
//! `m → m²` per step: `⌈log₂ l⌉` steps suffice.
//!
//! The lift target `p^l` is recomputed exactly (not overshot) by walking the
//! exponent schedule `1, …, ⌈l/4⌉, ⌈l/2⌉, l`, which is legal because the step
//! above is valid for *any* `m′` dividing `m²`.
//!
//! # 5. Mignotte bound and recombination
//!
//! **Landau–Mignotte.** If `g | f` in `ℤ[x]` with `deg g = k` then, writing
//! `M(g) = |lc g| · Π max(1, |β_j|)` for the Mahler measure over the roots of
//! `g`,
//!
//! ```text
//!   ‖g‖₁ = Σ_j |g_j| ≤ Σ_j C(k, j) · M(g) = 2^k · M(g)
//!        ≤ 2^k · |lc(g)/lc(f)| · M(f) ≤ 2^k · |lc(g)/lc(f)| · ‖f‖₂
//! ```
//!
//! using Landau's inequality `M(f) ≤ ‖f‖₂` and the fact that the roots of `g`
//! are a sub-multiset of those of `f` with `lc(g) | lc(f)`. Consequently, for
//! every divisor `g` of `f`,
//!
//! ```text
//!   ‖ (b / lc g) · g ‖₁  ≤  2^n · ‖f‖₂  =:  B .
//! ```
//!
//! **Why that is the right object.** If `f = g · w` over ℤ then, mod `p`,
//! `ḡ = lc(g) · Π_{i∈S} h̄_i` for some subset `S` (unique factorization in
//! `GF(p)[x]`), and by *uniqueness* of the Hensel lift applied to
//! `F ≡ (lc(g)^{-1} g) · (lc(w)^{-1} w)` we get `lc(g)^{-1} g ≡ Π_{i∈S} ĥ_i`
//! modulo `p^l`. Multiplying by `b`:
//!
//! ```text
//!   G_S := b · Π_{i∈S} ĥ_i  ≡  (b / lc g) · g   (mod p^l) .
//! ```
//!
//! Choosing `l` minimal with `p^l ≥ 2B + 1` makes the symmetric representative
//! of `G_S` *equal* to the integer polynomial `(b/lc g) · g`, whose primitive
//! part is exactly `g`. So the true factors are found by enumerating subsets,
//! symmetrizing, taking the primitive part and testing exact divisibility — no
//! rounding, no heuristic acceptance. Two sound filters make this cheap:
//!
//! * the constant coefficient of `(b/lc g)·g` divides `b · f(0)`
//!   (because `g(0) | f(0)`), and
//! * `‖sym(G_S)‖₁ ≤ B` must hold, by the Mignotte bound above.
//!
//! Subsets are tried by increasing size, so the first subset that divides is
//! necessarily *irreducible*: every proper sub-subset was already ruled out.
//! Once a factor is peeled off, `f`, `b` and the index pool shrink, which keeps
//! the later candidates small.
//!
//! # Complexity and honest failure
//!
//! Everything except recombination is polynomial time. Recombination is
//! worst-case exponential in `r` — the Swinnerton–Dyer polynomials are the
//! classical witnesses, since they split into linear factors modulo *every*
//! prime while being irreducible over ℚ. (Making that step polynomial requires
//! the van Hoeij / LLL knapsack reformulation, which is out of scope here.) The
//! number of subsets examined is capped by [`MAX_RECOMBINATION_SUBSETS`]; when
//! the cap, the prime search or the Cantor–Zassenhaus attempt budget is
//! exhausted, the functions return [`PolyError::FactorizationLimit`]. They never
//! return a wrong factorization and never report an unproven irreducibility.

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;

use super::modular::{
    MAX_MODULUS, ModPoly, MpPoly, Xorshift64, bigint_mod_inverse, is_small_prime,
};
use super::univariate::Poly;
use super::{Coeff, PolyError};

/// How many lucky primes are examined before committing to one.
pub const MAX_PRIME_CANDIDATES: usize = 5;

/// How many primes are tested before the search gives up.
///
/// A prime is unlucky only when it divides `lc(f) · disc(f)`, so exhausting this
/// many candidates requires that product to have more than
/// `MAX_PRIMES_TESTED` distinct prime divisors — astronomically improbable, but
/// reported honestly as [`PolyError::FactorizationLimit`] rather than guessed at.
pub const MAX_PRIMES_TESTED: usize = 4096;

/// Cantor–Zassenhaus attempts before a split is declared hopeless.
///
/// Each attempt splits with probability `≥ 1/2`, so the failure probability of
/// the whole budget is below `2^-1024`.
pub const MAX_EDF_ATTEMPTS: usize = 1024;

/// Upper bound on the number of subsets inspected during recombination.
///
/// This is the documented worst-case escape hatch of Zassenhaus' algorithm.
pub const MAX_RECOMBINATION_SUBSETS: usize = 1 << 20;

// ── Integer polynomial helpers (dense, ascending, high-order zeros stripped) ───

/// Strip high-order zero coefficients.
fn int_normalize(coeffs: &mut Vec<BigInt>) {
    while coeffs.last().is_some_and(|c| c.sign() == Sign::NoSign) {
        coeffs.pop();
    }
}

/// The (non-negative) content: the GCD of all coefficients.
fn int_content(coeffs: &[BigInt]) -> BigInt {
    let mut g = BigInt::from(0i32);
    for c in coeffs {
        g = g.gcd(c);
    }
    g
}

/// The primitive part, normalized to a positive leading coefficient.
fn int_primitive(coeffs: &[BigInt]) -> Vec<BigInt> {
    let mut out: Vec<BigInt> = coeffs.to_vec();
    int_normalize(&mut out);
    if out.is_empty() {
        return out;
    }
    let content = int_content(&out);
    if content.sign() != Sign::NoSign {
        for c in &mut out {
            *c /= &content;
        }
    }
    if out.last().is_some_and(|c| c.sign() == Sign::Minus) {
        for c in &mut out {
            *c = -&*c;
        }
    }
    out
}

/// The `L1` norm `Σ |a_i|`.
fn int_l1_norm(coeffs: &[BigInt]) -> BigInt {
    let mut sum = BigInt::from(0i32);
    for c in coeffs {
        sum += BigInt::from_biguint(Sign::Plus, c.magnitude().clone());
    }
    sum
}

/// Exact division in `ℤ[x]`: returns `f / g` when `g` divides `f` exactly, and
/// `None` otherwise (including when a single coefficient fails to divide).
///
/// Bails out at the first non-divisible coefficient, which makes it a very cheap
/// rejection test for a wrong recombination candidate.
fn int_divide_exact(f: &[BigInt], g: &[BigInt]) -> Option<Vec<BigInt>> {
    if g.is_empty() {
        return None;
    }
    if f.is_empty() {
        return Some(Vec::new());
    }
    let dg = g.len() - 1;
    let df = f.len() - 1;
    if df < dg {
        return None;
    }
    let lc = &g[dg];

    let mut rem: Vec<BigInt> = f.to_vec();
    let mut quo = vec![BigInt::from(0i32); df - dg + 1];

    for i in (dg..=df).rev() {
        let (q, r) = rem[i].div_rem(lc);
        if r.sign() != Sign::NoSign {
            return None;
        }
        if q.sign() == Sign::NoSign {
            continue;
        }
        let pos = i - dg;
        for (j, gj) in g.iter().enumerate() {
            rem[pos + j] -= &q * gj;
        }
        quo[pos] = q;
    }

    if rem.iter().any(|c| c.sign() != Sign::NoSign) {
        return None;
    }
    int_normalize(&mut quo);
    Some(quo)
}

/// Multiply two integer coefficient vectors modulo `m`.
fn mod_mul_coeffs(a: &[BigInt], b: &[BigInt], m: &BigInt) -> Vec<BigInt> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![BigInt::from(0i32); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.sign() == Sign::NoSign {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            if y.sign() == Sign::NoSign {
                continue;
            }
            out[i + j] = (&out[i + j] + x * y).mod_floor(m);
        }
    }
    int_normalize(&mut out);
    out
}

/// Symmetric representatives modulo the odd `m`, i.e. the unique integers of
/// absolute value `≤ (m − 1) / 2` congruent to each coefficient.
fn symmetric(coeffs: &[BigInt], m: &BigInt) -> Vec<BigInt> {
    coeffs
        .iter()
        .map(|c| {
            let r = c.mod_floor(m);
            if &r * 2i32 > *m { r - m } else { r }
        })
        .collect()
}

/// The primitive integer coefficient vector of a rational polynomial: clear
/// denominators, divide out the content, force a positive leading coefficient.
fn primitive_integer_coeffs(f: &Poly) -> Vec<BigInt> {
    let norm = f.normalized();
    if norm.coeffs.is_empty() {
        return Vec::new();
    }
    let mut lcm_denom = BigInt::from(1i32);
    for c in &norm.coeffs {
        lcm_denom = lcm_denom.lcm(c.denom());
    }
    let ints: Vec<BigInt> = norm
        .coeffs
        .iter()
        .map(|c| c.numer() * (&lcm_denom / c.denom()))
        .collect();
    int_primitive(&ints)
}

/// Rebuild a [`Poly`] from an integer coefficient vector.
fn poly_from_ints(coeffs: &[BigInt]) -> Poly {
    Poly::from_ratios(
        coeffs
            .iter()
            .map(|c| Coeff::from_integer(c.clone()))
            .collect(),
    )
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Factor a square-free rational polynomial into its irreducible factors over ℚ.
///
/// The input is first reduced to its primitive integer form (Gauss' lemma), so
/// the returned factors are **primitive integer polynomials with positive
/// leading coefficient** whose product is that primitive part. A constant (or
/// zero) input yields an empty factor list.
///
/// The input **must be square-free** — the caller ([`super::factor::Poly::factor`])
/// guarantees this by running Yun's decomposition first. Feeding a
/// non-square-free polynomial in would make every lucky-prime test fail and the
/// call would honestly report [`PolyError::FactorizationLimit`] rather than
/// return nonsense.
///
/// # Errors
///
/// Returns [`PolyError::FactorizationLimit`] if the lucky-prime search, the
/// Cantor–Zassenhaus attempt budget or the recombination subset cap is
/// exhausted. Returns [`PolyError::DivByZero`] only for internal invariant
/// violations, which are unreachable for a square-free input.
pub fn factor_square_free(f: &Poly) -> Result<Vec<Poly>, PolyError> {
    let ints = primitive_integer_coeffs(f);
    if ints.len() <= 1 {
        // The zero polynomial or a constant: no irreducible factors.
        return Ok(Vec::new());
    }
    if ints.len() == 2 {
        // Linear polynomials are irreducible.
        return Ok(vec![poly_from_ints(&ints)]);
    }

    let factors = factor_primitive_square_free(&ints)?;
    Ok(factors.iter().map(|c| poly_from_ints(c)).collect())
}

/// The integer-level driver: the five stages of the pipeline, in order.
fn factor_primitive_square_free(ints: &[BigInt]) -> Result<Vec<Vec<BigInt>>, PolyError> {
    // ── 1. lucky prime + distinct-degree factorization ────────────────────────
    let (prime, ddf) = select_prime(ints)?;

    let modular_count: usize = ddf.iter().map(|(d, g)| g.degree().unwrap_or(0) / d).sum();
    if modular_count <= 1 {
        // f is irreducible modulo p, hence irreducible over ℚ.
        return Ok(vec![ints.to_vec()]);
    }

    // ── 2/3. equal-degree factorization ───────────────────────────────────────
    let modular = equal_degree_factors(&ddf, prime)?;
    if modular.len() <= 1 {
        return Ok(vec![ints.to_vec()]);
    }

    // ── 4. Mignotte bound → lift exponent → Hensel lift ───────────────────────
    let bound = mignotte_bound(ints);
    let (modulus, exponent) = lift_exponent(prime, &bound);
    let lifted = hensel_lift(ints, &modular, prime, exponent)?;

    // ── 5. recombination ──────────────────────────────────────────────────────
    recombine(ints, &lifted, &modulus, &bound)
}

// ── 1. Prime selection ────────────────────────────────────────────────────────

/// Distinct-degree factorization output: `(degree, product of the irreducible
/// factors of that degree)`.
type DdfParts = Vec<(usize, ModPoly)>;

/// Find a lucky odd prime and return it together with the distinct-degree
/// factorization of `f mod p`.
///
/// Up to [`MAX_PRIME_CANDIDATES`] lucky primes are examined and the one with the
/// fewest modular factors is kept, because recombination is exponential in that
/// count. A prime that proves `f` irreducible modulo `p` short-circuits.
fn select_prime(ints: &[BigInt]) -> Result<(i128, DdfParts), PolyError> {
    let degree = ints.len() - 1;
    let lc = &ints[degree];

    let mut best: Option<(i128, DdfParts, usize)> = None;
    let mut lucky = 0usize;
    let mut tested = 0usize;
    let mut candidate: i128 = 3;

    while candidate < MAX_MODULUS && tested < MAX_PRIMES_TESTED {
        let prime = candidate;
        candidate += 2;
        if !is_small_prime(prime) {
            continue;
        }
        tested += 1;

        // p must not divide the leading coefficient, or the degree would drop.
        if lc.mod_floor(&BigInt::from(prime)).sign() == Sign::NoSign {
            continue;
        }

        let reduced = ModPoly::from_bigint_coeffs(ints, prime).monic()?;
        if reduced.degree() != Some(degree) {
            continue;
        }
        // A repeated modular factor would break the coprimality that Hensel
        // lifting requires.
        if !reduced.is_square_free()? {
            continue;
        }

        let parts = distinct_degree(&reduced)?;
        let count: usize = parts.iter().map(|(d, g)| g.degree().unwrap_or(0) / d).sum();

        lucky += 1;
        let improved = match &best {
            Some((_, _, best_count)) => count < *best_count,
            None => true,
        };
        if improved {
            best = Some((prime, parts, count));
        }
        if count == 1 || lucky >= MAX_PRIME_CANDIDATES {
            break;
        }
    }

    match best {
        Some((prime, parts, _)) => Ok((prime, parts)),
        None => Err(PolyError::FactorizationLimit),
    }
}

// ── 2. Distinct-degree factorization ──────────────────────────────────────────

/// Split a monic square-free `f ∈ GF(p)[x]` by the degrees of its irreducible
/// factors.
///
/// Returns `(d, g_d)` pairs where `g_d` is the product of all monic irreducible
/// factors of `f` of degree exactly `d`. Uses the Frobenius iteration
/// `w ← w^p mod f` starting from `w = x`, so that after the `d`-th round
/// `w = x^(p^d) mod f` and `gcd(f, w − x)` peels off the degree-`d` part.
fn distinct_degree(f: &ModPoly) -> Result<DdfParts, PolyError> {
    let prime = f.modulus();
    let exp = BigUint::from(prime.unsigned_abs());

    let mut parts: DdfParts = Vec::new();
    let mut remaining = f.clone();
    let mut w = ModPoly::identity(prime);
    let mut d = 1usize;

    while remaining.degree().unwrap_or(0) >= 2 * d {
        w = w.pow_mod(&exp, &remaining)?;
        let g = ModPoly::gcd(&remaining, &w.sub(&ModPoly::identity(prime)))?;
        if g.degree().unwrap_or(0) > 0 {
            parts.push((d, g.clone()));
            remaining = remaining.div_rem(&g)?.0;
            w = w.rem(&remaining)?;
        }
        d += 1;
    }

    if let Some(left) = remaining.degree() {
        if left > 0 {
            parts.push((left, remaining));
        }
    }
    Ok(parts)
}

// ── 3. Cantor–Zassenhaus equal-degree factorization ───────────────────────────

/// Split every distinct-degree part into individual monic irreducible factors.
///
/// The result is sorted canonically (by degree, then lexicographically by
/// coefficients) so that the whole factorization is a deterministic function of
/// its input, despite Cantor–Zassenhaus being randomized.
fn equal_degree_factors(parts: &DdfParts, prime: i128) -> Result<Vec<ModPoly>, PolyError> {
    let mut out: Vec<ModPoly> = Vec::new();

    for (degree, part) in parts {
        // Seed the PRNG from the part being split, so the random choices are
        // reproducible across runs and machines.
        let mut rng = Xorshift64::new(seed_from(part, prime, *degree));
        let exp =
            (BigUint::from(prime.unsigned_abs()).pow(*degree as u32) - BigUint::from(1u32)) >> 1u32;
        equal_degree_split(part, *degree, &exp, &mut rng, &mut out)?;
    }

    out.sort_by(|a, b| {
        a.coeffs()
            .len()
            .cmp(&b.coeffs().len())
            .then_with(|| a.coeffs().cmp(b.coeffs()))
    });
    Ok(out)
}

/// An FNV-1a hash of the polynomial, used to seed the equal-degree PRNG.
fn seed_from(f: &ModPoly, prime: i128, degree: usize) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(prime as u64);
    mix(degree as u64);
    for &c in f.coeffs() {
        mix(c as u64);
    }
    hash
}

/// Cantor–Zassenhaus: recursively split `g`, a product of `deg g / degree`
/// distinct monic irreducibles all of degree `degree`, over an odd `GF(p)`.
///
/// `exp` must be `(p^degree − 1) / 2`, precomputed once by the caller.
fn equal_degree_split(
    g: &ModPoly,
    degree: usize,
    exp: &BigUint,
    rng: &mut Xorshift64,
    out: &mut Vec<ModPoly>,
) -> Result<(), PolyError> {
    let n = match g.degree() {
        Some(0) | None => return Ok(()),
        Some(d) => d,
    };
    if n == degree {
        out.push(g.monic()?);
        return Ok(());
    }

    let prime = g.modulus();
    let one = ModPoly::one(prime);

    for _ in 0..MAX_EDF_ATTEMPTS {
        let a = random_poly(rng, n, prime);
        if a.degree().unwrap_or(0) < 1 {
            continue;
        }

        // A non-trivial gcd(a, g) is already a split (it means `a` lands on zero
        // in at least one — but not all — CRT components).
        let mut split = ModPoly::gcd(g, &a)?;
        if split.degree() == Some(0) {
            let b = a.pow_mod(exp, g)?;
            split = ModPoly::gcd(g, &b.sub(&one))?;
        }

        let split_degree = split.degree().unwrap_or(0);
        if split_degree > 0 && split_degree < n {
            let cofactor = g.div_rem(&split)?.0;
            equal_degree_split(&split.monic()?, degree, exp, rng, out)?;
            equal_degree_split(&cofactor.monic()?, degree, exp, rng, out)?;
            return Ok(());
        }
    }

    Err(PolyError::FactorizationLimit)
}

/// A pseudo-random polynomial of degree `< bound` over `GF(p)`.
fn random_poly(rng: &mut Xorshift64, bound: usize, prime: i128) -> ModPoly {
    let coeffs = (0..bound).map(|_| rng.next_residue(prime)).collect();
    ModPoly::new(coeffs, prime)
}

// ── 4. Mignotte bound, lift exponent and Hensel lifting ───────────────────────

/// The Landau–Mignotte bound `B = 2^n · ⌈‖f‖₂⌉`.
///
/// Every divisor `g` of `f` satisfies `‖(lc f / lc g) · g‖₁ ≤ B` — see the
/// module documentation for the derivation. Rounding `‖f‖₂` *up* to an integer
/// keeps the bound valid.
fn mignotte_bound(ints: &[BigInt]) -> BigInt {
    let degree = ints.len().saturating_sub(1);

    let mut sum_of_squares = BigInt::from(0i32);
    for c in ints {
        sum_of_squares += c * c;
    }

    // Integer ceiling of the square root.
    let root = sum_of_squares.sqrt();
    let norm2 = if &root * &root < sum_of_squares {
        root + 1i32
    } else {
        root
    };

    norm2 << degree
}

/// The smallest `l` with `p^l ≥ 2B + 1`, together with `p^l` itself.
///
/// Integers of absolute value `≤ B` then have a unique symmetric representative
/// modulo `p^l`, which is what makes the recombination test exact.
fn lift_exponent(prime: i128, bound: &BigInt) -> (BigInt, usize) {
    let prime_big = BigInt::from(prime);
    let target = bound * 2i32 + 1i32;

    let mut exponent = 1usize;
    let mut modulus = prime_big.clone();
    while modulus < target {
        modulus *= &prime_big;
        exponent += 1;
    }
    (modulus, exponent)
}

/// A node of the Hensel factor tree.
struct HenselTree {
    /// The current lift of the product of all leaves below this node.
    poly: MpPoly,
    kind: HenselKind,
}

enum HenselKind {
    /// Index of the modular factor this leaf carries.
    Leaf(usize),
    Internal(Box<HenselInternal>),
}

struct HenselInternal {
    left: HenselTree,
    right: HenselTree,
    /// Bézout cofactors with `s · left.poly + t · right.poly ≡ 1`.
    s: MpPoly,
    t: MpPoly,
}

/// Lift `f ≡ lc(f) · Π h_i (mod p)` to `f ≡ lc(f) · Π ĥ_i (mod p^exponent)`.
///
/// Returns the monic lifted factors `ĥ_i`, in the same order as `modular`, as
/// coefficient vectors reduced into `[0, p^exponent)`.
fn hensel_lift(
    ints: &[BigInt],
    modular: &[ModPoly],
    prime: i128,
    exponent: usize,
) -> Result<Vec<Vec<BigInt>>, PolyError> {
    let prime_big = BigInt::from(prime);
    let indices: Vec<usize> = (0..modular.len()).collect();
    let (mut tree, _) = build_tree(modular, &indices, &prime_big)?;

    // Exponent schedule 1, …, ⌈l/4⌉, ⌈l/2⌉, l. Consecutive entries satisfy
    // e_{i+1} ≤ 2·e_i, which is exactly the precondition of the Hensel step.
    let mut schedule = vec![exponent];
    let mut e = exponent;
    while e > 1 {
        e = e.div_ceil(2);
        schedule.push(e);
    }
    schedule.reverse();

    for &e in schedule.iter().skip(1) {
        let modulus = prime_big.pow(e as u32);
        let target = monic_image(ints, &modulus)?;
        lift_tree(&mut tree, &target, &modulus)?;
    }

    let mut leaves: Vec<(usize, MpPoly)> = Vec::with_capacity(modular.len());
    collect_leaves(&tree, &mut leaves);
    leaves.sort_by_key(|(index, _)| *index);
    Ok(leaves
        .into_iter()
        .map(|(_, poly)| poly.into_coeffs())
        .collect())
}

/// `lc(f)^{-1} · f (mod m)`: the monic image of `f` that the lift targets.
fn monic_image(ints: &[BigInt], modulus: &BigInt) -> Result<MpPoly, PolyError> {
    let lc = ints.last().ok_or(PolyError::DivByZero)?;
    let inverse = bigint_mod_inverse(lc, modulus).ok_or(PolyError::DivByZero)?;
    let coeffs = ints
        .iter()
        .map(|c| (c * &inverse).mod_floor(modulus))
        .collect();
    Ok(MpPoly::new(coeffs, modulus))
}

/// Build a balanced product tree over `modular[indices]`, with Bézout cofactors
/// at every internal node.
///
/// Returns the tree (with all polynomials reduced mod `p`) and the `GF(p)`
/// product of its leaves, which the parent needs for its own Bézout step.
fn build_tree(
    modular: &[ModPoly],
    indices: &[usize],
    prime_big: &BigInt,
) -> Result<(HenselTree, ModPoly), PolyError> {
    if indices.len() == 1 {
        let index = indices[0];
        let leaf = modular[index].clone();
        let tree = HenselTree {
            poly: MpPoly::from_mod_poly(&leaf, prime_big),
            kind: HenselKind::Leaf(index),
        };
        return Ok((tree, leaf));
    }

    let mid = indices.len() / 2;
    let (left, left_poly) = build_tree(modular, &indices[..mid], prime_big)?;
    let (right, right_poly) = build_tree(modular, &indices[mid..], prime_big)?;

    let (s, t) = bezout(&left_poly, &right_poly)?;
    let product = left_poly.mul(&right_poly);

    let tree = HenselTree {
        poly: MpPoly::from_mod_poly(&product, prime_big),
        kind: HenselKind::Internal(Box::new(HenselInternal {
            left,
            right,
            s: MpPoly::from_mod_poly(&s, prime_big),
            t: MpPoly::from_mod_poly(&t, prime_big),
        })),
    };
    Ok((tree, product))
}

/// Bézout cofactors `s·g + t·h = 1` over `GF(p)`, normalized to
/// `deg s < deg h` and `deg t < deg g`.
///
/// The Hensel step's degree analysis relies on exactly those bounds, so they are
/// enforced explicitly rather than inherited from the Euclidean remainder
/// sequence: `s` is reduced modulo `h` and `t` is recomputed as `(1 − s·g) / h`,
/// an exact division because `s·g ≡ 1 (mod h)`.
fn bezout(g: &ModPoly, h: &ModPoly) -> Result<(ModPoly, ModPoly), PolyError> {
    let prime = g.modulus();
    let (gcd, s, _) = ModPoly::xgcd(g, h)?;
    if gcd.degree() != Some(0) {
        // The modular factors are pairwise coprime because `f mod p` is
        // square-free; reaching this means a lucky-prime invariant was broken.
        return Err(PolyError::DivByZero);
    }

    let s = s.rem(h)?;
    let numerator = ModPoly::one(prime).sub(&s.mul(g));
    let (t, remainder) = numerator.div_rem(h)?;
    if !remainder.is_zero() {
        return Err(PolyError::DivByZero);
    }
    Ok((s, t))
}

/// Lift a whole subtree to the modulus `m`, given the lifted target for its own
/// product.
fn lift_tree(tree: &mut HenselTree, target: &MpPoly, modulus: &BigInt) -> Result<(), PolyError> {
    tree.poly = target.clone();

    if let HenselKind::Internal(node) = &mut tree.kind {
        let g = node.left.poly.with_modulus(modulus);
        let h = node.right.poly.with_modulus(modulus);
        let s = node.s.with_modulus(modulus);
        let t = node.t.with_modulus(modulus);

        let lifted = hensel_step(target, &g, &h, &s, &t)?;
        node.s = lifted.s;
        node.t = lifted.t;

        lift_tree(&mut node.left, &lifted.g, modulus)?;
        lift_tree(&mut node.right, &lifted.h, modulus)?;
    }
    Ok(())
}

/// The output of one quadratic Hensel step.
struct HenselStep {
    g: MpPoly,
    h: MpPoly,
    s: MpPoly,
    t: MpPoly,
}

/// One quadratic (Newton) Hensel step, raising the precision from `m` to any
/// `m′ | m²` — the modulus carried by `f`, `g`, `h`, `s` and `t`.
///
/// # Preconditions
///
/// `f ≡ g·h (mod m)`, `s·g + t·h ≡ 1 (mod m)`, `h` monic, `g` monic,
/// `deg s < deg h`, `deg t ≤ deg g`, all arguments already re-based to `m′`.
///
/// # Postconditions
///
/// `f ≡ g*·h* (mod m′)`, `s*·g* + t*·h* ≡ 1 (mod m′)`, `g*`, `h*` monic of the
/// same degrees, and everything is congruent to its input modulo `m`.
///
/// # Errors
///
/// Returns [`PolyError::DivByZero`] if a divisor is not monic, which the
/// preconditions rule out.
fn hensel_step(
    f: &MpPoly,
    g: &MpPoly,
    h: &MpPoly,
    s: &MpPoly,
    t: &MpPoly,
) -> Result<HenselStep, PolyError> {
    // Newton correction of the factorization.
    let error = f.sub(&g.mul(h));
    let (q, r) = s.mul(&error).div_rem_monic(h)?;

    let g_new = g.add(&t.mul(&error)).add(&q.mul(g));
    let h_new = h.add(&r);

    debug_assert_eq!(
        g_new.degree(),
        g.degree(),
        "Hensel step must preserve the degree of g"
    );
    debug_assert!(h_new.is_monic(), "Hensel step must preserve monicity of h");

    // Newton correction of the Bézout relation, so the next step can reuse it.
    let defect = s
        .mul(&g_new)
        .add(&t.mul(&h_new))
        .sub(&MpPoly::one(f.modulus()));
    let (u, v) = s.mul(&defect).div_rem_monic(&h_new)?;

    let s_new = s.sub(&v);
    let t_new = t.sub(&t.mul(&defect)).sub(&u.mul(&g_new));

    Ok(HenselStep {
        g: g_new,
        h: h_new,
        s: s_new,
        t: t_new,
    })
}

/// Gather the lifted leaves, tagged with their original index.
fn collect_leaves(tree: &HenselTree, out: &mut Vec<(usize, MpPoly)>) {
    match &tree.kind {
        HenselKind::Leaf(index) => out.push((*index, tree.poly.clone())),
        HenselKind::Internal(node) => {
            collect_leaves(&node.left, out);
            collect_leaves(&node.right, out);
        }
    }
}

// ── 5. Zassenhaus recombination ───────────────────────────────────────────────

/// The state a subset search needs; bundled so the recursion stays readable.
struct Recombiner<'a> {
    /// The lifted modular factors, mod `p^l`.
    lifted: &'a [Vec<BigInt>],
    /// Indices of the factors not yet used up.
    pool: &'a [usize],
    /// `p^l`.
    modulus: &'a BigInt,
    /// The Landau–Mignotte bound `B`.
    bound: &'a BigInt,
    /// Leading coefficient of the polynomial still to be factored.
    lead: &'a BigInt,
    /// The polynomial still to be factored.
    current: &'a [BigInt],
    /// Remaining subset budget.
    budget: &'a mut usize,
}

/// A successful recombination: the subset used, the irreducible factor found and
/// the cofactor left over.
struct Hit {
    subset: Vec<usize>,
    factor: Vec<BigInt>,
    cofactor: Vec<BigInt>,
}

impl Recombiner<'_> {
    /// Depth-first search over the size-`need` subsets of `pool[start..]`,
    /// carrying the running product of the chosen leaves modulo `p^l`.
    fn search(
        &mut self,
        start: usize,
        need: usize,
        product: &[BigInt],
        chosen: &mut Vec<usize>,
    ) -> Result<Option<Hit>, PolyError> {
        if need == 0 {
            return self.test(product, chosen);
        }
        if self.pool.len() < start + need {
            return Ok(None);
        }

        for position in start..=(self.pool.len() - need) {
            let index = self.pool[position];
            let next = mod_mul_coeffs(product, &self.lifted[index], self.modulus);
            chosen.push(index);
            let hit = self.search(position + 1, need - 1, &next, chosen)?;
            chosen.pop();
            if hit.is_some() {
                return Ok(hit);
            }
        }
        Ok(None)
    }

    /// Test one candidate subset.
    ///
    /// `product` is `Π_{i ∈ chosen} ĥ_i (mod p^l)`. Scaling by the current
    /// leading coefficient and symmetrizing recovers `(lc / lc g) · g` exactly
    /// whenever `chosen` is the subset belonging to a genuine factor `g`; the
    /// two filters below reject the rest cheaply, and exact division decides.
    fn test(&mut self, product: &[BigInt], chosen: &[usize]) -> Result<Option<Hit>, PolyError> {
        if *self.budget == 0 {
            return Err(PolyError::FactorizationLimit);
        }
        *self.budget -= 1;

        let scaled: Vec<BigInt> = product
            .iter()
            .map(|c| (c * self.lead).mod_floor(self.modulus))
            .collect();
        let candidate = symmetric(&scaled, self.modulus);
        if candidate.is_empty() {
            return Ok(None);
        }

        // Filter 1: the constant coefficient of (lc/lc g)·g divides lc · f(0),
        // because g(0) divides f(0).
        let constant = &self.current[0];
        if constant.sign() != Sign::NoSign {
            let candidate_constant = &candidate[0];
            if candidate_constant.sign() == Sign::NoSign {
                return Ok(None);
            }
            if (constant * self.lead).mod_floor(candidate_constant).sign() != Sign::NoSign {
                return Ok(None);
            }
        }

        // Filter 2: the Landau–Mignotte bound must hold for a genuine factor.
        if int_l1_norm(&candidate) > *self.bound {
            return Ok(None);
        }

        let factor = int_primitive(&candidate);
        if factor.len() <= 1 {
            return Ok(None);
        }

        match int_divide_exact(self.current, &factor) {
            Some(cofactor) => Ok(Some(Hit {
                subset: chosen.to_vec(),
                factor,
                cofactor,
            })),
            None => Ok(None),
        }
    }
}

/// Recombine the lifted modular factors into the irreducible integer factors.
///
/// Subsets are searched by increasing size, so the first divisor found for a
/// given size is irreducible (all smaller subsets of the current pool have
/// already been rejected). The loop stops when no subset of size `≤ |pool| / 2`
/// can be formed, at which point the leftover is irreducible.
///
/// # Errors
///
/// Returns [`PolyError::FactorizationLimit`] when more than
/// [`MAX_RECOMBINATION_SUBSETS`] subsets have been inspected.
fn recombine(
    ints: &[BigInt],
    lifted: &[Vec<BigInt>],
    modulus: &BigInt,
    bound: &BigInt,
) -> Result<Vec<Vec<BigInt>>, PolyError> {
    let mut pool: Vec<usize> = (0..lifted.len()).collect();
    let mut current: Vec<BigInt> = ints.to_vec();
    let mut lead: BigInt = current.last().cloned().ok_or(PolyError::DivByZero)?;
    let mut factors: Vec<Vec<BigInt>> = Vec::new();
    let mut budget = MAX_RECOMBINATION_SUBSETS;
    let mut size = 1usize;

    while 2 * size <= pool.len() {
        let hit = {
            let mut recombiner = Recombiner {
                lifted,
                pool: &pool,
                modulus,
                bound,
                lead: &lead,
                current: &current,
                budget: &mut budget,
            };
            let mut chosen: Vec<usize> = Vec::with_capacity(size);
            recombiner.search(0, size, &[BigInt::from(1i32)], &mut chosen)?
        };

        match hit {
            Some(hit) => {
                pool.retain(|index| !hit.subset.contains(index));
                current = hit.cofactor;
                lead = current.last().cloned().ok_or(PolyError::DivByZero)?;
                factors.push(hit.factor);
            }
            None => size += 1,
        }
    }

    // Whatever survives corresponds to the remaining modular factors and is
    // irreducible: any proper divisor would have shown up as a subset of size
    // at most |pool| / 2, and all of those were rejected.
    factors.push(int_primitive(&current));
    Ok(factors)
}
