use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul, Neg, Sub};

use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_maybe_rayon::prelude::*;

use crate::domain::CircleDomain;

/// Affine representation of a point on the circle.
/// x^2 + y^2 == 1
// _private is to prevent construction so we can debug assert the invariant
#[allow(clippy::manual_non_exhaustive)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Point<F> {
    pub x: F,
    pub y: F,
    _private: (),
}

impl<F: Field> Point<F> {
    #[inline]
    pub fn new(x: F, y: F) -> Self {
        debug_assert_eq!(x.square() + y.square(), F::ONE);
        Self { x, y, _private: () }
    }

    const ZERO: Self = Self {
        x: F::ONE,
        y: F::ZERO,
        _private: (),
    };

    /// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
    /// ```ignore
    /// (x, y) = ((1-t^2)/(1+t^2), 2t/(1+t^2))
    /// ```
    /// Panics if t^2 = -1, corresponding to either of the points at infinity
    /// (on the projective *circle*) (1 : ±i : 0)
    pub fn from_projective_line(t: F) -> Self {
        let t2 = t.square();
        let inv_denom = (F::ONE + t2).try_inverse().expect("t^2 = -1");
        Self::new((F::ONE - t2) * inv_denom, t.double() * inv_denom)
    }

    /// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
    /// ```ignore
    /// t = y / (x + 1)
    /// ```
    /// Returns None if self.x = -1, corresponding to Inf on the projective line
    ///
    /// This is also used as a selector polynomial, with a simple zero at (1,0)
    /// and a simple pole at (-1,0), which in the paper is called v_0
    /// Circle STARKs, Section 5.1, Lemma 11 (page 21 of the first revision PDF)
    pub fn to_projective_line(self) -> Option<F> {
        (self.x + F::ONE).try_inverse().map(|x| x * self.y)
    }

    /// The "squaring map", or doubling in additive notation, denoted π(x,y)
    /// Circle STARKs, Section 3.1, Equation 1: (page 5 of the first revision PDF)
    pub fn double(self) -> Self {
        Self::new(self.x.square().double() - F::ONE, self.x.double() * self.y)
    }

    /// Apply the doubling map `n` times: π^n(x,y)
    pub fn repeated_double(mut self, n: usize) -> Self {
        for _ in 0..n {
            self = self.double();
        }
        self
    }

    /// Evaluate the vanishing polynomial for the standard position coset of size 2^log_n
    /// at this point
    /// Circle STARKs, Section 3.3, Equation 8 (page 10 of the first revision PDF)
    pub fn v_n(mut self, log_n: usize) -> F {
        debug_assert!(log_n >= 1, "v_n requires log_n >= 1");
        for _ in 0..log_n.saturating_sub(1) {
            self.x = self.x.square().double() - F::ONE; // TODO: replace this by a custom field impl.
        }
        self.x
    }

    /// Compute a product of successive `v_n`'s.
    ///
    /// More explicitly this computes `(1..log_n).map(|i| self.v_n(i)).product()`
    /// but uses far fewer `self.x.square().double() - F::ONE` steps compared to the naive implementation.
    pub fn v_n_prod(mut self, log_n: usize) -> F {
        if log_n <= 1 {
            return F::ONE;
        }
        let mut output = self.x;
        for _ in 0..(log_n - 2) {
            self.x = self.x.square().double() - F::ONE; // TODO: replace this by a custom field impl.
            output *= self.x;
        }
        output
    }

    /// Evaluate the selector function which is zero at `self` and nonzero elsewhere, at `at`.
    /// Called v_0 . T_p⁻¹ or ṽ_p(x,y) in the paper, used for constraint selectors.
    /// Panics if p = -self, the pole.
    /// Section 5.1, Lemma 11 of Circle Starks (page 21 of first edition PDF)
    pub fn v_tilde_p<EF: ExtensionField<F>>(self, at: Point<EF>) -> EF {
        (at - self).to_projective_line().unwrap()
    }

    /// Return the numerator and denominator of the reciprocal selector `1 / v_tilde_p(self, at)`.
    ///
    /// More precisely, if `v_tilde_p(self, at) = denom / numer`, then its reciprocal is
    /// `numer / denom`. This form lets callers batch-invert only `denom` values.
    #[inline]
    pub(crate) fn recip_v_tilde_p_num_den<EF: ExtensionField<F>>(self, at: Point<EF>) -> (EF, EF) {
        let diff = at - self;
        (diff.x + EF::ONE, diff.y)
    }

    /// The concrete value of the selector s_P = v_n / (v_0 . T_p⁻¹) at P=self, used for normalization.
    /// Circle STARKs, Section 5.1, Remark 16 (page 22 of the first revision PDF)
    pub fn s_p_at_p(self, log_n: usize) -> F {
        debug_assert!(log_n >= 1, "s_p_at_p requires log_n >= 1");
        -self.v_n_prod(log_n).mul_2exp_u64((2 * log_n - 1) as u64) * self.y
    }

    /// Evaluate the alternate single-point vanishing function v_p(x), used for DEEP quotient.
    /// Returns (a, b), representing the complex number a + bi.
    /// Simple zero at p, simple pole at +-infinity.
    /// Circle STARKs, Section 3.3, Equation 11 (page 11 of the first edition PDF).
    pub fn v_p<EF: ExtensionField<F>>(self, at: Point<EF>) -> (EF, EF) {
        let diff = -at + self;
        (EF::ONE - diff.x, -diff.y)
    }
}

/// Compute Lagrange denominators for CFFT-ordered points of `domain`.
///
/// Let `k = domain.log_n` and let `g` generate the subgroup of order `2^(k-1)`. Then
/// `s_p_at_p(P, k) = -2^k * (2^(k-1) P).y`, and doubling `k - 1` times sends the half-coset
/// `shift + <g>` to `2^(k-1) shift` and the half-coset `-shift + <g>` to its negation, for any
/// `shift`. So `s_p_at_p` equals `s_p_at_p(shift, k)` on the first half and its negation on the
/// second, which avoids recomputing the `s_p` chain for every point.
///
/// `points[i]` must lie in `shift + <g>` exactly when `i` is even. CFFT ordering satisfies this;
/// any ordering that does not yields wrong denominators.
pub(crate) fn compute_lagrange_den_on_domain<F: ComplexExtendable, EF: ExtensionField<F>>(
    points: &[Point<F>],
    at: Point<EF>,
    domain: CircleDomain<F>,
) -> Vec<EF> {
    let n = points.len();
    debug_assert_eq!(n, 1 << domain.log_n);

    let s_p_at_shift = domain.shift.s_p_at_p(domain.log_n);
    let s_p_at_index = |i: usize| {
        if i & 1 == 0 {
            s_p_at_shift
        } else {
            -s_p_at_shift
        }
    };
    // Spot-check the parity precondition on both ends of the slice.
    debug_assert!(
        [0, 1, n - 2, n - 1]
            .into_iter()
            .all(|i| points[i].s_p_at_p(domain.log_n) == s_p_at_index(i)),
        "points do not alternate between the half-cosets of the domain"
    );

    let (numer, denom): (Vec<_>, Vec<_>) = points
        .par_iter()
        .enumerate()
        .map(|(i, &pt)| {
            let diff = at - pt;
            (diff.x + F::ONE, diff.y * s_p_at_index(i))
        })
        .unzip();

    let inv_d = batch_multiplicative_inverse(&denom);
    numer
        .par_iter()
        .zip(inv_d.par_iter())
        .map(|(&num, &inv_d)| num * inv_d)
        .collect()
}

impl<F: ComplexExtendable> Point<F> {
    pub fn generator(log_n: usize) -> Self {
        let g = F::circle_two_adic_generator(log_n);
        Self::new(g.real(), g.imag())
    }
}

/// Circle STARKs, Section 3.1, Equation 2: (page 5 of the first revision PDF)
/// The inverse map J(x,y) = (x,-y)
impl<F: Field> Neg for Point<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.y = -self.y;
        self
    }
}

impl<F: Field, EF: ExtensionField<F>> Add<Point<F>> for Point<EF> {
    type Output = Self;
    fn add(self, rhs: Point<F>) -> Self::Output {
        Self::new(
            self.x * rhs.x - self.y * rhs.y,
            self.x * rhs.y + self.y * rhs.x,
        )
    }
}

impl<F: Field> AddAssign for Point<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub<Point<F>> for Point<EF> {
    type Output = Self;
    fn sub(self, rhs: Point<F>) -> Self::Output {
        Self::new(
            self.x * rhs.x + self.y * rhs.y,
            self.y * rhs.x - self.x * rhs.y,
        )
    }
}

impl<F: Field> Mul<usize> for Point<F> {
    type Output = Self;
    fn mul(mut self, mut rhs: usize) -> Self::Output {
        let mut res = Self::ZERO;
        while rhs != 0 {
            if rhs & 1 == 1 {
                res += self;
            }
            rhs >>= 1;
            self = self.double();
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_mersenne_31::Mersenne31;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;
    type Pt = Point<F>;

    #[test]
    fn test_arithmetic() {
        let one = Pt::generator(3);
        assert_eq!(one - one, Pt::ZERO);
        assert_eq!(one + one, one * 2);
        assert_eq!(one + one + one, one * 3);
        assert_eq!(one * 7, -one);
        assert_eq!(one * 8, Pt::ZERO);

        let generator = Pt::generator(10);
        let log_n = 10;
        let vn_prod_gen = (1..log_n).map(|i| generator.v_n(i)).product();
        assert_eq!(generator.v_n_prod(log_n), vn_prod_gen);
    }

    #[test]
    fn recip_v_tilde_p_num_den_matches_selector_value() {
        let p = Pt::generator(8);
        let at = Point::<EF>::from_projective_line(EF::from(F::new(7)));

        let (numer, denom) = p.recip_v_tilde_p_num_den(at);
        let diff = at - p;

        assert_eq!(numer, diff.x + EF::ONE);
        assert_eq!(denom, diff.y);
        assert_eq!(p.v_tilde_p(at), diff.to_projective_line().unwrap());
        assert_eq!(diff.to_projective_line().unwrap() * numer, denom);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "v_n requires log_n >= 1")]
    fn test_v_n_underflow_log_n_0() {
        let p = Pt::generator(3);
        let _ = p.v_n(0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "s_p_at_p requires log_n >= 1")]
    fn test_s_p_at_p_underflow_log_n_0() {
        let p = Pt::generator(3);
        let _ = p.s_p_at_p(0);
    }

    /// Independent reference: the pre-batched formulation, one inversion per point.
    fn lagrange_den_scalar(points: &[Pt], at: Point<EF>, log_n: usize) -> Vec<EF> {
        points
            .iter()
            .map(|&pt| {
                let diff = at - pt;
                let numer = diff.x + F::ONE;
                let denom = diff.y * pt.s_p_at_p(log_n);
                numer * denom.inverse()
            })
            .collect()
    }

    proptest! {
        #[test]
        fn compute_lagrange_den_on_domain_matches_scalar(
            log_n in 1usize..11,
            at_seed in any::<u64>(),
        ) {
            let domain = crate::CircleDomain::standard(log_n);
            let points = crate::cfft_permute_slice(&domain.points().collect::<Vec<_>>());

            // A pseudo-random extension point stands in for the out-of-domain query.
            let mut rng = SmallRng::seed_from_u64(at_seed);
            let at = Point::<EF>::from_projective_line(rng.random());

            // Discard the measure-zero draws that would invert a zero denominator.
            let all_invertible = points
                .iter()
                .all(|&pt| (at - pt).y * pt.s_p_at_p(log_n) != EF::ZERO);
            prop_assume!(all_invertible);

            prop_assert_eq!(
                compute_lagrange_den_on_domain(&points, at, domain),
                lagrange_den_scalar(&points, at, log_n)
            );
        }
    }

    #[test]
    fn compute_lagrange_den_on_nonstandard_domain_matches_scalar() {
        for log_n in 1..10 {
            let domain = crate::CircleDomain::new(log_n, Pt::generator(log_n + 2));
            let points = crate::cfft_permute_slice(&domain.points().collect::<Vec<_>>());
            let at = Point::<EF>::from_projective_line(EF::from_u8(9));

            assert_eq!(
                compute_lagrange_den_on_domain(&points, at, domain),
                lagrange_den_scalar(&points, at, log_n),
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "points do not alternate between the half-cosets of the domain")]
    fn compute_lagrange_den_on_domain_rejects_reversed_points() {
        let domain = crate::CircleDomain::<F>::standard(4);
        let mut points = crate::cfft_permute_slice(&domain.points().collect::<Vec<_>>());
        points.reverse();
        let at = Point::<EF>::from_projective_line(EF::from_u8(9));

        let _ = compute_lagrange_den_on_domain(&points, at, domain);
    }
}
