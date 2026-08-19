use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
#[cfg(test)]
use p3_field::{batch_multiplicative_inverse, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use p3_util::log2_strict_usize;

use super::{Fraction, FractionGkrLayerProof, FractionGkrOutput, FractionGkrProof, SplitFraction};

struct InputLayer<'a, F: Field, EF: ExtensionField<F>> {
    fraction: &'a Fraction<Poly<F>, PolyMaybePacked<F, EF>>,
    eq_suffix: PolyMaybePacked<F, EF>,
}

struct Layer<F: Field, EF: ExtensionField<F>> {
    fraction: SplitFractionMaybePacked<F, EF>,
    eq_suffix: PolyMaybePacked<F, EF>,
    eq_prefix: EF,
}

enum SplitFractionMaybePacked<F: Field, EF: ExtensionField<F>> {
    Packed(SplitFraction<Poly<EF::ExtensionPacking>>),
    Scalar(SplitFraction<Poly<EF>>),
}

#[allow(clippy::too_many_arguments)]
fn mixed_round_polys<N, A>(
    eq_suffix: &[A],
    n0_lo: &[N],
    n0_hi: &[N],
    n1_lo: &[N],
    n1_hi: &[N],
    d0_lo: &[A],
    d0_hi: &[A],
    d1_lo: &[A],
    d1_hi: &[A],
    lambda: A,
) -> [A; 2]
where
    N: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<N> + Copy + Send + Sync,
{
    debug_assert_eq!(eq_suffix.len(), n0_lo.len());
    debug_assert_eq!(eq_suffix.len(), n0_hi.len());
    debug_assert_eq!(eq_suffix.len(), n1_lo.len());
    debug_assert_eq!(eq_suffix.len(), n1_hi.len());
    debug_assert_eq!(eq_suffix.len(), d0_lo.len());
    debug_assert_eq!(eq_suffix.len(), d0_hi.len());
    debug_assert_eq!(eq_suffix.len(), d1_lo.len());
    debug_assert_eq!(eq_suffix.len(), d1_hi.len());

    eq_suffix
        .par_iter()
        .zip(n0_lo.par_iter().zip(n0_hi.par_iter()))
        .zip(n1_lo.par_iter().zip(n1_hi.par_iter()))
        .zip(d0_lo.par_iter().zip(d0_hi.par_iter()))
        .zip(d1_lo.par_iter().zip(d1_hi.par_iter()))
        .par_fold_reduce(
            || [A::ZERO; 2],
            |mut acc, ((((&eq_suffix, n0), n1), d0), d1)| {
                let (&n0_lo, &n0_hi) = n0;
                let (&n1_lo, &n1_hi) = n1;
                let (&d0_lo, &d0_hi) = d0;
                let (&d1_lo, &d1_hi) = d1;
                let n0_dif = n0_hi - n0_lo;
                let n1_dif = n1_hi - n1_lo;
                let d0_dif = d0_hi - d0_lo;
                let d1_dif = d1_hi - d1_lo;
                let evaluate = |n0: N, n1: N, d0: A, d1: A| d1 * n0 + d0 * n1 + lambda * d0 * d1;

                acc[0] += eq_suffix * evaluate(n0_lo, n1_lo, d0_lo, d1_lo);

                let n0 = n0_hi + n0_dif;
                let n1 = n1_hi + n1_dif;
                let d0 = d0_hi + d0_dif;
                let d1 = d1_hi + d1_dif;

                acc[1] += eq_suffix * evaluate(n0, n1, d0, d1);

                acc
            },
            |mut lhs, rhs| {
                lhs.iter_mut().zip(rhs).for_each(|(lhs, rhs)| *lhs += rhs);
                lhs
            },
        )
}

fn reduce_fraction<N, A>(n0: &[N], n1: &[N], d0: &[A], d1: &[A]) -> SplitFraction<Poly<A>>
where
    N: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<N> + Copy + Send + Sync,
{
    debug_assert_eq!(n0.len(), n1.len());
    debug_assert_eq!(n0.len(), d0.len());
    debug_assert_eq!(n0.len(), d1.len());
    let (n00, n01) = n0.split_at(n0.len() / 2);
    let (n10, n11) = n1.split_at(n1.len() / 2);
    let (d00, d01) = d0.split_at(d0.len() / 2);
    let (d10, d11) = d1.split_at(d1.len() / 2);

    let reduce = |n0: &[N], n1: &[N], d0: &[A], d1: &[A]| {
        n0.par_iter()
            .zip(n1.par_iter())
            .zip(d0.par_iter())
            .zip(d1.par_iter())
            .map(|(((&n0, &n1), &d0), &d1)| (d1 * n0 + d0 * n1, d0 * d1))
            .unzip::<_, _, Vec<_>, Vec<_>>()
    };

    let (n0, d0) = reduce(n00, n10, d00, d10);
    let (n1, d1) = reduce(n01, n11, d01, d11);
    SplitFraction {
        n0: Poly::new(n0),
        d0: Poly::new(d0),
        n1: Poly::new(n1),
        d1: Poly::new(d1),
    }
}

impl<F: Field, EF: ExtensionField<F>> Fraction<Poly<F>, PolyMaybePacked<F, EF>> {
    fn reduce(&self) -> SplitFractionMaybePacked<F, EF> {
        match &self.d {
            PolyMaybePacked::Packed(denom) => {
                let numer = F::Packing::pack_slice(self.n.as_slice());
                let (n0, n1) = numer.split_at(numer.len() / 2);
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                SplitFractionMaybePacked::Packed(reduce_fraction(n0, n1, d0, d1))
            }
            PolyMaybePacked::Scalar(denom) => {
                let (n0, n1) = self.n.as_slice().split_at(self.n.num_evals() / 2);
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                SplitFractionMaybePacked::Scalar(reduce_fraction(n0, n1, d0, d1))
            }
        }
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> InputLayer<'a, F, EF> {
    fn new(fraction: &'a Fraction<Poly<F>, PolyMaybePacked<F, EF>>, point: &Point<EF>) -> Self {
        assert_eq!(fraction.d.num_variables(), point.num_variables() + 1);
        assert_eq!(fraction.n.num_variables(), point.num_variables() + 1);

        let eq_suffix = match &fraction.d {
            PolyMaybePacked::Packed(_) => PolyMaybePacked::Packed(Poly::new_packed_from_point(
                &point.as_slice()[1..],
                EF::ONE,
            )),
            PolyMaybePacked::Scalar(_) => {
                PolyMaybePacked::Scalar(Poly::new_from_point(&point.as_slice()[1..], EF::ONE))
            }
        };

        Self {
            fraction,
            eq_suffix,
        }
    }

    fn round_polys(&self, lambda: EF) -> [EF; 2] {
        let (n0, n1) = self
            .fraction
            .n
            .as_slice()
            .split_at(self.fraction.n.num_evals() / 2);

        match (&self.fraction.d, &self.eq_suffix) {
            (PolyMaybePacked::Packed(denom), PolyMaybePacked::Packed(eq_suffix)) => {
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                let (n0_lo, n0_hi) = n0.split_at(n0.len() / 2);
                let (n1_lo, n1_hi) = n1.split_at(n1.len() / 2);
                let (d0_lo, d0_hi) = d0.split_at(d0.len() / 2);
                let (d1_lo, d1_hi) = d1.split_at(d1.len() / 2);

                mixed_round_polys(
                    eq_suffix.as_slice(),
                    F::Packing::pack_slice(n0_lo),
                    F::Packing::pack_slice(n0_hi),
                    F::Packing::pack_slice(n1_lo),
                    F::Packing::pack_slice(n1_hi),
                    d0_lo,
                    d0_hi,
                    d1_lo,
                    d1_hi,
                    EF::ExtensionPacking::from(lambda),
                )
                .map(|value| EF::ExtensionPacking::to_ext_iter([value]).sum())
            }
            (PolyMaybePacked::Scalar(denom), PolyMaybePacked::Scalar(eq_suffix)) => {
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                let half = eq_suffix.num_evals();
                let (n0_lo, n0_hi) = n0.split_at(half);
                let (n1_lo, n1_hi) = n1.split_at(half);
                let (d0_lo, d0_hi) = d0.split_at(half);
                let (d1_lo, d1_hi) = d1.split_at(half);

                mixed_round_polys(
                    eq_suffix.as_slice(),
                    n0_lo,
                    n0_hi,
                    n1_lo,
                    n1_hi,
                    d0_lo,
                    d0_hi,
                    d1_lo,
                    d1_hi,
                    lambda,
                )
            }
            _ => unreachable!("input denominator and equality table use the same representation"),
        }
    }

    fn fold(self, coordinate: EF, r: EF) -> Layer<F, EF> {
        let Fraction { n, d } = self.fraction;
        let (n0, n1) = n.as_slice().split_at(n.num_evals() / 2);

        let (fraction, eq_suffix) = match (d, self.eq_suffix) {
            (PolyMaybePacked::Packed(denom), PolyMaybePacked::Packed(mut eq_suffix)) => {
                eq_suffix.sum_prefix_var_mut();
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                let r_packed = EF::ExtensionPacking::from(r);
                (
                    SplitFractionMaybePacked::Packed(SplitFraction {
                        n0: Poly::new(n0).fix_prefix_var_to_packed(r),
                        n1: Poly::new(n1).fix_prefix_var_to_packed(r),
                        d0: Poly::new(d0).fix_prefix_var(r_packed),
                        d1: Poly::new(d1).fix_prefix_var(r_packed),
                    }),
                    PolyMaybePacked::Packed(eq_suffix),
                )
            }
            (PolyMaybePacked::Scalar(denom), PolyMaybePacked::Scalar(mut eq_suffix)) => {
                eq_suffix.sum_prefix_var_mut();
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                (
                    SplitFractionMaybePacked::Scalar(SplitFraction {
                        n0: Poly::new(n0).fix_prefix_var(r),
                        n1: Poly::new(n1).fix_prefix_var(r),
                        d0: Poly::new(d0).fix_prefix_var(r),
                        d1: Poly::new(d1).fix_prefix_var(r),
                    }),
                    PolyMaybePacked::Scalar(eq_suffix),
                )
            }
            _ => unreachable!("input denominator and equality table use the same representation"),
        };

        let mut layer = Layer {
            fraction,
            eq_suffix,
            eq_prefix: (EF::ONE - coordinate) + r * (coordinate.double() - EF::ONE),
        };
        layer.transition();
        layer
    }
}

impl<F: Field, EF: ExtensionField<F>> Layer<F, EF> {
    fn new(fraction: SplitFractionMaybePacked<F, EF>, point: &Point<EF>) -> Self {
        let eq_suffix = match &fraction {
            SplitFractionMaybePacked::Packed(_) => PolyMaybePacked::Packed(
                Poly::new_packed_from_point(&point.as_slice()[1..], EF::ONE),
            ),
            SplitFractionMaybePacked::Scalar(_) => {
                PolyMaybePacked::Scalar(Poly::new_from_point(&point.as_slice()[1..], EF::ONE))
            }
        };
        let mut layer = Self {
            fraction,
            eq_suffix,
            eq_prefix: EF::ONE,
        };
        layer.transition();
        layer
    }

    fn num_variables(&self) -> usize {
        self.fraction.num_variables() - 1
    }

    /// Evaluate the equality-stripped quadratic round polynomial at `0` and `2`.
    fn round_polys(&self, lambda: EF) -> [EF; 2] {
        match (&self.fraction, &self.eq_suffix) {
            (SplitFractionMaybePacked::Packed(fraction), PolyMaybePacked::Packed(eq)) => fraction
                .round_polys(eq, EF::ExtensionPacking::from(lambda))
                .map(|value| EF::ExtensionPacking::to_ext_iter([value]).sum()),
            (SplitFractionMaybePacked::Scalar(fraction), PolyMaybePacked::Scalar(eq)) => {
                fraction.round_polys(eq, lambda)
            }
            _ => unreachable!("fraction and equality tables use the same representation"),
        }
    }

    fn fold(&mut self, coordinate: EF, r: EF) {
        match (&mut self.fraction, &mut self.eq_suffix) {
            (SplitFractionMaybePacked::Packed(fraction), PolyMaybePacked::Packed(eq_suffix)) => {
                let r = EF::ExtensionPacking::from(r);
                fraction.n0.fix_prefix_var_mut(r);
                fraction.n1.fix_prefix_var_mut(r);
                fraction.d0.fix_prefix_var_mut(r);
                fraction.d1.fix_prefix_var_mut(r);
                eq_suffix.sum_prefix_var_mut();
            }
            (SplitFractionMaybePacked::Scalar(fraction), PolyMaybePacked::Scalar(eq_suffix)) => {
                fraction.n0.fix_prefix_var_mut(r);
                fraction.n1.fix_prefix_var_mut(r);
                fraction.d0.fix_prefix_var_mut(r);
                fraction.d1.fix_prefix_var_mut(r);
                eq_suffix.sum_prefix_var_mut();
            }
            _ => unreachable!("fraction and equality tables use the same representation"),
        }
        self.eq_prefix *= (EF::ONE - coordinate) + r * (coordinate.double() - EF::ONE);
        self.transition();
    }

    fn into_claims(self) -> SplitFraction<EF> {
        assert_eq!(self.num_variables(), 0);

        match &self.fraction {
            SplitFractionMaybePacked::Scalar(fraction) => SplitFraction {
                n0: fraction.n0.as_constant().unwrap(),
                d0: fraction.d0.as_constant().unwrap(),
                n1: fraction.n1.as_constant().unwrap(),
                d1: fraction.d1.as_constant().unwrap(),
            },
            _ => unreachable!("fraction and equality tables use the same representation"),
        }
    }

    fn transition(&mut self) {
        if let (SplitFractionMaybePacked::Packed(fraction), PolyMaybePacked::Packed(eq_suffix)) =
            (&mut self.fraction, &mut self.eq_suffix)
            && eq_suffix.num_evals() == 1
        {
            self.fraction = SplitFractionMaybePacked::Scalar(SplitFraction {
                n0: fraction.n0.unpack::<F, EF>(),
                d0: fraction.d0.unpack::<F, EF>(),
                n1: fraction.n1.unpack::<F, EF>(),
                d1: fraction.d1.unpack::<F, EF>(),
            });
            self.eq_suffix = PolyMaybePacked::Scalar(eq_suffix.unpack::<F, EF>());
        }
    }
}

impl<A> SplitFraction<Poly<A>>
where
    A: PrimeCharacteristicRing + Copy + Send + Sync,
{
    fn num_variables(&self) -> usize {
        self.n0.num_variables() + 1
    }

    fn reduce(&self) -> Self {
        reduce_fraction(
            self.n0.as_slice(),
            self.n1.as_slice(),
            self.d0.as_slice(),
            self.d1.as_slice(),
        )
    }

    fn round_polys(&self, eq_suffix: &Poly<A>, lambda: A) -> [A; 2] {
        let half = eq_suffix.num_evals();
        let (n0_lo, n0_hi) = self.n0.as_slice().split_at(half);
        let (n1_lo, n1_hi) = self.n1.as_slice().split_at(half);
        let (d0_lo, d0_hi) = self.d0.as_slice().split_at(half);
        let (d1_lo, d1_hi) = self.d1.as_slice().split_at(half);

        mixed_round_polys(
            eq_suffix.as_slice(),
            n0_lo,
            n0_hi,
            n1_lo,
            n1_hi,
            d0_lo,
            d0_hi,
            d1_lo,
            d1_hi,
            lambda,
        )
    }

    fn into_constants(self) -> SplitFraction<A> {
        SplitFraction {
            n0: self
                .n0
                .as_constant()
                .expect("root numerator must be constant"),
            d0: self
                .d0
                .as_constant()
                .expect("root denominator must be constant"),
            n1: self
                .n1
                .as_constant()
                .expect("root numerator must be constant"),
            d1: self
                .d1
                .as_constant()
                .expect("root denominator must be constant"),
        }
    }
}

#[cfg(test)]
impl<A: Field> SplitFraction<Poly<A>> {
    pub(super) fn sum(&self) -> A {
        let mut denom = Vec::with_capacity(self.d0.num_evals() + self.d1.num_evals());
        denom.extend_from_slice(self.d0.as_slice());
        denom.extend_from_slice(self.d1.as_slice());
        let denom_inv = batch_multiplicative_inverse(&denom);

        dot_product(
            denom_inv.into_iter(),
            self.n0.as_slice().iter().chain(self.n1.as_slice()).copied(),
        )
    }

    pub(super) fn eval(&self, point: &Point<A>) -> (A, A) {
        assert_eq!(point.num_variables(), self.num_variables());
        let branch = point[0];
        let suffix = point.get_subpoint_over_range(1..);
        let n0 = self.n0.eval_ext::<A>(&suffix);
        let n1 = self.n1.eval_ext::<A>(&suffix);
        let d0 = self.d0.eval_ext::<A>(&suffix);
        let d1 = self.d1.eval_ext::<A>(&suffix);

        (n0 + branch * (n1 - n0), d0 + branch * (d1 - d0))
    }
}

impl<F: Field, EF: ExtensionField<F>> SplitFractionMaybePacked<F, EF> {
    fn num_variables(&self) -> usize {
        match self {
            Self::Packed(fraction) => {
                fraction.num_variables() + log2_strict_usize(F::Packing::WIDTH)
            }
            Self::Scalar(fraction) => fraction.num_variables(),
        }
    }

    fn reduce(&self) -> Self {
        let mut reduced = match self {
            Self::Packed(fraction) => Self::Packed(fraction.reduce()),
            Self::Scalar(fraction) => Self::Scalar(fraction.reduce()),
        };
        reduced.transition();
        reduced
    }

    /// Leave packed mode after its final useful SIMD reduction.
    fn transition(&mut self) {
        match self {
            Self::Packed(fraction) if fraction.n0.num_evals() == 1 => {
                *self = Self::Scalar(SplitFraction {
                    n0: fraction.n0.unpack::<F, EF>(),
                    d0: fraction.d0.unpack::<F, EF>(),
                    n1: fraction.n1.unpack::<F, EF>(),
                    d1: fraction.d1.unpack::<F, EF>(),
                });
            }
            _ => {}
        }
    }

    fn unpack(self) -> SplitFraction<Poly<EF>> {
        match self {
            Self::Packed(fraction) => SplitFraction {
                n0: fraction.n0.unpack::<F, EF>(),
                n1: fraction.n1.unpack::<F, EF>(),
                d0: fraction.d0.unpack::<F, EF>(),
                d1: fraction.d1.unpack::<F, EF>(),
            },
            Self::Scalar(fraction) => fraction,
        }
    }
}

/// Restore the current equality factor to the internally quadratic round polynomial.
///
/// The returned message contains the cubic evaluations expected by the existing verifier.
/// `q_sum` is the normalized quadratic sum `q(0) + q(1)` used only by the prover.
fn restore_equality_factor<EF: Field>(
    quadratic_evals: [EF; 2],
    normalized_sum: EF,
    coordinate: EF,
    equality_prefix: EF,
    interpolator: &RoundPolyInterpolator<EF>,
) -> ([EF; 3], EF) {
    let q1 = (normalized_sum - (EF::ONE - coordinate) * quadratic_evals[0]) * coordinate.inverse();
    let q_sum = quadratic_evals[0] + q1;
    let q3 = interpolator.eval(&quadratic_evals, q_sum, EF::from_u8(3));
    let equality_0 = EF::ONE - coordinate;
    let equality_step = coordinate.double() - EF::ONE;
    let equality_2 = equality_0 + EF::TWO * equality_step;
    let equality_3 = equality_2 + equality_step;
    let round_poly = [
        equality_prefix * equality_0 * quadratic_evals[0],
        equality_prefix * equality_2 * quadratic_evals[1],
        equality_prefix * equality_3 * q3,
    ];
    (round_poly, q_sum)
}

/// Proves that a table of fractions sums to zero via a binary-tree reduction.
///
/// It observes the root denominator and layer messages in `challenger` and
/// returns the numerator and denominator openings of the input tables at the
/// resulting transcript-derived point.
///
/// # Panics
///
/// Panics if the numerator and denominator have different variable counts, if
/// they have no variables, or if the fully reduced root denominator is zero.
pub fn prove_fractional_gkr<F, EF, Challenger>(
    fraction: &Fraction<Poly<F>, PolyMaybePacked<F, EF>>,
    challenger: &mut Challenger,
) -> (FractionGkrProof<EF>, FractionGkrOutput<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let num_variables = fraction.n.num_variables();
    assert_eq!(num_variables, fraction.d.num_variables());
    assert!(
        num_variables >= 1,
        "fraction GKR requires at least one input variable"
    );

    // With one input variable, the original two leaves are already the root's children. The
    // root layer is therefore also the input layer: it has no sumcheck rounds and directly opens
    // the input at the sampled branch coordinate.
    if num_variables == 1 {
        let denom = fraction.d.clone().unpack();
        let root_claims = SplitFraction {
            n0: EF::from(fraction.n.as_slice()[0]),
            d0: denom.as_slice()[0],
            n1: EF::from(fraction.n.as_slice()[1]),
            d1: denom.as_slice()[1],
        };
        let root_denominator = root_claims.d0 * root_claims.d1;
        assert_eq!(
            root_claims.d1 * root_claims.n0 + root_claims.d0 * root_claims.n1,
            EF::ZERO,
            "fraction GKR input fractions must sum to zero"
        );
        assert_ne!(
            root_denominator,
            EF::ZERO,
            "fraction GKR root denominator must be nonzero"
        );
        challenger.observe_algebra_element(root_denominator);

        let lambda: EF = challenger.sample_algebra_element();
        debug_assert_eq!(lambda * root_denominator, root_claims.gate(lambda));
        challenger.observe_algebra_slice(&[
            root_claims.n0,
            root_claims.d0,
            root_claims.n1,
            root_claims.d1,
        ]);
        let branch: EF = challenger.sample_algebra_element();
        let numerator = root_claims.n0 + branch * (root_claims.n1 - root_claims.n0);
        let denominator = root_claims.d0 + branch * (root_claims.d1 - root_claims.d0);

        return (
            FractionGkrProof {
                root_denominator,
                layers: vec![FractionGkrLayerProof {
                    round_polys: Vec::new(),
                    claims: root_claims,
                }],
            },
            FractionGkrOutput {
                point: Point::new(vec![branch]),
                numerator,
                denominator,
            },
        );
    }

    // The packed input path needs three backing variables: two for its initial
    // branch splits and one for dropping the first equality-suffix variable
    // before transitioning to scalar storage. With fewer, those logical
    // variables live entirely inside SIMD lanes, where the backing-table folds
    // cannot address them. At this tiny boundary, unpack once and reuse the
    // scalar prover so its proof and transcript stay identical to scalar storage.
    let scalar_fraction = if let PolyMaybePacked::Packed(denom) = &fraction.d
        && denom.num_variables() < 3
    {
        Some(Fraction {
            n: fraction.n.clone(),
            d: PolyMaybePacked::Scalar(denom.unpack::<F, EF>()),
        })
    } else {
        None
    };
    let fraction = scalar_fraction.as_ref().unwrap_or(fraction);

    // Build every reduced tree layer while retaining the original input for the
    // final mixed base/extension-field layer.
    let mut fractions = vec![fraction.reduce()];
    while fractions.last().unwrap().num_variables() > 1 {
        fractions.push(fractions.last().unwrap().reduce());
    }

    let root_claims = fractions.pop().unwrap().unpack().into_constants();
    let root_denominator = root_claims.d0 * root_claims.d1;
    assert_eq!(
        root_claims.d1 * root_claims.n0 + root_claims.d0 * root_claims.n1,
        EF::ZERO,
        "fraction GKR input fractions must sum to zero"
    );
    assert_ne!(
        root_denominator,
        EF::ZERO,
        "fraction GKR root denominator must be nonzero"
    );
    challenger.observe_algebra_element(root_denominator);
    let interpolator = RoundPolyInterpolator::new(2);
    let mut layer_proofs = Vec::with_capacity(num_variables);

    // Prove the final reduction from the two one-variable children to the scalar root. This
    // layer has no sumcheck variables, but its random linear combination binds the root
    // denominator to the zero-numerator statement.
    let lambda: EF = challenger.sample_algebra_element();
    debug_assert_eq!(lambda * root_denominator, root_claims.gate(lambda));
    challenger.observe_algebra_slice(&[
        root_claims.n0,
        root_claims.d0,
        root_claims.n1,
        root_claims.d1,
    ]);
    let branch: EF = challenger.sample_algebra_element();
    let mut numerator = root_claims.n0 + branch * (root_claims.n1 - root_claims.n0);
    let mut denominator = root_claims.d0 + branch * (root_claims.d1 - root_claims.d0);
    let mut point = Point::new(vec![branch]);
    layer_proofs.push(FractionGkrLayerProof {
        round_polys: Vec::new(),
        claims: root_claims,
    });

    // Reduced extension-field layers are consumed from the top of the tree down.
    for fraction in fractions.into_iter().rev() {
        let lambda: EF = challenger.sample_algebra_element();
        let mut normalized_sum = numerator + lambda * denominator;
        let mut layer = Layer::new(fraction, &point);
        let mut round_polys = Vec::with_capacity(point.num_variables());
        let mut round_point = Vec::with_capacity(point.num_variables());

        debug_assert_eq!(layer.num_variables(), point.num_variables());
        for round in 0..point.num_variables() {
            let coordinate = point[round];
            let quadratic_evals = layer.round_polys(lambda);
            let (round_poly, q_sum) = restore_equality_factor(
                quadratic_evals,
                normalized_sum,
                coordinate,
                layer.eq_prefix,
                &interpolator,
            );
            challenger.observe_algebra_slice(&round_poly);
            let challenge: EF = challenger.sample_algebra_element();
            round_polys.push(round_poly);
            round_point.push(challenge);
            normalized_sum = interpolator.eval(&quadratic_evals, q_sum, challenge);
            layer.fold(coordinate, challenge);
        }

        let claims = layer.into_claims();
        debug_assert_eq!(normalized_sum, claims.gate(lambda));
        challenger.observe_algebra_slice(&[claims.n0, claims.d0, claims.n1, claims.d1]);
        let branch: EF = challenger.sample_algebra_element();
        numerator = claims.n0 + branch * (claims.n1 - claims.n0);
        denominator = claims.d0 + branch * (claims.d1 - claims.d0);
        round_point.insert(0, branch);
        point = Point::new(round_point);
        layer_proofs.push(FractionGkrLayerProof {
            round_polys,
            claims,
        });
    }

    // The original lookup arrays are the only mixed base/extension-field layer.
    // Its first fold promotes the numerators; all later rounds use Layer.
    let lambda: EF = challenger.sample_algebra_element();
    let mut normalized_sum = numerator + lambda * denominator;
    let input_layer = InputLayer::new(fraction, &point);
    let mut round_polys = Vec::with_capacity(point.num_variables());
    let mut round_point = Vec::with_capacity(point.num_variables());

    let coordinate = point[0];
    let quadratic_evals = input_layer.round_polys(lambda);
    let (round_poly, q_sum) = restore_equality_factor(
        quadratic_evals,
        normalized_sum,
        coordinate,
        EF::ONE,
        &interpolator,
    );
    challenger.observe_algebra_slice(&round_poly);
    let challenge: EF = challenger.sample_algebra_element();
    round_polys.push(round_poly);
    round_point.push(challenge);
    normalized_sum = interpolator.eval(&quadratic_evals, q_sum, challenge);
    let mut layer = input_layer.fold(coordinate, challenge);

    debug_assert_eq!(layer.num_variables(), point.num_variables() - 1);
    for round in 1..point.num_variables() {
        let coordinate = point[round];
        let quadratic_evals = layer.round_polys(lambda);
        let (round_poly, q_sum) = restore_equality_factor(
            quadratic_evals,
            normalized_sum,
            coordinate,
            layer.eq_prefix,
            &interpolator,
        );
        challenger.observe_algebra_slice(&round_poly);
        let challenge: EF = challenger.sample_algebra_element();
        round_polys.push(round_poly);
        round_point.push(challenge);
        normalized_sum = interpolator.eval(&quadratic_evals, q_sum, challenge);
        layer.fold(coordinate, challenge);
    }

    let claims = layer.into_claims();
    debug_assert_eq!(normalized_sum, claims.gate(lambda));
    challenger.observe_algebra_slice(&[claims.n0, claims.d0, claims.n1, claims.d1]);
    let branch: EF = challenger.sample_algebra_element();
    let numerator = claims.n0 + branch * (claims.n1 - claims.n0);
    let denominator = claims.d0 + branch * (claims.d1 - claims.d0);
    round_point.insert(0, branch);
    let point = Point::new(round_point);
    layer_proofs.push(FractionGkrLayerProof {
        round_polys,
        claims,
    });

    (
        FractionGkrProof {
            root_denominator,
            layers: layer_proofs,
        },
        FractionGkrOutput {
            point,
            numerator,
            denominator,
        },
    )
}
