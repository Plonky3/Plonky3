//! Multilinear polynomial stored either as scalars or in SIMD-packed form.
//!
//! Both variants describe the same function on the Boolean hypercube.
//! They differ only in where the trailing variables live.
//!
//! `W = F::Packing::WIDTH` is the number of base-field elements per SIMD vector.
//! The packed variant moves the last `log2(W)` variables into the lanes:
//!
//! ```text
//! n = 5, W = 4:
//!
//!   Scalar  [ e0 ][ e1 ][ e2 ] ... [ e31 ]      32 stored, 5 stored variables
//!
//!   Packed  [ e0 e1 e2 e3 ][ e4 e5 e6 e7 ] ...   8 stored, 3 stored variables
//!             ^^^^^^^^^^^                         x_3, x_4 index the lanes
//! ```
//!
//! A packed [`Poly`] therefore reports only `n - log2(W)` variables.
//! Accessors here report the logical size `n` instead.

use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_util::log2_strict_usize;

use crate::point::Point;
use crate::poly::Poly;

/// Borrowed view of an extension-field polynomial in its current scalar or packed representation.
pub type PolyMaybePackedView<'a, F, EF> =
    PolyMaybePacked<F, EF, &'a [EF], &'a [<EF as ExtensionField<F>>::ExtensionPacking]>;

/// An extension-field polynomial stored either as scalar evaluations or in SIMD-packed form.
///
/// See the module docs for the lane layout the two variants share.
///
/// This type has no equality impl.
/// A derived one would compare the storage tag rather than the polynomial.
///
/// Unpack both sides with `unpack_into` to compare by value.
#[derive(Debug, Copy, Clone)]
#[must_use]
pub enum PolyMaybePacked<
    F,
    EF,
    ScalarStorage = Vec<EF>,
    PackedStorage = Vec<<EF as ExtensionField<F>>::ExtensionPacking>,
> where
    F: Field,
    EF: ExtensionField<F>,
{
    /// One extension-field element per Boolean-hypercube evaluation.
    Scalar(Poly<EF, ScalarStorage>),
    /// `F::Packing::WIDTH` consecutive evaluations per packed extension-field element.
    Packed(Poly<EF::ExtensionPacking, PackedStorage>),
}

impl<F, EF, ScalarStorage, PackedStorage> PolyMaybePacked<F, EF, ScalarStorage, PackedStorage>
where
    F: Field,
    EF: ExtensionField<F>,
    ScalarStorage: Borrow<[EF]>,
    PackedStorage: Borrow<[EF::ExtensionPacking]>,
{
    /// Number of variables in the represented multilinear polynomial.
    ///
    /// The packed variant adds back the `log2(W)` variables held in the SIMD lanes.
    #[inline]
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Scalar(poly) => poly.num_variables(),
            Self::Packed(poly) => poly.num_variables() + log2_strict_usize(F::Packing::WIDTH),
        }
    }

    /// Number of scalar evaluations the polynomial represents, `2^num_variables`.
    ///
    /// This counts logical evaluations rather than stored elements.
    /// [`Poly::num_evals`] reports the stored count, which is `W` times smaller when packed.
    #[inline]
    pub fn num_scalar_evals(&self) -> usize {
        1 << self.num_variables()
    }

    /// Borrow the represented polynomial without changing its storage format.
    #[inline]
    pub fn as_view(&self) -> PolyMaybePackedView<'_, F, EF> {
        match self {
            Self::Scalar(poly) => PolyMaybePacked::Scalar(poly.as_view()),
            Self::Packed(poly) => PolyMaybePacked::Packed(poly.as_view()),
        }
    }

    /// Writes all logical evaluations into `out` in index order.
    ///
    /// # Panics
    ///
    /// The output length must equal [`Self::num_scalar_evals`].
    pub fn unpack_into(&self, out: &mut [EF]) {
        assert_eq!(out.len(), self.num_scalar_evals());
        match self {
            Self::Packed(poly) => out
                .iter_mut()
                .zip(EF::ExtensionPacking::to_ext_iter(poly.iter().copied()))
                .for_each(|(out, value)| *out = value),
            Self::Scalar(poly) => out.copy_from_slice(poly.as_slice()),
        }
    }

    /// Evaluate the represented polynomial without changing its storage format.
    ///
    /// # Panics
    ///
    /// The point must have [`Self::num_variables`] coordinates.
    #[inline]
    pub fn eval(&self, point: &Point<EF>) -> EF {
        assert_eq!(self.num_variables(), point.num_variables());
        match self {
            Self::Scalar(poly) => poly.eval_ext::<F>(point),
            Self::Packed(poly) => poly.eval_packed::<F, EF>(point),
        }
    }
}

impl<F, EF> PolyMaybePacked<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Convert owned storage to scalar form, unpacking SIMD lanes only when necessary.
    #[inline]
    pub fn unpack(self) -> Poly<EF> {
        match self {
            Self::Scalar(poly) => poly,
            Self::Packed(poly) => poly.unpack::<F, EF>(),
        }
    }
}

impl<F, EF> From<Poly<EF>> for PolyMaybePacked<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    #[inline]
    fn from(poly: Poly<EF>) -> Self {
        Self::Scalar(poly)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{PolyMaybePacked, PolyMaybePackedView};
    use crate::point::Point;
    use crate::poly::Poly;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Number of variables the packed variant absorbs into the SIMD lanes.
    const K_PACK: usize = log2_strict_usize(<<F as Field>::Packing as PackedValue>::WIDTH);

    #[test]
    fn scalar_view_borrows_without_copying() {
        let poly = Poly::new(vec![EF::ZERO, EF::ONE]);
        let maybe = PolyMaybePacked::<F, EF>::Scalar(poly);
        let view: PolyMaybePackedView<'_, F, EF> = maybe.as_view();

        let PolyMaybePacked::Scalar(view_poly) = view else {
            panic!("scalar storage must produce a scalar view");
        };
        let PolyMaybePacked::Scalar(owned_poly) = &maybe else {
            unreachable!();
        };
        assert_eq!(
            view_poly.as_slice().as_ptr(),
            owned_poly.as_slice().as_ptr()
        );
    }

    #[test]
    #[should_panic]
    fn unpack_into_rejects_short_output() {
        let maybe = PolyMaybePacked::<F, EF>::from(Poly::new(vec![EF::ZERO, EF::ONE]));
        // The buffer must hold every logical evaluation, not one fewer.
        maybe.unpack_into(&mut [EF::ZERO]);
    }

    #[test]
    #[should_panic]
    fn eval_rejects_point_of_wrong_arity() {
        let maybe = PolyMaybePacked::<F, EF>::from(Poly::new(vec![EF::ZERO, EF::ONE]));
        // One variable is stored, so a two-coordinate point cannot apply.
        let _ = maybe.eval(&Point::new(vec![EF::ONE, EF::ONE]));
    }

    proptest! {
        #[test]
        fn prop_scalar_and_packed_agree(extra in 0usize..=4, seed in any::<u64>()) {
            // Stay at or above the packing threshold so `pack` applies.
            // `extra = 0` pins the single-packed-element boundary.
            let n = K_PACK + extra;
            let mut rng = SmallRng::seed_from_u64(seed);

            let scalar = Poly::<EF>::rand(&mut rng, n);
            let as_scalar = PolyMaybePacked::<F, EF>::from(scalar.clone());
            let as_packed = PolyMaybePacked::<F, EF>::Packed(scalar.pack::<F, EF>());

            // Shape is reported logically, so the storage tag must not show through.
            prop_assert_eq!(as_scalar.num_variables(), n);
            prop_assert_eq!(as_packed.num_variables(), n);
            prop_assert_eq!(as_scalar.num_scalar_evals(), 1 << n);
            prop_assert_eq!(as_packed.num_scalar_evals(), 1 << n);

            // Unpacking must recover index order within lanes and across elements.
            let mut from_scalar = EF::zero_vec(1 << n);
            let mut from_packed = EF::zero_vec(1 << n);
            as_scalar.unpack_into(&mut from_scalar);
            as_packed.unpack_into(&mut from_packed);
            prop_assert_eq!(from_scalar.as_slice(), scalar.as_slice());
            prop_assert_eq!(from_packed.as_slice(), scalar.as_slice());

            // Evaluation agrees with the scalar oracle in both representations.
            // At `extra = 0` the packed path falls back to unpacking internally.
            let point = Point::<EF>::rand(&mut rng, n);
            let expected = scalar.eval_ext::<F>(&point);
            prop_assert_eq!(as_scalar.eval(&point), expected);
            prop_assert_eq!(as_packed.eval(&point), expected);

            // Owned conversion to scalar form round-trips.
            let unpacked = as_packed.unpack();
            prop_assert_eq!(unpacked.as_slice(), scalar.as_slice());
        }
    }
}
