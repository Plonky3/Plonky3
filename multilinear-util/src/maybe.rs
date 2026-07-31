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
/// The two variants represent the same logical object. In the packed variant, the last
/// `log2(F::Packing::WIDTH)` Boolean variables live inside the SIMD lanes, so the backing
/// [`Poly`] has that many fewer stored variables. The accessors on this enum always report the
/// logical size rather than the backing-vector size.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
    #[inline]
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Scalar(poly) => poly.num_variables(),
            Self::Packed(poly) => poly.num_variables() + log2_strict_usize(F::Packing::WIDTH),
        }
    }

    /// Number of logical scalar evaluations represented by the backing storage.
    #[inline]
    pub fn num_evals(&self) -> usize {
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
    /// The output length must equal [`Self::num_evals`].
    pub fn unpack_into(&self, out: &mut [EF]) {
        assert_eq!(out.len(), self.num_evals());
        match self {
            Self::Packed(poly) => out
                .iter_mut()
                .zip(EF::ExtensionPacking::to_ext_iter(poly.iter().copied()))
                .for_each(|(out, value)| *out = value),
            Self::Scalar(poly) => out.copy_from_slice(poly.as_slice()),
        }
    }

    /// Evaluate the represented polynomial without changing its storage format.
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

    use super::{PolyMaybePacked, PolyMaybePackedView};
    use crate::poly::Poly;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

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
    fn packed_view_preserves_logical_shape_and_unpacks() {
        let width = <<F as Field>::Packing as PackedValue>::WIDTH;
        let scalar = Poly::new((0..width).map(|value| EF::from_u64(value as u64)).collect());
        let packed = scalar.pack::<F, EF>();
        let maybe = PolyMaybePacked::<F, EF>::Packed(packed);
        let view = maybe.as_view();

        assert_eq!(view.num_variables(), maybe.num_variables());
        assert_eq!(view.num_evals(), width);

        let mut unpacked = EF::zero_vec(width);
        view.unpack_into(&mut unpacked);
        assert_eq!(unpacked, scalar.into_evals());
    }
}
