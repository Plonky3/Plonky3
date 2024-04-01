use p3_field::{Field, PackedField, PackedValue};

pub(crate) trait Butterfly<F: Field>: Send + Sync {
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: &mut PF, x_2: &mut PF);

    #[inline]
    fn apply_to_rows(&self, row_1: &mut [F], row_2: &mut [F]) {
        let (shorts_1, suffix_1) = F::Packing::pack_slice_with_suffix_mut(row_1);
        let (shorts_2, suffix_2) = F::Packing::pack_slice_with_suffix_mut(row_2);
        for (x_1, x_2) in shorts_1.iter_mut().zip(shorts_2) {
            self.apply(x_1, x_2);
        }
        for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2) {
            self.apply(x_1, x_2);
        }
    }
}

pub(crate) struct DifButterfly<F>(pub F);
impl<F: Field> Butterfly<F> for DifButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: &mut PF, x_2: &mut PF) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff * self.0;
    }
}

pub(crate) struct DitButterfly<F>(pub F);
impl<F: Field> Butterfly<F> for DitButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: &mut PF, x_2: &mut PF) {
        let x_2_twiddle = *x_2 * self.0;
        let sum = *x_1 + x_2_twiddle;
        let diff = *x_1 - x_2_twiddle;
        *x_1 = sum;
        *x_2 = diff;
    }
}

/// Butterfly with twiddle factor 1 (works in either DIT or DIF).
pub(crate) struct TwiddleFreeButterfly;
impl<F: Field> Butterfly<F> for TwiddleFreeButterfly {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: &mut PF, x_2: &mut PF) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff;
    }
}
