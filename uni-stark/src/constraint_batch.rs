use p3_field::{Algebra, Field, PackedField, PrimeCharacteristicRing};

/// Batch size for constraint linear-combination chunks.
///
/// Chosen to match the typical SIMD accumulation width used in packed field
/// implementations. Operations are chunked in groups of this size so that
/// `sum_array` / `packed_linear_combination` can exploit instruction-level
/// parallelism.
pub const CONSTRAINT_BATCH: usize = 8;

/// Batched linear combination of packed extension field values with EF coefficients.
///
/// Extension-field analogue of [`PackedField::packed_linear_combination`]. Processes
/// `coeffs` and `values` in chunks of [`CONSTRAINT_BATCH`], then handles the remainder.
#[inline]
pub fn batched_ext_linear_combination<EF, PE>(coeffs: &[EF], values: &[PE]) -> PE
where
    EF: Field,
    PE: PrimeCharacteristicRing + Algebra<EF> + Copy,
{
    debug_assert_eq!(coeffs.len(), values.len());
    let len = coeffs.len();
    let mut acc = PE::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        let batch: [PE; CONSTRAINT_BATCH] =
            core::array::from_fn(|i| values[start + i] * coeffs[start + i]);
        acc += PE::sum_array::<CONSTRAINT_BATCH>(&batch);
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Batched linear combination of packed base field values with F coefficients.
///
/// Wraps [`PackedField::packed_linear_combination`] with batched chunking
/// and remainder handling, mirroring [`batched_ext_linear_combination`].
#[inline]
pub fn batched_base_linear_combination<P: PackedField>(coeffs: &[P::Scalar], values: &[P]) -> P {
    debug_assert_eq!(coeffs.len(), values.len());
    let len = coeffs.len();
    let mut acc = P::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        acc += P::packed_linear_combination::<CONSTRAINT_BATCH>(
            &coeffs[start..start + CONSTRAINT_BATCH],
            &values[start..start + CONSTRAINT_BATCH],
        );
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    fn naive_base_lc(coeffs: &[F], values: &[F]) -> F {
        coeffs.iter().zip(values).map(|(&c, &v)| v * c).sum()
    }

    fn naive_ext_lc(coeffs: &[EF], values: &[EF]) -> EF {
        coeffs.iter().zip(values).map(|(&c, &v)| v * c).sum()
    }

    #[test]
    fn batched_base_matches_naive_exact_multiple() {
        let n = CONSTRAINT_BATCH * 3;
        let coeffs: Vec<F> = (1..=n).map(|i| F::from_u32(i as u32)).collect();
        let values: Vec<F> = (2..=n + 1).map(|i| F::from_u32(i as u32)).collect();

        assert_eq!(
            batched_base_linear_combination(&coeffs, &values),
            naive_base_lc(&coeffs, &values),
        );
    }

    #[test]
    fn batched_base_matches_naive_with_remainder() {
        let n = CONSTRAINT_BATCH * 2 + 3;
        let coeffs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 + 1)).collect();
        let values: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 * 2 + 1)).collect();

        assert_eq!(
            batched_base_linear_combination(&coeffs, &values),
            naive_base_lc(&coeffs, &values),
        );
    }

    #[test]
    fn batched_base_matches_naive_shorter_than_batch() {
        let n = CONSTRAINT_BATCH - 1;
        let coeffs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 + 5)).collect();
        let values: Vec<F> = (0..n).map(|i| F::from_u32(i as u32 + 7)).collect();

        assert_eq!(
            batched_base_linear_combination(&coeffs, &values),
            naive_base_lc(&coeffs, &values),
        );
    }

    #[test]
    fn batched_base_empty() {
        assert_eq!(batched_base_linear_combination::<F>(&[], &[]), F::ZERO);
    }

    #[test]
    fn batched_ext_matches_naive_exact_multiple() {
        let n = CONSTRAINT_BATCH * 3;
        let coeffs: Vec<EF> = (1..=n).map(|i| EF::from(F::from_u32(i as u32))).collect();
        let values: Vec<EF> = (2..=n + 1)
            .map(|i| EF::from(F::from_u32(i as u32)))
            .collect();

        assert_eq!(
            batched_ext_linear_combination(&coeffs, &values),
            naive_ext_lc(&coeffs, &values),
        );
    }

    #[test]
    fn batched_ext_matches_naive_with_remainder() {
        let n = CONSTRAINT_BATCH + 5;
        let coeffs: Vec<EF> = (0..n)
            .map(|i| EF::from(F::from_u32(i as u32 + 3)))
            .collect();
        let values: Vec<EF> = (0..n)
            .map(|i| EF::from(F::from_u32(i as u32 * 3 + 1)))
            .collect();

        assert_eq!(
            batched_ext_linear_combination(&coeffs, &values),
            naive_ext_lc(&coeffs, &values),
        );
    }

    #[test]
    fn batched_ext_empty() {
        assert_eq!(batched_ext_linear_combination::<EF, EF>(&[], &[]), EF::ZERO);
    }
}
