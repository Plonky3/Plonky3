//! Witness cells the verifier knows, pinned by the committed proof.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::operator::encode_usize;

/// Public witness cells evenly spaced by a power of two, such as the constant one of an R1CS.
///
/// The cells span a sub-cube: their index bits are free on one window and fixed elsewhere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicRegion<F> {
    /// Index of the first cell.
    offset: usize,
    /// Spacing between consecutive cells, as a power of two.
    log_stride: usize,
    /// Cell values, a power of two of them.
    values: Vec<F>,
}

impl<F> PublicRegion<F> {
    /// The cells `offset + j * 2^log_stride`, one per value `j`.
    ///
    /// A stride of one with an aligned offset is a contiguous block.
    ///
    /// # Panics
    ///
    /// - The number of values is not a power of two, or the offset has a bit the cells index.
    #[must_use]
    pub fn new(offset: usize, log_stride: usize, values: Vec<F>) -> Self {
        assert!(
            values.len().is_power_of_two(),
            "a public region holds a power of two of cells"
        );
        let log_len = values.len().trailing_zeros() as usize;
        assert!(
            log_stride
                .checked_add(log_len)
                .is_some_and(|span| span < usize::BITS as usize),
            "the cell indices overflow usize"
        );
        let window = (values.len() - 1) << log_stride;
        assert_eq!(offset & window, 0, "the offset overlaps the cell indices");
        Self {
            offset,
            log_stride,
            values,
        }
    }

    /// Index of the first cell.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Spacing between consecutive cells, as a power of two.
    #[must_use]
    pub const fn log_stride(&self) -> usize {
        self.log_stride
    }

    /// Cell values.
    #[must_use]
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Number of variables indexing a cell of the region.
    #[must_use]
    pub const fn log_len(&self) -> usize {
        self.values.len().trailing_zeros() as usize
    }

    /// Whether a witness holds these values in these cells.
    #[must_use]
    pub fn matches(&self, z: &[F]) -> bool
    where
        F: PartialEq,
    {
        self.values
            .iter()
            .enumerate()
            .all(|(j, value)| z.get(self.offset | (j << self.log_stride)) == Some(value))
    }

    /// Whether every cell lies inside a witness of `2^log_cols` cells.
    pub(crate) const fn fits(&self, log_cols: usize) -> bool {
        // The window and the offset both fit the witness variables.
        self.log_stride + self.log_len() <= log_cols
            && (log_cols >= usize::BITS as usize || self.offset >> log_cols == 0)
    }

    /// Witness point of the cells at a point over the region's own variables.
    ///
    /// The offset's bits above the window lead, then the point, then its bits below.
    pub(crate) fn witness_point<EF: Field>(&self, log_cols: usize, inner: &[EF]) -> Point<EF> {
        let span = self.log_stride + self.log_len();
        let low = self.offset & ((1 << self.log_stride) - 1);
        let mut point = Point::hypercube(self.offset >> span, log_cols - span);
        point.extend(&Point::new(inner));
        point.extend(&Point::hypercube(low, self.log_stride));
        point
    }

    /// Multilinear extension of the values at a point over the region's variables.
    pub(crate) fn evaluate<EF: ExtensionField<F>>(&self, inner: &[EF]) -> EF
    where
        F: Field,
    {
        let eq = Poly::new_from_point(inner, EF::ONE);
        eq.iter().zip(&self.values).map(|(&e, &v)| e * v).sum()
    }

    /// Append the offset, the stride, the count and every value.
    pub(crate) fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        encode_usize(out, self.offset);
        encode_usize(out, self.log_stride);
        encode_usize(out, self.log_len());
        for value in &self.values {
            F::encode(value, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Base field.
    type F = BabyBear;
    /// Challenge field.
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn the_witness_extension_at_the_region_point_is_the_region_extension() {
        // Invariant: z~(high bits, r, low bits) is the values' extension at r when z holds them.
        //
        // Fixture state: 2^7 cells; contiguous, strided and single-cell regions.
        let mut rng = SmallRng::seed_from_u64(3);
        let mut z: Vec<F> = (0..128).map(|_| rng.random()).collect();
        for (offset, log_stride, log_len) in [(32, 0, 3), (5, 3, 2), (77, 0, 0), (3, 4, 3)] {
            let values: Vec<F> = (0..1 << log_len).map(|_| rng.random()).collect();
            let region = PublicRegion::new(offset, log_stride, values);
            for (j, &value) in region.values().iter().enumerate() {
                z[offset | (j << log_stride)] = value;
            }
            assert!(region.matches(&z));
            let span = region.log_stride() + region.log_len();
            assert!(region.fits(7) && span.checked_sub(1).is_none_or(|n| !region.fits(n)));

            let r: Vec<EF> = (0..log_len).map(|_| rng.random()).collect();
            let at = Poly::new(z.clone()).eval_base(&region.witness_point(7, &r));
            assert_eq!(at, region.evaluate(&r), "offset {offset}");
        }
    }

    #[test]
    #[should_panic(expected = "overlaps")]
    fn an_offset_inside_the_window_is_rejected() {
        // Mutation: offset 4 sets a bit that four cells two apart already index.
        let _ = PublicRegion::new(4, 1, vec![F::ONE; 4]);
    }
}
