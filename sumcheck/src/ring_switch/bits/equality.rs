//! The equality table of one run, held as the two factors it is a product of.

use p3_binary_field::BitCoordinates;
use p3_field::Field;
use p3_multilinear_util::poly::Poly;

use super::basis::{Coefficients, CoordinateSums};

/// `eq(point, .)` over one run, held as an outer factor times an inner factor.
///
/// # Overview
///
/// The equality polynomial factors over its variables, and the table indexes them with the
/// first coordinate most significant. Splitting the point after its leading coordinates
/// therefore splits the table into equal blocks:
///
/// ```text
///     point                 =  (outer coordinates, inner coordinates)
///     table[i * block + j]  =  outer[i] * inner[j]
/// ```
///
/// The dense table is `2^k` elements; the two factors together are `2^(k - m) + 2^m`.
///
/// # Why the factors and not the table
///
/// Every consumer here sweeps the table once, one block per task, so an entry is wanted
/// exactly once. Forming it from the factors costs one multiplication and no memory, where
/// the dense table costs a `2^k` allocation written once and read once.
///
/// The block is also the unit of parallelism, so a task reads one outer entry and the whole
/// inner factor. Sizing the inner factor to fit in cache keeps that read local.
#[derive(Clone, Debug)]
pub(crate) struct FactoredEquality<F> {
    /// `eq(outer coordinates, .)`, one entry per block.
    outer: Poly<F>,
    /// `eq(inner coordinates, .)`, one entry per element of a block.
    inner: Poly<F>,
}

impl<F: Field> FactoredEquality<F> {
    /// The equality table of `point`, in blocks of at most `1 << log_block` elements.
    ///
    /// The inner factor takes the trailing coordinates, which are the ones a block varies.
    #[must_use]
    pub(crate) fn new(point: &[F], log_block: usize) -> Self {
        let inner_variables = log_block.min(point.len());
        let (outer, inner) = point.split_at(point.len() - inner_variables);
        Self {
            outer: Poly::new_from_point(outer, F::ONE),
            inner: Poly::new_from_point(inner, F::ONE),
        }
    }

    /// Number of variables the table indexes.
    #[must_use]
    pub(crate) fn num_variables(&self) -> usize {
        self.outer.num_variables() + self.inner.num_variables()
    }

    /// Number of entries the table holds.
    #[must_use]
    pub(crate) fn num_evals(&self) -> usize {
        self.outer.num_evals() * self.inner.num_evals()
    }

    /// Number of entries one block holds.
    #[must_use]
    pub(crate) fn block_len(&self) -> usize {
        self.inner.num_evals()
    }

    /// One weight per block.
    #[must_use]
    pub(crate) fn outer(&self) -> &[F] {
        self.outer.as_slice()
    }

    /// One weight per element of a block.
    #[must_use]
    pub(crate) fn inner(&self) -> &[F] {
        self.inner.as_slice()
    }

    /// The entry one before block `index` begins, and zero at the first block.
    ///
    /// A sweep that reads the entry before the one it is at needs this at a block boundary.
    /// Block zero begins at entry zero, which has no entry before it.
    #[must_use]
    pub(crate) fn before_block(&self, index: usize) -> F {
        index.checked_sub(1).map_or(F::ZERO, |previous| {
            self.outer()[previous] * self.inner()[self.block_len() - 1]
        })
    }

    /// The table, written out entry by entry.
    #[cfg(test)]
    pub(crate) fn materialize(&self) -> Poly<F> {
        let mut table = Poly::zero(self.num_variables());
        for (block, &weight) in table
            .as_mut_slice()
            .chunks_mut(self.block_len())
            .zip(self.outer())
        {
            for (slot, &inner) in block.iter_mut().zip(self.inner()) {
                *slot = weight * inner;
            }
        }
        table
    }
}

/// Write into `out` the coordinate sums `sums` takes once the value it reads is scaled.
///
/// # Algorithm
///
/// Scaling is `F_2`-linear, so it commutes with the coordinate decomposition:
///
/// ```text
///     scale * x  =  sum over the set coordinates u of x  of  scale * beta_u
/// ```
///
/// The scaled sums are therefore the sums of one weight per coordinate again, that weight
/// being what the original reads on the scaled basis vector:
///
/// ```text
///     out.sum(x)  ==  sums.sum(scale * x)   for every x
/// ```
///
/// So a sweep whose values all carry the same scale multiplies once per coordinate, however
/// many values it then reads.
///
/// The scale is a field element, and the weights are whatever the sums hold, so a sweep
/// whose sums already live in another representation stays there.
///
/// # Performance
///
/// The table the caller already holds is written through, so a sweep rescaling once per
/// block allocates its `d/8 * 256` entries once rather than at every block.
pub(crate) fn scaled_sums_into<EF: BitCoordinates, A: Field>(
    out: &mut CoordinateSums<EF, A>,
    sums: &CoordinateSums<EF, A>,
    scale: EF,
) {
    out.overwrite(|coordinate| {
        let mut basis = Coefficients::<EF>::zero();
        basis.set(coordinate);
        sums.sum(scale * basis.element())
    });
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BinaryField128;

    #[test]
    fn the_factors_multiply_to_the_dense_table() {
        // Invariant: the factored entry is the dense entry, element for element.
        //
        // The two differ only in how the per-coordinate factors are grouped, and the
        // field's multiplication is associative, so nothing may differ at all.
        let mut rng = SmallRng::seed_from_u64(0xFAC7);
        for num_variables in 0..10 {
            let point = Point::<F>::rand(&mut rng, num_variables);
            let dense = Poly::new_from_point(point.as_slice(), F::ONE);
            for log_block in 0..=num_variables {
                let factored = FactoredEquality::new(point.as_slice(), log_block);
                assert_eq!(factored.num_variables(), num_variables);
                assert_eq!(factored.num_evals(), dense.num_evals());
                assert_eq!(
                    factored.materialize().as_slice(),
                    dense.as_slice(),
                    "{num_variables} variables in blocks of 2^{log_block}"
                );
            }
        }
    }

    #[test]
    fn a_block_wider_than_the_table_is_the_whole_table() {
        // A run shorter than one block leaves the outer factor a single unit weight.
        let mut rng = SmallRng::seed_from_u64(0xB10C);
        let point = Point::<F>::rand(&mut rng, 3);
        let factored = FactoredEquality::new(point.as_slice(), 9);

        assert_eq!(factored.outer(), &[F::ONE]);
        assert_eq!(factored.block_len(), 8);
        assert_eq!(
            factored.inner(),
            Poly::new_from_point(point.as_slice(), F::ONE).as_slice()
        );
    }

    #[test]
    fn the_entry_before_a_block_is_the_last_entry_of_the_one_before() {
        // Invariant: `before_block` reads the dense table one entry below the block start.
        let mut rng = SmallRng::seed_from_u64(0xBEF0);
        let point = Point::<F>::rand(&mut rng, 7);
        let dense = Poly::new_from_point(point.as_slice(), F::ONE);
        let factored = FactoredEquality::new(point.as_slice(), 3);

        assert_eq!(factored.before_block(0), F::ZERO);
        for index in 1..factored.outer().len() {
            assert_eq!(
                factored.before_block(index),
                dense.as_slice()[index * factored.block_len() - 1],
                "block {index}"
            );
        }
    }

    #[test]
    fn scaled_sums_read_the_scaled_value() {
        // Invariant: the scaled sums answer what the original answers on the scaled value.
        //
        //     out.sum(x)  ==  sums.sum(scale * x)
        //
        // That is what lets a block under one weight multiply per coordinate, not per entry.
        // One table serves every scale here, so a rescale must leave nothing of the one before.
        let mut rng = SmallRng::seed_from_u64(0x5CA1);
        let weights = (0..Coefficients::<F>::DIMENSION)
            .map(|_| rng.random())
            .collect::<Vec<F>>();
        let sums = CoordinateSums::new(&weights);

        let mut scaled = CoordinateSums::<F, F>::new(&weights);
        for scale in [F::ZERO, F::ONE, rng.random(), rng.random()] {
            scaled_sums_into(&mut scaled, &sums, scale);
            for value in [F::ZERO, F::ONE]
                .into_iter()
                .chain((0..32).map(|_| rng.random()))
            {
                assert_eq!(scaled.sum(value), sums.sum(scale * value));
            }
        }
    }

    #[test]
    fn a_point_of_no_variables_is_the_single_unit_entry() {
        // The table of no coordinates is one entry, and both factors hold it.
        let factored = FactoredEquality::<F>::new(&[], 4);

        assert_eq!(factored.num_variables(), 0);
        assert_eq!(factored.num_evals(), 1);
        assert_eq!(factored.materialize().as_slice(), &[F::ONE]);
    }

    #[test]
    fn every_block_reads_the_dense_run_it_covers() {
        // Invariant: block `i` is the dense entries `[i * block, (i + 1) * block)`.
        let mut rng = SmallRng::seed_from_u64(0x8100);
        let point = Point::<F>::rand(&mut rng, 8);
        let dense = Poly::new_from_point(point.as_slice(), F::ONE);
        let factored = FactoredEquality::new(point.as_slice(), 4);

        let rebuilt = factored
            .outer()
            .iter()
            .flat_map(|&weight| factored.inner().iter().map(move |&inner| weight * inner))
            .collect::<Vec<_>>();

        assert_eq!(rebuilt, dense.as_slice());
    }
}
