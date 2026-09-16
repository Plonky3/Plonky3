//! Table-driven low-degree extension of bit-valued rows onto the transmitted points.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_maybe_rayon::prelude::*;
use thiserror::Error;

use super::domain::SkipDomain;

/// Bits one lookup covers.
///
/// # Why this value
///
/// - A byte indexes with a single load and needs no unpacking.
/// - Four bits would double the lookups per row to shrink the table to a sixteenth.
/// - Sixteen bits would grow it by a factor of 256 and spill every cache level.
///
/// Table size is `256 * (2^(k+e) - 2^k)` elements, which nothing here bounds.
///
/// The six-bit skip of a degree-two composition makes that 16 KiB, which stays in L1.
///
/// Wider skips leave it, and the win becomes locality rather than residency.
pub const CHUNK_BITS: usize = 8;

/// Number of distinct values one chunk can take, and so the number of rows in the base table.
pub(crate) const TABLE_ROWS: usize = 1 << CHUNK_BITS;

/// Reasons the extension table cannot be built.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompressedLdeError {
    /// The subspace holds fewer points than one chunk covers bits.
    ///
    /// Chunking a row into bytes needs the row to be a whole number of bytes long.
    #[error("a dimension-{log_size} subspace is narrower than one {CHUNK_BITS}-bit chunk")]
    SubspaceTooNarrow {
        /// Dimension of the subspace whose points index the row.
        log_size: usize,
    },
}

/// Extends bit-valued rows from the skipped subspace onto the transmitted points.
///
/// # Overview
///
/// A univariate-skip round reads each row of the trace as a polynomial on the subspace.
///
/// What it needs is that polynomial's values on the transmitted points.
/// Both readings are `F_2`-linear in the row, so one fixed matrix carries the whole map:
///
/// ```text
///     row bits  in F_2^(2^k)   --M-->   transmitted values in F^(2^(k+e) - 2^k)
/// ```
///
/// Applying `M` with a transform pays full field arithmetic at every butterfly.
///
/// At these sizes a transform's asymptotics never pay that back.
/// Tabulating `M` instead turns each row into a handful of loads and exclusive-ors.
///
/// # Algorithm
///
/// Splitting the row into byte chunks splits the matrix into column blocks:
///
/// ```text
///     out[i] = sum_c M[i][c] * bit_c
///            = sum_b sum_{j<8} M[i][8b + j] * bit_{8b + j}
/// ```
///
/// The naive tabulation stores one table per chunk position.
///
/// For a six-bit skip that is eight tables of 16 KiB.
/// One identity collapses them to one:
///
/// ```text
///     M[i][8b + j] = M[i xor 8b][j]
/// ```
///
/// so every chunk position reads the same table, at an index offset by the chunk position:
///
/// ```text
///     out[i] = sum_b T[chunk_b][i xor 8b]
/// ```
///
/// Three facts make the identity hold:
///
/// - The matrix is a Cauchy matrix in the two point sets.
/// - The index-to-point map is `F_2`-linear.
/// - The vanishing polynomial in the numerator is constant along any shift by a subspace element.
///
/// See Bünz, Rothblum, Wang, *Flock*, Section 4.2 and Appendix D.
///
/// # Performance
///
/// The table is `256 * (2^(k+e) - 2^k)` field elements.
/// For a six-bit skip of a degree-two composition over a byte field that is 16 KiB, against 128
/// KiB for the per-position tabulation, which is the whole point: the shared table stays hot.
///
/// Per row the cost is one load and one exclusive-or per output entry per chunk.
///
/// There are no multiplications at all.
#[derive(Debug, Clone)]
pub struct CompressedLde<F> {
    /// One row of transmitted values per chunk value, in row-major order.
    table: Vec<F>,
    /// Number of transmitted values, which is the row stride of the table.
    stride: usize,
    /// Number of byte chunks one input row occupies.
    num_chunks: usize,
}

impl<F: TowerLevel> CompressedLde<F> {
    /// Tabulate the extension map of the given domain.
    ///
    /// # Errors
    ///
    /// Returns an error when the subspace is narrower than one chunk.
    ///
    /// A row would then not be a whole number of chunks.
    pub fn new(domain: &SkipDomain<F>) -> Result<Self, CompressedLdeError> {
        // A row is one bit per subspace point, so it must cover a whole number of chunks.
        if domain.size() < CHUNK_BITS {
            return Err(CompressedLdeError::SubspaceTooNarrow {
                log_size: domain.log_size(),
            });
        }

        let stride = domain.num_transmitted();
        let num_chunks = domain.size() / CHUNK_BITS;

        // Only the first chunk's worth of resampling columns is ever read.
        //
        // The identity above reaches the rest by offsetting the output index instead.
        //
        // The remaining columns are therefore never built.
        let columns = domain.resampling_prefix(CHUNK_BITS);

        // Each table row is the extension of a chunk sitting at position zero.
        //
        //     T[v][i] = sum over set bits j of v of M[i][j]
        //
        // Peeling the lowest set bit turns that sum into one vector addition per table row.
        let mut table = F::zero_vec(TABLE_ROWS * stride);
        for value in 1..TABLE_ROWS {
            // The column this bit selects, and the already-built row for the remaining bits.
            let bit = value.trailing_zeros() as usize;
            let (built, building) = table.split_at_mut(value * stride);
            let previous = &built[(value & (value - 1)) * stride..][..stride];

            // Add the selected column onto the row for the remaining bits.
            for (index, entry) in building[..stride].iter_mut().enumerate() {
                *entry = previous[index] + columns[index * CHUNK_BITS + bit];
            }
        }

        Ok(Self {
            table,
            stride,
            num_chunks,
        })
    }

    /// Number of transmitted values one extended row produces.
    #[must_use]
    pub const fn num_transmitted(&self) -> usize {
        self.stride
    }

    /// Number of bytes one packed input row occupies.
    #[must_use]
    pub const fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Number of field elements the table holds.
    #[must_use]
    pub const fn table_len(&self) -> usize {
        self.table.len()
    }

    /// The tabulated extension of one chunk value sitting at the lowest chunk position.
    fn chunk_row(&self, value: u8) -> &[F] {
        &self.table[usize::from(value) * self.stride..][..self.stride]
    }

    /// Extend one packed row onto the transmitted points.
    ///
    /// # Arguments
    ///
    /// - `row`: the row's bits, least significant bit first, one byte per chunk.
    ///   Bit `j` of byte `b` is the row's value at subspace point `8 * b + j`.
    /// - `out`: receives one field element per transmitted point, in index order.
    ///
    /// # Panics
    ///
    /// Panics if either slice has the wrong length for this table.
    pub fn extend(&self, row: &[u8], out: &mut [F]) {
        assert_eq!(row.len(), self.num_chunks, "one byte per chunk");
        assert_eq!(out.len(), self.stride, "one output per transmitted point");

        // Moving a chunk to position `chunk` offsets the output index by `8 * chunk`.
        //
        // That only ever flips bits above the third.
        //
        //     out block q  <-  table block (q xor chunk)
        //
        // Viewing both sides as fixed-size blocks turns that into one exclusive-or per block.
        //
        // It also lets the inner loop vectorize with no per-entry bounds check.
        let (blocks, tail) = out.as_chunks_mut::<CHUNK_BITS>();
        debug_assert!(
            tail.is_empty(),
            "the transmitted count is a multiple of the chunk"
        );

        // Every chunk contributes to every output, so the accumulator starts empty.
        blocks.fill([F::ZERO; CHUNK_BITS]);

        // Walk the chunks, accumulating each one's tabulated contribution.
        for (chunk, &value) in row.iter().enumerate() {
            let (source, _) = self.chunk_row(value).as_chunks::<CHUNK_BITS>();
            for (block, destination) in blocks.iter_mut().enumerate() {
                let contribution = &source[block ^ chunk];
                for index in 0..CHUNK_BITS {
                    destination[index] += contribution[index];
                }
            }
        }
    }

    /// Extend many packed rows onto the transmitted points.
    ///
    /// Rows are independent, so the work splits across threads with no coordination.
    ///
    /// # Arguments
    ///
    /// - `rows`: the packed rows back to back, each one chunk-count bytes long.
    /// - `out`: receives the transmitted values of every row, in the same order.
    ///
    /// # Panics
    ///
    /// Panics if either slice is not a whole number of rows, or if the two disagree on how many.
    pub fn extend_batch(&self, rows: &[u8], out: &mut [F])
    where
        F: Send + Sync,
    {
        assert_eq!(rows.len() % self.num_chunks, 0, "whole input rows");
        assert_eq!(out.len() % self.stride, 0, "whole output rows");
        assert_eq!(
            rows.len() / self.num_chunks,
            out.len() / self.stride,
            "one output row per input row"
        );

        // One row per task, with the table shared read-only across all of them.
        out.par_chunks_exact_mut(self.stride)
            .zip(rows.par_chunks_exact(self.num_chunks))
            .for_each(|(destination, row)| self.extend(row, destination));
    }
}

/// Extend one packed row by interpolating it directly, without a table.
///
/// This is the definition the tabulated path stands in for, and exists to test against.
///
/// It costs one inversion per subspace point per output, so it is far too slow to prove with.
///
/// Reaching it needs the test-utility feature, so no production path can pick it up by mistake.
///
/// # Panics
///
/// Panics if either slice has the wrong length for this domain.
#[cfg(any(test, feature = "test-util"))]
pub fn extend_reference<F: TowerLevel>(domain: &SkipDomain<F>, row: &[u8], out: &mut [F]) {
    assert_eq!(row.len() * CHUNK_BITS, domain.size(), "one byte per chunk");
    assert_eq!(
        out.len(),
        domain.num_transmitted(),
        "one output per transmitted point"
    );

    // Unpack the row into one field element per subspace point, least significant bit first.
    let values = (0..domain.size())
        .map(|index| {
            let bit = (row[index / CHUNK_BITS] >> (index % CHUNK_BITS)) & 1;
            if bit == 1 { F::ONE } else { F::ZERO }
        })
        .collect::<Vec<_>>();

    // Read the interpolating polynomial at each transmitted point.
    for (entry, &point) in out.iter_mut().zip(domain.transmitted()) {
        *entry = domain.interpolate(&values, point);
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField16};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    // Six skipped rounds over a byte field, one extra dimension for a degree-two composition.
    // This is the configuration a product-form zerocheck runs.
    //
    // It is also the one the 16 KiB claim is about.
    const LOG_SIZE: usize = 6;

    fn product_domain() -> SkipDomain<BinaryField8> {
        SkipDomain::for_degree(LOG_SIZE, 2).unwrap()
    }

    #[test]
    fn the_product_form_table_is_sixteen_kibibytes() {
        // Fixture state: 64 subspace points, 64 transmitted points, one byte per entry.
        //
        //     256 chunk values * 64 transmitted points * 1 byte = 16 KiB
        //
        // The per-position tabulation this replaces would be 8 tables of the same size.
        let lde = CompressedLde::new(&product_domain()).unwrap();
        assert_eq!(lde.num_chunks(), 8);
        assert_eq!(lde.num_transmitted(), 64);
        assert_eq!(lde.table_len() * size_of::<BinaryField8>(), 16 * 1024);
    }

    #[test]
    fn the_empty_row_extends_to_zero() {
        // The extension is F_2-linear, so the all-zero row must produce all-zero output.
        let domain = product_domain();
        let lde = CompressedLde::new(&domain).unwrap();
        let mut out = BinaryField8::zero_vec(lde.num_transmitted());
        lde.extend(&[0u8; 8], &mut out);
        assert!(out.iter().all(|&value| value == BinaryField8::ZERO));
    }

    #[test]
    fn a_single_bit_reproduces_its_lagrange_basis_polynomial() {
        // Fixture state: a row with exactly one bit set at subspace index c.
        //
        // The extension is then the Lagrange basis polynomial that index selects.
        //
        // Reading it at each transmitted point must reproduce the matrix column.
        //
        //     row = e_c  ->  out[i] = M[i][c]
        let domain = product_domain();
        let lde = CompressedLde::new(&domain).unwrap();
        let matrix = domain.resampling_matrix();

        let mut out = BinaryField8::zero_vec(lde.num_transmitted());
        for column in 0..domain.size() {
            // Set exactly the bit this subspace index maps to.
            let mut row = [0u8; 8];
            row[column / CHUNK_BITS] = 1 << (column % CHUNK_BITS);
            lde.extend(&row, &mut out);

            // Compare against the matrix column the bit selects.
            for (index, &value) in out.iter().enumerate() {
                assert_eq!(value, matrix[index * domain.size() + column], "c={column}");
            }
        }
    }

    #[test]
    fn extension_is_additive_over_the_row_bits() {
        // Invariant: the map is F_2-linear.
        //
        // The exclusive-or of two rows therefore extends to the sum of the two extensions.
        //
        // This is what licenses splitting a row into independent chunks in the first place.
        let mut rng = SmallRng::seed_from_u64(5);
        let lde = CompressedLde::new(&product_domain()).unwrap();

        let left: [u8; 8] = rng.random();
        let right: [u8; 8] = rng.random();
        let mixed: [u8; 8] = core::array::from_fn(|index| left[index] ^ right[index]);

        let mut a = BinaryField8::zero_vec(lde.num_transmitted());
        let mut b = BinaryField8::zero_vec(lde.num_transmitted());
        let mut c = BinaryField8::zero_vec(lde.num_transmitted());
        lde.extend(&left, &mut a);
        lde.extend(&right, &mut b);
        lde.extend(&mixed, &mut c);

        for index in 0..lde.num_transmitted() {
            assert_eq!(c[index], a[index] + b[index], "index={index}");
        }
    }

    #[test]
    fn rejects_a_subspace_narrower_than_one_chunk() {
        // Mutation: a dimension-2 subspace is 4 points, so a row is half a byte.
        let domain = SkipDomain::<BinaryField8>::new(2, 3).unwrap();
        assert_eq!(
            CompressedLde::new(&domain).unwrap_err(),
            CompressedLdeError::SubspaceTooNarrow { log_size: 2 }
        );
        // Dimension 3 is exactly one chunk, so it is the narrowest that works.
        let domain = SkipDomain::<BinaryField8>::new(3, 4).unwrap();
        assert!(CompressedLde::new(&domain).is_ok());
    }

    #[test]
    fn batching_agrees_with_extending_one_row_at_a_time() {
        // Fixture state: 37 random rows, an odd count so no power-of-two path is assumed.
        let mut rng = SmallRng::seed_from_u64(13);
        let lde = CompressedLde::new(&product_domain()).unwrap();
        let num_rows = 37;

        let rows = (0..num_rows * lde.num_chunks())
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();

        // Batch the whole block, then redo it row by row.
        let mut batched = BinaryField8::zero_vec(num_rows * lde.num_transmitted());
        lde.extend_batch(&rows, &mut batched);

        let mut single = BinaryField8::zero_vec(lde.num_transmitted());
        for (index, row) in rows.chunks_exact(lde.num_chunks()).enumerate() {
            lde.extend(row, &mut single);
            assert_eq!(
                &batched[index * lde.num_transmitted()..][..lde.num_transmitted()],
                &single[..],
                "row={index}"
            );
        }
    }

    proptest! {
        #[test]
        fn tabulated_extension_matches_direct_interpolation(seed: u64) {
            // Invariant: the table is only a faster way to evaluate the interpolating polynomial.
            //
            // Fixture state: a random row of 64 bits over the product-form domain.
            let mut rng = SmallRng::seed_from_u64(seed);
            let domain = product_domain();
            let lde = CompressedLde::new(&domain).unwrap();

            let row = (0..lde.num_chunks()).map(|_| rng.random::<u8>()).collect::<Vec<_>>();

            // The table path and the Lagrange path must agree entry for entry.
            let mut fast = BinaryField8::zero_vec(lde.num_transmitted());
            let mut slow = BinaryField8::zero_vec(lde.num_transmitted());
            lde.extend(&row, &mut fast);
            extend_reference(&domain, &row, &mut slow);
            prop_assert_eq!(fast, slow);
        }

        #[test]
        fn tabulated_extension_matches_across_domain_shapes(
            seed: u64,
            log_size in 3usize..=8,
            extra in 1usize..=4,
        ) {
            // Invariant: the shared table's identity holds at every dimension and width.
            //
            // The widest shapes here give several cosets.
            //
            // A block index then runs past `2^(k-3)`, and the offset must stay in its coset.
            //
            // It is not special to the product-form configuration.
            //
            // Fixture state: dimension `log_size` inside dimension `log_size + extra`, over a
            // 16-bit field so the widest combination still fits.
            let mut rng = SmallRng::seed_from_u64(seed);
            let domain = SkipDomain::<BinaryField16>::new(log_size, log_size + extra).unwrap();
            let lde = CompressedLde::new(&domain).unwrap();

            let row = (0..lde.num_chunks()).map(|_| rng.random::<u8>()).collect::<Vec<_>>();

            let mut fast = BinaryField16::zero_vec(lde.num_transmitted());
            let mut slow = BinaryField16::zero_vec(lde.num_transmitted());
            lde.extend(&row, &mut fast);
            extend_reference(&domain, &row, &mut slow);
            prop_assert_eq!(fast, slow);
        }
    }
}
