//! Small block repeated down the diagonal.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;

use super::{LinearOperator, SparseMatrix, axis_len, encode_usize};

/// Encoding tag.
const TAG: u8 = 2;

/// Block repeated once per data-parallel copy, with the copy index as the high bits.
///
/// ```text
/// M~((r_copy, r_in), (y_copy, y_in)) = eq(r_copy, y_copy) * B~(r_in, y_in)
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDiagonal<F, B = SparseMatrix<F>> {
    /// Copy variables.
    log_copies: usize,
    /// Block every copy applies.
    block: B,
    /// Coefficient field of the block; no auto trait depends on it.
    _field: PhantomData<fn() -> F>,
}

/// Bit shift inside every word, bit zero first; bits shifted out are lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitShift {
    /// Output bit i reads input bit i - s.
    LogicalLeft(usize),
    /// Output bit i reads input bit i + s.
    LogicalRight(usize),
    /// Output bit i reads input bit (i + s) mod w.
    RotateRight(usize),
}

impl BitShift {
    /// Input bit an output bit reads, if any.
    const fn source(self, bit: usize, word_bits: usize) -> Option<usize> {
        match self {
            Self::LogicalLeft(s) => bit.checked_sub(s),
            Self::LogicalRight(s) => match bit.checked_add(s) {
                Some(src) if src < word_bits => Some(src),
                _ => None,
            },
            Self::RotateRight(s) => Some((bit + s % word_bits) % word_bits),
        }
    }
}

impl<F: Field, B: LinearOperator<F>> BlockDiagonal<F, B> {
    /// Repeat a block once per copy.
    ///
    /// # Panics
    ///
    /// - An axis overflows usize.
    #[must_use]
    pub fn new(log_copies: usize, block: B) -> Self {
        for log_block in [block.log_rows(), block.log_cols()] {
            axis_len(
                log_copies
                    .checked_add(log_block)
                    .expect("axis length overflows usize"),
            );
        }
        Self {
            log_copies,
            block,
            _field: PhantomData,
        }
    }

    /// Copy variables.
    #[must_use]
    pub const fn log_copies(&self) -> usize {
        self.log_copies
    }

    /// Block every copy applies.
    #[must_use]
    pub const fn block(&self) -> &B {
        &self.block
    }
}

impl<F: Field> BlockDiagonal<F> {
    /// Shift the bits of every word, each word being 2^log_word_bits consecutive entries.
    ///
    /// # Panics
    ///
    /// - An axis overflows usize.
    #[must_use]
    pub fn bit_shift(log_words: usize, log_word_bits: usize, shift: BitShift) -> Self {
        // One unit entry per output bit with a source.
        let word_bits = axis_len(log_word_bits);
        let entries = (0..word_bits)
            .filter_map(|bit| shift.source(bit, word_bits).map(|src| (bit, src, F::ONE)))
            .collect();

        // One block per word.
        Self::new(
            log_words,
            SparseMatrix::new(log_word_bits, log_word_bits, entries),
        )
    }
}

impl<F: Field, B: LinearOperator<F>> LinearOperator<F> for BlockDiagonal<F, B> {
    fn log_rows(&self) -> usize {
        self.log_copies + self.block.log_rows()
    }

    fn log_cols(&self) -> usize {
        self.log_copies + self.block.log_cols()
    }

    fn apply(&self, z: &[F], out: &mut [F]) {
        assert_eq!(z.len(), 1 << self.log_cols(), "witness length");
        assert_eq!(out.len(), 1 << self.log_rows(), "output length");

        // out[copy t] = B * z[copy t], in parallel.
        let (block_rows, block_cols) = (1 << self.block.log_rows(), 1 << self.block.log_cols());
        out.par_chunks_exact_mut(block_rows)
            .zip(z.par_chunks_exact(block_cols))
            .with_min_task_bytes((block_rows + block_cols) * size_of::<F>())
            .for_each(|(out, z)| self.block.apply(z, out));
    }

    fn add_transpose<EF: ExtensionField<F>>(&self, weights: &[EF], scale: EF, out: &mut [EF]) {
        assert_eq!(weights.len(), 1 << self.log_rows(), "weights length");
        assert_eq!(out.len(), 1 << self.log_cols(), "output length");

        // out[copy t] += scale * B^T * w[copy t], in parallel.
        let (block_rows, block_cols) = (1 << self.block.log_rows(), 1 << self.block.log_cols());
        out.par_chunks_exact_mut(block_cols)
            .zip(weights.par_chunks_exact(block_rows))
            .with_min_task_bytes((block_rows + block_cols) * size_of::<EF>())
            .for_each(|(out, w)| self.block.add_transpose(w, scale, out));
    }

    fn evaluate<EF: ExtensionField<F>>(&self, row: &[EF], col: &[EF]) -> EF {
        assert_eq!(row.len(), self.log_rows(), "row point length");
        assert_eq!(col.len(), self.log_cols(), "column point length");

        // Copy bits lead both indices.
        let (row_copy, row_in) = row.split_at(self.log_copies);
        let (col_copy, col_in) = col.split_at(self.log_copies);

        // Nonzero only where the copies agree.
        Point::eval_eq(row_copy, col_copy) * self.block.evaluate(row_in, col_in)
    }

    fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        // Tag, copy count, then the self-delimiting block.
        out.push(TAG);
        encode_usize(out, self.log_copies);
        self.block.encode(out);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::operator::testing::assert_operator_contract;

    /// Base field.
    type F = BabyBear;
    /// Challenge field.
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn two_copies_of_the_r1cs_block_run_independently() {
        // Fixture state: the (x, x + y) block on copies (3, 4) and (4, 3).
        let block = SparseMatrix::new(1, 2, vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]);
        let m = BlockDiagonal::new(1, block);
        let z = [1, 3, 4, 0, 1, 4, 3, 0].map(F::from_u32);
        let mut out = [F::ZERO; 4];
        m.apply(&z, &mut out);
        assert_eq!(out, [3, 7, 4, 7].map(F::from_u32));
    }

    #[test]
    fn rotation_moves_every_bit_three_places_down() {
        // Fixture state: 0xb2 rotated right by 3 is 0x56.
        let input = [0, 1, 0, 0, 1, 1, 0, 1].map(F::from_u32);
        let mut out = [F::ZERO; 8];
        BlockDiagonal::bit_shift(0, 3, BitShift::RotateRight(3)).apply(&input, &mut out);
        assert_eq!(out, [0, 1, 1, 0, 1, 0, 1, 0].map(F::from_u32));
    }

    #[test]
    fn every_shift_satisfies_the_operator_contract() {
        // Invariant: apply, transpose and evaluation describe the same matrix.
        //
        // Fixture state: 4 words of 8 bits, every shift kind, every amount.
        for amount in 0..9 {
            for shift in [
                BitShift::LogicalLeft(amount),
                BitShift::LogicalRight(amount),
                BitShift::RotateRight(amount),
            ] {
                let m = BlockDiagonal::<F>::bit_shift(2, 3, shift);
                assert_operator_contract::<F, EF, _>(&m, amount as u64);
            }
        }
    }

    #[test]
    fn a_rectangular_block_satisfies_the_operator_contract() {
        // Fixture state: 8 copies of a 2 x 4 block with weighted entries.
        let block = SparseMatrix::new(
            1,
            2,
            vec![(0, 3, F::from_u32(9)), (1, 0, F::NEG_ONE), (1, 1, F::TWO)],
        );
        assert_operator_contract::<F, EF, _>(&BlockDiagonal::new(3, block), 5);
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn a_copy_count_that_overflows_an_axis_is_rejected() {
        // Mutation: usize::MAX copy bits plus one block bit wraps without a checked add.
        let block = SparseMatrix::<F>::new(1, 1, vec![]);
        let _ = BlockDiagonal::new(usize::MAX, block);
    }
}
