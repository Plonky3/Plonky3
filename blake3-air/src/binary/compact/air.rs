use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::Borrow;

use p3_air::utils::u32_to_bits_le;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::super::air::{CONSTRAINTS_PER_G, State, eval_g};
use super::super::{Blake3CompressionInput, G_PER_ROUND, G_SCHEDULE, iv_word};
use super::columns::{Blake3BinaryCompactCols, NUM_BLAKE3_BINARY_COMPACT_COLS};
use super::generation::{generate_binary_compact_trace_packed, generate_binary_compact_trace_rows};
use crate::constants::permute;

const NUM_ROUND_INPUT_BITS: usize = (8 + 16 + 16) * 32;
const NUM_CONSTRAINTS: usize = G_PER_ROUND * CONSTRAINTS_PER_G + 12 * 32 + 40 * 32;

/// A compact binary Blake-3 AIR with one round per row.
///
/// These constraints describe Blake-3 only over fields of characteristic two.
///
/// Each compression occupies eight rows: seven real rounds and a valid independent
/// padding round. Public period-eight selectors initialize each compression and link
/// its rounds. The padding row has no links to either compression; the generator fills
/// it with zero, but any valid round is accepted there. Every constraint has degree at most two.
///
/// The width is 2,800 cells, compared with 11,536 for [`super::super::Blake3BinaryAir`].
/// With Booleanity supplied by the commitment this AIR has 3,184 constraints, compared
/// with 10,640. For a standalone table at an equal number of compressions, it has eight times
/// as many rows and twice the padded commitment domain. Successor reads also disable
/// some prover optimizations available to the compression-per-row AIR.
///
/// Full successor openings preserve the Boolean PCS's column-batched route. This means
/// 5,600 opened values per row point, rather than the wide AIR's 11,536. This variant is
/// intended for callers trading prover work for a smaller recursive verifier.
///
/// ```
/// use core::borrow::Borrow;
/// use p3_binary_field::BinaryField128;
/// use p3_blake3_air::{
///     BLAKE3_BINARY_COMPACT_OUTPUT_ROW, Blake3BinaryCompactCols, Blake3CompressionInput,
///     generate_binary_compact_trace_rows,
/// };
/// use p3_matrix::Matrix;
///
/// let trace = generate_binary_compact_trace_rows::<BinaryField128>(
///     vec![Blake3CompressionInput::default()], 0,
/// );
/// let values = trace.row_slice(BLAKE3_BINARY_COMPACT_OUTPUT_ROW).unwrap();
/// let final_round: &Blake3BinaryCompactCols<BinaryField128> = (*values).borrow();
/// let output = final_round.compression_output();
/// ```
#[derive(Debug)]
pub struct Blake3BinaryCompactAir {
    constrain_booleanity: bool,
}

impl Default for Blake3BinaryCompactAir {
    fn default() -> Self {
        Self {
            constrain_booleanity: true,
        }
    }
}

impl Blake3BinaryCompactAir {
    /// Omit input Booleanity assertions when the commitment guarantees one bit per cell.
    ///
    /// This reports [`BaseAir::assumes_boolean_trace`]. A commitment to unrestricted
    /// field elements does not make this mode sound.
    pub const fn assuming_boolean_trace() -> Self {
        Self {
            constrain_booleanity: false,
        }
    }

    /// Generate eight rows per fixed-seed random compression, for benches/examples.
    ///
    /// Uses the same inputs as the wide AIR's random generator. For specific inputs,
    /// use [`generate_binary_compact_trace_rows`].
    pub fn generate_random_trace_rows<F: Field>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        generate_binary_compact_trace_rows(random_inputs(num_hashes), extra_capacity_bits)
    }

    /// Generate the same random inputs as [`Self::generate_random_trace_rows`],
    /// packed into one `u64` per 64 logical rows.
    pub fn generate_random_trace_packed<F: Field>(&self, num_hashes: usize) -> RowMajorMatrix<u64> {
        generate_binary_compact_trace_packed::<F>(random_inputs(num_hashes))
    }
}

fn random_inputs(num_hashes: usize) -> Vec<Blake3CompressionInput> {
    let mut rng = SmallRng::seed_from_u64(1);
    (0..num_hashes)
        .map(|i| Blake3CompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            counter: i as u64,
            block_len: 64,
            flags: 0,
        })
        .collect()
}

impl<F: PrimeCharacteristicRing> BaseAir<F> for Blake3BinaryCompactAir {
    fn width(&self) -> usize {
        NUM_BLAKE3_BINARY_COMPACT_COLS
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        // A partial list falls back to per-column proofs in the Boolean PCS.
        (0..NUM_BLAKE3_BINARY_COMPACT_COLS).collect()
    }

    fn assumes_boolean_trace(&self) -> bool {
        !self.constrain_booleanity
    }

    fn num_constraints(&self) -> Option<usize> {
        Some(
            NUM_CONSTRAINTS
                + if self.constrain_booleanity {
                    NUM_ROUND_INPUT_BITS
                } else {
                    0
                },
        )
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }

    fn num_periodic_columns(&self) -> usize {
        2
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        Cow::Owned(vec![
            vec![
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ],
            vec![
                F::ONE,
                F::ONE,
                F::ONE,
                F::ONE,
                F::ONE,
                F::ONE,
                F::ZERO,
                F::ZERO,
            ],
        ])
    }

    fn periodic_evaluations<EF>(&self, point: &[EF]) -> Option<Vec<EF>>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let [a, b, c] = point.get(point.len().checked_sub(3)?..)? else {
            return None;
        };
        Some(vec![
            (EF::ONE - *a) * (EF::ONE - *b) * (EF::ONE - *c),
            EF::ONE - *a * *b,
        ])
    }
}

impl<AB: AirBuilder> Air<AB> for Blake3BinaryCompactAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Blake3BinaryCompactCols<AB::Var> = main.current_slice().borrow();
        let next: &Blake3BinaryCompactCols<AB::Var> = main.next_slice().borrow();
        let first: AB::Expr = builder.periodic_values()[0].into();
        let continuing: AB::Expr = builder.periodic_values()[1].into();

        if self.constrain_booleanity {
            for word in local
                .chaining_value
                .iter()
                .chain(&local.block)
                .chain(&local.state)
            {
                builder.assert_bools(*word);
            }
        }

        let mut state = State::<AB> {
            a: array::from_fn(|i| local.state[i].map(Into::into)),
            b: array::from_fn(|i| local.state[4 + i]),
            c: array::from_fn(|i| local.state[8 + i].map(Into::into)),
            d: array::from_fn(|i| local.state[12 + i]),
        };
        // Quadratic G equations remain unconditional, including on the padding row.
        for (g, (cols, slots)) in local.round.iter().zip(G_SCHEDULE).enumerate() {
            eval_g(
                builder,
                &mut state,
                slots,
                &local.block[2 * g],
                &local.block[2 * g + 1],
                cols,
            );
        }

        let mut initialized = builder.when(first);
        for (word, cv) in local.state[..8].iter().zip(&local.chaining_value) {
            for (&bit, &cv_bit) in word.iter().zip(cv) {
                initialized.assert_eq(bit, cv_bit);
            }
        }
        for (i, word) in local.state[8..12].iter().enumerate() {
            let iv: [AB::Expr; 32] = u32_to_bits_le(iv_word(i));
            for (&bit, iv_bit) in word.iter().zip(iv) {
                initialized.assert_eq(bit, iv_bit);
            }
        }

        let mut linked = builder.when(continuing);
        for i in 0..4 {
            for bit in 0..32 {
                linked.assert_eq(next.state[i][bit], state.a[i][bit].clone());
                linked.assert_eq(next.state[4 + i][bit], state.b[i][bit]);
                linked.assert_eq(next.state[8 + i][bit], state.c[i][bit].clone());
                linked.assert_eq(next.state[12 + i][bit], state.d[i][bit]);
            }
        }
        let mut message = local.block;
        permute(&mut message);
        for (next_word, word) in next.block.iter().zip(message) {
            for (&next_bit, bit) in next_word.iter().zip(word) {
                linked.assert_eq(next_bit, bit);
            }
        }
        for (next_word, word) in next.chaining_value.iter().zip(&local.chaining_value) {
            for (&next_bit, &bit) in next_word.iter().zip(word) {
                linked.assert_eq(next_bit, bit);
            }
        }
    }
}
