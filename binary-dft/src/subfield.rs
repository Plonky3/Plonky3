//! The additive NTT of a message whose entries lie in a byte-aligned tower subfield.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::butterfly::ButterflyField;
use crate::encoder::padded_message_len;
use crate::lch::transform_stages;

/// Number of widest butterfly layers a subfield of `2^log_bits` bits is closed under.
///
/// - Layer `j` of a height-`2^l` unshifted transform scales by a point of `span(v_1..v_{l-1-j})`.
/// - A subfield of `b` bits holds the first `b` basis vectors and nothing above them.
/// - So the layer stays inside it exactly while `l - 1 - j < b`, the top `b` layers.
const fn closed_layers(log_bits: usize) -> usize {
    1 << log_bits
}

/// Bytes a message must carry before running its closed layers narrow pays.
///
/// Those layers are a transform of the message alone, at a fraction of the alphabet's width.
///
/// A short one is a few microseconds of work behind one thread handoff per layer.
/// The narrow element size does not save enough to cover those handoffs.
///
/// A hundred and twenty-eight kibibytes is the smallest message measured to come out ahead.
const NARROW_MESSAGE_BYTES: usize = 128 * 1024;

/// Bytes the codeword must carry before the widening pass is worth handing to the machine.
///
/// The pass is one read and one write per entry, with no arithmetic between them.
/// It is therefore limited by memory rather than by cores.
///
/// Four mebibytes is the smallest codeword measured to gain from being spread.
const PARALLEL_WIDEN_BYTES: usize = 4 * 1024 * 1024;

/// Embed every entry into the wider level, in one pass over the data.
fn widen<S: TowerLevel, F: TowerLevel + From<S>>(values: &[S]) -> Vec<F> {
    if size_of::<F>() * values.len() < PARALLEL_WIDEN_BYTES {
        return values.iter().map(|&value| F::from(value)).collect();
    }
    values.par_iter().map(|&value| F::from(value)).collect()
}

/// Evaluate a subfield message with the widest `head` layers run at the narrow element width.
///
/// # Algorithm
///
/// Write `l` for the domain dimension and `h` for the layers to run narrow.
/// A row index splits into its top `h` bits and its bottom `l - h` bits:
///
/// ```text
///     row = a * 2^(l-h) + b ,     a < 2^h ,  b < 2^(l-h)
/// ```
///
/// The top `h` layers pair rows that differ in `a` alone, so they leave `b` untouched:
///
/// ```text
///     phase 1   the buffer read as 2^h rows of 2^(l-h) * width columns, transformed whole
///     phase 2   the remaining l - h layers of the transform the buffer really is
/// ```
///
/// Both phases are unshifted and share block indices, so they agree twiddle for twiddle.
///
/// # Panics
///
/// Panics if `head` exceeds the layers the subfield is closed under.
/// Phase 1 would then ask the subfield for a basis vector above its own.
fn split_ntt_batch<S, F>(mut mat: RowMajorMatrix<S>, head: usize) -> RowMajorMatrix<F>
where
    S: ButterflyField,
    F: ButterflyField + From<S>,
{
    let width = mat.width();
    let log_n = log2_strict_usize(mat.height());
    assert!(log_n <= 1 << F::LOG_BITS, "domain exceeds field dimension");
    assert!(
        head <= closed_layers(S::LOG_BITS).min(log_n),
        "the subfield is not closed under that many layers"
    );
    let low = log_n - head;

    // Phase 1: every layer of a shorter, wider reading of the same buffer.
    transform_stages::<S, false>(&mut mat.values, width << low, head, S::ZERO);

    let mut values = widen::<S, F>(&mat.values);

    // Phase 2: the layers the subfield is not closed under, at the alphabet's own width.
    transform_stages::<F, false>(&mut values, width, low, F::ZERO);

    RowMajorMatrix::new(values, width)
}

/// Evaluate each column of a subfield message on the subspace of the wider level.
///
/// Widening the message first pays the wide element size at every layer.
/// The widest layers do not need it: their twiddles map the subfield into itself.
///
/// The layers left over run the tower transform, not whichever backend is fastest wide.
///
/// # Panics
///
/// Panics if the height is not a power of two.
/// Panics if the domain dimension exceeds the bit width of the wider level.
#[must_use]
pub fn subfield_ntt_batch<S, F>(mat: RowMajorMatrix<S>) -> RowMajorMatrix<F>
where
    S: ButterflyField,
    F: ButterflyField + From<S>,
{
    let log_n = log2_strict_usize(mat.height());

    // A message too small to pay for the narrow phase runs no layer narrow.
    // What is left is the ordinary transform of the widened message.
    let head = if size_of::<S>() * mat.values.len() >= NARROW_MESSAGE_BYTES {
        closed_layers(S::LOG_BITS).min(log_n)
    } else {
        0
    };
    split_ntt_batch(mat, head)
}

/// Reed–Solomon encode a subfield message over the additive NTT domain.
///
/// The message holds the low-index novel-basis coefficients of each column.
/// The codeword is their evaluation on the subspace `log_inv_rate` dimensions above.
///
/// Zero lies in the subfield, so the padding that extends the domain is paid narrow too.
///
/// # Panics
///
/// Panics under the same conditions as the plain transform.
/// Panics if the codeword length overflows the address space.
#[must_use]
pub fn subfield_encode_batch<S, F>(
    mut message: RowMajorMatrix<S>,
    log_inv_rate: usize,
) -> RowMajorMatrix<F>
where
    S: ButterflyField,
    F: ButterflyField + From<S>,
{
    let padded_len = padded_message_len(message.values.len(), log_inv_rate);
    let _ = log2_strict_usize(message.height());

    message.values.resize(padded_len, S::ZERO);
    subfield_ntt_batch(message)
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use p3_binary_field::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Gf2, TowerLevel,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::{closed_layers, split_ntt_batch, subfield_encode_batch, subfield_ntt_batch, widen};
    use crate::butterfly::ButterflyField;
    use crate::domain::{domain_point, subspace_polynomial};
    use crate::lch::LchNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Widths covering a single column, an odd row, and a row of several elements.
    const WIDTHS: [usize; 4] = [1, 3, 4, 16];

    /// A matrix of subfield entries that depend on both the position and the seed.
    fn matrix<S: TowerLevel>(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<S> {
        RowMajorMatrix::new(
            (0..(width << log_n))
                .map(|i| {
                    let bits = seed
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(i as u64 + 1)
                        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    S::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
                })
                .collect(),
            width,
        )
    }

    /// Every split depth against the unspecialised transform on the widened message.
    fn check_against_the_wide_transform<S, F>(log_n: usize, width: usize, seed: u64)
    where
        S: ButterflyField,
        F: ButterflyField + From<S>,
    {
        let message = matrix::<S>(log_n, width, seed);

        // The reference widens first and runs the ordinary transform.
        let wide = RowMajorMatrix::new(widen::<S, F>(&message.values), width);
        let expected = LchNtt::<F>::default().ntt_batch(wide);

        for head in 0..=closed_layers(S::LOG_BITS).min(log_n) {
            let label = format!("log_n={log_n} width={width} seed={seed} head={head}");
            assert_eq!(
                split_ntt_batch::<S, F>(message.clone(), head),
                expected,
                "{label}"
            );
        }

        // And the entry point itself, at whichever depth its own size rule picks.
        assert_eq!(subfield_ntt_batch::<S, F>(message), expected);
    }

    #[test]
    fn a_byte_message_transforms_like_the_widened_one() {
        // Heights on both sides of the eight layers a byte subfield is closed under.
        for width in WIDTHS {
            for log_n in 0..=10 {
                check_against_the_wide_transform::<BinaryField8, BinaryField128>(log_n, width, 7);
                check_against_the_wide_transform::<BinaryField8, BinaryField64>(log_n, width, 11);
                check_against_the_wide_transform::<BinaryField8, BinaryField32>(log_n, width, 13);
            }
        }
    }

    #[test]
    fn a_bit_message_transforms_like_the_widened_one() {
        // A single-bit subfield is closed under one layer, the narrowest split there is.
        for width in WIDTHS {
            for log_n in 0..=8 {
                check_against_the_wide_transform::<Gf2, BinaryField128>(log_n, width, 17);
            }
        }
    }

    #[test]
    fn a_two_and_four_byte_message_transforms_like_the_widened_one() {
        // Sixteen and thirty-two closed layers, so the heights here are all inside the head.
        for width in WIDTHS {
            for log_n in 0..=10 {
                check_against_the_wide_transform::<BinaryField16, BinaryField128>(log_n, width, 19);
                check_against_the_wide_transform::<BinaryField32, BinaryField128>(log_n, width, 23);
            }
        }
    }

    #[test]
    fn the_widest_subfield_is_the_alphabet_itself() {
        // Invariant: with nothing left over the head, phase 2 has no layers at all.
        for log_n in 0..=6 {
            check_against_the_wide_transform::<BinaryField64, BinaryField64>(log_n, 3, 29);
        }
    }

    #[test]
    #[should_panic = "the subfield is not closed under that many layers"]
    fn a_split_deeper_than_the_subfield_allows_is_refused() {
        // A byte subfield is closed under eight layers.
        // A ninth needs a basis vector it does not have, and would compute another map.
        let message = matrix::<BinaryField8>(10, 1, 0);
        let _ = split_ntt_batch::<BinaryField8, BinaryField128>(message, 9);
    }

    #[test]
    fn a_message_below_the_narrow_threshold_runs_no_narrow_layer() {
        // Invariant: the size rule is what decides, and it decides the same way twice.
        //
        // Fixture state: 2^6 rows of one byte column is 64 bytes, far below the threshold.
        let message = matrix::<BinaryField8>(6, 1, 41);
        assert_eq!(
            subfield_ntt_batch::<BinaryField8, BinaryField128>(message.clone()),
            split_ntt_batch::<BinaryField8, BinaryField128>(message, 0)
        );
    }

    /// The deepest split of one shape, against the novel basis read from its product definition.
    fn check_against_the_oracle(log_n: usize, width: usize) {
        let message = matrix::<BinaryField8>(log_n, width, 31);
        let wide = RowMajorMatrix::new(widen::<_, BinaryField128>(&message.values), width);
        let expected = NaiveAdditiveNtt::default().ntt_batch(wide);

        // The deepest split the height allows, which is where phase 1 does the most.
        let head = closed_layers(BinaryField8::LOG_BITS).min(log_n);
        assert_eq!(
            split_ntt_batch::<_, BinaryField128>(message, head),
            expected,
            "log_n={log_n} width={width}"
        );
    }

    #[test]
    fn a_byte_message_matches_the_reference_oracle() {
        // The oracle evaluates the novel basis straight from its product definition.
        // So it pins the split to that basis, not to another split of the same network.
        for width in [1usize, 3] {
            for log_n in 0..=6 {
                check_against_the_oracle(log_n, width);
            }
        }
    }

    #[test]
    fn a_byte_message_past_the_closed_layers_matches_the_reference_oracle() {
        // Invariant: both phases are pinned to the definition, not the narrow phase alone.
        // A byte subfield is closed under eight layers, so phase 2 is empty below 2^9.
        //
        //     log_n = 9   head = 8, phase 2 runs one layer at the wide width
        //     log_n = 10  head = 8, phase 2 runs two of them
        for log_n in [9usize, 10] {
            check_against_the_oracle(log_n, 1);
        }
    }

    #[test]
    fn the_closed_layer_count_is_the_subfield_bit_width() {
        // Fixture state: the layer count a level's own basis vectors allow.
        //
        // - one bit gives one layer
        // - eight bits give eight layers
        // - a hundred and twenty-eight bits give that many
        assert_eq!(closed_layers(Gf2::LOG_BITS), 1);
        assert_eq!(closed_layers(BinaryField8::LOG_BITS), 8);
        assert_eq!(closed_layers(BinaryField128::LOG_BITS), 128);
    }

    #[test]
    fn the_closed_layers_really_keep_their_twiddles_in_the_subfield() {
        // Invariant: every twiddle of the top `h` layers lies in the subfield.
        // Checked against the subspace-polynomial definition, not against the split.
        const LOG_N: usize = 12;
        let head = closed_layers(BinaryField8::LOG_BITS);

        for layer in LOG_N - head..LOG_N {
            // Layer `layer` pairs rows `2^layer` apart, so it has this many blocks.
            for block in 0..1usize << (LOG_N - 1 - layer) {
                let t = subspace_polynomial::<BinaryField128>(layer, BinaryField128::ZERO)
                    + domain_point::<BinaryField128>(block << 1);
                assert!(
                    t.to_repr() <= u128::from(u8::MAX),
                    "layer={layer} block={block}"
                );
            }
        }

        // And the layer just below is where that stops, for this subfield at this height.
        let t = domain_point::<BinaryField128>(1 << (head - 1) << 1);
        assert!(
            t.to_repr() > u128::from(u8::MAX),
            "the head could be one layer deeper"
        );
    }

    #[test]
    fn encoding_pads_the_coefficients_at_the_narrow_width() {
        // Fixture state: a 2^4-row message at rate 1/4, so the codeword has 2^6 rows.
        for width in WIDTHS {
            for log_inv_rate in 0..=3 {
                let message = matrix::<BinaryField8>(4, width, 37);

                // The reference pads at the wide width and transforms the whole thing.
                let mut padded = widen::<_, BinaryField128>(&message.values);
                padded.resize(padded.len() << log_inv_rate, BinaryField128::ZERO);
                let expected = LchNtt::<BinaryField128>::default()
                    .ntt_batch(RowMajorMatrix::new(padded, width));

                let actual = subfield_encode_batch::<_, BinaryField128>(message, log_inv_rate);
                assert_eq!(actual, expected, "width={width} rate={log_inv_rate}");
            }
        }
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn encoding_rejects_a_codeword_length_past_the_address_space() {
        let message = matrix::<BinaryField8>(1, 1, 0);
        let _ = subfield_encode_batch::<_, BinaryField128>(message, usize::BITS as usize - 1);
    }

    #[test]
    #[should_panic = "domain exceeds field dimension"]
    fn a_domain_past_the_alphabet_dimension_is_refused() {
        // A 2^9-row domain asks for nine Cantor basis vectors, and a byte level has eight.
        let message = matrix::<BinaryField8>(9, 1, 0);
        let _ = subfield_ntt_batch::<BinaryField8, BinaryField8>(message);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Random heights and widths, across the three interesting subfield widths.
        #[test]
        fn random_subfield_messages_transform_like_the_widened_ones(
            log_n in 0usize..=9,
            width in 1usize..=5,
            seed in any::<u64>(),
        ) {
            check_against_the_wide_transform::<Gf2, BinaryField128>(log_n, width, seed);
            check_against_the_wide_transform::<BinaryField8, BinaryField128>(log_n, width, seed);
            check_against_the_wide_transform::<BinaryField16, BinaryField64>(log_n, width, seed);
        }
    }
}
