use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_field::{
    BasedVectorSpace, PrimeField, PrimeField32, absorb_radix_bits, max_absorb_injective_limbs,
    reduce_packed, split_pf_to_field_order_limbs, squeeze_field_order_num_limbs,
};
use p3_symmetric::{CryptographicPermutation, Hash, MerkleCap};

use crate::{
    CanFinalizeDigest, CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger,
};

/// A challenger that samples in `F: PrimeField32` while the transcript sponge lives in `PF`.
///
/// Wraps [`DuplexChallenger<PF>`](DuplexChallenger): all permutations and `PF` rate state are
/// exactly those of `inner`. This type only adapts
///
/// - **`F` → `PF`**: pending scalars are packed with [`reduce_packed`] (radix
///   $2^{\texttt{absorb\_radix\_bits::<F>()}}$) into up to `RATE` `PF` rate slots, then
///   [`DuplexChallenger::absorb_rate_padded_with_tag`](DuplexChallenger::absorb_rate_padded_with_tag)
///   runs (zero-padded tail, length tag = number of `F`s absorbed).
/// - **`PF` → `F`**: after each duplex, each rate cell is split with
///   [`split_pf_to_field_order_limbs`] (base `|F|`, [`squeeze_field_order_num_limbs`] limbs per
///   cell) into a flat queue consumed by [`CanSample::sample`]. Each extracted limb is uniform
///   over the **entire** `F` domain (bias `< 1/|F|`). The inner `output_buffer` is then cleared
///   so the next empty batch triggers a new duplex like [`DuplexChallenger::sample`].
///
/// **`observe(Hash)` / `observe(MerkleCap)`** flush pending `F`s through that packed absorb, then
/// absorb digest words natively via the same `absorb_rate_padded_with_tag` (length tag = number of
/// `PF` words in the block)—no PF → `F` → repack detour.
#[derive(Clone, Debug)]
pub struct MultiField32Challenger<F, PF, P, const WIDTH: usize, const RATE: usize>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    /// The underlying `PF` duplex sponge.
    inner: DuplexChallenger<PF, P, WIDTH, RATE>,
    f_buffer: Vec<F>,
    /// Expanded `F` limbs from `inner.output_buffer` (same pop order as the pre-wrapper design).
    f_squeeze_buffer: Vec<F>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    /// Radix bit-width $b$ for packing observed `F` values via [`reduce_packed`]: the smallest
    /// `b` with `F::ORDER_U32 - 1 < 2^b` (see [`p3_field::absorb_radix_bits`]).
    #[inline]
    #[must_use]
    pub const fn absorb_radix_bits(&self) -> u32 {
        absorb_radix_bits::<F>()
    }

    /// Maximum number of `F` elements packed into a single `PF` rate slot injectively (see
    /// [`p3_field::max_absorb_injective_limbs`]). Pending scalars are absorbed in chunks of this
    /// size; at most `RATE` such packed words are written per duplex step.
    #[inline]
    #[must_use]
    pub fn absorb_num_f_elms(&self) -> usize {
        max_absorb_injective_limbs::<F, PF>()
    }

    /// Number of base-`|F|` limbs taken from each squeezed `PF` rate cell when refilling the
    /// `F` queue (see [`p3_field::squeeze_field_order_num_limbs`] and
    /// [`p3_field::split_pf_to_field_order_limbs`]). Chooses near-uniform limbs over `F` for
    /// uniform `PF`.
    #[inline]
    #[must_use]
    pub fn squeeze_num_f_elms(&self) -> usize {
        squeeze_field_order_num_limbs::<PF, F>()
    }

    /// Number of `F` challenges still queued from the current squeeze batch (after `sample` pops).
    #[inline]
    #[must_use]
    pub const fn pending_f_squeeze_len(&self) -> usize {
        self.f_squeeze_buffer.len()
    }

    pub fn new(permutation: P) -> Result<Self, String> {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }
        if RATE >= WIDTH {
            return Err(String::from("RATE must be less than WIDTH"));
        }
        // A full flush stamps up to limbs-per-slot * RATE scalars into a byte-sized length tag.
        // Past 255, lengths differing by 256 would share a tag and collide in the transcript.
        if max_absorb_injective_limbs::<F, PF>() * RATE > u8::MAX as usize {
            return Err(String::from(
                "absorb length tag must fit in a u8: max_absorb_injective_limbs * RATE must be at most 255",
            ));
        }

        Ok(Self {
            inner: DuplexChallenger::new(permutation),
            f_buffer: vec![],
            f_squeeze_buffer: vec![],
        })
    }

    fn flush_f_if_non_empty(&mut self) {
        if self.f_buffer.is_empty() {
            return;
        }
        let n_in = self.f_buffer.len();
        let absorb_n = self.absorb_num_f_elms();
        assert!(n_in <= absorb_n * RATE);
        let rb = self.absorb_radix_bits();
        let packed: Vec<PF> = self
            .f_buffer
            .chunks(absorb_n)
            .map(|chunk| reduce_packed(chunk, rb))
            .collect();
        // Invariant: the constructor bounds a full flush at 255 scalars, so this never truncates.
        let tag = u8::try_from(n_in).expect("absorb length tag must fit in a u8");
        self.inner.absorb_rate_padded_with_tag(&packed, tag);
        self.f_buffer.clear();
        self.f_squeeze_buffer.clear();
    }

    fn refill_f_squeeze_from_inner(&mut self) {
        self.f_squeeze_buffer.clear();
        let squeeze_n = self.squeeze_num_f_elms();
        for &pf in &self.inner.output_buffer {
            self.f_squeeze_buffer
                .extend(split_pf_to_field_order_limbs::<PF, F>(pf, squeeze_n));
        }
        // Match `DuplexChallenger` semantics: squeezing consumes the current rate row. Until these
        // `F` limbs are exhausted, `inner.output_buffer` must read as empty so the next `sample`
        // triggers a fresh duplex when needed.
        self.inner.output_buffer.clear();
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> FieldChallenger<F>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        self.inner.output_buffer.clear();
        self.f_squeeze_buffer.clear();
        self.f_buffer.push(value);
        if self.f_buffer.len() == self.absorb_num_f_elms() * RATE {
            self.flush_f_if_non_empty();
        }
    }
}

impl<F, PF, const N: usize, P, const WIDTH: usize, const RATE: usize> CanObserve<[F; N]>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, PF, const N: usize, P, const WIDTH: usize, const RATE: usize> CanObserve<Hash<F, PF, N>>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, values: Hash<F, PF, N>) {
        self.inner.output_buffer.clear();
        self.f_squeeze_buffer.clear();
        self.flush_f_if_non_empty();

        let words: &[PF; N] = values.as_ref();

        for chunk in words.chunks(RATE) {
            // Invariant: each block holds at most RATE words, bounded at 255 by the constructor.
            let tag = u8::try_from(chunk.len()).expect("absorb length tag must fit in a u8");
            self.inner.absorb_rate_padded_with_tag(chunk, tag);
            self.f_squeeze_buffer.clear();
        }
    }
}

impl<F, PF, const N: usize, P, const WIDTH: usize, const RATE: usize>
    CanObserve<&MerkleCap<F, [PF; N]>> for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, cap: &MerkleCap<F, [PF; N]>) {
        for digest in cap.roots() {
            self.observe(Hash::<F, PF, N>::from(*digest));
        }
    }
}

impl<F, PF, const N: usize, P, const WIDTH: usize, const RATE: usize>
    CanObserve<MerkleCap<F, [PF; N]>> for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, cap: MerkleCap<F, [PF; N]>) {
        self.observe(&cap);
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<Vec<F>>>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<F, EF, PF, P, const WIDTH: usize, const RATE: usize> CanSample<EF>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    EF: BasedVectorSpace<F>,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn sample(&mut self) -> EF {
        EF::from_basis_coefficients_fn(|_| {
            self.flush_f_if_non_empty();
            if self.f_squeeze_buffer.is_empty() {
                if !self.inner.input_buffer.is_empty() || self.inner.output_buffer.is_empty() {
                    self.inner.duplexing();
                }
                self.refill_f_squeeze_from_inner();
            }
            self.f_squeeze_buffer
                .pop()
                .expect("Output buffer should be non-empty")
        })
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> CanSampleBits<usize>
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    /// The sampled bits are not perfectly uniform, but we can bound the error: every sequence
    /// appears with probability 1/p-close to uniform (1/2^b).
    ///
    /// Proof:
    /// We denote p = F::ORDER_U32, and b = `bits`.
    /// If X follows a uniform distribution over F, if we consider the first b bits of X, each
    /// sequence appears either with probability P1 = ⌊p / 2^b⌋ / p or P2 = (1 + ⌊p / 2^b⌋) / p.
    /// We have 1/2^b - 1/p ≤ P1, P2 ≤ 1/2^b + 1/p
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        // Evaluate the bound in `u64` to keep the shift within its type width.
        // A `u32` shift by `bits >= 32` would wrap and zero the mask, accepting any witness.
        assert!(
            (1u64 << bits) < F::ORDER_U64,
            "requested bit count must fit within the field order"
        );
        let rand_f: F = self.sample();
        let rand_usize = rand_f.as_canonical_u32() as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    /// Build the proof-of-work acceptance predicate used by
    /// [`GrindingChallenger::grind`](crate::GrindingChallenger::grind).
    ///
    /// The predicate replays `check_witness` = `observe(witness)` + `sample_bits(bits)`
    /// on a stack copy of the sponge state:
    ///
    /// ```text
    ///     pending scalars | witness  ->  packed rate slots, zero tail, length tag
    ///                                ->  permute
    ///                                ->  last limb of last rate cell
    ///                                ->  accept iff low `bits` are zero
    /// ```
    ///
    /// Everything except the witness digit is candidate-independent and computed once.
    /// Each call then costs one state copy, one digit write, one permutation,
    /// and one limb split — no clone, no heap allocation.
    pub(crate) fn pow_check_fn(&self, bits: usize) -> impl Fn(u32) -> bool + Sync + '_ {
        let rb = self.absorb_radix_bits();
        let absorb_n = self.absorb_num_f_elms();
        let squeeze_n = self.squeeze_num_f_elms();

        // The witness joins the pending scalars at the next free position.
        let n_pending = self.f_buffer.len();
        // Invariant: the constructor bounds a full flush at 255 scalars, so this never truncates.
        let tag = u8::try_from(n_pending + 1).expect("absorb length tag must fit in a u8");

        // Packed-slot coordinates of the witness digit.
        let chunk_idx = n_pending / absorb_n;
        let pos_in_chunk = n_pending % absorb_n;

        // The packing is little-endian Horner: digit `j` of a chunk weighs `2^(rb * j)`.
        let shift = PF::from_u64(1u64 << rb).exp_u64(pos_in_chunk as u64);

        // Digits below the witness in its own chunk.
        let const_tail: PF = reduce_packed(&self.f_buffer[chunk_idx * absorb_n..], rb);

        // Candidate-independent pre-permutation state.
        // Mirrors `absorb_rate_padded_with_tag(packed_chunks, tag)` on the inner sponge.
        let base_state: [PF; WIDTH] = array::from_fn(|i| {
            if i < chunk_idx {
                // Full constant chunks fill the leading rate slots.
                reduce_packed(&self.f_buffer[i * absorb_n..(i + 1) * absorb_n], rb)
            } else if i < RATE {
                // The witness chunk (rebuilt per candidate) and the unused rate slots are zeroed.
                PF::ZERO
            } else if i == RATE {
                // The first capacity element carries the length tag.
                self.inner.sponge_state[RATE] + PF::from_u8(tag)
            } else {
                // The rest of the capacity carries forward unchanged.
                self.inner.sponge_state[i]
            }
        });

        // Accept when the low `bits` of the checked challenge are zero.
        let mask = (1u64 << bits) - 1;

        move |candidate| {
            // One stack copy, one digit write, one permutation per candidate.
            let mut state = base_state;
            state[chunk_idx] = const_tail + shift * PF::from_u32(candidate);
            self.inner.permutation.permute_mut(&mut state);

            // `sample_bits` pops the last limb split from the last rate cell.
            let limbs = split_pf_to_field_order_limbs::<PF, F>(state[RATE - 1], squeeze_n);
            (u64::from(limbs[squeeze_n - 1].as_canonical_u32()) & mask) == 0
        }
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> CanFinalizeDigest
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    type Digest = [PF; RATE];

    fn finalize(mut self) -> [PF; RATE] {
        // Match the previous single `duplexing()` in `finalize`: if there was pending `F`, the
        // absorb+permute happens in `flush_f_if_non_empty` only; otherwise run one empty absorb
        // round (permute), like `duplexing` with `n_in == 0`.
        let had_pending_f = !self.f_buffer.is_empty();
        self.flush_f_if_non_empty();
        if !had_pending_f {
            self.inner.duplexing();
        }
        self.inner.sponge_state[..RATE].try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{
        Field, PrimeCharacteristicRing, PrimeField, injective_pack_bits, split_pf_to_packed_limbs,
        squeeze_field_order_num_limbs,
    };
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use super::*;
    use crate::grinding_challenger::GrindingChallenger;

    const WIDTH: usize = 8;
    const RATE: usize = 4;

    type F = BabyBear;
    type PF = Goldilocks;

    #[derive(Clone)]
    struct TestPermutation;

    impl Permutation<[PF; WIDTH]> for TestPermutation {
        fn permute_mut(&self, input: &mut [PF; WIDTH]) {
            for (i, val) in input.iter_mut().enumerate() {
                *val = PF::from_u8((i + 1) as u8);
            }
        }
    }

    impl CryptographicPermutation<[PF; WIDTH]> for TestPermutation {}

    /// A permutation where each output depends on all inputs, suitable for
    /// tests that need to detect state changes (e.g. finalize).
    #[derive(Clone)]
    struct MixingPermutation;

    impl Permutation<[PF; WIDTH]> for MixingPermutation {
        fn permute_mut(&self, input: &mut [PF; WIDTH]) {
            let sum: PF = input.iter().copied().sum();
            for (i, val) in input.iter_mut().enumerate() {
                *val = sum + PF::from_u8((i + 1) as u8);
            }
        }
    }

    impl CryptographicPermutation<[PF; WIDTH]> for MixingPermutation {}

    /// A no-op permutation generic over the state width.
    /// Lets tests instantiate challengers at widths the fixed-width permutations cannot reach.
    #[derive(Clone)]
    struct WideIdentityPermutation;

    impl<const W: usize> Permutation<[PF; W]> for WideIdentityPermutation {
        fn permute_mut(&self, _input: &mut [PF; W]) {}
    }

    impl<const W: usize> CryptographicPermutation<[PF; W]> for WideIdentityPermutation {}

    #[test]
    fn test_new_rejects_length_tag_overflow() {
        // The capacity length tag is a single byte stamped per padded absorb.
        // A full flush absorbs up to limbs-per-slot * RATE scalars at once.
        //
        // Fixture state: BabyBear packs 2 limbs per Goldilocks rate slot.
        assert_eq!(max_absorb_injective_limbs::<F, PF>(), 2);

        // Mutation: push RATE past the byte boundary.
        //
        //     RATE = 128 → 2 * 128 = 256 > 255 → reject
        //     RATE = 127 → 2 * 127 = 254 ≤ 255 → accept
        let too_wide = MultiField32Challenger::<F, PF, _, 129, 128>::new(WideIdentityPermutation);
        assert_eq!(
            too_wide.err().as_deref(),
            Some(
                "absorb length tag must fit in a u8: max_absorb_injective_limbs * RATE must be at most 255"
            )
        );

        let in_range = MultiField32Challenger::<F, PF, _, 128, 127>::new(WideIdentityPermutation);
        assert!(in_range.is_ok());
    }

    #[test]
    fn test_packing() {
        let c = MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        assert_eq!(c.absorb_radix_bits(), 31);
        assert_eq!(c.absorb_num_f_elms(), 2);
        assert_eq!(c.squeeze_num_f_elms(), 1);
        assert_eq!(squeeze_field_order_num_limbs::<PF, F>(), 1);
    }

    #[test]
    fn test_output_buffer_excludes_capacity() {
        let permutation = TestPermutation;
        let mut challenger =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(permutation).unwrap();

        let squeeze_n = challenger.squeeze_num_f_elms();

        let _: F = challenger.sample();

        let expected_output_size = RATE * squeeze_n;

        assert_eq!(
            challenger.pending_f_squeeze_len(),
            expected_output_size - 1,
            "Pending F squeeze should be RATE * squeeze_num_f_elms minus one sample"
        );
        assert_eq!(
            challenger.inner.output_buffer.len(),
            0,
            "After refill, inner PF output buffer is drained like popped F outputs"
        );
    }

    #[test]
    fn test_finalize() {
        let new_chal =
            || MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();

        // Deterministic: same observations produce same digest.
        let mut c1 = new_chal();
        let mut c2 = new_chal();
        for i in 0..5u8 {
            c1.observe(F::from_u8(i));
            c2.observe(F::from_u8(i));
        }
        assert_eq!(c1.finalize(), c2.finalize());

        // Different observations produce different digests.
        let mut c1 = new_chal();
        let mut c2 = new_chal();
        for i in 0..5u8 {
            c1.observe(F::from_u8(i));
            c2.observe(F::from_u8(i + 1));
        }
        assert_ne!(c1.finalize(), c2.finalize());
    }

    /// Document how sampling interacts with finalize.
    ///
    /// Same principle as DuplexChallenger: sampling only pops from the
    /// output buffer without modifying sponge state. The digest changes
    /// when a sample triggers a new duplexing. Each duplexing produces
    /// `num_f_elms * RATE` output elements (here 1 * 4 = 4 BabyBear
    /// elements for Goldilocks/BabyBear), so the digest is stable within
    /// each batch of that many samples.
    #[test]
    fn test_finalize_sample_interaction() {
        let batch_size = {
            let c =
                MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
            c.squeeze_num_f_elms() * RATE
        };

        let digest = |n_samples: usize| {
            let mut c =
                MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
            for i in 0..3u8 {
                c.observe(F::from_u8(i));
            }
            for _ in 0..n_samples {
                let _: F = c.sample();
            }
            c.finalize()
        };

        // The first sample triggers duplexing (absorbs pending input),
        // so finalize's duplexing is an extra permutation — different digest.
        assert_ne!(digest(0), digest(1));

        // Samples within the same batch don't trigger another duplexing.
        assert_eq!(digest(1), digest(2));
        assert_eq!(digest(1), digest(batch_size));

        // Exhausting the output buffer triggers a fresh duplexing.
        assert_ne!(digest(batch_size), digest(batch_size + 1));

        // Stable within the second batch.
        assert_eq!(digest(batch_size + 1), digest(batch_size + 2));
    }

    #[test]
    fn test_partial_absorb_length_distinct_from_padded_equivalent() {
        let ne = {
            let c =
                MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
            c.absorb_num_f_elms()
        };
        assert_eq!(ne, 2);

        let mut a =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        a.observe(F::ONE);

        let mut b =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        b.observe(F::ONE);
        for _ in 1..ne {
            b.observe(F::ZERO);
        }

        assert_ne!(a.finalize(), b.finalize());
    }

    #[test]
    fn test_absorb_no_radix_overflow_collision() {
        let mut a =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        a.observe(F::from_u32(1 << 30));
        a.observe(F::ZERO);

        let mut b =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        b.observe(F::ZERO);
        b.observe(F::ONE);

        assert_ne!(a.finalize(), b.finalize());
    }

    #[test]
    fn test_duplexing_respects_rate() {
        let permutation = TestPermutation;
        let mut challenger =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(permutation).unwrap();

        let absorb_n = challenger.absorb_num_f_elms();

        for i in 0..(absorb_n * RATE) {
            challenger.observe(F::from_u8(i as u8));
        }

        assert_eq!(
            challenger.inner.output_buffer.len(),
            RATE,
            "After a full F batch flush, inner holds one rate row of PF elements"
        );
        assert_eq!(
            challenger.pending_f_squeeze_len(),
            0,
            "F limbs are produced on sample() via split_pf_to_packed_limbs, not on observe"
        );
    }

    #[test]
    fn test_squeeze_covers_full_f_range() {
        // With base-2^30, challenges are confined to [0, 2^30) ≈ 50% of BabyBear.
        // With base-|F|, the c0 limb = v mod p_BB is near-uniform over all of BabyBear.
        // Verify that values above 2^30 can appear as challenges by constructing a Goldilocks
        // rate output whose canonical form mod p_BB exceeds 2^30.
        //
        // injective_pack_bits::<BabyBear>() = 30, so [2^30, p_BB) was previously unreachable.
        use p3_field::split_pf_to_field_order_limbs;
        let pack_bits = injective_pack_bits::<F>();
        let threshold = 1u32 << pack_bits; // 2^30

        // Build a Goldilocks value v such that v mod p_BB > 2^30.
        // p_BB + 2^30 < Goldilocks::ORDER (since p_BB ≈ 2^30.9 and p_GL ≈ 2^64),
        // so v = p_BB + threshold + 1 is a valid small Goldilocks element.
        let v_raw = F::ORDER_U32 as u64 + threshold as u64 + 1;
        let pf_val = PF::from_u64(v_raw);
        let limbs = split_pf_to_field_order_limbs::<PF, F>(pf_val, 1);
        // c0 = v_raw mod p_BB = threshold + 1 (since v_raw = p_BB + threshold + 1 ≡ threshold + 1).
        assert_eq!(limbs[0].as_canonical_u32(), threshold + 1);
        assert!(
            limbs[0].as_canonical_u32() > threshold,
            "c0 must exceed the old base-2^30 ceiling"
        );
    }

    #[test]
    fn test_observe_hash_native_pf_high_bits_distinct() {
        use num_bigint::BigUint;
        use p3_bn254::Bn254;
        use p3_field::split_pf_to_packed_limbs;
        use p3_symmetric::Hash;

        type PF254 = Bn254;

        #[derive(Clone)]
        struct Bn254MixingPermutation;

        impl Permutation<[PF254; WIDTH]> for Bn254MixingPermutation {
            fn permute_mut(&self, input: &mut [PF254; WIDTH]) {
                let sum: PF254 = input.iter().copied().sum();
                for (i, val) in input.iter_mut().enumerate() {
                    *val = sum + PF254::from_u8((i + 1) as u8);
                }
            }
        }

        impl CryptographicPermutation<[PF254; WIDTH]> for Bn254MixingPermutation {}

        let pack_bits = injective_pack_bits::<F>();
        let observe_n = PF254::bits().div_ceil(pack_bits as usize);

        let a = PF254::from_biguint(BigUint::from(1u32)).unwrap();
        let b = PF254::from_biguint(BigUint::from(1u32) + (BigUint::from(1u32) << 200)).unwrap();
        assert_ne!(a, b);

        let digest = |h: PF254| {
            let mut c =
                MultiField32Challenger::<F, PF254, _, WIDTH, RATE>::new(Bn254MixingPermutation)
                    .unwrap();
            c.observe(Hash::<F, PF254, 1>::from([h]));
            c.finalize()
        };

        assert_ne!(digest(a), digest(b));

        let limbs_a = split_pf_to_packed_limbs::<PF254, F>(a, observe_n, pack_bits);
        let limbs_b = split_pf_to_packed_limbs::<PF254, F>(b, observe_n, pack_bits);
        assert_ne!(limbs_a, limbs_b);

        let d_a = a.as_canonical_biguint().to_u64_digits();
        let d_b = b.as_canonical_biguint().to_u64_digits();
        let take3 = |d: &[u64]| {
            let mut v = [0u64; 3];
            for (i, x) in d.iter().take(3).enumerate() {
                v[i] = *x;
            }
            v
        };
        assert_eq!(take3(&d_a), take3(&d_b));
    }

    #[test]
    fn test_observe_hash_native_vs_expanded_f_not_equal() {
        use p3_symmetric::Hash;

        let g = PF::from_u64(123456789);
        let mut native =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        native.observe(Hash::<F, PF, 1>::from([g]));

        let mut via_f =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        let pb = injective_pack_bits::<F>();
        let n = PF::bits().div_ceil(pb as usize);
        for f in split_pf_to_packed_limbs::<PF, F>(g, n, pb) {
            via_f.observe(f);
        }

        assert_ne!(native.finalize(), via_f.finalize());
    }

    #[test]
    fn test_inner_sponge_matches_manual_absorb_chain() {
        let mut m =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        for i in 0..8u8 {
            m.observe(F::from_u8(i));
        }
        let d_m = m.inner.sponge_state;

        let mut inner = DuplexChallenger::<PF, _, WIDTH, RATE>::new(MixingPermutation);
        let packed: Vec<PF> = (0..8)
            .step_by(2)
            .map(|j| {
                reduce_packed::<F, PF>(
                    &[F::from_u8(j), F::from_u8(j + 1)],
                    absorb_radix_bits::<F>(),
                )
            })
            .collect();
        inner.absorb_rate_padded_with_tag(&packed, 8);
        assert_eq!(d_m, inner.sponge_state);
    }

    #[test]
    fn test_grind_zero_bits_returns_zero() {
        // bits == 0: must short-circuit to ZERO without touching state.
        let mut challenger =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();

        // Snapshot to detect any transcript mutation.
        let before = challenger.clone();

        let witness = challenger.grind(0);

        assert_eq!(witness, F::ZERO);
        assert_eq!(challenger.inner.input_buffer, before.inner.input_buffer);
        assert_eq!(challenger.inner.output_buffer, before.inner.output_buffer);
        assert_eq!(challenger.inner.sponge_state, before.inner.sponge_state);
    }

    #[test]
    #[should_panic = "requested bit count must fit within the field order"]
    fn test_sample_bits_rejects_oversized_request() {
        // Sampled field is BabyBear, order ~2^30.9, so a 32-bit request must be rejected.
        // The bound is evaluated in u64, so the guard fires in every build profile.
        let mut challenger =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        let _ = challenger.sample_bits(32);
    }

    #[test]
    #[should_panic = "requested bit count must fit within the field order"]
    fn test_grind_rejects_oversized_request() {
        // Same guard, reached through the proof-of-work entry point.
        let mut challenger =
            MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
        let _ = challenger.grind(32);
    }

    #[test]
    fn test_grind_advances_state_like_direct_check() {
        // The fast grind precomputes the pre-permutation state once per search.
        // Its result and final transcript must match a direct `check_witness`.
        //
        // Pending counts cover every packing geometry (absorb_n = 2, RATE = 4):
        //
        //     0  ->  witness opens chunk 0          (chunk_idx 0, pos 0)
        //     1  ->  witness completes chunk 0      (chunk_idx 0, pos 1)
        //     4  ->  witness opens chunk 2          (chunk_idx 2, pos 0)
        //     7  ->  witness fills the whole batch  (flush fires inside observe)
        for pending in [0usize, 1, 4, 7] {
            let mut challenger =
                MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
            for i in 0..pending {
                challenger.observe(F::from_u8(i as u8));
            }

            // Shadow challenger consumes the witness through the plain verifier path.
            let mut direct = challenger.clone();

            let bits = 2;
            let witness = challenger.grind(bits);

            // The witness validates, and both transcripts land on the same state.
            assert!(direct.check_witness(bits, witness), "pending = {pending}");
            assert_eq!(challenger.inner.sponge_state, direct.inner.sponge_state);
            assert_eq!(challenger.inner.input_buffer, direct.inner.input_buffer);
            assert_eq!(challenger.inner.output_buffer, direct.inner.output_buffer);
            assert_eq!(challenger.f_buffer, direct.f_buffer);
            assert_eq!(challenger.f_squeeze_buffer, direct.f_squeeze_buffer);
        }
    }

    #[test]
    fn test_grind_advances_state_like_direct_check_bn254() {
        // Bn254 splits each squeezed cell into several BabyBear limbs,
        // so this pins the limb-extraction path the Goldilocks fixture cannot reach.
        use p3_bn254::Bn254;

        type PF254 = Bn254;

        #[derive(Clone)]
        struct Bn254MixingPermutation;

        impl Permutation<[PF254; WIDTH]> for Bn254MixingPermutation {
            fn permute_mut(&self, input: &mut [PF254; WIDTH]) {
                let sum: PF254 = input.iter().copied().sum();
                for (i, val) in input.iter_mut().enumerate() {
                    *val = sum + PF254::from_u8((i + 1) as u8);
                }
            }
        }

        impl CryptographicPermutation<[PF254; WIDTH]> for Bn254MixingPermutation {}

        let mut challenger =
            MultiField32Challenger::<F, PF254, _, WIDTH, RATE>::new(Bn254MixingPermutation)
                .unwrap();
        assert!(challenger.squeeze_num_f_elms() > 1);
        for i in 0..3u8 {
            challenger.observe(F::from_u8(i));
        }

        let mut direct = challenger.clone();

        let bits = 2;
        let witness = challenger.grind(bits);

        assert!(direct.check_witness(bits, witness));
        assert_eq!(challenger.inner.sponge_state, direct.inner.sponge_state);
        assert_eq!(challenger.f_buffer, direct.f_buffer);
        assert_eq!(challenger.f_squeeze_buffer, direct.f_squeeze_buffer);
    }

    proptest! {
        #[test]
        fn prop_pow_check_matches_check_witness(
            pending_values in prop::collection::vec(0u32..F::ORDER_U32, 0..8),
            candidate in 0u32..F::ORDER_U32,
            bits in 1usize..8,
        ) {
            // Specialization-vs-reference pin:
            // the precomputed-state predicate must agree with the plain
            // `observe + sample_bits` verifier path on every candidate.
            let mut challenger =
                MultiField32Challenger::<F, PF, _, WIDTH, RATE>::new(MixingPermutation).unwrap();
            for &v in &pending_values {
                challenger.observe(F::from_u32(v));
            }

            let fast = challenger.pow_check_fn(bits)(candidate);

            // Reference path on a clone; `candidate` is canonical by the range above.
            let witness = F::from_u32(candidate);
            let reference = challenger.clone().check_witness(bits, witness);

            prop_assert_eq!(fast, reference);
        }
    }
}
