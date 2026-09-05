//! A byte-sampling Fiat–Shamir challenger for the binary tower fields.

use alloc::vec::Vec;
use core::iter::{repeat, repeat_with};
use core::marker::PhantomData;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger,
    GrindingChallenger, HashChallenger, ResamplingError,
};
use p3_field::BasedVectorSpace;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, MerkleCap};
use tracing::instrument;

use crate::tower::TowerLevel;

/// The number of transcript bytes consumed by one bitstring sample.
///
/// Fixing this at eight keeps the transcript independent of the target's pointer width.
const BITS_SAMPLE_BYTES: usize = 8;

/// The headroom, in bits, that a grinding request must leave below the width of the
/// candidate space `grind` actually searches.
///
/// `grind` enumerates candidates with a 64-bit counter, so that space is `min(F::bits(),
/// 64)` bits wide even for fields wider than `u64`. A candidate satisfies a `bits`-wide
/// check with probability `2^-bits`, so a search space of `w` bits offers `2^(w - bits)`
/// expected successes. Eight bits of headroom keep that expectation at `>= 256`, so an
/// exhaustive search that finds nothing is astronomically unlikely rather than merely
/// improbable.
const GRIND_MARGIN_BITS: usize = 8;

/// A challenger for a binary tower field, driven by a challenger over bytes.
///
/// A tower level is a whole number of bytes wide, and every bit pattern is an element.
///
/// A transcript over the field is therefore a transcript over its bytes.
/// Nothing is encoded, decoded or rejection-sampled in either direction.
///
/// **Observing**: absorbs the little-endian bytes of the element into the inner challenger.
///
/// **Sampling**: reads the little-endian bytes of the element from the inner challenger.
/// Since the field has exactly `2^(8·NUM_BYTES)` elements, uniform bytes give a uniform
/// field element.
///
/// **Proof of work**: unlike the field and bit sampling above, which work at every level,
/// grinding is only viable in the wider ones. [`GrindingChallenger::grind`] enumerates
/// candidates with a 64-bit counter and demands `GRIND_MARGIN_BITS` of headroom below that
/// space, which puts a ceiling of `min(F::bits(), 64) − GRIND_MARGIN_BITS` on the width of a
/// single request:
///
/// | witness field | largest supported `bits` |
/// | --- | --- |
/// | [`BinaryField2`](crate::BinaryField2), [`BinaryField4`](crate::BinaryField4), [`BinaryField8`](crate::BinaryField8) | none: every nonzero request panics |
/// | [`BinaryField16`](crate::BinaryField16) | 8 |
/// | [`BinaryField32`](crate::BinaryField32) | 24 |
/// | [`BinaryField64`](crate::BinaryField64) | 56 |
/// | [`BinaryField128`](crate::BinaryField128) | 56, bounded by the counter rather than the field |
///
/// So the levels below `GF(2^32)` implement [`GrindingChallenger`] without being usable
/// proof-of-work witness types, and a 128-bit witness buys no more width than a 64-bit one.
#[derive(Clone, Debug)]
pub struct BinaryChallenger<F, Inner> {
    inner: Inner,
    _marker: PhantomData<F>,
}

impl<F, Inner> BinaryChallenger<F, Inner> {
    /// Wrap a challenger over bytes.
    pub const fn new(inner: Inner) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<F, H> BinaryChallenger<F, HashChallenger<u8, H, 32>>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    /// Wrap a fresh [`HashChallenger`] over the given hasher and initial transcript.
    pub const fn from_hasher(initial_state: Vec<u8>, hasher: H) -> Self {
        Self::new(HashChallenger::new(initial_state, hasher))
    }
}

impl<F: TowerLevel, Inner: CanObserve<u8>> CanObserve<F> for BinaryChallenger<F, Inner> {
    fn observe(&mut self, value: F) {
        for byte in value.into_bytes() {
            self.inner.observe(byte);
        }
    }
}

impl<F, const N: usize, Inner: CanObserve<u8>> CanObserve<Hash<F, u8, N>>
    for BinaryChallenger<F, Inner>
{
    fn observe(&mut self, values: Hash<F, u8, N>) {
        for value in values {
            self.inner.observe(value);
        }
    }
}

impl<F, const N: usize, Inner: CanObserve<u8>> CanObserve<&MerkleCap<F, [u8; N]>>
    for BinaryChallenger<F, Inner>
{
    fn observe(&mut self, cap: &MerkleCap<F, [u8; N]>) {
        for digest in cap.roots() {
            for value in digest {
                self.inner.observe(*value);
            }
        }
    }
}

impl<F, const N: usize, Inner: CanObserve<u8>> CanObserve<MerkleCap<F, [u8; N]>>
    for BinaryChallenger<F, Inner>
{
    fn observe(&mut self, cap: MerkleCap<F, [u8; N]>) {
        self.observe(&cap);
    }
}

impl<F, EF, Inner> CanSample<EF> for BinaryChallenger<F, Inner>
where
    F: TowerLevel,
    EF: BasedVectorSpace<F>,
    Inner: CanSample<u8>,
{
    fn sample(&mut self) -> EF {
        let inner = &mut self.inner;
        // Every bit pattern is a field element, so reading `NUM_BYTES` uniform bytes per
        // basis coefficient samples uniformly from `EF`.
        EF::from_basis_coefficients_fn(|_| F::from_le_byte_iter(repeat_with(|| inner.sample())))
    }
}

impl<F, Inner> CanSampleBits<usize> for BinaryChallenger<F, Inner>
where
    F: TowerLevel,
    Inner: CanSample<u8>,
{
    /// Sample a bitstring by masking bytes taken from the inner stream.
    ///
    /// A challenger that bit-decomposes a sampled field element can only extract as many
    /// bits as that element carries, and so has to bound the request by the field order.
    /// This one instead reads its entropy straight off the inner byte stream, so how many
    /// bits are available per call does not depend on `F`'s width at all: a challenger over
    /// a narrow level can still draw a query index far wider than one of its own elements.
    ///
    /// The one real bound is that the result must fit in a `usize`, which also keeps it
    /// within the `u64` the bytes are read into.
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(
            bits < (usize::BITS as usize),
            "requested bit count must fit within a usize"
        );
        let bytes: [u8; BITS_SAMPLE_BYTES] = self.inner.sample_array();
        let rand_u64 = u64::from_le_bytes(bytes);
        (rand_u64 & ((1u64 << bits) - 1)) as usize
    }
}

impl<F, Inner> CanSampleUniformBits<F> for BinaryChallenger<F, Inner>
where
    F: TowerLevel,
    Inner: CanSample<u8>,
{
    /// Sample uniform bits by masking bytes from the inner stream.
    ///
    /// # Overview
    ///
    /// The inner stream emits cryptographic-hash bytes uniform on `[0, 2^8)`.
    ///
    /// Reading 8 bytes as a 64-bit integer and masking the low `bits` is exactly uniform
    /// on `[0, 2^bits)`.
    ///
    /// No field-element decomposition occurs, so no rejection band exists. The const
    /// generic is therefore inert: this function never errors and never resamples.
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        // Byte-sourced sampling is uniform without rejection, so the result is always
        // valid and the error arm is unreachable.
        Ok(self.sample_bits(bits))
    }
}

/// The `index`-th grinding candidate: the element whose low bytes are those of `index`.
///
/// This is injective on the candidate range of every level: narrower levels stop before the
/// counter wraps their bit width, and wider ones are zero-extended.
#[inline]
fn candidate<F: TowerLevel>(index: u64) -> F {
    F::from_le_byte_iter(index.to_le_bytes().into_iter().chain(repeat(0)))
}

impl<F, Inner> GrindingChallenger for BinaryChallenger<F, Inner>
where
    F: TowerLevel,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
    type Witness = F;

    /// # Panics
    /// Panics unless `bits + GRIND_MARGIN_BITS <= min(F::bits(), 64)`. See the per-level
    /// ceiling tabulated on [`BinaryChallenger`].
    #[instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return F::ZERO;
        }

        assert!(bits < (usize::BITS as usize));
        assert!(
            bits + GRIND_MARGIN_BITS <= F::bits().min(u64::BITS as usize),
            "requested bit count leaves too small a margin against the witness field's size"
        );

        // Candidates are enumerated by a 64-bit counter, which exhausts the field only for
        // the levels narrower than `GF(2^64)`.
        let num_candidates = if F::bits() >= u64::BITS as usize {
            u64::MAX
        } else {
            1u64 << F::bits()
        };

        let witness = (0..num_candidates)
            .into_par_iter()
            .map(candidate)
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

impl<F, Inner> FieldChallenger<F> for BinaryChallenger<F, Inner>
where
    F: TowerLevel,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_challenger::{
        CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger,
        GrindingChallenger, HashChallenger,
    };
    use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, RawDataSerializable};
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::{Hash, MerkleCap};

    use crate::challenger::{BITS_SAMPLE_BYTES, BinaryChallenger};
    use crate::tower::TowerLevel;
    use crate::{BinaryField8, BinaryField32, BinaryField128, Gf2, Ghash128};

    type Inner = HashChallenger<u8, Keccak256Hash, 32>;

    /// A challenger over an empty transcript.
    fn mk<F>() -> BinaryChallenger<F, Inner> {
        BinaryChallenger::from_hasher(vec![], Keccak256Hash)
    }

    /// An inner challenger over the same empty transcript, to compare byte streams against.
    fn mk_inner() -> Inner {
        HashChallenger::new(vec![], Keccak256Hash)
    }

    /// A fixed nonzero element of `GF(2^128)`.
    fn a128() -> BinaryField128 {
        BinaryField128::from_le_bytes([
            0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56,
            0x34, 0x12,
        ])
    }

    #[test]
    fn the_polynomial_basis_field_shares_the_transcript_with_the_tower() {
        // Both representations are 16 bytes wide with every pattern a valid element, so a
        // transcript over one is byte-for-byte the transcript over the other.
        //
        //     observe tower element -> same bytes absorbed -> same bytes drawn
        //
        // What differs is only how those bytes are read as a field element.
        let element = a128();

        let mut tower = mk::<BinaryField128>();
        tower.observe(element);
        let drawn: BinaryField128 = tower.sample();

        let mut ghash = mk::<Ghash128>();
        ghash.observe(Ghash128::from_le_byte_iter(
            element.into_bytes().into_iter(),
        ));
        let mirrored: Ghash128 = ghash.sample();

        assert_eq!(
            drawn.into_bytes().into_iter().collect::<Vec<_>>(),
            mirrored.into_bytes().into_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn sampling_is_deterministic_and_observation_dependent() {
        // Identical (empty) histories produce identical challenges.
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        let (x, y): (BinaryField128, BinaryField128) = (a.sample(), b.sample());
        assert_eq!(x, y);

        // Identical observations produce identical challenges.
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        a.observe(a128());
        b.observe(a128());
        let (x, y): (BinaryField128, BinaryField128) = (a.sample(), b.sample());
        assert_eq!(x, y);

        // Different observations produce different challenges.
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        a.observe(BinaryField128::ONE);
        b.observe(BinaryField128::ZERO);
        let (x, y): (BinaryField128, BinaryField128) = (a.sample(), b.sample());
        assert_ne!(x, y);

        // Observing at all changes what is sampled next.
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        b.observe(BinaryField128::ZERO);
        let (x, y): (BinaryField128, BinaryField128) = (a.sample(), b.sample());
        assert_ne!(x, y);

        // The transcript advances: consecutive challenges differ, and the sequence is
        // reproducible from an identical history.
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        let xs: [BinaryField128; 4] = a.sample_array();
        let ys: [BinaryField128; 4] = b.sample_array();
        assert_eq!(xs, ys);
        assert_ne!(xs[0], xs[1]);
        assert_ne!(xs[1], xs[2]);
        assert_ne!(xs[2], xs[3]);
    }

    #[test]
    fn observing_an_element_absorbs_its_little_endian_bytes() {
        let a = a128();

        let mut challenger = mk::<BinaryField128>();
        challenger.observe(a);

        let mut inner = mk_inner();
        inner.observe_slice(&a.into_bytes().into_iter().collect::<Vec<_>>());

        // Compare a full digest worth of the resulting byte streams.
        for _ in 0..32 {
            let observed: u8 = challenger.inner.sample();
            let expected: u8 = inner.sample();
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn a_sampled_element_is_read_from_the_little_endian_byte_stream() {
        let mut challenger = mk::<BinaryField128>();
        let sampled: BinaryField128 = challenger.sample();

        let mut inner = mk_inner();
        let bytes: [u8; 16] = core::array::from_fn(|_| inner.sample());

        assert_eq!(sampled, BinaryField128::from_le_bytes(bytes));
    }

    #[test]
    fn a_sampled_element_of_a_narrow_level_reads_only_its_own_bytes() {
        let mut challenger = mk::<BinaryField32>();
        let sampled: BinaryField32 = challenger.sample();

        let mut inner = mk_inner();
        let bytes: [u8; 4] = core::array::from_fn(|_| inner.sample());

        assert_eq!(sampled, BinaryField32::from_le_bytes(bytes));
    }

    #[test]
    fn sampling_an_extension_element_reads_its_basis_coefficients_in_order() {
        let mut challenger = mk::<BinaryField8>();
        let sampled: BinaryField128 = challenger.sample();

        let mut coefficients = mk::<BinaryField8>();
        let coeffs: [BinaryField8; 16] = core::array::from_fn(|_| coefficients.sample());

        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField8>>::from_basis_coefficients_fn(|i| {
                coeffs[i]
            }),
            sampled
        );
    }

    #[test]
    fn observing_digests_matches_observing_their_bytes() {
        let digest = [1_u8, 2, 3, 4];

        let mut from_hash = mk::<BinaryField128>();
        from_hash.observe(Hash::<BinaryField128, u8, 4>::from(digest));

        let cap = MerkleCap::<BinaryField128, [u8; 4]>::new(vec![digest, [5, 6, 7, 8]]);
        let mut from_cap_ref = mk::<BinaryField128>();
        from_cap_ref.observe(&cap);
        let mut from_cap = mk::<BinaryField128>();
        from_cap.observe(cap);

        let mut hash_bytes = mk_inner();
        hash_bytes.observe_slice(&digest);
        let mut cap_bytes = mk_inner();
        cap_bytes.observe_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        for _ in 0..32 {
            let expected: u8 = hash_bytes.sample();
            assert_eq!(from_hash.inner.sample(), expected);

            let expected: u8 = cap_bytes.sample();
            assert_eq!(from_cap_ref.inner.sample(), expected);
            assert_eq!(from_cap.inner.sample(), expected);
        }
    }

    #[test]
    fn sampled_bits_fit_the_requested_width_and_are_reproducible() {
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        for bits in 0..32 {
            let value = a.sample_bits(bits);
            assert!(value < (1 << bits), "{value} does not fit in {bits} bits");
            assert_eq!(value, b.sample_bits(bits));
        }

        // The bit stream depends on the transcript.
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        a.observe(BinaryField128::ONE);
        assert_ne!(a.sample_bits(31), b.sample_bits(31));
    }

    #[test]
    fn uniform_bits_agree_with_plain_bits_and_never_reject() {
        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        for bits in [0, 1, 7, 8, 16, 31] {
            assert_eq!(
                a.sample_uniform_bits::<true>(bits).unwrap(),
                b.sample_bits(bits)
            );
        }

        let (mut a, mut b) = (mk::<BinaryField128>(), mk::<BinaryField128>());
        for bits in [0, 1, 7, 8, 16, 31] {
            assert_eq!(
                a.sample_uniform_bits::<false>(bits).unwrap(),
                b.sample_bits(bits)
            );
        }
    }

    /// The bits come from the inner byte stream, not from a sampled element, so a request
    /// wider than the field itself is well defined and must be served.
    #[test]
    fn sample_bits_is_not_limited_by_the_width_of_the_field() {
        // A `GF(2^8)` challenger sampling `GF(2^128)` challenges still has to produce
        // query indices of a realistic width.
        assert_eq!(<BinaryField8 as Field>::bits(), 8);
        let mut challenger = mk::<BinaryField8>();
        let sampled = challenger.sample_bits(24);
        assert!(sampled < (1 << 24));

        let mut inner = mk_inner();
        let bytes: [u8; BITS_SAMPLE_BYTES] = core::array::from_fn(|_| inner.sample());
        assert_eq!(sampled, (u64::from_le_bytes(bytes) & 0xff_ffff) as usize);

        // The uniform-bit entry point inherits the same reach.
        let mut challenger = mk::<BinaryField8>();
        assert_eq!(challenger.sample_uniform_bits::<true>(24).unwrap(), sampled);
        let mut challenger = mk::<BinaryField8>();
        assert_eq!(
            challenger.sample_uniform_bits::<false>(24).unwrap(),
            sampled
        );

        // Right up to the widest request a `usize` can hold.
        let widest = usize::BITS as usize - 1;
        let mut challenger = mk::<BinaryField8>();
        assert!(challenger.sample_bits(widest) < (1 << widest));
    }

    #[test]
    #[should_panic = "requested bit count must fit within a usize"]
    fn sample_bits_rejects_a_request_wider_than_a_usize() {
        let mut challenger = mk::<BinaryField128>();
        let _ = challenger.sample_bits(usize::BITS as usize);
    }

    #[test]
    #[should_panic = "requested bit count leaves too small a margin"]
    fn grind_rejects_a_request_wider_than_the_field() {
        let mut challenger = mk::<BinaryField32>();
        let _ = challenger.grind(32);
    }

    #[test]
    #[should_panic = "requested bit count leaves too small a margin"]
    fn grind_rejects_a_request_inside_the_margin() {
        // Narrower than the field, but too close to it: only `2^7` expected witnesses.
        let mut challenger = mk::<BinaryField32>();
        let _ = challenger.grind(25);
    }

    /// An inner challenger that always samples zero bytes, so every candidate trivially
    /// satisfies `check_witness`. This lets a test drive `grind` at bit widths whose real
    /// exhaustive search would otherwise be intractably slow to actually run.
    #[derive(Clone)]
    struct AlwaysZeroInner;

    impl CanObserve<u8> for AlwaysZeroInner {
        fn observe(&mut self, _value: u8) {}
    }

    impl CanSample<u8> for AlwaysZeroInner {
        fn sample(&mut self) -> u8 {
            0
        }
    }

    #[test]
    #[should_panic = "requested bit count leaves too small a margin"]
    fn grind_rejects_a_request_the_u64_cap_no_longer_allows() {
        // `BinaryField128`'s candidates are enumerated by a 64-bit counter, so the real
        // search space is 64 bits wide, not the field's 128. Before accounting for that
        // cap, this request wrongly passed the margin check (`63 + 8 <= 128`) while the
        // real candidate space only offered `2^(64 - 63) = 2` expected witnesses.
        let mut challenger = mk::<BinaryField128>();
        let _ = challenger.grind(63);
    }

    #[test]
    fn grind_accepts_the_widest_request_the_u64_cap_allows() {
        // The u64-capped search space is 64 bits wide, so `64 - GRIND_MARGIN_BITS = 56` is
        // the widest request the margin should still allow on `BinaryField128`. An inner
        // challenger that always samples zero bytes makes every candidate an accepted
        // witness, so the search completes immediately regardless of how wide `bits` is,
        // letting this exercise the real `grind` path instead of just its assertions.
        let mut challenger: BinaryChallenger<BinaryField128, AlwaysZeroInner> =
            BinaryChallenger::new(AlwaysZeroInner);
        let witness = challenger.grind(56);
        assert!(challenger.check_witness(56, witness));
    }

    #[test]
    fn a_bitstring_sample_reads_a_fixed_number_of_bytes() {
        // The width of one bitstring draw must not depend on the target's pointer width,
        // or transcripts would not agree across targets.
        let mut challenger = mk::<BinaryField128>();
        let sampled = challenger.sample_bits(31);

        let mut inner = mk_inner();
        let bytes: [u8; BITS_SAMPLE_BYTES] = core::array::from_fn(|_| inner.sample());

        assert_eq!(BITS_SAMPLE_BYTES, 8);
        assert_eq!(
            sampled,
            (u64::from_le_bytes(bytes) & ((1 << 31) - 1)) as usize
        );

        // The next draw continues from where the first one stopped.
        let next = challenger.sample_bits(8);
        let bytes: [u8; BITS_SAMPLE_BYTES] = core::array::from_fn(|_| inner.sample());
        assert_eq!(next, (u64::from_le_bytes(bytes) & 0xff) as usize);
    }

    #[test]
    fn grind_satisfies_check() {
        let mut challenger = mk::<BinaryField128>();
        let witness = challenger.clone().grind(8);
        assert!(challenger.check_witness(8, witness));
    }

    #[test]
    fn grind_over_a_narrow_level_satisfies_check() {
        let mut challenger = mk::<BinaryField32>();
        let witness = challenger.clone().grind(8);
        assert!(challenger.check_witness(8, witness));
    }

    #[test]
    fn grind_of_zero_bits_is_free() {
        let mut challenger = mk::<BinaryField128>();
        let mut shadow = challenger.clone();

        let witness = challenger.grind(0);
        assert_eq!(witness, BinaryField128::ZERO);

        // The transcript must be untouched.
        for _ in 0..32 {
            let expected: u8 = shadow.inner.sample();
            assert_eq!(challenger.inner.sample(), expected);
        }
    }

    #[test]
    fn grind_of_zero_bits_is_free_on_a_level_narrower_than_the_margin() {
        // `Gf2`'s single bit is narrower than `GRIND_MARGIN_BITS`: the zero-bits short
        // circuit must run before the margin assertion, or this would panic.
        let mut challenger = mk::<Gf2>();
        let mut shadow = challenger.clone();

        let witness = challenger.grind(0);
        assert_eq!(witness, Gf2::ZERO);

        // The transcript must be untouched.
        for _ in 0..32 {
            let expected: u8 = shadow.inner.sample();
            assert_eq!(challenger.inner.sample(), expected);
        }
    }

    #[test]
    fn a_wrong_witness_fails_the_check() {
        let mut challenger = mk::<BinaryField128>();
        let witness = challenger.clone().grind(8);

        // Flipping a bit of the witness invalidates it with probability 1 - 2^-8; the
        // transcript is fixed, so this is a deterministic statement about this witness.
        let mut wrong = challenger.clone();
        assert!(!wrong.check_witness(8, witness + BinaryField128::ONE));
        assert!(challenger.check_witness(8, witness));
    }

    /// The generic [`FieldChallenger`] helpers go through the base-field basis.
    #[test]
    fn field_challenger_algebra_helpers_round_trip() {
        let mut challenger = mk::<BinaryField8>();
        challenger.observe_algebra_element(a128());
        let sampled: BinaryField128 = challenger.sample_algebra_element();

        let mut expected = mk::<BinaryField8>();
        expected
            .observe_slice(BasedVectorSpace::<BinaryField8>::as_basis_coefficients_slice(&a128()));
        let expected: BinaryField128 = expected.sample();

        assert_eq!(sampled, expected);
    }
}
