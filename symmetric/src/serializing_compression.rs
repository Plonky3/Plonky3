use core::array;

use p3_field::{PackedField, PrimeCharacteristicRing, PrimeField32};

use crate::compression::PseudoCompressionFunction;

/// Lanes of a packed digest staged on the stack per call to the inner batched compression.
///
/// It covers the widest 32-bit packing, 16 lanes, in one call.
const PACKED_STAGED: usize = 16;

/// Scalar groups staged on the stack per call to the inner batched compression.
///
/// The byte hashes in this repository batch up to 32 messages per call, so 64 keeps every call full.
///
/// The stack buffer holds `SCALAR_STAGED * N * 32` input bytes.
const SCALAR_STAGED: usize = 64;

/// Runs a compression over 32-byte digests on digests of eight 32-bit prime field elements.
///
/// This gives a byte hash such as SHA-256, BLAKE3 or Keccak the digest type of an arithmetic hash such as Poseidon2.
/// The two can then share one Merkle tree through a [`MixedCompression`](crate::MixedCompression), with no conversion step between levels.
///
/// # Encoding
///
/// ```text
///     input  [F; 8] x N  --to_unique_u32, little endian-->  [u8; 32] x N
///     inner  [u8; 32] x N  ------------------------------>  [u8; 32]
///     output [u8; 32]  --little-endian u32, reduced mod p-->  [F; 8]
/// ```
///
/// The input side uses the field's own hashing encoding, [`PrimeField32::to_unique_u32`], the same one `SerializingHasher` uses for leaves.
/// It is injective, so a collision of this function is a collision of the inner one or of the output reduction.
/// For a Montgomery field it is the internal representation, which skips a reduction per element.
/// A recursive verifier rebuilding these bytes multiplies each element by the Montgomery constant before decomposing it.
///
/// # Collision resistance
///
/// The reduction maps `2^32` words onto `p < 2^32` elements, so it is not injective.
/// A compression needs collision resistance, which the reduction keeps when `p` is close to `2^32`.
/// For a 31-bit prime such as BabyBear, KoalaBear or Mersenne-31, each element has two or three preimages.
/// The reduced element then keeps over 30.8 of its at most 31 bits of collision entropy.
/// A digest keeps over 246 bits, against 248 for eight uniform 31-bit elements, so the birthday bound stays above `2^123`.
/// A smaller prime would shrink the digest to about `8 * log2(p)` bits, so fields with `p <= 2^30` are rejected at compile time.
///
/// # Cost
///
/// The conversions add a few nanoseconds per compression, mostly in turning output words into field elements.
/// That is significant next to the fastest byte hashes, and small next to an arithmetic compression such as Poseidon2.
///
/// # Levels
///
/// The tree level reaches the inner compression unchanged, so the inner function can itself be a [`MixedCompression`](crate::MixedCompression) of byte hashes.
///
/// # Lanes
///
/// A scalar batch is serialized a stack buffer at a time and sent to the inner [`compress_many`] in full calls.
/// It reports the inner [`LANES`], so a tree takes its batched path.
/// A packed digest stages one group per lane and sends the lanes to the inner `compress_many` together, so a packed tree keeps the inner batching too.
///
/// [`compress_many`]: PseudoCompressionFunction::compress_many
/// [`LANES`]: PseudoCompressionFunction::LANES
#[derive(Copy, Clone, Debug)]
pub struct SerializingCompression<Inner> {
    inner: Inner,
}

impl<Inner> SerializingCompression<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }

    /// One inner compression, at a tree level when one is given.
    #[inline]
    fn inner_one<const N: usize>(&self, group: [[u8; 32]; N], level: Option<usize>) -> [u8; 32]
    where
        Inner: PseudoCompressionFunction<[u8; 32], N>,
    {
        level.map_or_else(
            || self.inner.compress(group),
            |level| self.inner.compress_at_level(group, level),
        )
    }

    /// One inner batched compression, at a tree level when one is given.
    #[inline]
    fn inner_many<const N: usize>(
        &self,
        groups: &[[[u8; 32]; N]],
        out: &mut [[u8; 32]],
        level: Option<usize>,
    ) where
        Inner: PseudoCompressionFunction<[u8; 32], N>,
    {
        match level {
            Some(level) => self.inner.compress_many_at_level(groups, out, level),
            None => self.inner.compress_many(groups, out),
        }
    }

    /// Compress one group of possibly packed digests, at a tree level when one is given.
    fn compress_group<PF, const N: usize>(
        &self,
        input: [[PF; 8]; N],
        level: Option<usize>,
    ) -> [PF; 8]
    where
        PF: PackedField,
        PF::Scalar: PrimeField32,
        Inner: PseudoCompressionFunction<[u8; 32], N>,
    {
        // A one-lane packing is a scalar digest, so nothing is staged.
        if PF::WIDTH == 1 {
            let bytes = input.map(|digest| to_bytes(&digest.map(|x| x.as_slice()[0])));
            let out = from_bytes::<PF::Scalar>(&self.inner_one(bytes, level));
            return out.map(PF::from);
        }

        // Each lane is one group, and the lanes go to the inner function a stack buffer at a time.
        let mut groups = [[[0; 32]; N]; PACKED_STAGED];
        let mut outs = [[0; 32]; PACKED_STAGED];
        let mut digests = [[PF::Scalar::ZERO; 8]; PACKED_STAGED];
        let mut result = [PF::ZERO; 8];
        for start in (0..PF::WIDTH).step_by(PACKED_STAGED) {
            let lanes = PACKED_STAGED.min(PF::WIDTH - start);
            for (lane, group) in groups[..lanes].iter_mut().enumerate() {
                for (child, bytes) in group.iter_mut().enumerate() {
                    *bytes = to_bytes(&array::from_fn(|i| {
                        input[child][i].as_slice()[start + lane]
                    }));
                }
            }
            self.inner_many(&groups[..lanes], &mut outs[..lanes], level);
            for (digest, out) in digests[..lanes].iter_mut().zip(&outs[..lanes]) {
                *digest = from_bytes(out);
            }
            for (i, word) in result.iter_mut().enumerate() {
                let lanes_out = word.as_slice_mut();
                for lane in 0..lanes {
                    lanes_out[start + lane] = digests[lane][i];
                }
            }
        }
        result
    }

    /// Compress a batch of groups, at a tree level when one is given.
    fn compress_groups<PF, const N: usize>(
        &self,
        inputs: &[[[PF; 8]; N]],
        out: &mut [[PF; 8]],
        level: Option<usize>,
    ) where
        PF: PackedField,
        PF::Scalar: PrimeField32,
        Inner: PseudoCompressionFunction<[u8; 32], N>,
    {
        // A ragged batch is a caller bug, so it fails loudly rather than dropping groups.
        assert_eq!(
            inputs.len(),
            out.len(),
            "group count ({}) must equal the output count ({})",
            inputs.len(),
            out.len()
        );

        // A packed group already batches its lanes.
        if PF::WIDTH > 1 {
            for (output, group) in out.iter_mut().zip(inputs) {
                *output = self.compress_group(*group, level);
            }
            return;
        }

        // Scalar groups are serialized a stack buffer at a time, one full inner call per buffer.
        let mut groups = [[[0; 32]; N]; SCALAR_STAGED];
        let mut outs = [[0; 32]; SCALAR_STAGED];
        for (inputs, out) in inputs
            .chunks(SCALAR_STAGED)
            .zip(out.chunks_mut(SCALAR_STAGED))
        {
            let count = inputs.len();
            for (bytes, group) in groups[..count].iter_mut().zip(inputs) {
                *bytes = array::from_fn(|child| {
                    to_bytes(&array::from_fn(|i| group[child][i].as_slice()[0]))
                });
            }
            self.inner_many(&groups[..count], &mut outs[..count], level);
            for (digest, bytes) in out.iter_mut().zip(&outs[..count]) {
                *digest = from_bytes::<PF::Scalar>(bytes).map(PF::from);
            }
        }
    }
}

/// The 32-byte encoding of one field digest.
#[inline]
fn to_bytes<F: PrimeField32>(digest: &[F; 8]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (chunk, x) in bytes.chunks_exact_mut(4).zip(digest) {
        chunk.copy_from_slice(&x.to_unique_u32().to_le_bytes());
    }
    bytes
}

/// Reads a 32-byte output as eight little-endian words, each reduced into the field.
#[inline]
fn from_bytes<F: PrimeField32>(bytes: &[u8; 32]) -> [F; 8] {
    const {
        assert!(
            F::ORDER_U32 > 1 << 30,
            "SerializingCompression needs a prime above 2^30 to keep a 32-byte digest's collision resistance"
        );
    }
    let (words, _) = bytes.as_chunks::<4>();
    array::from_fn(|i| F::from_u32(u32::from_le_bytes(words[i])))
}

impl<PF, Inner, const N: usize> PseudoCompressionFunction<[PF; 8], N>
    for SerializingCompression<Inner>
where
    PF: PackedField,
    PF::Scalar: PrimeField32,
    Inner: PseudoCompressionFunction<[u8; 32], N>,
{
    const LANES: usize = Inner::LANES;

    fn compress(&self, input: [[PF; 8]; N]) -> [PF; 8] {
        self.compress_group(input, None)
    }

    fn compress_many(&self, inputs: &[[[PF; 8]; N]], out: &mut [[PF; 8]]) {
        self.compress_groups(inputs, out, None);
    }

    fn compress_at_level(&self, input: [[PF; 8]; N], level: usize) -> [PF; 8] {
        self.compress_group(input, Some(level))
    }

    fn compress_many_at_level(&self, inputs: &[[[PF; 8]; N]], out: &mut [[PF; 8]], level: usize) {
        self.compress_groups(inputs, out, Some(level));
    }

    fn lanes_at_level(&self, level: usize) -> usize {
        self.inner.lanes_at_level(level)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PackedValue;
    use p3_koala_bear::KoalaBear;

    use super::*;

    type F = KoalaBear;
    type P = <F as p3_field::Field>::Packing;

    /// A toy byte compression: a byte-wise wrapping sum of the children, then a rotation so the output words differ from any one input word.
    ///
    /// It counts as a batch of four lanes.
    #[derive(Clone, Debug)]
    struct ByteMix;

    impl PseudoCompressionFunction<[u8; 32], 2> for ByteMix {
        const LANES: usize = 4;

        fn compress(&self, input: [[u8; 32]; 2]) -> [u8; 32] {
            let mut out: [u8; 32] =
                array::from_fn(|i| input[0][i].wrapping_mul(3).wrapping_add(input[1][i]));
            out.rotate_left(5);
            out
        }
    }

    fn digest(seed: u32) -> [F; 8] {
        array::from_fn(|i| F::from_u32(seed.wrapping_mul(2_654_435_761).wrapping_add(i as u32)))
    }

    #[test]
    fn scalar_matches_the_documented_encoding() {
        let c = SerializingCompression::new(ByteMix);
        let (a, b) = (digest(1), digest(2));

        let bytes = |d: [F; 8]| -> [u8; 32] {
            let words: Vec<u8> = d
                .iter()
                .flat_map(|x| x.to_unique_u32().to_le_bytes())
                .collect();
            words.try_into().unwrap()
        };
        let raw = ByteMix.compress([bytes(a), bytes(b)]);
        let expected: [F; 8] = array::from_fn(|i| {
            F::from_u32(u32::from_le_bytes(
                raw[4 * i..4 * i + 4].try_into().unwrap(),
            ))
        });

        assert_eq!(c.compress([a, b]), expected);
    }

    #[test]
    fn reduction_wraps_words_above_the_modulus() {
        // A word at or above p must land on its residue, not fail or saturate.
        let mut bytes = [0u8; 32];
        bytes[..4].copy_from_slice(&(F::ORDER_U32 + 5).to_le_bytes());
        bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        let out = from_bytes::<F>(&bytes);
        assert_eq!(out[0], F::from_u32(5));
        assert_eq!(out[1], F::from_u32(u32::MAX - F::ORDER_U32));
    }

    #[test]
    fn packed_lanes_match_scalar() {
        // This exercises the staging loop only with a SIMD packing, as on the AVX2, AVX-512 and NEON builds.
        // With one lane it compares the scalar path with itself.
        let c = SerializingCompression::new(ByteMix);
        let left: Vec<[F; 8]> = (0..P::WIDTH as u32).map(digest).collect();
        let right: Vec<[F; 8]> = (100..100 + P::WIDTH as u32).map(digest).collect();

        let pack = |ds: &[[F; 8]]| -> [P; 8] { array::from_fn(|i| P::from_fn(|lane| ds[lane][i])) };
        let packed: [P; 8] = c.compress([pack(&left), pack(&right)]);

        for lane in 0..P::WIDTH {
            let scalar: [F; 8] = c.compress([left[lane], right[lane]]);
            let got: [F; 8] = array::from_fn(|i| packed[i].as_slice()[lane]);
            assert_eq!(got, scalar, "lane {lane}");
        }
    }

    #[test]
    fn batched_scalar_matches_one_at_a_time() {
        let c = SerializingCompression::new(ByteMix);
        // More groups than two stack buffers hold, and not a multiple of one.
        let inputs: Vec<[[F; 8]; 2]> = (0..150).map(|i| [digest(i), digest(i + 1000)]).collect();
        let mut out = alloc::vec![[F::ZERO; 8]; inputs.len()];
        c.compress_many(&inputs, &mut out);
        for (group, got) in inputs.iter().zip(&out) {
            assert_eq!(*got, c.compress(*group));
        }
        assert_eq!(
            <SerializingCompression<ByteMix> as PseudoCompressionFunction<[F; 8], 2>>::LANES,
            4
        );
    }

    /// A second toy byte compression, distinct from `ByteMix`, that batches eight lanes.
    #[derive(Clone, Debug)]
    struct ByteXor;

    impl PseudoCompressionFunction<[u8; 32], 2> for ByteXor {
        const LANES: usize = 8;

        fn compress(&self, input: [[u8; 32]; 2]) -> [u8; 32] {
            let mut out: [u8; 32] = array::from_fn(|i| input[0][i] ^ input[1][i].rotate_left(3));
            out.reverse();
            out
        }
    }

    #[test]
    fn the_level_reaches_a_mixed_inner_compression() {
        // An inner mix of two byte hashes switches at level 2, behind one field-digest bridge.
        let c = SerializingCompression::new(crate::MixedCompression::new(ByteMix, ByteXor, 2));
        let below = SerializingCompression::new(ByteMix);
        let above = SerializingCompression::new(ByteXor);
        let group = [digest(7), digest(8)];
        for level in 0..5 {
            let expected: [F; 8] = if level < 2 {
                below.compress(group)
            } else {
                above.compress(group)
            };
            assert_eq!(c.compress_at_level(group, level), expected, "level {level}");

            let inputs: Vec<[[F; 8]; 2]> = (0..70).map(|i| [digest(i), digest(i + 500)]).collect();
            let mut got = alloc::vec![[F::ZERO; 8]; inputs.len()];
            c.compress_many_at_level(&inputs, &mut got, level);
            for (input, got) in inputs.iter().zip(&got) {
                let want: [F; 8] = if level < 2 {
                    below.compress(*input)
                } else {
                    above.compress(*input)
                };
                assert_eq!(*got, want, "batched, level {level}");
            }

            let lanes =
                <SerializingCompression<_> as PseudoCompressionFunction<[F; 8], 2>>::lanes_at_level(
                    &c, level,
                );
            assert_eq!(lanes, if level < 2 { 4 } else { 8 }, "lanes, level {level}");
        }

        // The packed path forwards the level too.
        let pack = |ds: &[[F; 8]]| -> [P; 8] { array::from_fn(|i| P::from_fn(|lane| ds[lane][i])) };
        let left: Vec<[F; 8]> = (0..P::WIDTH as u32).map(digest).collect();
        let right: Vec<[F; 8]> = (50..50 + P::WIDTH as u32).map(digest).collect();
        let packed: [P; 8] = c.compress_at_level([pack(&left), pack(&right)], 3);
        for lane in 0..P::WIDTH {
            let want: [F; 8] = above.compress([left[lane], right[lane]]);
            let got: [F; 8] = array::from_fn(|i| packed[i].as_slice()[lane]);
            assert_eq!(got, want, "packed lane {lane}");
        }
    }
}
