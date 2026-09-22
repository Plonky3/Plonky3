//! Sums of exterior products against one fixed list of left factors.

use p3_binary_field::TowerLevel;

use super::tensor::{BitTensor, BitTensorBuckets};

/// Left factors read against many blocks of right factors, each block summed on its own.
///
/// # Overview
///
/// The equality sweeps read one inner factor against every block of the packing:
///
/// ```text
///     sum_j left[j] ⊗ right[j]        the same left for every block
/// ```
///
/// The left factors are therefore known before any block is.
/// A layout prepared from them once serves every block.
///
/// # Algorithm
///
/// Over `F_2` the sum is a matrix product, the left coordinates transposed against the right:
///
/// ```text
///     m[u][v] = sum_j left_j[u] * right_j[v]
/// ```
///
/// Where the target multiplies `8 x 8` bit matrices, eight terms at a time run as one bit block
/// per pair of byte positions. The left side's blocks are the prepared layout.
///
/// Elsewhere, and for a shape the kernel does not take, the terms go into byte buckets.
pub(crate) struct LeftFactors<'a, EF> {
    /// The factors on the first tensor leg, one per term.
    left: &'a [EF],
    /// The same factors in the bit-block layout, where the target has the kernel.
    prepared: Option<kernel::Prepared>,
}

impl<'a, EF: TowerLevel> LeftFactors<'a, EF> {
    /// Prepare the left factors of every block sum to come.
    pub(crate) fn new(left: &'a [EF]) -> Self {
        Self {
            left,
            prepared: kernel::Prepared::new(left),
        }
    }

    /// `sum_j left[j] ⊗ right[j]`, over the terms both lists hold.
    ///
    /// The buckets are the fallback's scratch, allocated on first use and kept for the next block.
    pub(crate) fn sum(
        &self,
        right: &[EF],
        scratch: &mut Option<BitTensorBuckets<EF>>,
    ) -> BitTensor<EF> {
        if let Some(prepared) = &self.prepared
            && right.len() == self.left.len()
        {
            return prepared.sum(right);
        }
        let buckets = scratch.get_or_insert_with(BitTensorBuckets::zero);
        buckets.clear();
        for (&left, &right) in self.left.iter().zip(right) {
            buckets.add_exterior_product(left, right);
        }
        buckets.tensor()
    }
}

/// The bit-block product, on a target with `8 x 8` bit-matrix multiplication and wide registers.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod kernel {
    use alloc::vec::Vec;
    use core::arch::x86_64::*;

    use p3_binary_field::TowerLevel;

    use super::super::basis::Coefficients;
    use super::super::tensor::BitTensor;

    /// Terms one bit block spans, one per bit of a byte.
    const GROUP: usize = 8;

    /// Quadword whose byte `i` is `1 << i`, the input that transposes a matrix operand.
    const UNIT: u64 = 0x8040_2010_0804_0201;

    /// Quadword whose byte `i` is `1 << (7 - i)`, the transpose with its bytes reversed.
    const REVERSED: u64 = 0x0102_0408_1020_4080;

    /// The quadword gather of one half of eight elements, quadword `t` from element `t`.
    ///
    /// Two registers hold the eight elements in memory order, four to a register, so quadword
    /// `half` of element `t` is quadword `2t + half` of the pair. Elements `2k` and `2k + 1`
    /// then share the 128-bit lane `k`.
    const fn halves(half: usize) -> [i64; 8] {
        let mut quadwords = [0i64; 8];
        let mut t = 0;
        while t < GROUP {
            quadwords[t] = (2 * t + half) as i64;
            t += 1;
        }
        quadwords
    }

    /// The quadword gathers of the low and the high eight bytes.
    const HALVES: [[i64; 8]; 2] = [halves(0), halves(1)];

    /// The in-lane byte shuffle interleaving a lane's two quadwords, byte `j` of each together.
    ///
    /// Word `j` of lane `k` is then `(e_2k[j], e_2k+1[j])`, the two bytes the output keeps side
    /// by side.
    const fn interleave() -> [i64; 8] {
        let mut quadwords = [0i64; 8];
        let mut q = 0;
        while q < 8 {
            let mut word = 0u64;
            let mut byte = 0;
            while byte < 8 {
                // Byte `b` of the lane takes byte `b / 2` of its first or its second quadword.
                let b = 8 * (q % 2) + byte;
                word |= ((b / 2 + 8 * (b % 2)) as u64) << (8 * byte);
                byte += 1;
            }
            quadwords[q] = word as i64;
            q += 1;
        }
        quadwords
    }

    /// See [`interleave`].
    const INTERLEAVE: [i64; 8] = interleave();

    /// The word permute taking word `j` of lane `k` to word `k` of quadword `j`.
    ///
    /// Quadword `j` of the result is then byte `j` of the half of every element in order:
    ///
    /// ```text
    ///     quadword j  =  (e_0[8h + j], e_1[8h + j], .., e_7[8h + j])
    /// ```
    const fn transpose_words() -> [i64; 8] {
        let mut quadwords = [0i64; 8];
        let mut j = 0;
        while j < 8 {
            let mut word = 0u64;
            let mut k = 0;
            while k < 4 {
                word |= ((8 * k + j) as u64) << (16 * k);
                k += 1;
            }
            quadwords[j] = word as i64;
            j += 1;
        }
        quadwords
    }

    /// See [`transpose_words`].
    const TRANSPOSE_WORDS: [i64; 8] = transpose_words();

    /// The left factors, as the transposed `8 x 8` bit blocks of every group of eight.
    ///
    /// Block `p` of a group holds byte position `p` of its eight factors, transposed: byte `i`
    /// collects coordinate `8p + i` of every factor, one bit per factor, the last factor lowest.
    pub(super) struct Prepared {
        /// One entry per group of eight factors, one block per byte position.
        groups: Vec<[u64; 16]>,
    }

    impl Prepared {
        /// The blocks of `left`, when its level is 128 bits wide and it splits into groups.
        pub(super) fn new<EF: TowerLevel>(left: &[EF]) -> Option<Self> {
            if Coefficients::<EF>::DIMENSION != 128
                || left.is_empty()
                || !left.len().is_multiple_of(GROUP)
            {
                return None;
            }
            let (groups, _) = left.as_chunks::<GROUP>();
            let groups = groups
                .iter()
                // SAFETY: this module is compiled only where the build enables every target
                // feature the kernel names.
                .map(|group| unsafe { left_blocks(group) })
                .collect();
            Some(Self { groups })
        }

        /// `sum_j left[j] ⊗ right[j]`, for exactly one right factor per prepared left factor.
        pub(super) fn sum<EF: TowerLevel>(&self, right: &[EF]) -> BitTensor<EF> {
            assert_eq!(
                right.len(),
                self.groups.len() * GROUP,
                "one right factor per prepared left factor"
            );
            // SAFETY: this module is compiled only where the build enables every target feature
            // the kernel names.
            let rows = unsafe { rows(&self.groups, right) };
            BitTensor::try_from(
                rows.iter()
                    .map(|row| EF::from_le_byte_iter(row.to_le_bytes().into_iter()))
                    .collect::<Vec<_>>(),
            )
            .expect("one row per coordinate of a 128-bit level")
        }
    }

    /// The transposed blocks of one group of eight left factors.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    fn left_blocks<EF: TowerLevel>(group: &[EF; GROUP]) -> [u64; 16] {
        let registers = registers(group);
        let mut blocks = [0u64; 16];
        for (half, blocks) in blocks.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let transposed = _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(UNIT as i64),
                gathered(registers, half),
            );
            *blocks = lanes(transposed);
        }
        blocks
    }

    /// The rows of `sum_j left[j] ⊗ right[j]`, the left factors given as their blocks.
    ///
    /// # Algorithm
    ///
    /// With both sides of a group transposed, bit `t` of a byte is term `7 - t`. The affine
    /// instruction's parity of a matrix byte against an input byte is then the group's sum
    /// over its eight terms, one coefficient of the product:
    ///
    /// ```text
    ///     byte i, bit k of block (p, q)  =  sum_t left_t[8p + i] * right_t[8q + k]
    /// ```
    ///
    /// A register holds eight byte positions `q` of the right side, so one instruction per
    /// left block `p` covers half a byte row of the matrix. Each half runs over the whole
    /// block with its sixteen partial sums held in registers.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    fn rows<EF: TowerLevel>(groups: &[[u64; 16]], right: &[EF]) -> [u128; 128] {
        let mut rows = [0u128; 128];
        for half in 0..2 {
            let mut sums = [_mm512_setzero_si512(); 16];
            for (group, blocks) in right.as_chunks::<GROUP>().0.iter().zip(groups) {
                let columns = _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(REVERSED as i64),
                    gathered(registers(group), half),
                );
                for (sum, &block) in sums.iter_mut().zip(blocks) {
                    let product = _mm512_gf2p8affine_epi64_epi8::<0>(
                        _mm512_set1_epi64(block as i64),
                        columns,
                    );
                    *sum = _mm512_xor_si512(*sum, product);
                }
            }
            // Quadword `lane` of sum `p` holds byte `8 * half + lane` of rows `8p .. 8p + 8`.
            for (p, &sum) in sums.iter().enumerate() {
                for (lane, word) in lanes(sum).into_iter().enumerate() {
                    for i in 0..8 {
                        let byte = (word >> (8 * i)) & 0xFF;
                        rows[8 * p + i] |= u128::from(byte) << (8 * (8 * half + lane));
                    }
                }
            }
        }
        rows
    }

    /// Eight elements in memory order, four to a register.
    #[target_feature(enable = "avx512f")]
    fn registers<EF: TowerLevel>(group: &[EF; GROUP]) -> [__m512i; 2] {
        let words: [u128; GROUP] = core::array::from_fn(|t| {
            let mut bytes = [0u8; 16];
            for (slot, byte) in bytes.iter_mut().zip(group[t].into_bytes()) {
                *slot = byte;
            }
            u128::from_le_bytes(bytes)
        });
        [register(&words[..4]), register(&words[4..])]
    }

    /// Four words in memory order, the low quadword of each first.
    #[target_feature(enable = "avx512f")]
    fn register(words: &[u128]) -> __m512i {
        _mm512_set_epi64(
            (words[3] >> 64) as i64,
            words[3] as i64,
            (words[2] >> 64) as i64,
            words[2] as i64,
            (words[1] >> 64) as i64,
            words[1] as i64,
            (words[0] >> 64) as i64,
            words[0] as i64,
        )
    }

    /// One half of eight elements, one quadword per byte position.
    ///
    /// Byte `t` of quadword `j` is byte `8 * half + j` of element `t`: a byte transpose, taken
    /// as a quadword gather, an in-lane byte interleave and a word permute.
    #[target_feature(enable = "avx512f,avx512bw")]
    fn gathered(registers: [__m512i; 2], half: usize) -> __m512i {
        let halves = _mm512_permutex2var_epi64(registers[0], constant(HALVES[half]), registers[1]);
        let pairs = _mm512_shuffle_epi8(halves, constant(INTERLEAVE));
        _mm512_permutexvar_epi16(constant(TRANSPOSE_WORDS), pairs)
    }

    /// A register of eight quadwords, the first lowest.
    #[target_feature(enable = "avx512f")]
    fn constant(quadwords: [i64; 8]) -> __m512i {
        let [q0, q1, q2, q3, q4, q5, q6, q7] = quadwords;
        _mm512_set_epi64(q7, q6, q5, q4, q3, q2, q1, q0)
    }

    /// The eight quadwords of a register, lowest first.
    #[target_feature(enable = "avx512f")]
    fn lanes(value: __m512i) -> [u64; 8] {
        let low = _mm512_castsi512_si256(value);
        let high = _mm512_extracti64x4_epi64::<1>(value);
        [
            _mm256_extract_epi64::<0>(low) as u64,
            _mm256_extract_epi64::<1>(low) as u64,
            _mm256_extract_epi64::<2>(low) as u64,
            _mm256_extract_epi64::<3>(low) as u64,
            _mm256_extract_epi64::<0>(high) as u64,
            _mm256_extract_epi64::<1>(high) as u64,
            _mm256_extract_epi64::<2>(high) as u64,
            _mm256_extract_epi64::<3>(high) as u64,
        ]
    }
}

/// No bit-block product on this target, so nothing is ever prepared.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
mod kernel {
    use p3_binary_field::TowerLevel;

    use super::super::tensor::BitTensor;

    /// A layout no value can take, so every sum takes the buckets.
    pub(super) enum Prepared {}

    // The kernel's methods run intrinsics, so they cannot be `const`.
    // Keeping these the same stops constness from leaking into callers on some targets only.
    #[allow(clippy::missing_const_for_fn)]
    impl Prepared {
        /// Refuses every list, the target having no kernel to prepare for.
        pub(super) fn new<EF>(_left: &[EF]) -> Option<Self> {
            None
        }

        /// Never called: no value of this type exists.
        // The empty match is the proof of that, so the dereference it names never runs.
        #[allow(clippy::uninhabited_references)]
        pub(super) fn sum<EF: TowerLevel>(&self, _right: &[EF]) -> BitTensor<EF> {
            match *self {}
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField16, BinaryField128};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// The sum one exterior product at a time.
    fn reference<EF: TowerLevel>(left: &[EF], right: &[EF]) -> BitTensor<EF> {
        let mut tensor = BitTensor::zero();
        for (&left, &right) in left.iter().zip(right) {
            tensor.add_exterior_product(left, right);
        }
        tensor
    }

    fn random<EF: TowerLevel>(rng: &mut SmallRng, len: usize) -> Vec<EF>
    where
        rand::distr::StandardUniform: rand::distr::Distribution<EF>,
    {
        (0..len).map(|_| rng.random()).collect()
    }

    #[test]
    fn a_block_sum_is_the_sum_of_its_exterior_products() {
        // Invariant: whichever route a shape takes, the sum is the term-by-term one.
        //
        // Lengths below, at and past one group, a group-aligned block and a ragged one.
        let mut rng = SmallRng::seed_from_u64(0x5EED_B10C);
        for len in [1, 7, 8, 16, 24, 1 << 10, (1 << 10) + 3] {
            let left = random::<BinaryField128>(&mut rng, len);
            let factors = LeftFactors::new(&left);
            let mut scratch = None;
            for _ in 0..3 {
                let right = random::<BinaryField128>(&mut rng, len);
                assert_eq!(
                    factors.sum(&right, &mut scratch),
                    reference(&left, &right),
                    "{len} terms"
                );
            }
        }
    }

    #[test]
    fn a_block_sum_reads_every_coordinate_on_both_legs() {
        // Invariant: a single set coordinate on each side lands on its one matrix entry.
        //
        // Random factors set about half of each; this pins every row and column index.
        let mut rng = SmallRng::seed_from_u64(0xC00D);
        let len = 64;
        for (u, v) in [
            (0, 0),
            (7, 8),
            (8, 7),
            (63, 64),
            (64, 63),
            (127, 127),
            (1, 126),
        ] {
            let term = rng.random_range(0..len);
            let mut left = alloc::vec![BinaryField128::from_repr(0); len];
            let mut right = alloc::vec![BinaryField128::from_repr(0); len];
            left[term] = BinaryField128::from_repr(1u128 << u);
            right[term] = BinaryField128::from_repr(1u128 << v);
            let sum = LeftFactors::new(&left).sum(&right, &mut None);
            assert_eq!(
                sum,
                reference(&left, &right),
                "entry ({u}, {v}) at term {term}"
            );
        }
    }

    #[test]
    fn a_narrow_level_takes_the_buckets() {
        // The bit-block kernel is shaped for 128 coordinates; a narrower level falls back.
        let mut rng = SmallRng::seed_from_u64(0x16B17);
        let left = random::<BinaryField16>(&mut rng, 64);
        let right = random::<BinaryField16>(&mut rng, 64);
        let factors = LeftFactors::new(&left);
        assert!(factors.prepared.is_none());
        assert_eq!(factors.sum(&right, &mut None), reference(&left, &right));
    }

    #[test]
    fn a_right_block_of_another_length_takes_the_buckets() {
        // The prepared layout covers exactly the left factors, so a shorter block is summed
        // over the terms both hold, as the buckets do.
        let mut rng = SmallRng::seed_from_u64(0x5407);
        let left = random::<BinaryField128>(&mut rng, 32);
        let right = random::<BinaryField128>(&mut rng, 24);
        assert_eq!(
            LeftFactors::new(&left).sum(&right, &mut None),
            reference(&left, &right)
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn the_kernel_target_prepares_a_grouped_wide_level() {
        // On a target with the kernel, the differential tests above run it and not the buckets.
        let mut rng = SmallRng::seed_from_u64(0x6F21);
        let left = random::<BinaryField128>(&mut rng, 64);
        assert!(LeftFactors::new(&left).prepared.is_some());
        assert!(LeftFactors::new(&left[..60]).prepared.is_none());
    }
}
