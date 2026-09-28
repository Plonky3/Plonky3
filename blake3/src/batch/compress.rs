//! The BLAKE3 compression function, on vectors of lanes.

use super::lanes::Word;

/// Words in one message block.
pub(super) const BLOCK_WORDS: usize = 16;

/// Words in a chaining value.
pub(super) const STATE_WORDS: usize = 8;

/// Rounds in one compression.
const ROUNDS: usize = 7;

/// The initialization vector, the same eight words SHA-256 starts from.
///
/// It is the key of the plain hash mode.
pub(crate) const IV: [u32; STATE_WORDS] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Flag on the first block of a chunk.
pub(super) const CHUNK_START: u32 = 1 << 0;

/// Flag on the last block of a chunk.
pub(super) const CHUNK_END: u32 = 1 << 1;

/// Flag on a compression that merges two child chaining values.
pub(super) const PARENT: u32 = 1 << 2;

/// Flag on the one compression whose chaining value is the digest.
pub(super) const ROOT: u32 = 1 << 3;

/// Where message word `i` moves between two rounds.
const PERMUTATION: [usize; BLOCK_WORDS] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Which message word each G input reads, in every round.
///
/// Round `r` reads the block permuted `r` times.
///
/// - Round 0 reads word `i` at position `i`.
/// - Round `r + 1` reads `SCHEDULE[r][PERMUTATION[i]]` at position `i`.
const SCHEDULE: [[usize; BLOCK_WORDS]; ROUNDS] = {
    let mut schedule = [[0; BLOCK_WORDS]; ROUNDS];
    let mut i = 0;
    while i < BLOCK_WORDS {
        schedule[0][i] = i;
        i += 1;
    }
    let mut r = 1;
    while r < ROUNDS {
        let mut i = 0;
        while i < BLOCK_WORDS {
            schedule[r][i] = schedule[r - 1][PERMUTATION[i]];
            i += 1;
        }
        r += 1;
    }
    schedule
};

/// Advance every lane of every group by one block.
///
/// All lanes share the counter, the block length and the flags.
///
/// That holds because they hash the same block of equal-length messages.
///
/// The new chaining value is the first half of the compression output:
///
/// ```text
///     h'[i] = v[i] ^ v[i + 8]
/// ```
#[inline(always)]
pub(super) fn compress<V: Word, const G: usize>(
    h: &mut [[V; STATE_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
    counter: u64,
    block_len: u32,
    flags: u32,
) {
    // The second half of the working vector, the same in every lane.
    let params = [
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    if V::compress_scheduled(h, m, &params) {
        return;
    }
    let v = working_vector(h, &params);
    rounds(h, v, m);
}

/// Advance every lane of every group by one block, each lane with its own counter.
///
/// Lanes that hash different chunks of one message differ only in the counter.
///
/// - `counters[g][0]` holds the low counter word of every lane of group `g`.
/// - `counters[g][1]` holds the high word.
///
/// The block length and the flags stay shared.
#[inline(always)]
pub(super) fn compress_counters<V: Word, const G: usize>(
    h: &mut [[V; STATE_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
    counters: &[[V; 2]; G],
    block_len: u32,
    flags: u32,
) {
    // The counter slots are overwritten per lane, so they start at zero here.
    let params = [IV[0], IV[1], IV[2], IV[3], 0, 0, block_len, flags];
    if V::compress_scheduled_counters(h, m, &params, counters) {
        return;
    }
    let mut v = working_vector(h, &params);

    // Words 12 and 13 of the working vector are the counter, low word first.
    for (v, counter) in v.iter_mut().zip(counters) {
        v[12] = counter[0];
        v[13] = counter[1];
    }
    rounds(h, v, m);
}

/// Run the seven rounds on the working vector, then fold it into the chaining value.
#[inline(always)]
fn rounds<V: Word, const G: usize>(
    h: &mut [[V; STATE_WORDS]; G],
    mut v: [[V; BLOCK_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
) {
    // Seven literal rounds, so every schedule index is a constant.
    //
    // Constant indices keep the working vector in registers instead of memory.
    round::<0, V, G>(&mut v, m);
    round::<1, V, G>(&mut v, m);
    round::<2, V, G>(&mut v, m);
    round::<3, V, G>(&mut v, m);
    round::<4, V, G>(&mut v, m);
    round::<5, V, G>(&mut v, m);
    round::<6, V, G>(&mut v, m);

    for (h, v) in h.iter_mut().zip(&v) {
        for i in 0..STATE_WORDS {
            h[i] = v[i].xor(v[i + STATE_WORDS]);
        }
    }
}

/// The chaining value, then the parameters of this compression.
///
/// - `v[0..8]` is `h`.
/// - `v[8..12]` is `IV[0..4]`.
/// - `v[12..14]` is the counter, low word first.
/// - `v[14]` is the block length in bytes.
/// - `v[15]` is the flags.
#[inline(always)]
fn working_vector<V: Word, const G: usize>(
    h: &[[V; STATE_WORDS]; G],
    params: &[u32; STATE_WORDS],
) -> [[V; BLOCK_WORDS]; G] {
    let second = params.map(V::splat);
    core::array::from_fn(|g| {
        core::array::from_fn(|i| {
            if i < STATE_WORDS {
                h[g][i]
            } else {
                second[i - STATE_WORDS]
            }
        })
    })
}

/// One round on every group: four column mixes, then four diagonal mixes.
#[inline(always)]
fn round<const R: usize, V: Word, const G: usize>(
    v: &mut [[V; BLOCK_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
) {
    // The message schedule of this round, fixed at compile time.
    let s = const { SCHEDULE[R] };

    // Columns.
    mix(v, m, [0, 4, 8, 12], [s[0], s[1]]);
    mix(v, m, [1, 5, 9, 13], [s[2], s[3]]);
    mix(v, m, [2, 6, 10, 14], [s[4], s[5]]);
    mix(v, m, [3, 7, 11, 15], [s[6], s[7]]);

    // Diagonals.
    mix(v, m, [0, 5, 10, 15], [s[8], s[9]]);
    mix(v, m, [1, 6, 11, 12], [s[10], s[11]]);
    mix(v, m, [2, 7, 8, 13], [s[12], s[13]]);
    mix(v, m, [3, 4, 9, 14], [s[14], s[15]]);
}

/// The mixing function G on every group, with the rotation distances 16, 12, 8 and 7.
///
/// Each step runs across all groups before the next one.
///
/// Groups share no data, so their steps fill each other's latency.
#[inline(always)]
fn mix<V: Word, const G: usize>(
    v: &mut [[V; BLOCK_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
    [a, b, c, d]: [usize; 4],
    [x, y]: [usize; 2],
) {
    // The message word is added first, so that addition sits off the dependency chain.
    for (v, m) in v.iter_mut().zip(m) {
        v[a] = v[a].add(m[x]).add(v[b]);
    }
    for v in v.iter_mut() {
        v[d] = v[d].xor(v[a]).rotr_16();
    }
    for v in v.iter_mut() {
        v[c] = v[c].add(v[d]);
    }
    for v in v.iter_mut() {
        v[b] = v[b].xor(v[c]).rotr_12();
    }
    for (v, m) in v.iter_mut().zip(m) {
        v[a] = v[a].add(m[y]).add(v[b]);
    }
    for v in v.iter_mut() {
        v[d] = v[d].xor(v[a]).rotr_8();
    }
    for v in v.iter_mut() {
        v[c] = v[c].add(v[d]);
    }
    for v in v.iter_mut() {
        v[b] = v[b].xor(v[c]).rotr_7();
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::uint32x4_t;
    #[cfg(all(
        target_arch = "wasm32",
        any(target_feature = "simd128", feature = "wasm32-simd")
    ))]
    use core::arch::wasm32::v128;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx2")
    ))]
    use core::arch::x86_64::__m128i;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx512f")
    ))]
    use core::arch::x86_64::__m256i;
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    use core::arch::x86_64::__m512i;

    use proptest::prelude::*;

    use super::*;
    use crate::batch::lanes::{Backend, from_lanes, to_lanes};

    /// Lanes in the widest register of any backend.
    const MAX_WIDTH: usize = 16;

    /// Lane words of two groups, wide enough for any backend.
    type Words<const N: usize> = [[[u32; MAX_WIDTH]; N]; 2];

    /// Check that backend `V` compresses two groups as it compresses each group alone.
    fn check_pair<V: Backend<W>, const W: usize>(
        h: &Words<STATE_WORDS>,
        m: &Words<BLOCK_WORDS>,
        counter: u64,
        block_len: u32,
        flags: u32,
    ) -> Result<(), TestCaseError> {
        // The running CPU may lack this backend.
        if !V::supported() {
            return Ok(());
        }

        // Each backend reads the first `W` lanes of every word.
        let vector =
            |lanes: &[u32; MAX_WIDTH]| from_lanes::<V, W>(core::array::from_fn(|l| lanes[l]));
        let h: [[V; STATE_WORDS]; 2] = h.each_ref().map(|group| group.each_ref().map(vector));
        let m: [[V; BLOCK_WORDS]; 2] = m.each_ref().map(|group| group.each_ref().map(vector));

        // Two groups may take a hand-scheduled kernel, and one group never does.
        let mut pair = h;
        compress(&mut pair, &m, counter, block_len, flags);

        for g in 0..2 {
            let mut single = [h[g]];
            compress(&mut single, &[m[g]], counter, block_len, flags);
            prop_assert_eq!(
                single[0].map(to_lanes::<V, W>),
                pair[g].map(to_lanes::<V, W>),
                "group {}",
                g
            );
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn two_groups_match_one_group_at_a_time(
            h in prop::array::uniform2(prop::array::uniform8(any::<[u32; MAX_WIDTH]>())),
            m in prop::array::uniform2(prop::array::uniform16(any::<[u32; MAX_WIDTH]>())),
            counter in any::<u64>(),
            block_len in 0u32..=64,
            flags in any::<u8>(),
        ) {
            let flags = u32::from(flags);

            // Every backend this build compiles, each on the CPUs that have it.
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            check_pair::<__m512i, 16>(&h, &m, counter, block_len, flags)?;
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(target_feature = "avx512f")))]
            check_pair::<__m256i, 8>(&h, &m, counter, block_len, flags)?;
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(target_feature = "avx2")))]
            check_pair::<__m128i, 4>(&h, &m, counter, block_len, flags)?;
            #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
            check_pair::<uint32x4_t, 4>(&h, &m, counter, block_len, flags)?;
            #[cfg(all(
                target_arch = "wasm32",
                any(target_feature = "simd128", feature = "wasm32-simd")
            ))]
            check_pair::<v128, 4>(&h, &m, counter, block_len, flags)?;
            #[cfg(not(any(
                all(target_arch = "x86_64", target_feature = "sse2"),
                all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"),
                all(
                    target_arch = "wasm32",
                    any(target_feature = "simd128", feature = "wasm32-simd")
                )
            )))]
            check_pair::<u32, 1>(&h, &m, counter, block_len, flags)?;
        }
    }
}
