//! The BLAKE2s compression function of RFC 7693 section 3.2, on vectors of lanes.

use crate::batch::lanes::Word;

/// Bytes in one compression block.
pub(super) const BLOCK_BYTES: usize = 64;

/// Words in one compression block.
pub(super) const BLOCK_WORDS: usize = 16;

/// Words in a chaining value.
pub(super) const STATE_WORDS: usize = 8;

pub(super) use crate::params::PARAM_BLOCK_0;
use crate::params::{IV, SIGMA};

/// The chaining value a message starts from, in every lane.
///
/// Sequential mode leaves every parameter word but the first at zero.
#[inline(always)]
pub(super) fn initial_state<V: Word, const G: usize>(param_0: u32) -> [[V; STATE_WORDS]; G] {
    // h_0 = IV ^ parameter block.
    let mut iv = IV;
    iv[0] ^= param_0;
    [iv.map(V::splat); G]
}

/// Advance every lane of every group by one block.
///
/// All lanes share the byte counter and the final-block flag.
/// That holds because they hash the same block index of equal-length messages.
#[inline(always)]
pub(super) fn compress<V: Word, const G: usize>(
    h: &mut [[V; STATE_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
    counter: u64,
    last: bool,
) {
    let mut v = working_vector(h, counter, last);

    // Ten rounds, each a literal instantiation, so every schedule index is a constant.
    //
    // Constant indices keep the working vector in registers instead of memory.
    round::<0, V, G>(&mut v, m);
    round::<1, V, G>(&mut v, m);
    round::<2, V, G>(&mut v, m);
    round::<3, V, G>(&mut v, m);
    round::<4, V, G>(&mut v, m);
    round::<5, V, G>(&mut v, m);
    round::<6, V, G>(&mut v, m);
    round::<7, V, G>(&mut v, m);
    round::<8, V, G>(&mut v, m);
    round::<9, V, G>(&mut v, m);

    // Feed-forward: h'[i] = h[i] ^ v[i] ^ v[i + 8].
    for (h, v) in h.iter_mut().zip(&v) {
        for i in 0..STATE_WORDS {
            h[i] = h[i].xor3(v[i], v[i + 8]);
        }
    }
}

/// The chaining value, then the IV with the byte counter and the final-block flag folded in.
///
/// ```text
///     v[12] ^= t_0     low word of the byte counter
///     v[13] ^= t_1     high word of the byte counter
///     v[14] ^= f_0     all ones on the last block, zero otherwise
/// ```
#[inline(always)]
fn working_vector<V: Word, const G: usize>(
    h: &[[V; STATE_WORDS]; G],
    counter: u64,
    last: bool,
) -> [[V; BLOCK_WORDS]; G] {
    // The upper half is the same in every group.
    let flag = if last { u32::MAX } else { 0 };
    let upper = [
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ counter as u32,
        IV[5] ^ (counter >> 32) as u32,
        IV[6] ^ flag,
        IV[7],
    ]
    .map(V::splat);
    core::array::from_fn(|g| {
        core::array::from_fn(|i| {
            if i < STATE_WORDS {
                h[g][i]
            } else {
                upper[i - STATE_WORDS]
            }
        })
    })
}

/// One round on every group: four column mixes, then four diagonal mixes.
///
/// The groups run one after the other.
/// They share no data, so the pipeline overlaps one group's dependency chains with the next.
#[inline(always)]
fn round<const R: usize, V: Word, const G: usize>(
    v: &mut [[V; BLOCK_WORDS]; G],
    m: &[[V; BLOCK_WORDS]; G],
) {
    // The message schedule of this round, fixed at compile time.
    let s = const { SIGMA[R] };
    for (v, m) in v.iter_mut().zip(m) {
        // Columns.
        mix(v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
        mix(v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
        mix(v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
        mix(v, [3, 7, 11, 15], m[s[6]], m[s[7]]);

        // Diagonals.
        mix(v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
        mix(v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
        mix(v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
        mix(v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
    }
}

/// The mixing function G, with the rotation distances 16, 12, 8 and 7.
#[inline(always)]
fn mix<V: Word>(v: &mut [V; BLOCK_WORDS], [a, b, c, d]: [usize; 4], x: V, y: V) {
    // The message word is added first, so that addition sits off the dependency chain.
    v[a] = v[a].add(x).add(v[b]);
    v[d] = v[d].xor(v[a]).rotr_16();
    v[c] = v[c].add(v[d]);
    v[b] = v[b].xor(v[c]).rotr_12();
    v[a] = v[a].add(y).add(v[b]);
    v[d] = v[d].xor(v[a]).rotr_8();
    v[c] = v[c].add(v[d]);
    v[b] = v[b].xor(v[c]).rotr_7();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::lanes::{GROUPS, Vector, to_lanes};
    use crate::params::ROUNDS;

    /// RFC 7693 appendix B: the working vector of BLAKE2s-256("abc") before the first round.
    const ABC_V_INITIAL: [u32; BLOCK_WORDS] = [
        0x6B08E647, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19, 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527C, 0x9B05688C,
        0xE07C2654, 0x5BE0CD19,
    ];

    /// RFC 7693 appendix B: the working vector of BLAKE2s-256("abc") after each round.
    const ABC_V_AFTER_ROUND: [[u32; BLOCK_WORDS]; ROUNDS] = [
        [
            0x16A3242E, 0xD7B5E238, 0xCE8CE24B, 0x927AEDE1, 0xA7B430D9, 0x93A4A14E, 0xA44E7C31,
            0x41D4759B, 0x95BF33D3, 0x9A99C181, 0x608A3A6B, 0xB666383E, 0x7A8DD50F, 0xBE378ED7,
            0x353D1EE6, 0x3BB44C6B,
        ],
        [
            0x3AE30FE3, 0x0982A96B, 0xE88185B4, 0x3E339B16, 0xF24338CD, 0x0E66D326, 0xE005ED0C,
            0xD591A277, 0x180B1F3A, 0xFCF43914, 0x30DB62D6, 0x4847831C, 0x7F00C58E, 0xFB847886,
            0xC544E836, 0x524AB0E2,
        ],
        [
            0x7A3BE783, 0x997546C1, 0xD45246DF, 0xEDB5F821, 0x7F98A742, 0x10E864E2, 0xD4AB70D0,
            0xC63CB1AB, 0x6038DA9E, 0x414594B0, 0xF2C218B5, 0x8DA0DCB7, 0xD7CD7AF5, 0xAB4909DF,
            0x85031A52, 0xC4EDFC98,
        ],
        [
            0x2A8B8CB7, 0x1ACA82B2, 0x14045D7F, 0xCC7258ED, 0x383CF67C, 0xE090E7F9, 0x3025D276,
            0x57D04DE4, 0x994BACF0, 0xF0982759, 0xF17EE300, 0xD48FC2D5, 0xDC854C10, 0x523898A9,
            0xC03A0F89, 0x47D6CD88,
        ],
        [
            0xC4AA2DDB, 0x111343A3, 0xD54A700A, 0x574A00A9, 0x857D5A48, 0xB1E11989, 0x6F5C52DF,
            0xDD2C53A3, 0x678E5F8E, 0x9718D4E9, 0x622CB684, 0x92976076, 0x0E41A517, 0x359DC2BE,
            0x87A87DDD, 0x643F9CEC,
        ],
        [
            0x3453921C, 0xD7595EE1, 0x592E776D, 0x3ED6A974, 0x4D997CB3, 0xDE9212C3, 0x35ADF5C9,
            0x9916FD65, 0x96562E89, 0x4EAD0792, 0xEBFC2712, 0x2385F5B2, 0xF34600FB, 0xD7BC20FB,
            0xEB452A7B, 0xECE1AA40,
        ],
        [
            0xBE851B2D, 0xA85F6358, 0x81E6FC3B, 0x0BB28000, 0xFA55A33A, 0x87BE1FAD, 0x4119370F,
            0x1E2261AA, 0xA1318FD3, 0xF4329816, 0x071783C2, 0x6E536A8D, 0x9A81A601, 0xE7EC80F1,
            0xACC09948, 0xF849A584,
        ],
        [
            0x07E5B85A, 0x069CC164, 0xF9DE3141, 0xA56F4680, 0x9E440AD2, 0x9AB659EA, 0x3C84B971,
            0x21DBD9CF, 0x46699F8C, 0x765257EC, 0xAF1D998C, 0x75E4C3B6, 0x523878DC, 0x30715015,
            0x397FEE81, 0x4F1FA799,
        ],
        [
            0x435148C4, 0xA5AA2D11, 0x4B354173, 0xD543BC9E, 0xBDA2591C, 0xBF1D2569, 0x4FCB3120,
            0x707ADA48, 0x565B3FDE, 0x32C9C916, 0xEAF4A1AB, 0xB1018F28, 0x8078D978, 0x68ADE4B5,
            0x9778FDA3, 0x2863B92E,
        ],
        [
            0xD9C994AA, 0xCFEC3AA6, 0x700D0AB2, 0x2C38670E, 0xAF6A1F66, 0x1D023EF3, 0x1D9EC27D,
            0x945357A5, 0x3E9FFEBD, 0x969FE811, 0xEF485E21, 0xA632797A, 0xDEEF082E, 0xAF3D80E1,
            0x4E86829B, 0x4DEAFD3A,
        ],
    ];

    /// RFC 7693 appendix B: the chaining value of BLAKE2s-256("abc") after its one compression.
    const ABC_H: [u32; STATE_WORDS] = [
        0x8C5E8C50, 0xE2147C32, 0xA32BA7E1, 0x2F45EB4E, 0x208B4537, 0x293AD69E, 0x4C9B994D,
        0x82596786,
    ];

    /// Assert that every lane of every group holds the expected words.
    fn assert_words<const N: usize>(actual: &[[Vector; N]; GROUPS], expected: &[u32; N], at: &str) {
        for group in actual {
            for (word, (&vector, &value)) in group.iter().zip(expected).enumerate() {
                let lanes = to_lanes(vector);
                assert!(
                    lanes.iter().all(|&lane| lane == value),
                    "word {word} {at}: expected {value:08X}, got {lanes:08X?}"
                );
            }
        }
    }

    #[test]
    fn abc_follows_the_rfc_7693_trace_round_by_round() {
        // "abc" is one final block of 3 bytes: m[0] = 0x00636261, every other word zero.
        let mut words = [0u32; BLOCK_WORDS];
        words[0] = 0x0063_6261;
        let m = [words.map(Vector::splat); GROUPS];
        let h = initial_state::<Vector, GROUPS>(PARAM_BLOCK_0);

        // The counter is 3 bytes, and the final-block flag is set.
        let mut v = working_vector(&h, 3, true);
        assert_words(&v, &ABC_V_INITIAL, "before round 1");

        // Each round lands on the working vector the appendix prints after it.
        macro_rules! rounds {
            ($($r:literal)*) => {$(
                round::<$r, Vector, GROUPS>(&mut v, &m);
                assert_words(&v, &ABC_V_AFTER_ROUND[$r], concat!("after round index ", $r));
            )*};
        }
        rounds!(0 1 2 3 4 5 6 7 8 9);

        // The full compression, feed-forward included, lands on the printed chaining value.
        let mut h = h;
        compress(&mut h, &m, 3, true);
        assert_words(&h, &ABC_H, "after the feed-forward");
    }
}
