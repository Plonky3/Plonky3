use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_andnot_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_sll_epi64, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi64, _mm256_xor_si256,
};
use core::mem::transmute;

#[cfg(target_feature = "avx2")]
use p3_symmetric::{CryptographicPermutation, Permutation};

#[cfg(target_feature = "avx2")]
use crate::KeccakF;
use crate::batch::{self, Lanes, State};

/// Keccak states interleaved in one register, one per 64-bit lane.
const WIDTH: usize = 4;

/// Keccak states the packed permutation advances at once.
#[cfg(target_feature = "avx2")]
pub const VECTOR_LEN: usize = WIDTH;

const RC: [__m256i; 24] = unsafe {
    transmute([
        [1u64; 4],
        [0x8082u64; 4],
        [0x800000000000808au64; 4],
        [0x8000000080008000u64; 4],
        [0x808bu64; 4],
        [0x80000001u64; 4],
        [0x8000000080008081u64; 4],
        [0x8000000000008009u64; 4],
        [0x8au64; 4],
        [0x88u64; 4],
        [0x80008009u64; 4],
        [0x8000000au64; 4],
        [0x8000808bu64; 4],
        [0x800000000000008bu64; 4],
        [0x8000000000008089u64; 4],
        [0x8000000000008003u64; 4],
        [0x8000000000008002u64; 4],
        [0x8000000000000080u64; 4],
        [0x800au64; 4],
        [0x800000008000000au64; 4],
        [0x8000000080008081u64; 4],
        [0x8000000000008080u64; 4],
        [0x80000001u64; 4],
        [0x8000000080008008u64; 4],
    ])
};

// SAFETY (every intrinsic below): the permutation runs only on a CPU with AVX2.
#[inline(always)]
fn rol_1(a: __m256i) -> __m256i {
    unsafe {
        let shl = _mm256_add_epi64(a, a);
        let shr = _mm256_srli_epi64::<63>(a);
        _mm256_or_si256(shl, shr)
    }
}

const ROL_8_CTRL: __m256i = unsafe {
    transmute::<[u8; 32], _>([
        0o07, 0o00, 0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o17, 0o10, 0o11, 0o12, 0o13, 0o14, 0o15,
        0o16, 0o07, 0o00, 0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o17, 0o10, 0o11, 0o12, 0o13, 0o14,
        0o15, 0o16,
    ])
};

#[inline(always)]
fn rol_8(a: __m256i) -> __m256i {
    unsafe { _mm256_shuffle_epi8(a, ROL_8_CTRL) }
}

const ROL_56_CTRL: __m256i = unsafe {
    transmute::<[u8; 32], _>([
        0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o07, 0o00, 0o11, 0o12, 0o13, 0o14, 0o15, 0o16, 0o17,
        0o10, 0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o07, 0o00, 0o11, 0o12, 0o13, 0o14, 0o15, 0o16,
        0o17, 0o10,
    ])
};

#[inline(always)]
fn rol_56(a: __m256i) -> __m256i {
    unsafe { _mm256_shuffle_epi8(a, ROL_56_CTRL) }
}

/// Rotate every 64-bit lane left by `N`.
///
/// Byte-sized rotations take one shuffle, the others two shifts and an OR.
#[inline(always)]
fn rol<const N: i32>(a: __m256i) -> __m256i {
    match N {
        0 => a,
        1 => rol_1(a),
        8 => rol_8(a),
        56 => rol_56(a),
        _ => unsafe {
            let shl = _mm256_sll_epi64(a, _mm_cvtsi64_si128(N as i64));
            let shr = _mm256_srl_epi64(a, _mm_cvtsi64_si128(64 - N as i64));
            _mm256_or_si256(shl, shr)
        },
    }
}

/// Theta's column correction, then rho's rotation by `N`, on one lane.
#[inline(always)]
fn theta_rho<const N: i32>(lane: __m256i, d: __m256i) -> __m256i {
    rol::<N>(unsafe { _mm256_xor_si256(lane, d) })
}

/// Chi on one plane: `b[j] ^ (!b[j + 1] & b[j + 2])`.
#[inline(always)]
fn chi(b: [__m256i; 5]) -> [__m256i; 5] {
    let mut out = b;
    for j in 0..5 {
        out[j] =
            unsafe { _mm256_xor_si256(b[j], _mm256_andnot_si256(b[(j + 1) % 5], b[(j + 2) % 5])) };
    }
    out
}

/// One round of Keccak-f on a state held as `[y][x]`.
///
/// Each output plane runs theta, rho, pi and chi back to back.
///
/// So only one plane of intermediate values is live at a time, against the 16 registers available.
#[inline(always)]
fn round(rc: __m256i, a: [[__m256i; 5]; 5]) -> [[__m256i; 5]; 5] {
    // SAFETY (whole body): the permutation runs only on a CPU with AVX2.
    unsafe {
        // Theta: C[x] = XOR of column x, then D[x] = C[x - 1] ^ rol(C[x + 1], 1).
        let mut c = [_mm256_setzero_si256(); 5];
        for x in 0..5 {
            let rows_01 = _mm256_xor_si256(a[0][x], a[1][x]);
            let rows_23 = _mm256_xor_si256(a[2][x], a[3][x]);
            c[x] = _mm256_xor_si256(_mm256_xor_si256(rows_01, rows_23), a[4][x]);
        }
        let mut d = [_mm256_setzero_si256(); 5];
        for x in 0..5 {
            d[x] = _mm256_xor_si256(c[(x + 4) % 5], rol_1(c[(x + 1) % 5]));
        }

        // Pi: plane y takes lane (j, j + 3y mod 5) at position j.
        let mut plane_0 = chi([
            theta_rho::<0>(a[0][0], d[0]),
            theta_rho::<44>(a[1][1], d[1]),
            theta_rho::<43>(a[2][2], d[2]),
            theta_rho::<21>(a[3][3], d[3]),
            theta_rho::<14>(a[4][4], d[4]),
        ]);

        // Iota right after plane 0: the next round waits on this lane, and the CPU issues oldest first.
        plane_0[0] = _mm256_xor_si256(plane_0[0], rc);

        [
            plane_0,
            chi([
                theta_rho::<28>(a[0][3], d[3]),
                theta_rho::<20>(a[1][4], d[4]),
                theta_rho::<3>(a[2][0], d[0]),
                theta_rho::<45>(a[3][1], d[1]),
                theta_rho::<61>(a[4][2], d[2]),
            ]),
            chi([
                theta_rho::<1>(a[0][1], d[1]),
                theta_rho::<6>(a[1][2], d[2]),
                theta_rho::<25>(a[2][3], d[3]),
                theta_rho::<8>(a[3][4], d[4]),
                theta_rho::<18>(a[4][0], d[0]),
            ]),
            chi([
                theta_rho::<27>(a[0][4], d[4]),
                theta_rho::<36>(a[1][0], d[0]),
                theta_rho::<10>(a[2][1], d[1]),
                theta_rho::<15>(a[3][2], d[2]),
                theta_rho::<56>(a[4][3], d[3]),
            ]),
            chi([
                theta_rho::<62>(a[0][2], d[2]),
                theta_rho::<55>(a[1][3], d[3]),
                theta_rho::<39>(a[2][4], d[4]),
                theta_rho::<41>(a[3][0], d[0]),
                theta_rho::<2>(a[4][1], d[1]),
            ]),
        ]
    }
}

/// Permute four interleaved Keccak states.
///
/// # Safety
///
/// The running CPU has AVX2.
#[inline(always)]
unsafe fn keccak_perm(buf: &mut [[u64; WIDTH]; 25]) {
    let mut state: [[__m256i; 5]; 5] = unsafe { transmute(*buf) };
    for rc in RC {
        state = round(rc, state);
    }
    *buf = unsafe { transmute::<[[__m256i; 5]; 5], [[u64; WIDTH]; 25]>(state) };
}

/// Whether the running CPU has AVX2.
fn supported() -> bool {
    cpufeatures::new!(has_avx2, "avx2");
    has_avx2::get()
}

/// Transpose a 4 x 4 matrix of 64-bit words held one row per register.
///
/// ```text
///     t_0 = [ r_0[0] r_1[0] | r_0[2] r_1[2] ]      t_1 = [ r_0[1] r_1[1] | r_0[3] r_1[3] ]
///     t_2 = [ r_2[0] r_3[0] | r_2[2] r_3[2] ]      t_3 = [ r_2[1] r_3[1] | r_2[3] r_3[3] ]
///
///     column 0 = [ low half of t_0 | low half of t_2 ]      column 2: the high halves
///     column 1 = [ low half of t_1 | low half of t_3 ]      column 3: the high halves
/// ```
#[inline(always)]
fn transpose(r: [__m256i; 4]) -> [__m256i; 4] {
    unsafe {
        // Interleave 64-bit words of neighbouring rows.
        let t0 = _mm256_unpacklo_epi64(r[0], r[1]);
        let t1 = _mm256_unpackhi_epi64(r[0], r[1]);
        let t2 = _mm256_unpacklo_epi64(r[2], r[3]);
        let t3 = _mm256_unpackhi_epi64(r[2], r[3]);

        // Join the matching 128-bit halves.
        [
            _mm256_permute2x128_si256::<0x20>(t0, t2),
            _mm256_permute2x128_si256::<0x20>(t1, t3),
            _mm256_permute2x128_si256::<0x31>(t0, t2),
            _mm256_permute2x128_si256::<0x31>(t1, t3),
        ]
    }
}

/// The steps of the batched sponge on this backend.
///
/// Messages enter through register transposes instead of word-by-word loads.
struct Backend;

impl Lanes<WIDTH> for Backend {
    #[inline(always)]
    unsafe fn permute(state: &mut State<WIDTH>) {
        // SAFETY: the caller runs this on a CPU with AVX2.
        unsafe { keccak_perm(state) };
    }

    #[inline(always)]
    unsafe fn absorb_words(state: &mut [[u64; WIDTH]], lanes: &[&[u8]; WIDTH], offset: usize) {
        // Four words at a time: one 32-byte load per lane, then a transpose.
        let (groups, rest) = state.as_chunks_mut::<4>();
        for (g, group) in groups.iter_mut().enumerate() {
            let at = offset + 32 * g;
            let mut rows = [unsafe { _mm256_setzero_si256() }; 4];
            for (row, lane) in rows.iter_mut().zip(lanes) {
                let bytes: &[u8; 32] = lane[at..][..32].try_into().unwrap();
                // SAFETY: the load reads exactly the 32 bytes just bounds-checked.
                *row = unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) };
            }
            for (word, column) in group.iter_mut().zip(transpose(rows)) {
                let word = word.as_mut_ptr().cast();
                // SAFETY: a state word is 32 bytes, and unaligned access is allowed.
                unsafe {
                    let sum = _mm256_xor_si256(_mm256_loadu_si256(word), column);
                    _mm256_storeu_si256(word, sum);
                }
            }
        }

        // Fewer than four words left: one word at a time.
        batch::absorb_words(rest, lanes, offset + 32 * groups.len());
    }
}

batch::kernel!("AVX2", Backend, WIDTH, "avx2");

#[cfg(target_feature = "avx2")]
impl Permutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {
    fn permute_mut(&self, state: &mut [[u64; VECTOR_LEN]; 25]) {
        // SAFETY: the build enables AVX2.
        unsafe { keccak_perm(state) };
    }
}

#[cfg(target_feature = "avx2")]
impl CryptographicPermutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {}

#[cfg(test)]
mod tests {
    use tiny_keccak::keccakf;

    use super::*;

    const STATES: [[u64; 25]; 4] = [
        [
            0xc22c4c11dbedc46a,
            0x317f74268c4f5cd0,
            0x838719da5aa295b6,
            0x9e9b17211985a3ba,
            0x92927b963ce29d69,
            0xf9a7169e38cc7216,
            0x639a594d6fbfe341,
            0x2335ebd8d15777bd,
            0x44e1abc0d022823b,
            0xb3657f9d16b36c13,
            0x26d9217c32b3010a,
            0x6e73d6e9c7e5bcc8,
            0x400aa469d130a391,
            0x1aa7c8a2cb97188a,
            0xdc3084a09bd0a6e3,
            0xbcfe3b656841baea,
            0x325f41887c840166,
            0x844656e313674bfe,
            0xd63de8bad19d156c,
            0x49ef0ac0ab52e147,
            0x8b92ee811c654ca9,
            0x42a9310fedf09bda,
            0x182dbdac03a5358e,
            0x3b4692ce58af8cb5,
            0x534da610f01b8fb3,
        ],
        [
            0x1c322ff4aea07d26,
            0xbd67bde061c97612,
            0x517565bd02ab410a,
            0xb251273ddc12a725,
            0x24f0979fe4f4fedc,
            0xc32d063a64f0bf03,
            0xd33c6709a7b103d2,
            0xaf33a8224b5c8828,
            0x6544ca066e997f1c,
            0xd53ad41e39f06d68,
            0x67695f6fb71d77d9,
            0xd6378cf19ee510f2,
            0x49472ea57abcbd08,
            0xcf3739df1eefbbb4,
            0x0fac1bf30e8ef101,
            0x7ff04c9b90de0f27,
            0xf3d63ec0e64cb2ab,
            0x76388c05f377d4bd,
            0x7886dd8f5b14ef5b,
            0xb036d289ba24a513,
            0x011e8fd6be65a408,
            0x695e2d20848eec67,
            0x31f9e80c5f45f8ee,
            0xcdf873daf7a5fdeb,
            0xfe98ff5bf28d560a,
        ],
        [
            0xed7423c3e4cda469,
            0x8bbbe52577993e33,
            0x93182a78487f96db,
            0x3c5209456d78dc73,
            0x8b66bde37b967146,
            0x226ae6e2148314fc,
            0x302aed4f30cd2db9,
            0x621a7c7751302084,
            0x4054874254bc66cc,
            0xb513327739d3c109,
            0x0ae03189599dd81a,
            0x457e6f324e512ed9,
            0x3870ea63c7367728,
            0xb08c7c0e401d2623,
            0xa1316890c9bb3ac1,
            0x0a313e02f34f6c7e,
            0x7c1325754df4dbf5,
            0x287e3d88240bedd2,
            0xc7c0f3c5058290bb,
            0x39471c62d065a4d1,
            0x050d8ecb5c7911bf,
            0x7a6cd7ca757186a7,
            0xed14a51934a17895,
            0x8a75418d7ffb98dd,
            0x8096f8d803188d57,
        ],
        [
            0x118d693606b316cd,
            0xc2614f04c0dfca91,
            0x5eb3da95450a214f,
            0x193eb69a8198e724,
            0xc24dea1c58e5fa6d,
            0xcf8630adb771d47c,
            0xe612253b23ade1df,
            0x281b2b53f2d5fe61,
            0x9a3f8fb149d7c419,
            0x7ac1eeffbd426464,
            0xb1bdb03caa7b20a3,
            0x4e38a03b709d47d1,
            0x35cafd22e9a91879,
            0x26c9ae757c5b9f80,
            0x9113b092720d90db,
            0xeb504ed5104a0a09,
            0x1405c220c45ee2b1,
            0x8055d2c37b02472b,
            0x545eeff9734b4b99,
            0x79eb67721385dff8,
            0x4133f750b0446360,
            0x7167e6c1fd8ff59b,
            0x576436ac7d46936e,
            0x8db22fbb547d0826,
            0xa855d775d64f0110,
        ],
    ];

    fn our_res() -> [[u64; 25]; 4] {
        let mut packed_result = [[0; 4]; 25];
        for (i, packed_res) in packed_result.iter_mut().enumerate() {
            *packed_res = [STATES[0][i], STATES[1][i], STATES[2][i], STATES[3][i]];
        }

        // SAFETY: the test runs only on a CPU with this backend.
        unsafe { keccak_perm(&mut packed_result) };

        let mut result = [[0; 25]; 4];
        for (i, packed_res) in packed_result.iter_mut().enumerate() {
            result[0][i] = packed_res[0];
            result[1][i] = packed_res[1];
            result[2][i] = packed_res[2];
            result[3][i] = packed_res[3];
        }
        result
    }

    fn tiny_keccak_res() -> [[u64; 25]; 4] {
        let mut result = STATES;
        keccakf(&mut result[0]);
        keccakf(&mut result[1]);
        keccakf(&mut result[2]);
        keccakf(&mut result[3]);
        result
    }

    #[test]
    fn test_vs_tiny_keccak() {
        // Every x86-64 build compiles this backend, but only a CPU with AVX2 runs it.
        if !supported() {
            return;
        }

        let expected = tiny_keccak_res();
        let computed = our_res();
        assert_eq!(expected, computed);
    }
}

#[cfg(test)]
mod lane_tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;
    use tiny_keccak::keccakf;

    use super::*;

    fn check(states: &[[u64; 25]]) {
        // Pack state l into lane l of every word.
        let mut packed: [[u64; WIDTH]; 25] =
            core::array::from_fn(|word| core::array::from_fn(|lane| states[lane][word]));
        // SAFETY: the test returns early on a CPU without the feature.
        unsafe { keccak_perm(&mut packed) };

        // Invariant: lane l of the packed permutation is the scalar permutation of state l.
        for (lane, state) in states.iter().enumerate() {
            let mut expected = *state;
            keccakf(&mut expected);
            let computed: [u64; 25] = core::array::from_fn(|word| packed[word][lane]);
            assert_eq!(computed, expected, "lane {lane}");
        }
    }

    #[test]
    fn every_lane_permutes_like_the_scalar_code_on_edge_states() {
        if !supported() {
            return;
        }

        // Fixture state: every lane holds its own state, so a swap of any two lanes shows.
        //
        //     lane 0: all zeros
        //     lane 1: all ones
        //     lane l >= 2: a single 1 at word 7l mod 25, bit l
        //
        //     lane 2 -> word 14, bit 2      lane 3 -> word 21, bit 3      lane 4 -> word 3, bit 4
        let states: Vec<[u64; 25]> = (0..WIDTH)
            .map(|lane| match lane {
                0 => [0; 25],
                1 => [u64::MAX; 25],
                _ => {
                    let mut state = [0; 25];
                    state[(7 * lane) % 25] = 1 << lane;
                    state
                }
            })
            .collect();

        // Invariant: no two lanes share a state, so lane routing is observable.
        for (i, a) in states.iter().enumerate() {
            for b in &states[i + 1..] {
                assert_ne!(a, b);
            }
        }

        check(&states);
    }

    proptest! {
        #[test]
        fn every_lane_permutes_like_the_scalar_code(words in prop::collection::vec(any::<u64>(), 25 * WIDTH)) {
            if !supported() {
                return Ok(());
            }

            // Fixture state: independent random states, one per lane.
            let states: Vec<[u64; 25]> = words.chunks(25).map(|c| c.try_into().unwrap()).collect();
            check(&states);
        }
    }
}
