use core::arch::x86_64::{
    __m128i, _mm_add_epi64, _mm_andnot_si128, _mm_cvtsi64_si128, _mm_or_si128, _mm_setzero_si128,
    _mm_sll_epi64, _mm_srl_epi64, _mm_srli_epi64, _mm_xor_si128,
};
use core::mem::transmute;

use p3_symmetric::{CryptographicPermutation, Permutation};

use crate::KeccakF;
use crate::batch::{self, Lanes, State};

pub const VECTOR_LEN: usize = 2;

const RC: [__m128i; 24] = unsafe {
    transmute([
        [1u64; 2],
        [0x8082u64; 2],
        [0x800000000000808au64; 2],
        [0x8000000080008000u64; 2],
        [0x808bu64; 2],
        [0x80000001u64; 2],
        [0x8000000080008081u64; 2],
        [0x8000000000008009u64; 2],
        [0x8au64; 2],
        [0x88u64; 2],
        [0x80008009u64; 2],
        [0x8000000au64; 2],
        [0x8000808bu64; 2],
        [0x800000000000008bu64; 2],
        [0x8000000000008089u64; 2],
        [0x8000000000008003u64; 2],
        [0x8000000000008002u64; 2],
        [0x8000000000000080u64; 2],
        [0x800au64; 2],
        [0x800000008000000au64; 2],
        [0x8000000080008081u64; 2],
        [0x8000000000008080u64; 2],
        [0x80000001u64; 2],
        [0x8000000080008008u64; 2],
    ])
};

#[inline(always)]
fn rol_1(a: __m128i) -> __m128i {
    unsafe {
        let shl = _mm_add_epi64(a, a);
        let shr = _mm_srli_epi64::<63>(a);
        _mm_or_si128(shl, shr)
    }
}

#[cfg(target_feature = "ssse3")]
#[inline(always)]
fn rol_8(a: __m128i) -> __m128i {
    use core::arch::x86_64::_mm_shuffle_epi8;
    const ROL_8_CTRL: __m128i = unsafe {
        transmute::<[u8; 16], _>([
            0o07, 0o00, 0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o17, 0o10, 0o11, 0o12, 0o13, 0o14,
            0o15, 0o16,
        ])
    };
    unsafe { _mm_shuffle_epi8(a, ROL_8_CTRL) }
}

#[cfg(not(target_feature = "ssse3"))]
#[inline(always)]
fn rol_8(a: __m128i) -> __m128i {
    rol_shift::<8>(a)
}

#[cfg(target_feature = "ssse3")]
#[inline(always)]
fn rol_56(a: __m128i) -> __m128i {
    use core::arch::x86_64::_mm_shuffle_epi8;
    const ROL_56_CTRL: __m128i = unsafe {
        transmute::<[u8; 16], _>([
            0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o07, 0o00, 0o11, 0o12, 0o13, 0o14, 0o15, 0o16,
            0o17, 0o10,
        ])
    };
    unsafe { _mm_shuffle_epi8(a, ROL_56_CTRL) }
}

#[cfg(not(target_feature = "ssse3"))]
#[inline(always)]
fn rol_56(a: __m128i) -> __m128i {
    rol_shift::<56>(a)
}

/// Rotate every 64-bit lane left by `N`, with two shifts and an OR.
#[inline(always)]
fn rol_shift<const N: i32>(a: __m128i) -> __m128i {
    unsafe {
        let shl = _mm_sll_epi64(a, _mm_cvtsi64_si128(N as i64));
        let shr = _mm_srl_epi64(a, _mm_cvtsi64_si128(64 - N as i64));
        _mm_or_si128(shl, shr)
    }
}

/// Rotate every 64-bit lane left by `N`.
///
/// Byte-sized rotations take one shuffle when SSSE3 is enabled.
#[inline(always)]
fn rol<const N: i32>(a: __m128i) -> __m128i {
    match N {
        0 => a,
        1 => rol_1(a),
        8 => rol_8(a),
        56 => rol_56(a),
        _ => rol_shift::<N>(a),
    }
}

/// Theta's column correction, then rho's rotation by `N`, on one lane.
#[inline(always)]
fn theta_rho<const N: i32>(lane: __m128i, d: __m128i) -> __m128i {
    rol::<N>(unsafe { _mm_xor_si128(lane, d) })
}

/// Chi on one plane: `b[j] ^ (!b[j + 1] & b[j + 2])`.
#[inline(always)]
fn chi(b: [__m128i; 5]) -> [__m128i; 5] {
    let mut out = b;
    for j in 0..5 {
        out[j] = unsafe { _mm_xor_si128(b[j], _mm_andnot_si128(b[(j + 1) % 5], b[(j + 2) % 5])) };
    }
    out
}

/// One round of Keccak-f on a state held as `[y][x]`.
///
/// Each output plane runs theta, rho, pi and chi back to back.
///
/// So only one plane of intermediate values is live at a time, against the 16 registers available.
#[inline(always)]
fn round(rc: __m128i, a: [[__m128i; 5]; 5]) -> [[__m128i; 5]; 5] {
    // SAFETY (whole body): the permutation runs on every CPU of the target.
    unsafe {
        // Theta: C[x] = XOR of column x, then D[x] = C[x - 1] ^ rol(C[x + 1], 1).
        let mut c = [_mm_setzero_si128(); 5];
        for x in 0..5 {
            let rows_01 = _mm_xor_si128(a[0][x], a[1][x]);
            let rows_23 = _mm_xor_si128(a[2][x], a[3][x]);
            c[x] = _mm_xor_si128(_mm_xor_si128(rows_01, rows_23), a[4][x]);
        }
        let mut d = [_mm_setzero_si128(); 5];
        for x in 0..5 {
            d[x] = _mm_xor_si128(c[(x + 4) % 5], rol_1(c[(x + 1) % 5]));
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
        plane_0[0] = _mm_xor_si128(plane_0[0], rc);

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

fn keccak_perm(buf: &mut [[u64; VECTOR_LEN]; 25]) {
    let mut state: [[__m128i; 5]; 5] = unsafe { transmute(*buf) };
    for rc in RC {
        state = round(rc, state);
    }
    *buf = unsafe { transmute::<[[__m128i; 5]; 5], [[u64; VECTOR_LEN]; 25]>(state) };
}

/// Every CPU of a build that compiles this backend has it.
const fn supported() -> bool {
    true
}

/// The steps of the batched sponge on this backend.
struct Backend;

impl Lanes<VECTOR_LEN> for Backend {
    #[inline(always)]
    unsafe fn permute(state: &mut State<VECTOR_LEN>) {
        keccak_perm(state);
    }
}

batch::kernel!("SSE2", Backend, VECTOR_LEN);

impl Permutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {
    fn permute_mut(&self, state: &mut [[u64; VECTOR_LEN]; 25]) {
        keccak_perm(state);
    }
}

impl CryptographicPermutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {}

#[cfg(test)]
mod tests {
    use tiny_keccak::keccakf;

    use super::*;

    const STATES: [[u64; 25]; 2] = [
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
    ];

    fn our_res() -> [[u64; 25]; 2] {
        let mut packed_result = [[0; 2]; 25];
        for (i, packed_res) in packed_result.iter_mut().enumerate() {
            *packed_res = [STATES[0][i], STATES[1][i]];
        }

        keccak_perm(&mut packed_result);

        let mut result = [[0; 25]; 2];
        for (i, packed_res) in packed_result.iter().enumerate() {
            result[0][i] = packed_res[0];
            result[1][i] = packed_res[1];
        }
        result
    }

    fn tiny_keccak_res() -> [[u64; 25]; 2] {
        let mut result = STATES;
        keccakf(&mut result[0]);
        keccakf(&mut result[1]);
        result
    }

    #[test]
    fn test_vs_tiny_keccak() {
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
        let mut packed: [[u64; VECTOR_LEN]; 25] =
            core::array::from_fn(|word| core::array::from_fn(|lane| states[lane][word]));
        keccak_perm(&mut packed);

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
        let states: Vec<[u64; 25]> = (0..VECTOR_LEN)
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
        fn every_lane_permutes_like_the_scalar_code(words in prop::collection::vec(any::<u64>(), 25 * VECTOR_LEN)) {
            if !supported() {
                return Ok(());
            }

            // Fixture state: independent random states, one per lane.
            let states: Vec<[u64; 25]> = words.chunks(25).map(|c| c.try_into().unwrap()).collect();
            check(&states);
        }
    }
}
