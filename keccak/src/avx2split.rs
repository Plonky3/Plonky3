use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_andnot_si256, _mm256_blend_epi32, _mm256_castpd_si256,
    _mm256_castsi256_pd, _mm256_or_si256, _mm256_shuffle_epi32, _mm256_shuffle_epi8,
    _mm256_shuffle_pd, _mm256_slli_epi64, _mm256_sllv_epi64, _mm256_srli_epi64, _mm256_srlv_epi64,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi64, _mm256_xor_si256,
};
use core::mem::transmute;

use p3_symmetric::{CryptographicPermutation, Permutation};

use crate::KeccakF;

pub const VECTOR_LEN: usize = 4;

const RC: [__m256i; 24] = unsafe {
    transmute::<[[[u64; 2]; 2]; 24], _>([
        [[1, 0]; 2],
        [[0x8082, 0]; 2],
        [[0x800000000000808a, 0]; 2],
        [[0x8000000080008000, 0]; 2],
        [[0x808b, 0]; 2],
        [[0x80000001, 0]; 2],
        [[0x8000000080008081, 0]; 2],
        [[0x8000000000008009, 0]; 2],
        [[0x8a, 0]; 2],
        [[0x88, 0]; 2],
        [[0x80008009, 0]; 2],
        [[0x8000000a, 0]; 2],
        [[0x8000808b, 0]; 2],
        [[0x800000000000008b, 0]; 2],
        [[0x8000000000008089, 0]; 2],
        [[0x8000000000008003, 0]; 2],
        [[0x8000000000008002, 0]; 2],
        [[0x8000000000000080, 0]; 2],
        [[0x800a, 0]; 2],
        [[0x800000008000000a, 0]; 2],
        [[0x8000000080008081, 0]; 2],
        [[0x8000000000008080, 0]; 2],
        [[0x80000001, 0]; 2],
        [[0x8000000080008008, 0]; 2],
    ])
};

#[inline(always)]
fn xor(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_xor_si256(a, b) }
}

#[inline(always)]
fn andn(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_andnot_si256(a, b) }
}

#[inline(always)]
fn rol_1(a: __m256i) -> __m256i {
    unsafe {
        let shl = _mm256_add_epi64(a, a);
        let shr = _mm256_srli_epi64::<63>(a);
        _mm256_or_si256(shl, shr)
    }
}

#[inline(always)]
fn rol_2(a: __m256i) -> __m256i {
    unsafe {
        let shl = _mm256_slli_epi64::<2>(a);
        let shr = _mm256_srli_epi64::<62>(a);
        _mm256_or_si256(shl, shr)
    }
}

#[inline(always)]
fn rol_8_56(a: __m256i) -> __m256i {
    unsafe {
        const CTRL: __m256i = unsafe {
            transmute::<[[u8; 16]; 2], _>(
                [[
                    0o07, 0o00, 0o01, 0o02, 0o03, 0o04, 0o05, 0o06, 0o11, 0o12, 0o13, 0o14, 0o15,
                    0o16, 0o17, 0o10,
                ]; 2],
            )
        };
        _mm256_shuffle_epi8(a, CTRL)
    }
}

#[inline(always)]
fn rol_var(a: __m256i, shl_amt: __m256i, shr_amt: __m256i) -> __m256i {
    unsafe {
        let shl = _mm256_sllv_epi64(a, shl_amt);
        let shr = _mm256_srlv_epi64(a, shr_amt);
        _mm256_or_si256(shl, shr)
    }
}

#[inline(always)]
fn rol_0_20(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[0u64, 20u64]; 2]),
            transmute([[64u64, 44u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_44_3(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[44u64, 3u64]; 2]),
            transmute([[20u64, 61u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_43_45(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[43u64, 45u64]; 2]),
            transmute([[21u64, 19u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_21_61(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[21u64, 61u64]; 2]),
            transmute([[43u64, 3u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_14_28(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[14u64, 28u64]; 2]),
            transmute([[50u64, 36u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_1_36(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[1u64, 36u64]; 2]),
            transmute([[63u64, 28u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_6_10(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[6u64, 10u64]; 2]),
            transmute([[58u64, 54u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_25_15(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[25u64, 15u64]; 2]),
            transmute([[39u64, 49u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_18_27(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[18u64, 27u64]; 2]),
            transmute([[46u64, 37u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_62_55(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[62u64, 55u64]; 2]),
            transmute([[2u64, 9u64]; 2]),
        )
    }
}

#[inline(always)]
fn rol_39_41(a: __m256i) -> __m256i {
    unsafe {
        rol_var(
            a,
            transmute([[39u64, 41u64]; 2]),
            transmute([[25u64, 23u64]; 2]),
        )
    }
}

#[inline(always)]
fn swap(a: __m256i) -> __m256i {
    unsafe { _mm256_shuffle_epi32::<0b01001110>(a) }
}

#[inline(always)]
fn dup_hi(a: __m256i) -> __m256i {
    unsafe { _mm256_unpackhi_epi64(a, a) }
}

#[inline(always)]
fn get_lo_lo(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_unpacklo_epi64(a, b) }
}

#[inline(always)]
fn get_lo_hi(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_blend_epi32::<0b11001100>(a, b) }
}

#[inline(always)]
fn get_hi_lo(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        _mm256_castpd_si256(_mm256_shuffle_pd::<0b0101>(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
        ))
    }
}

#[inline(always)]
fn get_hi_hi(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_unpackhi_epi64(a, b) }
}

const ZERO: __m256i = unsafe { transmute([0u64; 4]) };

#[derive(Copy, Clone)]
pub struct State {
    pub(self) a0b1: __m256i,
    pub(self) b0c1: __m256i,
    pub(self) c0d1: __m256i,
    pub(self) d0e1: __m256i,
    pub(self) e0a1: __m256i,
    pub(self) a2b3: __m256i,
    pub(self) b2c3: __m256i,
    pub(self) c2d3: __m256i,
    pub(self) d2e3: __m256i,
    pub(self) e2a3: __m256i,
    pub(self) a4b4: __m256i,
    pub(self) c4d4: __m256i,
    pub(self) e4zz: __m256i,
}

impl State {
    #[inline]
    pub const fn from_arrs(arrs: [[u64; 25]; 2]) -> State {
        unsafe {
            let a0b1 = transmute([arrs[0][0], arrs[0][6], arrs[1][0], arrs[1][6]]);
            let b0c1 = transmute([arrs[0][1], arrs[0][7], arrs[1][1], arrs[1][7]]);
            let c0d1 = transmute([arrs[0][2], arrs[0][8], arrs[1][2], arrs[1][8]]);
            let d0e1 = transmute([arrs[0][3], arrs[0][9], arrs[1][3], arrs[1][9]]);
            let e0a1 = transmute([arrs[0][4], arrs[0][5], arrs[1][4], arrs[1][5]]);
            let a2b3 = transmute([arrs[0][10], arrs[0][16], arrs[1][10], arrs[1][16]]);
            let b2c3 = transmute([arrs[0][11], arrs[0][17], arrs[1][11], arrs[1][17]]);
            let c2d3 = transmute([arrs[0][12], arrs[0][18], arrs[1][12], arrs[1][18]]);
            let d2e3 = transmute([arrs[0][13], arrs[0][19], arrs[1][13], arrs[1][19]]);
            let e2a3 = transmute([arrs[0][14], arrs[0][15], arrs[1][14], arrs[1][15]]);
            let a4b4 = transmute([arrs[0][20], arrs[0][21], arrs[1][20], arrs[1][21]]);
            let c4d4 = transmute([arrs[0][22], arrs[0][23], arrs[1][22], arrs[1][23]]);
            let e4zz = transmute([arrs[0][24], 0u64, arrs[1][24], 0u64]);
            State {
                a0b1,
                b0c1,
                c0d1,
                d0e1,
                e0a1,
                a2b3,
                b2c3,
                c2d3,
                d2e3,
                e2a3,
                a4b4,
                c4d4,
                e4zz,
            }
        }
    }

    #[inline]
    pub const fn to_arrs(self) -> [[u64; 25]; 2] {
        let a0b1: [u64; 4] = unsafe { transmute(self.a0b1) };
        let b0c1: [u64; 4] = unsafe { transmute(self.b0c1) };
        let c0d1: [u64; 4] = unsafe { transmute(self.c0d1) };
        let d0e1: [u64; 4] = unsafe { transmute(self.d0e1) };
        let e0a1: [u64; 4] = unsafe { transmute(self.e0a1) };
        let a2b3: [u64; 4] = unsafe { transmute(self.a2b3) };
        let b2c3: [u64; 4] = unsafe { transmute(self.b2c3) };
        let c2d3: [u64; 4] = unsafe { transmute(self.c2d3) };
        let d2e3: [u64; 4] = unsafe { transmute(self.d2e3) };
        let e2a3: [u64; 4] = unsafe { transmute(self.e2a3) };
        let a4b4: [u64; 4] = unsafe { transmute(self.a4b4) };
        let c4d4: [u64; 4] = unsafe { transmute(self.c4d4) };
        let e4zz: [u64; 4] = unsafe { transmute(self.e4zz) };

        [
            [
                a0b1[0], b0c1[0], c0d1[0], d0e1[0], e0a1[0], e0a1[1], a0b1[1], b0c1[1], c0d1[1],
                d0e1[1], a2b3[0], b2c3[0], c2d3[0], d2e3[0], e2a3[0], e2a3[1], a2b3[1], b2c3[1],
                c2d3[1], d2e3[1], a4b4[0], a4b4[1], c4d4[0], c4d4[1], e4zz[0],
            ],
            [
                a0b1[2], b0c1[2], c0d1[2], d0e1[2], e0a1[2], e0a1[3], a0b1[3], b0c1[3], c0d1[3],
                d0e1[3], a2b3[2], b2c3[2], c2d3[2], d2e3[2], e2a3[2], e2a3[3], a2b3[3], b2c3[3],
                c2d3[3], d2e3[3], a4b4[2], a4b4[3], c4d4[2], c4d4[3], e4zz[2],
            ],
        ]
    }
}

#[inline(always)]
#[allow(non_snake_case)]
fn round(i: usize, state: State) -> State {
    let State {
        a0b1: Aa0b1,
        b0c1: Ab0c1,
        c0d1: Ac0d1,
        d0e1: Ad0e1,
        e0a1: Ae0a1,
        a2b3: Aa2b3,
        b2c3: Ab2c3,
        c2d3: Ac2d3,
        d2e3: Ad2e3,
        e2a3: Ae2a3,
        a4b4: Aa4b4,
        c4d4: Ac4d4,
        e4zz: Ae4zz,
    } = state;

    let Bbc = xor(Ab0c1, Ab2c3);
    let Bde = xor(Ad0e1, Ad2e3);
    let Bab = xor(xor(Aa0b1, Aa2b3), Aa4b4);
    let Bcd = xor(xor(Ac0d1, Ac2d3), Ac4d4);
    let Bea = xor(xor(Ae0a1, Ae2a3), Ae4zz);
    let Bab = xor(Bab, get_hi_hi(Bea, ZERO));
    let Bab = xor(Bab, get_lo_lo(ZERO, Bbc));
    let Bcd = xor(Bcd, get_hi_hi(Bbc, ZERO));
    let Bcd = xor(Bcd, get_lo_lo(ZERO, Bde));
    let Bea = xor(Bea, get_hi_hi(Bde, ZERO));
    let Bea = get_lo_lo(Bea, Bab);
    let Bbc = get_hi_lo(Bab, Bcd);

    let Cbc = xor(Bab, rol_1(Bcd));
    let Cde = xor(Bcd, rol_1(Bea));
    let Cab = xor(Bea, rol_1(Bbc));
    let Cae = get_lo_hi(Cab, Cde);
    let Cba = swap(Cab);
    let Ccb = swap(Cbc);
    let Cdc = get_lo_lo(Cde, Ccb);
    let Ced = swap(Cde);

    let Da0b1 = rol_0_20(xor(get_lo_hi(Aa0b1, Ad0e1), Cae));
    let Db0c1 = rol_44_3(xor(get_hi_lo(Aa0b1, Aa2b3), Cba));
    let Dc0d1 = rol_43_45(xor(get_lo_hi(Ac2d3, Aa2b3), Ccb));
    let Dd0e1 = rol_21_61(xor(get_hi_lo(Ac2d3, Ac4d4), Cdc));
    let De0a1 = rol_14_28(xor(get_lo_lo(Ae4zz, Ad0e1), Ced));
    let Da2b3 = rol_1_36(xor(get_lo_hi(Ab0c1, Ae0a1), Cba));
    let Db2c3 = rol_6_10(xor(get_hi_lo(Ab0c1, Ab2c3), Ccb));
    let Dc2d3 = rol_25_15(xor(get_lo_hi(Ad2e3, Ab2c3), Cdc));
    let Dd2e3 = rol_8_56(xor(get_hi_hi(Ad2e3, Ac4d4), Ced));
    let De2a3 = rol_18_27(xor(get_lo_lo(Aa4b4, Ae0a1), Cae));
    let Da4b4 = rol_62_55(xor(Ac0d1, swap(Cdc)));
    let Dc4d4 = rol_39_41(xor(Ae2a3, swap(Cae)));
    let De4sa = get_lo_lo(rol_2(xor(dup_hi(Aa4b4), Cbc)), Da4b4);
    let Db4c4 = get_hi_lo(Da4b4, Dc4d4);
    let Dd4e4 = get_hi_lo(Dc4d4, De4sa);

    let Ea0b1 = xor(xor(Da0b1, andn(Db0c1, Dc0d1)), RC[i]);
    let Eb0c1 = xor(Db0c1, andn(Dc0d1, Dd0e1));
    let Ec0d1 = xor(Dc0d1, andn(Dd0e1, De0a1));
    let Ed0e1 = xor(Dd0e1, andn(De0a1, Da0b1));
    let Ee0a1 = xor(De0a1, andn(Da0b1, Db0c1));
    let Ea2b3 = xor(Da2b3, andn(Db2c3, Dc2d3));
    let Eb2c3 = xor(Db2c3, andn(Dc2d3, Dd2e3));
    let Ec2d3 = xor(Dc2d3, andn(Dd2e3, De2a3));
    let Ed2e3 = xor(Dd2e3, andn(De2a3, Da2b3));
    let Ee2a3 = xor(De2a3, andn(Da2b3, Db2c3));
    let Ea4b4 = xor(Da4b4, andn(Db4c4, Dc4d4));
    let Ec4d4 = xor(Dc4d4, andn(Dd4e4, De4sa));
    let Ee4zz = get_lo_hi(xor(De4sa, andn(Da4b4, Db4c4)), ZERO);

    State {
        a0b1: Ea0b1,
        b0c1: Eb0c1,
        c0d1: Ec0d1,
        d0e1: Ed0e1,
        e0a1: Ee0a1,
        a2b3: Ea2b3,
        b2c3: Eb2c3,
        c2d3: Ec2d3,
        d2e3: Ed2e3,
        e2a3: Ee2a3,
        a4b4: Ea4b4,
        c4d4: Ec4d4,
        e4zz: Ee4zz,
    }
}

fn keccak_perm(buf: &mut State) {
    let mut state = *buf;
    for i in 0..24 {
        state = round(i, state);
    }
    *buf = state;
}

impl Permutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {
    fn permute_mut(&self, state: &mut [[u64; VECTOR_LEN]; 25]) {
        keccak_perm(unsafe { transmute(state) });
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
        let mut state = State::from_arrs(STATES);
        keccak_perm(&mut state);
        state.to_arrs()
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
