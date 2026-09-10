//! Two-state Keccak-f permutation using WebAssembly SIMD128.

use core::arch::wasm32::{
    i8x16_shuffle, u64x2_shl, u64x2_shr, v128, v128_andnot, v128_or, v128_xor,
};
use core::mem::transmute;

use p3_symmetric::{CryptographicPermutation, Permutation};

use crate::KeccakF;

pub const VECTOR_LEN: usize = 2;

const RC: [v128; 24] = unsafe {
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
fn form_matrix(buf: [v128; 25]) -> [[v128; 5]; 5] {
    unsafe { transmute(buf) }
}

#[inline(always)]
fn flatten(mat: [[v128; 5]; 5]) -> [v128; 25] {
    unsafe { transmute(mat) }
}

#[inline(always)]
fn rol<const AMT: u32>(a: v128) -> v128 {
    v128_or(u64x2_shl(a, AMT), u64x2_shr(a, 64 - AMT))
}

#[inline(always)]
fn rol_8(a: v128) -> v128 {
    i8x16_shuffle::<7, 0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14>(a, a)
}

#[inline(always)]
fn rol_56(a: v128) -> v128 {
    i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 8>(a, a)
}

#[inline(always)]
fn theta(state: [[v128; 5]; 5]) -> [[v128; 5]; 5] {
    let parities: [v128; 5] = core::array::from_fn(|x| {
        v128_xor(
            v128_xor(v128_xor(state[0][x], state[1][x]), state[2][x]),
            v128_xor(state[3][x], state[4][x]),
        )
    });
    let deltas = [
        v128_xor(parities[4], rol::<1>(parities[1])),
        v128_xor(parities[0], rol::<1>(parities[2])),
        v128_xor(parities[1], rol::<1>(parities[3])),
        v128_xor(parities[2], rol::<1>(parities[4])),
        v128_xor(parities[3], rol::<1>(parities[0])),
    ];

    core::array::from_fn(|y| core::array::from_fn(|x| v128_xor(state[y][x], deltas[x])))
}

#[inline(always)]
fn rho(state: [[v128; 5]; 5]) -> [[v128; 5]; 5] {
    [
        [
            state[0][0],
            rol::<1>(state[0][1]),
            rol::<62>(state[0][2]),
            rol::<28>(state[0][3]),
            rol::<27>(state[0][4]),
        ],
        [
            rol::<36>(state[1][0]),
            rol::<44>(state[1][1]),
            rol::<6>(state[1][2]),
            rol::<55>(state[1][3]),
            rol::<20>(state[1][4]),
        ],
        [
            rol::<3>(state[2][0]),
            rol::<10>(state[2][1]),
            rol::<43>(state[2][2]),
            rol::<25>(state[2][3]),
            rol::<39>(state[2][4]),
        ],
        [
            rol::<41>(state[3][0]),
            rol::<45>(state[3][1]),
            rol::<15>(state[3][2]),
            rol::<21>(state[3][3]),
            rol_8(state[3][4]),
        ],
        [
            rol::<18>(state[4][0]),
            rol::<2>(state[4][1]),
            rol::<61>(state[4][2]),
            rol_56(state[4][3]),
            rol::<14>(state[4][4]),
        ],
    ]
}

#[inline(always)]
const fn pi(state: [[v128; 5]; 5]) -> [[v128; 5]; 5] {
    [
        [
            state[0][0],
            state[1][1],
            state[2][2],
            state[3][3],
            state[4][4],
        ],
        [
            state[0][3],
            state[1][4],
            state[2][0],
            state[3][1],
            state[4][2],
        ],
        [
            state[0][1],
            state[1][2],
            state[2][3],
            state[3][4],
            state[4][0],
        ],
        [
            state[0][4],
            state[1][0],
            state[2][1],
            state[3][2],
            state[4][3],
        ],
        [
            state[0][2],
            state[1][3],
            state[2][4],
            state[3][0],
            state[4][1],
        ],
    ]
}

#[inline(always)]
fn chi_row(row: [v128; 5]) -> [v128; 5] {
    [
        v128_xor(row[0], v128_andnot(row[2], row[1])),
        v128_xor(row[1], v128_andnot(row[3], row[2])),
        v128_xor(row[2], v128_andnot(row[4], row[3])),
        v128_xor(row[3], v128_andnot(row[0], row[4])),
        v128_xor(row[4], v128_andnot(row[1], row[0])),
    ]
}

#[inline(always)]
fn chi(state: [[v128; 5]; 5]) -> [[v128; 5]; 5] {
    [
        chi_row(state[0]),
        chi_row(state[1]),
        chi_row(state[2]),
        chi_row(state[3]),
        chi_row(state[4]),
    ]
}

#[inline(always)]
fn round(round_constant: v128, state: [v128; 25]) -> [v128; 25] {
    let state = chi(pi(rho(theta(form_matrix(state)))));
    let mut state = flatten(state);
    state[0] = v128_xor(state[0], round_constant);
    state
}

fn keccak_perm(buf: &mut [[u64; VECTOR_LEN]; 25]) {
    let mut state: [v128; 25] = unsafe { transmute(*buf) };
    for &round_constant in &RC {
        state = round(round_constant, state);
    }
    *buf = unsafe { transmute::<[v128; 25], [[u64; VECTOR_LEN]; 25]>(state) };
}

impl Permutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {
    fn permute_mut(&self, state: &mut [[u64; VECTOR_LEN]; 25]) {
        keccak_perm(state);
    }
}

impl CryptographicPermutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {}
