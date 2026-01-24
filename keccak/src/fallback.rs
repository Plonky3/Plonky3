//! This module should be included only when no target-specific implementations are available.
//! It provides a fallback implementation based on pure Rust Keccak.

use core::mem::transmute;

use p3_symmetric::{CryptographicPermutation, Permutation};

use crate::KeccakF;

pub const VECTOR_LEN: usize = 1;

impl Permutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {
    fn permute_mut(&self, state: &mut [[u64; VECTOR_LEN]; 25]) {
        let state: &mut [u64; 25] = unsafe { transmute(state) };
        self.permute_mut(state);
    }
}

impl CryptographicPermutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {}
