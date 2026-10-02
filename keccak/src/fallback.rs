//! This module should be included only when none of the more target-specific implementations are
//! available. It fills in a few things based on a pure Rust implementation of Keccak.

use p3_symmetric::{CryptographicPermutation, Permutation};
use tiny_keccak::keccakf;

use crate::KeccakF;
use crate::batch::{self, Lanes, State};

pub const VECTOR_LEN: usize = 1;

/// Permute one Keccak state, stored one word per row.
fn keccak_perm(state: &mut [[u64; VECTOR_LEN]; 25]) {
    // One lane per row makes the rows the plain 25-word state.
    let (words, _) = state.as_flattened_mut().as_chunks_mut::<25>();
    keccakf(&mut words[0]);
}

/// Every CPU runs the scalar permutation.
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

batch::kernel!("scalar", Backend, VECTOR_LEN);

impl Permutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {
    fn permute_mut(&self, input: &mut [[u64; VECTOR_LEN]; 25]) {
        keccak_perm(input);
    }
}

impl CryptographicPermutation<[[u64; VECTOR_LEN]; 25]> for KeccakF {}
