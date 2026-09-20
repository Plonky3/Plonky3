//! The vectorized permutation against the scalar reference, at whatever width the build selects.
//!
//! Each backend already checks itself against one fixed state.
//!
//! A single fixture misses a lane-crossing mistake that happens to be quiet on those inputs.
//!
//! So this sweeps random states instead.

use p3_keccak::{KeccakF, VECTOR_LEN};
use p3_symmetric::Permutation;
use proptest::prelude::*;
use tiny_keccak::keccakf;

/// The scalar permutation applied lane by lane, from the reference implementation.
fn reference(state: [[u64; VECTOR_LEN]; 25]) -> [[u64; VECTOR_LEN]; 25] {
    let mut out = state;
    for lane in 0..VECTOR_LEN {
        let mut scalar: [u64; 25] = core::array::from_fn(|i| state[i][lane]);
        keccakf(&mut scalar);
        for i in 0..25 {
            out[i][lane] = scalar[i];
        }
    }
    out
}

proptest! {
    #[test]
    fn the_vectorized_permutation_agrees_lane_by_lane(
        words in prop::array::uniform25(any::<[u64; VECTOR_LEN]>()),
    ) {
        let mut state = words;
        KeccakF.permute_mut(&mut state);
        prop_assert_eq!(state, reference(words));
    }
}

#[test]
fn distinct_lanes_do_not_mix() {
    // Every lane but one is zero, so anything leaking across lanes shows immediately.
    for lane in 0..VECTOR_LEN {
        let mut state = [[0u64; VECTOR_LEN]; 25];
        for (i, word) in state.iter_mut().enumerate() {
            word[lane] = 0x0f0e_0d0c_0b0a_0908u64.wrapping_mul(i as u64 + 1);
        }
        let expected = reference(state);
        KeccakF.permute_mut(&mut state);
        assert_eq!(state, expected, "lane {lane}");
    }
}

#[test]
fn the_all_zero_state_matches_the_published_first_word() {
    // The first state word after permuting an all-zero state, from the reference test vectors.
    let mut state = [0u64; 25];
    KeccakF.permute_mut(&mut state);
    assert_eq!(state[0], 0xf125_8f79_40e1_dde7);
}
