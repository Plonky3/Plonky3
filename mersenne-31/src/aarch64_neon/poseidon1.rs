//! NEON-optimized Poseidon1 permutation layers for Mersenne31.
//!
//! Provides [`Poseidon1ExternalLayerMersenne31`] (full rounds) and
//! [`Poseidon1InternalLayerMersenne31`] (partial rounds), implementing
//! [`FullRoundLayer`] and [`PartialRoundLayer`] for [`PackedMersenne31Neon`].
//!
//! # Optimization Strategy
//!
//! **Full rounds** — round constants are pre-packed into `uint32x4_t` NEON
//! vectors at construction time. Each full round fuses the constant addition
//! and the x^5 S-box into a single [`add_rc_and_sbox`] call, then applies
//! the circulant MDS via per-lane scalar Karatsuba convolution.
//!
//! **Partial rounds** — the sparse matrix decomposition from the Poseidon
//! paper (Appendix B) is used with scalar `Mersenne31` constants. The
//! S-box is applied only to `state[0]`; the remaining state elements are
//! updated via the cheap sparse matrix–vector product ([`cheap_matmul`]).

use alloc::vec::Vec;
use core::arch::aarch64;
use core::arch::aarch64::uint32x4_t;

use p3_field::InjectiveMonomial;
use p3_poseidon1::external::{
    FullRoundConstants, FullRoundLayer, FullRoundLayerConstructor, mds_multiply,
};
use p3_poseidon1::internal::{
    PartialRoundConstants, PartialRoundLayer, PartialRoundLayerConstructor, cheap_matmul,
    partial_permute_state,
};
use p3_symmetric::Permutation;

use super::utils::add_rc_and_sbox;
use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31Neon};

/// NEON-optimized external (full-round) layer for Mersenne31 Poseidon1.
///
/// Stores round constants in two forms:
/// - **Scalar** (`FullRoundConstants<Mersenne31, WIDTH>`) — used by the
///   scalar fallback path (`FullRoundLayer<Mersenne31, …>`).
/// - **Packed** (`Vec<[uint32x4_t; WIDTH]>`) — each constant broadcast to
///   all four NEON lanes, enabling the fused [`add_rc_and_sbox`] path.
#[derive(Clone)]
pub struct Poseidon1ExternalLayerMersenne31<const WIDTH: usize> {
    constants: FullRoundConstants<Mersenne31, WIDTH>,
    packed_initial_constants: Vec<[uint32x4_t; WIDTH]>,
    packed_terminal_constants: Vec<[uint32x4_t; WIDTH]>,
}

impl<const WIDTH: usize> FullRoundLayerConstructor<Mersenne31, WIDTH>
    for Poseidon1ExternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(constants: FullRoundConstants<Mersenne31, WIDTH>) -> Self {
        let pack_rc = |rcs: &[[Mersenne31; WIDTH]]| -> Vec<[uint32x4_t; WIDTH]> {
            rcs.iter()
                .map(|rc| rc.map(|c| unsafe { aarch64::vdupq_n_u32(c.value) }))
                .collect()
        };
        let packed_initial_constants = pack_rc(&constants.initial);
        let packed_terminal_constants = pack_rc(&constants.terminal);
        Self {
            constants,
            packed_initial_constants,
            packed_terminal_constants,
        }
    }
}

/// Apply a sequence of full rounds using NEON-packed constants.
///
/// For each round: fuse `add_rc + x^5` via [`add_rc_and_sbox`], then
/// apply the circulant MDS through per-lane scalar delegation.
#[inline]
fn full_rounds_packed<const WIDTH: usize>(
    state: &mut [PackedMersenne31Neon; WIDTH],
    packed_constants: &[[uint32x4_t; WIDTH]],
) where
    MdsMatrixMersenne31: Permutation<[PackedMersenne31Neon; WIDTH]>,
{
    let mds = MdsMatrixMersenne31;
    for rc in packed_constants {
        for (s, &c) in state.iter_mut().zip(rc.iter()) {
            add_rc_and_sbox(s, c);
        }
        mds.permute_mut(state);
    }
}

/// Packed NEON path: fused AddRC + S-box with pre-packed constants.
impl<const WIDTH: usize> FullRoundLayer<PackedMersenne31Neon, WIDTH, 5>
    for Poseidon1ExternalLayerMersenne31<WIDTH>
where
    MdsMatrixMersenne31: Permutation<[PackedMersenne31Neon; WIDTH]>,
{
    fn permute_state_initial(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        full_rounds_packed(state, &self.packed_initial_constants);
    }

    fn permute_state_terminal(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        full_rounds_packed(state, &self.packed_terminal_constants);
    }
}

/// Scalar fallback: standard AddRC → S-box → MDS on `Mersenne31` elements.
impl<const WIDTH: usize> FullRoundLayer<Mersenne31, WIDTH, 5>
    for Poseidon1ExternalLayerMersenne31<WIDTH>
where
    MdsMatrixMersenne31: Permutation<[Mersenne31; WIDTH]>,
{
    fn permute_state_initial(&self, state: &mut [Mersenne31; WIDTH]) {
        let mds = MdsMatrixMersenne31;
        for round_constants in &self.constants.initial {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                *s += rc;
            }
            for s in state.iter_mut() {
                *s = s.injective_exp_n();
            }
            mds.permute_mut(state);
        }
    }

    fn permute_state_terminal(&self, state: &mut [Mersenne31; WIDTH]) {
        let mds = MdsMatrixMersenne31;
        for round_constants in &self.constants.terminal {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                *s += rc;
            }
            for s in state.iter_mut() {
                *s = s.injective_exp_n();
            }
            mds.permute_mut(state);
        }
    }
}

/// NEON-optimized internal (partial-round) layer for Mersenne31 Poseidon1.
///
/// The internal layer uses the **sparse matrix decomposition** from the
/// Poseidon paper (Appendix B). Constants are stored as scalar
/// `Mersenne31` values; the `Algebra<Mersenne31>` impl on
/// `PackedMersenne31Neon` handles broadcasting during multiplication.
///
/// Each partial round applies the S-box only to `state[0]`, then performs
/// a cheap sparse matrix–vector product via [`cheap_matmul`].
#[derive(Clone)]
pub struct Poseidon1InternalLayerMersenne31<const WIDTH: usize> {
    constants: PartialRoundConstants<Mersenne31, WIDTH>,
}

impl<const WIDTH: usize> PartialRoundLayerConstructor<Mersenne31, WIDTH>
    for Poseidon1InternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(constants: PartialRoundConstants<Mersenne31, WIDTH>) -> Self {
        Self { constants }
    }
}

/// Packed NEON path: S-box on `state[0]` only, sparse matmul via scalar constants.
impl<const WIDTH: usize> PartialRoundLayer<PackedMersenne31Neon, WIDTH, 5>
    for Poseidon1InternalLayerMersenne31<WIDTH>
{
    fn permute_state(&self, state: &mut [PackedMersenne31Neon; WIDTH]) {
        // Add the full first-round constant vector (scalar → packed broadcast).
        for (s, &rc) in state
            .iter_mut()
            .zip(self.constants.first_round_constants.iter())
        {
            *s += rc;
        }

        // Dense transition matrix m_i, applied once before the partial rounds.
        mds_multiply(state, &self.constants.m_i);

        let rounds_p = self.constants.sparse_first_row.len();

        // Partial rounds 0..RP-2: S-box on state[0] + round constant + sparse matmul.
        for r in 0..rounds_p - 1 {
            state[0] = state[0].injective_exp_n();
            state[0] += self.constants.round_constants[r];
            cheap_matmul(
                state,
                &self.constants.sparse_first_row[r],
                &self.constants.v[r],
            );
        }

        // Last partial round: S-box on state[0] + sparse matmul (no round constant).
        state[0] = state[0].injective_exp_n();
        cheap_matmul(
            state,
            &self.constants.sparse_first_row[rounds_p - 1],
            &self.constants.v[rounds_p - 1],
        );
    }
}

/// Scalar fallback: delegates to the generic partial-round implementation.
impl<const WIDTH: usize> PartialRoundLayer<Mersenne31, WIDTH, 5>
    for Poseidon1InternalLayerMersenne31<WIDTH>
{
    fn permute_state(&self, state: &mut [Mersenne31; WIDTH]) {
        partial_permute_state::<Mersenne31, Mersenne31, WIDTH, 5>(state, &self.constants);
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon1::{default_mersenne31_poseidon1_16, default_mersenne31_poseidon1_32};
    use crate::{Mersenne31, PackedMersenne31Neon};

    type F = Mersenne31;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn poseidon1_neon_matches_scalar_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            let perm = default_mersenne31_poseidon1_16();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut neon_input = input.map(Into::<PackedMersenne31Neon>::into);
            perm.permute_mut(&mut neon_input);
            let neon_output = neon_input.map(|x| x.0[0]);

            prop_assert_eq!(neon_output, expected);
        }

        #[test]
        fn poseidon1_neon_matches_scalar_width_32(
            input in prop::array::uniform32(arb_f())
        ) {
            let perm = default_mersenne31_poseidon1_32();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut neon_input = input.map(Into::<PackedMersenne31Neon>::into);
            perm.permute_mut(&mut neon_input);
            let neon_output = neon_input.map(|x| x.0[0]);

            prop_assert_eq!(neon_output, expected);
        }
    }
}
