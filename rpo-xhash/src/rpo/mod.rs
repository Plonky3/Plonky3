//! RPO (Rescue Prime Optimized) hash for small fields.
//!
//! RPO-M31 follows the specification in "RPO-M31 and XHash-M31" (Ashur & Tariq).
//!
//! Each round consists of:
//!   - F_M (forward step): MDS, add RC, x^alpha
//!   - B_M (backward step): MDS, add RC, x^{1/alpha}
//!
//! After all rounds, a final linear layer applies MDS and adds RC.

use alloc::vec::Vec;
use p3_field::{Field, PackedValue, PrimeField};
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};

/// SIMD-accelerated `state += rc` using `<F as Field>::Packing`.
///
/// On aarch64 with our widths (24 for BB/KB/M31, 12 for GL), the state slice
/// divides cleanly into the packed width (4 or 2) and the entire add is SIMD.
/// On platforms where `WIDTH % Packing::WIDTH != 0` (e.g. AVX-512), the
/// trailing elements are added scalar.
#[inline(always)]
pub(crate) fn add_rc<F: PrimeField, const WIDTH: usize>(state: &mut [F; WIDTH], rc: &[F]) {
    let (state_packed, state_suffix) = <F as Field>::Packing::pack_slice_with_suffix_mut(state);
    let (rc_packed, rc_suffix) = <F as Field>::Packing::pack_slice_with_suffix(rc);
    for (s, &r) in state_packed.iter_mut().zip(rc_packed.iter()) {
        *s += r;
    }
    for (s, &r) in state_suffix.iter_mut().zip(rc_suffix.iter()) {
        *s += r;
    }
}

/// Forward and backward S-box pair for an RPO round.
pub trait RpoSbox<F, const WIDTH: usize>: Clone + Sync {
    fn forward(&self, state: &mut [F; WIDTH]);
    fn backward(&self, state: &mut [F; WIDTH]);
}

/// RPO hash permutation, generic over S-box and MDS layer.
///
/// Schedule for N rounds:
///   for i in 0..N:
///     state = sbox_forward(MDS(state) + RC_F\[i\])
///     state = sbox_backward(MDS(state) + RC_B\[i\])
///   state = MDS(state) + RC_L
#[derive(Clone, Debug)]
pub struct RpoHash<F, Sbox, Mds, const WIDTH: usize> {
    num_rounds: usize,
    /// Flat layout: [RC_F_0, RC_B_0, RC_F_1, RC_B_1, ..., RC_L].
    /// Total length: (2 * num_rounds + 1) * WIDTH.
    round_constants: Vec<F>,
    sbox: Sbox,
    mds: Mds,
}

impl<F, Sbox, Mds, const WIDTH: usize> RpoHash<F, Sbox, Mds, WIDTH>
where
    F: PrimeField,
    Sbox: RpoSbox<F, WIDTH> + Default,
    Mds: MdsPermutation<F, WIDTH> + Default,
{
    pub fn new_from_constants(num_rounds: usize, round_constants: Vec<F>) -> Self {
        let expected = (2 * num_rounds + 1) * WIDTH;
        assert_eq!(
            round_constants.len(),
            expected,
            "expected {expected} round constants, got {}",
            round_constants.len()
        );
        Self {
            num_rounds,
            round_constants,
            sbox: Sbox::default(),
            mds: Mds::default(),
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

impl<F, Sbox, Mds, const WIDTH: usize> Permutation<[F; WIDTH]> for RpoHash<F, Sbox, Mds, WIDTH>
where
    F: PrimeField,
    Sbox: RpoSbox<F, WIDTH>,
    Mds: MdsPermutation<F, WIDTH>,
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        let rc = &self.round_constants;
        let mut offset = 0;

        for _ in 0..self.num_rounds {
            // Forward step: MDS -> add RC -> x^alpha
            self.mds.permute_mut(state);
            add_rc(state, &rc[offset..offset + WIDTH]);
            offset += WIDTH;
            self.sbox.forward(state);

            // Backward step: MDS -> add RC -> x^{1/alpha}
            self.mds.permute_mut(state);
            add_rc(state, &rc[offset..offset + WIDTH]);
            offset += WIDTH;
            self.sbox.backward(state);
        }

        // Final linear layer: MDS -> add RC
        self.mds.permute_mut(state);
        add_rc(state, &rc[offset..offset + WIDTH]);
    }
}

impl<F, Sbox, Mds, const WIDTH: usize> CryptographicPermutation<[F; WIDTH]>
    for RpoHash<F, Sbox, Mds, WIDTH>
where
    F: PrimeField,
    Sbox: RpoSbox<F, WIDTH>,
    Mds: MdsPermutation<F, WIDTH>,
{
}

pub mod babybear;
pub mod goldilocks;
pub mod koalabear;
pub mod m31;
