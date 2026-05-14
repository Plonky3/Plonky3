//! XHash hash function: base-field F_M/B_M rounds interleaved with an
//! extension-field S-box round (P3_M).
//!
//! XHash is a distinct construction from RPO; the two share the F_M/B_M
//! round shape but XHash adds an extension-field S-box layer between them.
//! `Rpx256` from miden-crypto is XHash instantiated over Goldilocks.
//!
//! From "RPO-M31 and XHash-M31" (Ashur & Tariq), Section 3.1.2.
//!
//! Round structure (N rounds):
//!   for i in 0..N:
//!     F_M:  MDS → RC → x^d           (base field forward)
//!     B_M:  MDS → RC → x^{1/d}       (base field backward)
//!     P3_M: RC → X^d over F_{p^k}    (extension field, NO MDS)
//!   Final: MDS → RC

use alloc::vec::Vec;
use p3_field::PrimeField;
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};

use crate::rpo::{add_rc, RpoSbox};

/// XHash permutation, generic over base S-box, extension S-box, and MDS.
///
/// `BaseSbox`: forward (x^d) + backward (x^{1/d}) over F_p.
/// `ExtSbox`: forward X^d over an extension field F_{p^k} (stride-k layout on the
/// WIDTH-element state). All current instantiations use k=2 (BabyBear) or k=3
/// (KoalaBear, M31, Goldilocks).
#[derive(Clone, Debug)]
pub struct XHash<F, BaseSbox, ExtSbox, Mds, const WIDTH: usize> {
    num_rounds: usize,
    /// Flat layout: [RC_F_0, RC_B_0, RC_P3_0, ..., RC_L].
    /// Total: (3 * num_rounds + 1) * WIDTH.
    round_constants: Vec<F>,
    base_sbox: BaseSbox,
    ext_sbox: ExtSbox,
    mds: Mds,
}

impl<F, BaseSbox, ExtSbox, Mds, const WIDTH: usize> XHash<F, BaseSbox, ExtSbox, Mds, WIDTH>
where
    F: PrimeField,
    BaseSbox: RpoSbox<F, WIDTH> + Default,
    ExtSbox: Permutation<[F; WIDTH]> + Default,
    Mds: MdsPermutation<F, WIDTH> + Default,
{
    pub fn new_from_constants(num_rounds: usize, round_constants: Vec<F>) -> Self {
        let expected = (3 * num_rounds + 1) * WIDTH;
        assert_eq!(
            round_constants.len(),
            expected,
            "expected {expected} round constants, got {}",
            round_constants.len()
        );
        Self {
            num_rounds,
            round_constants,
            base_sbox: BaseSbox::default(),
            ext_sbox: ExtSbox::default(),
            mds: Mds::default(),
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

impl<F, BaseSbox, ExtSbox, Mds, const WIDTH: usize> Permutation<[F; WIDTH]>
    for XHash<F, BaseSbox, ExtSbox, Mds, WIDTH>
where
    F: PrimeField,
    BaseSbox: RpoSbox<F, WIDTH>,
    ExtSbox: Permutation<[F; WIDTH]>,
    Mds: MdsPermutation<F, WIDTH>,
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        let rc = &self.round_constants;
        let mut offset = 0;

        for _ in 0..self.num_rounds {
            // F_M: MDS → RC → base forward S-box
            self.mds.permute_mut(state);
            add_rc(state, &rc[offset..offset + WIDTH]);
            offset += WIDTH;
            self.base_sbox.forward(state);

            // B_M: MDS → RC → base backward S-box
            self.mds.permute_mut(state);
            add_rc(state, &rc[offset..offset + WIDTH]);
            offset += WIDTH;
            self.base_sbox.backward(state);

            // P3_M: RC → extension S-box (NO MDS)
            add_rc(state, &rc[offset..offset + WIDTH]);
            offset += WIDTH;
            self.ext_sbox.permute_mut(state);
        }

        // Final linear layer: MDS → RC
        self.mds.permute_mut(state);
        add_rc(state, &rc[offset..offset + WIDTH]);
    }
}

impl<F, BaseSbox, ExtSbox, Mds, const WIDTH: usize> CryptographicPermutation<[F; WIDTH]>
    for XHash<F, BaseSbox, ExtSbox, Mds, WIDTH>
where
    F: PrimeField,
    BaseSbox: RpoSbox<F, WIDTH>,
    ExtSbox: Permutation<[F; WIDTH]>,
    Mds: MdsPermutation<F, WIDTH>,
{
}

pub mod babybear;
pub mod goldilocks;
pub mod koalabear;
pub mod m31;
