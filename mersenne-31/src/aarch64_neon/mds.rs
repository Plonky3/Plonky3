//! Packed MDS permutation for Mersenne31 on aarch64 NEON.
//!
//! Implements [`Permutation`] and [`MdsPermutation`] for
//! `[PackedMersenne31Neon; WIDTH]` by applying the scalar circulant MDS
//! independently to each of the four NEON lanes.
//!
//! Each [`PackedMersenne31Neon`] element holds 4 `Mersenne31` values in a
//! `uint32x4_t` register. The MDS is applied per-lane: unpack → scalar
//! MDS → repack, ensuring correctness by reusing the optimised Karatsuba
//! convolution from [`MdsMatrixMersenne31`].

use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31Neon};

/// Apply the scalar MDS to each NEON lane independently.
///
/// Extracts one scalar state per lane, runs the circulant MDS
/// convolution, then writes the results back into the packed state.
#[inline]
fn mds_packed<const WIDTH: usize>(
    mds: &MdsMatrixMersenne31,
    input: &mut [PackedMersenne31Neon; WIDTH],
) where
    MdsMatrixMersenne31: Permutation<[Mersenne31; WIDTH]>,
{
    for lane in 0..4 {
        let mut scalar_state: [Mersenne31; WIDTH] = core::array::from_fn(|i| input[i].0[lane]);
        mds.permute_mut(&mut scalar_state);
        for i in 0..WIDTH {
            input[i].0[lane] = scalar_state[i];
        }
    }
}

impl Permutation<[PackedMersenne31Neon; 8]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31Neon; 8]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31Neon, 8> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31Neon; 12]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31Neon; 12]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31Neon, 12> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31Neon; 16]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31Neon; 16]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31Neon, 16> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31Neon; 32]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31Neon; 32]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31Neon, 32> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31Neon; 64]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31Neon; 64]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31Neon, 64> for MdsMatrixMersenne31 {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31Neon};

    type F = Mersenne31;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    macro_rules! proptest_neon_mds {
        ($name:ident, $width:literal, $uniform:ident) => {
            proptest! {
                #[test]
                fn $name(input in prop::array::$uniform(arb_f())) {
                    let mds = MdsMatrixMersenne31;
                    let expected = mds.permute(input);

                    let packed_input = input.map(Into::<PackedMersenne31Neon>::into);
                    let packed_output = mds.permute(packed_input);
                    let neon_output = packed_output.map(|x| x.0[0]);

                    prop_assert_eq!(neon_output, expected);
                }
            }
        };
    }

    proptest_neon_mds!(mds_neon_matches_scalar_8, 8, uniform8);
    proptest_neon_mds!(mds_neon_matches_scalar_12, 12, uniform12);
    proptest_neon_mds!(mds_neon_matches_scalar_16, 16, uniform16);
    proptest_neon_mds!(mds_neon_matches_scalar_32, 32, uniform32);

    proptest! {
        #[test]
        fn mds_neon_matches_scalar_64(
            a in prop::array::uniform32(arb_f()),
            b in prop::array::uniform32(arb_f()),
        ) {
            let mut input = [F::ZERO; 64];
            input[..32].copy_from_slice(&a);
            input[32..].copy_from_slice(&b);

            let mds = MdsMatrixMersenne31;
            let expected = mds.permute(input);

            let packed_input = input.map(Into::<PackedMersenne31Neon>::into);
            let packed_output = mds.permute(packed_input);
            let neon_output = packed_output.map(|x| x.0[0]);

            prop_assert_eq!(neon_output, expected);
        }
    }
}
