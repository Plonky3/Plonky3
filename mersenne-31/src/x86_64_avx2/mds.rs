//! Packed MDS permutation for Mersenne31 on x86_64 AVX2.
//!
//! Implements [`Permutation`] and [`MdsPermutation`] for
//! `[PackedMersenne31AVX2; WIDTH]` by applying the scalar circulant MDS
//! independently to each of the eight AVX2 lanes.
//!
//! Each [`PackedMersenne31AVX2`] element holds 8 `Mersenne31` values in a
//! `__m256i` register. The MDS is applied per-lane: unpack → scalar
//! MDS → repack, reusing the optimised Karatsuba convolution from
//! [`MdsMatrixMersenne31`].

use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31AVX2};

/// Apply the scalar MDS to each AVX2 lane independently.
///
/// Extracts one scalar state per lane, runs the circulant MDS
/// convolution, then writes the results back into the packed state.
#[inline]
fn mds_packed<const WIDTH: usize>(
    mds: &MdsMatrixMersenne31,
    input: &mut [PackedMersenne31AVX2; WIDTH],
) where
    MdsMatrixMersenne31: Permutation<[Mersenne31; WIDTH]>,
{
    for lane in 0..8 {
        let mut scalar_state: [Mersenne31; WIDTH] = core::array::from_fn(|i| input[i].0[lane]);
        mds.permute_mut(&mut scalar_state);
        for i in 0..WIDTH {
            input[i].0[lane] = scalar_state[i];
        }
    }
}

impl Permutation<[PackedMersenne31AVX2; 8]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX2; 8]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX2, 8> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX2; 12]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX2; 12]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX2, 12> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX2; 16]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX2; 16]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX2, 16> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX2; 32]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX2; 32]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX2, 32> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX2; 64]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX2; 64]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX2, 64> for MdsMatrixMersenne31 {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31AVX2};

    type F = Mersenne31;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    macro_rules! proptest_avx2_mds {
        ($name:ident, $width:literal, $uniform:ident) => {
            proptest! {
                #[test]
                fn $name(input in prop::array::$uniform(arb_f())) {
                    let mds = MdsMatrixMersenne31;
                    let expected = mds.permute(input);

                    let packed_input = input.map(Into::<PackedMersenne31AVX2>::into);
                    let packed_output = mds.permute(packed_input);
                    let avx2_output = packed_output.map(|x| x.0[0]);

                    prop_assert_eq!(avx2_output, expected);
                }
            }
        };
    }

    proptest_avx2_mds!(mds_avx2_matches_scalar_8, 8, uniform8);
    proptest_avx2_mds!(mds_avx2_matches_scalar_12, 12, uniform12);
    proptest_avx2_mds!(mds_avx2_matches_scalar_16, 16, uniform16);
    proptest_avx2_mds!(mds_avx2_matches_scalar_32, 32, uniform32);

    proptest! {
        #[test]
        fn mds_avx2_matches_scalar_64(
            a in prop::array::uniform32(arb_f()),
            b in prop::array::uniform32(arb_f()),
        ) {
            let mut input = [F::ZERO; 64];
            input[..32].copy_from_slice(&a);
            input[32..].copy_from_slice(&b);

            let mds = MdsMatrixMersenne31;
            let expected = mds.permute(input);

            let packed_input = input.map(Into::<PackedMersenne31AVX2>::into);
            let packed_output = mds.permute(packed_input);
            let avx2_output = packed_output.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }
    }
}
