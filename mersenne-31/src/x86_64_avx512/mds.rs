//! Packed MDS permutation for Mersenne31 on x86_64 AVX-512.
//!
//! Implements [`Permutation`] and [`MdsPermutation`] for
//! `[PackedMersenne31AVX512; WIDTH]` by applying the scalar circulant MDS
//! independently to each of the sixteen AVX-512 lanes.
//!
//! Each [`PackedMersenne31AVX512`] element holds 16 `Mersenne31` values in
//! a `__m512i` register. The MDS is applied per-lane: unpack → scalar
//! MDS → repack, reusing the optimised Karatsuba convolution from
//! [`MdsMatrixMersenne31`].

use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31AVX512};

/// Apply the scalar MDS to each AVX-512 lane independently.
///
/// Extracts one scalar state per lane, runs the circulant MDS
/// convolution, then writes the results back into the packed state.
#[inline]
fn mds_packed<const WIDTH: usize>(
    mds: &MdsMatrixMersenne31,
    input: &mut [PackedMersenne31AVX512; WIDTH],
) where
    MdsMatrixMersenne31: Permutation<[Mersenne31; WIDTH]>,
{
    for lane in 0..16 {
        let mut scalar_state: [Mersenne31; WIDTH] = core::array::from_fn(|i| input[i].0[lane]);
        mds.permute_mut(&mut scalar_state);
        for i in 0..WIDTH {
            input[i].0[lane] = scalar_state[i];
        }
    }
}

impl Permutation<[PackedMersenne31AVX512; 8]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX512; 8]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX512, 8> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX512; 12]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX512; 12]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX512, 12> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX512; 16]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX512; 16]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX512, 16> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX512; 32]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX512; 32]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX512, 32> for MdsMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX512; 64]> for MdsMatrixMersenne31 {
    fn permute_mut(&self, input: &mut [PackedMersenne31AVX512; 64]) {
        mds_packed(self, input);
    }
}
impl MdsPermutation<PackedMersenne31AVX512, 64> for MdsMatrixMersenne31 {}
