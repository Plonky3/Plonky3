//! MDS permutation for packed Goldilocks NEON state.

use p3_mds::MdsPermutation;
use p3_symmetric::Permutation;

use super::utils::{pack_lanes, unpack_lanes};
use crate::aarch64_neon::packing::PackedGoldilocksNeon;
use crate::{Goldilocks, MdsMatrixGoldilocks};

/// Apply the scalar MDS to each lane of a packed NEON state independently.
///
/// Each packed slot contains two Goldilocks elements. This function
/// separates them into two scalar arrays, applies the MDS to each,
/// and recombines.
///
/// The transmute from `[u64; WIDTH]` to `[Goldilocks; WIDTH]` is safe
/// because `Goldilocks` is `repr(transparent)` over `u64`.
#[inline]
fn mds_packed<const WIDTH: usize>(
    mds: &MdsMatrixGoldilocks,
    input: &mut [PackedGoldilocksNeon; WIDTH],
) where
    MdsMatrixGoldilocks: Permutation<[Goldilocks; WIDTH]>,
{
    // Separate the two lanes into independent scalar arrays.
    let (mut lane0, mut lane1) = unpack_lanes(input);

    // Transmute u64 arrays to Goldilocks arrays and apply the scalar MDS.
    // This is safe because Goldilocks is repr(transparent) over u64.
    unsafe {
        mds.permute_mut(&mut *(&mut lane0 as *mut [u64; WIDTH] as *mut [Goldilocks; WIDTH]));
        mds.permute_mut(&mut *(&mut lane1 as *mut [u64; WIDTH] as *mut [Goldilocks; WIDTH]));
    }

    // Recombine both lanes into the packed representation.
    pack_lanes(input, &lane0, &lane1);
}

impl Permutation<[PackedGoldilocksNeon; 8]> for MdsMatrixGoldilocks {
    fn permute_mut(&self, input: &mut [PackedGoldilocksNeon; 8]) {
        mds_packed(self, input);
    }
}

impl MdsPermutation<PackedGoldilocksNeon, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksNeon; 12]> for MdsMatrixGoldilocks {
    fn permute_mut(&self, input: &mut [PackedGoldilocksNeon; 12]) {
        mds_packed(self, input);
    }
}

impl MdsPermutation<PackedGoldilocksNeon, 12> for MdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{Goldilocks, MdsMatrixGoldilocks, PackedGoldilocksNeon};

    /// Verify that the packed NEON MDS produces the same result as the
    /// scalar MDS when each packed slot holds a single element in lane 0.
    ///
    /// Generates a random scalar input, computes the expected output via
    /// the scalar path, then wraps each element into a packed slot and
    /// checks that lane 0 of the packed output matches.
    macro_rules! test_neon_mds {
        ($name:ident, $width:literal) => {
            #[test]
            fn $name() {
                let mut rng = SmallRng::seed_from_u64(1);
                let mds = MdsMatrixGoldilocks;

                // Compute the expected result using the scalar MDS.
                let input: [Goldilocks; $width] = rng.random();
                let expected = mds.permute(input);

                // Wrap each scalar element into a packed slot.
                let packed_input = input.map(Into::<PackedGoldilocksNeon>::into);
                let packed_output = mds.permute(packed_input);

                // Lane 0 of the packed output must match the scalar result.
                let neon_output = packed_output.map(|x| x.0[0]);
                assert_eq!(neon_output, expected);
            }
        };
    }

    test_neon_mds!(test_neon_mds_width_8, 8);
    test_neon_mds!(test_neon_mds_width_12, 12);
}
