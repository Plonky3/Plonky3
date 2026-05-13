use p3_field::PrimeCharacteristicRing;
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::{
    mds_circulant_karatsuba_8, mds_circulant_karatsuba_12, mds_circulant_karatsuba_16,
};
use p3_mds::util::{apply_circulant, first_row_to_first_col};
use p3_symmetric::Permutation;

use crate::x86_64_avx512::packing::PackedGoldilocksAVX512;
use crate::{
    Goldilocks, MATRIX_CIRC_MDS_8_SML_ROW, MATRIX_CIRC_MDS_12_SML_ROW, MATRIX_CIRC_MDS_16_SML_ROW,
    MATRIX_CIRC_MDS_24_GOLDILOCKS, MdsMatrixGoldilocks,
};

/// Convert a `[i64; N]` row of small non-negative MDS coefficients into the
/// matching circulant first column as `[Goldilocks; N]`. Used at compile time
/// to feed the Karatsuba helpers.
const fn sml_row_to_goldilocks_col<const N: usize>(row: &[i64; N]) -> [Goldilocks; N] {
    let col_i64 = first_row_to_first_col(row);
    let mut col = [Goldilocks::ZERO; N];
    let mut i = 0;
    while i < N {
        col[i] = Goldilocks::new(col_i64[i] as u64);
        i += 1;
    }
    col
}

impl Permutation<[PackedGoldilocksAVX512; 8]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 8]) -> [PackedGoldilocksAVX512; 8] {
        const COL: [Goldilocks; 8] = sml_row_to_goldilocks_col(&MATRIX_CIRC_MDS_8_SML_ROW);
        let mut state = input;
        mds_circulant_karatsuba_8(&mut state, &COL);
        state
    }
}

impl MdsPermutation<PackedGoldilocksAVX512, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX512; 12]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 12]) -> [PackedGoldilocksAVX512; 12] {
        const COL: [Goldilocks; 12] = sml_row_to_goldilocks_col(&MATRIX_CIRC_MDS_12_SML_ROW);
        let mut state = input;
        mds_circulant_karatsuba_12(&mut state, &COL);
        state
    }
}

impl MdsPermutation<PackedGoldilocksAVX512, 12> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX512; 16]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 16]) -> [PackedGoldilocksAVX512; 16] {
        const COL: [Goldilocks; 16] = sml_row_to_goldilocks_col(&MATRIX_CIRC_MDS_16_SML_ROW);
        let mut state = input;
        mds_circulant_karatsuba_16(&mut state, &COL);
        state
    }
}

impl MdsPermutation<PackedGoldilocksAVX512, 16> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX512; 24]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 24]) -> [PackedGoldilocksAVX512; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_GOLDILOCKS, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX512, 24> for MdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{Goldilocks, MdsMatrixGoldilocks, PackedGoldilocksAVX512};

    macro_rules! test_avx512_mds {
        ($name:ident, $width:literal) => {
            #[test]
            fn $name() {
                let mut rng = SmallRng::seed_from_u64(1);
                let mds = MdsMatrixGoldilocks;

                let input: [Goldilocks; $width] = rng.random();
                let expected = mds.permute(input);

                let packed_input = input.map(Into::<PackedGoldilocksAVX512>::into);
                let packed_output = mds.permute(packed_input);

                let avx512_output = packed_output.map(|x| x.0[0]);
                assert_eq!(avx512_output, expected);
            }
        };
    }

    test_avx512_mds!(test_avx512_mds_width_8, 8);
    test_avx512_mds!(test_avx512_mds_width_12, 12);
    test_avx512_mds!(test_avx512_mds_width_16, 16);
    test_avx512_mds!(test_avx512_mds_width_24, 24);
}
