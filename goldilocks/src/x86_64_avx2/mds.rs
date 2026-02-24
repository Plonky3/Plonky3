use p3_mds::MdsPermutation;
use p3_mds::util::apply_circulant;
use p3_symmetric::Permutation;

use crate::x86_64_avx2::packing::PackedGoldilocksAVX2;
use crate::{
    MATRIX_CIRC_MDS_8_SML_ROW, MATRIX_CIRC_MDS_12_SML_ROW, MATRIX_CIRC_MDS_16_SML_ROW,
    MATRIX_CIRC_MDS_24_GOLDILOCKS, MdsMatrixGoldilocks,
};
const fn convert_array<const N: usize>(arr: [i64; N]) -> [u64; N] {
    let mut result: [u64; N] = [0; N];
    let mut i = 0;
    while i < N {
        result[i] = arr[i] as u64;
        i += 1;
    }
    result
}

impl Permutation<[PackedGoldilocksAVX2; 8]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX2; 8]) -> [PackedGoldilocksAVX2; 8] {
        const MATRIX_CIRC_MDS_8_SML_ROW_U64: [u64; 8] = convert_array(MATRIX_CIRC_MDS_8_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_8_SML_ROW_U64, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX2, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX2; 12]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX2; 12]) -> [PackedGoldilocksAVX2; 12] {
        const MATRIX_CIRC_MDS_12_SML_ROW_U64: [u64; 12] = convert_array(MATRIX_CIRC_MDS_12_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_12_SML_ROW_U64, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX2, 12> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX2; 16]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX2; 16]) -> [PackedGoldilocksAVX2; 16] {
        const MATRIX_CIRC_MDS_16_SML_ROW_U64: [u64; 16] = convert_array(MATRIX_CIRC_MDS_16_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_16_SML_ROW_U64, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX2, 16> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX2; 24]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX2; 24]) -> [PackedGoldilocksAVX2; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_GOLDILOCKS, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX2, 24> for MdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{Goldilocks, MdsMatrixGoldilocks, PackedGoldilocksAVX2};

    fn test_avx2_mds_width<const WIDTH: usize>() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mds = MdsMatrixGoldilocks;

        let input: [Goldilocks; WIDTH] = rng.random();

        let expected = mds.permute(input);

        let packed_input = input.map(Into::<PackedGoldilocksAVX2>::into);
        let packed_output = mds.permute(packed_input);

        let avx2_output = packed_output.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }

    #[test]
    fn test_avx2_mds_width_8() {
        test_avx2_mds_width::<8>();
    }

    #[test]
    fn test_avx2_mds_width_12() {
        test_avx2_mds_width::<12>();
    }

    #[test]
    fn test_avx2_mds_width_16() {
        test_avx2_mds_width::<16>();
    }

    #[test]
    fn test_avx2_mds_width_24() {
        test_avx2_mds_width::<24>();
    }
}
