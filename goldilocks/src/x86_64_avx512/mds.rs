use p3_mds::MdsPermutation;
use p3_mds::util::apply_circulant;
use p3_symmetric::Permutation;

use crate::x86_64_avx512::packing::PackedGoldilocksAVX512;
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

impl Permutation<[PackedGoldilocksAVX512; 8]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 8]) -> [PackedGoldilocksAVX512; 8] {
        const MATRIX_CIRC_MDS_8_SML_ROW_U64: [u64; 8] = convert_array(MATRIX_CIRC_MDS_8_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_8_SML_ROW_U64, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX512, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX512; 12]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 12]) -> [PackedGoldilocksAVX512; 12] {
        const MATRIX_CIRC_MDS_12_SML_ROW_U64: [u64; 12] = convert_array(MATRIX_CIRC_MDS_12_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_12_SML_ROW_U64, &input)
    }
}

impl MdsPermutation<PackedGoldilocksAVX512, 12> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksAVX512; 16]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksAVX512; 16]) -> [PackedGoldilocksAVX512; 16] {
        const MATRIX_CIRC_MDS_16_SML_ROW_U64: [u64; 16] = convert_array(MATRIX_CIRC_MDS_16_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_16_SML_ROW_U64, &input)
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

    fn test_avx512_mds_width<const WIDTH: usize>() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mds = MdsMatrixGoldilocks;

        let input: [Goldilocks; WIDTH] = rng.random();

        let expected = mds.permute(input);

        let packed_input = input.map(Into::<PackedGoldilocksAVX512>::into);
        let packed_output = mds.permute(packed_input);

        let avx512_output = packed_output.map(|x| x.0[0]);
        assert_eq!(avx512_output, expected);
    }

    #[test]
    fn test_avx512_mds_width_8() {
        test_avx512_mds_width::<8>();
    }

    #[test]
    fn test_avx512_mds_width_12() {
        test_avx512_mds_width::<12>();
    }

    #[test]
    fn test_avx512_mds_width_16() {
        test_avx512_mds_width::<16>();
    }

    #[test]
    fn test_avx512_mds_width_24() {
        test_avx512_mds_width::<24>();
    }
}
