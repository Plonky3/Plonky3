use p3_mds::MdsPermutation;
use p3_mds::util::apply_circulant;
use p3_symmetric::Permutation;
use crate::MdsMatrixGoldilocks;
use crate::x86_64_avx2::packing::PackedAvx2Goldilocks;
use crate::{MATRIX_CIRC_MDS_8_SML_ROW, MATRIX_CIRC_MDS_12_SML_ROW, MATRIX_CIRC_MDS_16_SML_ROW, MATRIX_CIRC_MDS_24_GOLDILOCKS};
const fn convert_array<const N: usize>(arr: [i64; N]) -> [u64; N] {
    let mut result: [u64; N] = [0; N];
    let mut i = 0;
    while i < N {
        result[i] = arr[i] as u64;
        i += 1;
    }
    result
}

impl Permutation<[PackedAvx2Goldilocks; 8]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedAvx2Goldilocks; 8]) -> [PackedAvx2Goldilocks; 8] {
        const MATRIX_CIRC_MDS_8_SML_ROW_U64: [u64;8] = convert_array(MATRIX_CIRC_MDS_8_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_8_SML_ROW_U64, input)
    }

    fn permute_mut(&self, input: &mut [PackedAvx2Goldilocks; 8]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<PackedAvx2Goldilocks, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedAvx2Goldilocks; 12]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedAvx2Goldilocks; 12]) -> [PackedAvx2Goldilocks; 12] {
        const MATRIX_CIRC_MDS_12_SML_ROW_U64: [u64;12] = convert_array(MATRIX_CIRC_MDS_12_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_12_SML_ROW_U64, input)
    }

    fn permute_mut(&self, input: &mut [PackedAvx2Goldilocks; 12]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<PackedAvx2Goldilocks, 12> for MdsMatrixGoldilocks {}

impl Permutation<[PackedAvx2Goldilocks; 16]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedAvx2Goldilocks; 16]) -> [PackedAvx2Goldilocks; 16] {
        const MATRIX_CIRC_MDS_16_SML_ROW_U64: [u64;16] = convert_array(MATRIX_CIRC_MDS_16_SML_ROW);
        apply_circulant(&MATRIX_CIRC_MDS_16_SML_ROW_U64, input)
    }

    fn permute_mut(&self, input: &mut [PackedAvx2Goldilocks; 16]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<PackedAvx2Goldilocks, 16> for MdsMatrixGoldilocks {}


impl Permutation<[PackedAvx2Goldilocks; 24]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedAvx2Goldilocks; 24]) -> [PackedAvx2Goldilocks; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_GOLDILOCKS, input)
    }

    fn permute_mut(&self, input: &mut [PackedAvx2Goldilocks; 24]) {
        *input = self.permute(*input);
    }
}

impl MdsPermutation<PackedAvx2Goldilocks, 24> for MdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use p3_field::AbstractField;
    use p3_poseidon::Poseidon;
    use p3_symmetric::Permutation;
    use crate::{Goldilocks, MdsMatrixGoldilocks};
    use crate::PackedAvx2Goldilocks;

    #[test]
    fn test_avx2_poseidon_width_8() {
        let mut rng = rand::thread_rng();
        type F = Goldilocks;
        type Perm = Poseidon<F, MdsMatrixGoldilocks, 8, 7>;
        let poseidon = Perm::new_from_rng(4, 22, MdsMatrixGoldilocks, &mut rand::thread_rng());

        let input: [F; 8] = rng.gen();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedAvx2Goldilocks::from_f);
        poseidon.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }

    #[test]
    fn test_avx2_poseidon_width_12() {
        let mut rng = rand::thread_rng();
        type F = Goldilocks;
        type Perm = Poseidon<F, MdsMatrixGoldilocks, 12, 7>;
        let poseidon = Perm::new_from_rng(4, 22, MdsMatrixGoldilocks, &mut rand::thread_rng());

        let input: [F; 12] = rng.gen();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedAvx2Goldilocks::from_f);
        poseidon.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }

    #[test]
    fn test_avx2_poseidon_width_16() {
        let mut rng = rand::thread_rng();
        type F = Goldilocks;
        type Perm = Poseidon<F, MdsMatrixGoldilocks, 16, 7>;
        let poseidon = Perm::new_from_rng(4, 22, MdsMatrixGoldilocks, &mut rand::thread_rng());

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedAvx2Goldilocks::from_f);
        poseidon.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }

    #[test]
    fn test_avx2_poseidon_width_24() {
        let mut rng = rand::thread_rng();
        type F = Goldilocks;
        type Perm = Poseidon<F, MdsMatrixGoldilocks, 24, 7>;
        let poseidon = Perm::new_from_rng(4, 22, MdsMatrixGoldilocks, &mut rand::thread_rng());

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedAvx2Goldilocks::from_f);
        poseidon.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }
}