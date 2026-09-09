use p3_field::PrimeCharacteristicRing;
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::{
    mds_circulant_karatsuba_8, mds_circulant_karatsuba_12, mds_circulant_karatsuba_16,
};
use p3_mds::util::{apply_circulant, first_row_to_first_col};
use p3_symmetric::Permutation;

use crate::wasm32_simd128::packing::PackedGoldilocksWasmSimd128;
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

impl Permutation<[PackedGoldilocksWasmSimd128; 8]> for MdsMatrixGoldilocks {
    fn permute(&self, input: [PackedGoldilocksWasmSimd128; 8]) -> [PackedGoldilocksWasmSimd128; 8] {
        const COL: [Goldilocks; 8] = sml_row_to_goldilocks_col(&MATRIX_CIRC_MDS_8_SML_ROW);
        let mut state = input;
        mds_circulant_karatsuba_8(&mut state, &COL);
        state
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 8> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksWasmSimd128; 12]> for MdsMatrixGoldilocks {
    fn permute(
        &self,
        input: [PackedGoldilocksWasmSimd128; 12],
    ) -> [PackedGoldilocksWasmSimd128; 12] {
        const COL: [Goldilocks; 12] = sml_row_to_goldilocks_col(&MATRIX_CIRC_MDS_12_SML_ROW);
        let mut state = input;
        mds_circulant_karatsuba_12(&mut state, &COL);
        state
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 12> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksWasmSimd128; 16]> for MdsMatrixGoldilocks {
    fn permute(
        &self,
        input: [PackedGoldilocksWasmSimd128; 16],
    ) -> [PackedGoldilocksWasmSimd128; 16] {
        const COL: [Goldilocks; 16] = sml_row_to_goldilocks_col(&MATRIX_CIRC_MDS_16_SML_ROW);
        let mut state = input;
        mds_circulant_karatsuba_16(&mut state, &COL);
        state
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 16> for MdsMatrixGoldilocks {}

impl Permutation<[PackedGoldilocksWasmSimd128; 24]> for MdsMatrixGoldilocks {
    fn permute(
        &self,
        input: [PackedGoldilocksWasmSimd128; 24],
    ) -> [PackedGoldilocksWasmSimd128; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_GOLDILOCKS, &input)
    }
}

impl MdsPermutation<PackedGoldilocksWasmSimd128, 24> for MdsMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{Goldilocks, MdsMatrixGoldilocks, PackedGoldilocksWasmSimd128};

    fn check<const WIDTH: usize>(lanes: [[Goldilocks; WIDTH]; 2])
    where
        MdsMatrixGoldilocks:
            Permutation<[Goldilocks; WIDTH]> + Permutation<[PackedGoldilocksWasmSimd128; WIDTH]>,
    {
        let packed =
            core::array::from_fn(|i| PackedGoldilocksWasmSimd128([lanes[0][i], lanes[1][i]]));
        let actual = MdsMatrixGoldilocks.permute(packed);
        for (lane, input) in lanes.into_iter().enumerate() {
            let expected = MdsMatrixGoldilocks.permute(input);
            assert_eq!(actual.map(|x| x.0[lane]), expected);
        }
    }

    macro_rules! test_wasm_mds {
        ($name:ident, $width:literal) => {
            #[test]
            fn $name() {
                let edges = [
                    0,
                    1,
                    (1 << 32) - 1,
                    1 << 63,
                    crate::P - 1,
                    crate::P,
                    crate::P + 1,
                    u64::MAX,
                ];
                for offset in 0..edges.len() {
                    check::<$width>(core::array::from_fn(|lane| {
                        core::array::from_fn(|i| {
                            Goldilocks::new(edges[(offset + i + lane * 3) % edges.len()])
                        })
                    }));
                    check::<$width>([[Goldilocks::new(edges[offset]); $width]; 2]);
                }
                let mut rng = SmallRng::seed_from_u64(1);
                for _ in 0..64 {
                    check::<$width>(core::array::from_fn(|_| {
                        core::array::from_fn(|_| Goldilocks::new(rng.random()))
                    }));
                }
            }
        };
    }

    fn check_poseidon1<const WIDTH: usize>(
        perm: impl Permutation<[Goldilocks; WIDTH]> + Permutation<[PackedGoldilocksWasmSimd128; WIDTH]>,
    ) {
        let mut rng = SmallRng::seed_from_u64(0x905E1);
        for _ in 0..16 {
            let lanes: [[Goldilocks; WIDTH]; 2] =
                core::array::from_fn(|_| core::array::from_fn(|_| Goldilocks::new(rng.random())));
            let packed =
                core::array::from_fn(|i| PackedGoldilocksWasmSimd128([lanes[0][i], lanes[1][i]]));
            let actual = perm.permute(packed);
            for (lane, input) in lanes.into_iter().enumerate() {
                assert_eq!(actual.map(|x| x.0[lane]), perm.permute(input));
            }
        }
    }

    #[test]
    fn packed_poseidon1_matches_scalar() {
        check_poseidon1(crate::poseidon1::default_goldilocks_poseidon1_8());
        check_poseidon1(crate::poseidon1::default_goldilocks_poseidon1_12());
    }

    test_wasm_mds!(test_wasm_mds_width_8, 8);
    test_wasm_mds!(test_wasm_mds_width_12, 12);
    test_wasm_mds!(test_wasm_mds_width_16, 16);
    test_wasm_mds!(test_wasm_mds_width_24, 24);
}
