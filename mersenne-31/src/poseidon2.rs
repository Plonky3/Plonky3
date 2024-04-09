use p3_field::PrimeField32;
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{from_u62, to_mersenne31_array, Mersenne31};

// Two optimised diffusion matrices for Mersenne31/16:

// Mersenne31:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
// Power of 2 entries: [-2,  1,   2,   4,   8,  16,  32,  64, 128, 256, 1024, 4096, 8192, 16384, 32768, 65536]
//                   = [?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^13,  2^14,  2^15, 2^16]

const MATRIX_DIAG_16_MERSENNE31_U32: [u32; 16] = [
    Mersenne31::ORDER_U32 - 2,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    1024,
    4096,
    8192,
    16384,
    32768,
    65536,
];

pub const MATRIX_DIAG_16_MERSENNE31: [Mersenne31; 16] =
    to_mersenne31_array(MATRIX_DIAG_16_MERSENNE31_U32);

// We make use of the fact that most entries are a power of 2.
// Note that this array is 1 element shorter than MATRIX_DIAG_16_MERSENNE31 as we do not include the first element.
const MATRIX_DIAG_16_MONTY_SHIFTS: [u64; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixMersenne31;

impl Permutation<[Mersenne31; 16]> for DiffusionMatrixMersenne31 {
    #[inline]
    fn permute_mut(&self, state: &mut [Mersenne31; 16]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (Mersenne31::ORDER_U32 - state[0].value) as u64;
        state[0] = from_u62(s0);
        for i in 1..16 {
            let si = full_sum + ((state[i].value as u64) << MATRIX_DIAG_16_MONTY_SHIFTS[i - 1]);
            state[i] = from_u62(si);
        }
    }
}

impl DiffusionPermutation<Mersenne31, 16> for DiffusionMatrixMersenne31 {}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

    use super::*;

    type F = Mersenne31;

    // Generate some fixed random constants using Sage.
    // These generated using:
    // set_random_seed(246810)
    // print([vector([ZZ.random_element(2**31) for _ in range(16)]) for _ in range(8)])
    // print(vector([ZZ.random_element(2**31) for _ in range(14)]))
    const EXTRNAL_POSEIDON2_CONSTANTS: [[u32; 16]; 8] = [
        [
            1777708182, 1721401758, 2112139575, 970766477, 1003159146, 2109481055, 1591645666,
            1253081731, 147790673, 1795993607, 418185859, 1354578103, 1652934702, 1108743982,
            1435244566, 1543814404,
        ],
        [
            654930200, 351316337, 1963308868, 38861083, 879582739, 2132987718, 1163315387,
            1362831467, 654205465, 1015011465, 1366066774, 1705258057, 1563940060, 1724600887,
            917326561, 638966780,
        ],
        [
            3384696, 1302649031, 2108321185, 457987989, 467145221, 2036332892, 719635900,
            618708116, 2116382241, 601241247, 700313800, 1175204338, 1430530974, 396015992,
            215098263, 960210005,
        ],
        [
            1078243556, 737500799, 157307924, 1991287991, 296086783, 86954821, 1631858947,
            1813358481, 2017811068, 1864361777, 1809775679, 584697386, 1396011569, 1656991903,
            157756641, 1571003831,
        ],
        [
            2109082198, 1282673068, 331533340, 155427572, 1362775299, 1204999857, 868494283,
            1084220124, 148106575, 806464112, 517366790, 1706152613, 93013513, 1937305141,
            1140571467, 1885203418,
        ],
        [
            1835920144, 1206936349, 3631345, 1673506215, 142350226, 991116072, 795722507,
            2014023057, 1240042960, 617506984, 1572505983, 883145032, 284483487, 393762478,
            1370696701, 1548190401,
        ],
        [
            593717366, 831183373, 1349533916, 132932556, 919497294, 1319399919, 1455908113,
            1518325178, 466178793, 1843875951, 820181398, 358956349, 547431065, 1169772005,
            1176952676, 1315938663,
        ],
        [
            439110515, 866287879, 675258191, 1182462343, 1663930657, 1669852806, 458624943,
            1671116203, 157114180, 601492859, 482098945, 725305402, 2121687200, 207978448,
            422038674, 1593677019,
        ],
    ];
    const INTERNAL_POSEIDON2_CONSTANTS: [u32; 14] = [
        439688492, 778564230, 1348784034, 28869688, 674449982, 1063896158, 215562190, 693226089,
        1852238031, 569836680, 995240347, 978793598, 1217858362, 357939656,
    ];

    // Our Poseidon2 Implementation
    fn poseidon2_mersenne31_width_16(input: &mut [F; 16]) {
        const WIDTH: usize = 16;
        const D: u64 = 5;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 14;

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            F,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            EXTRNAL_POSEIDON2_CONSTANTS
                .map(to_mersenne31_array)
                .to_vec(),
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            to_mersenne31_array(INTERNAL_POSEIDON2_CONSTANTS).to_vec(),
            DiffusionMatrixMersenne31,
        );

        poseidon2.permute_mut(input);
    }

    /// Test on the constant 0 input.
    #[test]
    fn test_poseidon2_width_16_zeroes() {
        let mut input: [F; 16] = [F::zero(); 16];

        let expected: [F; 16] = [
            1040993253, 2058700579, 511363496, 489533323, 208503827, 675841613, 904681360,
            595986756, 1739638800, 927645969, 1828588418, 1320272864, 1414251048, 1145941116,
            179329634, 872143503,
        ]
        .map(F::from_canonical_u32);
        poseidon2_mersenne31_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on the input [0, 1, ..., 15]
    #[test]
    fn test_poseidon2_width_16_range() {
        let mut input: [F; 16] = array::from_fn(|i| F::from_wrapped_u32(i as u32));

        let expected: [F; 16] = [
            561581441, 1097299251, 1959749978, 9711246, 1447531064, 1441858718, 1041864964,
            1028880727, 1652126221, 1143650501, 281857854, 1999066545, 822538192, 1276689616,
            1185513339, 2049007413,
        ]
        .map(F::from_canonical_u32);
        poseidon2_mersenne31_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            1179785652, 1291567559, 66272299, 471640172, 653876821, 478855335, 871063984,
            540251327, 1506944720, 1403776782, 770420443, 126472305, 1535928603, 1017977016,
            818646757, 359411429,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 16] = [
            483475180, 1072944079, 1734362791, 508307352, 67641563, 1167529837, 2106074116,
            1172761349, 1446415876, 815280570, 1541562292, 1421722086, 248493354, 404752383,
            1097623117, 1171663837,
        ]
        .map(F::from_canonical_u32);

        poseidon2_mersenne31_width_16(&mut input);
        assert_eq!(input, expected);
    }
}
