//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
//!
//! For the diffusion matrix, 1 + Diag(V), we perform a search to find an optimized
//! vector V composed of elements with efficient multiplication algorithms in AVX2/AVX512/NEON.
//!
//! This leads to using small values (e.g. 1, 2, 3, 4) where multiplication is implemented using addition
//! and inverse powers of 2 where it is possible to avoid monty reductions.
//! Additionally, for technical reasons, having the first entry be -2 is useful.
//!
//! Optimized Diagonal for KoalaBear16:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
//! Optimized Diagonal for KoalaBear24:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
//! See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.

use p3_field::PrimeCharacteristicRing;
use p3_monty_31::{
    GenericPoseidon2LinearLayersMonty31, InternalLayerBaseParameters, InternalLayerParameters,
    Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};
use p3_poseidon2::{ExternalLayerConstants, Poseidon2};

use crate::{KoalaBear, KoalaBearParameters};

pub type Poseidon2InternalLayerKoalaBear<const WIDTH: usize> =
    Poseidon2InternalLayerMonty31<KoalaBearParameters, WIDTH, KoalaBearInternalLayerParameters>;

pub type Poseidon2ExternalLayerKoalaBear<const WIDTH: usize> =
    Poseidon2ExternalLayerMonty31<KoalaBearParameters, WIDTH>;

/// Degree of the chosen permutation polynomial for KoalaBear, used as the Poseidon2 S-Box.
///
/// As p - 1 = 127 * 2^{24} we have a lot of choice in degree D satisfying gcd(p - 1, D) = 1.
/// Experimentation suggests that the optimal choice is the smallest available one, namely 3.
const KOALABEAR_S_BOX_DEGREE: u64 = 3;

/// An implementation of the Poseidon2 hash function specialised to run on the current architecture.
///
/// It acts on arrays of the form either `[KoalaBear::Packing; WIDTH]` or `[KoalaBear; WIDTH]`. For speed purposes,
/// wherever possible, input arrays should of the form `[KoalaBear::Packing; WIDTH]`.
pub type Poseidon2KoalaBear<const WIDTH: usize> = Poseidon2<
    KoalaBear,
    Poseidon2ExternalLayerKoalaBear<WIDTH>,
    Poseidon2InternalLayerKoalaBear<WIDTH>,
    WIDTH,
    KOALABEAR_S_BOX_DEGREE,
>;

/// An implementation of the matrix multiplications in the internal and external layers of Poseidon2.
///
/// This can act on `[A; WIDTH]` for any ring implementing `Algebra<BabyBear>`.
/// If you have either `[KoalaBear::Packing; WIDTH]` or `[KoalaBear; WIDTH]` it will be much faster
/// to use `Poseidon2KoalaBear<WIDTH>` instead of building a Poseidon2 permutation using this.
pub type GenericPoseidon2LinearLayersKoalaBear =
    GenericPoseidon2LinearLayersMonty31<KoalaBearParameters, KoalaBearInternalLayerParameters>;

/// Initial round constants for the 16-width Poseidon2 external layer on KoalaBear.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub const KOALABEAR_RC16_EXTERNAL_INITIAL: [[KoalaBear; 16]; 4] = KoalaBear::new_2d_array([
    [
        2128964168, 288780357, 316938561, 2126233899, 426817493, 1714118888, 1045008582,
        1738510837, 889721787, 8866516, 681576474, 419059826, 1596305521, 1583176088, 1584387047,
        1529751136,
    ],
    [
        1863858111, 1072044075, 517831365, 1464274176, 1138001621, 428001039, 245709561,
        1641420379, 1365482496, 770454828, 693167409, 757905735, 136670447, 436275702, 525466355,
        1559174242,
    ],
    [
        1030087950, 869864998, 322787870, 267688717, 948964561, 740478015, 679816114, 113662466,
        2066544572, 1744924186, 367094720, 1380455578, 1842483872, 416711434, 1342291586,
        1692058446,
    ],
    [
        1493348999, 1113949088, 210900530, 1071655077, 610242121, 1136339326, 2020858841,
        1019840479, 678147278, 1678413261, 1361743414, 61132629, 1209546658, 64412292, 1936878279,
        1980661727,
    ],
]);

/// Final round constants for the 16-width Poseidon2's external layer on KoalaBear.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub const KOALABEAR_RC16_EXTERNAL_FINAL: [[KoalaBear; 16]; 4] = KoalaBear::new_2d_array([
    [
        1423960925, 2101391318, 1915532054, 275400051, 1168624859, 1141248885, 356546469,
        1165250474, 1320543726, 932505663, 1204226364, 1452576828, 1774936729, 926808140,
        1184948056, 1186493834,
    ],
    [
        843181003, 185193011, 452207447, 510054082, 1139268644, 630873441, 669538875, 462500858,
        876500520, 1214043330, 383937013, 375087302, 636912601, 307200505, 390279673, 1999916485,
    ],
    [
        1518476730, 1606686591, 1410677749, 1581191572, 1004269969, 143426723, 1747283099,
        1016118214, 1749423722, 66331533, 1177761275, 1581069649, 1851371119, 852520128,
        1499632627, 1820847538,
    ],
    [
        150757557, 884787840, 619710451, 1651711087, 505263814, 212076987, 1482432120, 1458130652,
        382871348, 417404007, 2066495280, 1996518884, 902934924, 582892981, 1337064375, 1199354861,
    ],
]);

/// Round constants for the 16-width Poseidon2's internal layer on KoalaBear.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub const KOALABEAR_RC16_INTERNAL: [KoalaBear; 20] = KoalaBear::new_array([
    2102596038, 1533193853, 1436311464, 2012303432, 839997195, 1225781098, 2011967775, 575084315,
    1309329169, 786393545, 995788880, 1702925345, 1444525226, 908073383, 1811535085, 1531002367,
    1635653662, 1585100155, 867006515, 879151050,
]);

/// A default Poseidon2 for KoalaBear using the round constants from the original specification.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub fn default_koalabear_poseidon2_16() -> Poseidon2KoalaBear<16> {
    Poseidon2::new(
        ExternalLayerConstants::new(
            KOALABEAR_RC16_EXTERNAL_INITIAL.to_vec(),
            KOALABEAR_RC16_EXTERNAL_FINAL.to_vec(),
        ),
        KOALABEAR_RC16_INTERNAL.to_vec(),
    )
}

/// Initial round constants for the 24-width Poseidon2 external layer on KoalaBear.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub const KOALABEAR_RC24_EXTERNAL_INITIAL: [[KoalaBear; 24]; 4] = KoalaBear::new_2d_array([
    [
        487143900, 1829048205, 1652578477, 646002781, 1044144830, 53279448, 1519499836, 22697702,
        1768655004, 230479744, 1484895689, 705130286, 1429811285, 1695785093, 1417332623,
        1115801016, 1048199020, 878062617, 738518649, 249004596, 1601837737, 24601614, 245692625,
        364803730,
    ],
    [
        1857019234, 1906668230, 1916890890, 835590867, 557228239, 352829675, 515301498, 973918075,
        954515249, 1142063750, 1795549558, 608869266, 1850421928, 2028872854, 1197543771,
        1027240055, 1976813168, 963257461, 652017844, 2113212249, 213459679, 90747280, 1540619478,
        324138382,
    ],
    [
        1377377119, 294744504, 512472871, 668081958, 907306515, 518526882, 1907091534, 1152942192,
        1572881424, 720020214, 729527057, 1762035789, 86171731, 205890068, 453077400, 1201344594,
        986483134, 125174298, 2050269685, 1895332113, 749706654, 40566555, 742540942, 1735551813,
    ],
    [
        162985276, 1943496073, 1469312688, 703013107, 1979485151, 1278193166, 548674995,
        2118718736, 749596440, 1476142294, 1293606474, 918523452, 890353212, 1691895663,
        1932240646, 1180911992, 86098300, 1592168978, 895077289, 724819849, 1697986774, 1608418116,
        1083269213, 691256798,
    ],
]);

/// Final round constants for the 24-width Poseidon2's external layer on KoalaBear.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub const KOALABEAR_RC24_EXTERNAL_FINAL: [[KoalaBear; 24]; 4] = KoalaBear::new_2d_array([
    [
        328586442, 1572520009, 1375479591, 322991001, 967600467, 1172861548, 1973891356,
        1503625929, 1881993531, 40601941, 1155570620, 571547775, 1361622243, 1495024047,
        1733254248, 964808915, 763558040, 1887228519, 994888261, 718330940, 213359415, 603124968,
        1038411577, 2099454809,
    ],
    [
        949846777, 630926956, 1168723439, 222917504, 1527025973, 1009157017, 2029957881, 805977836,
        1347511739, 540019059, 589807745, 440771316, 1530063406, 761076336, 87974206, 1412686751,
        1230318064, 514464425, 1469011754, 1770970737, 1510972858, 965357206, 209398053, 778802532,
    ],
    [
        40567006, 1984217577, 1545851069, 879801839, 1611910970, 1215591048, 330802499, 1051639108,
        321036, 511927202, 591603098, 1775897642, 115598532, 278200718, 233743176, 525096211,
        1335507608, 830017835, 1380629279, 560028578, 598425701, 302162385, 567434115, 1859222575,
    ],
    [
        958294793, 1582225556, 1781487858, 1570246000, 1067748446, 526608119, 1666453343,
        1786918381, 348203640, 1860035017, 1489902626, 1904576699, 860033965, 1954077639,
        1685771567, 971513929, 1877873770, 137113380, 520695829, 806829080, 1408699405, 1613277964,
        793223662, 648443918,
    ],
]);

/// Round constants for the 24-width Poseidon2's internal layer on KoalaBear.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub const KOALABEAR_RC24_INTERNAL: [KoalaBear; 23] = KoalaBear::new_array([
    893435011, 403879071, 1363789863, 1662900517, 2043370, 2109755796, 931751726, 2091644718,
    606977583, 185050397, 946157136, 1350065230, 1625860064, 122045240, 880989921, 145137438,
    1059782436, 1477755661, 335465138, 1640704282, 1757946479, 1551204074, 681266718,
]);

/// A default Poseidon2 for KoalaBear using the round constants from the original specification.
///
/// See Poseidon paper for more details: https://eprint.iacr.org/2019/458
/// Sage script used to generate these constants: https://github.com/HorizenLabs/poseidon2/blob/main/poseidon2_rust_params.sage
pub fn default_koalabear_poseidon2_24() -> Poseidon2KoalaBear<24> {
    Poseidon2::new(
        ExternalLayerConstants::new(
            KOALABEAR_RC24_EXTERNAL_INITIAL.to_vec(),
            KOALABEAR_RC24_EXTERNAL_FINAL.to_vec(),
        ),
        KOALABEAR_RC24_INTERNAL.to_vec(),
    )
}

/// Contains data needed to define the internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Default)]
pub struct KoalaBearInternalLayerParameters;

impl InternalLayerBaseParameters<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {
    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul<R: PrimeCharacteristicRing>(state: &mut [R; 16], sum: R) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
        state[1] += sum.clone();
        state[2] = state[2].double() + sum.clone();
        state[3] = state[3].halve() + sum.clone();
        state[4] = sum.clone() + state[4].double() + state[4].clone();
        state[5] = sum.clone() + state[5].double().double();
        state[6] = sum.clone() - state[6].halve();
        state[7] = sum.clone() - (state[7].double() + state[7].clone());
        state[8] = sum.clone() - state[8].double().double();
        state[9] = state[9].div_2exp_u64(8);
        state[9] += sum.clone();
        state[10] = state[10].div_2exp_u64(3);
        state[10] += sum.clone();
        state[11] = state[11].div_2exp_u64(24);
        state[11] += sum.clone();
        state[12] = state[12].div_2exp_u64(8);
        state[12] = sum.clone() - state[12].clone();
        state[13] = state[13].div_2exp_u64(3);
        state[13] = sum.clone() - state[13].clone();
        state[14] = state[14].div_2exp_u64(4);
        state[14] = sum.clone() - state[14].clone();
        state[15] = state[15].div_2exp_u64(24);
        state[15] = sum - state[15].clone();
    }
}

impl InternalLayerBaseParameters<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {
    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul<R: PrimeCharacteristicRing>(state: &mut [R; 24], sum: R) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
        state[1] += sum.clone();
        state[2] = state[2].double() + sum.clone();
        state[3] = state[3].halve() + sum.clone();
        state[4] = sum.clone() + state[4].double() + state[4].clone();
        state[5] = sum.clone() + state[5].double().double();
        state[6] = sum.clone() - state[6].halve();
        state[7] = sum.clone() - (state[7].double() + state[7].clone());
        state[8] = sum.clone() - state[8].double().double();
        state[9] = state[9].div_2exp_u64(8);
        state[9] += sum.clone();
        state[10] = state[10].div_2exp_u64(2);
        state[10] += sum.clone();
        state[11] = state[11].div_2exp_u64(3);
        state[11] += sum.clone();
        state[12] = state[12].div_2exp_u64(4);
        state[12] += sum.clone();
        state[13] = state[13].div_2exp_u64(5);
        state[13] += sum.clone();
        state[14] = state[14].div_2exp_u64(6);
        state[14] += sum.clone();
        state[15] = state[15].div_2exp_u64(24);
        state[15] += sum.clone();
        state[16] = state[16].div_2exp_u64(8);
        state[16] = sum.clone() - state[16].clone();
        state[17] = state[17].div_2exp_u64(3);
        state[17] = sum.clone() - state[17].clone();
        state[18] = state[18].div_2exp_u64(4);
        state[18] = sum.clone() - state[18].clone();
        state[19] = state[19].div_2exp_u64(5);
        state[19] = sum.clone() - state[19].clone();
        state[20] = state[20].div_2exp_u64(6);
        state[20] = sum.clone() - state[20].clone();
        state[21] = state[21].div_2exp_u64(7);
        state[21] = sum.clone() - state[21].clone();
        state[22] = state[22].div_2exp_u64(9);
        state[22] = sum.clone() - state[22].clone();
        state[23] = state[23].div_2exp_u64(24);
        state[23] = sum - state[23].clone();
    }
}

impl InternalLayerParameters<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {}
impl InternalLayerParameters<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = KoalaBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([KB.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = KoalaBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        let expected: [F; 16] = KoalaBear::new_array([
            652590279, 1200629963, 1013089423, 1840372851, 19101828, 561050015, 1714865585,
            994637181, 498949829, 729884572, 1957973925, 263012103, 535029297, 2121808603,
            964663675, 1473622080,
        ]);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([KB.random_element() for t in range(24)]).
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = KoalaBear::new_array([
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 2026927696,
            449439011, 1131357108, 50869465,
        ]);

        let expected: [F; 24] = KoalaBear::new_array([
            3825456, 486989921, 613714063, 282152282, 1027154688, 1171655681, 879344953,
            1090688809, 1960721991, 1604199242, 1329947150, 1535171244, 781646521, 1156559780,
            1875690339, 368140677, 457503063, 304208551, 1919757655, 835116474, 1293372648,
            1254825008, 810923913, 1773631109,
        ]);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 16.
    #[test]
    fn test_generic_internal_linear_layer_16() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mut input1: [F; 16] = rng.random();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().copied().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        KoalaBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        KoalaBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 16.
    #[test]
    fn test_generic_internal_linear_layer_24() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mut input1: [F; 24] = rng.random();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().copied().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        KoalaBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        KoalaBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }
}
