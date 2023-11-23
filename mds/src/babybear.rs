//! MDS matrices over the BabyBear field, and permutations defined by them.
//!
//! NB: Not all sizes have fast implementations of their permutations.
//! Supported sizes: 8, 12, 16, 24, 32, 64.
//! Sizes 8 and 12 are from Plonky2. Other sizes are from Ulrich Haböck's database.

use p3_baby_bear::BabyBear;
use p3_dft::Radix2Bowers;
use p3_symmetric::Permutation;

use crate::util::{
    apply_circulant, apply_circulant_12_sml, apply_circulant_8_sml, apply_circulant_fft,
    first_row_to_first_col,
};
use crate::MdsPermutation;

#[derive(Clone, Default)]
pub struct MdsMatrixBabyBear;

const FFT_ALGO: Radix2Bowers = Radix2Bowers;

impl Permutation<[BabyBear; 8]> for MdsMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 8]) -> [BabyBear; 8] {
        apply_circulant_8_sml(input)
    }

    fn permute_mut(&self, input: &mut [BabyBear; 8]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BabyBear, 8> for MdsMatrixBabyBear {}

impl Permutation<[BabyBear; 12]> for MdsMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 12]) -> [BabyBear; 12] {
        apply_circulant_12_sml(input)
    }

    fn permute_mut(&self, input: &mut [BabyBear; 12]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BabyBear, 12> for MdsMatrixBabyBear {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_16_BABYBEAR: [u64; 16] = [
    0x0780_1000, 0x4ACA_AC32, 0x6A70_9B76, 0x2041_3E94,
    0x0092_8499, 0x31C3_4CA3, 0x03BB_C192, 0x3F20_868B,
    0x257F_FAAB, 0x5F05_F559, 0x55B4_3EA9, 0x2BC6_59ED,
    0x2C6D_7501, 0x1D11_0184, 0x0E1F_608D, 0x2032_F0C6,
];

impl Permutation<[BabyBear; 16]> for MdsMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 16]) -> [BabyBear; 16] {
        const ENTRIES: [u64; 16] = first_row_to_first_col(&MATRIX_CIRC_MDS_16_BABYBEAR);
        apply_circulant_fft(FFT_ALGO, ENTRIES, &input)
    }

    fn permute_mut(&self, input: &mut [BabyBear; 16]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BabyBear, 16> for MdsMatrixBabyBear {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_24_BABYBEAR: [u64; 24] = [
    0x2D0A_AAAB, 0x6485_0517, 0x17F5_551D, 0x04EC_BEB5,
    0x6D91_A8D5, 0x6070_3026, 0x18D6_F3CA, 0x7296_01A7,
    0x77CD_A9E2, 0x3C0F_5038, 0x26D5_2A61, 0x0360_405D,
    0x68FC_71C8, 0x2495_A71D, 0x5D57_AFC2, 0x1689_DD98,
    0x3C2C_3DBE, 0x0C23_DC41, 0x0524_C7F2, 0x6BE4_DF69,
    0x0A6E_572C, 0x5C77_90FA, 0x17E1_18F6, 0x0878_A07F,
];

impl Permutation<[BabyBear; 24]> for MdsMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 24]) -> [BabyBear; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_BABYBEAR, input)
    }

    fn permute_mut(&self, input: &mut [BabyBear; 24]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BabyBear, 24> for MdsMatrixBabyBear {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_32_BABYBEAR: [u64; 32] = [
    0x0BC0_0000, 0x2BED_8F81, 0x337E_0652, 0x4C45_35D1,
    0x4AF2_DC32, 0x2DB4_050F, 0x676A_7CE3, 0x3A06_B68E,
    0x5E95_C1B1, 0x2C5F_54A0, 0x2332_F13D, 0x58E7_57F1,
    0x3AA6_DCCE, 0x607E_E630, 0x4ED5_7FF0, 0x6E08_555B,
    0x4C15_5556, 0x587F_D0CE, 0x462F_1551, 0x032A_43CC,
    0x5E2E_43EA, 0x7160_9B02, 0x0ED9_7E45, 0x562C_A7E9,
    0x2CB7_0B1D, 0x4E94_1E23, 0x174A_61C1, 0x117A_9426,
    0x7356_2137, 0x5459_6086, 0x487C_560B, 0x68A4_ACAB,
];

impl Permutation<[BabyBear; 32]> for MdsMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 32]) -> [BabyBear; 32] {
        const ENTRIES: [u64; 32] = first_row_to_first_col(&MATRIX_CIRC_MDS_32_BABYBEAR);
        apply_circulant_fft(FFT_ALGO, ENTRIES, &input)
    }

    fn permute_mut(&self, input: &mut [BabyBear; 32]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BabyBear, 32> for MdsMatrixBabyBear {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_64_BABYBEAR: [u64; 64] = [
    0x3957_7778, 0x0072_F4E1, 0x0B1B_8404, 0x041E_9C88,
    0x32D2_2F9F, 0x4E4B_F946, 0x20C7_B6D7, 0x0587_C267,
    0x5587_7229, 0x4D18_6EC4, 0x4A19_FD23, 0x1A64_A20F,
    0x2965_CA4D, 0x16D9_8A5A, 0x471E_544A, 0x193D_5C8B,
    0x6E66_DF0C, 0x28BF_1F16, 0x26DB_0BC8, 0x5B06_CDDB,
    0x100D_CCA2, 0x65C2_68AD, 0x199F_09E7, 0x36BA_04BE,
    0x06C3_93F2, 0x51B0_6DFD, 0x6951_B0C4, 0x6683_A4C2,
    0x3B53_D11B, 0x26E5_134C, 0x45A5_F1C5, 0x6F4D_2433,
    0x3CE2_D82E, 0x3630_9A7D, 0x3DD9_B459, 0x6805_1E4C,
    0x5C3A_A720, 0x1164_0517, 0x0634_D995, 0x1B0F_6406,
    0x72A1_8430, 0x2651_3CC5, 0x67C0_B93C, 0x548A_B4A3,
    0x6395_D20D, 0x3E5D_BC41, 0x332A_F630, 0x3C5D_DCB3,
    0x0AA9_5792, 0x66EB_5492, 0x3F78_DDDC, 0x5AC4_1627,
    0x16CD_5124, 0x3564_DA96, 0x4618_67C9, 0x157B_4E11,
    0x1AA4_86C8, 0x0C50_95A9, 0x3833_C0C6, 0x008F_EBA5,
    0x52EC_BE2E, 0x1D17_8A67, 0x58B3_C04B, 0x6E95_CB51,
];

impl Permutation<[BabyBear; 64]> for MdsMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 64]) -> [BabyBear; 64] {
        const ENTRIES: [u64; 64] = first_row_to_first_col(&MATRIX_CIRC_MDS_64_BABYBEAR);
        apply_circulant_fft(FFT_ALGO, ENTRIES, &input)
    }

    fn permute_mut(&self, input: &mut [BabyBear; 64]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BabyBear, 64> for MdsMatrixBabyBear {}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;

    use super::MdsMatrixBabyBear;

    #[test]
    fn babybear8() {
        let input: [BabyBear; 8] = [
            391_474_477, 1_174_409_341, 666_967_492, 1_852_498_830, 1_801_235_316, 820_595_865, 585_587_525,
            1_348_326_858,
        ]
        .map(BabyBear::from_canonical_u64);

        let output = MdsMatrixBabyBear.permute(input);

        let expected: [BabyBear; 8] = [
            504_128_309, 1_915_631_392, 1_485_872_679, 1_192_473_153, 1_425_656_962, 634_837_116, 1_385_055_496,
            795_071_948,
        ]
        .map(BabyBear::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn babybear12() {
        let input: [BabyBear; 12] = [
            918_423_259, 673_549_090, 364_157_140, 9_832_898, 493_922_569, 1_171_855_651, 246_075_034, 1_542_167_926,
            1_787_615_541, 1_696_819_900, 1_884_530_130, 422_386_768,
        ]
        .map(BabyBear::from_canonical_u64);

        let output = MdsMatrixBabyBear.permute(input);

        let expected: [BabyBear; 12] = [
            772_551_966, 2_009_480_750, 430_187_688, 1_134_406_614, 351_991_333, 1_100_020_355, 777_201_441,
            109_334_185, 2_000_422_332, 226_001_108, 1_763_301_937, 631_922_975,
        ]
        .map(BabyBear::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn babybear16() {
        let input: [BabyBear; 16] = [
            1_983_708_094, 1_477_844_074, 1_638_775_686, 98_517_138, 70_746_308, 968_700_066, 275_567_720,
            1_359_144_511, 960_499_489, 1_215_199_187, 474_302_783, 79_320_256, 1_923_147_803, 1_197_733_438,
            1_638_511_323, 303_948_902,
        ]
        .map(BabyBear::from_canonical_u64);

        let output = MdsMatrixBabyBear.permute(input);

        let expected: [BabyBear; 16] = [
            556_401_834, 683_220_320, 1_810_464_928, 1_169_932_617, 638_040_805, 1_006_828_793, 1_808_829_293,
            1_614_898_838, 23_062_004, 622_101_715, 967_448_737, 519_782_760, 579_530_259, 157_817_176,
            1_439_772_057, 54_268_721,
        ]
        .map(BabyBear::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn babybear24() {
        let input: [BabyBear; 24] = [
            1_307_148_929, 1_603_957_607, 1_515_498_600, 1_412_393_512, 785_287_979, 988_718_522, 1_750_345_556,
            853_137_995, 534_387_281, 930_390_055, 1_600_030_977, 903_985_158, 1_141_020_507, 636_889_442,
            966_037_834, 1_778_991_639, 1_440_427_266, 1_379_431_959, 853_403_277, 959_593_575, 733_455_867,
            908_584_009, 817_124_993, 418_826_476,
        ]
        .map(BabyBear::from_canonical_u64);

        let output = MdsMatrixBabyBear.permute(input);

        let expected: [BabyBear; 24] = [
            1_537_871_777, 1_626_055_274, 1_705_000_179, 1_426_678_258, 1_688_760_658, 1_347_225_494, 1_291_221_794,
            1_224_656_589, 1_791_446_853, 1_978_133_881, 1_820_380_039, 1_366_829_700, 27_479_566, 409_595_531,
            1_223_347_944, 1_752_750_033, 594_548_873, 1_447_473_111, 1_385_412_872, 1_111_945_102, 1_366_585_917,
            138_866_947, 1_326_436_332, 656_898_133,
        ]
        .map(BabyBear::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn babybear32() {
        let input: [BabyBear; 32] = [
            1_346_087_634, 1_511_946_000, 1_883_470_964, 54_906_057, 233_060_279, 5_304_922, 1_881_494_193,
            743_728_289, 404_047_361, 1_148_556_479, 144_976_634, 1_726_343_008, 29_659_471, 1_350_407_160,
            1_636_652_429, 385_978_955, 327_649_601, 1_248_138_459, 1_255_358_242, 84_164_877, 1_005_571_393,
            1_713_215_328, 72_913_800, 1_683_904_606, 904_763_213, 316_800_515, 656_395_998, 788_184_609,
            1_824_512_025, 1_177_399_063, 1_358_745_087, 444_151_496,
        ]
        .map(BabyBear::from_canonical_u64);

        let output = MdsMatrixBabyBear.permute(input);

        let expected: [BabyBear; 32] = [
            1_359_576_919, 1_657_405_784, 1_031_581_836, 212_090_105, 699_048_671, 877_916_349, 205_627_787,
            1_211_567_750, 210_807_569, 1_696_391_051, 558_468_987, 161_148_427, 304_343_518, 76_611_896,
            532_792_005, 1_963_649_139, 1_283_500_358, 250_848_292, 1_109_842_541, 2_007_388_683, 433_801_252,
            1_189_712_914, 626_158_024, 1_436_409_738, 456_315_160, 1_836_818_120, 1_645_024_941, 925_447_491,
            1_599_571_860, 1_055_439_714, 353_537_136, 379_644_130,
        ]
        .map(BabyBear::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn babybear64() {
        let input: [BabyBear; 64] = [
            1_931_358_930, 1_322_576_114, 1_658_000_717, 134_388_215, 1_517_892_791, 1_486_447_670, 93_570_662,
            898_466_034, 1_576_905_917, 283_824_713, 1_433_559_150, 1_730_678_909, 155_340_881, 1_978_472_263,
            1_980_644_590, 1_814_040_165, 654_743_892, 849_954_227, 323_176_597, 146_970_735, 252_703_735,
            1_856_579_399, 162_749_290, 986_745_196, 352_038_183, 1_239_527_508, 828_473_247, 1_184_743_572,
            1_017_249_065, 36_804_843, 1_378_131_210, 1_286_724_687, 596_095_979, 1_916_924_908, 528_946_791,
            397_247_884, 23_477_278, 299_412_064, 415_288_430, 935_825_754, 1_218_003_667, 1_954_592_289,
            1_594_612_673, 664_096_455, 958_392_778, 497_208_288, 1_544_504_580, 1_829_423_324, 956_111_902,
            458_327_015, 1_736_664_598, 430_977_734, 599_887_171, 1_100_074_154, 1_197_653_896, 427_838_651,
            466_509_871, 1_236_918_100, 940_670_246, 1_421_951_147, 255_557_957, 1_374_188_100, 315_300_068,
            623_354_170,
        ]
        .map(BabyBear::from_canonical_u64);

        let output = MdsMatrixBabyBear.permute(input);

        let expected: [BabyBear; 64] = [
            442_300_274, 756_862_170, 167_612_495, 1_103_336_044, 546_496_433, 1_211_822_920, 329_094_196,
            1_334_376_959, 944_085_937, 977_350_947, 1_445_060_130, 918_469_957, 800_346_119, 1_957_918_170,
            739_098_112, 1_862_817_833, 1_831_589_884, 1_673_860_978, 698_081_523, 1_128_978_338, 387_929_536,
            1_106_772_486, 1_367_460_469, 1_911_237_185, 362_669_171, 819_949_894, 1_801_786_287, 1_943_505_026,
            586_738_185, 996_076_080, 1_641_277_705, 1_680_239_311, 1_005_815_192, 63_087_470, 593_010_310,
            364_673_774, 543_368_618, 1_576_179_136, 47_618_763, 1_990_080_335, 1_608_655_220, 499_504_830,
            861_863_262, 765_074_289, 139_277_832, 1_139_970_138, 1_510_286_607, 244_269_525, 43_042_067,
            119_733_624, 1_314_663_255, 893_295_811, 1_444_902_994, 914_930_267, 1_675_139_862, 1_148_717_487,
            1_601_328_192, 534_383_401, 296_215_929, 1_924_587_380, 1_336_639_141, 34_897_994, 2_005_302_060,
            1_780_337_352,
        ]
        .map(BabyBear::from_canonical_u64);

        assert_eq!(output, expected);
    }
}
