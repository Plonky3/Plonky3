//! MDS matrices over the Mersenne31 field, and permutations defined by them.
//!
//! NB: Not all sizes have fast implementations of their permutations.
//! Supported sizes: 8, 12, 16, 32, 64.
//! Sizes 8 and 12 are from Plonky2. Other sizes are from Ulrich Haböck's database.

use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;

use crate::util::{apply_circulant, apply_circulant_12_sml, apply_circulant_8_sml};
use crate::MdsPermutation;

#[derive(Clone, Default)]
pub struct MdsMatrixMersenne31;

impl Permutation<[Mersenne31; 8]> for MdsMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 8]) -> [Mersenne31; 8] {
        apply_circulant_8_sml(input)
    }

    fn permute_mut(&self, input: &mut [Mersenne31; 8]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Mersenne31, 8> for MdsMatrixMersenne31 {}

impl Permutation<[Mersenne31; 12]> for MdsMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 12]) -> [Mersenne31; 12] {
        apply_circulant_12_sml(input)
    }

    fn permute_mut(&self, input: &mut [Mersenne31; 12]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Mersenne31, 12> for MdsMatrixMersenne31 {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_16_MERSENNE31: [u64; 16] = [
    0x327A_CB92, 0x58C9_9138, 0x3AC4_86B5, 0x2512_3B13,
    0x2C74_BDE9, 0x108B_D51A, 0x4E91_1F9D, 0x19DD_8E68,
    0x0622_7198, 0x516E_E062, 0x0F74_2AE6, 0x738B_4216,
    0x7AED_C4EC, 0x653B_794A, 0x4736_6EC7, 0x6D85_346D
];

impl Permutation<[Mersenne31; 16]> for MdsMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 16]) -> [Mersenne31; 16] {
        apply_circulant(&MATRIX_CIRC_MDS_16_MERSENNE31, input)
    }

    fn permute_mut(&self, input: &mut [Mersenne31; 16]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Mersenne31, 16> for MdsMatrixMersenne31 {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_32_MERSENNE31: [u64; 32] = [
    0x1896_DC78, 0x559D_1E29, 0x04EB_D732, 0x3FF4_49D7,
    0x2DB0_E2CE, 0x2677_6B85, 0x7601_8E57, 0x1025_FA13,
    0x0648_6BAB, 0x3770_6EBA, 0x25EB_966B, 0x113C_24E5,
    0x2AE2_0EC4, 0x5A27_507C, 0x0CD3_8CF1, 0x761C_10E5,
    0x19E3_EF1A, 0x032C_730F, 0x35D8_AF83, 0x651D_F13B,
    0x7EC3_DB1A, 0x6A14_6994, 0x588F_9145, 0x09B7_9455,
    0x7FDA_05EC, 0x19FE_71A8, 0x6988_947A, 0x624F_1D31,
    0x500B_B628, 0x0B14_28CE, 0x3A62_E1D6, 0x7769_2387
];

impl Permutation<[Mersenne31; 32]> for MdsMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 32]) -> [Mersenne31; 32] {
        apply_circulant(&MATRIX_CIRC_MDS_32_MERSENNE31, input)
    }

    fn permute_mut(&self, input: &mut [Mersenne31; 32]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Mersenne31, 32> for MdsMatrixMersenne31 {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_64_MERSENNE31: [u64; 64] = [
    0x5702_27A5, 0x3702_983F, 0x4B7B_3B0A, 0x74F1_3DE3,
    0x4853_14B0, 0x0157_E2EC, 0x1AD2_E5DE, 0x7215_15E3,
    0x5452_ADA3, 0x0C74_B6C1, 0x67DA_9450, 0x33A4_8369,
    0x3BDB_EE06, 0x7C67_8D5E, 0x160F_16D3, 0x5488_8B8C,
    0x666C_7AA6, 0x113B_89E2, 0x2A40_3CE2, 0x18F9_DF42,
    0x2A68_5E84, 0x49EE_FDE5, 0x5D04_4806, 0x560A_41F8,
    0x69EF_1BD0, 0x2CD1_5786, 0x62E0_7766, 0x22A2_31E2,
    0x3CFC_F40C, 0x4E8F_63D8, 0x6965_7A15, 0x466B_4B2D,
    0x4194_B4D2, 0x1E9A_85EA, 0x3970_9C27, 0x4B03_0BF3,
    0x655D_CE1D, 0x251F_8899, 0x5B2E_A879, 0x1E10_E42F,
    0x31F5_BE07, 0x2AFB_B7F9, 0x3E11_021A, 0x5D97_A17B,
    0x6F06_20BD, 0x5DBF_C31D, 0x76C4_761D, 0x2193_8559,
    0x3377_7473, 0x71F0_E92C, 0x0B98_72A1, 0x4C24_11F9,
    0x545B_7C96, 0x2025_6BAF, 0x7B8B_493E, 0x33AD_525C,
    0x15EA_EA1C, 0x6D2D_1A21, 0x06A8_1D14, 0x3FAC_EB4F,
    0x130E_C21C, 0x3C84_C4F5, 0x50FD_67C0, 0x30FD_D85A,
];

impl Permutation<[Mersenne31; 64]> for MdsMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 64]) -> [Mersenne31; 64] {
        apply_circulant(&MATRIX_CIRC_MDS_64_MERSENNE31, input)
    }

    fn permute_mut(&self, input: &mut [Mersenne31; 64]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Mersenne31, 64> for MdsMatrixMersenne31 {}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::Permutation;

    use super::MdsMatrixMersenne31;

    #[test]
    fn mersenne8() {
        let input: [Mersenne31; 8] = [
            1_741_044_457, 327_154_658, 318_297_696, 1_528_828_225, 468_360_260, 1_271_368_222, 1_906_288_587,
            1_521_884_224,
        ]
        .map(Mersenne31::from_canonical_u64);

        let output = MdsMatrixMersenne31.permute(input);

        let expected: [Mersenne31; 8] = [
            1_796_260_072, 48_130_602, 971_886_692, 1_460_399_885, 745_498_940, 352_898_876, 223_078_564,
            2_090_539_234,
        ]
        .map(Mersenne31::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn mersenne12() {
        let input: [Mersenne31; 12] = [
            1_232_740_094, 661_555_540, 11_024_822, 1_620_264_994, 471_137_070, 276_755_041, 1_316_882_747,
            1_023_679_816, 1_675_266_989, 743_211_887, 44_774_582, 1_990_989_306,
        ]
        .map(Mersenne31::from_canonical_u64);

        let output = MdsMatrixMersenne31.permute(input);

        let expected: [Mersenne31; 12] = [
            492_952_161, 916_402_585, 1_541_871_876, 799_921_480, 707_671_572, 1_293_088_641, 866_554_196,
            1_471_029_895, 35_362_849, 2_107_961_577, 1_616_107_486, 762_379_007,
        ]
        .map(Mersenne31::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn mersenne16() {
        let input: [Mersenne31; 16] = [
            1_431_168_444, 963_811_518, 88_067_321, 381_314_132, 908_628_282, 1_260_098_295, 980_207_659,
            150_070_493, 357_706_876, 2_014_609_375, 387_876_458, 1_621_671_571, 183_146_044, 107_201_572,
            166_536_524, 2_078_440_788,
        ]
        .map(Mersenne31::from_canonical_u64);

        let output = MdsMatrixMersenne31.permute(input);

        let expected: [Mersenne31; 16] = [
            1_929_166_367, 1_352_685_756, 1_090_911_983, 379_953_343, 62_410_403, 637_712_268, 1_637_633_936,
            555_902_167, 850_536_312, 913_896_503, 2_070_446_350, 814_495_093, 651_934_716, 419_066_839,
            603_091_570, 1_453_848_863,
        ]
        .map(Mersenne31::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn mersenne32() {
        let input: [Mersenne31; 32] = [
            873_912_014, 1_112_497_426, 300_405_095, 4_255_553, 1_234_979_949, 156_402_357, 1_952_135_954,
            718_195_399, 1_041_748_465, 683_604_342, 184_275_751, 1_184_118_518, 214_257_054, 1_293_941_921,
            64_085_758, 710_448_062, 1_133_100_009, 350_114_887, 1_091_675_272, 671_421_879, 1_226_105_999,
            546_430_131, 1_298_443_967, 1_787_169_653, 2_129_310_791, 1_560_307_302, 471_771_931, 1_191_484_402,
            1_550_203_198, 1_541_319_048, 229_197_040, 839_673_789,
        ]
        .map(Mersenne31::from_canonical_u64);

        let output = MdsMatrixMersenne31.permute(input);

        let expected: [Mersenne31; 32] = [
            1_439_049_928, 890_642_852, 694_402_307, 713_403_244, 553_213_342, 1_049_445_650, 321_709_533,
            1_195_683_415, 2_118_492_257, 623_077_773, 96_734_062, 990_488_164, 1_674_607_608, 749_155_000,
            353_377_854, 966_432_998, 1_114_654_884, 1_370_359_248, 1_624_965_859, 685_087_760, 1_631_836_645,
            1_615_931_812, 2_061_986_317, 1_773_551_151, 1_449_911_206, 1_951_762_557, 545_742_785, 582_866_449,
            1_379_774_336, 229_242_759, 1_871_227_547, 752_848_413,
        ]
        .map(Mersenne31::from_canonical_u64);

        assert_eq!(output, expected);
    }

    #[test]
    fn mersenne64() {
        let input: [Mersenne31; 64] = [
            837_269_696, 1_509_031_194, 413_915_480, 1_889_329_185, 315_502_822, 1_529_162_228, 1_454_661_012,
            1_015_826_742, 973_381_409, 1_414_676_304, 1_449_029_961, 1_968_715_566, 2_027_226_497, 1_721_820_509,
            434_042_616, 1_436_005_045, 1_680_352_863, 651_591_867, 260_585_272, 1_078_022_153, 703_990_572,
            269_504_423, 1_776_357_592, 1_174_979_337, 1_142_666_094, 1_897_872_960, 1_387_995_838, 250_774_418,
            776_134_750, 73_930_096, 194_742_451, 1_860_060_380, 666_407_744, 669_566_398, 963_802_147,
            2_063_418_105, 1_772_573_581, 998_923_482, 701_912_753, 1_716_548_204, 860_820_931, 1_680_395_948,
            949_886_256, 1_811_558_161, 501_734_557, 1_671_977_429, 463_135_040, 1_911_493_108, 207_754_409,
            608_714_758, 1_553_060_084, 1_558_941_605, 980_281_686, 2_014_426_559, 650_527_801, 53_015_148,
            1_521_176_057, 720_530_872, 713_593_252, 88_228_433, 1_194_162_313, 1_922_416_934, 1_075_145_779,
            344_403_794,
        ]
        .map(Mersenne31::from_canonical_u64);

        let output = MdsMatrixMersenne31.permute(input);

        let expected: [Mersenne31; 64] = [
            1_599_981_950, 252_630_853, 1_171_557_270, 116_468_420, 1_269_245_345, 666_203_050, 46_155_642,
            1_701_131_520, 530_845_775, 508_460_407, 630_407_239, 1_731_628_135, 1_199_144_768, 295_132_047,
            77_536_342, 1_472_377_703, 30_752_443, 1_300_339_617, 18_647_556, 1_267_774_380, 1_194_573_079,
            1_624_665_024, 646_848_056, 1_667_216_490, 1_184_843_555, 1_250_329_476, 254_171_597, 1_902_035_936,
            1_706_882_202, 964_921_003, 952_266_538, 1_215_696_284, 539_510_504, 1_056_507_562, 1_393_151_480,
            733_644_883, 1_663_330_816, 1_100_715_048, 991_108_703, 1_671_345_065, 1_376_431_774, 408_310_416,
            313_176_996, 743_567_676, 304_660_642, 1_842_695_838, 958_201_635, 1_650_792_218, 541_570_244,
            968_523_062, 1_958_918_704, 1_866_282_698, 849_808_680, 1_193_306_222, 794_153_281, 822_835_360,
            135_282_913, 1_149_868_448, 2_068_162_123, 1_474_283_743, 2_039_088_058, 720_305_835, 746_036_736,
            671_006_610,
        ]
        .map(Mersenne31::from_canonical_u64);

        assert_eq!(output, expected);
    }
}
