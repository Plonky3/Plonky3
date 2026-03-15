//! MDS matrices over the KoalaBear field, and permutations defined by them.
//!
//! NB: Not all sizes have fast implementations of their permutations.
//! Supported sizes: 8, 12, 16, 24, 32, 64.
//! Sizes 8 and 12 are from Plonky2, size 16 was found as part of concurrent
//! work by Angus Gruen and Hamish Ivey-Law. Other sizes are from Ulrich Haböck's
//! database.

use p3_mds::util::first_row_to_first_col;
use p3_monty_31::{MDSUtils, MdsMatrixMontyField31};

#[derive(Clone, Default)]
pub struct MDSKoalaBearData;

impl MDSUtils for MDSKoalaBearData {
    const MATRIX_CIRC_MDS_8_COL: [i64; 8] = first_row_to_first_col(&[7, 1, 3, 8, 8, 3, 4, 9]);
    const MATRIX_CIRC_MDS_12_COL: [i64; 12] =
        first_row_to_first_col(&[1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10]);
    const MATRIX_CIRC_MDS_16_COL: [i64; 16] =
        first_row_to_first_col(&[1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3]);
    const MATRIX_CIRC_MDS_24_COL: [i64; 24] = first_row_to_first_col(&[
        0x2D0AAAAB, 0x64850517, 0x17F5551D, 0x04ECBEB5, 0x6D91A8D5, 0x60703026, 0x18D6F3CA,
        0x729601A7, 0x77CDA9E2, 0x3C0F5038, 0x26D52A61, 0x0360405D, 0x68FC71C8, 0x2495A71D,
        0x5D57AFC2, 0x1689DD98, 0x3C2C3DBE, 0x0C23DC41, 0x0524C7F2, 0x6BE4DF69, 0x0A6E572C,
        0x5C7790FA, 0x17E118F6, 0x0878A07F,
    ]);
    const MATRIX_CIRC_MDS_32_COL: [i64; 32] = first_row_to_first_col(&[
        0x0BC00000, 0x2BED8F81, 0x337E0652, 0x4C4535D1, 0x4AF2DC32, 0x2DB4050F, 0x676A7CE3,
        0x3A06B68E, 0x5E95C1B1, 0x2C5F54A0, 0x2332F13D, 0x58E757F1, 0x3AA6DCCE, 0x607EE630,
        0x4ED57FF0, 0x6E08555B, 0x4C155556, 0x587FD0CE, 0x462F1551, 0x032A43CC, 0x5E2E43EA,
        0x71609B02, 0x0ED97E45, 0x562CA7E9, 0x2CB70B1D, 0x4E941E23, 0x174A61C1, 0x117A9426,
        0x73562137, 0x54596086, 0x487C560B, 0x68A4ACAB,
    ]);
    const MATRIX_CIRC_MDS_64_COL: [i64; 64] = first_row_to_first_col(&[
        0x39577778, 0x0072F4E1, 0x0B1B8404, 0x041E9C88, 0x32D22F9F, 0x4E4BF946, 0x20C7B6D7,
        0x0587C267, 0x55877229, 0x4D186EC4, 0x4A19FD23, 0x1A64A20F, 0x2965CA4D, 0x16D98A5A,
        0x471E544A, 0x193D5C8B, 0x6E66DF0C, 0x28BF1F16, 0x26DB0BC8, 0x5B06CDDB, 0x100DCCA2,
        0x65C268AD, 0x199F09E7, 0x36BA04BE, 0x06C393F2, 0x51B06DFD, 0x6951B0C4, 0x6683A4C2,
        0x3B53D11B, 0x26E5134C, 0x45A5F1C5, 0x6F4D2433, 0x3CE2D82E, 0x36309A7D, 0x3DD9B459,
        0x68051E4C, 0x5C3AA720, 0x11640517, 0x0634D995, 0x1B0F6406, 0x72A18430, 0x26513CC5,
        0x67C0B93C, 0x548AB4A3, 0x6395D20D, 0x3E5DBC41, 0x332AF630, 0x3C5DDCB3, 0x0AA95792,
        0x66EB5492, 0x3F78DDDC, 0x5AC41627, 0x16CD5124, 0x3564DA96, 0x461867C9, 0x157B4E11,
        0x1AA486C8, 0x0C5095A9, 0x3833C0C6, 0x008FEBA5, 0x52ECBE2E, 0x1D178A67, 0x58B3C04B,
        0x6E95CB51,
    ]);
}

pub type MdsMatrixKoalaBear = MdsMatrixMontyField31<MDSKoalaBearData>;

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;

    use super::MdsMatrixKoalaBear;
    use crate::KoalaBear;

    #[test]
    fn koalabear8() {
        let input: [KoalaBear; 8] = KoalaBear::new_array([
            391474477, 1174409341, 666967492, 1852498830, 1801235316, 820595865, 585587525,
            1348326858,
        ]);

        let mds: MdsMatrixKoalaBear = Default::default();

        let output = mds.permute(input);

        let expected: [KoalaBear; 8] = KoalaBear::new_array([
            947631349, 1348484024, 1002291099, 1962469348, 831049401, 1648283812, 1017255940,
            589556689,
        ]);

        assert_eq!(output, expected);
    }

    #[test]
    fn koalabear12() {
        let input: [KoalaBear; 12] = KoalaBear::new_array([
            918423259, 673549090, 364157140, 9832898, 493922569, 1171855651, 246075034, 1542167926,
            1787615541, 1696819900, 1884530130, 422386768,
        ]);

        let mds: MdsMatrixKoalaBear = Default::default();

        let output = mds.permute(input);

        let expected: [KoalaBear; 12] = KoalaBear::new_array([
            3672342, 689021900, 1455700352, 1687414333, 1231524540, 1572686242, 42253424,
            696666080, 950244312, 678673484, 530048499, 135761510,
        ]);

        assert_eq!(output, expected);
    }

    #[test]
    fn koalabear16() {
        let input: [KoalaBear; 16] = KoalaBear::new_array([
            1983708094, 1477844074, 1638775686, 98517138, 70746308, 968700066, 275567720,
            1359144511, 960499489, 1215199187, 474302783, 79320256, 1923147803, 1197733438,
            1638511323, 303948902,
        ]);

        let mds: MdsMatrixKoalaBear = Default::default();

        let output = mds.permute(input);

        let expected: [KoalaBear; 16] = KoalaBear::new_array([
            54729128, 2128589920, 81963306, 842781423, 59798772, 1955488131, 274677035, 372631613,
            1610234661, 608093248, 1204230235, 1081779929, 873712545, 436245025, 339463618,
            255045423,
        ]);

        assert_eq!(output, expected);
    }
}
