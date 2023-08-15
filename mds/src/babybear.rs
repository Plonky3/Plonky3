use p3_field::AbstractField;
use p3_symmetric::mds::MDSPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};

use crate::BabyBear;

// NB: These four are MDS for M31, BabyBear and Goldilocks
//const MATRIX_CIRC_MDS_8_2EXP: [u64; 8] = [1, 1, 2, 1, 8, 32, 4, 256];
const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

//const MATRIX_CIRC_MDS_12_2EXP: [u64; 12] = [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024];
const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];

pub struct MDSMatrixBabyBear;

fn dot_vec<F: AbstractField, const N: usize>(u: &[F; N], v: &[F; N]) -> F {
    u.iter().zip(v).map(|(x, y)| x.clone() * y.clone()).sum()
}

// F: Field ==> need dot_vec for each field
fn apply_circulant<F: AbstractField, const N: usize>(
    circ_matrix: &[u64; N],
    input: [F; N],
) -> [F; N] {
    let mut matrix: [F; N] = circ_matrix.map(F::from_canonical_u64);

    let mut output = [F::ZERO; N];
    for i in 0..N - 1 {
        output[i] = dot_vec(&input, &matrix);
        matrix.rotate_right(1);
    }
    output[N - 1] = dot_vec(&input, &matrix);
    output
}

impl CryptographicPermutation<[BabyBear; 8]> for MDSMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 8]) -> [BabyBear; 8] {
        apply_circulant(&MATRIX_CIRC_MDS_8_SML, input)
    }
}
impl ArrayPermutation<BabyBear, 8> for MDSMatrixBabyBear {}
impl MDSPermutation<BabyBear, 8> for MDSMatrixBabyBear {}

impl CryptographicPermutation<[BabyBear; 12]> for MDSMatrixBabyBear {
    fn permute(&self, input: [BabyBear; 12]) -> [BabyBear; 12] {
        apply_circulant(&MATRIX_CIRC_MDS_12_SML, input)
    }
}
impl ArrayPermutation<BabyBear, 12> for MDSMatrixBabyBear {}
impl MDSPermutation<BabyBear, 12> for MDSMatrixBabyBear {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_16_BABYBEAR: [u64; 16] = [
    0x07801000, 0x4ACAAC32, 0x6A709B76, 0x20413E94,
    0x00928499, 0x31C34CA3, 0x03BBC192, 0x3F20868B,
    0x257FFAAB, 0x5F05F559, 0x55B43EA9, 0x2BC659ED,
    0x2C6D7501, 0x1D110184, 0x0E1F608D, 0x2032F0C6,
];

#[rustfmt::skip]
const MATRIX_CIRC_MDS_24_BABYBEAR: [u64; 24] = [
    0x2D0AAAAB, 0x64850517, 0x17F5551D, 0x04ECBEB5,
    0x6D91A8D5, 0x60703026, 0x18D6F3CA, 0x729601A7,
    0x77CDA9E2, 0x3C0F5038, 0x26D52A61, 0x0360405D,
    0x68FC71C8, 0x2495A71D, 0x5D57AFC2, 0x1689DD98,
    0x3C2C3DBE, 0x0C23DC41, 0x0524C7F2, 0x6BE4DF69,
    0x0A6E572C, 0x5C7790FA, 0x17E118F6, 0x0878A07F,
];

#[rustfmt::skip]
const MATRIX_CIRC_MDS_32_BABYBEAR: [u64; 32] = [
    0x0BC00000, 0x2BED8F81, 0x337E0652, 0x4C4535D1,
    0x4AF2DC32, 0x2DB4050F, 0x676A7CE3, 0x3A06B68E,
    0x5E95C1B1, 0x2C5F54A0, 0x2332F13D, 0x58E757F1,
    0x3AA6DCCE, 0x607EE630, 0x4ED57FF0, 0x6E08555B,
    0x4C155556, 0x587FD0CE, 0x462F1551, 0x032A43CC,
    0x5E2E43EA, 0x71609B02, 0x0ED97E45, 0x562CA7E9,
    0x2CB70B1D, 0x4E941E23, 0x174A61C1, 0x117A9426,
    0x73562137, 0x54596086, 0x487C560B, 0x68A4ACAB,
];

#[rustfmt::skip]
const MATRIX_CIRC_MDS_64_BABYBEAR: [u64; 64] = [
    0x39577778, 0x0072F4E1, 0x0B1B8404, 0x041E9C88,
    0x32D22F9F, 0x4E4BF946, 0x20C7B6D7, 0x0587C267,
    0x55877229, 0x4D186EC4, 0x4A19FD23, 0x1A64A20F,
    0x2965CA4D, 0x16D98A5A, 0x471E544A, 0x193D5C8B,
    0x6E66DF0C, 0x28BF1F16, 0x26DB0BC8, 0x5B06CDDB,
    0x100DCCA2, 0x65C268AD, 0x199F09E7, 0x36BA04BE,
    0x06C393F2, 0x51B06DFD, 0x6951B0C4, 0x6683A4C2,
    0x3B53D11B, 0x26E5134C, 0x45A5F1C5, 0x6F4D2433,
    0x3CE2D82E, 0x36309A7D, 0x3DD9B459, 0x68051E4C,
    0x5C3AA720, 0x11640517, 0x0634D995, 0x1B0F6406,
    0x72A18430, 0x26513CC5, 0x67C0B93C, 0x548AB4A3,
    0x6395D20D, 0x3E5DBC41, 0x332AF630, 0x3C5DDCB3,
    0x0AA95792, 0x66EB5492, 0x3F78DDDC, 0x5AC41627,
    0x16CD5124, 0x3564DA96, 0x461867C9, 0x157B4E11,
    0x1AA486C8, 0x0C5095A9, 0x3833C0C6, 0x008FEBA5,
    0x52ECBE2E, 0x1D178A67, 0x58B3C04B, 0x6E95CB51,
];
