use p3_mersenne_31::Mersenne31;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};

use crate::util::{apply_circulant, apply_circulant_prime32_12_sml, apply_circulant_prime32_8_sml};
use crate::MDSPermutation;

pub struct MDSMatrixMersenne31;

impl CryptographicPermutation<[Mersenne31; 8]> for MDSMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 8]) -> [Mersenne31; 8] {
        apply_circulant_prime32_8_sml(input)
    }
}
impl ArrayPermutation<Mersenne31, 8> for MDSMatrixMersenne31 {}
impl MDSPermutation<Mersenne31, 8> for MDSMatrixMersenne31 {}

impl CryptographicPermutation<[Mersenne31; 12]> for MDSMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 12]) -> [Mersenne31; 12] {
        apply_circulant_prime32_12_sml(input)
    }
}
impl ArrayPermutation<Mersenne31, 12> for MDSMatrixMersenne31 {}
impl MDSPermutation<Mersenne31, 12> for MDSMatrixMersenne31 {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_16_MERSENNE31: [u64; 16] = [
    0x327ACB92, 0x58C99138, 0x3AC486B5, 0x25123B13,
    0x2C74BDE9, 0x108BD51A, 0x4E911F9D, 0x19DD8E68,
    0x06227198, 0x516EE062, 0x0F742AE6, 0x738B4216,
    0x7AEDC4EC, 0x653B794A, 0x47366EC7, 0x6D85346D
];

impl CryptographicPermutation<[Mersenne31; 16]> for MDSMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 16]) -> [Mersenne31; 16] {
        apply_circulant(&MATRIX_CIRC_MDS_16_MERSENNE31, input)
    }
}
impl ArrayPermutation<Mersenne31, 16> for MDSMatrixMersenne31 {}
impl MDSPermutation<Mersenne31, 16> for MDSMatrixMersenne31 {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_32_MERSENNE31: [u64; 32] = [
    0x1896DC78, 0x559D1E29, 0x04EBD732, 0x3FF449D7,
    0x2DB0E2CE, 0x26776B85, 0x76018E57, 0x1025FA13,
    0x06486BAB, 0x37706EBA, 0x25EB966B, 0x113C24E5,
    0x2AE20EC4, 0x5A27507C, 0x0CD38CF1, 0x761C10E5,
    0x19E3EF1A, 0x032C730F, 0x35D8AF83, 0x651DF13B,
    0x7EC3DB1A, 0x6A146994, 0x588F9145, 0x09B79455,
    0x7FDA05EC, 0x19FE71A8, 0x6988947A, 0x624F1D31,
    0x500BB628, 0x0B1428CE, 0x3A62E1D6, 0x77692387
];

impl CryptographicPermutation<[Mersenne31; 32]> for MDSMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 32]) -> [Mersenne31; 32] {
        apply_circulant(&MATRIX_CIRC_MDS_32_MERSENNE31, input)
    }
}
impl ArrayPermutation<Mersenne31, 32> for MDSMatrixMersenne31 {}
impl MDSPermutation<Mersenne31, 32> for MDSMatrixMersenne31 {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_64_MERSENNE31: [u64; 64] = [
    0x570227A5, 0x3702983F, 0x4B7B3B0A, 0x74F13DE3,
    0x485314B0, 0x0157E2EC, 0x1AD2E5DE, 0x721515E3,
    0x5452ADA3, 0x0C74B6C1, 0x67DA9450, 0x33A48369,
    0x3BDBEE06, 0x7C678D5E, 0x160F16D3, 0x54888B8C,
    0x666C7AA6, 0x113B89E2, 0x2A403CE2, 0x18F9DF42,
    0x2A685E84, 0x49EEFDE5, 0x5D044806, 0x560A41F8,
    0x69EF1BD0, 0x2CD15786, 0x62E07766, 0x22A231E2,
    0x3CFCF40C, 0x4E8F63D8, 0x69657A15, 0x466B4B2D,
    0x4194B4D2, 0x1E9A85EA, 0x39709C27, 0x4B030BF3,
    0x655DCE1D, 0x251F8899, 0x5B2EA879, 0x1E10E42F,
    0x31F5BE07, 0x2AFBB7F9, 0x3E11021A, 0x5D97A17B,
    0x6F0620BD, 0x5DBFC31D, 0x76C4761D, 0x21938559,
    0x33777473, 0x71F0E92C, 0x0B9872A1, 0x4C2411F9,
    0x545B7C96, 0x20256BAF, 0x7B8B493E, 0x33AD525C,
    0x15EAEA1C, 0x6D2D1A21, 0x06A81D14, 0x3FACEB4F,
    0x130EC21C, 0x3C84C4F5, 0x50FD67C0, 0x30FDD85A,
];

impl CryptographicPermutation<[Mersenne31; 64]> for MDSMatrixMersenne31 {
    fn permute(&self, input: [Mersenne31; 64]) -> [Mersenne31; 64] {
        apply_circulant(&MATRIX_CIRC_MDS_64_MERSENNE31, input)
    }
}
impl ArrayPermutation<Mersenne31, 64> for MDSMatrixMersenne31 {}
impl MDSPermutation<Mersenne31, 64> for MDSMatrixMersenne31 {}
