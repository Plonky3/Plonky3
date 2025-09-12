//! MDS matrices over the Goldilocks Montgomery field, and permutations defined by them.
//!
//! This implements the same MDS matrices as the standard Goldilocks field but adapted for Montgomery arithmetic.
//! NB: Not all sizes have fast implementations of their permutations.
//! Supported sizes: 8, 12, 16, 24, 32, 64, 68.

use p3_dft::Radix2Bowers;
use p3_field::PrimeField64;
use p3_mds::MdsPermutation;
use p3_mds::karatsuba_convolution::Convolve;
use p3_mds::util::{apply_circulant, apply_circulant_fft, first_row_to_first_col};
use p3_symmetric::Permutation;

use crate::Goldilocks;

#[derive(Clone, Debug, Default)]
pub struct MdsMatrixGoldilocksMonty;

/// Instantiate convolution for "small" RHS vectors over Goldilocks Montgomery.
///
/// This is adapted from the standard Goldilocks implementation but works with Montgomery form values.
#[derive(Debug)]
pub struct SmallConvolveGoldilocksMonty;

impl Convolve<Goldilocks, i128, i64, i128> for SmallConvolveGoldilocksMonty {
    /// Return the lift of a Goldilocks element from Montgomery form.
    /// We convert from Montgomery form to standard form for arithmetic operations.
    #[inline(always)]
    fn read(input: Goldilocks) -> i128 {
        input.as_canonical_u64() as i128
    }

    /// Perform dot product with widened types to avoid overflow.
    #[inline(always)]
    fn parity_dot<const N: usize>(u: [i128; N], v: [i64; N]) -> i128 {
        let mut s = 0i128;
        for i in 0..N {
            s += u[i] * v[i] as i128;
        }
        s
    }

    /// Reduce the result back to Goldilocks Montgomery form.
    #[inline(always)]
    fn reduce(z: i128) -> Goldilocks {
        debug_assert!(z >= 0);
        // Convert to standard form, then back to Montgomery
        Goldilocks::new((z as u128 % (crate::GOLDILOCKS_PRIME as u128)) as u64)
    }
}

const FFT_ALGO: Radix2Bowers = Radix2Bowers;

// Use the same MDS matrix constants as the standard Goldilocks field
pub(crate) const MATRIX_CIRC_MDS_8_SML_ROW: [i64; 8] = [7, 1, 3, 8, 8, 3, 4, 9];

impl Permutation<[Goldilocks; 8]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 8]) -> [Goldilocks; 8] {
        const MATRIX_CIRC_MDS_8_SML_COL: [i64; 8] =
            first_row_to_first_col(&MATRIX_CIRC_MDS_8_SML_ROW);
        SmallConvolveGoldilocksMonty::apply(
            input,
            MATRIX_CIRC_MDS_8_SML_COL,
            SmallConvolveGoldilocksMonty::conv8,
        )
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 8]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 8> for MdsMatrixGoldilocksMonty {}

pub(crate) const MATRIX_CIRC_MDS_12_SML_ROW: [i64; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];

impl Permutation<[Goldilocks; 12]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 12]) -> [Goldilocks; 12] {
        const MATRIX_CIRC_MDS_12_SML_COL: [i64; 12] =
            first_row_to_first_col(&MATRIX_CIRC_MDS_12_SML_ROW);
        SmallConvolveGoldilocksMonty::apply(
            input,
            MATRIX_CIRC_MDS_12_SML_COL,
            SmallConvolveGoldilocksMonty::conv12,
        )
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 12]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 12> for MdsMatrixGoldilocksMonty {}

pub(crate) const MATRIX_CIRC_MDS_16_SML_ROW: [i64; 16] =
    [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];

impl Permutation<[Goldilocks; 16]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 16]) -> [Goldilocks; 16] {
        const MATRIX_CIRC_MDS_16_SML_COL: [i64; 16] =
            first_row_to_first_col(&MATRIX_CIRC_MDS_16_SML_ROW);
        SmallConvolveGoldilocksMonty::apply(
            input,
            MATRIX_CIRC_MDS_16_SML_COL,
            SmallConvolveGoldilocksMonty::conv16,
        )
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 16]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 16> for MdsMatrixGoldilocksMonty {}

#[rustfmt::skip]
pub(crate) const MATRIX_CIRC_MDS_24_GOLDILOCKS_MONTY: [u64; 24] = [
    0x5FFFFFFFA00AAAAB, 0x24021AB75BBFE656, 0x7BE9082D73B06DF5, 0x2282863E9C3A5A62,
    0xE0071C70DFFC71C8, 0x796CB65AB42A1A63, 0xDBBBBFFADFFDDDE3, 0x23B88EE217C5C9C2,
    0x20030C309FFB6DB7, 0x23C3C64763BE1E1D, 0x0F93B7C9CC51362E, 0xC697A1094BD0850A,
    0xDFFFFFFF1FFC71C8, 0xC15A4FD614950302, 0xC41D883A4C4DEDF2, 0x187879BC23C46462,
    0x5FFCF3CEDFFE79E8, 0x1C41DF105B82398E, 0x64444003DFFDDDDA, 0x76EDDBB6F7E51F95,
    0x1FF8E38E20038E39, 0x214139BD5C40A09D, 0x3065B7CCF3B3B621, 0x23B6F4622485CEDC,
];

impl Permutation<[Goldilocks; 24]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 24]) -> [Goldilocks; 24] {
        apply_circulant(&MATRIX_CIRC_MDS_24_GOLDILOCKS_MONTY, input)
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 24]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 24> for MdsMatrixGoldilocksMonty {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_32_GOLDILOCKS_MONTY: [u64; 32] = [
    0x0800000000000000, 0x69249248B4924925, 0x3ABD5EAF15EAF57B, 0x294A5294739CE73A,
    0x59E2D2CEB4B3C5A6, 0x087FBE00FF7C0220, 0xA554AA94A554AA96, 0xF00080FEFFDF8005,
    0x64CCCCCC6666699A, 0x5B13AD8973B139D9, 0xAD4A55ACA54AD5AA, 0xDA496DA3B492DB8A,
    0x4AD696955A5694B5, 0xA4A6B29A25B496D3, 0xA74EA162162BD3A9, 0xC698B3A5662CE98C,
    0xA7FFFFFF55555556, 0x4AAAAAAA5AAAAAAB, 0xB047DC113DC11F71, 0x8BA2E8B99B26C9B3,
    0xD259696C5A5B4D2E, 0xA7D540AA557EA9F6, 0x8B6E922D26DB249C, 0xFAAA805455602AAD,
    0xCB33333266666334, 0xD13B17619B13B277, 0x45B26D9326E9374A, 0x52AB552A5AA9556B,
    0x68ED2D2DB4B87697, 0x8B264C98A74E9D3B, 0x09EC23D83D847B09, 0x2C9A4D26669349A5,
];

impl Permutation<[Goldilocks; 32]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 32]) -> [Goldilocks; 32] {
        const ENTRIES: [u64; 32] = first_row_to_first_col(&MATRIX_CIRC_MDS_32_GOLDILOCKS_MONTY);
        // Convert to standard form for FFT operations
        let standard_input: [crate::Goldilocks; 32] =
            input.map(|x| crate::Goldilocks::new(x.as_canonical_u64()));
        let result = apply_circulant_fft(FFT_ALGO, ENTRIES, &standard_input);
        result.map(|x| Goldilocks::new(x.as_canonical_u64()))
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 32]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 32> for MdsMatrixGoldilocksMonty {}

// For larger sizes (64, 68), we use similar patterns but with Montgomery conversion

#[rustfmt::skip]
const MATRIX_CIRC_MDS_64_GOLDILOCKS_MONTY: [u64; 64] = [
    0x07FFFFFFFC000000, 0xFBFFFFFF04000001, 0x436DB6DB25B6DB6E, 0x4AAAAAAA5AAAAAAB,
    0x45B2D96C6D96CB66, 0x3BC7BC7B87BC7BC8, 0x6318C63125294A53, 0xCB3672CCCD9CB368,
    0xB43CB5A12D68796C, 0xFBFBFBFAFBFBFBFD, 0x883DBF107B7E2210, 0x8A7689B59B629DA3,
    0xF7FEFFDF00000001, 0x7B7C83BBC83BC47C, 0xEFF0410107EF7F83, 0x2CD8B3629CB272CA,
    0x9800019900CCCE67, 0xFBFFFBFF07FFFC01, 0x94EC4A758C4EC628, 0xDA5A5B4A6D2D2E1F,
    0xFFEFC080FC003FFF, 0xBC387BC2C783BC79, 0xB492DB686D24B6F3, 0x1DB6925B4B6E2477,
    0x7801E0EF87BFFF10, 0xFC0803FAFBFC0409, 0x3780FE03C086F21C, 0x8B749B224DB22D94,
    0x32648B36B76E9923, 0x3BC3C3C387C3C3C4, 0x79AF286B4FCA1AF3, 0x9E2762758B627628,
    0x52AAAAAA56AAAAAB, 0xFBFFFFFEFC000001, 0xF7FFFFFF08000001, 0x2CCCCCCC9CCCCCCD,
    0xCF286BC946BCA1B0, 0xBC483B7B883B7C49, 0xD9364D9287C1F07D, 0xAD5A94A8A95AD5AA,
    0xFF871002C400F1E1, 0xFC03FC02FC03FC05, 0xD29495A4D6D4B4A6, 0x6C926DD1DD24DB65,
    0x1EDC247B4DB64937, 0x7C7B843B47BC437D, 0xA55A95AAAD5AD52C, 0x4A96D5A45AD694A6,
    0xFE6664CBCD999801, 0xFC0003FF08000401, 0x1EC4F09D64EC4D8A, 0x9E1E1D2C8B4B4A5B,
    0xD9270937709B64DC, 0x3BB77C4448843B78, 0xFFFFFFDF03FF0021, 0x59D8761D2D8A6299,
    0xC3496878A5E5A4B5, 0xFBF80402FC0403F9, 0x5ECD9B360E142851, 0x6D925D6429D64976,
    0xA8AE615C19CC2B99, 0xBC44444388444445, 0xDFE3F1F81CFC7E40, 0xDA4924916D24924A,
];

impl Permutation<[Goldilocks; 64]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 64]) -> [Goldilocks; 64] {
        const ENTRIES: [u64; 64] = first_row_to_first_col(&MATRIX_CIRC_MDS_64_GOLDILOCKS_MONTY);
        let standard_input: [crate::Goldilocks; 64] =
            input.map(|x| crate::Goldilocks::new(x.as_canonical_u64()));
        let result = apply_circulant_fft(FFT_ALGO, ENTRIES, &standard_input);
        result.map(|x| Goldilocks::new(x.as_canonical_u64()))
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 64]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 64> for MdsMatrixGoldilocksMonty {}

#[rustfmt::skip]
const MATRIX_CIRC_MDS_68_GOLDILOCKS_MONTY: [u64; 68] = [
    0x03C3C3C3FC3C3C3C, 0x6799AFC54A69BC7D, 0xDA8C2C496A74B03B, 0x1E641D7AB35ED229,
    0x9239DA20DA3A2686, 0x6E23D41459EBA8C4, 0x7BC412896E2A6B3A, 0x9082059089ABD4FC,
    0x94A16FA8B0339EEE, 0x85650EC91BB519C9, 0x1600745267E94DE1, 0xFFFD8405C82020AB,
    0x21BDE80429DCED6A, 0x8ACE123AF754E343, 0xFFC7211605D2BDAE, 0xC21187AE15900F4D,
    0x9C4A889708568DC6, 0x65A5A726B5758D8E, 0x949DB90B9AC0D11A, 0x23B6CF7C368BBE52,
    0xD5128DDF59CB5A35, 0xF53BCC5BDADF3A0A, 0xBA7C5112F4BAB1CD, 0x4B93989C5B729351,
    0x6534B7E50E4AD1CB, 0x640061B54C918405, 0x0E66E1F90D2C9311, 0x31C8649B0FE7557F,
    0x0E9190D165F4A8F3, 0x52DF336BB708F919, 0x3C0F6697F14065A5, 0xBE8190942EC50031,
    0x60038E9ACC701118, 0x73F105909A55A88B, 0xFEBEBEBDABEBEBED, 0x6F52163A64B03467,
    0xFBAE131F23A12F56, 0x1950493BC70D0676, 0x2886550DB5A1BBBF, 0x15B003D6E58181D7,
    0x3A4E7D9D44F100F8, 0x6CC3AB896025E6A0, 0x7E23E68456F825E5, 0x079CDD570B591A16,
    0xEC15A830C3D2CCD1, 0xCF4C722D2C0F8A0E, 0xC1BB6F5591B59A26, 0xB63A5931A607BDE0,
    0x43A0AD0B71040187, 0x7E4B492889D1CEE0, 0x734153F3F0C31C5B, 0x98D8D756B2725A5B,
    0x5589D20D74BA00B8, 0xB2DF58DF0A312509, 0xFABC378690D64A3A, 0x700640AFC244B695,
    0xFFA652236547F3BE, 0x2B9CA498A001D059, 0x7DACA6F16787D5DE, 0xAAAD774FAC613EA3,
    0xA88583816975CD56, 0x78B71DC516FF49CA, 0xC7BF095DF702FFA6, 0x78A60B3F971783B3,
    0xCB158EF40BC75CAC, 0xA97E818DBC152B4C, 0x9FC8339D415C3999, 0x006A88C0A0D8201C,
];

impl Permutation<[Goldilocks; 68]> for MdsMatrixGoldilocksMonty {
    fn permute(&self, input: [Goldilocks; 68]) -> [Goldilocks; 68] {
        apply_circulant(&MATRIX_CIRC_MDS_68_GOLDILOCKS_MONTY, input)
    }

    fn permute_mut(&self, input: &mut [Goldilocks; 68]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<Goldilocks, 68> for MdsMatrixGoldilocksMonty {}
