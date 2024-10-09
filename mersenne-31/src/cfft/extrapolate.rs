//! Discrete Fourier Transform, in-place, decimation-in-time
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;

use p3_field::PackedField;

use crate::{
    cfft::{backward, forward},
    to_mersenne31_array, Mersenne31,
};

// Due to the nature of how we pad, we can combine a couple of the twiddle factors.
pub(crate) const EXTRAPOLATE_TWIDDLES_4: [Mersenne31; 2] =
    to_mersenne31_array([1759659628, 578122920]);

impl Mersenne31 {
    #[inline(always)]
    fn extrapolate_2_to_4<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
    ) {
        assert_eq!(input.len(), 2);
        assert_eq!(output.len(), 4);

        let s = input[0] + input[1];

        // We would usually multiply by the innermost twiddle (2^16) here but, due to the padding by zeroes,
        // we are able to fold this into the next layer.

        // Don't need to fully reduce here. Can do a custom field impl to get t_shift_0, t_shift_1.
        let t = input[0] - input[1];

        // The ratio of these twiddles is -(2^16 - 1) so it might be slightly faster to compute
        // t_shift_0 and (1 - 2^16) * t_shift_0 as 2^16 can be done with just shifts.
        let t_shift_0 = t * EXTRAPOLATE_TWIDDLES_4[0];
        let t_shift_1 = t * EXTRAPOLATE_TWIDDLES_4[1];

        output[0] = s + t_shift_0;
        output[1] = s + t_shift_1;
        output[2] = s - t_shift_0;
        output[3] = s - t_shift_1;
    }

    #[inline(always)]
    fn extrapolate_4_to_8<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
    ) {
        assert_eq!(input.len(), 4);
        assert_eq!(output.len(), 8);

        let a0 = input[0];
        let a1 = input[1];
        let a2 = input[2];
        let a3 = input[3];

        let a0_pos_02 = a0 + a2;
        let a1_pos_13 = a1 + a3;
        let a2_neg_02 = (a0 - a2) * backward::INV_TWIDDLES_4[0];
        let a3_neg_13 = (a1 - a3) * backward::INV_TWIDDLES_4[1];

        let b0 = a0_pos_02 + a1_pos_13;
        let b1 = a0_pos_02 - a1_pos_13;
        let b2 = a2_neg_02 + a3_neg_13;
        let b3 = a2_neg_02 - a3_neg_13;

        let b1_0 = b1 * EXTRAPOLATE_TWIDDLES_4[0];
        let b1_1 = b1 * EXTRAPOLATE_TWIDDLES_4[1];
        let b3_0 = b3 * EXTRAPOLATE_TWIDDLES_4[0];
        let b3_1 = b3 * EXTRAPOLATE_TWIDDLES_4[1];

        let c0 = b0 + b1_0;
        let c1 = b0 + b1_1;
        let c2 = b0 - b1_0;
        let c3 = b0 - b1_1;
        let c4 = (b2 + b3_0) * forward::TWIDDLES_8[0];
        let c5 = (b2 + b3_1) * forward::TWIDDLES_8[1];
        let c6 = (b2 - b3_0) * forward::TWIDDLES_8[2];
        let c7 = (b2 - b3_1) * forward::TWIDDLES_8[3];

        output[0] = c0 + c4;
        output[1] = c1 + c5;
        output[2] = c2 + c6;
        output[3] = c3 + c7;
        output[4] = c0 - c4;
        output[5] = c1 - c5;
        output[6] = c2 - c6;
        output[7] = c3 - c7;
    }

    #[inline(always)]
    fn extrapolate_8_to_16<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
    ) {
        assert_eq!(input.len(), 8);
        assert_eq!(output.len(), 16);

        Self::backward_pass(input, &backward::INV_TWIDDLES_8);

        // Safe because input.len() == 8
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 16
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        Self::extrapolate_4_to_8(input0, output0);
        Self::extrapolate_4_to_8(input1, output1);

        Self::forward_pass(output, &forward::TWIDDLES_16);
    }

    #[inline(always)]
    fn extrapolate_16_to_32<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
    ) {
        assert_eq!(input.len(), 16);
        assert_eq!(output.len(), 32);

        Self::backward_pass(input, &backward::INV_TWIDDLES_16);

        // Safe because input.len() == 16
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 32
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        Self::extrapolate_8_to_16(input0, output0);
        Self::extrapolate_8_to_16(input1, output1);

        Self::forward_pass(output, &forward::TWIDDLES_32);
    }

    #[inline(always)]
    fn extrapolate_32_to_64<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
    ) {
        assert_eq!(input.len(), 32);
        assert_eq!(output.len(), 64);

        Self::backward_pass(input, &backward::INV_TWIDDLES_32);

        // Safe because input.len() == 32
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 64
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        Self::extrapolate_16_to_32(input0, output0);
        Self::extrapolate_16_to_32(input1, output1);

        Self::forward_pass(output, &forward::TWIDDLES_64);
    }

    #[inline(always)]
    fn extrapolate_64_to_128<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
    ) {
        assert_eq!(input.len(), 64);
        assert_eq!(output.len(), 128);

        Self::backward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 64
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 128
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        Self::extrapolate_32_to_64(input0, output0);
        Self::extrapolate_32_to_64(input1, output1);

        Self::forward_pass(output, &twiddle_table[0]);
    }

    #[inline(always)]
    fn extrapolate_128_to_256<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
    ) {
        assert_eq!(input.len(), 128);
        assert_eq!(output.len(), 256);

        Self::backward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 128
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 256
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };

        Self::extrapolate_64_to_128(
            input0,
            output0,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
        );
        Self::extrapolate_64_to_128(
            input1,
            output1,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
        );

        Self::forward_pass(output, &twiddle_table[0]);
    }

    #[inline(always)]
    fn extrapolate_256_to_512<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
    ) {
        assert_eq!(input.len(), 256);
        assert_eq!(output.len(), 512);

        Self::backward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 256
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 512
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        Self::extrapolate_128_to_256(
            input0,
            output0,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
        );
        Self::extrapolate_128_to_256(
            input1,
            output1,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
        );

        Self::forward_pass(output, &twiddle_table[0]);
    }

    #[inline]
    pub fn extrapolate_fft<PF: PackedField<Scalar = Mersenne31>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
    ) {
        let n = input.len();
        if n == 1 {
            return;
        }

        assert_eq!(n, 1 << (inv_twiddle_table.len()));
        assert_eq!(output.len(), 1 << (twiddle_table.len()));
        match n {
            256 => Self::extrapolate_256_to_512(input, output, twiddle_table, inv_twiddle_table),
            128 => Self::extrapolate_128_to_256(input, output, twiddle_table, inv_twiddle_table),
            64 => Self::extrapolate_64_to_128(input, output, twiddle_table, inv_twiddle_table),
            32 => Self::extrapolate_32_to_64(input, output),
            16 => Self::extrapolate_16_to_32(input, output),
            8 => Self::extrapolate_8_to_16(input, output),
            4 => Self::extrapolate_4_to_8(input, output),
            2 => Self::extrapolate_2_to_4(input, output),
            _ => {
                debug_assert!(n > 64);

                Self::backward_pass(input, &inv_twiddle_table[0]);

                // Safe because a.len() > 64
                let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
                let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
                Self::extrapolate_fft(
                    input0,
                    output0,
                    &twiddle_table[1..],
                    &inv_twiddle_table[1..],
                );
                Self::extrapolate_fft(
                    input1,
                    output1,
                    &twiddle_table[1..],
                    &inv_twiddle_table[1..],
                );

                Self::forward_pass(output, &twiddle_table[0]);
            }
        }
    }
}
