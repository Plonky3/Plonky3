//! Discrete Fourier Transform, in-place, decimation-in-time
//!
//! Straightforward recursive algorithm, "unrolled" up to size 256.
//!
//! Inspired by Bernstein's djbfft: https://cr.yp.to/djbfft.html

extern crate alloc;
use alloc::vec::Vec;

use p3_field::PackedField;

use crate::{FieldParameters, MontyField31, TwoAdicData};

impl<MP: FieldParameters + TwoAdicData> MontyField31<MP> {
    #[inline(always)]
    fn extrapolate_2_to_4<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 2);
        assert_eq!(output.len(), 4);

        let s = input[0] + input[1];
        let t = input[0] - input[1];

        let s_shift = s * shifts[0];
        let t_shift = t * shifts[1];
        let t_shift_adjust = t_shift * MP::ROOTS_8.as_ref()[1];

        output[0] = s_shift + t_shift;
        output[1] = s_shift + t_shift_adjust;
        output[2] = s_shift - t_shift;
        output[3] = s_shift - t_shift_adjust;
    }

    #[inline(always)]
    fn extrapolate_4_to_8<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 4);
        assert_eq!(output.len(), 8);

        let a0 = input[0] + input[2];
        let a1 = input[1] + input[3];
        let a2 = input[0] - input[2];
        let a3 = (input[1] - input[3]) * MP::INV_ROOTS_8.as_ref()[1];

        let a00 = a0 + a1;
        let a01 = a0 - a1;
        let a10 = a2 + a3;
        let a11 = a2 - a3;

        let a00_shift = a00 * shifts[0];
        let a01_shift = a01 * shifts[1];
        let a01_shift_adjusted = a01_shift * MP::ROOTS_8.as_ref()[1];
        let a10_shift = a10 * shifts[2];
        let a11_shift = a11 * shifts[3];
        let a11_shift_adjusted = a11_shift * MP::ROOTS_8.as_ref()[1];

        output[0] = a00_shift + a01_shift;
        output[1] = a00_shift + a01_shift_adjusted;
        output[2] = a00_shift - a01_shift;
        output[3] = a00_shift - a01_shift_adjusted;
        output[4] = a10_shift + a11_shift;
        output[5] = a10_shift + a11_shift_adjusted;
        output[6] = a10_shift - a11_shift;
        output[7] = a10_shift - a11_shift_adjusted;

        Self::backward_pass(output, MP::ROOTS_8.as_ref());
    }

    #[inline(always)]
    fn extrapolate_8_to_16<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 8);
        assert_eq!(output.len(), 16);

        Self::forward_pass(input, MP::INV_ROOTS_8.as_ref());

        // Safe because input.len() == 8
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 16
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };
        Self::extrapolate_4_to_8(input0, output0, shifts0);
        Self::extrapolate_4_to_8(input1, output1, shifts1);

        Self::backward_pass(output, MP::ROOTS_16.as_ref());
    }

    #[inline(always)]
    fn extrapolate_16_to_32<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 16);
        assert_eq!(output.len(), 32);

        Self::forward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 16
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 32
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };
        Self::extrapolate_8_to_16(input0, output0, shifts0);
        Self::extrapolate_8_to_16(input1, output1, shifts1);

        Self::backward_pass(output, &twiddle_table[0]);
    }

    #[inline(always)]
    fn extrapolate_32_to_64<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 32);
        assert_eq!(output.len(), 64);

        Self::forward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 32
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 64
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };
        Self::extrapolate_16_to_32(
            input0,
            output0,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts0,
        );
        Self::extrapolate_16_to_32(
            input1,
            output1,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts1,
        );

        Self::backward_pass(output, &twiddle_table[0]);
    }

    #[inline(always)]
    fn extrapolate_64_to_128<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 64);
        assert_eq!(output.len(), 128);

        Self::forward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 64
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 128
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };
        Self::extrapolate_32_to_64(
            input0,
            output0,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts0,
        );
        Self::extrapolate_32_to_64(
            input1,
            output1,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts1,
        );

        Self::backward_pass(output, &twiddle_table[0]);
    }

    #[inline(always)]
    fn extrapolate_128_to_256<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 128);
        assert_eq!(output.len(), 256);

        Self::forward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 128
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 256
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };

        Self::extrapolate_64_to_128(
            input0,
            output0,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts0,
        );
        Self::extrapolate_64_to_128(
            input1,
            output1,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts1,
        );

        Self::backward_pass(output, &twiddle_table[0]);
    }

    #[inline(always)]
    fn extrapolate_256_to_512<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
        shifts: &[Self],
    ) {
        assert_eq!(input.len(), 256);
        assert_eq!(output.len(), 512);

        Self::forward_pass(input, &inv_twiddle_table[0]);

        // Safe because input.len() == 256
        let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
        // Safe because input.len() == 512
        let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
        let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };
        Self::extrapolate_128_to_256(
            input0,
            output0,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts0,
        );
        Self::extrapolate_128_to_256(
            input1,
            output1,
            &twiddle_table[1..],
            &inv_twiddle_table[1..],
            shifts1,
        );

        Self::backward_pass(output, &twiddle_table[0]);
    }

    #[inline]
    pub fn extrapolate_fft<PF: PackedField<Scalar = MontyField31<MP>>>(
        input: &mut [PF],
        output: &mut [PF],
        twiddle_table: &[Vec<Self>],
        inv_twiddle_table: &[Vec<Self>],
        shifts: &[Self],
    ) {
        let n = input.len();
        if n == 1 {
            return;
        }

        assert_eq!(n, 1 << (inv_twiddle_table.len() + 1));
        assert_eq!(output.len(), 1 << (twiddle_table.len() + 1));
        match n {
            256 => Self::extrapolate_256_to_512(
                input,
                output,
                twiddle_table,
                inv_twiddle_table,
                shifts,
            ),
            128 => Self::extrapolate_128_to_256(
                input,
                output,
                twiddle_table,
                inv_twiddle_table,
                shifts,
            ),
            64 => {
                Self::extrapolate_64_to_128(input, output, twiddle_table, inv_twiddle_table, shifts)
            }
            32 => {
                Self::extrapolate_32_to_64(input, output, twiddle_table, inv_twiddle_table, shifts)
            }
            16 => {
                Self::extrapolate_16_to_32(input, output, twiddle_table, inv_twiddle_table, shifts)
            }
            8 => Self::extrapolate_8_to_16(input, output, shifts),
            4 => Self::extrapolate_4_to_8(input, output, shifts),
            2 => Self::extrapolate_2_to_4(input, output, shifts),
            _ => {
                debug_assert!(n > 64);

                Self::forward_pass(input, &inv_twiddle_table[0]);

                // Safe because a.len() > 64
                let (input0, input1) = unsafe { input.split_at_mut_unchecked(input.len() / 2) };
                let (output0, output1) = unsafe { output.split_at_mut_unchecked(output.len() / 2) };
                let (shifts0, shifts1) = unsafe { shifts.split_at_unchecked(shifts.len() / 2) };
                Self::extrapolate_fft(
                    input0,
                    output0,
                    &twiddle_table[1..],
                    &inv_twiddle_table[1..],
                    shifts0,
                );
                Self::extrapolate_fft(
                    input1,
                    output1,
                    &twiddle_table[1..],
                    &inv_twiddle_table[1..],
                    shifts1,
                );

                Self::backward_pass(output, &twiddle_table[0]);
            }
        }
    }
}
