//! An implementation of the FFT for `BabyBear`
mod backward;
mod forward;

pub use crate::dft::backward::backward_fft;
pub use crate::dft::forward::{
    batch_forward_fft, forward_fft, four_step_fft, roots_of_unity_table,
};

// TODO: These are only pub for benches at the moment...
//pub mod backward;

const P: u32 = 0x78000001;

/// Copied from Rust nightly sources
#[inline(always)]
unsafe fn from_raw_parts_mut<'a, T>(data: *mut T, len: usize) -> &'a mut [T] {
    unsafe { &mut *core::ptr::slice_from_raw_parts_mut(data, len) }
}

/// Copied from Rust nightly sources
#[inline(always)]
pub(crate) unsafe fn split_at_mut_unchecked<T>(v: &mut [T], mid: usize) -> (&mut [T], &mut [T]) {
    let len = v.len();
    let ptr = v.as_mut_ptr();

    // SAFETY: Caller has to check that `0 <= mid <= self.len()`.
    //
    // `[ptr; mid]` and `[mid; len]` are not overlapping, so returning
    // a mutable reference is fine.
    unsafe {
        (
            from_raw_parts_mut(ptr, mid),
            from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;
    use core::iter::repeat_with;

    use p3_baby_bear::{BabyBear, BabyBearParameters};
    use p3_field::{AbstractField, Field, PrimeField32};
    use p3_util::reverse_slice_index_bits;
    use rand::{thread_rng, Rng};

    use crate::dft::*;

    fn naive_convolve(us: &[BabyBear], vs: &[BabyBear]) -> Vec<BabyBear> {
        let n = us.len();
        assert_eq!(n, vs.len());

        let mut conv = Vec::with_capacity(n);
        for i in 0..n {
            let mut t = BabyBear::zero();
            for j in 0..n {
                t = t + us[j] * vs[(n + i - j) % n];
            }
            conv.push(t);
        }
        conv
    }

    fn randcomplex() -> u32 {
        let mut rng = thread_rng();
        rng.gen::<u32>() % BabyBear::ORDER_U32
    }

    fn randvec(n: usize) -> Vec<u32> {
        repeat_with(randcomplex).take(n).collect::<Vec<_>>()
    }

    #[test]
    fn test_forward_16() {
        const NITERS: usize = 100;
        let len = 16;
        let root_table = roots_of_unity_table::<BabyBear>(len);

        for _ in 0..NITERS {
            let us = randvec(len);
            /*
            //let us = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            // monty form of [0..16)
            let us = vec![
                0, 268435454, 536870908, 805306362, 1073741816, 1342177270, 1610612724, 1879048178,
                134217711, 402653165, 671088619, 939524073, 1207959527, 1476394981, 1744830435,
                2013265889,
            ];
            */

            let mut vs = us.clone();
            forward_fft(&mut vs, &root_table);
            reverse_slice_index_bits(&mut vs);

            let mut ws = us.clone();
            four_step_fft(&mut ws, &root_table);

            assert!(vs.iter().zip(ws).all(|(&v, w)| v == w));
        }
    }

    #[test]
    fn forward_backward_is_identity() {
        const NITERS: usize = 100;
        let mut len = 16;
        loop {
            let root_table = roots_of_unity_table::<BabyBearParameters>(len);
            let root = root_table[0][0];
            //let root_inv = BabyBear { value: root }.inverse().value;
            let root_inv = BabyBear::new_monty(root).inverse().value;

            for _ in 0..NITERS {
                let us = randvec(len);
                let mut vs = us.clone();
                //forward_fft(&mut vs, &root_table);
                four_step_fft(&mut vs, &root_table);

                reverse_slice_index_bits(&mut vs);

                let mut ws = vs.clone();
                backward_fft(&mut ws, root_inv);

                assert!(us
                    .iter()
                    .zip(ws)
                    .all(|(&u, w)| w as u64 == (u as u64 * len as u64) % P as u64));
            }
            len *= 2;
            if len > 8192 {
                break;
            }
        }
    }

    #[test]
    fn convolution() {
        const NITERS: usize = 4;
        let mut len = 4;
        loop {
            let root_table = roots_of_unity_table(len);
            let root = root_table[0][0];
            let root_inv = BabyBear { value: root }.inverse().value;

            for _ in 0..NITERS {
                let us = randvec(len);
                let vs = randvec(len);

                let mut fft_us = us.clone();
                forward_fft(&mut fft_us, &root_table);

                let mut fft_vs = vs.clone();
                forward_fft(&mut fft_vs, &root_table);

                let mut pt_prods = fft_us
                    .iter()
                    .zip(fft_vs)
                    .map(|(&u, v)| {
                        let prod = BabyBear { value: u } * BabyBear { value: v };
                        prod.value
                    })
                    .collect::<Vec<_>>();

                backward_fft(&mut pt_prods, root_inv);

                let bus = us
                    .iter()
                    .map(|&u| BabyBear { value: u })
                    .collect::<Vec<_>>();
                let bvs = vs
                    .iter()
                    .map(|&v| BabyBear { value: v })
                    .collect::<Vec<_>>();
                let bconv = naive_convolve(&bus, &bvs);
                let conv = bconv
                    .iter()
                    .map(|&BabyBear { value }| value)
                    .collect::<Vec<_>>();

                assert!(conv
                    .iter()
                    .zip(pt_prods)
                    .all(|(&c, p)| p as u64 == (c as u64 * len as u64) % P as u64));
            }
            len *= 2;
            if len > 8192 {
                break;
            }
        }
    }
}
