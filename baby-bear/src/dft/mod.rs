//! An implementation of the FFT for `BabyBear`
mod backward;
mod forward;

pub use crate::dft::backward::backward_fft;
pub use crate::dft::forward::{forward_fft, roots_of_unity_vector};

// TODO: These are only pub for benches at the moment...
//pub mod backward;

type Real = i64;
const P: Real = 0x78000001;

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
    use alloc::vec::Vec;
    use core::iter::repeat_with;
    use p3_field::{AbstractField, Field, PrimeField64};
    use rand::{thread_rng, Rng};

    use super::{backward_fft, forward_fft, roots_of_unity_vector};
    use crate::{
        dft::{Real, P},
        BabyBear,
    };

    fn naive_convolve(us: &[Real], vs: &[Real]) -> Vec<Real> {
        let n = us.len();
        assert_eq!(n, vs.len());

        let mut conv = Vec::with_capacity(n);
        for i in 0..n {
            let mut t = 0i64;
            for j in 0..n {
                t = t + us[j] * vs[(n + i - j) % n];
            }
            conv.push(t);
        }
        conv
    }

    fn randcomplex() -> Real {
        let mut rng = thread_rng();
        (rng.gen::<u32>() % (P as u32)) as i64
    }

    fn randvec(n: usize) -> Vec<Real> {
        repeat_with(randcomplex).take(n).collect::<Vec<_>>()
    }

    #[test]
    fn forward_backward_is_identity() {
        const NITERS: usize = 100;
        let mut len = 4;
        loop {
            let roots = roots_of_unity_vector::<BabyBear>(len);
            let root = roots[1];
            let root_inv = BabyBear::from_canonical_u32(root as u32)
                .inverse()
                .as_canonical_u64() as i64;

            for _ in 0..NITERS {
                let us = randvec(len);
                let mut vs = us.clone();
                forward_fft(&mut vs, root);

                let mut ws = vs.clone();
                backward_fft(&mut ws, root_inv);

                assert!(us.iter().zip(ws).all(|(&u, w)| w == (u * len as i64) % P));
            }
            len *= 2;
            if len > 8192 {
                break;
            }
        }
    }

    /*
    #[test]
    fn convolution() {
        let mut len = 4;
        loop {
            let us = randvec(len);
            let vs = randvec(len);

            let mut fft_us = us.clone();
            forward_fft(&mut fft_us);

            let mut fft_vs = vs.clone();
            forward_fft(&mut fft_vs);

            let mut pt_prods = fft_us
                .iter()
                .zip(fft_vs)
                .map(|(&u, v)| u * v)
                .collect::<Vec<_>>();

            backward_fft(&mut pt_prods);

            let conv = naive_convolve(&us, &vs);
            assert!(conv.iter().zip(pt_prods).all(|(&c, p)| p == c * len as i64));
            len *= 2;
            if len > 8192 {
                break;
            }
        }
    }
    */
}
