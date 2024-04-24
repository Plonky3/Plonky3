use super::{split_at_mut_unchecked, Real, P};

#[inline]
fn backward_pass(a: &mut [Real], root: Real) {
    let half_n = a.len() / 2;
    let mut w = 1;
    for i in 0..half_n {
        let s = a[i];
        let tw = (a[i + half_n] * w) % P;
        a[i] = (s + tw) % P;
        a[i + half_n] = (P + s - tw) % P;
        w *= root;
        w %= P;
    }
}

#[inline]
pub fn backward_fft(a: &mut [Real], root: Real) {
    let n = a.len();

    if n > 1 {
        let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };
        let root_sqr = (root * root) % P;
        backward_fft(a0, root_sqr);
        backward_fft(a1, root_sqr);

        backward_pass(a, root);
    }
}
