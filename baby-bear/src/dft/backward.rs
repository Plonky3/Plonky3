use super::{split_at_mut_unchecked, Real, P};

const MONTY_BITS: u32 = 32;
const MONTY_MU: u32 = 0x88000001;

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline(always)]
fn monty_reduce(x: u64) -> u32 {
    const PP: u32 = 0x78000001;
    let t = x.wrapping_mul(MONTY_MU as u64) as u32 as u64;
    let u = t * (P as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MONTY_BITS) as u32;
    let corr = if over { PP } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Given x in [0, 2p), return the x mod p in [0, p)
#[inline(always)]
fn reduce_2p(x: i64) -> i64 {
    if x < P {
        x
    } else {
        x - P
    }
}

#[inline]
fn backward_pass(a: &mut [Real], root: Real) {
    let half_n = a.len() / 2;
    let mut w = MONTY_MU as i64;
    for i in 0..half_n {
        let s = a[i];
        //let tw = (a[i + half_n] * w) % P;
        let tw = monty_reduce(a[i + half_n] as u64 * w as u64) as i64;
        a[i] = reduce_2p(s + tw);
        a[i + half_n] = reduce_2p(P + s - tw);
        w *= root;
        w = monty_reduce(w as u64) as i64;
    }
}

#[inline]
pub fn backward_fft(a: &mut [Real], root: Real) {
    let n = a.len();

    if n > 1 {
        let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };
        let root_sqr = root * root;
        let root_sqr = monty_reduce(root_sqr as u64) as i64;
        backward_fft(a0, root_sqr);
        backward_fft(a1, root_sqr);

        backward_pass(a, root);
    }
}
