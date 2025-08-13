/// Add two arrays of integers modulo `P` using packing.
///
/// This is a fallback which should only be compiled in situations where packings are
/// unavailable.
///
/// Assumes that `P` is less than `2^31` and `a + b <= 2P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod P`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
///
/// Scalar add is assumed to be a function which implements `a + b % P` with the
/// same specifications as above.
#[inline(always)]
pub fn packed_mod_add<const WIDTH: usize>(
    a: &[u32; WIDTH],
    b: &[u32; WIDTH],
    res: &mut [u32; WIDTH],
    _p: u32,
    scalar_add: fn(u32, u32) -> u32,
) {
    res.iter_mut()
        .zip(a.iter().zip(b.iter()))
        .for_each(|(r, (&a, &b))| *r = scalar_add(a, b));
}

/// Subtract two arrays of integers modulo `P` using packing.
///
/// This is a fallback which should only be compiled in situations where packings are
/// unavailable.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
///
/// Scalar add is assumed to be a function which implements `a + b % P` with the
/// same specifications as above.
#[inline(always)]
pub fn packed_mod_sub<const WIDTH: usize>(
    a: &[u32; WIDTH],
    b: &[u32; WIDTH],
    res: &mut [u32; WIDTH],
    _p: u32,
    scalar_sub: fn(u32, u32) -> u32,
) {
    res.iter_mut()
        .zip(a.iter().zip(b.iter()))
        .for_each(|(r, (&a, &b))| *r = scalar_sub(a, b));
}
