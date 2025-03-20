use alloc::vec;
use alloc::vec::Vec;

use modinverse::modinverse;
use p3_field::PrimeField64;
use p3_util::log2_ceil_u64;
use p3_util::relatively_prime_u64;
use sha3::Shake256;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// Generate alpha, the smallest integer relatively prime to `p − 1`.
pub(crate) const fn get_alpha<F: PrimeField64>() -> u64 {
    let p = F::ORDER_U64;
    let mut a = 3;

    while a < p {
        if relatively_prime_u64(a, p - 1) {
            return a;
        }
        a += 1;
    }

    panic!("No valid alpha found. Rescue does not support fields of order 2 or 3.");
}

/// Given alpha, find its multiplicative inverse in `Z/⟨p − 1⟩`.
pub(crate) fn get_inverse<F: PrimeField64>(alpha: u64) -> u64 {
    let p = F::ORDER_U64 as i128;
    modinverse(alpha as i128, p - 1)
        .expect("x^alpha not a permutation")
        .unsigned_abs()
        .try_into()
        .unwrap()
}

/// Compute the SHAKE256 variant of SHA-3.
/// This is used to generate the round constants from a seed string.
pub(crate) fn shake256_hash(seed_bytes: &[u8], num_bytes: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(seed_bytes);
    let mut reader = hasher.finalize_xof();
    let mut result = vec![0u8; num_bytes];
    reader.read(&mut result);
    result
}

/// Return 2^x for x an f32.
///
/// This is a replacement for f32::powf() when libm isn't available;
/// it is slow and shouldn't be used for anything important.
///
/// Algorithm is a direct evaluation of the corresponding Taylor
/// series. As x increases, increase the precision P until the
/// accuracy is sufficient.
#[must_use]
fn pow2_no_std<const P: usize>(x: f32) -> f32 {
    const LN_2: f32 = 0.69314718056; // log(2)
    let y = x * LN_2;
    let mut t = 1.0; // ith Taylor term = (x ln(2))^i/i!
    let mut two_pow_x = t;
    for i in 1..P {
        t *= y / (i as f32);
        two_pow_x += t;
    }
    two_pow_x
}

/// Return log2(x) for x a u64.
///
/// This is a replacement for f64::log2() when libm isn't available;
/// it is slow and shouldn't be used for anything important.
///
/// At least for inputs up to a few hundred the accuracy of this function
/// is better than 0.001.
///
/// Algorithm is just three iterations of Newton-Raphson. This is
/// sufficient for the one use in this crate. It should be generalised
/// to multiple iterations (with a suitable analysis of the precision
/// passed to pow2_no_std) before being used more widely.
#[must_use]
fn log2_no_std(x: u64) -> f32 {
    const LOG2_E: f32 = 1.44269504;
    // Initial estimate x0 = floor(log2(x))
    let x0 = log2_ceil_u64(x + 1) - 1;
    let p0 = (1 << x0) as f32; // 2^x0
    let x1 = x0 as f32 - LOG2_E * (1.0 - x as f32 / p0);
    // precision 20 determined by experiment
    let p1 = pow2_no_std::<20>(x1);
    let x2 = x1 - LOG2_E * (1.0 - x as f32 / p1);
    let p2 = pow2_no_std::<20>(x2);
    let x3 = x2 - LOG2_E * (1.0 - x as f32 / p2);
    x3
}

/// Compute an approximation to log2(binomial(n, k)).
///
/// This calculation relies on a slow version of log2 and shouldn't be
/// used for anything important.
///
/// The algorithm uses the approximation
///
///   log2(binom(n,k)) ≈ n log2(n) - k log2(k) - (n-k) log2(n-k)
///               + (log2(n) - log2(k) - log2(n-k) - log2(2π))/2
///
/// coming from Stirling's approximation for n!.
pub(crate) fn log2_binom(n: u64, k: u64) -> f32 {
    const LOG2_2PI: f32 = 2.65149613;
    let log2_n = log2_no_std(n);
    let log2_k = log2_no_std(k);
    let log2_nmk = log2_no_std(n - k);

    n as f32 * log2_n - k as f32 * log2_k - (n - k) as f32 * log2_nmk
        + 0.5 * (log2_n - log2_k - log2_nmk - LOG2_2PI)
}

#[cfg(test)]
mod test {
    use super::log2_no_std;

    const TOLERANCE: f32 = 0.001;

    #[test]
    fn test_log2_no_std() {
        let inputs = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
            200,
        ];
        let expected = [
            3.321928094887362,
            4.321928094887363,
            4.906890595608519,
            5.321928094887363,
            5.643856189774724,
            5.906890595608519,
            6.129283016944966,
            6.321928094887363,
            6.491853096329675,
            6.643856189774724,
            6.78135971352466,
            6.906890595608519,
            7.022367813028454,
            7.129283016944966,
            7.22881869049588,
            7.321928094887363,
            7.409390936137702,
            7.491853096329675,
            7.569855608330948,
            7.643856189774724,
        ];
        for (&x, y) in inputs.iter().zip(expected) {
            assert!((log2_no_std(x) - y) < TOLERANCE);
        }
    }
}
