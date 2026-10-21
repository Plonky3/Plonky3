//! Probabilistic primality: trial division, Miller-Rabin, the strong Lucas test and BPSW.
//!
//! Every test runs in time that depends on its input, which must be public.

mod sample;

use alloc::vec;

use num_bigint::BigUint;

pub use self::sample::{PrimePolicy, PrimeSample, sample_prime};
use crate::error::PrimeError;
use crate::mont::MontModulus;
use crate::ring::ModRing;
use crate::uint::Uint;

/// The first 256 primes, from 2 to 1619.
pub const SMALL_PRIMES: [u16; 256] = small_primes();

const fn small_primes() -> [u16; 256] {
    let mut out = [0u16; 256];
    let mut found = 0;
    let mut c = 2u16;
    while found < 256 {
        let mut i = 0;
        let mut prime = true;
        while i < found {
            if c.is_multiple_of(out[i]) {
                prime = false;
                break;
            }
            i += 1;
        }
        if prime {
            out[found] = c;
            found += 1;
        }
        c += 1;
    }
    out
}

/// Products of consecutive odd small primes below `2^64`, and the index of each group's first prime.
struct Groups {
    product: [u64; 64],
    first: [u16; 65],
    len: usize,
}

const GROUPS: Groups = groups();

const fn groups() -> Groups {
    let mut g = Groups {
        product: [0; 64],
        first: [0; 65],
        len: 0,
    };
    // Prime 2 is handled by parity.
    let mut i = 1;
    while i < 256 {
        g.first[g.len] = i as u16;
        let mut product = 1u64;
        while i < 256 {
            let Some(next) = product.checked_mul(SMALL_PRIMES[i] as u64) else {
                break;
            };
            product = next;
            i += 1;
        }
        g.product[g.len] = product;
        g.len += 1;
    }
    g.first[g.len] = 256;
    g
}

/// For each odd small prime `p`: `p^-1 mod 2^64` and `floor((2^64 - 1) / p)`.
///
/// `r` is a multiple of `p` exactly when `r p^-1 mod 2^64` is at most the bound.
const DIVISIBILITY: [(u64, u64); 256] = divisibility();

const fn divisibility() -> [(u64, u64); 256] {
    let mut out = [(0, 0); 256];
    let mut i = 1;
    while i < 256 {
        let p = SMALL_PRIMES[i] as u64;
        let mut inv = 1u64;
        let mut k = 0;
        while k < 6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
            k += 1;
        }
        out[i] = (inv, u64::MAX / p);
        i += 1;
    }
    out
}

/// `n mod d` for a nonzero `d`.
fn rem_u64<const L: usize>(n: &Uint<L>, d: u64) -> u64 {
    n.0.iter().rev().fold(0u64, |r, &limb| {
        ((((r as u128) << 64) | limb as u128) % d as u128) as u64
    })
}

/// Settles `n` by the small primes: `Some(true)` for a small prime, `Some(false)` for a multiple of one or `n < 2`.
///
/// Returns `None` when `n` has no factor among them and is not one of them.
#[must_use]
pub fn trial_division<const L: usize>(n: &Uint<L>) -> Option<bool> {
    if n.bits() <= 11 {
        let v = n.0[0] as u16;
        if v <= SMALL_PRIMES[255] {
            return Some(SMALL_PRIMES.binary_search(&v).is_ok());
        }
    }
    if !n.is_odd() {
        return Some(false);
    }
    for g in 0..GROUPS.len {
        let r = rem_u64(n, GROUPS.product[g]);
        for &(inv, bound) in &DIVISIBILITY[GROUPS.first[g] as usize..GROUPS.first[g + 1] as usize] {
            if r.wrapping_mul(inv) <= bound {
                return Some(false);
            }
        }
    }
    None
}

/// Whether `n` is a strong probable prime to `base`.
///
/// A base divisible by `n` passes.
#[must_use]
pub fn miller_rabin<const L: usize>(n: &MontModulus<L>, base: &Uint<L>) -> bool {
    let a = n.from_uint(base);
    if a == n.zero() {
        return true;
    }
    let one = n.one();
    let minus_one = n.neg(one);
    let (n_minus_1, _) = n.modulus_uint().sbb(&Uint::ONE);
    let s = trailing_zeros(&n_minus_1);
    let d = n_minus_1.shr(s);
    let mut x = n.pow(a, &d.0);
    if x == one || x == minus_one {
        return true;
    }
    for _ in 1..s {
        x = n.square(x);
        if x == minus_one {
            return true;
        }
        if x == one {
            return false;
        }
    }
    false
}

/// The number of trailing zero bits of a nonzero `x`.
fn trailing_zeros<const L: usize>(x: &Uint<L>) -> u32 {
    let i = x.0.iter().position(|&w| w != 0).unwrap_or(0);
    64 * i as u32 + x.0[i].trailing_zeros()
}

/// The Jacobi symbol `(a / n)` for an odd `n`.
fn jacobi<const L: usize>(a: i64, n: &Uint<L>) -> i32 {
    let n_mod_8 = n.0[0] & 7;
    let mut sign = 1;
    // (-1 / n) = -1 exactly when n = 3 mod 4.
    if a < 0 && n_mod_8 & 3 == 3 {
        sign = -sign;
    }
    let mut a = a.unsigned_abs();
    if a == 0 {
        return 0;
    }
    // (2 / n) = -1 exactly when n = 3, 5 mod 8.
    while a & 1 == 0 {
        a >>= 1;
        if n_mod_8 == 3 || n_mod_8 == 5 {
            sign = -sign;
        }
    }
    // Reciprocity for odd a and n, then the single-limb symbol (n mod a / a).
    if a & 3 == 3 && n_mod_8 & 3 == 3 {
        sign = -sign;
    }
    sign * jacobi_u64(rem_u64(n, a), a)
}

/// The Jacobi symbol `(a / n)` for an odd `n`.
const fn jacobi_u64(mut a: u64, mut n: u64) -> i32 {
    let mut sign = 1;
    a %= n;
    while a != 0 {
        while a & 1 == 0 {
            a >>= 1;
            let r = n & 7;
            if r == 3 || r == 5 {
                sign = -sign;
            }
        }
        core::mem::swap(&mut a, &mut n);
        if a & 3 == 3 && n & 3 == 3 {
            sign = -sign;
        }
        a %= n;
    }
    if n == 1 { sign } else { 0 }
}

/// Whether `n` is a perfect square.
fn is_square<const L: usize>(n: &Uint<L>) -> bool {
    let n = BigUint::from(*n);
    let r = n.sqrt();
    &r * &r == n
}

/// Whether `n` is a strong Lucas probable prime with Selfridge's parameters.
///
/// `D` is the first of `5, -7, 9, -11, ...` with `(D / n) = -1`, `P = 1` and `Q = (1 - D) / 4`.
#[must_use]
pub fn strong_lucas_selfridge<const L: usize>(n: &MontModulus<L>) -> bool {
    let m = n.modulus_uint();
    let mut d: i64 = 5;
    let mut tries = 0;
    loop {
        match jacobi(d, m) {
            -1 => break,
            // A proper common factor with D proves n composite.
            0 if Uint::from_u64(d.unsigned_abs()) < *m => return false,
            _ => {}
        }
        tries += 1;
        // No D exists for a square, so test for one once the search runs long.
        if tries == 8 && is_square(m) {
            return false;
        }
        d = if d > 0 { -(d + 2) } else { 2 - d };
    }
    let q = (1 - d) / 4;
    let dd = n.from_i64(d);
    let qq = n.from_i64(q);

    // n + 1 = k 2^s with k odd; the spare bit keeps n + 1 in range.
    let (n_plus_1, _) = m.adc(&Uint::ONE);
    let s = trailing_zeros(&n_plus_1);
    let k = n_plus_1.shr(s);

    // U_1 = 1, V_1 = P = 1, then double and step along the bits of k.
    let mut u = n.one();
    let mut v = n.one();
    let mut qk = qq;
    for i in (0..k.bits() - 1).rev() {
        u = n.mul(u, v);
        v = n.sub(n.square(v), n.add(qk, qk));
        qk = n.square(qk);
        if k.bit(i) {
            let u1 = n.half(n.add(u, v));
            v = n.half(n.add(n.mul(dd, u), v));
            u = u1;
            qk = n.mul(qk, qq);
        }
    }
    let zero = n.zero();
    if u == zero || v == zero {
        return true;
    }
    for _ in 1..s {
        v = n.sub(n.square(v), n.add(qk, qk));
        qk = n.square(qk);
        if v == zero {
            return true;
        }
    }
    false
}

/// The Baillie-PSW test: trial division, Miller-Rabin to base 2, then the strong Lucas test.
#[must_use]
pub fn is_bpsw<const L: usize>(n: &MontModulus<L>) -> bool {
    if let Some(settled) = trial_division(n.modulus_uint()) {
        return settled;
    }
    miller_rabin(n, &Uint::from_u64(2)) && strong_lucas_selfridge(n)
}

/// A uniform integer in `[0, bound)` by rejection, for a nonzero `bound`.
pub(crate) fn draw_below<const L: usize>(
    fill: &mut impl FnMut(&mut [u8]),
    bound: &Uint<L>,
) -> Result<Uint<L>, PrimeError> {
    let (max, _) = bound.sbb(&Uint::ONE);
    let bits = max.bits();
    let mask = Uint::<L>::MAX.shr(Uint::<L>::BITS - bits);
    let mut buf = vec![0u8; bits.div_ceil(8) as usize];
    // Each draw lands below the bound with probability above 1/2.
    for _ in 0..128 {
        fill(&mut buf);
        let x = Uint::<L>::from_le_bytes(&buf).ok_or(PrimeError::Exhausted)?;
        let x = Uint(core::array::from_fn(|i| x.0[i] & mask.0[i]));
        if x < *bound {
            return Ok(x);
        }
    }
    Err(PrimeError::Exhausted)
}

/// BPSW, then `rounds` Miller-Rabin tests with bases drawn uniformly from `[2, n - 2]`.
///
/// A composite passes with probability at most `4^-rounds`.
pub(crate) fn is_prime_with<const L: usize>(
    n: &MontModulus<L>,
    rounds: u32,
    fill: &mut impl FnMut(&mut [u8]),
) -> Result<bool, PrimeError> {
    if let Some(settled) = trial_division(n.modulus_uint()) {
        return Ok(settled);
    }
    if !(miller_rabin(n, &Uint::from_u64(2)) && strong_lucas_selfridge(n)) {
        return Ok(false);
    }
    // n exceeds the small primes here, so [2, n - 2] is not empty.
    let (span, _) = n.modulus_uint().sbb(&Uint::from_u64(3));
    for _ in 0..rounds {
        let (base, _) = draw_below(fill, &span)?.adc(&Uint::from_u64(2));
        if !miller_rabin(n, &base) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// BPSW and 64 Miller-Rabin rounds, on bases from a fixed-seed SplitMix64 stream.
pub(crate) fn is_prime_fixed<const L: usize>(n: &MontModulus<L>) -> bool {
    let mut state = 0x7033_6d6f_6475_6c61u64;
    let mut fill = |out: &mut [u8]| {
        for chunk in out.chunks_mut(8) {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
        }
    };
    is_prime_with(n, 64, &mut fill).unwrap_or(false)
}
