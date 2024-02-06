use core::ops::{Add, AddAssign, Mul, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};

use p3_field::{Canonicalize, IntegerLike, NonCanonicalPrimeField32, PrimeField32};

use crate::{from_monty_u32, BabyBear};

const _P: u32 = 0x78000001;
const _MONTY_BITS: u32 = 31;
const _MONTY_MASK: u32 = (1 << _MONTY_BITS) - 1;
const _MONTY_MU: u32 = 0x8000001;

// Implementation of BabyBearNonCanonical

/// Roughly a wrapper for i64's but treated as non canonical representatives of elements in the Babybear field.
#[derive(Debug, Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct BabyBearNonCanonical {
    value: i64,
}

impl BabyBearNonCanonical {
    /// create a new `BabyBearNonCanonical` from an `i64`.
    #[inline]
    pub(crate) const fn new(n: i64) -> Self {
        Self { value: n }
    }
}

impl Add for BabyBearNonCanonical {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl AddAssign for BabyBearNonCanonical {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for BabyBearNonCanonical {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl SubAssign for BabyBearNonCanonical {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Shl<usize> for BabyBearNonCanonical {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self {
        Self {
            value: self.value << rhs,
        }
    }
}

impl ShlAssign<usize> for BabyBearNonCanonical {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = *self << rhs;
    }
}

impl Shr<usize> for BabyBearNonCanonical {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self {
        Self {
            value: self.value >> rhs,
        }
    }
}

impl ShrAssign<usize> for BabyBearNonCanonical {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

// Multiplying 2 generic elements of this field will result in an i128.
impl Mul for BabyBearNonCanonical {
    type Output = i128;

    #[inline]
    fn mul(self, rhs: Self) -> i128 {
        (self.value as i128) * (rhs.value as i128)
    }
}

// Given a non canonical field element and an element of the base field we implement a product such that |output| < max(2^40, |non canonical input|)
impl Mul<BabyBear> for BabyBearNonCanonical {
    type Output = BabyBearNonCanonical;

    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        // We use a variation of Mongomery Multiplication as we assume both values are in Monty Form.

        // This is currently a placeholder for the actual algorithm. DO NOT USE IN PRODUCTION IT IS CLEARLY WRONG. JUST HERE FOR SOME TESTING PURPOSES.

        let prod = self.value.wrapping_mul(rhs.value as i64); // This is an i96 at worst.
                                                              // let t = prod.wrapping_mul(MONTY_MU as i128) & (MONTY_MASK as i128);
                                                              // let u = (t as u64) * (P as u64);

        // let x_sub_u = prod - (u as i128);
        // let x_sub_u_hi = (x_sub_u >> 31) as i64;

        Self::from_i64(prod)
    }
}

// We add 2 specialised multiplication functions.

impl IntegerLike for BabyBearNonCanonical {}

impl NonCanonicalPrimeField32 for BabyBearNonCanonical {
    const ORDER_U32: u32 = (1 << 31) - (1 << 27) + 1; // BabyBear Prime

    #[inline]
    fn from_i64(input: i64) -> Self {
        Self::new(input)
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self.value
    }

    /// Given x, an i128 representing a field element with |x| < 2**80 computes x' satisfying:
    /// |x'| < 2**50
    /// x' = x mod p
    /// x' = x mod 2^10
    #[inline]
    unsafe fn from_small_i128(input: i128) -> Self {
        Self::new(barret_red_babybear(input))
    }
}

impl Canonicalize<BabyBear> for BabyBearNonCanonical {
    // Naive Implementation for now
    // As self.value >= -2**63 and P > 2**30, self.value + 2**33 P > 0 so the % returns a positive number.
    // This should clearly be improved at some point.
    #[inline]
    fn to_canonical(self) -> BabyBear {
        from_monty_u32(
            (((self.value as i128) + ((Self::ORDER_U32 as i128) << 33)) % (Self::ORDER_U32 as i128))
                as u32,
        )
    }

    /// Currently we take the "Monty" form.
    /// Need to be careful of this.
    #[inline]
    fn from_canonical(input: BabyBear) -> Self {
        Self::from_u32(input.value)
    }

    #[inline]
    fn from_canonical_to_i31(input: BabyBear) -> Self {
        let val = input.value;
        if val & (1 << 30) != 0 {
            Self::from_u32(val)
        } else {
            Self::from_i32((val as i32) - (BabyBear::ORDER_U32 as i32))
        }
    }

    // Naive Implementation for now
    #[inline]
    fn to_canonical_i_small(self) -> BabyBear {
        self.to_canonical()
    }

    // Naive Implementation for now but will work for any positive value.
    // Should improve this at some point.
    #[inline]
    fn to_canonical_u_small(self) -> BabyBear {
        from_monty_u32((self.value % Self::ORDER_U32 as i64) as u32)
    }
}

/// Given |x| < 2^80 compute x' such that:
/// |x'| < 2**50
/// x' = x mod p
/// x' = x mod 2^10
/// See Thm 1 (Below function) for a proof that this function is correct.
#[inline]
fn barret_red_babybear(input: i128) -> i64 {
    const N: usize = 40; // beta = 2^N, fixing N = 40 here
    const P: u32 = (1 << 31) - (1 << 27) + 1; // Babybear Prime
    const I: i64 = (((1_i128) << (2 * N)) / (P as i128)) as i64; // I = 2^80 / P => I < 2**50
                                                                 // I: i64 = 0x22222221d950c
    const MASK: i64 = !((1 << 10) - 1); // Lets us 0 out the bottom 10 digits of an i64.

    // input = input_low + beta*input_high
    // So input_high < 2**63 and fits in an i64.
    let input_high = (input >> N) as i64; // input_high < input / beta < 2**{80 - N}

    // I, input_high are i64's so this mulitiplication can't overflow.
    let quot = (((input_high as i128) * (I as i128)) >> N) as i64;

    // Replace quot by a close value which is divisibly by 2^10.
    let quot_2adic = quot & MASK;

    // quot_2adic, P are i64's so this can't overflow.
    // sub is by construction divisible by both P and 2^10.
    let sub = (quot_2adic as i128) * (P as i128);

    (input - sub) as i64
}

// Theorem 1:
// Given |x| < 2^80, barret_red(x) computes an x' such that:
//       x' = x mod p
//       x' = x mod 2^10
//       |x'| < 2**50.
///////////////////////////////////////////////////////////////////////////////////////
// PROOF:
// By construction P, 2**10 | sub and so we immediately see that
// x' = x mod p
// x' = x mod 2^10.
//
// It remains to prove that |x'| < 2**50.
//
// We start by introducing some simple inequalities and relations bewteen our variables:
//
// First consider the relationship between bitshift and division.
// It's easy to check that for all x:
// 1: (x >> N) <= x / 2**N <= 1 + (x >> N)
//
// Similarly, as our mask just 0's the last 10 bits,
// 2: x + 1 - 2^10 <= x & mask <= x
//
// Now if x, y are positive integers then
// (x / y) - 1 <= x // y <= x / y
// Where // denotes integer division.
//
// From this last inequality we immediately derive:
// 3: (2**{2N} / P) - 1 <= I <= (2**{2N} / P)
// 3a: 2**{2N} - P <= PI
//
// Finally, note that by definition:
// input = input_high*(2**N) + input_low
// Hence a simple rearrangment gets us
// 4: input_high*(2**N) = input - input_low
//
//
// We now need to split into cases depending on the sign of input.
// Note that if x = 0 then x' = 0 so that case is trivial.
///////////////////////////////////////////////////////////////////////////
// CASE 1: input > 0
//
// If input > 0 then:
// sub = Q*P = ((((input >> N) * I) >> N) & mask) * P <= P * (input / 2**{N}) * (2**{2N} / P) / 2**{N} = input
// So input - sub >= 0.
//
// We need to improve our bound on Q. Observe that:
// Q = (((input_high * I) >> N) & mask)
// --(2)   => Q + (2^10 - 1) >= (input_high * I) >> N)
// --(1)   => Q + 2^10 >= (I*x_high)/(2**N)
//         => (2**N)*Q + 2^10*(2**N) >= I*x_high
//
// Hence we find that:
// (2**N)*Q*P + 2^10*(2**N)*P >= input_high*I*P
// --(3a)                     >= input_high*2**{2N} - P*input_high
// --(4)                      >= (2**N)*input - (2**N)*input_low - (2**N)*input_high   (Assuming P < 2**N)
//
// Dividing by 2**N we get
// Q*P + 2^{10}*P >= input - input_low - input_high
// which rearranges to
// x' = input - Q*P <= 2^{10}*P + input_low + input_high
//
// Picking N = 40 we see that 2^{10}*P, input_low, input_high are all bounded by 2**40
// Hence x' < 2**42 < 2**50 as desired.
//
//
//
///////////////////////////////////////////////////////////////////////////
// CASE 2: input < 0
//
// This case will be similar but all our inequalities will change slightly as negatives complicate things.
// First observe that:
// (input >> N) * I   >= (input >> N) * 2**(2N) / P
//                    >= (1 + (input / 2**N)) * 2**(2N) / P
//                    >= (2**N + input) * 2**N / P
//
// Thus:
// Q = ((input >> N) * I) >> N >= ((2**N + input) * 2**N / P) >> N
//                             >= ((2**N + input) / P) - 1
//
// And so sub = Q*P >= 2**N - P + input.
// Hence input - sub < 2**N - P.
//
// Thus if input - sub > 0 then |input - sub| < 2**50.
// Thus we are left with bounding -(input - sub) = (sub - input).
// Again we will proceed by improving our bound on Q.
//
// Q = (((input_high * I) >> N) & mask)
// --(2)   => Q <= (input_high * I) >> N) <= (I*x_high)/(2**N)
// --(1)   => Q <= (I*x_high)/(2**N)
//         => (2**N)*Q <= I*x_high
//
// Hence we find that:
// (2**N)*Q*P <= input_high*I*P
// --(3a)     <= input_high*2**{2N} - P*input_high
// --(4)      <= (2**N)*input - (2**N)*input_low - (2**N)*input_high   (Assuming P < 2**N)
//
// Dividing by 2**N we get
// Q*P <= input - input_low - input_high
// which rearranges to
// -x' = -input + Q*P <= -input_high - input_low < 2**50
//
// This completes the proof.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
