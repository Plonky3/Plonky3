use core::ops::{Add, AddAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign, Mul};
use p3_mersenne_31::Mersenne31;
use p3_baby_bear::{BabyBear, from_monty_u32, to_non_canonical_u32};
use p3_field::{PrimeField32, AbstractField};

// A collection of methods able to be appied to simple integers.
pub trait IntegerLike: 
    Sized
    + Default
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + ShrAssign<usize>
    + Shr<usize, Output = Self>
    + ShlAssign<usize>
    + Shl<usize, Output = Self>
{
}

/// A implementation of Prime Fields where values are stored Non Canonically.
/// This allows for more control over when to perform modulo reductions.
/// Potentially should come with an "unsafe" label as improper use will lead to wraparound and cause errors.
/// Ensuring algorithms are correct is entirely left to the programmer.
/// Should only be used with small fields.
pub trait NonCanonicalPrimeField32:
    IntegerLike
    + Mul<Output = i128> // Multiplication of 2 field elements will yeild an i128.
{
    /// The order of the field.
    const ORDER_U32: u32;

    /// Return the internal field element value
    fn value(self) -> i64;

    /// Produce a new non canonical field element
    fn from_i64(input: i64) -> Self;

    /// Produce a new non canonical field element from some other types
    #[inline]
    fn from_i32(input: i32) -> Self {
        Self::from_i64(input as i64)
    }

    #[inline]
    fn from_u32(input: u32) -> Self {
        Self::from_i64(input as i64)
    }

    /// Return the zero of the Field
    #[inline]
    fn zero() -> Self {
        Self::from_i64(0)
    }

    /// Given x, an i128 representing a field element with |x| < 2**80 computes x' satisfying:
    /// |x'| < 2**50
    /// x' = x mod p
    /// x' = x mod 2^10
    /// This is important for large convolutions.
    fn from_small_i128(input: i128) -> Self;

    /// If we are sure a product will not overflow we don't need to pass to i128s.
    #[inline]
    fn mul_small(lhs: Self, rhs: Self) -> Self {
        Self::from_i64(Self::value(lhs) * Self::value(rhs))
    }

    /// If we want to immediately reduce a multiplication this is a simple shorthand.
    #[inline]
    fn mul_large(lhs: Self, rhs: Self) -> Self {
        Self::from_small_i128(lhs * rhs)
    }
}

/// This lets us pass from our non Canonical representatives back to canonical field elements.
/// We implement a couple of different methods to be used in different situtations.
pub trait Canonicalize<Base: PrimeField32>: NonCanonicalPrimeField32 {
   /// Given a generic non canonical field element, produce a canonical one.
   fn to_canonical(self) -> Base;

   /// Given an element in the field, produce a non canonical one
   fn from_canonical(val: Base) -> Self;

   /// Given a non canonical field element garunteed to be < n < 64 bits, produce a canonical one.
   /// The precise value of n will depend on the field. Should be faster than to_canonical for some fields.
   fn to_canonical_i_small(self) -> Base;

   /// Given a non canonical field element garunteed to be positive and < n < 64 bits, produce a canonical one.
   /// The precise value of n will depend on the field. Should be faster than to_canonical for some fields.
   fn to_canonical_u_small(self) -> Base;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Implementation of BabyBearNonCanonical 

/// Roughly a wrapper for i64's but treated as non canonical representatives of elements in the Babybear field.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
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
     Self { value: self.value + rhs.value}
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
     Self { value: self.value - rhs.value}
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
     Self { value: self.value << rhs }
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
     Self { value: self.value >> rhs }
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

// We add 2 specialised multiplication functions.

impl IntegerLike for BabyBearNonCanonical {}

impl NonCanonicalPrimeField32 for BabyBearNonCanonical {

    const ORDER_U32: u32 = (1 << 31) - (1 << 27) + 1; // BabyBear Prime

    #[inline]
    fn from_i64(input: i64) -> Self {
        Self::new(input)
    }

    #[inline]
    fn value(self) -> i64 {
        self.value
    }

    /// Given x, an i128 representing a field element with |x| < 2**80 computes x' satisfying:
    /// |x'| < 2**50
    /// x' = x mod p
    /// x' = x mod 2^10
    #[inline]
    fn from_small_i128(input: i128) -> Self {
        Self::new(barret_red_babybear(input))
    }
}

impl Canonicalize<BabyBear> for BabyBearNonCanonical {
   
    // Naive Implementation for now
    // As self.value >= -2**63 and P > 2**30, self.value + 2**33 P > 0 so the % returns a positive number.
    // This should clearly be improved at some point.
    #[inline]
    fn to_canonical(self) -> BabyBear {
        from_monty_u32((((self.value as i128) + ((Self::ORDER_U32 as i128) << 33)) % (Self::ORDER_U32 as i128)) as u32)
    }

    #[inline]
    fn from_canonical(input: BabyBear) -> Self {
        Self::from_u32(to_non_canonical_u32(input))
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
   const I: i64 = (((1 as i128) << (2*N))/(P as i128)) as i64; // I = 2^80 / P => I < 2**50
   // I: i64 = 0x22222221d950c
   const MASK: i64 = !((1 << 10) - 1); // Lets us 0 out the bottom 10 digits of an i64.

   // input = input_low + beta*input_high
   // So input_high < 2**63 and fits in an i64.
   let input_high = (input >> N) as i64; // input_high < input / beta < 2**{80 - N}

   // I, input_high are i64's so this mulitiplication can't overflow.
   let quot = (((input_high as i128)*(I as i128)) >> N) as i64;

   // Replace quot by a close value which is divisibly by 2^10.
   let quot_2adic = quot & MASK;

   // quot_2adic, P are i64's so this can't overflow.
   // sub is by construction divisible by both P and 2^10.
   let sub = (quot_2adic as i128)*(P as i128);

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


/// Roughly a wrapper for i64's but treated as non canonical representatives of elements in the Mersenne31 field.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct Mersenne31NonCanonical {
   value: i64,
}

impl Mersenne31NonCanonical {
    /// create a new `Mersenne31NonCanonical` from an `i64`.
   #[inline]
   pub(crate) const fn new(n: i64) -> Self {
     Self { value: n }
   }
}

impl Add for Mersenne31NonCanonical {
   type Output = Self;

   #[inline]
   fn add(self, rhs: Self) -> Self {
     Self { value: self.value + rhs.value}
   }
}

impl AddAssign for Mersenne31NonCanonical {
   #[inline]
   fn add_assign(&mut self, rhs: Self) {
     *self = *self + rhs;
   }
}

impl Sub for Mersenne31NonCanonical {
   type Output = Self;

   #[inline]
   fn sub(self, rhs: Self) -> Self {
     Self { value: self.value - rhs.value}
   }
}

impl SubAssign for Mersenne31NonCanonical {
   #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Shl<usize> for Mersenne31NonCanonical {
   type Output = Self;

   #[inline]
   fn shl(self, rhs: usize) -> Self {
     Self { value: self.value << rhs }
   }
}

impl ShlAssign<usize> for Mersenne31NonCanonical {
   #[inline]
   fn shl_assign(&mut self, rhs: usize) {
     *self = *self << rhs;
   }
}

impl Shr<usize> for Mersenne31NonCanonical {
   type Output = Self;

   #[inline]
   fn shr(self, rhs: usize) -> Self {
     Self { value: self.value >> rhs }
   }
}

impl ShrAssign<usize> for Mersenne31NonCanonical {
   #[inline]
   fn shr_assign(&mut self, rhs: usize) {
     *self = *self >> rhs;
   }
}

// Multiplying 2 generic elements of this field will result in an i128.
impl Mul for Mersenne31NonCanonical {
   type Output = i128;

   #[inline]
   fn mul(self, rhs: Self) -> i128 {
     (self.value as i128) * (rhs.value as i128)
   }
}

impl IntegerLike for Mersenne31NonCanonical {}

impl NonCanonicalPrimeField32 for Mersenne31NonCanonical {

   const ORDER_U32: u32 = (1 << 31) - 1; // Mersenne31 Prime

    #[inline]
    fn value(self) -> i64 {
        self.value
    }

    #[inline]
    fn from_i64(input: i64) -> Self {
        Self::new(input)
    }

    /// Given x, an i128 representing a field element with |x| < 2**80 computes x' satisfying:
    /// |x'| < 2**50
    /// x' = x mod p
    /// x' = x mod 2^10
    #[inline]
    fn from_small_i128(input: i128) -> Self {
        const LOWMASK: i128 = (1 << 42) - 1; // Gets the bits lower than 42.
        const HIGHMASK: i128 = !(LOWMASK); // Gets all bits higher than 42.

        let low_bits = (input & LOWMASK) as i64; // low_bits < 2**42
        let high_bits = ((input & HIGHMASK) >> 31) as i64; // |high_bits| < 2**(n - 31)

        // We quickly prove that low_bits + high_bits is what we want.

        // The individual bounds clearly show that low_bits + high_bits < 2**(n - 30).
        // Next observe that low_bits + high_bits = input - (2**31 - 1)(high_bits) = input mod P.
        // Finally note that 2**11 divides high_bits and so low_bits + high_bits = low_bits mod 2**11 = input mod 2**11.

        Self::new(low_bits + high_bits)
    }
}

impl Canonicalize<Mersenne31> for Mersenne31NonCanonical {

    #[inline]
    fn to_canonical(self) -> Mersenne31 {
        todo!()
    }

    #[inline]
    fn from_canonical(input: Mersenne31) -> Self {
        Self::from_u32(input.as_canonical_u32())
    }

    /// Reduces an i64 in the range -2^61 <= x < 2^61 to its (almost) canonical representative mod 2^31 - 1.
    /// Not technically canonical as we allow P as an output.
    #[inline]
    fn to_canonical_i_small(self) -> Mersenne31 {
        debug_assert!((-(1 << 61)..(1 << 61)).contains(&self.value));

        const MASK: i64 = (1 << 31) - 1;

        // Morally, our value is a i62 not a i64 as the top 3 bits are garunteed to be equal.
        let low_bits = Mersenne31::from_canonical_u32((self.value & MASK) as u32); // Get the bottom 31 bits, 0 <= low_bits < 2**31.
        let high_bits = ((self.value >> 31) & MASK) as i32; // Get the top 31 bits. 0 <= high_bits < 2**31.
        let sign_bits = (self.value >> 62) as i32; // sign_bits = 0 or -1

        // Note that high_bits + sign_bits > 0 as by assumption b[63] = b[61].

        let high = Mersenne31::from_canonical_u32((high_bits + sign_bits) as u32); // 0 <= high <= P so we can do our usual algorithm from here.

        low_bits + high
    }

    /// Reduces an i64 in the range 0 <= x < 2^62 to its (almost) canonical representative mod 2^31 - 1.
    /// Not technically canonical as we allow P as an output.
    #[inline]
    fn to_canonical_u_small(self) -> Mersenne31 {
        debug_assert!((0..(1 << 62)).contains(&self.value));

        const MASK: i64 = (1 << 31) - 1;

        let low_bits = Mersenne31::from_canonical_u32((self.value & MASK) as u32); // Get the bottom 31 bits, 0 <= low_bits <= P
        let high_bits = Mersenne31::from_canonical_u32((self.value >> 31) as u32); // Get the top 31 bits, 0 <= high_bits <= P.

        low_bits + high_bits
    }
}