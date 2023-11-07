use core::ops::{Add, AddAssign, Mul, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};

const P: i64 = (1 << 31) - (1 << 27) + 1; // BabyBear Prime

/// This type is a wrapper for i64 but will let us specialise the karatsuba_convolution code to the BabyBear Field.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct SignedBabyBearNonCanonical {
   value: i64,
}

impl SignedBabyBearNonCanonical {
    /// create a new `SignedBabyBearNonCanonical` from an `i64`.
   #[inline]
   pub(crate) const fn new(n: i64) -> Self {
     Self { value: n }
   }
}

impl Add for SignedBabyBearNonCanonical {
   type Output = Self;

   #[inline]
   fn add(self, rhs: Self) -> Self {
     Self { value: self.value + rhs.value}
   }
}

impl AddAssign for SignedBabyBearNonCanonical {
   #[inline]
   fn add_assign(&mut self, rhs: Self) {
     *self = *self + rhs;
   }
}

impl Sub for SignedBabyBearNonCanonical {
   type Output = Self;

   #[inline]
   fn sub(self, rhs: Self) -> Self {
     Self { value: self.value - rhs.value}
   }
}

impl SubAssign for SignedBabyBearNonCanonical {
   #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for SignedBabyBearNonCanonical {
   type Output = SignedBabyBearLarge;

   /// Multiplying two i64's will give us an i128.
   /// This forces us to change type here.
   #[inline]
   fn mul(self, rhs: Self) -> SignedBabyBearLarge {
     SignedBabyBearLarge { value: (self.value as i128) + (rhs.value as i128)}
   }
}


// Similarly to SignedBabyBearNonCanonical, SignedBabyBearLarge is essentially a wrapper for an i128.
// The key additionaly property it has is that it implements an efficient algorithm to reduce back to an i64.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct SignedBabyBearLarge {
    value: i128,
}

impl SignedBabyBearLarge {
   /// create a new `SignedBabyBearLarge` from an `i128`.
   #[inline]
   pub const fn new(n: i128) -> Self {
     Self { value: n }
   }

   /// Given |x| < 2^80 compute x' such that:
   /// |x'| < 2^50
   /// x' = x mod p
   /// x' = x mod 2^10
   /// See Thm 1 (Below function) for a proof that this function is correct.
   #[inline]
   pub fn barret_red(self) -> SignedBabyBearNonCanonical {
      const N: usize = 40; // beta = 2^N, fixing N = 40 here
      const I: i64 = (((1 as i128) << (2*N))/(P as i128)) as i64; // I = 2^80 / P => I < 2**50
      const MASK: i64 = !((1 << 10) - 1); // Lets us 0 out the bottom 10 digits of an i64.

      // input = input_low + beta*input_high

      let input_high = (self.value >> N) as i64; // input_high < input / beta < 2**{80 - N}

      // I, input_high are i64's so this can't overflow.
      // quot = I*input_high / beta < 2**{130 - 2N}.

      let quot = (((input_high as i128)*(I as i128)) >> N) as i64;

      // quot - 2^{10} < quot_2adic < quot.
      // Importantly, 2^10 divides quot_2adic.
      let quot_2adic = quot & MASK;

      // quot_2adic, P are i64's so this can't overflow.
      // sub is by construction divisible by both P and 2^10.
      let sub = (quot_2adic as i128)*(P as i128);

      SignedBabyBearNonCanonical::new((self.value - sub) as i64)
   }

   // Theorem 1:
   // Given |x| < 2^80, barret_red(x) computes an x' such that:
   //       x' = x mod p
   //       x' = x mod 2^10
   //       |x'| < 2^50.
   ///////////////////////////////////////////////////////////////////////////////////////
   // PROOF:
   // By construction P, 2**10 | sub and so we immediately see that
   // x' = x mod p
   // x' = x mod 2^10.
   //
   // It remains to prove that |x'| < 2^50.
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
}
