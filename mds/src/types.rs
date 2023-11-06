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
   #[inline]
   pub fn barret_red(self) -> SignedBabyBearNonCanonical {
      const N: usize = 40; // beta = 2^N
      const I: i64 = (((1 as i128) << (2*N))/(P as i128)) as i64; // 2^80 / P, I < 2**50
      const MASK: i64 = !((1 << 10) - 1);

      // input = input_low + beta*input_high

      let input_high = self.value >> N as i64; // input_high < 2**N

      // I*input_high < 2**90
      let quot = (((input_high as i128)*(I as i128)) >> N) as i64; // quot < 2**N

      let quot_2adic = quot & MASK;

      let sub = (quot_2adic as i128)*(P as i128);

      SignedBabyBearNonCanonical::new((self.value - sub) as i64)
   }

   // Proof that the above function computes what we claim:
   // By construction 2**10 | quot_2adic and so P*2**10 | sub = P*quot_2adic.
   // Hence we immediately see that
   // x' = x mod p
   // x' = x mod 2^10.

   // It remains to prove that |x'| < 2^50.

   // Note that for all x, (x << N) <= x / 2**N <= 1 + (x << N)

   // First assume that input > 0.
   // sub = Q*P = ((input >> N) * (2**N * 2**N // P) >> N) * P <= input
   // So input - sub > 0.

   // On the other hand, if input < 0 then, due to the negative,
   // (input >> N) * (2**N * 2**N // P)   >= (input >> N) * 2**(2N) / P 
   //                                     >= (1 + input / 2**N) * 2**(2N) / P
   //                                     >= (2**N + input) * 2**N / P

   // Thus:
   // Q = ((input >> N) * (2**N * 2**N // P)) >> N >= ((2**N + input) * 2**N / P) >> N
   //                                              >= 1 + ((2**N + input) / P)

   // And so sub = Q*P >= P + 2**N + input.
   // Hence input - sub > 0.

   // Note that, regardless of if  or x < 0, we always have
   // 2**N * (x << N) < x

   // Hence as sub = Q*P = ((input >> N) * (2**N * 2**N // P) >> N) * P


   // 2**N * input_high <= input.
   // Now if x > 0 then 
   // Hence: sub = Q*P <= ((input / 2**N) * (2**N * 2**N / P) / 2**N) * P < input so input - sub > 0.

   // If input < 0, then input_high < input / (2**N) so:

   // Definitionally we have input = input_high*beta + input_low => input_high*beta = input - input_low

   // First observe that:
   // I = beta^2 // B 
   //    => I + 1 > beta^2 / B
   //    => B*I + B > beta^2

   // To bound Q we need to first deal with signs. If x is positive then (I*x_high)//beta = (I*x_high) >> N and:
   // Q = ((I*x_high)//beta) & !(2^10 - 1)
   //    => Q + (2^10 - 1) >= (I*x_high)//beta
   //    => Q + 2^10 > (I*x_high)/beta
   //    => Q*beta + 2^10*beta > I*x_high

   // On the other hand, if x is negative, (_ >> N) rounds the opposite way to (_ // beta) (Away from not towards 0).
   // Thus for x < 0, (x >> N) <= x/beta <= x//beta.
   // But we can simply shift by 1 to get (x >> N) + 1 >= x/beta.
   // Additionally, even for negative x we have (x & !(2^10 - 1)) >= x - (2^10 - 1).

   // Thus our bound is idential!
   // Q = ((I*x_high)//beta) & !(2^10 - 1)
   //    => Q + (2^10 - 1) >= (I*x_high)//beta
   //    => Q + 2^10 > (I*x_high)/beta
   //    => Q*beta + 2^10*beta > I*x_high

   // 
   // Q*beta*B + 2^10*beta*B > x_high*I*B
   //                        > x_high(beta^2 - B) 
   //                        = beta*beta*x_high - B x_high
   //                        > beta*(x - x_low) - beta x_high
   // Now we can divide by beta to get

   // Q*B + 2^10*B > x - x_low - x_high

   // Which rearranges to

   // x' = x - Q*B < 2^10*B + x_low + x_high
   // The terms on the left hand side are bounded by 2^41, beta and x//beta respectively which for  beta = 2^31 or 2^40 gives us exactly the bounds we want.

}
