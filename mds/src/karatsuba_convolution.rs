use core::ops::{Add, AddAssign, ShrAssign, Sub, SubAssign};

pub trait RngElt:
    Add<Output = Self>
    + AddAssign
    + Copy
    + Default
    + ShrAssign<u32>
    + Sub<Output = Self>
    + SubAssign
    + Eq
{
}

impl RngElt for i64 {}
impl RngElt for i128 {}

/// In practice, one of the parameters to the convolution will be
/// constant (the MDS matrix); after inspecting Godbolt output, it
/// seems that the compiler does indeed generate single constants as
/// inputs to the multiplication, rather than doing all that
/// arithmetic on the constant values every time. Note however that
/// these compile-time generated constants will be about N times
/// bigger than they need to be in principle, which could be a
/// potential avenue for some minor improvements.
///
/// NB: Note that the convolution code does some `ShrAssign`s after
/// calling `mul`, so if `mul` does an intermediate/partial reduction,
/// then the definition of `ShrAssign` will have to be replaced with
/// the corresponding field "divide-by-2" function, rather than the
/// primitive "bit-shift-to-the-right" which relies on the knowledge
/// that the input is even if no reduction has taken place.
pub trait Convolve<T: RngElt, U: RngElt, V: RngElt> {
    fn mul(x: T, y: U) -> V;

    /// Compute the convolution of two vectors of length N.
    /// output(x) = lhs(x)rhs(x) mod x^N - 1
    /// We split this into a convolution and signed convolution of size N/2
    #[inline(always)]
    fn conv_n<
        const N: usize,
        const HALF_N: usize,
        C: Fn([T; HALF_N], [U; HALF_N], &mut [V]),
        SC: Fn([T; HALF_N], [U; HALF_N], &mut [V]),
    >(
        lhs: [T; N],
        rhs: [U; N],
        output: &mut [V],
        inner_conv: C,
        inner_signed_conv: SC,
    ) {
        debug_assert_eq!(2 * HALF_N, N);
        // NB: The compiler is smart enough not to initialise these arrays.
        let mut lhs_pos = [T::default(); HALF_N]; // lhs_pos = lhs(x) mod x^{N/2} - 1
        let mut lhs_neg = [T::default(); HALF_N]; // lhs_neg = lhs(x) mod x^{N/2} + 1
        let mut rhs_pos = [U::default(); HALF_N]; // rhs_pos = lhs(x) mod x^{N/2} - 1
        let mut rhs_neg = [U::default(); HALF_N]; // rhs_neg = lhs(x) mod x^{N/2} + 1

        // Could mutably change the inputs? Can't use transmute as compiler doesn't know that 2 * HALF_N = N.

        for i in 0..HALF_N {
            let s = lhs[i];
            let t = lhs[i + HALF_N];

            lhs_pos[i] = s + t;
            lhs_neg[i] = s - t;

            let s = rhs[i];
            let t = rhs[i + HALF_N];

            rhs_pos[i] = s + t;
            rhs_neg[i] = s - t;
        }

        let (left, right) = output.split_at_mut(HALF_N);

        inner_signed_conv(lhs_neg, rhs_neg, left); // left = lhs(x)rhs(x) mod x^{N/2} + 1
        inner_conv(lhs_pos, rhs_pos, right); // right = lhs(x)rhs(x) mod x^{N/2} - 1

        for i in 0..HALF_N {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /// Compute the signed convolution of two vectors of length N.
    /// output(x) = lhs(x)rhs(x) mod x^N + 1
    #[inline(always)]
    fn signed_conv_n<
        const N: usize,
        const HALF_N: usize,
        SC: Fn([T; HALF_N], [U; HALF_N], &mut [V]),
    >(
        lhs: [T; N],
        rhs: [U; N],
        output: &mut [V],
        inner_signed_conv: SC,
    ) {
        debug_assert_eq!(2 * HALF_N, N);
        // NB: The compiler is smart enough not to initialise these arrays.
        let mut lhs_even = [T::default(); HALF_N];
        let mut lhs_odd = [T::default(); HALF_N];
        let mut lhs_sum = [T::default(); HALF_N];
        let mut rhs_even = [U::default(); HALF_N];
        let mut rhs_odd = [U::default(); HALF_N];
        let mut rhs_sum = [U::default(); HALF_N];

        for i in 0..HALF_N {
            let s = lhs[2 * i];
            let t = lhs[2 * i + 1];
            lhs_even[i] = s;
            lhs_odd[i] = t;
            lhs_sum[i] = s + t;

            let s = rhs[2 * i];
            let t = rhs[2 * i + 1];
            rhs_even[i] = s;
            rhs_odd[i] = t;
            rhs_sum[i] = s + t;
        }

        // Could make some scratch space to draw from here?

        let mut even_s_conv = [V::default(); HALF_N];
        let (left, right) = output.split_at_mut(HALF_N);

        inner_signed_conv(lhs_even, rhs_even, &mut even_s_conv);
        inner_signed_conv(lhs_odd, rhs_odd, left);
        inner_signed_conv(lhs_sum, rhs_sum, right);

        // First we get the correct values in right and even_s_conv respectively.
        right[0] -= even_s_conv[0] + left[0];
        even_s_conv[0] -= left[HALF_N - 1];

        for i in 1..HALF_N {
            right[i] -= even_s_conv[i] + left[i];
            even_s_conv[i] += left[i - 1];
        }

        // We need to interleave even_s_conv and right in the output.
        for i in 0..HALF_N {
            output[2 * i] = even_s_conv[i];
            output[2 * i + 1] = output[i + HALF_N];
        }
    }

    #[inline(always)]
    fn conv3(lhs: [T; 3], rhs: [U; 3], output: &mut [V]) {
        output[0] =
            Self::mul(lhs[0], rhs[0]) + Self::mul(lhs[1], rhs[2]) + Self::mul(lhs[2], rhs[1]);
        output[1] =
            Self::mul(lhs[0], rhs[1]) + Self::mul(lhs[1], rhs[0]) + Self::mul(lhs[2], rhs[2]);
        output[2] =
            Self::mul(lhs[0], rhs[2]) + Self::mul(lhs[1], rhs[1]) + Self::mul(lhs[2], rhs[0]);
    }

    /// Compute the signed convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 + 1
    #[inline(always)]
    fn signed_conv3(lhs: [T; 3], rhs: [U; 3], output: &mut [V]) {
        output[0] =
            Self::mul(lhs[0], rhs[0]) - Self::mul(lhs[1], rhs[2]) - Self::mul(lhs[2], rhs[1]);
        output[1] =
            Self::mul(lhs[0], rhs[1]) + Self::mul(lhs[1], rhs[0]) - Self::mul(lhs[2], rhs[2]);
        output[2] =
            Self::mul(lhs[0], rhs[2]) + Self::mul(lhs[1], rhs[1]) + Self::mul(lhs[2], rhs[0]);
    }

    #[inline(always)]
    fn conv4(lhs: [T; 4], rhs: [U; 4], output: &mut [V]) {
        let u_p = [lhs[0] + lhs[2], lhs[1] + lhs[3]]; // v_0(x)
        let u_m = [lhs[0] - lhs[2], lhs[1] - lhs[3]]; // v_1(x)
        let v_p = [rhs[0] + rhs[2], rhs[1] + rhs[3]]; // u_0(x)
        let v_m = [rhs[0] - rhs[2], rhs[1] - rhs[3]]; // u_1(x)

        output[0] = Self::mul(u_m[0], v_m[0]) - Self::mul(u_m[1], v_m[1]);
        output[1] = Self::mul(u_m[0], v_m[1]) + Self::mul(u_m[1], v_m[0]);
        output[2] = Self::mul(u_p[0], v_p[0]) + Self::mul(u_p[1], v_p[1]);
        output[3] = Self::mul(u_p[0], v_p[1]) + Self::mul(u_p[1], v_p[0]);

        output[0] += output[2];
        output[1] += output[3];

        output[0] >>= 1;
        output[1] >>= 1;

        output[2] -= output[0];
        output[3] -= output[1];
    }

    #[inline(always)]
    fn signed_conv4(lhs: [T; 4], rhs: [U; 4], output: &mut [V]) {
        output[0] = Self::mul(lhs[0], rhs[0])
            - Self::mul(lhs[1], rhs[3])
            - Self::mul(lhs[2], rhs[2])
            - Self::mul(lhs[3], rhs[1]);
        output[1] = Self::mul(lhs[0], rhs[1]) + Self::mul(lhs[1], rhs[0])
            - Self::mul(lhs[2], rhs[3])
            - Self::mul(lhs[3], rhs[2]);
        output[2] =
            Self::mul(lhs[0], rhs[2]) + Self::mul(lhs[1], rhs[1]) + Self::mul(lhs[2], rhs[0])
                - Self::mul(lhs[3], rhs[3]);
        output[3] = Self::mul(lhs[0], rhs[3])
            + Self::mul(lhs[1], rhs[2])
            + Self::mul(lhs[2], rhs[1])
            + Self::mul(lhs[3], rhs[0]);
    }

    #[inline(always)]
    fn conv6(lhs: [T; 6], rhs: [U; 6], output: &mut [V]) {
        Self::conv_n::<6, 3, _, _>(lhs, rhs, output, Self::conv3, Self::signed_conv3)
    }

    #[inline(always)]
    fn signed_conv6(lhs: [T; 6], rhs: [U; 6], output: &mut [V]) {
        output[0] = Self::mul(lhs[0], rhs[0])
            - Self::mul(lhs[1], rhs[5])
            - Self::mul(lhs[2], rhs[4])
            - Self::mul(lhs[3], rhs[3])
            - Self::mul(lhs[4], rhs[2])
            - Self::mul(lhs[5], rhs[1]);

        output[1] = Self::mul(lhs[0], rhs[1]) + Self::mul(lhs[1], rhs[0])
            - Self::mul(lhs[2], rhs[5])
            - Self::mul(lhs[3], rhs[4])
            - Self::mul(lhs[4], rhs[3])
            - Self::mul(lhs[5], rhs[2]);

        output[2] =
            Self::mul(lhs[0], rhs[2]) + Self::mul(lhs[1], rhs[1]) + Self::mul(lhs[2], rhs[0])
                - Self::mul(lhs[3], rhs[5])
                - Self::mul(lhs[4], rhs[4])
                - Self::mul(lhs[5], rhs[3]);

        output[3] = Self::mul(lhs[0], rhs[3])
            + Self::mul(lhs[1], rhs[2])
            + Self::mul(lhs[2], rhs[1])
            + Self::mul(lhs[3], rhs[0])
            - Self::mul(lhs[4], rhs[5])
            - Self::mul(lhs[5], rhs[4]);

        output[4] = Self::mul(lhs[0], rhs[4])
            + Self::mul(lhs[1], rhs[3])
            + Self::mul(lhs[2], rhs[2])
            + Self::mul(lhs[3], rhs[1])
            + Self::mul(lhs[4], rhs[0])
            - Self::mul(lhs[5], rhs[5]);

        output[5] = Self::mul(lhs[0], rhs[5])
            + Self::mul(lhs[1], rhs[4])
            + Self::mul(lhs[2], rhs[3])
            + Self::mul(lhs[3], rhs[2])
            + Self::mul(lhs[4], rhs[1])
            + Self::mul(lhs[5], rhs[0]);
    }

    #[inline(always)]
    fn conv8(lhs: [T; 8], rhs: [U; 8], output: &mut [V]) {
        Self::conv_n::<8, 4, _, _>(lhs, rhs, output, Self::conv4, Self::signed_conv4)
    }

    #[inline(always)]
    fn signed_conv8(lhs: [T; 8], rhs: [U; 8], output: &mut [V]) {
        Self::signed_conv_n::<8, 4, _>(lhs, rhs, output, Self::signed_conv4)
    }

    #[inline(always)]
    fn conv12(lhs: [T; 12], rhs: [U; 12], output: &mut [V]) {
        Self::conv_n::<12, 6, _, _>(lhs, rhs, output, Self::conv6, Self::signed_conv6)
    }

    #[inline(always)]
    fn conv16(lhs: [T; 16], rhs: [U; 16], output: &mut [V]) {
        Self::conv_n::<16, 8, _, _>(lhs, rhs, output, Self::conv8, Self::signed_conv8)
    }

    #[inline(always)]
    fn signed_conv16(lhs: [T; 16], rhs: [U; 16], output: &mut [V]) {
        Self::signed_conv_n::<16, 8, _>(lhs, rhs, output, Self::signed_conv8)
    }

    #[inline(always)]
    fn conv32(lhs: [T; 32], rhs: [U; 32], output: &mut [V]) {
        Self::conv_n::<32, 16, _, _>(lhs, rhs, output, Self::conv16, Self::signed_conv16)
    }

    #[inline(always)]
    fn signed_conv32(lhs: [T; 32], rhs: [U; 32], output: &mut [V]) {
        Self::signed_conv_n::<32, 16, _>(lhs, rhs, output, Self::signed_conv16)
    }

    #[inline(always)]
    fn conv64(lhs: [T; 64], rhs: [U; 64], output: &mut [V]) {
        Self::conv_n::<64, 32, _, _>(lhs, rhs, output, Self::conv32, Self::signed_conv32)
    }
}

pub struct SmallConvolvePrimeField32;
impl Convolve<i64, i64, i64> for SmallConvolvePrimeField32 {
    #[inline(always)]
    fn mul(x: i64, y: i64) -> i64 {
        x * y
    }
}

pub struct LargeConvolvePrimeField32;
impl Convolve<i64, i64, i128> for LargeConvolvePrimeField32 {
    #[inline(always)]
    fn mul(x: i64, y: i64) -> i128 {
        x as i128 * y as i128
    }
}

pub struct SmallConvolvePrimeField64;
impl Convolve<i128, i64, i128> for SmallConvolvePrimeField64 {
    #[inline(always)]
    fn mul(x: i128, y: i64) -> i128 {
        x * y as i128
    }
}

// FIXME: Move to mds/goldilocks.rs
/*
struct LargeConvolveGoldilocks;
impl Convolve<i128, i128, i128> for LargeConvolveGoldilocks {
    #[inline(always)]
    fn mul(x: i128, y: i128) -> i128 {
        let x = Goldilocks::from_wrapped_i128(x);
        let y = Goldilocks::from_wrapped_i128(y);
        let xy = x * y;
        xy.0 as i128
    }
}
*/

/*

// We want to compute a convolution lhs * rhs of 31 bit field elements.
// For any sensible algorithm we use, the size of elements computed in intermediate steps is bounded by
// Sum(lhs) * Sum(rhs) ~ N^2 2**62 for a convolution of size N and generic field elements.
// Thus this will overflow an i64 and so we need to we need to include reductions and/or pass to i128's breifly.
struct LargeConvolution;

// We proivide a quick proof that our strategy for LargeConvolution will produce the correct result.
// We focus on the N = 64 case as any case involving smaller N will be strictly easier.

// Entries start as field elements, so i32's.
// In each reduction step we add or subtract some entries so by the time we have gotten to size 4 convs the entries are i36's.
// The size 4 convs involve products and adding together 4 things so our entries become i74's.
// Now we apply a reduction step which reduces i80's -> i50's.
// Moving back up, n each karatsuba step we have to compute a difference of 3 elements.
// We have 3 karatsuba steps making the entries size i55's.
// CRT steps don't increase the size due to the division by 2 so our entries will remain i55's which will not overflow.

impl Convolution for LargeConvolution {
    /// Compute the convolution of two vectors of length 4.
    /// output(x) = lhs(x)rhs(x) mod x^4 - 1 in Fp[X]
    /// Coefficients will be non canonical representatives.
    fn conv4<T: NonCanonicalPrimeField32>(lhs: [T; 4], rhs: [T; 4], output: &mut [T]) {
        // Even at this small size, doing the FFT decomposition seems to produce shorter compiled code using godbolt.
        // In particular testing the code produced for conv8.

        let lhs_p = [lhs[0] + lhs[2], lhs[1] + lhs[3]]; // v_0(x)
        let lhs_m = [lhs[0] - lhs[2], lhs[1] - lhs[3]]; // v_1(x)

        let rhs_p = [rhs[0] + rhs[2], rhs[1] + rhs[3]]; // u_0(x)
        let rhs_m = [rhs[0] - rhs[2], rhs[1] - rhs[3]]; // u_1(x)

        // This is safe for all convolutions of size < 512 as we get
        // |lhs_m|, |rhs_m|, |lhs_p|, |rhs_p| < 256*2**31 < 2**39.
        // Thus T::from_small_i128 is only called on inputs with |input| < 2**80
        unsafe {
            output[0] = T::from_small_i128(lhs_m[0] * rhs_m[0] - lhs_m[1] * rhs_m[1]);
            output[1] = T::from_small_i128(lhs_m[0] * rhs_m[1] + lhs_m[1] * rhs_m[0]); // output[0, 1] = w_1 = v_1(x)u_1(x) mod x^2 + 1
            output[2] = T::from_small_i128(lhs_p[0] * rhs_p[0] + lhs_p[1] * rhs_p[1]);
            output[3] = T::from_small_i128(lhs_p[0] * rhs_p[1] + lhs_p[1] * rhs_p[0]);
            // output[2, 3] = w_0 = v_0(x)u_0(x) mod x^2 - 1
        }

        output[0] += output[2];
        output[1] += output[3]; // output[0, 1] = w_1 + w_0

        output[0] >>= 1;
        output[1] >>= 1; // output[0, 1] = (w_1 + w_0)/2)

        output[2] -= output[0];
        output[3] -= output[1]; // output[2, 3] = w_0 - (w_1 + w_0)/2) = (w_0 - w_1)/2
    }

    fn signed_conv4<T: NonCanonicalPrimeField32>(lhs: [T; 4], rhs: [T; 4], output: &mut [T]) {
        let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

        // This is safe for all convolutions of size < 512 as we get
        // |lhs|, |rhs| < 128*2**31 < 2**38.
        // Thus T::dot_large will not overflow an i128.
        // Thus T::from_small_i128 is only called on inputs with |input| < 2**80
        unsafe {
            output[0] =
                T::from_small_i128((lhs[0] * rhs_rev[3]) - T::dot_large(&lhs[1..], &rhs_rev[..3])); // v_0u_0 - (v_1u_3 + v_2u_2 + v_3u_1)
            output[1] = T::from_small_i128(
                T::dot_large(&lhs[..2], &rhs_rev[2..]) - T::dot_large(&lhs[2..], &rhs_rev[..2]),
            ); // v_0u_1 + v_1u_0 - (v_2u_3 + v_2u_3)
            output[2] =
                T::from_small_i128(T::dot_large(&lhs[..3], &rhs_rev[1..]) - (lhs[3] * rhs_rev[0])); // v_0u_2 + v_1u_1 + v_2u_0 - v_3u_3
            output[3] = T::from_small_i128(T::dot_large(&lhs, &rhs_rev)); // v_0u_3 + v_1u_2 + v_2u_1 + v_3u_0
        }

        // This might not be the best way to compute this.
        // Another approach is to define
        // [rhs[0], -rhs[3], -rhs[2], -rhs[1]]
        // [rhs[1], rhs[0], -rhs[3], -rhs[2]]
        // [rhs[2], rhs[1], rhs[0], -rhs[3]]
        // [rhs[3], rhs[2], rhs[1], rhs[0]]
        // And then take dot products.
        // Might also be other methods in particular we might be able to pick MDS matrices to make this simpler.
    }

}

// If we can add the assumption that Sum(lhs) < 2**20 then
// Sum(lhs)*Sum(rhs) < N * 2**{51} and so, for small N we can work with i64's and ignore overflow.
// This assumption is not checked inside the code so it is up to the programmer to ensure it is satisfied.
struct SmallConvolution;

impl Convolution for SmallConvolution {
    /// Compute the convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 - 1
    #[inline]
    fn conv3<T: NonCanonicalPrimeField32>(lhs: [T; 3], rhs: [T; 3], output: &mut [T]) {
        // This is small enough we just explicitely write down the answer.

        // Provided the initial inputs are as small as claimed this is safe.
        unsafe {
            output[0] = T::mul_small(lhs[0], rhs[0])
                + T::mul_small(lhs[1], rhs[2])
                + T::mul_small(lhs[2], rhs[1]);
            output[1] = T::mul_small(lhs[0], rhs[1])
                + T::mul_small(lhs[1], rhs[0])
                + T::mul_small(lhs[2], rhs[2]);
            output[2] = T::mul_small(lhs[0], rhs[2])
                + T::mul_small(lhs[1], rhs[1])
                + T::mul_small(lhs[2], rhs[0]);
        }
    }

    /// Compute the signed convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 + 1
    #[inline]
    fn signed_conv3<T: NonCanonicalPrimeField32>(lhs: [T; 3], rhs: [T; 3], output: &mut [T]) {
        // This is small enough we just explicitely write down the answer.

        // Provided the initial inputs are as small as claimed this is safe.
        unsafe {
            output[0] = T::mul_small(lhs[0], rhs[0])
                - T::mul_small(lhs[1], rhs[2])
                - T::mul_small(lhs[2], rhs[1]);
            output[1] = T::mul_small(lhs[0], rhs[1]) + T::mul_small(lhs[1], rhs[0])
                - T::mul_small(lhs[2], rhs[2]);
            output[2] = T::mul_small(lhs[0], rhs[2])
                + T::mul_small(lhs[1], rhs[1])
                + T::mul_small(lhs[2], rhs[0]);
        }
    }

    /// Compute the convolution of two vectors of length 4. We assume we can ignore overflow so
    /// output(x) = lhs(x)rhs(x) mod x^4 - 1 in Z[X]
    #[inline]
    fn conv4<T: NonCanonicalPrimeField32>(lhs: [T; 4], rhs: [T; 4], output: &mut [T]) {
        // Even at this small size, doing the FFT decomposition seems to produce shorter compiled code using godbolt.
        // In particular testing the code produced for conv8.
        let lhs_p = [lhs[0] + lhs[2], lhs[1] + lhs[3]]; // v_0(x)
        let lhs_m = [lhs[0] - lhs[2], lhs[1] - lhs[3]]; // v_1(x)

        let rhs_p = [rhs[0] + rhs[2], rhs[1] + rhs[3]]; // u_0(x)
        let rhs_m = [rhs[0] - rhs[2], rhs[1] - rhs[3]]; // u_1(x)

        // Provided the initial inputs are as small as claimed this is safe.
        unsafe {
            output[0] = T::mul_small(lhs_m[0], rhs_m[0]) - T::mul_small(lhs_m[1], rhs_m[1]);
            output[1] = T::mul_small(lhs_m[0], rhs_m[1]) + T::mul_small(lhs_m[1], rhs_m[0]); // output[0, 1] = w_1 = v_1(x)u_1(x) mod x^2 + 1
            output[2] = T::mul_small(lhs_p[0], rhs_p[0]) + T::mul_small(lhs_p[1], rhs_p[1]);
            output[3] = T::mul_small(lhs_p[0], rhs_p[1]) + T::mul_small(lhs_p[1], rhs_p[0]);
        }

        output[0] += output[2];
        output[1] += output[3]; // output[0, 1] = w_1 + w_0

        output[0] >>= 1;
        output[1] >>= 1; // output[0, 1] = (w_1 + w_0)/2)

        output[2] -= output[0];
        output[3] -= output[1]; // output[2, 3] = w_0 - (w_1 + w_0)/2) = (w_0 - w_1)/2
    }

    /// Compute the signed convolution of two vectors of length 4.
    /// output(x) = lhs(x)rhs(x) mod x^4 + 1
    #[inline]
    fn signed_conv4<T: NonCanonicalPrimeField32>(lhs: [T; 4], rhs: [T; 4], output: &mut [T]) {
        let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

        // Provided the initial inputs are as small as claimed this is safe.
        unsafe {
            output[0] = T::mul_small(lhs[0], rhs[0]) - T::dot_small(&lhs[1..], &rhs_rev[..3]); // v_0u_0 - (v_1u_3 + v_2u_2 + v_3u_1)
            output[1] =
                T::dot_small(&lhs[..2], &rhs_rev[2..]) - T::dot_small(&lhs[2..], &rhs_rev[..2]); // v_0u_1 + v_1u_0 - (v_2u_3 + v_2u_3)
            output[2] = T::dot_small(&lhs[..3], &rhs_rev[1..]) - T::mul_small(lhs[3], rhs[3]); // v_0u_2 + v_1u_1 + v_2u_0 - v_3u_3
            output[3] = T::dot_small(&lhs, &rhs_rev); // v_0u_3 + v_1u_2 + v_2u_1 + v_3u_0
        }

        // This might not be the best way to compute this.
        // Another approach is to define
        // [rhs[0], -rhs[3], -rhs[2], -rhs[1]]
        // [rhs[1], rhs[0], -rhs[3], -rhs[2]]
        // [rhs[2], rhs[1], rhs[0], -rhs[3]]
        // [rhs[3], rhs[2], rhs[1], rhs[0]]
        // And then take dot products.
        // Might also be other methods in particular we might be able to pick MDS matrices to make this simpler.
    }

    /// Compute the signed convolution of two vectors of length 6.
    /// output(x) = lhs(x)rhs(x) mod x^6 + 1
    #[inline]
    fn signed_conv6<T: NonCanonicalPrimeField32>(lhs: [T; 6], rhs: [T; 6], output: &mut [T]) {
        let rhs_rev = [rhs[5], rhs[4], rhs[3], rhs[2], rhs[1], rhs[0]];

        // Provided the initial inputs are as small as claimed this is safe.
        unsafe {
            output[0] = T::mul_small(lhs[0], rhs[0]) - T::dot_small(&lhs[1..], &rhs_rev[..5]);
            output[1] =
                T::dot_small(&lhs[..2], &rhs_rev[4..]) - T::dot_small(&lhs[2..], &rhs_rev[..4]);
            output[2] =
                T::dot_small(&lhs[..3], &rhs_rev[3..]) - T::dot_small(&lhs[3..], &rhs_rev[..3]);
            output[3] =
                T::dot_small(&lhs[..4], &rhs_rev[2..]) - T::dot_small(&lhs[4..], &rhs_rev[..2]);
            output[4] = T::dot_small(&lhs[..5], &rhs_rev[1..]) - T::mul_small(lhs[5], rhs[5]);
            output[5] = T::dot_small(&lhs, &rhs_rev);
        }
    }
}

// Given a vector v \in F^N, let v(x) \in F[X] denote the polynomial v_0 + v_1 x + ... + v_{N - 1} x^{N - 1}.
// Then w is equal to the convolution v * u if and only if w(x) = v(x)u(x) mod x^N - 1.
// Additionally, define the signed convolution by w(x) = v(x)u(x) mod x^N + 1.

// Using the chinese remainder theorem we can compute w(x) from:
//                      w_0 = v(x)u(x) mod x^{N/2} - 1
//                      w_1 = v(x)u(x) mod x^{N/2} + 1
// Via:
//                      w(x) = 1/2 (w_0(x) + w_1(x)) + x^{N/2}/2 (w_0(x) - w_1(x))

// To compute w_0 and w_1 we first compute
//                  v_0(x) = v(x) mod x^{N/2} - 1
//                  v_1(x) = v(x) mod x^{N/2} + 1
//                  u_0(x) = v(x) mod x^{N/2} - 1
//                  u_1(x) = v(x) mod x^{N/2} + 1

// Now w_0 is the convolution of v_0 and u_0 so this can be applied recursively.
// For w_1 we compute the signed convolution v_1(x)u_1(x) mod x^{N/2} + 1 using Karatsuba.

// There are 2 possible approaches to this karatsuba which mirror the DIT vs DIF approaches to FFT's.

// Option 1: left/right decomposition.
// Write v = (v_l, v_r) so that v(x) = (v_l(x) + x^{N/2}v_r(x)).
// Then v(x)u(x) mod x^N + 1 = (v_l(x)u_l(x) - v_r(x)u_r(x)) + x^{N/2}((v_l(x) + v_r(x))(u_l(x) + u_r(x)) - (v_l(x)u_l(x) + v_r(x)u_r(x))) mod X^N + 1

// As v_l(x), v_r(x), u_l(x), u_r(x) all have degree < N/2 no product will have degree > N - 1.
// The only place we need to deal with the mod operation is after the multipication by x^{N/2} and this is easy to do.
// Thus this reduces the problem to 3 polynomial multiplications of size N/2 and we can use the standard karatsuba for this.

// Option 2: even/odd decomposition.
// Define the even v_e and odd v_o parts so that v(x) = (v_e(x^2) + xv_o(x^2)).
// Then v(x)u(x) = (v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2)) + x ((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - (v_e(x^2)u_e(x^2) + v_o(x^2)u_o(x^2))
// This reduces the problem to 3 signed convolutions of size N/2 and we can do this recursively.

// Option 2: seems to involve less total operations and so should be faster hence it is the once we have implmented for now.
// The main issue is that we are currently doing quite a bit of data manipulation. (Needing to split vectors into odd and even parts and recombining afterwards)
// Would be good to try and find a way to cut down on this.

// Once we get down to small sizes we use the O(n^2) approach.

*/
