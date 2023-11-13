// use core::mem::transmute;

use p3_field::{PrimeField32, Canonicalize, IntegerLike, NonCanonicalPrimeField32};

/// Computes the convolution of input and MATRIX_CIRC_MDS_8_SML.
/// Input must be an array of field elements of length 8.
/// Only works with Mersenne31 and Babybear31
pub(crate) fn apply_circulant_8_karat<Base: PrimeField32, F: Canonicalize<Base>>(
    input: [Base; 8],
    mds_const: [i64; 8],
) -> [Base; 8] {
    // The numbers we will encounter through our algorithm are (roughly) bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_8_SML)
    // <= (8 * 2**31) * 33 < 2**40 << 2**63
    // Hence we can work with i64's with garuntees of no overflow occuring.
    let input_non_canonical = input.map(F::from_canonical);
    let mds_non_canonical = mds_const.map(F::from_i64);

    // Compute the convolution.
    // Currently might? not taking full advantage of MATRIX_CIRC_MDS_8_SML_I64 being constant.
    let mut output: [F; 8] = [F::zero(); 8];
    SmallConvolution::conv8(input_non_canonical, mds_non_canonical, &mut output);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive and is bounded by 2**40.
    output.map(F::to_canonical_u_small)
}

pub(crate) fn apply_circulant_12_karat<Base: PrimeField32, F: Canonicalize<Base>>(
    input: [Base; 12],
    mds_const: [i64; 12],
) -> [Base; 12] {
    // The numbers we will encounter through our algorithm are bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_8_SML) <= (12 * 2**31) * 33 < 2**40 << 2**63
    // Hence we can work with i64's with garuntees of no overflow occuring.
    let input_non_canonical = input.map(F::from_canonical);
    let mds_non_canonical = mds_const.map(F::from_i64);

    // Compute the convolution.
    // Currently might? not taking full advantage of MATRIX_CIRC_MDS_12_SML_I64 being constant.
    let mut output: [F; 12] = [F::zero(); 12];
    SmallConvolution::conv12(input_non_canonical, mds_non_canonical, &mut output);
    // let output = conv12(input_non_canonical.map(F::to_i64), matrix_circ_mds_12_sml_i64.map(F::to_i64)).map(F::from_i64);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive and is bounded by 2**40.
    output.map(F::to_canonical_u_small)
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_16_SML.
/// Input must be an array of field elements of length 16.
/// Only works with Mersenne31 and Babybear31
pub(crate) fn apply_circulant_16_karat<Base: PrimeField32, F: Canonicalize<Base>>(
    input: [Base; 16],
    mds_const: [i64; 16],
) -> [Base; 16] {
    // The numbers we will encounter through our algorithm are (roughly) bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_8_SML)
    // <= (8 * 2**31) * 33 < 2**40 << 2**63
    // Hence we can work with i64's with garuntees of no overflow occuring.
    let input_non_canonical = input.map(F::from_canonical);
    let mds_non_canonical = mds_const.map(F::from_i64);

    // Compute the convolution.
    // Currently might? not taking full advantage of MATRIX_CIRC_MDS_8_SML_I64 being constant.
    let mut output: [F; 16] = [F::zero(); 16];
    SmallConvolution::conv16(input_non_canonical, mds_non_canonical, &mut output);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive and is bounded by 2**40.
    // output.map(|x| F::from_wrapped_u64(x as u64))
    output.map(F::to_canonical_u_small)
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_32_MERSENNE31.
/// Input must be an array of Mersenne31 field elements of length 32.
pub(crate) fn apply_circulant_32_karat<Base: PrimeField32, F: Canonicalize<Base>>(
    input: [Base; 32],
    mds_const: [i64; 32],
) -> [Base; 32] {
    // The numbers we will encounter through our algorithm are > 2**64 as
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_32_MERSENNE31) <= (32 * 2**31)**2 < 2**72.
    // Hence we need to do some intermediate reductions.
    let input_non_canonical = input.map(F::from_canonical);
    let mds_non_canonical = mds_const.map(F::from_i64);

    // Compute the convolution.
    // Currently might? not taking full advantage of MATRIX_CIRC_MDS_8_SML_I64 being constant.
    let mut output: [F; 32] = [F::zero(); 32];
    LargeConvolution::conv32(input_non_canonical, mds_non_canonical, &mut output);

    // x is an i49 => (P << 20) + x is positive.
    output.map(F::to_canonical_i_small)
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_64_MERSENNE31.
/// Input must be an array of Mersenne31 field elements of length 64.
pub(crate) fn apply_circulant_64_karat<Base: PrimeField32, F: Canonicalize<Base>>(
    input: [Base; 64],
    mds_const: [i64; 64],
) -> [Base; 64] {
    // The numbers we will encounter through our algorithm are > 2**64 as
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_64_MERSENNE31) < (64 * 2**31)**2 < 2**74 << 2**127
    // Hence we need to do some intermediate reductions.
    let input_non_canonical = input.map(F::from_canonical);
    let mds_non_canonical = mds_const.map(F::from_i64);

    // Compute the convolution.
    // Currently might? not taking full advantage of MATRIX_CIRC_MDS_8_SML_I64 being constant.
    let mut output: [F; 64] = [F::zero(); 64];
    LargeConvolution::conv64(input_non_canonical, mds_non_canonical, &mut output);

    // x is an i49 => (P << 20) + x is positive.
    output.map(F::to_canonical_i_small)
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

// Several functions here to encode some simple vector addition and similar.

/// Performs vector addition on slices saving the result in the first slice
#[inline]
fn add_mut<T: IntegerLike>(lhs: &mut [T], rhs: &[T]) {
    let n = rhs.len();
    for i in 0..n {
        lhs[i] += rhs[i]
    }
}

// Performs vector addition on slices returning a new array.
// This should be a const function but I can't work out how to do it.
// Need to say that Add and Default are constant. (I have a workaround for Default but not Add)
#[inline]
fn add_vec<T: IntegerLike, const N: usize>(lhs: &[T], rhs: &[T]) -> [T; N] {
    let mut output: [T; N] = [T::default(); N];
    let mut i = 0;
    loop {
        output[i] = lhs[i] + rhs[i];
        i += 1;
        if i == N {
            break;
        }
    }
    output
}

/// Performs vector subtraction on slices saving the result in the first slice
#[inline]
fn sub_mut<T: IntegerLike>(lhs: &mut [T], sub: &[T]) {
    let n = sub.len();
    for i in 0..n {
        lhs[i] -= sub[i]
    }
}

// Performs vector addition on slices returning a new array.
// This should be a const function but I can't work out how to do it.
// Need to say that Sub and Default are constant. (I have a workaround for Default but not Sub)
#[inline]
fn sub_vec<T: IntegerLike, const N: usize>(lhs: &[T], sub: &[T]) -> [T; N] {
    let mut output: [T; N] = [T::default(); N];
    let mut i = 0;
    loop {
        output[i] = lhs[i] - sub[i];
        i += 1;
        if i == N {
            break;
        }
    }
    output
}

/// Takes the dot product of two vectors whose products would overflow an i64.
/// Computes the result as i128's and returns that.
#[inline]
fn dot_i128<T: NonCanonicalPrimeField32>(lhs: &[T], rhs: &[T]) -> i128 {
    let n = lhs.len();
    let mut sum = lhs[0] * rhs[0];
    for i in 1..n {
        sum += lhs[i] * rhs[i];
    }
    sum
}

/// Takes the dot product of two vectors which we are sure will not overflow an i64.
#[inline]
fn dot_i64<T: NonCanonicalPrimeField32>(lhs: &[T], rhs: &[T]) -> T {
    let n = lhs.len();
    let mut sum = T::mul_small(lhs[0], rhs[0]);
    for i in 1..n {
        sum += T::mul_small(lhs[i], rhs[i]);
    }
    sum
}

/// Split an array of length N into two subarrays of length N/2.
/// Return the sum and difference of the arrays.
fn split_add_sub<T: IntegerLike, const N: usize, const HALF: usize>(
    input: [T; N],
) -> ([T; HALF], [T; HALF]) {
    // N = 2*HALF

    // Could make this whole thing mutable so we use less space?
    // Should be easy, just need to use a for loop.

    // Really want something like the following to work:
    // let [input_left, input_right] = unsafe{ core::mem::transmute::<[T;N], [[T; HALF]; 2]>(input)};

    let (input_left, input_right) = input.split_at(HALF);

    let input_p = add_vec(input_left, input_right);
    let input_m = sub_vec(input_left, input_right);

    (input_p, input_m)
}

/// This will package all our basic convolution functions but allow for us to slightly modify implementations
/// to suit our purposes.
trait Convolution {
    // For the smallest sizes we implement two different algorithms depending on whether it's possible to overflow and i64.

    /// Compute the convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 - 1
    fn conv3<T: NonCanonicalPrimeField32>(lhs: [T; 3], rhs: [T; 3], output: &mut [T]);

    /// Compute the signed convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 - 1
    fn signed_conv3<T: NonCanonicalPrimeField32>(lhs: &[T; 3], rhs: &[T; 3], output: &mut [T]);

    /// Compute the convolution of two vectors of length 4.
    /// output(x) = lhs(x)rhs(x) mod x^4 - 1
    fn conv4<T: NonCanonicalPrimeField32>(lhs: [T; 4], rhs: [T; 4], output: &mut [T]);

    /// Compute the signed convolution of two vectors of length 4.
    /// output(x) = lhs(x)rhs(x) mod x^4 + 1
    /// Eventually we will remove this and only have the mutable version.
    fn signed_conv4<T: NonCanonicalPrimeField32>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4];

    /// Compute the signed convolution of two vectors of length 4 and save in output.
    /// output(x) = lhs(x)rhs(x) mod x^4 + 1
    fn signed_conv4_mut<T: NonCanonicalPrimeField32>(lhs: &[T; 4], rhs: &[T; 4], output: &mut [T]);

    /// Compute the signed convolution of two vectors of length 6.
    /// output(x) = lhs(x)rhs(x) mod x^6 - 1
    /// This should likely be replaced by a mutable version.
    fn signed_conv6<T: NonCanonicalPrimeField32>(lhs: &[T; 6], rhs: &[T; 6], output: &mut [T]);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // We will have 2 different implementations of the above functions depending on whether our types can overflow.
    // In all otehr cases we can ignore overflow as we only deal with addition and subtractions.

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Length 6

    /// Compute the convolution of 2 vectors of length 6.
    /// output(x) = lhs(x)rhs(x) mod x^6 - 1  <=>  output = lhs * rhs
    /// Use the FFT Trick to split into a convolution of length 4 and a signed convolution of length 4.
    #[inline]
    fn conv6<T: NonCanonicalPrimeField32>(lhs: [T; 6], rhs: [T; 6], output: &mut [T]) {
        const N: usize = 6;
        const HALF: usize = N / 2;

        // Compute lhs(x) mod x^3 - 1, lhs(x) mod x^3 + 1
        let (lhs_p, lhs_m) = split_add_sub(lhs);

        // rhs will always be constant. Not sure how to tell the compiler this though.
        // Compute rhs(x) mod x^3 - 1, rhs(x) mod x^3 + 1
        let (rhs_p, rhs_m) = split_add_sub(rhs);

        let (left, right) = output.split_at_mut(HALF);

        Self::signed_conv3(&lhs_m, &rhs_m, left); // left = w_1 = lhs*rhs mod x^3 + 1
        Self::conv3(lhs_p, rhs_p, right); // right = w_0 = lhs*rhs mod x^3 - 1

        for i in 0..HALF {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Length 8

    /// Compute the convolution of 2 vectors of length 8.
    /// output(x) = lhs(x)rhs(x) mod x^8 - 1  <=>  output = lhs * rhs
    /// Use the FFT Trick to split into a convolution of length 4 and a signed convolution of length 4.
    #[inline]
    fn conv8<T: NonCanonicalPrimeField32>(lhs: [T; 8], rhs: [T; 8], output: &mut [T]) {
        const N: usize = 8;
        const HALF: usize = N / 2;

        // Compute lhs(x) mod x^4 - 1, lhs(x) mod x^4 + 1
        let (lhs_p, lhs_m) = split_add_sub(lhs);

        // rhs will always be constant. Not sure how to tell the compiler this though.
        // Compute rhs(x) mod x^4 - 1, rhs(x) mod x^4 + 1
        let (rhs_p, rhs_m) = split_add_sub(rhs);

        let (left, right) = output.split_at_mut(HALF);

        Self::signed_conv4_mut(&lhs_m, &rhs_m, left); // left = w_1 = lhs*rhs mod x^4 + 1
        Self::conv4(lhs_p, rhs_p, right); // right = w_0 = lhs*rhs mod x^4 - 1

        for i in 0..HALF {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /// Compute the signed convolution of 2 vectors of length 8.
    /// output(x) = lhs(x)rhs(x) mod x^8 + 1
    /// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
    #[inline]
    fn signed_conv8<T: NonCanonicalPrimeField32>(lhs: &[T; 8], rhs: &[T; 8]) -> [T; 8] {
        const N: usize = 8;
        const HALF: usize = N / 2;

        // The algorithm is relatively simple:
        // v(x)u(x) mod x^8 + 1 = (v_e(x^2) + xv_o(x^2))(u_e(x^2) + xu_o(x^2)) mod x^8 + 1
        //          = v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2) + x((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - v_e(x^2)u_e(x^2) - v_o(x^2)u_o(x^2))

        // Now computing v_e(x^2)u_e(x^2) mod x^8 + 1 is equivalent to computing v_e(x)u_e(x) mod x^4 + 1 and similarly for the other products.

        // Clearly there should be a cleaner way to get this decomposition but everything I've tried has been slower.
        // Also seems like we are doing quite a bit of data fiddiling. Would be nice to avoid this.
        let mut lhs_even = [lhs[0], lhs[2], lhs[4], lhs[6]]; // v_e
        let lhs_odd = [lhs[1], lhs[3], lhs[5], lhs[7]]; // v_o
        let mut rhs_even = [rhs[0], rhs[2], rhs[4], rhs[6]]; // u_e
        let rhs_odd = [rhs[1], rhs[3], rhs[5], rhs[7]]; // u_o

        let mut prod_even = Self::signed_conv4(&lhs_even, &rhs_even); // v_e(x)u_e(x) mod x^4 + 1
        let prod_odd = Self::signed_conv4(&lhs_odd, &rhs_odd); // v_o(x)u_o(x) mod x^4 + 1

        // Add the two halves together, storing the result in lhs_even/rhs_even.
        add_mut(&mut lhs_even, &lhs_odd); // v_e + v_o
        add_mut(&mut rhs_even, &rhs_odd); // u_e + u_o

        let mut prod_mix = Self::signed_conv4(&lhs_even, &rhs_even); // (v_e(x) + v_o(x))(u_e(x) + u_o(x))
        sub_mut(&mut prod_mix, &prod_even);
        sub_mut(&mut prod_mix, &prod_odd); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) - v_e(x)u_e(x) - v_o(x)u_o(x)

        add_mut(&mut prod_even[1..], &prod_odd[..(HALF - 1)]);
        prod_even[0] -= prod_odd[HALF - 1]; // v_e(x)u_e(x) + xv_o(x)u_o(x) mod x^4 + 1

        // An annoying amount of data fiddiling. It's possible to get around this by choosing the "right" initial ordering
        // But implementing the relation between prod_even and prod_odd will become complicated in that case.
        [
            prod_even[0],
            prod_mix[0],
            prod_even[1],
            prod_mix[1],
            prod_even[2],
            prod_mix[2],
            prod_even[3],
            prod_mix[3],
        ] // Intertwining the result. Again this is some annoying data fiddiling. Must be a way to avoid some of this.
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Length 12

    /// Compute the convolution of 2 vectors of length 12.
    /// output(x) = lhs(x)rhs(x) mod x^12 - 1  <=>  output = lhs * rhs
    /// Use the FFT Trick to split into a convolution of length 6 and a signed convolution of length 6.
    #[inline]
    fn conv12<T: NonCanonicalPrimeField32>(lhs: [T; 12], rhs: [T; 12], output: &mut [T]) {
        const N: usize = 12;
        const HALF: usize = N / 2;

        // Compute lhs(x) mod x^6 - 1, lhs(x) mod x^6 + 1
        let (lhs_p, lhs_m) = split_add_sub(lhs);

        // rhs will always be constant. Not sure how to tell the compiler this though.
        // Compute rhs(x) mod x^6 - 1, rhs(x) mod x^6 + 1
        let (rhs_p, rhs_m) = split_add_sub(rhs);

        let (left, right) = output.split_at_mut(HALF);
        Self::signed_conv6(&lhs_m, &rhs_m, left); // left = w_1 = lhs*rhs mod x^12 + 1
        Self::conv6(lhs_p, rhs_p, right); // right = w_0 = lhs*rhs mod x^12 - 1

        for i in 0..HALF {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Length 16

    /// Compute the convolution of 2 vectors of length 16.
    /// output(x) = lhs(x)rhs(x) mod x^16 - 1  <=>  output = lhs * rhs
    /// Use the FFT Trick to split into a convolution of length 8 and a signed convolution of length 8.
    #[inline]
    fn conv16<T: NonCanonicalPrimeField32>(lhs: [T; 16], rhs: [T; 16], output: &mut [T]) {
        const N: usize = 16;
        const HALF: usize = N / 2;

        // Compute lhs(x) mod x^16 - 1, lhs(x) mod x^16 + 1
        let (lhs_p, lhs_m) = split_add_sub(lhs);

        // rhs will always be constant. Not sure how to tell the compiler this though.
        // Compute rhs(x) mod x^16 - 1, rhs(x) mod x^16 + 1
        let (rhs_p, rhs_m) = split_add_sub(rhs);

        let (left, right) = output.split_at_mut(HALF);
        left.clone_from_slice(&Self::signed_conv8(&lhs_m, &rhs_m)); // left = w_1 = lhs*rhs mod x^16 + 1
        Self::conv8(lhs_p, rhs_p, right); // right = w_0 = lhs*rhs mod x^16 - 1
        for i in 0..HALF {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /// Compute the signed convolution of 2 vectors of length 16.
    /// output(x) = lhs(x)rhs(x) mod x^16 + 1
    /// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
    #[inline]
    fn signed_conv16<T: NonCanonicalPrimeField32>(lhs: &[T; 16], rhs: &[T; 16]) -> [T; 16] {
        const N: usize = 16;
        const HALF: usize = N / 2;

        // The algorithm is relatively simple:
        // v(x)u(x) mod x^16 + 1 = (v_e(x^2) + xv_o(x^2))(u_e(x^2) + xu_o(x^2)) mod x^16 + 1
        //          = v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2) + x((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - v_e(x^2)u_e(x^2) - v_o(x^2)u_o(x^2))

        // Now computing v_e(x^2)u_e(x^2) mod x^16 + 1 is equivalent to computing v_e(x)u_e(x) mod x^8 + 1 and similarly for the other products.

        // Clearly there should be a cleaner way to get this decomposition but everything I've tried has been slower.
        // Also seems like we are doing quite a bit of data fiddiling. Would be nice to avoid this.
        let mut lhs_even = [
            lhs[0], lhs[2], lhs[4], lhs[6], lhs[8], lhs[10], lhs[12], lhs[14], // v_e
        ];
        let lhs_odd = [
            lhs[1], lhs[3], lhs[5], lhs[7], lhs[9], lhs[11], lhs[13], lhs[15], // v_o
        ];
        let mut rhs_even = [
            rhs[0], rhs[2], rhs[4], rhs[6], rhs[8], rhs[10], rhs[12], rhs[14], // u_e
        ];
        let rhs_odd = [
            rhs[1], rhs[3], rhs[5], rhs[7], rhs[9], rhs[11], rhs[13], rhs[15], // u_o
        ];

        let mut prod_even = Self::signed_conv8(&lhs_even, &rhs_even); // v_e(x)u_e(x) mod x^8 + 1
        let prod_odd = Self::signed_conv8(&lhs_odd, &rhs_odd); // v_o(x)u_o(x) mod x^8 + 1

        // Add the two halves together, storing the result in lhs_even/rhs_even.
        add_mut(&mut lhs_even, &lhs_odd); // v_e + v_o
        add_mut(&mut rhs_even, &rhs_odd); // u_e + u_o

        let mut prod_mix = Self::signed_conv8(&lhs_even, &rhs_even); // (v_e(x) + v_o(x))(u_e(x) + u_o(x))
        sub_mut(&mut prod_mix, &prod_even);
        sub_mut(&mut prod_mix, &prod_odd); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) - v_e(x)u_e(x) - v_o(x)u_o(x)

        add_mut(&mut prod_even[1..], &prod_odd[..(HALF - 1)]);
        prod_even[0] -= prod_odd[HALF - 1]; // v_e(x)u_e(x) + xv_o(x)u_o(x) mod x^8 + 1

        [
            prod_even[0],
            prod_mix[0],
            prod_even[1],
            prod_mix[1],
            prod_even[2],
            prod_mix[2],
            prod_even[3],
            prod_mix[3],
            prod_even[4],
            prod_mix[4],
            prod_even[5],
            prod_mix[5],
            prod_even[6],
            prod_mix[6],
            prod_even[7],
            prod_mix[7],
        ] // Intertwining the result. Again this is some annoying data fiddiling. Must be a way to avoid some of this.
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Length 32

    /// Compute the convolution of 2 vectors of length 32.
    /// output(x) = lhs(x)rhs(x) mod x^32 - 1  <=>  output = lhs * rhs
    /// Use the FFT Trick to split into a convolution of length 16 and a signed convolution of length 16.
    #[inline]
    fn conv32<T: NonCanonicalPrimeField32>(lhs: [T; 32], rhs: [T; 32], output: &mut [T]) {
        const N: usize = 32;
        const HALF: usize = N / 2;

        // Compute lhs(x) mod x^16 - 1, lhs(x) mod x^16 + 1
        let (lhs_p, lhs_m) = split_add_sub(lhs);

        // rhs will always be constant. Not sure how to tell the compiler this though.
        // Compute rhs(x) mod x^16 - 1, rhs(x) mod x^16 + 1
        let (rhs_p, rhs_m) = split_add_sub(rhs);

        let (left, right) = output.split_at_mut(HALF);
        left.clone_from_slice(&Self::signed_conv16(&lhs_m, &rhs_m)); // left = w_1 = lhs*rhs mod x^16 + 1
        Self::conv16(lhs_p, rhs_p, right); // right = w_0 = lhs*rhs mod x^16 - 1
        for i in 0..HALF {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /// Compute the signed convolution of 2 vectors of length 16.
    /// output(x) = lhs(x)rhs(x) mod x^16 + 1
    /// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
    #[inline]
    fn signed_conv32<T: NonCanonicalPrimeField32>(lhs: &[T; 32], rhs: &[T; 32]) -> [T; 32] {
        const N: usize = 32;
        const HALF: usize = N / 2;

        // The algorithm is simple:
        // v(x)u(x) mod x^32 + 1 = (v_l(x) + x^4v_h(x))(u_l(x) + x^4u_h(x)) mod x^32 + 1
        //          = v_l(x)u_l(x) - v_h(x)u_h(x) + x^4((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))

        // Now computing v_e(x^2)u_e(x^2) mod x^32 + 1 is equivalent to computing v_e(x)u_e(x) mod x^16 + 1 and similarly for the other products.

        // Clearly there should be a cleaner way to get this decomposition but everything I've tried has been slower.
        // Also seems like we are doing quite a bit of data fiddiling. Would be nice to avoid this.
        let mut lhs_even = [
            lhs[0], lhs[2], lhs[4], lhs[6], lhs[8], lhs[10], lhs[12], lhs[14], lhs[16],
            lhs[18], // v_e
            lhs[20], lhs[22], lhs[24], lhs[26], lhs[28], lhs[30],
        ];
        let lhs_odd = [
            lhs[1], lhs[3], lhs[5], lhs[7], lhs[9], lhs[11], lhs[13], lhs[15], lhs[17],
            lhs[19], // v_o
            lhs[21], lhs[23], lhs[25], lhs[27], lhs[29], lhs[31],
        ];
        let mut rhs_even = [
            rhs[0], rhs[2], rhs[4], rhs[6], rhs[8], rhs[10], rhs[12], rhs[14], rhs[16],
            rhs[18], // u_e
            rhs[20], rhs[22], rhs[24], rhs[26], rhs[28], rhs[30],
        ];
        let rhs_odd = [
            rhs[1], rhs[3], rhs[5], rhs[7], rhs[9], rhs[11], rhs[13], rhs[15], rhs[17],
            rhs[19], // u_o
            rhs[21], rhs[23], rhs[25], rhs[27], rhs[29], rhs[31],
        ];

        let mut prod_even = Self::signed_conv16(&lhs_even, &rhs_even); // v_e(x)u_e(x) mod x^16 + 1
        let prod_odd = Self::signed_conv16(&lhs_odd, &rhs_odd); // v_o(x)u_o(x) mod x^16 + 1

        // Add the two halves together, storing the result in lhs_even/rhs_even.
        add_mut(&mut lhs_even, &lhs_odd); // v_e + v_o
        add_mut(&mut rhs_even, &rhs_odd); // u_e + u_o

        let mut prod_mix = Self::signed_conv16(&lhs_even, &rhs_even); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) mod x^16 + 1
        sub_mut(&mut prod_mix, &prod_even);
        sub_mut(&mut prod_mix, &prod_odd); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) - v_e(x)u_e(x) - v_o(x)u_o(x)

        add_mut(&mut prod_even[1..], &prod_odd[..(HALF - 1)]);
        prod_even[0] -= prod_odd[HALF - 1]; // v_e(x)u_e(x) + xv_o(x)u_o(x) mod x^16 + 1

        [
            prod_even[0],
            prod_mix[0],
            prod_even[1],
            prod_mix[1],
            prod_even[2],
            prod_mix[2],
            prod_even[3],
            prod_mix[3],
            prod_even[4],
            prod_mix[4],
            prod_even[5],
            prod_mix[5],
            prod_even[6],
            prod_mix[6],
            prod_even[7],
            prod_mix[7],
            prod_even[8],
            prod_mix[8],
            prod_even[9],
            prod_mix[9],
            prod_even[10],
            prod_mix[10],
            prod_even[11],
            prod_mix[11],
            prod_even[12],
            prod_mix[12],
            prod_even[13],
            prod_mix[13],
            prod_even[14],
            prod_mix[14],
            prod_even[15],
            prod_mix[15],
        ] // Intertwining the result. Again this is some annoying data fiddiling. Must be a way to avoid some of this.
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Length 64

    /// Compute the convolution of 2 vectors of length 64.
    /// output(x) = lhs(x)rhs(x) mod x^64 - 1  <=>  output = lhs * rhs
    /// Use the FFT Trick to split into a convolution of length 32 and a signed convolution of length 32.
    #[inline]
    fn conv64<T: NonCanonicalPrimeField32>(lhs: [T; 64], rhs: [T; 64], output: &mut [T]) {
        const N: usize = 64;
        const HALF: usize = N / 2;

        // Compute lhs(x) mod x^32 - 1, lhs(x) mod x^32 + 1
        let (lhs_p, lhs_m) = split_add_sub(lhs);

        // rhs will always be constant. Not sure how to tell the compiler this though.
        // Compute rhs(x) mod x^32 - 1, rhs(x) mod x^32 + 1
        let (rhs_p, rhs_m) = split_add_sub(rhs);

        let (left, right) = output.split_at_mut(HALF);
        left.clone_from_slice(&Self::signed_conv32(&lhs_m, &rhs_m)); // left = w_1 = lhs*rhs mod x^32 + 1
        Self::conv32(lhs_p, rhs_p, right); // right = w_0 = lhs*rhs mod x^32 - 1
        for i in 0..HALF {
            left[i] += right[i]; // w_0 + w_1
            left[i] >>= 1; // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }
}

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
    // We will need to implement this if we want to handle convolutions of size 24/48 but for now we ignore this.
    fn conv3<T: NonCanonicalPrimeField32>(_: [T; 3], _: [T; 3], _: &mut [T]) {
        todo!()
    }
    fn signed_conv3<T: NonCanonicalPrimeField32>(_: &[T; 3], _: &[T; 3], _: &mut [T]) {
        todo!()
    }

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

        output[0] = T::from_small_i128(lhs_m[0] * rhs_m[0] - lhs_m[1] * rhs_m[1]);
        output[1] = T::from_small_i128(lhs_m[0] * rhs_m[1] + lhs_m[1] * rhs_m[0]); // output[0, 1] = w_1 = v_1(x)u_1(x) mod x^2 + 1
        output[2] = T::from_small_i128(lhs_p[0] * rhs_p[0] + lhs_p[1] * rhs_p[1]);
        output[3] = T::from_small_i128(lhs_p[0] * rhs_p[1] + lhs_p[1] * rhs_p[0]); // output[2, 3] = w_0 = v_0(x)u_0(x) mod x^2 - 1

        output[0] += output[2];
        output[1] += output[3]; // output[0, 1] = w_1 + w_0

        output[0] >>= 1;
        output[1] >>= 1; // output[0, 1] = (w_1 + w_0)/2)

        output[2] -= output[0];
        output[3] -= output[1]; // output[2, 3] = w_0 - (w_1 + w_0)/2) = (w_0 - w_1)/2
    }

    fn signed_conv4_mut<T: NonCanonicalPrimeField32>(lhs: &[T; 4], rhs: &[T; 4], output: &mut [T]) {
        let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

        output[0] = T::from_small_i128((lhs[0] * rhs_rev[3]) - dot_i128(&lhs[1..], &rhs_rev[..3])); // v_0u_0 - (v_1u_3 + v_2u_2 + v_3u_1)
        output[1] = T::from_small_i128(
            dot_i128(&lhs[..2], &rhs_rev[2..]) - dot_i128(&lhs[2..], &rhs_rev[..2]),
        ); // v_0u_1 + v_1u_0 - (v_2u_3 + v_2u_3)
        output[2] = T::from_small_i128(dot_i128(&lhs[..3], &rhs_rev[1..]) - (lhs[3] * rhs_rev[0])); // v_0u_2 + v_1u_1 + v_2u_0 - v_3u_3
        output[3] = T::from_small_i128(dot_i128(lhs, &rhs_rev)); // v_0u_3 + v_1u_2 + v_2u_1 + v_3u_0

        // This might not be the best way to compute this.
        // Another approach is to define
        // [rhs[0], -rhs[3], -rhs[2], -rhs[1]]
        // [rhs[1], rhs[0], -rhs[3], -rhs[2]]
        // [rhs[2], rhs[1], rhs[0], -rhs[3]]
        // [rhs[3], rhs[2], rhs[1], rhs[0]]
        // And then take dot products.
        // Might also be other methods in particular we might be able to pick MDS matrices to make this simpler.
    }

    fn signed_conv4<T: NonCanonicalPrimeField32>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4] {
        let mut output = [T::zero(); 4];

        Self::signed_conv4_mut(lhs, rhs, &mut output);

        output
    }

    // We will need to implement this if we want to handle convolutions of size 24/48 but for now we ignore this.
    fn conv6<T: NonCanonicalPrimeField32>(_: [T; 6], _: [T; 6], _: &mut [T]) {
        todo!()
    }
    fn signed_conv6<T: NonCanonicalPrimeField32>(_: &[T; 6], _: &[T; 6], _: &mut [T]) {
        todo!()
    }
}

// If we can add the assumption that Sum(lhs) < 2**20 then
// Sum(lhs)*Sum(rhs) < N * 2**{51} and so, for small N we can work with i64's and ignore overflow.
struct SmallConvolution;

impl Convolution for SmallConvolution {
    /// Compute the convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 - 1
    #[inline]
    fn conv3<T: NonCanonicalPrimeField32>(lhs: [T; 3], rhs: [T; 3], output: &mut [T]) {
        // This is small enough we just explicitely write down the answer.
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

    /// Compute the signed convolution of two vectors of length 3.
    /// output(x) = lhs(x)rhs(x) mod x^3 + 1
    #[inline]
    fn signed_conv3<T: NonCanonicalPrimeField32>(lhs: &[T; 3], rhs: &[T; 3], output: &mut [T]) {
        // This is small enough we just explicitely write down the answer.
        output[0] = T::mul_small(lhs[0], rhs[0])
            - T::mul_small(lhs[1], rhs[2])
            - T::mul_small(lhs[2], rhs[1]);
        output[1] = T::mul_small(lhs[0], rhs[1]) + T::mul_small(lhs[1], rhs[0])
            - T::mul_small(lhs[2], rhs[2]);
        output[2] = T::mul_small(lhs[0], rhs[2])
            + T::mul_small(lhs[1], rhs[1])
            + T::mul_small(lhs[2], rhs[0]);
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

        output[0] = T::mul_small(lhs_m[0], rhs_m[0]) - T::mul_small(lhs_m[1], rhs_m[1]);
        output[1] = T::mul_small(lhs_m[0], rhs_m[1]) + T::mul_small(lhs_m[1], rhs_m[0]); // output[0, 1] = w_1 = v_1(x)u_1(x) mod x^2 + 1
        output[2] = T::mul_small(lhs_p[0], rhs_p[0]) + T::mul_small(lhs_p[1], rhs_p[1]);
        output[3] = T::mul_small(lhs_p[0], rhs_p[1]) + T::mul_small(lhs_p[1], rhs_p[0]);

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
    fn signed_conv4_mut<T: NonCanonicalPrimeField32>(lhs: &[T; 4], rhs: &[T; 4], output: &mut [T]) {
        let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

        output[0] = T::mul_small(lhs[0], rhs[0]) - dot_i64(&lhs[1..], &rhs_rev[..3]); // v_0u_0 - (v_1u_3 + v_2u_2 + v_3u_1)
        output[1] = dot_i64(&lhs[..2], &rhs_rev[2..]) - dot_i64(&lhs[2..], &rhs_rev[..2]); // v_0u_1 + v_1u_0 - (v_2u_3 + v_2u_3)
        output[2] = dot_i64(&lhs[..3], &rhs_rev[1..]) - T::mul_small(lhs[3], rhs[3]); // v_0u_2 + v_1u_1 + v_2u_0 - v_3u_3
        output[3] = dot_i64(lhs, &rhs_rev); // v_0u_3 + v_1u_2 + v_2u_1 + v_3u_0

        // This might not be the best way to compute this.
        // Another approach is to define
        // [rhs[0], -rhs[3], -rhs[2], -rhs[1]]
        // [rhs[1], rhs[0], -rhs[3], -rhs[2]]
        // [rhs[2], rhs[1], rhs[0], -rhs[3]]
        // [rhs[3], rhs[2], rhs[1], rhs[0]]
        // And then take dot products.
        // Might also be other methods in particular we might be able to pick MDS matrices to make this simpler.
    }

    /// Compute the signed convolution of two vectors of length 4.
    /// output(x) = lhs(x)rhs(x) mod x^4 + 1
    #[inline]
    fn signed_conv4<T: NonCanonicalPrimeField32>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4] {
        let mut output = [T::zero(); 4];

        Self::signed_conv4_mut(lhs, rhs, &mut output);

        output
    }

    /// Compute the signed convolution of two vectors of length 6.
    /// output(x) = lhs(x)rhs(x) mod x^6 + 1
    #[inline]
    fn signed_conv6<T: NonCanonicalPrimeField32>(lhs: &[T; 6], rhs: &[T; 6], output: &mut [T]) {
        let rhs_rev = [rhs[5], rhs[4], rhs[3], rhs[2], rhs[1], rhs[0]];

        output[0] = T::mul_small(lhs[0], rhs[0]) - dot_i64(&lhs[1..], &rhs_rev[..5]);
        output[1] = dot_i64(&lhs[..2], &rhs_rev[4..]) - dot_i64(&lhs[2..], &rhs_rev[..4]);
        output[2] = dot_i64(&lhs[..3], &rhs_rev[3..]) - dot_i64(&lhs[3..], &rhs_rev[..3]);
        output[3] = dot_i64(&lhs[..4], &rhs_rev[2..]) - dot_i64(&lhs[4..], &rhs_rev[..2]);
        output[4] = dot_i64(&lhs[..5], &rhs_rev[1..]) - T::mul_small(lhs[5], rhs[5]);
        output[5] = dot_i64(lhs, &rhs_rev);
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