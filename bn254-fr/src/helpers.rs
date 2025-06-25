use p3_field::Field;

/// Compute `base^{2^num_sq} * mul`
fn sq_and_mul<F: Field>(base: F, num_sq: usize, mul: F) -> F {
    base.exp_power_of_2(num_sq) * mul
}

/// Compute the exponential `x -> x^21888242871839275222246405745257275088548364400416034343698204186575808495615` using a custom addition chain.
///
/// This map computes the inverse in the BN254 field.
pub(crate) fn exp_bn_inv<F: Field>(val: F) -> F {
    // Note the binary expansion: 21888242871839275222246405745257275088548364400416034343698204186575808495615
    //  = 1100000110010001001110011100101110000100110001101000000010100110111000010100000100010110110110100000
    //       0110000001010110000101110100101000001100111110100001001000011110011011100101110000100100010100001
    //       111100001111101011001001111101111111111111111111111111111.
    // This uses 250 Squares + 54 Multiplications => 306 Operations total.
    // It is likely that this could be improved through further effort.

    // 10101

    // The basic idea we follow here is to create some simple binary building blocks to save on multiplications.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10 * p1;
    let p101 = p11 * p10;
    let p111 = p101 * p10;
    let p1111 = sq_and_mul(p111, 1, p1);
    let p11111 = sq_and_mul(p1111, 1, p1);
    let p1111100 = p11111.exp_power_of_2(2);
    let p1111101 = p1111100 * p1;
    let p1111111 = p1111101 * p10;
    let p10000001 = p1111111 * p10;
    let mut output = sq_and_mul(p10000001, 1, p10000001);

    // output now agrees with the first 9 digits of the binary expansion. We just have to do the
    // remaining 245...

    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 4, p1); // 0001
    output = sq_and_mul(output, 5, p111); // 00111
    output = sq_and_mul(output, 5, p111); // 00111
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 4, p11); // 0011
    output = sq_and_mul(output, 5, p11); // 00011
    output = sq_and_mul(output, 2, p1); // 01
    output = sq_and_mul(output, 10, p101); // 0000000101
    output = sq_and_mul(output, 4, p11); // 0011
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 7, p101); // 0000101
    output = sq_and_mul(output, 6, p1); // 000001
    output = sq_and_mul(output, 6, p101); // 000101
    output = sq_and_mul(output, 3, p101); // 101
    output = sq_and_mul(output, 3, p101); // 101
    output = sq_and_mul(output, 3, p101); // 101
    output = sq_and_mul(output, 8, p11); // 00000011
    output = sq_and_mul(output, 9, p101); // 000000101
    output = sq_and_mul(output, 3, p11); // 011
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 2, p1); // 01
    output = sq_and_mul(output, 5, p101); // 00101
    output = sq_and_mul(output, 7, p11); // 0000011
    output = sq_and_mul(output, 9, p1111101); // 001111101
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 8, p1111); // 00001111
    output = sq_and_mul(output, 4, p11); // 0011
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 4, p111); // 0111
    output = sq_and_mul(output, 5, p1); // 00001
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 6, p101); // 000101
    output = sq_and_mul(output, 9, p11111); // 000011111
    output = sq_and_mul(output, 11, p1111101); // 00001111101
    output = sq_and_mul(output, 3, p11); // 011
    output = sq_and_mul(output, 3, p1); // 001
    output = sq_and_mul(output, 7, p11111); // 0011111
    output = sq_and_mul(output, 8, p1111111); // 01111111
    output = sq_and_mul(output, 7, p1111111); // 1111111
    output = sq_and_mul(output, 7, p1111111); // 1111111
    output = sq_and_mul(output, 7, p1111111); // 1111111
    output
}
