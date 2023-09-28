use crate::AbstractField;

pub fn exp_u64_by_squaring<AF: AbstractField>(val: AF, power: u64) -> AF {
    let mut current = val;
    let mut product = AF::ONE;

    for j in 0..bits_u64(power) {
        if (power >> j & 1) != 0 {
            product *= current.clone();
        }
        current = current.square();
    }
    product
}

const fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}

pub fn exp_1717986917<AF: AbstractField>(val: AF) -> AF {
    // Note that 5 * 1717986917 = 4*(2^31 - 2) + 1 = 1 mod p - 1.
    // Thus as a^{p - 1} = 1 for all a \in F_p, (a^{1717986917})^5 = a.
    // Note the binary expansion: 1717986917 = 1100110011001100110011001100101_2
    // This uses 30 Squares + 7 Multiplications => 37 Operations total.
    // Suspect it's possible to improve this with enough effort. For example 1717986918 takes only 4 Multiplications.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10.clone() * p1;
    let p101 = p10 * p11.clone();
    let p110000 = p11.exp_power_of_2(4);
    let p110011 = p110000 * p11.clone();
    let p11001100000000 = p110011.exp_power_of_2(8);
    let p11001100110011 = p11001100000000.clone() * p110011;
    let p1100110000000000000000 = p11001100000000.exp_power_of_2(8);
    let p1100110011001100110011 = p1100110000000000000000 * p11001100110011;
    let p11001100110011001100110000 = p1100110011001100110011.exp_power_of_2(4);
    let p11001100110011001100110011 = p11001100110011001100110000 * p11;
    let p1100110011001100110011001100000 = p11001100110011001100110011.exp_power_of_2(5);
    p1100110011001100110011001100000 * p101
}

pub fn exp_1725656503<AF: AbstractField>(val: AF) -> AF {
    // Note that 7 * 1725656503 = 6*(2^31 - 2^27) + 1 = 1 mod (p - 1).
    // Thus as a^{p - 1} = 1 for all a \in F_p, (a^{1725656503})^7 = a.
    // Note the binary expansion: 1725656503 = 1100110110110110110110110110111_2
    // This uses 29 Squares + 8 Multiplications => 37 Operations total.
    // Suspect it's possible to improve this with enough effort.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10 * p1.clone();
    let p110 = p11.square();
    let p111 = p110.clone() * p1;
    let p11000 = p110.exp_power_of_2(2);
    let p11011 = p11000.clone() * p11;
    let p11000000 = p11000.exp_power_of_2(3);
    let p11011011 = p11000000.clone() * p11011;
    let p110011011 = p11011011.clone() * p11000000;
    let p110011011000000000 = p110011011.exp_power_of_2(9);
    let p110011011011011011 = p110011011000000000 * p11011011.clone();
    let p110011011011011011000000000 = p110011011011011011.exp_power_of_2(9);
    let p110011011011011011011011011 = p110011011011011011000000000 * p11011011;
    let p1100110110110110110110110110000 = p110011011011011011011011011.exp_power_of_2(4);
    p1100110110110110110110110110000 * p111
}

pub fn exp_10540996611094048183<AF: AbstractField>(val: AF) -> AF {
    // Note that 7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1).
    // Thus as a^{p - 1} = 1 for all a \in F_p, (a^{10540996611094048183})^7 = a.
    // Also: 10540996611094048183 = 1001001001001001001001001001000110110110110110110110110110110111_2.
    // This uses 63 Squares + 9 Multiplications => 72 Operations total.
    // Suspect it's possible to improve this a little with enough effort.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10.clone() * p1.clone();
    let p100 = p10.square();
    let p111 = p100.clone() * p11;
    let p1000 = p100.square();
    let p1001 = p1000 * p1;
    let p1001000000 = p1001.exp_power_of_2(6);
    let p1001001001 = p1001000000 * p1001.clone();
    let p1001001001000000 = p1001001001.exp_power_of_2(6);
    let p1001001001001001 = p1001001001000000.clone() * p1001;
    let p1001001001000000000000000000 = p1001001001000000.exp_power_of_2(12);
    let p1001001001001001001001001001 = p1001001001000000000000000000 * p1001001001001001;
    let p10010010010010010010010010010 = p1001001001001001001001001001.square();
    let p100100100100100100100100100100000000000000000000000000000000 =
        p10010010010010010010010010010.exp_power_of_2(31);
    let p11011011011011011011011011011 =
        p10010010010010010010010010010 * p1001001001001001001001001001;
    let p100100100100100100100100100100011011011011011011011011011011 =
        p100100100100100100100100100100000000000000000000000000000000
            * p11011011011011011011011011011;

    let p1001001001001001001001001001000110110110110110110110110110110000 =
        p100100100100100100100100100100011011011011011011011011011011.exp_power_of_2(4);

    p1001001001001001001001001001000110110110110110110110110110110000 * p111
}
