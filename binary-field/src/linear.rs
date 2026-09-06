//! Byte tables for linear maps in the tower representation.

/// Multiply by the highest tower generator, using only XOR and shifts.
const fn mul_alpha(a: u128, bits: usize) -> u128 {
    if bits == 1 {
        return a;
    }
    let half = bits / 2;
    let lo = a & (u128::MAX >> (128 - half));
    let hi = a >> half;
    hi | ((lo ^ mul_alpha(hi, half)) << half)
}

/// Squaring follows `(a + bX)^2 = a^2 + b^2 + alpha*b^2*X`.
const fn square(a: u128, bits: usize) -> u128 {
    if bits == 1 {
        return a;
    }
    let half = bits / 2;
    let lo = square(a & (u128::MAX >> (128 - half)), half);
    let hi = square(a >> half, half);
    (lo ^ hi) | (mul_alpha(hi, half) << half)
}

const fn square_table<const BYTES: usize>() -> [[u128; 256]; BYTES] {
    let mut table = [[0; 256]; BYTES];
    let mut i = 0;
    while i < BYTES {
        let mut b = 0;
        while b < 8 {
            let image = square(1 << (i * 8 + b), BYTES * 8);
            let mut v = 0;
            while v < 1 << b {
                table[i][v | (1 << b)] = table[i][v] ^ image;
                v += 1;
            }
            b += 1;
        }
        i += 1;
    }
    table
}

macro_rules! square_map {
    ($name:ident, $bytes:literal, $repr:ty) => {
        #[inline]
        pub(crate) fn $name(value: $repr) -> $repr {
            static TABLE: [[$repr; 256]; $bytes] = {
                let wide = square_table::<$bytes>();
                let mut table = [[0; 256]; $bytes];
                let mut i = 0;
                while i < $bytes {
                    let mut j = 0;
                    while j < 256 {
                        table[i][j] = wide[i][j] as $repr;
                        j += 1;
                    }
                    i += 1;
                }
                table
            };
            let mut result = 0;
            for (i, row) in TABLE.iter().enumerate() {
                result ^= row[((value >> (8 * i)) & 255) as usize];
            }
            result
        }
    };
}

square_map!(square_16, 2, u16);
square_map!(square_64, 8, u64);
square_map!(square_128, 16, u128);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{BinaryField16, BinaryField64, BinaryField128, TowerLevel};

    proptest! {
        #[test]
        fn square_maps_match_reference(a: u128) {
            macro_rules! check {
                ($field:ty, $repr:ty, $map:ident) => {{
                    let x = <$field>::from_repr(a as $repr);
                    prop_assert_eq!(super::$map(a as $repr), x.reference_mul(x).to_repr());
                }};
            }
            check!(BinaryField16, u16, square_16);
            check!(BinaryField64, u64, square_64);
            check!(BinaryField128, u128, square_128);
        }
    }
}
