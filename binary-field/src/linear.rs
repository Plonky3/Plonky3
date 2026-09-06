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

/// Invert multiplication by the highest generator: `(a,b) -> (b + alpha*a,a)`.
const fn div_alpha(a: u128, bits: usize) -> u128 {
    if bits == 1 {
        return a;
    }
    let half = bits / 2;
    let lo = a & (u128::MAX >> (128 - half));
    let hi = a >> half;
    (hi ^ mul_alpha(lo, half)) | (lo << half)
}

const fn sqrt(a: u128, bits: usize) -> u128 {
    if bits == 1 {
        return a;
    }
    let half = bits / 2;
    let hi = sqrt(div_alpha(a >> half, half), half);
    let lo = sqrt(a & (u128::MAX >> (128 - half)), half) ^ hi;
    lo | (hi << half)
}

const fn unary_table<const BYTES: usize>(inverse: bool) -> [[u128; 256]; BYTES] {
    let mut table = [[0; 256]; BYTES];
    let mut i = 0;
    while i < BYTES {
        let mut b = 0;
        while b < 8 {
            let bit = 1 << (i * 8 + b);
            let image = if inverse {
                sqrt(bit, BYTES * 8)
            } else {
                square(bit, BYTES * 8)
            };
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

macro_rules! unary_map {
    ($name:ident, $bytes:literal, $repr:ty, $inverse:literal) => {
        #[inline]
        pub(crate) fn $name(value: $repr) -> $repr {
            static TABLE: [[$repr; 256]; $bytes] = {
                let wide = unary_table::<$bytes>($inverse);
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

unary_map!(square_16, 2, u16, false);
unary_map!(square_64, 8, u64, false);
unary_map!(square_128, 16, u128, false);
unary_map!(sqrt_16, 2, u16, true);
unary_map!(sqrt_32, 4, u32, true);
unary_map!(sqrt_64, 8, u64, true);
unary_map!(sqrt_128, 16, u128, true);

/// Columns of squaring raised to successive powers of two, composed over GF(2).
const fn frobenius_columns<const BITS: usize, const LOG: usize>() -> [[u128; BITS]; LOG] {
    let mut maps = [[0; BITS]; LOG];
    let mut i = 0;
    while i < BITS {
        maps[0][i] = square(1 << i, BITS);
        i += 1;
    }
    let mut k = 1;
    while k < LOG {
        i = 0;
        while i < BITS {
            let mut bits = maps[k - 1][i];
            let mut result = 0;
            while bits != 0 {
                result ^= maps[k - 1][bits.trailing_zeros() as usize];
                bits &= bits - 1;
            }
            maps[k][i] = result;
            i += 1;
        }
        k += 1;
    }
    maps
}

macro_rules! frobenius_map {
    ($name:ident, $bits:literal, $log:literal, $repr:ty) => {
        #[inline]
        pub(crate) fn $name(mut value: $repr, power: usize) -> $repr {
            // Nibbles keep the family of maps small; square and sqrt have separate byte tables.
            static TABLES: [[[$repr; 16]; $bits / 4]; $log] = {
                let columns = frobenius_columns::<$bits, $log>();
                let mut tables = [[[0; 16]; $bits / 4]; $log];
                let mut k = 0;
                while k < $log {
                    let mut i = 0;
                    while i < $bits / 4 {
                        let mut b = 0;
                        while b < 4 {
                            let mut v = 0;
                            while v < 1 << b {
                                tables[k][i][v | (1 << b)] =
                                    tables[k][i][v] ^ columns[k][4 * i + b] as $repr;
                                v += 1;
                            }
                            b += 1;
                        }
                        i += 1;
                    }
                    k += 1;
                }
                tables
            };
            for (k, table) in TABLES.iter().enumerate() {
                if (power >> k) & 1 != 0 {
                    let mut result = 0;
                    for (i, row) in table.iter().enumerate() {
                        result ^= row[((value >> (4 * i)) & 15) as usize];
                    }
                    value = result;
                }
            }
            value
        }
    };
}

frobenius_map!(frobenius_16, 16, 3, u16);
frobenius_map!(frobenius_32, 32, 4, u32);
frobenius_map!(frobenius_64, 64, 5, u64);
frobenius_map!(frobenius_128, 128, 6, u128);

#[cfg(test)]
mod tests {
    use p3_field::{Field, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use crate::{BinaryField16, BinaryField32, BinaryField64, BinaryField128, TowerLevel};

    #[test]
    fn frobenius_maps_and_sqrt_match_repeated_reference_squares() {
        macro_rules! check {
            ($field:ty, $repr:ty, $bits:literal) => {{
                for value in [0, 1, 2, <$repr>::MAX, 0x9876 as $repr] {
                    let x = <$field>::from_repr(value);
                    let mut expected = x;
                    for power in 0..=$bits {
                        assert_eq!(x.exp_power_of_2(power), expected);
                        if power == $bits - 1 {
                            assert_eq!(x.try_sqrt(), Some(expected));
                        }
                        expected = expected.reference_mul(expected);
                    }
                    assert_eq!(
                        x.exp_power_of_2(usize::MAX),
                        x.exp_power_of_2(usize::MAX % $bits)
                    );
                }
            }};
        }
        check!(BinaryField16, u16, 16);
        check!(BinaryField32, u32, 32);
        check!(BinaryField64, u64, 64);
        check!(BinaryField128, u128, 128);
    }

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
