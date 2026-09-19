//! The Frobenius orbit and linearized polynomials over the AES field.
//!
//! Squaring is additive in characteristic 2, so every power of it is an `F_2`-linear map.
//!
//! So is any sum `c_0 x + c_1 x^2 + c_2 x^4 + ...`, which is what linearized means.
//!
//! Tabulated as a matrix, any of them costs one instruction per register.
//!
//! Evaluated term by term instead, each element costs a chain of squarings and products.

use serde::{Deserialize, Serialize};

use super::engine::ByteMatrix;
use super::{Rijndael8b, mul_bytes};

/// The number of distinct powers of squaring, which is the field's degree over `GF(2)`.
const ORBIT: usize = 8;

/// The squaring map, as a matrix.
const SQUARE: ByteMatrix = {
    let mut images = [0u8; 8];
    let mut b = 0;
    while b < 8 {
        images[b] = mul_bytes(1 << b, 1 << b);
        b += 1;
    }
    ByteMatrix::from_images(images)
};

/// The map raising to `2^k`, for every `k` below the field's degree.
///
/// Squaring has order eight as a map, so entry zero is the identity and the orbit closes.
const FROBENIUS: [ByteMatrix; ORBIT] = {
    let mut maps = [ByteMatrix::IDENTITY; ORBIT];
    let mut k = 1;
    while k < ORBIT {
        maps[k] = SQUARE.compose(maps[k - 1]);
        k += 1;
    }
    maps
};

impl Rijndael8b {
    /// The map raising every element to the power `2^k`, for `k` the argument.
    ///
    /// The orbit closes after eight steps, so the argument is taken modulo that.
    pub const fn frobenius_map(power_log: usize) -> ByteMatrix {
        FROBENIUS[power_log % ORBIT]
    }

    /// This element raised to `2^k`, for every `k` below the field's degree.
    ///
    /// These are the element's conjugates with multiplicity, so a subfield element repeats.
    pub fn frobenius_orbit(self) -> [Self; ORBIT] {
        core::array::from_fn(|k| Self::from_byte(FROBENIUS[k].apply(self.to_byte())))
    }
}

/// An `F_2`-linear map written as `sum_j c_j x^(2^j)`.
///
/// Every `F_2`-linear map on the field has exactly one such form.
///
/// Both sides are 64-dimensional over `GF(2)`, and the correspondence is injective.
///
/// Held as coefficients rather than as a matrix, because that is the form a protocol states a
/// Frobenius-twisted weight in.
///
/// The width in the name is the field's, since the form is specific to it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
#[must_use]
pub struct LinearizedPoly8b([Rijndael8b; ORBIT]);

impl LinearizedPoly8b {
    /// The map with the given coefficients, in order of increasing squaring power.
    pub const fn new(coefficients: [Rijndael8b; ORBIT]) -> Self {
        Self(coefficients)
    }

    /// The coefficients, in order of increasing squaring power.
    pub const fn coefficients(&self) -> [Rijndael8b; ORBIT] {
        self.0
    }

    /// The value at one point, summed term by term.
    ///
    /// This is the definition, kept for a single point and for checking the tabulated form.
    pub fn eval(&self, x: Rijndael8b) -> Rijndael8b {
        let orbit = x.frobenius_orbit();
        self.0
            .iter()
            .zip(orbit)
            .fold(Rijndael8b::from_byte(0), |acc, (&c, power)| acc + c * power)
    }

    /// The same map, tabulated for the byte-wise engine.
    ///
    /// # Algorithm
    ///
    /// A linear map is fixed by the images of the eight basis vectors.
    ///
    /// Each image is that vector's own Frobenius orbit summed against the coefficients.
    ///
    /// ```text
    ///     image(v)  =  sum over j of  c_j * v^(2^j)
    /// ```
    pub const fn to_matrix(&self) -> ByteMatrix {
        let mut images = [0u8; 8];
        let mut b = 0;
        while b < 8 {
            let vector = 1u8 << b;
            let mut sum = 0u8;
            let mut j = 0;
            while j < ORBIT {
                sum ^= mul_bytes(self.0[j].to_byte(), FROBENIUS[j].apply(vector));
                j += 1;
            }
            images[b] = sum;
            b += 1;
        }
        ByteMatrix::from_images(images)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::{FROBENIUS, LinearizedPoly8b, ORBIT};
    use crate::aes::mul_bytes;
    use crate::{PackedRijndael8b, Rijndael8b};

    /// The `k`-th power of squaring, by squaring `k` times through the field product.
    ///
    /// The tabulated maps are built by composing matrices instead.
    ///
    /// The two therefore share nothing beyond the field product.
    fn repeated_square(mut byte: u8, count: usize) -> u8 {
        for _ in 0..count {
            byte = mul_bytes(byte, byte);
        }
        byte
    }

    #[test]
    fn the_tabulated_orbit_matches_repeated_squaring() {
        // Exhaustive over the field: 8 maps by 256 inputs settles every case.
        for (k, map) in FROBENIUS.iter().enumerate() {
            for byte in 0..=u8::MAX {
                assert_eq!(map.apply(byte), repeated_square(byte, k), "power {k}");
            }
        }
    }

    #[test]
    fn the_orbit_closes_and_starts_at_the_identity() {
        // Invariant: squaring has order 8 as a map, since the field has 2^8 elements.
        assert_eq!(FROBENIUS[0], crate::ByteMatrix::IDENTITY);
        for byte in 0..=u8::MAX {
            assert_eq!(repeated_square(byte, ORBIT), byte, "{byte:#x}");
        }

        // The orbit of an element is its conjugate set, starting from the element itself.
        let x = Rijndael8b::from_byte(0x53);
        let orbit = x.frobenius_orbit();
        assert_eq!(orbit[0], x);
        for k in 1..ORBIT {
            assert_eq!(orbit[k], orbit[k - 1].square(), "power {k}");
        }
    }

    #[test]
    fn the_frobenius_fixes_only_the_prime_subfield() {
        // Invariant: x^2 = x holds exactly on the roots of `x^2 + x`, which are 0 and 1.
        let fixed: Vec<u8> = (0..=u8::MAX)
            .filter(|&b| Rijndael8b::frobenius_map(1).apply(b) == b)
            .collect();
        assert_eq!(fixed, alloc::vec![0, 1]);
    }

    #[test]
    fn the_square_root_is_the_seventh_power_of_squaring() {
        // Invariant: squaring has order 8, so its inverse is its seventh power.
        for byte in 0..=u8::MAX {
            let x = Rijndael8b::from_byte(byte);
            let root = x
                .try_sqrt()
                .expect("every element of a binary field is a square");
            assert_eq!(root.square(), x, "{byte:#x}");
            assert_eq!(root.to_byte(), Rijndael8b::frobenius_map(7).apply(byte));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The tabulated map must agree with summing the linearized terms at every point.
        #[test]
        fn the_tabulated_linearized_map_matches_its_definition(coefficients: [u8; ORBIT], x: u8) {
            let poly = LinearizedPoly8b::new(coefficients.map(Rijndael8b::from_byte));
            let point = Rijndael8b::from_byte(x);
            prop_assert_eq!(poly.to_matrix().apply(x), poly.eval(point).to_byte());
        }

        /// A linearized polynomial is additive, which is the whole point of the form.
        #[test]
        fn a_linearized_polynomial_is_additive(coefficients: [u8; ORBIT], x: u8, y: u8) {
            let poly = LinearizedPoly8b::new(coefficients.map(Rijndael8b::from_byte));
            let (a, b) = (Rijndael8b::from_byte(x), Rijndael8b::from_byte(y));
            prop_assert_eq!(poly.eval(a + b), poly.eval(a) + poly.eval(b));
        }

        /// A single constant coefficient must be scaling, which is tabulated elsewhere.
        ///
        /// Neither the coefficient order nor the exponent of the orbit reaches this route.
        #[test]
        fn one_constant_coefficient_is_scaling(c: u8) {
            let scalar = Rijndael8b::from_byte(c);
            let mut coefficients = [Rijndael8b::ZERO; ORBIT];
            coefficients[0] = scalar;
            prop_assert_eq!(
                LinearizedPoly8b::new(coefficients).to_matrix(),
                scalar.scaling_matrix()
            );
        }

        /// The packed sweep must agree with the scalar map, position by position.
        #[test]
        fn the_packed_sweep_matches_the_scalar_map(
            coefficients: [u8; ORBIT],
            values in prop::collection::vec(any::<u8>(), 64),
            power_log in 0usize..16,
        ) {
            let poly = LinearizedPoly8b::new(coefficients.map(Rijndael8b::from_byte));
            let block: PackedRijndael8b<64> =
                PackedValue::from_fn(|i| Rijndael8b::from_byte(values[i]));

            let expected: PackedRijndael8b<64> =
                PackedValue::from_fn(|i| poly.eval(Rijndael8b::from_byte(values[i])));
            prop_assert_eq!(block.apply(poly.to_matrix()), expected);

            let expected: PackedRijndael8b<64> = PackedValue::from_fn(|i| {
                Rijndael8b::from_byte(repeated_square(values[i], power_log % ORBIT))
            });
            prop_assert_eq!(block.frobenius(power_log), expected);

            // The packed override must match the scalar one it mirrors.
            prop_assert_eq!(block.exp_power_of_2(power_log), expected);
        }
    }

    #[test]
    fn a_single_unit_coefficient_is_one_map_of_the_orbit() {
        // Invariant: with `c_j = 1` and every other coefficient zero the sum is `x^(2^j)`.
        //
        // A reversed coefficient order would place the map at `7 - j` instead.
        for j in 0..ORBIT {
            let mut coefficients = [Rijndael8b::ZERO; ORBIT];
            coefficients[j] = Rijndael8b::ONE;
            assert_eq!(
                LinearizedPoly8b::new(coefficients).to_matrix(),
                Rijndael8b::frobenius_map(j),
                "term {j}"
            );
        }
    }

    #[test]
    fn a_known_answer_with_coefficients_outside_the_prime_subfield() {
        // A reversed coefficient order and a `c_j^(2^j)` misreading agree on `GF(2)`.
        //
        // So the weight below has four coefficients that are neither zero nor one.
        //
        // Fixture state: `0x02 x + 0x03 x^2 + 0x8d x^8 + 0x1f x^64`, evaluated elsewhere.
        //
        // - x = 0x01 gives 0x93, and x = 0x53 gives 0xa2,
        // - x = 0xff gives 0x34, and x = 0x80 gives 0x30.
        let poly = LinearizedPoly8b::new(
            [0x02, 0x03, 0x00, 0x8d, 0x00, 0x00, 0x1f, 0x00].map(Rijndael8b::from_byte),
        );
        for (x, want) in [(0x01, 0x93), (0x53, 0xa2), (0xff, 0x34), (0x80, 0x30)] {
            assert_eq!(
                poly.eval(Rijndael8b::from_byte(x)).to_byte(),
                want,
                "{x:#04x}"
            );
            assert_eq!(poly.to_matrix().apply(x), want, "{x:#04x}");
        }
    }

    #[test]
    fn repeated_squaring_matches_the_reference_at_every_exponent() {
        // Invariant: the orbit closes after eight steps, so the exponent reduces modulo eight.
        //
        // The bound sweeps one full turn and a little past it.
        //
        // The largest index a caller can pass is covered separately.
        for byte in [0u8, 1, 2, 0x1b, 0x80, 0x53, u8::MAX] {
            let x = Rijndael8b::from_byte(byte);
            for power_log in 0..=(2 * ORBIT + 3) {
                assert_eq!(
                    x.exp_power_of_2(power_log).to_byte(),
                    repeated_square(byte, power_log % ORBIT),
                    "{byte:#04x} at {power_log}"
                );
            }
            assert_eq!(
                x.exp_power_of_2(usize::MAX),
                x.exp_power_of_2(usize::MAX % ORBIT),
                "{byte:#04x} at the largest exponent"
            );
        }
    }

    /// A worked example small enough to check by hand.
    ///
    /// The polynomial `x + x^2` is the Artin-Schreier map, whose kernel is the prime subfield.
    #[test]
    fn a_hand_checkable_linearized_polynomial() {
        let one = Rijndael8b::ONE;
        let zero = Rijndael8b::ZERO;
        let poly = LinearizedPoly8b::new([one, one, zero, zero, zero, zero, zero, zero]);

        // Fixture state: the map sends `x` to `x + x^2`.
        assert_eq!(poly.eval(zero), zero);
        assert_eq!(poly.eval(one), zero);

        // x = 2: 2 + 4 = 6, since squaring the polynomial variable is still below the modulus.
        assert_eq!(poly.eval(Rijndael8b::from_byte(2)).to_byte(), 6);

        // The tabulated form must be the same map.
        assert_eq!(poly.to_matrix().apply(2), 6);
    }

    #[test]
    fn the_coefficients_round_trip_through_serde() {
        // A protocol states a twisted weight in this form, so the encoding has to carry it.
        let poly = LinearizedPoly8b::new(
            [0x02, 0x03, 0x00, 0x8d, 0x00, 0x00, 0x1f, 0x00].map(Rijndael8b::from_byte),
        );
        let encoded = serde_json::to_string(&poly).unwrap();
        assert_eq!(
            serde_json::from_str::<LinearizedPoly8b>(&encoded).unwrap(),
            poly
        );
        assert_eq!(poly.coefficients()[3].to_byte(), 0x8d);
    }
}
