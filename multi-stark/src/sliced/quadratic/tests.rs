use alloc::vec::Vec;

use p3_binary_field::{BinaryField2, BinaryField128, Ghash128, TowerLevel};
use p3_field::PrimeCharacteristicRing;
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::sliced::SlicedGf4;

type F = BinaryField128;
type Quadratic = SlicedQuadratic<F>;
type Bit = SlicedBit<F>;
type Gf4 = SlicedGf4<F, BinaryField2>;
type Prime = <F as PrimeCharacteristicRing>::PrimeSubfield;

/// Inputs of the test quadratic.
const INPUTS: usize = 4;

/// Monomials `x_i x_j` with `i <= j` of the test quadratic.
const PAIRS: usize = INPUTS * (INPUTS + 1) / 2;

/// A quadratic in four inputs, its coefficients bits, written with the ring operations alone.
#[derive(Clone, Copy, Debug)]
struct TestQuadratic {
    pairs: [bool; PAIRS],
    singles: [bool; INPUTS],
    constant: bool,
}

impl TestQuadratic {
    fn eval<E: PrimeCharacteristicRing + Copy>(&self, x: [E; INPUTS]) -> E {
        let mut value = E::from_bool(self.constant);
        let mut pair = 0;
        for i in 0..INPUTS {
            if self.singles[i] {
                value += x[i];
            }
            for j in i..INPUTS {
                if self.pairs[pair] {
                    value += x[i] * x[j];
                }
                pair += 1;
            }
        }
        value
    }
}

fn arb_quadratic() -> impl Strategy<Value = TestQuadratic> {
    (
        prop::array::uniform10(any::<bool>()),
        prop::array::uniform4(any::<bool>()),
        any::<bool>(),
    )
        .prop_map(|(pairs, singles, constant)| TestQuadratic {
            pairs,
            singles,
            constant,
        })
}

fn arb_words() -> impl Strategy<Value = CellWords> {
    prop::array::uniform4(any::<u64>())
}

/// Words of every input, one per evaluation.
fn arb_inputs() -> impl Strategy<Value = [CellWords; INPUTS]> {
    prop::array::uniform4(arb_words())
}

fn arb_parts() -> impl Strategy<Value = Quadratic> {
    (arb_words(), arb_words(), any::<bool>()).prop_map(|(quadratic, linear, constant)| {
        Quadratic::from_parts(quadratic, linear, u64::from(constant).wrapping_neg())
    })
}

/// The three parts and the poison flag of a value.
fn parts(x: Quadratic) -> (CellWords, CellWords, u64, bool) {
    (x.quadratic, x.linear, x.constant, x.poisoned)
}

/// Every lane of a sliced `GF(4)` value.
fn gf4_lanes(x: Gf4) -> Vec<BinaryField2> {
    (0..SLICED_LANES).map(|lane| x.lane(lane)).collect()
}

/// Every lane of one word, as bits of `GF(4)`.
fn bit_lanes(word: u64) -> Vec<BinaryField2> {
    (0..SLICED_LANES)
        .map(|lane| BinaryField2::from_bool((word >> lane) & 1 == 1))
        .collect()
}

proptest! {
    #[test]
    fn the_quadratic_part_is_the_coefficient_of_the_square(
        quadratic in arb_quadratic(),
        lo in arb_inputs(),
        hi in arb_inputs(),
    ) {
        // On the line lo + v (hi - lo) the quadratic has degree two in v, and its value at the
        // node g of GF(4) interpolates its values at 0 and 1 and its coefficient of v^2:
        //
        //     q(g) = q(0) (1 + g) + q(1) g + q(inf) (g^2 + g),        g^2 + g = 1
        let delta: [CellWords; INPUTS] = core::array::from_fn(|i| xor(lo[i], hi[i]));
        let leading = quadratic.eval(delta.map(|words| Quadratic::from(Bit::new(words))));
        prop_assert!(!leading.is_poisoned());
        for cell in 0..SLICED_CELLS {
            let at = |low: &[CellWords; INPUTS], high: Option<&[CellWords; INPUTS]>| {
                quadratic.eval(core::array::from_fn(|i| {
                    Gf4::from_planes(low[i][cell], high.map_or(0, |high| high[i][cell]))
                }))
            };
            let on_node = at(&lo, Some(&delta));
            let interpolated = at(&lo, None).scale(true, true)
                + at(&hi, None).scale(false, true)
                + Gf4::from_planes(leading.quadratic()[cell], 0);
            prop_assert_eq!(gf4_lanes(on_node), gf4_lanes(interpolated));
        }
    }

    #[test]
    fn the_whole_value_is_the_quadratic_on_the_bits(
        quadratic in arb_quadratic(),
        bits in arb_inputs(),
    ) {
        let whole = quadratic.eval(bits.map(|words| Quadratic::from(Bit::new(words))));
        for (cell, value) in whole.value().into_iter().enumerate() {
            let reference =
                quadratic.eval(core::array::from_fn(|i| Gf4::from_planes(bits[i][cell], 0)));
            prop_assert_eq!(bit_lanes(value), gf4_lanes(reference));
        }
    }

    #[test]
    fn input_arithmetic_matches_the_general_product(
        a in arb_words(),
        b in arb_words(),
        x in arb_parts(),
    ) {
        let (a, b) = (Bit::new(a), Bit::new(b));
        let (ea, eb) = (Quadratic::from(a), Quadratic::from(b));
        prop_assert_eq!(parts(a + b), parts(ea + eb));
        prop_assert_eq!(parts(a - b), parts(ea - eb));
        prop_assert_eq!(parts(a * b), parts(ea * eb));
        prop_assert_eq!(parts(a + x), parts(ea + x));
        prop_assert_eq!(parts(a - x), parts(ea - x));
        prop_assert_eq!(parts(x + a), parts(x + ea));
        prop_assert_eq!(parts(x - a), parts(x - ea));
        prop_assert_eq!(parts(a * x), parts(ea * x));
        prop_assert_eq!(parts(x * a), parts(x * ea));
        for constant in [F::ZERO, F::ONE] {
            let narrowed = Quadratic::narrow(constant);
            prop_assert_eq!(parts(a + constant), parts(ea + narrowed));
            prop_assert_eq!(parts(a - constant), parts(ea - narrowed));
            prop_assert_eq!(parts(a * constant), parts(ea * narrowed));
            prop_assert_eq!(parts(x + constant), parts(x + narrowed));
            prop_assert_eq!(parts(x * constant), parts(x * narrowed));
        }
    }

    #[test]
    fn squaring_and_booleanity_match_the_product(x in arb_parts()) {
        prop_assert_eq!(parts(x.square()), parts(x * x));
        prop_assert_eq!(parts(x.bool_check()), parts(x * x - x));
        prop_assert_eq!(parts(x.double()), parts(x + x));
        prop_assert_eq!(parts(-x), parts(x));
    }

    #[test]
    fn bit_lane_sums_weight_every_set_lane(bits in any::<u64>(), seed in any::<u64>()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = (0..SLICED_LANES).map(|_| rng.random()).collect::<Vec<Ghash128>>();
        let sums = BitLaneSums::new(&weights);
        let expected = weights
            .iter()
            .enumerate()
            .filter(|&(lane, _)| (bits >> lane) & 1 == 1)
            .map(|(_, &weight)| weight)
            .sum::<Ghash128>();
        prop_assert_eq!(sums.sum(bits), expected);
    }
}

#[test]
fn constants_narrow_to_bits_or_poison() {
    let zero = [0; SLICED_CELLS];
    assert_eq!(parts(Quadratic::narrow(F::ZERO)), (zero, zero, 0, false));
    assert_eq!(
        parts(Quadratic::narrow(F::ONE)),
        (zero, zero, u64::MAX, false)
    );
    for bits in 2..8_u128 {
        assert!(Quadratic::narrow(F::from_repr(bits)).is_poisoned());
    }
    // Poison survives every operation it takes part in.
    let outside = Quadratic::narrow(F::from_repr(2));
    let clean = Quadratic::from(Bit::new([0x0123_4567_89ab_cdef; SLICED_CELLS]));
    let input = Bit::new([u64::MAX; SLICED_CELLS]);
    assert!((clean * outside).is_poisoned());
    assert!((outside + clean).is_poisoned());
    assert!((outside * input).is_poisoned());
    assert!((input + outside).is_poisoned());
    assert!(outside.square().is_poisoned());
    assert!(outside.bool_check().is_poisoned());
    assert!((clean + F::from_repr(3)).is_poisoned());
    assert!((input * F::from_repr(3)).is_poisoned());
}

#[test]
fn ring_constants_are_those_of_gf2() {
    let zero = [0; SLICED_CELLS];
    assert_eq!(parts(Quadratic::ZERO), (zero, zero, 0, false));
    assert_eq!(parts(Quadratic::ONE), (zero, zero, u64::MAX, false));
    assert_eq!(parts(Quadratic::TWO), parts(Quadratic::ZERO));
    assert_eq!(parts(Quadratic::NEG_ONE), parts(Quadratic::ONE));
    assert_eq!(
        parts(Quadratic::from_prime_subfield(Prime::ONE)),
        parts(Quadratic::ONE)
    );
    assert_eq!(
        parts(Quadratic::from_prime_subfield(Prime::ZERO)),
        parts(Quadratic::ZERO)
    );
}
