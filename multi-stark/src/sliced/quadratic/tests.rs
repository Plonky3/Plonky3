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

fn arb_parts() -> impl Strategy<Value = Quadratic> {
    (any::<u64>(), any::<u64>(), any::<bool>()).prop_map(|(quadratic, linear, constant)| {
        Quadratic::from_parts(quadratic, linear, u64::from(constant).wrapping_neg())
    })
}

/// The three parts and the poison flag of a value.
fn parts(x: Quadratic) -> (u64, u64, u64, bool) {
    (x.quadratic, x.linear, x.constant, x.poisoned)
}

proptest! {
    #[test]
    fn the_quadratic_part_is_the_coefficient_of_the_square(
        quadratic in arb_quadratic(),
        lo in prop::array::uniform4(any::<u64>()),
        hi in prop::array::uniform4(any::<u64>()),
    ) {
        // On the line lo + v (hi - lo) the quadratic has degree two in v, and its value at the
        // node g of GF(4) interpolates its values at 0 and 1 and its coefficient of v^2:
        //
        //     q(g) = q(0) (1 + g) + q(1) g + q(inf) (g^2 + g),        g^2 + g = 1
        let at = |low: [u64; INPUTS], high: [u64; INPUTS]| {
            quadratic.eval(core::array::from_fn(|i| Gf4::from_planes(low[i], high[i])))
        };
        let delta: [u64; INPUTS] = core::array::from_fn(|i| lo[i] ^ hi[i]);
        let on_node = at(lo, delta);
        let (at_zero, at_one) = (at(lo, [0; INPUTS]), at(hi, [0; INPUTS]));
        let leading = quadratic.eval(delta.map(|bits| Quadratic::from(Bit::new(bits))));
        prop_assert!(!leading.is_poisoned());
        let interpolated = at_zero.scale(true, true)
            + at_one.scale(false, true)
            + Gf4::from_planes(leading.quadratic(), 0);
        prop_assert_eq!(
            (0..SLICED_LANES).map(|lane| on_node.lane(lane)).collect::<Vec<_>>(),
            (0..SLICED_LANES).map(|lane| interpolated.lane(lane)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_whole_value_is_the_quadratic_on_the_bits(
        quadratic in arb_quadratic(),
        bits in prop::array::uniform4(any::<u64>()),
    ) {
        let whole = quadratic.eval(bits.map(|bits| Quadratic::from(Bit::new(bits))));
        let reference = quadratic.eval(bits.map(|bits| Gf4::from_planes(bits, 0)));
        prop_assert_eq!(
            (0..SLICED_LANES)
                .map(|lane| BinaryField2::from_bool((whole.value() >> lane) & 1 == 1))
                .collect::<Vec<_>>(),
            (0..SLICED_LANES).map(|lane| reference.lane(lane)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn input_arithmetic_matches_the_general_product(
        a in any::<u64>(),
        b in any::<u64>(),
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
    assert_eq!(parts(Quadratic::narrow(F::ZERO)), (0, 0, 0, false));
    assert_eq!(parts(Quadratic::narrow(F::ONE)), (0, 0, u64::MAX, false));
    for bits in 2..8_u128 {
        assert!(Quadratic::narrow(F::from_repr(bits)).is_poisoned());
    }
    // Poison survives every operation it takes part in.
    let outside = Quadratic::narrow(F::from_repr(2));
    let clean = Quadratic::from(Bit::new(0x0123_4567_89ab_cdef));
    let input = Bit::new(u64::MAX);
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
    assert_eq!(parts(Quadratic::ZERO), (0, 0, 0, false));
    assert_eq!(parts(Quadratic::ONE), (0, 0, u64::MAX, false));
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
