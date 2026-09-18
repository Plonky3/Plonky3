use alloc::vec::Vec;

use p3_binary_field::{BinaryField2, BinaryField128, Ghash128, TowerLevel};
use p3_field::{Field, PrimeCharacteristicRing};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;

type F = BinaryField128;
type S = BinaryField2;
type Sliced = SlicedGf4<F, S>;

/// Every lane of `value` as an element of `S`.
fn lanes(value: Sliced) -> Vec<S> {
    (0..SLICED_LANES).map(|lane| value.lane(lane)).collect()
}

/// The sliced value whose lanes are `values`.
fn from_lanes(values: &[S]) -> Sliced {
    let (mut low, mut high) = (0, 0);
    for (lane, &value) in values.iter().enumerate() {
        let (l, h) = gf4_coordinates(value).expect("a GF(4) element");
        low |= u64::from(l) << lane;
        high |= u64::from(h) << lane;
    }
    Sliced::from_planes(low, high)
}

fn arb_sliced() -> impl Strategy<Value = Sliced> {
    (any::<u64>(), any::<u64>()).prop_map(|(low, high)| Sliced::from_planes(low, high))
}

/// The element of `S` with coordinates `(low, high)`.
fn element(low: bool, high: bool) -> S {
    S::from_bool(low) + S::from_bool(high) * S::GENERATOR
}

#[test]
fn the_tower_gf4_is_the_sliced_field() {
    assert!(is_gf4::<S>());
    assert!(!is_gf4::<F>());
    // The coordinates name every element exactly once.
    let elements = [(false, false), (true, false), (false, true), (true, true)]
        .map(|(low, high)| element(low, high));
    for (i, &a) in elements.iter().enumerate() {
        for &b in &elements[i + 1..] {
            assert_ne!(a, b);
        }
    }
    for low in [false, true] {
        for high in [false, true] {
            assert_eq!(gf4_coordinates(element(low, high)), Some((low, high)));
        }
    }
}

proptest! {
    #[test]
    fn arithmetic_is_lanewise_gf4(a in arb_sliced(), b in arb_sliced()) {
        let (x, y) = (lanes(a), lanes(b));
        let lanewise = |op: fn(S, S) -> S| {
            x.iter().zip(&y).map(|(&x, &y)| op(x, y)).collect::<Vec<_>>()
        };
        prop_assert_eq!(lanes(a + b), lanewise(|x, y| x + y));
        prop_assert_eq!(lanes(a - b), lanewise(|x, y| x - y));
        prop_assert_eq!(lanes(a * b), lanewise(|x, y| x * y));
        prop_assert_eq!(lanes(-a), x.iter().map(|&x| -x).collect::<Vec<_>>());
        prop_assert_eq!(lanes(a.square()), x.iter().map(|&x| x.square()).collect::<Vec<_>>());
        prop_assert_eq!(lanes(a.double()), x.iter().map(|&x| x.double()).collect::<Vec<_>>());
        prop_assert_eq!(
            lanes(a.bool_check()),
            x.iter().map(|&x| x * (x - S::ONE)).collect::<Vec<_>>()
        );
        for low in [false, true] {
            for high in [false, true] {
                let c = element(low, high);
                prop_assert_eq!(
                    lanes(a.scale(low, high)),
                    x.iter().map(|&x| c * x).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn lane_sums_weight_every_set_lane(value in arb_sliced(), seed in any::<u64>()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = (0..SLICED_LANES).map(|_| rng.random()).collect::<Vec<Ghash128>>();
        let generator = Ghash128::from(F::from(S::GENERATOR));
        let sums = LaneSums::new(&weights, generator);
        let (low, high) = value.planes();
        let expected = weights
            .iter()
            .enumerate()
            .map(|(lane, &weight)| {
                let (l, h) = ((low >> lane) & 1 == 1, (high >> lane) & 1 == 1);
                weight * (Ghash128::from_bool(l) + Ghash128::from_bool(h) * generator)
            })
            .sum::<Ghash128>();
        prop_assert_eq!(sums.sum(low, high), expected);
    }
}

#[test]
fn constants_narrow_or_poison() {
    for bits in 0..4_u128 {
        let value = Sliced::narrow(F::from_repr(bits));
        assert!(!value.is_poisoned());
        let expected = F::from_repr(bits).as_subfield().expect("a GF(4) cell");
        assert!(lanes(value).iter().all(|&lane| lane == expected));
    }
    let outside = Sliced::narrow(F::from_repr(4));
    assert!(outside.is_poisoned());
    // Poison survives every operation it takes part in.
    let clean = from_lanes(&[S::ONE; SLICED_LANES]);
    assert!((clean * outside).is_poisoned());
    assert!((outside + clean).is_poisoned());
    assert!(outside.square().is_poisoned());
    assert!((clean + F::from_repr(7)).is_poisoned());
}

#[test]
fn ring_constants_are_those_of_gf4() {
    assert_eq!(lanes(Sliced::ONE), [S::ONE; SLICED_LANES]);
    assert_eq!(lanes(Sliced::ZERO), [S::ZERO; SLICED_LANES]);
    assert_eq!(lanes(Sliced::TWO), [S::TWO; SLICED_LANES]);
    assert_eq!(lanes(Sliced::NEG_ONE), [S::NEG_ONE; SLICED_LANES]);
}
