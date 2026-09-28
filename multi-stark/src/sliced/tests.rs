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

/// The byte encoding of `value`, read as a little-endian word.
fn encoding<R: Field>(value: R) -> u128 {
    let mut bytes = [0u8; 16];
    for (slot, byte) in bytes.iter_mut().zip(value.into_bytes()) {
        *slot = byte;
    }
    u128::from_le_bytes(bytes)
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
        prop_assert_eq!(lanes([a, b, a * b].into_iter().sum()), lanewise(|x, y| x + y + x * y));
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

    // The sliced kernel adds field elements as the XOR of their byte encodings.
    #[test]
    fn byte_encodings_add_as_xor(seed in any::<u64>()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (a, b): (Ghash128, Ghash128) = (rng.random(), rng.random());
        prop_assert_eq!(encoding(a + b), encoding(a) ^ encoding(b));
        let (a, b): (F, F) = (rng.random(), rng.random());
        prop_assert_eq!(encoding(a + b), encoding(a) ^ encoding(b));
    }

    #[test]
    fn lane_sums_weight_every_set_lane(value in arb_sliced(), seed in any::<u64>()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = (0..SLICED_LANES).map(|_| rng.random()).collect::<Vec<Ghash128>>();
        let generator = Ghash128::from(F::from(S::GENERATOR));
        let sums = LaneSums::new(&weights, generator);
        let (low, high) = (value.low, value.high);
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

/// The byte-sliced kernel, on the targets that have it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod kernel_tests {
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::{Ghash128, Poly64, Poly192};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing, RawDataSerializable};
    use proptest::prelude::*;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{F, S, Sliced, encoding};
    use crate::selectors::BoundaryEvals;
    use crate::sliced::{
        LaneSums, PreparedPowers, PreparedSums, SLICED_LANES, SlicedEvaluation, SlicedFolder,
        kernel,
    };

    /// `powers` laid out for the kernel, however few they are.
    fn prepared<R: Field>(powers: &[R], generator: R) -> PreparedPowers<R> {
        let basis =
            kernel::coordinate_basis::<R>().expect("the kernel takes a 128-bit binary field");
        PreparedPowers(kernel::Prepared::new(powers, generator, Arc::from(basis)))
    }

    /// The generator of `S` as an element of `R`.
    fn generator<R: Field + From<F>>() -> R {
        R::from(F::from(S::GENERATOR))
    }

    /// Random lane weights and their tables.
    fn random_lanes<R>(rng: &mut SmallRng) -> LaneSums<R>
    where
        R: Field + From<F>,
        StandardUniform: Distribution<R>,
    {
        let weights = (0..SLICED_LANES).map(|_| rng.random()).collect::<Vec<R>>();
        LaneSums::new(&weights, generator())
    }

    /// Planes of a constraint in block `index / 8`, the blocks taking turns being zero, low
    /// plane only, high plane only, and full but for every third constraint.
    fn planes(rng: &mut SmallRng, index: usize) -> (u64, u64) {
        match ((index / 8) % 4, index % 3) {
            (0, _) | (3, 0) => (0, 0),
            (1, _) => (rng.random(), 0),
            (2, _) => (0, rng.random()),
            _ => (rng.random(), rng.random()),
        }
    }

    /// The kernel's sum of `count` random constraints, then `extra` more past the prepared
    /// powers, beside the one-at-a-time sum of the first `count`.
    fn kernel_and_lane_sums<R>(seed: u64, count: usize, extra: usize) -> (R, R)
    where
        R: Field + From<F>,
        StandardUniform: Distribution<R>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let lanes = random_lanes::<R>(&mut rng);
        let powers = (0..count).map(|_| rng.random()).collect::<Vec<R>>();
        let prepared = prepared(&powers, generator());
        let mut sums = PreparedSums::new();
        let mut expected = R::ZERO;
        for index in 0..count + extra {
            let (low, high) = planes(&mut rng, index);
            sums.add(&prepared, index, low, high);
            if let Some(&power) = powers.get(index) {
                expected += power * lanes.sum(low, high);
            }
        }
        (sums.finish(&prepared, &lanes), expected)
    }

    proptest! {
        #[test]
        fn prepared_powers_sum_as_the_lane_sums_do(
            seed in any::<u64>(),
            count in 0_usize..70,
            extra in 0_usize..20,
        ) {
            let (kernel, expected) = kernel_and_lane_sums::<Ghash128>(seed, count, extra);
            prop_assert_eq!(kernel, expected);
            let (kernel, expected) = kernel_and_lane_sums::<F>(seed, count, extra);
            prop_assert_eq!(kernel, expected);
        }
    }

    /// Asserts every main column, then its product with the next one.
    struct Columns(usize);

    impl<T> BaseAir<T> for Columns {
        fn width(&self) -> usize {
            self.0
        }
    }

    impl<AB: AirBuilder> Air<AB> for Columns {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            for (column, &value) in local.iter().enumerate() {
                builder.assert_zero(value);
                builder.assert_zero(value * local[(column + 1) % local.len()]);
            }
        }
    }

    /// One evaluation of `Columns(width)` over random planes against `powers` random powers,
    /// with the kernel and without it.
    fn folder_sums(
        seed: u64,
        width: usize,
        powers: usize,
    ) -> (SlicedEvaluation<Ghash128>, SlicedEvaluation<Ghash128>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let local = (0..width)
            .map(|column| {
                let (low, high) = planes(&mut rng, 2 * column);
                Sliced::from_planes(low, high)
            })
            .collect::<Vec<_>>();
        let lanes = random_lanes::<Ghash128>(&mut rng);
        let powers = (0..powers).map(|_| rng.random()).collect::<Vec<Ghash128>>();
        let prepared = prepared(&powers, generator());
        let boundary = BoundaryEvals {
            first: Sliced::ZERO,
            last: Sliced::ZERO,
            transition: Sliced::ONE,
        };
        let folder = || SlicedFolder::new(&local, &local, boundary, &[], &powers, &lanes);
        let air = Columns(width);
        (
            folder().with_prepared_powers(&prepared).eval_air(&air),
            folder().eval_air(&air),
        )
    }

    proptest! {
        #[test]
        fn the_folder_sums_as_well_with_the_kernel(seed in any::<u64>(), width in 1_usize..40) {
            let (kernel, lanes) = folder_sums(seed, width, 2 * width);
            prop_assert!(!kernel.poisoned && !lanes.poisoned);
            prop_assert_eq!(kernel.value, lanes.value);
        }
    }

    #[test]
    #[should_panic(
        expected = "attached alpha powers must match the number of asserted constraints"
    )]
    fn a_constraint_past_the_prepared_powers_fails_the_count_check() {
        let _ = folder_sums(1, 12, 23);
    }

    #[test]
    #[should_panic(
        expected = "attached alpha powers must match the number of asserted constraints"
    )]
    fn a_prepared_power_left_over_fails_the_count_check() {
        let _ = folder_sums(2, 12, 25);
    }

    #[test]
    fn only_an_air_asserting_enough_constraints_takes_the_kernel() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut powers = |len| (0..len).map(|_| rng.random()).collect::<Vec<Ghash128>>();
        let airs = [
            powers(kernel::MIN_CONSTRAINTS - 1),
            powers(kernel::MIN_CONSTRAINTS),
            Vec::new(),
        ];
        let prepared = PreparedPowers::per_air(&airs, generator());
        assert!(prepared[0].is_none());
        assert!(prepared[1].is_some());
        assert!(prepared[2].is_none());
    }

    #[test]
    fn a_binary_field_of_128_coordinates_has_a_coordinate_basis() {
        let ghash = kernel::coordinate_basis::<Ghash128>().expect("GHASH is linear over F_2");
        let tower = kernel::coordinate_basis::<F>().expect("the tower is linear over F_2");
        for bit in 0..128 {
            assert_eq!(encoding(ghash[bit]), 1 << bit);
            assert_eq!(encoding(tower[bit]), 1 << bit);
        }
    }

    #[test]
    fn a_field_of_another_size_or_characteristic_has_no_coordinate_basis() {
        assert_eq!(Poly64::NUM_BYTES, 8);
        assert!(kernel::coordinate_basis::<Poly64>().is_none());
        assert_eq!(Poly192::NUM_BYTES, 24);
        assert!(kernel::coordinate_basis::<Poly192>().is_none());
        // Sixteen bytes, but odd characteristic.
        type Quartic = BinomialExtensionField<BabyBear, 4>;
        assert_eq!(Quartic::NUM_BYTES, 16);
        assert!(kernel::coordinate_basis::<Quartic>().is_none());
    }

    #[test]
    fn a_nonlinear_encoding_has_no_basis() {
        assert!(kernel::basis_under::<Ghash128>(encoding).is_some());
        // Affine: zero does not encode to zero.
        assert!(kernel::basis_under::<Ghash128>(|x| encoding(x) ^ 1).is_none());
        // Zero to zero, but the top bit takes the product of the two lowest coordinates.
        assert!(
            kernel::basis_under::<Ghash128>(|x| {
                let bits = encoding(x);
                bits ^ ((bits & (bits >> 1) & 1) << 127)
            })
            .is_none()
        );
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
    assert!(
        [clean, outside, clean]
            .into_iter()
            .sum::<Sliced>()
            .is_poisoned()
    );
    assert!((clean + F::from_repr(7)).is_poisoned());
}

#[test]
fn ring_constants_are_those_of_gf4() {
    assert_eq!(lanes(Sliced::ONE), [S::ONE; SLICED_LANES]);
    assert_eq!(lanes(Sliced::ZERO), [S::ZERO; SLICED_LANES]);
    assert_eq!(lanes(Sliced::TWO), [S::TWO; SLICED_LANES]);
    assert_eq!(lanes(Sliced::NEG_ONE), [S::NEG_ONE; SLICED_LANES]);
}
