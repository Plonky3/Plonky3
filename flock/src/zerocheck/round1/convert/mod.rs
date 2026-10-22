//! The per-window convert of the round-1 sweep: medium rows of bytes to challenge-field sums per lane.
//!
//! Lane `s` of every row is weighted by `g^b phi8(byte)` at medium position `b`.

#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
mod portable;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
mod x86_64_avx512;

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
)))]
pub(crate) use self::portable::{Convert, ConvertTables};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
pub(crate) use self::x86_64_avx512::{Convert, ConvertTables};

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::Poly192;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::skip::{SKIP_SIZE, phi8};
    use crate::zerocheck::fixed::gamma_powers;

    /// Lane `lane` of medium rows by the definition.
    fn reference(rows: &[[u8; 64]], lane: usize) -> Poly192 {
        rows.iter()
            .zip(gamma_powers())
            .map(|(row, g)| g * phi8(row[lane]))
            .sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn the_dispatched_convert_is_the_definition(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let tables = ConvertTables::new();
            let mut convert = Convert::new();
            let (mut want_ab, mut want_c) = ([Poly192::ZERO; SKIP_SIZE], [Poly192::ZERO; SKIP_SIZE]);
            for n in [16, 16, 0, 7, 15] {
                let rows = |rng: &mut SmallRng| -> Vec<[u8; 64]> {
                    (0..n).map(|_| core::array::from_fn(|_| rng.random())).collect()
                };
                let (ab, c) = (rows(&mut rng), rows(&mut rng));
                let eq: Poly192 = rng.random();
                convert.accumulate(&tables, &ab, &c, eq);
                for lane in 0..SKIP_SIZE {
                    want_ab[lane] += reference(&ab, lane) * eq;
                    want_c[lane] += reference(&c, lane) * eq;
                }
            }
            prop_assert_eq!(convert.values(), (want_ab, want_c));
        }

        #[test]
        fn the_portable_convert_is_the_definition(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let tables = portable::ConvertTables::new();
            let mut convert = portable::Convert::new();
            let rows: Vec<[u8; 64]> = (0..11).map(|_| core::array::from_fn(|_| rng.random())).collect();
            let eq: Poly192 = rng.random();
            convert.accumulate(&tables, &rows, &rows, eq);
            let want: [Poly192; SKIP_SIZE] = core::array::from_fn(|lane| reference(&rows, lane) * eq);
            prop_assert_eq!(convert.values(), (want, want));
        }
    }
}
