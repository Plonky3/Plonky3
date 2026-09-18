//! `Powers::collect_n` against the scalar powers, with the parallel split taken over
//! Rayon pools of several sizes.

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::test_powers_collect;
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;

fn check_fields() {
    test_powers_collect::<BabyBear>();
    test_powers_collect::<KoalaBear>();
    test_powers_collect::<Goldilocks>();
    test_powers_collect::<Mersenne31>();
    test_powers_collect::<BinomialExtensionField<BabyBear, 4>>();
}

#[test]
fn collect_n_matches_scalar_powers_for_each_pool_size() {
    #[cfg(feature = "parallel")]
    for threads in [1, 3, 4, 7] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(check_fields);
    }
    #[cfg(not(feature = "parallel"))]
    check_fields();
}
