use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
use p3_field::extension::BinomialExtensionField;
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::fiat_shamir::domain_separator::DomainSeparator;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

fn make_challenger() -> MyChallenger {
    let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
    DuplexChallenger::new(perm)
}

proptest! {
    /// Tests that `observe_domain_separator` produces the same challenger state
    /// as manually observing the same field elements.
    #[test]
    fn test_observe_domain_separator(seed in any::<u64>(), pattern_len in 1usize..16) {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Create a domain separator with random pattern elements
        let pattern: Vec<F> = (0..pattern_len).map(|_| rng.random()).collect();
        let domsep = DomainSeparator::<EF, F>::new(pattern.clone());

        // Create two identical challengers
        let mut challenger1 = make_challenger();
        let mut challenger2 = make_challenger();

        // Use observe_domain_separator on the first challenger
        domsep.observe_domain_separator(&mut challenger1);

        // Manually observe the same field elements on the second challenger
        challenger2.observe_slice(&pattern);

        // Verify that both challengers produce the same challenge values
        let sample1: F = challenger1.sample();
        let sample2: F = challenger2.sample();
        prop_assert_eq!(sample1, sample2, "Challengers should produce identical challenges after observing the same domain separator");

        // Sample a few more values to ensure consistency
        for _ in 0..3 {
            let s1: F = challenger1.sample();
            let s2: F = challenger2.sample();
            prop_assert_eq!(s1, s2, "All subsequent samples should match");
        }
    }
}
