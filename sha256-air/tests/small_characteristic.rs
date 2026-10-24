//! The AIR over prime fields too small for its addition checks.

use p3_air::check_constraints;
use p3_monty_31::{FieldParameters, MontyField31, MontyParameters, PackedMontyParameters};
use p3_sha256_air::{SHA256_IV, Sha256Air, generate_trace_rows};

/// Define a Montgomery prime field for tests, with the packing parameters of every target.
macro_rules! test_prime_field {
    ($params:ident, $field:ident, $prime:expr, $mu:expr, $generator:expr) => {
        #[derive(Copy, Clone, Default, Debug, Eq, Hash, PartialEq)]
        struct $params;

        impl MontyParameters for $params {
            const PRIME: u32 = $prime;
            const MONTY_BITS: u32 = 32;
            const MONTY_MU: u32 = $mu;
        }

        impl PackedMontyParameters for $params {}

        impl FieldParameters for $params {
            const MONTY_GEN: MontyField31<Self> = MontyField31::new($generator);
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        impl p3_monty_31::MontyParametersNeon for $params {
            const PACKED_P: core::arch::aarch64::uint32x4_t =
                unsafe { core::mem::transmute([$prime; 4]) };
            const PACKED_MU: core::arch::aarch64::int32x4_t =
                unsafe { core::mem::transmute([$mu; 4]) };
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ))]
        impl p3_monty_31::MontyParametersAVX2 for $params {
            const PACKED_P: core::arch::x86_64::__m256i =
                unsafe { core::mem::transmute([$prime; 8]) };
            const PACKED_MU: core::arch::x86_64::__m256i =
                unsafe { core::mem::transmute([$mu; 8]) };
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        impl p3_monty_31::MontyParametersAVX512 for $params {
            const PACKED_P: core::arch::x86_64::__m512i =
                unsafe { core::mem::transmute([$prime; 16]) };
            const PACKED_MU: core::arch::x86_64::__m512i =
                unsafe { core::mem::transmute([$mu; 16]) };
        }

        type $field = MontyField31<$params>;
    };
}

// The prime 262147, just above 2^18.
test_prime_field!(SmallParameters, Small, 262147u32, 0x71c6aaabu32, 2);

// The prime 458789, the first above 7 * 2^16.
test_prime_field!(EnoughParameters, Enough, 458789u32, 0xeced1badu32, 2);

/// The zero block on the initial hash value, twice.
fn zero_block_inputs() -> Vec<[u32; 24]> {
    let mut input = [0u32; 24];
    input[16..].copy_from_slice(&SHA256_IV);
    vec![input; 2]
}

#[test]
#[should_panic(expected = "characteristic must exceed 7 * 2^16")]
fn rejects_a_characteristic_at_most_7_times_2_16() {
    // Invariant: the unchecked intermediate sums are sound only above 7 * 2^16.
    //
    // Fixture state: 262147 lies in (2^18, 7 * 2^16].
    //
    //     2^16 = 65536 < 262147 <= 458752 = 7 * 2^16
    //
    // Over such a prime a forged round can satisfy every constraint.
    // So the evaluation refuses the field, even on an honest trace.
    type F = Small;
    let trace = generate_trace_rows::<F>(zero_block_inputs(), 0);
    check_constraints(&Sha256Air, &trace, &[]);
}

#[test]
fn accepts_an_honest_trace_just_above_the_bound() {
    // Invariant: 458789 > 7 * 2^16 = 458752, so the evaluation runs.
    type F = Enough;
    let trace = generate_trace_rows::<F>(zero_block_inputs(), 0);
    check_constraints(&Sha256Air, &trace, &[]);
}
