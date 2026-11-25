//! The AIR over fields whose characteristic is below 2^16.

use p3_air::{AirLayout, check_constraints, get_symbolic_constraints};
use p3_binary_field::BinaryField128;
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS, generate_trace_rows};
use p3_monty_31::{FieldParameters, MontyField31, MontyParameters, PackedMontyParameters};

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

// The largest prime below 2^16.
test_prime_field!(SmallParameters, Small, 65521u32, 0xcdf01111u32, 17);

// The smallest prime at least 2^16.
test_prime_field!(EnoughParameters, Enough, 65537u32, 0xffff0001u32, 3);

#[test]
#[should_panic(expected = "characteristic must be at least 2^16")]
fn rejects_a_characteristic_below_2_16() {
    // Over p = 65521 < 2^16, limb 65521 reduces to 0, colliding with an all-zero preimage.
    type F = Small;
    let mut input = [0u64; 25];
    input[0] = 65521;
    let trace = generate_trace_rows::<F>(vec![input], 0);
    check_constraints(&KeccakAir {}, &trace, &[]);
}

#[test]
#[should_panic(expected = "characteristic must be at least 2^16")]
fn rejects_characteristic_2() {
    let layout = AirLayout {
        main_width: NUM_KECCAK_COLS,
        ..Default::default()
    };
    let _ = get_symbolic_constraints::<BinaryField128, _>(&KeccakAir {}, layout);
}

#[test]
fn accepts_an_honest_trace_at_the_bound() {
    // 65537 >= 2^16, so all 16-bit limbs are distinct field elements.
    type F = Enough;
    let trace = generate_trace_rows::<F>(vec![[0u64; 25]], 0);
    check_constraints(&KeccakAir {}, &trace, &[]);
}
