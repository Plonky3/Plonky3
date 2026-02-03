#![no_std]

#[cfg(test)]
extern crate alloc;

#[cfg(test)]
mod extension_test;
mod koala_bear;
mod poseidon2;

pub use koala_bear::*;
// Re-export quintic extension field type for convenience.
pub use p3_field::extension::QuinticExtensionField;
pub use poseidon2::*;

/// Quintic extension of KoalaBear using the trinomial `X^5 + X^2 - 1`.
///
/// This is the degree-5 extension field `KoalaBear[X]/(X^5 + X^2 - 1)`.
/// Elements are polynomials of degree at most 4 with KoalaBear coefficients.
///
/// # Example
///
/// ```
/// use p3_koala_bear::{KoalaBear, KoalaBear5};
/// use p3_field::{Field, PrimeCharacteristicRing};
///
/// let a = KoalaBear5::from(KoalaBear::new(5));
/// let b = a * a;  // Squaring works
/// let c = b.inverse();  // Inversion works
/// assert_eq!(b * c, KoalaBear5::ONE);
/// ```
pub type KoalaBear5 = QuinticExtensionField<KoalaBear>;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use x86_64_avx2::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod x86_64_avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use x86_64_avx512::*;
