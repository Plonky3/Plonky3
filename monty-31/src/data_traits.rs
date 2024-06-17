use core::fmt::Debug;
use core::hash::Hash;

use p3_field::{AbstractField, Field};

use crate::{to_monty, to_monty_from_array};

/// MontyParameters contains constants needed to convert elements into MONTY form and perform MONTY reductions.
pub trait MontyParameters {
    // A 31-bit prime.
    const PRIME: u32;

    // The log_2 of the MONTY constant we use for faster multiplication.
    const MONTY_BITS: u32;

    // We define MU = P^-1 (mod 2^MONTY_BITS). This is different from the usual convention
    // (MU = -P^-1 (mod 2^MONTY_BITS)) but it avoids a carry.
    const MONTY_MU: u32;
    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;
}

/// BarrettParameters contains constants needed for the Barrett reduction used in the MDS code.
pub trait BarrettParameters: MontyParameters {
    const N: usize = 40; // beta = 2^N, fixing N = 40 here
    const PRIME_I128: i128 = Self::PRIME as i128;
    const PSEUDO_INV: i64 = (((1_i128) << (2 * Self::N)) / Self::PRIME_I128) as i64; // I = 2^80 / P => I < 2**50
    const MASK: i64 = !((1 << 10) - 1); // Lets us 0 out the bottom 10 digits of an i64.
}

/// FieldConstants contains constants and methods needed to imply AbstractField and Field for MontyField31.
pub trait FieldConstants: MontyParameters + Sized {
    // Simple field constants.
    const MONTY_ZERO: u32 = 0; // The monty form of 0 is always 0.
    const MONTY_ONE: u32 = to_monty::<Self>(1);
    const MONTY_TWO: u32 = to_monty::<Self>(2);
    const MONTY_NEG_ONE: u32 = Self::PRIME - Self::MONTY_ONE; // As MONTY_ONE =/= 0, MONTY_NEG_ONE = P - MONTY_ONE.

    // TODO: Don't save both GEN and MONTY_GEN. Only one of these should be in the trait.
    const GEN: u32; // A generator of the fields multiplicative group.
    const MONTY_GEN: u32 = to_monty::<Self>(Self::GEN); // Generator saved in MONTY form

    const HALF_P_PLUS_1: u32 = (Self::PRIME + 1) >> 1;

    fn exp_u64_generic<AF: AbstractField>(val: AF, power: u64) -> AF;
    fn try_inverse<F: Field>(p1: F) -> Option<F>;
}

pub trait TwoAdicData {
    // Largest n such that 2^n divides p - 1.
    const TWO_ADICITY: usize;

    // ArrayLike should be [u32; TWO_ADICITY + 1].
    type ArrayLike: AsRef<[u32]> + Sized;

    // TWO_ADIC_GENERATORS needs to be in MONTY FORM.
    const TWO_ADIC_GENERATORS: Self::ArrayLike;
}

// TODO: This should be deleted long term once we have improved our API for defining extension fields.
pub trait BinomialExtensionData<const DEG: usize>: MontyParameters + Sized {
    // W is a value such that (x^DEG - WN) is irreducible.
    const W: u32;
    const MONTY_W: u32 = to_monty::<Self>(Self::W); // The MONTY form of W4

    // DTH_ROOTN = W^((p - 1)/DEG)
    const DTH_ROOT: u32;
    const MONTY_DTH_ROOT: u32 = to_monty::<Self>(Self::DTH_ROOT);

    const EXT_GENERATOR: [u32; DEG];
    const MONTY_EXT_GENERATOR: [u32; DEG] = to_monty_from_array::<DEG, Self>(Self::EXT_GENERATOR);

    const EXT_TWO_ADICITY: usize;

    fn u32_ext_two_adic_generator(bits: usize) -> [u32; DEG];
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub trait FieldParameters:
    Copy
    + Clone
    + Default
    + Debug
    + Eq
    + PartialEq
    + Sync
    + Send
    + Hash
    + 'static
    + MontyParameters
    + FieldConstants
    + TwoAdicData
    + BarrettParameters
    + BinomialExtensionData<4>
    + BinomialExtensionData<5>
    + crate::FieldParametersNeon
{
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(feature = "nightly-features", target_feature = "avx512f"))
))]
pub trait FieldParameters:
    Copy
    + Clone
    + Default
    + Debug
    + Eq
    + PartialEq
    + Sync
    + Send
    + Hash
    + 'static
    + MontyParameters
    + FieldConstants
    + TwoAdicData
    + BarrettParameters
    + BinomialExtensionData<4>
    + BinomialExtensionData<5>
    + crate::FieldParametersAVX2
{
}
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub trait FieldParameters:
    Copy
    + Clone
    + Default
    + Debug
    + Eq
    + PartialEq
    + Sync
    + Send
    + Hash
    + 'static
    + MontyParameters
    + FieldConstants
    + TwoAdicData
    + BarrettParameters
    + BinomialExtensionData<4>
    + BinomialExtensionData<5>
    + crate::FieldParametersAVX2
    + crate::FieldParametersAVX512
{
}
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ),
    all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
)))]
pub trait FieldParameters:
    Copy
    + Clone
    + Default
    + Debug
    + Eq
    + PartialEq
    + Sync
    + Send
    + Hash
    + 'static
    + MontyParameters
    + FieldConstants
    + TwoAdicData
    + BarrettParameters
    + BinomialExtensionData<4>
    + BinomialExtensionData<5>
{
}
