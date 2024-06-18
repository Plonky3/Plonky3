use core::fmt::Debug;
use core::hash::Hash;

use p3_field::{AbstractField, Field};

use crate::{to_monty_elem, to_monty_from_array, MontyField31};

/// MontyParameters contains the prime P along with constants needed to convert elements intoand out of MONTY form.
pub trait MontyParameters:
    Copy + Clone + Default + Debug + Eq + PartialEq + Sync + Send + Hash + 'static
{
    // A 31-bit prime.
    const PRIME: u32;

    // The log_2 of the MONTY constant we use for faster multiplication.
    const MONTY_BITS: u32;

    // We define MU = P^-1 (mod 2^MONTY_BITS). This is different from the usual convention
    // (MU = -P^-1 (mod 2^MONTY_BITS)) but it avoids a carry.
    const MONTY_MU: u32;
    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;
}

/// PackedMontyParameters constains constants needed for MONTY operations for packings of Monty31 fields.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub trait PackedMontyParameters: crate::FieldParametersNeon + MontyParameters {}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(feature = "nightly-features", target_feature = "avx512f"))
))]
/// PackedMontyParameters constains constants needed for MONTY operations for packings of Monty31 fields.
pub trait PackedMontyParameters: crate::MontyParametersAVX2 + MontyParameters {}
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
/// PackedMontyParameters constains constants needed for MONTY operations for packings of Monty31 fields.
pub trait PackedMontyParameters:
    crate::FieldParametersAVX2 + crate::FieldParametersAVX512 + MontyParameters
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
/// PackedMontyParameters constains constants needed for MONTY operations for packings of Monty31 fields.
pub trait PackedMontyParameters: MontyParameters {}

/// BarrettParameters contains constants needed for the Barrett reduction used in the MDS code.
pub trait BarrettParameters: MontyParameters {
    const N: usize = 40; // beta = 2^N, fixing N = 40 here
    const PRIME_I128: i128 = Self::PRIME as i128;
    const PSEUDO_INV: i64 = (((1_i128) << (2 * Self::N)) / Self::PRIME_I128) as i64; // I = 2^80 / P => I < 2**50
    const MASK: i64 = !((1 << 10) - 1); // Lets us 0 out the bottom 10 digits of an i64.
}

/// FieldParameters contains constants and methods needed to imply AbstractField and Field for MontyField31.
pub trait FieldParameters: PackedMontyParameters + Sized {
    // Simple field constants.
    const MONTY_ZERO: MontyField31<Self> = to_monty_elem(0); // The monty form of 0 is always 0.
    const MONTY_ONE: MontyField31<Self> = to_monty_elem(1);
    const MONTY_TWO: MontyField31<Self> = to_monty_elem(2);
    const MONTY_NEG_ONE: MontyField31<Self> = to_monty_elem(Self::PRIME - 1);

    // A generator of the fields multiplicative group. Needs to be given in Monty Form.
    const MONTY_GEN: MontyField31<Self>;

    const HALF_P_PLUS_1: u32 = (Self::PRIME + 1) >> 1;

    fn exp_u64_generic<AF: AbstractField>(val: AF, power: u64) -> AF;
    fn try_inverse<F: Field>(p1: F) -> Option<F>;
}

/// TwoAdicData contains constants needed to imply TwoAdicField for Monty31 fields.
pub trait TwoAdicData: MontyParameters {
    // Largest n such that 2^n divides p - 1.
    const TWO_ADICITY: usize;

    // ArrayLike should be [u32; TWO_ADICITY + 1].
    type ArrayLike: AsRef<[MontyField31<Self>]> + Sized;

    // TWO_ADIC_GENERATORS needs to be in MONTY FORM.
    const TWO_ADIC_GENERATORS: Self::ArrayLike;
}

/// TODO: This should be deleted long term once we have improved our API for defining extension fields.
/// This allows us to implement Binomial Extensions over Monty31 fields.
pub trait BinomialExtensionData<const DEG: usize>: MontyParameters + Sized {
    // W is a value such that (x^DEG - WN) is irreducible.
    const W: u32;
    const MONTY_W: MontyField31<Self> = to_monty_elem(Self::W); // The MONTY form of W

    // DTH_ROOTN = W^((p - 1)/DEG)
    const DTH_ROOT: u32;
    const MONTY_DTH_ROOT: MontyField31<Self> = to_monty_elem(Self::DTH_ROOT);

    const EXT_GENERATOR: [u32; DEG];
    const MONTY_EXT_GENERATOR: [u32; DEG] = to_monty_from_array::<DEG, Self>(Self::EXT_GENERATOR);

    const EXT_TWO_ADICITY: usize;

    fn u32_ext_two_adic_generator(bits: usize) -> [u32; DEG];
}
