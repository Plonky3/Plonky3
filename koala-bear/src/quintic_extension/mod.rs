use p3_field::{Algebra, Field, PrimeCharacteristicRing, packed_mod_add, packed_mod_sub};
use p3_monty_31::{MontyParameters, base_mul_packed, monty_add, monty_sub};

use crate::packed_quintic_extension::PackedQuinticExtensionField;
use crate::packing::quintic_mul_packed;
use crate::quintic_extension::quintic_extension::QuinticExtensionField;
use crate::{KoalaBear, KoalaBearParameters};

pub(crate) mod packed_quintic_extension;
pub(crate) mod packing;
pub(crate) mod quintic_extension;

pub type QuinticExtensionFieldKB = QuinticExtensionField<KoalaBear>;
pub type PackedQuinticExtensionFieldKB =
    PackedQuinticExtensionField<KoalaBear, <KoalaBear as Field>::Packing>;

impl QuinticExtendable for KoalaBear {
    const FROBENIUS_MATRIX: [[Self; 5]; 4] = [
        [
            Self::new(1576402667),
            Self::new(1173144480),
            Self::new(1567662457),
            Self::new(1206866823),
            Self::new(2428146),
        ],
        [
            Self::new(1680345488),
            Self::new(1381986),
            Self::new(615237464),
            Self::new(1380104858),
            Self::new(295431824),
        ],
        [
            Self::new(441230756),
            Self::new(323126830),
            Self::new(704986542),
            Self::new(1445620072),
            Self::new(503505220),
        ],
        [
            Self::new(1364444097),
            Self::new(1144738982),
            Self::new(2008416047),
            Self::new(143367062),
            Self::new(1027410849),
        ],
    ];

    const EXT_GENERATOR: [Self; 5] = Self::new_array([2, 1, 0, 0, 0]);
}

impl QuinticExtendableAlgebra<KoalaBear> for KoalaBear {
    #[inline(always)]
    fn quintic_mul(a: &[Self; 5], b: &[Self; 5], res: &mut [Self; 5]) {
        quintic_mul_packed(a, b, res);
    }

    #[inline(always)]
    fn quintic_add(a: &[Self; 5], b: &[Self; 5]) -> [Self; 5] {
        let mut res = [Self::ZERO; 5];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; 5] = &*(a.as_ptr() as *const [u32; 5]);
            let b: &[u32; 5] = &*(b.as_ptr() as *const [u32; 5]);
            let res: &mut [u32; 5] = &mut *(res.as_mut_ptr() as *mut [u32; 5]);

            packed_mod_add(
                a,
                b,
                res,
                KoalaBearParameters::PRIME,
                monty_add::<KoalaBearParameters>,
            );
        }
        res
    }

    #[inline(always)]
    fn quintic_sub(a: &[Self; 5], b: &[Self; 5]) -> [Self; 5] {
        let mut res = [Self::ZERO; 5];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; 5] = &*(a.as_ptr() as *const [u32; 5]);
            let b: &[u32; 5] = &*(b.as_ptr() as *const [u32; 5]);
            let res: &mut [u32; 5] = &mut *(res.as_mut_ptr() as *mut [u32; 5]);

            packed_mod_sub(
                a,
                b,
                res,
                KoalaBearParameters::PRIME,
                monty_sub::<KoalaBearParameters>,
            );
        }
        res
    }

    #[inline(always)]
    fn quintic_base_mul(lhs: [Self; 5], rhs: Self) -> [Self; 5] {
        let mut res = [Self::ZERO; 5];
        base_mul_packed(lhs, rhs, &mut res);
        res
    }
}

/// Trait for fields that support binomial extension of the form: `F[X]/(X^5 + X^2 - 1)`
pub trait QuinticExtendable: Field + QuinticExtendableAlgebra<Self> {
    const FROBENIUS_MATRIX: [[Self; 5]; 4];

    /// A generator for the extension field, expressed as a degree-`D` polynomial.
    ///
    /// This is an array of size `D`, where each entry is a base field element.
    const EXT_GENERATOR: [Self; 5];
}

pub trait QuinticExtendableAlgebra<F: Field>: Algebra<F> {
    /// Multiplication in the algebra extension ring `A<X> / (X^D - W)`.
    ///
    /// Some algebras may want to reimplement this with faster methods.
    fn quintic_mul(a: &[Self; 5], b: &[Self; 5], res: &mut [Self; 5]);

    /// Addition of elements in the algebra extension ring `A<X> / (X^D - W)`.
    ///
    /// As addition has no dependence on `W` so this is equivalent
    /// to an algorithm for adding arrays of elements of `A`.
    ///
    /// Some algebras may want to reimplement this with faster methods.
    #[must_use]
    fn quintic_add(a: &[Self; 5], b: &[Self; 5]) -> [Self; 5];

    /// Subtraction of elements in the algebra extension ring `A<X> / (X^D - W)`.
    ///
    /// As subtraction has no dependence on `W` so this is equivalent
    /// to an algorithm for subtracting arrays of elements of `A`.
    ///
    /// Some algebras may want to reimplement this with faster methods.
    #[must_use]
    fn quintic_sub(a: &[Self; 5], b: &[Self; 5]) -> [Self; 5];

    fn quintic_base_mul(lhs: [Self; 5], rhs: Self) -> [Self; 5];
}
