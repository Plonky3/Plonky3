//! Extension products whose coefficient-field reductions can be postponed across sums.

use core::ops::{Add, AddAssign};

use crate::clmul::{raw_product_64, reduce_64};
use crate::{Poly64, Poly192};

/// Three polynomial coordinates, folded modulo `y^3 + y + 1` but unreduced over `GF(2^64)`.
///
/// Products and partial sums can be added before [`Self::reduce`] performs three reductions.
#[derive(Clone, Copy, Debug, Default)]
#[must_use]
pub struct Poly192Unreduced(
    /// Unreduced coordinates in the basis `1, y, y^2`.
    pub(crate) [u128; 3],
);

impl Poly192Unreduced {
    /// The empty sum of products.
    pub const ZERO: Self = Self([0; 3]);

    /// Reduce the complete sum to an extension-field element.
    #[inline]
    pub fn reduce(self) -> Poly192 {
        Poly192::new(self.0.map(|coordinate| Poly64::new(reduce_64(coordinate))))
    }
}

impl Add for Poly192Unreduced {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }
}

impl AddAssign for Poly192Unreduced {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Poly192 {
    /// Multiply two extension elements, postponing the coefficient-field reductions.
    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> Poly192Unreduced {
        let [a0, a1, a2] = *self.limbs();
        let [b0, b1, b2] = *rhs.limbs();
        let c0 = raw_product_64(a0, b0);
        let c1 = raw_product_64(a1, b1);
        let c2 = raw_product_64(a2, b2);
        let d01 = raw_product_64(a0 ^ a1, b0 ^ b1);
        let d02 = raw_product_64(a0 ^ a2, b0 ^ b2);
        let d12 = raw_product_64(a1 ^ a2, b1 ^ b2);
        Poly192Unreduced([c0 ^ c1 ^ c2 ^ d12, c0 ^ d01 ^ d12, c0 ^ c1 ^ d02])
    }

    /// Multiply by a coefficient-field element without reducing the three coordinates.
    #[inline]
    pub fn mul_base_unreduced(self, rhs: Poly64) -> Poly192Unreduced {
        Poly192Unreduced(
            self.limbs()
                .map(|coordinate| raw_product_64(coordinate, rhs.to_bits())),
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn deferred_products_preserve_sums(a in any::<[[u64; 3]; 9]>(), b in any::<[[u64; 3]; 9]>(), k in any::<[u64; 9]>()) {
            let a = a.map(|x| Poly192::new(x.map(Poly64::new)));
            let b = b.map(|x| Poly192::new(x.map(Poly64::new)));
            let k = k.map(Poly64::new);
            let mut sum = Poly192Unreduced::ZERO;
            let mut expected = Poly192::ZERO;
            for i in 0..a.len() {
                sum += a[i].mul_unreduced(b[i]);
                sum += a[i].mul_base_unreduced(k[i]);
                expected += a[i] * b[i] + a[i] * k[i];
            }
            prop_assert_eq!(sum.reduce(), expected);
            prop_assert_eq!((sum + sum).reduce(), Poly192::ZERO);
        }
    }
}
