//! The reduction and the split multiplier, written once over an abstract lane backend.
//!
//! One backend is whichever register the target provides.
//!
//! The other is a scalar model compiled under `cfg(test)`, so every leg checks the algebra.
//! Otherwise only a target with the wide carryless multiply would exercise any of it.

use crate::clmul::TAIL_128;

/// Selects the low quadword of both operands.
///
/// Bit 0 picks the half of the first argument, bit 4 the half of the second.
pub(crate) const LOW_BY_LOW: i32 = 0x00;

/// Selects the high quadword of both operands.
pub(crate) const HIGH_BY_HIGH: i32 = 0x11;

/// Selects the high quadword of the first operand and the low quadword of the second.
pub(crate) const HIGH_BY_LOW: i32 = 0x01;

/// Selects the low quadword of the first operand and the high quadword of the second.
pub(crate) const LOW_BY_HIGH: i32 = 0x10;

/// The lane operations the algebra below is written against.
///
/// Every method acts on each 128-bit lane independently.
pub(crate) trait Lanes: Copy {
    /// All lanes zero.
    fn zero() -> Self;

    /// The same field element in every lane.
    fn broadcast(value: u128) -> Self;

    /// The modulus tail in the low quadword of every lane.
    #[inline(always)]
    fn tail() -> Self {
        Self::broadcast(TAIL_128)
    }

    /// Bitwise exclusive or.
    fn xor(self, other: Self) -> Self;

    /// The low quadword of each operand, paired within each lane.
    fn unpack_low_64(self, other: Self) -> Self;

    /// The carryless product of one quadword of each operand, in every lane.
    fn clmul<const IMM: i32>(self, other: Self) -> Self;
}

/// One Horner step of the reduction, in every lane at once.
///
/// Splitting the second argument into its 64-bit halves rewrites the part that overflows:
///
/// ```text
///     T  = x^7 + x^2 + x + 1                    the modulus tail, since x^128 = T
///     t1 = t1_lo + t1_hi x^64
///
///     t1 x^64 = t1_lo x^64 + t1_hi T
/// ```
///
/// Both rewritten terms stay below `x^128`, so one step is exact here.
///
/// One step lowers the second argument's weight by `x^64`.
///
/// A product spanning more than three 64-bit limbs therefore needs one step per extra limb.
#[inline(always)]
pub(crate) fn fold_shifted<L: Lanes>(t0: L, t1: L) -> L {
    // Interleaving against zero moves the low quadword up, scaling by x^64.
    let raised = L::zero().unpack_low_64(t1);

    // The high quadword times the tail is what the modulus rewrites.
    let folded = t1.clmul::<HIGH_BY_LOW>(L::tail());

    t0.xor(raised.xor(folded))
}

/// A multiplier held fixed across a slice, in every lane, beside its shifted companion.
///
/// # Algorithm
///
/// The field is `GF(2)[x]` modulo `p = x^128 + x^7 + x^2 + x + 1`.
///
/// So `x^128 = T`, writing `T = x^7 + x^2 + x + 1` for the modulus tail.
///
/// Reduction modulo `p` is a ring homomorphism.
///
/// A shift may therefore move off the varying operand and onto the fixed one:
///
/// ```text
///     v   = v_0 + v_1 x^64                deg v_0 < 64,  deg v_1 < 64
///     u   = t x^64 mod p                  one companion per multiplier
///
///     t v = t v_0 + t v_1 x^64
///         = t v_0 + (t x^64) v_1
///         = t v_0 + u v_1                 (mod p)
/// ```
///
/// Both surviving products are `128 x 64` rather than `128 x 128`.
///
/// Writing `t = t_0 + t_1 x^64` and `u = u_0 + u_1 x^64`, that is two pieces:
///
/// ```text
///     low  = t_0 v_0 + u_0 v_1            degree <= 126
///     high = t_1 v_0 + u_1 v_1            degree <= 126, weighted by x^64
///
///     t v  = low + high x^64              degree <= 190, so three 64-bit limbs
/// ```
///
/// A four-limb product needs two Horner steps in `x^64` to come back under `x^128`.
///
/// Three limbs need one.
///
/// That step is exact because the piece it shifts is only 128 bits wide:
///
/// ```text
///     high = h_0 + h_1 x^64               both parts below x^64
///
///     high x^64 = h_0 x^64 + h_1 x^128    degree <= 127
///               = h_0 x^64 + h_1 T        degree <= 70
/// ```
///
/// Neither term reaches `x^128`, so nothing spills over the top a second time.
#[derive(Clone, Copy)]
pub(crate) struct SplitScalar<L> {
    /// The multiplier, repeated in every lane.
    t: L,
    /// The multiplier shifted by `x^64` and reduced, repeated in every lane.
    t_x64: L,
}

impl<L: Lanes> SplitScalar<L> {
    /// Broadcast a multiplier and derive its companion.
    #[inline(always)]
    pub(crate) fn new(scalar: u128) -> Self {
        let t = L::broadcast(scalar);

        // The companion is one Horner step with nothing underneath the shifted part.
        let t_x64 = fold_shifted(L::zero(), t);

        Self { t, t_x64 }
    }

    /// The reduced product of the multiplier with every lane of the argument.
    #[inline(always)]
    pub(crate) fn apply(self, v: L) -> L {
        // Limbs 0 and 1: the half products that carry no further shift.
        let low = v
            .clmul::<LOW_BY_LOW>(self.t)
            .xor(v.clmul::<HIGH_BY_LOW>(self.t_x64));

        // Limbs 1 and 2: the half products weighted by `x^64`.
        let high = v
            .clmul::<LOW_BY_HIGH>(self.t)
            .xor(v.clmul::<HIGH_BY_HIGH>(self.t_x64));

        // Three limbs, so one Horner step in `x^64` finishes the reduction.
        fold_shifted(low, high)
    }
}

/// A scalar stand-in for a register, so the algebra above is checked on every target.
///
/// Without it only a build with the wide carryless multiply would exercise any of the split.
#[cfg(test)]
pub(crate) mod model {
    use super::Lanes;

    /// One 128-bit field element per lane, held in plain integers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Model<const LANES: usize>(pub(crate) [u128; LANES]);

    /// The carryless product of two quadwords, bit-serial from the low bit up.
    ///
    /// The result spans 127 bits, so it never leaves the lane.
    fn carryless_64(x: u64, y: u64) -> u128 {
        let mut acc = 0u128;
        for i in 0..64 {
            if (y >> i) & 1 == 1 {
                acc ^= (x as u128) << i;
            }
        }
        acc
    }

    impl<const LANES: usize> Lanes for Model<LANES> {
        fn zero() -> Self {
            Self([0; LANES])
        }

        fn broadcast(value: u128) -> Self {
            Self([value; LANES])
        }

        fn xor(self, other: Self) -> Self {
            Self(core::array::from_fn(|i| self.0[i] ^ other.0[i]))
        }

        fn unpack_low_64(self, other: Self) -> Self {
            // The low quadword of the first operand, then the low quadword of the second.
            Self(core::array::from_fn(|i| {
                (self.0[i] & u64::MAX as u128) | (other.0[i] << 64)
            }))
        }

        fn clmul<const IMM: i32>(self, other: Self) -> Self {
            Self(core::array::from_fn(|i| {
                // Bit 0 picks the half of the first argument, bit 4 the half of the second.
                let x = if IMM & 0x01 == 0 {
                    self.0[i] as u64
                } else {
                    (self.0[i] >> 64) as u64
                };
                let y = if IMM & 0x10 == 0 {
                    other.0[i] as u64
                } else {
                    (other.0[i] >> 64) as u64
                };
                carryless_64(x, y)
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::model::Model;
    use super::{Lanes, SplitScalar, fold_shifted};
    use crate::clmul;

    /// Lanes in the model, enough that a lane-crossing operation shows up as a mismatch.
    const LANES: usize = 2;

    /// The multipliers a random search is unlikely to reach.
    ///
    /// Each one drives the split of the multiplier, or the fold of the modulus, to an extreme:
    ///
    /// ```text
    ///     0                 the whole product vanishes
    ///     1                 the identity
    ///     all ones          every coefficient of every half product is live
    ///     x^127             the highest degree, so the companion spills furthest
    ///     x^64              the companion is the multiplier's own shift, already reduced
    ///     lower half only   the companion is the only source of upper limbs
    ///     0x87              the modulus tail itself
    /// ```
    const CORNERS: [u128; 7] = [0, 1, u128::MAX, 1 << 127, 1 << 64, (1u128 << 64) - 1, 0x87];

    /// The lanes of the split product against the scalar backend, element by element.
    fn split_agrees(scalar: u128, values: [u128; LANES]) -> Result<(), TestCaseError> {
        let got = SplitScalar::new(scalar).apply(Model::<LANES>(values));
        let want = core::array::from_fn(|i| clmul::poly_mul_128(scalar, values[i]));
        prop_assert_eq!(got, Model(want));
        Ok(())
    }

    #[test]
    fn the_split_product_agrees_with_the_scalar_backend_at_the_corners() {
        // Every corner as the multiplier, against every corner as the value.
        for scalar in CORNERS {
            for value in CORNERS {
                for other in CORNERS {
                    split_agrees(scalar, [value, other])
                        .unwrap_or_else(|e| panic!("scalar {scalar:#x}: {e}"));
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn the_split_product_agrees_with_the_scalar_backend(
            scalar in any::<u128>(),
            values in any::<[u128; LANES]>(),
        ) {
            split_agrees(scalar, values)?;
        }

        /// The companion is the multiplier scaled by `x^64`, which fixes the four immediates.
        #[test]
        fn the_companion_is_the_shifted_multiplier(scalar in any::<u128>()) {
            let companion = fold_shifted(Model::<LANES>::zero(), Model::broadcast(scalar));
            let want = clmul::poly_mul_128(scalar, 1 << 64);
            prop_assert_eq!(companion, Model::broadcast(want));
        }
    }
}
