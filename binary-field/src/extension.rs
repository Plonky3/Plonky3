//! Extension-field structure between the byte-aligned levels of the Wiedemann tower.
//!
//! Each level `GF(2^b)` of the tower contains every level below it, so for byte-aligned levels
//! `Lower = GF(2^l)` and `Upper = GF(2^u)` with `l < u`, `Upper` is a degree-`u/l` extension of
//! `Lower`. The basis is the tower basis: writing an element of `Upper` as an unsigned integer,
//! basis coefficient `i` is the `l`-bit chunk at bit `i * l`, and basis element `i` is `1 << (i * l)`.
//!
//! On a little-endian target that chunking is exactly the in-memory byte layout, so the basis
//! coefficients of an element can be borrowed in place rather than copied out. The levels below
//! `GF(2^8)` are excluded because they do not fill their backing integer, and so their chunks
//! would not all be canonical representatives.

use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};
use core::{ptr, slice};

use p3_field::extension::HasFrobenius;
use p3_field::op_assign_macros::{impl_add_base_field, impl_mul_base_field, impl_sub_base_field};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, Powers,
    PrimeCharacteristicRing,
};

use crate::tower::TowerLevel;
use crate::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128};

/// Make one byte-aligned tower level an extension of a byte-aligned level below it.
///
/// The parameters are the upper level with its backing integer, then the lower level with its
/// backing integer.
macro_rules! binary_tower_extension {
    ($upper:ty, $upper_repr:ty, $lower:ty, $lower_repr:ty) => {
        impl From<$lower> for $upper {
            /// The lower level occupies the low `BITS` bits; the other coefficients vanish.
            #[inline]
            fn from(x: $lower) -> Self {
                Self::from_repr(<$upper_repr>::from(x.to_repr()))
            }
        }

        impl_add_base_field!($upper, $lower);
        impl_sub_base_field!($upper, $lower);
        impl_mul_base_field!($upper, $lower);

        impl Algebra<$lower> for $upper {}

        impl BasedVectorSpace<$lower> for $upper {
            const DIMENSION: usize = <$upper>::BITS / <$lower>::BITS;

            #[inline]
            fn as_basis_coefficients_slice(&self) -> &[$lower] {
                const {
                    assert!(
                        cfg!(target_endian = "little"),
                        "the tower basis coincides with the memory layout only on little-endian targets"
                    );
                    // The lower level fills its backing integer, so every chunk of `self` is
                    // already a canonical representative.
                    assert!(<$lower>::BITS == <$lower_repr>::BITS as usize);
                    assert!(
                        size_of::<$upper>()
                            == <$upper as BasedVectorSpace<$lower>>::DIMENSION
                                * size_of::<$lower>()
                    );
                    assert!(align_of::<$lower>() <= align_of::<$upper>());
                }

                // SAFETY: both levels are `repr(transparent)` over unsigned integers, so `self`
                // is `DIMENSION` contiguous, padding-free `$lower` values, aligned at least as
                // strictly as `$lower` requires; the assertions above check the size, alignment
                // and endianness this relies on, and that every chunk is canonical. The returned
                // slice borrows `self`, so it cannot outlive it.
                unsafe {
                    slice::from_raw_parts(
                        ptr::from_ref(self).cast::<$lower>(),
                        <$upper as BasedVectorSpace<$lower>>::DIMENSION,
                    )
                }
            }

            #[inline]
            fn from_basis_coefficients_fn<Fn: FnMut(usize) -> $lower>(mut f: Fn) -> Self {
                let mut repr: $upper_repr = 0;
                for i in 0..<$upper as BasedVectorSpace<$lower>>::DIMENSION {
                    repr |= <$upper_repr>::from(f(i).to_repr()) << (i * <$lower>::BITS);
                }
                Self::from_repr(repr)
            }

            #[inline]
            fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = $lower>>(
                iter: I,
            ) -> Option<Self> {
                let dimension = <$upper as BasedVectorSpace<$lower>>::DIMENSION;
                (iter.len() == dimension).then(|| {
                    let mut repr: $upper_repr = 0;
                    // Zipping against the dimension bounds the shift even if `len` lied.
                    for (i, c) in (0..dimension).zip(iter) {
                        repr |= <$upper_repr>::from(c.to_repr()) << (i * <$lower>::BITS);
                    }
                    Self::from_repr(repr)
                })
            }

            /// The tower basis coincides with the byte layout, so the whole buffer already *is*
            /// its coefficients: flattening is one copy, not one per element.
            ///
            /// The inverse direction is not symmetric, and `reconstitute_from_base` keeps the
            /// default that builds a fresh buffer: a `Vec<$lower>` is allocated at `$lower`'s
            /// alignment, which is weaker than `$upper`'s, and freeing an allocation under a
            /// `Layout` whose alignment differs from the one it was made with is undefined
            /// behaviour.
            #[inline]
            fn flatten_to_base(vec: Vec<Self>) -> Vec<$lower> {
                let d = <$upper as BasedVectorSpace<$lower>>::DIMENSION;

                // SAFETY: exactly the argument in `as_basis_coefficients_slice` above, extended
                // to the whole buffer: a `Vec<$upper>` is `vec.len()` contiguous, padding-free
                // `$upper` values, each of which is `d` contiguous `$lower` values in tower-basis
                // order. The const assertions there cover the endianness, size, alignment and
                // canonicity this relies on. The slice borrows `vec`, so it cannot outlive it.
                let flat: &[$lower] =
                    unsafe { slice::from_raw_parts(vec.as_ptr().cast::<$lower>(), vec.len() * d) };
                flat.to_vec()
            }
        }

        impl ExtensionField<$lower> for $upper {
            type ExtensionPacking = Self;

            #[inline]
            fn is_in_basefield(&self) -> bool {
                // The base field is exactly the elements whose coefficients above the
                // constant one vanish.
                (self.to_repr() >> <$lower>::BITS) == 0
            }

            #[inline]
            fn as_base(&self) -> Option<$lower> {
                <Self as ExtensionField<$lower>>::is_in_basefield(self)
                    .then(|| <$lower>::from_repr(self.to_repr() as $lower_repr))
            }
        }

        /// A tower level packs one element per vector, so every lane index is `0`.
        impl PackedFieldExtension<$lower, $upper> for $upper {
            #[inline]
            fn from_ext_fn(f: impl Fn(usize) -> Self) -> Self {
                f(0)
            }

            #[inline]
            fn from_ext_slice(slice: &[Self]) -> Self {
                assert_eq!(slice.len(), 1);
                slice[0]
            }

            #[inline]
            fn extract(&self, lane: usize) -> Self {
                assert_eq!(lane, 0, "lane index out of range");
                *self
            }

            #[inline]
            fn add_assign_lane(&mut self, lane: usize, value: Self) {
                assert_eq!(lane, 0, "lane index out of range");
                *self += value;
            }

            #[inline]
            fn packed_ext_powers(base: Self) -> Powers<Self> {
                base.powers()
            }
        }

        impl HasFrobenius<$lower> for $upper {
            /// `x ↦ x^n` for `n = 2^BITS` the order of the base field.
            #[inline]
            fn frobenius(&self) -> Self {
                // Every level of the tower is twice as wide as the one below, so a dimension of
                // `2` holds exactly when the base field is the level immediately below, and only
                // then is `$upper` the quadratic extension `X² + αX + 1` whose `split`, `join`
                // and `mul_alpha` are phrased in terms of. `DIMENSION` is a constant, so the
                // branch is settled at compile time.
                if <$upper as BasedVectorSpace<$lower>>::DIMENSION == 2 {
                    // The two roots of `X² + αX + 1` sum to `α`, so the automorphism fixing the
                    // level below sends `X` to `X + α`, and `a0 + a1·X` to `(a0 + α·a1) + a1·X`.
                    let (a0, a1) = self.split();
                    Self::join(a0 + a1.mul_alpha(), a1)
                } else {
                    self.exp_power_of_2(<$lower>::BITS)
                }
            }

            #[inline]
            fn repeated_frobenius(&self, count: usize) -> Self {
                // The Galois group is cyclic of order `DIMENSION`, so `count` reduces modulo it.
                let count = count % <$upper as BasedVectorSpace<$lower>>::DIMENSION;
                if <$upper as BasedVectorSpace<$lower>>::DIMENSION == 2 {
                    if count == 0 {
                        *self
                    } else {
                        <Self as HasFrobenius<$lower>>::frobenius(self)
                    }
                } else {
                    self.exp_power_of_2(<$lower>::BITS * count)
                }
            }

            #[inline]
            fn pseudo_inv(&self) -> Self {
                // Inversion through the norm to the level below is cheaper than the
                // `n^D - 2` exponentiation the contract is phrased in terms of.
                self.try_inverse().unwrap_or(Self::ZERO)
            }
        }
    };
}

binary_tower_extension!(BinaryField16, u16, BinaryField8, u8);
binary_tower_extension!(BinaryField32, u32, BinaryField8, u8);
binary_tower_extension!(BinaryField64, u64, BinaryField8, u8);
binary_tower_extension!(BinaryField128, u128, BinaryField8, u8);
binary_tower_extension!(BinaryField32, u32, BinaryField16, u16);
binary_tower_extension!(BinaryField64, u64, BinaryField16, u16);
binary_tower_extension!(BinaryField128, u128, BinaryField16, u16);
binary_tower_extension!(BinaryField64, u64, BinaryField32, u32);
binary_tower_extension!(BinaryField128, u128, BinaryField32, u32);
binary_tower_extension!(BinaryField128, u128, BinaryField64, u64);

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::extension::HasFrobenius;
    use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use crate::tower::TowerLevel;
    use crate::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128};

    /// A fixed nonzero element of `GF(2^128)` used across the tests.
    const A128: u128 = 0x1234_5678_9abc_def0_0fed_cba9_8765_4321;

    /// Check that the basis coefficients over `$lower` really are coordinates in the basis
    /// returned by `ith_basis_element`, and that both reassembly routes recover the element.
    macro_rules! assert_basis_round_trip {
        ($upper:ty, $lower:ty, $val:expr) => {{
            let a = <$upper>::from_repr($val);
            let dim = <$upper as BasedVectorSpace<$lower>>::DIMENSION;
            let coeffs = BasedVectorSpace::<$lower>::as_basis_coefficients_slice(&a);
            assert_eq!(coeffs.len(), dim);

            // `a = Σ coeffs[i] · basis[i]` is the defining property of a coordinate map.
            let recomposed = (0..dim)
                .map(|i| {
                    <$upper as BasedVectorSpace<$lower>>::ith_basis_element(i).unwrap() * coeffs[i]
                })
                .sum::<$upper>();
            assert_eq!(recomposed, a);

            assert_eq!(
                <$upper as BasedVectorSpace<$lower>>::from_basis_coefficients_slice(coeffs),
                Some(a)
            );
            assert_eq!(
                <$upper as BasedVectorSpace<$lower>>::from_basis_coefficients_fn(|i| coeffs[i]),
                a
            );
        }};
    }

    /// Check that the mixed-type operators agree with embedding the scalar first.
    macro_rules! assert_mixed_ops_match_embedding {
        ($upper:ty, $lower:ty, $val:expr, $scalar:expr) => {{
            let a = <$upper>::from_repr($val);
            let s = <$lower>::from_repr($scalar);
            let s_up = <$upper>::from(s);

            assert_eq!(a + s, a + s_up);
            assert_eq!(a - s, a - s_up);
            assert_eq!(a * s, a * s_up);

            let mut acc = a;
            acc += s;
            assert_eq!(acc, a + s_up);
            let mut acc = a;
            acc -= s;
            assert_eq!(acc, a - s_up);
            let mut acc = a;
            acc *= s;
            assert_eq!(acc, a * s_up);
        }};
    }

    /// Check that Frobenius over `$lower` fixes `$lower` pointwise, has order `DIMENSION`, and
    /// that `repeated_frobenius` iterates it.
    macro_rules! assert_frobenius_over {
        ($upper:ty, $lower:ty, $val:expr) => {{
            let dim = <$upper as BasedVectorSpace<$lower>>::DIMENSION;

            // Frobenius fixes the base field: `x ↦ x^|lower|` is the identity on `lower`.
            for i in 0..=u8::MAX {
                let embedded = <$upper>::from(<$lower>::from_repr(i.into()));
                assert_eq!(
                    HasFrobenius::<$lower>::frobenius(&embedded),
                    embedded,
                    "Frobenius must fix the embedded base field"
                );
            }

            // It is not the identity away from the base field: the generator of `$upper` over
            // `$lower` is the first basis element beyond the base field.
            let x = <$upper as BasedVectorSpace<$lower>>::ith_basis_element(1).unwrap();
            assert_ne!(
                HasFrobenius::<$lower>::frobenius(&x),
                x,
                "Frobenius must move elements outside the base field"
            );

            // `repeated_frobenius(count)` iterates `frobenius` and has order `DIMENSION`.
            let a = <$upper>::from_repr($val);
            let mut acc = a;
            for count in 0..(2 * dim + 3) {
                assert_eq!(HasFrobenius::<$lower>::repeated_frobenius(&a, count), acc);
                acc = HasFrobenius::<$lower>::frobenius(&acc);
            }

            // `pseudo_inv` is inversion, extended by `0 ↦ 0`.
            assert_eq!(
                HasFrobenius::<$lower>::pseudo_inv(&<$upper>::ZERO),
                <$upper>::ZERO
            );
            assert_eq!(a * HasFrobenius::<$lower>::pseudo_inv(&a), <$upper>::ONE);
        }};
    }

    #[test]
    fn gf2_128_over_gf2_8_basis_coefficients_are_bytes() {
        let a = BinaryField128::from_repr(0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100);
        let coeffs = BasedVectorSpace::<BinaryField8>::as_basis_coefficients_slice(&a);
        assert_eq!(coeffs.len(), 16);
        for (i, c) in coeffs.iter().enumerate() {
            assert_eq!(*c, BinaryField8::from_repr(i as u8));
        }
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField8>>::from_basis_coefficients_slice(
                coeffs
            ),
            Some(a)
        );
    }

    #[test]
    fn mixed_mul_matches_embedding() {
        let a = BinaryField128::from_repr(A128);
        let s = BinaryField8::from_repr(0xa7);
        assert_eq!(a * s, a * BinaryField128::from(s));
    }

    #[test]
    fn dimension_is_the_ratio_of_bit_widths() {
        assert_eq!(
            <BinaryField16 as BasedVectorSpace<BinaryField8>>::DIMENSION,
            2
        );
        assert_eq!(
            <BinaryField32 as BasedVectorSpace<BinaryField8>>::DIMENSION,
            4
        );
        assert_eq!(
            <BinaryField64 as BasedVectorSpace<BinaryField8>>::DIMENSION,
            8
        );
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField8>>::DIMENSION,
            16
        );
        assert_eq!(
            <BinaryField32 as BasedVectorSpace<BinaryField16>>::DIMENSION,
            2
        );
        assert_eq!(
            <BinaryField64 as BasedVectorSpace<BinaryField16>>::DIMENSION,
            4
        );
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField16>>::DIMENSION,
            8
        );
        assert_eq!(
            <BinaryField64 as BasedVectorSpace<BinaryField32>>::DIMENSION,
            2
        );
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField32>>::DIMENSION,
            4
        );
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField64>>::DIMENSION,
            2
        );
    }

    #[test]
    fn basis_coefficients_round_trip_on_every_pair() {
        assert_basis_round_trip!(BinaryField16, BinaryField8, 0xfedc);
        assert_basis_round_trip!(BinaryField32, BinaryField8, 0xfedc_ba98);
        assert_basis_round_trip!(BinaryField64, BinaryField8, 0xfedc_ba98_7654_3210);
        assert_basis_round_trip!(BinaryField128, BinaryField8, A128);
        assert_basis_round_trip!(BinaryField32, BinaryField16, 0xfedc_ba98);
        assert_basis_round_trip!(BinaryField64, BinaryField16, 0xfedc_ba98_7654_3210);
        assert_basis_round_trip!(BinaryField128, BinaryField16, A128);
        assert_basis_round_trip!(BinaryField64, BinaryField32, 0xfedc_ba98_7654_3210);
        assert_basis_round_trip!(BinaryField128, BinaryField32, A128);
        assert_basis_round_trip!(BinaryField128, BinaryField64, A128);
    }

    #[test]
    fn mixed_ops_match_embedding_on_every_pair() {
        assert_mixed_ops_match_embedding!(BinaryField16, BinaryField8, 0xfedc, 0xa7);
        assert_mixed_ops_match_embedding!(BinaryField32, BinaryField8, 0xfedc_ba98, 0xa7);
        assert_mixed_ops_match_embedding!(BinaryField64, BinaryField8, 0xfedc_ba98_7654_3210, 0xa7);
        assert_mixed_ops_match_embedding!(BinaryField128, BinaryField8, A128, 0xa7);
        assert_mixed_ops_match_embedding!(BinaryField32, BinaryField16, 0xfedc_ba98, 0xa73b);
        assert_mixed_ops_match_embedding!(
            BinaryField64,
            BinaryField16,
            0xfedc_ba98_7654_3210,
            0xa73b
        );
        assert_mixed_ops_match_embedding!(BinaryField128, BinaryField16, A128, 0xa73b);
        assert_mixed_ops_match_embedding!(
            BinaryField64,
            BinaryField32,
            0xfedc_ba98_7654_3210,
            0xa73b_c91d
        );
        assert_mixed_ops_match_embedding!(BinaryField128, BinaryField32, A128, 0xa73b_c91d);
        assert_mixed_ops_match_embedding!(
            BinaryField128,
            BinaryField64,
            A128,
            0xa73b_c91d_5566_7788
        );
    }

    #[test]
    fn frobenius_fixes_exactly_the_lower_level_on_every_pair() {
        assert_frobenius_over!(BinaryField16, BinaryField8, 0xfedc);
        assert_frobenius_over!(BinaryField32, BinaryField8, 0xfedc_ba98);
        assert_frobenius_over!(BinaryField64, BinaryField8, 0xfedc_ba98_7654_3210);
        assert_frobenius_over!(BinaryField128, BinaryField8, A128);
        assert_frobenius_over!(BinaryField32, BinaryField16, 0xfedc_ba98);
        assert_frobenius_over!(BinaryField64, BinaryField16, 0xfedc_ba98_7654_3210);
        assert_frobenius_over!(BinaryField128, BinaryField16, A128);
        assert_frobenius_over!(BinaryField64, BinaryField32, 0xfedc_ba98_7654_3210);
        assert_frobenius_over!(BinaryField128, BinaryField32, A128);
        assert_frobenius_over!(BinaryField128, BinaryField64, A128);
    }

    /// The order of the base field, not its bit width, is the Frobenius exponent: over
    /// `GF(2^8)` the map is `x ↦ x^256`, which fixes all 256 elements of `GF(2^8)`.
    #[test]
    fn frobenius_exponent_is_the_order_of_the_base_field() {
        let a = BinaryField128::from_repr(A128);
        assert_eq!(
            HasFrobenius::<BinaryField8>::frobenius(&a),
            a.exp_power_of_2(8)
        );
        assert_eq!(
            HasFrobenius::<BinaryField64>::frobenius(&a),
            a.exp_power_of_2(64)
        );
    }

    #[test]
    fn is_in_basefield_detects_the_embedded_subfield() {
        let low = BinaryField128::from(BinaryField8::from_repr(0xa7));
        assert!(ExtensionField::<BinaryField8>::is_in_basefield(&low));
        assert_eq!(
            ExtensionField::<BinaryField8>::as_base(&low),
            Some(BinaryField8::from_repr(0xa7))
        );

        // Sixteen bits wide: inside `GF(2^16)`, outside `GF(2^8)`.
        let mid = BinaryField128::from_repr(0xa73b);
        assert!(!ExtensionField::<BinaryField8>::is_in_basefield(&mid));
        assert_eq!(ExtensionField::<BinaryField8>::as_base(&mid), None);
        assert!(ExtensionField::<BinaryField16>::is_in_basefield(&mid));
        assert_eq!(
            ExtensionField::<BinaryField16>::as_base(&mid),
            Some(BinaryField16::from_repr(0xa73b))
        );
    }

    #[test]
    fn from_basis_coefficients_iter_checks_the_length() {
        let coeffs = [BinaryField64::ONE, BinaryField64::ZERO];
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField64>>::from_basis_coefficients_iter(
                coeffs.into_iter()
            ),
            Some(BinaryField128::ONE)
        );
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField64>>::from_basis_coefficients_iter(
                coeffs[..1].iter().copied()
            ),
            None
        );
        assert_eq!(
            <BinaryField128 as BasedVectorSpace<BinaryField64>>::from_basis_coefficients_slice(&[]),
            None
        );
    }

    /// Flattening concatenates each element's coefficient bytes in memory order, and
    /// reconstituting inverts it.
    #[test]
    fn flatten_and_reconstitute_round_trip() {
        let elems: Vec<BinaryField128> = (0..64u128)
            .map(|i| {
                BinaryField128::from_repr(i.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834))
            })
            .collect();

        // Every coefficient position must take a nonzero value somewhere in the fixture,
        // or a shift that lands in the wrong position could OR in zeros and still round-trip.
        for pos in 0..16 {
            assert!(
                elems.iter().any(|e| {
                    BasedVectorSpace::<BinaryField8>::as_basis_coefficients_slice(e)[pos]
                        != BinaryField8::ZERO
                }),
                "coefficient position {pos} is never nonzero"
            );
        }

        let flat =
            <BinaryField128 as BasedVectorSpace<BinaryField8>>::flatten_to_base(elems.clone());
        assert_eq!(flat.len(), elems.len() * 16);
        for (i, e) in elems.iter().enumerate() {
            assert_eq!(&flat[i * 16..(i + 1) * 16], e.as_basis_coefficients_slice());
        }

        let back = <BinaryField128 as BasedVectorSpace<BinaryField8>>::reconstitute_from_base(flat);
        assert_eq!(back, elems);
    }

    /// Every byte of every `GF(2^8)` value survives a pack and an unpack.
    #[test]
    fn reconstitute_is_exhaustive_over_the_byte_level() {
        let all: Vec<BinaryField8> = (0..=u8::MAX).map(BinaryField8::from_repr).collect();
        let packed =
            <BinaryField16 as BasedVectorSpace<BinaryField8>>::reconstitute_from_base(all.clone());
        assert_eq!(packed.len(), 128);
        let flat = <BinaryField16 as BasedVectorSpace<BinaryField8>>::flatten_to_base(packed);
        assert_eq!(flat, all);
    }

    /// Check the conjugation `frobenius` takes over the level immediately below against the
    /// repeated squaring that defines it, for one element.
    macro_rules! assert_frobenius_is_repeated_squaring {
        ($upper:ty, $lower:ty, $val:expr) => {{
            let a = <$upper>::from_repr($val);
            let dim = <$upper as BasedVectorSpace<$lower>>::DIMENSION;
            assert_eq!(dim, 2, "the specialisation only covers adjacent levels");

            prop_assert_eq!(
                HasFrobenius::<$lower>::frobenius(&a),
                a.exp_power_of_2(<$lower>::BITS)
            );
            for count in 0..(2 * dim + 3) {
                prop_assert_eq!(
                    HasFrobenius::<$lower>::repeated_frobenius(&a, count),
                    a.exp_power_of_2(<$lower>::BITS * (count % dim))
                );
            }
        }};
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// Frobenius over the level immediately below is conjugation rather than a chain of
        /// squarings; the two must agree at each of the four levels that take that route.
        #[test]
        fn frobenius_over_the_adjacent_level_is_repeated_squaring(
            a in any::<u128>(),
        ) {
            assert_frobenius_is_repeated_squaring!(BinaryField16, BinaryField8, a as u16);
            assert_frobenius_is_repeated_squaring!(BinaryField32, BinaryField16, a as u32);
            assert_frobenius_is_repeated_squaring!(BinaryField64, BinaryField32, a as u64);
            assert_frobenius_is_repeated_squaring!(BinaryField128, BinaryField64, a);
        }

        /// The levels whose base field is further down keep the repeated squaring, so the
        /// specialisation must not have leaked into them.
        #[test]
        fn frobenius_over_a_deeper_level_is_repeated_squaring(a in any::<u128>()) {
            macro_rules! assert_unspecialised {
                ($upper:ty, $lower:ty, $val:expr) => {{
                    let x = <$upper>::from_repr($val);
                    prop_assert_eq!(
                        HasFrobenius::<$lower>::frobenius(&x),
                        x.exp_power_of_2(<$lower>::BITS)
                    );
                }};
            }

            assert_unspecialised!(BinaryField32, BinaryField8, a as u32);
            assert_unspecialised!(BinaryField64, BinaryField8, a as u64);
            assert_unspecialised!(BinaryField128, BinaryField8, a);
            assert_unspecialised!(BinaryField64, BinaryField16, a as u64);
            assert_unspecialised!(BinaryField128, BinaryField16, a);
            assert_unspecialised!(BinaryField128, BinaryField32, a);
        }
    }

    /// The embedding is multiplicative and additive, i.e. a ring homomorphism.
    #[test]
    fn embedding_is_a_ring_homomorphism() {
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                let (x, y) = (BinaryField8::from_repr(a), BinaryField8::from_repr(b));
                assert_eq!(
                    BinaryField128::from(x * y),
                    BinaryField128::from(x) * BinaryField128::from(y)
                );
                assert_eq!(
                    BinaryField128::from(x + y),
                    BinaryField128::from(x) + BinaryField128::from(y)
                );
            }
        }
        assert_eq!(BinaryField128::from(BinaryField8::ONE), BinaryField128::ONE);
    }
}
