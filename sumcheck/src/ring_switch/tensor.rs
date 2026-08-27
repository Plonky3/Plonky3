//! The tensor algebra `A = EF ⊗_F EF`.
//!
//! Fix the `F`-basis `(β_0, ..., β_{D-1})` of `EF` given by [`BasedVectorSpace`], with
//! `β_0 = 1`. An element of `A` is held as a `D × D` matrix `m` over `F`, where `m[u][v]` is
//! the coefficient of `β_u ⊗ β_v`. The matrix admits two readings as an `EF`: column `v` is
//! `Σ_u m[u][v] · β_u`, and row `u` is `Σ_v m[u][v] · β_v`.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::AddAssign;

use p3_field::{BasedVectorSpace, ExtensionField, Field};
use serde::{Deserialize, Serialize};

use super::RingSwitchError;

/// An element of `EF ⊗_F EF`, held as a `DIMENSION × DIMENSION` matrix over `F`.
///
/// See the module documentation for the basis convention this relies on.
///
/// Both readings index the matrix as `DIMENSION × DIMENSION`, so every value of this type
/// carries exactly `DIMENSION²` coefficients. The only route from untrusted bytes to a value
/// is [`TryFrom<Vec<F>>`], which the [`Deserialize`] impl goes through, so a deserialized
/// element is already the right shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    into = "Vec<F>",
    try_from = "Vec<F>",
    bound(
        serialize = "F: Field, EF: ExtensionField<F>",
        deserialize = "F: Field, EF: ExtensionField<F>"
    )
)]
pub struct TensorAlgebra<F, EF> {
    /// Row-major `DIMENSION × DIMENSION`: `coeffs[u * DIMENSION + v]` is the coefficient of
    /// `β_u ⊗ β_v`.
    coeffs: Vec<F>,
    _marker: PhantomData<EF>,
}

impl<F: Field, EF: ExtensionField<F>> TensorAlgebra<F, EF> {
    /// The dimension of `EF` as a vector space over `F`.
    pub const DIMENSION: usize = <EF as BasedVectorSpace<F>>::DIMENSION;

    /// The additive identity: the all-zero matrix.
    pub fn zero() -> Self {
        Self {
            coeffs: vec![F::ZERO; Self::DIMENSION * Self::DIMENSION],
            _marker: PhantomData,
        }
    }

    /// The multiplicative identity `1 ⊗ 1`.
    ///
    /// Since `β_0 = 1`, `1 ⊗ 1` has coefficient `1` at `(u, v) = (0, 0)` and `0` everywhere
    /// else.
    ///
    /// # Panics
    /// Debug builds check the `β_0 = 1` convention this placement relies on.
    pub fn one() -> Self {
        debug_assert!(
            <EF as BasedVectorSpace<F>>::ith_basis_element(0) == Some(EF::ONE),
            "the tensor algebra places 1 ⊗ 1 at (0, 0), which needs the basis convention β_0 = 1"
        );
        let mut t = Self::zero();
        t.coeffs[0] = F::ONE;
        t
    }

    /// `a ⊗ b`.
    pub fn exterior_product(a: EF, b: EF) -> Self {
        let d = Self::DIMENSION;
        let a_coeffs = a.as_basis_coefficients_slice();
        let b_coeffs = b.as_basis_coefficients_slice();
        let mut coeffs = vec![F::ZERO; d * d];
        for u in 0..d {
            for v in 0..d {
                coeffs[u * d + v] = a_coeffs[u] * b_coeffs[v];
            }
        }
        Self {
            coeffs,
            _marker: PhantomData,
        }
    }

    /// Whether the element carries the `DIMENSION²` coefficients both readings index.
    ///
    /// Every constructor here produces a well-formed element; this is the check a consumer
    /// applies to an element it did not build itself.
    pub const fn is_well_formed(&self) -> bool {
        self.coeffs.len() == Self::DIMENSION * Self::DIMENSION
    }

    /// The `F`-coefficients in row-major order: entry `u * DIMENSION + v` is the coefficient
    /// of `β_u ⊗ β_v`.
    ///
    /// These are what a transcript should absorb, since they are what crosses the wire; the
    /// two readings below are derived from them.
    pub fn coefficients(&self) -> &[F] {
        &self.coeffs
    }

    /// Adds `delta` into the coefficient at `index`, to exercise the rejection paths that a
    /// tampered element must trigger.
    #[cfg(test)]
    pub(crate) fn perturb_for_test(&mut self, index: usize, delta: F) {
        self.coeffs[index] += delta;
    }

    /// Drops all but the first `len` coefficients, to exercise the malformed-element path.
    #[cfg(test)]
    pub(crate) fn truncate_for_test(&mut self, len: usize) {
        self.coeffs.truncate(len);
    }

    /// The columns, each read as an `EF`: column `v` is `Σ_u m[u][v] · β_u`.
    pub fn columns(&self) -> Vec<EF> {
        let d = Self::DIMENSION;
        (0..d)
            .map(|v| EF::from_basis_coefficients_fn(|u| self.coeffs[u * d + v]))
            .collect()
    }

    /// The rows, each read as an `EF`: row `u` is `Σ_v m[u][v] · β_v`.
    pub fn rows(&self) -> Vec<EF> {
        let d = Self::DIMENSION;
        (0..d)
            .map(|u| EF::from_basis_coefficients_fn(|v| self.coeffs[u * d + v]))
            .collect()
    }

    /// `φ0(a) · self`: scales each column, read as an `EF`, by `a`.
    pub fn scale_columns(&mut self, a: EF) {
        let d = Self::DIMENSION;
        for (v, col) in self.columns().into_iter().enumerate() {
            let scaled = col * a;
            for (u, coeff) in scaled.as_basis_coefficients_slice().iter().enumerate() {
                self.coeffs[u * d + v] = *coeff;
            }
        }
    }

    /// `φ1(a) · self`: scales each row, read as an `EF`, by `a`.
    pub fn scale_rows(&mut self, a: EF) {
        let d = Self::DIMENSION;
        for (u, row) in self.rows().into_iter().enumerate() {
            let scaled = (row * a).as_basis_coefficients_slice().to_vec();
            self.coeffs[u * d..(u + 1) * d].copy_from_slice(&scaled);
        }
    }
}

/// The checked constructor: `coeffs` must be the `DIMENSION²` row-major coefficients.
impl<F: Field, EF: ExtensionField<F>> TryFrom<Vec<F>> for TensorAlgebra<F, EF> {
    type Error = RingSwitchError;

    fn try_from(coeffs: Vec<F>) -> Result<Self, Self::Error> {
        let expected = Self::DIMENSION * Self::DIMENSION;
        if coeffs.len() == expected {
            Ok(Self {
                coeffs,
                _marker: PhantomData,
            })
        } else {
            Err(RingSwitchError::MalformedTensor {
                expected,
                actual: coeffs.len(),
            })
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> From<TensorAlgebra<F, EF>> for Vec<F> {
    fn from(tensor: TensorAlgebra<F, EF>) -> Self {
        tensor.coeffs
    }
}

impl<F: Field, EF> AddAssign for TensorAlgebra<F, EF> {
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(
            self.coeffs.len(),
            rhs.coeffs.len(),
            "adding tensor elements of different shapes would truncate to the shorter one"
        );
        for (c, r) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *c += r;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::TowerLevel;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    use super::TensorAlgebra;
    use crate::ring_switch::RingSwitchError;
    use crate::ring_switch::test_util::{EF, F};

    /// `a ⊗ b` read by columns is `a` scaled by each coefficient of `b`, and by rows is `b`
    /// scaled by each coefficient of `a`.
    #[test]
    fn exterior_product_readings_are_the_scaled_operands() {
        let a = EF::from_repr(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        let b = EF::from_repr(0x1111_2222_3333_4444_5555_6666_7777_8888);
        let t = TensorAlgebra::<F, EF>::exterior_product(a, b);

        for (v, col) in t.columns().iter().enumerate() {
            assert_eq!(
                *col,
                a * EF::from(BasedVectorSpace::<F>::as_basis_coefficients_slice(&b)[v])
            );
        }
        for (u, row) in t.rows().iter().enumerate() {
            assert_eq!(
                *row,
                b * EF::from(BasedVectorSpace::<F>::as_basis_coefficients_slice(&a)[u])
            );
        }
    }

    /// Scaling columns is `φ0`, scaling rows is `φ1`, and the two commute.
    #[test]
    fn column_and_row_scaling_commute() {
        let a = EF::from_repr(0xdead_beef_0000_0000_0000_0000_cafe_babe);
        let b = EF::from_repr(0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100);
        let (x, y) = (EF::from_repr(7), EF::from_repr(11));

        let mut lhs = TensorAlgebra::<F, EF>::exterior_product(a, b);
        lhs.scale_columns(x);
        lhs.scale_rows(y);

        let mut rhs = TensorAlgebra::<F, EF>::exterior_product(a, b);
        rhs.scale_rows(y);
        rhs.scale_columns(x);

        assert_eq!(lhs, rhs);
        assert_eq!(lhs, TensorAlgebra::<F, EF>::exterior_product(a * x, b * y));
    }

    /// `1 ⊗ 1` reads as one in the first slot and zero elsewhere, which is what makes the
    /// equality-element recurrence start correctly. Relies on `β_0 = 1`.
    #[test]
    fn one_is_the_exterior_product_of_ones() {
        let t = TensorAlgebra::<F, EF>::one();
        assert_eq!(
            t,
            TensorAlgebra::<F, EF>::exterior_product(EF::ONE, EF::ONE)
        );
        assert_eq!(t.columns()[0], EF::ONE);
        assert!(t.columns()[1..].iter().all(|c| *c == EF::ZERO));
        assert_eq!(t.rows()[0], EF::ONE);
        assert!(t.rows()[1..].iter().all(|r| *r == EF::ZERO));
    }

    /// A coefficient vector of the wrong length has no `DIMENSION × DIMENSION` reading, so
    /// the constructor rejects it rather than building an element that indexes out of range.
    #[test]
    fn a_wrong_length_coefficient_vector_is_rejected() {
        let dimension = TensorAlgebra::<F, EF>::DIMENSION;
        let expected = dimension * dimension;

        let well_formed = TensorAlgebra::<F, EF>::try_from(vec![F::ZERO; expected]).unwrap();
        assert!(well_formed.is_well_formed());

        assert_eq!(
            TensorAlgebra::<F, EF>::try_from(vec![F::ZERO; expected - 1]),
            Err(RingSwitchError::MalformedTensor {
                expected,
                actual: expected - 1,
            })
        );
    }

    /// Serialization round-trips through the checked constructor, so the shape invariant
    /// survives the wire.
    #[test]
    fn serialization_round_trips() {
        let t = TensorAlgebra::<F, EF>::exterior_product(EF::from_repr(3), EF::from_repr(5));
        let bytes = serde_json::to_vec(&t).unwrap();
        let decoded: TensorAlgebra<F, EF> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(t, decoded);
    }

    /// A short coefficient vector on the wire is rejected by the deserializer itself.
    #[test]
    fn deserializing_a_short_coefficient_vector_fails() {
        let bytes = serde_json::to_vec(&vec![F::ZERO; 1]).unwrap();
        assert!(serde_json::from_slice::<TensorAlgebra<F, EF>>(&bytes).is_err());
    }
}
