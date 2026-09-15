//! The tensor algebra `EF ⊗_{F_2} EF`, held as a bit matrix.

use alloc::vec::Vec;
use core::ops::AddAssign;

use p3_binary_field::TowerLevel;
use serde::{Deserialize, Serialize};

use super::basis::Coefficients;

/// An element of `EF ⊗_{F_2} EF`, held as the `d` rows of a `d x d` bit matrix.
///
/// # Overview
///
/// Fix the `F_2`-basis the coordinates define.
/// The element is the matrix `m`, with `m[u][v]` the `beta_u ⊗ beta_v` term.
/// A row of bits is an element of `EF`, so the matrix is `d` elements:
///
/// ```text
///     row u     =  sum_v m[u][v] * beta_v
///     column v  =  sum_u m[u][v] * beta_u
/// ```
///
/// Rows are what this type stores, so the row reading is free.
/// The column reading is one bit transpose away.
///
/// # What crosses the wire
///
/// The rows, as `d` elements of `EF`: the whole element, one bit per entry.
/// 2 KB at `d = 128`, against the 16 KB a byte per coefficient would cost.
///
/// The only route from untrusted data to this type checks the row count.
/// A deserialized element is therefore already the right shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    into = "Vec<EF>",
    try_from = "Vec<EF>",
    bound(serialize = "EF: TowerLevel", deserialize = "EF: TowerLevel")
)]
pub struct BitTensor<EF> {
    /// Row `u` as an element: bit `v` is the coefficient of `beta_u ⊗ beta_v`.
    rows: Vec<EF>,
}

impl<EF: TowerLevel> BitTensor<EF> {
    /// The side length of the matrix: the field's dimension over `F_2`.
    pub const DIMENSION: usize = Coefficients::<EF>::DIMENSION;

    /// The additive identity: the all-zero matrix.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            rows: alloc::vec![EF::ZERO; Self::DIMENSION],
        }
    }

    /// The multiplicative identity `1 ⊗ 1`.
    ///
    /// Formed from the coordinates of one rather than placed at the corner.
    ///
    /// That is correct whichever basis the level carries.
    #[must_use]
    pub fn one() -> Self {
        Self::exterior_product(EF::ONE, EF::ONE)
    }

    /// Adds `a ⊗ b` into this element, without forming the product separately.
    ///
    /// Coordinates are bits, so a term is an addition, not a multiplication.
    ///
    /// Row `u` takes `b` exactly when coordinate `u` of `a` is set.
    pub fn add_exterior_product(&mut self, a: EF, b: EF) {
        for u in Coefficients::of(a).iter_set() {
            self.rows[u] += b;
        }
    }

    /// `a ⊗ b`.
    #[must_use]
    pub fn exterior_product(a: EF, b: EF) -> Self {
        let mut out = Self::zero();
        out.add_exterior_product(a, b);
        out
    }

    /// Whether the element carries the row count both readings index.
    ///
    /// Every constructor here produces a well-formed element.
    ///
    /// This is the check a consumer applies to one it did not build itself.
    #[must_use]
    pub const fn is_well_formed(&self) -> bool {
        self.rows.len() == Self::DIMENSION
    }

    /// The rows, each read as an element.
    ///
    /// These are what a transcript absorbs, being what crosses the wire.
    #[must_use]
    pub fn rows(&self) -> &[EF] {
        &self.rows
    }

    /// The columns, each read as an element.
    ///
    /// The matrix transposed, which is `d^2` bit moves.
    ///
    /// Taken once per use rather than maintained alongside the rows.
    #[must_use]
    pub fn columns(&self) -> Vec<EF> {
        let d = Self::DIMENSION;

        // Read each row's coordinates once, so the transpose is one gather.
        let source = self
            .rows
            .iter()
            .map(|&row| Coefficients::of(row))
            .collect::<Vec<_>>();

        (0..d)
            .map(|v| {
                let mut column = Coefficients::<EF>::zero();
                for (u, row) in source.iter().enumerate() {
                    if row.get(v) {
                        column.set(u);
                    }
                }
                column.element()
            })
            .collect()
    }

    /// Scales the row reading: row `u` becomes `b * row u`.
    ///
    /// This is multiplication by `1 ⊗ b`, acting on the second tensor leg.
    pub fn scale_rows(&mut self, b: EF) {
        for row in &mut self.rows {
            *row *= b;
        }
    }

    /// Scales the column reading: column `v` becomes `a * column v`.
    ///
    /// This is multiplication by `a ⊗ 1`, acting on the first tensor leg.
    /// The stored rows are the wrong reading for it.
    /// So the matrix is transposed, scaled, and transposed back.
    pub fn scale_columns(&mut self, a: EF) {
        let mut columns = self.columns();
        for column in &mut columns {
            *column *= a;
        }
        self.rows = Self { rows: columns }.columns();
    }
}

impl<EF: TowerLevel> AddAssign<&Self> for BitTensor<EF> {
    fn add_assign(&mut self, rhs: &Self) {
        for (row, &other) in self.rows.iter_mut().zip(&rhs.rows) {
            *row += other;
        }
    }
}

impl<EF: TowerLevel> AddAssign for BitTensor<EF> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<EF> From<BitTensor<EF>> for Vec<EF> {
    fn from(value: BitTensor<EF>) -> Self {
        value.rows
    }
}

impl<EF: TowerLevel> TryFrom<Vec<EF>> for BitTensor<EF> {
    type Error = MalformedBitTensor;

    fn try_from(rows: Vec<EF>) -> Result<Self, Self::Error> {
        // Both readings index a square matrix, so a wrong count defines none.
        if rows.len() == Self::DIMENSION {
            Ok(Self { rows })
        } else {
            Err(MalformedBitTensor {
                expected: Self::DIMENSION,
                actual: rows.len(),
            })
        }
    }
}

/// A tensor element whose row count is not the field's dimension over `F_2`.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("a bit tensor element carries {actual} rows, expected {expected}")]
pub struct MalformedBitTensor {
    /// The row count both readings index.
    pub expected: usize,
    /// The count the data supplied.
    pub actual: usize,
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField16, BinaryField128};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;

    /// A tensor element built from a handful of exterior products.
    fn element(seed: u64) -> BitTensor<EF> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut out = BitTensor::zero();
        for _ in 0..8 {
            out.add_exterior_product(rng.random(), rng.random());
        }
        out
    }

    /// The transpose, taken by reading the columns back as rows.
    fn transpose(element: &BitTensor<EF>) -> BitTensor<EF> {
        BitTensor::try_from(element.columns()).unwrap()
    }

    #[test]
    fn the_matrix_is_square_in_the_fields_dimension() {
        // Fixture state: 16 bits give a 16 x 16 matrix, 128 bits a 128 x 128.
        assert_eq!(BitTensor::<BinaryField16>::DIMENSION, 16);
        assert_eq!(BitTensor::<BinaryField128>::DIMENSION, 128);
        assert_eq!(BitTensor::<BinaryField16>::zero().rows().len(), 16);
    }

    #[test]
    fn an_exterior_product_is_the_outer_product_of_the_coordinates() {
        // Invariant: entry `(u, v)` is coordinate `u` of `a` times `v` of `b`.
        //
        // Over `F_2` that is the two bits being set together.
        let mut rng = SmallRng::seed_from_u64(0x0117);
        for _ in 0..32 {
            let (a, b) = (rng.random::<EF>(), rng.random::<EF>());
            let product = BitTensor::exterior_product(a, b);
            let (left, right) = (Coefficients::of(a), Coefficients::of(b));

            for (u, row) in product.rows().iter().enumerate() {
                for (v, entry) in Coefficients::of(*row).iter().enumerate() {
                    assert_eq!(entry, left.get(u) && right.get(v), "entry ({u}, {v})");
                }
            }
        }
    }

    #[test]
    fn the_identity_is_one_tensor_one() {
        // The identity seeds the equality recurrence.
        //
        // A wrong value there is a silent completeness failure, no rejection.
        assert_eq!(
            BitTensor::<EF>::one(),
            BitTensor::exterior_product(EF::ONE, EF::ONE)
        );
    }

    #[test]
    fn transposing_twice_is_the_identity() {
        // The column reading is the transpose, so twice must be the identity.
        let original = element(0x7A5);

        assert_eq!(transpose(&transpose(&original)), original);
    }

    #[test]
    fn the_column_reading_transposes_the_row_reading() {
        // Invariant: coordinate `u` of column `v` is coordinate `v` of row `u`.
        let original = element(0xC01);
        let columns = original.columns();

        for (u, row) in original.rows().iter().enumerate() {
            let row_bits = Coefficients::of(*row);
            for (v, column) in columns.iter().enumerate() {
                assert_eq!(
                    Coefficients::of(*column).get(u),
                    row_bits.get(v),
                    "entry ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn scaling_the_rows_scales_every_row_reading() {
        // Multiplying by `1 ⊗ b` acts on the second leg, the row reading.
        let mut scaled = element(0x505);
        let before = scaled.rows().to_vec();

        let b = SmallRng::seed_from_u64(0x506).random::<EF>();
        scaled.scale_rows(b);

        for (after, &original) in scaled.rows().iter().zip(&before) {
            assert_eq!(*after, original * b);
        }
    }

    #[test]
    fn scaling_the_columns_scales_every_column_reading() {
        // Multiplying by `a ⊗ 1` acts on the first leg, the column reading.
        //
        // The stored rows are the other reading, so this is the transpose path.
        let mut scaled = element(0xC015);
        let before = scaled.columns();

        let a = SmallRng::seed_from_u64(0xC016).random::<EF>();
        scaled.scale_columns(a);

        for (after, &original) in scaled.columns().iter().zip(&before) {
            assert_eq!(*after, original * a);
        }
    }

    #[test]
    fn the_two_scalings_commute() {
        // The equality recurrence scales both legs, in either order.
        let original = element(0xC0119);
        let mut rng = SmallRng::seed_from_u64(0xC011A);
        let (a, b) = (rng.random::<EF>(), rng.random::<EF>());

        let mut rows_first = original.clone();
        rows_first.scale_rows(b);
        rows_first.scale_columns(a);

        let mut columns_first = original;
        columns_first.scale_columns(a);
        columns_first.scale_rows(b);

        assert_eq!(rows_first, columns_first);
    }

    #[test]
    fn a_wrong_row_count_is_refused() {
        // The only route from untrusted data checks the shape it indexes.
        assert_eq!(
            BitTensor::<EF>::try_from(alloc::vec![EF::ZERO; 15]).unwrap_err(),
            MalformedBitTensor {
                expected: 16,
                actual: 15,
            }
        );
        assert!(BitTensor::<EF>::try_from(alloc::vec![EF::ZERO; 16]).is_ok());
        assert!(BitTensor::<EF>::zero().is_well_formed());
    }

    #[test]
    fn the_wire_form_round_trips() {
        // The rows cross the wire, so reading them back must rebuild it.
        let original = element(0x5E4);

        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: BitTensor<EF> = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, original);
    }

    proptest! {
        #[test]
        fn addition_is_entrywise(a: u16, b: u16, c: u16, d: u16) {
            // Adding two elements adds their matrices, a row-wise add here.
            let mut left = BitTensor::exterior_product(EF::from_repr(a), EF::from_repr(b));
            let right = BitTensor::exterior_product(EF::from_repr(c), EF::from_repr(d));
            let expected: Vec<EF> = left
                .rows()
                .iter()
                .zip(right.rows())
                .map(|(&x, &y)| x + y)
                .collect();

            left += &right;
            prop_assert_eq!(left.rows(), expected.as_slice());
        }
    }
}
