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
    pub fn scale_columns(&mut self, a: EF) {
        let mut scaled = Self::zero();
        scaled.add_scaled_columns(self, a);
        *self = scaled;
    }

    /// Adds `(a ⊗ 1) * other` into this element.
    ///
    /// # Algorithm
    ///
    /// The row reading is `other = sum_u beta_u ⊗ row_u`, and the first leg carries the
    /// basis vectors alone. Scaling it therefore leaves the rows where they are:
    ///
    /// ```text
    ///     (a ⊗ 1) * other  =  sum_u (a * beta_u) ⊗ row_u
    /// ```
    ///
    /// That is one multiplication per coordinate, whatever the element was accumulated from.
    /// A zero row scales to nothing, so its multiplication is never formed.
    pub fn add_scaled_columns(&mut self, other: &Self, a: EF) {
        for (u, &row) in other.rows.iter().enumerate() {
            if row != EF::ZERO {
                let mut basis = Coefficients::<EF>::zero();
                basis.set(u);
                self.add_exterior_product(a * basis.element(), row);
            }
        }
    }

    /// Multiplies by `1 + a ⊗ 1 + 1 ⊗ b`, the equality polynomial lifted into the algebra.
    ///
    /// In characteristic two `eq(X, Y) = XY + (1 + X)(1 + Y) = 1 + X + Y`.
    /// `X` lands on the first leg and `Y` on the second, so one factor is two scalings.
    pub fn mul_equality_factor(&mut self, a: EF, b: EF) {
        let mut first = self.clone();
        first.scale_columns(a);
        let mut second = self.clone();
        second.scale_rows(b);
        *self += first;
        *self += second;
    }

    /// `sum_x done(point, x) ⊗ eq(x, other)`, the successor weight lifted into the algebra.
    ///
    /// # Overview
    ///
    /// `done(point, x)` is `eq(point, x - 1)` for `x >= 1` and zero at `x = 0`.
    /// It is the successor on these coordinates with no row repeating, the settled
    /// accumulator of `Point::eval_next`.
    ///
    /// # Algorithm
    ///
    /// The `eval_next` recurrence, with the point on the first leg and `other` on the second.
    /// Folding from the lowest coordinate up:
    ///
    /// ```text
    ///     done     <- done * (1 + a ⊗ 1 + 1 ⊗ b) + (carry_a (1 + a)) ⊗ (carry_b b)
    ///     carry_a  <- carry_a * a
    ///     carry_b  <- carry_b * (1 + b)
    /// ```
    ///
    /// The carry stays one exterior product, so it is held as its two legs.
    ///
    /// # Panics
    ///
    /// Panics unless the two points name the same number of coordinates.
    #[must_use]
    pub fn successor_element(point: &[EF], other: &[EF]) -> Self {
        assert_eq!(
            point.len(),
            other.len(),
            "the successor element pairs one coordinate of each point"
        );
        let (mut carry_a, mut carry_b) = (EF::ONE, EF::ONE);
        let mut done = Self::zero();
        for (&a, &b) in point.iter().zip(other).rev() {
            done.mul_equality_factor(a, b);
            done.add_exterior_product(carry_a * (EF::ONE + a), carry_b * b);
            carry_a *= a;
            carry_b *= EF::ONE + b;
        }
        done
    }

    /// Column `v` alone, read as an element.
    ///
    /// Coordinate `u` of the column is coordinate `v` of row `u`, so this is `d` bit reads.
    #[must_use]
    pub fn column(&self, v: usize) -> EF {
        let mut column = Coefficients::<EF>::zero();
        for (u, &row) in self.rows.iter().enumerate() {
            if Coefficients::of(row).get(v) {
                column.set(u);
            }
        }
        column.element()
    }
}

/// A sum of exterior products, accumulated one bucket per byte value of its left factor.
///
/// # Algorithm
///
/// The row reading of `sum_w a_w (x) b_w` adds `b_w` into row `u` for every coordinate `u` the
/// left factor sets, so half the rows on average. Bucketing by whole bytes of that factor adds
/// each term once per byte instead, and a row is the sum of the buckets whose byte sets it:
///
/// ```text
///     bucket[k][s] = sum of b_w over the w whose byte k of a_w is s
///     row 8k + j   = sum of bucket[k][s] over the s with bit j set
/// ```
///
/// The closing pass over the buckets is `d/8 * 256` entries, whatever the sum was over.
///
/// The buckets are an accumulation detail of the reductions here, not a wire or API type.
#[derive(Clone, Debug)]
pub(crate) struct BitTensorBuckets<EF> {
    /// Per byte position of the left factor, one sum per value that byte takes.
    buckets: Vec<[EF; 256]>,
}

impl<EF: TowerLevel> BitTensorBuckets<EF> {
    /// Empty buckets, which read back as the zero element.
    pub(crate) fn zero() -> Self {
        Self {
            buckets: alloc::vec![[EF::ZERO; 256]; EF::NUM_BYTES],
        }
    }

    /// Add `a (x) b` to the sum.
    #[inline]
    pub(crate) fn add_exterior_product(&mut self, a: EF, b: EF) {
        for (bucket, byte) in self.buckets.iter_mut().zip(a.into_bytes()) {
            bucket[usize::from(byte)] += b;
        }
    }

    /// Forget every term added so far, keeping the allocation.
    ///
    /// A sweep that reads back one partial sum per block accumulates them in turn.
    pub(crate) fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.fill(EF::ZERO);
        }
    }

    /// The element the buckets hold.
    pub(crate) fn tensor(&self) -> BitTensor<EF> {
        let mut tensor = BitTensor::zero();
        for (position, bucket) in self.buckets.iter().enumerate() {
            for (value, &sum) in bucket.iter().enumerate() {
                // Byte value `value` sets coordinate `8 * position + bit` for each of its bits.
                let mut bits = value as u8;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;

                    // A level narrower than its byte has no coordinate up there.
                    // No term can have set one either, so those sums are zero.
                    if let Some(row) = tensor.rows.get_mut(position * 8 + bit) {
                        *row += sum;
                    }
                }
            }
        }
        tensor
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
    use p3_binary_field::{BinaryField16, BinaryField128, Gf2};
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;

    #[test]
    fn the_buckets_hold_the_sum_of_the_exterior_products() {
        let mut rng = SmallRng::seed_from_u64(0xB0C5);
        let terms = (0..32)
            .map(|_| (rng.random::<EF>(), rng.random::<EF>()))
            .collect::<Vec<_>>();

        let mut buckets = BitTensorBuckets::<EF>::zero();
        let mut tensor = BitTensor::<EF>::zero();
        for &(a, b) in &terms {
            buckets.add_exterior_product(a, b);
            tensor.add_exterior_product(a, b);
        }
        assert_eq!(buckets.tensor(), tensor);
    }

    #[test]
    fn a_level_narrower_than_its_byte_reads_its_own_rows_back() {
        // Gf2 holds one coordinate in a byte of eight, so seven bucket bits address no row.
        let mut buckets = BitTensorBuckets::<Gf2>::zero();
        buckets.add_exterior_product(Gf2::ONE, Gf2::ONE);
        assert_eq!(
            buckets.tensor(),
            BitTensor::exterior_product(Gf2::ONE, Gf2::ONE)
        );
    }

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
    fn adding_a_scaled_element_scales_its_column_reading() {
        // Invariant: the accumulating form is the standalone scaling, added.
        //
        //     out += (a ⊗ 1) * other
        //
        // A sum accumulated under one scale therefore need not be scaled term by term.
        let mut rng = SmallRng::seed_from_u64(0x5CA7);
        let a = rng.random::<EF>();

        let mut expected = element(0x5CA8);
        let mut scaled = element(0x5CA9);
        scaled.scale_columns(a);
        expected += scaled;

        let mut accumulated = element(0x5CA8);
        accumulated.add_scaled_columns(&element(0x5CA9), a);

        assert_eq!(accumulated, expected);
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
    fn the_equality_factor_is_agree_plus_disagree() {
        // Invariant: in characteristic two, eq(a, b) = ab + (1 + a)(1 + b) = 1 + a + b.
        //
        //     agree     x * (a ⊗ b)
        //     disagree  x * ((1 + a) ⊗ (1 + b))
        let mut rng = SmallRng::seed_from_u64(0xE0F);
        for _ in 0..8 {
            let original = element(rng.random());
            let (a, b) = (rng.random::<EF>(), rng.random::<EF>());

            let mut agree = original.clone();
            agree.scale_columns(a);
            agree.scale_rows(b);
            let mut disagree = original.clone();
            disagree.scale_columns(EF::ONE + a);
            disagree.scale_rows(EF::ONE + b);
            agree += disagree;

            let mut factored = original;
            factored.mul_equality_factor(a, b);
            assert_eq!(factored, agree);
        }
    }

    #[test]
    fn the_successor_element_matches_its_hypercube_definition() {
        // Invariant: the recurrence is linear in the variables; the definition is exponential.
        //
        //     e = sum_x done(point, x) ⊗ eq(x, other)
        //     done(point, x) = eq(point, x - 1) for x >= 1, and 0 at x = 0
        //
        // The done weight is read off `Point::eval_next`, the recurrence the zerocheck uses.
        // Its closed form is cross-checked too, so a wrong semantics fails here, not later.
        let mut rng = SmallRng::seed_from_u64(0x5CC);
        for num_variables in 0..6 {
            let point = Point::<EF>::rand(&mut rng, num_variables);
            let other = Point::<EF>::rand(&mut rng, num_variables);
            let eq_point = Poly::<EF>::new_from_point(point.as_slice(), EF::ONE);
            let eq_other = Poly::<EF>::new_from_point(other.as_slice(), EF::ONE);

            let mut expected = BitTensor::zero();
            for x in 0..1usize << num_variables {
                let row = Point::<EF>::hypercube(x, num_variables);
                let (_, done, _) = Point::eval_next(point.as_slice(), row.as_slice());
                let closed = if x == 0 {
                    EF::ZERO
                } else {
                    eq_point.as_slice()[x - 1]
                };
                assert_eq!(done, closed, "{num_variables} variables, row {x}");
                expected.add_exterior_product(done, eq_other.as_slice()[x]);
            }

            assert_eq!(
                BitTensor::successor_element(point.as_slice(), other.as_slice()),
                expected,
                "{num_variables} variables"
            );
        }
    }

    #[test]
    fn one_column_is_that_column_of_the_transpose() {
        // Invariant: reading one column agrees with the full transpose, column by column.
        let original = element(0xC07);
        let columns = original.columns();
        for (v, &column) in columns.iter().enumerate() {
            assert_eq!(original.column(v), column, "column {v}");
        }
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
