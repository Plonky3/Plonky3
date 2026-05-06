//! Physical placement of source tables inside the stacked committed polynomial.

use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::reverse_bits_len;

use crate::sumcheck::layout::plan::{LayoutShape, plan_layout};
use crate::sumcheck::table::TableShape;

/// Identifies one slot inside the stacked polynomial.
#[derive(Debug, Clone, Copy)]
pub struct Selector {
    /// Bit-width of the slot address, carved from the stacked variables.
    num_variables: usize,
    /// Slot index, interpreted as an integer in `0..2^num_variables`.
    index: usize,
}

impl Selector {
    /// Builds a selector over `num_variables` bits pointing at slot `index`.
    ///
    /// # Panics
    ///
    /// - Slot index must fit in `num_variables` bits.
    pub const fn new(num_variables: usize, index: usize) -> Self {
        // Bounds check: slot index must address a valid hypercube point.
        assert!(index < (1 << num_variables));
        Self {
            num_variables,
            index,
        }
    }

    /// Returns the hypercube point that addresses this slot.
    #[inline(always)]
    pub fn point<F: Field>(&self) -> Point<F> {
        Point::hypercube(self.index, self.num_variables)
    }

    /// Returns the number of selector bits.
    #[inline(always)]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the slot index.
    #[inline(always)]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns this selector with its bitstring reversed.
    #[inline(always)]
    pub const fn reverse(&mut self) {
        self.index = reverse_bits_len(self.index, self.num_variables);
    }

    /// Prefixes `other` with the selector bits.
    #[inline(always)]
    pub fn lift_prefix<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        // Expand the slot index as selector bits; single allocation.
        let mut out: Point<Ext> = self.point();
        // Append the local coordinates to finish the stacked-space point.
        out.extend(other);
        out
    }

    /// Appends the selector bits after the local coordinates.
    #[inline(always)]
    pub fn lift_suffix<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        let mut out = other.clone();
        out.extend(&self.point());
        out
    }
}

/// A column-major table of multilinear polynomials sharing a common arity.
///
/// # Invariants
///
/// - At least one column.
/// - Every column has the same number of variables.
#[derive(Debug, Clone)]
pub struct Table<F: Field>(Vec<Poly<F>>);

impl<F: Field> Table<F> {
    /// Creates a table from one or more polynomials.
    ///
    /// # Panics
    ///
    /// - Column list must not be empty.
    /// - Every column must share the same arity.
    pub fn new(polys: Vec<Poly<F>>) -> Self {
        // Structural precondition: at least one column.
        assert!(!polys.is_empty());
        // Consistency precondition: every column shares the same arity.
        assert!(
            polys.iter().map(Poly::num_variables).all_equal(),
            "every column of a table must share the same arity",
        );
        Self(polys)
    }

    /// Returns the polynomial at column `id`.
    pub fn poly(&self, id: usize) -> &Poly<F> {
        &self.0[id]
    }

    /// Returns the number of columns.
    pub const fn num_polys(&self) -> usize {
        self.0.len()
    }

    /// Returns the shared number of variables.
    pub fn num_variables(&self) -> usize {
        // Invariant (set by the constructor): every column shares this value.
        self.0[0].num_variables()
    }

    /// Returns the verifier shape of this table.
    pub fn shape(&self) -> TableShape {
        TableShape::new(self.num_variables(), self.num_polys())
    }

    /// Pads every column with zeros until the table has at least `num_variables`.
    fn pad_zeros(&mut self, num_variables: usize) {
        let current_num_variables = self.num_variables();
        if current_num_variables < num_variables {
            self.0
                .iter_mut()
                .for_each(|poly| poly.pad_zeros(num_variables));
        }
    }
}

/// Placement metadata for one table inside the stacked polynomial.
#[derive(Debug, Clone)]
pub struct TablePlacement {
    /// Source table index this placement refers back to.
    pub(super) idx: usize,
    /// One selector per column, addressing the column's slot.
    pub(super) selectors: Vec<Selector>,
}

impl TablePlacement {
    /// Creates placement metadata for table index `idx` with the given selectors.
    pub const fn new(idx: usize, selectors: Vec<Selector>) -> Self {
        Self { idx, selectors }
    }

    /// Reverses every selector bitstring in this placement.
    pub fn reverse_selectors(&mut self) {
        self.selectors.iter_mut().for_each(Selector::reverse);
    }

    /// Returns the number of columns placed for this table.
    pub const fn num_polys(&self) -> usize {
        self.selectors.len()
    }

    /// Returns the source table index.
    pub const fn idx(&self) -> usize {
        self.idx
    }

    /// Returns the selector assigned to each column.
    pub fn selectors(&self) -> &[Selector] {
        &self.selectors
    }
}

/// Owns the source tables together with the stacked committed polynomial.
#[derive(Debug, Clone)]
pub struct Witness<F: Field> {
    /// Source tables behind the stacked polynomial.
    pub(super) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(super) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(super) num_variables: usize,
    /// Preprocessing depth (number of rounds the protocol folds upfront).
    pub(super) folding: usize,
    /// Stacked committed polynomial.
    pub(super) poly: Poly<F>,
}

impl<F: Field> Witness<F> {
    /// Stacks the given tables into a single committed polynomial.
    ///
    /// # Algorithm
    ///
    /// - Sort tables by arity ascending; reverse-iterate to place largest first.
    /// - Each column occupies one slot of size `2^arity`.
    /// - Selector bit-width equals the stacked arity minus the table arity.
    /// - Total stacked size is rounded up to the next power of two.
    /// - Unused tail entries stay zero.
    ///
    /// # Panics
    ///
    /// - Table list must be non-empty.
    /// - Tables below the preprocessing depth are zero-padded to that depth.
    #[tracing::instrument(skip_all)]
    pub fn new(mut tables: Vec<Table<F>>, folding: usize) -> Self {
        // Precondition: need at least one source table to stack.
        assert!(
            !tables.is_empty(),
            "Witness requires at least one source table"
        );
        // Normalize small tables to the committed arity used by the protocol.
        tables.iter_mut().for_each(|table| table.pad_zeros(folding));

        // Delegate slot assignment to the shared planner (same routine as the verifier).
        let shapes: Vec<LayoutShape> = tables
            .iter()
            .map(|t| LayoutShape {
                arity: t.num_variables(),
                width: t.num_polys(),
            })
            .collect();
        let (num_variables, placements) = plan_layout(&shapes);

        // Stacked buffer starts zero; unused tail entries stay zero.
        let mut stacked = Poly::<F>::zero(num_variables);

        // Copy each source column into its planner-assigned slot.
        for placement in &placements {
            let table = &tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let poly = table.poly(poly_idx);
                let dst = selector.index << poly.num_variables();
                stacked.as_mut_slice()[dst..dst + poly.num_evals()]
                    .copy_from_slice(poly.as_slice());
            }
        }

        Self {
            tables,
            placements,
            num_variables,
            folding,
            poly: stacked,
        }
    }

    /// Stacks the given tables with local variables before selector variables.
    ///
    /// # Layout
    ///
    /// Current `new` stores each column contiguously as:
    ///
    /// ```text
    ///     P(selector_bits, local_bits)
    /// ```
    ///
    /// This constructor stores each column strided by selector bits as:
    ///
    /// ```text
    ///     P(local_bits, selector_bits)
    /// ```
    ///
    /// Local table evaluation order is preserved. Only selector bitstrings are
    /// reversed from the prefix-oriented planner so mixed-arity selector codes
    /// remain suffix-disjoint.
    ///
    /// # Panics
    ///
    /// - Table list must be non-empty.
    /// - Tables below the preprocessing depth are zero-padded to that depth.
    #[tracing::instrument(skip_all)]
    pub fn new_interleaved(mut tables: Vec<Table<F>>, folding: usize) -> Self {
        assert!(
            !tables.is_empty(),
            "Witness requires at least one source table"
        );
        tables.iter_mut().for_each(|table| table.pad_zeros(folding));

        let shapes: Vec<LayoutShape> = tables
            .iter()
            .map(|t| LayoutShape {
                arity: t.num_variables(),
                width: t.num_polys(),
            })
            .collect();
        let (num_variables, mut placements) = plan_layout(&shapes);
        placements
            .iter_mut()
            .for_each(TablePlacement::reverse_selectors);

        let mut stacked = Poly::<F>::zero(num_variables);

        for placement in &placements {
            let table = &tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let poly = table.poly(poly_idx);

                for (local_idx, &value) in poly.as_slice().iter().enumerate() {
                    let dst = (local_idx << selector.num_variables) | selector.index;
                    stacked.as_mut_slice()[dst] = value;
                }
            }
        }

        Self {
            tables,
            placements,
            num_variables,
            folding,
            poly: stacked,
        }
    }

    /// Returns the number of variables of the stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns verifier table shapes after witness normalization/padding.
    pub fn table_shapes(&self) -> Vec<TableShape> {
        self.tables.iter().map(Table::shape).collect()
    }

    /// Returns the stacked committed polynomial.
    pub const fn poly(&self) -> &Poly<F> {
        &self.poly
    }

    /// Splits the witness into its owned components for downstream use.
    pub(super) fn into_parts(self) -> WitnessParts<F> {
        // Hand each field to the caller verbatim; no normalisation is needed.
        WitnessParts {
            tables: self.tables,
            placements: self.placements,
            num_variables: self.num_variables,
            folding: self.folding,
            poly: self.poly,
        }
    }
}

/// Owned components of a stacked-layout commitment.
#[derive(Debug, Clone)]
pub(super) struct WitnessParts<F: Field> {
    /// Source tables stacked into the committed polynomial.
    pub(super) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(super) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(super) num_variables: usize,
    /// Number of preprocessing rounds folded upfront.
    pub(super) folding: usize,
    /// Stacked committed polynomial.
    pub(super) poly: Poly<F>,
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_util::log2_ceil_usize;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::sumcheck::layout::{Layout, PrefixProver, SuffixProver};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn selector_new_stores_num_variables_and_index() {
        // Invariant:
        //     Constructor stores both fields verbatim.
        //
        // Fixture state:
        //     num_variables = 3, index = 5
        let sel = Selector::new(3, 5);

        // Check: getters return what was passed in.
        assert_eq!(sel.num_variables(), 3);
        assert_eq!(sel.index(), 5);
    }

    #[test]
    #[should_panic]
    fn selector_new_panics_on_index_out_of_range() {
        // Invariant:
        //     A slot index that does not fit in num_variables bits is rejected.
        //
        // Fixture state:
        //     num_variables = 2 → legal indices are 0..=3; use 4 to trigger panic.
        let _ = Selector::new(2, 4);
    }

    #[test]
    fn selector_point_returns_boolean_vector_of_right_length() {
        // Invariant:
        //     point() returns a num_variables-long vector of 0/1 field elements.
        //
        // Fixture state:
        //     num_variables = 3, index = 5 = 0b101 → point bits are {1, 0, 1}.
        let sel = Selector::new(3, 5);
        let point: Point<F> = sel.point();

        // Check: length matches the bit-width.
        assert_eq!(point.num_variables(), 3);

        // Check: every coordinate is either 0 or 1.
        for &bit in point.as_slice() {
            assert!(bit == F::ZERO || bit == F::ONE);
        }
    }

    #[test]
    fn selector_lift_prefixes_selector_bits_onto_other() {
        // Invariant:
        //     lift(other) returns a point with the selector bits prepended,
        //     total length = selector.num_variables() + other.num_variables().
        //
        // Fixture state:
        //     selector: 3-bit, index 5 → 3 prefix coordinates.
        //     other:    2-variable point over EF.
        let sel = Selector::new(3, 5);
        let other: Point<EF> = Point::new(vec![EF::from_u64(7), EF::from_u64(11)]);
        let lifted = sel.lift_prefix(&other);

        // Check: total length is the concatenation length.
        assert_eq!(lifted.num_variables(), 5);

        // Check: the suffix matches the original point element-wise.
        for i in 0..2 {
            assert_eq!(lifted.as_slice()[3 + i], other.as_slice()[i]);
        }
    }

    #[test]
    fn selector_lift_suffix_appends_selector_bits_after_other() {
        // Invariant:
        //     lift_suffix(other) returns a point with the local coordinates
        //     first and the selector bits appended, total length =
        //     other.num_variables() + selector.num_variables().
        //
        // Fixture state:
        //     selector: 3-bit, index 5 = 0b101 → bits {1, 0, 1}.
        //     other:    2-variable point over EF.
        let sel = Selector::new(3, 5);
        let other: Point<EF> = Point::new(vec![EF::from_u64(7), EF::from_u64(11)]);
        let lifted = sel.lift_suffix(&other);

        // Check: total length is the concatenation length.
        assert_eq!(lifted.num_variables(), 5);

        // Check: the prefix matches the original point element-wise.
        assert_eq!(lifted.as_slice()[0], other.as_slice()[0]);
        assert_eq!(lifted.as_slice()[1], other.as_slice()[1]);

        // Check: the suffix is the boolean expansion of the slot index.
        let selector_bits: Point<EF> = sel.point();
        assert_eq!(lifted.as_slice()[2], selector_bits.as_slice()[0]);
        assert_eq!(lifted.as_slice()[3], selector_bits.as_slice()[1]);
        assert_eq!(lifted.as_slice()[4], selector_bits.as_slice()[2]);
    }

    #[test]
    fn selector_reverse_swaps_msb_and_lsb_within_width() {
        // Invariant:
        //     reverse() flips the slot index bitstring within num_variables bits,
        //     leaving num_variables itself untouched.
        //
        // Fixture state:
        //     num_variables = 4, index = 0b0010 = 2 → reversed: 0b0100 = 4.
        let mut sel = Selector::new(4, 0b0010);
        sel.reverse();

        // Check: bit-width is preserved.
        assert_eq!(sel.num_variables(), 4);
        // Check: index has been bit-reversed within those 4 bits.
        assert_eq!(sel.index(), 0b0100);
    }

    #[test]
    fn selector_reverse_is_idempotent_under_double_application() {
        // Invariant:
        //     Reversing a selector twice restores the original index.
        //
        // Fixture state:
        //     num_variables = 5, index = 13 → reverse → reverse → 13.
        let mut sel = Selector::new(5, 13);
        let original_index = sel.index();

        sel.reverse();
        sel.reverse();

        assert_eq!(sel.num_variables(), 5);
        assert_eq!(sel.index(), original_index);
    }

    #[test]
    #[should_panic]
    fn table_new_panics_on_empty_polys() {
        // Invariant:
        //     A table with no columns is rejected.
        let _: Table<F> = Table::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "must share the same arity")]
    fn table_new_panics_on_mixed_arities() {
        // Invariant:
        //     Mixing columns of different arities is rejected.
        //
        // Fixture state:
        //     poly_a: 3 variables, poly_b: 4 variables → must panic.
        let poly_a = Poly::<F>::zero(3);
        let poly_b = Poly::<F>::zero(4);
        let _ = Table::new(vec![poly_a, poly_b]);
    }

    #[test]
    fn table_accessors_report_shape() {
        // Invariant:
        //     num_polys, num_variables, size, and poly(id) all agree with the input.
        //
        // Fixture state:
        //     two columns of arity 3 → num_polys = 2, num_variables = 3, size = 2^3 * 2 = 16.
        let table = Table::new(vec![Poly::<F>::zero(3), Poly::<F>::zero(3)]);

        // Check: all shape queries match the fixture.
        assert_eq!(table.num_polys(), 2);
        assert_eq!(table.num_variables(), 3);
        // Check: column lookup returns a ref to the i-th poly with matching arity.
        assert_eq!(table.poly(0).num_variables(), 3);
    }

    #[test]
    fn table_placement_accessors_return_stored_values() {
        // Invariant:
        //     TablePlacement forwards the table index and the selector slice.
        //
        // Fixture state:
        //     idx = 7, two selectors with num_variables = 2.
        let selectors = vec![Selector::new(2, 0), Selector::new(2, 1)];
        let placement = TablePlacement::new(7, selectors);

        // Check: idx() forwards the constructor argument.
        assert_eq!(placement.idx(), 7);
        // Check: num_polys() matches the selector count.
        assert_eq!(placement.num_polys(), 2);
        // Check: selectors() exposes the underlying slice.
        assert_eq!(placement.selectors().len(), 2);
    }

    #[test]
    fn table_placement_reverse_selectors_flips_each_in_place() {
        // Invariant:
        //     reverse_selectors() bit-reverses every selector index in place,
        //     leaving each selector's width and the placement's table index
        //     untouched.
        //
        // Fixture state:
        //     idx = 4, three selectors of width 3 with indices {0b001, 0b010, 0b110}.
        //     Reversed indices within 3 bits: {0b100, 0b010, 0b011} = {4, 2, 3}.
        let selectors = vec![
            Selector::new(3, 0b001),
            Selector::new(3, 0b010),
            Selector::new(3, 0b110),
        ];
        let mut placement = TablePlacement::new(4, selectors);

        placement.reverse_selectors();

        // Check: the table index is preserved.
        assert_eq!(placement.idx(), 4);
        // Check: every selector's width is preserved.
        for selector in placement.selectors() {
            assert_eq!(selector.num_variables(), 3);
        }
        // Check: each selector index has been bit-reversed within 3 bits.
        let indices: Vec<usize> = placement.selectors().iter().map(Selector::index).collect();
        assert_eq!(indices, vec![0b100, 0b010, 0b011]);
    }

    #[test]
    fn table_shape_reports_arity_and_width() {
        // Invariant:
        //     shape() returns the (num_variables, width) pair derived from the
        //     table's columns.
        //
        // Fixture state:
        //     three columns of arity 4 → expected shape: (4, 3).
        let table = Table::new(vec![
            Poly::<F>::zero(4),
            Poly::<F>::zero(4),
            Poly::<F>::zero(4),
        ]);

        let shape = table.shape();

        assert_eq!(shape.num_variables(), 4);
        assert_eq!(shape.width(), 3);
    }

    // Builds a deterministic witness with two tables of arities (4, 3) and
    // two columns each. Used by several tests to avoid repeating the setup.
    fn fixture_witness() -> Witness<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        // Table 0: arity 3, two columns.
        let t0 = Table::new(vec![
            Poly::<F>::rand(&mut rng, 3),
            Poly::<F>::rand(&mut rng, 3),
        ]);
        // Table 1: arity 4, two columns.
        let t1 = Table::new(vec![
            Poly::<F>::rand(&mut rng, 4),
            Poly::<F>::rand(&mut rng, 4),
        ]);
        Witness::new(vec![t0, t1], 1)
    }

    #[test]
    fn witness_new_places_largest_table_first() {
        // Invariant:
        //     Placement order is tables sorted by arity ascending, reversed.
        //
        // Fixture state:
        //     table 0: arity 3, table 1: arity 4.
        //     expected placement order: [table 1 (arity 4), table 0 (arity 3)]
        let w = fixture_witness();

        // Check: first placement targets the larger table.
        assert_eq!(w.placements[0].idx(), 1);
        // Check: second placement targets the smaller table.
        assert_eq!(w.placements[1].idx(), 0);
    }

    #[test]
    fn witness_num_variables_is_stacked_arity_rounded_up() {
        // Invariant:
        //     num_variables equals log2_ceil of total stacked size.
        //
        // Fixture state:
        //     total size = 2^3 * 2 + 2^4 * 2 = 16 + 32 = 48 → ceil(log2) = 6.
        let w = fixture_witness();
        assert_eq!(w.num_variables(), 6);
    }

    #[test]
    fn witness_new_copies_column_evals_into_slots() {
        // Invariant:
        //     Every column's evaluations appear in the stacked polynomial
        //     at the offset computed by its selector.
        //
        // Fixture state:
        //     two columns per table, two tables; each column occupies its
        //     own slot; destination = selector.index << arity.
        let w = fixture_witness();

        // Walk every placement; for each column, compare the slot slice to the source column.
        for placement in &w.placements {
            let table = &w.tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let col = table.poly(poly_idx);
                let dst = selector.index() << col.num_variables();
                let slot = &w.poly.as_slice()[dst..dst + col.num_evals()];
                // Check: slot contents match the source column evaluations.
                assert_eq!(slot, col.as_slice());
            }
        }
    }

    #[test]
    fn witness_new_interleaves_by_suffix_selectors() {
        // Invariant:
        //     Local-first stacking stores P(local_bits, selector_bits), preserving
        //     each table's local evaluation order.
        //
        // Fixture state:
        //     A has arity 2: [a0, a1, a2, a3]
        //     B has arity 1: [b0, b1]
        //     Expected storage: [a0, b0, a1, 0, a2, b1, a3, 0].
        let a0 = F::from_u64(10);
        let a1 = F::from_u64(11);
        let a2 = F::from_u64(12);
        let a3 = F::from_u64(13);
        let b0 = F::from_u64(20);
        let b1 = F::from_u64(21);

        let table_a = Table::new(vec![Poly::new(vec![a0, a1, a2, a3])]);
        let table_b = Table::new(vec![Poly::new(vec![b0, b1])]);
        let witness = Witness::new_interleaved(vec![table_a, table_b], 0);

        assert_eq!(witness.num_variables(), 3);
        assert_eq!(
            witness.poly.as_slice(),
            &[a0, b0, a1, F::ZERO, a2, b1, a3, F::ZERO],
        );
    }

    #[test]
    fn witness_new_pads_tables_below_folding() {
        // Invariant:
        //     A table smaller than the preprocessing depth is committed as the
        //     zero-padded polynomial with arity equal to folding.
        let a0 = F::from_u64(10);
        let a1 = F::from_u64(11);
        let table = Table::new(vec![Poly::new(vec![a0, a1])]);

        let witness = Witness::new(vec![table], 3);

        assert_eq!(witness.tables[0].num_variables(), 3);
        assert_eq!(witness.num_variables(), 3);
        assert_eq!(
            witness.tables[0].poly(0).as_slice(),
            &[a0, a1, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        );
        assert_eq!(
            witness.poly.as_slice(),
            &[a0, a1, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        );
    }

    #[test]
    fn witness_table_shapes_returns_shapes_in_source_order() {
        // Invariant:
        //     table_shapes() returns one shape per source table, in the order
        //     the witness was constructed (independent of placement order).
        //
        // Fixture state:
        //     table 0: arity 3, two columns → shape (3, 2).
        //     table 1: arity 4, two columns → shape (4, 2).
        //     Source order: [table 0, table 1].
        let w = fixture_witness();

        let shapes = w.table_shapes();

        assert_eq!(shapes, vec![TableShape::new(3, 2), TableShape::new(4, 2)],);
    }

    #[test]
    fn witness_poly_returns_the_stacked_polynomial() {
        // Invariant:
        //     poly() returns a borrow of the stacked committed polynomial.
        //     Its arity equals num_variables() and its contents match the
        //     internal field used by every consumer.
        //
        // Fixture state:
        //     two-table fixture; expected stacked arity = 6.
        let w = fixture_witness();

        let stacked = w.poly();

        assert_eq!(stacked.num_variables(), w.num_variables());
        assert_eq!(stacked.as_slice(), w.poly.as_slice());
    }

    #[test]
    fn witness_into_parts_preserves_every_field() {
        // Invariant:
        //     into_parts() destructures the witness while preserving every
        //     field byte-for-byte: source tables (in source order), placement
        //     metadata, stacked arity, folding depth, and stacked polynomial.
        //
        // Fixture state:
        //     two-table fixture; folding = 1.
        let w = fixture_witness();
        // Snapshot every field before consumption.
        let expected_num_variables = w.num_variables();
        let expected_folding = w.folding;
        let expected_table_shapes = w.table_shapes();
        let expected_placement_idx: Vec<usize> =
            w.placements.iter().map(TablePlacement::idx).collect();
        let expected_poly = w.poly.clone();

        let parts = w.into_parts();

        // Check: scalar fields survive the move.
        assert_eq!(parts.num_variables, expected_num_variables);
        assert_eq!(parts.folding, expected_folding);
        // Check: source tables are carried over in source order.
        let actual_table_shapes: Vec<TableShape> = parts.tables.iter().map(Table::shape).collect();
        assert_eq!(actual_table_shapes, expected_table_shapes);
        // Check: placement order and table indices survive verbatim.
        let actual_placement_idx: Vec<usize> =
            parts.placements.iter().map(TablePlacement::idx).collect();
        assert_eq!(actual_placement_idx, expected_placement_idx);
        // Check: the stacked polynomial is bit-for-bit identical.
        assert_eq!(parts.poly.as_slice(), expected_poly.as_slice());
    }

    #[test]
    fn prefix_prover_from_witness_carries_stacked_state() {
        // Invariant:
        //     Handing the witness to the prefix prover preserves the
        //     stacked polynomial and the per-table shapes.
        let w = fixture_witness();
        let stacked_copy = w.poly.clone();
        let num_variables = w.num_variables();

        // Build a prefix-mode prover from the witness.
        let prover = PrefixProver::<F, EF>::from_witness(w);

        // Check: arity matches the original stacked polynomial.
        assert_eq!(prover.num_variables(), num_variables);
        // Check: the committed polynomial is bit-for-bit identical.
        assert_eq!(prover.poly.as_slice(), stacked_copy.as_slice());
    }

    #[test]
    fn suffix_prover_from_witness_carries_stacked_state() {
        // Invariant:
        //     Handing the witness to the suffix prover preserves the
        //     stacked arity and the per-table data layout.
        //
        //     The suffix prover does not retain the stacked polynomial —
        //     it walks per-table evaluations on demand — so this test only
        //     checks the structural fields that survive the move.
        let w = fixture_witness();
        let num_variables = w.num_variables();
        let table_shapes = w.table_shapes();

        // Build a suffix-mode prover from the witness.
        let prover = SuffixProver::<F, EF>::from_witness(w);

        // Check: stacked arity matches.
        assert_eq!(prover.num_variables(), num_variables);
        // Check: every source table is carried over with its original shape.
        let prover_shapes: Vec<TableShape> = prover.tables.iter().map(Table::shape).collect();
        assert_eq!(prover_shapes, table_shapes);
    }

    // Proptest strategy: random table shapes within safe bounds.
    //
    //     1..=3 tables, each with arity in 2..=5 and 1..=3 columns.
    //     folding = 1 fits since every arity is at least 2.
    fn arb_table_shapes() -> impl Strategy<Value = Vec<(usize, usize)>> {
        prop::collection::vec((2usize..=5, 1usize..=3), 1..=3)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

        // Invariant:
        //     For any valid set of table shapes:
        //     - every column's evaluations live at selector.index << arity
        //     - num_variables equals log2_ceil of total size
        //     - trailing slots past the concatenation stay zero
        #[test]
        fn witness_stacks_columns_and_zeros_the_tail(shapes in arb_table_shapes()) {
            let mut rng = SmallRng::seed_from_u64(123);

            // Build one table per (arity, width) entry with random evaluations.
            let tables: Vec<Table<F>> = shapes
                .iter()
                .map(|&(arity, width)| {
                    let polys = (0..width).map(|_| Poly::<F>::rand(&mut rng, arity)).collect();
                    Table::new(polys)
                })
                .collect();

            // Total stacked size (before power-of-two rounding).
            let total_used: usize = tables
                .iter()
                .map(|t| (1 << t.num_variables()) * t.num_polys())
                .sum();

            // Folding = 1 is always safe since the strategy guarantees arity >= 2.
            let witness = Witness::new(tables, 1);

            // Check: stacked arity equals log2_ceil of total occupied size.
            assert_eq!(witness.num_variables(), log2_ceil_usize(total_used));

            // Check: each column's evaluations land at the predicted slot.
            let mut used = 0usize;
            for placement in &witness.placements {
                let table = &witness.tables[placement.idx()];
                for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                    let col = table.poly(poly_idx);
                    let dst = selector.index() << col.num_variables();
                    let slot = &witness.poly.as_slice()[dst..dst + col.num_evals()];
                    assert_eq!(slot, col.as_slice());
                    used += col.num_evals();
                }
            }

            // Check: the counted occupied region matches total_used.
            assert_eq!(used, total_used);

            // Check: every entry past the concatenation is the zero element.
            let stacked_len = 1usize << witness.num_variables();
            for &v in &witness.poly.as_slice()[used..stacked_len] {
                // The specific region of "used" is contiguous here only because
                // placements are emitted largest-first and slot offsets are the
                // cursor value. That matches the "unused tail stays zero" rule.
                assert_eq!(v, F::ZERO);
            }
        }
    }
}
