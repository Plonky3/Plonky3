//! Physical placement of source tables inside the stacked committed polynomial.

use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::sumcheck::layout::plan::{LayoutShape, plan_layout};
use crate::sumcheck::layout::prover::{PrefixProver, SuffixProver};

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
    pub fn point<F: Field>(&self) -> Point<F> {
        Point::hypercube(self.index, self.num_variables)
    }

    /// Returns the number of selector bits.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the slot index.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Prefixes `other` with the selector bits.
    pub fn lift<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        // Expand the slot index as selector bits; single allocation.
        let mut out: Point<Ext> = self.point();
        // Append the local coordinates to finish the stacked-space point.
        out.extend(other);
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

    /// Returns the total number of stacked evaluations contributed.
    pub fn num_total_values(&self) -> usize {
        (1 << self.num_variables()) * self.num_polys()
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
    /// - Every table arity must exceed the preprocessing depth.
    pub fn new(tables: Vec<Table<F>>, folding: usize) -> Self {
        // Precondition: need at least one source table to stack.
        assert!(
            !tables.is_empty(),
            "Witness requires at least one source table"
        );
        // Precondition: every table must have at least one variable left after folding.
        assert!(tables.iter().all(|table| table.num_variables() > folding));

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
                let dst = selector.index() << poly.num_variables();
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

    /// Returns the number of variables of the stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Consumes the witness and hands it to a prefix-mode prover.
    pub fn as_prefix_prover<EF: ExtensionField<F>>(self) -> PrefixProver<F, EF> {
        // One empty concrete-claim list per source table.
        let num_tables = self.tables.len();
        PrefixProver {
            tables: self.tables,
            placements: self.placements,
            num_variables: self.num_variables,
            folding: self.folding,
            poly: self.poly,
            claim_map: (0..num_tables).map(|_| Vec::new()).collect(),
            virtual_claims: Vec::new(),
        }
    }

    /// Consumes the witness and hands it to a suffix-mode prover.
    pub fn as_suffix_prover<EF: ExtensionField<F>>(self) -> SuffixProver<F, EF> {
        // One empty concrete-claim list per source table.
        let num_tables = self.tables.len();
        SuffixProver {
            tables: self.tables,
            placements: self.placements,
            num_variables: self.num_variables,
            folding: self.folding,
            poly: self.poly,
            claim_map: (0..num_tables).map(|_| Vec::new()).collect(),
            virtual_claims: Vec::new(),
        }
    }
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
        let lifted = sel.lift(&other);

        // Check: total length is the concatenation length.
        assert_eq!(lifted.num_variables(), 5);

        // Check: the suffix matches the original point element-wise.
        for i in 0..2 {
            assert_eq!(lifted.as_slice()[3 + i], other.as_slice()[i]);
        }
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
        assert_eq!(table.num_total_values(), 16);
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
    fn witness_as_prefix_prover_carries_stacked_state() {
        // Invariant:
        //     Handing the witness to the prefix prover preserves the
        //     stacked polynomial and the per-table shapes.
        let w = fixture_witness();
        let stacked_copy = w.poly.clone();
        let num_variables = w.num_variables();

        // Hand off the witness to a prefix-mode prover.
        let prover: PrefixProver<F, EF> = w.as_prefix_prover();

        // Check: arity matches the original stacked polynomial.
        assert_eq!(prover.num_variables(), num_variables);
        // Check: the committed polynomial is bit-for-bit identical.
        assert_eq!(prover.poly().as_slice(), stacked_copy.as_slice());
    }

    #[test]
    fn witness_as_suffix_prover_carries_stacked_state() {
        // Invariant:
        //     Handing the witness to the suffix prover preserves the
        //     stacked polynomial and the per-table shapes.
        let w = fixture_witness();
        let stacked_copy = w.poly.clone();
        let num_variables = w.num_variables();

        // Hand off the witness to a suffix-mode prover.
        let prover: SuffixProver<F, EF> = w.as_suffix_prover();

        // Check: arity matches the original stacked polynomial.
        assert_eq!(prover.num_variables(), num_variables);
        // Check: the committed polynomial is bit-for-bit identical.
        assert_eq!(prover.poly().as_slice(), stacked_copy.as_slice());
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
            let total_used: usize = tables.iter().map(Table::num_total_values).sum();

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
