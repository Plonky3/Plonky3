//! Single-sourced stacked-layout planner shared by prover and verifier.
//!
//! # Why shared
//!
//! - Prover and verifier must agree on slot assignment bit-for-bit.
//! - Letting each side re-derive the layout independently makes that
//!   agreement a maintenance invariant that can silently decay on refactor.
//! - Running the same planner on both sides removes the invariant entirely.

use alloc::vec::Vec;

use p3_util::log2_ceil_usize;

use crate::layout::witness::{Selector, TablePlacement};
use crate::table::TableShape;

/// Per-table shape input to the layout planner.
///
/// # Fields
///
/// - `arity`: log base two of the row count per column.
/// - `width`: number of columns in the table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayoutShape {
    pub(crate) arity: usize,
    pub(crate) width: usize,
}

/// Plans the stacked-polynomial layout for a set of source tables.
///
/// # Returns
///
/// - Stacked arity: `log2_ceil` of the total cell count across all tables.
/// - One placement per source table, carrying its slot selectors.
///
/// # Algorithm
///
/// - Sort source tables by arity ascending.
/// - Iterate reversed so the largest tables land at the lowest stacked slots.
/// - Each column gets one contiguous slot of size `2^arity`.
/// - Selector bit-width is the stacked arity minus the table arity.
///
/// # Output ordering
///
/// - Placements are emitted largest-first.
/// - Selectors inside a placement appear in source-column order.
pub(crate) fn plan_layout(shapes: &[LayoutShape]) -> (usize, Vec<TablePlacement>) {
    // Sort indices by arity ascending; reverse-iterate to place largest first.
    let mut order = (0..shapes.len()).collect::<Vec<usize>>();
    order.sort_by_key(|&i| shapes[i].arity);

    // Stacked arity: log2_ceil of the total cell count.
    let k = log2_ceil_usize(
        shapes
            .iter()
            .map(|s| s.width * (1usize << s.arity))
            .sum::<usize>(),
    );

    // Running cursor into the stacked polynomial, in scalar units.
    let mut offset = 0usize;
    let mut placements = Vec::with_capacity(shapes.len());

    // Lay out largest tables first; each column claims one slot.
    for &table_idx in order.iter().rev() {
        let shape = &shapes[table_idx];
        // Slot size per column: 2^arity.
        let slot_size = 1usize << shape.arity;

        // Assign one selector per column, advancing the cursor after each.
        let selectors = (0..shape.width)
            .map(|_| {
                // Selector bit-width is stacked arity minus table arity.
                let selector = Selector::new(k - shape.arity, offset >> shape.arity);
                // Advance the cursor by one slot.
                offset += slot_size;
                selector
            })
            .collect();

        placements.push(TablePlacement::new(table_idx, selectors));
    }

    (k, placements)
}

/// Plan the stacked layout of a set of committed table shapes.
///
/// A caller that packs a witness itself needs the same slot assignment the schemes use.
///
/// Reading it from here is what keeps the two from drifting apart.
///
/// # Returns
///
/// - Stacked arity, the log of the padded cell count across every table.
/// - One placement per source table, largest arity first, carrying its slot selectors.
#[must_use]
pub fn plan_stacked_layout(shapes: &[TableShape]) -> (usize, Vec<TablePlacement>) {
    // The planner reads arity and width only, so the shapes translate one for one.
    let shapes = shapes
        .iter()
        .map(|shape| LayoutShape {
            arity: shape.num_variables(),
            width: shape.width(),
        })
        .collect::<Vec<_>>();
    plan_layout(&shapes)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn the_public_planner_lays_out_what_the_private_one_does() {
        // Invariant: a caller packing its own witness must land on the very same slots.
        //
        // Fixture state: two tables, of two columns each.
        //
        //     table 0   arity 3, width 2   ->  16 cells
        //     table 1   arity 5, width 2   ->  64 cells
        //     stacked   log2_ceil(80) = 7
        let shapes = vec![TableShape::new(3, 2), TableShape::new(5, 2)];
        let private = plan_layout(&[
            LayoutShape { arity: 3, width: 2 },
            LayoutShape { arity: 5, width: 2 },
        ]);
        let public = plan_stacked_layout(&shapes);

        assert_eq!(public.0, private.0);
        assert_eq!(public.0, 7);
        // Table order and slot addresses agree entry by entry.
        for (public, private) in public.1.iter().zip(&private.1) {
            assert_eq!(public.idx(), private.idx());
            let addresses = |placement: &TablePlacement| {
                placement
                    .selectors()
                    .iter()
                    .map(|selector| (selector.num_variables(), selector.index()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(addresses(public), addresses(private));
        }
    }

    #[test]
    fn plan_layout_empty_returns_zero_arity() {
        // Invariant:
        //     With no tables, the stacked polynomial has arity 0 and no placements.
        let (k, placements) = plan_layout(&[]);
        assert_eq!(k, 0);
        assert!(placements.is_empty());
    }

    #[test]
    fn plan_layout_single_table_places_at_origin() {
        // Invariant:
        //     One table with arity k and w columns → stacked arity = log2_ceil(w * 2^k),
        //     w slots starting at offset 0.
        //
        // Fixture state:
        //     arity = 3, width = 2 → total = 16, stacked arity = 4.
        let (k, placements) = plan_layout(&[LayoutShape { arity: 3, width: 2 }]);
        assert_eq!(k, 4);
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].idx(), 0);
        // Column 0 at offset 0, column 1 at offset 8.
        assert_eq!(placements[0].selectors()[0].index(), 0);
        assert_eq!(placements[0].selectors()[1].index(), 1);
    }

    #[test]
    fn plan_layout_sorts_largest_first() {
        // Invariant:
        //     Placements are emitted in arity-descending order.
        //
        // Fixture state:
        //     table 0 arity 3, table 1 arity 5 → placement order [1, 0].
        let shapes = vec![
            LayoutShape { arity: 3, width: 1 },
            LayoutShape { arity: 5, width: 1 },
        ];
        let (_, placements) = plan_layout(&shapes);
        assert_eq!(placements[0].idx(), 1);
        assert_eq!(placements[1].idx(), 0);
    }

    #[test]
    fn plan_layout_offsets_are_contiguous_and_aligned() {
        // Invariant:
        //     Slot offsets for a column of arity k are multiples of 2^k, and
        //     consecutive columns carve disjoint ranges of the stacked poly.
        //
        // Fixture state:
        //     two tables: arity 5 × 2 cols, arity 3 × 2 cols.
        //     stacked arity = log2_ceil(64 + 16) = 7.
        //
        // Expected offsets (scalar units):
        //
        //     largest first (arity 5): slots at 0, 32
        //     then (arity 3):          slots at 64, 72
        let shapes = vec![
            LayoutShape { arity: 3, width: 2 },
            LayoutShape { arity: 5, width: 2 },
        ];
        let (k, placements) = plan_layout(&shapes);
        assert_eq!(k, 7);
        // Largest table (table 1, arity 5) lands first.
        assert_eq!(placements[0].idx(), 1);
        assert_eq!(placements[0].selectors()[0].index() << 5, 0);
        assert_eq!(placements[0].selectors()[1].index() << 5, 32);
        // Smaller table (table 0, arity 3) follows.
        assert_eq!(placements[1].idx(), 0);
        assert_eq!(placements[1].selectors()[0].index() << 3, 64);
        assert_eq!(placements[1].selectors()[1].index() << 3, 72);
    }
}
