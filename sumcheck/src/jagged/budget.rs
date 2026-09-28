//! Cells a layout provisions for a set of column heights.
//!
//! Stacking rounds every column up on its own, and a jagged commitment rounds up once at the end.
//!
//! The difference between the two is what a sparse commitment exists to avoid paying.

use p3_util::log2_ceil_usize;

use super::layout::JaggedLayout;

/// Live cells and provisioned cells for one set of column heights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellBudget {
    /// Cells a trace actually fills.
    live: usize,
    /// Cells a commitment to that trace binds.
    provisioned: usize,
}

impl CellBudget {
    /// Returns what one validated jagged geometry provisions.
    ///
    /// Only the final round-up to the envelope is dead, whatever the individual heights are.
    #[must_use]
    pub fn of(layout: &JaggedLayout) -> Self {
        Self {
            live: layout.area(),
            provisioned: layout.dense_capacity(),
        }
    }

    /// Returns what a stacking that rounds every column up to a power of two provisions.
    ///
    /// This is what a commitment pays when unequal columns share one polynomial without a reduction.
    #[must_use]
    pub fn stacked(heights: &[usize]) -> Self {
        // One slot per column, each the smallest power of two that holds it, then one final round-up.
        let slots = heights
            .iter()
            .map(|&height| 1usize << log2_ceil_usize(height.max(1)))
            .sum::<usize>();

        Self {
            live: heights.iter().sum(),
            provisioned: slots.max(1).next_power_of_two(),
        }
    }

    /// Returns the cells the trace fills.
    #[must_use]
    pub const fn live(&self) -> usize {
        self.live
    }

    /// Returns the cells the commitment binds.
    #[must_use]
    pub const fn provisioned(&self) -> usize {
        self.provisioned
    }

    /// Returns the provisioned cells no trace cell reaches.
    #[must_use]
    pub const fn dead(&self) -> usize {
        self.provisioned - self.live
    }

    /// Returns the provisioned cells per live cell, scaled by the given denominator.
    ///
    /// A ratio is reported as an integer so that no floating point enters a report.
    #[must_use]
    pub const fn provisioned_per_live(&self, scale: usize) -> usize {
        if self.live == 0 {
            return 0;
        }
        self.provisioned * scale / self.live
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::layout::plan_stacked_layout;
    use crate::table::TableShape;

    // The stacking constructor must be the one the planner the commitment schemes share implements.
    // A hand-rolled formula that drifted from it would make every number below fiction.
    fn planner_provisioned(heights: &[usize]) -> usize {
        let shapes = heights
            .iter()
            .map(|&height| TableShape::new(log2_ceil_usize(height.max(1)), 1))
            .collect::<Vec<_>>();
        1usize << plan_stacked_layout(&shapes).0
    }

    #[test]
    fn a_jagged_envelope_rounds_up_once_and_a_stacking_rounds_up_per_column() {
        // Fixture state: four columns whose heights are each one above a power of two.
        //
        // ```text
        //     heights      [5, 9, 17, 33]
        //     live         64
        //     stacked      8 + 16 + 32 + 64 = 120, rounded to 128
        //     jagged       64, already a power of two
        // ```
        let heights = [5, 9, 17, 33];
        let stacked = CellBudget::stacked(&heights);
        assert_eq!(stacked.live(), 64);
        assert_eq!(stacked.provisioned(), 128);
        assert_eq!(stacked.dead(), 64);

        let jagged = CellBudget::of(&JaggedLayout::new(6, &heights).unwrap());
        assert_eq!(jagged.live(), 64);
        assert_eq!(jagged.provisioned(), 64);
        assert_eq!(jagged.dead(), 0);
    }

    #[test]
    fn many_short_columns_are_where_the_two_models_part() {
        // Sixty-four columns of three rows each fill one hundred and ninety-two cells.
        // A stacking gives each of them four, and a jagged envelope gives the whole trace 256.
        let heights = vec![3usize; 64];
        assert_eq!(CellBudget::stacked(&heights).provisioned(), 256);
        assert_eq!(CellBudget::stacked(&heights).dead(), 64);

        let jagged = CellBudget::of(&JaggedLayout::new(2, &heights).unwrap());
        assert_eq!(jagged.provisioned(), 256);
        assert_eq!(jagged.dead(), 64);
    }

    #[test]
    fn a_ratio_is_reported_without_floating_point() {
        // Nine live cells inside a sixteen-cell envelope is one and seven ninths.
        let budget = CellBudget::of(&JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap());
        assert_eq!(budget.provisioned_per_live(1000), 1777);

        // An empty trace has no ratio to report.
        let empty = CellBudget::of(&JaggedLayout::new(3, &[0, 0]).unwrap());
        assert_eq!(empty.provisioned_per_live(1000), 0);
    }

    proptest! {
        #[test]
        fn the_stacking_model_agrees_with_the_planner_the_schemes_share(
            heights in prop::collection::vec(0usize..=40, 1..=16),
        ) {
            prop_assert_eq!(
                CellBudget::stacked(&heights).provisioned(),
                planner_provisioned(&heights)
            );
        }

        #[test]
        fn a_jagged_envelope_never_provisions_more_than_a_stacking(
            heights in prop::collection::vec(0usize..=64, 1..=32),
        ) {
            // Padded to a power-of-two column count, which a jagged geometry requires.
            let mut heights = heights;
            heights.resize(heights.len().next_power_of_two(), 0);

            let jagged = CellBudget::of(&JaggedLayout::new(6, &heights).unwrap());
            let stacked = CellBudget::stacked(&heights);
            prop_assert_eq!(jagged.live(), stacked.live());
            prop_assert!(jagged.provisioned() <= stacked.provisioned());
        }
    }
}
