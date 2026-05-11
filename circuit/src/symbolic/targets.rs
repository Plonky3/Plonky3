//! Circuit target identifiers for row selectors and trace columns.

use p3_air::{BaseEntry, ExtEntry};

use crate::ExprId;

/// Circuit expression identifiers for the Lagrange row selector flags.
#[derive(Clone, Copy, Debug)]
pub struct RowSelectorsTargets {
    /// Selector that equals `1` on the first trace row.
    pub is_first_row: ExprId,
    /// Selector that equals `1` on the last trace row.
    pub is_last_row: ExprId,
    /// Selector that equals `1` on every row except the last.
    pub is_transition: ExprId,
}

/// Circuit expression identifiers for every column category in an AIR.
///
/// Each slice maps positionally to the symbolic variables produced
/// by the AIR's constraint generation.
#[derive(Clone, Debug)]
pub struct ColumnsTargets<'a> {
    /// Challenge values drawn from the verifier.
    pub challenges: &'a [ExprId],
    /// Public input values.
    pub public_values: &'a [ExprId],
    /// Permutation column values at the current row.
    pub permutation_local_values: &'a [ExprId],
    /// Permutation column values at the next row.
    pub permutation_next_values: &'a [ExprId],
    /// Global lookup expected cumulated values.
    pub permutation_values: &'a [ExprId],
    /// Preprocessed column values at the current row.
    pub local_prep_values: &'a [ExprId],
    /// Preprocessed column values at the next row.
    pub next_prep_values: &'a [ExprId],
    /// Main trace column values at the current row.
    pub local_values: &'a [ExprId],
    /// Main trace column values at the next row.
    pub next_values: &'a [ExprId],
}

impl ColumnsTargets<'_> {
    /// Map a base-field variable to its circuit target.
    ///
    /// Supports preprocessed, main trace, and public value entries.
    ///
    /// # Panics
    ///
    /// - If the row offset exceeds 1 (only current and next rows are supported).
    /// - If the entry is a periodic column (not yet supported).
    pub fn resolve_base_var(&self, entry: &BaseEntry, index: usize) -> ExprId {
        match entry {
            // Preprocessed columns: offset 0 = current row, offset 1 = next row.
            BaseEntry::Preprocessed { offset: 0 } => self.local_prep_values[index],
            BaseEntry::Preprocessed { offset: 1 } => self.next_prep_values[index],
            // Main trace columns: offset 0 = current row, offset 1 = next row.
            BaseEntry::Main { offset: 0 } => self.local_values[index],
            BaseEntry::Main { offset: 1 } => self.next_values[index],
            // Public values are not row-dependent.
            BaseEntry::Public => self.public_values[index],
            BaseEntry::Periodic => {
                unimplemented!("Periodic values are not supported.")
            }
            // Only two-row windows (current + next) are supported.
            _ => panic!("Cannot have expressions involving more than two rows."),
        }
    }

    /// Map an extension-field variable to its circuit target.
    ///
    /// Supports permutation column, challenge, and cumulated value entries.
    ///
    /// # Panics
    ///
    /// Panics if the row offset exceeds 1.
    pub fn resolve_ext_var(&self, entry: &ExtEntry, index: usize) -> ExprId {
        match entry {
            // Permutation columns: offset 0 = current row, offset 1 = next row.
            ExtEntry::Permutation { offset: 0 } => self.permutation_local_values[index],
            ExtEntry::Permutation { offset: 1 } => self.permutation_next_values[index],
            // Verifier challenges are global (not row-dependent).
            ExtEntry::Challenge => self.challenges[index],
            // Global lookup cumulated values.
            ExtEntry::PermutationValue => self.permutation_values[index],
            // Only two-row windows (current + next) are supported.
            _ => panic!("Cannot have expressions involving more than two rows."),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{BaseEntry, ExtEntry};

    use super::{ColumnsTargets, ExprId};

    /// Build a `ColumnsTargets` with numbered `ExprId`s for easy assertions.
    ///
    /// Layout (2 columns per category):
    /// - challenges:              [0, 1]
    /// - public_values:           [2, 3]
    /// - permutation_local:       [4, 5]
    /// - permutation_next:        [6, 7]
    /// - permutation_values:      [8, 9]
    /// - local_prep:              [10, 11]
    /// - next_prep:               [12, 13]
    /// - local_values (main):     [14, 15]
    /// - next_values (main):      [16, 17]
    fn make_ids() -> [ExprId; 18] {
        core::array::from_fn(|i| ExprId(i as u32))
    }

    fn make_columns(ids: &[ExprId; 18]) -> ColumnsTargets<'_> {
        ColumnsTargets {
            challenges: &ids[0..2],
            public_values: &ids[2..4],
            permutation_local_values: &ids[4..6],
            permutation_next_values: &ids[6..8],
            permutation_values: &ids[8..10],
            local_prep_values: &ids[10..12],
            next_prep_values: &ids[12..14],
            local_values: &ids[14..16],
            next_values: &ids[16..18],
        }
    }

    #[test]
    fn resolve_base_var_preprocessed() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Preprocessed { offset: 0 }, 0),
            ids[10]
        );
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Preprocessed { offset: 0 }, 1),
            ids[11]
        );
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Preprocessed { offset: 1 }, 0),
            ids[12]
        );
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Preprocessed { offset: 1 }, 1),
            ids[13]
        );
    }

    #[test]
    fn resolve_base_var_main() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Main { offset: 0 }, 0),
            ids[14]
        );
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Main { offset: 0 }, 1),
            ids[15]
        );
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Main { offset: 1 }, 0),
            ids[16]
        );
        assert_eq!(
            cols.resolve_base_var(&BaseEntry::Main { offset: 1 }, 1),
            ids[17]
        );
    }

    #[test]
    fn resolve_base_var_public() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        assert_eq!(cols.resolve_base_var(&BaseEntry::Public, 0), ids[2]);
        assert_eq!(cols.resolve_base_var(&BaseEntry::Public, 1), ids[3]);
    }

    #[test]
    #[should_panic(expected = "Periodic values are not supported")]
    fn resolve_base_var_periodic_panics() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        cols.resolve_base_var(&BaseEntry::Periodic, 0);
    }

    #[test]
    #[should_panic(expected = "Cannot have expressions involving more than two rows")]
    fn resolve_base_var_bad_offset_panics() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        cols.resolve_base_var(&BaseEntry::Main { offset: 2 }, 0);
    }

    #[test]
    fn resolve_ext_var_permutation() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        assert_eq!(
            cols.resolve_ext_var(&ExtEntry::Permutation { offset: 0 }, 0),
            ids[4]
        );
        assert_eq!(
            cols.resolve_ext_var(&ExtEntry::Permutation { offset: 0 }, 1),
            ids[5]
        );
        assert_eq!(
            cols.resolve_ext_var(&ExtEntry::Permutation { offset: 1 }, 0),
            ids[6]
        );
        assert_eq!(
            cols.resolve_ext_var(&ExtEntry::Permutation { offset: 1 }, 1),
            ids[7]
        );
    }

    #[test]
    fn resolve_ext_var_challenge() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        assert_eq!(cols.resolve_ext_var(&ExtEntry::Challenge, 0), ids[0]);
        assert_eq!(cols.resolve_ext_var(&ExtEntry::Challenge, 1), ids[1]);
    }

    #[test]
    fn resolve_ext_var_permutation_value() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        assert_eq!(cols.resolve_ext_var(&ExtEntry::PermutationValue, 0), ids[8]);
        assert_eq!(cols.resolve_ext_var(&ExtEntry::PermutationValue, 1), ids[9]);
    }

    #[test]
    #[should_panic(expected = "Cannot have expressions involving more than two rows")]
    fn resolve_ext_var_bad_offset_panics() {
        let ids = make_ids();
        let cols = make_columns(&ids);
        cols.resolve_ext_var(&ExtEntry::Permutation { offset: 2 }, 0);
    }
}
