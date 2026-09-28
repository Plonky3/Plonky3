//! Placement of every fraction block inside the padded leaf table.

use alloc::vec::Vec;
use core::cmp::Reverse;

use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_util::log2_ceil_usize;

use super::{TableLookup, position};

/// What a block of leaves carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockRole {
    /// One reader's fractions, one per row of that reader.
    Reader {
        /// Position of the reader within its own table.
        index: usize,
    },
    /// One table's fractions, one per table entry.
    Table,
}

/// One contiguous run of leaves owned by a reader or by a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    /// Position of the owning table in statement order.
    pub(crate) table: usize,
    /// What this block carries.
    pub(crate) role: BlockRole,
    /// Base-two logarithm of the block's height.
    pub(crate) num_variables: usize,
    /// First leaf this block owns.
    pub(crate) offset: usize,
}

impl Block {
    /// Equality weight selecting this block at a point of the padded table.
    ///
    /// The leading coordinates address the block and the trailing ones address a leaf inside it.
    ///
    /// Every block is a power of two placed at a multiple of its own height.
    ///
    /// So the leading coordinates take one hypercube value across the whole block.
    pub(crate) fn weight<F: Field, EF: ExtensionField<F>>(
        &self,
        point: &Point<EF>,
        num_variables: usize,
    ) -> EF {
        let prefix = num_variables - self.num_variables;
        let vertex = Point::<F>::hypercube(self.offset >> self.num_variables, prefix);
        Point::eval_eq(vertex.as_slice(), &point.as_slice()[..prefix])
    }

    /// Trailing coordinates of a point that address a leaf inside this block.
    pub(crate) fn subpoint<EF: Field>(&self, point: &Point<EF>, num_variables: usize) -> Point<EF> {
        point.get_subpoint_over_range(num_variables - self.num_variables..num_variables)
    }
}

/// Shape of one table in a reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TableShape {
    /// Base-two logarithm of the number of table entries.
    pub(crate) num_variables: usize,
    /// Number of columns each entry carries.
    pub(crate) width: usize,
    /// Base-two logarithm of each reader's row count, in the table's reader order.
    pub(crate) readers: Vec<usize>,
}

/// Where every fraction block sits, and how wide the padded leaf table is.
///
/// Both sides build this from the statement alone.
///
/// No number here ever comes out of a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupStarPlan {
    /// Every block, in placement order.
    pub(crate) blocks: Vec<Block>,
    /// One entry per table, in statement order.
    pub(crate) tables: Vec<TableShape>,
    /// Variable count of the padded leaf table.
    pub(crate) num_variables: usize,
    /// Largest reader row count in the whole reduction, as a base-two logarithm.
    pub(crate) max_reader_variables: usize,
    /// Largest table entry count in the whole reduction, as a base-two logarithm.
    pub(crate) max_table_variables: usize,
}

impl LogupStarPlan {
    /// Lay the statement out into blocks.
    ///
    /// # Panics
    ///
    /// Panics if the statement has no table.
    ///
    /// Panics if a table has no reader, since nothing would then be proved about it.
    ///
    /// Panics if a table has no entries, since the reduction needs a variable to split on.
    ///
    /// Panics if two readers of one table disagree on how many columns they pull.
    ///
    /// Panics if a table has more entries than the base field embeds injectively.
    pub fn new<F: Field, EF: ExtensionField<F>>(lookups: &[TableLookup<'_, EF>]) -> Self {
        assert!(!lookups.is_empty(), "a reduction needs at least one table");

        // Read the shape off the statement, checking as we go that it describes a reduction.
        let tables = lookups
            .iter()
            .map(|lookup| {
                assert!(
                    !lookup.readers.is_empty(),
                    "a table with no reader has nothing to prove"
                );
                assert!(
                    lookup.num_variables > 0,
                    "a table needs at least one variable for the reduction to split on"
                );
                assert!(
                    position::fits::<F>(lookup.num_variables),
                    "a table of 2^{} entries does not embed injectively in this field",
                    lookup.num_variables
                );

                // Every reader pulls the whole entry, so one width serves the table.
                let width = lookup.width();
                assert!(width > 0, "a table entry must carry at least one column");
                assert!(
                    lookup
                        .readers
                        .iter()
                        .all(|reader| reader.claims.len() == width),
                    "every reader of one table pulls the same columns"
                );

                TableShape {
                    num_variables: lookup.num_variables,
                    width,
                    readers: lookup
                        .readers
                        .iter()
                        .map(|reader| reader.point.num_variables())
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        // Enumerate the blocks in one canonical order, readers of a table before the table.
        //
        //     table 0: reader 0, reader 1, entries | table 1: reader 0, entries | ...
        let mut blocks = tables
            .iter()
            .enumerate()
            .flat_map(|(table, shape)| {
                let readers =
                    shape
                        .readers
                        .iter()
                        .enumerate()
                        .map(move |(index, &num_variables)| Block {
                            table,
                            role: BlockRole::Reader { index },
                            num_variables,
                            offset: 0,
                        });
                readers.chain(core::iter::once(Block {
                    table,
                    role: BlockRole::Table,
                    num_variables: shape.num_variables,
                    offset: 0,
                }))
            })
            .collect::<Vec<_>>();

        // Tallest blocks first, so a block's offset always lands on a multiple of its height.
        //
        // Every earlier block is at least as tall.
        //
        // So the running total is already a multiple of this block's height.
        //
        // That is what lets the leading coordinates address it.
        //
        // The sort is stable, so equal heights keep the canonical order above.
        //
        // Both sides therefore land on the same layout.
        blocks.sort_by_key(|block| Reverse(block.num_variables));

        // Hand each block the next aligned run of leaves.
        let mut height = 0;
        for block in &mut blocks {
            block.offset = height;
            height += 1 << block.num_variables;
        }

        let max_reader_variables = tables
            .iter()
            .flat_map(|shape| shape.readers.iter().copied())
            .max()
            .expect("every table has at least one reader");
        let max_table_variables = tables
            .iter()
            .map(|shape| shape.num_variables)
            .max()
            .expect("the statement has at least one table");

        Self {
            blocks,
            tables,
            num_variables: log2_ceil_usize(height),
            max_reader_variables,
            max_table_variables,
        }
    }

    /// Variable count of the padded fraction table the reduction runs over.
    ///
    /// This is also its layer count, and what its soundness error scales with.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Coordinates the position-column claims are drawn from.
    pub const fn position_point_variables(&self) -> usize {
        self.max_reader_variables
    }

    /// Coordinates the table-column claims are drawn from.
    ///
    /// This is also how many rounds the product sumcheck runs for.
    pub const fn table_point_variables(&self) -> usize {
        self.max_table_variables
    }

    /// Number of tables in the reduction.
    pub(crate) const fn num_tables(&self) -> usize {
        self.tables.len()
    }

    /// Number of readers across every table.
    pub(crate) fn num_readers(&self) -> usize {
        self.tables.iter().map(|shape| shape.readers.len()).sum()
    }

    /// Largest reader count any single table has.
    ///
    /// Readers are weighted by powers of one challenge.
    ///
    /// Only readers of the same table are ever combined, so this is how many powers exist.
    pub(crate) fn max_readers_per_table(&self) -> usize {
        self.tables
            .iter()
            .map(|shape| shape.readers.len())
            .max()
            .expect("the statement has at least one table")
    }

    /// Position of one reader among all readers, counting tables in statement order.
    pub(crate) fn reader_offset(&self, table: usize) -> usize {
        self.tables[..table]
            .iter()
            .map(|shape| shape.readers.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::logup_star::Reader;

    type Binary = BinaryField128;

    /// Claim points of the given heights, owned so a statement can borrow them.
    fn points(heights: &[usize]) -> Vec<Point<Binary>> {
        heights
            .iter()
            .map(|&height| Point::new(vec![Binary::ONE; height]))
            .collect()
    }

    /// One reader per point, each pulling `width` columns out of a shared claim buffer.
    fn readers_over<'a>(
        points: &'a [Point<Binary>],
        claims: &'a [Binary],
    ) -> Vec<Reader<'a, Binary>> {
        points
            .iter()
            .map(|point| Reader { point, claims })
            .collect()
    }

    #[test]
    fn blocks_are_laid_down_tallest_first() {
        // Fixture state: readers of heights 2^1, 2^5, 2^2 over a table of 2^3 entries.
        //
        // Placement order must be by descending height, whatever order they arrive in:
        //
        //     2^5 reader | 2^3 table | 2^2 reader | 2^1 reader
        let claims = [Binary::ONE];
        let owned = points(&[1, 5, 2]);
        let readers = readers_over(&owned, &claims);
        let plan = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 3,
            readers: &readers,
        }]);

        let heights = plan
            .blocks
            .iter()
            .map(|block| block.num_variables)
            .collect::<Vec<_>>();
        assert_eq!(heights, vec![5, 3, 2, 1]);
    }

    #[test]
    fn every_block_starts_at_a_multiple_of_its_own_height() {
        // Alignment is what lets the leading coordinates of a point address a block.
        //
        // Without it the block would straddle two subcubes and no equality weight selects it.
        let claims = [Binary::ONE];
        let owned = points(&[5, 2, 3, 2]);
        let readers = readers_over(&owned, &claims);
        let plan = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 4,
            readers: &readers,
        }]);

        for block in &plan.blocks {
            assert_eq!(
                block.offset % (1 << block.num_variables),
                0,
                "a block must start at a multiple of its own height"
            );
        }
    }

    #[test]
    fn blocks_tile_the_used_leaves_without_overlapping() {
        // A shared leaf double-counts a fraction.
        //
        // A gap leaves one uncounted.
        //
        // Either way the reduction proves the wrong sum.
        let claims = [Binary::ONE];
        let owned = points(&[5, 2, 3, 2]);
        let readers = readers_over(&owned, &claims);
        let plan = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 4,
            readers: &readers,
        }]);

        let mut covered = vec![false; 1 << plan.num_variables];
        for block in &plan.blocks {
            let span = block.offset..block.offset + (1 << block.num_variables);
            assert!(
                covered[span.clone()].iter().all(|&leaf| !leaf),
                "two blocks claim the same leaf"
            );
            covered[span].fill(true);
        }

        // Tallest first means the used leaves form one run from the bottom.
        let used = plan
            .blocks
            .iter()
            .map(|block| 1usize << block.num_variables)
            .sum::<usize>();
        assert!(covered[..used].iter().all(|&leaf| leaf));
        assert!(covered[used..].iter().all(|&leaf| !leaf));
    }

    #[test]
    fn the_padded_table_is_the_smallest_one_that_fits() {
        // Fixture state: one 2^2 reader and one 2^2 table, so four plus four leaves are used.
        //
        // Eight is already a power of two, so nothing is padded.
        let claims = [Binary::ONE];
        let owned = points(&[2]);
        let readers = readers_over(&owned, &claims);
        let exact = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 2,
            readers: &readers,
        }]);
        assert_eq!(exact.num_variables, 3);

        // Adding a 2^1 reader takes the total to ten, which rounds up to sixteen.
        let owned = points(&[2, 1]);
        let readers = readers_over(&owned, &claims);
        let padded = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 2,
            readers: &readers,
        }]);
        assert_eq!(padded.num_variables, 4);
    }

    #[test]
    fn readers_are_numbered_table_by_table() {
        // Position claims travel in one flat list.
        //
        // Both sides must agree on where each table's readers start in it.
        //
        //     table 0: readers 0, 1 | table 1: reader 2 | table 2: readers 3, 4, 5
        let claims = [Binary::ONE];
        let first = points(&[1, 1]);
        let second = points(&[1]);
        let third = points(&[1, 1, 1]);
        let groups = [
            readers_over(&first, &claims),
            readers_over(&second, &claims),
            readers_over(&third, &claims),
        ];
        let lookups = groups
            .iter()
            .map(|readers| TableLookup {
                num_variables: 2,
                readers,
            })
            .collect::<Vec<_>>();
        let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);

        assert_eq!(plan.num_readers(), 6);
        assert_eq!(plan.reader_offset(0), 0);
        assert_eq!(plan.reader_offset(1), 2);
        assert_eq!(plan.reader_offset(2), 3);
        assert_eq!(plan.max_readers_per_table(), 3);
    }

    #[test]
    fn the_same_statement_always_lays_out_the_same_way() {
        // Neither side sends its layout.
        //
        // Both must derive an identical one or their transcripts diverge.
        //
        // Equal-height blocks are the case at risk.
        //
        // Only a stable sort keeps them in the order the statement gave.
        let claims = [Binary::ONE];
        let owned = points(&[2, 2, 2]);
        let readers = readers_over(&owned, &claims);
        let lookups = [TableLookup {
            num_variables: 2,
            readers: &readers,
        }];

        assert_eq!(
            LogupStarPlan::new::<Binary, Binary>(&lookups),
            LogupStarPlan::new::<Binary, Binary>(&lookups)
        );
    }

    #[test]
    #[should_panic(expected = "a reduction needs at least one table")]
    fn rejects_a_statement_with_no_table() {
        let _ = LogupStarPlan::new::<Binary, Binary>(&[]);
    }

    #[test]
    #[should_panic(expected = "a table with no reader has nothing to prove")]
    fn rejects_a_table_nobody_reads() {
        let _ = LogupStarPlan::new::<Binary, Binary>(&[TableLookup::<Binary> {
            num_variables: 2,
            readers: &[],
        }]);
    }

    #[test]
    #[should_panic(expected = "a table needs at least one variable")]
    fn rejects_a_table_with_one_entry() {
        // A single-entry table leaves the reduction no variable to split on.
        let claims = [Binary::ONE];
        let owned = points(&[1]);
        let readers = readers_over(&owned, &claims);
        let _ = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 0,
            readers: &readers,
        }]);
    }

    #[test]
    #[should_panic(expected = "does not embed injectively in this field")]
    fn rejects_a_table_the_field_cannot_index() {
        // Past the field's width two entries would share an embedding.
        //
        // A pushforward could then move weight between them unseen.
        //
        // Catching it here beats panicking deep inside the verifier's own evaluation.
        let claims = [Binary::ONE];
        let owned = points(&[1]);
        let readers = readers_over(&owned, &claims);
        let _ = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 128,
            readers: &readers,
        }]);
    }

    #[test]
    #[should_panic(expected = "every reader of one table pulls the same columns")]
    fn rejects_readers_that_disagree_on_width() {
        // One pushforward serves every reader of a table, so they must pull the same entry.
        let owned = points(&[1, 1]);
        let wide = [Binary::ONE, Binary::ONE];
        let narrow = [Binary::ONE];
        let readers = vec![
            Reader {
                point: &owned[0],
                claims: &wide,
            },
            Reader {
                point: &owned[1],
                claims: &narrow,
            },
        ];
        let _ = LogupStarPlan::new::<Binary, Binary>(&[TableLookup {
            num_variables: 2,
            readers: &readers,
        }]);
    }
}
