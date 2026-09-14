//! Prover tables: the equality weights, the pushforwards, and the padded fractions.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::plan::{BlockRole, LogupStarPlan};
use super::{TableLookup, TableWitness, position};

/// The equality weights each reader contributes, and the pushforwards they scatter into.
pub(crate) struct Weights<EF> {
    /// One weight table per reader, tables in statement order and readers within them.
    ///
    /// A reader carries the equality tensor at its own claim point.
    ///
    /// Its position within the table earns it one power of the batching challenge.
    pub(crate) readers: Vec<Poly<EF>>,
    /// One pushforward per table, in statement order.
    ///
    /// Entry `v` holds the total weight of every row, of every reader of that table, naming `v`.
    pub(crate) pushforwards: Vec<Vec<EF>>,
}

/// Build the equality weights and scatter them into one pushforward per table.
///
/// The pushforward is the only object the reduction adds.
///
/// It is as wide as the table rather than as wide as the readers.
///
/// # Panics
///
/// Panics if a reader's row count disagrees with its claim point.
///
/// Panics if a row names an entry the table does not have.
pub(crate) fn weights<F, EF>(
    plan: &LogupStarPlan,
    lookups: &[TableLookup<'_, EF>],
    witness: &[TableWitness<'_, F>],
    batching: EF,
) -> Weights<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Readers of one table are weighted by consecutive powers of the batching challenge.
    //
    // Only readers of the same table are ever combined.
    //
    // So the largest table fixes how many powers the whole reduction needs.
    let scales = batching
        .powers()
        .take(plan.max_readers_per_table())
        .collect();

    // Expand one equality tensor per reader, its scale folded into the expansion.
    //
    // Seeding the expansion with the scale keeps this to a single pass over a single table.
    let readers = lookups
        .iter()
        .flat_map(|lookup| lookup.readers.iter().zip(&scales))
        .map(|(reader, &scale)| Poly::new_from_point(reader.point.as_slice(), scale))
        .collect::<Vec<_>>();

    // Scatter each table's readers onto its entries and sum.
    //
    //     Y[v] = sum over readers, over rows i naming v, of that row's weight
    let pushforwards = lookups
        .iter()
        .zip(witness)
        .enumerate()
        .map(|(table, (lookup, table_witness))| {
            let num_entries = 1 << lookup.num_variables;
            let first = plan.reader_offset(table);

            table_witness.readers.iter().enumerate().fold(
                EF::zero_vec(num_entries),
                |mut pushforward, (index, reader_witness)| {
                    let weights = readers[first + index].as_slice();
                    assert_eq!(
                        reader_witness.positions.len(),
                        weights.len(),
                        "a reader's row count must match its claim point"
                    );

                    // Rows split across threads, each split filling its own copy of the table.
                    //
                    // Scattering in place would let two rows naming one entry race.
                    //
                    // The splits merge afterwards, since addition ignores order.
                    let scattered = reader_witness
                        .positions
                        .par_iter()
                        .zip(weights.par_iter())
                        .par_fold_reduce(
                            || EF::zero_vec(num_entries),
                            |mut split, (&entry, &weight)| {
                                *split
                                    .get_mut(entry)
                                    .expect("a row names an entry outside its table") += weight;
                                split
                            },
                            |mut left, right| {
                                EF::add_slices(&mut left, &right);
                                left
                            },
                        );

                    EF::add_slices(&mut pushforward, &scattered);
                    pushforward
                },
            )
        })
        .collect();

    Weights {
        readers,
        pushforwards,
    }
}

/// Materialize the padded fraction tables the reduction runs over.
///
/// Every block writes one fraction per leaf it owns:
///
/// ```text
///     reader block   weight of the row        /  challenge - position named by the row
///     table  block   weight over the entry    /  position of the entry - challenge
///     padding        0                        /  1
/// ```
///
/// The table side enters negated, so an honest statement makes the whole sum vanish.
///
/// The reduction is then handed a numerator of zero rather than one read off a proof.
///
/// Padding contributes a zero over a one, which adds nothing and divides by nothing.
pub(crate) fn leaf_tables<F, EF>(
    plan: &LogupStarPlan,
    witness: &[TableWitness<'_, F>],
    weights: &Weights<EF>,
    entry_challenges: &[EF],
) -> (Poly<EF>, Poly<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
{
    let height = 1 << plan.num_variables;
    let mut numerator = EF::zero_vec(height);
    let mut denominator = vec![EF::ONE; height];

    for block in &plan.blocks {
        let span = block.offset..block.offset + (1 << block.num_variables);
        let challenge = entry_challenges[block.table];
        let numerator = &mut numerator[span.clone()];
        let denominator = &mut denominator[span];

        match block.role {
            BlockRole::Reader { index } => {
                let positions = witness[block.table].readers[index].positions;
                let weights = weights.readers[plan.reader_offset(block.table) + index].as_slice();

                numerator.copy_from_slice(weights);
                denominator
                    .par_iter_mut()
                    .zip(positions.par_iter())
                    .for_each(|(denominator, &entry)| {
                        *denominator = challenge - position::embed::<F>(entry);
                    });
            }
            BlockRole::Table => {
                numerator.copy_from_slice(&weights.pushforwards[block.table]);
                denominator
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(entry, denominator)| {
                        *denominator = EF::from(position::embed::<F>(entry)) - challenge;
                    });
            }
        }
    }

    (Poly::new(numerator), Poly::new(denominator))
}

#[cfg(test)]
mod tests {
    use core::iter;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::logup_star::{Reader, ReaderWitness};

    type B = BinaryField128;

    /// A statement and witness over one table, owned so both can be borrowed from.
    struct Fixture {
        table_variables: usize,
        points: Vec<Point<B>>,
        positions: Vec<Vec<usize>>,
        column: Vec<B>,
    }

    impl Fixture {
        /// One table read by as many readers as there are position lists.
        fn new(rng: &mut SmallRng, table_variables: usize, positions: Vec<Vec<usize>>) -> Self {
            let points = positions
                .iter()
                .map(|rows| Point::<B>::rand(rng, log2_of(rows.len())))
                .collect();
            Self {
                table_variables,
                points,
                positions,
                column: Poly::<B>::rand(rng, table_variables).as_slice().to_vec(),
            }
        }

        fn readers(&self) -> Vec<Reader<'_, B>> {
            self.points
                .iter()
                .map(|point| Reader {
                    point,
                    claims: &self.column[..1],
                })
                .collect()
        }

        fn reader_witnesses(&self) -> Vec<ReaderWitness<'_>> {
            self.positions
                .iter()
                .map(|positions| ReaderWitness { positions })
                .collect()
        }
    }

    /// Base-two logarithm of a power-of-two length.
    fn log2_of(length: usize) -> usize {
        length.trailing_zeros() as usize
    }

    #[test]
    fn the_pushforward_is_the_scatter_of_the_reader_weights() {
        // Fixture state: four entries, two readers of four rows each.
        //
        // Reader 0 is unscaled and reader 1 carries the batching challenge.
        //
        // A pushforward that dropped or swapped a scale shows up here.
        let mut rng = SmallRng::seed_from_u64(0x0_5CA7);
        let fixture = Fixture::new(&mut rng, 2, vec![vec![0, 1, 1, 3], vec![2, 2, 0, 1]]);
        let batching = B::interpolation_node(5);

        let readers = fixture.readers();
        let lookups = [TableLookup {
            num_variables: fixture.table_variables,
            readers: &readers,
        }];
        let plan = LogupStarPlan::new(&lookups);
        let reader_witnesses = fixture.reader_witnesses();
        let columns = [fixture.column.as_slice()];
        let witness = [TableWitness {
            columns: &columns,
            readers: &reader_witnesses,
        }];

        let built = weights(&plan, &lookups, &witness, batching);

        // Scatter the same weights by hand, one row at a time.
        let mut expected = B::zero_vec(1 << fixture.table_variables);
        for (index, (point, positions)) in
            iter::zip(&fixture.points, &fixture.positions).enumerate()
        {
            let scale = batching.exp_u64(index as u64);
            let tensor = Poly::<B>::new_from_point(point.as_slice(), B::ONE);
            for (row, &entry) in positions.iter().enumerate() {
                expected[entry] += scale * tensor.as_slice()[row];
            }
        }

        assert_eq!(built.pushforwards[0], expected);
    }

    #[test]
    fn an_entry_nobody_reads_stays_empty() {
        // Entry 3 is named by no row, so nothing scatters onto it.
        //
        // Every other entry carries the weight of the rows that named it.
        //
        // One equality tensor sums to one over the cube, so the whole table does too.
        let mut rng = SmallRng::seed_from_u64(0x0_E4917);
        let fixture = Fixture::new(&mut rng, 2, vec![vec![0, 1, 2, 0]]);

        let readers = fixture.readers();
        let lookups = [TableLookup {
            num_variables: 2,
            readers: &readers,
        }];
        let plan = LogupStarPlan::new(&lookups);
        let reader_witnesses = fixture.reader_witnesses();
        let columns = [fixture.column.as_slice()];
        let witness = [TableWitness {
            columns: &columns,
            readers: &reader_witnesses,
        }];

        let built = weights(&plan, &lookups, &witness, B::ONE);

        assert_eq!(built.pushforwards[0][3], B::ZERO);
        assert_eq!(
            built.pushforwards[0].iter().copied().sum::<B>(),
            B::ONE,
            "an equality tensor sums to one over the cube"
        );
    }

    #[test]
    fn each_block_lands_where_the_plan_puts_it() {
        // The verifier rebuilds each block's value from its own copy of the layout.
        //
        // The fractions have to sit exactly where that layout says.
        //
        // Fixture state: four entries and one reader of four rows.
        //
        // Eight leaves are used, so nothing is padded.
        let mut rng = SmallRng::seed_from_u64(0x0_B10C);
        let fixture = Fixture::new(&mut rng, 2, vec![vec![3, 1, 0, 2]]);
        let challenge = B::interpolation_node(9);

        let readers = fixture.readers();
        let lookups = [TableLookup {
            num_variables: 2,
            readers: &readers,
        }];
        let plan = LogupStarPlan::new(&lookups);
        let reader_witnesses = fixture.reader_witnesses();
        let columns = [fixture.column.as_slice()];
        let witness = [TableWitness {
            columns: &columns,
            readers: &reader_witnesses,
        }];

        let built = weights(&plan, &lookups, &witness, B::ONE);
        let (numerator, denominator) = leaf_tables(&plan, &witness, &built, &[challenge]);

        for block in &plan.blocks {
            let offset = block.offset;
            match block.role {
                BlockRole::Reader { index } => {
                    let tensor =
                        Poly::<B>::new_from_point(fixture.points[index].as_slice(), B::ONE);
                    for (row, &entry) in fixture.positions[index].iter().enumerate() {
                        assert_eq!(numerator.as_slice()[offset + row], tensor.as_slice()[row]);
                        assert_eq!(
                            denominator.as_slice()[offset + row],
                            challenge - position::embed::<B>(entry)
                        );
                    }
                }
                BlockRole::Table => {
                    for entry in 0..1 << fixture.table_variables {
                        assert_eq!(
                            numerator.as_slice()[offset + entry],
                            built.pushforwards[0][entry]
                        );
                        // Negated, which is what makes an honest statement sum to zero.
                        assert_eq!(
                            denominator.as_slice()[offset + entry],
                            position::embed::<B>(entry) - challenge
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn padding_carries_a_zero_over_a_one() {
        // Fixture state: one reader of two rows over a table of four entries.
        //
        //     used leaves   : 2 + 4 = 6
        //     padded table  : 8
        //     padding       : leaves 6 and 7
        //
        // A padded leaf must add nothing to the sum and divide by nothing.
        let mut rng = SmallRng::seed_from_u64(0x0_9AD0);
        let fixture = Fixture::new(&mut rng, 2, vec![vec![1, 2]]);

        let readers = fixture.readers();
        let lookups = [TableLookup {
            num_variables: 2,
            readers: &readers,
        }];
        let plan = LogupStarPlan::new(&lookups);
        assert_eq!(plan.num_variables, 3);

        let reader_witnesses = fixture.reader_witnesses();
        let columns = [fixture.column.as_slice()];
        let witness = [TableWitness {
            columns: &columns,
            readers: &reader_witnesses,
        }];
        let built = weights(&plan, &lookups, &witness, B::ONE);
        let (numerator, denominator) =
            leaf_tables(&plan, &witness, &built, &[B::interpolation_node(9)]);

        for leaf in 6..8 {
            assert_eq!(numerator.as_slice()[leaf], B::ZERO);
            assert_eq!(denominator.as_slice()[leaf], B::ONE);
        }
    }

    #[test]
    fn an_honest_leaf_table_sums_to_zero() {
        // This is the identity the whole reduction rests on.
        //
        // Reader and table fractions carry opposite signs.
        //
        // An honest pushforward makes them cancel entry by entry.
        let mut rng = SmallRng::seed_from_u64(0x0_2E20);
        let fixture = Fixture::new(&mut rng, 2, vec![vec![0, 3, 3, 1], vec![2, 0]]);
        let batching = B::interpolation_node(6);

        let readers = fixture.readers();
        let lookups = [TableLookup {
            num_variables: 2,
            readers: &readers,
        }];
        let plan = LogupStarPlan::new(&lookups);
        let reader_witnesses = fixture.reader_witnesses();
        let columns = [fixture.column.as_slice()];
        let witness = [TableWitness {
            columns: &columns,
            readers: &reader_witnesses,
        }];

        let built = weights(&plan, &lookups, &witness, batching);
        let (numerator, denominator) =
            leaf_tables(&plan, &witness, &built, &[B::interpolation_node(13)]);

        let sum = iter::zip(numerator.as_slice(), denominator.as_slice())
            .map(|(&numerator, &denominator)| numerator * denominator.inverse())
            .sum::<B>();
        assert_eq!(sum, B::ZERO);
    }
}
