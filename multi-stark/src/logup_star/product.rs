//! The sumcheck that binds every table to its pushforward.
//!
//! Each table owes one claim per column:
//!
//! ```text
//!     sum_v Y(v) * T_k(v) = e_k
//! ```
//!
//! Every column of one table is read through the same pushforward.
//!
//! So one challenge folds the columns together and the table owes a single product claim.
//!
//! Tables need not agree on a size.
//!
//! A smaller one is summed against an equality factor on the leading variables.
//!
//! That factor sums to one over the cube, so the table's own claim is untouched.
//!
//! Leading variables are bound first.
//!
//! A small table contributes only a scalar until the reduction reaches its own.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;

use super::TableLookup;

/// One table's share of the batched sumcheck.
struct TableState<EF> {
    /// Leading rounds during which this table contributes only a scalar.
    padding_rounds: usize,
    /// Equality weight of the padding coordinates bound so far.
    weight: EF,
    /// The table's pushforward, folded once the padding rounds are past.
    pushforward: Poly<EF>,
    /// The table's columns under one challenge, folded alongside the pushforward.
    columns: Poly<EF>,
    /// Inner product of the two tables above, which does not move while padding.
    sum: EF,
}

/// Prover state for the sumcheck binding every table to its pushforward.
pub(crate) struct ProductProver<EF> {
    /// One entry per table, in statement order.
    tables: Vec<TableState<EF>>,
    /// Variables bound so far.
    round: usize,
}

impl<EF: Field> ProductProver<EF> {
    /// Build the state from each table's pushforward and its columns.
    ///
    /// # Arguments
    ///
    /// - `num_rounds`: variable count of the widest table, which every table is padded up to.
    /// - `pushforwards`: one pushforward per table, in statement order.
    /// - `columns`: one column list per table, in statement order.
    /// - `batching`: the challenge separating one column claim from the next.
    ///
    /// # Returns
    ///
    /// The prover state, and the sum it starts from.
    pub(crate) fn new<F: Field>(
        num_rounds: usize,
        pushforwards: &[Vec<EF>],
        columns: &[&[&[F]]],
        batching: EF,
    ) -> (Self, EF)
    where
        EF: ExtensionField<F>,
    {
        // Every column claim in the reduction earns its own power, tables in statement order.
        let mut power = EF::ONE;

        let tables = pushforwards
            .iter()
            .zip(columns)
            .map(|(pushforward, columns)| {
                let num_entries = pushforward.len();

                // Fold the columns together under the batching challenge.
                //
                // All of a table's columns are read through one pushforward.
                //
                // Folding them first turns its several product claims into a single one.
                let mut combined = EF::zero_vec(num_entries);
                for column in *columns {
                    assert_eq!(column.len(), num_entries, "a column must cover every entry");
                    combined
                        .par_iter_mut()
                        .zip(column.par_iter())
                        .for_each(|(combined, &value)| *combined += power * value);
                    power *= batching;
                }

                let sum = pushforward
                    .par_iter()
                    .zip(combined.par_iter())
                    .par_fold_reduce(
                        || EF::ZERO,
                        |acc, (&weight, &column)| acc + weight * column,
                        |left, right| left + right,
                    );

                TableState {
                    padding_rounds: num_rounds - log2_of(num_entries),
                    weight: EF::ONE,
                    pushforward: Poly::new(pushforward.clone()),
                    columns: Poly::new(combined),
                    sum,
                }
            })
            .collect::<Vec<_>>();

        // The padding factor sums to one over the cube, so a padded table still owes its own sum.
        let claimed_sum = tables.iter().map(|table| table.sum).sum();

        (Self { tables, round: 0 }, claimed_sum)
    }
}

/// Base-two logarithm of a power-of-two length.
const fn log2_of(length: usize) -> usize {
    length.trailing_zeros() as usize
}

impl<EF: Field> RoundProver<EF> for ProductProver<EF> {
    fn round_poly(&self) -> Vec<EF> {
        // The summand is a product of two multilinears, so the round polynomial is quadratic.
        //
        // Two values fix it, once the running sum supplies the third.
        let node = EF::interpolation_node(2);

        self.tables
            .iter()
            .fold([EF::ZERO; 2], |acc, table| {
                let [at_zero, at_node] = if self.round < table.padding_rounds {
                    // Still padding, so the equality factor is all that moves.
                    //
                    // The table's own sum rides along unchanged.
                    [table.sum, (EF::ONE - node) * table.sum]
                } else {
                    let half = table.pushforward.num_evals() / 2;
                    let (y_low, y_high) = table.pushforward.as_slice().split_at(half);
                    let (c_low, c_high) = table.columns.as_slice().split_at(half);

                    // One pass over the halves accumulates both values of the line.
                    y_low
                        .par_iter()
                        .zip(y_high.par_iter())
                        .zip(c_low.par_iter().zip(c_high.par_iter()))
                        .par_fold_reduce(
                            || [EF::ZERO; 2],
                            |mut acc, ((&y0, &y1), (&c0, &c1))| {
                                acc[0] += y0 * c0;
                                acc[1] += (y0 + node * (y1 - y0)) * (c0 + node * (c1 - c0));
                                acc
                            },
                            |mut left, right| {
                                left.iter_mut().zip(right).for_each(|(l, r)| *l += r);
                                left
                            },
                        )
                };

                [
                    acc[0] + table.weight * at_zero,
                    acc[1] + table.weight * at_node,
                ]
            })
            .to_vec()
    }

    fn fold(&mut self, r: EF) {
        for table in &mut self.tables {
            if self.round < table.padding_rounds {
                // Binding a padding variable only scales what the table contributes.
                table.weight *= EF::ONE - r;
            } else {
                table.pushforward.fix_prefix_var_mut(r);
                table.columns.fix_prefix_var_mut(r);
            }
        }
        self.round += 1;
    }
}

/// The sum the statement says the product sumcheck must start from.
///
/// Readers of one table are weighted by powers of the first challenge.
///
/// Every column claim in the reduction is weighted by powers of the second.
pub(crate) fn claimed_sum<EF: Field>(
    lookups: &[TableLookup<'_, EF>],
    reader_batching: EF,
    column_batching: EF,
) -> EF {
    let mut power = EF::ONE;
    let mut total = EF::ZERO;

    for lookup in lookups {
        for column in 0..lookup.width() {
            // Combine this table's readers on this column.
            //
            // Then give the result its own place in the column batching.
            let combined = lookup
                .readers
                .iter()
                .zip(reader_batching.powers())
                .map(|(reader, scale)| scale * reader.claims[column])
                .sum::<EF>();
            total += power * combined;
            power *= column_batching;
        }
    }

    total
}

/// Rebuild the value the sumcheck reduces to, from the claims a verifier holds.
///
/// Each table contributes its padding weight times its pushforward times its columns.
///
/// All three are taken at the point the reduction landed on.
///
/// # Arguments
///
/// - `point`: where the sumcheck ended, most significant coordinate first.
/// - `pushforwards`: one pushforward per table, which the verifier evaluates itself.
/// - `column_claims`: one claim per column, tables in statement order.
/// - `batching`: the challenge separating one column claim from the next.
pub(crate) fn final_value<F, EF>(
    point: &Point<EF>,
    pushforwards: &[Vec<EF>],
    column_claims: &[Vec<EF>],
    batching: EF,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let num_rounds = point.num_variables();
    let mut power = EF::ONE;

    pushforwards
        .iter()
        .zip(column_claims)
        .map(|(pushforward, claims)| {
            let num_variables = log2_of(pushforward.len());
            let padding_rounds = num_rounds - num_variables;

            // The padding coordinates are the leading ones, each weighted as if it were zero.
            let weight = point.as_slice()[..padding_rounds]
                .iter()
                .map(|&coordinate| EF::ONE - coordinate)
                .product::<EF>();

            // The table's own coordinates are what is left.
            let own = point.get_subpoint_over_range(padding_rounds..num_rounds);
            let pushforward = Poly::new(pushforward.as_slice()).eval_ext::<F>(&own);

            // The columns were folded under the batching challenge, so their claims are too.
            let columns = claims
                .iter()
                .map(|&claim| {
                    let weighted = power * claim;
                    power *= batching;
                    weighted
                })
                .sum::<EF>();

            weight * pushforward * columns
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::iter;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type B = BinaryField128;

    /// One table as the oracle sees it: its pushforward and its columns.
    struct Table {
        pushforward: Vec<B>,
        columns: Vec<Vec<B>>,
    }

    /// The columns of one table folded under the batching challenge.
    ///
    /// This is the definition the prover is checked against, written out directly.
    fn combine(table: &Table, first_power: B, batching: B) -> Vec<B> {
        let mut combined = B::zero_vec(table.pushforward.len());
        let mut power = first_power;
        for column in &table.columns {
            for (combined, &value) in combined.iter_mut().zip(column) {
                *combined += power * value;
            }
            power *= batching;
        }
        combined
    }

    /// What one table contributes at one full point of the padded cube.
    ///
    /// Padding coordinates are the leading ones, each weighted as if it were zero.
    fn summand(table: &Table, combined: &[B], num_rounds: usize, point: &[B]) -> B {
        let num_variables = log2_of(table.pushforward.len());
        let padding = num_rounds - num_variables;

        let weight = point[..padding]
            .iter()
            .map(|&coordinate| B::ONE - coordinate)
            .product::<B>();
        let own = Point::new(point[padding..].to_vec());

        weight
            * Poly::new(table.pushforward.clone()).eval_ext::<B>(&own)
            * Poly::new(combined.to_vec()).eval_ext::<B>(&own)
    }

    /// The batched sum with the leading variables bound and the next one set.
    ///
    /// Everything below is enumerated over the cube, so nothing here is folded.
    fn oracle(
        tables: &[Table],
        combined: &[Vec<B>],
        num_rounds: usize,
        bound: &[B],
        value: Option<B>,
    ) -> B {
        let fixed = bound.len() + usize::from(value.is_some());
        let free = num_rounds - fixed;

        (0..1usize << free)
            .map(|suffix| {
                let mut point = bound.to_vec();
                point.extend(value);
                point.extend(Point::<B>::hypercube(suffix, free).as_slice());

                iter::zip(tables, combined)
                    .map(|(table, combined)| summand(table, combined, num_rounds, &point))
                    .sum::<B>()
            })
            .sum()
    }

    /// Two tables of different heights and widths, so the padding path is exercised.
    fn fixture(rng: &mut SmallRng) -> Vec<Table> {
        vec![
            Table {
                pushforward: Poly::<B>::rand(rng, 2).as_slice().to_vec(),
                columns: vec![
                    Poly::<B>::rand(rng, 2).as_slice().to_vec(),
                    Poly::<B>::rand(rng, 2).as_slice().to_vec(),
                ],
            },
            Table {
                pushforward: Poly::<B>::rand(rng, 1).as_slice().to_vec(),
                columns: vec![Poly::<B>::rand(rng, 1).as_slice().to_vec()],
            },
        ]
    }

    #[test]
    fn the_starting_sum_is_the_batched_inner_product() {
        // The verifier rebuilds this sum from the statement.
        //
        // A prover starting anywhere else lines up with nothing downstream.
        let mut rng = SmallRng::seed_from_u64(0x0D_0C01);
        let tables = fixture(&mut rng);
        let batching = B::interpolation_node(7);

        let mut power = B::ONE;
        let combined = tables
            .iter()
            .map(|table| {
                let combined = combine(table, power, batching);
                power *= batching.exp_u64(table.columns.len() as u64);
                combined
            })
            .collect::<Vec<_>>();

        let columns = tables
            .iter()
            .map(|table| table.columns.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let views = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let pushforwards = tables
            .iter()
            .map(|table| table.pushforward.clone())
            .collect::<Vec<_>>();

        let (_, claimed_sum) = ProductProver::new::<B>(2, &pushforwards, &views, batching);
        assert_eq!(claimed_sum, oracle(&tables, &combined, 2, &[], None));
    }

    #[test]
    fn every_round_polynomial_matches_the_definition() {
        // The round polynomial is where padding, folding and the quadratic all meet.
        //
        // It is checked here against the same sum written out longhand.
        //
        // Fixture state: tables of 2^2 and 2^1 entries.
        //
        // The smaller one spends the first round contributing nothing but a scalar.
        let mut rng = SmallRng::seed_from_u64(0x0D_0C02);
        let tables = fixture(&mut rng);
        let batching = B::interpolation_node(11);
        let num_rounds = 2;

        let mut power = B::ONE;
        let combined = tables
            .iter()
            .map(|table| {
                let combined = combine(table, power, batching);
                power *= batching.exp_u64(table.columns.len() as u64);
                combined
            })
            .collect::<Vec<_>>();

        let columns = tables
            .iter()
            .map(|table| table.columns.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let views = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let pushforwards = tables
            .iter()
            .map(|table| table.pushforward.clone())
            .collect::<Vec<_>>();

        let (mut prover, claimed_sum) =
            ProductProver::new::<B>(num_rounds, &pushforwards, &views, batching);

        let node = B::interpolation_node(2);
        let mut bound = Vec::new();
        let mut running = claimed_sum;

        for round in 0..num_rounds {
            let sent = prover.round_poly();
            let at_zero = oracle(&tables, &combined, num_rounds, &bound, Some(B::ZERO));
            let at_node = oracle(&tables, &combined, num_rounds, &bound, Some(node));

            assert_eq!(sent, vec![at_zero, at_node], "round {round}");

            // The verifier recovers the value at one from the running sum.
            //
            // So the two halves of the cube must add up to it.
            let at_one = oracle(&tables, &combined, num_rounds, &bound, Some(B::ONE));
            assert_eq!(at_zero + at_one, running, "round {round}");

            let challenge = B::interpolation_node(round + 3);
            running = oracle(&tables, &combined, num_rounds, &bound, Some(challenge));
            bound.push(challenge);
            prover.fold(challenge);
        }

        // The sumcheck ends on the value the verifier rebuilds from the claims it holds.
        let point = Point::new(bound);
        let column_claims = tables
            .iter()
            .map(|table| {
                let num_variables = log2_of(table.pushforward.len());
                let own = point.get_subpoint_over_range(num_rounds - num_variables..num_rounds);
                table
                    .columns
                    .iter()
                    .map(|column| Poly::new(column.clone()).eval_ext::<B>(&own))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            running,
            final_value::<B, B>(&point, &pushforwards, &column_claims, batching)
        );
    }

    #[test]
    fn a_padded_table_contributes_its_whole_sum() {
        // Padding multiplies a table by an equality factor on the variables it does not have.
        //
        // That factor sums to one over the cube.
        //
        // A short table still owes exactly its own inner product.
        let mut rng = SmallRng::seed_from_u64(0x0D_0C03);
        let short = Table {
            pushforward: Poly::<B>::rand(&mut rng, 1).as_slice().to_vec(),
            columns: vec![Poly::<B>::rand(&mut rng, 1).as_slice().to_vec()],
        };

        let own_sum = iter::zip(&short.pushforward, &short.columns[0])
            .map(|(&weight, &column)| weight * column)
            .sum::<B>();

        let columns = vec![short.columns[0].as_slice()];
        let views = vec![columns.as_slice()];
        let pushforwards = vec![short.pushforward];

        // Padded up to three variables, four more than it has entries for.
        let (_, claimed_sum) = ProductProver::new::<B>(3, &pushforwards, &views, B::ONE);
        assert_eq!(claimed_sum, own_sum);
    }
}
