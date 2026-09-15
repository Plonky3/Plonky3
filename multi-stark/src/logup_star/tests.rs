use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::generic_degree::RoundProver;
use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::plan::BlockRole;
use super::product::{self, ProductProver};
use super::prover::PRODUCT_DEGREE;
use super::transcript::{LogupStarProverTranscript, LogupStarShape};
use super::witness::Weights;
use super::{
    LogupStarError, LogupStarPlan, LogupStarProof, Reader, ReaderWitness, TableLookup,
    TableWitness, position,
};
use crate::fractional_gkr::{Fraction, LeafNumerator, prove_fractional_gkr};

type Small = BabyBear;
type SmallExt = BinomialExtensionField<Small, 4>;
type SmallChallenger = DuplexChallenger<Small, Poseidon2BabyBear<16>, 16, 8>;

type Binary = BinaryField128;
type BinaryChal = BinaryChallenger<Binary, HashChallenger<u8, Keccak256Hash, 32>>;

fn small_challenger() -> SmallChallenger {
    // Fixed seed so both sides of a test walk the same sponge.
    let mut rng = SmallRng::seed_from_u64(0x1065_7A20);
    SmallChallenger::new(Poseidon2BabyBear::new_from_rng_128(&mut rng))
}

fn binary_challenger() -> BinaryChal {
    BinaryChal::from_hasher(b"p3-logup-star-test".to_vec(), Keccak256Hash)
}

/// One table's shape: its entry count, its width, and how tall each of its readers is.
type Spec<'a> = &'a [(usize, usize, &'a [usize])];

/// Everything one test reduction owns, so the statement and the witness can borrow from it.
struct Instance<F, EF> {
    /// Base-two logarithm of each table's entry count.
    table_variables: Vec<usize>,
    /// Columns of each table, grouped by table.
    columns: Vec<Vec<Vec<F>>>,
    /// Claim point of each reader, grouped by table.
    points: Vec<Vec<Point<EF>>>,
    /// Entry each reader row names, grouped by table.
    positions: Vec<Vec<Vec<usize>>>,
    /// Value each reader pulled, grouped by table, then reader, then column.
    claims: Vec<Vec<Vec<EF>>>,
}

impl<F: Field, EF: ExtensionField<F>> Instance<F, EF>
where
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    /// Draw a random honest instance matching the shape.
    fn random(rng: &mut SmallRng, spec: Spec<'_>) -> Self {
        let mut instance = Self {
            table_variables: spec.iter().map(|&(entries, ..)| entries).collect(),
            columns: Vec::new(),
            points: Vec::new(),
            positions: Vec::new(),
            claims: Vec::new(),
        };

        for &(table_variables, width, readers) in spec {
            let num_entries = 1 << table_variables;

            // Table entries are unconstrained, so random values exercise the general case.
            instance.columns.push(
                (0..width)
                    .map(|_| Poly::<F>::rand(rng, table_variables).as_slice().to_vec())
                    .collect(),
            );

            // Each reader names a random entry on every one of its rows.
            instance.positions.push(
                readers
                    .iter()
                    .map(|&reader_variables| {
                        (0..1 << reader_variables)
                            .map(|_| rng.random_range(0..num_entries))
                            .collect()
                    })
                    .collect(),
            );
            instance.points.push(
                readers
                    .iter()
                    .map(|&reader_variables| Point::<EF>::rand(rng, reader_variables))
                    .collect(),
            );
            instance.claims.push(Vec::new());
        }

        // The claims are what the reduction is about, so they come from the witness itself.
        instance.recompute_claims();
        instance
    }

    /// Recompute every claim from the current tables and positions.
    fn recompute_claims(&mut self) {
        for table in 0..self.columns.len() {
            self.claims[table] = (0..self.positions[table].len())
                .map(|reader| {
                    self.columns[table]
                        .iter()
                        .map(|column| {
                            pulled_claim(
                                column,
                                &self.positions[table][reader],
                                &self.points[table][reader],
                            )
                        })
                        .collect()
                })
                .collect();
        }
    }

    fn readers(&self) -> Vec<Vec<Reader<'_, EF>>> {
        self.points
            .iter()
            .zip(&self.claims)
            .map(|(points, claims)| {
                points
                    .iter()
                    .zip(claims)
                    .map(|(point, claims)| Reader { point, claims })
                    .collect()
            })
            .collect()
    }

    fn lookups<'a>(&self, readers: &'a [Vec<Reader<'a, EF>>]) -> Vec<TableLookup<'a, EF>> {
        self.table_variables
            .iter()
            .zip(readers)
            .map(|(&num_variables, readers)| TableLookup {
                num_variables,
                readers,
            })
            .collect()
    }

    fn column_views(&self) -> Vec<Vec<&[F]>> {
        self.columns
            .iter()
            .map(|columns| columns.iter().map(Vec::as_slice).collect())
            .collect()
    }

    fn reader_witnesses(&self) -> Vec<Vec<ReaderWitness<'_>>> {
        self.positions
            .iter()
            .map(|positions| {
                positions
                    .iter()
                    .map(|positions| ReaderWitness { positions })
                    .collect()
            })
            .collect()
    }

    fn witness<'a>(
        columns: &'a [Vec<&'a [F]>],
        readers: &'a [Vec<ReaderWitness<'a>>],
    ) -> Vec<TableWitness<'a, F>> {
        columns
            .iter()
            .zip(readers)
            .map(|(columns, readers)| TableWitness { columns, readers })
            .collect()
    }
}

/// The value one reader pulled out of one column, computed straight from the witness.
///
/// This is the multilinear extension of the pulled vector.
///
/// Discharging it without ever materializing that vector is the whole point of the reduction.
fn pulled_claim<F: Field, EF: ExtensionField<F>>(
    column: &[F],
    positions: &[usize],
    point: &Point<EF>,
) -> EF {
    let pulled = positions
        .iter()
        .map(|&entry| EF::from(column[entry]))
        .collect::<Vec<_>>();
    Poly::new(pulled).eval_ext::<F>(point)
}

/// Sum a table of fractions directly, with no reduction involved.
fn fraction_sum<EF: Field>(numerator: &[EF], denominator: &[EF]) -> EF {
    numerator
        .iter()
        .zip(denominator)
        .map(|(&numerator, &denominator)| numerator * denominator.inverse())
        .sum()
}

/// Run one honest reduction end to end and check every claim it hands back.
fn round_trip<F, EF, Challenger>(spec: Spec<'_>, seed: u64, challenger: impl Fn() -> Challenger)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    let instance = Instance::<F, EF>::random(&mut rng, spec);

    let readers = instance.readers();
    let lookups = instance.lookups(&readers);
    let columns = instance.column_views();
    let reader_witnesses = instance.reader_witnesses();
    let witness = Instance::<F, EF>::witness(&columns, &reader_witnesses);

    let mut prover_challenger = challenger();
    let (proof, prover_output) = LogupStarProof::prove(&lookups, &witness, &mut prover_challenger);

    let mut verifier_challenger = challenger();
    let verifier_output = proof
        .verify(&lookups, &mut verifier_challenger)
        .expect("an honest reduction verifies");
    assert_eq!(verifier_output, prover_output);

    // Whatever runs after the reduction must find one shared sponge state.
    let prover_next: EF = prover_challenger.sample_algebra_element();
    let verifier_next: EF = verifier_challenger.sample_algebra_element();
    assert_eq!(prover_next, verifier_next);

    // The reduction is worth nothing unless the claims it leaves are the true evaluations.
    let plan = LogupStarPlan::new::<F, EF>(&lookups);
    for (table, output) in verifier_output.tables.iter().enumerate() {
        // A table is claimed at the last coordinates of the shared table point.
        let table_variables = instance.table_variables[table];
        let own = verifier_output.table_point.get_subpoint_over_range(
            plan.max_table_variables - table_variables..plan.max_table_variables,
        );
        for (column, &claim) in instance.columns[table].iter().zip(&output.column_claims) {
            let lifted = column.iter().copied().map(EF::from).collect::<Vec<_>>();
            assert_eq!(Poly::new(lifted).eval_ext::<F>(&own), claim);
        }

        // A reader is claimed at the last coordinates of the shared position point.
        for (reader, &claim) in output.position_claims.iter().enumerate() {
            let reader_variables = instance.points[table][reader].num_variables();
            let own = verifier_output.position_point.get_subpoint_over_range(
                plan.max_reader_variables - reader_variables..plan.max_reader_variables,
            );
            let embedded = instance.positions[table][reader]
                .iter()
                .map(|&entry| EF::from(position::embed::<F>(entry)))
                .collect::<Vec<_>>();
            assert_eq!(Poly::new(embedded).eval_ext::<F>(&own), claim);
        }
    }
}

#[test]
fn round_trip_over_a_binary_field() {
    // The characteristic-two target: one table of eight entries, two columns, two readers.
    round_trip::<Binary, Binary, _>(&[(3, 2, &[4, 3])], 1, binary_challenger);
}

#[test]
fn round_trip_over_a_prime_field() {
    // The same statement in odd characteristic, since the reduction is field-generic.
    round_trip::<Small, SmallExt, _>(&[(3, 2, &[4, 3])], 1, small_challenger);
}

#[test]
fn round_trip_over_several_tables_of_unequal_size() {
    // Unequal heights exercise the padding of both the layout and the product sumcheck.
    //
    // The single-reader table exercises the degenerate batching case.
    let spec: Spec<'_> = &[(4, 1, &[5, 2]), (2, 3, &[3]), (1, 1, &[1])];
    round_trip::<Binary, Binary, _>(spec, 2, binary_challenger);
    round_trip::<Small, SmallExt, _>(spec, 2, small_challenger);
}

#[test]
fn round_trip_when_a_reader_is_shorter_than_its_table() {
    // A reader with fewer rows than the table has entries leaves most of the pushforward zero.
    round_trip::<Binary, Binary, _>(&[(4, 1, &[1])], 3, binary_challenger);
}

#[test]
fn reading_one_entry_an_even_number_of_times_does_not_cancel() {
    // This is the failure that rules a logarithmic-derivative lookup out over a binary field.
    //
    // There an entry carries an integer count, read back modulo the characteristic.
    //
    // A count of two is then indistinguishable from a count of zero.
    //
    // Fixture state: four entries, one reader of four rows, naming 1, 0, 1, 0.
    //
    // A point is most significant coordinate first.
    //
    // So the rows naming entry 1 are those whose low bit is clear:
    //
    //     weight over entry 1 = eq_r(00) + eq_r(10) = (1 - r_0)(1 - r_1) + r_0 (1 - r_1)
    //                         = 1 - r_1
    //     weight over entry 0 = eq_r(01) + eq_r(11) = r_1
    //
    // Both are field elements fixed by the point, not counts, so neither vanishes.
    let mut rng = SmallRng::seed_from_u64(0x0E4E_0001);
    let point = Point::<Binary>::rand(&mut rng, 2);

    let readers = vec![Reader {
        point: &point,
        claims: &[Binary::ONE],
    }];
    let lookups = vec![TableLookup {
        num_variables: 2,
        readers: &readers,
    }];
    let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);

    let positions = vec![1usize, 0, 1, 0];
    let reader_witnesses = vec![ReaderWitness {
        positions: &positions,
    }];
    let column = vec![Binary::ONE; 4];
    let columns = vec![column.as_slice()];
    let witness = vec![TableWitness {
        columns: &columns,
        readers: &reader_witnesses,
    }];

    let weights = Weights::build(&plan, &lookups, &witness, Binary::ONE);
    let pushforward = &weights.pushforwards[0];

    assert_eq!(pushforward[1], Binary::ONE - point[1]);
    assert_eq!(pushforward[0], point[1]);
    assert_ne!(pushforward[1], Binary::ZERO);

    // Entries nobody read stay empty.
    assert_eq!(pushforward[2], Binary::ZERO);
    assert_eq!(pushforward[3], Binary::ZERO);
}

#[test]
fn one_challenge_per_table_is_what_stops_two_tables_cancelling() {
    // Fixture state: two tables of four entries, one reader of two rows each.
    //
    // Mutation: move a unit of weight off one table's entry 1 and onto the other's.
    //
    //     honest   : Y_0[1] = y      Y_1[1] = z
    //     cheating : Y_0[1] = y - 1  Y_1[1] = z + 1
    //
    // Both table sides enter the sum negated, so the two errors are
    //
    //     -1 / (iota(1) - c_0)   and   +1 / (iota(1) - c_1)
    //
    // which cancel for every value of a shared challenge, and for no other pair.
    //
    // Integer constants are useless here.
    //
    // Characteristic two collapses them onto zero and one.
    //
    // The challenges are drawn from the field's own enumeration instead.
    let mut rng = SmallRng::seed_from_u64(0x0C0F_FEE0);
    let points = [
        Point::<Binary>::rand(&mut rng, 1),
        Point::<Binary>::rand(&mut rng, 1),
    ];
    let readers = points
        .iter()
        .map(|point| {
            vec![Reader {
                point,
                claims: &[Binary::ONE],
            }]
        })
        .collect::<Vec<_>>();
    let lookups = readers
        .iter()
        .map(|readers| TableLookup {
            num_variables: 2,
            readers,
        })
        .collect::<Vec<_>>();
    let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);

    let positions = [vec![0usize, 1], vec![1usize, 2]];
    let reader_witnesses = positions
        .iter()
        .map(|positions| vec![ReaderWitness { positions }])
        .collect::<Vec<_>>();
    let column = vec![Binary::ONE; 4];
    let columns = [vec![column.as_slice()], vec![column.as_slice()]];
    let witness = columns
        .iter()
        .zip(&reader_witnesses)
        .map(|(columns, readers)| TableWitness { columns, readers })
        .collect::<Vec<_>>();

    let mut weights = Weights::build(&plan, &lookups, &witness, Binary::ONE);

    // An honest run sums to zero whatever the challenges are.
    let distinct = [Binary::interpolation_node(5), Binary::interpolation_node(9)];
    let (numerator, denominator) = weights.leaf_tables(&plan, &witness, &distinct);
    assert_eq!(
        fraction_sum(numerator.as_slice(), denominator.as_slice()),
        Binary::ZERO
    );

    weights.pushforwards[0][1] -= Binary::ONE;
    weights.pushforwards[1][1] += Binary::ONE;

    // One challenge per table puts the two errors over different denominators.
    let (numerator, denominator) = weights.leaf_tables(&plan, &witness, &distinct);
    assert_ne!(
        fraction_sum(numerator.as_slice(), denominator.as_slice()),
        Binary::ZERO,
        "distinct challenges must expose weight moved between tables"
    );

    // One shared challenge puts them over the same denominator, where they cancel exactly.
    let shared = [Binary::interpolation_node(5); 2];
    let (numerator, denominator) = weights.leaf_tables(&plan, &witness, &shared);
    assert_eq!(
        fraction_sum(numerator.as_slice(), denominator.as_slice()),
        Binary::ZERO,
        "a shared challenge is exactly what would let the two errors cancel"
    );
}

#[test]
#[should_panic(expected = "fraction GKR input fractions must sum to zero")]
fn weight_moved_between_two_tables_cannot_be_hidden() {
    // The companion test above shows the algebra.
    //
    // Under one shared challenge the two errors sit over the same denominator and cancel.
    //
    // Under one challenge per table they do not.
    //
    // This one drives the real prover, so it also guards the code that keeps them apart.
    //
    // Fixture state: two tables of four entries, one reader of two rows each.
    //
    // Mutation: move a unit of weight off one table's entry 1 and onto the other's.
    //
    // It moves after the pushforwards are built and before they are bound.
    //
    // The fractions then no longer sum to zero and the reduction refuses to run.
    //
    // Under a shared challenge the sum would vanish and the cheat would go through.
    let mut rng = SmallRng::seed_from_u64(0x0_C4EA7);
    let instance = Instance::<Binary, Binary>::random(&mut rng, &[(2, 1, &[1]), (2, 1, &[1])]);

    let readers = instance.readers();
    let lookups = instance.lookups(&readers);
    let columns = instance.column_views();
    let reader_witnesses = instance.reader_witnesses();
    let witness = Instance::<Binary, Binary>::witness(&columns, &reader_witnesses);

    let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);
    let shape = LogupStarShape::new(&plan);
    let mut sponge = binary_challenger();
    let mut transcript = LogupStarProverTranscript::<_, Binary, Binary>::new(&mut sponge, &shape);

    transcript.statement(&TableLookup::statement(&lookups));
    let reader_batching = transcript.reader_batching();
    let mut built = Weights::build(&plan, &lookups, &witness, reader_batching);

    built.pushforwards[0][1] -= Binary::ONE;
    built.pushforwards[1][1] += Binary::ONE;

    for pushforward in &built.pushforwards {
        transcript.pushforward(pushforward);
    }
    let entry_challenges = transcript.entry_challenges(plan.num_tables());

    let (numerator, denominator) = built.leaf_tables(&plan, &witness, &entry_challenges);
    let numerator = PolyMaybePacked::Scalar(numerator);
    let denominator = PolyMaybePacked::Scalar(denominator);
    transcript.fraction_reduction(|challenger| {
        prove_fractional_gkr(
            Fraction {
                n: LeafNumerator::Ext(&numerator),
                d: &denominator,
            },
            challenger,
        )
    });
}

#[test]
#[should_panic(expected = "a row names an entry outside its table")]
fn a_position_outside_the_table_is_rejected_while_scattering() {
    // A position column is fixed before the claim point is drawn.
    //
    // An entry no table has leaves a pole the table side cannot match.
    //
    // The identity fails on its own, so no range constraint is needed.
    //
    // The prover catches it first, while scattering.
    let point = Point::<Binary>::new(vec![Binary::ONE]);
    let readers = vec![Reader {
        point: &point,
        claims: &[Binary::ONE],
    }];
    let lookups = vec![TableLookup {
        num_variables: 2,
        readers: &readers,
    }];
    let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);

    // Entry four is one past the last entry of a four-entry table.
    let positions = vec![0usize, 4];
    let reader_witnesses = vec![ReaderWitness {
        positions: &positions,
    }];
    let column = vec![Binary::ONE; 4];
    let columns = vec![column.as_slice()];
    let witness = vec![TableWitness {
        columns: &columns,
        readers: &reader_witnesses,
    }];

    let _ = Weights::build(&plan, &lookups, &witness, Binary::ONE);
}

#[test]
fn no_pushforward_can_close_the_pole_an_out_of_range_read_opens() {
    // Fixture state: one four-entry table, one reader of two rows, rows naming 0 and 4.
    //
    // Entry 4 is one past the last entry.
    //
    // The reader's second fraction therefore sits over the pole `c - iota(4)`.
    //
    // Every table fraction sits over some `iota(e) - c` with `e < 4`.
    //
    //     reader:   w_0 / (c - iota(0))    w_1 / (c - iota(4))   <- nothing sits here
    //     table:   -Y_e / (c - iota(e))    for e in 0, 1, 2, 3
    //
    // The embedding is injective, so no table entry reaches the second pole.
    //
    // The pushforward is the only part of this a prover chooses.
    //
    // Two scatters stand in for every choice.
    //
    // One drops the offending row's weight.
    //
    // The other moves it onto entry 0, where it would cancel if the poles were confused.
    //
    // Neither sum vanishes, which is why a position column carries no range constraint.
    let mut rng = SmallRng::seed_from_u64(0x004A_1101);
    let point = Point::<Binary>::rand(&mut rng, 1);
    let challenge: Binary = rng.random();

    let readers = vec![Reader {
        point: &point,
        claims: &[Binary::ONE],
    }];
    let lookups = vec![TableLookup {
        num_variables: 2,
        readers: &readers,
    }];
    let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);

    let positions = vec![0usize, 4];
    let reader_witnesses = vec![ReaderWitness {
        positions: &positions,
    }];
    let column = vec![Binary::ONE; 4];
    let columns = vec![column.as_slice()];
    let witness = vec![TableWitness {
        columns: &columns,
        readers: &reader_witnesses,
    }];

    // The equality weights of the two rows, which no prover chooses.
    let row_weights = Poly::<Binary>::new_from_point(point.as_slice(), Binary::ONE);
    let [first, second] = [row_weights.as_slice()[0], row_weights.as_slice()[1]];

    for scatter in [
        [first, Binary::ZERO, Binary::ZERO, Binary::ZERO],
        [first + second, Binary::ZERO, Binary::ZERO, Binary::ZERO],
    ] {
        let built = Weights {
            readers: vec![Poly::<Binary>::new_from_point(
                point.as_slice(),
                Binary::ONE,
            )],
            pushforwards: vec![scatter.to_vec()],
        };
        let (numerator, denominator) = built.leaf_tables(&plan, &witness, &[challenge]);

        assert_ne!(
            fraction_sum(numerator.as_slice(), denominator.as_slice()),
            Binary::ZERO,
        );
    }
}

#[test]
fn an_out_of_range_read_hidden_behind_a_zero_weight_is_rejected() {
    // The companion test above shows such fractions never sum to zero.
    //
    // The reduction refuses to run on a sum that does not vanish.
    //
    // So the only proof a prover can build for such a read lies about the numerator side.
    //
    // This test drives the real verifier against that proof.
    //
    // Fixture state: one four-entry table of one column, one reader of two rows.
    //
    // The rows name 0 and 4, and entry 4 is one past the table.
    //
    // Mutation: the offending row is given weight zero, on both sides.
    //
    // That is what makes the leaf fractions sum to zero.
    //
    //     honest row weights   [w_0, w_1]
    //     what is proved       [w_0,   0]
    //
    // Everything else is built honestly.
    //
    // The position claim still opens a column naming entry 4.
    //
    // A verifier rebuilds every reader weight from the claim point alone.
    //
    // The forged numerator is caught where the reduction is tied back to the statement.
    let mut rng = SmallRng::seed_from_u64(0x004A_1102);
    let point = Point::<Binary>::rand(&mut rng, 1);
    let column = Poly::<Binary>::rand(&mut rng, 2).as_slice().to_vec();

    // The weight of the row that stays, and the claim the forged pushforward supports.
    let row_weights = Poly::<Binary>::new_from_point(point.as_slice(), Binary::ONE);
    let kept = row_weights.as_slice()[0];
    let claims = vec![kept * column[0]];

    let readers = vec![Reader {
        point: &point,
        claims: &claims,
    }];
    let lookups = vec![TableLookup {
        num_variables: 2,
        readers: &readers,
    }];

    let positions = vec![0usize, 4];
    let reader_witnesses = vec![ReaderWitness {
        positions: &positions,
    }];
    let columns = vec![column.as_slice()];
    let witness = vec![TableWitness {
        columns: &columns,
        readers: &reader_witnesses,
    }];

    let plan = LogupStarPlan::new::<Binary, Binary>(&lookups);
    let shape = LogupStarShape::new(&plan);
    let mut sponge = binary_challenger();
    let mut transcript = LogupStarProverTranscript::<_, Binary, Binary>::new(&mut sponge, &shape);

    transcript.statement(&TableLookup::statement(&lookups));
    let reader_batching = transcript.reader_batching();

    // One reader earns the zeroth power of the batching challenge.
    //
    // The weights below are the honest ones with the offending row struck out.
    let forged = Weights {
        readers: vec![Poly::new(vec![kept, Binary::ZERO])],
        pushforwards: vec![vec![kept, Binary::ZERO, Binary::ZERO, Binary::ZERO]],
    };
    for pushforward in &forged.pushforwards {
        transcript.pushforward(pushforward);
    }
    let entry_challenges = transcript.entry_challenges(plan.num_tables());

    let (numerator, denominator) = forged.leaf_tables(&plan, &witness, &entry_challenges);
    let numerator = PolyMaybePacked::Scalar(numerator);
    let denominator = PolyMaybePacked::Scalar(denominator);
    let (fraction_gkr, gkr_output) = transcript.fraction_reduction(|challenger| {
        prove_fractional_gkr(
            Fraction {
                n: LeafNumerator::Ext(&numerator),
                d: &denominator,
            },
            challenger,
        )
    });

    // The position column is opened honestly: its second row still names entry 4.
    let block = plan
        .blocks
        .iter()
        .find(|block| matches!(block.role, BlockRole::Reader { .. }))
        .expect("the plan lays out the single reader");
    let own = block.subpoint(&gkr_output.point, plan.num_variables);
    let embedded = positions
        .iter()
        .map(|&entry| position::embed::<Binary>(entry))
        .collect::<Vec<_>>();
    let position_claims = vec![
        SplitEq::<Binary, Binary>::new_packed(&own, Binary::ONE)
            .eval_base(Poly::new(embedded.as_slice())),
    ];
    transcript.position_claims(&position_claims);

    let column_batching = transcript.column_batching();
    let all_columns = vec![witness[0].columns];
    let (mut product_prover, claimed_sum) = ProductProver::new::<Binary>(
        plan.max_table_variables,
        &forged.pushforwards,
        &all_columns,
        column_batching,
    );

    // The forged pushforward was built to support the claim the statement carries.
    //
    // The product sumcheck therefore starts from the sum a verifier asks for.
    assert_eq!(
        claimed_sum,
        product::claimed_sum(&lookups, reader_batching, column_batching)
    );

    let (product, product_point) = transcript.product_sumcheck(|challenger| {
        product_prover.prove::<Binary, _>(
            challenger,
            plan.max_table_variables,
            PRODUCT_DEGREE,
            0,
            claimed_sum,
        )
    });

    let column_claims = vec![vec![
        SplitEq::<Binary, Binary>::new_packed(&product_point, Binary::ONE)
            .eval_base(Poly::new(column.as_slice())),
    ]];
    transcript.column_claims(&column_claims.concat());
    transcript.finish();

    let proof = LogupStarProof {
        pushforwards: forged.pushforwards,
        fraction_gkr,
        position_claims,
        product,
        column_claims,
    };

    let mut verifier_challenger = binary_challenger();
    assert_eq!(
        proof.verify(&lookups, &mut verifier_challenger).map(|_| ()),
        Err(LogupStarError::LeafNumerator)
    );
}

/// Build one honest binary-field proof, mutate it, and return the verifier's verdict.
fn tampered_verdict(
    mutate: impl FnOnce(&mut LogupStarProof<Binary, Binary>),
) -> Result<(), LogupStarError> {
    let mut rng = SmallRng::seed_from_u64(0x0BAD_5EED);
    let instance = Instance::<Binary, Binary>::random(&mut rng, &[(3, 2, &[4, 2])]);

    let readers = instance.readers();
    let lookups = instance.lookups(&readers);
    let columns = instance.column_views();
    let reader_witnesses = instance.reader_witnesses();
    let witness = Instance::<Binary, Binary>::witness(&columns, &reader_witnesses);

    let mut prover_challenger = binary_challenger();
    let (mut proof, _) = LogupStarProof::prove(&lookups, &witness, &mut prover_challenger);
    mutate(&mut proof);

    let mut verifier_challenger = binary_challenger();
    proof.verify(&lookups, &mut verifier_challenger).map(|_| ())
}

#[test]
fn rejects_a_tampered_pushforward() {
    // The pushforward is what the whole reduction pins down.
    //
    // Moving one entry of it has to land somewhere.
    //
    // Where it lands says which property is doing the work.
    //
    //     absorbed before the entry challenges  ->  every later draw moves
    //                                           ->  the reduction's own replay diverges
    //
    //     not absorbed                          ->  the draws are unchanged
    //                                           ->  only the rebuilt opening disagrees
    //
    // So pinning the reduction as the rejecting party is what pins the ordering.
    assert!(matches!(
        tampered_verdict(|proof| proof.pushforwards[0][0] += Binary::ONE),
        Err(LogupStarError::FractionGkr(_))
    ));
}

#[test]
fn rejects_a_tampered_position_claim() {
    // Position values are the one part of the denominator side a verifier cannot derive.
    //
    // A wrong one therefore surfaces where the lookup identity is checked.
    assert_eq!(
        tampered_verdict(|proof| proof.position_claims[0] += Binary::ONE),
        Err(LogupStarError::LeafDenominator)
    );
}

#[test]
fn rejects_a_tampered_column_claim() {
    // Column values close the product claim, so a wrong one breaks the sumcheck's final check.
    assert_eq!(
        tampered_verdict(|proof| proof.column_claims[0][0] += Binary::ONE),
        Err(LogupStarError::ProductFinalValue)
    );
}

#[test]
fn rejects_a_product_sumcheck_claiming_its_own_sum() {
    // The statement fixes what the sumcheck starts from; a prover may not choose it.
    assert_eq!(
        tampered_verdict(|proof| proof.product.claimed_sum += Binary::ONE),
        Err(LogupStarError::ProductClaimedSum)
    );
}

#[test]
fn rejects_a_pushforward_of_the_wrong_width() {
    // Widths decide how the transcript is walked, so they are checked before it is touched.
    assert_eq!(
        tampered_verdict(|proof| {
            proof.pushforwards[0].pop();
        }),
        Err(LogupStarError::PushforwardWidth {
            table: 0,
            expected: 8,
            actual: 7,
        })
    );
}

/// Digest of one proof's encoding, so a change in what the run derives is visible.
fn proof_digest(proof: &LogupStarProof<Binary, Binary>) -> [u8; 32] {
    let bytes = postcard::to_allocvec(proof).expect("a proof serializes");
    Keccak256Hash.hash_iter(bytes)
}

/// Lowercase hexadecimal, so a digest reads as one string in an assertion.
fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn a_fixed_statement_always_produces_the_same_proof() {
    // Every challenge in the run is derived.
    //
    // The whole proof is therefore a function of the statement and the witness.
    //
    // Pinning it catches a change in any derivation.
    //
    // That includes one keeping prover and verifier agreed, which no round trip would show.
    //
    // Fixture state: two tables of four entries, one reader each, fixed seed.
    //
    // A failure here is not automatically a bug.
    //
    // It means the transcript changed.
    //
    // The new digest is right exactly when that change was intended.
    let mut rng = SmallRng::seed_from_u64(0x0_901D);
    let instance = Instance::<Binary, Binary>::random(&mut rng, &[(2, 1, &[2]), (2, 2, &[1])]);

    let readers = instance.readers();
    let lookups = instance.lookups(&readers);
    let columns = instance.column_views();
    let reader_witnesses = instance.reader_witnesses();
    let witness = Instance::<Binary, Binary>::witness(&columns, &reader_witnesses);

    let mut challenger = binary_challenger();
    let (proof, _) = LogupStarProof::prove(&lookups, &witness, &mut challenger);

    assert_eq!(
        hex(&proof_digest(&proof)),
        "460d8984f75ef5c7bcad06133f058e3fbe051dcf21124462500b4259d4bb2f8d"
    );
}

#[test]
fn a_proof_survives_a_round_trip_through_serialization() {
    // A proof crosses a wire, so a decoded one has to be worth exactly what the original was.
    //
    // Verifying the decoded proof is a stronger check than comparing fields.
    //
    // It also catches a field that survives the encoding but lands in the wrong place.
    let mut rng = SmallRng::seed_from_u64(0x0_5E4D);
    let instance = Instance::<Binary, Binary>::random(&mut rng, &[(3, 2, &[4, 2])]);

    let readers = instance.readers();
    let lookups = instance.lookups(&readers);
    let columns = instance.column_views();
    let reader_witnesses = instance.reader_witnesses();
    let witness = Instance::<Binary, Binary>::witness(&columns, &reader_witnesses);

    let mut challenger = binary_challenger();
    let (proof, output) = LogupStarProof::prove(&lookups, &witness, &mut challenger);

    let bytes = postcard::to_allocvec(&proof).expect("a proof serializes");
    let decoded: LogupStarProof<Binary, Binary> =
        postcard::from_bytes(&bytes).expect("a proof deserializes");

    let mut challenger = binary_challenger();
    assert_eq!(
        decoded
            .verify(&lookups, &mut challenger)
            .expect("a decoded proof verifies"),
        output
    );
}

#[test]
fn rejects_claims_the_witness_never_produced() {
    // Fixture state: an honest proof over an honest statement.
    //
    // Mutation: verify it against a statement claiming the reader pulled something else.
    //
    // The shapes are unchanged, so only the claims differ.
    //
    // The claims are bound before any challenge is drawn.
    //
    // A different statement draws a different challenge stream.
    //
    // The reduction then stops agreeing with itself.
    let mut rng = SmallRng::seed_from_u64(0x0FA1_5E00);
    let mut instance = Instance::<Binary, Binary>::random(&mut rng, &[(3, 1, &[4])]);

    let (proof, _) = {
        let readers = instance.readers();
        let lookups = instance.lookups(&readers);
        let columns = instance.column_views();
        let reader_witnesses = instance.reader_witnesses();
        let witness = Instance::<Binary, Binary>::witness(&columns, &reader_witnesses);
        let mut challenger = binary_challenger();
        LogupStarProof::prove(&lookups, &witness, &mut challenger)
    };

    instance.claims[0][0][0] += Binary::ONE;
    let readers = instance.readers();
    let lookups = instance.lookups(&readers);

    let mut challenger = binary_challenger();
    assert_eq!(
        proof.verify(&lookups, &mut challenger).map(|_| ()),
        Err(LogupStarError::FractionGkr(
            crate::fractional_gkr::FractionGkrError::LayerConsistency { layer: 1 }
        ))
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn honest_reductions_always_verify(
        table_variables in 1usize..=3,
        width in 1usize..=3,
        first_reader in 1usize..=4,
        second_reader in 1usize..=4,
        seed in any::<u64>(),
    ) {
        // Shapes vary independently.
        //
        // A reader may be shorter or taller than its table.
        //
        // Two readers of one table need not agree either.
        round_trip::<Binary, Binary, _>(
            &[(table_variables, width, &[first_reader, second_reader])],
            seed,
            binary_challenger,
        );
    }
}
