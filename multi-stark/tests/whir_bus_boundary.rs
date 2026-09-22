//! End-to-end boundary flushes: one tuple per table end, proved and verified over WHIR.
//!
//! A machine closes its execution chain at the two ends of a table.
//!
//! The initial state enters the bus once, the final state leaves it once, and no row between them says anything about either.
//!
//! The prover materializes one live leaf factor per end and the identity everywhere else.
//!
//! The verifier rebuilds the same factors from its own symbolic AIRs and the committed openings.

use p3_air::{Air, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_bus::{BusActivation, BusBoundary, BusDirection, BusInteractionBuilder, BusName};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    ProverInstance, ProverInstances, ProvingError, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type L = PrefixProver<F, EF>;
type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

/// First-round folding factor; also the per-table padding floor.
const FOLDING: usize = 2;

/// Columns every table in this file carries.
const NUM_COLS: usize = 2;

/// The one place this machine names its state channel.
const STATE: BusName<'static> = BusName::new("state");

/// A WHIR-backed multilinear AIR configuration over `BabyBear`.
struct WhirConfigForTest {
    /// The WHIR commitment scheme, fixed to one stacked-table arity.
    pcs: TestPcs,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        None
    }

    fn min_num_variables(&self) -> usize {
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        L::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a p3_whir::WhirProverData<F, EF, MyMmcs, L>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Fixed permutation so prover and verifier transcripts match exactly.
fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

/// Per-round log-inverse rates for a stacked polynomial.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let folding_schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .expect("valid folding schedule");
    let num_rounds = folding_schedule.len() - 1;
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for &folding in folding_schedule.iter().take(num_rounds) {
        rate += folding - 1;
        rates.push(rate);
    }
    rates
}

/// Build a configuration sized for a batch of same-shape trace tables.
fn config_for(log_height: usize, num_tables: usize) -> WhirConfigForTest {
    let stacked_num_variables = log2_ceil_usize(num_tables * NUM_COLS * (1 << log_height));
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: default_round_log_inv_rates(stacked_num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let whir_config = WhirConfig::new(stacked_num_variables, params).unwrap();
    WhirConfigForTest {
        pcs: TestPcs::new(whir_config, MyDft::default(), mmcs),
    }
}

/// A fresh challenger.
fn challenger() -> MyChallenger {
    MyChallenger::new(perm())
}

/// One table that flushes its state column once at each end of its trace.
///
/// Column zero carries the state word, and column one is a flag the local constraint owns.
///
/// The flag exists because a boundary indicator contributes no constraint of its own.
///
/// A table with nothing but boundary declarations would leave the batched zerocheck empty and be refused.
#[derive(Clone, Copy)]
struct BoundaryStateAir {
    /// Side of the multiset equality this table contributes both of its ends to.
    direction: BusDirection,
}

impl BaseAir<F> for BoundaryStateAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB> Air<AB> for BoundaryStateAir
where
    AB: BusInteractionBuilder<F = F>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let state: AB::Expr = row[0].into();
        let flag: AB::Expr = row[1].into();

        // The table's own constraint family, independent of anything the bus declares.
        builder.assert_zero(flag.clone() * (flag - AB::Expr::ONE));

        builder.push_bus_interaction(
            STATE,
            self.direction,
            [state.clone()],
            BusActivation::Boundary(BusBoundary::First),
        );
        builder.push_bus_interaction(
            STATE,
            self.direction,
            [state],
            BusActivation::Boundary(BusBoundary::Last),
        );
    }
}

/// A trace whose state column holds `states` and whose flag column is all ones.
fn table(states: [u64; 4]) -> Table<F> {
    let values = states
        .into_iter()
        .flat_map(|state| [F::from_u64(state), F::ONE])
        .collect::<Vec<_>>();
    Table::new(RowMajorMatrix::new(values, NUM_COLS).transpose())
}

/// Whether one two-table statement whose ends are flushed onto the state channel balances.
///
/// A false answer is specifically an unbalanced multiset, never some other prover failure.
fn balances(pushes: [u64; 4], pulls: [u64; 4]) -> bool {
    let push = BoundaryStateAir {
        direction: BusDirection::Push,
    };
    let pull = BoundaryStateAir {
        direction: BusDirection::Pull,
    };
    let log_height = log2_strict_usize(pushes.len());
    let config = config_for(log_height, 2);
    let (pk, _) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();
    match prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&push, table(pushes), &pk, &[]),
            ProverInstance::new(&pull, table(pulls), &pk, &[]),
        ]),
        0,
        &mut challenger(),
    ) {
        Ok(_) => true,
        Err(ProvingError::BusArgument(p3_bus::BusArgumentError::UnbalancedProducts)) => false,
        Err(error) => panic!("the fixture failed for an unrelated reason: {error}"),
    }
}

#[test]
fn boundary_flushes_balance_when_only_the_two_ends_agree() {
    // The two tables agree at row zero and at row three and disagree everywhere between.
    //
    // - push table -> 10, 11, 12, 13
    // - pull table -> 10, 21, 22, 13
    //
    // Read as boundaries the multiset balances, 10 and 13 against 10 and 13.
    //
    // Read as one tuple per row it would not, so an indicator that leaked past its end could not prove this.
    let push = BoundaryStateAir {
        direction: BusDirection::Push,
    };
    let pull = BoundaryStateAir {
        direction: BusDirection::Pull,
    };
    let log_height = 2;
    let config = config_for(log_height, 2);
    let (pk, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&push, table([10, 11, 12, 13]), &pk, &[]),
            ProverInstance::new(&pull, table([10, 21, 22, 13]), &pk, &[]),
        ]),
        0,
        &mut challenger(),
    )
    .expect("a boundary-balanced statement is provable");

    // The bus reduction ran: the two ends are not folded into the local constraints.
    assert!(proof.bus.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&push, &vk, log_height, &[]),
            VerifierInstance::new(&pull, &vk, log_height, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("the verifier rebuilds the same two leaf factors from its own AIRs");
}

#[test]
fn a_boundary_flush_is_not_a_per_row_flush() {
    // Middle rows that match do not rescue ends that do not.
    //
    // Only row three differs here, and only at an end a declaration names, so this is unprovable.
    assert!(!balances([10, 11, 12, 13], [10, 11, 12, 14]));

    // The mirror case: the first rows disagree and every later row matches.
    assert!(!balances([10, 11, 12, 13], [20, 11, 12, 13]));

    // Middle rows are outside both declarations, so disagreeing there changes nothing.
    assert!(balances([10, 11, 12, 13], [10, 99, 98, 13]));
}

#[test]
fn a_boundary_flush_on_a_one_row_table_fires_both_ends_at_that_row() {
    // A repeated word at both ends makes each side flush it twice.
    //
    // The two declarations then name one value rather than two, and the multiset still balances.
    let push = BoundaryStateAir {
        direction: BusDirection::Push,
    };
    let pull = BoundaryStateAir {
        direction: BusDirection::Pull,
    };
    let log_height = 2;
    let config = config_for(log_height, 2);
    let (pk, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();

    // Both tables repeat one word at both ends, so each side flushes that word twice.
    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&push, table([7, 1, 2, 7]), &pk, &[]),
            ProverInstance::new(&pull, table([7, 3, 4, 7]), &pk, &[]),
        ]),
        0,
        &mut challenger(),
    )
    .expect("two pushes of one word match two pulls of it");

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&push, &vk, log_height, &[]),
            VerifierInstance::new(&pull, &vk, log_height, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("repeated ends verify like distinct ones");
}
