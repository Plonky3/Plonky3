//! One table holding both bit columns and field-element columns.
//!
//! An addition row, `out = a + b` on `w`-bit words, beside a clock `t`:
//!
//! ```text
//!     bit region    a (w bits), b (w bits), out (w bits), c_1 .. c_{w-1} (carries)
//!     dense region  t
//!
//!     constraints   out_i   = a_i + b_i + c_i            bits only
//!                   c_{i+1} = maj(a_i, b_i, c_i)         bits only
//!                   t'      = t * g                      dense only
//!     bus message   (t, word(a), word(b), word(out))     both regions
//! ```
//!
//! The reference splits the row into a bit table and a dense table, joined on a second bus.
//!
//! Both statements must accept the same witnesses and reject the same forgeries.

use std::time::{Duration, Instant};

use p3_air::utils::word_view;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{
    BinaryPcsConfig, BinaryPcsParams, BooleanTracePcs, GroupedCodewordMmcs, MixedTraceCommitment,
    MixedTraceData, MixedTracePcs, committed_shapes, coordinate_basis,
};
use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName};
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::contract::HeightRange;
use p3_multi_stark::{
    ProverInstance, ProverInstances, TableDeclaration, VerifierInstance, VerifierInstances,
    prove_with_security, setup, verify_with_security,
};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MerkleMmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Mmcs = GroupedCodewordMmcs<MerkleMmcs>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Coordinates one committed element absorbs.
const ABSORBED: usize = 7;

/// Security target every proof is graded against.
const SECURITY_BITS: usize = 100;

/// Bus the addition rows are consumed on.
const ADD: &str = "add";

/// Bus the reference joins its two halves on.
const JOIN: &str = "join";

/// Bit columns of one addition row: three words and the carries into bits `1 .. w`.
const fn row_bits(word: usize) -> usize {
    4 * word - 1
}

/// One table of either statement.
#[derive(Clone)]
enum Kind {
    /// The mixed row: the bit region, then `t`.
    Adder,
    /// The reference's bit half: the bit region alone.
    Bits,
    /// The reference's dense half: `t`, then the three words.
    Clock,
    /// The consumer both statements share: `t`, the three words, and an activation.
    Consumer,
}

/// A table of the addition statement, over `w`-bit words.
#[derive(Clone)]
struct AdderAir {
    kind: Kind,
    /// The basis a word view reads its bits over.
    basis: Vec<F>,
}

impl AdderAir {
    fn new(kind: Kind, word: usize) -> Self {
        Self {
            kind,
            basis: coordinate_basis::<F>()[..word].to_vec(),
        }
    }

    const fn word(&self) -> usize {
        self.basis.len()
    }

    /// Carries, sums, and the word views of `a`, `b`, `out`, from a row's bit region.
    fn eval_bits<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        bits: &[AB::Var],
    ) -> [AB::Expr; 3] {
        let w = self.word();
        let (a, rest) = bits.split_at(w);
        let (b, rest) = rest.split_at(w);
        let (out, carries) = rest.split_at(w);
        // c_0 = 0, so bit 0 has no carry column.
        let carry = |i: usize| -> AB::Expr {
            if i == 0 {
                AB::Expr::ZERO
            } else {
                carries[i - 1].into()
            }
        };
        for i in 0..w {
            let (x, y, c) = (a[i].into(), b[i].into(), carry(i));
            builder.assert_eq(out[i], x.clone() + y.clone() + c.clone());
            if i + 1 < w {
                // maj(x, y, c) = x*y + x*c + y*c in characteristic two.
                builder.assert_eq(carry(i + 1), x.clone() * y.clone() + x * c.clone() + y * c);
            }
        }
        [a, b, out].map(|bits| word_view::<AB::Expr, F, _>(bits, &self.basis))
    }
}

impl BaseAir<F> for AdderAir {
    fn width(&self) -> usize {
        let bits = row_bits(self.word());
        match self.kind {
            Kind::Adder => bits + 1,
            Kind::Bits => bits,
            Kind::Clock => 4,
            Kind::Consumer => 5,
        }
    }

    fn boolean_columns(&self) -> usize {
        match self.kind {
            Kind::Adder | Kind::Bits => row_bits(self.word()),
            Kind::Clock | Kind::Consumer => 0,
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        // Only the clock is read one row ahead.
        match self.kind {
            Kind::Adder => vec![row_bits(self.word())],
            Kind::Clock => vec![0],
            Kind::Bits | Kind::Consumer => Vec::new(),
        }
    }
}

impl<AB: BusInteractionBuilder<F = F>> Air<AB> for AdderAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice().to_vec();
        let next = main.next_slice().to_vec();
        let add = BusName::new(ADD);
        let join = BusName::new(JOIN);
        let clock = |builder: &mut AB, t: AB::Var, t_next: AB::Var| {
            builder
                .when_transition()
                .assert_eq(t_next, t.into() * F::GENERATOR);
        };
        match self.kind {
            Kind::Adder => {
                let t = row_bits(self.word());
                let [a, b, out] = self.eval_bits(builder, &local[..t]);
                clock(builder, local[t], next[t]);
                let message = [local[t].into(), a, b, out];
                builder.push_bus_interaction(
                    add,
                    BusDirection::Push,
                    message,
                    BusActivation::Always,
                );
            }
            Kind::Bits => {
                let words = self.eval_bits(builder, &local);
                builder.push_bus_interaction(
                    join,
                    BusDirection::Push,
                    words,
                    BusActivation::Always,
                );
            }
            Kind::Clock => {
                clock(builder, local[0], next[0]);
                builder.push_bus_interaction(
                    join,
                    BusDirection::Pull,
                    local[1..4].to_vec(),
                    BusActivation::Always,
                );
                builder.push_bus_interaction(
                    add,
                    BusDirection::Push,
                    local[..4].to_vec(),
                    BusActivation::Always,
                );
            }
            Kind::Consumer => {
                builder.push_bus_interaction(
                    add,
                    BusDirection::Pull,
                    local[..4].to_vec(),
                    BusActivation::Boolean(local[4].into()),
                );
            }
        }
    }
}

/// A commitment scheme sized for one statement's tables.
struct MixedConfig {
    pcs: MixedTracePcs<F, Mmcs, Mmcs>,
}

impl MixedConfig {
    /// Split every table where its declaration says, and size the bit witness for them.
    fn new(airs: &[AdderAir], log_height: usize) -> Self {
        let bits: Vec<usize> = airs
            .iter()
            .map(|air| {
                TableDeclaration::from_constraints::<F, F, _>(
                    air,
                    HeightRange::exactly(log_height as u32),
                )
                .columns()
                .boolean
            })
            .collect();
        let shapes: Vec<TableShape> = airs
            .iter()
            .map(|air| TableShape::new(log_height, air.width()))
            .collect();
        let (arity, _) = plan_stacked_layout(&committed_shapes::<F>(&shapes, &bits));
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: SECURITY_BITS,
        };
        let config = BinaryPcsConfig::try_new::<F, F>(arity - ABSORBED, params)
            .unwrap()
            .try_with_folding(3.min(arity - ABSORBED))
            .unwrap();
        let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
        let mmcs = Mmcs::for_folding(merkle, &config);
        let inner = BooleanTracePcs::new(config, mmcs.clone(), mmcs, arity).unwrap();
        Self {
            pcs: MixedTraceCommitment::new(inner, bits),
        }
    }

    /// Committed bytes: one bit per committed cell, the stack padded to a power of two.
    fn committed_bytes(&self) -> usize {
        (1usize << self.pcs.inner().num_variables()) / 8
    }
}

impl MultiStarkConfig for MixedConfig {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = MixedTracePcs<F, Mmcs, Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // Keccak-256 is shared by the transcript and the Merkle tree.
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Vec<Table<F>> {
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a MixedTraceData<F, p3_binary_pcs::BinaryPcsProverData<F, F, Mmcs>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

fn challenger() -> Challenger {
    Challenger::from_hasher(b"p3-multi-stark-mixed-width-v1".to_vec(), Keccak256Hash)
}

/// What a forger changes in an otherwise honest witness.
#[derive(Clone, Copy, Debug)]
enum Forgery {
    /// Nothing: the honest witness.
    Honest,
    /// One bit of `out` is flipped.
    OutBit,
    /// One carry is flipped.
    Carry,
    /// One clock reading is moved.
    Clock,
    /// The consumer's word disagrees with the bits it came from.
    BusWord,
    /// A bit cell holds a field element that is not a bit.
    NonBit,
}

/// One addition statement's rows, in trace order.
struct Rows {
    word: usize,
    /// The basis every word view of this statement reads over.
    basis: Vec<F>,
    a: Vec<u64>,
    b: Vec<u64>,
    t: Vec<F>,
}

impl Rows {
    fn random(seed: u64, word: usize, log_height: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mask = if word == 64 {
            u64::MAX
        } else {
            (1 << word) - 1
        };
        let rows = 1 << log_height;
        let a = (0..rows).map(|_| rng.random::<u64>() & mask).collect();
        let b = (0..rows).map(|_| rng.random::<u64>() & mask).collect();
        let mut t = vec![F::ONE];
        for _ in 1..rows {
            t.push(*t.last().unwrap() * F::GENERATOR);
        }
        let basis = coordinate_basis::<F>()[..word].to_vec();
        Self {
            word,
            basis,
            a,
            b,
            t,
        }
    }

    /// Bit region of row `j`: `a`, `b`, `out`, then the carries into bits `1 .. w`.
    fn bits(&self, j: usize) -> Vec<F> {
        let w = self.word;
        let (a, b) = (self.a[j], self.b[j]);
        let bit = |x: u64, i: usize| (x >> i) & 1 == 1;
        let mut carries = Vec::with_capacity(w);
        let mut c = false;
        let mut out = 0u64;
        for i in 0..w {
            let (x, y) = (bit(a, i), bit(b, i));
            out |= u64::from(x ^ y ^ c) << i;
            c = (x & y) | (x & c) | (y & c);
            carries.push(c);
        }
        carries.pop();
        (0..w)
            .map(|i| bit(a, i))
            .chain((0..w).map(|i| bit(b, i)))
            .chain((0..w).map(|i| bit(out, i)))
            .chain(carries)
            .map(F::from_bool)
            .collect()
    }

    /// The word a run of bit cells reads as.
    fn word_of(&self, bits: &[F]) -> F {
        word_view(bits, &self.basis)
    }

    const fn height(&self) -> usize {
        self.t.len()
    }
}

/// One table as a column-major trace.
fn table(rows: &[Vec<F>]) -> Table<F> {
    let height = rows.len();
    let width = rows[0].len();
    let columns = (0..width)
        .flat_map(|column| rows.iter().map(move |row| row[column]))
        .collect();
    Table::new(RowMajorMatrix::new(columns, height))
}

/// The traces of both statements, with one forgery applied to each in the same place.
///
/// Returns `(mixed, reference)`: `[adder, consumer]` and `[bits, clock, consumer]`.
fn traces(rows: &Rows, forgery: Forgery, at: usize) -> ([Table<F>; 2], [Table<F>; 3]) {
    let w = rows.word;
    let mut bits: Vec<Vec<F>> = (0..rows.height()).map(|j| rows.bits(j)).collect();
    let mut t = rows.t.clone();
    // The consumer reads what the honest bits say, before any forgery.
    let words = |bits: &[F]| [0, 1, 2].map(|k| rows.word_of(&bits[k * w..(k + 1) * w]));
    let mut consumer: Vec<Vec<F>> = bits
        .iter()
        .zip(&t)
        .map(|(row, &t)| {
            let [a, b, out] = words(row);
            vec![t, a, b, out, F::ONE]
        })
        .collect();

    match forgery {
        Forgery::Honest => {}
        Forgery::OutBit => bits[at][2 * w] += F::ONE,
        Forgery::Carry => bits[at][3 * w] += F::ONE,
        Forgery::Clock => t[at] += F::ONE,
        Forgery::BusWord => consumer[at][3] += F::ONE,
        Forgery::NonBit => bits[at][0] = F::GENERATOR,
    }

    let adder: Vec<Vec<F>> = bits
        .iter()
        .zip(&t)
        .map(|(row, &t)| row.iter().copied().chain([t]).collect())
        .collect();
    // The dense half copies the words the bit half's cells read as.
    let clock: Vec<Vec<F>> = bits
        .iter()
        .zip(&t)
        .map(|(row, &t)| {
            let [a, b, out] = words(row);
            vec![t, a, b, out]
        })
        .collect();
    let consumer = table(&consumer);
    (
        [table(&adder), consumer.clone()],
        [table(&bits), table(&clock), consumer],
    )
}

/// Prove and verify one statement, returning the proof size and prove time, or why it failed.
fn run(
    airs: &[AdderAir],
    tables: Vec<Table<F>>,
    log_height: usize,
) -> Result<(usize, Duration), String> {
    let config = MixedConfig::new(airs, log_height);
    let refs: Vec<&AdderAir> = airs.iter().collect();
    let (pk, vk) = setup(&config, &refs, &mut challenger()).map_err(|e| format!("{e:?}"))?;
    let instances = ProverInstances::new(
        airs.iter()
            .zip(tables)
            .map(|(air, table)| ProverInstance::new(air, table, &pk, &[]))
            .collect(),
    );
    let start = Instant::now();
    let proof = prove_with_security(&config, instances, 0, SECURITY_BITS, &mut challenger())
        .map_err(|e| format!("{e:?}"))?;
    let elapsed = start.elapsed();
    let verifier = VerifierInstances::new(
        airs.iter()
            .map(|air| VerifierInstance::new(air, &vk, log_height, &[]))
            .collect(),
    );
    verify_with_security(
        &config,
        verifier,
        &proof,
        0,
        SECURITY_BITS,
        &mut challenger(),
    )
    .map_err(|e| format!("{e:?}"))?;
    Ok((postcard::to_allocvec(&proof).unwrap().len(), elapsed))
}

/// The mixed statement's tables: one addition table and the consumer.
fn mixed_airs(word: usize) -> [AdderAir; 2] {
    [
        AdderAir::new(Kind::Adder, word),
        AdderAir::new(Kind::Consumer, word),
    ]
}

/// The reference statement's tables: the two halves and the consumer.
fn reference_airs(word: usize) -> [AdderAir; 3] {
    [
        AdderAir::new(Kind::Bits, word),
        AdderAir::new(Kind::Clock, word),
        AdderAir::new(Kind::Consumer, word),
    ]
}

/// Whether each statement accepts the witness this forgery leaves.
fn verdicts(
    seed: u64,
    word: usize,
    log_height: usize,
    forgery: Forgery,
    at: usize,
) -> (bool, bool) {
    let rows = Rows::random(seed, word, log_height);
    let (mixed, reference) = traces(&rows, forgery, at);
    let mixed = run(&mixed_airs(word), mixed.to_vec(), log_height).is_ok();
    let reference = run(&reference_airs(word), reference.to_vec(), log_height).is_ok();
    (mixed, reference)
}

#[test]
fn the_declaration_carries_both_regions() {
    // Eight-bit words: 31 bit columns, then the clock.
    let air = AdderAir::new(Kind::Adder, 8);
    let declared = TableDeclaration::from_constraints::<F, F, _>(&air, HeightRange::exactly(4));
    assert_eq!(declared.columns().committed, 32);
    assert_eq!(declared.columns().boolean, 31);
}

#[test]
fn an_honest_mixed_table_proves() {
    assert_eq!(verdicts(1, 8, 4, Forgery::Honest, 0), (true, true));
}

#[test]
fn a_forged_word_view_is_rejected() {
    // The consumer's word for row 3 is one off the word its bits read as.
    assert_eq!(verdicts(2, 8, 4, Forgery::BusWord, 3), (false, false));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn the_mixed_table_agrees_with_the_two_table_reference(
        seed in any::<u64>(),
        log_height in 2usize..5,
        forgery in prop_oneof![
            Just(Forgery::Honest),
            Just(Forgery::OutBit),
            Just(Forgery::Carry),
            Just(Forgery::Clock),
            Just(Forgery::BusWord),
            Just(Forgery::NonBit),
        ],
        at in any::<prop::sample::Index>(),
    ) {
        let at = at.index(1 << log_height);
        let (mixed, reference) = verdicts(seed, 4, log_height, forgery, at);
        // Both accept exactly the honest witness.
        prop_assert_eq!(mixed, reference);
        prop_assert_eq!(mixed, matches!(forgery, Forgery::Honest));
    }
}

#[test]
#[ignore = "benchmark: cargo test --release -p p3-multi-stark --test mixed_width -- --ignored --nocapture"]
fn mixed_against_two_tables() {
    // 64-bit words: 255 bits and one clock per row, against a bit table and a four-column clock table.
    println!("log_height  statement  committed_kib  prove_ms  proof_kib");
    for log_height in [10, 12, 14] {
        let rows = Rows::random(7, 64, log_height);
        let (mixed, reference) = traces(&rows, Forgery::Honest, 0);
        for (name, airs, tables) in [
            ("mixed", mixed_airs(64).to_vec(), mixed.to_vec()),
            ("two-table", reference_airs(64).to_vec(), reference.to_vec()),
        ] {
            let committed = MixedConfig::new(&airs, log_height).committed_bytes();
            let (proof, elapsed) = run(&airs, tables, log_height).unwrap();
            let elapsed = elapsed.as_secs_f64() * 1e3;
            println!(
                "{log_height:>10}  {name:>9}  {:>13}  {elapsed:>8.1}  {:>9.1}",
                committed / 1024,
                proof as f64 / 1024.0
            );
        }
    }
}
