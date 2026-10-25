//! A circuit table beside an ordinary one, joined on a bus.
//!
//! ```text
//!     requests   one 64-bit addition per row, its words as bits
//!     adder      the circuit out = a + b, one instance per row
//!
//!     requests --add--> adder     bus: (a, b, out) as words
//! ```

use p3_air::utils::word_view;
use p3_air::{Air, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::whir::{
    BinaryWhirDomain, BinaryWhirProfile, BooleanWhirData, BooleanWhirPcs, BooleanWhirProver,
    BooleanWhirTracePcs, recommended_cap_height,
};
use p3_binary_pcs::{BooleanTraceCommitmentData, coordinate_basis};
use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName};
use p3_challenger::HashChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_flock::{Builder, Circuit, Wire, Witness};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    CircuitError, MultiStarkProof, ProverInstance, ProverInstances, ProvingKey, VerificationError,
    VerifierInstance, VerifierInstances, VerifyingKey, prove, security_report, setup, verify,
};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Poly64;
type EF = Poly192;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type Domain = BinaryWhirDomain<F>;
type PackedPcs = BooleanWhirTracePcs<F, EF, Domain, Mmcs, Challenger>;

/// Per-term target of the proximity schedule.
const TERM_BITS: usize = 140;

/// Base-two logarithm of the inverse code rate.
const LOG_INV_RATE: usize = 2;

/// Rows of both tables, as a power of two.
const LOG_ROWS: usize = 5;

/// Bits of one word.
const WORD: usize = 64;

/// Bus the additions travel on.
const ADD: &str = "add";

/// The two tables.
enum Chip {
    /// One addition per row: the bits of `a`, `b` and `out`.
    Requests,
    /// The adder circuit's ports: `a`, `b` and `out` as words.
    Adder,
}

impl BaseAir<F> for Chip {
    fn width(&self) -> usize {
        match self {
            Self::Requests => 3 * WORD,
            Self::Adder => 3,
        }
    }

    fn boolean_columns(&self) -> usize {
        match self {
            Self::Requests => 3 * WORD,
            Self::Adder => 0,
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB: BusInteractionBuilder<F = F>> Air<AB> for Chip {
    fn eval(&self, builder: &mut AB) {
        let local = builder.main().current_slice().to_vec();
        let add = BusName::new(ADD);
        match self {
            Self::Requests => {
                let (a, rest) = local.split_at(WORD);
                let (b, out) = rest.split_at(WORD);
                // The low bit of a sum carries nothing in: out_0 = a_0 + b_0.
                builder.assert_eq(out[0], a[0].into() + b[0].into());
                let basis = coordinate_basis::<F>();
                let words = [a, b, out].map(|bits| word_view::<AB::Expr, F, _>(bits, &basis));
                builder.push_bus_interaction(add, BusDirection::Push, words, BusActivation::Always);
            }
            Self::Adder => {
                // The circuit proves the sum, so the AIR only takes the words off the bus.
                builder.push_bus_interaction(
                    add,
                    BusDirection::Pull,
                    local[..3].to_vec(),
                    BusActivation::Always,
                );
            }
        }
    }
}

/// `out = a + flip(b)` on 64-bit words, `flip` inverting bit 0 when asked.
///
/// Both variants have the same gates in the same slots, so they differ only in what they compute.
fn adder(flip: bool) -> Circuit {
    let mut c = Builder::new(&[WORD, WORD], &[WORD]);
    let a = c.input(0).to_vec();
    let mut b = c.input(1).to_vec();
    if flip {
        b[0] = c.not(b[0]);
    } else {
        // The same free gate, so both circuits share their slot layout.
        b[0] = c.xor(b[0], Wire::ZERO);
    }
    let sum = c.wrapping_sum(&[&a, &b]);
    for (bit, &wire) in sum.iter().enumerate() {
        c.output(0, bit, wire);
    }
    c.finish()
}

/// Random operand pairs, one per row.
fn operands(seed: u64) -> Vec<(u64, u64)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..1 << LOG_ROWS)
        .map(|_| (rng.random(), rng.random()))
        .collect()
}

/// The circuit's witness on every row.
fn witness(circuit: &Circuit, rows: &[(u64, u64)]) -> Witness {
    Witness::generate(circuit, LOG_ROWS, rows, &(0, 0), |&(a, b), words| {
        words[0] = a;
        words[1] = b;
    })
}

/// The requests table: each row's `a`, `b` and the sum the witness computed, bit by bit.
fn requests(rows: &[(u64, u64)], sums: &[u64]) -> Table<F> {
    let height = 1 << LOG_ROWS;
    // One polynomial per word bit, every row's bit in order.
    let mut columns = F::zero_vec(3 * WORD * height);
    for (row, (&(a, b), &sum)) in rows.iter().zip(sums).enumerate() {
        for (word, value) in [a, b, sum].into_iter().enumerate() {
            for bit in 0..WORD {
                columns[(word * WORD + bit) * height + row] = F::from_bool((value >> bit) & 1 == 1);
            }
        }
    }
    Table::new(RowMajorMatrix::new(columns, height))
}

/// Each row's output word, read off the witness bits.
fn sums(circuit: &Circuit, witness: &Witness) -> Vec<u64> {
    let stride = (1 << circuit.k_log()) / 64;
    witness
        .z()
        .chunks_exact(stride)
        .map(|instance| circuit.output_words(instance)[0])
        .collect()
}

/// Every cell a bit, packed sixty-four to a committed element.
struct PackedConfig {
    pcs: PackedPcs,
}

impl PackedConfig {
    /// Wire the scheme for the requests table and the adder's witness column.
    fn new(circuit: &Circuit) -> Self {
        Self::with_shapes(&[
            TableShape::new(LOG_ROWS, 3 * WORD),
            TableShape::new(LOG_ROWS + circuit.k_log(), 1),
        ])
    }

    /// Wire the scheme for tables committed at these shapes.
    fn with_shapes(shapes: &[TableShape]) -> Self {
        let (arity, _) = plan_stacked_layout(shapes);
        let absorbed = BitRingSwitch::<F, EF>::ABSORBED;
        let domain = Domain::default();
        let config = BinaryWhirProfile::unique_decoding(TERM_BITS, LOG_INV_RATE, 1)
            .config::<EF, F, Challenger, _>(arity - absorbed, &domain)
            .expect("the schedule fits the packed arity");
        let cap_height = recommended_cap_height(&config);
        let mmcs = Mmcs::new(
            Hash::new(Keccak256Hash),
            Compress::new(Keccak256Hash),
            cap_height,
        );
        let prover = BooleanWhirProver::new(config, domain, mmcs);
        let bits = BooleanWhirPcs::new(prover, arity).expect("the schedule commits the packing");
        Self {
            pcs: PackedPcs::from_commitment(bits),
        }
    }
}

impl MultiStarkConfig for PackedConfig {
    type Val = F;
    type Challenge = EF;
    type Challenger = Challenger;
    type Pcs = PackedPcs;

    fn pcs(&self) -> &PackedPcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
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
        prover_data: &'a BooleanTraceCommitmentData<F, BooleanWhirData<F, EF, Mmcs>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// A fresh transcript, the same on both sides.
const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The two tables, the requests first.
const CHIPS: [Chip; 2] = [Chip::Requests, Chip::Adder];

/// Keys for the two tables.
fn keys(config: &PackedConfig) -> (ProvingKey<PackedConfig>, VerifyingKey<PackedConfig>) {
    setup(config, &[&CHIPS[0], &CHIPS[1]], &mut challenger()).unwrap()
}

/// Prove the machine: `proved` runs every row, the requests holding the sums `computed` made.
fn prove_machine(
    config: &PackedConfig,
    pk: &ProvingKey<PackedConfig>,
    proved: &Circuit,
    computed: &Circuit,
    rows: &[(u64, u64)],
) -> MultiStarkProof<PackedConfig> {
    let witness = witness(computed, rows);
    let requests = requests(rows, &sums(computed, &witness));
    prove(
        config,
        ProverInstances::new(vec![
            ProverInstance::new(&CHIPS[0], requests, pk, &[]),
            ProverInstance::circuit(&CHIPS[1], proved, witness, pk, &[]),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap()
}

/// The verifier's view of the machine, the adder checked against `circuit`.
fn instances<'a>(
    vk: &'a VerifyingKey<PackedConfig>,
    circuit: &'a Circuit,
) -> VerifierInstances<'a, PackedConfig, Chip> {
    VerifierInstances::new(vec![
        VerifierInstance::new(&CHIPS[0], vk, LOG_ROWS, &[]),
        VerifierInstance::circuit(&CHIPS[1], circuit, vk, LOG_ROWS, &[]),
    ])
}

#[test]
fn a_circuit_table_proves_beside_an_ordinary_table() {
    // Fixture state: 32 random additions, the adder proving each one.
    let circuit = adder(false);
    let config = PackedConfig::new(&circuit);
    let (pk, vk) = keys(&config);
    let rows = operands(1);

    // Sanity: the witness computed the wrapping sums the requests table carries.
    let witness = witness(&circuit, &rows);
    let expected = rows
        .iter()
        .map(|&(a, b)| a.wrapping_add(b))
        .collect::<Vec<_>>();
    assert_eq!(sums(&circuit, &witness), expected);

    // One argument for the one circuit table, carrying one value per port word.
    let proof = prove_machine(&config, &pk, &circuit, &circuit, &rows);
    assert_eq!(proof.circuits.len(), 1);
    assert_eq!(proof.circuits[0].ports.len(), 3);
    verify(
        &config,
        instances(&vk, &circuit),
        &proof,
        0,
        &mut challenger(),
    )
    .unwrap();

    // The report charges both of Flock's reductions and clears 128 bits but for the hash.
    let report = security_report(&config, &instances(&vk, &circuit)).unwrap();
    for label in [p3_flock::ZEROCHECK_LABEL, p3_flock::LINCHECK_LABEL] {
        let term = report
            .terms()
            .iter()
            .find(|term| term.label == label)
            .expect("the report charges every Flock reduction");
        assert!(term.bits.bits() > 160.0, "{label}: {}", term.bits.bits());
    }
    let composed = report.security_bits().expect("every component is assessed");
    assert!(composed > 127.99, "composed {composed}");
}

#[test]
fn a_witness_of_another_circuit_is_rejected() {
    // Fixture state: the witness computes a + (b xor 1), and the requests carry those sums.
    //
    // Mutation: the proof claims the plain adder ran on every row.
    //
    //     bus       balanced: both tables hold the same wrong sums
    //     circuit   unsatisfied: the committed products are not a + b's
    //     -> the lincheck's walk of the adder's matrices misses the zerocheck's claims
    let circuit = adder(false);
    let config = PackedConfig::new(&circuit);
    let (pk, vk) = keys(&config);
    let proof = prove_machine(&config, &pk, &circuit, &adder(true), &operands(2));

    let error = verify(
        &config,
        instances(&vk, &circuit),
        &proof,
        0,
        &mut challenger(),
    )
    .expect_err("a witness of another circuit must not verify");
    assert!(
        matches!(
            error,
            VerificationError::Circuit { instance: 1, ref error }
                if matches!(**error, CircuitError::Flock(_))
        ),
        "{error:?}"
    );
}

#[test]
fn a_forged_port_value_is_rejected() {
    // Fixture state: an honest proof of the plain adder.
    let circuit = adder(false);
    let config = PackedConfig::new(&circuit);
    let (pk, vk) = keys(&config);
    let mut proof = prove_machine(&config, &pk, &circuit, &circuit, &operands(3));

    // Mutation: the output port's value at the bound point moves by one.
    //
    //     zerocheck closing   reads the moved value through the bus share
    //     port claim          weighs the witness and no longer meets it
    proof.circuits[0].ports[2] += EF::ONE;
    assert!(
        verify(
            &config,
            instances(&vk, &circuit),
            &proof,
            0,
            &mut challenger()
        )
        .is_err()
    );
}

#[test]
fn a_verifier_holding_another_circuit_rejects() {
    // Fixture state: an honest proof of the plain adder.
    let circuit = adder(false);
    let config = PackedConfig::new(&circuit);
    let (pk, vk) = keys(&config);
    let proof = prove_machine(&config, &pk, &circuit, &circuit, &operands(4));

    // Mutation: the verifier checks the adder that flips a bit of b.
    //
    // Its digest seeds the statement, so every challenge moves and the replay fails.
    let other = adder(true);
    assert_ne!(circuit.digest(), other.digest());
    assert!(
        verify(
            &config,
            instances(&vk, &other),
            &proof,
            0,
            &mut challenger()
        )
        .is_err()
    );
}

#[test]
fn port_weights_read_packed_bits_as_their_word() {
    // The port claims weigh bit v of a word by X^v, which must be the packing's own basis.
    let basis = coordinate_basis::<F>();
    let word = 0x0123_4567_89ab_cdef_u64;
    let rebuilt = (0..WORD)
        .filter(|&bit| (word >> bit) & 1 == 1)
        .map(|bit| basis[bit])
        .sum::<F>();
    assert_eq!(rebuilt, F::new(word));
}

#[test]
fn a_lone_circuit_table_proves() {
    // Fixture state: the adder alone, no other table and no bus.
    //
    // Its stage holds one AIR with no constraint of its own, so the circuit is the whole statement.
    struct Ports;
    impl BaseAir<F> for Ports {
        fn width(&self) -> usize {
            3
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            Vec::new()
        }
    }
    impl<AB: BusInteractionBuilder<F = F>> Air<AB> for Ports {
        fn eval(&self, _builder: &mut AB) {}
    }

    let circuit = adder(false);
    let shapes = [TableShape::new(LOG_ROWS + circuit.k_log(), 1)];
    let config = PackedConfig::with_shapes(&shapes);
    let (pk, vk) = setup(&config, &[&Ports], &mut challenger()).unwrap();
    let rows = operands(5);
    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::circuit(
            &Ports,
            &circuit,
            witness(&circuit, &rows),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();
    let instances = || {
        VerifierInstances::new(vec![VerifierInstance::circuit(
            &Ports,
            &circuit,
            &vk,
            LOG_ROWS,
            &[],
        )])
    };
    verify(&config, instances(), &proof, 0, &mut challenger()).unwrap();

    // The circuit is the whole statement, and the report still clears 128 bits but for the hash.
    let report = security_report(&config, &instances()).unwrap();
    let composed = report.security_bits();
    assert!(
        composed.is_some_and(|bits| bits > 127.99),
        "{composed:?}: {:#?} unassessed {:?}",
        report.terms(),
        report.unassessed_components()
    );
}

#[test]
fn a_circuit_table_read_by_an_air_of_another_width_is_refused() {
    // Fixture state: an honest proof of the plain adder, its AIR reading three port words.
    let circuit = adder(false);
    let config = PackedConfig::new(&circuit);
    let (pk, vk) = keys(&config);
    let proof = prove_machine(&config, &pk, &circuit, &circuit, &operands(6));

    // Mutation: the verifier pairs the adder with the requests AIR, 192 columns for 3 ports.
    //
    //     ports carried   3
    //     columns read    192
    //     -> refused before any column is read past the ports
    let instances = VerifierInstances::new(vec![
        VerifierInstance::new(&CHIPS[0], &vk, LOG_ROWS, &[]),
        VerifierInstance::circuit(&CHIPS[0], &circuit, &vk, LOG_ROWS, &[]),
    ]);
    assert!(matches!(
        verify(&config, instances, &proof, 0, &mut challenger()),
        Err(VerificationError::CircuitShape { instance: 1 })
    ));
}
