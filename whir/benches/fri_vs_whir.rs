//! Head-to-head FRI vs WHIR comparison.
//!
//! Configuration:
//! - Merkle hash: Poseidon1 (arithmetic, field-element digests).
//! - Message size: 2^22 elements.
//!
//! FRI is reported in two matrix shapes:
//! - single-polynomial (width 1) — natural pair to WHIR's single multilinear,
//!   but not plonky3 FRI's optimised regime;
//! - batch (width 2^LOG_FRI_BATCH_WIDTH) — the regime plonky3 FRI is tuned for,
//!   matching real STARK trace shapes.
//!
//! Run with: `cargo bench -p p3-whir --bench fri_vs_whir`

use std::time::Instant;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, Pcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PackedValue};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{
    KoalaBear, Poseidon1KoalaBear, default_koalabear_poseidon1_16, default_koalabear_poseidon1_24,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::constraints::statement::EqStatement;
use p3_whir::constraints::statement::initial::InitialStatement;
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy, WhirConfig,
};
use p3_whir::pcs::committer::reader::CommitmentReader;
use p3_whir::pcs::committer::writer::CommitmentWriter;
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::prover::WhirProver;
use p3_whir::pcs::verifier::WhirVerifier;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// Shared substrate that does not depend on the hash function.

/// Base field used for the committed message.
type F = KoalaBear;

/// Challenge field used for Fiat-Shamir challenges and out-of-domain samples.
type EF = BinomialExtensionField<F, 4>;

/// DFT backend used by both protocols.
type Dft = Radix2DFTSmallBatch<F>;

// Polynomial shape: FRI is univariate, WHIR is multilinear.
//
// Both protocols commit to a message of 2^m field elements:
//
//     FRI    coefficients of a univariate polynomial of degree < 2^m
//     WHIR   evaluations of an m-variate multilinear polynomial
//
// Shared substrate (matched across both sides):
//
//     code rate          rho = 2^(-log_blowup)
//     evaluation domain  |L| = 2^(m + log_blowup)
//     hash, MMCS, challenger, DFT, RNG seed
//
// Opening points live in different spaces:
//
//     FRI    z in EF          (one field element)
//     WHIR   z in EF^m        (m-tuple on the Boolean cube)
//
// Both encode the same claim: f(z) = v at one random query.
// A univariate evaluation at z equals the multilinear evaluation at
//
//     (z, z^2, z^4, ..., z^(2^(m-1)))
//
// so the two views are interchangeable.
//
// Anchor: vary m, lock every other knob.

// Soundness regimes for Reed-Solomon proximity testing:
//
//     regime           status        proximity bound
//     unique-decoding  proven        (1 - rho) / 2
//     Johnson          proven        1 - sqrt(rho)
//     capacity         conjectured   1 - rho - eta
//
// Constraint: plonky3 FRI exposes only the capacity-regime formula
//             `log_blowup * num_queries + pow_bits`. Stricter
//             regimes are not surfaced by the crate.
//
// Choice:     WHIR runs in the capacity regime to match.
//             A stricter WHIR setting would force more queries
//             than FRI pays for, biasing the comparison.
//
// Outcome:    same soundness bits under the same conjecture.

/// Target soundness in bits for both protocols.
///
/// Interpreted under the capacity-regime conjecture (see security framing above).
const SECURITY_LEVEL: usize = 100;

/// Proof-of-work grinding bits.
///
/// Smallest budget that lets WHIR's summed per-round PoW reach the target soundness
/// across the full message-size sweep. Shared by both protocols.
const POW_BITS: usize = 22;

/// log_2 of the inverse code rate.
///
/// `1` corresponds to rate `rho = 1/2`, the most common production setting.
/// - On the FRI side this is `log_blowup`;
/// - On the WHIR side it is `starting_log_inv_rate`.
const LOG_BLOWUP: usize = 1;

/// WHIR folding parameter `k`.
///
/// The paper recommends `k = 4` as a good trade-off between query count and per-round cost.
const WHIR_FOLDING_FACTOR: usize = 4;

/// Initial WHIR domain reduction factor before the first fold.
///
/// `1` halves the domain on the first round, matching the steady-state behaviour
/// of subsequent rounds.
const WHIR_RS_INITIAL_REDUCTION: usize = 1;

/// FRI folding arity log.
///
/// `1` is binary folding (arity 2).
const FRI_MAX_LOG_ARITY: usize = 1;

/// FRI final-polynomial truncation log.
///
/// `0` means the protocol opens the final polynomial at one point.
const FRI_LOG_FINAL_POLY_LEN: usize = 0;

/// FRI commit-phase PoW bits.
///
/// We keep all PoW on the query phase to match `new_benchmark` conventions.
const FRI_COMMIT_POW_BITS: usize = 0;

/// FRI per-query PoW bits.
const FRI_QUERY_POW_BITS: usize = POW_BITS;

/// Minimum FRI query count that reaches the target soundness under the ethSTARK conjecture.
///
/// Solves `log_blowup * queries + query_pow_bits >= security_level` for `queries`.
const FRI_NUM_QUERIES: usize = SECURITY_LEVEL.div_ceil(LOG_BLOWUP) - FRI_QUERY_POW_BITS;

/// log_2 of the matrix width for the production-shape FRI variant.
///
/// plonky3 FRI is implemented as a batch FRI tuned for wide matrices (typical
/// AIR traces have hundreds to thousands of columns). Single-polynomial FRI
/// is therefore not its optimised regime.
///
/// `8` gives a width of 256, which is:
///
/// - wide enough to put plonky3 FRI in its tuned regime;
/// - small enough that, at the lowest message size in the sweep, the
///   per-column polynomial still has 2^10 evaluations (so per-round FRI
///   work is not dominated by overhead).
///
/// 1024 (closer to a real Poseidon2-trace width) is also defensible; we
/// keep 256 to stay safe at the small end of the sweep.
const LOG_FRI_BATCH_WIDTH: usize = 8;

/// Message-size sweep.
///
/// `m = 22` matches the largest case in the WHIR paper Table 2 first row.
const M_VALUES: [usize; 1] = [22];

/// RNG seed used to make the bench deterministic across runs.
const BENCH_SEED: u64 = 0xA17_5C0DE;

/// Bound bundle the WHIR helpers require on the Merkle MMCS type.
///
/// Every concrete hash backend in this bench satisfies these bounds.
trait WhirMmcs: Mmcs<F> + Clone + Send + Sync {}

impl<T: Mmcs<F> + Clone + Send + Sync> WhirMmcs for T {}

/// Bound bundle the WHIR helpers require on the Fiat-Shamir challenger.
trait WhirChallenger<MT: Mmcs<F>>:
    FieldChallenger<F>
    + GrindingChallenger<Witness = F>
    + CanSampleUniformBits<F>
    + CanObserve<MT::Commitment>
    + Clone
{
}
impl<MT: Mmcs<F>, T> WhirChallenger<MT> for T where
    T: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>
        + Clone
{
}

/// Carrier for one fully-prepared WHIR setup at a given `num_variables`.
///
/// The MMCS and challenger types are generic so the same struct can carry any
/// hash backend.
struct WhirRig<MT: Mmcs<F>, Ch> {
    /// Logarithm of the message length.
    num_variables: usize,
    /// Derived per-round protocol configuration.
    config: WhirConfig<EF, F, MT, Ch>,
    /// Protocol parameters retained because the proof builder takes them by reference.
    params: ProtocolParameters<MT>,
    /// Pre-allocated DFT with twiddle factors for the maximum FFT size used by this rig.
    dft: Dft,
    /// Initial statement: holds the random message and its evaluation constraints.
    statement: InitialStatement<F, EF>,
    /// Equality-form statement consumed by the verifier.
    verifier_statement: EqStatement<EF>,
    /// Domain separator used for both prover and verifier transcripts.
    domain_separator: DomainSeparator<EF, F>,
    /// Base challenger; both prover and verifier clone from this.
    challenger: Ch,
}

/// Build a fresh WHIR rig for the given message size.
fn whir_setup<MT, Ch>(num_variables: usize, mmcs: MT, base_challenger: Ch) -> WhirRig<MT, Ch>
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    // WHIR protocol knobs collected into one struct.
    let params = ProtocolParameters {
        security_level: SECURITY_LEVEL,
        pow_bits: POW_BITS,
        folding_factor: FoldingFactor::Constant(WHIR_FOLDING_FACTOR),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: LOG_BLOWUP,
        rs_domain_initial_reduction_factor: WHIR_RS_INITIAL_REDUCTION,
    };

    // Per-round protocol layout: query counts, OOD samples, PoW bits per round.
    let config = WhirConfig::<EF, F, MT, Ch>::new(num_variables, params.clone());

    // Per-rig RNG: distinct seed per `num_variables` so two rigs can't accidentally collide.
    let mut rng = SmallRng::seed_from_u64(BENCH_SEED ^ ((num_variables as u64) << 16));

    // Random multilinear polynomial laid out as one coefficient per Boolean cube point.
    let polynomial = Poly::<F>::new((0..(1 << num_variables)).map(|_| rng.random()).collect());

    // Build the initial statement and register one random evaluation claim.
    let mut statement = config.initial_statement(polynomial, SumcheckStrategy::default());
    let z = Point::<EF>::rand(&mut rng, num_variables);
    let _ = statement.evaluate(&z);

    // Equality-form statement consumed by the verifier (same data, normalised layout).
    let verifier_statement = statement.clone().normalize();

    // Pre-allocate twiddle factors up to the largest FFT size used by this rig.
    let dft = Dft::new(1 << config.max_fft_size());

    // Domain separator: encodes the protocol shape into the transcript so challenges
    // are bound to this exact configuration.
    let mut domain_separator = DomainSeparator::new(vec![]);
    domain_separator.commit_statement::<_, _, 32>(&config);
    domain_separator.add_whir_proof::<_, _, 32>(&config);

    WhirRig {
        num_variables,
        config,
        params,
        dft,
        statement,
        verifier_statement,
        domain_separator,
        challenger: base_challenger,
    }
}

/// Run one full WHIR proving cycle (commit + open).
fn whir_prove_full<MT, Ch>(rig: &WhirRig<MT, Ch>) -> (WhirProof<F, EF, MT>, u128, u128)
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    // Fresh prover challenger seeded with the domain separator.
    let mut prover_challenger = rig.challenger.clone();
    rig.domain_separator
        .observe_domain_separator(&mut prover_challenger);

    // Each iteration mutates a fresh statement and proof.
    let mut statement = rig.statement.clone();
    let mut proof =
        WhirProof::<F, EF, MT>::from_protocol_parameters(&rig.params, rig.num_variables);

    // Phase 1: commit (DFT + Merkle + OOD samples).
    let committer = CommitmentWriter::new(&rig.config);
    let t = Instant::now();
    let prover_data = committer
        .commit(&rig.dft, &mut proof, &mut prover_challenger, &mut statement)
        .expect("WHIR commit failed");
    let commit_ms = t.elapsed().as_millis();

    // Phase 2: open (multi-round sumcheck + STIR queries + PoW).
    let prover = WhirProver(&rig.config);
    let t = Instant::now();
    prover
        .prove(
            &rig.dft,
            &mut proof,
            &mut prover_challenger,
            &statement,
            prover_data,
        )
        .expect("WHIR prove failed");
    let open_ms = t.elapsed().as_millis();

    (proof, commit_ms, open_ms)
}

/// Run one full WHIR verification cycle and assert it accepts.
///
/// `W` is the digest element type and `DIG` the count, both decided by the MMCS:
/// - for arithmetic hashes `(W = F, DIG = 8)`,
/// - for byte hashes `(W = u8, DIG = 32)`.
fn whir_verify_full<MT, Ch, W, const DIG: usize>(
    rig: &WhirRig<MT, Ch>,
    proof: &WhirProof<F, EF, MT>,
) -> u128
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
    W: PackedValue<Value = W> + Eq + Copy,
{
    let mut verifier_challenger = rig.challenger.clone();
    rig.domain_separator
        .observe_domain_separator(&mut verifier_challenger);

    let commitment_reader = CommitmentReader::new(&rig.config);
    let parsed_commitment =
        commitment_reader.parse_commitment::<W, DIG>(proof, &mut verifier_challenger);

    let verifier = WhirVerifier::new(&rig.config);
    let t = Instant::now();
    verifier
        .verify(
            proof,
            &mut verifier_challenger,
            &parsed_commitment,
            rig.verifier_statement.clone(),
        )
        .expect("WHIR verify failed");
    t.elapsed().as_micros()
}

/// Bound bundle the FRI helpers require on the input MMCS type.
trait FriInputMmcs: Mmcs<F, Proof: Sync, Error: Sync> + Clone + Send + Sync {}
impl<T: Mmcs<F, Proof: Sync, Error: Sync> + Clone + Send + Sync> FriInputMmcs for T {}

/// Bound bundle the FRI helpers require on the per-round MMCS type.
trait FriChallengeMmcs: Mmcs<EF> + Clone + Send + Sync {}
impl<T: Mmcs<EF> + Clone + Send + Sync> FriChallengeMmcs for T {}

/// Bound bundle the FRI helpers require on the Fiat-Shamir challenger.
trait FriChal<InMmcs: Mmcs<F>, ChMmcs: Mmcs<EF>>:
    FieldChallenger<F>
    + GrindingChallenger<Witness = F>
    + CanObserve<<InMmcs as Mmcs<F>>::Commitment>
    + CanObserve<<ChMmcs as Mmcs<EF>>::Commitment>
    + Clone
{
}
impl<InMmcs: Mmcs<F>, ChMmcs: Mmcs<EF>, T> FriChal<InMmcs, ChMmcs> for T where
    T: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<<InMmcs as Mmcs<F>>::Commitment>
        + CanObserve<<ChMmcs as Mmcs<EF>>::Commitment>
        + Clone
{
}

/// One prepared FRI setup at a fixed message size and matrix width.
///
/// # Layout
///
/// The committed matrix has shape:
///
/// ```text
///     rows    = 2^(log_n - log_width)
///     columns = 2^log_width
///     total committed elements = 2^log_n   (independent of log_width)
/// ```
///
/// # Width regimes
///
/// - `log_width = 0` — one polynomial of size `2^log_n`. The classic FRI statement.
///   This pairs naturally with WHIR's single-multilinear shape, but is **not**
///   plonky3 FRI's optimised regime.
/// - `log_width > 0` — batched FRI on `2^log_width` polynomials of size
///   `2^(log_n - log_width)`, tested jointly via random linear combination.
///   This is the shape plonky3 FRI is actually tuned for and what real STARK
///   traces look like.
struct FriRig<InMmcs, ChMmcs, Ch>
where
    InMmcs: Mmcs<F>,
    ChMmcs: Mmcs<EF>,
{
    pcs: TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs>,
    domain: TwoAdicMultiplicativeCoset<F>,
    message: RowMajorMatrix<F>,
    challenger: Ch,
}

/// FRI proof type for this bench's MMCS configuration.
type FriProofTy<InMmcs, ChMmcs> = p3_fri::FriProof<EF, ChMmcs, F, Vec<BatchOpening<F, InMmcs>>>;

/// FRI commitment type for an input MMCS.
type FriCommitTy<InMmcs> = <InMmcs as Mmcs<F>>::Commitment;

/// Build a fresh FRI rig for the given message size and matrix width.
///
/// See the rig type's documentation for the meaning of `log_n` and `log_width`.
fn fri_setup<InMmcs, ChMmcs, Ch>(
    log_n: usize,
    log_width: usize,
    val_mmcs: InMmcs,
    challenge_mmcs: ChMmcs,
    base_challenger: Ch,
) -> FriRig<InMmcs, ChMmcs, Ch>
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    // Protocol parameters matched to the same target soundness as WHIR.
    let fri_params = FriParameters {
        log_blowup: LOG_BLOWUP,
        log_final_poly_len: FRI_LOG_FINAL_POLY_LEN,
        max_log_arity: FRI_MAX_LOG_ARITY,
        num_queries: FRI_NUM_QUERIES,
        commit_proof_of_work_bits: FRI_COMMIT_POW_BITS,
        query_proof_of_work_bits: FRI_QUERY_POW_BITS,
        mmcs: challenge_mmcs,
    };

    // Matrix shape:
    //
    //     rows per column = 2^(log_n - log_width)
    //     columns         = 2^log_width
    //     LDE rows        = 2^(log_n - log_width + log_blowup)
    //
    // Total committed = 2^log_n field elements regardless of log_width.
    let log_height = log_n - log_width;
    let width = 1 << log_width;

    // Pre-size the twiddle table to the per-column LDE row count.
    let dft = Dft::new(1 << (log_height + LOG_BLOWUP));
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

    // RNG seed mixes the width so width-1 and width-N rigs at the same `log_n`
    // do not share polynomial samples.
    let mut rng = SmallRng::seed_from_u64(
        BENCH_SEED ^ ((log_n as u64) << 16) ^ ((log_width as u64) << 8) ^ 0xF1,
    );

    // Random row-major matrix sized exactly as the layout above prescribes.
    let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, width);

    // UFCS spelling pins the challenger generic so trait selection is unambiguous.
    let domain = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::natural_domain_for_degree(
        &pcs,
        1 << log_height,
    );

    FriRig {
        pcs,
        domain,
        message,
        challenger: base_challenger,
    }
}

/// Run one full FRI proving cycle (commit + open).
///
/// # Returns
///
/// - the commitment Merkle root;
/// - the FRI proof;
/// - the opening point sampled from the transcript;
/// - one evaluation per committed column at that opening point;
/// - the commit-phase wall-clock in milliseconds;
/// - the open-phase wall-clock in milliseconds.
#[allow(clippy::type_complexity)]
fn fri_prove_full<InMmcs, ChMmcs, Ch>(
    rig: &FriRig<InMmcs, ChMmcs, Ch>,
) -> (
    FriCommitTy<InMmcs>,
    FriProofTy<InMmcs, ChMmcs>,
    EF,
    Vec<EF>,
    u128,
    u128,
)
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    // Each iteration starts from a fresh prover transcript.
    let mut prover_challenger = rig.challenger.clone();

    // Phase 1 — commit: coset LDE of the message, then a Merkle tree over the LDE.
    let t = Instant::now();
    let (commit, prover_data) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::commit(
        &rig.pcs,
        [(rig.domain, rig.message.clone())],
    );
    let commit_ms = t.elapsed().as_millis();

    // Bind the Merkle root into the transcript before drawing the opening point,
    // so the verifier can re-derive the same point during verification.
    prover_challenger.observe(commit.clone());
    let zeta: EF = prover_challenger.sample_algebra_element();

    // Open-call argument shape:
    //
    //     data_and_points : one entry per commitment           (here: 1)
    //         per entry   : one list per matrix in commitment  (here: 1)
    //             per list: one entry per opening point        (here: [zeta])
    let data_and_points = vec![(&prover_data, vec![vec![zeta]])];

    // Phase 2 — open: FRI commit phase + query phase + proof-of-work grind.
    let t = Instant::now();
    let (openings, proof) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::open(
        &rig.pcs,
        data_and_points,
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    // `openings` is a 4-level nested Vec indexed by:
    //
    //     openings[round][matrix][point][column]
    //
    // For our shape (1 commitment, 1 matrix, 1 point, `width` columns) the inner
    // slice carries every committed polynomial's evaluation at `zeta`.
    let values = openings[0][0][0].clone();

    (commit, proof, zeta, values, commit_ms, open_ms)
}

/// Run one full FRI verification cycle and assert it accepts.
fn fri_verify_full<InMmcs, ChMmcs, Ch>(
    rig: &FriRig<InMmcs, ChMmcs, Ch>,
    commit: &FriCommitTy<InMmcs>,
    proof: &FriProofTy<InMmcs, ChMmcs>,
    zeta: EF,
    values: &[EF],
) -> u128
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    // Each verification call starts from a fresh transcript clone.
    let mut verifier_challenger = rig.challenger.clone();

    // Re-derive the opening point and confirm it matches the prover's. A drift
    // here means the bench wired up the challengers wrong.
    verifier_challenger.observe(commit.clone());
    let derived: EF = verifier_challenger.sample_algebra_element();
    assert_eq!(derived, zeta, "verifier challenger drifted from prover");

    // Verify-call argument shape:
    //
    //     claims      : one entry per commitment           (here: 1)
    //         per entry  : one list per matrix             (here: 1)
    //             per list: (point, per-column evaluations) at that point
    let claims = vec![(
        commit.clone(),
        vec![(rig.domain, vec![(zeta, values.to_vec())])],
    )];

    let t = Instant::now();
    <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::verify(
        &rig.pcs,
        claims,
        proof,
        &mut verifier_challenger,
    )
    .expect("FRI verify failed");
    t.elapsed().as_micros()
}

/// Poseidon1-backed Merkle + duplex challenger.
mod poseidon1 {
    use super::*;

    pub const NAME: &str = "poseidon1";

    pub type Perm16 = Poseidon1KoalaBear<16>;
    pub type Perm24 = Poseidon1KoalaBear<24>;
    pub type MerkleHash = PaddingFreeSponge<Perm24, 24, 16, 8>;
    pub type MerkleCompress = TruncatedPermutation<Perm16, 2, 8, 16>;
    pub type PackedF = <F as Field>::Packing;
    pub type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    pub type Challenger = DuplexChallenger<F, Perm16, 16, 8>;

    /// Digest element type; for arithmetic hashes the digest is in the base field.
    pub type DigestW = F;
    /// Number of base-field elements per digest.
    pub const DIGEST_ELEMS: usize = 8;

    /// Build the Merkle backend, the lifted challenge MMCS, and a base challenger.
    pub fn build_kit() -> (Challenger, ValMmcs, ChallengeMmcs) {
        // Round constants are shipped by the koala-bear crate; no RNG needed.
        let perm16 = default_koalabear_poseidon1_16();
        let perm24 = default_koalabear_poseidon1_24();
        let merkle_hash = MerkleHash::new(perm24);
        let merkle_compress = MerkleCompress::new(perm16.clone());
        let val_mmcs = ValMmcs::new(merkle_hash, merkle_compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let challenger = Challenger::new(perm16);
        (challenger, val_mmcs, challenge_mmcs)
    }
}

/// Diagnostic table: per-phase timings and proof bytes for every (hash, m, protocol) triple.
fn print_diagnostic_table() {
    println!();
    println!(
        "=== FRI vs WHIR diagnostic ({SECURITY_LEVEL}-bit security, rho = 2^-{LOG_BLOWUP}) ==="
    );
    println!("  hash      |  m | proto | commit ms | open ms | verify us | proof bytes | queries");
    println!("------------+----+-------+-----------+---------+-----------+-------------+--------");

    // Per-hash dispatch. A macro is needed because each hash module exposes a
    // different set of type aliases (MMCS, challenger, digest type), so each
    // call site monomorphises differently.
    //
    // Three protocols are printed per `(hash, m)` cell:
    //
    //   - fri-single : single-polynomial FRI (matrix width 1). Apples-to-apples
    //                  shape with WHIR's single-multilinear opening, but not
    //                  plonky3 FRI's optimised regime.
    //   - fri-batch  : batched FRI with matrix width 2^LOG_FRI_BATCH_WIDTH.
    //                  This is plonky3 FRI's tuned regime and the shape real
    //                  STARK traces have.
    //   - whir       : WHIR as a polynomial commitment scheme.
    macro_rules! diag_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();

                // Single-polynomial FRI (width 1) — apples-to-apples with WHIR's shape.
                let fri_rig = fri_setup(
                    m,
                    0,
                    val_mmcs.clone(),
                    challenge_mmcs.clone(),
                    challenger.clone(),
                );
                let (commit, fri_proof, zeta, values, fri_commit_ms, fri_open_ms) =
                    fri_prove_full(&fri_rig);
                let fri_verify_us = fri_verify_full(&fri_rig, &commit, &fri_proof, zeta, &values);
                let fri_bytes = postcard::to_allocvec(&fri_proof)
                    .expect("postcard FRI")
                    .len();

                println!(
                    " {:<10} | {:>2} | fri-s | {:>9} | {:>7} | {:>9} | {:>11} | {} (single-shot)",
                    $module::NAME,
                    m,
                    fri_commit_ms,
                    fri_open_ms,
                    fri_verify_us,
                    fri_bytes,
                    FRI_NUM_QUERIES,
                );

                drop(fri_proof);
                drop(fri_rig);

                // Batch FRI (matrix width 2^LOG_FRI_BATCH_WIDTH) — plonky3 FRI's
                // tuned regime; same total committed elements as the width-1 case.
                let frib_rig = fri_setup(
                    m,
                    LOG_FRI_BATCH_WIDTH,
                    val_mmcs.clone(),
                    challenge_mmcs,
                    challenger.clone(),
                );
                let (
                    frib_commit,
                    frib_proof,
                    frib_zeta,
                    frib_values,
                    frib_commit_ms,
                    frib_open_ms,
                ) = fri_prove_full(&frib_rig);
                let frib_verify_us = fri_verify_full(
                    &frib_rig,
                    &frib_commit,
                    &frib_proof,
                    frib_zeta,
                    &frib_values,
                );
                let frib_bytes = postcard::to_allocvec(&frib_proof)
                    .expect("postcard FRI batch")
                    .len();

                println!(
                    " {:<10} | {:>2} | fri-b | {:>9} | {:>7} | {:>9} | {:>11} | {} (single-shot, {} cols)",
                    $module::NAME,
                    m,
                    frib_commit_ms,
                    frib_open_ms,
                    frib_verify_us,
                    frib_bytes,
                    FRI_NUM_QUERIES,
                    1 << LOG_FRI_BATCH_WIDTH,
                );

                drop(frib_proof);
                drop(frib_rig);

                // WHIR as a polynomial commitment scheme.
                let whir_rig = whir_setup(m, val_mmcs, challenger);
                let (whir_proof, whir_commit_ms, whir_open_ms) = whir_prove_full(&whir_rig);
                let whir_verify_us =
                    whir_verify_full::<_, _, $module::DigestW, { $module::DIGEST_ELEMS }>(
                        &whir_rig,
                        &whir_proof,
                    );
                let whir_bytes = postcard::to_allocvec(&whir_proof)
                    .expect("postcard WHIR")
                    .len();

                let whir_queries = whir_rig
                    .config
                    .round_parameters
                    .iter()
                    .map(|r| r.num_queries.to_string())
                    .join(",");

                println!(
                    " {:<10} | {:>2} | whir  | {:>9} | {:>7} | {:>9} | {:>11} | [{}]",
                    $module::NAME,
                    m,
                    whir_commit_ms,
                    whir_open_ms,
                    whir_verify_us,
                    whir_bytes,
                    whir_queries,
                );

                assert!(
                    whir_rig.config.check_pow_bits(),
                    "WHIR PoW budget below {SECURITY_LEVEL}-bit target at m = {m}"
                );

                drop(whir_proof);
                drop(whir_rig);
            }
        }};
    }

    diag_block!(poseidon1);

    println!();
}

// FRI soundness sanity is hash-agnostic and constant: enforce at compile time.
const _: () = assert!(
    LOG_BLOWUP * FRI_NUM_QUERIES + FRI_QUERY_POW_BITS >= SECURITY_LEVEL,
    "FRI soundness budget below target"
);

/// Bench total proving time (commit + open) for both protocols across all hash backends.
fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcs/fri_vs_whir/prove");
    group.sample_size(10);

    macro_rules! prove_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();

                // Single-polynomial FRI.
                let fri_rig = fri_setup(
                    m,
                    0,
                    val_mmcs.clone(),
                    challenge_mmcs.clone(),
                    challenger.clone(),
                );
                let label_fri = format!("fri-single/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_fri, m), &m, |b, _| {
                    b.iter_batched(
                        || (),
                        |()| {
                            let _ = fri_prove_full(&fri_rig);
                        },
                        BatchSize::LargeInput,
                    );
                });

                // Batch FRI in plonky3's tuned regime.
                let frib_rig = fri_setup(
                    m,
                    LOG_FRI_BATCH_WIDTH,
                    val_mmcs.clone(),
                    challenge_mmcs,
                    challenger.clone(),
                );
                let label_frib = format!("fri-batch/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_frib, m), &m, |b, _| {
                    b.iter_batched(
                        || (),
                        |()| {
                            let _ = fri_prove_full(&frib_rig);
                        },
                        BatchSize::LargeInput,
                    );
                });

                let whir_rig = whir_setup(m, val_mmcs, challenger);
                let label_whir = format!("whir/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_whir, m), &m, |b, _| {
                    b.iter_batched(
                        || (),
                        |()| {
                            let _ = whir_prove_full(&whir_rig);
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
        }};
    }

    prove_block!(poseidon1);

    group.finish();
}

/// Bench verification time for both protocols across all hash backends.
fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcs/fri_vs_whir/verify");

    macro_rules! verify_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();

                // Single-polynomial FRI: pre-prove once so the inner loop only
                // measures verification.
                let fri_rig = fri_setup(
                    m,
                    0,
                    val_mmcs.clone(),
                    challenge_mmcs.clone(),
                    challenger.clone(),
                );
                let (commit, fri_proof, zeta, values, _, _) = fri_prove_full(&fri_rig);
                let label_fri = format!("fri-single/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_fri, m), &m, |b, _| {
                    b.iter(|| {
                        fri_verify_full(&fri_rig, &commit, &fri_proof, zeta, &values);
                    });
                });

                // Batch FRI: same pre-prove pattern.
                let frib_rig = fri_setup(
                    m,
                    LOG_FRI_BATCH_WIDTH,
                    val_mmcs.clone(),
                    challenge_mmcs,
                    challenger.clone(),
                );
                let (frib_commit, frib_proof, frib_zeta, frib_values, _, _) =
                    fri_prove_full(&frib_rig);
                let label_frib = format!("fri-batch/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_frib, m), &m, |b, _| {
                    b.iter(|| {
                        fri_verify_full(
                            &frib_rig,
                            &frib_commit,
                            &frib_proof,
                            frib_zeta,
                            &frib_values,
                        );
                    });
                });

                let whir_rig = whir_setup(m, val_mmcs, challenger);
                let (whir_proof, _, _) = whir_prove_full(&whir_rig);
                let label_whir = format!("whir/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_whir, m), &m, |b, _| {
                    b.iter(|| {
                        whir_verify_full::<_, _, $module::DigestW, { $module::DIGEST_ELEMS }>(
                            &whir_rig,
                            &whir_proof,
                        );
                    });
                });
            }
        }};
    }

    verify_block!(poseidon1);

    group.finish();
}

/// Wrapper that runs the diagnostic table once before criterion takes over.
fn bench_with_diagnostic(c: &mut Criterion) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(print_diagnostic_table);

    bench_prove(c);
    bench_verify(c);
}

criterion_group!(benches, bench_with_diagnostic);
criterion_main!(benches);
