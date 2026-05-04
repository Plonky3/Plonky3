//! Head-to-head FRI vs WHIR comparison across three Merkle hash backends.
//!
//! Three hash modes are swept:
//! - Poseidon1, Poseidon2 (arithmetic; field-element digests),
//! - Blake3 (byte-oriented; 32-byte digests).
//!
//! For each hash the protocol-level parameters are otherwise locked.
//!
//! So the comparison isolates the hash-function effect.
//!
//! Run with: `cargo bench -p p3-whir --bench fri_vs_whir`

use std::time::Instant;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use p3_blake3::Blake3;
use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    HashChallenger, SerializingChallenger32,
};
use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, Pcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PackedValue};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{
    KoalaBear, Poseidon1KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon1_16,
    default_koalabear_poseidon1_24,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
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

/// Message-size sweep.
///
/// `m = 18` is a small but realistic SNARK trace width.
/// `m = 22` matches the largest case in the WHIR paper Table 2 first row.
const M_VALUES: [usize; 3] = [18, 20, 22];

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

/// Carrier for one fully-prepared FRI setup at a given message size.
///
/// The message is stored as a width-1 matrix: one polynomial, one column,
/// so the comparison mirrors WHIR's single-polynomial commitment.
///
/// Concrete domain type sidesteps the `Pcs<EF, Ch>` projection so the struct
/// definition does not need to repeat every Pcs-impl bound.
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

/// Concrete FRI proof type, expanded so the struct/alias does not need every
/// Pcs<EF, Ch> trait bound just to mention the type.
type FriProofTy<InMmcs, ChMmcs> = p3_fri::FriProof<EF, ChMmcs, F, Vec<BatchOpening<F, InMmcs>>>;

/// Concrete FRI commitment type for an input MMCS.
type FriCommitTy<InMmcs> = <InMmcs as Mmcs<F>>::Commitment;

/// Build a fresh FRI rig for the given message size.
fn fri_setup<InMmcs, ChMmcs, Ch>(
    log_n: usize,
    val_mmcs: InMmcs,
    challenge_mmcs: ChMmcs,
    base_challenger: Ch,
) -> FriRig<InMmcs, ChMmcs, Ch>
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    // FRI protocol knobs matched to the same target soundness as WHIR.
    let fri_params = FriParameters {
        log_blowup: LOG_BLOWUP,
        log_final_poly_len: FRI_LOG_FINAL_POLY_LEN,
        max_log_arity: FRI_MAX_LOG_ARITY,
        num_queries: FRI_NUM_QUERIES,
        commit_proof_of_work_bits: FRI_COMMIT_POW_BITS,
        query_proof_of_work_bits: FRI_QUERY_POW_BITS,
        mmcs: challenge_mmcs,
    };

    // Pre-allocate twiddle factors up to the LDE size = 2^(log_n + log_blowup).
    let dft = Dft::new(1 << (log_n + LOG_BLOWUP));
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

    // Per-rig RNG, distinct from the WHIR rig at the same size.
    let mut rng = SmallRng::seed_from_u64(BENCH_SEED ^ ((log_n as u64) << 16) ^ 0xF1);

    // Width-1 row-major matrix: 1 column x 2^log_n rows.
    let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_n, 1);

    // UFCS pins the challenger type so trait selection is unambiguous.
    let domain = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::natural_domain_for_degree(
        &pcs,
        1 << log_n,
    );

    FriRig {
        pcs,
        domain,
        message,
        challenger: base_challenger,
    }
}

/// Run one full FRI proving cycle (commit + open) and return the produced proof.
#[allow(clippy::type_complexity)]
fn fri_prove_full<InMmcs, ChMmcs, Ch>(
    rig: &FriRig<InMmcs, ChMmcs, Ch>,
) -> (
    FriCommitTy<InMmcs>,
    FriProofTy<InMmcs, ChMmcs>,
    EF,
    EF,
    u128,
    u128,
)
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    let mut prover_challenger = rig.challenger.clone();

    // Phase 1: commit (coset LDE + Merkle).
    let t = Instant::now();
    let (commit, prover_data) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::commit(
        &rig.pcs,
        [(rig.domain, rig.message.clone())],
    );
    let commit_ms = t.elapsed().as_millis();

    // Bind the commitment to the transcript so the FRI challenges are committed.
    prover_challenger.observe(commit.clone());
    let zeta: EF = prover_challenger.sample_algebra_element();

    // open() takes one entry per commitment;
    // Each entry pairs the prover data with a list of opening points per matrix.
    //
    //   - outer Vec : one entry per commitment       (here: 1)
    //   - middle Vec: one entry per matrix           (here: 1)
    //   - inner Vec : opening points for that matrix (here: [zeta])
    let data_and_points = vec![(&prover_data, vec![vec![zeta]])];

    // Phase 2: open (FRI commit phase + query phase + PoW).
    let t = Instant::now();
    let (openings, proof) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::open(
        &rig.pcs,
        data_and_points,
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    // openings is a 4-level nested Vec, indexed [round][matrix][point][column]:
    //
    //   - level 0 : one entry per round   (here: 1)
    //   - level 1 : one entry per matrix  (here: 1)
    //   - level 2 : one entry per point   (here: 1)
    //   - level 3 : one entry per column  (here: 1)
    let value = openings[0][0][0][0];

    (commit, proof, zeta, value, commit_ms, open_ms)
}

/// Run one full FRI verification cycle and assert it accepts.
fn fri_verify_full<InMmcs, ChMmcs, Ch>(
    rig: &FriRig<InMmcs, ChMmcs, Ch>,
    commit: &FriCommitTy<InMmcs>,
    proof: &FriProofTy<InMmcs, ChMmcs>,
    zeta: EF,
    value: EF,
) -> u128
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    let mut verifier_challenger = rig.challenger.clone();
    verifier_challenger.observe(commit.clone());
    let derived: EF = verifier_challenger.sample_algebra_element();
    assert_eq!(derived, zeta, "verifier challenger drifted from prover");

    // verify() takes one entry per commitment;
    // Each entry pairs the commitment with a list of (domain, claims) per matrix.
    //
    //   - outer Vec  : one entry per commitment        (here: 1)
    //   - middle Vec : one entry per matrix            (here: 1)
    //   - claims Vec : (point, per-column evaluations) (here: (zeta, [value]))
    let claims = vec![(
        commit.clone(),
        vec![(rig.domain, vec![(zeta, vec![value])])],
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

/// Poseidon2-backed Merkle + duplex challenger.
mod poseidon2 {
    use super::*;

    pub const NAME: &str = "poseidon2";

    pub type Perm16 = Poseidon2KoalaBear<16>;
    pub type Perm24 = Poseidon2KoalaBear<24>;
    pub type MerkleHash = PaddingFreeSponge<Perm24, 24, 16, 8>;
    pub type MerkleCompress = TruncatedPermutation<Perm16, 2, 8, 16>;
    pub type PackedF = <F as Field>::Packing;
    pub type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    pub type Challenger = DuplexChallenger<F, Perm16, 16, 8>;

    pub type DigestW = F;
    pub const DIGEST_ELEMS: usize = 8;

    pub fn build_kit() -> (Challenger, ValMmcs, ChallengeMmcs) {
        // Round constants are derived from a fixed seed for reproducibility.
        let mut rng = SmallRng::seed_from_u64(BENCH_SEED);
        let perm16 = Perm16::new_from_rng_128(&mut rng);
        let perm24 = Perm24::new_from_rng_128(&mut rng);
        let merkle_hash = MerkleHash::new(perm24);
        let merkle_compress = MerkleCompress::new(perm16.clone());
        let val_mmcs = ValMmcs::new(merkle_hash, merkle_compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let challenger = Challenger::new(perm16);
        (challenger, val_mmcs, challenge_mmcs)
    }
}

/// Blake3-backed Merkle + serializing byte challenger.
mod blake3_hash {
    use super::*;

    pub const NAME: &str = "blake3";

    /// Field-element hasher built by serialising to bytes and feeding into Blake3.
    pub type FieldHash = SerializingHasher<Blake3>;
    /// 2-to-1 compression of two 32-byte digests through Blake3.
    pub type MyCompress = CompressionFunctionFromHasher<Blake3, 2, 32>;
    pub type ValMmcs = MerkleTreeMmcs<F, u8, FieldHash, MyCompress, 2, 32>;
    pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    pub type Challenger = SerializingChallenger32<F, HashChallenger<u8, Blake3, 32>>;

    /// Digest element type; for byte hashes the digest is a u8 array.
    pub type DigestW = u8;
    /// Number of bytes per digest (Blake3 outputs 32 bytes).
    pub const DIGEST_ELEMS: usize = 32;

    pub fn build_kit() -> (Challenger, ValMmcs, ChallengeMmcs) {
        let byte_hash = Blake3;
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let challenger = Challenger::from_hasher(vec![], byte_hash);
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

    // Each closure runs the full diagnostic for one hash mode and prints its rows.
    // Closures are needed because the type universe differs across hashes.
    macro_rules! diag_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();
                let fri_rig = fri_setup(
                    m,
                    val_mmcs.clone(),
                    challenge_mmcs.clone(),
                    challenger.clone(),
                );
                let (commit, fri_proof, zeta, value, fri_commit_ms, fri_open_ms) =
                    fri_prove_full(&fri_rig);
                let fri_verify_us = fri_verify_full(&fri_rig, &commit, &fri_proof, zeta, value);
                let fri_bytes = postcard::to_allocvec(&fri_proof)
                    .expect("postcard FRI")
                    .len();

                println!(
                    " {:<10} | {:>2} | fri   | {:>9} | {:>7} | {:>9} | {:>11} | {} (single-shot)",
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
    diag_block!(poseidon2);
    diag_block!(blake3_hash);

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

                let fri_rig = fri_setup(m, val_mmcs.clone(), challenge_mmcs, challenger.clone());
                let label_fri = format!("fri/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_fri, m), &m, |b, _| {
                    b.iter_batched(
                        || (),
                        |()| {
                            let _ = fri_prove_full(&fri_rig);
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
    prove_block!(poseidon2);
    prove_block!(blake3_hash);

    group.finish();
}

/// Bench verification time for both protocols across all hash backends.
fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcs/fri_vs_whir/verify");

    macro_rules! verify_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();

                let fri_rig = fri_setup(m, val_mmcs.clone(), challenge_mmcs, challenger.clone());
                let (commit, fri_proof, zeta, value, _, _) = fri_prove_full(&fri_rig);
                let label_fri = format!("fri/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_fri, m), &m, |b, _| {
                    b.iter(|| {
                        fri_verify_full(&fri_rig, &commit, &fri_proof, zeta, value);
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
    verify_block!(poseidon2);
    verify_block!(blake3_hash);

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
