//! Head-to-head FRI vs WHIR comparison.
//!
//! # What the bench measures
//!
//! - Prover wall-clock (commit + open).
//! - Verifier wall-clock.
//! - Postcard-serialised proof size.
//!
//! # Substrate (matched on both sides)
//!
//! - Field            : KoalaBear, with quartic extension `EF`.
//! - Merkle hash      : swept over Poseidon1, Poseidon2 (field-element digests), Blake3 (byte digests).
//! - Message          : `2^22` base-field elements.
//! - Code rate        : `ρ = 2^-1` (`log_blowup = 1`).
//! - Soundness target : 100 bits, capacity-regime conjecture.
//!
//! # Workload (matched claim shape)
//!
//! Both protocols arrange the `2^22` elements as `256 = 2^8` polynomials
//! of size `2^14`, and open all 256 at one common point.
//!
//! `fri-batch`:
//! - polynomials  : 256 univariates, each of degree `< 2^14`
//! - opening point: `z ∈ EF`
//! - reveals      : `v_i = f_i(z)` for `i ∈ {0..256}`
//!
//! `whir`:
//! - polynomial   : 1 multilinear in 22 variables (= 256 stacked 14-var multilinears)
//! - opening point: 256 points sharing the trailing `ζ ∈ EF^14`
//! - reveals      : `v_i = p̃(i_binary, ζ)` for `i ∈ {0..256}`
//!
//! # Why the two claims are equivalent
//!
//! WHIR paper §1.1 bridge:
//!
//! ```text
//!     ζ = (z, z^2, z^4, …, z^{2^13})    ⇒    f_i(z) = p̃(i_binary, ζ)
//! ```
//!
//! - Same data shape    : `2^22` base-field elements committed.
//! - Same opening shape : 256 evaluations at one common point.
//! - Same scalar count  : 256 EF values revealed by the prover.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p p3-whir --bench fri_vs_whir
//! ```

use std::time::Instant;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, MultilinearPcs, Pcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{
    KoalaBear, Poseidon1KoalaBear, default_koalabear_poseidon1_16, default_koalabear_poseidon1_24,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::proof::PcsProof;
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_whir::sumcheck::{OpeningProtocol, TableShape, TableSpec};
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Shared substrate that does not depend on the hash function.

/// Base field used for the committed message.
type F = KoalaBear;

/// Challenge field used for Fiat-Shamir challenges and out-of-domain samples.
type EF = BinomialExtensionField<F, 4>;

/// DFT backend used by both protocols.
type Dft = Radix2DFTSmallBatch<F>;

// Polynomial worlds — FRI is univariate, WHIR is multilinear.
//
// FRI:
//   - message   : univariate of degree < 2^m
//   - committed : coefficient vector
//   - open at   : z ∈ EF
//
// WHIR:
//   - message   : multilinear in m variables
//   - committed : evaluation table on {0,1}^m
//   - open at   : z ∈ EF^m
//
// Both encode the same claim "f at one random query equals v".
// The §1.1 bridge identifies the two opening shapes:
//
//     univariate at z    ==    multilinear at (z, z^2, z^4, …, z^{2^(m-1)})
//
// Soundness regimes for RS proximity testing:
//
//   - unique-decoding : proven       , bound (1 - ρ) / 2
//   - Johnson         : proven       , bound 1 - √ρ
//   - capacity        : conjectured  , bound 1 - ρ - η
//
// plonky3 FRI exposes only the capacity-regime formula
//
//     soundness_bits = log_blowup * num_queries + pow_bits
//
// so WHIR is also configured at `SecurityAssumption::CapacityBound`.
// A stricter WHIR setting would force more queries than FRI pays
// for, biasing the comparison.

/// Target soundness in bits for both protocols.
///
/// Interpreted under the capacity-regime conjecture (see security framing above).
const SECURITY_LEVEL: usize = 100;

/// Proof-of-work grinding bits, shared by both protocols.
///
/// # Why this value
///
/// - The WHIR side needs every per-round grinding requirement to fit under this budget. `20` is the smallest budget that clears that floor.
const POW_BITS: usize = 20;

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
const FRI_NUM_QUERIES: usize = (SECURITY_LEVEL - FRI_QUERY_POW_BITS).div_ceil(LOG_BLOWUP);

/// Common batching log for both protocols.
///
/// # Role
///
/// Splits the `2^22` committed elements into `2^k` polynomials of `2^(22 - k)` each:
///
/// ```text
///     FRI    2^k univariate columns of degree < 2^(22 - k)
///            opened at one common z ∈ EF
///
///     WHIR   2^k stacked columns of 2^(22 - k) evaluations each,
///            all opened at one common point
/// ```
///
/// # Why `8` (width 256)
///
/// - Wide enough to put plonky3 FRI in its tuned batch-FRI regime.
/// - Per-column size at `m = 22` is still `2^14`, so FRI's folding work is not overhead-bound.
/// - Matches the column-count order of magnitude of real STARK traces.
///
/// # Alternatives
///
/// - `10` (width 1024) — closer to a real Poseidon2 trace width.
/// - `8` is the safer pick at the small end of any future `m` sweep.
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

/// Layout binding mode for the WHIR stacked polynomial.
///
/// The suffix variant binds the trailing variables first, which matches the way the
/// benchmark's single common opening point is shared across all columns.
type WhirLayout = SuffixProver<F, EF>;

/// Concrete WHIR PCS for a given Merkle backend and challenger.
type WhirPcsTy<MT, Ch> = WhirProver<EF, F, Dft, MT, Ch, WhirLayout>;

/// WHIR opening proof for a given Merkle backend.
type WhirProofTy<MT> = PcsProof<F, EF, MT>;

/// WHIR commitment for a given Merkle backend.
type WhirCommitTy<MT> = <MT as Mmcs<F>>::Commitment;

/// Per-round inverse-rate schedule matching the default protocol parameters.
///
/// # Returns
///
/// - One log-2 inverse rate per intermediate WHIR round.
/// - Round zero starts at inverse rate one.
/// - Each round adds the count of variables it folds away, minus one.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    // One entry per intermediate round; the trailing direct-send round has no entry.
    let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_variables);
    let mut rates = Vec::with_capacity(num_rounds);
    // Start at the base rate of the first committed codeword.
    let mut rate = 1;
    for round in 0..num_rounds {
        // Each folded variable beyond the first raises the inverse rate by one.
        rate += folding_factor.at_round(round) - 1;
        rates.push(rate);
    }
    rates
}

/// Carrier for one fully-prepared WHIR setup.
///
/// # Shape
///
/// - One table of `2^log_width` columns, each a multilinear in `num_variables - log_width` variables.
/// - The columns stack into one committed multilinear in `num_variables` variables.
/// - One opening point opens all `2^log_width` columns at once.
/// - Mirrors batched FRI's "open `2^log_width` univariates at one common point".
struct WhirRig<MT: Mmcs<F>, Ch> {
    /// Fully-instantiated WHIR PCS: derived config, FFT engine, and Merkle backend.
    pcs: WhirPcsTy<MT, Ch>,
    /// Stacked committed witness built from the table of columns.
    witness: Witness<F>,
    /// Public opening schedule: one common point opening every column.
    protocol: OpeningProtocol,
    /// Domain separator used for both prover and verifier transcripts.
    domain_separator: DomainSeparator<EF, F>,
    /// Base challenger; both prover and verifier clone from this.
    challenger: Ch,
}

/// Build a fresh WHIR rig matching batched FRI's `(num_variables, log_width)` shape.
///
/// # Arguments
///
/// - `num_variables` — total log-size of the committed multilinear (`m`).
/// - `log_width` — log of the number of columns opened.
///
/// # Layout at the bench's standard `(m, log_width) = (22, 8)`
///
/// Committed:
/// - 256 columns, each a 14-variable multilinear.
/// - Stacked into one 22-variable multilinear, `2^22` evaluations total.
///
/// Opening:
/// - One sampled point opens all 256 columns at once.
/// - Reveals 256 extension-field evaluations at one common point.
///
/// # Equivalence with batched FRI
///
/// `fri-batch` opens `2^log_width` univariates at one common `z ∈ EF`.
///
/// `whir` opens `2^log_width` stacked columns at one common point.
///
/// Both reveal `2^log_width` extension-field evaluations of the same committed data.
fn whir_setup<MT, Ch, const DIGEST_ELEMS: usize>(
    num_variables: usize,
    log_width: usize,
    mmcs: MT,
    base_challenger: Ch,
) -> WhirRig<MT, Ch>
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    assert!(
        log_width <= num_variables,
        "log_width = {log_width} cannot exceed num_variables = {num_variables}"
    );

    // Per-column arity: each column is a multilinear in this many variables.
    let log_height = num_variables - log_width;
    // Column count, equal to batched FRI's matrix width.
    let width = 1 << log_width;
    // Variables folded away per WHIR round.
    let folding = WHIR_FOLDING_FACTOR;
    let folding_factor = FoldingFactor::Constant(folding);

    // WHIR protocol knobs collected into one struct.
    let params = ProtocolParameters {
        security_level: SECURITY_LEVEL,
        pow_bits: POW_BITS,
        round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: LOG_BLOWUP,
    };

    // Per-round protocol layout: query counts, OOD samples, PoW bits per round.
    let config = WhirConfig::<EF, F, Ch>::new(num_variables, params);

    // Per-rig RNG: distinct seed per `(num_variables, log_width)` so two rigs
    // cannot accidentally collide on polynomial samples.
    let mut rng = SmallRng::seed_from_u64(
        BENCH_SEED ^ ((num_variables as u64) << 16) ^ ((log_width as u64) << 8),
    );

    // One table of `width` random columns, each a `log_height`-variable multilinear.
    let columns = (0..width)
        .map(|_| Poly::<F>::rand(&mut rng, log_height))
        .collect();
    let table = Table::new(columns);
    // Stack the columns into the single committed multilinear in `num_variables` variables.
    let witness = WhirLayout::new_witness(vec![table], folding);

    // Opening schedule: a single point that opens every one of the `width` columns.
    // This is the multilinear analogue of batched FRI's single common opening point.
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_height, width),
        vec![(0..width).collect()],
    )])
    .pad_to_min_num_variables(folding);

    // Pre-allocate twiddle factors up to the largest FFT size used by this rig.
    let dft = Dft::new(1 << config.max_fft_size());
    // Bundle config, FFT engine, and Merkle backend into the PCS.
    let pcs = WhirPcsTy::<MT, Ch>::new(config, dft, mmcs);

    // Domain separator: encodes the protocol shape into the transcript so challenges
    // are bound to this exact configuration.
    let mut domain_separator = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<DIGEST_ELEMS>(&mut domain_separator);

    WhirRig {
        pcs,
        witness,
        protocol,
        domain_separator,
        challenger: base_challenger,
    }
}

/// Run one full WHIR proving cycle (commit + open).
///
/// # Returns
///
/// - the Merkle-root commitment;
/// - the opening proof, which carries the claimed evaluations;
/// - the commit-phase wall-clock in milliseconds;
/// - the open-phase wall-clock in milliseconds.
fn whir_prove_full<MT, Ch>(rig: &WhirRig<MT, Ch>) -> (WhirCommitTy<MT>, WhirProofTy<MT>, u128, u128)
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    // Fresh prover challenger seeded with the domain separator.
    let mut prover_challenger = rig.challenger.clone();
    rig.domain_separator
        .observe_domain_separator(&mut prover_challenger);

    // Phase 1: commit (DFT + Merkle + OOD samples).
    let t = Instant::now();
    let (commitment, prover_data) = <WhirPcsTy<MT, Ch> as MultilinearPcs<EF, Ch>>::commit(
        &rig.pcs,
        rig.witness.clone(),
        &mut prover_challenger,
    );
    let commit_ms = t.elapsed().as_millis();

    // Phase 2: open (multi-round sumcheck + STIR queries + PoW).
    let t = Instant::now();
    let proof = <WhirPcsTy<MT, Ch> as MultilinearPcs<EF, Ch>>::open(
        &rig.pcs,
        prover_data,
        rig.protocol.clone(),
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    (commitment, proof, commit_ms, open_ms)
}

/// Run one full WHIR verification cycle and assert it accepts.
fn whir_verify_full<MT, Ch>(
    rig: &WhirRig<MT, Ch>,
    commitment: &WhirCommitTy<MT>,
    proof: &WhirProofTy<MT>,
) -> u128
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    // Each verification call starts from a fresh transcript clone.
    let mut verifier_challenger = rig.challenger.clone();
    rig.domain_separator
        .observe_domain_separator(&mut verifier_challenger);

    let t = Instant::now();
    <WhirPcsTy<MT, Ch> as MultilinearPcs<EF, Ch>>::verify(
        &rig.pcs,
        commitment,
        proof,
        &mut verifier_challenger,
        rig.protocol.clone(),
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
/// # Matrix shape
///
/// - rows    : `2^(log_n - log_width)`
/// - columns : `2^log_width`
/// - total   : `2^log_n` field elements (independent of `log_width`)
///
/// # Workload at the bench's standard `(log_n, log_width) = (22, 8)`
///
/// - 256 columns × 2^14 rows — plonky3 batched-FRI's tuned regime.
/// - All 256 columns opened at one common `z ∈ EF`.
/// - Reveals 256 univariate evaluations.
/// - Matches `whir`'s 256 multilinear evaluations on a 22-var polynomial sharing 14 coordinates.
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
    //   - rows per column : 2^(log_n - log_width)
    //   - columns         : 2^log_width
    //   - LDE rows        : 2^(log_n - log_width + log_blowup)
    //   - total committed : 2^log_n field elements (independent of log_width)
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
    //   - outer : one entry per commitment           (here: 1)
    //   - mid   : one list per matrix in commitment  (here: 1)
    //   - inner : one entry per opening point        (here: [zeta])
    let data_and_points = vec![(&prover_data, vec![vec![zeta]])];

    // Phase 2 — open: FRI commit phase + query phase + proof-of-work grind.
    let t = Instant::now();
    let (openings, proof) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::open(
        &rig.pcs,
        data_and_points,
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    // `openings` is a 4-level nested Vec indexed by `[round][matrix][point][column]`.
    // For our shape (1 commitment, 1 matrix, 1 point, `width` columns) the
    // inner slice carries every committed polynomial's evaluation at `zeta`.
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
    //   - outer : one entry per commitment    (here: 1)
    //   - mid   : one list per matrix         (here: 1)
    //   - inner : (point, per-column evaluations at that point)
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

    /// Number of base-field elements per digest, used to size the domain separator.
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
    use p3_koala_bear::Poseidon2KoalaBear;

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

    /// Number of base-field elements per digest, used to size the domain separator.
    pub const DIGEST_ELEMS: usize = 8;

    /// Build the Merkle backend, the lifted challenge MMCS, and a base challenger.
    pub fn build_kit() -> (Challenger, ValMmcs, ChallengeMmcs) {
        // Unlike Poseidon1 there are no shipped constants, so draw them from a fixed seed.
        // The fixed seed keeps the sampled round constants identical on every run.
        let mut rng = SmallRng::seed_from_u64(BENCH_SEED);
        // Width-16 permutation drives both 2-to-1 compression and the duplex challenger.
        let perm16 = Perm16::new_from_rng_128(&mut rng);
        // Width-24 permutation hashes the Merkle leaves through a padding-free sponge.
        let perm24 = Perm24::new_from_rng_128(&mut rng);
        // Leaf hasher: absorb a row, squeeze an 8-element field digest.
        let merkle_hash = MerkleHash::new(perm24);
        // Internal-node hasher: compress two child digests into one.
        let merkle_compress = MerkleCompress::new(perm16.clone());
        // Base-field Merkle commitment scheme over the message matrix.
        let val_mmcs = ValMmcs::new(merkle_hash, merkle_compress, 0);
        // Extension-field commitment scheme, reusing the base-field one for the FRI rounds.
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        // Duplex sponge challenger seeded with the same width-16 permutation.
        let challenger = Challenger::new(perm16);
        (challenger, val_mmcs, challenge_mmcs)
    }
}

/// Blake3-backed Merkle + byte-serialising challenger.
mod blake3 {
    use p3_blake3::Blake3;
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    use super::*;

    pub const NAME: &str = "blake3";

    /// Raw byte hash: `[u8; 32]` digests.
    pub type ByteHash = Blake3;
    /// Field-element hasher: serialise to bytes, then Blake3.
    pub type FieldHash = SerializingHasher<Blake3>;
    /// 2-to-1 compression over 32-byte digests.
    pub type Compress = CompressionFunctionFromHasher<Blake3, 2, 32>;
    pub type ValMmcs = MerkleTreeMmcs<F, u8, FieldHash, Compress, 2, 32>;
    pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    pub type Challenger = SerializingChallenger32<F, HashChallenger<u8, Blake3, 32>>;

    /// Number of bytes per digest, used to size the domain separator.
    pub const DIGEST_ELEMS: usize = 32;

    /// Build the Merkle backend, the lifted challenge MMCS, and a base challenger.
    pub fn build_kit() -> (Challenger, ValMmcs, ChallengeMmcs) {
        // Blake3 is stateless, so no RNG or round constants are needed.
        // Leaf hasher: serialise a row to bytes, then Blake3 it into a 32-byte digest.
        let field_hash = FieldHash::new(ByteHash {});
        // Internal-node hasher: Blake3-compress two child digests into one.
        let compress = Compress::new(ByteHash {});
        // Base-field Merkle commitment scheme over the message matrix.
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        // Extension-field commitment scheme, reusing the base-field one for the FRI rounds.
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        // Byte-stream challenger: an empty Blake3 sponge feeding the Fiat-Shamir transcript.
        let challenger = Challenger::new(HashChallenger::new(vec![], ByteHash {}));
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

    // Why a macro
    // - Each hash module exposes its own MMCS, challenger, and digest aliases.
    // - The macro forces a fresh monomorphisation at every call site.
    //
    // Two protocols printed per `(hash, m)` cell, both at width 2^LOG_FRI_BATCH_WIDTH.
    //
    // fri-batch:
    //   - 256 univariates of degree < 2^14
    //   - opened at one common z ∈ EF
    //
    // whir:
    //   - 256 columns of 2^14 evaluations, stacked into one 22-variable multilinear
    //   - all 256 columns opened at one common point
    //     (the same `z` lifted via the WHIR §1.1 univariate-multilinear bridge)
    macro_rules! diag_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();

                // Batch FRI (matrix width 2^LOG_FRI_BATCH_WIDTH) — plonky3 FRI's
                // tuned regime, opened at one common point.
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

                // WHIR with 256 stacked columns opened at one common point,
                // matching fri-batch's claim shape.
                let whir_rig = whir_setup::<_, _, { $module::DIGEST_ELEMS }>(
                    m,
                    LOG_FRI_BATCH_WIDTH,
                    val_mmcs,
                    challenger,
                );
                let (whir_commit, whir_proof, whir_commit_ms, whir_open_ms) =
                    whir_prove_full(&whir_rig);
                let whir_verify_us = whir_verify_full(&whir_rig, &whir_commit, &whir_proof);
                let whir_bytes = postcard::to_allocvec(&whir_proof)
                    .expect("postcard WHIR")
                    .len();

                let whir_queries = whir_rig
                    .pcs
                    .config
                    .round_parameters
                    .iter()
                    .map(|r| r.num_queries.to_string())
                    .join(",");

                println!(
                    " {:<10} | {:>2} | whir  | {:>9} | {:>7} | {:>9} | {:>11} | [{}] ({} columns)",
                    $module::NAME,
                    m,
                    whir_commit_ms,
                    whir_open_ms,
                    whir_verify_us,
                    whir_bytes,
                    whir_queries,
                    1 << LOG_FRI_BATCH_WIDTH,
                );

                assert!(
                    whir_rig.pcs.config.check_pow_bits(),
                    "WHIR PoW budget below {SECURITY_LEVEL}-bit target at m = {m}"
                );

                drop(whir_proof);
                drop(whir_rig);
            }
        }};
    }

    diag_block!(poseidon1);
    diag_block!(poseidon2);
    diag_block!(blake3);

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

                // Batch FRI in plonky3's tuned regime: 256 columns of 2^14 rows,
                // opened at one common point.
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

                // WHIR with 256 stacked columns opened at one common point,
                // mirroring fri-batch's claim shape.
                let whir_rig = whir_setup::<_, _, { $module::DIGEST_ELEMS }>(
                    m,
                    LOG_FRI_BATCH_WIDTH,
                    val_mmcs,
                    challenger,
                );
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
    prove_block!(blake3);

    group.finish();
}

/// Bench verification time for both protocols across all hash backends.
fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcs/fri_vs_whir/verify");

    macro_rules! verify_block {
        ($module:ident) => {{
            for &m in &M_VALUES {
                let (challenger, val_mmcs, challenge_mmcs) = $module::build_kit();

                // Pre-prove once for batch FRI so the inner loop only times verification.
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

                // Same for WHIR: prove once, then time the verifier in isolation.
                let whir_rig = whir_setup::<_, _, { $module::DIGEST_ELEMS }>(
                    m,
                    LOG_FRI_BATCH_WIDTH,
                    val_mmcs,
                    challenger,
                );
                let (whir_commit, whir_proof, _, _) = whir_prove_full(&whir_rig);
                let label_whir = format!("whir/{}", $module::NAME);
                group.bench_with_input(BenchmarkId::new(label_whir, m), &m, |b, _| {
                    b.iter(|| {
                        whir_verify_full(&whir_rig, &whir_commit, &whir_proof);
                    });
                });
            }
        }};
    }

    verify_block!(poseidon1);
    verify_block!(poseidon2);
    verify_block!(blake3);

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
