//! Shared fixtures for the HVZK sumcheck test modules.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_zk_codes::reed_solomon::ReedSolomonZkEncoding;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::layout::{PrefixProver, SuffixProver, Table, TableShape};
use crate::strategy::VariableOrder;
use crate::table::OpeningBatch;
use crate::zk::{ZkLayout, ZkProver, ZkSumcheckData, ZkVerifier};

/// Base field used across the test suite.
pub type F = BabyBear;
/// Extension field used for sumcheck randomness and Lagrange weights.
pub type EF = BinomialExtensionField<F, 4>;
/// Hash permutation used by the Merkle commitment scheme.
pub type Perm = Poseidon2BabyBear<16>;
/// Sponge wrapper around the permutation for leaf hashing.
pub type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
/// Two-to-one compression wrapper for Merkle inner nodes.
pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
/// Fiat-Shamir challenger driven by the same permutation.
pub type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
/// Packed flavour of the base field, used for SIMD Merkle leaves.
pub type PackedF = <F as Field>::Packing;
/// Base-field Merkle commitment scheme backing the mask oracles.
pub type BaseMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
/// Extension-field Merkle commitment scheme used for the mask oracles.
pub type MyMmcs = ExtensionMmcs<F, EF, BaseMmcs>;
/// Two-adic DFT backing the Reed-Solomon encoding.
pub type MyDft = Radix2DFTSmallBatch<EF>;
/// Reed-Solomon zero-knowledge encoding driving the mask code.
pub type MyEnc = ReedSolomonZkEncoding<EF, MyDft>;

/// Randomness symbols appended to each mask before encoding.
///
/// # Why this value
///
/// - Encoding is `t`-query private.
/// - Sumcheck itself opens zero positions, so any `t >= 0` is sound.
/// - `t = 2` keeps the codeword small (cheap setup) and still covers one even + one odd slot for downstream composition tests.
pub const T: usize = 2;

/// Builds the per-test setup triple from a single seed.
///
/// # Arguments
///
/// - `seed`: drives the permutation initialisation.
/// - `ell_zk`: message length of the Reed-Solomon mask code.
///
/// # Returns
///
/// - Permutation seeded deterministically from the input.
/// - Merkle commitment scheme built from that permutation.
/// - Reed-Solomon zero-knowledge encoding sized for the mask.
pub fn make_setup(seed: u64, ell_zk: usize) -> (Perm, MyMmcs, MyEnc) {
    // Deterministic permutation.
    //
    // Same seed => same permutation state on both challengers in a test.
    let mut perm_rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut perm_rng);

    // Leaf hash + inner-node compression wrappers.
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm.clone());
    // Zero salt: deterministic commitments for reproducibility.
    let base_mmcs = BaseMmcs::new(merkle_hash, merkle_compress, 0);
    // Lift the base-field MMCS to commit extension-field mask codewords.
    let mmcs: MyMmcs = ExtensionMmcs::new(base_mmcs);

    // Codeword length, rounded up to a power of two for the butterfly schedule:
    //
    //     m = next_pow2( ell_zk + t )
    let m = (ell_zk + T).next_power_of_two();
    let dft = MyDft::default();
    let encoding = MyEnc::new(T, ell_zk, m, dft);

    (perm, mmcs, encoding)
}

/// Builds an HVZK prover and a matching verifier for the given binding mode.
///
/// The binding mode is selected by the layout type parameter.
/// The witness shape is delivered by the layout's own factory:
///
/// - prefix layout  ⇒  interleaved-stacked witness  (local bits first)
/// - suffix layout  ⇒  contiguous stacked witness   (selector bits first)
///
/// # Returns
///
/// - HVZK prover wrapping a fresh inner prover of the chosen binding mode.
/// - Verifier registered with the matching layout strategy.
/// - Variable count of the polynomial, useful for shape checks.
#[allow(clippy::type_complexity)]
pub fn build_prover_verifier<L>(
    evals: Vec<F>,
    folding_factor: usize,
    encoding: MyEnc,
    mmcs: MyMmcs,
) -> (ZkProver<F, EF, MyEnc, MyMmcs, L>, ZkVerifier<F, EF>, usize)
where
    L: ZkLayout<F, EF>,
{
    // Table arity:
    //
    //     |evals| = 2^n_vars
    let n_vars = p3_util::log2_strict_usize(evals.len());

    // One-column table built from the witness polynomial.
    let poly = Poly::new(evals);
    let table = Table::new(vec![poly]);

    // Witness shape comes from the layout factory; binding mode is encoded
    // in the layout type itself.
    let witness = L::new_witness(vec![table], folding_factor);

    // Layout-binding inner prover + HVZK overlay.
    let inner = L::from_witness(witness);
    let prover = ZkProver::new(inner, encoding, mmcs);

    // Verifier strategy must match the layout binding mode.
    let shapes = [TableShape::new(n_vars, 1)];
    let verifier = match L::strategy().variable_order {
        VariableOrder::Prefix => ZkVerifier::<F, EF>::new_prefix(&shapes),
        VariableOrder::Suffix => ZkVerifier::<F, EF>::new_suffix(&shapes),
    };

    (prover, verifier, n_vars)
}

/// Verifier and proof artefacts produced by an honest prover run.
///
/// Returned by the per-mode prover helper.
/// The verifier replay step is shared across modes and consumes one of these.
pub struct ProverRun {
    /// HVZK verifier with the claim phase already absorbed.
    pub verifier: ZkVerifier<F, EF>,
    /// Verifier-side challenger paired with the prover one.
    pub verifier_challenger: MyChallenger,
    /// Per-round zero-knowledge transcript artefacts.
    pub zk_data: ZkSumcheckData<F, EF>,
    /// Batch mask commitment forwarded to the verifier.
    pub mask_commitment: <MyMmcs as p3_commit::Mmcs<EF>>::Commitment,
    /// Per-round folding randomness emitted by the prover.
    pub prover_randomness: p3_multilinear_util::point::Point<EF>,
    /// Virtual evaluations sampled during the claim phase.
    ///
    /// Used by simulator-coupling tests to rebuild a parallel verifier mirroring the same claim absorption.
    pub virtual_evals: Vec<EF>,
}

/// Runs the prover side of a roundtrip and returns the proof artefacts.
///
/// Branches on the binding direction internally.
/// The post-prover verifier replay is shared across modes and lives in [`replay_verifier`].
#[allow(clippy::too_many_arguments)]
pub fn run_prover(
    binding: VariableOrder,
    n_vars: usize,
    folding_factor: usize,
    ell_zk: usize,
    num_concrete: usize,
    num_virtual: usize,
    pow_bits: usize,
    seed: u64,
) -> ProverRun {
    // Phase 1: deterministic setup.
    let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

    // Phase 2: witness vector.
    //
    // Distinct seed so draws never collide with later RNG use.
    let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

    // Phase 3: parallel challengers from the same permutation state.
    let prover_challenger = MyChallenger::new(perm.clone());
    let verifier_challenger = MyChallenger::new(perm);

    // Phase 4: matched prover + verifier on this witness, dispatching on the
    //          binding direction.
    let zk_data = ZkSumcheckData::<F, EF>::default();
    let prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));

    // The per-mode body is identical once the prover and verifier are typed.
    // Dispatch once on `binding` to pick the layout type; the rest is shared.
    match binding {
        VariableOrder::Prefix => drive_prover_run::<PrefixProver<F, EF>>(
            evals,
            folding_factor,
            encoding,
            mmcs,
            num_concrete,
            num_virtual,
            pow_bits,
            prover_challenger,
            verifier_challenger,
            zk_data,
            prover_rng,
        ),
        VariableOrder::Suffix => drive_prover_run::<SuffixProver<F, EF>>(
            evals,
            folding_factor,
            encoding,
            mmcs,
            num_concrete,
            num_virtual,
            pow_bits,
            prover_challenger,
            verifier_challenger,
            zk_data,
            prover_rng,
        ),
    }
}

/// Builds the prover/verifier pair and drives a full claim-recording roundtrip.
///
/// Generic over the binding-mode layout.
/// The body is identical for both binding modes; only the type parameter changes.
#[allow(clippy::too_many_arguments)]
fn drive_prover_run<L>(
    evals: Vec<F>,
    folding_factor: usize,
    encoding: MyEnc,
    mmcs: MyMmcs,
    num_concrete: usize,
    num_virtual: usize,
    pow_bits: usize,
    mut prover_challenger: MyChallenger,
    mut verifier_challenger: MyChallenger,
    mut zk_data: ZkSumcheckData<F, EF>,
    mut prover_rng: SmallRng,
) -> ProverRun
where
    L: ZkLayout<F, EF>,
{
    let (mut prover, mut verifier, _n_vars) =
        build_prover_verifier::<L>(evals, folding_factor, encoding, mmcs);

    // Concrete opening claims.
    for _ in 0..num_concrete {
        let batch = OpeningBatch::new(vec![0], Vec::new());
        let evals = prover.eval(0, &batch, &mut prover_challenger);
        verifier
            .add_claim(0, &batch, &evals, &mut verifier_challenger)
            .unwrap();
    }

    // Virtual evaluation claims.
    let mut virtual_evals = Vec::with_capacity(num_virtual);
    for _ in 0..num_virtual {
        let eval = prover.add_virtual_eval(&mut prover_challenger);
        verifier.add_virtual_eval(eval, &mut verifier_challenger);
        virtual_evals.push(eval);
    }

    // Prover-side sumcheck; consumes `prover`.
    let prover_handoff = prover.into_sumcheck(
        &mut zk_data,
        pow_bits,
        &mut prover_challenger,
        &mut prover_rng,
    );
    let mask_commitment = prover_handoff.mask_oracle.0.clone();

    ProverRun {
        verifier,
        verifier_challenger,
        zk_data,
        mask_commitment,
        prover_randomness: prover_handoff.randomness,
        virtual_evals,
    }
}

/// Drives the verifier replay against a prover run and returns the per-round folding randomness recovered.
pub fn replay_verifier(
    mut run: ProverRun,
    ell_zk: usize,
    folding_factor: usize,
    pow_bits: usize,
) -> Result<p3_multilinear_util::point::Point<EF>, &'static str> {
    let verifier_handoff = run
        .verifier
        .into_sumcheck::<MyMmcs, _>(
            &run.zk_data,
            &run.mask_commitment,
            ell_zk,
            folding_factor,
            pow_bits,
            &mut run.verifier_challenger,
        )
        .map_err(|_| "verifier rejected honest prover output")?;
    Ok(verifier_handoff.randomness)
}

/// End-to-end honest prover/verifier driver, parameterised by binding direction.
///
/// # Phases
///
/// 1. Build matched prover and verifier on a random witness (per mode).
/// 2. Record concrete opening claims and virtual evaluation claims in lockstep on both challengers.
/// 3. Run the prover's HVZK sumcheck.
/// 4. Replay the transcript on the verifier and compare challenges.
///
/// Concrete and virtual claims feed two separate accumulator branches joined by the alpha-power split.
/// Pass at least one of each to exercise both.
///
/// # Returns
///
/// - `Ok(())` when prover and verifier derive the same per-round challenges.
/// - `Err(msg)` otherwise, with context for a proptest failure report.
pub fn run_roundtrip(
    binding: VariableOrder,
    n_vars: usize,
    folding_factor: usize,
    ell_zk: usize,
    num_concrete: usize,
    num_virtual: usize,
    seed: u64,
) -> Result<(), &'static str> {
    // Grinding enabled to exercise the PoW path on both sides.
    let pow_bits = 4;

    let run = run_prover(
        binding,
        n_vars,
        folding_factor,
        ell_zk,
        num_concrete,
        num_virtual,
        pow_bits,
        seed,
    );
    let prover_randomness = run.prover_randomness.clone();

    let verifier_point = replay_verifier(run, ell_zk, folding_factor, pow_bits)?;

    // Parallel challengers must derive the same per-round challenges:
    //
    //     prover_randomness == verifier_point
    let prover_randomness_vec: Vec<EF> = prover_randomness.iter().copied().collect();
    let verifier_randomness_vec: Vec<EF> = verifier_point.iter().copied().collect();
    if prover_randomness_vec != verifier_randomness_vec {
        return Err("prover/verifier disagreed on sumcheck randomness");
    }
    Ok(())
}
