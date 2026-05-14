//! Shared fixtures for the HVZK sumcheck test modules.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_zk_codes::reed_solomon::ReedSolomonZkEncoding;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::sumcheck::layout::{Layout, PrefixProver, Table, TableShape, Witness};
use crate::sumcheck::zk::{ZkPrefixProver, ZkSumcheckData, ZkVerifier};

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
/// Merkle commitment scheme used for the mask oracles.
pub type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
/// Two-adic DFT backing the Reed-Solomon encoding.
pub type MyDft = Radix2DFTSmallBatch<F>;
/// Reed-Solomon zero-knowledge encoding driving the mask code.
pub type MyEnc = ReedSolomonZkEncoding<F, MyDft>;

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
    let mmcs: MyMmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    // Codeword length, rounded up to a power of two for the butterfly schedule:
    //
    //     m = next_pow2( ell_zk + t )
    let m = (ell_zk + T).next_power_of_two();
    let dft = MyDft::default();
    let encoding = MyEnc::new(T, ell_zk, m, dft);

    (perm, mmcs, encoding)
}

/// Builds a prover and matching verifier on the same witness.
///
/// # Returns
///
/// - HVZK prover wrapping a freshly built prefix-binding inner prover.
/// - Verifier registered with the matching table shape.
/// - The polynomial's variable count, useful for downstream shape checks.
pub fn build_prover_verifier(
    evals: Vec<F>,
    folding_factor: usize,
    encoding: MyEnc,
    mmcs: MyMmcs,
) -> (
    ZkPrefixProver<F, EF, MyEnc, MyMmcs>,
    ZkVerifier<F, EF>,
    usize,
) {
    // Table arity:
    //
    //     |evals| = 2^n_vars
    let n_vars = p3_util::log2_strict_usize(evals.len());

    // One-column table built from the witness polynomial.
    let poly = Poly::new(evals);
    let table = Table::new(vec![poly]);

    // Interleave into the stacked layout the inner prover expects.
    let witness: Witness<F> = Witness::new_interleaved(vec![table], folding_factor);

    // Prefix-binding inner prover + HVZK overlay.
    let inner = PrefixProver::<F, EF>::from_witness(witness);
    let prover = ZkPrefixProver::new(inner, encoding, mmcs);

    // Verifier registered with the matching table shape (keeps claim recording in sync).
    let verifier = ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
    (prover, verifier, n_vars)
}

/// End-to-end honest prover -- verifier driver.
///
/// # Phases
///
/// 1. Build matched prover and verifier on a random witness.
/// 2. Record `num_concrete` opening claims, then `num_virtual` evaluation claims, in lockstep on both challengers.
/// 3. Run the prover's HVZK sumcheck.
/// 4. Replay the transcript on the verifier and compare challenges.
///
/// # Claim kinds
///
/// Concrete and virtual claims feed two separate accumulator branches joined by the alpha-power split.
/// Pass at least one of each to exercise both.
///
/// # Returns
///
/// - `Ok(())` when prover and verifier derive the same per-round challenges.
/// - `Err(msg)` otherwise, with context for a proptest failure report.
pub fn run_roundtrip(
    n_vars: usize,
    folding_factor: usize,
    ell_zk: usize,
    num_concrete: usize,
    num_virtual: usize,
    seed: u64,
) -> Result<(), &'static str> {
    // Phase 1: deterministic setup.
    let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

    // Phase 2: witness vector.
    //
    // Distinct seed so draws never collide with later RNG use.
    let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

    // Phase 3: matched prover + verifier on this witness.
    let (mut prover, mut verifier, _n_vars) =
        build_prover_verifier(evals, folding_factor, encoding, mmcs);

    // Phase 4: parallel challengers from the same permutation state.
    let mut prover_challenger = MyChallenger::new(perm.clone());
    let mut verifier_challenger = MyChallenger::new(perm);

    // Phase 5a: concrete opening claims.
    //
    //     prover  : samples opening point, returns evaluations
    //     verifier: mirrors the same draws to stay in lockstep
    for _ in 0..num_concrete {
        let openings = prover.eval(0, &[0], &mut prover_challenger);
        verifier.add_claim(0, &[0], &openings, &mut verifier_challenger);
    }

    // Phase 5b: virtual evaluation claims.
    //
    //     prover  : advances transcript, returns the value
    //     verifier: observes the value to mirror challenger state
    for _ in 0..num_virtual {
        let eval = prover.add_virtual_eval(&mut prover_challenger);
        verifier.add_virtual_eval(eval, &mut verifier_challenger);
    }

    // Phase 6: prover-side sumcheck with grinding enabled.
    //
    //     pow_bits > 0 => exercises the PoW path on both sides
    let pow_bits = 4;
    let mut zk_data = ZkSumcheckData::<F, EF>::default();
    // Distinct seed so mask sampling does not collide with witness draws.
    let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
    let (_residual_prover, prover_randomness, mask_oracles) = prover.into_sumcheck(
        &mut zk_data,
        pow_bits,
        &mut prover_challenger,
        &mut prover_rng,
    );

    // Public mask commits forwarded to the verifier; prover-side data discarded.
    let mask_commits: Vec<_> = mask_oracles.iter().map(|(c, _)| c.clone()).collect();

    // Phase 7: verifier replay.
    //
    // A mismatch in any shape or PoW check returns an error.
    let (verifier_point, _final_target) = verifier
        .into_sumcheck::<MyMmcs, _>(
            &zk_data,
            &mask_commits,
            ell_zk,
            folding_factor,
            pow_bits,
            &mut verifier_challenger,
        )
        .map_err(|_| "verifier rejected honest prover output")?;

    // Phase 8: coupling certificate.
    //
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

/// True when an extension-field element lifts a single base-field element.
///
/// # Why this check exists
///
/// Honest and simulated wire coordinates with index `>= 2` are produced as
///
/// ```text
///     2^{k - j} * mask_coeff
/// ```
///
/// lifted from F into EF, so both views land in the F-subspace.
///
/// Without enforcing this on the simulator, a distinguisher could read off the basis decomposition and separate the two views.
///
/// # Encoding
///
/// ```text
///     x in F  <=>  basis_coeffs(x) = [ x_0, 0, 0, ..., 0 ]
/// ```
pub fn ef_in_f_subspace(x: EF) -> bool {
    // Decompose into the EF basis.
    let coeffs: &[F] = EF::as_basis_coefficients_slice(&x);
    // First slot carries the lifted base value; all higher slots must be zero.
    coeffs[1..].iter().all(|c| *c == F::ZERO)
}
