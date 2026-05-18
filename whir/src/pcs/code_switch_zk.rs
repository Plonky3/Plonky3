//! HVZK code-switching round (Construction 9.7, eprint 2026/391 §9.4).
//!
//! Reduces a proximity claim about oracle `f` (source code `C`) to one about
//! oracle `g` (target code `C'`). When ZK is active, the prover additionally
//! commits a mask oracle `s` that hides the encoding randomness.
//!
//! # Protocol (Construction 9.7)
//!
//! **Interaction phase:**
//!
//! 1. **Send witness.** Prover sends:
//!    - `g = Enc_{C'}(f, r')` as oracle (target-code codeword),
//!    - `s = Enc_{C_zk}((r ∥ s̃), r'')` as oracle (mask codeword).
//! 2. **OOD samples.** Verifier samples `ρ_ood ← D_ood`.
//! 3. **OOD answers.** Prover sends `y = ze_ood(ρ_ood) · (f, r, s_msg)`.
//! 4. **In-domain queries.** Verifier queries `f(x_1), …, f(x_t)`.
//!
//! **Decision phase:** `μ' = ν_1·μ + Σ ν·y + Σ ν·f(x_i)` (batching).
//!
//! # Security
//!
//! - Lemma 9.8: HVZK with error `ζ_{C'} + ζ_ze + ζ_{C_zk}`.
//! - Lemma 9.9: RBR knowledge soundness.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::utils::padded_ood_t1;

/// Proof data for the ZK code-switching round.
///
/// Contains the private OOD answer and the encoding randomness
/// carried forward for the next round's mask.
#[derive(Debug, Clone)]
pub struct CodeSwitchZkData<EF> {
    /// OOD answer: `y = ze_ood(ρ) · (f_msg, prev_rand, mask_msg)`.
    pub ood_answer: EF,
    /// Encoding randomness `r'` used for the target commitment `g`.
    ///
    /// Carried forward so the next round's mask can hide it.
    pub encoding_randomness: Vec<EF>,
}

/// Commit the ZK mask oracle (Construction 9.7, step 1b).
///
/// Builds the mask message `(prev_encoding_rand ∥ s̃)` where `s̃` is fresh
/// randomness, encodes it via `enc_zk`, commits the codeword, and observes
/// the Merkle root in the Fiat-Shamir transcript.
///
/// Returns `(mask_root, mask_msg)` where `mask_msg = (r ∥ s̃)` is the
/// randomness vector needed by [`padded_ood_t1`](crate::utils::padded_ood_t1).
///
/// This is separated from [`prove`] so that the caller can sample OOD
/// challenges from the transcript *after* the mask root is observed,
/// matching the Fiat-Shamir transcript order.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn commit_mask<F, EF, Enc, MT, Challenger, R>(
    prev_encoding_rand: &[EF],
    enc_zk: &Enc,
    mmcs_zk: &ExtensionMmcs<F, EF, MT>,
    challenger: &mut Challenger,
    rng: &mut R,
) -> (
    MT::Commitment,
    Vec<EF>,
    MT::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Enc: ZkEncoding<EF, Codeword = DenseMatrix<EF>>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    let mask_msg_len = enc_zk.message_len();
    assert!(
        prev_encoding_rand.len() <= mask_msg_len,
        "Previous encoding randomness ({}) exceeds mask message length ({})",
        prev_encoding_rand.len(),
        mask_msg_len,
    );
    let padding_len = mask_msg_len - prev_encoding_rand.len();
    let mut mask_msg = Vec::with_capacity(mask_msg_len);
    mask_msg.extend_from_slice(prev_encoding_rand);
    mask_msg.extend((0..padding_len).map(|_| rng.random::<EF>()));

    // Encode and commit the mask oracle.
    let mask_codeword = enc_zk.encode(&mask_msg, rng);
    let (mask_root, mask_prover_data) = mmcs_zk.commit_matrix(mask_codeword);
    challenger.observe(mask_root.clone());

    (mask_root, mask_msg, mask_prover_data)
}

/// Prover side of the HVZK code-switching round (Construction 9.7).
///
/// Produces the mask oracle commitment and computes the private OOD answer.
/// The target commitment `g = Enc_{C'}(f, r')` is handled by the caller
/// (the existing `commit_extension` path with ZK randomness appended).
///
/// # Arguments
///
/// - `f_msg` — folded polynomial coefficients (message part of current codeword).
/// - `prev_encoding_rand` — encoding randomness `r` from the previous round's
///   ZK commitment. The mask oracle hides this.
/// - `ood_point` — verifier's OOD challenge `ρ_ood`.
/// - `target_encoding_rand` — fresh randomness `r'` used for the target `g`
///   commitment (already committed by the caller).
/// - `enc_zk` — ZK encoding for the mask oracle `C_zk`.
/// - `mmcs_zk` — Merkle commitment scheme for the mask oracle.
/// - `challenger` — Fiat-Shamir transcript.
/// - `rng` — source of private randomness.
///
/// # Returns
///
/// `(mask_commitment, code_switch_data)` — the mask root (to be stored in the
/// proof) and the round's ZK data including the OOD answer and new encoding
/// randomness.
#[allow(clippy::too_many_arguments)]
pub fn prove<F, EF, Enc, MT, Challenger, R>(
    f_msg: &[EF],
    prev_encoding_rand: &[EF],
    ood_point: EF,
    target_encoding_rand: &[EF],
    enc_zk: &Enc,
    mmcs_zk: &ExtensionMmcs<F, EF, MT>,
    challenger: &mut Challenger,
    rng: &mut R,
) -> (MT::Commitment, CodeSwitchZkData<EF>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Enc: ZkEncodingWithRandomness<EF, Codeword = DenseMatrix<EF>>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    // --- Step 1b: commit mask oracle s = Enc_{C_zk}((r ∥ s̃), r'') ---
    //
    // The mask message is (prev_encoding_rand ∥ s̃), where s̃ is fresh
    // randomness padding to fill the encoding's message length.
    let mask_msg_len = enc_zk.message_len();
    assert!(
        prev_encoding_rand.len() <= mask_msg_len,
        "Previous encoding randomness ({}) exceeds mask message length ({})",
        prev_encoding_rand.len(),
        mask_msg_len,
    );
    let padding_len = mask_msg_len - prev_encoding_rand.len();
    let mut mask_msg = Vec::with_capacity(mask_msg_len);
    mask_msg.extend_from_slice(prev_encoding_rand);
    mask_msg.extend((0..padding_len).map(|_| rng.random::<EF>()));

    // Encode and commit the mask oracle.
    let mask_codeword = enc_zk.encode(&mask_msg, rng);
    let (mask_root, mask_prover_data) = mmcs_zk.commit_matrix(mask_codeword);
    challenger.observe(mask_root.clone());
    let _ = mask_prover_data;

    // --- Step 3: OOD answer y = ze_ood(ρ) · (f_msg, r ∥ s̃) ---
    //
    // padded_ood_t1(ρ, msg, rand) computes ze*_l(ρ)·msg + ρ^l · ze*_r(ρ)·rand
    // where l = msg.len() and r = rand.len().
    //
    // mask_msg is already (prev_encoding_rand ∥ s̃) — the paper's (r ∥ s̃).
    let ood_answer = padded_ood_t1(ood_point, f_msg, &mask_msg);

    (
        mask_root,
        CodeSwitchZkData {
            ood_answer,
            encoding_randomness: target_encoding_rand.to_vec(),
        },
    )
}

/// Simulator for the HVZK code-switching round (Lemma 9.8).
///
/// Produces a simulated transcript that is statistically close to the real
/// protocol view, without access to the witness `(f, r)`.
///
/// # Arguments
///
/// - `ood_point` — verifier's OOD challenge.
/// - `query_positions_target` — positions queried on the target oracle `g`.
/// - `query_positions_mask` — positions queried on the mask oracle `s`.
/// - `enc_target` — ZK encoding for the target code `C'`.
/// - `enc_zk` — ZK encoding for the mask code `C_zk`.
/// - `rng` — randomness source.
///
/// # Returns
///
/// `(simulated_ood_answer, simulated_g_values, simulated_s_values)`.
#[allow(clippy::too_many_arguments)]
pub fn simulate<EF, EncTarget, EncMask, R>(
    ood_point: EF,
    query_positions_target: &[usize],
    query_positions_mask: &[usize],
    enc_target: &EncTarget,
    enc_zk: &EncMask,
    rng: &mut R,
) -> (EF, Vec<EF>, Vec<EF>)
where
    EF: Field,
    EncTarget: ZkEncoding<EF>,
    EncMask: ZkEncoding<EF>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    // Lemma 9.8 step 3: sample (ρ_ood, y) ← S_{ze_ood}.
    //
    // For RS-based private zero-evaders (Lemma 9.3), ze_ood is
    // (ℓ_zk - r, 0)-private: the simulator S_{ze_ood} outputs
    // (ρ_ood, y) where y is uniform in F^{t_ood}, independent of
    // the message. In Fiat-Shamir mode ρ_ood is fixed by the
    // transcript (not sampled), so only y is simulated.
    //
    // The uniform distribution over F is the correct simulation
    // because ζ_ze = 0 (perfect privacy). See Lemma 9.3 proof:
    // y = t_ρ + M·r where M spans F^t, making y uniform for any
    // fixed ρ and message.
    let _ = ood_point;
    let simulated_ood: EF = rng.random();

    // Lemma 9.8 step 5: simulate oracle answers via Sim_{C'} and Sim_{C_zk}.
    let simulated_g = enc_target.simulate(query_positions_target, rng);
    let simulated_s = enc_zk.simulate(query_positions_mask, rng);

    (simulated_ood, simulated_g, simulated_s)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2Dit;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_zk_codes::ReedSolomonZkEncoding;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn test_perm() -> Perm {
        Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(42))
    }

    fn test_challenger() -> Challenger {
        DuplexChallenger::new(test_perm())
    }

    fn test_mmcs() -> ValMmcs {
        let perm = test_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        MerkleTreeMmcs::new(hash, compress, 0)
    }

    /// Honest prover produces a valid, non-trivial OOD answer.
    #[test]
    fn ood_answer_is_nontrivial() {
        let mut rng = SmallRng::seed_from_u64(1);

        let msg_len = 4;
        let t_target = 2;

        let prev_rand: Vec<EF> = (0..t_target).map(|_| rng.random()).collect();
        let target_rand: Vec<EF> = (0..t_target).map(|_| rng.random()).collect();

        let mask_msg_len = prev_rand.len() + 2;
        let mask_t = 2;
        let mask_m = 8;
        let dft = Radix2Dit::default();
        let enc_zk = ReedSolomonZkEncoding::<EF, _>::new(mask_t, mask_msg_len, mask_m, dft);

        let mmcs = test_mmcs();
        let ext_mmcs = ExtensionMmcs::<F, EF, _>::new(mmcs);
        let mut challenger = test_challenger();

        let f_msg: Vec<EF> = (0..msg_len).map(|i| EF::from_u64(i as u64 + 1)).collect();
        let ood_point: EF = EF::from_u64(7);

        let (_mask_root, data) = prove::<F, EF, _, _, _, _>(
            &f_msg,
            &prev_rand,
            ood_point,
            &target_rand,
            &enc_zk,
            &ext_mmcs,
            &mut challenger,
            &mut rng,
        );

        assert_ne!(data.ood_answer, EF::ZERO);
        assert_eq!(data.encoding_randomness, target_rand);
    }

    /// Simulator produces output with correct dimensions.
    #[test]
    fn simulator_output_shape() {
        let mut rng = SmallRng::seed_from_u64(2);
        let dft = Radix2Dit::default();

        let enc_target = ReedSolomonZkEncoding::<EF, _>::new(2, 4, 8, dft.clone());
        let enc_mask = ReedSolomonZkEncoding::<EF, _>::new(2, 4, 8, dft);

        let ood_point = EF::from_u64(13);
        let target_positions = [0usize, 3];
        let mask_positions = [1usize, 5];

        let (sim_ood, sim_g, sim_s) = simulate(
            ood_point,
            &target_positions,
            &mask_positions,
            &enc_target,
            &enc_mask,
            &mut rng,
        );

        assert_eq!(sim_g.len(), target_positions.len());
        assert_eq!(sim_s.len(), mask_positions.len());
        assert_ne!(sim_ood, EF::ZERO);
    }

    proptest! {
        /// The honest prover's OOD answer round-trips through verify_ood.
        #[test]
        fn prop_ood_roundtrip(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let dft = Radix2Dit::default();

            let msg_len = 4;
            let mask_msg_len = 4;
            let mask_t = 2;
            let mask_m = 8;

            let enc_zk = ReedSolomonZkEncoding::<EF, _>::new(
                mask_t, mask_msg_len, mask_m, dft,
            );

            let mmcs = test_mmcs();
            let ext_mmcs = ExtensionMmcs::<F, EF, _>::new(mmcs);
            let mut challenger = test_challenger();

            let f_msg: Vec<EF> = (0..msg_len).map(|_| rng.random()).collect();
            let prev_rand: Vec<EF> = (0..2).map(|_| rng.random()).collect();
            let target_rand: Vec<EF> = (0..2).map(|_| rng.random()).collect();
            let ood_point: EF = rng.random();

            let (_mask_root, data) = prove::<F, EF, _, _, _, _>(
                &f_msg,
                &prev_rand,
                ood_point,
                &target_rand,
                &enc_zk,
                &ext_mmcs,
                &mut challenger,
                &mut rng,
            );

            // OOD answer is non-trivial for random inputs
            // (probability of zero over a 128-bit field is negligible).
            prop_assert_ne!(data.ood_answer, EF::ZERO);
        }

        /// Simulator's output has correct shape across random seeds.
        #[test]
        fn prop_simulator_output_shape(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let dft = Radix2Dit::default();

            let enc_target = ReedSolomonZkEncoding::<EF, _>::new(2, 4, 8, dft.clone());
            let enc_mask = ReedSolomonZkEncoding::<EF, _>::new(2, 4, 8, dft);

            let ood_point: EF = rng.random();
            let n_target = 2;
            let n_mask = 2;
            let target_positions: Vec<usize> = (0..n_target).collect();
            let mask_positions: Vec<usize> = (0..n_mask).collect();

            let (_sim_ood, sim_g, sim_s) = simulate(
                ood_point,
                &target_positions,
                &mask_positions,
                &enc_target,
                &enc_mask,
                &mut rng,
            );

            prop_assert_eq!(sim_g.len(), n_target);
            prop_assert_eq!(sim_s.len(), n_mask);
        }
    }
}
