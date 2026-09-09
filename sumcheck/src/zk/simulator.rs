//! Witness-free simulators for the HVZK sumcheck.
//!
//! One per prelude, over the single description every party drives.
//!
//! ```text
//!     recorded claims  ->  mu is derived from the claims a verifier recorded
//!     inherited claim  ->  mu arrives as public input and is bound first
//! ```
//!
//! Everything after the prelude is prelude-agnostic, so one body plays it for both.
//!
//! # Witness-freeness
//!
//! Definition 5.8 of eprint 2026/391 makes the sumcheck target public input.
//!
//! The inherited entry point therefore takes no verifier at all.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, HornerIter};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncodingWithRandomness;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::data::ZkSumcheckData;
use super::prover::common::{mask_endpoints, sample_masks};
use super::transcript::{ZkProverTranscript, ZkSumcheckShape};
use super::verifier::ZkVerifier;

/// Simulate the masking prelude and the round chain of one batch.
///
/// The transcript arrives with its prelude already played.
///
/// The two entry points reach the target differently, so only what follows is shared.
///
/// # Arguments
///
/// - `transcript`: driver positioned just after the prelude, and the source of the shape.
/// - `claimed_sum`: the scalar the batch runs against.
/// - `encoding`: mask code the masks are drawn from and encoded under.
/// - `mmcs`: commitment scheme carrying the interleaved mask oracle.
/// - `rng`: source of the mask messages and of every wire coordinate.
///
/// # Returns
///
/// - Simulated proof record.
/// - The batch mask commitment.
/// - Per-round challenges.
fn simulate_claim<F, EF, Enc, M, Challenger, R>(
    transcript: &mut ZkProverTranscript<'_, Challenger, F, EF>,
    claimed_sum: EF,
    encoding: &Enc,
    mmcs: &M,
    rng: &mut R,
) -> (ZkSumcheckData<F, EF>, M::Commitment, Point<EF>)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Enc: ZkEncodingWithRandomness<EF>,
    Enc::Codeword: Matrix<EF>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    // The shape the driver was seeded with, so the wire width cannot drift from the description.
    let shape = transcript.shape();
    let k = shape.num_rounds;

    // Sample, encode and commit the masks (Construction 6.3 step 1 replay).
    //
    // Uniform messages give uniform codewords, so the codeword distribution is exactly the prover's.
    //
    // The commit is a root over that whole codeword, which fixes the masks information-theoretically.
    // Indistinguishability of the commits is therefore computational, as the crate's zk module docs record.
    //
    // The prover's own helper draws them, so the two draw sequences cannot drift.
    let (masks, _mask_randomness, (mask_commitment, _prover_data)) =
        sample_masks::<EF, _, _, _>(k, encoding, mmcs, rng);

    // The endpoint sum via the closed form (Construction 6.3 step 2 replay).
    //
    //     mu_tilde = 2^{k-1} * sum_l ( s_l(0) + s_l(1) )
    //
    // Byte-equivalent to the honest prover under matched RNG seeds.
    let (mu_tilde, _endpoints) = mask_endpoints::<EF>(&masks, k);

    // Bind the oracle and mu_tilde, then draw the combining challenge.
    let eps: EF = transcript.masks(mask_commitment.clone(), mu_tilde);

    // Per-round wire sampling, simulating Construction 6.3 step 4.
    //
    // Every coordinate is drawn uniformly over the extension field.
    //
    // Wire shape (linear coefficient c_1 dropped):
    //
    //     wire_size = max(ell_zk, 3) - 1
    //
    //     wire = [ c_0, c_2, c_3, ..., c_d ]
    let wire_size = shape.wire_len();

    // Output container.
    //
    // The metadata fields are populated up front.
    let mut zk_data = ZkSumcheckData::<F, EF> {
        mu_tilde,
        round_coefficients: Vec::with_capacity(k),
        pow_witnesses: Vec::with_capacity(if shape.pow_bits > 0 { k } else { 0 }),
    };

    // Per-round challenges and running target mirroring the verifier.
    let mut randomness: Vec<EF> = Vec::with_capacity(k);
    let mut target: EF = eps * claimed_sum + mu_tilde;

    for _ in 0..k {
        // Every wire coordinate is uniform over the full extension field (Lemma 6.4 with `F := EF`).
        let wire: Vec<EF> = (0..wire_size).map(|_| rng.random::<EF>()).collect();

        // One call binds the wire, grinds when enabled, and draws the challenge.
        let (gamma_j, witness) = transcript.round(&wire);
        zk_data.pow_witnesses.extend(witness);

        // Reconstruct c_1:
        //
        //     2 * c_0 + c_1 + sum_{i>=2} c_i = target
        //     => c_1 = target - 2 * c_0 - sum_{i>=2} c_i
        let c0 = wire[0];
        let high_sum: EF = wire[1..].iter().copied().sum();
        let c1 = target - c0.double() - high_sum;

        // Horner-evaluate h_j at gamma_j to derive the next target:
        //
        //     h_j(gamma) = c_0 + gamma * (c_1 + gamma * (c_2 + ... + gamma * c_d))
        target = core::iter::once(c0)
            .chain(core::iter::once(c1))
            .chain(wire[1..].iter().copied())
            .horner(gamma_j);

        // Record wire and challenge.
        zk_data.round_coefficients.push(wire);
        randomness.push(gamma_j);
    }

    (zk_data, mask_commitment, Point::new(randomness))
}

/// Witness-free HVZK simulator over the claims a verifier recorded (Lemma 6.4).
///
/// Produces a transcript indistinguishable from the honest prover's view by reading only the verifier's recorded claims; never the witness itself.
///
/// # Method
///
/// - Sample alpha and derive `mu` from the verifier's claim batching.
/// - Sample and commit fresh masks just like the prover.
/// - Sample each wire coordinate uniformly over `EF` (the honest joint distribution).
/// - Reconstruct the dropped `c_1` from the affine identity, so every simulated wire verifies by construction.
///
/// # Indistinguishability
///
/// The transcript is played through the prover's own description.
///
/// It runs over the prover's own mask helpers, not a copy of them.
///
/// ```text
///     shape    ->  the batching description, exactly as the layout prover builds it
///     prelude  ->  the same drawn batching challenge
///     masks    ->  sample_masks, then mask_endpoints
///     rounds   ->  the same wire, grinding and challenge steps
/// ```
///
/// A step this simulator plays out of order is a pattern failure, not a silent loss of hiding.
///
/// # Scope
///
/// This simulates one Construction 6.3 sumcheck batch, not the composed
/// HVZK-WHIR protocol. It discards the mask oracle's prover data and does not
/// return the mask messages or encoding randomness needed for later
/// code-switch and base-case openings.
///
/// # Wire distribution
///
/// The construction is instantiated over `EF`, so every wire coordinate is
/// uniform over the full extension field (Lemma 6.4 with `F := EF`). Each sent
/// wire `[c_0, c_2, c_3, ..., c_d]` (`d = max(ell_zk - 1, 2)`) is drawn
/// uniformly from `EF`; the dropped linear coefficient `c_1` is recovered from
/// the affine identity.
///
/// # Returns
///
/// - Simulated transcript.
/// - The batch mask commitment.
/// - Per-round challenges.
///
/// # Panics
///
/// When the configuration cannot describe a masked batch, exactly as the prover panics.
pub fn simulate_classic_unpacked<F, EF, Enc, M, Challenger, R>(
    challenger: &mut Challenger,
    verifier: &ZkVerifier<F, EF>,
    folding_factor: usize,
    pow_bits: usize,
    encoding: &Enc,
    mmcs: &M,
    rng: &mut R,
) -> (ZkSumcheckData<F, EF>, M::Commitment, Point<EF>)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Enc: ZkEncodingWithRandomness<EF>,
    Enc::Codeword: Matrix<EF>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    // The same description the layout prover builds from the same three numbers.
    let shape = ZkSumcheckShape::new_batching(folding_factor, encoding.message_len(), pow_bits);
    shape
        .validate::<F>()
        .expect("a simulator's own configuration must describe a masked batch");

    let mut transcript = ZkProverTranscript::<Challenger, F, EF>::new(challenger, shape);

    // Draw alpha and derive mu (replays the Construction 6.3 prelude of the layout path).
    //
    // Honest prover's transcript order after the claim phase:
    //
    //     alpha  ->  mask commit  ->  mu_tilde  ->  eps  ->  wires
    //
    // `verifier.sum(alpha)` reads only the recorded claims, so Lemma 6.4's witness-freeness is preserved.
    let alpha: EF = transcript.batching_challenge();
    let mu = verifier.sum(alpha);

    let simulated = simulate_claim(&mut transcript, mu, encoding, mmcs, rng);

    // Every described step has been played, so the sponge goes back to the caller.
    transcript.finish();

    simulated
}

/// Witness-free HVZK simulator over a claim the batch inherits (Lemma 6.4).
///
/// This is the prelude the residual prover plays, and so the one a WHIR round ships.
///
/// # Public input
///
/// The target is a parameter, matching Definition 5.8 of eprint 2026/391.
///
/// ```text
///     paper       ->  mu = <f, sl(st)> + sum_i <xi_i, sl_i(st_i)>
///     this crate  ->  claimed_sum + aux_claim, the scalar the prover binds
/// ```
///
/// Nothing else about the instance is read, so this entry point takes no verifier.
///
/// # Method
///
/// - Bind the inherited claim, exactly where the prover binds it.
/// - Sample and commit fresh masks just like the prover.
/// - Sample each wire coordinate uniformly over `EF` (the honest joint distribution).
/// - Reconstruct the dropped `c_1` from the affine identity, so every simulated wire verifies by construction.
///
/// # Why the auxiliary constant needs no separate treatment
///
/// It is constant in `X` and independent of the masks.
///
/// ```text
///     aux * 2^{-j}  ->  summed into the constant slot before masking
///     rank of the linear map on the witness  ->  unchanged
/// ```
///
/// So it lands in the affine offset the target already carries.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, driven to the pre-batch state.
/// - `claimed_sum`: the scalar the batch runs against, matching what the prover binds.
/// - `folding_factor`: number of rounds the batch runs.
/// - `pow_bits`: grinding difficulty per round, or zero to omit grinding.
/// - `encoding`: mask code the masks are drawn from and encoded under.
/// - `mmcs`: commitment scheme carrying the interleaved mask oracle.
/// - `rng`: source of the mask messages and of every wire coordinate.
///
/// # Scope
///
/// Same as the recorded-claims entry point: one Construction 6.3 batch, not the composed HVZK-WHIR protocol.
///
/// The mask oracle's prover data is discarded, so the mask messages and encoding randomness are not returned.
///
/// # Returns
///
/// - Simulated transcript.
/// - The batch mask commitment.
/// - Per-round challenges.
///
/// # Panics
///
/// When the configuration cannot describe a masked batch, exactly as the prover panics.
pub fn simulate_classic_unpacked_claim<F, EF, Enc, M, Challenger, R>(
    challenger: &mut Challenger,
    claimed_sum: EF,
    folding_factor: usize,
    pow_bits: usize,
    encoding: &Enc,
    mmcs: &M,
    rng: &mut R,
) -> (ZkSumcheckData<F, EF>, M::Commitment, Point<EF>)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Enc: ZkEncodingWithRandomness<EF>,
    Enc::Codeword: Matrix<EF>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    // The same description the residual prover builds from the same three numbers.
    let shape = ZkSumcheckShape::new_inherited(folding_factor, encoding.message_len(), pow_bits);
    shape
        .validate::<F>()
        .expect("a simulator's own configuration must describe a masked batch");

    let mut transcript = ZkProverTranscript::<Challenger, F, EF>::new(challenger, shape);

    // The claim opens the description, ahead of the masking prelude.
    //
    //     claim  ->  mask commit  ->  mu_tilde  ->  eps  ->  wires
    transcript.bind_claim(claimed_sum);

    let simulated = simulate_claim(&mut transcript, claimed_sum, encoding, mmcs, rng);

    // Every described step has been played, so the sponge goes back to the caller.
    transcript.finish();

    simulated
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing, dot_product};
    use p3_multilinear_util::poly::Poly;
    use p3_zk_codes::ZkEncoding;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::layout::TableShape;
    use crate::product_polynomial::ProductPolynomial;
    use crate::strategy::{SumcheckProver, VariableOrder};
    use crate::zk::ZkVerifier;
    use crate::zk::test_helpers::{EF, F, MyChallenger, MyMmcs, make_setup, run_prover};

    /// True when an extension element has a non-zero coordinate above the base slot.
    ///
    /// Every honest and simulated wire coordinate is uniform over `EF` (the
    /// construction is instantiated over `EF`), so each coordinate escapes the
    /// base-field subspace except with probability `|F|^{-(D-1)}`.
    fn escapes_f_subspace(x: EF) -> bool {
        let coeffs: &[F] = EF::as_basis_coefficients_slice(&x);
        coeffs[1..].iter().any(|c| *c != F::ZERO)
    }

    /// Lemma 6.4 acceptance and mask-prelude coupling driver for
    /// Reed-Solomon mask encoding.
    ///
    /// # Invariants per `(binding, n_vars, folding, ell_zk, num_eqs)` case
    ///
    /// 1. Verifier accepts both transcripts (soundness floor).
    /// 2. `mu_tilde` and mask commits match bit-for-bit under matched seeds (deterministic equality, not a distributional test).
    /// 3. Wire coordinates with index `>= 2` escape the base-field subspace on both sides (the masks are extension-valued, so the witness leak of an `F`-valued mask is absent).
    /// 4. The mask encoding's own simulator returns the correct shape (RS simulator error = 0).
    ///
    /// The binding parameter only affects the real run; the simulator output depends only on the wire schema, which both binding modes share.
    ///
    /// Check 3 reads the band above the two plain-bearing slots, which the wire width decides:
    ///
    /// ```text
    ///     ell_zk = 3  ->  wire_len = 2, so the band is empty and check 3 reads nothing
    ///     ell_zk > 3  ->  wire_len = ell_zk - 1, so the band is ell_zk - 3 wide
    /// ```
    ///
    /// The shortest legal mask is therefore a case check 3 cannot fail on.
    /// The proptests below draw above it, and a test of its own covers the floor.
    fn run_acceptance_and_mask_prelude_coupling(
        binding: VariableOrder,
        n_vars: usize,
        folding_factor: usize,
        ell_zk: usize,
        num_eqs: usize,
        seed: u64,
    ) -> Result<(), &'static str> {
        // Real run via the binding-parameterised helper.
        //
        // Internally, `run_prover` reuses the same `seed.wrapping_add(2)`
        // RNG seed we re-create below for the simulator, which is what
        // makes the mu_tilde / mask-commits coupling certificate exact.
        let pow_bits = 0;
        let mut real_run = run_prover(
            binding,
            n_vars,
            folding_factor,
            ell_zk,
            0,
            num_eqs,
            pow_bits,
            seed,
        );

        // Snapshot virtual evals so the simulator-side verifier can mirror
        // the same claim phase before being handed to the simulator.
        let virtual_evals = real_run.virtual_evals.clone();
        let zk_data_real = real_run.zk_data.clone();
        let mask_commitment_real = real_run.mask_commitment.clone();

        // Honest verifier replay.
        let _ = real_run
            .verifier
            .into_sumcheck::<MyMmcs, _>(
                &zk_data_real,
                &mask_commitment_real,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut real_run.verifier_challenger,
            )
            .map_err(|_| "real prover transcript rejected by verifier")?;

        // === Simulator run (matched mask-RNG seed) ===
        //
        // Re-derive the setup from the same seed so both runs reach the
        // same MMCS state and the matched-RNG coupling is meaningful.
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

        // The simulator is binding-mode-agnostic, so the verifier we hand
        // it carries the strategy of the real prover for symmetric
        // selector lifting.
        let mut verifier_sim = match binding {
            VariableOrder::Prefix => ZkVerifier::<F, EF>::new_prefix(&[TableShape::new(n_vars, 1)]),
            VariableOrder::Suffix => ZkVerifier::<F, EF>::new_suffix(&[TableShape::new(n_vars, 1)]),
        };
        let mut sim_ch = MyChallenger::new(perm);
        for &eval in &virtual_evals {
            verifier_sim.add_virtual_eval(eval, &mut sim_ch);
        }
        let mut verifier_sim_ch = sim_ch.clone();

        // Matched seed with the real prover RNG; needed by the coupling certificate below.
        let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (zk_data_sim, mask_commitment_sim, _gammas_sim) =
            simulate_classic_unpacked::<F, EF, _, _, _, _>(
                &mut sim_ch,
                &verifier_sim,
                folding_factor,
                pow_bits,
                &encoding,
                &mmcs,
                &mut sim_rng,
            );

        // Verifier replay against the simulated proof.
        let _ = verifier_sim
            .into_sumcheck::<MyMmcs, _>(
                &zk_data_sim,
                &mask_commitment_sim,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut verifier_sim_ch,
            )
            .map_err(|_| "simulator transcript rejected by verifier")?;

        // === Coupling certificate ===

        if zk_data_real.mu_tilde != zk_data_sim.mu_tilde {
            return Err("matched-RNG coupling: mu_tilde differs");
        }
        if mask_commitment_real != mask_commitment_sim {
            return Err("matched-RNG coupling: mask commitment differs");
        }

        // === Extension-valued wires on both sides ===
        //
        // The masks are extension-valued, so every wire coordinate is uniform
        // over EF. We check the index-`>= 2` coordinates (mask-only, no plain
        // piece) escape the base-field subspace on both the real and simulated
        // sides; an `F`-valued mask would pin them to the base field and
        // reintroduce the witness leak.
        for wire in &zk_data_real.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !escapes_f_subspace(c) {
                    return Err("real-prover wire[i >= 2] collapsed into the F-subspace");
                }
            }
        }
        for wire in &zk_data_sim.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !escapes_f_subspace(c) {
                    return Err("simulator wire[i >= 2] collapsed into the F-subspace");
                }
            }
        }

        // === Mask oracle queries via ZkEncoding::simulate (Lemma 6.4 step 5) ===
        //
        // One distinct query set per mask, sized within the encoding's randomness budget so the RS simulator does not panic on too-many queries.
        let t_zk = encoding.randomness_len();
        let m = encoding.codeword_len();
        let mut query_rng = SmallRng::seed_from_u64(seed.wrapping_add(5));
        let mut sim_ans_rng = SmallRng::seed_from_u64(seed.wrapping_add(6));
        for _ in 0..folding_factor {
            let q_size = query_rng.random_range(1..=t_zk);
            let mut positions: Vec<usize> = Vec::with_capacity(q_size);
            while positions.len() < q_size {
                let p = query_rng.random_range(0..m);
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
            let sim_answers: Vec<EF> = encoding.simulate(&positions, &mut sim_ans_rng);
            if sim_answers.len() != positions.len() {
                return Err("ZkEncoding::simulate returned wrong number of answers");
            }
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_simulator_accepts_and_couples_mask_prelude_rs_prefix(
            n_vars in 3usize..=8,
            ell_zk in 4usize..=5,
            num_eqs in 1usize..=3,
            seed in 0u64..1024,
        ) {
            // Invariant: both transcripts accept and their mask preludes
            // couple under matched RNG seeds across the parameter cube for
            // RS mask encoding on the prefix path.
            //
            // Per-case invariants pinned by
            // run_acceptance_and_mask_prelude_coupling (see docstring).
            //
            // Why ell_zk >= 4: lengths 2 and 3 both give a 2-coordinate wire, on which the driver's check 3 reads nothing.
            // The shortest legal mask is covered by a test of its own instead.

            // Compression step requires the folded polynomial to retain at
            // least one full packed lane. Packing width depends on the ISA.
            let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));

            prop_assert!(
                run_acceptance_and_mask_prelude_coupling(
                    VariableOrder::Prefix,
                    n_vars,
                    folding_factor,
                    ell_zk,
                    num_eqs,
                    seed,
                ).is_ok()
            );
        }

        #[test]
        fn prop_simulator_accepts_and_couples_mask_prelude_rs_suffix(
            n_vars in 3usize..=8,
            ell_zk in 4usize..=5,
            num_eqs in 1usize..=3,
            seed in 0u64..1024,
        ) {
            // Same invariant on the suffix path. Suffix mode never packs
            // the residual factor, so the parameter window is wider:
            // folding can go up to `n_vars - 1` instead of `n_vars - k_pack`.
            //
            // The mask length starts at 4 for the same reason as the prefix draw above.
            let folding_factor = 1 + (seed as usize % (n_vars - 1).max(1));

            prop_assert!(
                run_acceptance_and_mask_prelude_coupling(
                    VariableOrder::Suffix,
                    n_vars,
                    folding_factor,
                    ell_zk,
                    num_eqs,
                    seed,
                ).is_ok()
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_simulator_invariants(
            n_vars in 3usize..=6,
            ell_zk in 4usize..=6,
            num_eqs in 1usize..=2,
            seed in 0u64..256,
        ) {
            // Invariants asserted on every (n_vars, folding, ell_zk, num_eqs) draw:
            //
            //   1. Shape      - output sizes match `folding_factor`, `ell_zk`, and `pow_bits = 0`.
            //   2. EF wires   - every wire coordinate with index >= 2 escapes the base-field subspace.
            //   3. Acceptance - a fresh verifier replay accepts the simulated transcript (Lemma 6.4).
            //
            // The construction is instantiated over EF, so the wire (linear
            // coefficient c_1 dropped) has all coordinates uniform over EF:
            //
            //     wire[0] = c_0, wire[1] = c_2, wire[2] = c_3, ...
            //
            // Why ell_zk >= 4: lengths 2 and 3 produce a 2-coordinate wire and wire[2..] is empty, so check 2 has no work to do.
            //
            // Check 2 is the regression guard for the witness leak: an F-valued
            // mask would pin the mask-only coordinates (index >= 2) to the base
            // field, and the plain-bearing coordinates wire[0], wire[1] to a
            // base-field coset, which a distinguisher separates (paper §6.1).
            let folding_factor = 1 + (seed as usize % n_vars);

            let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

            let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
            let mut sim_challenger = MyChallenger::new(perm);

            // Phase: claim absorption.
            //
            // - Drives the simulator-side challenger to the same post-claim state a real verifier would reach.
            // - Strategy: prefix (arbitrary — simulator output depends only on the wire schema).
            //
            // Why one binding is enough here:
            //
            //     coverage source                 | what it pins
            //     --------------------------------+----------------------------------
            //     acceptance/coupling tests       | prefix + suffix coverage
            //     this test (prefix only)         | shape + stratification invariants
            let mut verifier = ZkVerifier::<F, EF>::new_prefix(&[TableShape::new(n_vars, 1)]);
            for _ in 0..num_eqs {
                let eval: EF = data_rng.random();
                verifier.add_virtual_eval(eval, &mut sim_challenger);
            }

            // Snapshot the post-claim challenger state.
            // The simulator advances `sim_challenger`; the verifier replay needs an independent copy from the same state.
            let mut verifier_replay_ch = sim_challenger.clone();

            // Run the simulator under matched RNGs.
            let pow_bits = 0;
            let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
            let (sim_zk_data, mask_commitment, gammas) =
                simulate_classic_unpacked::<F, EF, _, _, _, _>(
                    &mut sim_challenger,
                    &verifier,
                    folding_factor,
                    pow_bits,
                    &encoding,
                    &mmcs,
                    &mut sim_rng,
                );

            // Check 1: shape invariants.
            //
            //     wire_size = max(ell_zk, 3) - 1
            let expected_wire_size = ell_zk.max(3) - 1;
            prop_assert_eq!(
                sim_zk_data.round_coefficients.len(),
                folding_factor,
                "one wire per sumcheck round",
            );
            for (round_idx, wire) in sim_zk_data.round_coefficients.iter().enumerate() {
                prop_assert_eq!(
                    wire.len(),
                    expected_wire_size,
                    "wire length mismatch in round {}",
                    round_idx,
                );
            }
            prop_assert!(
                sim_zk_data.pow_witnesses.is_empty(),
                "pow_witnesses must be empty when pow_bits == 0",
            );
            prop_assert_eq!(
                gammas.as_slice().len(),
                folding_factor,
                "one challenge per round",
            );

            // Check 2: extension-valued wires (witness-leak regression guard).
            //
            // Each mask-only coordinate (index >= 2) is uniform over EF, so it
            // escapes the base-field subspace except with probability
            // `|F|^{-(D-1)}`. An F-valued mask would pin these to the base
            // field and leak the witness; this check rejects that regression.
            for (round_idx, wire) in sim_zk_data.round_coefficients.iter().enumerate() {
                for (pos, &coeff) in wire.iter().enumerate().skip(2) {
                    prop_assert!(
                        escapes_f_subspace(coeff),
                        "simulator wire[{pos}] in round {round_idx} collapsed into the F-subspace",
                    );
                }
            }

            // Check 3: verifier accepts the simulated transcript.
            //
            // Test form of Lemma 6.4:
            //
            //     simulator output  -->  verifier replay  -->  Ok
            //
            // The replay runs the same shape, target, and PoW checks the honest path triggers.
            let replay = verifier
                .into_sumcheck::<MyMmcs, _>(
                    &sim_zk_data,
                    &mask_commitment,
                    ell_zk,
                    folding_factor,
                    pow_bits,
                    &mut verifier_replay_ch,
                );
            prop_assert!(
                replay.is_ok(),
                "verifier rejected the simulated transcript: {:?}",
                replay.err(),
            );
        }
    }

    /// Replay one recorded masked batch through a bare prover-side driver.
    ///
    /// Reads nothing but the description and the values the batch bound.
    /// Returns the per-round challenges the driver draws.
    ///
    /// `claim` selects the prelude: `None` draws the batching challenge, `Some` binds the inherited scalar.
    fn replay_through_the_description(
        challenger: &mut MyChallenger,
        shape: ZkSumcheckShape,
        claim: Option<EF>,
        mask_commitment: <MyMmcs as Mmcs<EF>>::Commitment,
        zk_data: &ZkSumcheckData<F, EF>,
    ) -> Vec<EF> {
        let mut transcript = ZkProverTranscript::<MyChallenger, F, EF>::new(challenger, shape);
        match claim {
            None => {
                let _alpha: EF = transcript.batching_challenge();
            }
            Some(claim) => transcript.bind_claim(claim),
        }
        let _eps = transcript.masks(mask_commitment, zk_data.mu_tilde);
        let gammas = zk_data
            .round_coefficients
            .iter()
            .map(|wire| transcript.round(wire).0)
            .collect();
        transcript.finish();
        gammas
    }

    #[test]
    fn the_simulator_and_the_prover_play_one_description() {
        // Invariant: honest-verifier zero knowledge needs both parties on one step sequence.
        //
        // Neither party keeps a private copy of the masking prelude.
        //
        // Both build the same shape from the same three numbers, and both drive it.
        //
        // What this pins is that each party's challenge stream is reproduced by a bare driver
        // over that description alone, fed only the values the party bound.
        //
        //     party      | bound values                          | stream
        //     -----------+---------------------------------------+---------------
        //     prover     | its commitment, mu_tilde, its wires   | its gammas
        //     simulator  | its commitment, mu_tilde, its wires   | its gammas
        //
        // A step either party played out of order would make the bare driver panic on the
        // description, or land on a different stream.
        //
        // Fixture state: n_vars = 6, folding = 2, ell_zk = 4, num_virtual = 1, seed = 11.
        //
        // PoW is disabled, so the replay draws no witness of its own.
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let num_virtual = 1;
        let pow_bits = 0;
        let seed = 11u64;

        let shape = ZkSumcheckShape::new_batching(folding_factor, ell_zk, pow_bits);

        // Honest run.
        //
        // The verifier-side challenger sits at the post-claim state both parties
        // start their batch from.
        let real_run = run_prover(
            VariableOrder::Prefix,
            n_vars,
            folding_factor,
            ell_zk,
            0,
            num_virtual,
            pow_bits,
            seed,
        );
        let real_gammas: Vec<EF> = real_run.prover_randomness.iter().copied().collect();

        // The layout prover's own stream, reproduced from the description alone.
        let mut real_replay_ch = real_run.verifier_challenger.clone();
        assert_eq!(
            replay_through_the_description(
                &mut real_replay_ch,
                shape,
                None,
                real_run.mask_commitment.clone(),
                &real_run.zk_data,
            ),
            real_gammas,
            "the layout prover's stream must come out of the shared description",
        );

        // Simulator run from a challenger driven to the same post-claim state.
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let mut verifier_sim = ZkVerifier::<F, EF>::new_prefix(&[TableShape::new(n_vars, 1)]);
        let mut sim_ch = MyChallenger::new(perm);
        for &eval in &real_run.virtual_evals {
            verifier_sim.add_virtual_eval(eval, &mut sim_ch);
        }
        let mut sim_replay_ch = sim_ch.clone();
        let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (sim_zk_data, sim_commitment, sim_gammas) =
            simulate_classic_unpacked::<F, EF, _, _, _, _>(
                &mut sim_ch,
                &verifier_sim,
                folding_factor,
                pow_bits,
                &encoding,
                &mmcs,
                &mut sim_rng,
            );

        // The simulator's own stream, reproduced from the very same description.
        assert_eq!(
            replay_through_the_description(
                &mut sim_replay_ch,
                shape,
                None,
                sim_commitment.clone(),
                &sim_zk_data,
            ),
            sim_gammas.iter().copied().collect::<Vec<_>>(),
            "the simulator's stream must come out of the shared description",
        );

        // Matched RNG seeds, so the masking prelude couples value for value.
        assert_eq!(real_run.zk_data.mu_tilde, sim_zk_data.mu_tilde);
        assert_eq!(real_run.mask_commitment, sim_commitment);

        // The wires themselves are not equal, and Lemma 6.4 does not claim they are.
        //
        //     prover     ->  the round polynomial its masked witness produces
        //     simulator  ->  a uniform draw over the same space
        //
        // What the two share is the description, which is what the assertions above pin.
        assert_eq!(
            real_run.zk_data.round_coefficients.len(),
            sim_zk_data.round_coefficients.len(),
        );
    }

    #[test]
    fn simulator_with_pow_replays() {
        // The simulator must emit one valid grinding witness per round when
        // the sumcheck configuration enables PoW.
        let seed = 0x5eed;
        let n_vars = 4;
        let folding_factor = 2;
        let ell_zk = 4;
        let pow_bits = 4;
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

        let mut simulator_challenger = MyChallenger::new(perm);
        let mut verifier = ZkVerifier::<F, EF>::new_prefix(&[TableShape::new(n_vars, 1)]);
        verifier.add_virtual_eval(EF::from_u64(7), &mut simulator_challenger);
        let mut verifier_challenger = simulator_challenger.clone();
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(2));

        let (zk_data, mask_commitment, _) = simulate_classic_unpacked::<F, EF, _, _, _, _>(
            &mut simulator_challenger,
            &verifier,
            folding_factor,
            pow_bits,
            &encoding,
            &mmcs,
            &mut rng,
        );

        assert_eq!(zk_data.pow_witnesses.len(), folding_factor);
        verifier
            .into_sumcheck::<MyMmcs, _>(
                &zk_data,
                &mask_commitment,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut verifier_challenger,
            )
            .expect("verifier must accept the simulator's PoW witnesses");
    }

    /// Acceptance, mask-prelude coupling and wire-shape driver for the inherited-claim prelude.
    ///
    /// # Invariants per `(order, n_vars, folding, ell_zk, aux)` case
    ///
    /// 1. The verifier accepts both the residual prover's and the simulator's transcript.
    /// 2. Each side's challenge stream is the one the verifier redraws from the claim it was handed.
    /// 3. `mu_tilde` and the mask commit match bit-for-bit under matched seeds (deterministic equality, not a distributional test).
    /// 4. The simulated record has one wire of width `max(ell_zk, 3) - 1` per round, and no grinding witness.
    /// 5. Wire coordinates with index `>= 2` escape the base-field subspace on both sides.
    ///
    /// Check 2 is what a wrong bound claim trips.
    ///
    /// Check 5 reads the band above the two plain-bearing slots, which the wire width decides:
    ///
    /// ```text
    ///     ell_zk = 3  ->  wire_len = 2, so the band is empty and check 5 reads nothing
    ///     ell_zk > 3  ->  wire_len = ell_zk - 1, so the band is ell_zk - 3 wide
    /// ```
    ///
    /// The shortest legal mask is therefore a case check 5 cannot fail on.
    /// The proptest below draws above it, and a test of its own covers the floor.
    ///
    /// Grinding is off here, so the replay accepts any well-shaped record and only the redrawn stream moves.
    ///
    /// The simulator receives the target as public input and no verifier at all.
    #[allow(clippy::too_many_arguments)]
    fn run_inherited_acceptance_and_mask_prelude_coupling(
        order: VariableOrder,
        n_vars: usize,
        folding_factor: usize,
        ell_zk: usize,
        aux_claim: EF,
        seed: u64,
    ) -> Result<(), &'static str> {
        let pow_bits = 0;

        // Honest arm: a product polynomial and the claim it really sums to.
        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals = Poly::<EF>::rand(&mut data_rng, n_vars);
        let weights = Poly::<EF>::rand(&mut data_rng, n_vars);
        let claimed_sum = dot_product::<EF, _, _>(
            evals.as_slice().iter().copied(),
            weights.as_slice().iter().copied(),
        );

        // The scalar both sides bind, and the only instance data the simulator reads:
        //
        //     joint_claim = <f, w> + aux
        let joint_claim = claimed_sum + aux_claim;

        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let poly = ProductPolynomial::<F, EF>::new_unpacked(order, evals, weights);

        let mut real_ch = MyChallenger::new(perm.clone());
        let mut real_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let mut zk_data_real = ZkSumcheckData::<F, EF>::default();
        let real_handoff = SumcheckProver::new(poly, claimed_sum).into_zk_sumcheck(
            &mut zk_data_real,
            &encoding,
            &mmcs,
            folding_factor,
            pow_bits,
            aux_claim,
            &mut real_ch,
            &mut real_rng,
        );
        let commitment_real = real_handoff.mask_oracle.0.clone();

        // Honest verifier replay of the residual arm.
        let mut real_verifier_ch = MyChallenger::new(perm.clone());
        let real_replay = ZkVerifier::<F, EF>::verify_claim::<MyMmcs, _>(
            &zk_data_real,
            &commitment_real,
            ell_zk,
            folding_factor,
            pow_bits,
            joint_claim,
            &mut real_verifier_ch,
        )
        .map_err(|_| "real residual transcript rejected by verifier")?;
        if real_replay.randomness != real_handoff.randomness {
            return Err("verifier redrew a different stream than the residual prover's");
        }

        // === Simulator run (matched mask-RNG seed) ===
        //
        // Same fresh sponge state, same encoding, same commitment scheme.
        let mut sim_ch = MyChallenger::new(perm.clone());
        let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (zk_data_sim, commitment_sim, gammas_sim) =
            simulate_classic_unpacked_claim::<F, EF, _, _, _, _>(
                &mut sim_ch,
                joint_claim,
                folding_factor,
                pow_bits,
                &encoding,
                &mmcs,
                &mut sim_rng,
            );

        let mut sim_verifier_ch = MyChallenger::new(perm);
        let sim_replay = ZkVerifier::<F, EF>::verify_claim::<MyMmcs, _>(
            &zk_data_sim,
            &commitment_sim,
            ell_zk,
            folding_factor,
            pow_bits,
            joint_claim,
            &mut sim_verifier_ch,
        )
        .map_err(|_| "simulator transcript rejected by verifier")?;
        if sim_replay.randomness != gammas_sim {
            return Err("verifier redrew a different stream than the simulator's");
        }

        // === Coupling certificate ===

        if zk_data_real.mu_tilde != zk_data_sim.mu_tilde {
            return Err("matched-RNG coupling: mu_tilde differs");
        }
        if commitment_real != commitment_sim {
            return Err("matched-RNG coupling: mask commitment differs");
        }

        // === Wire shape ===
        //
        //     wire_len = max(ell_zk, 3) - 1
        let wire_len = ell_zk.max(3) - 1;
        if zk_data_sim.round_coefficients.len() != folding_factor {
            return Err("simulator emitted a wire count other than the round count");
        }
        if gammas_sim.as_slice().len() != folding_factor {
            return Err("simulator emitted a challenge count other than the round count");
        }
        if zk_data_sim
            .round_coefficients
            .iter()
            .any(|wire| wire.len() != wire_len)
        {
            return Err("simulator emitted a wire of the wrong width");
        }
        if !zk_data_sim.pow_witnesses.is_empty() {
            return Err("simulator emitted a grinding witness with pow_bits == 0");
        }

        // === Extension-valued wires on both sides ===
        //
        // The masks are extension-valued, so every wire coordinate is uniform over EF.
        //
        // The index-`>= 2` coordinates are mask-only.
        // An `F`-valued mask would pin them to the base field and reintroduce the witness leak.
        for wire in &zk_data_real.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !escapes_f_subspace(c) {
                    return Err("residual-prover wire[i >= 2] collapsed into the F-subspace");
                }
            }
        }
        for wire in &zk_data_sim.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !escapes_f_subspace(c) {
                    return Err("simulator wire[i >= 2] collapsed into the F-subspace");
                }
            }
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_inherited_simulator_accepts_and_couples_mask_prelude_rs(
            n_vars in 2usize..=8,
            ell_zk in 4usize..=5,
            order_bit in 0usize..2,
            seed in 0u64..1024,
        ) {
            // Invariant: on the prelude a WHIR round plays, both transcripts verify.
            //
            // Their mask preludes also couple value for value under matched RNG seeds.
            //
            // Per-case invariants are pinned by the driver's docstring.
            //
            // Why ell_zk >= 4: lengths 2 and 3 both give a 2-coordinate wire, on which the driver's check 5 reads nothing.
            // The shortest legal mask is covered by a test of its own instead.
            //
            // Fixture state: no auxiliary claim.
            //
            // A live carry is exercised by a test of its own.
            //
            // The residual arm never packs, so folding may reach every variable.
            let folding_factor = 1 + (seed as usize % n_vars);
            let order = if order_bit == 0 {
                VariableOrder::Prefix
            } else {
                VariableOrder::Suffix
            };

            prop_assert!(
                run_inherited_acceptance_and_mask_prelude_coupling(
                    order,
                    n_vars,
                    folding_factor,
                    ell_zk,
                    EF::ZERO,
                    seed,
                ).is_ok()
            );
        }
    }

    #[test]
    fn both_simulators_hold_at_the_shortest_legal_mask() {
        // Invariant: the shortest mask a batch may run under still accepts and still couples.
        //
        // Fixture state: ell_zk = 3, 8 variables, 2 rounds, both binding orders, no auxiliary claim.
        //
        //     wire_len = max(3, 3) - 1 = 2
        //
        // The wire is the two plain-bearing slots alone, so the mask-only band above them is empty.
        // Both drivers' stratification check therefore reads no coordinate at this length.
        // What this pins is acceptance, the redrawn challenge stream, and the matched-seed coupling.
        //
        // The proptests draw from 4 so that band always has work.
        // This is the case that keeps the floor covered while they do.
        let ell_zk = 3;
        let n_vars = 8;
        let folding_factor = 2;
        let seed = 0xF10;

        for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
            let batching = run_acceptance_and_mask_prelude_coupling(
                order,
                n_vars,
                folding_factor,
                ell_zk,
                1,
                seed,
            );
            assert_eq!(
                batching,
                Ok(()),
                "recorded-claims prelude, order = {order:?}: {}",
                batching.err().unwrap_or_default(),
            );

            let inherited = run_inherited_acceptance_and_mask_prelude_coupling(
                order,
                n_vars,
                folding_factor,
                ell_zk,
                EF::ZERO,
                seed,
            );
            assert_eq!(
                inherited,
                Ok(()),
                "inherited-claim prelude, order = {order:?}: {}",
                inherited.err().unwrap_or_default(),
            );
        }
    }

    #[test]
    fn the_inherited_simulator_runs_against_a_non_zero_auxiliary_claim() {
        // Invariant: the auxiliary constant is part of the public target, not a second input.
        //
        // Definition 5.8 of eprint 2026/391 puts it inside `mu`:
        //
        //     mu = <f, sl(st)> + sum_i <xi_i, sl_i(st_i)>
        //        = claimed_sum + aux
        //
        // WHIR's per-round call site passes a running mask-claim total there.
        // The deployed simulator therefore has to hold with the constant switched on.
        //
        // It is constant in X and independent of the masks.
        // So it rides the affine offset and never reaches the linear map on the witness.
        //
        // Fixture state: 6 variables, 3 rounds, mask length 4, both binding orders.
        //
        //     aux = 1           ->  smallest live carry
        //     aux = 7           ->  the value the honest-side deferred-binding test uses
        //     aux = 12345       ->  an arbitrary interior value
        //     aux = 2013265920  ->  p - 1 for p = 2^{31} - 2^{27} + 1, the largest element
        let n_vars = 6;
        let folding_factor = 3;
        let ell_zk = 4;

        for aux in [1u64, 7, 12_345, 2_013_265_920] {
            for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                let aux_claim = EF::from_u64(aux);
                assert_ne!(
                    aux_claim,
                    EF::ZERO,
                    "the fixture must exercise a live carry"
                );

                let outcome = run_inherited_acceptance_and_mask_prelude_coupling(
                    order,
                    n_vars,
                    folding_factor,
                    ell_zk,
                    aux_claim,
                    0x5EED + aux,
                );
                assert_eq!(
                    outcome,
                    Ok(()),
                    "aux = {aux}, order = {order:?}: {}",
                    outcome.err().unwrap_or_default(),
                );
            }
        }
    }

    #[test]
    fn the_inherited_simulator_and_the_residual_prover_play_one_description() {
        // Invariant: honest-verifier zero knowledge needs both parties on one step sequence.
        //
        // This is the inherited-claim prelude.
        // The description therefore opens by binding the scalar rather than by drawing a challenge.
        //
        //     party      | bound values                                  | stream
        //     -----------+-----------------------------------------------+---------------
        //     prover     | joint claim, its commitment, mu_tilde, wires  | its gammas
        //     simulator  | joint claim, its commitment, mu_tilde, wires  | its gammas
        //
        // A step either party played out of order would panic the bare driver, or move its stream.
        //
        // Fixture state: 6 variables, 2 rounds, mask length 4, aux = 7, seed = 11.
        //
        // PoW is disabled, so the replay draws no witness of its own.
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let pow_bits = 0;
        let seed = 11u64;
        let aux_claim = EF::from_u64(7);

        let shape = ZkSumcheckShape::new_inherited(folding_factor, ell_zk, pow_bits);

        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals = Poly::<EF>::rand(&mut data_rng, n_vars);
        let weights = Poly::<EF>::rand(&mut data_rng, n_vars);
        let claimed_sum = dot_product::<EF, _, _>(
            evals.as_slice().iter().copied(),
            weights.as_slice().iter().copied(),
        );
        let joint_claim = claimed_sum + aux_claim;

        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let poly = ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, evals, weights);

        // Honest residual run.
        let mut real_ch = MyChallenger::new(perm.clone());
        let mut real_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let mut zk_data_real = ZkSumcheckData::<F, EF>::default();
        let real_handoff = SumcheckProver::new(poly, claimed_sum).into_zk_sumcheck(
            &mut zk_data_real,
            &encoding,
            &mmcs,
            folding_factor,
            pow_bits,
            aux_claim,
            &mut real_ch,
            &mut real_rng,
        );
        let real_gammas: Vec<EF> = real_handoff.randomness.iter().copied().collect();

        // The residual prover's own stream, reproduced from the description alone.
        let mut real_replay_ch = MyChallenger::new(perm.clone());
        assert_eq!(
            replay_through_the_description(
                &mut real_replay_ch,
                shape,
                Some(joint_claim),
                real_handoff.mask_oracle.0.clone(),
                &zk_data_real,
            ),
            real_gammas,
            "the residual prover's stream must come out of the shared description",
        );

        // Simulator run from an identically seeded sponge.
        let mut sim_ch = MyChallenger::new(perm.clone());
        let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (zk_data_sim, commitment_sim, gammas_sim) =
            simulate_classic_unpacked_claim::<F, EF, _, _, _, _>(
                &mut sim_ch,
                joint_claim,
                folding_factor,
                pow_bits,
                &encoding,
                &mmcs,
                &mut sim_rng,
            );

        // The simulator's own stream, reproduced from the very same description.
        let mut sim_replay_ch = MyChallenger::new(perm);
        assert_eq!(
            replay_through_the_description(
                &mut sim_replay_ch,
                shape,
                Some(joint_claim),
                commitment_sim.clone(),
                &zk_data_sim,
            ),
            gammas_sim.iter().copied().collect::<Vec<_>>(),
            "the simulator's stream must come out of the shared description",
        );

        // Matched RNG seeds, so the masking prelude couples value for value.
        assert_eq!(zk_data_real.mu_tilde, zk_data_sim.mu_tilde);
        assert_eq!(real_handoff.mask_oracle.0, commitment_sim);

        // The wires themselves are not equal, and Lemma 6.4 does not claim they are.
        assert_eq!(
            zk_data_real.round_coefficients.len(),
            zk_data_sim.round_coefficients.len(),
        );
    }

    #[test]
    fn the_inherited_simulator_with_pow_replays() {
        // Invariant: one valid grinding witness per round when the configuration enables PoW.
        //
        // Fixture state: 2 rounds, mask length 4, 4 grinding bits, claim = 9, seed = 0x5eed.
        let seed = 0x5eed;
        let folding_factor = 2;
        let ell_zk = 4;
        let pow_bits = 4;
        let joint_claim = EF::from_u64(9);
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

        let mut sim_ch = MyChallenger::new(perm.clone());
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (zk_data, mask_commitment, _) = simulate_classic_unpacked_claim::<F, EF, _, _, _, _>(
            &mut sim_ch,
            joint_claim,
            folding_factor,
            pow_bits,
            &encoding,
            &mmcs,
            &mut rng,
        );

        assert_eq!(zk_data.pow_witnesses.len(), folding_factor);

        let mut verifier_ch = MyChallenger::new(perm);
        ZkVerifier::<F, EF>::verify_claim::<MyMmcs, _>(
            &zk_data,
            &mask_commitment,
            ell_zk,
            folding_factor,
            pow_bits,
            joint_claim,
            &mut verifier_ch,
        )
        .expect("verifier must accept the simulator's PoW witnesses");
    }
}
