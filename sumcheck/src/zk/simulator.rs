//! Witness-free simulator for the HVZK sumcheck.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncodingWithRandomness;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::data::ZkSumcheckData;
use super::prover::stack_codewords;
use super::verifier::ZkVerifier;

/// Witness-free HVZK simulator (Lemma 6.4).
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
/// Same precondition checks as the prover.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncodingWithRandomness<EF>,
    Enc::Codeword: Matrix<EF>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    R: Rng,
    StandardUniform: Distribution<EF>,
{
    // Protocol shape.
    let k = folding_factor;
    let ell_zk = encoding.message_len();

    // Lemma 6.4 hypotheses.
    assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
    assert!(
        ell_zk >= 3,
        "mask degree ell_zk - 1 must cover the degree-2 plain piece (ell_zk >= 3)",
    );
    assert!(k >= 1, "sumcheck requires at least one round");

    // Phase 1: sample alpha and derive mu (replays Construction 6.3 prelude).
    //
    // Honest prover's transcript order after the claim phase:
    //
    //     alpha  ->  mask commits  ->  mu_tilde  ->  eps  ->  wires
    //
    // `verifier.sum(alpha)` reads only the recorded claims, so Lemma 6.4's witness-freeness is preserved.
    let alpha: EF = challenger.sample_algebra_element();
    let mu = verifier.sum(alpha);

    // Phase 2: sample, encode, commit, observe masks (Construction 6.3 step 1 replay).
    //
    // Uniform messages produce uniform Reed-Solomon codewords.
    // Their Merkle commits are indistinguishable from the honest prover's (zero simulator error for RS).

    let masks: Vec<Vec<EF>> = (0..k).map(|_| encoding.sample_message(rng)).collect();

    // Encode with the same explicit draw order as the prover, stack the
    // batch column-wise, and commit once.
    let codewords: Vec<Enc::Codeword> = masks
        .iter()
        .map(|mask| {
            let randomness = encoding.sample_randomness(rng);
            encoding.encode_with_randomness(mask, &randomness)
        })
        .collect();
    let (mask_commitment, _prover_data) = mmcs.commit_matrix(stack_codewords(&codewords));
    challenger.observe(mask_commitment.clone());

    // Phase 3: mu_tilde via the closed form (Construction 6.3 step 2 replay).
    //
    //     mu_tilde = 2^{k-1} * sum_l ( s_l(0) + s_l(1) )
    //              = 2^{k-1} * sum_l ( mask[0].double() + sum(mask[1..]) )
    //
    // Byte-equivalent to the honest prover under matched RNG seeds.
    let two_to_k_minus_1 = EF::TWO.exp_u64((k - 1) as u64);
    let mu_tilde: EF = two_to_k_minus_1
        * masks
            .iter()
            .map(|m| m[0].double() + m[1..].iter().copied().sum::<EF>())
            .sum::<EF>();

    // Observe mu_tilde and sample the combining challenge.
    challenger.observe_algebra_element(mu_tilde);
    let eps: EF = challenger.sample_algebra_element();

    // Phase 4: per-round wire sampling (Construction 6.3 step 4 simulated; every coordinate uniform over EF).
    //
    // Wire shape (linear coefficient c_1 dropped):
    //
    //     h_size    = max(ell_zk, 3)
    //     wire_size = h_size - 1
    //
    //     wire = [ c_0, c_2, c_3, ..., c_d ]
    let h_size = ell_zk.max(3);
    let wire_size = h_size - 1;

    // Output container; metadata fields populated up front.
    let mut zk_data = ZkSumcheckData::<F, EF> {
        mu_tilde,
        ell_zk,
        round_coefficients: Vec::with_capacity(k),
        pow_witnesses: Vec::with_capacity(if pow_bits > 0 { k } else { 0 }),
    };

    // Per-round challenges and running target mirroring the verifier.
    let mut randomness: Vec<EF> = Vec::with_capacity(k);
    let mut target: EF = eps * mu + mu_tilde;

    for _ in 0..k {
        // Every wire coordinate is uniform over the full extension field
        // (Lemma 6.4 with `F := EF`).
        let wire: Vec<EF> = (0..wire_size).map(|_| rng.random::<EF>()).collect();

        // Absorb the wire on the transcript.
        challenger.observe_algebra_slice(&wire);

        // Drive the grind step when enabled.
        if pow_bits > 0 {
            zk_data.pow_witnesses.push(challenger.grind(pow_bits));
        }

        // Sample the per-round challenge.
        let gamma_j: EF = challenger.sample_algebra_element();

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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
    use p3_zk_codes::ZkEncoding;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::layout::TableShape;
    use crate::strategy::VariableOrder;
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

    /// Lemma 6.4 view-match driver for Reed-Solomon mask encoding.
    ///
    /// # Invariants per `(binding, n_vars, folding, ell_zk, num_eqs)` case
    ///
    /// 1. Verifier accepts both transcripts (soundness floor).
    /// 2. `mu_tilde` and mask commits match bit-for-bit under matched seeds (deterministic equality, not a distributional test).
    /// 3. Wire coordinates with index `>= 2` escape the base-field subspace on both sides (the masks are extension-valued, so the witness leak of an `F`-valued mask is absent).
    /// 4. The mask encoding's own simulator returns the correct shape (RS simulator error = 0).
    ///
    /// The binding parameter only affects the real run; the simulator output depends only on the wire schema, which both binding modes share.
    fn run_view_match_rs(
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
        let m = encoding.m;
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
        fn prop_simulator_view_matches_real_rs_prefix(
            n_vars in 3usize..=8,
            ell_zk in 3usize..=5,
            num_eqs in 1usize..=3,
            seed in 0u64..1024,
        ) {
            // Invariant: simulator view matches honest view across the
            //            (n_vars, folding, ell_zk, num_eqs) cube for RS mask
            //            encoding on the prefix path.
            //
            // Per-case invariants pinned by run_view_match_rs (see docstring).

            // Compression step requires the folded polynomial to retain at
            // least one full packed lane. Packing width depends on the ISA.
            let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));

            prop_assert!(
                run_view_match_rs(VariableOrder::Prefix, n_vars, folding_factor, ell_zk, num_eqs, seed).is_ok()
            );
        }

        #[test]
        fn prop_simulator_view_matches_real_rs_suffix(
            n_vars in 3usize..=8,
            ell_zk in 3usize..=5,
            num_eqs in 1usize..=3,
            seed in 0u64..1024,
        ) {
            // Same invariant on the suffix path. Suffix mode never packs
            // the residual factor, so the parameter window is wider:
            // folding can go up to `n_vars - 1` instead of `n_vars - k_pack`.
            let folding_factor = 1 + (seed as usize % (n_vars - 1).max(1));

            prop_assert!(
                run_view_match_rs(VariableOrder::Suffix, n_vars, folding_factor, ell_zk, num_eqs, seed).is_ok()
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
            //     view-match proptests (both)     | prefix + suffix coupling
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
            prop_assert_eq!(sim_zk_data.ell_zk, ell_zk, "ell_zk header must match the encoding");
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
}
