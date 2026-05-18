//! Witness-free simulator for the HVZK sumcheck.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::data::ZkSumcheckData;
use super::verifier::ZkVerifier;

/// Witness-free HVZK simulator (Lemma 6.4).
///
/// Produces a transcript indistinguishable from the honest prover's view by reading only the verifier's recorded claims; never the witness itself.
///
/// # Method
///
/// - Sample alpha and derive `mu` from the verifier's claim batching.
/// - Sample and commit fresh masks just like the prover.
/// - Sample each wire coordinate at the tier matching the honest joint distribution.
/// - Reconstruct the dropped `c_1` from the affine identity, so every simulated wire verifies by construction.
///
/// # Wire stratification
///
/// Honest layout per round (`d = max(ell_zk - 1, 2)`):
///
/// ```text
///     wire[0] = c_0       in EF             (eps-scaled plain)
///     wire[1] = c_2       in EF             (eps-scaled plain)
///     wire[i] = c_{i+1}   in F-subspace     for i >= 2 (mask-only)
/// ```
///
/// Sampling tiers:
///
/// - `wire[0..2]`: uniform in EF.
/// - `wire[2..]`: uniform in F, lifted into EF.
///
/// Without the split, a distinguisher could check whether `c_3, c_4, ...`
/// fall in the base-field subspace and separate the views (paper §6.1).
///
/// # Returns
///
/// - Simulated transcript.
/// - One mask commitment per round.
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
) -> (ZkSumcheckData<F, EF>, Vec<M::Commitment>, Point<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    R: Rng,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    // Protocol shape.
    let k = folding_factor;
    let ell_zk = encoding.message_len();

    // Lemma 6.4 hypotheses.
    assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
    assert!(ell_zk >= 2, "Lemma 6.4 requires ell_zk >= 2");
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

    let masks: Vec<Vec<F>> = (0..k).map(|_| encoding.sample_message(rng)).collect();

    let mut mask_commits: Vec<M::Commitment> = Vec::with_capacity(k);
    for mask in &masks {
        let codeword = encoding.encode(mask, rng);
        let (commit, _prover_data) = mmcs.commit_matrix(codeword);
        challenger.observe(commit.clone());
        mask_commits.push(commit);
    }

    // Phase 3: mu_tilde via the closed form (Construction 6.3 step 2 replay).
    //
    //     mu_tilde = 2^{k-1} * sum_l ( s_l(0) + s_l(1) )
    //              = 2^{k-1} * sum_l ( mask[0].double() + sum(mask[1..]) )
    //
    // Byte-equivalent to the honest prover under matched RNG seeds.
    let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
    let mu_tilde: F = two_to_k_minus_1
        * masks
            .iter()
            .map(|m| m[0].double() + m[1..].iter().copied().sum::<F>())
            .sum::<F>();

    // Observe mu_tilde (lifted to EF) and sample the combining challenge.
    challenger.observe_algebra_element(EF::from(mu_tilde));
    let eps: EF = challenger.sample_algebra_element();

    // Phase 4: per-round wire sampling (Construction 6.3 step 4 simulated; Lemma 6.4 stratification below).
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
        // Stratified wire sample:
        //
        //     wire[0..2] in EF             (eps-scaled plain in honest)
        //     wire[2..]  in F-subspace     (mask-only in honest)
        let wire: Vec<EF> = (0..wire_size)
            .map(|i| {
                if i < 2 {
                    rng.random::<EF>()
                } else {
                    EF::from(rng.random::<F>())
                }
            })
            .collect();

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
        let mut coeffs: Vec<EF> = Vec::with_capacity(h_size);
        coeffs.push(c0);
        coeffs.push(c1);
        coeffs.extend_from_slice(&wire[1..]);
        target = coeffs
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |acc, c| acc * gamma_j + c);

        // Record wire and challenge.
        zk_data.round_coefficients.push(wire);
        randomness.push(gamma_j);
    }

    (zk_data, mask_commits, Point::new(randomness))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{Field, PackedValue};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::layout::TableShape;
    use crate::strategy::VariableOrder;
    use crate::zk::test_helpers::{
        EF, F, MyChallenger, MyMmcs, ef_in_f_subspace, make_setup, run_prover,
    };
    use crate::zk::ZkVerifier;

    /// Lemma 6.4 view-match driver for Reed-Solomon mask encoding.
    ///
    /// # Invariants per `(binding, n_vars, folding, ell_zk, num_eqs)` case
    ///
    /// 1. Verifier accepts both transcripts (soundness floor).
    /// 2. `mu_tilde` and mask commits match bit-for-bit under matched seeds (deterministic equality, not a distributional test).
    /// 3. Wire coordinates with index `>= 2` lie in the base-field subspace on both sides (closes paper §6.1's distinguisher).
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
        let mask_commits_real = real_run.mask_commits.clone();

        // Honest verifier replay.
        let _ = real_run
            .verifier
            .into_sumcheck::<MyMmcs, _>(
                &zk_data_real,
                &mask_commits_real,
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
        let (zk_data_sim, mask_commits_sim, _gammas_sim) =
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
                &mask_commits_sim,
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
        if mask_commits_real != mask_commits_sim {
            return Err("matched-RNG coupling: mask commits differ");
        }

        // === F-subspace stratification on both sides ===

        for wire in &zk_data_real.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !ef_in_f_subspace(c) {
                    return Err("real-prover wire[i >= 2] escapes the F-subspace");
                }
            }
        }
        for wire in &zk_data_sim.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !ef_in_f_subspace(c) {
                    return Err("simulator wire[i >= 2] escapes the F-subspace");
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
            let sim_answers: Vec<F> = encoding.simulate(&positions, &mut sim_ans_rng);
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
            ell_zk in 2usize..=5,
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
            ell_zk in 2usize..=5,
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
            //   1. Shape   - output sizes match `folding_factor`, `ell_zk`, and `pow_bits = 0`.
            //   2. Stratification (+) - every wire coordinate with index >= 2 lies in the F-subspace.
            //   3. Stratification (-) - at least one wire[0..2] coordinate escapes the F-subspace.
            //   4. Acceptance  - a fresh verifier replay accepts the simulated transcript (Lemma 6.4).
            //
            // Wire layout (linear coefficient c_1 dropped):
            //
            //     wire[0] = c_0    in EF (eps-scaled plain piece)
            //     wire[1] = c_2    in EF (eps-scaled plain piece)
            //     wire[2] = c_3    in F-subspace (mask-only)
            //     wire[3] = c_4    in F-subspace (mask-only)
            //     ...
            //
            // Why ell_zk >= 4: lengths 2 and 3 produce a 2-coordinate wire and wire[2..] is empty, so check 2 has no work to do.
            //
            // Without check 2 a distinguisher trivially separates real from simulated views (paper §6.1).
            // Without check 3 a regression that collapses both tiers to F would still pass check 2.
            // Without check 4 a shape-correct but verifier-rejected transcript would slip through.
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
            let (sim_zk_data, commits, gammas) =
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
                commits.len(),
                folding_factor,
                "one mask commitment per round",
            );
            prop_assert_eq!(
                gammas.as_slice().len(),
                folding_factor,
                "one challenge per round",
            );

            // Check 2: positive stratification.
            //
            // wire[i] for i >= 2 must lie in the F-subspace.
            for (round_idx, wire) in sim_zk_data.round_coefficients.iter().enumerate() {
                for (pos, &coeff) in wire.iter().enumerate().skip(2) {
                    prop_assert!(
                        ef_in_f_subspace(coeff),
                        "simulator wire[{pos}] in round {round_idx} must live in F-subspace",
                    );
                }
            }

            // Check 3: negative stratification.
            //
            // Per coord, expected escape probability:
            //
            //     P[wire[0..2] not in F] = 1 - |F|^{-3}
            //
            // Aggregated false-negative bound:
            //
            //     <= 2^{-32 * folding_factor}
            //
            // Catches a regression that collapses both tiers to F.
            let any_ef_coord_escapes = sim_zk_data
                .round_coefficients
                .iter()
                .flat_map(|wire| wire.iter().take(2))
                .any(|&c| !ef_in_f_subspace(c));
            prop_assert!(
                any_ef_coord_escapes,
                "EF-tier samples collapsed into F-subspace across all rounds",
            );

            // Check 4: verifier accepts the simulated transcript.
            //
            // Test form of Lemma 6.4:
            //
            //     simulator output  -->  verifier replay  -->  Ok
            //
            // The replay runs the same shape, target, and PoW checks the honest path triggers.
            let replay = verifier
                .into_sumcheck::<MyMmcs, _>(
                    &sim_zk_data,
                    &commits,
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
