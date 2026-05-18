//! HVZK base-case IOPP (Construction 7.2, eprint 2026/391 §7).
//!
//! Non-succinct HVZK check for the relation `R_{C, C_zk, sl}` at the terminal
//! leaf of WHIR's recursive code-switching. The non-ZK path sends the polynomial
//! in the clear; this module masks it with fresh ZK-encoded masks.
//!
//! # Protocol (Construction 7.2)
//!
//! **Interaction phase:**
//!
//! 1. **New masks.** Prover sends:
//!    - `g = Enc_C(g̃, r')` as oracle (base-code mask),
//!    - `s_i = Enc_{C_zk}(s̃_i, r'_i)` for `i ∈ [n]` (per-constraint masks),
//!    - `μ' = ⟨g̃, W⟩ + Σ_i ⟨s̃_i, W_i⟩` (masked aggregate target),
//!    - `μ'_i = ⟨w_{o,i}, s̃_i⟩` for each `i` (per-constraint masked targets).
//! 2. **Combination randomness.** Verifier samples `γ`.
//! 3. **Answer.** Prover sends `f* = g̃ + γ·f`, `r* = r' + γ·r`,
//!    and `ξ*_i = s̃_i + γ·ξ_i`, `r*_i = r'_i + γ·r_i` for each `i`.
//!
//! **Decision phase:**
//!
//! - Target: `⟨f*, W⟩ + Σ_i ⟨ξ*_i, W_i⟩ = μ' + γ·μ`
//! - Per-constraint: `⟨w_{o,i}, ξ*_i⟩ = μ'_i + γ·μ_i`
//! - C spot checks: `Enc_C(f*,r*)(x_j) = g(x_j) + γ·f(x_j)` at `t` positions
//! - C_zk spot checks: `Enc_{C_zk}(ξ*_i,r*_i)(y_j) = s_i(y_j) + γ·ξ_i(y_j)`
//!
//! # Security
//!
//! - Lemma 7.3: HVZK with error `ζ_C + n·ζ_{C_zk}` (= 0 for RS encodings).
//! - Lemma 7.4: RBR with errors `(ε_mca + list_sizes/|F|, max{(1-δ)^t, (1-δ_zk)^{t_zk}})`.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_multilinear_util::poly::Poly;
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

/// Per-constraint witness: message `ξ_i`, randomness `r_i`, weight vector `W_i`,
/// output weight vector `w_{o,i}`, and target `μ_i`.
#[derive(Debug, Clone)]
pub struct ConstraintWitness<EF> {
    /// `ξ_i` — per-constraint witness message.
    pub msg: Vec<EF>,
    /// `r_i` — per-constraint witness encoding randomness.
    pub rand: Vec<EF>,
    /// `W_i` — per-constraint weight vector (inner product target for aggregate check).
    pub weights: Vec<EF>,
    /// `w_{o,i}` — per-constraint output weight vector (for per-constraint target check).
    pub output_weights: Vec<EF>,
    /// `μ_i` — per-constraint claimed evaluation.
    pub mu: EF,
}

/// Per-constraint proof data sent by the prover.
#[derive(Debug, Clone)]
pub struct ConstraintProofData<EF> {
    /// `μ'_i = ⟨w_{o,i}, s̃_i⟩` — masked per-constraint target.
    pub mu_prime: EF,
    /// `ξ*_i = s̃_i + γ·ξ_i` — masked per-constraint message.
    pub xi_star: Vec<EF>,
    /// `r*_i = r'_i + γ·r_i` — masked per-constraint randomness.
    pub r_star: Vec<EF>,
}

/// Proof artefact for the HVZK base-case IOPP (Construction 7.2).
#[derive(Debug, Clone)]
pub struct BaseCaseZkData<F, EF> {
    /// `μ'` — masked aggregate target.
    pub mu_prime: EF,
    /// `f* = g̃ + γ·f` — masked witness message.
    pub f_star: Vec<EF>,
    /// `r* = r' + γ·r` — masked witness randomness.
    pub r_star: Vec<EF>,
    /// Per-constraint proof data.
    pub constraints: Vec<ConstraintProofData<EF>>,
    /// PoW witness.
    pub pow_witness: F,
}

impl<F: Field, EF: ExtensionField<F>> Default for BaseCaseZkData<F, EF> {
    fn default() -> Self {
        Self {
            mu_prime: EF::ZERO,
            f_star: Vec::new(),
            r_star: Vec::new(),
            constraints: Vec::new(),
            pow_witness: F::ZERO,
        }
    }
}

/// Namespace for the HVZK base-case IOPP.
pub struct BaseCaseZk;

impl BaseCaseZk {
    /// Prover side of Construction 7.2.
    ///
    /// Implements all protocol steps: base-code mask, per-constraint masks,
    /// μ' and μ'_i computation, f*/r*/ξ*_i/r*_i answer.
    #[allow(clippy::too_many_arguments, unused_variables)]
    pub fn prove<F, EF, EncC, EncZk, MC, MZk, Challenger, R>(
        f_msg: &[EF],
        f_rand: &[EF],
        weights: &Poly<EF>,
        mu: EF,
        constraint_witnesses: &[ConstraintWitness<EF>],
        challenger: &mut Challenger,
        pow_bits: usize,
        enc_c: &EncC,
        enc_zk: &EncZk,
        mmcs_c: &MC,
        mmcs_zk: &MZk,
        rng: &mut R,
    ) -> (
        BaseCaseZkData<F, EF>,
        MC::Commitment,
        Vec<MZk::Commitment>,
        EF,
    )
    where
        F: Field,
        EF: ExtensionField<F>,
        EncC: ZkEncodingWithRandomness<F>,
        EncC::Codeword: Matrix<F>,
        EncZk: ZkEncodingWithRandomness<F>,
        EncZk::Codeword: Matrix<F>,
        MC: Mmcs<F>,
        MZk: Mmcs<F>,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MC::Commitment>
            + CanObserve<MZk::Commitment>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let ell = enc_c.message_len();
        let t = enc_c.randomness_len();
        let ell_zk = enc_zk.message_len();
        let t_zk = enc_zk.randomness_len();
        let n = constraint_witnesses.len();

        assert_eq!(f_msg.len(), ell);
        assert_eq!(f_rand.len(), t);
        assert_eq!(weights.num_evals(), ell);

        // --- Step 1: Base-code mask g ---
        let g_msg: Vec<F> = (0..ell).map(|_| rng.random()).collect();
        let g_rand: Vec<F> = (0..t).map(|_| rng.random()).collect();
        let g_codeword = enc_c.encode_with_randomness(&g_msg, &g_rand);
        let (g_commit, _g_data) = mmcs_c.commit_matrix(g_codeword);
        challenger.observe(g_commit.clone());

        // --- Step 1: Per-constraint masks s_i ---
        let mut s_msgs: Vec<Vec<F>> = Vec::with_capacity(n);
        let mut s_rands: Vec<Vec<F>> = Vec::with_capacity(n);
        let mut s_commits: Vec<MZk::Commitment> = Vec::with_capacity(n);
        for _ in 0..n {
            let s_msg: Vec<F> = (0..ell_zk).map(|_| rng.random()).collect();
            let s_rand: Vec<F> = (0..t_zk).map(|_| rng.random()).collect();
            let s_codeword = enc_zk.encode_with_randomness(&s_msg, &s_rand);
            let (s_commit, _s_data) = mmcs_zk.commit_matrix(s_codeword);
            challenger.observe(s_commit.clone());
            s_msgs.push(s_msg);
            s_rands.push(s_rand);
            s_commits.push(s_commit);
        }

        // --- Step 1: μ' = ⟨W, g̃⟩ + Σ_i ⟨W_i, s̃_i⟩ ---
        let mut mu_prime: EF = weights
            .iter()
            .zip(g_msg.iter())
            .map(|(&w, &g)| w * EF::from(g))
            .sum();
        for (cw, s_msg) in constraint_witnesses.iter().zip(s_msgs.iter()) {
            let contrib: EF = cw
                .weights
                .iter()
                .zip(s_msg.iter())
                .map(|(&w, &s)| w * EF::from(s))
                .sum();
            mu_prime += contrib;
        }
        challenger.observe_algebra_element(mu_prime);

        // --- Step 1: μ'_i = ⟨w_{o,i}, s̃_i⟩ for each i ---
        let mut constraint_data: Vec<ConstraintProofData<EF>> = Vec::with_capacity(n);
        for (cw, s_msg) in constraint_witnesses.iter().zip(s_msgs.iter()) {
            let mu_prime_i: EF = cw
                .output_weights
                .iter()
                .zip(s_msg.iter())
                .map(|(&w, &s)| w * EF::from(s))
                .sum();
            challenger.observe_algebra_element(mu_prime_i);
            constraint_data.push(ConstraintProofData {
                mu_prime: mu_prime_i,
                xi_star: Vec::new(),
                r_star: Vec::new(),
            });
        }

        // --- Step 2: Sample γ ---
        let gamma: EF = challenger.sample_algebra_element();

        // --- Step 3: f* = g̃ + γ·f, r* = r' + γ·r ---
        let f_star: Vec<EF> = g_msg
            .iter()
            .zip(f_msg.iter())
            .map(|(&g, &f)| EF::from(g) + gamma * f)
            .collect();
        let r_star: Vec<EF> = g_rand
            .iter()
            .zip(f_rand.iter())
            .map(|(&g, &r)| EF::from(g) + gamma * r)
            .collect();
        challenger.observe_algebra_slice(&f_star);
        challenger.observe_algebra_slice(&r_star);

        // --- Step 3: ξ*_i = s̃_i + γ·ξ_i, r*_i = r'_i + γ·r_i ---
        for (i, cw) in constraint_witnesses.iter().enumerate() {
            let xi_star: Vec<EF> = s_msgs[i]
                .iter()
                .zip(cw.msg.iter())
                .map(|(&s, &xi)| EF::from(s) + gamma * xi)
                .collect();
            let r_star_i: Vec<EF> = s_rands[i]
                .iter()
                .zip(cw.rand.iter())
                .map(|(&s, &r)| EF::from(s) + gamma * r)
                .collect();
            challenger.observe_algebra_slice(&xi_star);
            challenger.observe_algebra_slice(&r_star_i);
            constraint_data[i].xi_star = xi_star;
            constraint_data[i].r_star = r_star_i;
        }

        let mut data = BaseCaseZkData {
            mu_prime,
            f_star,
            r_star,
            constraints: constraint_data,
            pow_witness: F::ZERO,
        };
        if pow_bits > 0 {
            data.pow_witness = challenger.grind(pow_bits);
        }

        // Prover-side sanity: aggregate target check.
        #[cfg(debug_assertions)]
        {
            let mut lhs: EF = weights
                .iter()
                .zip(data.f_star.iter())
                .map(|(&w, &f)| w * f)
                .sum();
            for (cw, cd) in constraint_witnesses.iter().zip(data.constraints.iter()) {
                let c: EF = cw
                    .weights
                    .iter()
                    .zip(cd.xi_star.iter())
                    .map(|(&w, &x)| w * x)
                    .sum();
                lhs += c;
            }
            debug_assert_eq!(lhs, mu_prime + gamma * mu, "aggregate target check failed");
        }

        (data, g_commit, s_commits, gamma)
    }

    /// Verifier side of Construction 7.2.
    ///
    /// Verifies aggregate target check and per-constraint target checks.
    /// Spot checks (C and C_zk codeword consistency) are returned as a
    /// verification obligation for the caller, which holds the Merkle data.
    #[allow(clippy::too_many_arguments)]
    pub fn verify<F, EF, MC, MZk, Challenger>(
        data: &BaseCaseZkData<F, EF>,
        weights: &Poly<EF>,
        mu: EF,
        constraint_params: &[(Vec<EF>, Vec<EF>, EF)], // (W_i, w_{o,i}, μ_i) per constraint
        g_commit: &MC::Commitment,
        s_commits: &[MZk::Commitment],
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<EF, BaseCaseZkError>
    where
        F: Field,
        EF: ExtensionField<F>,
        MC: Mmcs<F>,
        MZk: Mmcs<F>,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MC::Commitment>
            + CanObserve<MZk::Commitment>,
    {
        let n = constraint_params.len();
        if data.constraints.len() != n {
            return Err(BaseCaseZkError::ConstraintCountMismatch {
                expected: n,
                actual: data.constraints.len(),
            });
        }
        if s_commits.len() != n {
            return Err(BaseCaseZkError::ConstraintCountMismatch {
                expected: n,
                actual: s_commits.len(),
            });
        }

        // Replay step 1: observe g commitment.
        challenger.observe(g_commit.clone());

        // Observe per-constraint mask commitments.
        for commit in s_commits {
            challenger.observe(commit.clone());
        }

        // Observe μ'.
        challenger.observe_algebra_element(data.mu_prime);

        // Observe μ'_i for each constraint.
        for cd in &data.constraints {
            challenger.observe_algebra_element(cd.mu_prime);
        }

        // Step 2: sample γ.
        let gamma: EF = challenger.sample_algebra_element();

        // Replay step 3: observe f*, r*.
        challenger.observe_algebra_slice(&data.f_star);
        challenger.observe_algebra_slice(&data.r_star);

        // Observe ξ*_i, r*_i for each constraint.
        for cd in &data.constraints {
            challenger.observe_algebra_slice(&cd.xi_star);
            challenger.observe_algebra_slice(&cd.r_star);
        }

        // PoW check.
        if pow_bits > 0 && !challenger.check_witness(pow_bits, data.pow_witness) {
            return Err(BaseCaseZkError::InvalidPowWitness);
        }

        // Aggregate target check: ⟨W, f*⟩ + Σ_i ⟨W_i, ξ*_i⟩ = μ' + γ·μ
        let mut lhs: EF = weights
            .iter()
            .zip(data.f_star.iter())
            .map(|(&w, &f)| w * f)
            .sum();
        for (cp, cd) in constraint_params.iter().zip(data.constraints.iter()) {
            let c: EF =
                cp.0.iter()
                    .zip(cd.xi_star.iter())
                    .map(|(&w, &x)| w * x)
                    .sum();
            lhs += c;
        }
        if lhs != data.mu_prime + gamma * mu {
            return Err(BaseCaseZkError::AggregateTargetCheckFailed);
        }

        // Per-constraint target checks: ⟨w_{o,i}, ξ*_i⟩ = μ'_i + γ·μ_i
        for (i, (cp, cd)) in constraint_params
            .iter()
            .zip(data.constraints.iter())
            .enumerate()
        {
            let lhs_i: EF =
                cp.1.iter()
                    .zip(cd.xi_star.iter())
                    .map(|(&w, &x)| w * x)
                    .sum();
            if lhs_i != cd.mu_prime + gamma * cp.2 {
                return Err(BaseCaseZkError::PerConstraintTargetCheckFailed { index: i });
            }
        }

        Ok(gamma)
    }

    /// Simulator for the HVZK base case (Lemma 7.3).
    ///
    /// Produces a transcript indistinguishable from the real prover's without
    /// access to the witness. For RS encodings the simulation error is 0.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn simulate<F, EF, EncC, EncZk, MC, MZk, Challenger, R>(
        weights: &Poly<EF>,
        mu: EF,
        constraint_params: &[(Vec<EF>, Vec<EF>, EF)], // (W_i, w_{o,i}, μ_i)
        challenger: &mut Challenger,
        pow_bits: usize,
        enc_c: &EncC,
        enc_zk: &EncZk,
        mmcs_c: &MC,
        mmcs_zk: &MZk,
        rng: &mut R,
    ) -> (
        BaseCaseZkData<F, EF>,
        MC::Commitment,
        Vec<MZk::Commitment>,
        EF,
    )
    where
        F: Field,
        EF: ExtensionField<F>,
        EncC: ZkEncoding<F>,
        EncC::Codeword: Matrix<F>,
        EncZk: ZkEncoding<F>,
        EncZk::Codeword: Matrix<F>,
        MC: Mmcs<F>,
        MZk: Mmcs<F>,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MC::Commitment>
            + CanObserve<MZk::Commitment>,
        R: Rng,
        StandardUniform: Distribution<F> + Distribution<EF>,
    {
        let ell = enc_c.message_len();
        let t = enc_c.randomness_len();
        let ell_zk = enc_zk.message_len();
        let t_zk = enc_zk.randomness_len();
        let n = constraint_params.len();

        // Step 1: real masks (same distribution as prover for RS).
        let g_msg: Vec<F> = (0..ell).map(|_| rng.random()).collect();
        let g_codeword = enc_c.encode(&g_msg, rng);
        let (g_commit, _) = mmcs_c.commit_matrix(g_codeword);
        challenger.observe(g_commit.clone());

        let mut s_msgs: Vec<Vec<F>> = Vec::with_capacity(n);
        let mut s_commits: Vec<MZk::Commitment> = Vec::with_capacity(n);
        for _ in 0..n {
            let s_msg: Vec<F> = (0..ell_zk).map(|_| rng.random()).collect();
            let s_codeword = enc_zk.encode(&s_msg, rng);
            let (s_commit, _) = mmcs_zk.commit_matrix(s_codeword);
            challenger.observe(s_commit.clone());
            s_msgs.push(s_msg);
            s_commits.push(s_commit);
        }

        // μ' from masks (same computation as prover).
        let mut mu_prime: EF = weights
            .iter()
            .zip(g_msg.iter())
            .map(|(&w, &g)| w * EF::from(g))
            .sum();
        for (cp, s_msg) in constraint_params.iter().zip(s_msgs.iter()) {
            let c: EF =
                cp.0.iter()
                    .zip(s_msg.iter())
                    .map(|(&w, &s)| w * EF::from(s))
                    .sum();
            mu_prime += c;
        }
        challenger.observe_algebra_element(mu_prime);

        // μ'_i from masks.
        let mut constraint_data: Vec<ConstraintProofData<EF>> = Vec::with_capacity(n);
        for (cp, s_msg) in constraint_params.iter().zip(s_msgs.iter()) {
            let mu_prime_i: EF =
                cp.1.iter()
                    .zip(s_msg.iter())
                    .map(|(&w, &s)| w * EF::from(s))
                    .sum();
            challenger.observe_algebra_element(mu_prime_i);
            constraint_data.push(ConstraintProofData {
                mu_prime: mu_prime_i,
                xi_star: Vec::new(),
                r_star: Vec::new(),
            });
        }

        // Step 2: sample γ.
        let gamma: EF = challenger.sample_algebra_element();

        // Step 4: sample f* conditioned on aggregate target.
        // ⟨W, f*⟩ + Σ_i ⟨W_i, ξ*_i⟩ = μ' + γ·μ
        // First sample ξ*_i conditioned on per-constraint targets,
        // then sample f* conditioned on the residual aggregate target.

        // Per-constraint: sample ξ*_i conditioned on ⟨w_{o,i}, ξ*_i⟩ = μ'_i + γ·μ_i
        for (i, cp) in constraint_params.iter().enumerate() {
            let target_i = constraint_data[i].mu_prime + gamma * cp.2;
            let mut xi_star: Vec<EF> = (0..ell_zk).map(|_| rng.random::<EF>()).collect();
            // Find pivot in w_{o,i} to condition on.
            if let Some(pivot) = cp.1.iter().rposition(|&w| w != EF::ZERO) {
                let partial: EF =
                    cp.1.iter()
                        .zip(xi_star.iter())
                        .enumerate()
                        .filter(|&(j, _)| j != pivot)
                        .map(|(_, (&w, &x))| w * x)
                        .sum();
                xi_star[pivot] = (target_i - partial) * cp.1[pivot].try_inverse().unwrap();
            }
            let r_star_i: Vec<EF> = (0..t_zk).map(|_| rng.random::<EF>()).collect();
            challenger.observe_algebra_slice(&xi_star);
            challenger.observe_algebra_slice(&r_star_i);
            constraint_data[i].xi_star = xi_star;
            constraint_data[i].r_star = r_star_i;
        }

        // f*: sample conditioned on aggregate target minus per-constraint contributions.
        let aggregate_target = mu_prime + gamma * mu;
        let constraint_contrib: EF = constraint_params
            .iter()
            .zip(constraint_data.iter())
            .map(|(cp, cd)| {
                cp.0.iter()
                    .zip(cd.xi_star.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<EF>()
            })
            .sum();
        let f_star_target = aggregate_target - constraint_contrib;

        let mut f_star: Vec<EF> = (0..ell).map(|_| rng.random::<EF>()).collect();
        if let Some(pivot) = weights.iter().rposition(|&w| w != EF::ZERO) {
            let partial: EF = weights
                .iter()
                .zip(f_star.iter())
                .enumerate()
                .filter(|&(j, _)| j != pivot)
                .map(|(_, (&w, &f))| w * f)
                .sum();
            f_star[pivot] =
                (f_star_target - partial) * weights.as_slice()[pivot].try_inverse().unwrap();
        }
        let r_star: Vec<EF> = (0..t).map(|_| rng.random::<EF>()).collect();

        challenger.observe_algebra_slice(&f_star);
        challenger.observe_algebra_slice(&r_star);

        let mut data = BaseCaseZkData {
            mu_prime,
            f_star,
            r_star,
            constraints: constraint_data,
            pow_witness: F::ZERO,
        };
        if pow_bits > 0 {
            data.pow_witness = challenger.grind(pow_bits);
        }

        (data, g_commit, s_commits, gamma)
    }
}

/// Verify C spot checks: `Enc_C(f*, r*)(x_j) = g(x_j) + γ·f(x_j)`.
///
/// The caller provides the evaluated values at each queried position.
/// This function checks the linear relationship. The Merkle opening
/// infrastructure (sampling positions, opening proofs) is the caller's
/// responsibility.
///
/// # Arguments
///
/// - `f_star_evals` — `Enc_C(f*, r*)(x_j)` at each queried position
/// - `g_evals` — `g(x_j)` at each queried position (from Merkle open)
/// - `f_evals` — `f(x_j)` at each queried position (from Merkle open)
/// - `gamma` — combination challenge
pub fn verify_c_spot_checks<EF: ExtensionField<impl Field>>(
    f_star_evals: &[EF],
    g_evals: &[EF],
    f_evals: &[EF],
    gamma: EF,
) -> Result<(), BaseCaseZkError> {
    for (j, ((&fs, &g), &f)) in f_star_evals
        .iter()
        .zip(g_evals.iter())
        .zip(f_evals.iter())
        .enumerate()
    {
        if fs != g + gamma * f {
            return Err(BaseCaseZkError::CSpotCheckFailed { position: j });
        }
    }
    Ok(())
}

/// Verify C_zk spot checks for constraint `i`:
/// `Enc_{C_zk}(ξ*_i, r*_i)(y_j) = s_i(y_j) + γ·ξ_i(y_j)`.
pub fn verify_czk_spot_checks<EF: ExtensionField<impl Field>>(
    constraint_idx: usize,
    xi_star_evals: &[EF],
    s_evals: &[EF],
    xi_evals: &[EF],
    gamma: EF,
) -> Result<(), BaseCaseZkError> {
    for (j, ((&xs, &s), &xi)) in xi_star_evals
        .iter()
        .zip(s_evals.iter())
        .zip(xi_evals.iter())
        .enumerate()
    {
        if xs != s + gamma * xi {
            return Err(BaseCaseZkError::CzkSpotCheckFailed {
                constraint: constraint_idx,
                position: j,
            });
        }
    }
    Ok(())
}

/// Errors from the HVZK base-case verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseCaseZkError {
    /// Aggregate target check failed.
    AggregateTargetCheckFailed,
    /// Per-constraint target check failed.
    PerConstraintTargetCheckFailed { index: usize },
    /// PoW witness invalid.
    InvalidPowWitness,
    /// Constraint count mismatch.
    ConstraintCountMismatch { expected: usize, actual: usize },
    /// C spot check failed.
    CSpotCheckFailed { position: usize },
    /// C_zk spot check failed.
    CzkSpotCheckFailed { constraint: usize, position: usize },
}

impl core::fmt::Display for BaseCaseZkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AggregateTargetCheckFailed => write!(f, "aggregate target check failed"),
            Self::PerConstraintTargetCheckFailed { index } => {
                write!(f, "per-constraint target check failed at {index}")
            }
            Self::InvalidPowWitness => write!(f, "invalid PoW witness"),
            Self::ConstraintCountMismatch { expected, actual } => {
                write!(f, "constraint count {actual}, expected {expected}")
            }
            Self::CSpotCheckFailed { position } => write!(f, "C spot check failed at {position}"),
            Self::CzkSpotCheckFailed {
                constraint,
                position,
            } => write!(
                f,
                "C_zk spot check failed: constraint {constraint}, position {position}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_zk_codes::ZkEncodingWithRandomness;
    use p3_zk_codes::reed_solomon::ReedSolomonZkEncoding;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type PackedF = <F as Field>::Packing;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    fn test_perm() -> Perm {
        Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(0))
    }

    fn test_mmcs(perm: &Perm) -> ValMmcs {
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        ValMmcs::new(hash, compress, 0)
    }

    /// Encoding linearity: Enc(a+b, r1+r2) = Enc(a,r1) + Enc(b,r2).
    /// This is the mathematical foundation of spot checks.
    #[test]
    fn encoding_linearity() {
        let msg_len = 4;
        let t = 2;
        let m = 8;
        let dft = Radix2DFTSmallBatch::default();
        let enc = ReedSolomonZkEncoding::<F, _>::new(t, msg_len, m, dft);
        let mut rng = SmallRng::seed_from_u64(99);

        let f: Vec<F> = (0..msg_len).map(|_| rng.random()).collect();
        let rf: Vec<F> = (0..t).map(|_| rng.random()).collect();
        let g: Vec<F> = (0..msg_len).map(|_| rng.random()).collect();
        let rg: Vec<F> = (0..t).map(|_| rng.random()).collect();
        let gamma = F::from_u64(42);

        let sum_msg: Vec<F> = g
            .iter()
            .zip(f.iter())
            .map(|(&a, &b)| a + gamma * b)
            .collect();
        let sum_rand: Vec<F> = rg
            .iter()
            .zip(rf.iter())
            .map(|(&a, &b)| a + gamma * b)
            .collect();

        let enc_f = enc.encode_with_randomness(&f, &rf);
        let enc_g = enc.encode_with_randomness(&g, &rg);
        let enc_sum = enc.encode_with_randomness(&sum_msg, &sum_rand);

        for j in 0..m {
            let lhs = enc_sum.row_slice(j).unwrap()[0];
            let rhs = enc_g.row_slice(j).unwrap()[0] + gamma * enc_f.row_slice(j).unwrap()[0];
            assert_eq!(lhs, rhs, "linearity at position {j}");
        }
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn make_test_data(
        msg_len: usize,
        t: usize,
        n_constraints: usize,
        ell_zk: usize,
        t_zk: usize,
        rng: &mut SmallRng,
    ) -> (Vec<EF>, Vec<EF>, Poly<EF>, EF, Vec<ConstraintWitness<EF>>) {
        let f_msg: Vec<EF> = (0..msg_len).map(|_| EF::from(rng.random::<F>())).collect();
        let f_rand: Vec<EF> = (0..t).map(|_| EF::from(rng.random::<F>())).collect();
        let w_vals: Vec<EF> = (0..msg_len).map(|_| EF::from(rng.random::<F>())).collect();
        let weights = Poly::new(w_vals.clone());
        let mu: EF = w_vals.iter().zip(f_msg.iter()).map(|(&w, &f)| w * f).sum();

        let cws: Vec<ConstraintWitness<EF>> = (0..n_constraints)
            .map(|_| {
                let msg: Vec<EF> = (0..ell_zk).map(|_| EF::from(rng.random::<F>())).collect();
                let rand: Vec<EF> = (0..t_zk).map(|_| EF::from(rng.random::<F>())).collect();
                let wi: Vec<EF> = (0..ell_zk).map(|_| EF::from(rng.random::<F>())).collect();
                let wo: Vec<EF> = (0..ell_zk).map(|_| EF::from(rng.random::<F>())).collect();
                let mu_i: EF = wo.iter().zip(msg.iter()).map(|(&w, &m)| w * m).sum();
                ConstraintWitness {
                    msg,
                    rand,
                    weights: wi,
                    output_weights: wo,
                    mu: mu_i,
                }
            })
            .collect();

        (f_msg, f_rand, weights, mu, cws)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_roundtrip(
        msg_len: usize,
        t: usize,
        m: usize,
        n_constraints: usize,
        ell_zk: usize,
        t_zk: usize,
        m_zk: usize,
        seed: u64,
    ) -> Result<(), &'static str> {
        let perm = test_perm();
        let mmcs = test_mmcs(&perm);
        let dft = Radix2DFTSmallBatch::default();
        let enc_c = ReedSolomonZkEncoding::<F, _>::new(t, msg_len, m, dft.clone());
        let enc_zk = ReedSolomonZkEncoding::<F, _>::new(t_zk, ell_zk, m_zk, dft);
        let pow_bits = 0;

        let mut rng = SmallRng::seed_from_u64(seed);
        let (f_msg, f_rand, weights, mu, cws) =
            make_test_data(msg_len, t, n_constraints, ell_zk, t_zk, &mut rng);

        // Compute aggregate mu including per-constraint contributions.
        let mut total_mu = mu;
        for cw in &cws {
            let c: EF = cw
                .weights
                .iter()
                .zip(cw.msg.iter())
                .map(|(&w, &m)| w * m)
                .sum();
            total_mu += c;
        }

        let mut prover_ch = MyChallenger::new(perm.clone());
        let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let (data, g_commit, s_commits, prover_gamma) = BaseCaseZk::prove::<F, EF, _, _, _, _, _, _>(
            &f_msg,
            &f_rand,
            &weights,
            total_mu,
            &cws,
            &mut prover_ch,
            pow_bits,
            &enc_c,
            &enc_zk,
            &mmcs,
            &mmcs,
            &mut prover_rng,
        );

        let cp: Vec<(Vec<EF>, Vec<EF>, EF)> = cws
            .iter()
            .map(|cw| (cw.weights.clone(), cw.output_weights.clone(), cw.mu))
            .collect();

        let mut verifier_ch = MyChallenger::new(perm);
        let verifier_gamma = BaseCaseZk::verify::<F, EF, ValMmcs, ValMmcs, _>(
            &data,
            &weights,
            total_mu,
            &cp,
            &g_commit,
            &s_commits,
            &mut verifier_ch,
            pow_bits,
        )
        .map_err(|_| "verifier rejected")?;

        if prover_gamma != verifier_gamma {
            return Err("gamma mismatch");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_simulator(
        msg_len: usize,
        t: usize,
        m: usize,
        n_constraints: usize,
        ell_zk: usize,
        t_zk: usize,
        m_zk: usize,
        seed: u64,
    ) -> Result<(), &'static str> {
        let perm = test_perm();
        let mmcs = test_mmcs(&perm);
        let dft = Radix2DFTSmallBatch::default();
        let enc_c = ReedSolomonZkEncoding::<F, _>::new(t, msg_len, m, dft.clone());
        let enc_zk = ReedSolomonZkEncoding::<F, _>::new(t_zk, ell_zk, m_zk, dft);
        let pow_bits = 0;

        let mut rng = SmallRng::seed_from_u64(seed);
        let (_, _, weights, mu, cws) =
            make_test_data(msg_len, t, n_constraints, ell_zk, t_zk, &mut rng);

        let mut total_mu = mu;
        for cw in &cws {
            let c: EF = cw
                .weights
                .iter()
                .zip(cw.msg.iter())
                .map(|(&w, &m)| w * m)
                .sum();
            total_mu += c;
        }

        let cp: Vec<(Vec<EF>, Vec<EF>, EF)> = cws
            .iter()
            .map(|cw| (cw.weights.clone(), cw.output_weights.clone(), cw.mu))
            .collect();

        let mut sim_ch = MyChallenger::new(perm.clone());
        let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (sim_data, sim_g, sim_s, _) = BaseCaseZk::simulate::<F, EF, _, _, _, _, _, _>(
            &weights,
            total_mu,
            &cp,
            &mut sim_ch,
            pow_bits,
            &enc_c,
            &enc_zk,
            &mmcs,
            &mmcs,
            &mut sim_rng,
        );

        let mut verifier_ch = MyChallenger::new(perm);
        let _gamma = BaseCaseZk::verify::<F, EF, ValMmcs, ValMmcs, _>(
            &sim_data,
            &weights,
            total_mu,
            &cp,
            &sim_g,
            &sim_s,
            &mut verifier_ch,
            pow_bits,
        )
        .map_err(|_| "verifier rejected simulator")?;
        Ok(())
    }

    /// Spot check validation: re-encode f*/r* and compare against g + γ·f.
    #[test]
    fn spot_check_validation() {
        let msg_len = 4;
        let t = 2;
        let m = 8;
        let dft = Radix2DFTSmallBatch::default();
        let enc = ReedSolomonZkEncoding::<F, _>::new(t, msg_len, m, dft);
        let mut rng = SmallRng::seed_from_u64(77);

        let f: Vec<F> = (0..msg_len).map(|_| rng.random()).collect();
        let rf: Vec<F> = (0..t).map(|_| rng.random()).collect();
        let g: Vec<F> = (0..msg_len).map(|_| rng.random()).collect();
        let rg: Vec<F> = (0..t).map(|_| rng.random()).collect();
        let gamma = F::from_u64(13);

        let f_star: Vec<F> = g
            .iter()
            .zip(f.iter())
            .map(|(&a, &b)| a + gamma * b)
            .collect();
        let r_star: Vec<F> = rg
            .iter()
            .zip(rf.iter())
            .map(|(&a, &b)| a + gamma * b)
            .collect();

        let enc_f_star = enc.encode_with_randomness(&f_star, &r_star);
        let enc_g = enc.encode_with_randomness(&g, &rg);
        let enc_f = enc.encode_with_randomness(&f, &rf);

        // Collect evals at all positions.
        let fs_evals: Vec<F> = (0..m)
            .map(|j| enc_f_star.row_slice(j).unwrap()[0])
            .collect();
        let g_evals: Vec<F> = (0..m).map(|j| enc_g.row_slice(j).unwrap()[0]).collect();
        let f_evals: Vec<F> = (0..m).map(|j| enc_f.row_slice(j).unwrap()[0]).collect();

        // Spot check should pass at every position.
        verify_c_spot_checks(&fs_evals, &g_evals, &f_evals, gamma).unwrap();

        // Tamper and verify rejection.
        let mut bad_fs = fs_evals;
        bad_fs[3] += F::ONE;
        assert!(verify_c_spot_checks(&bad_fs, &g_evals, &f_evals, gamma).is_err());
    }

    #[test]
    fn smoke_n0() {
        run_roundtrip(4, 2, 8, 0, 2, 1, 4, 42).unwrap();
    }

    #[test]
    fn smoke_n2() {
        run_roundtrip(4, 2, 8, 2, 3, 1, 4, 42).unwrap();
    }

    #[test]
    fn smoke_simulator_n0() {
        run_simulator(4, 2, 8, 0, 2, 1, 4, 42).unwrap();
    }

    #[test]
    fn smoke_simulator_n2() {
        run_simulator(4, 2, 8, 2, 3, 1, 4, 42).unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_completeness(
            k in 1usize..=3usize,
            t in 1usize..=3usize,
            n in 0usize..=3usize,
            seed in any::<u64>(),
        ) {
            let msg_len = 1 << k;
            let m = (msg_len + t).next_power_of_two();
            let ell_zk: usize = 2;
            let t_zk: usize = 1;
            let m_zk = (ell_zk + t_zk).next_power_of_two().max(4);
            prop_assert!(
                run_roundtrip(msg_len, t, m, n, ell_zk, t_zk, m_zk, seed).is_ok(),
                "k={k}, t={t}, n={n}, seed={seed}",
            );
        }

        #[test]
        fn prop_simulator(
            k in 1usize..=3usize,
            t in 1usize..=3usize,
            n in 0usize..=3usize,
            seed in any::<u64>(),
        ) {
            let msg_len = 1 << k;
            let m = (msg_len + t).next_power_of_two();
            let ell_zk: usize = 2;
            let t_zk: usize = 1;
            let m_zk = (ell_zk + t_zk).next_power_of_two().max(4);
            prop_assert!(
                run_simulator(msg_len, t, m, n, ell_zk, t_zk, m_zk, seed).is_ok(),
                "k={k}, t={t}, n={n}, seed={seed}",
            );
        }

        #[test]
        fn prop_tamper_f_star(
            k in 1usize..=3usize,
            t in 1usize..=3usize,
            n in 0usize..=2usize,
            seed in any::<u64>(),
            pos in 0usize..8usize,
        ) {
            let msg_len = 1 << k;
            let m = (msg_len + t).next_power_of_two();
            let ell_zk: usize = 2;
            let t_zk: usize = 1;
            let m_zk = (ell_zk + t_zk).next_power_of_two().max(4);
            let perm = test_perm();
            let mmcs = test_mmcs(&perm);
            let dft = Radix2DFTSmallBatch::default();
            let enc_c = ReedSolomonZkEncoding::<F, _>::new(t, msg_len, m, dft.clone());
            let enc_zk = ReedSolomonZkEncoding::<F, _>::new(t_zk, ell_zk, m_zk, dft);

            let mut rng = SmallRng::seed_from_u64(seed);
            let (f_msg, f_rand, weights, mu, cws) =
                make_test_data(msg_len, t, n, ell_zk, t_zk, &mut rng);
            let mut total_mu = mu;
            for cw in &cws {
                total_mu += cw.weights.iter().zip(cw.msg.iter()).map(|(&w, &m)| w * m).sum::<EF>();
            }

            let mut ch = MyChallenger::new(perm.clone());
            let mut prng = SmallRng::seed_from_u64(seed.wrapping_add(1));
            let (mut data, g, s, _) = BaseCaseZk::prove::<F, EF, _, _, _, _, _, _>(
                &f_msg, &f_rand, &weights, total_mu, &cws,
                &mut ch, 0, &enc_c, &enc_zk, &mmcs, &mmcs, &mut prng,
            );

            let idx = pos % data.f_star.len();
            data.f_star[idx] += EF::ONE;

            let cp: Vec<_> = cws.iter().map(|cw| (cw.weights.clone(), cw.output_weights.clone(), cw.mu)).collect();
            let mut vch = MyChallenger::new(perm);
            let result = BaseCaseZk::verify::<F, EF, ValMmcs, ValMmcs, _>(
                &data, &weights, total_mu, &cp, &g, &s, &mut vch, 0,
            );
            prop_assert!(result.is_err());
        }
    }
}
