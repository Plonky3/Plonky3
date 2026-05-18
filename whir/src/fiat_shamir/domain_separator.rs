//! Domain separator construction for the WHIR Fiat-Shamir transcript.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::fiat_shamir::pattern::{Hint, Observe, Pattern, Sample};
use crate::parameters::{FoldingFactor, WhirConfig};

/// Configuration for a sumcheck phase in the protocol.
#[derive(Debug)]
pub(crate) struct SumcheckParams {
    /// Number of sumcheck rounds.
    ///
    /// Each round corresponds to one prover polynomial and one verifier challenge.
    pub rounds: usize,

    /// Proof-of-work difficulty in bits.
    ///
    /// - Zero disables PoW.
    /// - Positive values insert a grinding step after each round.
    pub pow_bits: usize,
}

/// Encodes the structure of an interactive protocol as a sequence of field elements.
///
/// # Overview
///
/// Before any protocol execution, both prover and verifier build identical
/// domain separators that describe every transcript operation (observe, sample,
/// hint, PoW) in the exact order they occur. This sequence is absorbed into
/// the challenger at the start, binding the sponge state to the protocol
/// structure and preventing cross-protocol attacks.
///
/// # Transcript Operation Encoding
///
/// Each transcript step is encoded as a single field element:
///
/// ```text
///     element = pattern_tag + sub_label + count
/// ```
///
/// where:
/// - `pattern_tag` distinguishes observe / sample / hint.
/// - `sub_label` identifies the semantic role (e.g., Merkle digest, folding randomness).
/// - `count` is the number of field elements involved (omitted for hints).
///
/// # Evaluation Claims Are Not Absorbed Here
///
/// The WHIR protocol takes a constrained Reed-Solomon code as a public input.
/// When used as a polynomial commitment scheme, the evaluation claim
/// (point and value) is encoded into that constraint.
///
/// This separator only encodes the **internal** transcript structure:
/// - Merkle commitments
/// - Sumcheck polynomials
/// - Out-of-domain samples
///
/// It does **not** absorb the evaluation point or claimed value,
/// because those are external inputs to the protocol.
///
/// For Fiat-Shamir soundness (BCS transformation), the caller **must**
/// absorb the evaluation point and claimed value into the challenger
/// before any WHIR challenges are derived.
/// The PCS layer is typically responsible for this.
/// Omitting this step allows proof replay across different claims.
///
/// # Protocol Structure
///
/// The full WHIR proof transcript, as encoded by this separator, follows
/// this order (matching Construction 5.1 of the WHIR paper):
#[derive(Clone, Debug)]
pub struct DomainSeparator<EF, F> {
    /// Field-element encoding of the protocol transcript pattern.
    pattern: Vec<F>,

    /// Phantom marker for the extension field type.
    _extension_field: PhantomData<EF>,
}

impl<EF, F> DomainSeparator<EF, F>
where
    EF: ExtensionField<F>,
    F: Field,
{
    /// Create a domain separator from an existing pattern vector.
    #[must_use]
    pub const fn new(pattern: Vec<F>) -> Self {
        Self {
            pattern,
            _extension_field: PhantomData,
        }
    }

    /// Record that the prover observes `count` field elements into the sponge.
    pub fn observe(&mut self, count: usize, pattern: Observe) {
        self.pattern.push(
            pattern.as_field_element::<F>()
                + F::from_usize(count)
                + Pattern::Observe.as_field_element::<F>(),
        );
    }

    /// Record that the verifier samples `count` field elements from the sponge.
    pub fn sample(&mut self, count: usize, pattern: Sample) {
        self.pattern.push(
            pattern.as_field_element::<F>()
                + F::from_usize(count)
                + Pattern::Sample.as_field_element::<F>(),
        );
    }

    /// Encode a public protocol parameter into the domain separator.
    ///
    /// Pushes two field elements:
    /// 1. A constant marker identifying this entry as a protocol parameter.
    /// 2. The raw parameter value.
    ///
    /// This binds the Fiat-Shamir transcript to the specific protocol
    /// configuration, preventing cross-protocol transcript reuse.
    fn protocol_param(&mut self, value: usize) {
        // Constant marker: observe tag + protocol-param sub-label.
        self.pattern.push(
            Observe::ProtocolParam.as_field_element::<F>()
                + Pattern::Observe.as_field_element::<F>(),
        );
        // Raw parameter value.
        self.pattern.push(F::from_usize(value));
    }

    /// Record a non-binding hint from the prover.
    pub fn hint(&mut self, pattern: Hint) {
        self.pattern
            .push(pattern.as_field_element::<F>() + Pattern::Hint.as_field_element::<F>());
    }

    /// Absorb the entire domain separator pattern into the challenger.
    ///
    /// Must be called before any protocol-specific transcript operations
    /// so the sponge state is bound to the protocol structure.
    pub fn observe_domain_separator<Challenger>(&self, challenger: &mut Challenger)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        challenger.observe_slice(&self.pattern);
    }

    /// Append an out-of-domain (OOD) sampling step.
    ///
    /// Encodes sampling `num_samples` OOD evaluation points followed by
    /// observing their answers. Skipped when `num_samples` is zero.
    pub fn add_ood(&mut self, num_samples: usize) {
        if num_samples > 0 {
            self.sample(num_samples, Sample::OodQuery);
            self.observe(num_samples, Observe::OodAnswers);
        }
    }

    /// Append the commitment phase of the protocol.
    ///
    /// # Algorithm
    ///
    /// 1. Encode public protocol parameters that uniquely identify this
    ///    protocol instance. This prevents an adversary from replaying a
    ///    proof generated for one parameter set against a verifier
    ///    configured with different parameters.
    /// 2. Observe the Merkle root of the committed polynomial.
    /// 3. Optionally, encode an OOD sampling step.
    ///
    /// # Safety
    ///
    /// Does **not** absorb the evaluation point or claimed value.
    /// The caller must observe these public inputs into the challenger
    /// before any challenges are sampled.
    /// See the struct-level documentation for the rationale.
    pub fn commit_statement<Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        config: &WhirConfig<EF, F, Challenger>,
    ) {
        // Bind the transcript to the protocol configuration.
        self.protocol_param(config.num_variables);
        self.protocol_param(config.security_level);
        self.protocol_param(config.starting_log_inv_rate);
        self.protocol_param(config.pow_bits);

        // Encode the soundness assumption as its discriminant.
        self.protocol_param(config.soundness_type as usize);

        // Encode the folding strategy: discriminant followed by inner values.
        match config.folding_factor {
            FoldingFactor::Constant(f) => {
                self.protocol_param(0);
                self.protocol_param(f);
            }
            FoldingFactor::ConstantFromSecondRound(first, rest) => {
                self.protocol_param(1);
                self.protocol_param(first);
                self.protocol_param(rest);
            }
        }

        self.observe(DIGEST_ELEMS, Observe::MerkleDigest);
        self.add_ood(config.commitment_ood_samples);
    }

    /// Append the full WHIR proof transcript to the domain separator.
    ///
    /// # Safety
    ///
    /// Does **not** absorb the evaluation point or claimed value.
    /// The caller must observe these public inputs into the challenger
    /// before any challenges are sampled.
    /// See the struct-level documentation for the rationale.
    ///
    /// # Algorithm
    ///
    /// 1. Sample initial combination randomness and run the first sumcheck.
    /// 2. For each intermediate round:
    ///    - Observe the new Merkle commitment and optional OOD answers.
    ///    - Perform PoW (before queries, per the WHIR security argument).
    ///    - Draw a transcript checkpoint, then STIR query positions.
    ///    - Record hints for query data and Merkle proofs.
    ///    - Sample combination randomness and run the next sumcheck.
    /// 3. For the final round:
    ///    - Observe the final polynomial coefficients.
    ///    - Perform PoW, then draw final query positions.
    ///    - Record hints and run the final sumcheck.
    ///    - Record deferred weight evaluation hints.
    pub fn add_whir_proof<Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        config: &WhirConfig<EF, F, Challenger>,
    ) where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        EF: TwoAdicField,
        F: TwoAdicField,
    {
        // Initial combination randomness and first sumcheck phase.
        self.sample(1, Sample::InitialCombinationRandomness);
        self.add_sumcheck(&SumcheckParams {
            rounds: config.folding_factor.at_round(0),
            pow_bits: config.starting_folding_pow_bits,
        });

        // Intermediate rounds: commitment → OOD → PoW → checkpoint → queries → sumcheck.
        let mut domain_size = config.starting_domain_size();
        for (round, r) in config.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> config.folding_factor.at_round(round);
            // Byte length needed to encode a position in the folded domain.
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

            // Observe the new Merkle root and optional OOD evaluations.
            self.observe(DIGEST_ELEMS, Observe::MerkleDigest);
            self.add_ood(r.ood_samples);

            // PoW must precede query generation to prevent commitment shopping.
            self.pow(r.pow_bits);

            // Transcript checkpoint: a dummy sample that synchronizes the
            // domain separator with the prover/verifier `challenger.sample()` call
            // that occurs between PoW and query generation.
            self.sample(1, Sample::TranscriptCheckpoint);

            // Draw STIR query positions and provide opening data.
            self.sample(r.num_queries * domain_size_bytes, Sample::StirQueries);
            self.hint(Hint::StirQueries);
            self.hint(Hint::MerkleProof);

            // Combination randomness for the next polynomial, then sumcheck.
            self.sample(1, Sample::CombinationRandomness);

            self.add_sumcheck(&SumcheckParams {
                rounds: config.folding_factor.at_round(round + 1),
                pow_bits: r.folding_pow_bits,
            });
            domain_size >>= config.rs_reduction_factor(round);
        }

        // Final round: coefficients → PoW → queries → sumcheck → deferred hints.
        let folded_domain_size = domain_size
            >> config
                .folding_factor
                .at_round(config.round_parameters.len());
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        // Observe all coefficients of the final folded polynomial.
        self.observe(1 << config.final_sumcheck_rounds, Observe::FinalCoeffs);

        // PoW before final query generation (no transcript checkpoint in final round).
        self.pow(config.final_pow_bits);
        self.sample(
            domain_size_bytes * config.final_queries,
            Sample::FinalQueries,
        );
        self.hint(Hint::StirAnswers);
        self.hint(Hint::MerkleProof);

        // Final sumcheck and deferred weight evaluations.
        self.add_sumcheck(&SumcheckParams {
            rounds: config.final_sumcheck_rounds,
            pow_bits: config.final_folding_pow_bits,
        });
        self.hint(Hint::DeferredWeightEvaluations);
    }

    /// Append a sumcheck sub-protocol to the domain separator.
    ///
    /// # Algorithm
    ///
    /// For each round:
    /// 1. Observe 2 coefficients of the degree-2 round polynomial (c_0 and c_2).
    ///    The third coefficient c_1 = claimed_sum - c_0 is derived by the verifier.
    /// 2. Sample one folding randomness challenge.
    /// 3. Optionally perform a PoW step.
    pub(crate) fn add_sumcheck(&mut self, params: &SumcheckParams) {
        let SumcheckParams { rounds, pow_bits } = *params;

        for _ in 0..rounds {
            // Absorb c_0 and c_2; the verifier reconstructs c_1.
            self.observe(2, Observe::SumcheckPoly);
            // Verifier draws the folding challenge for this variable.
            self.sample(1, Sample::FoldingRandomness);
            // Optional grinding step after each sumcheck round.
            self.pow(pow_bits);
        }
    }

    /// Optionally append a proof-of-work challenge.
    ///
    /// When `bits` is positive, encodes:
    /// 1. Sampling a 32-byte challenge from the sponge.
    /// 2. Observing an 8-byte nonce that satisfies the grinding condition.
    ///
    /// When `bits` is zero, nothing is appended.
    pub fn pow(&mut self, bits: usize) {
        if bits > 0 {
            // Sample a 32-byte PoW challenge preimage.
            self.sample(32, Sample::PowQueries);
            // Observe the nonce that solves the challenge.
            self.observe(8, Observe::PowNonce);
        }
    }
}
