//! Challenge target structures for STARK verification circuits.

use alloc::vec;
use alloc::vec::Vec;

use p3_circuit::CircuitBuilder;
use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::Target;
use crate::traits::{Recursive, RecursiveChallenger};
use crate::types::ProofTargets;
use crate::verifier::ObservableCommitment;

/// Base STARK challenges (independent of PCS choice).
#[derive(Debug, Clone)]
pub struct StarkChallenges {
    /// Alpha: challenge for folding all constraint polynomials
    pub alpha: Target,
    /// Zeta: out-of-domain evaluation point
    pub zeta: Target,
    /// Zeta next: evaluation point for next row (zeta * g in the trace domain)
    pub zeta_next: Target,
}

/// Parameters for STARK challenge allocation that match native challenger behavior.
pub(crate) struct StarkChallengeParams<'a, SC: StarkGenericConfig, Comm> {
    /// Log₂ of trace domain size
    pub degree_bits: usize,
    /// is_zk flag (0 or 1)
    pub is_zk: usize,
    /// Width of preprocessed trace (0 if none)
    pub preprocessed_width: usize,
    /// Preprocessed commitment targets (if preprocessed_width > 0)
    pub preprocessed_commit: &'a Option<Comm>,
    /// Generator of the init trace domain (for computing zeta_next = zeta * generator)
    pub trace_domain_generator: SC::Challenge,
}

impl StarkChallenges {
    /// Allocate base STARK challenge targets using Fiat-Shamir transform.
    ///
    /// This method follows the exact native DuplexChallenger protocol ordering:
    /// 1. Observe degree_bits
    /// 2. Observe degree_bits - is_zk (init trace domain log size)
    /// 3. Observe preprocessed_width
    /// 4. Observe trace commitment
    /// 5. If preprocessed_width > 0: observe preprocessed commitment
    /// 6. Observe public values
    /// 7. Sample alpha
    /// 8. Observe quotient commitment
    /// 9. If ZK mode: observe random commitment
    /// 10. Sample zeta
    /// 11. Compute zeta_next = zeta * trace_domain_generator (NOT sampled!)
    ///
    /// The challenger state is mutated and can be used for further PCS challenge sampling.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder
    /// - `challenger`: Fiat-Shamir challenger (will be mutated)
    /// - `proof_targets`: Proof structure with commitments
    /// - `public_values`: AIR public input values
    /// - `params`: Challenge parameters matching native behavior
    ///
    /// # Returns
    /// The three base STARK challenges
    pub(crate) fn allocate<SC, Comm, OpeningProof>(
        circuit: &mut CircuitBuilder<SC::Challenge>,
        challenger: &mut impl RecursiveChallenger<Val<SC>, SC::Challenge>,
        proof_targets: &ProofTargets<SC, Comm, OpeningProof>,
        public_values: &[Target],
        params: &StarkChallengeParams<'_, SC, Comm>,
    ) -> Self
    where
        SC: StarkGenericConfig,
        Val<SC>: PrimeField64,
        SC::Challenge: ExtensionField<Val<SC>> + PrimeCharacteristicRing,
        Comm: Recursive<SC::Challenge> + ObservableCommitment,
        OpeningProof: Recursive<SC::Challenge>,
    {
        // Extract commitment targets from proof
        let trace_comm_targets = proof_targets
            .commitments_targets
            .trace_targets
            .to_observation_targets();
        let quotient_comm_targets = proof_targets
            .commitments_targets
            .quotient_chunks_targets
            .to_observation_targets();
        let random_comm_targets = proof_targets
            .commitments_targets
            .random_commit
            .as_ref()
            .map(|c| c.to_observation_targets());
        let preprocessed_comm_targets = params
            .preprocessed_commit
            .as_ref()
            .map(|c| c.to_observation_targets());

        // 1. Observe degree_bits (base field element)
        let degree_bits_target =
            circuit.alloc_const(SC::Challenge::from_usize(params.degree_bits), "degree bits");
        challenger.observe(circuit, degree_bits_target);

        // 2. Observe degree_bits - is_zk (init trace domain log size, base field element)
        let init_trace_log_size = params.degree_bits.saturating_sub(params.is_zk);
        let init_trace_log_size_target = circuit.alloc_const(
            SC::Challenge::from_usize(init_trace_log_size),
            "init trace log size",
        );
        challenger.observe(circuit, init_trace_log_size_target);

        // 3. Observe preprocessed_width (base field element)
        let preprocessed_width_target = circuit.alloc_const(
            SC::Challenge::from_usize(params.preprocessed_width),
            "preprocessed width",
        );
        challenger.observe(circuit, preprocessed_width_target);

        // 4. Observe trace commitment (base field elements)
        challenger.observe_slice(circuit, &trace_comm_targets);

        // 5. If preprocessed_width > 0: observe preprocessed commitment
        if params.preprocessed_width > 0
            && let Some(prep_comm) = &preprocessed_comm_targets
        {
            challenger.observe_slice(circuit, prep_comm);
        }

        // 6. Observe public values (base field elements)
        challenger.observe_slice(circuit, public_values);

        // 7. Sample alpha challenge (extension field element)
        let alpha = challenger.sample_ext(circuit);

        // 8. Observe quotient chunks commitment (base field elements)
        challenger.observe_slice(circuit, &quotient_comm_targets);

        // 9. Observe random commitment if in ZK mode
        if let Some(random_comm) = random_comm_targets {
            challenger.observe_slice(circuit, &random_comm);
        }

        // 10. Sample zeta (extension field element)
        let zeta = challenger.sample_ext(circuit);

        // 11. Compute zeta_next = zeta * trace_domain_generator (NOT sampled!)
        // This matches native behavior: zeta_next = init_trace_domain.next_point(zeta)
        let generator_const = circuit.define_const(params.trace_domain_generator);
        let zeta_next = circuit.mul(zeta, generator_const);

        Self {
            alpha,
            zeta,
            zeta_next,
        }
    }

    /// Convert to flat vector: [alpha, zeta, zeta_next]
    pub fn to_vec(&self) -> Vec<Target> {
        vec![self.alpha, self.zeta, self.zeta_next]
    }

    /// Get the alpha challenge (for constraint folding).
    pub const fn alpha(&self) -> Target {
        self.alpha
    }

    /// Get the zeta challenge (OOD evaluation point).
    pub const fn zeta(&self) -> Target {
        self.zeta
    }

    /// Get the zeta_next challenge (next row evaluation point).
    pub const fn zeta_next(&self) -> Target {
        self.zeta_next
    }
}

#[cfg(test)]
mod tests {
    use p3_circuit::ExprId;

    use super::*;

    #[test]
    fn test_stark_challenges_to_vec() {
        let alpha = ExprId(1);
        let zeta = ExprId(2);
        let zeta_next = ExprId(3);
        let challenges = StarkChallenges {
            alpha,
            zeta,
            zeta_next,
        };

        let vec = challenges.to_vec();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], alpha);
        assert_eq!(vec[1], zeta);
        assert_eq!(vec[2], zeta_next);
    }
}
