//! Checked players for the full HVZK run, including its terminal base case.

#[cfg(test)]
#[path = "player_tests.rs"]
mod tests;

use core::marker::PhantomData;

use p3_challenger::fs::{
    FieldToFieldCodec, ProverState, TranscriptBound, TranscriptError, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleUniformBits, GrindingChallenger};

use super::*;
use crate::transcript::Sumcheck;

/// Prover-side player. Dropping an unfinished run is a programming error.
pub struct ZkWhirProverTranscript<'a, C, F: PrimeField64, EF> {
    state: ProverState<&'a mut C, Alphabet<F>>,
    _ef: PhantomData<EF>,
}

/// Verifier-side player. Every rejecting path must explicitly abort the run.
pub struct ZkWhirVerifierTranscript<'a, C, F: PrimeField64, EF> {
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    _ef: PhantomData<EF>,
}

macro_rules! common {
    ($name:ident) => {
        impl<'a, C, F, EF> $name<'a, C, F, EF>
        where
            F: PrimeField64,
            EF: ExtensionField<F>,
            C: CanObserve<F>
                + CanSample<F>
                + CanSampleUniformBits<F>
                + GrindingChallenger<Witness = F>,
        {
            /// Start a full run, binding its trusted claim count and configuration.
            pub fn new(challenger: &'a mut C, shape: ZkWhirShape) -> Self {
                Self::from_separator(challenger, &shape.domain_separator::<F, EF>())
            }

            pub(crate) fn commitment<Com: Clone>(&mut self, label: &'static str, value: Com)
            where
                C: CanObserve<Com>,
            {
                self.state.observe_opaque(label, value);
            }

            pub(crate) fn observe(&mut self, label: &'static str, value: EF) {
                self.state
                    .observe_extension::<F, EF, FieldToFieldCodec<F>>(label, &value);
            }

            pub(crate) fn challenge(&mut self, label: &'static str) -> EF {
                self.state
                    .challenge_extension::<F, EF, FieldToFieldCodec<F>>(label)
                    .into_inner()
            }

            pub(crate) fn ood_point(&mut self, previous: &[EF]) -> EF {
                // The pad Vandermonde matrix must be invertible before an answer
                // is released. Rejected candidates consume sponge output but no
                // additional pattern step, identically on both sides.
                self.state
                    .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                        OOD_POINT,
                        1,
                        |point, _| !point.is_zero() && !previous.contains(point),
                    )
                    .pop()
                    .unwrap()
                    .into_inner()
            }

            pub(crate) fn indices(
                &mut self,
                label: &'static str,
                domain_size: usize,
                count: usize,
            ) -> Vec<usize> {
                if count >= domain_size {
                    return (0..domain_size).collect();
                }
                if count == 0 {
                    return Vec::new();
                }
                self.state
                    .challenge_uniform_bits::<F>(label, log2_strict_usize(domain_size), count)
                    .into_iter()
                    .map(TranscriptBound::into_inner)
                    .collect()
            }

            pub(crate) fn sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
                self.state.begin_protocol::<Sumcheck>("masked_sumcheck");
                let result = run(self.state.challenger_mut());
                self.state.end_protocol::<Sumcheck>("masked_sumcheck");
                result
            }
        }
    };
}
common!(ZkWhirProverTranscript);
common!(ZkWhirVerifierTranscript);

impl<'a, C, F, EF> ZkWhirProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn from_separator(
        challenger: &'a mut C,
        separator: &DomainSeparator<Alphabet<F>>,
    ) -> Self {
        Self {
            state: ProverState::new(challenger, separator),
            _ef: PhantomData,
        }
    }

    pub(crate) fn observe_slice(&mut self, label: &'static str, values: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(label, values);
    }

    pub(crate) fn pow(&mut self, label: &'static str, bits: usize) -> F {
        if bits == 0 {
            F::ZERO
        } else {
            self.state.observe_pow(label, bits)
        }
    }

    /// Require every step to have been played.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "HVZK carries its own proof values"
        );
    }
}

impl<'a, C, F, EF> ZkWhirVerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn from_separator(
        challenger: &'a mut C,
        separator: &DomainSeparator<Alphabet<F>>,
    ) -> Self {
        Self {
            state: VerifierState::new(challenger, separator, &[]),
            _ef: PhantomData,
        }
    }

    pub(crate) fn observe_slice(
        &mut self,
        label: &'static str,
        values: &[EF],
    ) -> Result<(), TranscriptError> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(label, values)
            .map(|_| ())
    }

    pub(crate) fn pow(&mut self, label: &'static str, bits: usize, witness: F) -> bool {
        if bits == 0 {
            witness == F::ZERO
        } else {
            self.state.observe_pow(label, bits, witness).is_ok()
        }
    }

    /// Disarm completeness checking only when rejecting a proof.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Require every step to have been replayed.
    pub fn finish(self) {
        self.state.finalize().expect("HVZK reads an empty wire");
    }
}
