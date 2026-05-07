//! Shared Fiat-Shamir transcript operations for batch-STARK prover and verifier.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::{Kind, Lookup, LookupData, LookupProtocol};

use crate::common::GlobalPreprocessed;
use crate::config::{Challenge, Commitment, StarkGenericConfig as SGC, Val};

/// Wrapper around a Fiat-Shamir challenger.
pub struct BatchTranscript<SC: SGC> {
    /// The underlying challenger.
    pub challenger: SC::Challenger,
}

impl<SC: SGC> BatchTranscript<SC> {
    /// Create a new transcript from a fresh challenger.
    pub const fn new(challenger: SC::Challenger) -> Self {
        Self { challenger }
    }

    /// Bind the instance count so the transcript cannot be reinterpreted
    /// with a different partitioning of opened values.
    pub fn observe_instance_count(&mut self, n: usize) {
        self.observe_usize(n);
    }

    /// Observe a single instance's structural binding data.
    pub fn observe_instance_binding(
        &mut self,
        log_ext_degree: usize,
        log_degree: usize,
        width: usize,
        num_quotient_chunks: usize,
    ) {
        self.observe_usize(log_ext_degree);
        self.observe_usize(log_degree);
        self.observe_usize(width);
        self.observe_usize(num_quotient_chunks);
    }

    /// Observe the main trace commitment and per-instance public values.
    pub fn observe_main<PV>(&mut self, main_commitment: &Commitment<SC>, public_values: &[PV])
    where
        PV: AsRef<[Val<SC>]>,
    {
        self.challenger.observe(main_commitment.clone());
        for pv in public_values {
            self.challenger.observe_slice(pv.as_ref());
        }
    }

    /// Observe preprocessed column widths and the optional global preprocessed commitment.
    pub fn observe_preprocessed(
        &mut self,
        preprocessed_widths: &[usize],
        preprocessed: Option<&GlobalPreprocessed<SC>>,
    ) {
        for &w in preprocessed_widths {
            self.observe_usize(w);
        }
        if let Some(global) = preprocessed {
            self.challenger.observe(global.commitment.clone());
        }
    }

    /// Sample per-instance permutation challenges.
    ///
    /// Global lookups with the same name share the same challenges; local
    /// lookups get fresh independent challenges.
    pub fn sample_perm_challenges<LG, L>(
        &mut self,
        all_lookups: &[L],
        lookup_gadget: &LG,
    ) -> Vec<Vec<SC::Challenge>>
    where
        LG: LookupProtocol,
        L: AsRef<[Lookup<Val<SC>>]>,
    {
        let n = lookup_gadget.num_challenges();
        let mut global = HashMap::new();

        all_lookups
            .iter()
            .map(|contexts| {
                contexts
                    .as_ref()
                    .iter()
                    .flat_map(|ctx| match &ctx.kind {
                        Kind::Global(name) => global
                            .entry(name)
                            .or_insert_with(|| self.sample_n_challenges(n))
                            .clone(),
                        Kind::Local => self.sample_n_challenges(n),
                    })
                    .collect()
            })
            .collect()
    }

    /// Observe the permutation commitment and cumulated lookup data, then
    /// sample the constraint-folding challenge alpha.
    pub fn observe_perm_and_sample_alpha(
        &mut self,
        perm_commitment: Option<&Commitment<SC>>,
        lookup_data: &[Vec<LookupData<Challenge<SC>>>],
    ) -> Challenge<SC> {
        if let Some(commit) = perm_commitment {
            self.challenger.observe(commit.clone());
            // Observe cumulated lookup sums so the verifier can check them.
            for data in lookup_data.iter().flatten() {
                self.challenger.observe_algebra_element(data.cumulative_sum);
            }
        }
        self.challenger.sample_algebra_element()
    }

    /// Observe the quotient chunks commitment.
    pub fn observe_quotient_commitment(&mut self, commitment: &Commitment<SC>) {
        self.challenger.observe(commitment.clone());
    }

    /// Observe the optional ZK randomization commitment.
    pub fn observe_random_commitment(&mut self, commitment: &Commitment<SC>) {
        self.challenger.observe(commitment.clone());
    }

    /// Sample the out-of-domain evaluation point zeta.
    pub fn sample_zeta(&mut self) -> Challenge<SC> {
        self.challenger.sample_algebra_element()
    }

    #[inline]
    fn observe_usize(&mut self, v: usize) {
        self.challenger
            .observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(v));
    }

    fn sample_n_challenges(&mut self, n: usize) -> Vec<SC::Challenge> {
        (0..n)
            .map(|_| self.challenger.sample_algebra_element())
            .collect()
    }
}
