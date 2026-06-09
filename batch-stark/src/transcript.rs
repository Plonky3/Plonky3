//! Shared Fiat-Shamir transcript operations for batch-STARK prover and verifier.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::{Challenges, Kind, Lookup, LookupProtocol, LookupTerminal};

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

    /// Sample the batch's lookup challenges and lay them out per instance.
    ///
    /// # Overview
    ///
    /// - One pair is drawn for the whole batch, not one per bus.
    /// - Each bus is separated by an additive offset from that pair.
    /// - Local lookups get a unique bus, so they balance on their own.
    /// - Global lookups sharing a name get one bus, so sends and receives cancel.
    ///
    /// # Soundness
    ///
    /// - The pair is sampled after the main commitment, so the trace cannot adapt to it.
    /// - Distinct buses occupy distinct cosets, so any imbalance survives with overwhelming probability.
    ///
    /// # Returns
    ///
    /// - One challenge vector per instance.
    /// - Each lookup contributes a pair: its bus offset, then the shared combiner.
    /// - The gadget reads that pair exactly as it read its former per-lookup pair.
    pub fn sample_perm_challenges<LG, L>(
        &mut self,
        all_lookups: &[L],
        lookup_gadget: &LG,
    ) -> Vec<Vec<SC::Challenge>>
    where
        LG: LookupProtocol,
        L: AsRef<[Lookup<Val<SC>>]>,
    {
        // The gadget reads two challenges per lookup: a denominator base and a combiner.
        // The single-pair scheme below relies on exactly that width.
        assert_eq!(
            lookup_gadget.num_challenges(),
            2,
            "single-pair domain separation expects a two-challenge gadget"
        );

        // No lookups means no squeeze, matching a batch that never had lookups.
        let any_lookup = all_lookups.iter().any(|c| !c.as_ref().is_empty());
        if !any_lookup {
            return all_lookups.iter().map(|_| Vec::new()).collect();
        }

        // Draw the single (alpha, beta) pair for the whole batch.
        // This is the only lookup squeeze: two draws, not two per bus.
        let alpha: SC::Challenge = self.challenger.sample_algebra_element();
        let beta: SC::Challenge = self.challenger.sample_algebra_element();

        // Assign each bus a global index and measure the widest payload.
        //
        // - Global buses share an index by name, so cross-instance messages cancel.
        // - Local buses take a fresh index each, so nothing else can cancel them.
        // - The widest payload fixes where the bus offset sits, one power above it.
        let mut global_index: HashMap<&str, usize> = HashMap::new();
        let mut next_bus = 0usize;
        let mut max_message_width = 1usize;
        let bus_ids: Vec<Vec<usize>> = all_lookups
            .iter()
            .map(|contexts| {
                contexts
                    .as_ref()
                    .iter()
                    .map(|ctx| {
                        // A lookup's payload width is the largest tuple it carries.
                        for tuple in &ctx.elements {
                            max_message_width = max_message_width.max(tuple.len());
                        }
                        match &ctx.kind {
                            Kind::Global(name) => *global_index.entry(name).or_insert_with(|| {
                                let id = next_bus;
                                next_bus += 1;
                                id
                            }),
                            Kind::Local => {
                                let id = next_bus;
                                next_bus += 1;
                                id
                            }
                        }
                    })
                    .collect()
            })
            .collect();

        // Precompute every bus offset once from the sampled pair.
        let challenges = Challenges::new(alpha, beta, max_message_width, next_bus);

        // Lay the challenges out per instance, one pair per lookup.
        //
        //     [ prefix[bus_0], beta, prefix[bus_1], beta, ... ]
        //
        // The gadget computes `base - combined`.
        // Passing `prefix[bus]` as the base yields the domain-separated denominator.
        bus_ids
            .iter()
            .map(|instance_buses| {
                instance_buses
                    .iter()
                    .flat_map(|&bus| [challenges.bus_prefix[bus], beta])
                    .collect()
            })
            .collect()
    }

    /// Observe the permutation commitment and per-AIR lookup terminals,
    /// then sample the constraint-folding challenge alpha.
    pub fn observe_perm_and_sample_alpha(
        &mut self,
        perm_commitment: Option<&Commitment<SC>>,
        lookup_terminals: &[Option<LookupTerminal<Challenge<SC>>>],
    ) -> Challenge<SC> {
        if let Some(commit) = perm_commitment {
            self.challenger.observe(commit.clone());
            // Observe per-AIR lookup terminals so the verifier can check the cross-AIR sum.
            for terminal in lookup_terminals.iter().flatten() {
                self.challenger.observe_algebra_element(terminal.0);
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
}
