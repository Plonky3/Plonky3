//! Shared Fiat-Shamir transcript operations for batch-STARK prover and verifier.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::{
    Challenges, Kind, Lookup, LookupProtocol, LookupTerminal, assert_uniform_tuple_width,
};

use crate::common::GlobalPreprocessed;
use crate::config::{Challenge, Commitment, StarkGenericConfig as SGC, Val};

/// Why a proof's lookup-challenge proof of work was rejected.
///
/// The three cases are distinguished because they mean different things: a bad
/// witness is a forgery or a difficulty mismatch, while a missing or unexpected
/// one means the proof disagrees with the batch about whether lookups exist at
/// all.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InvalidLookupPow {
    /// The witness does not satisfy the proof-of-work predicate.
    BadWitness,
    /// The batch has lookups but the proof carries no witness.
    MissingWitness,
    /// The batch has no lookups, so no challenge is sampled and no witness
    /// should be present.
    UnexpectedWitness,
}

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

    /// Whether any instance in the batch declares a lookup.
    ///
    /// A batch with none samples no lookup challenge, so it also grinds
    /// nothing and carries no witness.
    pub fn any_lookup<L>(all_lookups: &[L]) -> bool
    where
        L: AsRef<[Lookup<Val<SC>>]>,
    {
        all_lookups.iter().any(|c| !c.as_ref().is_empty())
    }

    /// Prover side: grind `pow_bits` before the lookup challenges, then sample
    /// them.
    ///
    /// The grind is sited after the main-trace commitment and every public
    /// value has been observed, so the witness commits the prover to the trace;
    /// a prover hunting for a favourable `(alpha, beta)` pays `2^pow_bits` per
    /// candidate. Returns the witness alongside the challenges, or `None` when
    /// the batch has no lookups and therefore no challenge to guard.
    pub fn grind_and_sample_perm_challenges<LG, L>(
        &mut self,
        all_lookups: &[L],
        lookup_gadget: &LG,
        pow_bits: usize,
    ) -> (Vec<Vec<SC::Challenge>>, Option<Val<SC>>)
    where
        LG: LookupProtocol,
        L: AsRef<[Lookup<Val<SC>>]>,
        SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    {
        if !Self::any_lookup(all_lookups) {
            return (all_lookups.iter().map(|_| Vec::new()).collect(), None);
        }
        let witness = self.challenger.grind(pow_bits);
        (
            self.sample_perm_challenges_inner(all_lookups, lookup_gadget),
            Some(witness),
        )
    }

    /// Verifier side: check the witness guarding the lookup challenges, then
    /// resample them.
    ///
    /// A batch without lookups must carry no witness, and one with lookups must
    /// carry a valid one; both mismatches are rejections rather than a silently
    /// unguarded sample.
    pub fn check_and_sample_perm_challenges<LG, L>(
        &mut self,
        all_lookups: &[L],
        lookup_gadget: &LG,
        pow_bits: usize,
        witness: Option<Val<SC>>,
    ) -> Result<Vec<Vec<SC::Challenge>>, InvalidLookupPow>
    where
        LG: LookupProtocol,
        L: AsRef<[Lookup<Val<SC>>]>,
        SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    {
        if !Self::any_lookup(all_lookups) {
            return match witness {
                None => Ok(all_lookups.iter().map(|_| Vec::new()).collect()),
                Some(_) => Err(InvalidLookupPow::UnexpectedWitness),
            };
        }
        let witness = witness.ok_or(InvalidLookupPow::MissingWitness)?;
        if !self.challenger.check_witness(pow_bits, witness) {
            return Err(InvalidLookupPow::BadWitness);
        }
        Ok(self.sample_perm_challenges_inner(all_lookups, lookup_gadget))
    }

    /// Sample the batch's lookup challenges and lay them out per instance,
    /// once the proof of work guarding them has been produced or checked.
    ///
    /// Split out from the two public entry points so the grind sits between
    /// "this batch has lookups" and the first squeeze, with prover and verifier
    /// sharing one body. The caller must have established that at least one
    /// instance has lookups; with none there is no squeeze at all.
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
    fn sample_perm_challenges_inner<LG, L>(
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

        debug_assert!(
            Self::any_lookup(all_lookups),
            "caller must check for lookups before sampling"
        );

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
        // Every tuple ever folded onto a given global bus must agree on width — see
        // `assert_uniform_tuple_width`'s doc for why a mismatch would let two
        // differently-shaped payloads fingerprint identically.
        let mut global_width: HashMap<&str, usize> = HashMap::new();
        let mut next_bus = 0usize;
        let mut max_message_width = 1usize;
        let bus_ids: Vec<Vec<usize>> = all_lookups
            .iter()
            .map(|contexts| {
                contexts
                    .as_ref()
                    .iter()
                    .map(|ctx| {
                        // A lookup's own tuples must already agree on width, and its
                        // payload width feeds the bus-offset power below.
                        let ctx_width = assert_uniform_tuple_width(&ctx.elements, "lookup");
                        max_message_width = max_message_width.max(ctx_width);

                        match &ctx.kind {
                            Kind::Global(name) => {
                                let id = *global_index.entry(name).or_insert_with(|| {
                                    let id = next_bus;
                                    next_bus += 1;
                                    id
                                });
                                let expected = *global_width.entry(name).or_insert(ctx_width);
                                assert_eq!(
                                    expected, ctx_width,
                                    "bus {name:?}: tuple widths {expected} and {ctx_width} \
                                     differ; every interaction sharing a bus must use the \
                                     same payload width, or a shorter tuple can alias a \
                                     longer one",
                                );
                                id
                            }
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

    /// Prover side: grind `pow_bits` before the out-of-domain point, then sample it.
    ///
    /// Every earlier commitment and lookup terminal is already in the transcript, so the witness
    /// commits the prover to the whole batch: a prover hunting for a favourable `zeta` pays
    /// `2^pow_bits` work per candidate.
    pub fn grind_and_sample_zeta(&mut self, pow_bits: usize) -> (Challenge<SC>, Val<SC>)
    where
        SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    {
        let witness = self.challenger.grind(pow_bits);
        (self.challenger.sample_algebra_element(), witness)
    }

    /// Verifier side: check the witness guarding the out-of-domain point, then resample it.
    ///
    /// `None` means the witness did not satisfy the proof-of-work predicate — either it was
    /// forged, or the prover and verifier disagree on
    /// [`p3_uni_stark::StarkGenericConfig::ood_proof_of_work_bits`].
    pub fn check_and_sample_zeta(
        &mut self,
        pow_bits: usize,
        witness: Val<SC>,
    ) -> Option<Challenge<SC>>
    where
        SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    {
        if !self.challenger.check_witness(pow_bits, witness) {
            return None;
        }
        Some(self.challenger.sample_algebra_element())
    }

    #[inline]
    fn observe_usize(&mut self, v: usize) {
        self.challenger
            .observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(v));
    }
}
