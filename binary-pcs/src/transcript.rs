//! Fiat-Shamir transcript of one binary-tower PCS run.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleUniformBits, GrindingChallenger};
use p3_field::{AlgebraIdentity, Field};
use p3_util::log2_strict_usize;

use crate::params::BinaryPcsConfig;
use crate::verifier::num_distinct_queries;

/// Version byte bound into the transcript seed.
///
/// Bumping it separates two revisions of this protocol.
///
/// It separates them even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-binary-pcs";

/// Step label of one folded oracle's Merkle root.
const ORACLE_COMMITMENT: &str = "oracle_commitment";

/// Step label of the final codeword, sent in the clear.
const FINAL_CODEWORD: &str = "final_codeword";

/// Step label of the grinding step guarding the query positions.
const QUERY_POW: &str = "query_pow";

/// Step label of the query positions.
const QUERY_INDICES: &str = "query_indices";

/// Sponge alphabet of a challenger that speaks the committed tower level natively.
type Alphabet<F> = FieldUnit<F>;

/// Numbers that fix the transcript of one binary-tower PCS run.
///
/// Both sides derive this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsShape {
    /// Arity of the committed polynomial.
    pub num_variables: usize,
    /// Variables each fold batch consumes.
    pub log_folding_factor: usize,
    /// Folded oracles committed between the fold batches.
    ///
    /// The last batch sends its codeword in the clear, so it commits nothing.
    pub num_oracles: usize,
    /// Symbols the final codeword carries.
    pub final_codeword_len: usize,
    /// Grinding difficulty guarding the query positions.
    pub pow_bits: usize,
    /// Bit width of one sampled fold pair.
    pub pair_bits: usize,
    /// Distinct pairs the run opens.
    pub num_pairs: usize,
}

impl BinaryPcsShape {
    /// Derive the shape of one run from its configuration.
    ///
    /// # Arguments
    ///
    /// - `config`: the parameter set both sides agreed on.
    ///
    /// # Panics
    ///
    /// When the base domain does not hold a whole number of fold pairs.
    #[must_use]
    pub const fn new(config: &BinaryPcsConfig) -> Self {
        // Queries index pairs, so the sampled domain is one bit narrower.
        let shift = config.log_folding_factor() - 1;
        let pair_domain = config.domain_size() >> shift;

        Self {
            num_variables: config.num_variables(),
            log_folding_factor: config.log_folding_factor(),
            // Every batch but the last commits its folded oracle.
            num_oracles: config.num_fold_batches() - 1,
            final_codeword_len: 1 << config.log_final_len(),
            pow_bits: config.pow_bits(),
            pair_bits: log2_strict_usize(pair_domain / 2),
            // A run asking for more pairs than the domain holds opens every pair.
            num_pairs: num_distinct_queries(pair_domain, config.num_queries()),
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// The fold rounds are absent: each seeds its own sumcheck sub-transcript.
    ///
    /// ```text
    ///     per oracle  ->  one Merkle root
    ///     closing     ->  final codeword, grind, query positions
    /// ```
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A flat sequence of leaf steps always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut steps = Vec::with_capacity(self.num_oracles + 3);

        // One root per folded oracle, so the oracle count is the step count.
        for _ in 0..self.num_oracles {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                ORACLE_COMMITMENT,
                Length::Scalar,
            ));
        }

        // The last word is uncommitted, so every symbol of it is bound.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            FINAL_CODEWORD,
            Length::Fixed(self.final_codeword_len),
        ));

        // A zero difficulty describes no work, so it contributes no step.
        if self.pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                QUERY_POW,
                Length::Fixed(self.pow_bits),
            ));
        }

        // The positions are drawn distinct, so the count is what is kept.
        steps.push(Interaction::uniform_bits(
            Hierarchy::Atomic,
            Kind::Challenge,
            QUERY_INDICES,
            self.pair_bits,
            Length::Fixed(self.num_pairs),
        ));

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity and this shape into a seed.
    ///
    /// # Soundness
    ///
    /// The step counts and widths ride the fingerprint on their own.
    ///
    /// The arity and the folding factor move no step, so they are bound as chunks.
    ///
    /// Without them, a run folding every variable at once shares a description.
    ///
    /// Every other arity at the same rate lands on it.
    ///
    /// ```text
    ///     nv = 8, k = 8   ->  one batch, no oracle, rate-wide final word
    ///     nv = 9, k = 9   ->  the same three numbers
    /// ```
    ///
    /// The surrounding layout binds the arity too.
    ///
    /// This makes the seed self-contained rather than closing a reachable gap.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        separator
            .instance(&(self.num_variables as u64).to_be_bytes())
            .instance(&(self.log_folding_factor as u64).to_be_bytes());

        separator
    }
}

/// Prover-side transcript of one binary-tower PCS run.
///
/// The challenger is borrowed, not consumed.
///
/// The run sits inside a larger protocol, whose transcript continues where this one stops.
pub struct BinaryPcsProverTranscript<'a, F: TranscriptField, EF, C> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: BinaryPcsShape,
    /// Marker for the field the folded codewords live in.
    _challenge: PhantomData<EF>,
}

impl<'a, F, EF, C> BinaryPcsProverTranscript<'a, F, EF, C>
where
    F: TranscriptField,
    EF: Field + AlgebraIdentity<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this run.
    /// - `shape`: the numbers that fix this run's transcript.
    pub fn new(challenger: &'a mut C, shape: BinaryPcsShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _challenge: PhantomData,
        }
    }

    /// Lend the sponge to the delegated sumcheck rounds of one fold batch.
    ///
    /// A sumcheck round seeds a sub-transcript of its own, so no step of it belongs here.
    pub fn fold_batch<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        run(self.state.challenger_mut())
    }

    /// Bind one folded oracle's Merkle root.
    pub fn oracle_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(ORACLE_COMMITMENT, commitment);
    }

    /// Bind the final codeword, which travels in the clear.
    ///
    /// # Panics
    ///
    /// When the codeword is not the length the run was described with.
    pub fn final_codeword(&mut self, codeword: &[EF]) {
        assert_eq!(
            codeword.len(),
            self.shape.final_codeword_len,
            "the final codeword must carry the described number of symbols",
        );
        let _bound = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_CODEWORD, codeword);
    }

    /// Grind the site guarding the query positions.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn query_pow(&mut self) -> F {
        if self.shape.pow_bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(QUERY_POW, self.shape.pow_bits)
    }

    /// Draw the distinct fold pairs this run opens.
    ///
    /// # Returns
    ///
    /// Each pair's low-indexed position, in ascending order.
    pub fn query_pairs(&mut self) -> Vec<usize> {
        let kept = self.state.challenge_uniform_bits_rejecting::<F>(
            QUERY_INDICES,
            self.shape.pair_bits,
            self.shape.num_pairs,
            |candidate, kept| !kept.contains(&candidate),
        );
        sorted_pairs(kept)
    }

    /// Release the completeness check without playing the rest of the description.
    ///
    /// A caller that deliberately runs only part of a run closes it this way.
    ///
    /// Dropping an unfinished driver otherwise panics.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "a binary PCS run carries its own proof values",
        );
    }
}

/// Verifier-side transcript of one binary-tower PCS run.
///
/// Mirrors the prover driver step for step, so the two walk one description.
pub struct BinaryPcsVerifierTranscript<'a, F: TranscriptField, EF, C> {
    /// Driver walking the description and holding the borrowed sponge.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: BinaryPcsShape,
    /// Marker for the field the folded codewords live in.
    _challenge: PhantomData<EF>,
}

impl<'a, F, EF, C> BinaryPcsVerifierTranscript<'a, F, EF, C>
where
    F: TranscriptField,
    EF: Field + AlgebraIdentity<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this run.
    /// - `shape`: the numbers that fix this run's transcript.
    pub fn new(challenger: &'a mut C, shape: BinaryPcsShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _challenge: PhantomData,
        }
    }

    /// Lend the sponge to the delegated sumcheck rounds of one fold batch.
    pub fn fold_batch<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        run(self.state.challenger_mut())
    }

    /// Replay one folded oracle's Merkle root.
    pub fn oracle_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(ORACLE_COMMITMENT, commitment);
    }

    /// Replay the final codeword the proof carries.
    ///
    /// # Errors
    ///
    /// When the codeword is not the length the run was described with.
    pub fn final_codeword(&mut self, codeword: &[EF]) -> Result<(), TranscriptFailure<F>> {
        // The length comes from the proof, so a mismatch is a rejection.
        //
        // Releasing the completeness check keeps this the only failure.
        if codeword.len() != self.shape.final_codeword_len {
            self.state.abort();
            return Err(TranscriptFailure::FinalCodewordLength {
                expected: self.shape.final_codeword_len,
                got: codeword.len(),
            });
        }

        let _bound = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_CODEWORD, codeword);

        Ok(())
    }

    /// Replay the grind guarding the query positions.
    ///
    /// # Errors
    ///
    /// - The witness is not zero where the site asks for no work.
    /// - The witness misses the difficulty the site requires.
    pub fn query_pow(&mut self, witness: F) -> Result<(), TranscriptFailure<F>> {
        if self.shape.pow_bits == 0 {
            // A zero-difficulty site plays no step, so nothing else reads this field.
            //
            // Pinning it here is what stops any value from riding along unbound.
            if witness != F::ZERO {
                // Releasing the completeness check keeps this rejection the only failure.
                self.state.abort();
                return Err(TranscriptFailure::NonCanonicalPowWitness { actual: witness });
            }
            return Ok(());
        }

        self.state
            .observe_pow(QUERY_POW, self.shape.pow_bits, witness)
            .map_err(|_| TranscriptFailure::PowWitness {
                bits: self.shape.pow_bits,
            })
    }

    /// Redraw the distinct fold pairs this run opens.
    pub fn query_pairs(&mut self) -> Vec<usize> {
        let kept = self.state.challenge_uniform_bits_rejecting::<F>(
            QUERY_INDICES,
            self.shape.pair_bits,
            self.shape.num_pairs,
            |candidate, kept| !kept.contains(&candidate),
        );
        sorted_pairs(kept)
    }

    /// Release the completeness check because the proof is being rejected.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("a binary PCS run reads an empty wire");
    }
}

/// A transcript step the proof failed to satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TranscriptFailure<F> {
    /// The final codeword carries a symbol count the run never described.
    #[error("final codeword length mismatch: expected {expected}, got {got}")]
    FinalCodewordLength {
        /// Symbol count the run was described with.
        expected: usize,
        /// Symbol count the proof carries.
        got: usize,
    },
    /// A grinding witness did not meet the difficulty its step requires.
    #[error("query grinding witness clears fewer than {bits} bits")]
    PowWitness {
        /// Difficulty the site requires, in bits.
        bits: usize,
    },
    /// A grinding witness is not the value a zero difficulty admits.
    ///
    /// At zero bits the site reads no witness, so nothing else binds the field.
    ///
    /// Zero is the only value an honest prover emits, so zero is the only one accepted.
    #[error("query grinding witness is {actual} at zero difficulty, expected zero")]
    NonCanonicalPowWitness {
        /// The witness the proof carries.
        actual: F,
    },
}

/// Turn the drawn pair indices into ascending low-indexed positions.
///
/// The draw keeps them distinct but in draw order.
///
/// Both sides sort, so the opened positions are one canonical list.
fn sorted_pairs(kept: Vec<p3_challenger::fs::TranscriptBound<usize>>) -> Vec<usize> {
    let mut pairs: Vec<usize> = kept
        .into_iter()
        .map(p3_challenger::fs::TranscriptBound::into_inner)
        .collect();
    pairs.sort_unstable();
    // A pair indexes two adjacent symbols, so its position is the low one.
    pairs.into_iter().map(|pair| pair << 1).collect()
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;

    use p3_binary_field::BinaryField128;
    use p3_challenger::CanSample;
    use p3_challenger::fs::PROTOCOL_ID_LEN;
    use p3_challenger::testing::{assert_seeds_pairwise_distinct, pow_difficulties, seed_digest};
    use p3_field::PrimeCharacteristicRing;
    use p3_security::grinding::{
        UNPRICED_GRINDING_SITES, grinding_step, is_unpriced_grinding_site,
    };

    use super::*;
    use crate::params::BinaryPcsParams;

    /// The protocol name as the security vocabulary spells it.
    const NAME_STR: &str = "p3-binary-pcs";
    use crate::test_util::challenger;

    /// The configuration every shape below is derived from.
    fn base_config() -> BinaryPcsConfig {
        BinaryPcsConfig::try_new::<BinaryField128, BinaryField128>(
            8,
            BinaryPcsParams {
                log_inv_rate: 2,
                pow_bits: 4,
                security_level: 100,
            },
        )
        .unwrap()
        .try_with_folding(1)
        .unwrap()
    }

    #[test]
    fn every_knob_of_a_run_reaches_the_seed() {
        // Invariant: one field changed moves the seed, and no two changes collide.
        //
        // Fixture state: the shape derived from the reference configuration.
        //
        // Exhaustiveness check: every field named, none elided by a rest pattern.
        //
        // A field added to the shape stops this from compiling.
        let BinaryPcsShape {
            num_variables: _,
            log_folding_factor: _,
            num_oracles: _,
            final_codeword_len: _,
            pow_bits: _,
            pair_bits: _,
            num_pairs: _,
        } = BinaryPcsShape::new(&base_config());

        let digest = |shape: BinaryPcsShape| {
            seed_digest(&shape.domain_separator::<BinaryField128, BinaryField128>())
        };
        let base = BinaryPcsShape::new(&base_config());

        let mut seeds = vec![(String::from("baseline"), digest(base))];

        // The arity moves no step, so it is bound as a chunk instead.
        let mut wider = base;
        wider.num_variables += 1;
        seeds.push((String::from("num variables"), digest(wider)));

        // So does the folding factor.
        let mut coarser = base;
        coarser.log_folding_factor += 1;
        seeds.push((String::from("folding factor"), digest(coarser)));

        // One more folded oracle is one more described step.
        let mut more_oracles = base;
        more_oracles.num_oracles += 1;
        seeds.push((String::from("oracle count"), digest(more_oracles)));

        // The final word travels in the clear, so its width is a step width.
        let mut longer_final = base;
        longer_final.final_codeword_len += 1;
        seeds.push((String::from("final codeword"), digest(longer_final)));

        // The grinding difficulty rides inside its own step.
        let mut harder = base;
        harder.pow_bits += 1;
        seeds.push((String::from("pow bits"), digest(harder)));

        // A narrower index draws from a smaller domain, which is a different claim.
        let mut narrower = base;
        narrower.pair_bits -= 1;
        seeds.push((String::from("pair bits"), digest(narrower)));

        // One more opened pair is one more position the proof must answer.
        let mut more_pairs = base;
        more_pairs.num_pairs += 1;
        seeds.push((String::from("pair count"), digest(more_pairs)));

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn a_zero_difficulty_run_describes_no_grinding_step() {
        // Invariant: a zero-difficulty site contributes no step.
        //
        // The two cases therefore cannot share a description.
        //
        //     bits = 0  ->  no grinding step at all
        //     bits = 4  ->  one step carrying the difficulty
        let mut unground = BinaryPcsShape::new(&base_config());
        unground.pow_bits = 0;
        assert!(pow_difficulties(&unground.pattern::<BinaryField128, BinaryField128>()).is_empty());

        let ground = BinaryPcsShape::new(&base_config());
        assert_eq!(
            pow_difficulties(&ground.pattern::<BinaryField128, BinaryField128>()),
            vec![(QUERY_POW, 4)]
        );
    }

    #[test]
    fn the_protocol_name_fits_the_identifier_and_extends_nothing() {
        // Invariant: the name is what separates this protocol from every other.
        //
        // It must fit the identifier, which carries its length in the last byte.
        assert!(NAME.len() < PROTOCOL_ID_LEN - 1);

        // No workspace protocol name starts with this one.
        //
        // No prefix relation therefore needs a length byte to part it from a longer name.
        assert!(!NAME.starts_with(b"p3-binary-field"));
    }

    #[test]
    fn the_grinding_sites_and_the_unpriced_table_name_each_other() {
        // Invariant: a proof-of-work step is a soundness parameter in two places.
        //
        //     transcript  ->  the difficulty the pattern describes
        //     model       ->  the difficulty the security report credits
        //
        // A site in neither vocabulary is a difficulty nobody compares.
        //
        // The workspace suite runs both directions for every protocol it can reach.
        //
        // This one seeds over a binary tower field.
        //
        // Its separator has a different sponge alphabet, so it cannot join that sweep.
        let shape = BinaryPcsShape::new(&base_config());

        for (label, _bits) in pow_difficulties(&shape.pattern::<BinaryField128, BinaryField128>()) {
            let budgeted = grinding_step(NAME_STR, label).is_some();
            let priced_elsewhere = is_unpriced_grinding_site(NAME_STR, label);

            assert!(
                budgeted || priced_elsewhere,
                "p3-binary-pcs/{label} grinds, but no vocabulary classifies it",
            );
            assert!(
                !(budgeted && priced_elsewhere),
                "p3-binary-pcs/{label} is both compared against the model and priced elsewhere",
            );
        }

        // The other direction: a listed row must name a step this run describes.
        //
        // Without it, dropping the grind would leave the row silently stale.
        let described: Vec<&str> =
            pow_difficulties(&shape.pattern::<BinaryField128, BinaryField128>())
                .into_iter()
                .map(|(label, _)| label)
                .collect();

        for &(protocol, label) in &UNPRICED_GRINDING_SITES {
            if protocol != NAME_STR {
                continue;
            }
            assert!(
                described.contains(&label),
                "p3-binary-pcs/{label} is listed as priced elsewhere, but no step describes it",
            );
        }
    }

    #[test]
    fn a_perturbed_final_codeword_moves_the_state_the_caller_continues_from() {
        // Invariant: every symbol of the uncommitted final word is bound.
        //
        // Changing one moves the grind and every position drawn after it.
        //
        // Fixture state: a four-symbol codeword.
        //
        // Mutation: bump the last symbol.
        let shape = BinaryPcsShape::new(&base_config());

        let run = |codeword: &[BinaryField128]| {
            let mut ch = challenger();
            let mut transcript = BinaryPcsProverTranscript::new(&mut ch, shape);
            for _ in 0..shape.num_oracles {
                transcript.oracle_commitment(
                    p3_symmetric::MerkleCap::<BinaryField128, [u8; 32]>::new(vec![[0u8; 32]]),
                );
            }
            transcript.final_codeword(codeword);
            let _witness = transcript.query_pow();
            let positions = transcript.query_pairs();
            transcript.finish();
            (positions, CanSample::<BinaryField128>::sample(&mut ch))
        };

        let honest = vec![BinaryField128::ZERO; shape.final_codeword_len];
        let mut tampered = honest.clone();
        *tampered.last_mut().unwrap() = BinaryField128::ONE;

        assert_ne!(run(&honest), run(&tampered));
    }
}
