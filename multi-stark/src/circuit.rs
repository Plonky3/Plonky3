//! Tables whose rows are instances of one Boolean circuit, proved by Flock (ePrint 2026/1329).
//!
//! A circuit table commits its witness bits as one column, one instance after another.
//!
//! Its AIR reads only the port words, one `GF(2^64)` column per word, and nothing commits those.
//!
//! A port word is a claim on the witness, its 64 bits weighed by the powers of `X`:
//!
//! ```text
//!     port p at rho  =  sum_v X^v * witness(rho, p, v)
//! ```
//!
//! The first `2^a` words of an instance, ports first, fold under a drawn point `r` into one claim.
//!
//! Flock proves every row satisfies the circuit and leaves one more claim on the witness.

use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192, fold_rows};
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, SymmetricSteps, TranscriptBound, TranscriptError, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_flock::piop::PiopProof;
use p3_flock::ports::PublicPorts;
use p3_flock::{Batch, Circuit, QuirkyClaim, Witness, lincheck, zerocheck};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_security::{ErrorBits, SecurityTerm};
use p3_sumcheck::layout::Table;
use p3_sumcheck::{FoldedColumn, WeightedColumnClaim};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bits one committed word holds, as a power of two.
const LOG_WORD_BITS: usize = 6;

/// Protocol version bound into the word transcript's seed.
const VERSION: u8 = 1;

/// Protocol name bound into the word transcript's seed.
const NAME: &[u8] = b"p3-multi-stark-circuit-words";

/// Word values label.
const WORDS: &str = "words";

/// Word folding point label.
const BATCH: &str = "batch";

/// Label of the word folding's soundness term.
pub(crate) const WORD_BATCHING_LABEL: &str = "circuit-word-batching";

/// What one circuit table adds to a proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct CircuitProof<EF> {
    /// Each port column at the point the zerocheck bound, port order.
    pub ports: Vec<EF>,
    /// The words after the ports at that point, up to the next power of two.
    pub padding: Vec<EF>,
    /// Flock's zerocheck and lincheck.
    pub piop: PiopProof,
}

/// Why a circuit table's argument is rejected.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CircuitError {
    /// The proof does not carry one value per word the claim folds.
    #[error("{actual} word values, expected {expected}")]
    WordCount {
        /// Ports, then the words up to the next power of two.
        expected: usize,
        /// Values carried.
        actual: usize,
    },
    /// The word values could not be bound.
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    /// Flock's zerocheck or lincheck refused.
    #[error(transparent)]
    Flock(#[from] p3_flock::VerificationError),
}

/// A circuit's Flock argument, at the field and transcript types of the batch that holds it.
///
/// Only the `GF(2^64)` values and `GF(2^192)` challenges Flock runs at implement it.
pub(crate) trait CircuitArgument<F, EF, Ch>: Sync {
    /// The circuit every row runs.
    fn circuit(&self) -> &Circuit;

    /// Bind the words at `rho`, prove every row, and return the two claims left on the witness column.
    #[allow(clippy::too_many_arguments)]
    fn prove(
        &self,
        ports: Vec<EF>,
        rho: &Point<EF>,
        witness: &Witness,
        bits: &[u64],
        table: usize,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (CircuitProof<EF>, [WeightedColumnClaim<EF>; 2]);

    /// Bind the words at `rho`, replay the argument, and return the two claims left on the witness column.
    ///
    /// # Errors
    ///
    /// A word count other than the circuit's, or a reduction that refuses.
    fn verify(
        &self,
        proof: &CircuitProof<EF>,
        rho: &Point<EF>,
        table: usize,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> Result<[WeightedColumnClaim<EF>; 2], CircuitError>;
}

impl<Ch> CircuitArgument<Poly64, Poly192, Ch> for Circuit
where
    Ch: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    fn circuit(&self) -> &Circuit {
        self
    }

    fn prove(
        &self,
        ports: Vec<Poly192>,
        rho: &Point<Poly192>,
        witness: &Witness,
        bits: &[u64],
        table: usize,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (CircuitProof<Poly192>, [WeightedColumnClaim<Poly192>; 2]) {
        let batch = batch(self, rho.num_variables());

        // The words past the ports, read at rho as the ports were.
        let padding = (ports.len()..folded_words(self))
            .map(|word| word_column(self, bits, word).eval_base(rho))
            .collect::<Vec<_>>();

        // Phase 1: every word bound, then the point folding them.
        let values = [ports.as_slice(), &padding].concat();
        let mut state = ProverState::new(&mut *challenger, &word_separator(self));
        let _ =
            state.observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(WORDS, &values);
        let r = draw_batch(&mut state, values.len());
        assert!(state.finalize().is_empty(), "the words carry no wire bytes");

        // Phase 2: every row satisfies the circuit, reduced to claims at one point.
        let (zerocheck, claims) = zerocheck::prove(&batch, witness, pow_bits, challenger);

        // Phase 3: those claims, reduced to one claim on the committed bits.
        let (lincheck, claim, folded) =
            lincheck::prove_folded(&batch, bits, &claims, &PublicPorts::default(), challenger);

        // Both claims fold the instances away, so each hands the opening its column already folded.
        //
        //     word claim     the instances folded at rho, one pass over the bits
        //     witness claim  the instances folded at Flock's point, which the lincheck kept
        let log_instances = rho.num_variables();
        let instances_at_rho = fold_rows(
            bits,
            self.k_log(),
            self.useful_bits(),
            Poly::new_from_point(rho.as_slice(), Poly192::ONE).as_slice(),
        );
        let claims = [
            word_claim(
                self,
                rho,
                &r,
                &values,
                table,
                Some(FoldedColumn {
                    log_outer: log_instances,
                    values: instances_at_rho,
                }),
            ),
            witness_claim(
                claim,
                table,
                Some(FoldedColumn {
                    log_outer: log_instances,
                    values: folded,
                }),
            ),
        ];
        (
            CircuitProof {
                ports,
                padding,
                piop: PiopProof {
                    zerocheck,
                    lincheck,
                },
            },
            claims,
        )
    }

    fn verify(
        &self,
        proof: &CircuitProof<Poly192>,
        rho: &Point<Poly192>,
        table: usize,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> Result<[WeightedColumnClaim<Poly192>; 2], CircuitError> {
        let words = folded_words(self);
        let values = [proof.ports.as_slice(), &proof.padding].concat();
        if proof.ports.len() != port_words(self) || values.len() != words {
            return Err(CircuitError::WordCount {
                expected: words,
                actual: values.len(),
            });
        }
        let batch = batch(self, rho.num_variables());

        // Bind every word and draw the folding point where the prover did.
        let mut state = VerifierState::new(&mut *challenger, &word_separator(self), &[]);
        if let Err(error) =
            state.observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(WORDS, &values)
        {
            // The driver refuses to be dropped mid-pattern, so it is released first.
            state.abort();
            return Err(error.into());
        }
        let r = draw_batch(&mut state, words);
        state.finalize()?;

        // Replay the zerocheck and the lincheck down to one witness claim.
        let claims = zerocheck::verify(&batch, &proof.piop.zerocheck, pow_bits, challenger)
            .map_err(p3_flock::VerificationError::from)?;
        let claim = lincheck::verify(
            &batch,
            &claims,
            &PublicPorts::default(),
            &proof.piop.lincheck,
            challenger,
        )
        .map_err(p3_flock::VerificationError::from)?;
        Ok([
            word_claim(self, rho, &r, &values, table, None),
            witness_claim(claim, table, None),
        ])
    }
}

/// Port words of one instance: the inputs, then the outputs.
pub(crate) const fn port_words(circuit: &Circuit) -> usize {
    circuit.const_pos() / 64
}

/// Words the word claim folds: the ports, rounded up to a power of two.
const fn folded_words(circuit: &Circuit) -> usize {
    port_words(circuit).next_power_of_two()
}

/// The witness column's row variables for `2^log_instances` instances.
pub(crate) const fn witness_variables(circuit: &Circuit, log_instances: usize) -> usize {
    circuit.k_log() + log_instances
}

/// The soundness terms of one circuit table of `2^log_instances` rows.
///
/// Flock's two reductions, then the folding of the words under one drawn point.
pub(crate) fn security_terms(
    circuit: &Circuit,
    log_instances: usize,
    field_bits: usize,
) -> [SecurityTerm; 3] {
    let [zerocheck, lincheck] = batch(circuit, log_instances).security_terms();
    // A wrong word survives the fold only where a nonzero multilinear in `a` coordinates vanishes.
    let coordinates = u128::from(folded_words(circuit).ilog2()).max(1);
    let batching = SecurityTerm::new(
        WORD_BATCHING_LABEL,
        ErrorBits::from_numerator(field_bits, coordinates),
    );
    [zerocheck, lincheck, batching]
}

/// Whether `2^log_instances` rows of this circuit make a batch Flock proves, read by an AIR of this width.
pub(crate) const fn fits(circuit: &Circuit, log_instances: usize, width: usize) -> bool {
    width == port_words(circuit) && Batch::new(circuit, log_instances).is_ok()
}

/// The batch of `2^log_instances` instances.
///
/// # Panics
///
/// When the instances and the circuit's bits are too few for the zerocheck, or too many to address.
fn batch(circuit: &Circuit, log_instances: usize) -> Batch<'_> {
    Batch::new(circuit, log_instances).expect("a circuit table holds a batch Flock can prove")
}

/// The witness bits as one committed column, instance after instance.
pub(crate) fn witness_table(bits: Vec<u64>, num_variables: usize) -> Table<Poly64> {
    Table::from_packed_bits(RowMajorMatrix::new(bits, 1), num_variables)
}

/// The port words as the AIR's columns, one per word, one row per instance.
pub(crate) fn port_table(circuit: &Circuit, bits: &[u64]) -> Table<Poly64> {
    let rows = bits.len() >> (circuit.k_log() - LOG_WORD_BITS);
    let columns = (0..port_words(circuit))
        .flat_map(|word| word_column(circuit, bits, word).into_evals())
        .collect();
    Table::new(RowMajorMatrix::new(columns, rows))
}

/// Word `word` of every instance, as one multilinear over the instances.
fn word_column(circuit: &Circuit, bits: &[u64], word: usize) -> Poly<Poly64> {
    // Instance `i` holds its words from `i * stride` on.
    let stride = 1 << (circuit.k_log() - LOG_WORD_BITS);
    Poly::new(
        bits.iter()
            .skip(word)
            .step_by(stride)
            .map(|&bits| Poly64::new(bits))
            .collect(),
    )
}

/// The claim folding every word at `rho` under the drawn point `r`.
///
/// ```text
///     high   = (rho, zeros above the folded words, r)
///     weight = X^v on bit v of a word, which reads 64 packed bits as their word
///     value  = sum_p eq(r, p) * word_p(rho)
/// ```
fn word_claim(
    circuit: &Circuit,
    rho: &Point<Poly192>,
    r: &[Poly192],
    values: &[Poly192],
    table: usize,
    folded: Option<FoldedColumn<Poly192>>,
) -> WeightedColumnClaim<Poly192> {
    let word_variables = circuit.k_log() - LOG_WORD_BITS;
    let mut high = rho.as_slice().to_vec();
    high.resize(high.len() + word_variables - r.len(), Poly192::ZERO);
    high.extend_from_slice(r);
    let fold = Poly::new_from_point(r, Poly192::ONE);
    WeightedColumnClaim {
        table,
        column: 0,
        high: Point::new(high),
        low_weights: (0..1 << LOG_WORD_BITS)
            .map(|v| Poly192::from(Poly64::new(1 << v)))
            .collect(),
        value: fold.iter().zip(values).map(|(&w, &value)| w * value).sum(),
        folded,
    }
}

/// Flock's claim as a weighted claim on the witness column of one table.
fn witness_claim(
    claim: QuirkyClaim,
    table: usize,
    folded: Option<FoldedColumn<Poly192>>,
) -> WeightedColumnClaim<Poly192> {
    let weighted = claim.into_weighted();
    WeightedColumnClaim {
        table,
        column: 0,
        high: weighted.opening.high,
        low_weights: weighted.opening.low_weights,
        value: weighted.value,
        folded,
    }
}

/// Draw the point folding `words` words, one coordinate per halving.
fn draw_batch<S>(state: &mut S, words: usize) -> Vec<Poly192>
where
    S: SymmetricSteps,
    S::Challenger: CanObserve<Poly64> + CanSample<Poly64>,
{
    let coordinates = words.ilog2() as usize;
    if coordinates == 0 {
        return Vec::new();
    }
    // Accepting every candidate draws exactly the plain list.
    state
        .challenge_extensions_rejecting::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(
            BATCH,
            coordinates,
            |_, _| true,
        )
        .into_iter()
        .map(TranscriptBound::into_inner)
        .collect()
}

/// The word transcript of one table: its circuit, its word values, then the folding point.
fn word_separator(circuit: &Circuit) -> DomainSeparator<FieldUnit<Poly64>> {
    let words = folded_words(circuit);
    let mut steps = alloc::vec![Interaction::algebra::<Poly64, Poly192>(
        Hierarchy::Atomic,
        Kind::Message,
        WORDS,
        Length::Fixed(words),
    )];
    if words > 1 {
        steps.push(Interaction::algebra::<Poly64, Poly192>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BATCH,
            Length::Fixed(words.ilog2() as usize),
        ));
    }
    let pattern = InteractionPattern::new(steps).expect("a flat pattern is always well formed");
    let mut separator = DomainSeparator::new(VERSION, NAME, pattern);
    separator.instance(circuit.digest());
    separator
}
