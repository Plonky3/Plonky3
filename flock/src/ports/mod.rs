//! Public ports: port words of every instance the statement fixes (ePrint 2026/1329, Appendix A, step 4).
//!
//! Ports fill the first words of an instance, so they lie in the region of its first `2^p` words.
//!
//! The lincheck draws a skip coordinate and region coordinates, and weighs every public word of the witness folded at its instance coordinates.
//!
//! The verifier sums the public table under the same weights itself, so the lincheck's one opened claim also binds the ports.
//!
//! A wrong public word makes those weighted sums differ as polynomials in the drawn coordinates, of degree `log_instances + 63 + p`.
//!
//! The statement is bound before the commitment, so neither side can be chosen after a challenge.

#[cfg(test)]
mod tests;

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::fold_rows;
use p3_challenger::CanObserve;
use p3_challenger::fs::{DomainSeparator, FieldUnit, InteractionPattern};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::CryptographicHasher;
use p3_util::log2_strict_usize;
use thiserror::Error;

use crate::circuit::Port;
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::{Batch, Challenge, Val, Witness};

/// Instances a statement chunk hashes on its own.
const DIGEST_INSTANCES: usize = 1 << 10;

/// The statement every proof binds before its first challenge: the circuit, the batch, and the public port words.
const STATEMENT: &[u8] = b"p3-flock-statement";

/// Statement version bound into the seed.
const VERSION: u8 = 1;

/// The public port words of every instance of a batch.
///
/// The default makes no port public, and a proof then only shows that some witness satisfies every instance.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublicPorts {
    /// Public words of one instance, ascending.
    words: Vec<usize>,
    /// Instances, as a power of two.
    log_instances: usize,
    /// Each instance's public words, `stride` words apart, zero past them.
    values: Vec<u64>,
    /// Words per instance in `values`, the public ones rounded up to a power of two.
    stride: usize,
    /// Keccak-256 of the instance count, the public words and their values; zero when no word is public.
    digest: [u8; 32],
}

/// Why public ports are refused.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PortsError {
    /// The public ports were made for another number of instances.
    #[error("the public ports cover 2^{actual} instances, the batch 2^{expected}")]
    InstanceCount {
        /// The batch's instances, as a power of two.
        expected: usize,
        /// The ports', as a power of two.
        actual: usize,
    },
    /// A public word lies past the circuit's ports.
    #[error("public word {word} lies past the {port_words} port words")]
    WordOutOfRange {
        /// The word.
        word: usize,
        /// Words the circuit's ports span.
        port_words: usize,
    },
    /// An instance's witness disagrees with its public words.
    #[error("instance {instance} disagrees with its public words")]
    Mismatch {
        /// One such instance.
        instance: usize,
    },
}

impl PublicPorts {
    /// The given ports of every instance of a batch, `fill(i, words)` writing instance `i`'s port words.
    ///
    /// The port words are all inputs, then all outputs, as the circuit lays them out from word zero.
    ///
    /// # Panics
    ///
    /// - A port is not one of the circuit's.
    #[must_use]
    pub fn new(batch: &Batch<'_>, ports: &[Port], fill: impl Fn(usize, &mut [u64]) + Sync) -> Self {
        let circuit = batch.circuit();
        let mut words: Vec<usize> = ports
            .iter()
            .flat_map(|&port| circuit.port(port).expect("a port of the circuit"))
            .collect();
        words.sort_unstable();
        words.dedup();
        if words.is_empty() {
            return Self::default();
        }
        let stride = words.len().next_power_of_two();
        let port_words = circuit.const_pos() / 64;
        let mut values = vec![0; stride << batch.log_instances()];
        values
            .par_chunks_exact_mut(stride)
            .enumerate()
            .for_each_init(
                || vec![0; port_words],
                |row, (i, out)| {
                    row.fill(0);
                    fill(i, row);
                    for (o, &w) in out.iter_mut().zip(&words) {
                        *o = row[w];
                    }
                },
            );
        let mut public = Self {
            words,
            log_instances: batch.log_instances(),
            values,
            stride,
            digest: [0; 32],
        };
        public.digest = public.compute_digest();
        public
    }

    /// Keccak-256 of the shape, then of the digests of fixed chunks of instances.
    ///
    /// The chunks hash in parallel, so binding a large statement costs little.
    fn compute_digest(&self) -> [u8; 32] {
        let len = self.words.len();
        let chunks: Vec<[u8; 32]> = self
            .values
            .par_chunks(self.stride * DIGEST_INSTANCES)
            .map(|chunk| {
                let mut bytes = Vec::with_capacity(8 * len * DIGEST_INSTANCES);
                for instance in chunk.chunks_exact(self.stride) {
                    for w in &instance[..len] {
                        bytes.extend_from_slice(&w.to_le_bytes());
                    }
                }
                Keccak256Hash.hash_slice(&bytes)
            })
            .collect();
        let mut shape = Vec::with_capacity(8 * (2 + len));
        for v in [self.log_instances, len]
            .into_iter()
            .chain(self.words.iter().copied())
        {
            shape.extend_from_slice(&(v as u64).to_be_bytes());
        }
        Keccak256Hash.hash_iter_slices(
            core::iter::once(shape.as_slice()).chain(chunks.iter().map(|c| c.as_slice())),
        )
    }

    /// Public words of one instance, ascending.
    #[must_use]
    pub fn words(&self) -> &[usize] {
        &self.words
    }

    /// Keccak-256 digest of the shape and every public value.
    pub(crate) const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Whether no word is public.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Instance `i`'s public words, in the order of `words`.
    ///
    /// # Panics
    ///
    /// - The batch has no instance `i`.
    #[must_use]
    pub fn instance(&self, i: usize) -> &[u64] {
        &self.values[i * self.stride..][..self.words.len()]
    }

    /// Check the ports against a batch: its instance count and its port words.
    ///
    /// # Errors
    ///
    /// The ports were made for another batch size, or name a word past the circuit's ports.
    pub(crate) fn check(&self, batch: &Batch<'_>) -> Result<(), PortsError> {
        if self.is_empty() {
            return Ok(());
        }
        if self.log_instances != batch.log_instances() {
            return Err(PortsError::InstanceCount {
                expected: batch.log_instances(),
                actual: self.log_instances,
            });
        }
        let port_words = batch.circuit().const_pos() / 64;
        match self.words.last() {
            Some(&word) if word >= port_words => {
                Err(PortsError::WordOutOfRange { word, port_words })
            }
            _ => Ok(()),
        }
    }

    /// Check the ports against a batch, then against the witness of every instance.
    ///
    /// # Errors
    ///
    /// The ports do not fit the batch, or an instance disagrees with its public words.
    pub(crate) fn check_witness(
        &self,
        batch: &Batch<'_>,
        witness: &Witness,
    ) -> Result<(), PortsError> {
        self.check(batch)?;
        if self.is_empty() {
            return Ok(());
        }
        let instance_words = 1 << (batch.circuit().k_log() - K_SKIP);
        let mismatch = witness
            .z()
            .par_chunks_exact(instance_words)
            .zip(self.values.par_chunks_exact(self.stride))
            .enumerate()
            .find_map_any(|(i, (z, public))| {
                self.words
                    .iter()
                    .zip(public)
                    .any(|(&w, &v)| z[w] != v)
                    .then_some(i)
            });
        mismatch.map_or(Ok(()), |instance| Err(PortsError::Mismatch { instance }))
    }

    /// Bind the whole statement into the sponge, before any challenge: the circuit and the public words' digest.
    ///
    /// # Errors
    ///
    /// The ports do not fit the batch.
    pub(crate) fn bind<C: CanObserve<Val>>(
        &self,
        batch: &Batch<'_>,
        challenger: &mut C,
    ) -> Result<(), PortsError> {
        self.check(batch)?;
        let pattern = InteractionPattern::new(Vec::new()).expect("an empty pattern is well formed");
        let mut separator = DomainSeparator::<FieldUnit<Val>>::new(VERSION, STATEMENT, pattern);
        separator.instance(batch.circuit().digest());
        separator.instance(&self.digest);
        separator.seed(challenger);
        Ok(())
    }

    /// The region's word bits: enough for every public word.
    pub(crate) fn region_bits(&self) -> usize {
        self.words
            .last()
            .map_or(0, |&w| log2_strict_usize((w + 1).next_power_of_two()))
    }

    /// Each public word with its weight `eq(region, w)`.
    pub(crate) fn weights<'a>(
        &'a self,
        region: &[Challenge],
    ) -> impl Iterator<Item = (usize, Challenge)> + 'a {
        let eq = Poly::new_from_point(region, Challenge::ONE);
        self.words.iter().map(move |&w| (w, eq.as_slice()[w]))
    }

    /// The public table's weighted sum: each word folded over the instances, at the skip coordinate, under its weight.
    ///
    /// The cost is linear in the table.
    pub(crate) fn evaluate(
        &self,
        outer: &[Challenge],
        skip: Challenge,
        region: &[Challenge],
    ) -> Challenge {
        let eq_outer = Poly::new_from_point(outer, Challenge::ONE);
        let folded = fold_rows(
            &self.values,
            log2_strict_usize(self.stride) + K_SKIP,
            SKIP_SIZE * self.words.len(),
            eq_outer.as_slice(),
        );
        let lagrange = lagrange_weights(skip, SKIP_SIZE);
        self.weights(region)
            .zip(folded.as_chunks::<SKIP_SIZE>().0)
            .map(|((_, weight), word)| {
                weight
                    * word
                        .iter()
                        .zip(&lagrange)
                        .map(|(&v, &l)| v * l)
                        .sum::<Challenge>()
            })
            .sum()
    }
}
