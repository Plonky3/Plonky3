//! Many signatures over keyed BLAKE3, verified in one binary-field proof.
//!
//! One verification is a Boolean circuit of BLAKE3 compressions, which Flock proves on every signature of a batch (ePrint 2026/1329).
//!
//! The circuit's shape depends only on the parameters: every chain takes all `w - 1` steps, and digit-driven selects keep or drop each one.
//!
//! Each signature's key, epoch and message are public inputs, and its computed root and digit sum are public outputs.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsArith};
use p3_flock::gadgets::blake3::compress;
use p3_flock::opening::{Commitment, FlockMmcs};
use p3_flock::{
    Batch, BatchError, Builder, Challenge, Circuit, FlockArith, FlockChallenger, FlockConfig,
    FlockHash, FlockProof, Port, ProvingError, PublicPorts, Val, VerificationError, Wire, Witness,
    Word,
};
use p3_util::log2_ceil_usize;
use p3_verifier_arith::Native;

use crate::blake3_hash::{
    CHAIN, DIGEST_BYTES, LEAF, MESSAGE, MESSAGE_BYTES, NODE, RANDOMNESS_BYTES,
};
use crate::{Blake3Hash, Claim, Params, Statement};

/// Input port of the hash parameter.
const PARAMETER_PORT: usize = 0;

/// Input port of the epoch.
const EPOCH_PORT: usize = 1;

/// Input port of the message.
const MESSAGE_PORT: usize = 2;

/// Input port of the encoding randomness.
const RANDOMNESS_PORT: usize = 3;

/// Input port of the chain values.
const CHAINS_PORT: usize = 4;

/// Input port of the Merkle path.
const PATH_PORT: usize = 5;

/// Output port of the computed root.
const ROOT_PORT: usize = 0;

/// Output port of the digit sum.
const SUM_PORT: usize = 1;

/// Bits of the digit sum, enough for any parameters.
const SUM_BITS: usize = 16;

/// Words of a digest.
const DIGEST_WORDS: usize = DIGEST_BYTES / 4;

/// The first block of a chunk, a BLAKE3 flag (BLAKE3 specification, Table 3).
const CHUNK_START: u64 = 1;

/// The last block of a chunk.
const CHUNK_END: u64 = 2;

/// The block whose output is the hash.
const ROOT: u64 = 8;

/// Every block of the keyed mode.
const KEYED_HASH: u64 = 16;

/// Why signatures cannot be aggregated, or their proof is rejected.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AggregationError {
    /// There is no signature.
    #[error("no signature to aggregate")]
    Empty,
    /// The batch is out of Flock's range.
    #[error(transparent)]
    Batch(#[from] BatchError),
    /// The proof cannot be made, for instance because a signature does not verify.
    #[error(transparent)]
    Proving(#[from] ProvingError),
    /// The proof is rejected.
    #[error(transparent)]
    Verification(#[from] VerificationError),
}

/// The verification circuit of one parameter set, and the Flock proofs of batches of it.
#[derive(Clone, Debug)]
pub struct BinaryAggregator {
    params: Params,
    circuit: Circuit,
    config: FlockConfig,
}

impl BinaryAggregator {
    /// The aggregator for signatures under `params`, proving with `config`.
    ///
    /// # Panics
    ///
    /// When the chain ends exceed one BLAKE3 chunk, or the digits one digest.
    pub fn new(params: Params, config: FlockConfig) -> Self {
        Self {
            params,
            circuit: circuit(&params),
            config,
        }
    }

    /// The verification circuit of one signature.
    pub const fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// The proof that every claim's signature verifies.
    ///
    /// The batch is padded to a power of two by repeating the last claim.
    ///
    /// # Errors
    ///
    /// When there is no claim, the batch is out of range, or a signature does not verify.
    ///
    /// # Panics
    ///
    /// When a signature does not have the shape the parameters give.
    pub fn prove<H: FlockHash>(
        &self,
        claims: &[Claim<Blake3Hash>],
        challenger: &mut FlockChallenger<H>,
    ) -> Result<FlockProof<H>, AggregationError> {
        let (last, batch) = self.batch(claims.len())?;
        for claim in claims {
            assert_eq!(
                claim.signature.chains.len(),
                self.params.num_chains(),
                "one value per chain"
            );
            assert_eq!(
                claim.signature.path.len(),
                self.params.log_lifetime(),
                "one sibling per level"
            );
        }
        let witness = Witness::generate(
            &self.circuit,
            batch.log_instances(),
            claims,
            &claims[last],
            |claim, words| self.write_input(claim, words),
        );
        let statements: Vec<Statement<Blake3Hash>> = claims.iter().map(Claim::statement).collect();
        let public = self.public(&batch, &statements);
        Ok(p3_flock::prove(
            &self.config,
            &batch,
            &public,
            witness,
            challenger,
        )?)
    }

    /// Check that `proof` shows a valid signature for every statement.
    ///
    /// # Errors
    ///
    /// When there is no statement, the batch is out of range, or the proof is rejected.
    pub fn verify<H: FlockHash>(
        &self,
        statements: &[Statement<Blake3Hash>],
        proof: &FlockProof<H>,
        challenger: &mut FlockChallenger<H>,
    ) -> Result<(), AggregationError> {
        self.verify_in(&mut Native::new(), statements, proof, challenger)
    }

    /// As [`Self::verify`], within an arithmetic.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn verify_in<A, Ch, H>(
        &self,
        arith: &mut A,
        statements: &[Statement<Blake3Hash>],
        proof: &FlockProof<H>,
        challenger: &mut Ch,
    ) -> Result<(), AggregationError>
    where
        A: FlockArith,
        Ch: FieldChallenger<Val>
            + GrindingChallenger<Witness = Val>
            + CanSampleUniformBits<Val>
            + CanObserve<Commitment>,
        H: FlockHash,
        FlockMmcs<H>: MmcsArith<
                A,
                A::F,
                Commitment = Commitment,
                MultiProof = <FlockMmcs<H> as Mmcs<Val>>::MultiProof,
            >,
        for<'s> ExtensionMmcs<Val, Challenge, &'s FlockMmcs<H>>: MmcsArith<
                A,
                A::EF,
                MultiProof = <FlockMmcs<H> as Mmcs<Val>>::MultiProof,
                Target = <FlockMmcs<H> as MmcsArith<A, A::F>>::Target,
            >,
    {
        let (_, batch) = self.batch(statements.len())?;
        let public = self.public(&batch, statements);
        Ok(p3_flock::verify_in(
            arith,
            &self.config,
            &batch,
            &public,
            proof,
            challenger,
        )?)
    }

    /// The last item's index and the batch of `count` items rounded up to a power of two.
    fn batch(&self, count: usize) -> Result<(usize, Batch<'_>), AggregationError> {
        let last = count.checked_sub(1).ok_or(AggregationError::Empty)?;
        Ok((last, Batch::new(&self.circuit, log2_ceil_usize(count))?))
    }

    /// The public words of every instance, padded with the last statement.
    fn public(&self, batch: &Batch<'_>, statements: &[Statement<Blake3Hash>]) -> PublicPorts {
        let ports = [
            Port::Input(PARAMETER_PORT),
            Port::Input(EPOCH_PORT),
            Port::Input(MESSAGE_PORT),
            Port::Output(ROOT_PORT),
            Port::Output(SUM_PORT),
        ];
        let last = statements.len() - 1;
        PublicPorts::new(batch, &ports, |i, words| {
            let statement = &statements[i.min(last)];
            let key = &statement.public_key;
            self.put(words, Port::Input(PARAMETER_PORT), &key.parameter);
            self.put(
                words,
                Port::Input(EPOCH_PORT),
                &statement.epoch.to_le_bytes(),
            );
            self.put(words, Port::Input(MESSAGE_PORT), &statement.message);
            self.put(words, Port::Output(ROOT_PORT), &key.root);
            let sum = self.params.target_sum() as u64;
            self.put(words, Port::Output(SUM_PORT), &sum.to_le_bytes());
        })
    }

    /// One instance's input words.
    fn write_input(&self, claim: &Claim<Blake3Hash>, words: &mut [u64]) {
        let signature = &claim.signature;
        self.put(
            words,
            Port::Input(PARAMETER_PORT),
            &claim.public_key.parameter,
        );
        self.put(words, Port::Input(EPOCH_PORT), &claim.epoch.to_le_bytes());
        self.put(words, Port::Input(MESSAGE_PORT), &claim.message);
        self.put(words, Port::Input(RANDOMNESS_PORT), &signature.randomness);
        self.put(
            words,
            Port::Input(CHAINS_PORT),
            signature.chains.as_flattened(),
        );
        self.put(words, Port::Input(PATH_PORT), signature.path.as_flattened());
    }

    /// Write `bytes` little-endian into a port's words.
    fn put(&self, words: &mut [u64], port: Port, bytes: &[u8]) {
        let range = self.circuit.port(port).expect("a port of the circuit");
        for (word, chunk) in words[range].iter_mut().zip(bytes.chunks(8)) {
            let mut le = [0; 8];
            le[..chunk.len()].copy_from_slice(chunk);
            *word = u64::from_le_bytes(le);
        }
    }
}

/// One signature's verification as a Boolean circuit, with the layout of [`Blake3Hash`].
///
/// # Panics
///
/// When the chain ends exceed one BLAKE3 chunk, or the digits one digest.
fn circuit(params: &Params) -> Circuit {
    let (height, chains, bits) = (
        params.log_lifetime(),
        params.num_chains(),
        params.chunk_bits(),
    );
    assert!(
        chains * DIGEST_BYTES <= 1024,
        "the chain ends fit in one chunk"
    );
    assert!(
        chains * bits <= 8 * DIGEST_BYTES,
        "the digits fit in one digest"
    );
    let digest = 8 * DIGEST_BYTES;
    let mut c = Builder::new(
        &[
            digest,
            height,
            8 * MESSAGE_BYTES,
            8 * RANDOMNESS_BYTES,
            chains * digest,
            height * digest,
        ],
        &[digest, SUM_BITS],
    );
    let input = |c: &Builder, port: usize| Word::split_u32(c.input(port));
    let parameter = input(&c, PARAMETER_PORT);
    let epoch: Word = c
        .input(EPOCH_PORT)
        .iter()
        .copied()
        .chain(core::iter::repeat(Wire::ZERO))
        .take(32)
        .collect();
    let (message, randomness) = (input(&c, MESSAGE_PORT), input(&c, RANDOMNESS_PORT));
    let (values, path) = (input(&c, CHAINS_PORT), input(&c, PATH_PORT));
    let literal = |c: &Builder, value: usize| c.literal(value as u64, 32);

    // The digits, read from the message digest, must sum to the target.
    let tweak = [
        literal(&c, MESSAGE as usize),
        epoch.clone(),
        literal(&c, 0),
        literal(&c, 0),
    ];
    let encoded: Vec<Word> = randomness.iter().chain(&message).cloned().collect();
    let digest_bits: Vec<Wire> = keyed_hash(&mut c, &parameter, &tweak, &encoded)
        .iter()
        .flat_map(|word| word.iter().copied())
        .collect();
    let digits: Vec<&[Wire]> = digest_bits.chunks(bits).take(chains).collect();
    let padded: Vec<Word> = digits
        .iter()
        .map(|digit| {
            digit
                .iter()
                .copied()
                .chain(core::iter::repeat(Wire::ZERO))
                .take(SUM_BITS)
                .collect()
        })
        .collect();
    let operands: Vec<&[Wire]> = padded.iter().map(|w| &w[..]).collect();
    let sum = c.wrapping_sum(&operands);
    for (bit, &wire) in sum.iter().enumerate() {
        c.output(SUM_PORT, bit, wire);
    }

    // Every chain takes all its steps from the signed value, and a step counts once its start reaches the digit.
    let mut ends = Vec::with_capacity(chains * DIGEST_WORDS);
    for (chain, (digit, value)) in digits.iter().zip(values.chunks(DIGEST_WORDS)).enumerate() {
        let mut state = value.to_vec();
        let mut reached = Wire::ZERO;
        for slot in 0..params.chain_len() {
            // The digit is `slot` when each of its bits matches the slot's.
            let mut equal = c.one();
            for (i, &bit) in digit.iter().enumerate() {
                let matched = if (slot >> i) & 1 == 1 {
                    bit
                } else {
                    c.not(bit)
                };
                equal = c.and(equal, matched);
            }
            // At most one slot equals the digit, so the sum is an OR.
            reached = c.xor(reached, equal);
            let tweak = [
                literal(&c, CHAIN as usize),
                epoch.clone(),
                literal(&c, chain),
                literal(&c, slot + 1),
            ];
            let step = keyed_hash(&mut c, &parameter, &tweak, &state);
            state = state
                .iter()
                .zip(&step)
                .map(|(kept, stepped)| {
                    kept.iter()
                        .zip(stepped)
                        .map(|(&kept, &stepped)| c.mux(reached, stepped, kept))
                        .collect()
                })
                .collect();
        }
        ends.extend(state);
    }
    let tweak = [
        literal(&c, LEAF as usize),
        epoch.clone(),
        literal(&c, 0),
        literal(&c, 0),
    ];
    let leaf = keyed_hash(&mut c, &parameter, &tweak, &ends);

    // The epoch's bits steer the path, and its parent positions tweak each node.
    let root =
        path.chunks(DIGEST_WORDS)
            .enumerate()
            .fold(leaf.to_vec(), |node, (level, sibling)| {
                let bit = epoch[level];
                let (left, right): (Vec<Word>, Vec<Word>) = node
                    .iter()
                    .zip(sibling)
                    .map(|(node, sibling)| {
                        let left: Word = node
                            .iter()
                            .zip(sibling)
                            .map(|(&n, &s)| c.mux(bit, s, n))
                            .collect();
                        let swap = c.xor_word(node, sibling);
                        let right = c.xor_word(&left, &swap);
                        (left, right)
                    })
                    .unzip();
                let tweak = [
                    literal(&c, NODE as usize),
                    literal(&c, level + 1),
                    &epoch >> (level + 1),
                    literal(&c, 0),
                ];
                let children: Vec<Word> = left.into_iter().chain(right).collect();
                keyed_hash(&mut c, &parameter, &tweak, &children).to_vec()
            });
    for (bit, &wire) in root.iter().flat_map(|word| word.iter()).enumerate() {
        c.output(ROOT_PORT, bit, wire);
    }
    c.finish()
}

/// The keyed BLAKE3 hash of `input`, one chunk at most, truncated to a digest.
///
/// The key is `parameter || tweak`, and the input's length is a whole number of words.
fn keyed_hash(
    c: &mut Builder,
    parameter: &[Word],
    tweak: &[Word; 4],
    input: &[Word],
) -> [Word; DIGEST_WORDS] {
    let mut cv: [Word; 8] = core::array::from_fn(|i| {
        if i < 4 {
            parameter[i].clone()
        } else {
            tweak[i - 4].clone()
        }
    });
    let blocks = input.len().div_ceil(16);
    for (j, block) in input.chunks(16).enumerate() {
        let mut flags = KEYED_HASH;
        if j == 0 {
            flags |= CHUNK_START;
        }
        if j + 1 == blocks {
            flags |= CHUNK_END | ROOT;
        }
        let words =
            core::array::from_fn(|i| block.get(i).cloned().unwrap_or_else(|| c.literal(0, 32)));
        let tail = [
            c.literal(0, 32),
            c.literal(0, 32),
            c.literal(4 * block.len() as u64, 32),
            c.literal(flags, 32),
        ];
        let out = compress(c, &cv, &words, &tail);
        cv = core::array::from_fn(|i| out[i].clone());
    }
    core::array::from_fn(|i| cv[i].clone())
}
