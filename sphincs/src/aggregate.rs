//! Many signatures verified in one binary-field Flock proof (ePrint 2026/1329).

use alloc::vec::Vec;

use p3_flock::{
    Batch, BatchError, Circuit, FlockChallenger, FlockConfig, FlockProof, Port, ProvingError,
    PublicPorts, VerificationError, Witness,
};
use p3_util::log2_ceil_usize;

use crate::circuit::{
    COMPUTED_ROOT_PORT, MESSAGE_BYTES, MESSAGE_PORT, ROOT_PORT, SEED_PORT, SIGNATURE_PORT, circuit,
};
use crate::{PublicKey, Signature};

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

/// What a proof shows a signature for: a public key and a message, signed under an empty context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Statement {
    /// The signer's key.
    pub public_key: PublicKey,
    /// The signed message.
    pub message: [u8; MESSAGE_BYTES],
}

/// A statement and the signature that makes it true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// The signer's key and message.
    pub statement: Statement,
    /// The signature.
    pub signature: Signature,
}

/// The verification circuit of one signature, and the Flock proofs of batches of it.
#[derive(Clone, Debug)]
pub struct Aggregator {
    circuit: Circuit,
    compressions: usize,
    config: FlockConfig,
}

impl Aggregator {
    /// The aggregator proving with `config`.
    #[must_use]
    pub fn new(config: FlockConfig) -> Self {
        let (circuit, compressions) = circuit();
        Self {
            circuit,
            compressions,
            config,
        }
    }

    /// The verification circuit of one signature.
    #[must_use]
    pub const fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// SHA-256 compressions the circuit runs per signature.
    #[must_use]
    pub const fn compressions(&self) -> usize {
        self.compressions
    }

    /// The proof that every claim's signature verifies.
    ///
    /// The batch is padded to a power of two by repeating the last claim.
    ///
    /// # Errors
    ///
    /// When there is no claim, the batch is out of range, or a signature does not verify.
    pub fn prove(
        &self,
        claims: &[Claim],
        challenger: &mut FlockChallenger,
    ) -> Result<FlockProof, AggregationError> {
        let (last, batch) = self.batch(claims.len())?;
        let witness = Witness::generate(
            &self.circuit,
            batch.log_instances(),
            claims,
            &claims[last],
            |claim, words| {
                self.put_statement(words, &claim.statement);
                self.put(
                    words,
                    Port::Input(SIGNATURE_PORT),
                    claim.signature.as_bytes(),
                );
            },
        );
        let statements: Vec<Statement> = claims.iter().map(|claim| claim.statement).collect();
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
    pub fn verify(
        &self,
        statements: &[Statement],
        proof: &FlockProof,
        challenger: &mut FlockChallenger,
    ) -> Result<(), AggregationError> {
        let (_, batch) = self.batch(statements.len())?;
        let public = self.public(&batch, statements);
        Ok(p3_flock::verify(
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

    /// The public words of every instance, padded with the last statement: the key and message in, the root out.
    fn public(&self, batch: &Batch<'_>, statements: &[Statement]) -> PublicPorts {
        let ports = [
            Port::Input(SEED_PORT),
            Port::Input(ROOT_PORT),
            Port::Input(MESSAGE_PORT),
            Port::Output(COMPUTED_ROOT_PORT),
        ];
        let last = statements.len() - 1;
        PublicPorts::new(batch, &ports, |i, words| {
            let statement = &statements[i.min(last)];
            self.put_statement(words, statement);
            self.put(
                words,
                Port::Output(COMPUTED_ROOT_PORT),
                &statement.public_key.root,
            );
        })
    }

    /// Write a statement's input ports.
    fn put_statement(&self, words: &mut [u64], statement: &Statement) {
        self.put(words, Port::Input(SEED_PORT), &statement.public_key.seed);
        self.put(words, Port::Input(ROOT_PORT), &statement.public_key.root);
        self.put(words, Port::Input(MESSAGE_PORT), &statement.message);
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
