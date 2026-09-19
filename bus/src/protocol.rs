//! Commitment-independent reduction of a planned bus to terminal leaf claims.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{BusDirection, BusPlan, ProductGkrError, ProductGkrOutput, ProductGkrProof};

const VERSION: u8 = 1;
const NAME: &[u8] = b"p3-bus-argument";
const FINGERPRINT: &str = "fingerprint";
const OFFSET: &str = "offset";
const PRODUCT: &str = "product_gkr";

type Alphabet<F> = FieldUnit<F>;

struct ProductReduction;

/// Verifier challenges defining every bus leaf factor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusChallenges<EF> {
    /// Point evaluating the padded tuple's multilinear extension.
    pub fingerprint: Vec<EF>,
    /// Random shift applied to every tuple fingerprint.
    pub offset: EF,
}

/// Product proof for one verifier-derived bus plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusProof<EF> {
    /// Reduction of the push and pull product trees to one shared point.
    pub product: ProductGkrProof<EF>,
}

/// Unauthenticated terminal claims returned by the bus reduction.
#[must_use = "terminal claims remain unauthenticated until checked against PCS openings"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusReductionOutput<EF> {
    /// Challenges defining the reduced leaf polynomials.
    pub challenges: BusChallenges<EF>,
    /// Product roots, terminal point, and terminal leaf evaluations.
    pub product: ProductGkrOutput<EF>,
}

/// Failure to construct or verify a planned bus reduction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusArgumentError {
    /// One materialized side has a statement-inconsistent leaf count.
    #[error("binary-bus {direction:?} leaf count is {actual}, expected {expected}")]
    LeafCountMismatch {
        /// Side whose materialization has the wrong length.
        direction: BusDirection,
        /// Count derived from the public bus plan.
        expected: usize,
        /// Count supplied by materialization.
        actual: usize,
    },
    /// Honest-prover inputs do not satisfy the multiset equality.
    #[error("binary-bus push and pull products differ")]
    UnbalancedProducts,
    /// The product reduction is malformed or inconsistent.
    #[error(transparent)]
    Product(#[from] ProductGkrError),
}

impl BusPlan {
    /// Prove the equality of this plan's materialized push and pull multisets.
    ///
    /// The returned terminal claims are not authenticated here.
    /// A caller must reconstruct them from commitment-bound evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong materialized shape or unequal products.
    pub fn prove<F, EF, Challenger>(
        &self,
        materialize: impl FnOnce(&BusChallenges<EF>) -> [Vec<EF>; 2],
        challenger: &mut Challenger,
    ) -> Result<(BusProof<EF>, BusReductionOutput<EF>), BusArgumentError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Sample the leaf polynomial only after the surrounding commitment is bound.
        let mut transcript = BusProverTranscript::<_, F, EF>::new(challenger, self);
        let challenges = transcript.challenges();
        let [pushes, pulls] = materialize(&challenges);

        // Validate every derived witness length before product reduction allocates by shape.
        if let Err(error) = self
            .validate_leaf_count(BusDirection::Push, pushes.len())
            .and_then(|()| self.validate_leaf_count(BusDirection::Pull, pulls.len()))
        {
            transcript.abort();
            return Err(error);
        }

        // Shared-root encoding is a statement, so reject a false witness explicitly.
        let push_root = pushes.iter().copied().product::<EF>();
        let pull_root = pulls.iter().copied().product::<EF>();
        if push_root != pull_root {
            transcript.abort();
            return Err(BusArgumentError::UnbalancedProducts);
        }
        let (product, output) = transcript.product(|challenger| {
            ProductGkrProof::prove::<F, _>(
                &[pushes.as_slice(), pulls.as_slice()],
                self.product_shape(),
                challenger,
            )
        });
        transcript.finish();

        Ok((
            BusProof { product },
            BusReductionOutput {
                challenges,
                product: output,
            },
        ))
    }

    /// Verify this plan's product reduction and return unauthenticated terminal claims.
    ///
    /// # Errors
    ///
    /// Returns an error when the product proof is malformed or inconsistent.
    pub fn verify<F, EF, Challenger>(
        &self,
        proof: &BusProof<EF>,
        challenger: &mut Challenger,
    ) -> Result<BusReductionOutput<EF>, BusArgumentError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Replay exactly the plan-derived challenge and product schedule.
        let mut transcript = BusVerifierTranscript::<_, F, EF>::new(challenger, self);
        let challenges = transcript.challenges();
        let product = match transcript.product(|challenger| {
            proof
                .product
                .verify::<F, _>(self.product_shape(), challenger)
        }) {
            Ok(product) => product,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };
        transcript.finish();

        Ok(BusReductionOutput {
            challenges,
            product,
        })
    }

    /// Reconstruct one identity-padded terminal leaf evaluation from block evaluations.
    ///
    /// `block_values` must follow [`Self::terminal_shares`] for the selected direction.
    #[must_use]
    pub fn terminal_value<EF: Field>(
        &self,
        direction: BusDirection,
        point: &[EF],
        block_values: &[EF],
    ) -> Option<EF> {
        // A malformed caller-owned vector has no statement-defined interpretation.
        let shares = self.terminal_shares(direction).collect::<Vec<_>>();
        if point.len() != self.product_shape().log_height() || block_values.len() != shares.len() {
            return None;
        }

        // Identity padding contributes one globally; each live block replaces its share of it.
        let mut value = EF::ONE;
        for (share, &block_value) in shares.iter().zip(block_values) {
            let prefix = &point[..share.prefix_variables];
            let weight = equality_at_vertex(prefix, share.prefix_index);
            value += weight * (block_value - EF::ONE);
        }
        Some(value)
    }

    const fn validate_leaf_count(
        &self,
        direction: BusDirection,
        actual: usize,
    ) -> Result<(), BusArgumentError> {
        // Physical blocks exactly partition the explicit prefix on each side.
        let expected = self.security_geometry().non_padding_leaf_counts()[match direction {
            BusDirection::Push => 0,
            BusDirection::Pull => 1,
        }];
        if actual != expected {
            return Err(BusArgumentError::LeafCountMismatch {
                direction,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The plan fixes both challenge length and the delegated reduction shape.
        InteractionPattern::new(vec![
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FINGERPRINT,
                Length::Fixed(self.security_geometry().tuple_variables()),
            ),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OFFSET,
                Length::Scalar,
            ),
            Interaction::marker::<ProductReduction>(Hierarchy::Begin, Kind::Protocol, PRODUCT),
            Interaction::marker::<ProductReduction>(Hierarchy::End, Kind::Protocol, PRODUCT),
        ])
        .expect("one matched product-reduction bracket is well formed")
    }

    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Bind every verifier-derived field that changes a tuple or product-tree position.
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        separator
            .instance(&(self.domains().len() as u64).to_be_bytes())
            .instance(&(self.payload_slots() as u64).to_be_bytes())
            .instance(&(self.domain_slots() as u64).to_be_bytes())
            .instance(&(self.fingerprint_width() as u64).to_be_bytes());
        for domain in self.domains() {
            separator
                .instance(&(domain.name.len() as u64).to_be_bytes())
                .instance(domain.name.as_bytes())
                .instance(&(domain.payload_width as u64).to_be_bytes())
                .instance(&(domain.identity as u64).to_be_bytes());
        }
        for direction in [BusDirection::Push, BusDirection::Pull] {
            let blocks = self.blocks(direction);
            separator.instance(&(blocks.len() as u64).to_be_bytes());
            for block in blocks {
                separator
                    .instance(&(block.bus as u64).to_be_bytes())
                    .instance(&(block.owner.air as u64).to_be_bytes())
                    .instance(&(block.owner.declaration as u64).to_be_bytes())
                    .instance(&(block.log_height as u64).to_be_bytes())
                    .instance(&(block.offset as u64).to_be_bytes());
            }
        }
        separator
    }
}

fn equality_at_vertex<F: Field>(point: &[F], vertex: usize) -> F {
    // Coordinates and vertex bits both follow most-significant-variable-first order.
    point
        .iter()
        .enumerate()
        .map(|(coordinate, &challenge)| {
            let bit = (vertex >> (point.len() - 1 - coordinate)) & 1;
            if bit == 0 {
                F::ONE - challenge
            } else {
                challenge
            }
        })
        .product()
}

struct BusProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Typed pattern player borrowing the surrounding challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Number of tuple coordinates fixed by the bus plan.
    fingerprint_variables: usize,
    /// Challenge-field marker used by the transcript codec.
    _ef: core::marker::PhantomData<EF>,
}

impl<'a, C, F, EF> BusProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    fn new(challenger: &'a mut C, plan: &BusPlan) -> Self {
        // Seed from the complete verifier-derived bus statement.
        Self {
            state: ProverState::new(challenger, &plan.domain_separator::<F, EF>()),
            fingerprint_variables: plan.security_geometry().tuple_variables(),
            _ef: core::marker::PhantomData,
        }
    }

    fn challenges(&mut self) -> BusChallenges<EF> {
        // The tuple point precedes the independent product shift.
        let fingerprint = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                FINGERPRINT,
                self.fingerprint_variables,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        let offset = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OFFSET)
            .into_inner();
        BusChallenges {
            fingerprint,
            offset,
        }
    }

    fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The delegated proof owns all product-reduction messages and challenges.
        self.state.begin_protocol::<ProductReduction>(PRODUCT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductReduction>(PRODUCT);
        output
    }

    fn finish(self) {
        // This protocol carries every message in its proof object.
        assert!(self.state.finalize().is_empty());
    }

    fn abort(&mut self) {
        // Release the typed completeness check before returning a witness-shape error.
        self.state.abort();
    }
}

struct BusVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Typed pattern player borrowing the surrounding challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Number of tuple coordinates fixed by the bus plan.
    fingerprint_variables: usize,
    /// Challenge-field marker used by the transcript codec.
    _ef: core::marker::PhantomData<EF>,
}

impl<'a, C, F, EF> BusVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    fn new(challenger: &'a mut C, plan: &BusPlan) -> Self {
        // The verifier reads no wire values at this protocol level.
        Self {
            state: VerifierState::new(challenger, &plan.domain_separator::<F, EF>(), &[]),
            fingerprint_variables: plan.security_geometry().tuple_variables(),
            _ef: core::marker::PhantomData,
        }
    }

    fn challenges(&mut self) -> BusChallenges<EF> {
        // Replay the same statement-derived challenge count as the prover.
        let fingerprint = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                FINGERPRINT,
                self.fingerprint_variables,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        let offset = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OFFSET)
            .into_inner();
        BusChallenges {
            fingerprint,
            offset,
        }
    }

    fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // Product verification runs inside the same explicit delegation bracket.
        self.state.begin_protocol::<ProductReduction>(PRODUCT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductReduction>(PRODUCT);
        output
    }

    fn finish(self) {
        // A complete replay consumes the entire empty typed wire.
        self.state
            .finalize()
            .expect("the bus argument reads an empty wire");
    }

    fn abort(&mut self) {
        // Release the outer typed completeness check after delegated verification rejects.
        self.state.abort();
    }
}
