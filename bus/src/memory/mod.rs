//! Read-only offline memory checking over the binary-native bus.
//!
//! Every array entry is pushed once at unit count and pulled once at the count it reached.
//!
//! Every read pulls its current count and pushes the next generator multiple of it.
//!
//! Balance proves membership when every read count is nonzero and fewer reads occur than the generator orbit allows.
//!
//! The read helper enforces both obligations itself.
//!
//! It constrains every count against a supplied inverse, and its named-array handle refuses a field whose orbit a representable row count could span.
//!
//! Declarations are therefore sound under any reduction proving the bus balance, including the two-tree one a whole plan derives.
//!
//! Seeding and finalizing the array is left to the declaring AIR, which must place every entry at its own verifier-derived generator power.
//!
//! The materialized path here carries no AIR constraints, so a third product tree rejects a zero read count in their place.
//!
//! Reduction returns leaf claims that still require commitment authentication.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::num::NonZeroUsize;

use num_bigint::BigUint;
use p3_challenger::FieldChallenger;
use p3_challenger::fs::{DomainSeparator, FieldUnit, InteractionPattern, TranscriptField};
use p3_field::{Dup, ExtensionField, Field};
use p3_security::SecurityTerm;
use p3_security::bus::{BusSecurityModel, ProductGkrSecurityProfile};

use crate::leaf::equality_weights;
use crate::{
    BusActivation, BusDirection, BusInteractionBuilder, BusPlan, BusTupleSlot, ProductGkrOutput,
    ProductGkrProof, ProductGkrRootShape, ProductGkrShape,
};

mod error;

pub use error::ReadOnlyMemoryError;

/// Version byte bound into the memory statement seed.
const STATEMENT_VERSION: u8 = 1;

/// Protocol name bound into the memory statement seed.
const STATEMENT_NAME: &[u8] = b"p3-bus-read-only-memory";

/// Named read-only array whose count orbit no representable row count can span.
///
/// A shorter orbit would let a full forged count cycle balance the bus on its own, and a declaration cannot see how many rows will exist.
///
/// Every read declaration goes through this handle so that bound is checked once, before any AIR can name the array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlyMemoryBus<F: Field> {
    /// Channel shared by every declaration of this array.
    name: String,
    /// Bind the checked generator orbit to this handle.
    marker: PhantomData<fn() -> F>,
}

impl<F: Field> ReadOnlyMemoryBus<F> {
    /// Names one read-only array and checks its field against the forged-cycle bound.
    ///
    /// # Errors
    ///
    /// Returns an error when a machine-word row count could span the whole count orbit.
    pub fn new(name: &str) -> Result<Self, ReadOnlyMemoryError> {
        // Only an orbit no row index can reach is safe without knowing the trace heights.
        let orbit = F::order() - BigUint::from(1u8);
        if orbit <= BigUint::from(usize::MAX) {
            return Err(ReadOnlyMemoryError::CountOrbitReachable {
                orbit_len: usize::try_from(orbit).expect("a rejected orbit fits in a machine word"),
            });
        }

        Ok(Self {
            name: name.to_string(),
            marker: PhantomData,
        })
    }

    /// Channel shared by every declaration of this array.
    #[must_use]
    pub fn name(&self) -> &str {
        // The enclosing bus plan groups declarations by this name.
        &self.name
    }
}

/// AIR interface for one read-only array access.
///
/// A read consumes its current count and produces the next generator-orbit count.
///
/// These declarations are not the leaves the reduction proves, only the tuple layout and the read count it covers.
pub trait ReadOnlyMemoryInteractionBuilder: BusInteractionBuilder
where
    Self::F: Field,
{
    /// Declares one paired read transition on a checked array handle.
    ///
    /// Every row of the declaring trace issues exactly one read.
    ///
    /// The supplied inverse is constrained against the count, which rules out the self-cancelling read a zero count would otherwise contribute to both directions.
    ///
    /// Conditional reads are unsupported because the materialized leaves carry no selector.
    fn read_only_memory(
        &mut self,
        bus: &ReadOnlyMemoryBus<Self::F>,
        address: Self::Expr,
        count: Self::Expr,
        count_inverse: Self::Expr,
        values: impl IntoIterator<Item = Self::Expr>,
    ) {
        // An invertible count differs from its generator multiple, so the two sides cannot cancel.
        self.assert_one(count.dup() * count_inverse);

        // Retain value expressions once so both directions use identical payloads.
        let values = values.into_iter().collect::<Vec<_>>();
        let pull = core::iter::once(address.dup())
            .chain(core::iter::once(count.dup()))
            .chain(values.iter().map(Dup::dup));
        self.push_bus_interaction(bus.name(), BusDirection::Pull, pull, BusActivation::Always);

        // Multiplying by the full-order generator advances one logical count.
        let push = core::iter::once(address)
            .chain(core::iter::once(count * Self::F::GENERATOR))
            .chain(values);
        self.push_bus_interaction(bus.name(), BusDirection::Push, push, BusActivation::Always);
    }
}

impl<T> ReadOnlyMemoryInteractionBuilder for T
where
    T: BusInteractionBuilder,
    T::F: Field,
{
}

/// Value components of the seeded array, one borrowed column per component.
#[derive(Clone, Copy, Debug)]
pub struct MemoryTableValues<'a, F>(pub &'a [&'a [F]]);

/// Count reached by each array entry after all reads.
#[derive(Clone, Copy, Debug)]
pub struct MemoryFinalCounts<'a, F>(pub &'a [F]);

/// Address returned for each read event.
#[derive(Clone, Copy, Debug)]
pub struct MemoryReadAddresses<'a, F>(pub &'a [F]);

/// Count held by each read event before it advances.
#[derive(Clone, Copy, Debug)]
pub struct MemoryReadCounts<'a, F>(pub &'a [F]);

/// Value components returned by the reads, one borrowed column per component.
#[derive(Clone, Copy, Debug)]
pub struct MemoryReadValues<'a, F>(pub &'a [&'a [F]]);

/// Borrowed columns for one read-only array and all reads made from it.
///
/// The first two are indexed by array entry and the last three by read event.
///
/// Each role has its own wrapper because two columns of equal height would otherwise be interchangeable at the call site.
#[derive(Clone, Copy, Debug)]
pub struct ReadOnlyMemoryColumns<'a, F> {
    /// Value components of the seeded array.
    pub table: MemoryTableValues<'a, F>,
    /// Count reached by each array entry after all reads.
    pub final_counts: MemoryFinalCounts<'a, F>,
    /// Address returned for each read event.
    pub read_addresses: MemoryReadAddresses<'a, F>,
    /// Count before each read event advances it.
    pub read_counts: MemoryReadCounts<'a, F>,
    /// Value components returned for each read event.
    pub read_values: MemoryReadValues<'a, F>,
}

/// Product leaves for bus balance and the read-count nonzero check.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ReadOnlyMemoryLeaves<EF> {
    /// Factors produced by seeds and count-advancing reads.
    pushes: Vec<EF>,
    /// Factors consumed by reads and finalization.
    pulls: Vec<EF>,
    /// Read counts whose product must not vanish.
    counts: Vec<EF>,
}

impl<EF: Field> ReadOnlyMemoryLeaves<EF> {
    /// Returns the three product inputs in their protocol order.
    ///
    /// The first two roots must be equal and the final root must be nonzero.
    fn product_inputs(&self) -> [&[EF]; 3] {
        // Keep the semantic order fixed for product-root sharing.
        [&self.pushes, &self.pulls, &self.counts]
    }

    /// Checks the two deterministic root obligations before proving.
    fn check_products(&self) -> Result<(), ReadOnlyMemoryError> {
        // Bus balance is equality of the first two product roots.
        let push_product = self.pushes.iter().copied().product::<EF>();
        let pull_product = self.pulls.iter().copied().product::<EF>();
        if push_product != pull_product {
            return Err(ReadOnlyMemoryError::UnbalancedProducts);
        }

        // A field product is nonzero exactly when every factor is nonzero.
        let count_product = self.counts.iter().copied().product::<EF>();
        if count_product == EF::ZERO {
            return Err(ReadOnlyMemoryError::ZeroCountProduct);
        }

        Ok(())
    }
}

/// Verifier challenges defining every memory leaf factor.
///
/// Both are drawn from a transcript that already carries the public dimensions of the statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlyMemoryChallenges<EF> {
    /// Point evaluating the padded tuple's multilinear extension.
    pub fingerprint: Vec<EF>,
    /// Random shift applied to every tuple fingerprint.
    pub offset: EF,
}

/// Authenticated claims still owed after the product reduction.
///
/// Every evaluation belongs to a distinct table padded with ones up to the shared product height, and the three prefixes differ.
///
/// Read factors start at a nonzero index on both bus sides while their counts start at index zero.
///
/// Authenticating a count against the bus-side prefix accepts a value unrelated to the counts that appear in the bus factors.
///
/// The address of every seed and finalization factor is a verifier-derived power of the field generator, never a committed column.
///
/// An all-equal committed address column would make the array multiset-valued and let one read return any stored value.
#[must_use = "leaf claims must be tied to committed columns"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlyMemoryClaims<EF> {
    /// Challenges that defined the reduced leaf polynomials.
    pub challenges: ReadOnlyMemoryChallenges<EF>,
    /// Shared multilinear point for all three leaf tables.
    pub point: Vec<EF>,
    /// Evaluation of the produced-factor table.
    pub push: EF,
    /// Evaluation of the consumed-factor table.
    pub pull: EF,
    /// Evaluation of the read-count table.
    pub counts: EF,
    /// Non-padding prefix length of each table, in the order the product inputs are given.
    pub prefix_lens: [usize; 3],
    /// Index of the first read factor on both bus sides, which is index zero of the count table.
    pub read_offset: usize,
}

/// Verifier-derived layout and orbit bounds for one read-only array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlyMemoryPlan<F: Field> {
    /// Meaning of every padded tuple slot for this named array.
    tuple_slots: Vec<BusTupleSlot>,
    /// Number of seeded array entries.
    table_len: usize,
    /// Number of read events.
    read_len: usize,
    /// Number of field components in one array value.
    value_width: usize,
    /// Variables in the tuple fingerprint table.
    tuple_variables: usize,
    /// Product reduction for push, pull, and count leaves.
    product_shape: ProductGkrShape,
    /// Bind the checked generator and field order to this plan.
    marker: PhantomData<fn() -> F>,
}

impl<F: Field> ReadOnlyMemoryPlan<F> {
    /// Derives a read-only memory statement from one named bus.
    ///
    /// Payload slots are interpreted as address, count, then value components, all over powers of the field's multiplicative generator.
    ///
    /// The table size may fill the generator orbit exactly, while the read count must be strictly smaller so that a forged count cycle cannot wrap.
    ///
    /// The read count must also equal the number of rows declared on that bus in each direction.
    ///
    /// That rejects a statement unrelated to the reads the AIRs declare, and a bus already carrying seed or finalization declarations of its own.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing bus, a malformed tuple, a read count the bus does not declare, or an unsafe orbit bound.
    pub fn new(
        bus_plan: &BusPlan,
        bus_name: &str,
        table_len: usize,
        read_len: usize,
    ) -> Result<Self, ReadOnlyMemoryError> {
        // A nonempty seed set defines the valid address and value pairs.
        if table_len == 0 {
            return Err(ReadOnlyMemoryError::EmptyTable);
        }

        // Stable bus identities come from the shared verifier-derived plan.
        let bus = bus_plan
            .domains()
            .iter()
            .position(|domain| domain.name == bus_name)
            .ok_or_else(|| ReadOnlyMemoryError::UnknownBus {
                name: bus_name.to_string(),
            })?;
        let payload_width = bus_plan.domains()[bus].payload_width;
        if payload_width < 2 {
            return Err(ReadOnlyMemoryError::PayloadTooNarrow {
                name: bus_name.to_string(),
                actual: payload_width,
                minimum: 2,
            });
        }

        // The helper declares one pull and one push per read, so each side must total the read count.
        for direction in [BusDirection::Push, BusDirection::Pull] {
            let declared = bus_plan
                .blocks(direction)
                .iter()
                .filter(|block| block.bus == bus)
                .try_fold(0usize, |total, block| {
                    total.checked_add(1usize << block.log_height)
                })
                .ok_or(ReadOnlyMemoryError::FactorCountOverflow)?;
            if declared != read_len {
                return Err(ReadOnlyMemoryError::DeclaredReadCountMismatch {
                    name: bus_name.to_string(),
                    expected: read_len,
                    actual: declared,
                });
            }
        }

        // The field contract makes its distinguished element a full-order generator.
        let orbit = F::order() - BigUint::from(1u8);
        if BigUint::from(table_len) > orbit {
            return Err(ReadOnlyMemoryError::AddressOrbitTooShort {
                table_len,
                orbit_len: usize::try_from(orbit)
                    .expect("a rejected orbit is smaller than a machine-word table length"),
            });
        }
        if BigUint::from(read_len) >= orbit {
            return Err(ReadOnlyMemoryError::CountOrbitTooShort {
                read_len,
                orbit_len: usize::try_from(orbit)
                    .expect("a rejected orbit is no larger than a machine-word read length"),
            });
        }

        // Seeds and reads occupy one factor each on both multiset sides.
        let factor_count = table_len
            .checked_add(read_len)
            .ok_or(ReadOnlyMemoryError::FactorCountOverflow)?;
        let capacity = factor_count
            .checked_next_power_of_two()
            .ok_or(ReadOnlyMemoryError::FactorCountOverflow)?;
        let log_height = capacity.trailing_zeros() as usize;
        let product_shape =
            ProductGkrShape::new(log_height, 3, ProductGkrRootShape::FirstTwoShared)
                .expect("a machine-word factor count yields a valid three-tree shape");

        Ok(Self {
            tuple_slots: (0..bus_plan.fingerprint_width())
                .map(|slot| {
                    bus_plan
                        .tuple_slot(bus, slot)
                        .expect("the slot range comes from the checked bus plan")
                })
                .collect(),
            table_len,
            read_len,
            value_width: payload_width - 2,
            tuple_variables: bus_plan.fingerprint_width().trailing_zeros() as usize,
            product_shape,
            marker: PhantomData,
        })
    }

    /// Number of seeded array entries.
    #[must_use]
    pub const fn table_len(&self) -> usize {
        // Expose the verifier-derived dimension without witness data.
        self.table_len
    }

    /// Number of read events.
    #[must_use]
    pub const fn read_len(&self) -> usize {
        // Expose the verifier-derived dimension without witness data.
        self.read_len
    }

    /// Number of field components in one array value.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        // The first two payload slots are reserved for address and count.
        self.value_width
    }

    /// Binds every public dimension of this statement into a transcript.
    ///
    /// The reduction transcript otherwise separates only on the padded tree height, so two statements whose factor counts round to the same power of two would share it.
    fn observe_statement<Challenger>(&self, challenger: &mut Challenger)
    where
        F: TranscriptField,
        Challenger: FieldChallenger<F>,
    {
        // An empty step sequence contributes a seed and no message schedule.
        let mut separator = DomainSeparator::<FieldUnit<F>>::new(
            STATEMENT_VERSION,
            STATEMENT_NAME,
            InteractionPattern::new(Vec::new()).expect("an empty step sequence is well formed"),
        );

        // Length-delimited bytes bind each dimension injectively even in characteristic two.
        for dimension in [
            self.table_len,
            self.read_len,
            self.value_width,
            self.tuple_variables,
        ] {
            separator.instance(&(dimension as u64).to_le_bytes());
        }
        separator.seed(challenger);
    }

    /// Materializes the two bus sides and the count-product leaves.
    ///
    /// The named-bus identity comes from the enclosing bus plan, and direction stays structural through separate output vectors.
    ///
    /// # Errors
    ///
    /// Returns an error when witness columns disagree with public statement dimensions.
    fn materialize<EF>(
        &self,
        columns: ReadOnlyMemoryColumns<'_, F>,
        challenges: &ReadOnlyMemoryChallenges<EF>,
    ) -> Result<ReadOnlyMemoryLeaves<EF>, ReadOnlyMemoryError>
    where
        EF: ExtensionField<F>,
    {
        // Validate every borrowed slice before allocating challenge-sized tables.
        self.validate_columns(columns)?;
        debug_assert_eq!(challenges.fingerprint.len(), self.tuple_variables);

        // One equality table supplies the coefficient of every padded tuple slot.
        let weights = equality_weights(&challenges.fingerprint);
        debug_assert_eq!(weights.len(), self.tuple_slots.len());

        // Fixed domain bits contribute the same fingerprint term to every row.
        let domain_term = weights
            .iter()
            .enumerate()
            .filter_map(|(slot, &weight)| {
                matches!(self.tuple_slots[slot], BusTupleSlot::DomainBit(true)).then_some(weight)
            })
            .sum::<EF>();

        // Payload positions are direct because every named bus is left-aligned.
        let address_weight = weights[0];
        let count_weight = weights[1];
        let value_weights = &weights[2..2 + self.value_width];
        let factor = |address: F, count: F, values: &[&[F]], row: usize| {
            let value_term = values
                .iter()
                .zip(value_weights)
                .map(|(column, &weight)| weight * column[row])
                .sum::<EF>();
            let fingerprint =
                domain_term + address_weight * address + count_weight * count + value_term;
            challenges.offset - fingerprint
        };

        // Both bus sides contain one boundary row per entry and one row per read.
        let factor_count = self.table_len + self.read_len;
        let mut pushes = Vec::with_capacity(factor_count);
        let mut pulls = Vec::with_capacity(factor_count);
        let mut counts = Vec::with_capacity(self.read_len);

        // Seed and finalize each valid pair at matching generator-derived addresses.
        let mut address = F::ONE;
        for row in 0..self.table_len {
            pushes.push(factor(address, F::ONE, columns.table.0, row));
            pulls.push(factor(
                address,
                columns.final_counts.0[row],
                columns.table.0,
                row,
            ));
            address *= F::GENERATOR;
        }

        // A read consumes its current count and produces the next orbit element.
        for row in 0..self.read_len {
            let address = columns.read_addresses.0[row];
            let count = columns.read_counts.0[row];
            pulls.push(factor(address, count, columns.read_values.0, row));
            pushes.push(factor(
                address,
                F::GENERATOR * count,
                columns.read_values.0,
                row,
            ));
            counts.push(EF::from(count));
        }

        Ok(ReadOnlyMemoryLeaves {
            pushes,
            pulls,
            counts,
        })
    }

    /// Binds this statement, reduces the three trees, and returns the claims still owed.
    ///
    /// Challenges are drawn here rather than supplied, so no caller can fingerprint the witness before its dimensions are bound.
    ///
    /// # Errors
    ///
    /// Returns an error when the columns disagree with the statement or the leaves fail their deterministic root obligations.
    pub fn prove<EF, Challenger>(
        &self,
        columns: ReadOnlyMemoryColumns<'_, F>,
        challenger: &mut Challenger,
    ) -> Result<(ProductGkrProof<EF>, ReadOnlyMemoryClaims<EF>), ReadOnlyMemoryError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Reject a malformed witness before it can perturb the shared transcript.
        self.validate_columns(columns)?;
        let challenges = self.sample_challenges(challenger);
        let leaves = self.materialize(columns, &challenges)?;

        // The product prover panics on a false shared-root statement, so reject one here.
        leaves.check_products()?;

        // Leaf lengths come from the statement itself, so they always fit the checked shape.
        debug_assert_eq!(
            leaves.product_inputs().map(<[_]>::len),
            [
                self.table_len + self.read_len,
                self.table_len + self.read_len,
                self.read_len
            ]
        );
        let (proof, output) = ProductGkrProof::prove::<F, _>(
            &leaves.product_inputs(),
            self.product_shape,
            challenger,
        );
        Ok((proof, self.claims(challenges, output)?))
    }

    /// Binds this statement, verifies the three-tree reduction, and returns the claims still owed.
    ///
    /// This checks bus balance and the nonzero count root, and authenticates no leaf evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed proof, a failed reduction, or a failed root obligation.
    pub fn verify<EF, Challenger>(
        &self,
        proof: &ProductGkrProof<EF>,
        challenger: &mut Challenger,
    ) -> Result<ReadOnlyMemoryClaims<EF>, ReadOnlyMemoryError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Prover and verifier bind the same dimensions before replaying the reduction.
        let challenges = self.sample_challenges(challenger);
        let output = proof.verify::<F, _>(self.product_shape, challenger)?;
        self.claims(challenges, output)
    }

    /// Binds this statement and draws the fingerprint challenges from the bound transcript.
    fn sample_challenges<EF, Challenger>(
        &self,
        challenger: &mut Challenger,
    ) -> ReadOnlyMemoryChallenges<EF>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Sampling lives inside the reduction so the ordering cannot be skipped.
        self.observe_statement(challenger);
        ReadOnlyMemoryChallenges {
            fingerprint: (0..self.tuple_variables)
                .map(|_| challenger.sample_algebra_element())
                .collect(),
            offset: challenger.sample_algebra_element(),
        }
    }

    /// Builds the union-bound term consumed by a protocol security report.
    ///
    /// The result covers tuple compression and the three-tree product reduction, while count-orbit safety and root checks are deterministic.
    ///
    /// Commitment binding and leaf-claim authentication remain caller obligations.
    #[must_use]
    pub fn security_term(&self, field_bits: NonZeroUsize) -> SecurityTerm {
        // Compose every random experiment as one protocol extra.
        self.security_model(field_bits).combined_term()
    }

    /// Builds separately labelled terms for diagnostic reporting.
    ///
    /// These terms describe one union bound and must not be charged again separately.
    #[must_use]
    pub fn security_components(&self, field_bits: NonZeroUsize) -> Vec<SecurityTerm> {
        // Preserve labels so parameter reports show every error source.
        self.security_model(field_bits).components()
    }

    /// Converts a verified three-tree reduction into claims for commitment authentication.
    fn claims<EF: ExtensionField<F>>(
        &self,
        challenges: ReadOnlyMemoryChallenges<EF>,
        output: ProductGkrOutput<EF>,
    ) -> Result<ReadOnlyMemoryClaims<EF>, ReadOnlyMemoryError> {
        // Attacker-controlled vector lengths must be checked before indexing.
        if output.roots.len() != 3 {
            return Err(ReadOnlyMemoryError::RootCountMismatch {
                expected: 3,
                actual: output.roots.len(),
            });
        }
        if output.values.len() != 3 {
            return Err(ReadOnlyMemoryError::LeafClaimCountMismatch {
                expected: 3,
                actual: output.values.len(),
            });
        }
        if output.point.len() != self.product_shape.log_height() {
            return Err(ReadOnlyMemoryError::ReductionPointDimensionMismatch {
                expected: self.product_shape.log_height(),
                actual: output.point.len(),
            });
        }

        // The shared-root encoding of this plan already forces the two bus roots to agree.
        // This check is therefore defensive against an output produced under a different root shape.
        if output.roots[0] != output.roots[1] {
            return Err(ReadOnlyMemoryError::UnbalancedProducts);
        }
        if output.roots[2] == EF::ZERO {
            return Err(ReadOnlyMemoryError::ZeroCountProduct);
        }

        // Each tree pads a different prefix, so the obligation travels with the claims.
        let bus_prefix = self.table_len + self.read_len;
        Ok(ReadOnlyMemoryClaims {
            challenges,
            point: output.point,
            push: output.values[0],
            pull: output.values[1],
            counts: output.values[2],
            prefix_lens: [bus_prefix, bus_prefix, self.read_len],
            read_offset: self.table_len,
        })
    }

    /// Checks all borrowed column dimensions before leaf generation.
    fn validate_columns(
        &self,
        columns: ReadOnlyMemoryColumns<'_, F>,
    ) -> Result<(), ReadOnlyMemoryError> {
        // Table and read values must carry the tuple's declared component count.
        if columns.table.0.len() != self.value_width {
            return Err(ReadOnlyMemoryError::TableWidthMismatch {
                expected: self.value_width,
                actual: columns.table.0.len(),
            });
        }
        if columns.read_values.0.len() != self.value_width {
            return Err(ReadOnlyMemoryError::ReadWidthMismatch {
                expected: self.value_width,
                actual: columns.read_values.0.len(),
            });
        }

        // Each value component spans its corresponding public row domain.
        for (column, values) in columns.table.0.iter().enumerate() {
            if values.len() != self.table_len {
                return Err(ReadOnlyMemoryError::TableHeightMismatch {
                    column,
                    expected: self.table_len,
                    actual: values.len(),
                });
            }
        }
        for (column, values) in columns.read_values.0.iter().enumerate() {
            if values.len() != self.read_len {
                return Err(ReadOnlyMemoryError::ReadHeightMismatch {
                    column,
                    expected: self.read_len,
                    actual: values.len(),
                });
            }
        }

        // Scalar metadata columns follow the same public dimensions.
        if columns.read_addresses.0.len() != self.read_len {
            return Err(ReadOnlyMemoryError::AddressHeightMismatch {
                expected: self.read_len,
                actual: columns.read_addresses.0.len(),
            });
        }
        if columns.read_counts.0.len() != self.read_len {
            return Err(ReadOnlyMemoryError::CountHeightMismatch {
                expected: self.read_len,
                actual: columns.read_counts.0.len(),
            });
        }
        if columns.final_counts.0.len() != self.table_len {
            return Err(ReadOnlyMemoryError::FinalCountHeightMismatch {
                expected: self.table_len,
                actual: columns.final_counts.0.len(),
            });
        }

        Ok(())
    }

    /// Derives exact random-error dimensions from the executed product schedule.
    fn security_model(&self, field_bits: NonZeroUsize) -> BusSecurityModel {
        // Radix-four layers contribute their actual sumcheck and collapse counts.
        let layers = self.product_shape.layers();
        let sumcheck_rounds = layers.iter().map(|(_, rounds)| rounds).sum();
        let collapse_challenges = layers
            .iter()
            .map(|(arity, _)| arity.trailing_zeros() as usize)
            .sum();
        let profile = ProductGkrSecurityProfile::new(
            self.product_shape.log_height(),
            self.product_shape.num_trees(),
            sumcheck_rounds,
            layers.len(),
            collapse_challenges,
        )
        .expect("a checked memory plan has valid product security dimensions");

        // Both multiset sides contain exactly one factor per seed or read.
        let factors = self.table_len + self.read_len;
        BusSecurityModel::new(
            field_bits.get(),
            self.tuple_variables,
            [factors, factors],
            profile,
        )
        .expect("a checked memory plan has valid bus security dimensions")
    }
}

#[cfg(test)]
mod tests;
