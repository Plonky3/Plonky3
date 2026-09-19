//! Read-only offline memory checking over the binary-native bus.
//!
//! Every array entry pushes `(address, 1, value)` once.
//! Every read pulls `(address, count, value)` and pushes `(address, g * count, value)`.
//! Every array entry finally pulls `(address, final_count, value)` once.
//!
//! Balance proves membership when every read count is nonzero and fewer than `ord(g)` reads occur.
//!
//! Product reduction returns leaf claims that still require commitment authentication.

use alloc::string::ToString;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::num::NonZeroUsize;

use num_bigint::BigUint;
use p3_field::{Dup, ExtensionField, Field, PrimeCharacteristicRing};
use p3_security::SecurityTerm;
use p3_security::bus::{BusSecurityModel, ProductGkrSecurityProfile};

use crate::{BusActivation, BusDirection, BusInteractionBuilder, RecordToken};
use crate::{BusPlan, BusTupleSlot, ProductGkrOutput, ProductGkrRootShape, ProductGkrShape};

mod error;

pub use error::ReadOnlyMemoryError;

/// AIR interface for one read-only array access.
///
/// A read consumes its current count and produces the next generator-orbit count.
/// Both interactions retain structural direction metadata in characteristic two.
pub trait ReadOnlyMemoryInteractionBuilder: BusInteractionBuilder
where
    Self::F: Field,
{
    /// Declares one paired read transition on a named array bus.
    ///
    /// A conditional activation is constrained to zero or one exactly once.
    fn read_only_memory(
        &mut self,
        bus_name: &str,
        address: Self::Expr,
        count: Self::Expr,
        values: impl IntoIterator<Item = Self::Expr>,
        activation: BusActivation<Self::Expr>,
    ) {
        // One activation controls both sides of the same semantic read.
        if let BusActivation::Boolean(selector) = &activation {
            self.assert_zero(selector.dup().bool_check());
        }

        // Retain value expressions once so both directions use identical payloads.
        let values = values.into_iter().collect::<Vec<_>>();
        let pull = core::iter::once(address.dup())
            .chain(core::iter::once(count.dup()))
            .chain(values.iter().map(Dup::dup));
        let pull_activation = match &activation {
            BusActivation::Always => BusActivation::Always,
            BusActivation::Boolean(selector) => BusActivation::Boolean(selector.dup()),
        };
        self.record_bus_interaction(
            RecordToken(()),
            bus_name,
            BusDirection::Pull,
            pull,
            pull_activation,
        );

        // Multiplying by the full-order generator advances one logical count.
        let push = core::iter::once(address)
            .chain(core::iter::once(count * Self::F::GENERATOR))
            .chain(values);
        self.record_bus_interaction(
            RecordToken(()),
            bus_name,
            BusDirection::Push,
            push,
            activation,
        );
    }
}

impl<T> ReadOnlyMemoryInteractionBuilder for T
where
    T: BusInteractionBuilder,
    T::F: Field,
{
}

/// Borrowed columns for one read-only array and all reads made from it.
#[derive(Clone, Copy, Debug)]
pub struct ReadOnlyMemoryColumns<'a, F> {
    /// Value components of the seeded array.
    pub table: &'a [&'a [F]],
    /// Address returned for each read event.
    pub read_addresses: &'a [F],
    /// Count before each read event advances it.
    pub read_counts: &'a [F],
    /// Value components returned for each read event.
    pub read_values: &'a [&'a [F]],
    /// Count reached by each array entry after all reads.
    pub final_counts: &'a [F],
}

/// Product leaves for bus balance and the read-count nonzero check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlyMemoryLeaves<EF> {
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
    /// The first two roots must be equal.
    /// The final root must be nonzero.
    /// Callers should check both conditions before invoking a prover that assumes shared roots.
    #[must_use]
    pub fn product_inputs(&self) -> [&[EF]; 3] {
        // Keep the semantic order fixed for product-root sharing.
        [&self.pushes, &self.pulls, &self.counts]
    }

    /// Checks the two deterministic root obligations before proving.
    ///
    /// # Errors
    ///
    /// Returns an error when bus products differ or a read count is zero.
    pub fn check_products(&self) -> Result<(), ReadOnlyMemoryError> {
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

/// Authenticated claims still owed after the product reduction.
#[must_use = "leaf claims must be tied to committed columns"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlyMemoryClaims<EF> {
    /// Shared multilinear point for all three leaf tables.
    pub point: Vec<EF>,
    /// Evaluation of the produced-factor table.
    pub push: EF,
    /// Evaluation of the consumed-factor table.
    pub pull: EF,
    /// Evaluation of the read-count table.
    pub counts: EF,
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
    /// Payload slots are interpreted as address, count, then value components.
    /// Array addresses and counts use powers of the field's multiplicative generator.
    ///
    /// The table size may fill the generator orbit exactly.
    /// The read count must be strictly smaller than that orbit.
    /// This strict bound prevents a forged count cycle from wrapping.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing bus, a malformed tuple, or an unsafe orbit bound.
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

    /// Checked product shape for push, pull, and count trees.
    #[must_use]
    pub const fn product_shape(&self) -> ProductGkrShape {
        // The root encoding forces equality of the first two trees.
        self.product_shape
    }

    /// Materializes the two bus sides and the count-product leaves.
    ///
    /// The named-bus identity comes from the enclosing bus plan.
    /// Direction remains structural metadata through separate output vectors.
    /// Both fingerprint challenges must be sampled after every source column is committed.
    ///
    /// # Errors
    ///
    /// Returns an error when witness columns disagree with public statement dimensions.
    pub fn materialize<EF>(
        &self,
        columns: ReadOnlyMemoryColumns<'_, F>,
        fingerprint_point: &[EF],
        offset: EF,
    ) -> Result<ReadOnlyMemoryLeaves<EF>, ReadOnlyMemoryError>
    where
        EF: ExtensionField<F>,
    {
        // Validate every borrowed slice before allocating challenge-sized tables.
        self.validate_columns(columns)?;
        if fingerprint_point.len() != self.tuple_variables {
            return Err(ReadOnlyMemoryError::FingerprintDimensionMismatch {
                expected: self.tuple_variables,
                actual: fingerprint_point.len(),
            });
        }

        // One equality table supplies the coefficient of every padded tuple slot.
        let weights = equality_weights(fingerprint_point);
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
            offset - fingerprint
        };

        // Both bus sides contain one boundary row per entry and one row per read.
        let factor_count = self.table_len + self.read_len;
        let mut pushes = Vec::with_capacity(factor_count);
        let mut pulls = Vec::with_capacity(factor_count);
        let mut counts = Vec::with_capacity(self.read_len);

        // Seed and finalize each valid pair at matching generator-derived addresses.
        let mut address = F::ONE;
        for row in 0..self.table_len {
            pushes.push(factor(address, F::ONE, columns.table, row));
            pulls.push(factor(
                address,
                columns.final_counts[row],
                columns.table,
                row,
            ));
            address *= F::GENERATOR;
        }

        // A read consumes its current count and produces the next orbit element.
        for row in 0..self.read_len {
            let address = columns.read_addresses[row];
            let count = columns.read_counts[row];
            pulls.push(factor(address, count, columns.read_values, row));
            pushes.push(factor(
                address,
                F::GENERATOR * count,
                columns.read_values,
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

    /// Converts a verified three-tree reduction into claims for commitment authentication.
    ///
    /// This checks bus balance and the nonzero count root.
    /// It does not authenticate any returned leaf evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed dimensions or failed root obligations.
    pub fn claims<EF: ExtensionField<F>>(
        &self,
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

        // Root sharing is structural in a verified reduction.
        if output.roots[0] != output.roots[1] {
            return Err(ReadOnlyMemoryError::UnbalancedProducts);
        }
        if output.roots[2] == EF::ZERO {
            return Err(ReadOnlyMemoryError::ZeroCountProduct);
        }

        Ok(ReadOnlyMemoryClaims {
            point: output.point,
            push: output.values[0],
            pull: output.values[1],
            counts: output.values[2],
        })
    }

    /// Builds the union-bound term consumed by a protocol security report.
    ///
    /// The result covers tuple compression and the three-tree product reduction.
    /// Count-orbit safety and root checks are deterministic obligations.
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

    /// Checks all borrowed column dimensions before leaf generation.
    fn validate_columns(
        &self,
        columns: ReadOnlyMemoryColumns<'_, F>,
    ) -> Result<(), ReadOnlyMemoryError> {
        // Table and read values must carry the tuple's declared component count.
        if columns.table.len() != self.value_width {
            return Err(ReadOnlyMemoryError::TableWidthMismatch {
                expected: self.value_width,
                actual: columns.table.len(),
            });
        }
        if columns.read_values.len() != self.value_width {
            return Err(ReadOnlyMemoryError::ReadWidthMismatch {
                expected: self.value_width,
                actual: columns.read_values.len(),
            });
        }

        // Each value component spans its corresponding public row domain.
        for (column, values) in columns.table.iter().enumerate() {
            if values.len() != self.table_len {
                return Err(ReadOnlyMemoryError::TableHeightMismatch {
                    column,
                    expected: self.table_len,
                    actual: values.len(),
                });
            }
        }
        for (column, values) in columns.read_values.iter().enumerate() {
            if values.len() != self.read_len {
                return Err(ReadOnlyMemoryError::ReadHeightMismatch {
                    column,
                    expected: self.read_len,
                    actual: values.len(),
                });
            }
        }

        // Scalar metadata columns follow the same public dimensions.
        if columns.read_addresses.len() != self.read_len {
            return Err(ReadOnlyMemoryError::AddressHeightMismatch {
                expected: self.read_len,
                actual: columns.read_addresses.len(),
            });
        }
        if columns.read_counts.len() != self.read_len {
            return Err(ReadOnlyMemoryError::CountHeightMismatch {
                expected: self.read_len,
                actual: columns.read_counts.len(),
            });
        }
        if columns.final_counts.len() != self.table_len {
            return Err(ReadOnlyMemoryError::FinalCountHeightMismatch {
                expected: self.table_len,
                actual: columns.final_counts.len(),
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

/// Evaluates the Boolean-cube equality polynomial at every tuple slot.
fn equality_weights<F: Field>(point: &[F]) -> Vec<F> {
    // An empty point addresses the sole slot of a width-one tuple.
    let mut weights = alloc::vec![F::ONE];

    for &coordinate in point {
        // Existing coordinates remain the low-order address bits.
        let old_len = weights.len();
        weights.resize(old_len * 2, F::ZERO);
        for index in 0..old_len {
            let weight = weights[index];
            weights[index] = weight * (F::ONE - coordinate);
            weights[old_len + index] = weight * coordinate;
        }
    }

    weights
}

#[cfg(test)]
mod tests;
