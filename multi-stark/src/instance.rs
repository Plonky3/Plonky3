//! Public instance wrappers for batched multi-STARK proving and verification.
//!
//! Setup takes only AIR references, since public values and main trace tables are proof-time data.
//! A proving instance adds the committed main trace table, the shared proving key, and public values.
//! A verifying instance adds the shared verifying key, the trace height, and public values.

use alloc::vec::Vec;
use core::ops::Deref;

use p3_air::BaseAir;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};

use crate::config::MultiStarkConfig;
pub use crate::keys::{ProvingKey, VerifyingKey, setup};
pub use crate::proof::MultiStarkProof;
pub use crate::prover::prove;
pub use crate::verifier::{VerificationError, verify};

/// Build an opening protocol for a sequence of committed tables.
///
/// Every table contributes one opening batch: all current-row columns plus the
/// successor-view columns declared by its AIR.
pub(super) fn opening_protocol<I>(tables: I) -> OpeningProtocol
where
    I: IntoIterator<Item = (usize, usize, Vec<usize>)>,
{
    OpeningProtocol::new(
        tables
            .into_iter()
            .map(|(log_height, width, next_columns)| {
                TableSpec::new(
                    TableShape::new(log_height, width),
                    alloc::vec![OpeningBatch::new(
                        (0..width).collect::<Vec<_>>(),
                        next_columns.to_vec(),
                    )],
                )
            })
            .collect(),
    )
}

/// One AIR statement proved inside a batched committed proof.
pub struct ProverInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// AIR whose constraints are proved.
    air: &'a A,
    /// Already-transposed trace table: one row per AIR column.
    table: Table<C::Val>,
    /// Proving key carrying optional reusable preprocessed data.
    proving_key: &'a ProvingKey<C>,
    /// Public values forwarded to the AIR.
    public_values: &'a [C::Val],
}

/// Collection of prover-side AIR instances proved with one shared proof.
///
/// All contained instances must use the same proving key. Main trace tables are
/// committed in this collection's order.
pub struct ProverInstances<'a, C, A>(Vec<ProverInstance<'a, C, A>>)
where
    C: MultiStarkConfig;

/// One AIR statement checked inside a batched proof.
pub struct VerifierInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// AIR whose constraints are checked.
    air: &'a A,
    /// Verifying key carrying optional reusable preprocessed data.
    verifying_key: &'a VerifyingKey<C>,
    /// Base-two logarithm of this instance's main trace height.
    num_variables: usize,
    /// Public values forwarded to the AIR.
    public_values: &'a [C::Val],
}

/// Collection of verifier-side AIR instances checked against one shared proof.
///
/// All contained instances must use the same verifying key. The order must match
/// the prover-side instance order used to create the proof.
pub struct VerifierInstances<'a, C, A>(Vec<VerifierInstance<'a, C, A>>)
where
    C: MultiStarkConfig;

pub(super) struct Instance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// AIR whose constraints are proved.
    pub(super) air: &'a A,
    /// Public values forwarded to the AIR.
    public_values: &'a [C::Val],
    /// Base-two logarithm of this instance's main trace height.
    num_variables: usize,
}

pub(super) struct Instances<'a, C, A>(Vec<Instance<'a, C, A>>)
where
    C: MultiStarkConfig;

/// A prover-side batch split into the parts the proving flow consumes.
pub(super) struct ProverParts<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Proving key shared by every instance in the batch.
    pub(super) proving_key: &'a ProvingKey<C>,
    /// Owned main trace tables, in batch order, ready to commit.
    pub(super) tables: Vec<Table<C::Val>>,
    /// Per-instance metadata retained for the zerocheck and openings.
    pub(super) instances: Instances<'a, C, A>,
}

impl<'a, C, A> ProverInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Create a prover-side AIR instance from proof-time data.
    ///
    /// The table must already be transposed into the multilinear layout: one row
    /// per AIR column.
    pub const fn new(
        air: &'a A,
        table: Table<C::Val>,
        proving_key: &'a ProvingKey<C>,
        public_values: &'a [C::Val],
    ) -> Self {
        Self {
            air,
            table,
            proving_key,
            public_values,
        }
    }
}

impl<'a, C, A> VerifierInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Create a verifier-side AIR instance from verification-time data.
    ///
    /// `num_variables` is the base-two logarithm of this instance's main trace height.
    pub const fn new(
        air: &'a A,
        verifying_key: &'a VerifyingKey<C>,
        num_variables: usize,
        public_values: &'a [C::Val],
    ) -> Self {
        Self {
            air,
            verifying_key,
            num_variables,
            public_values,
        }
    }
}

impl<'a, C, A> Deref for ProverInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    type Target = [ProverInstance<'a, C, A>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, C, A> Deref for VerifierInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    type Target = [VerifierInstance<'a, C, A>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, C, A> Deref for Instances<'a, C, A>
where
    C: MultiStarkConfig,
{
    type Target = [Instance<'a, C, A>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, C, A> ProverInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Create a prover-side batch in proof order.
    ///
    /// The order must match the AIR order used at setup.
    /// Main and preprocessed tables are committed and opened in this order.
    pub const fn new(instances: Vec<ProverInstance<'a, C, A>>) -> Self {
        Self(instances)
    }

    pub(super) fn proving_key(&self) -> &'a ProvingKey<C> {
        // Every instance must reference one shared proving key.
        // Adjacent-pair equality is transitively all-equal.
        assert!(
            self.0
                .windows(2)
                .all(|pair| core::ptr::eq(pair[0].proving_key, pair[1].proving_key)),
            "all prover instances must use the same proving key"
        );
        self.0
            .first()
            .expect("prover instances cannot be empty")
            .proving_key
    }

    pub(super) fn into_parts(self) -> ProverParts<'a, C, A> {
        let proving_key = self.proving_key();

        // One pass over the owned instances.
        // Read the borrowed metadata, then move the table out.
        let mut instances = Vec::with_capacity(self.0.len());
        let mut tables = Vec::with_capacity(self.0.len());
        for instance in self.0 {
            instances.push(Instance {
                air: instance.air,
                public_values: instance.public_values,
                num_variables: instance.table.num_variables(),
            });
            tables.push(instance.table);
        }

        ProverParts {
            proving_key,
            tables,
            instances: Instances(instances),
        }
    }
}

impl<'a, C, A> VerifierInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Create a verifier-side batch in proof order.
    ///
    /// The order must match the prover-side batch order and the AIR order used at setup.
    /// Openings are replayed in this order.
    pub const fn new(instances: Vec<VerifierInstance<'a, C, A>>) -> Self {
        Self(instances)
    }

    pub(super) fn verifying_key(&self) -> &'a VerifyingKey<C> {
        // Every instance must reference one shared verifying key.
        // Adjacent-pair equality is transitively all-equal.
        assert!(
            self.0
                .windows(2)
                .all(|pair| core::ptr::eq(pair[0].verifying_key, pair[1].verifying_key)),
            "all verifier instances must use the same verifying key"
        );
        self.0
            .first()
            .expect("verifier instances cannot be empty")
            .verifying_key
    }

    pub(super) fn into_parts(self) -> (&'a VerifyingKey<C>, Instances<'a, C, A>) {
        let verifying_key = self.verifying_key();
        let instances = self
            .0
            .into_iter()
            .map(|instance| Instance {
                air: instance.air,
                public_values: instance.public_values,
                num_variables: instance.num_variables,
            })
            .collect();

        (verifying_key, Instances(instances))
    }
}

impl<'a, C, A> Instances<'a, C, A>
where
    C: MultiStarkConfig,
    A: BaseAir<C::Val>,
{
    pub(super) fn num_variables(&self) -> Vec<usize> {
        self.0
            .iter()
            .map(|instance| instance.num_variables)
            .collect()
    }

    pub(super) fn widths(&self) -> Vec<usize> {
        self.0.iter().map(|instance| instance.air.width()).collect()
    }

    pub(super) fn next_columns(&self) -> Vec<Vec<usize>> {
        self.0
            .iter()
            .map(|instance| instance.air.main_next_row_columns())
            .collect()
    }

    pub(super) fn airs(&self) -> Vec<&A> {
        self.0
            .iter()
            .map(|instance| instance.air)
            .collect::<Vec<_>>()
    }

    pub(super) fn public_values(&self) -> Vec<&[C::Val]> {
        self.0
            .iter()
            .map(|instance| instance.public_values)
            .collect()
    }

    pub(super) fn opening_protocol(&self) -> OpeningProtocol {
        opening_protocol(
            self.num_variables()
                .iter()
                .zip(self.widths().iter())
                .zip(self.next_columns())
                .map(|((&log_height, &width), next_columns)| (log_height, width, next_columns)),
        )
    }

    pub(super) fn preprocessed_opening_protocol(&self) -> OpeningProtocol {
        opening_protocol(
            self.iter()
                .filter(|instance| instance.air.preprocessed_width() != 0)
                .map(|instance| {
                    (
                        instance.num_variables,
                        instance.air.preprocessed_width(),
                        instance.air.preprocessed_next_row_columns(),
                    )
                }),
        )
    }

    pub(super) fn preprocessed_next_columns(&self) -> Vec<Vec<usize>> {
        self.iter()
            .filter(|instance| instance.air.preprocessed_width() != 0)
            .map(|instance| instance.air.preprocessed_next_row_columns())
            .collect()
    }

    pub(super) fn max_num_variables(&self) -> usize {
        self.num_variables().iter().cloned().max().unwrap()
    }

    pub(super) fn main_points(&self, point: &Point<C::Challenge>) -> Vec<Point<C::Challenge>> {
        let max_num_var = self.max_num_variables();
        self.num_variables()
            .iter()
            .map(|num_var| point.split_at(max_num_var - num_var).1)
            .collect()
    }

    pub(super) fn preprocessed_points(
        &self,
        point: &Point<C::Challenge>,
    ) -> Vec<Point<C::Challenge>> {
        let max_num_var = self.max_num_variables();
        self.iter()
            .filter(|instance| instance.air.preprocessed_width() != 0)
            .map(|instance| point.split_at(max_num_var - instance.num_variables).1)
            .collect()
    }
}
