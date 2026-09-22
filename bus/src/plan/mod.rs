//! Verifier-derived layouts for binary-native bus declarations.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Reverse;

use hashbrown::HashSet;
use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_util::log2_ceil_usize;

mod error;

pub use error::BusPlanError;

use crate::{BusDirection, BusName, ProductGkrRootShape, ProductGkrShape, SymbolicBusInteraction};

/// Symbolic bus declarations belonging to one AIR instance.
#[derive(Clone, Copy, Debug)]
pub struct BusPlanInput<'a, F: Field> {
    /// Base-two logarithm of the instance's trace height.
    pub log_height: usize,
    /// Declarations emitted by the instance in AIR order.
    pub interactions: &'a [SymbolicBusInteraction<F>],
}

/// Stable owner of one materialized declaration block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusBlockOwner {
    /// AIR position in the statement.
    pub air: usize,
    /// Declaration position within that AIR.
    pub declaration: usize,
}

/// One aligned block in a direction-specific product tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusBlock {
    /// Named-bus position in lexicographic name order.
    pub bus: usize,
    /// Side of the multiset equality containing the block.
    pub direction: BusDirection,
    /// AIR and declaration that own the block.
    pub owner: BusBlockOwner,
    /// Base-two logarithm of the block height, shared by every block one AIR emits.
    pub log_height: usize,
    /// First leaf in the direction-specific tree.
    pub offset: usize,
}

/// One named bus and its automatically assigned tuple-domain identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusDomain {
    /// Name shared by every declaration in this group.
    pub name: String,
    /// Number of payload expressions required by this group.
    pub payload_width: usize,
    /// Nonzero identity encoded into the reserved domain slots.
    pub identity: usize,
}

impl BusDomain {
    /// Checked channel name of this group.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is outside the alphabet.
    ///
    /// Every field here is public, so this is reachable for a domain built by hand rather than by planning.
    pub const fn bus_name(&self) -> Result<BusName<'_>, crate::BusNameError> {
        BusName::try_new(self.name.as_str())
    }

    /// Read one little-endian bit of the nonzero domain identity.
    #[must_use]
    pub const fn identity_bit(&self, bit: usize) -> bool {
        bit < usize::BITS as usize && ((self.identity >> bit) & 1) != 0
    }
}

/// One block's contribution to the shifted terminal leaf evaluation.
///
/// Identity padding disappears after subtracting one from every leaf factor.
/// The authenticated relation is therefore:
///
/// `L(q) - 1 = sum_b eq(prefix_b, q_prefix) * (L_b(q_rows) - 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusTerminalShare {
    /// Named-bus position in lexicographic name order.
    pub bus: usize,
    /// Side of the multiset equality containing the share.
    pub direction: BusDirection,
    /// AIR and declaration whose expression rebuilds the share.
    pub owner: BusBlockOwner,
    /// Variables addressing rows inside the owning declaration.
    pub row_variables: usize,
    /// Leading variables selecting this block from the logical leaf tree.
    pub prefix_variables: usize,
    /// Boolean vertex selected by the leading variables.
    pub prefix_index: usize,
}

impl BusTerminalShare {
    /// Evaluate this block's selector at a terminal product-tree point.
    ///
    /// Point coordinates and vertex bits are interpreted most-significant first.
    ///
    /// Returns no value when the point or public share fields are inconsistent.
    #[must_use]
    pub fn prefix_weight<F: Field>(&self, point: &[F]) -> Option<F> {
        // The leading coordinates address the aligned block containing this share.
        let prefix = point.get(..self.prefix_variables)?;
        // Coordinate zero binds the most significant bit of the block address.
        Point::new(prefix).equality_at_vertex(self.prefix_index)
    }
}

/// Exact security-relevant dimensions of one bus plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusSecurityGeometry {
    /// Variables in the power-of-two tuple fingerprint table.
    tuple_variables: usize,
    /// Non-padding push and pull leaf positions.
    non_padding_leaf_counts: [usize; 2],
    /// Power-of-two capacity shared by both product trees.
    logical_leaf_count: usize,
    /// Variables in each product tree.
    log_logical_leaf_count: usize,
    /// Product trees reduced in lockstep.
    tree_count: usize,
    /// Root-to-leaf product-reduction layers.
    layer_count: usize,
}

impl BusSecurityGeometry {
    /// Variables in the power-of-two tuple fingerprint table.
    #[must_use]
    pub const fn tuple_variables(&self) -> usize {
        self.tuple_variables
    }

    /// Non-padding push and pull leaf positions.
    #[must_use]
    pub const fn non_padding_leaf_counts(&self) -> [usize; 2] {
        self.non_padding_leaf_counts
    }

    /// Non-padding leaf positions on one side of the multiset equality.
    #[must_use]
    pub const fn non_padding_leaf_count(&self, direction: BusDirection) -> usize {
        self.non_padding_leaf_counts[direction.index()]
    }

    /// Power-of-two capacity shared by both product trees.
    #[must_use]
    pub const fn logical_leaf_count(&self) -> usize {
        self.logical_leaf_count
    }

    /// Variables in each product tree.
    #[must_use]
    pub const fn log_logical_leaf_count(&self) -> usize {
        self.log_logical_leaf_count
    }

    /// Product trees reduced in lockstep.
    #[must_use]
    pub const fn tree_count(&self) -> usize {
        self.tree_count
    }

    /// Root-to-leaf product-reduction layers.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
}

/// Meaning of one slot in a padded fingerprint tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusTupleSlot {
    /// Payload expression at the enclosed bus-local position.
    Payload(usize),
    /// Fixed bit of the named-bus domain identity.
    DomainBit(bool),
    /// Constant zero used for width equalization or power-of-two padding.
    Zero,
}

/// A verifier-derived layout for every named binary-native bus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusPlan {
    /// Named groups in lexicographic name order.
    domains: Vec<BusDomain>,
    /// Payload positions reserved before the domain identity bits.
    payload_slots: usize,
    /// Bits reserved for a nonzero named-bus identity.
    domain_slots: usize,
    /// Payload and domain slots before zero padding.
    logical_tuple_width: usize,
    /// Power-of-two width consumed by tuple fingerprinting.
    fingerprint_width: usize,
    /// Push blocks in physical leaf order.
    pushes: Vec<BusBlock>,
    /// Pull blocks in physical leaf order.
    pulls: Vec<BusBlock>,
    /// Dimensions consumed by security accounting.
    geometry: BusSecurityGeometry,
    /// Product-reduction shape checked while the plan is built.
    product_shape: ProductGkrShape,
}

impl BusPlan {
    /// Build a deterministic layout from trusted symbolic AIR declarations.
    ///
    /// Empty batches produce no plan.
    ///
    /// Named groups are independent of AIR caller order.
    ///
    /// Blocks are ordered by descending height before their named domain.
    ///
    /// This keeps every mixed-height block aligned to its own Boolean subcube.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed tuple shapes, oversized layouts, or unsupported accesses.
    pub fn build<F: Field>(inputs: &[BusPlanInput<'_, F>]) -> Result<Option<Self>, BusPlanError> {
        let declaration_count = inputs
            .iter()
            .try_fold(0usize, |count, input| {
                count.checked_add(input.interactions.len())
            })
            .ok_or(BusPlanError::DeclarationCountOverflow)?;
        if declaration_count == 0 {
            return Ok(None);
        }

        let mut widths = BTreeMap::<String, usize>::new();
        for (air, input) in inputs.iter().enumerate() {
            if input.log_height >= usize::BITS as usize {
                return Err(BusPlanError::HeightOverflow { air });
            }
            for (declaration, interaction) in input.interactions.iter().enumerate() {
                if interaction.fields.is_empty() {
                    return Err(BusPlanError::EmptyTuple { air, declaration });
                }

                // The plan owns channel identity, so it rechecks every name it is handed.
                // A profile built through the declaration surface always passes.
                interaction
                    .bus()
                    .map_err(|source| BusPlanError::InvalidBusName {
                        air,
                        declaration,
                        source,
                    })?;
                validate_interaction(air, declaration, interaction)?;

                match widths.get(&interaction.bus_name) {
                    Some(&expected) if expected != interaction.fields.len() => {
                        return Err(BusPlanError::PayloadWidthMismatch {
                            name: interaction.bus_name.clone(),
                            expected,
                            actual: interaction.fields.len(),
                        });
                    }
                    Some(_) => {}
                    None => {
                        widths.insert(interaction.bus_name.clone(), interaction.fields.len());
                    }
                }
            }
        }

        let domain_count = widths.len();
        let identity_limit = domain_count
            .checked_add(1)
            .ok_or(BusPlanError::DomainCountOverflow)?;
        let domain_slots = log2_ceil_usize(identity_limit);
        let payload_slots = widths.values().copied().max().unwrap_or(0);
        let logical_tuple_width = payload_slots
            .checked_add(domain_slots)
            .ok_or(BusPlanError::TupleWidthOverflow)?;
        let fingerprint_width = logical_tuple_width
            .checked_next_power_of_two()
            .ok_or(BusPlanError::TupleWidthOverflow)?;
        let tuple_variables = fingerprint_width.trailing_zeros() as usize;

        let domains = widths
            .into_iter()
            .enumerate()
            .map(|(bus, (name, payload_width))| BusDomain {
                name,
                payload_width,
                identity: bus + 1,
            })
            .collect::<Vec<_>>();
        let indices = domains
            .iter()
            .enumerate()
            .map(|(index, domain)| (domain.name.as_str(), index))
            .collect::<BTreeMap<_, _>>();

        let mut pending = Vec::with_capacity(declaration_count);
        for (air, input) in inputs.iter().enumerate() {
            for (declaration, interaction) in input.interactions.iter().enumerate() {
                pending.push(PendingBlock {
                    bus: indices[interaction.bus_name.as_str()],
                    direction: interaction.direction,
                    owner: BusBlockOwner { air, declaration },
                    log_height: input.log_height,
                });
            }
        }

        pending.sort_by_key(|block| {
            (
                block.direction.index(),
                Reverse(block.log_height),
                block.bus,
                block.owner.air,
                block.owner.declaration,
            )
        });

        let mut pushes = Vec::new();
        let mut pulls = Vec::new();
        let mut non_padding_leaf_counts = [0usize; 2];
        for block in pending {
            let side = block.direction.index();
            let len = 1usize << block.log_height;
            let offset = non_padding_leaf_counts[side];
            debug_assert_eq!(offset % len, 0);
            non_padding_leaf_counts[side] =
                offset
                    .checked_add(len)
                    .ok_or(BusPlanError::LeafCountOverflow {
                        direction: block.direction,
                    })?;
            let block = BusBlock {
                bus: block.bus,
                direction: block.direction,
                owner: block.owner,
                log_height: block.log_height,
                offset,
            };
            match block.direction {
                BusDirection::Push => pushes.push(block),
                BusDirection::Pull => pulls.push(block),
            }
        }

        let (largest_direction, used) = if non_padding_leaf_counts[0] >= non_padding_leaf_counts[1]
        {
            (BusDirection::Push, non_padding_leaf_counts[0])
        } else {
            (BusDirection::Pull, non_padding_leaf_counts[1])
        };
        let used = used.max(1);
        let logical_leaf_count =
            used.checked_next_power_of_two()
                .ok_or(BusPlanError::LeafCountOverflow {
                    direction: largest_direction,
                })?;
        let log_logical_leaf_count = logical_leaf_count.trailing_zeros() as usize;
        let product_shape = ProductGkrShape::new(
            log_logical_leaf_count,
            2,
            ProductGkrRootShape::FirstTwoShared,
        )?;
        let geometry = BusSecurityGeometry {
            tuple_variables,
            non_padding_leaf_counts,
            logical_leaf_count,
            log_logical_leaf_count,
            tree_count: product_shape.num_trees(),
            layer_count: product_shape.layers().len(),
        };

        Ok(Some(Self {
            domains,
            payload_slots,
            domain_slots,
            logical_tuple_width,
            fingerprint_width,
            pushes,
            pulls,
            geometry,
            product_shape,
        }))
    }

    /// Named bus domains in their stable identity order.
    #[must_use]
    pub fn domains(&self) -> &[BusDomain] {
        &self.domains
    }

    /// Position of one channel in this plan's identity order.
    ///
    /// This is the index every other method here takes as its bus argument.
    ///
    /// Resolving a channel through the plan, rather than by scanning its domains, keeps identity where the verifier assigns it.
    ///
    /// Returns no value when this statement declares nothing on that channel.
    #[must_use]
    pub fn domain_index(&self, bus: BusName<'_>) -> Option<usize> {
        // Domains are sorted by name, so the lookup is a binary search rather than a scan.
        self.domains
            .binary_search_by(|domain| domain.name.as_str().cmp(bus.as_str()))
            .ok()
    }

    /// Identity and payload width of one channel.
    ///
    /// Returns no value when this statement declares nothing on that channel.
    #[must_use]
    pub fn domain(&self, bus: BusName<'_>) -> Option<&BusDomain> {
        self.domain_index(bus).map(|index| &self.domains[index])
    }

    /// Maximum payload width reserved in every fingerprint tuple.
    #[must_use]
    pub const fn payload_slots(&self) -> usize {
        self.payload_slots
    }

    /// Number of reserved named-domain identity bits.
    #[must_use]
    pub const fn domain_slots(&self) -> usize {
        self.domain_slots
    }

    /// Payload and named-domain slots before power-of-two padding.
    #[must_use]
    pub const fn logical_tuple_width(&self) -> usize {
        self.logical_tuple_width
    }

    /// Power-of-two tuple width consumed by the fingerprint MLE.
    #[must_use]
    pub const fn fingerprint_width(&self) -> usize {
        self.fingerprint_width
    }

    /// Physical blocks for one side of the multiset equality.
    #[must_use]
    pub fn blocks(&self, direction: BusDirection) -> &[BusBlock] {
        match direction {
            BusDirection::Push => &self.pushes,
            BusDirection::Pull => &self.pulls,
        }
    }

    /// Meaning of one fingerprint slot for one named bus.
    ///
    /// Returns no value when either index lies outside this plan.
    #[must_use]
    pub fn tuple_slot(&self, bus: usize, slot: usize) -> Option<BusTupleSlot> {
        let domain = self.domains.get(bus)?;
        if slot >= self.fingerprint_width {
            return None;
        }
        if slot < domain.payload_width {
            return Some(BusTupleSlot::Payload(slot));
        }
        if slot < self.payload_slots {
            return Some(BusTupleSlot::Zero);
        }
        if slot < self.logical_tuple_width {
            return Some(BusTupleSlot::DomainBit(
                domain.identity_bit(slot - self.payload_slots),
            ));
        }
        Some(BusTupleSlot::Zero)
    }

    /// Terminal shares in the product tree's physical block order.
    ///
    /// The leading point coordinates select one aligned block.
    ///
    /// The trailing coordinates evaluate the owning AIR expression over its rows.
    ///
    /// Each owning expression contributes its leaf factor minus one.
    ///
    /// Materialized declarations must follow this exact order on each direction.
    pub fn terminal_shares(
        &self,
        direction: BusDirection,
    ) -> impl ExactSizeIterator<Item = BusTerminalShare> + '_ {
        self.blocks(direction).iter().map(|block| BusTerminalShare {
            bus: block.bus,
            direction: block.direction,
            owner: block.owner,
            row_variables: block.log_height,
            prefix_variables: self.geometry.log_logical_leaf_count - block.log_height,
            prefix_index: block.offset >> block.log_height,
        })
    }

    /// Exact dimensions for later soundness accounting.
    #[must_use]
    pub const fn security_geometry(&self) -> BusSecurityGeometry {
        self.geometry
    }

    /// Checked product-reduction shape fixed by this layout.
    #[must_use]
    pub const fn product_shape(&self) -> ProductGkrShape {
        self.product_shape
    }
}

/// Symbolic expression position rejected during planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusExpressionLocation {
    /// Tuple payload expression at this position.
    Field(usize),
    /// Boolean row-activation expression.
    Activation,
}

/// Expression access unavailable to the later opening-based evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedBusAccess {
    /// A main-trace row other than the current row.
    MainOffset(usize),
    /// A preprocessed-trace row other than the current row.
    PreprocessedOffset(usize),
    /// A periodic column whose period is absent from this plan.
    Periodic,
}

#[derive(Clone, Copy, Debug)]
struct PendingBlock {
    /// Named-bus position in lexicographic name order.
    bus: usize,
    /// Side of the multiset equality receiving the block.
    direction: BusDirection,
    /// AIR and declaration that own the block.
    owner: BusBlockOwner,
    /// Base-two logarithm of the block height, taken from the table not the declaration.
    log_height: usize,
}

fn validate_interaction<F: Field>(
    air: usize,
    declaration: usize,
    interaction: &SymbolicBusInteraction<F>,
) -> Result<(), BusPlanError> {
    for (field, expression) in interaction.fields.iter().enumerate() {
        validate_expression(
            air,
            declaration,
            BusExpressionLocation::Field(field),
            expression,
        )?;
    }
    // A boundary indicator names no expression, so only a caller selector needs validating.
    if let crate::BusActivation::Boolean(expression) = &interaction.activation {
        validate_expression(
            air,
            declaration,
            BusExpressionLocation::Activation,
            expression,
        )?;
    }
    Ok(())
}

fn validate_expression<F: Field>(
    air: usize,
    declaration: usize,
    location: BusExpressionLocation,
    expression: &SymbolicExpression<F>,
) -> Result<(), BusPlanError> {
    // Arithmetic nodes share their operands, so the expression is a graph rather than a tree.
    // Walking it per path costs time exponential in the depth, which a bit recomposition reaches immediately.
    let mut seen = HashSet::<*const SymbolicExpression<F>>::new();
    let mut pending = alloc::vec![expression];
    while let Some(expression) = pending.pop() {
        if !seen.insert(core::ptr::from_ref(expression)) {
            continue;
        }
        match expression {
            SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => {
                let access = match variable.entry {
                    BaseEntry::Main { offset: 0 }
                    | BaseEntry::Preprocessed { offset: 0 }
                    | BaseEntry::Public => None,
                    BaseEntry::Main { offset } => Some(UnsupportedBusAccess::MainOffset(offset)),
                    BaseEntry::Preprocessed { offset } => {
                        Some(UnsupportedBusAccess::PreprocessedOffset(offset))
                    }
                    BaseEntry::Periodic => Some(UnsupportedBusAccess::Periodic),
                };
                if let Some(access) = access {
                    return Err(BusPlanError::UnsupportedExpression {
                        air,
                        declaration,
                        location,
                        access,
                    });
                }
            }
            SymbolicExpr::Leaf(_) => {}
            SymbolicExpr::Add { x, y, .. }
            | SymbolicExpr::Sub { x, y, .. }
            | SymbolicExpr::Mul { x, y, .. } => {
                pending.push(x);
                pending.push(y);
            }
            SymbolicExpr::Neg { x, .. } => pending.push(x),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;
    use crate::BusActivation;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn variable(entry: BaseEntry, index: usize) -> SymbolicExpression<F> {
        SymbolicVariable::new(entry, index).into()
    }

    fn interaction(name: &str, direction: BusDirection, width: usize) -> SymbolicBusInteraction<F> {
        SymbolicBusInteraction {
            bus_name: name.to_string(),
            direction,
            fields: (0..width)
                .map(|index| variable(BaseEntry::Main { offset: 0 }, index))
                .collect(),
            activation: BusActivation::Always,
        }
    }

    fn layout_signature(plan: &BusPlan) -> Vec<(BusDirection, usize, usize, usize)> {
        BusDirection::ALL
            .into_iter()
            .flat_map(|direction| {
                plan.blocks(direction)
                    .iter()
                    .map(move |block| (direction, block.bus, block.log_height, block.offset))
            })
            .collect()
    }

    fn evaluate<T: Field>(values: &[T], point: &[T]) -> T {
        assert_eq!(values.len(), 1usize << point.len());
        values
            .iter()
            .enumerate()
            .map(|(vertex, &value)| {
                point
                    .iter()
                    .enumerate()
                    .map(|(coordinate, &challenge)| {
                        let bit = (vertex >> (point.len() - 1 - coordinate)) & 1;
                        if bit == 0 {
                            T::ONE - challenge
                        } else {
                            challenge
                        }
                    })
                    .product::<T>()
                    * value
            })
            .sum()
    }

    /// Build a deterministic prime-field transcript for the protocol integration test.
    fn challenger() -> DuplexChallenger<F, Poseidon2BabyBear<16>, 16, 8> {
        // Matching seeds keep the prover and verifier transcript streams identical.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0xB055_700D);
        DuplexChallenger::new(Poseidon2BabyBear::new_from_rng_128(&mut rng))
    }

    #[test]
    fn scalar_layout_matches_aligned_mixed_height_reference() {
        let tall = vec![
            interaction("memory", BusDirection::Push, 3),
            interaction("dispatch", BusDirection::Pull, 1),
        ];
        let short = vec![
            interaction("dispatch", BusDirection::Push, 1),
            interaction("memory", BusDirection::Pull, 3),
        ];
        let plan = BusPlan::build(&[
            BusPlanInput {
                log_height: 4,
                interactions: &tall,
            },
            BusPlanInput {
                log_height: 2,
                interactions: &short,
            },
        ])
        .unwrap()
        .unwrap();

        assert_eq!(plan.domains()[0].name, "dispatch");
        assert_eq!(plan.domains()[0].identity, 1);
        assert_eq!(plan.domains()[1].name, "memory");
        assert_eq!(plan.domains()[1].identity, 2);
        assert_eq!(plan.payload_slots(), 3);
        assert_eq!(plan.domain_slots(), 2);
        assert_eq!(plan.logical_tuple_width(), 5);
        assert_eq!(plan.fingerprint_width(), 8);
        assert_eq!(plan.tuple_slot(0, 0), Some(BusTupleSlot::Payload(0)));
        assert_eq!(plan.tuple_slot(0, 1), Some(BusTupleSlot::Zero));
        assert_eq!(plan.tuple_slot(0, 3), Some(BusTupleSlot::DomainBit(true)));
        assert_eq!(plan.tuple_slot(0, 7), Some(BusTupleSlot::Zero));
        assert_eq!(plan.tuple_slot(0, 8), None);
        assert_eq!(plan.tuple_slot(2, 0), None);

        assert_eq!(
            layout_signature(&plan),
            vec![
                (BusDirection::Push, 1, 4, 0),
                (BusDirection::Push, 0, 2, 16),
                (BusDirection::Pull, 0, 4, 0),
                (BusDirection::Pull, 1, 2, 16),
            ]
        );
        assert_eq!(
            plan.security_geometry(),
            BusSecurityGeometry {
                tuple_variables: 3,
                non_padding_leaf_counts: [20, 20],
                logical_leaf_count: 32,
                log_logical_leaf_count: 5,
                tree_count: 2,
                layer_count: 3,
            }
        );
    }

    #[test]
    fn input_permutations_keep_domains_geometry_and_physical_layout() {
        let a = vec![interaction("z", BusDirection::Pull, 2)];
        let b = vec![interaction("a", BusDirection::Push, 1)];
        let c = vec![interaction("z", BusDirection::Push, 2)];
        let inputs = [
            BusPlanInput {
                log_height: 1,
                interactions: &a,
            },
            BusPlanInput {
                log_height: 4,
                interactions: &b,
            },
            BusPlanInput {
                log_height: 2,
                interactions: &c,
            },
        ];
        let expected = BusPlan::build(&inputs).unwrap().unwrap();

        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let permuted = order.map(|index| inputs[index]);
            let actual = BusPlan::build(&permuted).unwrap().unwrap();
            assert_eq!(actual.domains(), expected.domains());
            assert_eq!(actual.security_geometry(), expected.security_geometry());
            assert_eq!(layout_signature(&actual), layout_signature(&expected));
        }
    }

    #[test]
    fn ownership_and_direction_survive_physical_reordering() {
        let first = vec![interaction("bus", BusDirection::Pull, 1)];
        let second = vec![interaction("bus", BusDirection::Push, 1)];
        let plan = BusPlan::build(&[
            BusPlanInput {
                log_height: 1,
                interactions: &first,
            },
            BusPlanInput {
                log_height: 3,
                interactions: &second,
            },
        ])
        .unwrap()
        .unwrap();

        assert_eq!(
            plan.blocks(BusDirection::Push)[0].owner,
            BusBlockOwner {
                air: 1,
                declaration: 0,
            }
        );
        assert_eq!(
            plan.blocks(BusDirection::Pull)[0].owner,
            BusBlockOwner {
                air: 0,
                declaration: 0,
            }
        );

        assert_eq!(
            plan.terminal_shares(BusDirection::Push).collect::<Vec<_>>(),
            vec![BusTerminalShare {
                bus: 0,
                direction: BusDirection::Push,
                owner: BusBlockOwner {
                    air: 1,
                    declaration: 0,
                },
                row_variables: 3,
                prefix_variables: 0,
                prefix_index: 0,
            }]
        );
    }

    #[test]
    fn terminal_shares_match_the_identity_padded_scalar_table() {
        let tall = vec![interaction("bus", BusDirection::Push, 1)];
        let short = vec![interaction("bus", BusDirection::Push, 1)];
        let plan = BusPlan::build(&[
            BusPlanInput {
                log_height: 2,
                interactions: &tall,
            },
            BusPlanInput {
                log_height: 1,
                interactions: &short,
            },
        ])
        .unwrap()
        .unwrap();

        let point = [F::from_u8(2), F::from_u8(3), F::from_u8(4)];
        let dense = [
            F::from_u8(5),
            F::from_u8(6),
            F::from_u8(7),
            F::from_u8(8),
            F::from_u8(9),
            F::from_u8(10),
            F::ONE,
            F::ONE,
        ];
        let blocks = [&dense[..4], &dense[4..6]];

        let reconstructed = plan
            .terminal_shares(BusDirection::Push)
            .zip(blocks)
            .map(|(share, block)| {
                let row_point = &point[share.prefix_variables..];
                let weight = share.prefix_weight(&point).unwrap();
                weight * (evaluate(block, row_point) - F::ONE)
            })
            .sum::<F>();

        assert_eq!(reconstructed, evaluate(&dense, &point) - F::ONE);
    }

    #[test]
    fn planned_block_order_closes_on_the_real_product_reduction() {
        // Two mixed-height owners each emit matching push and pull declarations.
        let tall = vec![
            interaction("bus", BusDirection::Push, 1),
            interaction("bus", BusDirection::Pull, 1),
        ];
        let short = tall.clone();
        let inputs = [
            BusPlanInput {
                log_height: 2,
                interactions: &tall,
            },
            BusPlanInput {
                log_height: 1,
                interactions: &short,
            },
        ];
        let plan = BusPlan::build(&inputs).unwrap().unwrap();
        let fingerprint_point = [EF::from_u8(7)];
        let offset = EF::from_u8(11);

        // Materialize declarations in the plan's physical order on each direction.
        let weights = crate::BusChallenges {
            fingerprint: fingerprint_point.to_vec(),
            offset,
        }
        .fingerprint_weights();
        let materialize_direction = |direction| {
            let mut leaves = Vec::new();
            for block in plan.blocks(direction) {
                let owner = if block.owner.air == 0 { &tall } else { &short };
                let factor = plan
                    .compile_factor(block.bus, &owner[block.owner.declaration], &weights, offset)
                    .unwrap();
                let mut scratch = Vec::new();
                for row in 0..1usize << block.log_height {
                    let payload = [F::from_usize(block.owner.air * 16 + row + 2)];
                    leaves.push(
                        factor
                            .evaluate(
                                &mut scratch,
                                crate::BusEvaluation {
                                    main: &payload,
                                    preprocessed: &[],
                                    public: &[],
                                    is_first_row: F::ZERO,
                                    is_last_row: F::ZERO,
                                    is_transition: F::ZERO,
                                },
                            )
                            .unwrap(),
                    );
                }
            }
            leaves
        };
        let pushes = materialize_direction(BusDirection::Push);
        let pulls = materialize_direction(BusDirection::Pull);
        assert_eq!(pushes, pulls);

        // The real prover returns the repository-wide leading-prefix point order.
        let mut prover_challenger = challenger();
        let (proof, prover_output) = crate::ProductGkrProof::prove::<F, _>(
            &[pushes.as_slice(), pulls.as_slice()],
            plan.product_shape(),
            &mut prover_challenger,
        );
        let mut verifier_challenger = challenger();
        let verifier_output = proof
            .verify::<F, _>(plan.product_shape(), &mut verifier_challenger)
            .unwrap();
        assert_eq!(verifier_output, prover_output);

        // Terminal shares reconstruct the ones-padded leaf evaluation block by block.
        let mut dense = pushes.clone();
        dense.resize(plan.security_geometry().logical_leaf_count(), EF::ONE);
        let reconstructed = plan
            .terminal_shares(BusDirection::Push)
            .zip(plan.blocks(BusDirection::Push))
            .map(|(share, block)| {
                assert_eq!(share.owner, block.owner);
                let rows = 1usize << block.log_height;
                let values = &pushes[block.offset..block.offset + rows];
                let row_point = &prover_output.point[share.prefix_variables..];
                let prefix_weight = share.prefix_weight(&prover_output.point).unwrap();
                prefix_weight * (evaluate(values, row_point) - EF::ONE)
            })
            .sum::<EF>();
        assert_eq!(
            reconstructed,
            evaluate(&dense, &prover_output.point) - EF::ONE
        );
    }

    #[test]
    fn terminal_share_coordinates_address_prefixes_most_significant_first() {
        // Prefix index 01 selects the low half and then its high child.
        let share = BusTerminalShare {
            bus: 0,
            direction: BusDirection::Push,
            owner: BusBlockOwner {
                air: 0,
                declaration: 0,
            },
            row_variables: 1,
            prefix_variables: 2,
            prefix_index: 1,
        };
        let a = F::from_u8(2);
        let b = F::from_u8(3);

        // Trailing row coordinates do not enter the aligned-block selector.
        assert_eq!(
            share.prefix_weight(&[a, b, F::from_u8(5)]),
            Some((F::ONE - a) * b)
        );

        // Publicly constructible malformed shares fail without shifting by an invalid amount.
        let malformed = BusTerminalShare {
            prefix_index: 4,
            ..share
        };
        assert_eq!(malformed.prefix_weight(&[a, b]), None);
        assert_eq!(share.prefix_weight(&[a]), None);
    }

    #[test]
    fn malformed_shapes_and_unopenable_expressions_are_rejected() {
        let empty = vec![interaction("empty", BusDirection::Push, 0)];
        assert!(matches!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &empty,
            }]),
            Err(BusPlanError::EmptyTuple { .. })
        ));

        let narrow = vec![interaction("same", BusDirection::Push, 1)];
        let wide = vec![interaction("same", BusDirection::Pull, 2)];
        assert!(matches!(
            BusPlan::build(&[
                BusPlanInput {
                    log_height: 1,
                    interactions: &narrow,
                },
                BusPlanInput {
                    log_height: 1,
                    interactions: &wide,
                },
            ]),
            Err(BusPlanError::PayloadWidthMismatch { .. })
        ));

        for (entry, expected) in [
            (
                BaseEntry::Main { offset: 1 },
                UnsupportedBusAccess::MainOffset(1),
            ),
            (
                BaseEntry::Preprocessed { offset: 1 },
                UnsupportedBusAccess::PreprocessedOffset(1),
            ),
            (BaseEntry::Periodic, UnsupportedBusAccess::Periodic),
        ] {
            let invalid = vec![SymbolicBusInteraction {
                bus_name: "bad".to_string(),
                direction: BusDirection::Push,
                fields: vec![variable(entry, 0) + F::ONE],
                activation: BusActivation::Always,
            }];
            assert!(matches!(
                BusPlan::build(&[BusPlanInput {
                    log_height: 1,
                    interactions: &invalid,
                }]),
                Err(BusPlanError::UnsupportedExpression { access, .. }) if access == expected
            ));
        }

        // A doubling chain of this depth has two paths out of each of its nodes.
        // Re-walking the graph per path would take about thirteen seconds per declaration.
        const DEPTH: usize = 32;
        let mut supported = variable(BaseEntry::Main { offset: 0 }, 0);
        let mut unsupported = variable(BaseEntry::Periodic, 0);
        for _ in 0..DEPTH {
            supported = supported.clone() + supported;
            unsupported = unsupported.clone() + unsupported;
        }
        let deep = vec![SymbolicBusInteraction {
            bus_name: "deep".to_string(),
            direction: BusDirection::Push,
            fields: vec![supported],
            activation: BusActivation::Always,
        }];
        assert!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &deep,
            }])
            .is_ok()
        );

        // Sharing must not swallow a rejection buried under the same depth.
        let deep_invalid = vec![SymbolicBusInteraction {
            bus_name: "deep".to_string(),
            direction: BusDirection::Push,
            fields: vec![unsupported],
            activation: BusActivation::Always,
        }];
        assert!(matches!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &deep_invalid,
            }]),
            Err(BusPlanError::UnsupportedExpression {
                access: UnsupportedBusAccess::Periodic,
                ..
            })
        ));

        let enormous = vec![
            interaction("large", BusDirection::Push, 1),
            interaction("large", BusDirection::Push, 1),
        ];
        assert!(matches!(
            BusPlan::build(&[BusPlanInput {
                log_height: usize::BITS as usize - 1,
                interactions: &enormous,
            }]),
            Err(BusPlanError::LeafCountOverflow {
                direction: BusDirection::Push,
            })
        ));

        let ordinary = vec![interaction("height", BusDirection::Push, 1)];
        assert!(matches!(
            BusPlan::build(&[BusPlanInput {
                log_height: usize::BITS as usize,
                interactions: &ordinary,
            }]),
            Err(BusPlanError::HeightOverflow { air: 0 })
        ));
    }

    #[test]
    fn a_shared_operand_graph_is_walked_once_per_node_rather_than_once_per_path() {
        // Fixture state: forty doublings share their operand, giving eighty nodes and 2^40 root-to-leaf paths.
        const DEPTH: usize = 40;
        let deep = |entry: BaseEntry| {
            let mut field = variable(entry, 0);
            for _ in 0..DEPTH {
                field = field.clone() + field;
            }
            vec![SymbolicBusInteraction {
                bus_name: "deep".to_string(),
                direction: BusDirection::Push,
                fields: vec![field],
                activation: BusActivation::Always,
            }]
        };

        // A path-wise walk would not finish, so reaching the assertion at all is the property under test.
        let accepted = deep(BaseEntry::Main { offset: 0 });
        assert!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &accepted,
            }])
            .unwrap()
            .is_some()
        );

        // Skipping repeated nodes must not skip the rejection buried under the same sharing.
        let rejected = deep(BaseEntry::Main { offset: 1 });
        assert!(matches!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &rejected,
            }]),
            Err(BusPlanError::UnsupportedExpression {
                access: UnsupportedBusAccess::MainOffset(1),
                ..
            })
        ));
    }
}

#[cfg(test)]
mod width_tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::{BusActivation, BusName, BusNameError};

    type F = BabyBear;

    /// Channel names a machine of this shape would define once and import everywhere.
    const STATE: BusName<'static> = BusName::new("state");
    const MEMORY: BusName<'static> = BusName::new("memory");
    const BYTECODE: BusName<'static> = BusName::new("bytecode");
    const RANGE: BusName<'static> = BusName::new("range");

    fn declaration(
        bus: BusName<'_>,
        direction: BusDirection,
        width: usize,
    ) -> SymbolicBusInteraction<F> {
        SymbolicBusInteraction {
            bus_name: bus.as_str().to_string(),
            direction,
            fields: (0..width)
                .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into())
                .collect(),
            activation: BusActivation::Always,
        }
    }

    /// Widths and heights of the order a machine reaches.
    ///
    /// A narrow range channel, a state channel, an address-count-value memory channel, and a wide instruction channel.
    fn machine_plan() -> BusPlan {
        let cpu = [
            declaration(STATE, BusDirection::Push, 3),
            declaration(STATE, BusDirection::Pull, 3),
            declaration(BYTECODE, BusDirection::Pull, 40),
            declaration(MEMORY, BusDirection::Pull, 12),
            declaration(RANGE, BusDirection::Pull, 2),
        ];
        let memory = [
            declaration(MEMORY, BusDirection::Push, 12),
            declaration(MEMORY, BusDirection::Pull, 12),
        ];
        let tables = [declaration(BYTECODE, BusDirection::Push, 40)];
        BusPlan::build(&[
            BusPlanInput {
                log_height: 7,
                interactions: &cpu,
            },
            BusPlanInput {
                log_height: 5,
                interactions: &memory,
            },
            BusPlanInput {
                log_height: 3,
                interactions: &tables,
            },
        ])
        .unwrap()
        .unwrap()
    }

    #[test]
    fn four_channels_of_machine_width_share_one_fingerprint_table() {
        let plan = machine_plan();

        // Four names need three identity bits above the widest payload.
        assert_eq!(plan.domains().len(), 4);
        assert_eq!(plan.payload_slots(), 40);
        assert_eq!(plan.domain_slots(), 3);
        assert_eq!(plan.logical_tuple_width(), 43);

        // The padded table is the next power of two, so it costs six challenge coordinates.
        assert_eq!(plan.fingerprint_width(), 64);
        assert_eq!(plan.security_geometry().tuple_variables(), 6);
    }

    #[test]
    fn a_narrow_channel_is_zero_padded_up_to_the_widest_one() {
        let plan = machine_plan();
        let range = plan.domain_index(RANGE).unwrap();
        assert_eq!(plan.domains()[range].payload_width, 2);

        // Its own payload occupies the leading slots.
        for slot in 0..2 {
            assert_eq!(
                plan.tuple_slot(range, slot),
                Some(BusTupleSlot::Payload(slot))
            );
        }

        // Everything up to the widest payload is equalizing zero, not another channel's data.
        for slot in 2..40 {
            assert_eq!(plan.tuple_slot(range, slot), Some(BusTupleSlot::Zero));
        }

        // The identity bits follow, then the power-of-two padding.
        let identity = plan.domains()[range].identity;
        for bit in 0..3 {
            assert_eq!(
                plan.tuple_slot(range, 40 + bit),
                Some(BusTupleSlot::DomainBit((identity >> bit) & 1 == 1))
            );
        }
        for slot in 43..64 {
            assert_eq!(plan.tuple_slot(range, slot), Some(BusTupleSlot::Zero));
        }
        assert_eq!(plan.tuple_slot(range, 64), None);
    }

    #[test]
    fn no_two_channels_share_a_padded_tuple_space() {
        let plan = machine_plan();

        // Identities are distinct and nonzero.
        //
        // A tuple of one channel therefore cannot be replayed as a tuple of another, however it is padded.
        let identities = plan
            .domains()
            .iter()
            .map(|domain| domain.identity)
            .collect::<Vec<_>>();
        assert!(identities.iter().all(|&identity| identity != 0));
        for (index, &identity) in identities.iter().enumerate() {
            assert!(!identities[..index].contains(&identity));
        }

        // Every identity fits the slots reserved for it.
        assert!(identities.iter().all(|&identity| identity < 1 << 3));
    }

    #[test]
    fn a_channel_resolves_through_the_plan_that_assigned_its_identity() {
        let plan = machine_plan();

        // Identity order is lexicographic over names, not the order the AIRs declared them.
        assert_eq!(
            plan.domains()
                .iter()
                .map(|domain| domain.name.as_str())
                .collect::<Vec<_>>(),
            vec!["bytecode", "memory", "range", "state"]
        );
        for (index, bus) in [BYTECODE, MEMORY, RANGE, STATE].into_iter().enumerate() {
            assert_eq!(plan.domain_index(bus), Some(index));
            assert_eq!(plan.domain(bus).unwrap().bus_name(), Ok(bus));
            assert_eq!(plan.domain(bus).unwrap().identity, index + 1);
        }

        // A channel this statement never declares has no identity to resolve.
        assert_eq!(plan.domain_index(BusName::new("unused")), None);
        assert!(plan.domain(BusName::new("unused")).is_none());
    }

    #[test]
    fn the_plan_rechecks_every_name_it_is_handed() {
        // The declaration surface cannot produce this, but a profile built by hand can.
        let mut malformed = declaration(STATE, BusDirection::Push, 1);
        malformed.bus_name = "state machine".to_string();
        assert_eq!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &[malformed],
            }])
            .unwrap_err(),
            BusPlanError::InvalidBusName {
                air: 0,
                declaration: 0,
                source: BusNameError::Byte {
                    index: 5,
                    byte: b' ',
                },
            }
        );

        // An empty name would otherwise index a domain nothing can name.
        let mut empty = declaration(STATE, BusDirection::Push, 1);
        empty.bus_name = String::new();
        assert!(matches!(
            BusPlan::build(&[BusPlanInput {
                log_height: 1,
                interactions: &[empty],
            }]),
            Err(BusPlanError::InvalidBusName {
                source: BusNameError::Empty,
                ..
            })
        ));
    }

    #[test]
    fn one_channel_declared_at_two_widths_is_refused() {
        // Two chips that disagree on a channel's arity is the mistake the plan has to name.
        let cpu = [declaration(MEMORY, BusDirection::Pull, 12)];
        let memory = [declaration(MEMORY, BusDirection::Push, 11)];
        assert_eq!(
            BusPlan::build(&[
                BusPlanInput {
                    log_height: 3,
                    interactions: &cpu,
                },
                BusPlanInput {
                    log_height: 3,
                    interactions: &memory,
                },
            ])
            .unwrap_err(),
            BusPlanError::PayloadWidthMismatch {
                name: "memory".to_string(),
                expected: 12,
                actual: 11,
            }
        );
    }
}
