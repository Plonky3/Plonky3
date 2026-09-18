//! Verifier-derived layout for binary-native bus declarations.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Reverse;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::Field;
use thiserror::Error;

use crate::{
    BusDirection, ProductGkrRootShape, ProductGkrShape, ProductGkrShapeError,
    SymbolicBusInteraction,
};

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
    /// Base-two logarithm of the block height.
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

/// Exact security-relevant dimensions of one bus plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusSecurityGeometry {
    /// Variables in the power-of-two tuple fingerprint table.
    pub tuple_variables: usize,
    /// Non-padding push and pull leaf positions.
    pub non_padding_leaf_counts: [usize; 2],
    /// Power-of-two capacity shared by both product trees.
    pub logical_leaf_count: usize,
    /// Variables in each product tree.
    pub log_logical_leaf_count: usize,
    /// Product trees reduced in lockstep.
    pub tree_count: usize,
    /// Root-to-leaf product-reduction layers.
    pub layer_count: usize,
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
}

impl BusPlan {
    /// Build a deterministic layout from trusted symbolic AIR declarations.
    ///
    /// Empty batches produce no plan.
    /// Named groups are independent of AIR caller order.
    /// Blocks are ordered by descending height before their named domain.
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
        let domain_slots = log2_ceil(identity_limit);
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
                direction_index(block.direction),
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
            let side = direction_index(block.direction);
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
        }))
    }

    /// Named bus domains in their stable identity order.
    #[must_use]
    pub fn domains(&self) -> &[BusDomain] {
        &self.domains
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

    /// Terminal shares in the product tree's physical block order.
    ///
    /// The leading point coordinates select one aligned block.
    /// The trailing coordinates evaluate the owning AIR expression over its rows.
    /// Each owning expression contributes its leaf factor minus one.
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
    pub fn product_shape(&self) -> ProductGkrShape {
        ProductGkrShape::new(
            self.geometry.log_logical_leaf_count,
            self.geometry.tree_count,
            ProductGkrRootShape::FirstTwoShared,
        )
        .expect("a checked bus plan retains a valid product shape")
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

/// Invalid statement shapes rejected before transcript construction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusPlanError {
    /// The total number of symbolic declarations overflowed.
    #[error("binary-bus declaration count overflows usize")]
    DeclarationCountOverflow,
    /// One AIR's trace height cannot be represented.
    #[error("binary-bus AIR {air} height overflows usize")]
    HeightOverflow {
        /// AIR position in statement order.
        air: usize,
    },
    /// A tuple has no payload expression.
    #[error("binary-bus AIR {air} declaration {declaration} has an empty tuple")]
    EmptyTuple {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
    },
    /// Two declarations on one named bus disagree on payload width.
    #[error("binary bus {name} has payload widths {expected} and {actual}")]
    PayloadWidthMismatch {
        /// Shared bus name.
        name: String,
        /// Width fixed by the first declaration.
        expected: usize,
        /// Width carried by the conflicting declaration.
        actual: usize,
    },
    /// The number of named domains overflowed its nonzero encoding.
    #[error("binary-bus domain count overflows usize")]
    DomainCountOverflow,
    /// Tuple slots or their power-of-two table overflowed.
    #[error("binary-bus fingerprint tuple width overflows usize")]
    TupleWidthOverflow,
    /// One direction's materialized leaf count overflowed.
    #[error("binary-bus {direction:?} leaf count overflows usize")]
    LeafCountOverflow {
        /// Side whose blocks overflowed.
        direction: BusDirection,
    },
    /// A symbolic expression reads data the terminal evaluator cannot reconstruct.
    #[error(
        "binary-bus AIR {air} declaration {declaration} {location:?} uses unsupported {access:?}"
    )]
    UnsupportedExpression {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Payload or activation expression containing the access.
        location: BusExpressionLocation,
        /// Unsupported access encountered in the expression tree.
        access: UnsupportedBusAccess,
    },
    /// The derived product-tree shape is invalid.
    #[error(transparent)]
    ProductShape(#[from] ProductGkrShapeError),
}

#[derive(Clone, Copy, Debug)]
struct PendingBlock {
    bus: usize,
    direction: BusDirection,
    owner: BusBlockOwner,
    log_height: usize,
}

const fn direction_index(direction: BusDirection) -> usize {
    match direction {
        BusDirection::Push => 0,
        BusDirection::Pull => 1,
    }
}

const fn log2_ceil(value: usize) -> usize {
    usize::BITS as usize - value.saturating_sub(1).leading_zeros() as usize
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
    let mut pending = alloc::vec![expression];
    while let Some(expression) = pending.pop() {
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
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::BusActivation;

    type F = BabyBear;

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
        [BusDirection::Push, BusDirection::Pull]
            .into_iter()
            .flat_map(|direction| {
                plan.blocks(direction)
                    .iter()
                    .map(move |block| (direction, block.bus, block.log_height, block.offset))
            })
            .collect()
    }

    fn evaluate(values: &[F], point: &[F]) -> F {
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
                            F::ONE - challenge
                        } else {
                            challenge
                        }
                    })
                    .product::<F>()
                    * value
            })
            .sum()
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
                let prefix = &point[..share.prefix_variables];
                let row_point = &point[share.prefix_variables..];
                let prefix_vertex = (0..share.prefix_variables)
                    .map(|coordinate| {
                        ((share.prefix_index >> (share.prefix_variables - 1 - coordinate)) & 1)
                            as u8
                    })
                    .map(F::from_u8)
                    .collect::<Vec<_>>();
                let weight = prefix
                    .iter()
                    .zip(prefix_vertex)
                    .map(|(&challenge, bit)| {
                        (F::ONE - challenge) * (F::ONE - bit) + challenge * bit
                    })
                    .product::<F>();
                weight * (evaluate(block, row_point) - F::ONE)
            })
            .sum::<F>();

        assert_eq!(reconstructed, evaluate(&dense, &point) - F::ONE);
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
}
