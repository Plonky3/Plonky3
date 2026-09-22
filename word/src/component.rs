//! Word gadgets declared once and instantiated many times.
//!
//! A component is one word gadget, written against word slots of its own.
//!
//! A call pairs that gadget with a checked instance count.
//!
//! A composition lays several calls out into one statement.
//!
//! # Addressing
//!
//! Each call owns a contiguous, disjoint block of both word segments.
//!
//! Inside a block the words are packed by instance:
//!
//! ```text
//! public  word = public base   + instance * interface slots + slot
//! private word = private base  + instance * private slots   + slot
//! relation     = relation base + instance * relations       + relation
//! ```
//!
//! Every stride is the block's own width.
//!
//! Two instances can therefore never reach the same word or the same relation.
//!
//! That follows from the arithmetic, so no runtime check has to enforce it.
//!
//! # Why instance-major
//!
//! Binius64 stores a repeated gadget's witness the other way round, instance index minor.
//!
//! That lets its per-gadget commitment split the sumcheck point along the instance axis.
//!
//! Nothing here commits a gadget on its own, so there is no such split to gain.
//!
//! Instance-major instead gives every instance one contiguous run of words.
//!
//! A caller can then be handed one writable slice per instance, and no two of them overlap.
//!
//! # Public inputs and outputs
//!
//! A gadget's public segment is its interface.
//!
//! The leading slots are its inputs and the trailing slots its outputs.
//!
//! Both are verifier-known words of the composed statement.
//!
//! Their positions follow from the declared interface and the instance counts alone.
//!
//! Nothing the prover supplies can move them.

use alloc::vec::Vec;

use thiserror::Error;

use crate::constraint::{AndConstraint, IntegerMulConstraint, Operand, ZeroConstraint};
use crate::index::{Segment, ValueIndex};
use crate::system::{ConstraintKind, ConstraintSystem, ShapeError, SystemError};
use crate::word::Word;

/// Number of relation families a statement can declare.
const FAMILIES: usize = 3;

/// A gadget body whose interface does not describe its public segment.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ComponentError {
    /// The declared interface and the body's public segment disagree.
    #[error(
        "the interface declares {inputs} inputs and {outputs} outputs, but the body has {body} public words"
    )]
    InterfaceMismatch {
        /// Declared public input slots.
        inputs: usize,
        /// Declared public output slots.
        outputs: usize,
        /// Public words the body actually addresses.
        body: usize,
    },
    /// The interface widths do not fit the compact slot representation.
    #[error("the interface declares {inputs} inputs and {outputs} outputs, which cannot be added")]
    InterfaceTooWide {
        /// Declared public input slots.
        inputs: usize,
        /// Declared public output slots.
        outputs: usize,
    },
}

/// A composition that cannot be laid out, or a lookup that names nothing.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum CompositionError {
    /// A composed segment exceeds the compact address space.
    #[error("the composed {segment:?} segment would hold {len} words, which exceeds u32::MAX")]
    SegmentTooLong {
        /// The oversized segment.
        segment: Segment,
        /// The rejected length.
        len: usize,
    },
    /// A composed relation family exceeds the compact address space.
    #[error("the composed {kind:?} family would hold {len} relations, which exceeds u32::MAX")]
    TooManyConstraints {
        /// The oversized relation family.
        kind: ConstraintKind,
        /// The rejected relation count.
        len: usize,
    },
    /// A call index names no call of this composition.
    #[error("call {call} does not exist in a composition of {calls} calls")]
    UnknownCall {
        /// The rejected call index.
        call: usize,
        /// The number of declared calls.
        calls: usize,
    },
    /// An instance index names no instance of the selected call.
    #[error("call {call} declares {instances} instances, so instance {instance} does not exist")]
    UnknownInstance {
        /// The call whose instance was requested.
        call: usize,
        /// The rejected instance index.
        instance: usize,
        /// The number of declared instances.
        instances: usize,
    },
}

/// A word gadget written once against component-local slots.
///
/// Its public segment is the interface one instance exposes.
///
/// Its witness segment is the private storage one instance keeps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Component<W: Word> {
    /// Relations over component-local slots.
    body: ConstraintSystem<W>,
    /// Interface slots the instance reads.
    public_inputs: u32,
    /// Interface slots the instance writes.
    public_outputs: u32,
}

impl<W: Word> Component<W> {
    /// Declares a gadget whose interface covers its whole public segment.
    ///
    /// # Errors
    ///
    /// Returns an error when the two interface widths do not cover the body's public segment.
    pub fn new(
        body: ConstraintSystem<W>,
        public_inputs: usize,
        public_outputs: usize,
    ) -> Result<Self, ComponentError> {
        // A slot that is neither an input nor an output would be unreachable.
        let declared =
            public_inputs
                .checked_add(public_outputs)
                .ok_or(ComponentError::InterfaceTooWide {
                    inputs: public_inputs,
                    outputs: public_outputs,
                })?;
        if declared != body.public_len() {
            return Err(ComponentError::InterfaceMismatch {
                inputs: public_inputs,
                outputs: public_outputs,
                body: body.public_len(),
            });
        }

        // The body already bounded its own segments to the compact index space.
        Ok(Self {
            body,
            public_inputs: public_inputs as u32,
            public_outputs: public_outputs as u32,
        })
    }

    /// Returns the relations, addressed in component-local slot space.
    #[inline]
    pub const fn body(&self) -> &ConstraintSystem<W> {
        &self.body
    }

    /// Returns the number of per-instance public input slots.
    #[inline]
    pub const fn public_inputs(&self) -> usize {
        self.public_inputs as usize
    }

    /// Returns the number of per-instance public output slots.
    #[inline]
    pub const fn public_outputs(&self) -> usize {
        self.public_outputs as usize
    }

    /// Returns the number of per-instance interface slots.
    #[inline]
    pub const fn interface_slots(&self) -> usize {
        self.public_inputs() + self.public_outputs()
    }

    /// Returns the number of per-instance committed slots.
    #[inline]
    pub const fn local_slots(&self) -> usize {
        self.body.witness_len()
    }

    /// Returns the relation counts in zero, AND, then product order.
    #[inline]
    pub fn relation_counts(&self) -> [usize; FAMILIES] {
        [
            self.body.zero_constraints().len(),
            self.body.and_constraints().len(),
            self.body.integer_mul_constraints().len(),
        ]
    }
}

/// One component together with the number of instances it is used for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentCall<W: Word> {
    /// The gadget compiled once for every instance.
    component: Component<W>,
    /// The number of live instances.
    instances: usize,
}

impl<W: Word> ComponentCall<W> {
    /// Declares a number of instances of one component.
    #[inline]
    pub const fn new(component: Component<W>, instances: usize) -> Self {
        // The count is checked against the composed address space in `Composition::new`.
        Self {
            component,
            instances,
        }
    }

    /// Returns the compiled gadget.
    #[inline]
    pub const fn component(&self) -> &Component<W> {
        &self.component
    }

    /// Returns the declared instance count.
    #[inline]
    pub const fn instances(&self) -> usize {
        self.instances
    }
}

/// Where one call's words and relations begin in the composed statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CallBase {
    /// First public word of the call's block.
    public: usize,
    /// First committed word of the call's block.
    witness: usize,
    /// First relation of the call's block, one entry per family.
    constraint: [usize; FAMILIES],
}

/// Several component calls laid out into one statement.
///
/// Each call owns a disjoint, instance-packed block of both segments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Composition<W: Word> {
    /// The calls in layout order.
    calls: Vec<ComponentCall<W>>,
    /// The block start of each call.
    bases: Vec<CallBase>,
    /// Total public words.
    public_len: u32,
    /// Total committed words.
    witness_len: u32,
    /// Total relations per family.
    relation_counts: [u32; FAMILIES],
}

impl<W: Word> Composition<W> {
    /// Lays out the calls, checking every instance count against the address space.
    ///
    /// # Errors
    ///
    /// Returns an error when a composed segment or family outgrows the compact address space.
    pub fn new(calls: Vec<ComponentCall<W>>) -> Result<Self, CompositionError> {
        let mut public_len = 0_usize;
        let mut witness_len = 0_usize;
        let mut relation_counts = [0_usize; FAMILIES];
        let mut bases = Vec::with_capacity(calls.len());

        for call in &calls {
            bases.push(CallBase {
                public: public_len,
                witness: witness_len,
                constraint: relation_counts,
            });

            // Every block is the instance count times one instance's own width.
            let component = &call.component;
            public_len = extend(
                public_len,
                component.interface_slots(),
                call.instances,
                Segment::Public,
            )?;
            witness_len = extend(
                witness_len,
                component.local_slots(),
                call.instances,
                Segment::Witness,
            )?;
            for (family, count) in component.relation_counts().into_iter().enumerate() {
                let kind = ConstraintKind::from_code(family as u8)
                    .expect("relation counts are indexed by assigned family codes");
                relation_counts[family] =
                    extend_constraints(relation_counts[family], count, call.instances, kind)?;
            }
        }

        // A composed statement is addressed with exactly the flat index space.
        let public = bound_segment(public_len, Segment::Public)?;
        let witness = bound_segment(witness_len, Segment::Witness)?;
        let mut counts = [0_u32; FAMILIES];
        for (family, count) in relation_counts.into_iter().enumerate() {
            let kind = ConstraintKind::from_code(family as u8)
                .expect("relation counts are indexed by assigned family codes");
            counts[family] = u32::try_from(count)
                .map_err(|_| CompositionError::TooManyConstraints { kind, len: count })?;
        }

        Ok(Self {
            calls,
            bases,
            public_len: public,
            witness_len: witness,
            relation_counts: counts,
        })
    }

    /// Returns the calls in layout order.
    #[inline]
    pub fn calls(&self) -> &[ComponentCall<W>] {
        &self.calls
    }

    /// Returns the number of composed public words.
    #[inline]
    pub const fn public_len(&self) -> usize {
        self.public_len as usize
    }

    /// Returns the number of composed committed words.
    #[inline]
    pub const fn witness_len(&self) -> usize {
        self.witness_len as usize
    }

    /// Returns the composed relation counts in zero, AND, then product order.
    #[inline]
    pub const fn relation_counts(&self) -> [usize; FAMILIES] {
        [
            self.relation_counts[0] as usize,
            self.relation_counts[1] as usize,
            self.relation_counts[2] as usize,
        ]
    }

    /// Returns the first public word of one call's block.
    ///
    /// # Errors
    ///
    /// Returns an error when the index names no call.
    pub fn public_base(&self, call: usize) -> Result<usize, CompositionError> {
        Ok(self.base(call)?.public)
    }

    /// Returns the first committed word of one call's block.
    ///
    /// # Errors
    ///
    /// Returns an error when the index names no call.
    pub fn witness_base(&self, call: usize) -> Result<usize, CompositionError> {
        Ok(self.base(call)?.witness)
    }

    /// Returns the first relation of one call's block within a family.
    ///
    /// # Errors
    ///
    /// Returns an error when the index names no call.
    pub fn constraint_base(
        &self,
        call: usize,
        kind: ConstraintKind,
    ) -> Result<usize, CompositionError> {
        Ok(self.base(call)?.constraint[kind.code() as usize])
    }

    /// Checks the composed segment lengths against a supplied statement.
    ///
    /// # Errors
    ///
    /// Returns an error naming the segment whose length differs.
    pub const fn check_shape(
        &self,
        public_len: usize,
        witness_len: usize,
    ) -> Result<(), ShapeError> {
        if public_len != self.public_len() {
            return Err(ShapeError {
                segment: Segment::Public,
                expected: self.public_len(),
                actual: public_len,
            });
        }
        if witness_len != self.witness_len() {
            return Err(ShapeError {
                segment: Segment::Witness,
                expected: self.witness_len(),
                actual: witness_len,
            });
        }
        Ok(())
    }

    /// Returns one instance's interface words inside a composed public segment.
    ///
    /// The slice is the inputs followed by the outputs.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn interface<'a>(
        &self,
        public: &'a [W],
        call: usize,
        instance: usize,
    ) -> Result<&'a [W], CompositionError> {
        let range = self.interface_range(call, instance)?;
        Ok(&public[range.0..range.1])
    }

    /// Returns one instance's public input words.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn inputs<'a>(
        &self,
        public: &'a [W],
        call: usize,
        instance: usize,
    ) -> Result<&'a [W], CompositionError> {
        let inputs = self
            .check_instance(call, instance)?
            .component
            .public_inputs();
        let slice = self.interface(public, call, instance)?;
        Ok(&slice[..inputs])
    }

    /// Returns one instance's public output words.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn outputs<'a>(
        &self,
        public: &'a [W],
        call: usize,
        instance: usize,
    ) -> Result<&'a [W], CompositionError> {
        let inputs = self
            .check_instance(call, instance)?
            .component
            .public_inputs();
        let slice = self.interface(public, call, instance)?;
        Ok(&slice[inputs..])
    }

    /// Returns one instance's private words inside a composed witness segment.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn locals<'a>(
        &self,
        witness: &'a [W],
        call: usize,
        instance: usize,
    ) -> Result<&'a [W], CompositionError> {
        let (start, end) = self.local_range(call, instance)?;
        Ok(&witness[start..end])
    }

    /// Returns one instance's interface words for writing.
    ///
    /// Distinct instances yield disjoint slices.
    ///
    /// One instance's word cannot be written through another instance's handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn interface_mut<'a>(
        &self,
        public: &'a mut [W],
        call: usize,
        instance: usize,
    ) -> Result<&'a mut [W], CompositionError> {
        let (start, end) = self.interface_range(call, instance)?;
        Ok(&mut public[start..end])
    }

    /// Returns one instance's private words for writing.
    ///
    /// Distinct instances yield disjoint slices.
    ///
    /// One instance's word cannot be written through another instance's handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn locals_mut<'a>(
        &self,
        witness: &'a mut [W],
        call: usize,
        instance: usize,
    ) -> Result<&'a mut [W], CompositionError> {
        let (start, end) = self.local_range(call, instance)?;
        Ok(&mut witness[start..end])
    }

    /// Materialises the equivalent flat statement.
    ///
    /// This is the duplicated form the composition exists to avoid.
    ///
    /// It is worth building as a reference, or for a caller that wants the flat path.
    ///
    /// Its relation order matches the composed addressing exactly.
    ///
    /// # Errors
    ///
    /// Returns an error when the composed statement outgrows the flat address space.
    pub fn lower(&self) -> Result<ConstraintSystem<W>, SystemError> {
        let counts = self.relation_counts();
        let mut zero = Vec::with_capacity(counts[0]);
        let mut and = Vec::with_capacity(counts[1]);
        let mut product = Vec::with_capacity(counts[2]);

        for (call, base) in self.calls.iter().zip(&self.bases) {
            let body = call.component.body();
            for instance in 0..call.instances {
                // One affine map readdresses every term of this instance.
                let remap = |index: ValueIndex| -> ValueIndex {
                    let position = self.position(base, &call.component, instance, index);
                    match index.segment() {
                        Segment::Public => ValueIndex::public(position),
                        Segment::Witness => ValueIndex::witness(position),
                    }
                    .expect("a composed position was bounded when the layout was built")
                };

                // Families stay in their own order, so each block is contiguous.
                zero.extend(
                    body.zero_constraints().iter().map(|relation| {
                        ZeroConstraint::new(remap_operand(relation.value(), remap))
                    }),
                );
                and.extend(body.and_constraints().iter().map(|relation| {
                    AndConstraint::new(
                        remap_operand(relation.left(), remap),
                        remap_operand(relation.right(), remap),
                        remap_operand(relation.output(), remap),
                    )
                }));
                product.extend(body.integer_mul_constraints().iter().map(|relation| {
                    IntegerMulConstraint::new(
                        remap_operand(relation.left(), remap),
                        remap_operand(relation.right(), remap),
                        remap_operand(relation.low(), remap),
                        remap_operand(relation.high(), remap),
                    )
                }));
            }
        }

        ConstraintSystem::new(self.public_len(), self.witness_len(), zero, and, product)
    }

    /// Readdresses one component-local index into the composed segment.
    ///
    /// # Errors
    ///
    /// Returns an error when the call or the instance does not exist.
    pub fn resolve(
        &self,
        call: usize,
        instance: usize,
        index: ValueIndex,
    ) -> Result<ValueIndex, CompositionError> {
        let base = *self.base(call)?;
        let component = &self.check_instance(call, instance)?.component;
        let position = self.position(&base, component, instance, index);
        Ok(match index.segment() {
            Segment::Public => ValueIndex::public(position),
            Segment::Witness => ValueIndex::witness(position),
        }
        .expect("a composed position was bounded when the layout was built"))
    }

    /// Computes the composed word position of one component-local index.
    const fn position(
        &self,
        base: &CallBase,
        component: &Component<W>,
        instance: usize,
        index: ValueIndex,
    ) -> usize {
        // The instance stride is the block's own width, so blocks never overlap.
        let (start, stride) = match index.segment() {
            Segment::Public => (base.public, component.interface_slots()),
            Segment::Witness => (base.witness, component.local_slots()),
        };
        start + instance * stride + index.position() as usize
    }

    fn base(&self, call: usize) -> Result<&CallBase, CompositionError> {
        self.bases.get(call).ok_or(CompositionError::UnknownCall {
            call,
            calls: self.calls.len(),
        })
    }

    fn check_instance(
        &self,
        call: usize,
        instance: usize,
    ) -> Result<&ComponentCall<W>, CompositionError> {
        let entry = self.calls.get(call).ok_or(CompositionError::UnknownCall {
            call,
            calls: self.calls.len(),
        })?;
        if instance >= entry.instances {
            return Err(CompositionError::UnknownInstance {
                call,
                instance,
                instances: entry.instances,
            });
        }
        Ok(entry)
    }

    fn interface_range(
        &self,
        call: usize,
        instance: usize,
    ) -> Result<(usize, usize), CompositionError> {
        let base = self.base(call)?.public;
        let entry = self.check_instance(call, instance)?;
        let width = entry.component.interface_slots();
        let start = base + instance * width;
        Ok((start, start + width))
    }

    fn local_range(
        &self,
        call: usize,
        instance: usize,
    ) -> Result<(usize, usize), CompositionError> {
        let base = self.base(call)?.witness;
        let entry = self.check_instance(call, instance)?;
        let width = entry.component.local_slots();
        let start = base + instance * width;
        Ok((start, start + width))
    }
}

/// Rebuilds one operand at readdressed word positions.
fn remap_operand<W: Word>(
    operand: &Operand<W>,
    remap: impl Fn(ValueIndex) -> ValueIndex,
) -> Operand<W> {
    // Term order and multiplicity are preserved, so cancellation is unchanged.
    Operand::new(
        operand
            .terms()
            .iter()
            .map(|term| term.with_index(remap(term.index())))
            .collect(),
    )
}

/// Grows one segment by an instance-packed block.
fn extend(
    total: usize,
    slots: usize,
    instances: usize,
    segment: Segment,
) -> Result<usize, CompositionError> {
    slots
        .checked_mul(instances)
        .and_then(|block| total.checked_add(block))
        .ok_or(CompositionError::SegmentTooLong {
            segment,
            len: usize::MAX,
        })
}

/// Grows one relation family by an instance-packed block.
fn extend_constraints(
    total: usize,
    count: usize,
    instances: usize,
    kind: ConstraintKind,
) -> Result<usize, CompositionError> {
    count
        .checked_mul(instances)
        .and_then(|block| total.checked_add(block))
        .ok_or(CompositionError::TooManyConstraints {
            kind,
            len: usize::MAX,
        })
}

/// Bounds one composed segment to the compact protocol address space.
fn bound_segment(len: usize, segment: Segment) -> Result<u32, CompositionError> {
    u32::try_from(len).map_err(|_| CompositionError::SegmentTooLong { segment, len })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::shift::ShiftedValue;
    use crate::word::Word64;

    /// `out = in AND local`, with the input and output public.
    fn masker() -> Component<Word64> {
        let input = ValueIndex::public(0).unwrap();
        let output = ValueIndex::public(1).unwrap();
        let local = ValueIndex::witness(0).unwrap();
        let body = ConstraintSystem::new(
            2,
            1,
            vec![],
            vec![AndConstraint::new(
                Operand::single(ShiftedValue::plain(input)),
                Operand::single(ShiftedValue::plain(local)),
                Operand::single(ShiftedValue::plain(output)),
            )],
            vec![],
        )
        .unwrap();
        Component::new(body, 1, 1).unwrap()
    }

    #[test]
    fn an_interface_must_cover_the_whole_public_segment() {
        let body = ConstraintSystem::<Word64>::new(2, 0, vec![], vec![], vec![]).unwrap();
        assert_eq!(
            Component::new(body, 1, 0),
            Err(ComponentError::InterfaceMismatch {
                inputs: 1,
                outputs: 0,
                body: 2,
            })
        );
    }

    #[test]
    fn instance_blocks_are_disjoint_across_calls_and_instances() {
        // Fixture state: two calls of the same gadget, three and two instances.
        let composition = Composition::new(vec![
            ComponentCall::new(masker(), 3),
            ComponentCall::new(masker(), 2),
        ])
        .unwrap();

        // Every (call, instance, slot) triple must name its own word.
        let mut public = vec![];
        let mut witness = vec![];
        for (call, instances) in [(0, 3), (1, 2)] {
            for instance in 0..instances {
                for slot in 0..2 {
                    let index = ValueIndex::public(slot).unwrap();
                    public.push(composition.resolve(call, instance, index).unwrap());
                }
                let index = ValueIndex::witness(0).unwrap();
                witness.push(composition.resolve(call, instance, index).unwrap());
            }
        }

        let mut sorted = public
            .iter()
            .map(|index| index.position())
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), public.len());
        assert_eq!(sorted.len(), composition.public_len());

        let mut sorted = witness
            .iter()
            .map(|index| index.position())
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), witness.len());
        assert_eq!(sorted.len(), composition.witness_len());
    }

    #[test]
    fn lowering_matches_the_declared_addressing() {
        let composition = Composition::new(vec![ComponentCall::new(masker(), 2)]).unwrap();
        let flat = composition.lower().unwrap();

        // Two instances, two public words and one committed word each.
        assert_eq!(flat.public_len(), 4);
        assert_eq!(flat.witness_len(), 2);
        assert_eq!(flat.and_constraints().len(), 2);

        // Instance one reads exactly the words the affine map names.
        let relation = &flat.and_constraints()[1];
        assert_eq!(relation.left().terms()[0].index().position(), 2);
        assert_eq!(relation.right().terms()[0].index().position(), 1);
        assert_eq!(relation.output().terms()[0].index().position(), 3);
    }

    #[test]
    fn a_lowered_composition_verifies_against_its_own_instances() {
        let composition = Composition::new(vec![ComponentCall::new(masker(), 2)]).unwrap();
        let flat = composition.lower().unwrap();

        let mut public = vec![Word64::new(0); composition.public_len()];
        let mut witness = vec![Word64::new(0); composition.witness_len()];
        for (instance, (input, mask)) in [
            (0xffff_0000_ffff_0000, 0x00ff_00ff_00ff_00ff),
            (0x1234_5678_9abc_def0, 0xf0f0_f0f0_f0f0_f0f0),
        ]
        .into_iter()
        .enumerate()
        {
            let slots = composition.interface_mut(&mut public, 0, instance).unwrap();
            slots[0] = Word64::new(input);
            slots[1] = Word64::new(input & mask);
            composition.locals_mut(&mut witness, 0, instance).unwrap()[0] = Word64::new(mask);
        }

        assert_eq!(flat.verify(&public, &witness), Ok(()));

        // Giving instance one instance zero's mask breaks only that instance.
        let borrowed = composition.locals(&witness, 0, 0).unwrap()[0];
        composition.locals_mut(&mut witness, 0, 1).unwrap()[0] = borrowed;
        assert!(flat.verify(&public, &witness).is_err());
    }

    #[test]
    fn unknown_calls_and_instances_are_named_in_the_error() {
        let composition = Composition::new(vec![ComponentCall::new(masker(), 2)]).unwrap();
        let index = ValueIndex::witness(0).unwrap();

        assert_eq!(
            composition.resolve(1, 0, index),
            Err(CompositionError::UnknownCall { call: 1, calls: 1 })
        );
        assert_eq!(
            composition.resolve(0, 2, index),
            Err(CompositionError::UnknownInstance {
                call: 0,
                instance: 2,
                instances: 2,
            })
        );
    }

    #[test]
    fn a_zero_instance_call_contributes_nothing() {
        let composition = Composition::new(vec![
            ComponentCall::new(masker(), 0),
            ComponentCall::new(masker(), 1),
        ])
        .unwrap();

        assert_eq!(composition.public_len(), 2);
        assert_eq!(composition.witness_len(), 1);
        assert_eq!(composition.public_base(1), Ok(0));
        assert_eq!(composition.relation_counts(), [0, 1, 0]);
    }
}
