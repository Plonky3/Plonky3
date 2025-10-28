//! Builder for collecting interactions from AIR evaluation.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

use crate::interaction::{Interaction, InteractionKind, MessageBuilder};

/// A builder that collects interactions.
#[derive(Debug, Default)]
pub struct InteractionCollector<F: Field, K: InteractionKind> {
    /// All collected interactions, with multiplicities correctly signed.
    interactions: Vec<Interaction<F, K>>,
}

impl<F: Field, K: InteractionKind> InteractionCollector<F, K> {
    /// Creates a new interaction collector.
    pub const fn new() -> Self {
        Self {
            interactions: vec![],
        }
    }

    /// Consumes the builder and returns all collected interactions.
    #[must_use]
    pub fn into_interactions(self) -> Vec<Interaction<F, K>> {
        self.interactions
    }

    /// Returns a reference to the collected interactions.
    pub fn interactions(&self) -> &[Interaction<F, K>] {
        &self.interactions
    }
}

impl<F: Field, K: InteractionKind> MessageBuilder<F, K> for InteractionCollector<F, K> {
    fn send(&mut self, mut interaction: Interaction<F, K>) {
        // Make the multiplicity negative for sends
        //
        // send = negative contribution to LogUp sum
        //
        // TODO: in the future, we may want to allow users to specify
        // the multiplicity of a send interaction.
        interaction.multiplicity = -interaction.multiplicity;
        self.interactions.push(interaction);
    }

    fn receive(&mut self, interaction: Interaction<F, K>) {
        // Multiplicity stays positive for receives
        //
        // TODO: in the future, we may want to allow users to specify
        // the multiplicity of a send interaction.
        self.interactions.push(interaction);
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_uni_stark::SymbolicExpression;

    use super::*;

    type F = BabyBear;

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    enum TestKind {
        Lookup1,
        Lookup2,
    }

    #[test]
    fn test_collector_basic() {
        let mut collector = InteractionCollector::<F, TestKind>::new();

        // Send an interaction
        collector.send(Interaction {
            values: vec![SymbolicExpression::Constant(F::ONE)],
            multiplicity: SymbolicExpression::Constant(F::TWO),
            kind: TestKind::Lookup1,
        });

        // Receive an interaction
        collector.receive(Interaction {
            values: vec![SymbolicExpression::Constant(F::new(42))],
            multiplicity: SymbolicExpression::Constant(F::ONE),
            kind: TestKind::Lookup2,
        });

        let interactions = collector.into_interactions();

        assert_eq!(interactions.len(), 2);

        // Verify that send multiplicity was negated
        if let SymbolicExpression::Constant(m) = interactions[0].multiplicity {
            assert_eq!(m, -F::TWO);
        } else {
            panic!("Expected constant multiplicity");
        }

        // Verify receive multiplicity stayed positive
        if let SymbolicExpression::Constant(m) = interactions[1].multiplicity {
            assert_eq!(m, F::ONE);
        } else {
            panic!("Expected constant multiplicity");
        }
    }

    #[test]
    fn test_multiple_interactions() {
        let mut collector = InteractionCollector::<F, TestKind>::new();

        // Multiple sends and receives
        for i in 0..5 {
            collector.send(Interaction {
                values: vec![SymbolicExpression::Constant(F::new(i))],
                multiplicity: SymbolicExpression::Constant(F::ONE),
                kind: TestKind::Lookup1,
            });

            collector.receive(Interaction {
                values: vec![SymbolicExpression::Constant(F::new(i * 10))],
                multiplicity: SymbolicExpression::Constant(F::ONE),
                kind: TestKind::Lookup2,
            });
        }

        let interactions = collector.interactions();

        assert_eq!(interactions.len(), 10);
    }
}
