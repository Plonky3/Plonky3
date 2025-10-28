//! Builder for collecting interactions from AIR evaluation.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

use crate::interaction::{Interaction, InteractionKind, MessageBuilder};

/// A builder that collects interactions.
#[derive(Debug, Default)]
pub struct InteractionCollector<F: Field, K: InteractionKind> {
    /// Collected send interactions
    sends: Vec<Interaction<F, K>>,
    /// Collected receive interactions
    receives: Vec<Interaction<F, K>>,
}

impl<F: Field, K: InteractionKind> InteractionCollector<F, K> {
    /// Creates a new interaction collector.
    pub const fn new() -> Self {
        Self {
            sends: vec![],
            receives: vec![],
        }
    }

    /// Consumes the builder and returns the collected interactions.
    ///
    /// # Returns
    ///
    /// A tuple of `(sends, receives)` where each is a vector of interactions.
    pub fn into_interactions(self) -> (Vec<Interaction<F, K>>, Vec<Interaction<F, K>>) {
        (self.sends, self.receives)
    }

    /// Returns references to the collected interactions without consuming the builder.
    pub fn interactions(&self) -> (&[Interaction<F, K>], &[Interaction<F, K>]) {
        (&self.sends, &self.receives)
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
        self.sends.push(interaction);
    }

    fn receive(&mut self, interaction: Interaction<F, K>) {
        // Multiplicity stays positive for receives
        //
        // TODO: in the future, we may want to allow users to specify
        // the multiplicity of a send interaction.
        self.receives.push(interaction);
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

        let (sends, receives) = collector.into_interactions();

        assert_eq!(sends.len(), 1);
        assert_eq!(receives.len(), 1);

        // Verify that send multiplicity was negated
        if let SymbolicExpression::Constant(m) = sends[0].multiplicity {
            assert_eq!(m, -F::TWO);
        } else {
            panic!("Expected constant multiplicity");
        }

        // Verify receive multiplicity stayed positive
        if let SymbolicExpression::Constant(m) = receives[0].multiplicity {
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

        let (sends, receives) = collector.interactions();

        assert_eq!(sends.len(), 5);
        assert_eq!(receives.len(), 5);
    }
}
