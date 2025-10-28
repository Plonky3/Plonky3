//! Builder for collecting interactions from AIR evaluation.

use alloc::vec;
use alloc::vec::Vec;

use p3_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Entry, SymbolicExpression, SymbolicVariable};

use crate::interaction::{AirBuilderWithInteractions, Interaction, InteractionKind};

/// A builder that collects all interactions defined in an AIR's `eval` method.
pub struct InteractionCollector<F: Field, K: InteractionKind> {
    /// Symbolic variables for the preprocessed trace
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    /// Symbolic variables for the main trace
    main: RowMajorMatrix<SymbolicVariable<F>>,
    /// Collected send interactions
    sends: Vec<Interaction<F, K>>,
    /// Collected receive interactions
    receives: Vec<Interaction<F, K>>,
    /// Public values
    public_values: Vec<F>,
}

impl<F: Field, K: InteractionKind> InteractionCollector<F, K> {
    /// Creates a new interaction collector with the given trace widths.
    ///
    /// # Arguments
    ///
    /// - `preprocessed_width`: Width of the preprocessed trace
    /// - `main_width`: Width of the main trace
    pub fn new(preprocessed_width: usize, main_width: usize) -> Self {
        let preprocessed_width = preprocessed_width.max(1);

        let prep_values = (0..preprocessed_width)
            .map(move |column| SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, column))
            .collect();

        let main_values = (0..main_width)
            .map(move |column| SymbolicVariable::new(Entry::Main { offset: 0 }, column))
            .collect();

        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            sends: vec![],
            receives: vec![],
            public_values: vec![F::ZERO; 0], // Empty by default
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

impl<F: Field, K: InteractionKind> AirBuilder for InteractionCollector<F, K> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        unimplemented!();
    }

    fn is_last_row(&self) -> Self::Expr {
        unimplemented!();
    }

    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        unimplemented!();
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {
        // Do nothing - we only collect interactions, not constraints
    }
}

impl<F: Field, K: InteractionKind> PairBuilder for InteractionCollector<F, K> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field, K: InteractionKind> AirBuilderWithPublicValues for InteractionCollector<F, K> {
    type PublicVar = F;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field, K: InteractionKind> AirBuilderWithInteractions<K> for InteractionCollector<F, K> {
    fn send(&mut self, mut interaction: Interaction<Self::F, K>) {
        // Make the multiplicity negative for sends
        //
        // This is the key semantic: send = negative contribution to LogUp sum
        interaction.multiplicity = -interaction.multiplicity;
        self.sends.push(interaction);
    }

    fn receive(&mut self, interaction: Interaction<Self::F, K>) {
        // Multiplicity stays positive for receives
        self.receives.push(interaction);
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    enum TestKind {
        Lookup1,
        Lookup2,
    }

    #[test]
    fn test_collector_basic() {
        let mut collector = InteractionCollector::<F, TestKind>::new(4, 8);

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
        let mut collector = InteractionCollector::<F, TestKind>::new(0, 4);

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
